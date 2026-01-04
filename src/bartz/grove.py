# bartz/src/bartz/grove.py
#
# Copyright (c) 2024-2026, The Bartz Contributors
#
# This file is part of bartz.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Functions to create and manipulate binary decision trees."""

import math
from functools import partial
from typing import Protocol

from jax import jit, lax, vmap
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float32, Int32, Shaped, UInt

try:
    from numpy.lib.array_utils import normalize_axis_tuple  # numpy 2
except ImportError:
    from numpy.core.numeric import normalize_axis_tuple  # numpy 1

from bartz.jaxext import minimal_unsigned_dtype, vmap_nodoc


class TreeHeaps(Protocol):
    """A protocol for dataclasses that represent trees.

    A tree is represented with arrays as a heap. The root node is at index 1.
    The children nodes of a node at index :math:`i` are at indices :math:`2i`
    (left child) and :math:`2i + 1` (right child). The array element at index 0
    is unused.

    Arrays may have additional initial axes to represent multple trees.

    Parameters
    ----------
    leaf_tree
        The values in the leaves of the trees. This array can be dirty, i.e.,
        unused nodes can have whatever value. It may have an additional axis
        for multivariate leaves.
    var_tree
        The axes along which the decision nodes operate. This array can be
        dirty but for the always unused node at index 0 which must be set to 0.
    split_tree
        The decision boundaries of the trees. The boundaries are open on the
        right, i.e., a point belongs to the left child iff x < split. Whether a
        node is a leaf is indicated by the corresponding 'split' element being
        0. Unused nodes also have split set to 0. This array can't be dirty.

    Notes
    -----
    Since the nodes at the bottom can only be leaves and not decision nodes,
    `var_tree` and `split_tree` are half as long as `leaf_tree`.
    """

    leaf_tree: (
        Float32[Array, '*batch_shape 2**d'] | Float32[Array, '*batch_shape k 2**d']
    )
    var_tree: UInt[Array, '*batch_shape 2**(d-1)']
    split_tree: UInt[Array, '*batch_shape 2**(d-1)']


def make_tree(
    depth: int, dtype: DTypeLike, batch_shape: tuple[int, ...] = ()
) -> Shaped[Array, '*batch_shape 2**{depth}']:
    """
    Make an array to represent a binary tree.

    Parameters
    ----------
    depth
        The maximum depth of the tree. Depth 1 means that there is only a root
        node.
    dtype
        The dtype of the array.
    batch_shape
        The leading shape of the array, to represent multiple trees and/or
        multivariate trees.

    Returns
    -------
    An array of zeroes with the appropriate shape.
    """
    shape = (*batch_shape, 2**depth)
    return jnp.zeros(shape, dtype)


def tree_depth(tree: Shaped[Array, '*batch_shape 2**d']) -> int:
    """
    Return the maximum depth of a tree.

    Parameters
    ----------
    tree
        A tree created by `make_tree`. If the array is ND, the tree structure is
        assumed to be along the last axis.

    Returns
    -------
    The maximum depth of the tree.
    """
    return round(math.log2(tree.shape[-1]))


def traverse_tree(
    x: UInt[Array, ' p'],
    var_tree: UInt[Array, ' 2**(d-1)'],
    split_tree: UInt[Array, ' 2**(d-1)'],
) -> UInt[Array, '']:
    """
    Find the leaf where a point falls into.

    Parameters
    ----------
    x
        The coordinates to evaluate the tree at.
    var_tree
        The decision axes of the tree.
    split_tree
        The decision boundaries of the tree.

    Returns
    -------
    The index of the leaf.
    """
    carry = (
        jnp.zeros((), bool),
        jnp.ones((), minimal_unsigned_dtype(2 * var_tree.size - 1)),
    )

    def loop(carry, _):
        leaf_found, index = carry

        split = split_tree[index]
        var = var_tree[index]

        leaf_found |= split == 0
        child_index = (index << 1) + (x[var] >= split)
        index = jnp.where(leaf_found, index, child_index)

        return (leaf_found, index), None

    depth = tree_depth(var_tree)
    (_, index), _ = lax.scan(loop, carry, None, depth, unroll=16)
    return index


@jit
@partial(jnp.vectorize, excluded=(0,), signature='(hts),(hts)->(n)')
@partial(vmap_nodoc, in_axes=(1, None, None))
def traverse_forest(
    X: UInt[Array, 'p n'],
    var_trees: UInt[Array, '*forest_shape 2**(d-1)'],
    split_trees: UInt[Array, '*forest_shape 2**(d-1)'],
) -> UInt[Array, '*forest_shape n']:
    """
    Find the leaves where points falls into for each tree in a set.

    Parameters
    ----------
    X
        The coordinates to evaluate the trees at.
    var_trees
        The decision axes of the trees.
    split_trees
        The decision boundaries of the trees.

    Returns
    -------
    The indices of the leaves.
    """
    return traverse_tree(X, var_trees, split_trees)


@partial(jit, static_argnames=('sum_batch_axis',))
def evaluate_forest(
    X: UInt[Array, 'p n'],
    trees: TreeHeaps,
    *,
    sum_batch_axis: int | tuple[int, ...] = (),
) -> (
    Float32[Array, '*reduced_batch_size n'] | Float32[Array, '*reduced_batch_size n k']
):
    """
    Evaluate an ensemble of trees at an array of points.

    Parameters
    ----------
    X
        The coordinates to evaluate the trees at.
    trees
        The trees.
    sum_batch_axis
        The batch axes to sum over. By default, no summation is performed.
        Note that negative indices count from the end of the batch dimensions,
        the core dimensions n and k can't be summed over by this function.

    Returns
    -------
    The (sum of) the values of the trees at the points in `X`.
    """
    indices: UInt[Array, '*forest_shape n']
    indices = traverse_forest(X, trees.var_tree, trees.split_tree)

    is_mv = trees.leaf_tree.ndim != trees.var_tree.ndim

    bc_indices: UInt[Array, '*forest_shape n 1'] | UInt[Array, '*forest_shape n 1 1']
    bc_indices = indices[..., None, None] if is_mv else indices[..., None]

    bc_leaf_tree: (
        Float32[Array, '*forest_shape 1 tree_size']
        | Float32[Array, '*forest_shape 1 k tree_size']
    )
    bc_leaf_tree = (
        trees.leaf_tree[..., None, :, :] if is_mv else trees.leaf_tree[..., None, :]
    )

    bc_leaves: (
        Float32[Array, '*forest_shape n 1'] | Float32[Array, '*forest_shape n k 1']
    )
    bc_leaves = jnp.take_along_axis(bc_leaf_tree, bc_indices, -1)

    leaves: Float32[Array, '*forest_shape n'] | Float32[Array, '*forest_shape n k']
    leaves = jnp.squeeze(bc_leaves, -1)

    axis = normalize_axis_tuple(sum_batch_axis, trees.var_tree.ndim - 1)
    return jnp.sum(leaves, axis=axis)


def is_actual_leaf(
    split_tree: UInt[Array, ' 2**(d-1)'], *, add_bottom_level: bool = False
) -> Bool[Array, ' 2**(d-1)'] | Bool[Array, ' 2**d']:
    """
    Return a mask indicating the leaf nodes in a tree.

    Parameters
    ----------
    split_tree
        The splitting points of the tree.
    add_bottom_level
        If True, the bottom level of the tree is also considered.

    Returns
    -------
    The mask marking the leaf nodes. Length doubled if `add_bottom_level` is True.
    """
    size = split_tree.size
    is_leaf = split_tree == 0
    if add_bottom_level:
        size *= 2
        is_leaf = jnp.concatenate([is_leaf, jnp.ones_like(is_leaf)])
    index = jnp.arange(size, dtype=minimal_unsigned_dtype(size - 1))
    parent_index = index >> 1
    parent_nonleaf = split_tree[parent_index].astype(bool)
    parent_nonleaf = parent_nonleaf.at[1].set(True)
    return is_leaf & parent_nonleaf


def is_leaves_parent(split_tree: UInt[Array, ' 2**(d-1)']) -> Bool[Array, ' 2**(d-1)']:
    """
    Return a mask indicating the nodes with leaf (and only leaf) children.

    Parameters
    ----------
    split_tree
        The decision boundaries of the tree.

    Returns
    -------
    The mask indicating which nodes have leaf children.
    """
    index = jnp.arange(
        split_tree.size, dtype=minimal_unsigned_dtype(2 * split_tree.size - 1)
    )
    left_index = index << 1  # left child
    right_index = left_index + 1  # right child
    left_leaf = split_tree.at[left_index].get(mode='fill', fill_value=0) == 0
    right_leaf = split_tree.at[right_index].get(mode='fill', fill_value=0) == 0
    is_not_leaf = split_tree.astype(bool)
    return is_not_leaf & left_leaf & right_leaf
    # the 0-th item has split == 0, so it's not counted


def tree_depths(tree_length: int) -> Int32[Array, ' {tree_length}']:
    """
    Return the depth of each node in a binary tree.

    Parameters
    ----------
    tree_length
        The length of the tree array, i.e., 2 ** d.

    Returns
    -------
    The depth of each node.

    Notes
    -----
    The root node (index 1) has depth 0. The depth is the position of the most
    significant non-zero bit in the index. The first element (the unused node)
    is marked as depth 0.
    """
    depths = []
    depth = 0
    for i in range(tree_length):
        if i == 2**depth:
            depth += 1
        depths.append(depth - 1)
    depths[0] = 0
    return jnp.array(depths, minimal_unsigned_dtype(max(depths)))


@partial(jnp.vectorize, signature='(half_tree_size)->(tree_size)')
def is_used(
    split_tree: UInt[Array, '*batch_shape 2**(d-1)'],
) -> Bool[Array, '*batch_shape 2**d']:
    """
    Return a mask indicating the used nodes in a tree.

    Parameters
    ----------
    split_tree
        The decision boundaries of the tree.

    Returns
    -------
    A mask indicating which nodes are actually used.
    """
    internal_node = split_tree.astype(bool)
    internal_node = jnp.concatenate([internal_node, jnp.zeros_like(internal_node)])
    actual_leaf = is_actual_leaf(split_tree, add_bottom_level=True)
    return internal_node | actual_leaf


@jit
def forest_fill(split_tree: UInt[Array, '*batch_shape 2**(d-1)']) -> Float32[Array, '']:
    """
    Return the fraction of used nodes in a set of trees.

    Parameters
    ----------
    split_tree
        The decision boundaries of the trees.

    Returns
    -------
    Number of tree nodes over the maximum number that could be stored.
    """
    used = is_used(split_tree)
    count = jnp.count_nonzero(used)
    batch_size = split_tree.size // split_tree.shape[-1]
    return count / (used.size - batch_size)


@partial(jit, static_argnames=('p', 'sum_batch_axis'))
def var_histogram(
    p: int,
    var_tree: UInt[Array, '*batch_shape 2**(d-1)'],
    split_tree: UInt[Array, '*batch_shape 2**(d-1)'],
    *,
    sum_batch_axis: int | tuple[int, ...] = (),
) -> Int32[Array, '*reduced_batch_shape {p}']:
    """
    Count how many times each variable appears in a tree.

    Parameters
    ----------
    p
        The number of variables (the maximum value that can occur in
        `var_tree` is ``p - 1``).
    var_tree
        The decision axes of the tree.
    split_tree
        The decision boundaries of the tree.
    sum_batch_axis
        The batch axes to sum over. By default, no summation is performed.
        Note that negative indices count from the end of the batch dimensions,
        the core dimension p can't be summed over by this function.

    Returns
    -------
    The histogram(s) of the variables used in the tree.
    """
    is_internal = split_tree.astype(bool)

    def scatter_add(
        var_tree: UInt[Array, '*summed_batch_axes half_tree_size'],
        is_internal: Bool[Array, '*summed_batch_axes half_tree_size'],
    ) -> Int32[Array, ' p']:
        return jnp.zeros(p, int).at[var_tree].add(is_internal)

    # vmap scatter_add over non-batched dims
    batch_ndim = var_tree.ndim - 1
    axes = normalize_axis_tuple(sum_batch_axis, batch_ndim)
    for i in reversed(range(batch_ndim)):
        neg_i = i - var_tree.ndim
        if i not in axes:
            scatter_add = vmap(scatter_add, in_axes=(neg_i, neg_i))

    return scatter_add(var_tree, is_internal)
