# bartz/src/bartz/grove/_grove.py
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
from dataclasses import fields
from functools import partial
from typing import Literal, Protocol, runtime_checkable

from equinox import Module, tree_at
from jax import jit, lax, vmap
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float32, Int32, Shaped, UInt
from numpy.lib.array_utils import normalize_axis_tuple

from bartz._jaxext import autobatch, minimal_unsigned_dtype, vmap_nodoc


@runtime_checkable
class TreeHeaps(Protocol):
    """A protocol for dataclasses that represent trees.

    A tree is represented with arrays as a heap. The root node is at index 1.
    The children nodes of a node at index :math:`i` are at indices :math:`2i`
    (left child) and :math:`2i + 1` (right child). The array element at index 0
    is unused.

    Since the nodes at the bottom can only be leaves and not decision nodes,
    `var_tree` and `split_tree` are half as long as `leaf_tree`.

    Arrays may have additional initial axes to represent multiple trees.
    """

    leaf_tree: (
        Float32[Array, '*batch_shape 2*half_tree_size']
        | Float32[Array, '*batch_shape k 2*half_tree_size']
    )
    """The values in the leaves of the trees. This array can be dirty, i.e.,
    unused nodes can have whatever value. It may have an additional axis
    for multivariate leaves."""

    var_tree: UInt[Array, '*batch_shape half_tree_size']
    """The axes along which the decision nodes operate. This array can be
    dirty but for the always unused node at index 0 which must be set to 0."""

    split_tree: UInt[Array, '*batch_shape half_tree_size']
    """The decision boundaries of the trees. The boundaries are open on the
    right, i.e., a point belongs to the left child iff x < split. Whether a
    node is a leaf is indicated by the corresponding 'split' element being
    0. Unused nodes also have split set to 0. This array can't be dirty."""


class HeapArrays(Module):
    """Mixin providing shared behavior for `TreeHeaps` dataclasses.

    Subclasses must declare the `leaf_tree`, `var_tree` and `split_tree` heap
    arrays (see `TreeHeaps`); this mixin adds no fields, only the derived
    quantities that are the same regardless of how the leading batch axes are
    laid out.
    """

    @property
    def is_multivariate(self) -> bool:
        """Whether the leaves are vector-valued (an extra `k` axis on `leaf_tree`)."""
        return self.leaf_tree.ndim > self.var_tree.ndim


class TreesTrace(HeapArrays):
    """Implementation of `bartz.grove.TreeHeaps` for an MCMC trace."""

    # `var_tree`/`split_tree` are declared before `leaf_tree` so their single
    # (union-free) annotations bind the variadic `*batch_shape` axis first;
    # otherwise the runtime typechecker (which evaluates union members in a
    # hash-randomized order) can mis-bind it against the `k` axis of
    # `leaf_tree`'s union for a multivariate tree (the layouts are
    # rank-ambiguous). See `bartz.mcmcstep._state.Forest`. The leaf-bearing axis
    # is `2*half_tree_size` rather than `tree_size`, so the half-of-leaf
    # relationship is still checked here: `half_tree_size` is bound first by the
    # anchors, then `leaf_tree` is checked against twice it.
    var_tree: UInt[Array, '*batch_shape half_tree_size']
    split_tree: UInt[Array, '*batch_shape half_tree_size']
    leaf_tree: (
        Float32[Array, '*batch_shape 2*half_tree_size']
        | Float32[Array, '*batch_shape k 2*half_tree_size']
    )

    @classmethod
    def from_dataclass(cls, obj: TreeHeaps) -> 'TreesTrace':
        """Create a `TreesTrace` from any `bartz.grove.TreeHeaps`."""
        return cls(**{f.name: getattr(obj, f.name) for f in fields(cls)})

    def axes_from_dataclass(self, obj: TreeHeaps) -> 'TreesTrace':
        """Project the per-field vmap axis specs of `obj` onto this template.

        `self` supplies the (array) pytree; the same-named fields of `obj`
        (axis specs, i.e. ints or `None`) replace its leaves. Built with
        `tree_at`, which bypasses the type-checked `__init__`, so the
        deliberately off-type axis values are allowed.
        """
        names = [f.name for f in fields(type(self))]
        return tree_at(
            lambda t: [getattr(t, name) for name in names],
            self,
            [getattr(obj, name) for name in names],
        )


def tree_depth(tree: Shaped[Array, '*batch_shape tree_size']) -> int:
    """
    Return the maximum depth of a tree.

    Parameters
    ----------
    tree
        A tree array like those in a `TreeHeaps`. If the array is ND, the tree
        structure is assumed to be along the last axis.

    Returns
    -------
    The maximum depth of the tree.
    """
    return round(math.log2(tree.shape[-1]))


def traverse_tree(
    x: UInt[Array, ' p'],
    var_tree: UInt[Array, ' half_tree_size'],
    split_tree: UInt[Array, ' half_tree_size'],
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

    def loop(
        carry: tuple[Bool[Array, ''], UInt[Array, '']], _: None
    ) -> tuple[tuple[Bool[Array, ''], UInt[Array, '']], None]:
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
def traverse_forest(
    X: UInt[Array, 'p n'],
    var_trees: UInt[Array, '*forest_shape half_tree_size'],
    split_trees: UInt[Array, '*forest_shape half_tree_size'],
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
    return _traverse_forest(X, var_trees, split_trees)


@partial(jnp.vectorize, excluded=(0,), signature='(hts),(hts)->(n)')
@partial(vmap_nodoc, in_axes=(1, None, None))
def _traverse_forest(
    X: UInt[Array, ' p'],
    var_trees: UInt[Array, ' half_tree_size'],
    split_trees: UInt[Array, ' half_tree_size'],
) -> UInt[Array, '']:
    """Implement `traverse_forest`."""
    return traverse_tree(X, var_trees, split_trees)


@partial(jit, static_argnames=('sum_batch_axis',))
def evaluate_forest(
    X: UInt[Array, 'p n'],
    trees: TreeHeaps,
    *,
    sum_batch_axis: int | tuple[int, ...] = (),
) -> (
    Float32[Array, '*reduced_batch_size n'] | Float32[Array, '*reduced_batch_size k n']
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

    is_mv = trees.is_multivariate

    bc_indices: UInt[Array, '*forest_shape n 1'] | UInt[Array, '*forest_shape 1 n 1']
    bc_indices = indices[..., None, :, None] if is_mv else indices[..., None]

    bc_leaf_tree: (
        Float32[Array, '*forest_shape 1 tree_size']
        | Float32[Array, '*forest_shape k 1 tree_size']
    )
    bc_leaf_tree = (
        trees.leaf_tree[..., :, None, :] if is_mv else trees.leaf_tree[..., None, :]
    )

    bc_leaves: (
        Float32[Array, '*forest_shape n 1'] | Float32[Array, '*forest_shape k n 1']
    )
    bc_leaves = jnp.take_along_axis(bc_leaf_tree, bc_indices, -1)

    leaves: Float32[Array, '*forest_shape n'] | Float32[Array, '*forest_shape k n']
    leaves = jnp.squeeze(bc_leaves, -1)

    axis = normalize_axis_tuple(sum_batch_axis, trees.var_tree.ndim - 1)
    return jnp.sum(leaves, axis=axis)


def is_actual_leaf(
    split_tree: UInt[Array, ' half_tree_size'], *, add_bottom_level: bool = False
) -> Bool[Array, ' half_tree_size'] | Bool[Array, ' 2*half_tree_size']:
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


def is_leaves_parent(
    split_tree: UInt[Array, ' half_tree_size'],
) -> Bool[Array, ' half_tree_size']:
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


def tree_depths(tree_size: int) -> UInt[Array, ' {tree_size}']:
    """
    Return the depth of each node in a binary tree.

    Parameters
    ----------
    tree_size
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
    for i in range(tree_size):
        if i == 2**depth:
            depth += 1
        depths.append(depth - 1)
    depths[0] = 0
    return jnp.array(depths, minimal_unsigned_dtype(max(depths)))


@jit
def forest_mean_leaves(
    split_tree: UInt[Array, '*batch_shape half_tree_size'],
) -> Float32[Array, '']:
    """
    Return the average number of leaves per tree in a set of trees.

    Parameters
    ----------
    split_tree
        The decision boundaries of the trees.

    Returns
    -------
    The mean number of leaves across the trees.
    """
    # a tree with k internal nodes (the nonzero entries of split_tree) has k + 1
    # leaves; the maximum possible is split_tree.shape[-1]
    num_internal = jnp.count_nonzero(split_tree, axis=-1)
    return (num_internal + 1).mean()


@partial(jit, static_argnames=('p', 'sum_batch_axis'))
def var_histogram(
    p: int,
    var_tree: UInt[Array, '*batch_shape half_tree_size'],
    split_tree: UInt[Array, '*batch_shape half_tree_size'],
    *,
    sum_batch_axis: int | tuple[int, ...] = (),
) -> Int32[Array, '*reduced_batch_shape {p}']:
    """
    Count how many times each variable appears in a tree.

    Parameters
    ----------
    p
        The number of variables (the maximum value that can occur in `var_tree`
        is ``p - 1``).
    var_tree
        The decision axes of the tree.
    split_tree
        The decision boundaries of the tree.
    sum_batch_axis
        The batch axes to sum over. By default, no summation is performed. Note
        that negative indices count from the end of the batch dimensions, the
        core dimension p can't be summed over by this function.

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
            scatter_add = vmap(scatter_add, in_axes=neg_i)

    return scatter_add(var_tree, is_internal)


def _format_leaf(leaf: Float32[Array, ''] | Float32[Array, ' k'], is_mv: bool) -> str:
    """Format a (possibly multivariate) leaf value to 2 significant digits."""
    if is_mv:
        return '[' + ', '.join(f'{v:#.2g}' for v in leaf) + ']'
    return f'{leaf:#.2g}'


def format_tree(tree: TreeHeaps, *, print_all: bool = False) -> str:
    """Convert a tree to a human-readable string.

    Parameters
    ----------
    tree
        A single tree to format.
    print_all
        If `True`, also print the contents of unused node slots in the arrays.

    Returns
    -------
    A string representation of the tree.
    """
    tee = '├──'
    corner = '└──'
    join = '│  '
    space = '   '
    down = '┐'
    bottom = '╢'  # '┨' #

    *_, tree_size = tree.leaf_tree.shape
    is_mv = tree.is_multivariate

    def traverse_tree(
        lines: list[str],
        index: int,
        depth: int,
        indent: str,
        first_indent: str,
        next_indent: str,
        unused: bool,
    ) -> None:
        if index >= tree_size:
            return

        var: int = tree.var_tree.at[index].get(mode='fill', fill_value=0).item()
        split: int = tree.split_tree.at[index].get(mode='fill', fill_value=0).item()

        is_leaf = split == 0
        left_child = 2 * index
        right_child = 2 * index + 1

        if print_all:
            if unused:
                category = 'unused'
            elif is_leaf:
                category = 'leaf'
            else:
                category = 'decision'
            node_str = f'{category}({var}, {split}, {tree.leaf_tree[..., index]})'
        else:
            assert not unused
            if is_leaf:
                node_str = _format_leaf(tree.leaf_tree[..., index], is_mv)
            else:
                node_str = f'x{var} < {split}'

        if not is_leaf or (print_all and left_child < tree_size):
            link = down
        elif not print_all and left_child >= tree_size:
            link = bottom
        else:
            link = ' '

        max_number = tree_size - 1
        ndigits = len(str(max_number))
        number = str(index).rjust(ndigits)

        lines.append(f' {number} {indent}{first_indent}{link}{node_str}')

        indent += next_indent
        unused = unused or is_leaf

        if unused and not print_all:
            return

        traverse_tree(lines, left_child, depth + 1, indent, tee, join, unused)
        traverse_tree(lines, right_child, depth + 1, indent, corner, space, unused)

    lines = []
    traverse_tree(lines, 1, 0, '', '', '', False)
    return '\n'.join(lines)


def tree_actual_depth(split_tree: UInt[Array, ' half_tree_size']) -> UInt[Array, '']:
    """Measure the depth of the tree.

    Parameters
    ----------
    split_tree
        The cutpoints of the decision rules.

    Returns
    -------
    The depth of the deepest leaf in the tree. The root is at depth 0.
    """
    # this could be done just with split_tree != 0
    is_leaf = is_actual_leaf(split_tree, add_bottom_level=True)
    depth = tree_depths(is_leaf.size)
    depth = jnp.where(is_leaf, depth, 0)
    return jnp.max(depth)


@jit
@partial(jnp.vectorize, signature='(nt,hts)->(d)')
def forest_depth_distr(
    split_tree: UInt[Array, '*batch_shape num_trees half_tree_size'],
) -> Int32[Array, '*batch_shape d']:
    """Histogram the depths of a set of trees.

    Parameters
    ----------
    split_tree
        The cutpoints of the decision rules of the trees.

    Returns
    -------
    An integer vector where the i-th element counts how many trees have depth i.
    """
    depth = tree_depth(split_tree) + 1
    depths = vmap(tree_actual_depth)(split_tree)
    return jnp.bincount(depths, length=depth)


@partial(jit, static_argnames=('node_type', 'sum_batch_axis'))
def points_per_node_distr(
    X: UInt[Array, 'p n'],
    var_tree: UInt[Array, '*batch_shape half_tree_size'],
    split_tree: UInt[Array, '*batch_shape half_tree_size'],
    node_type: Literal['leaf', 'leaf-parent'],
    *,
    sum_batch_axis: int | tuple[int, ...] = (),
) -> Int32[Array, '*reduced_batch_shape n+1']:
    """Histogram points-per-node counts in a set of trees.

    Count how many nodes in a tree select each possible amount of points,
    over a certain subset of nodes.

    Parameters
    ----------
    X
        The set of points to count.
    var_tree
        The variables of the decision rules.
    split_tree
        The cutpoints of the decision rules.
    node_type
        The type of nodes to consider. Can be:

        'leaf'
            Count only leaf nodes.
        'leaf-parent'
            Count only parent-of-leaf nodes.
    sum_batch_axis
        Aggregate the histogram over these batch axes, counting how many nodes
        have each possible amount of points over subsets of trees instead of
        in each tree separately.

    Returns
    -------
    A vector where the i-th element counts how many nodes have i points.
    """
    batch_ndim = var_tree.ndim - 1
    axes = normalize_axis_tuple(sum_batch_axis, batch_ndim)

    def func(
        var_tree: UInt[Array, '*batch_shape half_tree_size'],
        split_tree: UInt[Array, '*batch_shape half_tree_size'],
    ) -> Int32[Array, '*reduced_batch_shape n_plus_1']:
        indices: UInt[Array, '*batch_shape n']
        indices = traverse_forest(X, var_tree, split_tree)

        @partial(jnp.vectorize, signature='(hts),(n)->(ts_or_hts),(ts_or_hts)')
        def count_points(
            split_tree: UInt[Array, '*batch_shape half_tree_size'],
            indices: UInt[Array, '*batch_shape n'],
        ) -> (
            tuple[
                Int32[Array, '*batch_shape 2*half_tree_size'],
                Bool[Array, '*batch_shape 2*half_tree_size'],
            ]
            | tuple[
                Int32[Array, '*batch_shape half_tree_size'],
                Bool[Array, '*batch_shape half_tree_size'],
            ]
        ):
            if node_type == 'leaf-parent':
                indices >>= 1
                predicate = is_leaves_parent(split_tree)
            elif node_type == 'leaf':
                predicate = is_actual_leaf(split_tree, add_bottom_level=True)
            else:
                raise ValueError(node_type)
            count_tree = jnp.zeros(predicate.size, int).at[indices].add(1).at[0].set(0)
            return count_tree, predicate

        count_tree, predicate = count_points(split_tree, indices)

        def count_nodes(
            count_tree: Int32[Array, '*summed_batch_axes half_tree_size'],
            predicate: Bool[Array, '*summed_batch_axes half_tree_size'],
        ) -> Int32[Array, ' n_plus_1']:
            return jnp.zeros(X.shape[1] + 1, int).at[count_tree].add(predicate)

        # vmap count_nodes over non-batched dims
        for i in reversed(range(batch_ndim)):
            neg_i = i - var_tree.ndim
            if i not in axes:
                count_nodes = vmap(count_nodes, in_axes=neg_i)

        return count_nodes(count_tree, predicate)

    # automatically batch over all batch dimensions
    max_io_nbytes = 2**27  # 128 MiB
    out_dim_shift = len(axes)
    for i in reversed(range(batch_ndim)):
        if i in axes:
            out_dim_shift -= 1
        else:
            func = autobatch(func, max_io_nbytes, i, i - out_dim_shift)
    assert out_dim_shift == 0

    return func(var_tree, split_tree)
