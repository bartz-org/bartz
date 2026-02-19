# bartz/src/bartz/debug/_check.py
#
# Copyright (c) 2026, The Bartz Contributors
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

"""Implement functions to check validity of trees."""

from typing import Protocol

from jax import jit
from jax import numpy as jnp
from jaxtyping import Array, Bool, Integer, UInt

from bartz.grove import TreeHeaps, is_actual_leaf
from bartz.jaxext import autobatch, minimal_unsigned_dtype
from bartz.mcmcloop import TreesTrace

CHECK_FUNCTIONS = []


class CheckFunc(Protocol):
    """Protocol for functions that check whether a tree is valid."""

    def __call__(
        self, tree: TreeHeaps, max_split: UInt[Array, ' p'], /
    ) -> bool | Bool[Array, '']:
        """Check whether a tree is valid.

        Parameters
        ----------
        tree
            The tree to check.
        max_split
            The maximum split value for each variable.

        Returns
        -------
        A boolean scalar indicating whether the tree is valid.
        """
        ...


def check(func: CheckFunc) -> CheckFunc:
    """Add a function to a list of functions used to check trees.

    Use to decorate functions that check whether a tree is valid in some way.
    These functions are invoked automatically by `check_tree`, `check_trace` and
    `debug_gbart`.

    Parameters
    ----------
    func
        The function to add to the list. It must accept a `TreeHeaps` and a
        `max_split` argument, and return a boolean scalar that indicates if the
        tree is ok.

    Returns
    -------
    The function unchanged.
    """
    CHECK_FUNCTIONS.append(func)
    return func


@check
def check_types(tree: TreeHeaps, max_split: UInt[Array, ' p']) -> bool:
    """Check that integer types are as small as possible and coherent."""
    expected_var_dtype = minimal_unsigned_dtype(max_split.size - 1)
    expected_split_dtype = max_split.dtype
    return (
        tree.var_tree.dtype == expected_var_dtype
        and tree.split_tree.dtype == expected_split_dtype
        and jnp.issubdtype(max_split.dtype, jnp.unsignedinteger)
    )


@check
def check_sizes(tree: TreeHeaps, _max_split: UInt[Array, ' p']) -> bool:
    """Check that array sizes are coherent."""
    return tree.leaf_tree.size == 2 * tree.var_tree.size == 2 * tree.split_tree.size


@check
def check_unused_node(
    tree: TreeHeaps, _max_split: UInt[Array, ' p']
) -> Bool[Array, '']:
    """Check that the unused node slot at index 0 is not dirty."""
    return (tree.var_tree[0] == 0) & (tree.split_tree[0] == 0)


@check
def check_leaf_values(
    tree: TreeHeaps, _max_split: UInt[Array, ' p']
) -> Bool[Array, '']:
    """Check that all leaf values are not inf of nan."""
    return jnp.all(jnp.isfinite(tree.leaf_tree))


@check
def check_stray_nodes(
    tree: TreeHeaps, _max_split: UInt[Array, ' p']
) -> Bool[Array, '']:
    """Check if there is any marked-non-leaf node with a marked-leaf parent."""
    index = jnp.arange(
        2 * tree.split_tree.size,
        dtype=minimal_unsigned_dtype(2 * tree.split_tree.size - 1),
    )
    parent_index = index >> 1
    is_not_leaf = tree.split_tree.at[index].get(mode='fill', fill_value=0) != 0
    parent_is_leaf = tree.split_tree[parent_index] == 0
    stray = is_not_leaf & parent_is_leaf
    stray = stray.at[1].set(False)
    return ~jnp.any(stray)


@check
def check_rule_consistency(
    tree: TreeHeaps, max_split: UInt[Array, ' p']
) -> bool | Bool[Array, '']:
    """Check that decision rules define proper subsets of ancestor rules."""
    if tree.var_tree.size < 4:
        return True

    # initial boundaries of decision rules. use extreme integers instead of 0,
    # max_split to avoid checking if there is something out of bounds.
    dtype = tree.split_tree.dtype
    small = jnp.iinfo(dtype).min
    large = jnp.iinfo(dtype).max
    lower = jnp.full(max_split.size, small, dtype)
    upper = jnp.full(max_split.size, large, dtype)
    # the split must be in (lower[var], upper[var]]

    def _check_recursive(
        node: int, lower: UInt[Array, ' p'], upper: UInt[Array, ' p']
    ) -> Bool[Array, '']:
        # read decision rule
        var = tree.var_tree[node]
        split = tree.split_tree[node]

        # get rule boundaries from ancestors. use fill value in case var is
        # out of bounds, we don't want to check out of bounds in this function
        lower_var = lower.at[var].get(mode='fill', fill_value=small)
        upper_var = upper.at[var].get(mode='fill', fill_value=large)

        # check rule is in bounds
        bad = jnp.where(split, (split <= lower_var) | (split > upper_var), False)

        # recurse
        if node < tree.var_tree.size // 2:
            idx = jnp.where(split, var, max_split.size)
            bad |= _check_recursive(2 * node, lower, upper.at[idx].set(split - 1))
            bad |= _check_recursive(2 * node + 1, lower.at[idx].set(split), upper)

        return bad

    return ~_check_recursive(1, lower, upper)


@check
def check_num_nodes(tree: TreeHeaps, max_split: UInt[Array, ' p']) -> Bool[Array, '']:  # noqa: ARG001
    """Check that #leaves = 1 + #(internal nodes)."""
    is_leaf = is_actual_leaf(tree.split_tree, add_bottom_level=True)
    num_leaves = jnp.count_nonzero(is_leaf)
    num_internal = jnp.count_nonzero(tree.split_tree)
    return num_leaves == num_internal + 1


@check
def check_var_in_bounds(
    tree: TreeHeaps, max_split: UInt[Array, ' p']
) -> Bool[Array, '']:
    """Check that variables are in [0, max_split.size)."""
    decision_node = tree.split_tree.astype(bool)
    in_bounds = (tree.var_tree >= 0) & (tree.var_tree < max_split.size)
    return jnp.all(in_bounds | ~decision_node)


@check
def check_split_in_bounds(
    tree: TreeHeaps, max_split: UInt[Array, ' p']
) -> Bool[Array, '']:
    """Check that splits are in [0, max_split[var]]."""
    max_split_var = (
        max_split.astype(jnp.int32)
        .at[tree.var_tree]
        .get(mode='fill', fill_value=jnp.iinfo(jnp.int32).max)
    )
    return jnp.all((tree.split_tree >= 0) & (tree.split_tree <= max_split_var))


def check_tree(tree: TreeHeaps, max_split: UInt[Array, ' p']) -> UInt[Array, '']:
    """Check the validity of a tree.

    Use `describe_error` to parse the error code returned by this function.

    Parameters
    ----------
    tree
        The tree to check.
    max_split
        The maximum split value for each variable.

    Returns
    -------
    An integer where each bit indicates whether a check failed.
    """
    error_type = minimal_unsigned_dtype(2 ** len(CHECK_FUNCTIONS) - 1)
    error = error_type(0)
    for i, func in enumerate(CHECK_FUNCTIONS):
        ok = func(tree, max_split)
        ok = jnp.bool_(ok)
        bit = (~ok) << i
        error |= bit
    return error


def describe_error(error: int | Integer[Array, '']) -> list[str]:
    """Describe an error code returned by `check_trace`.

    Parameters
    ----------
    error
        An error code returned by `check_trace`.

    Returns
    -------
    A list of the function names that implement the failed checks.
    """
    return [func.__name__ for i, func in enumerate(CHECK_FUNCTIONS) if error & (1 << i)]


@jit
def check_trace(
    trace: TreeHeaps, max_split: UInt[Array, ' p']
) -> UInt[Array, '*batch_shape']:
    """Check the validity of a set of trees.

    Use `describe_error` to parse the error codes returned by this function.

    Parameters
    ----------
    trace
        The set of trees to check. This object can have additional attributes
        beyond the tree arrays, they are ignored.
    max_split
        The maximum split value for each variable.

    Returns
    -------
    A tensor of error codes for each tree.
    """
    # vectorize check_tree over all batch dimensions
    unpack_check_tree = lambda l, v, s: check_tree(TreesTrace(l, v, s), max_split)
    is_mv = trace.leaf_tree.ndim > trace.split_tree.ndim
    signature = '(k,ts),(hts),(hts)->()' if is_mv else '(ts),(hts),(hts)->()'
    vec_check_tree = jnp.vectorize(unpack_check_tree, signature=signature)

    # automatically batch over all batch dimensions
    max_io_nbytes = 2**24  # 16 MiB
    batch_ndim = trace.split_tree.ndim - 1
    batched_check_tree = vec_check_tree
    for i in reversed(range(batch_ndim)):
        batched_check_tree = autobatch(batched_check_tree, max_io_nbytes, i, i)

    return batched_check_tree(trace.leaf_tree, trace.var_tree, trace.split_tree)
