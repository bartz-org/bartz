# bartz/src/bartz/debug.py
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

"""Debugging utilities. The main functionality is the class `debug_mc_gbart`."""

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from math import ceil, log2
from re import fullmatch

import numpy
from equinox import Module, field
from jax import jit, lax, random, vmap
from jax import numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import Array, Bool, Float32, Int32, Integer, Key, UInt

from bartz.BART import FloatLike, gbart, mc_gbart
from bartz.grove import (
    TreeHeaps,
    evaluate_forest,
    is_actual_leaf,
    is_leaves_parent,
    traverse_tree,
    tree_depth,
    tree_depths,
)
from bartz.jaxext import minimal_unsigned_dtype, vmap_nodoc
from bartz.jaxext import split as split_key
from bartz.mcmcloop import TreesTrace
from bartz.mcmcstep._moves import randint_masked


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

    def traverse_tree(
        lines: list[str],
        index: int,
        depth: int,
        indent: str,
        first_indent: str,
        next_indent: str,
        unused: bool,
    ):
        if index >= len(tree.leaf_tree):
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
            node_str = f'{category}({var}, {split}, {tree.leaf_tree[index]})'
        else:
            assert not unused
            if is_leaf:
                node_str = f'{tree.leaf_tree[index]:#.2g}'
            else:
                node_str = f'x{var} < {split}'

        if not is_leaf or (print_all and left_child < len(tree.leaf_tree)):
            link = down
        elif not print_all and left_child >= len(tree.leaf_tree):
            link = bottom
        else:
            link = ' '

        max_number = len(tree.leaf_tree) - 1
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


def tree_actual_depth(split_tree: UInt[Array, ' 2**(d-1)']) -> Int32[Array, '']:
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


def forest_depth_distr(
    split_tree: UInt[Array, 'num_trees 2**(d-1)'],
) -> Int32[Array, ' d']:
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


@jit
def trace_depth_distr(
    split_tree: UInt[Array, 'trace_length num_trees 2**(d-1)'],
) -> Int32[Array, 'trace_length d']:
    """Histogram the depths of a sequence of sets of trees.

    Parameters
    ----------
    split_tree
        The cutpoints of the decision rules of the trees.

    Returns
    -------
    A matrix where element (t,i) counts how many trees have depth i in set t.
    """
    return vmap(forest_depth_distr)(split_tree)


@vmap_nodoc
def chains_depth_distr(
    split_tree: UInt[Array, 'nchains trace_length num_trees 2**(d-1)'],
) -> Int32[Array, 'nchains trace_length d']:
    """Histogram the depths of chains of forests of trees.

    Parameters
    ----------
    split_tree
        The cutpoints of the decision rules of the trees.

    Returns
    -------
    A tensor where element (c,t,i) counts how many trees have depth i in forest t in chain c.
    """
    return trace_depth_distr(split_tree)


def points_per_decision_node_distr(
    var_tree: UInt[Array, ' 2**(d-1)'],
    split_tree: UInt[Array, ' 2**(d-1)'],
    X: UInt[Array, 'p n'],
) -> Int32[Array, ' n+1']:
    """Histogram points-per-node counts.

    Count how many parent-of-leaf nodes in a tree select each possible amount
    of points.

    Parameters
    ----------
    var_tree
        The variables of the decision rules.
    split_tree
        The cutpoints of the decision rules.
    X
        The set of points to count.

    Returns
    -------
    A vector where the i-th element counts how many next-to-leaf nodes have i points.
    """
    traverse_tree_X = vmap(traverse_tree, in_axes=(1, None, None))
    indices = traverse_tree_X(X, var_tree, split_tree)
    indices >>= 1
    count_tree = jnp.zeros(split_tree.size, int).at[indices].add(1).at[0].set(0)
    is_parent = is_leaves_parent(split_tree)
    return jnp.zeros(X.shape[1] + 1, int).at[count_tree].add(is_parent)


def forest_points_per_decision_node_distr(
    trees: TreeHeaps, X: UInt[Array, 'p n']
) -> Int32[Array, ' n+1']:
    """Histogram points-per-node counts for a set of trees.

    Count how many parent-of-leaf nodes in a set of trees select each possible
    amount of points.

    Parameters
    ----------
    trees
        The set of trees. The variables must have broadcast shape (num_trees,).
    X
        The set of points to count.

    Returns
    -------
    A vector where the i-th element counts how many next-to-leaf nodes have i points.
    """
    distr = jnp.zeros(X.shape[1] + 1, int)

    def loop(distr, heaps: tuple[Array, Array]):
        return distr + points_per_decision_node_distr(*heaps, X), None

    distr, _ = lax.scan(loop, distr, (trees.var_tree, trees.split_tree))
    return distr


@jit
@partial(vmap_nodoc, in_axes=(0, None))
def chains_points_per_decision_node_distr(
    chains: TreeHeaps, X: UInt[Array, 'p n']
) -> Int32[Array, 'nchains trace_length n+1']:
    """Separately histogram points-per-node counts over chains of forests of trees.

    For each set of trees, count how many parent-of-leaf nodes select each
    possible amount of points.

    Parameters
    ----------
    chains
        The chains of forests of trees. The variables must have broadcast shape
        (nchains, trace_length, num_trees).
    X
        The set of points to count.

    Returns
    -------
    A tensor where element (c,t,i) counts how many next-to-leaf nodes have i points in forest t in chain c.
    """

    def loop(_, forests):
        return None, forest_points_per_decision_node_distr(forests, X)

    _, distr = lax.scan(loop, None, chains)
    return distr


def points_per_leaf_distr(
    var_tree: UInt[Array, ' 2**(d-1)'],
    split_tree: UInt[Array, ' 2**(d-1)'],
    X: UInt[Array, 'p n'],
) -> Int32[Array, ' n+1']:
    """Histogram points-per-leaf counts in a tree.

    Count how many leaves in a tree select each possible amount of points.

    Parameters
    ----------
    var_tree
        The variables of the decision rules.
    split_tree
        The cutpoints of the decision rules.
    X
        The set of points to count.

    Returns
    -------
    A vector where the i-th element counts how many leaves have i points.
    """
    traverse_tree_X = vmap(traverse_tree, in_axes=(1, None, None))
    indices = traverse_tree_X(X, var_tree, split_tree)
    count_tree = jnp.zeros(2 * split_tree.size, int).at[indices].add(1)
    is_leaf = is_actual_leaf(split_tree, add_bottom_level=True)
    return jnp.zeros(X.shape[1] + 1, int).at[count_tree].add(is_leaf)


def forest_points_per_leaf_distr(
    trees: TreeHeaps, X: UInt[Array, 'p n']
) -> Int32[Array, ' n+1']:
    """Histogram points-per-leaf counts over a set of trees.

    Count how many leaves in a set of trees select each possible amount of points.

    Parameters
    ----------
    trees
        The set of trees. The variables must have broadcast shape (num_trees,).
    X
        The set of points to count.

    Returns
    -------
    A vector where the i-th element counts how many leaves have i points.
    """
    distr = jnp.zeros(X.shape[1] + 1, int)

    def loop(distr, heaps: tuple[Array, Array]):
        return distr + points_per_leaf_distr(*heaps, X), None

    distr, _ = lax.scan(loop, distr, (trees.var_tree, trees.split_tree))
    return distr


@jit
@partial(vmap_nodoc, in_axes=(0, None))
def chains_points_per_leaf_distr(
    chains: TreeHeaps, X: UInt[Array, 'p n']
) -> Int32[Array, 'nchains trace_length n+1']:
    """Separately histogram points-per-leaf counts over chains of forests of trees.

    For each set of trees, count how many leaves select each possible amount of
    points.

    Parameters
    ----------
    chains
        The chains of forests of trees. The variables must have broadcast shape
        (nchains, trace_length, num_trees).
    X
        The set of points to count.

    Returns
    -------
    A matrix where element (t,i) counts how many leaves have i points in set t.
    """

    def loop(_, forests):
        return None, forest_points_per_leaf_distr(forests, X)

    _, distr = lax.scan(loop, None, chains)
    return distr


check_functions = []


CheckFunc = Callable[[TreeHeaps, UInt[Array, ' p']], bool | Bool[Array, '']]


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
    check_functions.append(func)
    return func


@check
def check_types(tree: TreeHeaps, max_split: UInt[Array, ' p']) -> bool:
    """Check that integer types are as small as possible and coherent."""
    expected_var_dtype = minimal_unsigned_dtype(max_split.size - 1)
    expected_split_dtype = max_split.dtype
    return (
        tree.var_tree.dtype == expected_var_dtype
        and tree.split_tree.dtype == expected_split_dtype
    )


@check
def check_sizes(tree: TreeHeaps, max_split: UInt[Array, ' p']) -> bool:  # noqa: ARG001
    """Check that array sizes are coherent."""
    return tree.leaf_tree.size == 2 * tree.var_tree.size == 2 * tree.split_tree.size


@check
def check_unused_node(tree: TreeHeaps, max_split: UInt[Array, ' p']) -> Bool[Array, '']:  # noqa: ARG001
    """Check that the unused node slot at index 0 is not dirty."""
    return (tree.var_tree[0] == 0) & (tree.split_tree[0] == 0)


@check
def check_leaf_values(tree: TreeHeaps, max_split: UInt[Array, ' p']) -> Bool[Array, '']:  # noqa: ARG001
    """Check that all leaf values are not inf of nan."""
    return jnp.all(jnp.isfinite(tree.leaf_tree))


@check
def check_stray_nodes(tree: TreeHeaps, max_split: UInt[Array, ' p']) -> Bool[Array, '']:  # noqa: ARG001
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
    small = jnp.iinfo(jnp.int32).min
    large = jnp.iinfo(jnp.int32).max
    lower = jnp.full(max_split.size, small, jnp.int32)
    upper = jnp.full(max_split.size, large, jnp.int32)
    # specify the type explicitly, otherwise they are weakly types and get
    # implicitly converted to split.dtype (typically uint8) in the expressions

    def _check_recursive(node, lower, upper):
        # read decision rule
        var = tree.var_tree[node]
        split = tree.split_tree[node]

        # get rule boundaries from ancestors. use fill value in case var is
        # out of bounds, we don't want to check out of bounds in this function
        lower_var = lower.at[var].get(mode='fill', fill_value=small)
        upper_var = upper.at[var].get(mode='fill', fill_value=large)

        # check rule is in bounds
        bad = jnp.where(split, (split <= lower_var) | (split >= upper_var), False)

        # recurse
        if node < tree.var_tree.size // 2:
            idx = jnp.where(split, var, max_split.size)
            bad |= _check_recursive(2 * node, lower, upper.at[idx].set(split))
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
    error_type = minimal_unsigned_dtype(2 ** len(check_functions) - 1)
    error = error_type(0)
    for i, func in enumerate(check_functions):
        ok = func(tree, max_split)
        ok = jnp.bool_(ok)
        bit = (~ok) << i
        error |= bit
    return error


def describe_error(error: int | Integer[Array, '']) -> list[str]:
    """Describe the error code returned by `check_tree`.

    Parameters
    ----------
    error
        The error code returned by `check_tree`.

    Returns
    -------
    A list of the function names that implement the failed checks.
    """
    return [func.__name__ for i, func in enumerate(check_functions) if error & (1 << i)]


@jit
@partial(vmap_nodoc, in_axes=(0, None))
def check_trace(
    trace: TreeHeaps, max_split: UInt[Array, ' p']
) -> UInt[Array, 'trace_length num_trees']:
    """Check the validity of a sequence of sets of trees.

    Use `describe_error` to parse the error codes returned by this function.

    Parameters
    ----------
    trace
        The sequence of sets of trees to check. The tree arrays must have
        broadcast shape (trace_length, num_trees). This object can have
        additional attributes beyond the tree arrays, they are ignored.
    max_split
        The maximum split value for each variable.

    Returns
    -------
    A matrix of error codes for each tree.
    """
    trees = TreesTrace.from_dataclass(trace)
    return lax.map(partial(check_tree, max_split=max_split), trees)


@partial(vmap_nodoc, in_axes=(0, None))
def check_chains(
    chains: TreeHeaps, max_split: UInt[Array, ' p']
) -> UInt[Array, 'nchains trace_length num_trees']:
    """Check the validity of sequences of sets of trees.

    Use `describe_error` to parse the error codes returned by this function.

    Parameters
    ----------
    chains
        The sequences of sets of trees to check. The tree arrays must have
        broadcast shape (nchains, trace_length, num_trees). This object can have
        additional attributes beyond the tree arrays, they are ignored.
    max_split
        The maximum split value for each variable.

    Returns
    -------
    A tensor of error codes for each tree.
    """
    return check_trace(chains, max_split)


def _get_next_line(s: str, i: int) -> tuple[str, int]:
    """Get the next line from a string and the new index."""
    i_new = s.find('\n', i)
    if i_new == -1:
        return s[i:], len(s)
    return s[i:i_new], i_new + 1


class BARTTraceMeta(Module):
    """Metadata of R BART tree traces.

    Parameters
    ----------
    ndpost
        The number of posterior draws.
    ntree
        The number of trees in the model.
    numcut
        The maximum split value for each variable.
    heap_size
        The size of the heap required to store the trees.
    """

    ndpost: int = field(static=True)
    ntree: int = field(static=True)
    numcut: UInt[Array, ' p']
    heap_size: int = field(static=True)


def scan_BART_trees(trees: str) -> BARTTraceMeta:
    """Scan an R BART tree trace checking for errors and parsing metadata.

    Parameters
    ----------
    trees
        The string representation of a trace of trees of the R BART package.
        Can be accessed from ``mc_gbart(...).treedraws['trees']``.

    Returns
    -------
    An object containing the metadata.

    Raises
    ------
    ValueError
        If the string is malformed or contains leftover characters.
    """
    # parse first line
    line, i_char = _get_next_line(trees, 0)
    i_line = 1
    match = fullmatch(r'(\d+) (\d+) (\d+)', line)
    if match is None:
        msg = f'Malformed header at {i_line=}'
        raise ValueError(msg)
    ndpost, ntree, p = map(int, match.groups())

    # initial values for maxima
    max_heap_index = 0
    numcut = numpy.zeros(p, int)

    # cycle over iterations and trees
    for i_iter in range(ndpost):
        for i_tree in range(ntree):
            # parse first line of tree definition
            line, i_char = _get_next_line(trees, i_char)
            i_line += 1
            match = fullmatch(r'(\d+)', line)
            if match is None:
                msg = f'Malformed tree header at {i_iter=} {i_tree=} {i_line=}'
                raise ValueError(msg)
            num_nodes = int(line)

            # cycle over nodes
            for i_node in range(num_nodes):
                # parse node definition
                line, i_char = _get_next_line(trees, i_char)
                i_line += 1
                match = fullmatch(
                    r'(\d+) (\d+) (\d+) (-?\d+(\.\d+)?(e(\+|-|)\d+)?)', line
                )
                if match is None:
                    msg = f'Malformed node definition at {i_iter=} {i_tree=} {i_node=} {i_line=}'
                    raise ValueError(msg)
                i_heap = int(match.group(1))
                var = int(match.group(2))
                split = int(match.group(3))

                # update maxima
                numcut[var] = max(numcut[var], split)
                max_heap_index = max(max_heap_index, i_heap)

    assert i_char <= len(trees)
    if i_char < len(trees):
        msg = f'Leftover {len(trees) - i_char} characters in string'
        raise ValueError(msg)

    # determine minimal integer type for numcut
    numcut += 1  # because BART is 0-based
    split_dtype = minimal_unsigned_dtype(numcut.max())
    numcut = jnp.array(numcut.astype(split_dtype))

    # determine minimum heap size to store the trees
    heap_size = 2 ** ceil(log2(max_heap_index + 1))

    return BARTTraceMeta(ndpost=ndpost, ntree=ntree, numcut=numcut, heap_size=heap_size)


class TraceWithOffset(Module):
    """Implementation of `bartz.mcmcloop.Trace`."""

    leaf_tree: Float32[Array, 'ndpost ntree 2**d']
    var_tree: UInt[Array, 'ndpost ntree 2**(d-1)']
    split_tree: UInt[Array, 'ndpost ntree 2**(d-1)']
    offset: Float32[Array, ' ndpost']

    @classmethod
    def from_trees_trace(
        cls, trees: TreeHeaps, offset: Float32[Array, '']
    ) -> 'TraceWithOffset':
        """Create a `TraceWithOffset` from a `TreeHeaps`."""
        ndpost, _, _ = trees.leaf_tree.shape
        return cls(
            leaf_tree=trees.leaf_tree,
            var_tree=trees.var_tree,
            split_tree=trees.split_tree,
            offset=jnp.full(ndpost, offset),
        )


def trees_BART_to_bartz(
    trees: str, *, min_maxdepth: int = 0, offset: FloatLike | None = None
) -> tuple[TraceWithOffset, BARTTraceMeta]:
    """Convert trees from the R BART format to the bartz format.

    Parameters
    ----------
    trees
        The string representation of a trace of trees of the R BART package.
        Can be accessed from ``mc_gbart(...).treedraws['trees']``.
    min_maxdepth
        The maximum tree depth of the output will be set to the maximum
        observed depth in the input trees. Use this parameter to require at
        least this maximum depth in the output format.
    offset
        The trace returned by `bartz.mcmcloop.run_mcmc` contains an offset to be
        summed to the sum of trees. To match that behavior, this function
        returns an offset as well, zero by default. Set with this parameter
        otherwise.

    Returns
    -------
    trace : TraceWithOffset
        A representation of the trees compatible with the trace returned by
        `bartz.mcmcloop.run_mcmc`.
    meta : BARTTraceMeta
        The metadata of the trace, containing the number of iterations, trees,
        and the maximum split value.
    """
    # scan all the string checking for errors and determining sizes
    meta = scan_BART_trees(trees)

    # skip first line
    _, i_char = _get_next_line(trees, 0)

    heap_size = max(meta.heap_size, 2**min_maxdepth)
    leaf_trees = numpy.zeros((meta.ndpost, meta.ntree, heap_size), dtype=numpy.float32)
    var_trees = numpy.zeros(
        (meta.ndpost, meta.ntree, heap_size // 2),
        dtype=minimal_unsigned_dtype(meta.numcut.size - 1),
    )
    split_trees = numpy.zeros(
        (meta.ndpost, meta.ntree, heap_size // 2), dtype=meta.numcut.dtype
    )

    # cycle over iterations and trees
    for i_iter in range(meta.ndpost):
        for i_tree in range(meta.ntree):
            # parse first line of tree definition
            line, i_char = _get_next_line(trees, i_char)
            num_nodes = int(line)

            is_internal = numpy.zeros(heap_size // 2, dtype=bool)

            # cycle over nodes
            for _ in range(num_nodes):
                # parse node definition
                line, i_char = _get_next_line(trees, i_char)
                values = line.split()
                i_heap = int(values[0])
                var = int(values[1])
                split = int(values[2])
                leaf = float(values[3])

                # update values
                leaf_trees[i_iter, i_tree, i_heap] = leaf
                is_internal[i_heap // 2] = True
                if i_heap < heap_size // 2:
                    var_trees[i_iter, i_tree, i_heap] = var
                    split_trees[i_iter, i_tree, i_heap] = split + 1

            is_internal[0] = False
            split_trees[i_iter, i_tree, ~is_internal] = 0

    return TraceWithOffset(
        leaf_tree=jnp.array(leaf_trees),
        var_tree=jnp.array(var_trees),
        split_tree=jnp.array(split_trees),
        offset=jnp.zeros(meta.ndpost)
        if offset is None
        else jnp.full(meta.ndpost, offset),
    ), meta


class SamplePriorStack(Module):
    """Represent the manually managed stack used in `sample_prior`.

    Each level of the stack represents a recursion into a child node in a
    binary tree of maximum depth `d`.

    Parameters
    ----------
    nonterminal
        Whether the node is valid or the recursion is into unused node slots.
    lower
    upper
        The available cutpoints along ``var`` are in the integer range
        ``[1 + lower[var], 1 + upper[var])``.
    var
    split
        The variable and cutpoint of a decision node.
    """

    nonterminal: Bool[Array, ' d-1']
    lower: UInt[Array, 'd-1 p']
    upper: UInt[Array, 'd-1 p']
    var: UInt[Array, ' d-1']
    split: UInt[Array, ' d-1']

    @classmethod
    def initial(
        cls, p_nonterminal: Float32[Array, ' d-1'], max_split: UInt[Array, ' p']
    ) -> 'SamplePriorStack':
        """Initialize the stack.

        Parameters
        ----------
        p_nonterminal
            The prior probability of a node being non-terminal conditional on
            its ancestors and on having available decision rules, at each depth.
        max_split
            The number of cutpoints along each variable.

        Returns
        -------
        A `SamplePriorStack` initialized to start the recursion.
        """
        var_dtype = minimal_unsigned_dtype(max_split.size - 1)
        return cls(
            nonterminal=jnp.ones(p_nonterminal.size, bool),
            lower=jnp.zeros((p_nonterminal.size, max_split.size), max_split.dtype),
            upper=jnp.broadcast_to(max_split, (p_nonterminal.size, max_split.size)),
            var=jnp.zeros(p_nonterminal.size, var_dtype),
            split=jnp.zeros(p_nonterminal.size, max_split.dtype),
        )


class SamplePriorTrees(Module):
    """Object holding the trees generated by `sample_prior`.

    Parameters
    ----------
    leaf_tree
    var_tree
    split_tree
        The arrays representing the trees, see `bartz.grove`.
    """

    leaf_tree: Float32[Array, '* 2**d']
    var_tree: UInt[Array, '* 2**(d-1)']
    split_tree: UInt[Array, '* 2**(d-1)']

    @classmethod
    def initial(
        cls,
        key: Key[Array, ''],
        sigma_mu: Float32[Array, ''],
        p_nonterminal: Float32[Array, ' d-1'],
        max_split: UInt[Array, ' p'],
    ) -> 'SamplePriorTrees':
        """Initialize the trees.

        The leaves are already correct and do not need to be changed.

        Parameters
        ----------
        key
            A jax random key.
        sigma_mu
            The prior standard deviation of each leaf.
        p_nonterminal
            The prior probability of a node being non-terminal conditional on
            its ancestors and on having available decision rules, at each depth.
        max_split
            The number of cutpoints along each variable.

        Returns
        -------
        Trees initialized with random leaves and stub tree structures.
        """
        heap_size = 2 ** (p_nonterminal.size + 1)
        return cls(
            leaf_tree=sigma_mu * random.normal(key, (heap_size,)),
            var_tree=jnp.zeros(
                heap_size // 2, dtype=minimal_unsigned_dtype(max_split.size - 1)
            ),
            split_tree=jnp.zeros(heap_size // 2, dtype=max_split.dtype),
        )


class SamplePriorCarry(Module):
    """Object holding values carried along the recursion in `sample_prior`.

    Parameters
    ----------
    key
        A jax random key used to sample decision rules.
    stack
        The stack used to manage the recursion.
    trees
        The output arrays.
    """

    key: Key[Array, '']
    stack: SamplePriorStack
    trees: SamplePriorTrees

    @classmethod
    def initial(
        cls,
        key: Key[Array, ''],
        sigma_mu: Float32[Array, ''],
        p_nonterminal: Float32[Array, ' d-1'],
        max_split: UInt[Array, ' p'],
    ) -> 'SamplePriorCarry':
        """Initialize the carry object.

        Parameters
        ----------
        key
            A jax random key.
        sigma_mu
            The prior standard deviation of each leaf.
        p_nonterminal
            The prior probability of a node being non-terminal conditional on
            its ancestors and on having available decision rules, at each depth.
        max_split
            The number of cutpoints along each variable.

        Returns
        -------
        A `SamplePriorCarry` initialized to start the recursion.
        """
        keys = split_key(key)
        return cls(
            keys.pop(),
            SamplePriorStack.initial(p_nonterminal, max_split),
            SamplePriorTrees.initial(keys.pop(), sigma_mu, p_nonterminal, max_split),
        )


class SamplePriorX(Module):
    """Object representing the recursion scan in `sample_prior`.

    The sequence of nodes to visit is pre-computed recursively once, unrolling
    the recursion schedule.

    Parameters
    ----------
    node
        The heap index of the node to visit.
    depth
        The depth of the node.
    next_depth
        The depth of the next node to visit, either the left child or the right
        sibling of the node or of an ancestor.
    """

    node: Int32[Array, ' 2**(d-1)-1']
    depth: Int32[Array, ' 2**(d-1)-1']
    next_depth: Int32[Array, ' 2**(d-1)-1']

    @classmethod
    def initial(cls, p_nonterminal: Float32[Array, ' d-1']) -> 'SamplePriorX':
        """Initialize the sequence of nodes to visit.

        Parameters
        ----------
        p_nonterminal
            The prior probability of a node being non-terminal conditional on
            its ancestors and on having available decision rules, at each depth.

        Returns
        -------
        A `SamplePriorX` initialized with the sequence of nodes to visit.
        """
        seq = cls._sequence(p_nonterminal.size)
        assert len(seq) == 2**p_nonterminal.size - 1
        node = [node for node, depth in seq]
        depth = [depth for node, depth in seq]
        next_depth = [*depth[1:], p_nonterminal.size]
        return cls(
            node=jnp.array(node),
            depth=jnp.array(depth),
            next_depth=jnp.array(next_depth),
        )

    @classmethod
    def _sequence(
        cls, max_depth: int, depth: int = 0, node: int = 1
    ) -> tuple[tuple[int, int], ...]:
        """Recursively generate a sequence [(node, depth), ...]."""
        if depth < max_depth:
            out = ((node, depth),)
            out += cls._sequence(max_depth, depth + 1, 2 * node)
            out += cls._sequence(max_depth, depth + 1, 2 * node + 1)
            return out
        return ()


def sample_prior_onetree(
    key: Key[Array, ''],
    max_split: UInt[Array, ' p'],
    p_nonterminal: Float32[Array, ' d-1'],
    sigma_mu: Float32[Array, ''],
) -> SamplePriorTrees:
    """Sample a tree from the BART prior.

    Parameters
    ----------
    key
        A jax random key.
    max_split
        The maximum split value for each variable.
    p_nonterminal
        The prior probability of a node being non-terminal conditional on
        its ancestors and on having available decision rules, at each depth.
    sigma_mu
        The prior standard deviation of each leaf.

    Returns
    -------
    An object containing a generated tree.
    """
    carry = SamplePriorCarry.initial(key, sigma_mu, p_nonterminal, max_split)
    xs = SamplePriorX.initial(p_nonterminal)

    def loop(carry: SamplePriorCarry, x: SamplePriorX):
        keys = split_key(carry.key, 4)

        # get variables at current stack level
        stack = carry.stack
        nonterminal = stack.nonterminal[x.depth]
        lower = stack.lower[x.depth, :]
        upper = stack.upper[x.depth, :]

        # sample a random decision rule
        available: Bool[Array, ' p'] = lower < upper
        allowed = jnp.any(available)
        var = randint_masked(keys.pop(), available)
        split = 1 + random.randint(keys.pop(), (), lower[var], upper[var])

        # cast to shorter integer types
        var = var.astype(carry.trees.var_tree.dtype)
        split = split.astype(carry.trees.split_tree.dtype)

        # decide whether to try to grow the node if it is growable
        pnt = p_nonterminal[x.depth]
        try_nonterminal: Bool[Array, ''] = random.bernoulli(keys.pop(), pnt)
        nonterminal &= try_nonterminal & allowed

        # update trees
        trees = carry.trees
        trees = replace(
            trees,
            var_tree=trees.var_tree.at[x.node].set(var),
            split_tree=trees.split_tree.at[x.node].set(
                jnp.where(nonterminal, split, 0)
            ),
        )

        def write_push_stack() -> SamplePriorStack:
            """Update the stack to go to the left child."""
            return replace(
                stack,
                nonterminal=stack.nonterminal.at[x.next_depth].set(nonterminal),
                lower=stack.lower.at[x.next_depth, :].set(lower),
                upper=stack.upper.at[x.next_depth, :].set(upper.at[var].set(split - 1)),
                var=stack.var.at[x.depth].set(var),
                split=stack.split.at[x.depth].set(split),
            )

        def pop_push_stack() -> SamplePriorStack:
            """Update the stack to go to the right sibling, possibly at lower depth."""
            var = stack.var[x.next_depth - 1]
            split = stack.split[x.next_depth - 1]
            lower = stack.lower[x.next_depth - 1, :]
            upper = stack.upper[x.next_depth - 1, :]
            return replace(
                stack,
                lower=stack.lower.at[x.next_depth, :].set(lower.at[var].set(split)),
                upper=stack.upper.at[x.next_depth, :].set(upper),
            )

        # update stack
        stack = lax.cond(x.next_depth > x.depth, write_push_stack, pop_push_stack)

        # update carry
        carry = replace(carry, key=keys.pop(), stack=stack, trees=trees)
        return carry, None

    carry, _ = lax.scan(loop, carry, xs)
    return carry.trees


@partial(vmap_nodoc, in_axes=(0, None, None, None))
def sample_prior_forest(
    keys: Key[Array, ' num_trees'],
    max_split: UInt[Array, ' p'],
    p_nonterminal: Float32[Array, ' d-1'],
    sigma_mu: Float32[Array, ''],
) -> SamplePriorTrees:
    """Sample a set of independent trees from the BART prior.

    Parameters
    ----------
    keys
        A sequence of jax random keys, one for each tree. This determined the
        number of trees sampled.
    max_split
        The maximum split value for each variable.
    p_nonterminal
        The prior probability of a node being non-terminal conditional on
        its ancestors and on having available decision rules, at each depth.
    sigma_mu
        The prior standard deviation of each leaf.

    Returns
    -------
    An object containing the generated trees.
    """
    return sample_prior_onetree(keys, max_split, p_nonterminal, sigma_mu)


@partial(jit, static_argnums=(1, 2))
def sample_prior(
    key: Key[Array, ''],
    trace_length: int,
    num_trees: int,
    max_split: UInt[Array, ' p'],
    p_nonterminal: Float32[Array, ' d-1'],
    sigma_mu: Float32[Array, ''],
) -> SamplePriorTrees:
    """Sample independent trees from the BART prior.

    Parameters
    ----------
    key
        A jax random key.
    trace_length
        The number of iterations.
    num_trees
        The number of trees for each iteration.
    max_split
        The number of cutpoints along each variable.
    p_nonterminal
        The prior probability of a node being non-terminal conditional on
        its ancestors and on having available decision rules, at each depth.
        This determines the maximum depth of the trees.
    sigma_mu
        The prior standard deviation of each leaf.

    Returns
    -------
    An object containing the generated trees, with batch shape (trace_length, num_trees).
    """
    keys = random.split(key, trace_length * num_trees)
    trees = sample_prior_forest(keys, max_split, p_nonterminal, sigma_mu)
    return tree_map(lambda x: x.reshape(trace_length, num_trees, -1), trees)


class debug_mc_gbart(mc_gbart):
    """A subclass of `mc_gbart` that adds debugging functionality.

    Parameters
    ----------
    *args
        Passed to `mc_gbart`.
    check_trees
        If `True`, check all trees with `check_trace` after running the MCMC,
        and assert that they are all valid. Set to `False` to allow jax tracing.
    **kw
        Passed to `mc_gbart`.
    """

    def __init__(self, *args, check_trees: bool = True, **kw):
        super().__init__(*args, **kw)
        if check_trees:
            bad = self.check_trees()
            bad_count = jnp.count_nonzero(bad)
            assert bad_count == 0

    def print_tree(
        self, i_chain: int, i_sample: int, i_tree: int, print_all: bool = False
    ):
        """Print a single tree in human-readable format.

        Parameters
        ----------
        i_chain
            The index of the MCMC chain.
        i_sample
            The index of the (post-burnin) sample in the chain.
        i_tree
            The index of the tree in the sample.
        print_all
            If `True`, also print the content of unused node slots.
        """
        tree = TreesTrace.from_dataclass(self._main_trace)
        tree = tree_map(lambda x: x[i_chain, i_sample, i_tree, :], tree)
        s = format_tree(tree, print_all=print_all)
        print(s)  # noqa: T201, this method is intended for debug

    def sigma_harmonic_mean(self, prior: bool = False) -> Float32[Array, ' mc_cores']:
        """Return the harmonic mean of the error variance.

        Parameters
        ----------
        prior
            If `True`, use the prior distribution, otherwise use the full
            conditional at the last MCMC iteration.

        Returns
        -------
        The harmonic mean 1/E[1/sigma^2] in the selected distribution.
        """
        bart = self._mcmc_state
        assert bart.error_cov_df is not None
        assert bart.z is None
        # inverse gamma prior: alpha = df / 2, beta = scale / 2
        if prior:
            alpha = bart.error_cov_df / 2
            beta = bart.error_cov_scale / 2
        else:
            alpha = bart.error_cov_df / 2 + bart.resid.size / 2
            norm2 = jnp.einsum('ij,ij->i', bart.resid, bart.resid)
            beta = bart.error_cov_scale / 2 + norm2 / 2
        error_cov_inv = alpha / beta
        return jnp.sqrt(lax.reciprocal(error_cov_inv))

    def compare_resid(
        self,
    ) -> tuple[Float32[Array, 'mc_cores n'], Float32[Array, 'mc_cores n']]:
        """Re-compute residuals to compare them with the updated ones.

        Returns
        -------
        resid1 : Float32[Array, 'mc_cores n']
            The final state of the residuals updated during the MCMC.
        resid2 : Float32[Array, 'mc_cores n']
            The residuals computed from the final state of the trees.
        """
        bart = self._mcmc_state
        resid1 = bart.resid

        forests = TreesTrace.from_dataclass(bart.forest)
        trees = evaluate_forest(bart.X, forests, sum_batch_axis=-1)

        if bart.z is not None:
            ref = bart.z
        else:
            ref = bart.y
        resid2 = ref - (trees + bart.offset)

        return resid1, resid2

    def avg_acc(
        self,
    ) -> tuple[Float32[Array, ' mc_cores'], Float32[Array, ' mc_cores']]:
        """Compute the average acceptance rates of tree moves.

        Returns
        -------
        acc_grow : Float32[Array, 'mc_cores']
            The average acceptance rate of grow moves.
        acc_prune : Float32[Array, 'mc_cores']
            The average acceptance rate of prune moves.
        """
        trace = self._main_trace

        def acc(prefix):
            acc = getattr(trace, f'{prefix}_acc_count')
            prop = getattr(trace, f'{prefix}_prop_count')
            return acc.sum(axis=1) / prop.sum(axis=1)

        return acc('grow'), acc('prune')

    def avg_prop(
        self,
    ) -> tuple[Float32[Array, ' mc_cores'], Float32[Array, ' mc_cores']]:
        """Compute the average proposal rate of grow and prune moves.

        Returns
        -------
        prop_grow : Float32[Array, 'mc_cores']
            The fraction of times grow was proposed instead of prune.
        prop_prune : Float32[Array, 'mc_cores']
            The fraction of times prune was proposed instead of grow.

        Notes
        -----
        This function does not take into account cases where no move was
        proposed.
        """
        trace = self._main_trace

        def prop(prefix):
            return getattr(trace, f'{prefix}_prop_count').sum(axis=1)

        pgrow = prop('grow')
        pprune = prop('prune')
        total = pgrow + pprune
        return pgrow / total, pprune / total

    def avg_move(
        self,
    ) -> tuple[Float32[Array, ' mc_cores'], Float32[Array, ' mc_cores']]:
        """Compute the move rate.

        Returns
        -------
        rate_grow : Float32[Array, 'mc_cores']
            The fraction of times a grow move was proposed and accepted.
        rate_prune : Float32[Array, 'mc_cores']
            The fraction of times a prune move was proposed and accepted.
        """
        agrow, aprune = self.avg_acc()
        pgrow, pprune = self.avg_prop()
        return agrow * pgrow, aprune * pprune

    def depth_distr(self) -> Float32[Array, 'mc_cores ndpost/mc_cores d']:
        """Histogram of tree depths for each state of the trees.

        Returns
        -------
        A matrix where each row contains a histogram of tree depths.
        """
        return chains_depth_distr(self._main_trace.split_tree)

    def points_per_decision_node_distr(
        self,
    ) -> Float32[Array, 'mc_cores ndpost/mc_cores n+1']:
        """Histogram of number of points belonging to parent-of-leaf nodes.

        Returns
        -------
        A matrix where each row contains a histogram of number of points.
        """
        return chains_points_per_decision_node_distr(
            self._main_trace, self._mcmc_state.X
        )

    def points_per_leaf_distr(self) -> Float32[Array, 'mc_cores ndpost/mc_cores n+1']:
        """Histogram of number of points belonging to leaves.

        Returns
        -------
        A matrix where each row contains a histogram of number of points.
        """
        return chains_points_per_leaf_distr(self._main_trace, self._mcmc_state.X)

    def check_trees(self) -> UInt[Array, 'mc_cores ndpost/mc_cores ntree']:
        """Apply `check_trace` to all the tree draws."""
        return check_chains(self._main_trace, self._mcmc_state.forest.max_split)

    def tree_goes_bad(self) -> Bool[Array, 'mc_cores ndpost/mc_cores ntree']:
        """Find iterations where a tree becomes invalid.

        Returns
        -------
        A where (i,j) is `True` if tree j is invalid at iteration i but not i-1.
        """
        bad = self.check_trees().astype(bool)
        bad_before = jnp.pad(bad[:, :-1, :], [(0, 0), (1, 0), (0, 0)])
        return bad & ~bad_before


class debug_gbart(debug_mc_gbart, gbart):
    """A subclass of `gbart` that adds debugging functionality.

    Parameters
    ----------
    *args
        Passed to `gbart`.
    check_trees
        If `True`, check all trees with `check_trace` after running the MCMC,
        and assert that they are all valid. Set to `False` to allow jax tracing.
    **kw
        Passed to `gbart`.
    """
