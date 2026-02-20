# bartz/src/bartz/debug/_traceconv.py
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

"""Debugging utilities. The main functionality is the class `debug_mc_gbart`."""

from math import ceil, log2
from re import fullmatch

import numpy
from equinox import Module, field
from jax import numpy as jnp
from jaxtyping import Array, Float32, UInt

from bartz.BART._gbart import FloatLike
from bartz.grove import TreeHeaps
from bartz.jaxext import minimal_unsigned_dtype


def _get_next_line(s: str, i: int) -> tuple[str, int]:
    """Get the next line from a string and the new index."""
    i_new = s.find('\n', i)
    if i_new == -1:
        return s[i:], len(s)
    return s[i:i_new], i_new + 1


class BARTTraceMeta(Module):
    """Metadata of R BART tree traces."""

    ndpost: int = field(static=True)
    """The number of posterior draws."""

    ntree: int = field(static=True)
    """The number of trees in the model."""

    numcut: UInt[Array, ' p']
    """The maximum split value for each variable."""

    heap_size: int = field(static=True)
    """The size of the heap required to store the trees."""


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
