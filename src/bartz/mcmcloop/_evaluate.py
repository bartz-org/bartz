# bartz/src/bartz/mcmcloop/_evaluate.py
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

"""Post-processing of MCMC traces: predictions and predictor usage counts."""

import math
from collections.abc import Callable
from functools import partial
from typing import Literal, Protocol, runtime_checkable

from jax import ShapeDtypeStruct, jit, lax, shard_map, tree, vmap
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float32, Int32, UInt
from numpy.lib.array_utils import normalize_axis_index

from bartz._jaxext import autobatch
from bartz.grove import TreesTrace, evaluate_forest, var_histogram
from bartz.mcmcstep._state import (
    CHAIN_AXIS,
    chain_vmap_axes,
    chainful_axis,
    partition_specs,
)


@runtime_checkable
class EvaluableTrace(Protocol):
    """Structural type of the traces accepted by `evaluate_trace`.

    Both `bartz.mcmcloop.MainTrace` and `bartz.debug.TraceWithOffset` satisfy
    it. The runtime check is structural (attribute presence only, not the
    annotated shapes), so it also matches the axis-spec trees `chain_vmap_axes`
    derives from a trace.
    """

    leaf_tree: (
        Float32[Array, '*chains_and_samples num_trees tree_size']
        | Float32[Array, '*chains_and_samples num_trees k tree_size']
    )
    var_tree: UInt[Array, '*chains_and_samples num_trees tree_size//2']
    split_tree: UInt[Array, '*chains_and_samples num_trees tree_size//2']
    offset: Float32[Array, ''] | Float32[Array, ' k']
    has_chains: bool
    mesh: Mesh | None


@partial(
    jit,
    static_argnames=(
        'flatten_chains',
        'out_chain_axis_w_trees',
        'test_points',
        'max_io_nbytes',
    ),
)
def evaluate_trace(
    X: UInt[Array, 'p n'],
    trace: EvaluableTrace,
    *,
    flatten_chains: bool = False,
    out_chain_axis_w_trees: int = CHAIN_AXIS,
    test_points: Literal['none', 'autobatch', 'shard_and_autobatch'] = 'none',
    max_io_nbytes: int = 2**27,  # 128 MiB, per device
) -> Float32[Array, '*trace_shape n'] | Float32[Array, '*trace_shape k n']:
    """
    Compute predictions for all iterations of the BART MCMC.

    Parameters
    ----------
    X
        The predictors matrix, with `p` predictors and `n` observations.
    trace
        A main trace of the BART MCMC, as returned by `run_mcmc`.
    flatten_chains
        If `True` and `trace` has a chain axis, collapse it into the sample
        axis, so the leading dimension of the output is ``num_chains *
        num_samples`` instead of ``(num_chains, num_samples)``.
    out_chain_axis_w_trees
        Position of the chain axis in the output. Interpreted against the
        intermediate, pre-tree-reduction layout ``(sample, tree, *k, n)``;
        after summing over the tree axis the chain ends up one position
        earlier if it was after the tree axis. Negative values count from the
        end. Ignored when `trace` has no chain axis or `flatten_chains` is
        True.
    test_points
        How to handle the observation (``n``) axis of `X` across devices. The
        sharding of `X` can't be read at trace time, so the caller declares it:

        - ``'none'`` (default): leave `X` alone, neither sharding nor batching
          its ``n`` axis. Safe whatever the sharding of `X` is.
        - ``'autobatch'``: loop over the ``n`` axis to bound memory. Assumes
          `X` is *not* sharded over the mesh ``'data'`` axis; batching a
          sharded axis would serialize the devices.
        - ``'shard_and_autobatch'``: shard the ``n`` axis of `X` over the mesh
          ``'data'`` axis with a manual `shard_map` and batch the per-device
          chunk. Falls back to ``'autobatch'`` if the mesh has no ``'data'``
          axis.
    max_io_nbytes
        Soft limit, in bytes, on the input plus output of each batch of the
        autobatching loops (per device). Lower it to reduce peak memory at the
        cost of more iterations.

    Returns
    -------
    The predictions for each chain and iteration of the MCMC.
    """
    mesh = trace.mesh
    shard_chains = mesh is not None and 'chains' in mesh.axis_names and trace.has_chains
    shard_data = (
        test_points == 'shard_and_autobatch'
        and mesh is not None
        and 'data' in mesh.axis_names
    )
    batch_test_points = test_points in ('autobatch', 'shard_and_autobatch')

    # `_evaluate_trace` batches the test-point (`n`) axis itself as an inner
    # loop (see there); when sharding it runs per-device shard inside the
    # `shard_map`, so the chunk it loops over is already the per-device one
    fun = partial(
        _evaluate_trace,
        flatten_chains=flatten_chains,
        out_chain_axis_w_trees=out_chain_axis_w_trees,
        batch_test_points=batch_test_points,
        max_io_nbytes=max_io_nbytes,
    )

    # parallelize chains (and, on request, test points) across devices with a
    # manual `shard_map`; the bulk runs per-device shard, so its sizes come out
    # per-device on their own
    if shard_chains or shard_data:
        out_ndim, out_chain_axis, n_axis = _output_layout(
            trace, out_chain_axis_w_trees, flatten_chains
        )
        fun = _shard_map_eval(
            fun,
            trace,
            mesh,
            shard_chains=shard_chains,
            shard_data=shard_data,
            out_chain_axis=out_chain_axis,
            n_axis=n_axis,
            out_ndim=out_ndim,
        )

    return fun(X, trace)


def _evaluate_trace(
    X: UInt[Array, 'p n'],
    trace: EvaluableTrace,
    *,
    flatten_chains: bool,
    out_chain_axis_w_trees: int,
    batch_test_points: bool,
    max_io_nbytes: int,
) -> Float32[Array, '*trace_shape n'] | Float32[Array, '*trace_shape k n']:
    """Evaluate `trace` on `X` for a single device shard.

    This holds the bulk of `evaluate_trace` and is unaware of the mesh: all
    batch sizes are read from the array shapes, so under the `shard_map` applied
    by `evaluate_trace` they come out per-device automatically.

    If `batch_test_points`, the ``n`` axis of `X` is also looped over (see the
    autobatch nesting below). `max_io_nbytes` bounds each batch's I/O.
    """
    # extract only the trees from the trace, this will be the input to `evaluate_forest`
    trees = TreesTrace.from_dataclass(trace)
    batched_eval = evaluate_forest  # we will transform `batched_eval`

    # determine batching axes
    trace_chain_axes = chain_vmap_axes(trace)
    # WORKAROUND(python<3.14): use operator.is_none
    is_none = lambda x: x is None
    sample_axes = tree.map(partial(chainful_axis, 0), trace_chain_axes, is_leaf=is_none)
    tree_axes = tree.map(partial(chainful_axis, 1), trace_chain_axes, is_leaf=is_none)

    # output axis positions (size-invariant)
    out_chain_axis_w_trees, tree_axis, out_chain_axis, sample_axis = _output_axes(
        trace, out_chain_axis_w_trees
    )

    # sizes, read from the arrays (hence per-device under the `shard_map`)
    # leaf_tree has shape (sample, tree, *k, ts)
    k_axis = chainful_axis(2, trace_chain_axes.leaf_tree)
    is_mv = trace.leaf_tree.ndim > trace.split_tree.ndim
    kshape = trace.leaf_tree.shape[k_axis : k_axis + is_mv]
    _, n = X.shape
    num_samples = trace.leaf_tree.shape[sample_axes.leaf_tree]
    num_trees = trace.leaf_tree.shape[tree_axes.leaf_tree]

    # vmap over chains
    if trace.has_chains:
        batched_eval = vmap(
            batched_eval,
            in_axes=(None, trees.axes_from_dataclass(trace_chain_axes)),
            out_axes=out_chain_axis_w_trees,
        )

    # batch and sum over trees (innermost loop)
    batched_eval = autobatch(
        batched_eval,
        max_io_nbytes,
        in_axes=(None, trees.axes_from_dataclass(tree_axes)),
        out_axes=tree_axis,
        reduce_ufunc=jnp.add,
    )

    # output shape after reducing trees
    out_shape = (num_samples, *kshape, n)
    if trace.has_chains:
        num_chains = trace.leaf_tree.shape[trace_chain_axes.leaf_tree]
        out_shape = (
            *out_shape[:out_chain_axis],
            num_chains,
            *out_shape[out_chain_axis:],
        )

    # adjust memory limit keeping into account that trees are summed over
    max_io_nbytes = _tree_sum_budget(
        max_io_nbytes, trace, trace_chain_axes, kshape, n, num_trees
    )

    # batch over the test points (the `n` axis of `X`). It sits outside the tree
    # reduction (so each tree-sum only materializes a chunk of `n`) but inside
    # the sample/chain loops, to mirror the output layout where `n` is the
    # innermost axis. The chunks are concatenated back, no reduction.
    if batch_test_points:
        n_axis = chainful_axis(1 + is_mv, out_chain_axis if trace.has_chains else None)
        batched_eval = autobatch(
            batched_eval,
            max_io_nbytes,
            in_axes=(1, None),
            out_axes=n_axis,
            warn_on_overflow=False,  # the inner autobatch will handle it
        )

    # the outermost loop is the only one whose per-call output is statically the
    # full output, so it alone can take `result_shape_dtype` (skips a trace)
    full_shape = dict(result_shape_dtype=ShapeDtypeStruct(out_shape, jnp.float32))

    # batch over mcmc samples
    batched_eval = autobatch(
        batched_eval,
        max_io_nbytes,
        in_axes=(None, trees.axes_from_dataclass(sample_axes)),
        out_axes=sample_axis,
        warn_on_overflow=False,  # the inner autobatch will handle it
        **({} if trace.has_chains else full_shape),
    )

    # batch over chains (outermost loop, mirroring the output layout); the chain
    # axis is manual inside the `shard_map`, so looping over it stays within the
    # per-device shard
    if trace.has_chains:
        batched_eval = autobatch(
            batched_eval,
            max_io_nbytes,
            in_axes=(None, trees.axes_from_dataclass(trace_chain_axes)),
            out_axes=out_chain_axis,
            warn_on_overflow=False,  # the inner autobatch will handle it
            **full_shape,
        )

    # prepare offset for broadcasting
    # chainless y shape = (num_samples, *kshape, n), offset shape = kshape
    offset = trace.offset[None, ..., None]
    if trace.has_chains:
        offset = jnp.expand_dims(offset, out_chain_axis)

    # evaluate trees
    y = batched_eval(X, trees) + offset
    if flatten_chains and trace.has_chains:
        y = jnp.moveaxis(y, out_chain_axis, 0)
        y = lax.collapse(y, 0, 2)
    return y


def _output_axes(
    trace: EvaluableTrace, out_chain_axis_w_trees: int
) -> tuple[int, int, int, int]:
    """Axis positions in the output layout, derived from array ranks only.

    The positions are independent of the array sizes (hence of the sharding),
    so `evaluate_trace` can build the `shard_map` output spec from them.

    Parameters
    ----------
    trace
        A main trace of the BART MCMC.
    out_chain_axis_w_trees
        Requested chain position in the pre-tree-reduction layout.

    Returns
    -------
    out_chain_axis_w_trees : int
        Chain position in the pre-tree-reduction layout ``(sample, tree, *k, n)``.
    tree_axis : int
        Position of the tree axis to reduce over.
    out_chain_axis : int
        Chain position after the tree axis is summed away.
    sample_axis : int
        Position of the sample axis in the final output.
    """
    tree_axis = 1
    out_chain_axis = sample_axis = 0
    if trace.has_chains:
        # pre-tree-reduction output ndim: chain + (sample, tree, *k, n)
        is_mv = trace.leaf_tree.ndim > trace.split_tree.ndim
        out_chain_axis_w_trees = normalize_axis_index(out_chain_axis_w_trees, 4 + is_mv)
        tree_axis = chainful_axis(tree_axis, out_chain_axis_w_trees)
        out_chain_axis = out_chain_axis_w_trees - (out_chain_axis_w_trees > tree_axis)
        sample_axis = chainful_axis(0, out_chain_axis)
    return out_chain_axis_w_trees, tree_axis, out_chain_axis, sample_axis


def _output_layout(
    trace: EvaluableTrace, out_chain_axis_w_trees: int, flatten_chains: bool
) -> tuple[int, int | None, int]:
    """Rank and axis positions of the `evaluate_trace` output.

    Like `_output_axes`, but for the final (post-tree-reduction) layout and
    accounting for `flatten_chains`. Used to build the `shard_map` specs and
    the test-point autobatch.

    Parameters
    ----------
    trace
        A main trace of the BART MCMC.
    out_chain_axis_w_trees
        Requested chain position in the pre-tree-reduction layout.
    flatten_chains
        Whether the chain axis is folded into the leading sample axis.

    Returns
    -------
    out_ndim : int
        The number of axes of the output.
    out_chain_axis : int | None
        Position of the chain axis, or `None` if `trace` has no chain axis.
    n_axis : int
        Position of the observation (``n``) axis.
    """
    is_mv = trace.leaf_tree.ndim > trace.split_tree.ndim
    chainless_ndim = 2 + is_mv  # (sample, *k, n)
    n_core = chainless_ndim - 1  # `n` is the last chainless axis
    if trace.has_chains and not flatten_chains:
        out_ndim = chainless_ndim + 1
        *_, out_chain_axis, _ = _output_axes(trace, out_chain_axis_w_trees)
        # `n` shifts only if the chain was inserted at or before it
        n_axis = chainful_axis(n_core, out_chain_axis)
    else:
        out_ndim = chainless_ndim
        # when flattening, chains are folded into the leading (sample) axis
        out_chain_axis = 0 if trace.has_chains else None
        n_axis = n_core
    return out_ndim, out_chain_axis, n_axis


def _tree_sum_budget(
    max_io_nbytes: int,
    trace: EvaluableTrace,
    chain_axes: EvaluableTrace,
    kshape: tuple[int, ...],
    n: int,
    num_trees: int,
) -> int:
    """Shrink the I/O budget to leave room for the per-batch tree-sum intermediate.

    The tree-summing `autobatch` materializes an intermediate scaling with the
    number of trees; reducing the budget for the outer (chain/sample) batches
    keeps the total within `max_io_nbytes` (see `autobatch`).
    """
    # split_tree has shape (sample, tree, hts)
    hts = trace.split_tree.shape[chainful_axis(2, chain_axes.split_tree)]
    k = math.prod(kshape)
    out_size = k * n * jnp.float32.dtype.itemsize  # the value of the forest
    core_io_size = (
        num_trees
        * hts
        * (
            2 * k * trace.leaf_tree.itemsize
            + trace.var_tree.itemsize
            + trace.split_tree.itemsize
        )
        + out_size
    )
    core_int_size = (num_trees - 1) * out_size
    return max(1, math.floor(max_io_nbytes / (1 + core_int_size / core_io_size)))


def _shard_map_eval(
    fun: Callable[[UInt[Array, 'p n'], EvaluableTrace], Float32[Array, '*out']],
    trace: EvaluableTrace,
    mesh: Mesh,
    *,
    shard_chains: bool,
    shard_data: bool,
    out_chain_axis: int | None,
    n_axis: int,
    out_ndim: int,
) -> Callable[[UInt[Array, 'p n'], EvaluableTrace], Float32[Array, '*out']]:
    """Wrap ``fun(X, trace)`` in a `shard_map` manual over ``'chains'``/``'data'``.

    When `shard_chains`, the trace is sharded over its chain axis and the
    output over `out_chain_axis`. When `shard_data`, the ``n`` axis of `X` and
    the output are sharded over ``'data'``; the trace, which has no data axis,
    is replicated. Mesh axes not made manual here stay automatic, so the
    corresponding output axes follow whatever sharding the computation produces
    without forcing any inter-device movement.
    """
    axis_names = set()
    x_spec = [None, None]  # (p, n)
    out_spec = [None] * out_ndim
    if shard_chains:
        axis_names.add('chains')
        out_spec[out_chain_axis] = 'chains'
    if shard_data:
        axis_names.add('data')
        x_spec[1] = 'data'
        out_spec[n_axis] = 'data'

    # drop trailing Nones to match jax's canonical specs (eases sharding
    # comparisons during debugging)
    for spec in (x_spec, out_spec):
        while spec and spec[-1] is None:
            spec.pop()

    return shard_map(
        fun,
        mesh=mesh,
        axis_names=axis_names,
        in_specs=(PartitionSpec(*x_spec), partition_specs(trace, mesh)),
        out_specs=PartitionSpec(*out_spec),
        # `traverse_tree`'s `lax.scan` carry starts replicated and becomes
        # chain/data-varying, which the VMA checker rejects; the evaluation is
        # embarrassingly parallel over chains and points, so the check adds no
        # safety
        check_vma=False,
    )


@partial(jit, static_argnames=('p', 'out_chain_axis'))
def compute_varcount(
    p: int, trace: EvaluableTrace, *, out_chain_axis: int = CHAIN_AXIS
) -> Int32[Array, '*trace_shape {p}']:
    """
    Count how many times each predictor is used in each MCMC state.

    Parameters
    ----------
    p
        The number of predictors.
    trace
        A main trace of the BART MCMC, as returned by `run_mcmc`.
    out_chain_axis
        Position of the chain axis in the output, whose chainless shape is
        ``(samples, p)``. Negative values count from the end. Ignored when
        `trace` has no chain axis.

    Returns
    -------
    Histogram of predictor usage in each MCMC state.
    """
    chain_axes = chain_vmap_axes(trace)

    def histogram(
        var_tree: UInt[Array, 'samples trees nodes'],
        split_tree: UInt[Array, 'samples trees nodes'],
    ) -> Int32[Array, 'samples p']:
        return var_histogram(p, var_tree, split_tree, sum_batch_axis=-1)

    if trace.has_chains:
        # chainless output ndim is 2 (samples, p); the chain adds one
        out_axis = normalize_axis_index(out_chain_axis, 3)
        histogram = vmap(
            histogram,
            in_axes=(chain_axes.var_tree, chain_axes.split_tree),
            out_axes=out_axis,
        )

    return histogram(trace.var_tree, trace.split_tree)
