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

from jax import ShapeDtypeStruct, lax, shard_map, tree, vmap
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float, Float32, Int32, UInt

from bartz._jaxext import autobatch, forbid_x64, jit, project
from bartz._typing import kwdict
from bartz.grove import TreesTrace, evaluate_forest, is_multivariate, var_histogram
from bartz.mcmcstep._axes import chain_vmap_axes, trace_sample_axes
from bartz.mcmcstep._state import partition_specs


@runtime_checkable
class EvaluableTrace(Protocol):
    """Structural type of the traces accepted by `evaluate_trace`.

    Both `bartz.mcmcloop.MainTrace` and `bartz.debug.MinimalTrace` satisfy
    it. The runtime check is structural (attribute presence only, not the
    annotated shapes), so it also matches the axis-spec trees `chain_vmap_axes`
    derives from a trace. Implementations must be `equinox.Module` dataclasses
    whose per-sample fields carry ``samples`` axis markers (see
    `bartz.mcmcstep._axes.field`), read by `evaluate_trace` to locate the
    sample axis.
    """

    leaf_tree: (
        Float[Array, '*chains_and_samples num_trees max_num_leaves']
        | Float[Array, '*chains_and_samples num_trees k max_num_leaves']
    )
    var_tree: UInt[Array, '*chains_and_samples num_trees max_num_leaves']
    split_tree: UInt[Array, '*chains_and_samples num_trees max_num_leaves']
    offset: Float32[Array, ''] | Float32[Array, ' k']
    leaf_unit: Float32[Array, ''] | Float32[Array, ' k']

    @property
    def has_chains(self) -> bool: ...

    @property
    def mesh(self) -> Mesh | None: ...


@jit(static_argnames=('flatten_chains', 'test_points', 'max_io_nbytes'))
@forbid_x64
def evaluate_trace(
    X: UInt[Array, 'p n'],
    trace: EvaluableTrace,
    *,
    flatten_chains: bool = False,
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
    test_points
        How to handle the observation (``n``) axis of `X` across devices. The
        sharding of `X` can't be read at trace time, so the caller declares it:

        - ``'none'`` (default): leave `X` alone, neither sharding nor batching
          its ``n`` axis. Safe whatever the sharding of `X` is.
        - ``'autobatch'``: loop over the ``n`` axis to bound memory. Assumes
          `X` is *not* sharded over the mesh ``'data'`` axis; batching a
          sharded axis would serialize the devices.
        - ``'shard_and_autobatch'``: shard the ``n`` axis of `X` over the mesh
          ``'data'`` axis with a manual `jax.shard_map` and batch the per-device
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

    # the tree-sum reduction carry varies over whichever manual axes are active;
    # marking its seed varying (see autobatch's `reduce_vary_axes`) lets the VMA
    # checker pass
    reduce_vary_axes = ('chains',) * shard_chains + ('data',) * shard_data

    # `_evaluate_trace` batches the test-point (`n`) axis itself as an inner
    # loop (see there); when sharding it runs per-device shard inside the
    # `shard_map`, so the chunk it loops over is already the per-device one
    fun = partial(
        _evaluate_trace,
        flatten_chains=flatten_chains,
        batch_test_points=batch_test_points,
        max_io_nbytes=max_io_nbytes,
        reduce_vary_axes=reduce_vary_axes,
    )

    # parallelize chains (and, on request, test points) across devices with a
    # manual `shard_map`; the bulk runs per-device shard, so its sizes come out
    # per-device on their own
    if shard_chains or shard_data:
        fun = _shard_map_eval(
            fun,
            trace,
            mesh,
            shard_chains=shard_chains,
            shard_data=shard_data,
            out_ndim=_output_ndim(trace, flatten_chains),
        )

    return fun(X, trace)


def _evaluate_trace(
    X: UInt[Array, 'p n'],
    trace: EvaluableTrace,
    *,
    flatten_chains: bool,
    batch_test_points: bool,
    max_io_nbytes: int,
    reduce_vary_axes: tuple[str, ...] = (),
) -> Float32[Array, '*trace_shape n'] | Float32[Array, '*trace_shape k n']:
    """Evaluate `trace` on `X` for a single device shard.

    This holds the bulk of `evaluate_trace` and is unaware of the mesh: all
    batch sizes are read from the array shapes, so under the `shard_map` applied
    by `evaluate_trace` they come out per-device automatically.

    If `batch_test_points`, the ``n`` axis of `X` is also looped over (see the
    autobatch nesting below). `max_io_nbytes` bounds each batch's I/O.
    `reduce_vary_axes` is forwarded to the tree-sum reduction so its scan carry
    is cast to vary over those manual axes (the only mesh-aware bit here).
    """
    # extract only the trees from the trace, this will be the input to `evaluate_forest`
    trees = project(TreesTrace, trace)
    batched_eval = evaluate_forest  # we will transform `batched_eval`

    # determine batching axes from the `samples` field markers; fields without
    # a sample axis (e.g., `leaf_unit`) get `None` and are broadcast rather
    # than batched
    trace_chain_axes = chain_vmap_axes(trace)
    sample_axes = trace_sample_axes(trace)
    # WORKAROUND(python<3.14): use operator.is_none
    is_none = lambda x: x is None
    tree_axes = tree.map(
        # the tree axis follows the sample axis
        lambda sample_axis: None if sample_axis is None else sample_axis + 1,
        sample_axes,
        is_leaf=is_none,
    )

    # output axes before the tree reduction, laid out as (*chains, sample, tree, *k, n)
    sample_axis = int(trace.has_chains)
    tree_axis = sample_axis + 1

    # sizes, read from the arrays (hence per-device under the `shard_map`)
    # leaf_tree has shape (*chains, sample, tree, *k, max_num_leaves)
    kshape = trace.leaf_tree.shape[tree_axes.leaf_tree + 1 : -1]
    _, n = X.shape
    num_samples = trace.leaf_tree.shape[sample_axes.leaf_tree]
    num_trees = trace.leaf_tree.shape[tree_axes.leaf_tree]

    # vmap over chains
    if trace.has_chains:
        batched_eval = vmap(
            batched_eval, in_axes=(None, trees.axes_from_dataclass(trace_chain_axes))
        )

    # batch and sum over trees (innermost loop)
    batched_eval = autobatch(
        batched_eval,
        max_io_nbytes,
        in_axes=(None, trees.axes_from_dataclass(tree_axes)),
        out_axes=tree_axis,
        reduce_ufunc=jnp.add,
        reduce_vary_axes=reduce_vary_axes,
    )

    # output shape after reducing trees
    out_shape = (num_samples, *kshape, n)
    if trace.has_chains:
        num_chains, *_ = trace.leaf_tree.shape
        out_shape = (num_chains, *out_shape)

    # adjust memory limit keeping into account that trees are summed over
    max_io_nbytes = _tree_sum_budget(
        max_io_nbytes, trace, kshape=kshape, n=n, num_trees=num_trees
    )

    # batch over the test points (the `n` axis of `X`). It sits outside the tree
    # reduction (so each tree-sum only materializes a chunk of `n`) but inside
    # the sample/chain loops, to mirror the output layout where `n` is the
    # innermost axis. The chunks are concatenated back, no reduction.
    if batch_test_points:
        batched_eval = autobatch(
            batched_eval,
            max_io_nbytes,
            in_axes=(1, None),
            out_axes=-1,
            warn_on_overflow=False,  # the inner autobatch will handle it
        )

    # the outermost loop is the only one whose per-call output is statically the
    # full output, so it alone can take `result_shape_dtype` (skips a trace)
    full_shape: kwdict = dict(
        result_shape_dtype=ShapeDtypeStruct(out_shape, jnp.float32)
    )

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
            warn_on_overflow=False,  # the inner autobatch will handle it
            **full_shape,
        )

    # evaluate trees; `evaluate_forest` applies the leaf scale itself
    y = batched_eval(X, trees) + trace.offset[..., None]
    if flatten_chains and trace.has_chains:
        y = lax.collapse(y, 0, 2)
    return y


def _output_ndim(trace: EvaluableTrace, flatten_chains: bool) -> int:
    """Rank of the `evaluate_trace` output ``(*chains, sample, *k, n)``."""
    has_chain_axis = trace.has_chains and not flatten_chains
    return has_chain_axis + 2 + is_multivariate(trace)


def _tree_sum_budget(
    max_io_nbytes: int,
    trace: EvaluableTrace,
    *,
    kshape: tuple[int, ...],
    n: int,
    num_trees: int,
) -> int:
    """Shrink the I/O budget to leave room for the per-batch tree-sum intermediate.

    The tree-summing `autobatch` materializes an intermediate scaling with the
    number of trees; reducing the budget for the outer (chain/sample) batches
    keeps the total within `max_io_nbytes` (see `autobatch`).
    """
    *_, max_num_leaves = trace.split_tree.shape
    k = math.prod(kshape)
    out_size = k * n * jnp.float32.dtype.itemsize  # the value of the forest
    core_io_size = (
        num_trees
        * max_num_leaves
        * (
            k * trace.leaf_tree.itemsize
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
    out_ndim: int,
) -> Callable[[UInt[Array, 'p n'], EvaluableTrace], Float32[Array, '*out']]:
    """Wrap ``fun(X, trace)`` in a `shard_map` manual over ``'chains'``/``'data'``.

    When `shard_chains`, the trace and the output are sharded over their leading
    chain axis. When `shard_data`, the ``n`` axis of `X` and the output are
    sharded over ``'data'``; the trace, which has no data axis, is replicated.
    Mesh axes not made manual here stay automatic, so the corresponding output
    axes follow whatever sharding the computation produces without forcing any
    inter-device movement.

    `fun` is expected to cast its reduction seeds to varying (see
    `evaluate_trace`) so the VMA check passes.
    """
    axis_names = set()
    x_spec = [None, None]  # (p, n)
    out_spec = [None] * out_ndim
    if shard_chains:
        axis_names.add('chains')
        out_spec[0] = 'chains'
    if shard_data:
        axis_names.add('data')
        x_spec[1] = 'data'
        out_spec[-1] = 'data'

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
    )


@jit(static_argnames='p')
@forbid_x64
def compute_varcount(p: int, trace: EvaluableTrace) -> Int32[Array, '*trace_shape {p}']:
    """
    Count how many times each predictor is used in each MCMC state.

    Parameters
    ----------
    p
        The number of predictors.
    trace
        A main trace of the BART MCMC, as returned by `run_mcmc`.

    Returns
    -------
    Histogram of predictor usage in each MCMC state.
    """
    return var_histogram(p, trace.var_tree, trace.split_tree, sum_batch_axis=-1)
