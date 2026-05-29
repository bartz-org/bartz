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

from jax import ShapeDtypeStruct, jit, lax, shard_map, tree, vmap
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float32, Int32, UInt
from numpy.lib.array_utils import normalize_axis_index

from bartz._jaxext import autobatch
from bartz.grove import TreesTrace, evaluate_forest, var_histogram
from bartz.mcmcloop._trace import MainTrace
from bartz.mcmcstep._state import (
    CHAIN_AXIS,
    chain_vmap_axes,
    chainful_axis,
    get_axis_size,
    partition_specs,
)


@partial(jit, static_argnames=('flatten_chains', 'out_chain_axis_w_trees'))
def evaluate_trace(
    X: UInt[Array, 'p n'],
    trace: MainTrace,
    *,
    flatten_chains: bool = False,
    out_chain_axis_w_trees: int = CHAIN_AXIS,
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

    Returns
    -------
    The predictions for each chain and iteration of the MCMC.
    """
    # per-device memory limit
    max_io_nbytes = 2**27  # 128 MiB

    # the mesh the trace is sharded on (`None` if single-device); the chain
    # axis is parallelized below with a manual `shard_map`
    mesh = trace.mesh

    # extract only the trees from the trace, this will be the input to `evaluate_forest`
    trees = TreesTrace.from_dataclass(trace)
    batched_eval = evaluate_forest  # we will transform `batched_eval`

    # determine batching axes
    trace_chain_axes = chain_vmap_axes(trace)
    # WORKAROUND(python<3.14): use operator.is_none
    is_none = lambda x: x is None
    sample_axes = tree.map(partial(chainful_axis, 0), trace_chain_axes, is_leaf=is_none)
    tree_axes = tree.map(partial(chainful_axis, 1), trace_chain_axes, is_leaf=is_none)

    # stuff to determine output shapes
    # leaf_tree has shape (sample, tree, *k, ts)
    k_axis = chainful_axis(2, trace_chain_axes.leaf_tree)
    is_mv = trace.leaf_tree.ndim > trace.split_tree.ndim
    kshape = trace.leaf_tree.shape[k_axis : k_axis + is_mv]
    _, n = X.shape
    num_samples = trace.leaf_tree.shape[sample_axes.leaf_tree]
    num_trees = trace.leaf_tree.shape[tree_axes.leaf_tree]
    # the chain axis is parallelized with a manual `shard_map` (see below), so
    # the per-device computation only sees `num_chains` divided by the number
    # of chain devices; the data axis is left automatic and follows the
    # sharding of `X`
    if trace.has_chains:
        num_chains = trace.leaf_tree.shape[trace_chain_axes.leaf_tree]
        local_num_chains = num_chains // get_axis_size(mesh, 'chains')

    def expand_shape(shape: tuple[int, ...], chain_axis: int) -> tuple[int, ...]:
        return (*shape[:chain_axis], local_num_chains, *shape[chain_axis:])

    # tree axis to reduce over
    out_shape_w_trees = (num_samples, num_trees, *kshape, n)
    tree_axis = 1
    if trace.has_chains:
        out_chain_axis_w_trees = normalize_axis_index(
            out_chain_axis_w_trees, 1 + len(out_shape_w_trees)
        )
        tree_axis = chainful_axis(tree_axis, out_chain_axis_w_trees)

    # vmap over chains
    if trace.has_chains:
        batched_eval = vmap(
            batched_eval,
            in_axes=(None, TreesTrace.from_dataclass(trace_chain_axes)),
            out_axes=out_chain_axis_w_trees,
        )

    # batch and sum over trees
    batched_eval = autobatch(
        batched_eval,
        max_io_nbytes,
        in_axes=(None, TreesTrace.from_dataclass(tree_axes)),
        out_axes=tree_axis,
        reduce_ufunc=jnp.add,
    )

    # output shape after reducing trees
    out_shape = (num_samples, *kshape, n)
    sample_axis = 0
    if trace.has_chains:
        out_chain_axis = out_chain_axis_w_trees - (out_chain_axis_w_trees > tree_axis)
        sample_axis = chainful_axis(0, out_chain_axis)
        out_shape = expand_shape(out_shape, out_chain_axis)

    # adjust memory limit keeping into account that trees are summed over
    # split_tree has shape (sample, tree, hts)
    hts = trace.split_tree.shape[chainful_axis(2, trace_chain_axes.split_tree)]
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
    max_io_nbytes = max(
        1, math.floor(max_io_nbytes / (1 + core_int_size / core_io_size))
    )

    # batch over mcmc samples
    batched_eval = autobatch(
        batched_eval,
        max_io_nbytes,
        in_axes=(None, TreesTrace.from_dataclass(sample_axes)),
        out_axes=sample_axis,
        warn_on_overflow=False,  # the inner autobatch will handle it
        result_shape_dtype=ShapeDtypeStruct(out_shape, jnp.float32),
    )

    # parallelize the chain axis across devices with a manual `shard_map`
    if mesh is not None and 'chains' in mesh.axis_names and trace.has_chains:
        batched_eval = _shard_map_over_chains(
            batched_eval, trace, mesh, len(out_shape), out_chain_axis
        )

    # prepare offset for broadcasting
    # chainless y shape = (num_samples, *kshape, n), offset shape = kshape
    offset = trace.offset[None, ..., None]
    if trace.has_chains:
        offset = jnp.expand_dims(offset, out_chain_axis)

    # evaluate trees
    y_centered = batched_eval(X, trees)
    y = y_centered + offset
    if flatten_chains and trace.has_chains:
        y = jnp.moveaxis(y, out_chain_axis, 0)
        y = lax.collapse(y, 0, 2)
    return y


def _shard_map_over_chains(
    fun: Callable[[UInt[Array, 'p n'], TreesTrace], Float32[Array, '*out']],
    trace: MainTrace,
    mesh: Mesh,
    out_ndim: int,
    out_chain_axis: int,
) -> Callable[[UInt[Array, 'p n'], TreesTrace], Float32[Array, '*out']]:
    """Wrap ``fun(X, trees)`` in a `shard_map` that is manual over ``'chains'`` only.

    The trees are sharded over their chain axis; `X` is replicated over
    ``'chains'``. The data axis is left as an automatic mesh axis, so the
    output's data axis follows the sharding of `X` (data-sharded for training
    data, replicated for test data) without forcing any inter-device movement.
    """
    spec = [None] * out_ndim
    spec[out_chain_axis] = 'chains'
    trees_spec = TreesTrace.from_dataclass(partition_specs(trace, mesh))
    return shard_map(
        fun,
        mesh=mesh,
        axis_names={'chains'},
        in_specs=(PartitionSpec(), trees_spec),
        out_specs=PartitionSpec(*spec),
        # `traverse_tree`'s `lax.scan` carry starts replicated and becomes
        # chain-varying, which the VMA checker rejects; the evaluation is
        # embarrassingly parallel over chains, so the check adds no safety
        check_vma=False,
    )


@partial(jit, static_argnames=('p', 'out_chain_axis'))
def compute_varcount(
    p: int, trace: MainTrace, *, out_chain_axis: int = CHAIN_AXIS
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
    ) -> Int32[Array, 'samples {p}']:
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
