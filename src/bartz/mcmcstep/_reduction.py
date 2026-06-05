# bartz/src/bartz/mcmcstep/_reduction.py
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

"""Indexed-reduce (scatter-add) configs, one per algorithm, and the core ops."""

import math
from abc import abstractmethod
from functools import partial
from typing import Literal

from equinox import Module, field
from jax import ShapeDtypeStruct, lax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.typing import DTypeLike
from jaxtyping import Array, Float32, Int32, Integer, Shaped

# target number of datapoint batches on cpu when batching is resolved
# automatically; unlike the gpu target it does not vary across reductions
_AUTO_CPU_TARGET = 16

# target number of elements in the per-instance one-hot tile of `PallasReduction`
# when its block size is resolved automatically
_AUTO_PALLAS_TILE = 2**12


class ReductionConfig(Module):
    """Select and configure an indexed-reduce (scatter-add) implementation.

    Each concrete subclass identifies a reduction algorithm and carries its
    options. Pass instances to `init` to control how the residuals, counts and
    likelihood precisions are summed over the datapoints in each leaf.
    """

    @abstractmethod
    def _reduce(
        self,
        values: Float32[Array, '*batch_shape n'] | int,
        indices: Integer[Array, ' n'],
        /,
        *,
        size: int,
        dtype: DTypeLike,
        data_sharded: bool,
    ) -> Shaped[Array, '*batch_shape {size}']:
        """Indexed reduce along the last axis of `values` into `size` bins."""
        ...


class BatchedReduction(ReductionConfig):
    """Segment-sum with optional batching along the datapoints.

    The default; fastest at the usual tree sizes.
    """

    num_batches: int | None | Literal['auto'] = field(static=True, default='auto')
    """The number of datapoint batches. If `None`, the reduce is unbatched. If
    'auto', resolved per-platform at run time via `jax.lax.platform_dependent`."""

    auto_gpu_target: int = field(static=True, default=1024)
    """Target number of batches on gpu when `num_batches` is 'auto'."""

    batches_inner: bool = field(static=True, default=True)
    """Whether the batch axis sits on the scatter buffer's inner, contiguous axis
    (``size``-by-``num_batches``) or its outer axis (``num_batches``-by-``size``);
    the two layouts give the backend different memory access patterns. `True` (the
    default) matches the historical layout. No effect when `num_batches` is `None`."""

    contiguous: bool = field(static=True, default=False)
    """How datapoints are assigned to batches. `False` (the default) strides them,
    sending datapoint ``i`` to batch ``i % num_batches``; `True` splits them into
    contiguous chunks, sending ``i`` to batch ``i // batch_size``. No effect when
    `num_batches` is `None`."""

    def _reduce(
        self,
        values: Float32[Array, '*batch_shape n'] | int,
        indices: Integer[Array, ' n'],
        /,
        *,
        size: int,
        dtype: DTypeLike,
        data_sharded: bool,
    ) -> Shaped[Array, '*batch_shape {size}']:
        values = jnp.asarray(values)
        assert values.ndim == 0 or values.shape[-1:] == indices.shape

        def impl(num_batches: int | None) -> Shaped[Array, '*batch_shape size']:
            return _batched_scatter_add(
                values,
                indices,
                size=size,
                dtype=dtype,
                num_batches=num_batches,
                final_psum=data_sharded,
                batches_inner=self.batches_inner,
                contiguous=self.contiguous,
            )

        if self.num_batches != 'auto':
            return impl(self.num_batches)

        # `n` is the local shard size when data-sharded, which is exactly what
        # the batch-count heuristic wants. Defer the cpu/gpu choice to XLA: it
        # traces both branches but the compiler keeps only the target one.
        (n,) = indices.shape
        cpu_nb = _final_round(n, _AUTO_CPU_TARGET)
        gpu_nb = _final_round(n, self.auto_gpu_target)
        if cpu_nb == gpu_nb:
            return impl(cpu_nb)
        return lax.platform_dependent(
            default=partial(impl, cpu_nb),
            cuda=partial(impl, gpu_nb),
            rocm=partial(impl, gpu_nb),
        )


def _batched_scatter_add(
    values: Float32[Array, '*batch_shape n'] | Int32[Array, ''],
    indices: Integer[Array, ' n'],
    /,
    *,
    size: int,
    dtype: DTypeLike,
    num_batches: int | None,
    final_psum: bool = False,
    batches_inner: bool,
    contiguous: bool,
) -> Shaped[Array, '*batch_shape {size}']:
    batch_shape = values.shape[:-1]
    if num_batches is None:
        out = jnp.zeros((*batch_shape, size), dtype).at[..., indices].add(values)

    else:
        # in the sharded case, n is the size of the local shard, not the full size
        (n,) = indices.shape
        # unsigned avoids a negative-index normalization select in the scatter
        iota = jnp.arange(n, dtype=jnp.uint32)
        if contiguous:
            batch_size = -(-n // num_batches)  # ceil, so the last batch is partial
            batch_indices = iota // batch_size
        else:
            batch_indices = iota % num_batches
        if batches_inner:
            out = (
                jnp.zeros((*batch_shape, size, num_batches), dtype)
                .at[..., indices, batch_indices]
                .add(values)
                .sum(axis=-1)
            )
        else:
            out = (
                jnp.zeros((*batch_shape, num_batches, size), dtype)
                .at[..., batch_indices, indices]
                .add(values)
                .sum(axis=-2)
            )

    if final_psum:
        out = lax.psum(out, 'data')
    return out


def _final_round(n: int, num: float | int) -> int | None:
    """Bound batch size, round number of batches to a power of 2, and disable batching if there's only 1 batch."""
    # at least some elements per batch
    num = min(n // 32, num)

    # round to the nearest power of 2 because I guess XLA and the hardware
    # will like that (not sure about this, maybe just multiple of 32?)
    num = 2 ** round(math.log2(num)) if num else 0

    # disable batching if the batch is as large as the whole dataset
    return num if num > 1 else None


class OneHotReduction(ReductionConfig):
    """Dense one-hot reduction.

    Materializes the membership of each datapoint in its leaf as a one-hot
    matrix over the `size` bins and contracts it against the values. Beats
    `BatchedReduction` only when `size` is very small (e.g. a single leaf pair),
    or on gpu for multivariate residuals.
    """

    method: Literal['matmul', 'multiply', 'scatter_set'] = field(
        static=True, default='matmul'
    )
    """How to contract the values against the one-hot leaf-membership matrix:

    'matmul'
        Contract the values with the one-hot matrix via a dot. Faster on gpu,
        especially for multivariate residuals.
    'multiply'
        Elementwise-multiply by the one-hot matrix and reduce over the
        datapoints; whether the ``n``-by-``size`` product is fused into the
        reduction or materialized is left to the backend. Faster on cpu.
    'scatter_set'
        Scatter the values into a dense buffer with unique (non-atomic) writes,
        then sum over the datapoints.
    """

    n_inner: bool = field(static=True, default=True)
    """Whether the datapoints sit on the one-hot's inner, contiguous axis
    (``size``-by-``n``) or its outer axis (``n``-by-``size``); the two layouts
    give the backend different memory access patterns. `True` (the default)
    fuses better on gpu."""

    def _reduce(
        self,
        values: Float32[Array, '*batch_shape n'] | int,
        indices: Integer[Array, ' n'],
        /,
        *,
        size: int,
        dtype: DTypeLike,
        data_sharded: bool,
    ) -> Shaped[Array, '*batch_shape {size}']:
        values = jnp.asarray(values)
        assert values.ndim == 0 or values.shape[-1:] == indices.shape

        # a scalar value is the count case, weighting each datapoint by `values`;
        # it broadcasts in the scatter/multiply paths, only matmul needs a vector
        scalar = values.ndim == 0
        (n,) = indices.shape
        batch_shape = values.shape[:-1]
        bins = jnp.arange(size, dtype=indices.dtype)
        iota = jnp.arange(n)

        match self.method, self.n_inner:
            case 'scatter_set', True:
                out = (
                    jnp.zeros((*batch_shape, size, n), dtype)
                    .at[..., indices, iota]
                    .set(values, unique_indices=True)
                    .sum(axis=-1)
                )
            case 'scatter_set', False:
                out = (
                    jnp.zeros((*batch_shape, n, size), dtype)
                    .at[..., iota, indices]
                    .set(values, unique_indices=True)
                    .sum(axis=-2)
                )
            case 'matmul', True:
                onehot = (bins[:, None] == indices).astype(dtype)  # (size, n)
                vec = jnp.broadcast_to(values.astype(dtype), (n,)) if scalar else values
                out = jnp.einsum('...n,sn->...s', vec, onehot)
            case 'matmul', False:
                onehot = (indices[:, None] == bins).astype(dtype)  # (n, size)
                vec = jnp.broadcast_to(values.astype(dtype), (n,)) if scalar else values
                out = jnp.einsum('...n,ns->...s', vec, onehot)
            case 'multiply', True:
                onehot = bins[:, None] == indices  # (size, n)
                if scalar:
                    out = values * onehot.sum(axis=-1, dtype=dtype)
                else:
                    out = (values[..., None, :] * onehot).sum(axis=-1, dtype=dtype)
            case 'multiply', False:
                onehot = indices[:, None] == bins  # (n, size)
                if scalar:
                    out = values * onehot.sum(axis=-2, dtype=dtype)
                else:
                    out = (values[..., :, None] * onehot).sum(axis=-2, dtype=dtype)

        if data_sharded:
            out = lax.psum(out, 'data')
        return out


class PallasReduction(ReductionConfig):
    """Blocked one-hot scatter-add written as a Pallas kernel.

    Splits the datapoints into blocks and, for each block, contracts the values
    against a one-hot leaf-membership matrix held in fast memory, accumulating
    the block partials. Unlike `OneHotReduction`, the one-hot product is
    guaranteed to stay fused (it is never written back to main memory). Targets
    gpu/tpu; on cpu it falls back to Pallas interpret mode, which is slow and
    meant only for testing. Like `OneHotReduction`, it is competitive only when
    `size` (the number of leaves) is small. Does not support sharding the
    datapoints across devices.

    On gpu the kernel is lowered through Triton or Mosaic GPU; see `backend`.
    """

    block_size: int | Literal['auto'] = field(static=True, default='auto')
    """Datapoints contracted per kernel iteration, i.e., the width of the one-hot
    tile in fast memory. If 'auto', chosen to keep that tile small. Should be a
    power of 2 on gpu."""

    num_blocks: int | Literal['auto'] = field(static=True, default='auto')
    """Number of kernel instances (grid size) the datapoints are split across,
    each looping over its share. More instances raise occupancy but enlarge the
    partial-sum buffer. If 'auto', resolved per-platform at trace time."""

    auto_gpu_target: int = field(static=True, default=1024)
    """Cap on the number of kernel instances on gpu when `num_blocks` is 'auto'."""

    backend: Literal['triton', 'cpu', 'default'] = field(static=True, default='triton')
    """How to lower the kernel. The run platform is not known here at trace time,
    so it cannot be selected automatically:

    'triton'
        Pass Triton compiler params; the default, compiles on every CUDA/ROCm gpu.
    'cpu'
        Pallas interpret mode, the only mode that runs on cpu (slow; for testing).
    'default'
        Pass nothing, leaving jax to pick its own gpu backend: that is Mosaic GPU,
        which only compiles on Hopper and newer (compute capability 9.0+).
    """

    def _reduce(
        self,
        values: Float32[Array, '*batch_shape n'] | int,
        indices: Integer[Array, ' n'],
        /,
        *,
        size: int,
        dtype: DTypeLike,
        data_sharded: bool,
    ) -> Shaped[Array, '*batch_shape {size}']:
        if data_sharded:
            # the kernel trips the vma checks of `shard_map`, in jax-version-
            # dependent ways, even in interpret mode
            msg = 'PallasReduction does not support a sharded data axis'
            raise NotImplementedError(msg)

        values = jnp.asarray(values)
        assert values.ndim == 0 or values.shape[-1:] == indices.shape
        (n,) = indices.shape
        num_rows = 1 if values.ndim == 0 else math.prod(values.shape[:-1])

        # the grid size, tile width and interpret flag are all static; they are
        # resolved from `backend` (cpu vs gpu) rather than the trace-time platform,
        # which need not match the run platform
        interpret = self.backend == 'cpu'
        compiler_params = _resolve_pallas_backend(self.backend)
        if self.block_size == 'auto':
            block_size = _auto_block_size(n, size, num_rows)
        else:
            block_size = self.block_size
        if self.num_blocks == 'auto':
            target = _AUTO_CPU_TARGET if interpret else self.auto_gpu_target
            num_blocks = max(1, min(-(-n // block_size), target))
        else:
            num_blocks = self.num_blocks

        return _pallas_scatter_add(
            values,
            indices,
            size=size,
            dtype=dtype,
            num_blocks=num_blocks,
            block_size=block_size,
            interpret=interpret,
            compiler_params=compiler_params,
        )


def _resolve_pallas_backend(
    backend: Literal['triton', 'cpu', 'default'],
) -> pl.CompilerParams | None:
    """`compiler_params` for `pallas_call`, or `None` to use its own default.

    Only 'triton' passes params (it compiles on every CUDA/ROCm gpu); 'cpu'
    (interpret mode) and 'default' (Mosaic GPU, Hopper-only) pass nothing.
    """
    if backend != 'triton':
        return None
    # WORKAROUND(jax<0.7.0): the public Triton compiler params live at
    # `pallas.triton.CompilerParams` since jax 0.7 (earlier the class has
    # another name); older jax already defaulted its gpu Pallas backend to
    # Triton, so passing nothing is equivalent.
    try:
        from jax.experimental.pallas import triton as pallas_triton  # noqa: PLC0415

        cls = pallas_triton.CompilerParams
    except (ImportError, AttributeError):
        return None
    return cls()


def _auto_block_size(n: int, size: int, num_rows: int) -> int:
    """Power-of-2 datapoint tile keeping the kernel's one-hot working set small."""
    area = max(1, _AUTO_PALLAS_TILE // (size * num_rows))
    block_size = 1 << round(math.log2(area))  # nearest power of 2
    ceil_pow2_n = 1 << (max(1, n) - 1).bit_length()  # avoid padding beyond `n`
    return min(block_size, ceil_pow2_n)


def _pallas_scatter_add(
    values: Float32[Array, '*batch_shape n'] | Int32[Array, ''],
    indices: Integer[Array, ' n'],
    /,
    *,
    size: int,
    dtype: DTypeLike,
    num_blocks: int,
    block_size: int,
    interpret: bool,
    compiler_params: pl.CompilerParams | None = None,
) -> Shaped[Array, '*batch_shape {size}']:
    """Blocked one-hot indexed reduce via a Pallas kernel; see `PallasReduction`."""
    scalar = values.ndim == 0
    (n,) = indices.shape
    if scalar:
        # a scalar value (the count case) weights every datapoint equally
        batch_shape = ()
        rows = jnp.broadcast_to(values.astype(dtype), (1, n))
    else:
        batch_shape = values.shape[:-1]
        rows = values.astype(dtype).reshape(-1, n)
    num_rows, _ = rows.shape

    # the Triton backend requires every array dimension to be a power of 2. `size`
    # (the tree array size) and `block_size` already are; pad the rows axis (its
    # length is the product of the value's batch shape, e.g. k or k*k, so any k)
    # to a power of 2 with zero rows, whose zero output is sliced off below.
    padded_rows = 1 << (max(1, num_rows) - 1).bit_length()

    # each instance scans `iters` sub-blocks of `block_size` datapoints; pad the
    # datapoint axis so it splits evenly. The padded datapoints are zero in every
    # row, so they contribute nothing whatever bin they fall in; the out-of-range
    # `size` index is just a tidy default (it may wrap in a narrow index dtype).
    iters = -(-n // (num_blocks * block_size))
    chunk = iters * block_size
    pad = num_blocks * chunk - n
    indices = jnp.pad(indices, (0, pad), constant_values=size)
    rows = jnp.pad(rows, ((0, padded_rows - num_rows), (0, pad)))

    # the kernel operates on `Ref`s, not arrays, so it carries no array
    # annotations (which would also trip runtime shape typechecking)
    def kernel(rows_ref, indices_ref, out_ref):  # noqa: ANN001, ANN202
        bins = jnp.arange(size, dtype=indices_ref.dtype)

        def accumulate(i, acc):  # noqa: ANN001, ANN202
            block = pl.ds(i * block_size, block_size)
            onehot = (bins[:, None] == indices_ref[block]).astype(dtype)
            return acc + (rows_ref[:, block][:, None, :] * onehot).sum(axis=-1)

        out_ref[0] = lax.fori_loop(
            0, iters, accumulate, jnp.zeros((padded_rows, size), dtype)
        )

    out = pl.pallas_call(
        kernel,
        out_shape=ShapeDtypeStruct((num_blocks, padded_rows, size), dtype),
        grid=(num_blocks,),
        in_specs=[
            pl.BlockSpec((padded_rows, chunk), lambda p: (0, p)),
            pl.BlockSpec((chunk,), lambda p: (p,)),
        ],
        out_specs=pl.BlockSpec((1, padded_rows, size), lambda p: (p, 0, 0)),
        interpret=interpret,
        compiler_params=compiler_params,
        name='scatter_add',
    )(rows, indices)
    out = out.sum(axis=0)[:num_rows]  # drop the power-of-2 padding rows

    if scalar:
        return out.reshape(size)
    return out.reshape(*batch_shape, size)
