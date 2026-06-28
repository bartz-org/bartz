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

import jax
from equinox import Module, field
from jax import ShapeDtypeStruct, lax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.extend.backend import backends
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int32, Integer, Shaped, UInt

# target number of datapoint batches on cpu, and minimum datapoints per batch,
# when batching is resolved automatically; unlike the gpu heuristic these are
# flat (the cpu has no SM-count analog to scale with)
_AUTO_CPU_TARGET = 16
_AUTO_CPU_MIN_BATCH = 32

# target number of elements in the per-instance one-hot tile of `PallasReduction`
# when its block size is resolved automatically
_AUTO_PALLAS_TILE = 2**12

# SM count used to trace `AutoBatchedReduction`'s gpu branch when no cuda backend
# is visible. `lax.platform_dependent` traces that branch even with no gpu, only
# to discard it at lowering, so this value never sizes a real gpu's batch grid;
# it just keeps the dead trace valid (a present cuda backend reports its own).
_MOOT_GPU_SM = 1


class ReductionConfig(Module):
    """Select and configure an indexed-reduce (scatter-add) implementation.

    Each concrete subclass identifies a reduction algorithm and carries its
    options. Pass instances to `init` to control how the residuals, counts and
    likelihood precisions are summed over the datapoints in each leaf.
    """

    @abstractmethod
    def _reduce(
        self,
        values: Float[Array, '*batch_shape n'] | int,
        indices: UInt[Array, ' n'],
        /,
        *,
        size: int,
        subset_start: Integer[Array, ''] | None = None,
        subset_length: int | None = None,
        dtype: DTypeLike,
        data_sharded: bool,
        # the output's trailing axis is the number of reduced bins: the range
        # length, or `size` without a subset. jaxtyping evals the `{...}` dim
        # against the arguments but forbids spaces and str-formats arrays, so
        # this indexes a tuple by a bool instead of `... if ... else ...`. The
        # bool reads a zero-length range as no subset, mislabeling the dim, but
        # the only caller passes the nonempty two-element child pair.
    ) -> Shaped[Array, '*batch_shape {(size,subset_length)[bool(subset_length)]}']:
        """Indexed reduce along the last axis of `values`.

        Parameters
        ----------
        values
            The values to sum into bins, or a scalar `int` weighting every
            datapoint equally (used to count the datapoints in each bin).
        indices
            The bin index each datapoint falls into.
        size
            The static excluded upper bound on the values of `indices`, and
            the number of output bins when no subset is given.
        subset_start
            If given (with `subset_length`), reduce only into the contiguous
            bin range ``[subset_start, subset_start + subset_length)``,
            ignoring datapoints whose index falls elsewhere. The range may run
            past `size`; those out-of-domain bins reduce to zero.
        subset_length
            The static length of the bin range, or `None` to reduce into all
            `size` bins.
        dtype
            The dtype of the output and of the accumulation; the values are
            kept in their own, possibly narrower, dtype until accumulated.
        data_sharded
            Whether the data axis is sharded; if true, the result is
            psum-reduced across the ``'data'`` axis of the enclosing
            `shard_map`.

        Returns
        -------
        The per-bin sums, with the same leading dimensions as `values` and the bins on the trailing axis.
        """
        ...


def _resolve_range(
    indices: UInt[Array, ' n'],
    size: int,
    subset_start: Integer[Array, ''] | None,
    subset_length: int | None,
) -> tuple[int, UInt[Array, ' n']]:
    """Reduce the contiguous-range subset to the full case for scatter algorithms.

    Parameters
    ----------
    indices
        The bin index each datapoint falls into, in ``[0, size)``.
    size
        The number of bins.
    subset_start
        The first bin of the range to reduce into, or `None` for all bins.
    subset_length
        The static number of bins in the range, or `None` for all bins.

    Returns
    -------
    out_size : int
        The number of output bins: `subset_length`, or `size` without a subset.
    indices : UInt[Array, ' n']
        The scatter indices into the output bins: unchanged without a subset,
        else each datapoint's offset from `subset_start`, in the indices' own
        unsigned dtype, so that indices outside ``[subset_start, subset_start +
        subset_length)`` land out of bounds, where the scatter drops them.
    """
    if subset_length is None:
        return size, indices
    else:
        # the subtraction is unsigned: indices below `subset_start` underflow to
        # a large value rather than going negative (which the scatter would read
        # as wrap-around indexing), and together with indices ``>= subset_start +
        # subset_length`` they fall outside the output, where scatters drop them.
        # Exact while the range fits the index dtype (``subset_start +
        # subset_length <= 2 ** bits``), as it does for the per-move child pair.
        assert subset_start is not None  # set together with subset_length
        assert jnp.issubdtype(indices.dtype, jnp.unsignedinteger)
        offset = indices - subset_start.astype(indices.dtype)
        return subset_length, offset


class BatchedReduction(ReductionConfig):
    """Segment-sum with optional batching along the datapoints.

    Fastest at the usual tree sizes. See `AutoBatchedReduction` to resolve
    `num_batches` automatically per platform.
    """

    num_batches: int | None = field(static=True, default=None)
    """The number of datapoint batches, or `None` (the default) for an unbatched
    reduce."""

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
        values: Float[Array, '*batch_shape n'] | int,
        indices: UInt[Array, ' n'],
        /,
        *,
        size: int,
        subset_start: Integer[Array, ''] | None = None,
        subset_length: int | None = None,
        dtype: DTypeLike,
        data_sharded: bool,
    ) -> Shaped[Array, '*batch_shape {(size,subset_length)[bool(subset_length)]}']:
        values = jnp.asarray(values)
        assert values.ndim == 0 or values.shape[-1:] == indices.shape
        size, indices = _resolve_range(indices, size, subset_start, subset_length)
        batch_shape = values.shape[:-1]

        if self.num_batches is None:
            out = jnp.zeros((*batch_shape, size), dtype).at[..., indices].add(values)
        else:
            # in the sharded case, n is the size of the local shard, not the full size
            (n,) = indices.shape
            # unsigned avoids a negative-index normalization select in the scatter
            iota = jnp.arange(n, dtype=jnp.uint32)
            if self.contiguous:
                batch_size = -(-n // self.num_batches)  # ceil: last batch is partial
                batch_indices = iota // batch_size
            else:
                batch_indices = iota % self.num_batches
            if self.batches_inner:
                out = (
                    jnp.zeros((*batch_shape, size, self.num_batches), dtype)
                    .at[..., indices, batch_indices]
                    .add(values)
                    .sum(axis=-1)
                )
            else:
                out = (
                    jnp.zeros((*batch_shape, self.num_batches, size), dtype)
                    .at[..., batch_indices, indices]
                    .add(values)
                    .sum(axis=-2)
                )

        if data_sharded:
            out = lax.psum(out, 'data')
        return out


class AutoBatchedReduction(ReductionConfig):
    """`BatchedReduction` that picks `num_batches` automatically per platform.

    A flat target on cpu, and on gpu a count scaling with the SM count and the
    multivariate outcome size. Only cpu and cuda are supported; any other
    platform raises at lowering.
    """

    min_batch_size: float = field(static=True, default=128.0)
    """Minimum datapoints per batch on gpu: caps the batch count at
    ``n / min_batch_size``."""

    beta_sm: float = field(static=True, default=48.0)
    """Batches per streaming multiprocessor on gpu: the batch count saturates at
    ``beta_sm * n_sms * m ** -gamma``, with `n_sms` the gpu's SM count and ``m``
    the multivariate work per datapoint."""

    gamma: float = field(static=True, default=0.4)
    """Exponent by which multivariate outcomes (``m`` values per datapoint) shrink
    the saturation batch count on gpu."""

    batches_inner: bool = field(static=True, default=True)
    """Same as `BatchedReduction`."""

    contiguous: bool = field(static=True, default=False)
    """Same as `BatchedReduction`."""

    def _reduce(
        self,
        values: Float[Array, '*batch_shape n'] | int,
        indices: UInt[Array, ' n'],
        /,
        *,
        size: int,
        subset_start: Integer[Array, ''] | None = None,
        subset_length: int | None = None,
        dtype: DTypeLike,
        data_sharded: bool,
    ) -> Shaped[Array, '*batch_shape {(size,subset_length)[bool(subset_length)]}']:
        # defer the cpu/gpu choice to XLA: both branches are traced, but only the
        # one for the run platform is lowered. With no `default`, an untested
        # platform (rocm, tpu) errors at lowering instead of silently falling back.
        kwargs = dict(
            size=size,
            subset_start=subset_start,
            subset_length=subset_length,
            dtype=dtype,
            data_sharded=data_sharded,
        )
        return lax.platform_dependent(
            cpu=partial(self._reduce_cpu, values, indices, **kwargs),
            cuda=partial(self._reduce_gpu, values, indices, **kwargs),
        )

    def _reduce_cpu(
        self,
        values: Float[Array, '*batch_shape n'] | int,
        indices: UInt[Array, ' n'],
        /,
        *,
        size: int,
        subset_start: Integer[Array, ''] | None = None,
        subset_length: int | None = None,
        dtype: DTypeLike,
        data_sharded: bool,
    ) -> Shaped[Array, '*batch_shape {(size,subset_length)[bool(subset_length)]}']:
        # flat target: the cpu has no SM-count analog to scale the batch count with
        (n,) = indices.shape
        num_batches = _final_round(n, _AUTO_CPU_TARGET, _AUTO_CPU_MIN_BATCH)
        return self._delegate(num_batches)._reduce(  # noqa: SLF001
            values,
            indices,
            size=size,
            subset_start=subset_start,
            subset_length=subset_length,
            dtype=dtype,
            data_sharded=data_sharded,
        )

    def _reduce_gpu(
        self,
        values: Float[Array, '*batch_shape n'] | int,
        indices: UInt[Array, ' n'],
        /,
        *,
        size: int,
        subset_start: Integer[Array, ''] | None = None,
        subset_length: int | None = None,
        dtype: DTypeLike,
        data_sharded: bool,
    ) -> Shaped[Array, '*batch_shape {(size,subset_length)[bool(subset_length)]}']:
        # `n` is the local shard size when data-sharded, which is exactly what the
        # heuristic wants. `m` is the multivariate batch size: the product of the
        # values' leading (non-datapoint) axes (1 for the scalar count case), i.e.
        # how much vector work rides on each scatter slot; clamped to >=1 so a
        # zero-size leading axis (e.g. k=0 outcome components) leaves a moot, empty
        # reduce with the m=1 baseline cap.
        (n,) = indices.shape
        m = max(1, math.prod(jnp.shape(values)[:-1]))
        # the gpu cap scales up with the SM count and down, sublinearly, with the
        # multivariate work per slot
        sm_cap = self.beta_sm * _gpu_sm_count() * m ** (-self.gamma)
        num_batches = _final_round(n, sm_cap, self.min_batch_size)
        return self._delegate(num_batches)._reduce(  # noqa: SLF001
            values,
            indices,
            size=size,
            subset_start=subset_start,
            subset_length=subset_length,
            dtype=dtype,
            data_sharded=data_sharded,
        )

    def _delegate(self, num_batches: int | None) -> BatchedReduction:
        """Build a `BatchedReduction` with the resolved count and this config's layout."""
        return BatchedReduction(
            num_batches=num_batches,
            batches_inner=self.batches_inner,
            contiguous=self.contiguous,
        )


def _final_round(
    n: int, target: float | int, min_batch_size: float | int
) -> int | None:
    """Cap batches to keep them above `min_batch_size`, round to a power of 2, and disable batching if there's only 1 batch."""
    # at least `min_batch_size` elements per batch
    num = min(n / min_batch_size, target)

    # round to the nearest power of 2 because I guess XLA and the hardware
    # will like that (not sure about this, maybe just multiple of 32?)
    num = 2 ** round(math.log2(num)) if num > 0 else 0

    # disable batching if the batch is as large as the whole dataset
    return num if num > 1 else None


def _gpu_sm_count() -> int:
    """Streaming-multiprocessor count shared by the visible cuda gpus.

    Read by `AutoBatchedReduction` to size the gpu batch grid. Since
    `lax.platform_dependent` only runs the gpu branch on cuda, this trusts
    `jax.devices('cuda')` and each device's `core_count` rather than guessing,
    and raises if the gpus report differing counts (a mixed gpu set is
    unsupported).
    """
    if 'cuda' not in backends():
        # no cuda backend: lax.platform_dependent still traces the gpu branch
        # here, only to discard it at lowering, so the count is never used
        return _MOOT_GPU_SM
    counts = {device.core_count for device in jax.devices('cuda')}
    if len(counts) > 1:
        msg = (
            f'visible cuda gpus report differing SM counts {sorted(counts)}; '
            'AutoBatchedReduction assumes a single gpu model'
        )
        raise ValueError(msg)
    (count,) = counts
    return count


class OneHotReduction(ReductionConfig):
    """Dense one-hot reduction.

    Materializes the membership of each datapoint in its leaf as a one-hot
    matrix over the output bins and contracts it against the values. Beats
    `BatchedReduction` only when the number of bins is very small (e.g. a
    single leaf pair), or on gpu for multivariate residuals.
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
        values: Float[Array, '*batch_shape n'] | int,
        indices: UInt[Array, ' n'],
        /,
        *,
        size: int,
        subset_start: Integer[Array, ''] | None = None,
        subset_length: int | None = None,
        dtype: DTypeLike,
        data_sharded: bool,
    ) -> Shaped[Array, '*batch_shape {(size,subset_length)[bool(subset_length)]}']:
        values = jnp.asarray(values)
        assert values.ndim == 0 or values.shape[-1:] == indices.shape

        # a scalar value is the count case, weighting each datapoint by `values`;
        # it broadcasts in the scatter/multiply paths, only matmul needs a vector
        scalar = values.ndim == 0
        (n,) = indices.shape
        batch_shape = values.shape[:-1]
        size, bins, indices = _resolve_range_bins(
            indices,
            size,
            subset_start,
            subset_length,
            remap=self.method == 'scatter_set',
        )
        # unsigned avoids a negative-index normalization select in the scatter
        iota = jnp.arange(n, dtype=jnp.uint32)

        # one-hots and scatter buffers hold the values, so they are built in
        # the values' dtype; only the reduction accumulates in `dtype`. The
        # scalar count case has no input precision to preserve and uses `dtype`.
        values_dtype = dtype if scalar else values.dtype

        match self.method, self.n_inner:
            case 'scatter_set', True:
                out = (
                    jnp.zeros((*batch_shape, size, n), values_dtype)
                    .at[..., indices, iota]
                    .set(values, unique_indices=True)
                    .sum(axis=-1, dtype=dtype)
                )
            case 'scatter_set', False:
                out = (
                    jnp.zeros((*batch_shape, n, size), values_dtype)
                    .at[..., iota, indices]
                    .set(values, unique_indices=True)
                    .sum(axis=-2, dtype=dtype)
                )
            case 'matmul', True:
                onehot = (bins[:, None] == indices).astype(values_dtype)  # (size, n)
                vec = jnp.broadcast_to(values.astype(dtype), (n,)) if scalar else values
                out = jnp.einsum(
                    '...n,sn->...s', vec, onehot, preferred_element_type=dtype
                )
            case 'matmul', False:
                onehot = (indices[:, None] == bins).astype(values_dtype)  # (n, size)
                vec = jnp.broadcast_to(values.astype(dtype), (n,)) if scalar else values
                out = jnp.einsum(
                    '...n,ns->...s', vec, onehot, preferred_element_type=dtype
                )
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


def _resolve_range_bins(
    indices: UInt[Array, ' n'],
    size: int,
    subset_start: Integer[Array, ''] | None,
    subset_length: int | None,
    *,
    remap: bool,
) -> tuple[int, UInt[Array, ' out_size'], UInt[Array, ' n']]:
    """Resolve the range subset into output size, comparison bins, and scatter indices.

    The comparison methods of `OneHotReduction` reduce against the range's bins
    directly, while its scatter method (`remap`) indexes bins by position, so
    the indices are offset like in `_resolve_range`.
    """
    if subset_length is None:
        return size, jnp.arange(size, dtype=indices.dtype), indices
    assert subset_start is not None  # set together with subset_length
    # uint32, not the possibly narrow `indices.dtype`, so bins past `size` do
    # not wrap and alias a real bin in the comparison
    bins = subset_start.astype(jnp.uint32) + jnp.arange(subset_length, dtype=jnp.uint32)
    if remap:
        out_size, indices = _resolve_range(indices, size, subset_start, subset_length)
        return out_size, bins, indices
    else:
        return subset_length, bins, indices


class AutoOneHotReduction(ReductionConfig):
    """`OneHotReduction` that picks `method` and `n_inner` automatically.

    Resolves both knobs from trace-time information per site and platform, then
    delegates to a plain `OneHotReduction`. Uses `matmul` only for wide-bin
    multivariate reductions and `multiply` otherwise; lays the datapoints on the
    outer axis except on the two small-bin sites where the opposite wins (cpu
    precision, cuda count). Those two sites support only cpu and cuda, raising at
    lowering elsewhere.

    The site is recovered from the value: scalar is the count, a wide output the
    residual, a narrow non-scalar output the precision.

    Known limitation: the wide-bin univariate residual on cpu past ~10^6
    datapoints prefers a layout this picks against (up to ~2x slower).
    """

    min_matmul_bins: int = field(static=True, default=8)
    """Minimum output bins for `matmul`; below it `multiply` is always used."""

    def _reduce(
        self,
        values: Float[Array, '*batch_shape n'] | int,
        indices: UInt[Array, ' n'],
        /,
        *,
        size: int,
        subset_start: Integer[Array, ''] | None = None,
        subset_length: int | None = None,
        dtype: DTypeLike,
        data_sharded: bool,
    ) -> Shaped[Array, '*batch_shape {(size,subset_length)[bool(subset_length)]}']:
        out_size = size if subset_length is None else subset_length
        m = max(1, math.prod(jnp.shape(values)[:-1]))
        method = 'matmul' if m >= 2 and out_size >= self.min_matmul_bins else 'multiply'

        if jnp.ndim(values) == 0:  # count
            cpu_inner, cuda_inner = False, True
        elif out_size <= 2:  # precision
            cpu_inner, cuda_inner = True, False
        else:  # residual
            cpu_inner, cuda_inner = False, False

        args = (values, indices)
        kwargs: dict = dict(
            size=size,
            subset_start=subset_start,
            subset_length=subset_length,
            dtype=dtype,
            data_sharded=data_sharded,
        )
        if cpu_inner == cuda_inner:
            # the layout matches on every platform, so no platform split is
            # needed and the reduction also runs on untested platforms (tpu/rocm)
            return OneHotReduction(method=method, n_inner=cpu_inner)._reduce(  # noqa: SLF001
                *args, **kwargs
            )
        else:
            # defer the cpu/gpu choice to XLA: both branches are traced, but only
            # the run platform's is lowered. With no `default`, an untested
            # platform errors at lowering instead of silently falling back.
            return lax.platform_dependent(
                cpu=partial(
                    OneHotReduction(method=method, n_inner=cpu_inner)._reduce,  # noqa: SLF001
                    *args,
                    **kwargs,
                ),
                cuda=partial(
                    OneHotReduction(method=method, n_inner=cuda_inner)._reduce,  # noqa: SLF001
                    *args,
                    **kwargs,
                ),
            )


class PallasReduction(ReductionConfig):
    """Blocked one-hot scatter-add written as a Pallas kernel.

    Splits the datapoints into blocks and, for each block, contracts the values
    against a one-hot leaf-membership matrix held in fast memory, accumulating
    the block partials. Unlike `OneHotReduction`, the one-hot product is
    guaranteed to stay fused (it is never written back to main memory). Targets
    gpu/tpu; on cpu it falls back to Pallas interpret mode, which is slow and
    meant only for testing. Like `OneHotReduction`, it is competitive only when
    the number of output bins is small. Does not support sharding the
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
        values: Float[Array, '*batch_shape n'] | int,
        indices: UInt[Array, ' n'],
        /,
        *,
        size: int,
        subset_start: Integer[Array, ''] | None = None,
        subset_length: int | None = None,
        dtype: DTypeLike,
        data_sharded: bool,
    ) -> Shaped[Array, '*batch_shape {(size,subset_length)[bool(subset_length)]}']:
        if data_sharded:
            # the kernel trips the vma checks of `shard_map`, in jax-version-
            # dependent ways, even in interpret mode
            msg = 'PallasReduction does not support a sharded data axis'
            raise NotImplementedError(msg)

        values = jnp.asarray(values)
        assert values.ndim == 0 or values.shape[-1:] == indices.shape
        (n,) = indices.shape
        num_rows = 1 if values.ndim == 0 else math.prod(values.shape[:-1])
        # the kernel compares the indices against the range's bins, so the
        # subset case costs nothing more than the full one (`bins=None`)
        if subset_length is None:
            out_size = size
            bins = None
        else:
            assert subset_start is not None  # set together with subset_length
            out_size = subset_length
            # uint32 so bins past `size` do not wrap and alias a real bin
            bins = subset_start.astype(jnp.uint32) + jnp.arange(
                subset_length, dtype=jnp.uint32
            )

        # the grid size, tile width and interpret flag are all static; they are
        # resolved from `backend` (cpu vs gpu) rather than the trace-time platform,
        # which need not match the run platform
        interpret = self.backend == 'cpu'
        compiler_params = _resolve_pallas_backend(self.backend)
        if self.block_size == 'auto':
            block_size = _auto_block_size(n, out_size, num_rows)
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
            bins=bins,
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
    else:
        from jax.experimental.pallas import triton as pallas_triton  # noqa: PLC0415

        return pallas_triton.CompilerParams()


def _ceil_pow2(n: int) -> int:
    """Smallest power of 2 >= `n`."""
    return 1 << max(0, n - 1).bit_length()


def _auto_block_size(n: int, size: int, num_rows: int) -> int:
    """Power-of-2 datapoint tile keeping the kernel's one-hot working set small."""
    area = max(1, _AUTO_PALLAS_TILE // (size * num_rows))
    block_size = 1 << round(math.log2(area))  # nearest power of 2
    return min(block_size, _ceil_pow2(n))  # avoid padding beyond `n`


def _pallas_scatter_add(
    values: Float[Array, '*batch_shape n'] | Int32[Array, ''],
    indices: UInt[Array, ' n'],
    /,
    *,
    size: int,
    bins: Integer[Array, ' sub_size'] | None,
    dtype: DTypeLike,
    num_blocks: int,
    block_size: int,
    interpret: bool,
    compiler_params: pl.CompilerParams | None = None,
) -> Shaped[Array, '*batch_shape {getattr(bins,"size",size)}']:
    """Blocked one-hot indexed reduce via a Pallas kernel; see `PallasReduction`.

    `bins` are the bin indices to reduce into, `None` meaning all of ``0, 1,
    ..., size - 1``; the kernel compares the indices against them, so the
    subset case costs nothing more than the full one.
    """
    scalar = values.ndim == 0
    (n,) = indices.shape
    if scalar:
        # a scalar value (the count case) weights every datapoint equally
        batch_shape = ()
        rows = jnp.broadcast_to(values.astype(dtype), (1, n))
    else:
        # the rows stay in the values' dtype; the kernel accumulates in `dtype`
        batch_shape = values.shape[:-1]
        rows = values.reshape(-1, n)
    num_rows, _ = rows.shape
    if bins is None:
        out_size = size
    else:
        (out_size,) = bins.shape

    # the Triton backend requires every array dimension to be a power of 2;
    # `block_size` already is. Pad the rows axis (its length is the product of
    # the value's batch shape, e.g. k or k*k, so any k) with zero rows and the
    # bins axis (the full `size` is a power of 2, but a range subset has any
    # length) with the out-of-domain bin `size`; both pads are sliced off below.
    # `bins` is uint32, so `size` cannot wrap and alias a real bin.
    padded_rows = _ceil_pow2(num_rows)
    padded_size = _ceil_pow2(out_size)
    if bins is not None:
        bins = jnp.pad(bins, (0, padded_size - out_size), constant_values=size)

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
    def kernel(rows_ref, indices_ref, *refs):  # noqa: ANN001, ANN002, ANN202
        if bins is None:
            (out_ref,) = refs
            kernel_bins = jnp.arange(padded_size, dtype=indices_ref.dtype)
        else:
            bins_ref, out_ref = refs
            kernel_bins = bins_ref[:]

        def accumulate(i, acc):  # noqa: ANN001, ANN202
            block = pl.ds(i * block_size, block_size)
            onehot = kernel_bins[:, None] == indices_ref[block]
            # the one-hot product is exact in the values' dtype; the block
            # reduction accumulates in `dtype`
            prod = rows_ref[:, block][:, None, :] * onehot.astype(rows_ref.dtype)
            return acc + prod.sum(axis=-1, dtype=dtype)

        out_ref[0] = lax.fori_loop(
            0, iters, accumulate, jnp.zeros((padded_rows, padded_size), dtype)
        )

    in_specs = [
        pl.BlockSpec((padded_rows, chunk), lambda p: (0, p)),
        pl.BlockSpec((chunk,), lambda p: (p,)),
    ]
    args = [rows, indices]
    if bins is not None:
        # every instance reads the whole (tiny) bins array
        in_specs.append(pl.BlockSpec((padded_size,), lambda _p: (0,)))
        args.append(bins)

    out = pl.pallas_call(
        kernel,
        out_shape=ShapeDtypeStruct((num_blocks, padded_rows, padded_size), dtype),
        grid=(num_blocks,),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((1, padded_rows, padded_size), lambda p: (p, 0, 0)),
        interpret=interpret,
        compiler_params=compiler_params,
        name='scatter_add',
    )(*args)
    # drop the power-of-2 padding rows and bins
    out = out.sum(axis=0)[:num_rows, :out_size]

    if scalar:
        return out.reshape(out_size)
    return out.reshape(*batch_shape, out_size)
