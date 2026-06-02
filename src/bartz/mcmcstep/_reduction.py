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
from jax import lax
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float32, Int32, Integer, Shaped

# target number of datapoint batches on cpu when batching is resolved
# automatically; unlike the gpu target it does not vary across reductions
_AUTO_CPU_TARGET = 16


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
    """Segment-sum with optional batching along the datapoints."""

    num_batches: int | None | Literal['auto'] = field(static=True, default='auto')
    """The number of datapoint batches. If `None`, the reduce is unbatched. If
    'auto', resolved per-platform at run time via `lax.platform_dependent`."""

    auto_gpu_target: int = field(static=True, default=1024)
    """Target number of batches on gpu when `num_batches` is 'auto'."""

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
) -> Shaped[Array, '*batch_shape {size}']:
    batch_shape = values.shape[:-1]
    if num_batches is None:
        out = jnp.zeros((*batch_shape, size), dtype).at[..., indices].add(values)

    else:
        # in the sharded case, n is the size of the local shard, not the full size
        (n,) = indices.shape
        # unsigned avoids a negative-index normalization select in the scatter
        batch_indices = jnp.arange(n, dtype=jnp.uint32) % num_batches
        out = (
            jnp.zeros((*batch_shape, size, num_batches), dtype)
            .at[..., indices, batch_indices]
            .add(values)
            .sum(axis=-1)
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
    """Dense one-hot reduction, intended for small `n`.

    Materializes the membership of each datapoint in its leaf as a one-hot
    matrix over the `size` bins and contracts it against the values. This trades
    the scatter atomics of `BatchedReduction` for a dense ``size``-by-``n``
    pass, which can win when `n` is not much larger than `size`, e.g. on gpu
    with heavily data-sharded high-dimensional problems.
    """

    method: Literal[
        'scatter_set',
        'matmul_n_outer',
        'matmul_n_inner',
        'multiply_n_outer',
        'multiply_n_inner',
    ] = field(static=True, default='multiply_n_inner')
    """How to contract the values against the one-hot leaf-membership matrix:

    'scatter_set'
        Scatter the values into a dense ``size``-by-``n`` buffer with unique
        (non-atomic) writes, then sum over the datapoints.
    'matmul_n_outer'
        Contract the values with the ``n``-by-``size`` one-hot matrix via a dot.
    'matmul_n_inner'
        Same with the transposed ``size``-by-``n`` one-hot, which gets a
        different operand layout for the dot.
    'multiply_n_outer'
        Elementwise-multiply by the ``n``-by-``size`` one-hot and reduce over
        the datapoints; whether the product is fused into the reduction or
        materialized is left to the backend.
    'multiply_n_inner' (default)
        Same with the ``size``-by-``n`` one-hot, reducing along its contiguous
        axis.
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
        values = jnp.asarray(values)
        assert values.ndim == 0 or values.shape[-1:] == indices.shape

        # a scalar value (the count case) weights each datapoint by `values`
        scalar = values.ndim == 0
        (n,) = indices.shape
        bins = jnp.arange(size, dtype=indices.dtype)

        if self.method == 'scatter_set':
            iota = jnp.arange(n)
            # a scalar `values` broadcasts in the scatter, no need to expand it
            out = (
                jnp.zeros((*values.shape[:-1], size, n), dtype)
                .at[..., indices, iota]
                .set(values, unique_indices=True)
                .sum(axis=-1)
            )
        elif self.method == 'matmul_n_outer':
            onehot = (indices[:, None] == bins).astype(dtype)  # (n, size)
            vec = jnp.broadcast_to(values.astype(dtype), (n,)) if scalar else values
            out = jnp.einsum('...n,ns->...s', vec, onehot)
        elif self.method == 'matmul_n_inner':
            onehot = (bins[:, None] == indices).astype(dtype)  # (size, n)
            vec = jnp.broadcast_to(values.astype(dtype), (n,)) if scalar else values
            out = jnp.einsum('...n,sn->...s', vec, onehot)
        elif self.method == 'multiply_n_outer':
            onehot = indices[:, None] == bins  # (n, size)
            if scalar:
                out = values * onehot.sum(axis=-2, dtype=dtype)
            else:
                out = (values[..., :, None] * onehot).sum(axis=-2)
        elif self.method == 'multiply_n_inner':
            onehot = bins[:, None] == indices  # (size, n)
            if scalar:
                out = values * onehot.sum(axis=-1, dtype=dtype)
            else:
                out = (values[..., None, :] * onehot).sum(axis=-1)

        if data_sharded:
            out = lax.psum(out, 'data')
        return out


# default reduction configs for `init`; the per-reduction gpu batch targets were
# tuned on an A4000
DEFAULT_RESID_REDUCTION = BatchedReduction(auto_gpu_target=1024)
DEFAULT_COUNT_REDUCTION = BatchedReduction(auto_gpu_target=2048)
DEFAULT_PREC_REDUCTION = BatchedReduction(auto_gpu_target=1024)
