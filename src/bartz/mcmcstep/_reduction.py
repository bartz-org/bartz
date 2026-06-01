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
            return _scatter_add_impl(
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


# default reduction configs for `init`; the per-reduction gpu batch targets were
# tuned on an A4000
DEFAULT_RESID_REDUCTION = BatchedReduction(auto_gpu_target=1024)
DEFAULT_COUNT_REDUCTION = BatchedReduction(auto_gpu_target=2048)
DEFAULT_PREC_REDUCTION = BatchedReduction(auto_gpu_target=1024)


def _scatter_add_impl(
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
