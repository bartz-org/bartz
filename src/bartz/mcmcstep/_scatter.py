# bartz/src/bartz/mcmcstep/_scatter.py
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

"""Indexed-reduce (scatter-add) core op and its per-platform batch-count heuristic."""

import math
from functools import partial
from typing import Literal

from jax import lax
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float32, Int32, Integer, Shaped


def _scatter_add(
    values: Float32[Array, '*batch_shape n'] | int,
    indices: Integer[Array, ' n'],
    size: int,
    dtype: DTypeLike,
    batch_size: int | None | Literal['auto'],
    which: Literal['resid', 'count', 'prec'],
    data_sharded: bool,
) -> Shaped[Array, '*batch_shape {size}']:
    """Indexed reduce along the last axis of `values`, with optional batching.

    When `batch_size` is 'auto', the number of batches is chosen per-platform at
    run time via `lax.platform_dependent`. When `data_sharded` is True the
    result is psum-reduced across the ``'data'`` axis of the current `shard_map`
    region.
    """
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

    if batch_size != 'auto':
        return impl(batch_size)

    # `n` is the local shard size when data-sharded, which is exactly what the
    # batch-count heuristic wants. Defer the cpu/gpu choice to XLA: it traces
    # both branches but the compiler keeps only the one for the target platform.
    (n,) = indices.shape
    cpu_nb = _auto_num_batches('cpu', n, which)
    gpu_nb = _auto_num_batches('gpu', n, which)
    if cpu_nb == gpu_nb:
        return impl(cpu_nb)
    return lax.platform_dependent(
        default=partial(impl, cpu_nb),
        cuda=partial(impl, gpu_nb),
        rocm=partial(impl, gpu_nb),
    )


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
        batch_indices = jnp.arange(n) % num_batches
        out = (
            jnp.zeros((*batch_shape, size, num_batches), dtype)
            .at[..., indices, batch_indices]
            .add(values)
            .sum(axis=-1)
        )

    if final_psum:
        out = lax.psum(out, 'data')
    return out


def _auto_num_batches(
    platform: Literal['cpu', 'gpu'], n: int, which: Literal['resid', 'count', 'prec']
) -> int | None:
    """Pick the number of datapoint batches for a reduction on a given platform."""
    if platform == 'cpu':
        return _final_round(n, 16)
    else:
        nb = dict(resid=1024, count=2048, prec=1024)[which]  # on an A4000
        return _final_round(n, nb)


def _final_round(n: int, num: float | int) -> int | None:
    """Bound batch size, round number of batches to a power of 2, and disable batching if there's only 1 batch."""
    # at least some elements per batch
    num = min(n // 32, num)

    # round to the nearest power of 2 because I guess XLA and the hardware
    # will like that (not sure about this, maybe just multiple of 32?)
    num = 2 ** round(math.log2(num)) if num else 0

    # disable batching if the batch is as large as the whole dataset
    return num if num > 1 else None
