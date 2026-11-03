# bartz/src/bartz/_jaxext/random/_categorical.py
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

"""Categorical sampler faster than `jax.random.categorical` on cpu."""

from jax import numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int32, Key


def categorical(key: Key[Array, ''], logits: Float[Array, ' n']) -> Int32[Array, '']:
    """Draw like `jax.random.categorical` with 1d `logits`, with the same result."""
    # xla:cpu does not vectorize the argmax reduction, nor any computation of
    # `logits` fused into it, while it vectorizes max and min
    (n,) = logits.shape

    # like argmax, pick the first nan, to propagate it, else the first maximum
    # WORKAROUND(jax<0.12.1): keep nans out of max, as large xla:cpu reductions
    # drop them, https://github.com/jax-ml/jax/issues/41335; with the fix, use
    # `min(where((z == max(z)) | isnan(z), index, n))` (the gumbel noise is
    # finite, so z has nan where logits has)
    is_nan = jnp.isnan(logits)
    z = random.gumbel(key, (n,), logits.dtype) + jnp.where(is_nan, -jnp.inf, logits)
    index = jnp.arange(n)
    priority = jnp.where(is_nan, index, jnp.where(z == jnp.max(z), n + index, 2 * n))
    return jnp.min(priority) % n
