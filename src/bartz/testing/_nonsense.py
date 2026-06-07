# bartz/src/bartz/testing/_nonsense.py
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


"""Define `gen_nonsense_data`, a cheap synthetic data generator for benchmarks."""

from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, Float32, UInt8

from bartz._jaxext import jit


@jit(static_argnums=(0, 1, 2))
def gen_nonsense_data(
    p: int, n: int, k: int | None
) -> tuple[
    UInt8[Array, '{p} {n}'],
    Float32[Array, ' {n}'] | Float32[Array, '{k} {n}'],
    UInt8[Array, ' {p}'],
]:
    """Generate pretty nonsensical data."""
    X = jnp.arange(p * n, dtype=jnp.uint8).reshape(p, n)
    X = vmap(jnp.roll)(X, jnp.arange(p))
    max_split = jnp.full(p, 255, jnp.uint8)
    shift = 0 if k is None else jnp.linspace(0, 2 * jnp.pi, k, endpoint=False)[:, None]
    y = jnp.cos(jnp.linspace(0, 2 * jnp.pi / 32 * n, n) + shift)
    return X, y, max_split
