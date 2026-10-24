# bartz/src/bartz/_jaxext/_linalg.py
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

"""Cholesky and triangular solve, unrolled for small matrices."""

from typing import Literal

from jax import numpy as jnp
from jax.scipy.linalg import (
    solve_triangular as library_solve_triangular,  # noqa: TID251
)
from jaxtyping import Array, Float

# largest matrix size handled by the unrolled implementations; above this, the
# unrolled code is still faster at runtime in some configurations, but it gets
# slow to compile. Empty matrices go to the library too.
UNROLL_MAX_SIZE = 4


def cholesky(a: Float[Array, '*batch k k']) -> Float[Array, '*batch k k']:
    """Lower Cholesky factor, like `jax.numpy.linalg.cholesky`."""
    *_, k, _ = a.shape
    if 0 < k <= UNROLL_MAX_SIZE:
        return cholesky_unrolled(a)
    else:
        return jnp.linalg.cholesky(a)  # noqa: TID251


def solve_triangular(
    a: Float[Array, '*batch_a k k'],
    b: Float[Array, '*batch_b k n'],
    *,
    trans: Literal[0, 1, 'N', 'T'] = 0,
    lower: bool = False,
) -> Float[Array, '*batch k n']:
    """Solve a triangular system, like `jax.scipy.linalg.solve_triangular`."""
    *_, k, _ = a.shape
    if 0 < k <= UNROLL_MAX_SIZE:
        return solve_triangular_unrolled(a, b, trans=trans, lower=lower)
    else:
        # WORKAROUND(jax<0.10.0): the library does not broadcast batch axes
        batch = jnp.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        a = jnp.broadcast_to(a, (*batch, *a.shape[-2:]))
        b = jnp.broadcast_to(b, (*batch, *b.shape[-2:]))
        return library_solve_triangular(a, b, trans=trans, lower=lower)


def cholesky_unrolled(a: Float[Array, '*batch k k']) -> Float[Array, '*batch k k']:
    """Unrolled `cholesky`."""
    *_, k, _ = a.shape
    L = {}
    pivots = []
    for j in range(k):
        pivot = a[..., j, j]
        for m in range(j):
            pivot -= jnp.square(L[j, m])
        pivots.append(pivot)
        L[j, j] = jnp.sqrt(pivot)
        inv_diag = jnp.reciprocal(L[j, j])
        for i in range(j + 1, k):
            # symmetrize like the library
            x = 0.5 * (a[..., i, j] + a[..., j, i])
            for m in range(j):
                x -= L[i, m] * L[j, m]
            L[i, j] = x * inv_diag

    zero = jnp.zeros_like(a[..., 0, 0])
    rows = [jnp.stack([L.get((i, j), zero) for j in range(k)], -1) for i in range(k)]
    out = jnp.stack(rows, -2)
    # like the library, fill the lower triangle with nan if not positive definite
    ok = jnp.all(jnp.stack(pivots) > 0, axis=0)
    keep = ok[..., None, None] | ~jnp.tri(k, dtype=bool)
    return jnp.where(keep, out, jnp.nan)


def solve_triangular_unrolled(
    a: Float[Array, '*batch_a k k'],
    b: Float[Array, '*batch_b k n'],
    *,
    trans: Literal[0, 1, 'N', 'T'] = 0,
    lower: bool = False,
) -> Float[Array, '*batch k n']:
    """Unrolled `solve_triangular`."""
    if trans in (1, 'T'):
        a = a.mT
        lower = not lower
    elif trans not in (0, 'N'):
        msg = f'invalid {trans=}'
        raise ValueError(msg)

    *_, k, _ = a.shape
    order = range(k) if lower else range(k - 1, -1, -1)
    x = {}
    for i in order:
        xi = b[..., i, :]
        for m, xm in x.items():
            xi -= a[..., i, m, None] * xm
        x[i] = xi / a[..., i, i, None]
    return jnp.stack([x[i] for i in range(k)], -2)
