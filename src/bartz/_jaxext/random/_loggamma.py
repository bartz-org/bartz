# bartz/src/bartz/_jaxext/random/_loggamma.py
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

"""Sampler for the log of gamma random variables."""

from collections.abc import Sequence
from functools import partial

from jax import jit, lax, random
from jax import numpy as jnp
from jax.dtypes import canonicalize_dtype
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Key


@partial(jit, static_argnums=(2, 3), static_argnames=('n_uniforms',))
def loggamma(
    key: Key[Array, ''],
    a: Float[Array, '...'],
    shape: Sequence[int] | None = None,
    dtype: DTypeLike | None = None,
    *,
    n_uniforms: int = 4,
) -> Float[Array, '...']:
    r"""Sample the log of Gamma(a, 1) random values.

    This is a loop-free replacement for `jax.random.loggamma`. It draws a base
    sample with a chi-square quantile expansion and corrects it with Stuart's
    boosting identity, avoiding the nested rejection loops of the jax version
    while preserving the log-space accuracy for small `a`.

    Parameters
    ----------
    key
        JAX random key.
    a
        The shape parameter of the distribution, broadcasted with `shape`.
    shape
        Shape of the output. If not specified, it is `a.shape`.
    dtype
        Floating point dtype of the output. Defaults to the canonical float.
        Internally the computation always runs in at least float32, so a narrower
        dtype only rounds the output and yields no speedup.
    n_uniforms
        The number of boosting steps, i.e., of uniform variates used to move the
        base shape away from zero. Larger values increase accuracy at small `a`.
        The default keeps the base shape at :math:`\geq 4` as `a` decreases,
        where the base approximation is accurate and its floor below is hit with
        negligible probability (:math:`\approx 3 \times 10^{-9}`).

    Returns
    -------
    Array of `log(Gamma(a, 1))` samples.

    Notes
    -----
    The sampler relies on Stuart's theorem, which iterated `n_uniforms` times
    gives, with :math:`U_k` independent uniforms on :math:`(0, 1)`,

    .. math::
        \log G_a = \log G_{a + n} + \sum_{k=0}^{n-1} \frac{\log U_k}{a + k}.

    The base draw :math:`G_{a + n}` lands at shape :math:`\geq n`, where the
    chi-square quantile expansion is accurate, while the sum carries the small-`a`
    behavior exactly. :math:`\log U_k` is drawn as :math:`-\mathrm{Exponential}(1)`
    to avoid evaluating `log` at zero.

    Samples below the smallest value representable in `dtype` underflow to
    :math:`-\infty`.
    """
    dtype = canonicalize_dtype(float if dtype is None else dtype)
    # compute in at least float32: a narrower dtype gives no speedup (the cost is
    # the rng and transcendentals, which run in float32 anyway) and only costs
    # accuracy in the random draws, so we only narrow at the very end.
    compute_dtype = jnp.promote_types(dtype, jnp.float32)
    shape = jnp.shape(a) if shape is None else tuple(shape)
    a = jnp.broadcast_to(jnp.asarray(a, compute_dtype), shape)

    key_base, key_boost = random.split(key)

    # base draw at the boosted shape a + n_uniforms, where the approximation is good
    log_base = _loggamma_chisquare(key_base, a + n_uniforms)

    # boosting correction: sum_k log(U_k) / (a + k), with log(U_k) = -Exp(1)
    k = jnp.arange(n_uniforms, dtype=compute_dtype).reshape(
        (n_uniforms, *len(shape) * (1,))
    )
    log_u = -random.exponential(key_boost, (n_uniforms, *shape), compute_dtype)
    correction = jnp.sum(log_u / (a + k), axis=0)

    return (log_base + correction).astype(dtype)


# Coefficients of the chi-square quantile approximation (18.37) in Johnson, Kotz
# & Balakrishnan, Continuous Univariate Distributions, vol. 1, 2nd ed. Each inner
# tuple k (k=0..6) holds the coefficients of x^k nu^(-k/2), as a quadratic in
# 1/nu: entry [k][j] multiplies nu^(-j) for j=0..2.
#
# The leading constant is set to exactly 1 rather than the published 1.0000886.
# As nu -> inf the cube root (chi2/nu)^(1/3) -> 1, so this term must be 1; the
# published value is a fit artifact that multiplies every draw by 1.0000886**3,
# adding a constant 3*log(1.0000886) ~ 2.7e-4 bias to log(Gamma). That bias is
# negligible against the O(1/sqrt(a)) spread for small a, but at large a the
# spread shrinks below it and it dominates the sampling error.
_CHI2_QUANTILE_COEF = (
    (1.0, -0.2237368, -0.01513904),
    (0.4713941, 0.02607083, -0.008986007),
    (0.0001348028, 0.01128186, 0.02277679),
    (-0.008553069, -0.01153761, -0.01323293),
    (0.00312558, 0.005169654, -0.006950356),
    (-0.0008426812, 0.00253001, 0.001060438),
    (0.00009780499, -0.001450117, 0.001565326),
)


def _loggamma_chisquare(
    key: Key[Array, ''], a: Float[Array, '...']
) -> Float[Array, '...']:
    r"""Log of a Gamma(a, 1) draw via a high-order chi-square quantile expansion.

    Cubes a 6th-degree polynomial in a standard normal (eq. 18.37 of Johnson,
    Kotz & Balakrishnan, Continuous Univariate Distributions, vol. 1), applied to
    chi-square with :math:`\nu = 2a` degrees of freedom, since
    :math:`\mathrm{Gamma}(a, 1) = \chi^2_{2a} / 2`. The polynomial is evaluated by
    Horner's method, both in the normal variate and in :math:`1/\nu`.
    """
    nu = 2 * a
    x = random.normal(key, jnp.shape(a), a.dtype)
    u = 1 / nu
    t = x * lax.rsqrt(nu)  # since x^k nu^(-k/2) = t^k, poly is a degree-6 Horner in t

    def row_coef(c: tuple[float, float, float]) -> Float[Array, '...']:
        """Evaluate a coefficient row as a quadratic in 1/nu, by Horner."""
        return c[0] + u * (c[1] + u * c[2])

    poly = row_coef(_CHI2_QUANTILE_COEF[-1])
    for row in reversed(_CHI2_QUANTILE_COEF[:-1]):
        poly = poly * t + row_coef(row)

    # chi2 = nu poly^3 and Gamma(a) = chi2 / 2 = a poly^3, so the log gains log(a)
    poly = jnp.maximum(poly, jnp.finfo(a.dtype).tiny)
    return jnp.log(a) + 3 * jnp.log(poly)
