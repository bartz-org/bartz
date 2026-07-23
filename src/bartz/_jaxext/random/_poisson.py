# bartz/src/bartz/_jaxext/random/_poisson.py
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

"""Loop-free sampler for Poisson random variables."""

from collections.abc import Sequence

from jax import custom_jvp, jvp, random
from jax import numpy as jnp
from jax.dtypes import canonicalize_dtype
from jax.scipy.special import ndtr
from jax.scipy.stats import norm
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Integer, Key

from bartz._jaxext import jit

# below this mean, sample by inverting the truncated cdf directly; above,
# invert the Peizer-Pratt normal approximation of the cdf, whose max cdf error
# at the split is ~ 6e-5 (chi-square-detectable only beyond ~3e7 samples)
THETA_SPLIT = 7.0

# number of pmf terms for the truncated cdf; the truncated tail mass
# Pr[X > 22 | theta=7] ~ 3e-6 is below the Peizer-Pratt error at the split
NUM_TERMS = 23

# Peizer-Pratt continuity correction constant (0 for simplicity, 0.02 for more
# accuracy)
EPS = 0.02


@jit(static_argnums=(2, 3))
def poisson(
    key: Key[Array, ''],
    lambda_: Float[Array, '...'] | float,
    shape: Sequence[int] | None = None,
    dtype: DTypeLike = int,
) -> Integer[Array, '...'] | Float[Array, '...']:
    """Faster approximate Poisson sampler.

    Loop-free replacement for `jax.random.poisson`. It is approximate: the
    total variation distance from the exact distribution is below 1e-4, worst
    around ``lambda_ = 7`` (see Notes).

    Parameters
    ----------
    key
        JAX random key.
    lambda_
        The mean parameter, broadcasted with `shape`.
    shape
        Shape of the output. If not specified, it is `lambda_.shape`.
    dtype
        Dtype of the output. Defaults to the canonical int. Pass a float dtype
        to propagate derivatives w.r.t. `lambda_` (see Notes).

    Returns
    -------
    Array of Poisson(lambda_) samples.

    Raises
    ------
    ValueError
        If `lambda_` does not broadcast to `shape`.

    Notes
    -----
    Below a mean of 7 the pmf is truncated and inverted directly; above, the
    sample is obtained by inverting the very accurate normal approximation of
    the cdf by Peizer and Pratt [1]_, as presented in [2]_.

    The samples carry an approximate derivative w.r.t. `lambda_`, defined by
    implicit differentiation of a continuous relaxation of the cdf before
    rounding: single-sample derivatives are biased, but averaged over samples
    they estimate derivatives of expected values, at any ``lambda_ > 0``. The
    integer cast blocks them, so they only flow with a float `dtype`.

    References
    ----------
    .. [1] Peizer, D. B., & Pratt, J. W. (1968). A Normal Approximation for
       Binomial, F, Beta, and other Common, Related Tail Probabilities, I.
       Journal of the American Statistical Association, 63(324), 1416-1456.
       https://doi.org/10.1080/01621459.1968.10480938
    .. [2] Johnson, N. L., Kemp, A. W., & Kotz, S. (2005). Univariate discrete
       distributions (3rd ed). Wiley. Page 168.
    """
    dtype = canonicalize_dtype(dtype)
    compute_dtype = jnp.promote_types(jnp.result_type(lambda_), jnp.float32)
    shape = jnp.shape(lambda_) if shape is None else tuple(shape)
    if jnp.broadcast_shapes(shape, jnp.shape(lambda_)) != shape:
        msg = f'lambda_ shape {jnp.shape(lambda_)} does not broadcast to {shape}'
        raise ValueError(msg)
    lambda_ = jnp.asarray(lambda_, compute_dtype)

    z = random.normal(key, shape, compute_dtype)
    return poisson_from_normal(z, lambda_).astype(dtype)


@custom_jvp
def poisson_from_normal(
    z: Float[Array, '*shape'], lambda_: Float[Array, '...']
) -> Float[Array, '*shape']:
    """Map a standard normal variate to a Poisson(lambda_) variate."""
    small = poisson_cdf_inversion(ndtr(z), lambda_)
    large = poisson_peizer_pratt(z, lambda_)
    return jnp.where(lambda_ < THETA_SPLIT, small, large)


@poisson_from_normal.defjvp
def poisson_from_normal_jvp(
    primals: tuple[Float[Array, '*shape'], Float[Array, '*lambda_shape']],
    tangents: tuple[Float[Array, '*shape'], Float[Array, '*lambda_shape']],
) -> tuple[Float[Array, '*shape'], Float[Array, '*shape']]:
    """Implicit differentiation of a continuous relaxation of the sampler.

    Each branch applies the implicit function theorem to the relaxed root of
    its own cdf representation: the linearly interpolated truncated cdf below
    `THETA_SPLIT`, the Peizer-Pratt approximation above.
    """
    z, lambda_ = primals
    z_dot, lambda_dot = tangents
    sample = poisson_from_normal(z, lambda_)
    small = poisson_cdf_inversion_jvp(z, lambda_, sample, z_dot, lambda_dot)
    large = poisson_peizer_pratt_jvp(z, lambda_, z_dot, lambda_dot)
    return sample, jnp.where(lambda_ < THETA_SPLIT, small, large)


def poisson_cdf_inversion_jvp(
    z: Float[Array, '*shape'],
    lambda_: Float[Array, '*lambda_shape'],
    k: Float[Array, '*shape'],
    z_dot: Float[Array, '*shape'],
    lambda_dot: Float[Array, '*lambda_shape'],
) -> Float[Array, '*shape']:
    """Tangent of the root of the linearly interpolated truncated cdf.

    In the cell x in (k - 1, k] the relaxed cdf is F(k - 1) + frac pmf(k),
    frac = x - (k - 1), and the relaxed root solves
    F(x, lambda_) = ndtr(z) F(K), K = NUM_TERMS - 1; the implicit function
    theorem with d F(j)/dlambda_ = -pmf(j) and pmf(k - 1)/pmf(k) = k/lambda_
    then gives dx elementwise.
    """
    # re-run the pmf recurrence, gathering the terms at the sampled value
    pmf = jnp.exp(-lambda_)
    cdf = pmf
    pmf_k = jnp.where(k == 0, pmf, 0.0)
    cdf_km1 = jnp.zeros_like(k)
    for j in range(1, NUM_TERMS):
        cdf_km1 = jnp.where(k == j, cdf, cdf_km1)
        pmf *= lambda_ / j
        pmf_k = jnp.where(k == j, pmf, pmf_k)
        cdf += pmf

    # now cdf and pmf are at K; clip frac against tail cancellation errors.
    # guard pmf_k == 0: above THETA_SPLIT the sampled k can exceed the
    # truncation so no gather fires; the caller discards this branch, but inf
    # coefficients would still poison reverse mode through the select transpose
    # (inf * 0 = nan)
    pmf_k = jnp.where(pmf_k == 0.0, 1.0, pmf_k)
    u = ndtr(z) * cdf
    frac = jnp.clip((u - cdf_km1) / pmf_k, 0.0, 1.0)

    dx_z = norm.pdf(z) * cdf / pmf_k
    # guard k/lambda_ to keep the k = 0 term zero also at lambda_ = 0
    k_over_lambda = jnp.where(k == 0, 0.0, k / lambda_)
    dx_lambda = frac + (1 - frac) * k_over_lambda - ndtr(z) * pmf / pmf_k
    return dx_z * z_dot + dx_lambda * lambda_dot


def poisson_peizer_pratt_jvp(
    z: Float[Array, '*shape'],
    lambda_: Float[Array, '*lambda_shape'],
    z_dot: Float[Array, '*shape'],
    lambda_dot: Float[Array, '*lambda_shape'],
) -> Float[Array, '*shape']:
    """Tangent of the root of the Peizer-Pratt relaxation.

    The continuous root x(z, lambda_) of `peizer_pratt_z(x, lambda_) = z`
    satisfies, by the implicit function theorem,
    `dx = (dz - dz/dlambda_ dlambda_) / (dz/dx)`. The partials are evaluated at
    the root guess.
    """
    # keep the evaluation point inside the domain x > -1/2 of z(x); beyond the
    # bound the sample is pinned at 0 anyway
    x = jnp.maximum(peizer_pratt_root(z, lambda_), -0.4)

    # each partial closes over the other variable, so jax treats it as a
    # constant instead of carrying around a zero tangent
    _, dz_lambda = jvp(lambda ell: peizer_pratt_z(x, ell), (lambda_,), (lambda_dot,))
    _, dz_x = jvp(lambda x: peizer_pratt_z(x, lambda_), (x,), (jnp.ones_like(x),))

    return (z_dot - dz_lambda) / dz_x


def poisson_cdf_inversion(
    u: Float[Array, '*shape'], lambda_: Float[Array, '...']
) -> Float[Array, '*shape']:
    """Invert the cdf truncated to the first `NUM_TERMS` values."""
    # unrolled pmf recurrence instead of cumsum(exp(logpmf(arange))): keeps
    # every op elementwise at the broadcast shape, so xla fuses the inversion
    # into one kernel instead of materializing shape x NUM_TERMS intermediates
    pmf = jnp.exp(-lambda_)
    cdf = pmf
    cdfs = [cdf]
    for k in range(1, NUM_TERMS):
        pmf *= lambda_ / k
        cdf += pmf
        cdfs.append(cdf)

    # re-normalize because of truncation and float rounding
    u *= cdf

    count = jnp.zeros_like(u)
    for cdf in cdfs:
        count += cdf < u
    return count


def poisson_peizer_pratt(
    z: Float[Array, '*shape'], lambda_: Float[Array, '...']
) -> Float[Array, '*shape']:
    """Invert the Peizer-Pratt cdf approximation Pr[X <= x] ~ Phi(z(x)).

    The sample is the smallest integer x with z(x) >= z, i.e., ceil of the real
    root of z(x) = z. The exact z(x) is evaluated at 2 integers around an
    analytical guess to select the right value, correcting the guess as long as
    it falls within +/-1 of the root.
    """
    # the guess error stays within (-1 + 0.6, 1 + 0.6) of the root over
    # theta >= THETA_SPLIT, |z| <= 8.4 (the float64 range of random.normal),
    # so shifting by 0.6 centers it in the +/-1 window around ceil
    x = peizer_pratt_root(z, lambda_) + 0.6

    # select the smallest integer c with z(c) >= z among {c-1, c, c+1},
    # c = ceil(x); z at negative integers is -inf since Pr[X <= x] = 0 there
    c = jnp.ceil(x)

    def zint(c: Float[Array, '*shape']) -> Float[Array, '*shape']:
        return jnp.where(c < 0, -jnp.inf, peizer_pratt_z(jnp.maximum(c, 0.0), lambda_))

    out = c + 1 - (zint(c) >= z) - (zint(c - 1) >= z)
    return jnp.maximum(out, 0.0)


def peizer_pratt_root(
    z: Float[Array, '*shape'], lambda_: Float[Array, '...']
) -> Float[Array, '*shape']:
    """Cornish-Fisher approximation of the root of `peizer_pratt_z(x, .) = z`."""
    s = jnp.sqrt(lambda_)
    return (
        lambda_ + s * z - 2 / 3 + jnp.square(z) / 6 + z * (1 - jnp.square(z)) / (72 * s)
    )


def peizer_pratt_z(
    x: Float[Array, '*shape'], lambda_: Float[Array, '...']
) -> Float[Array, '*shape']:
    """Peizer-Pratt z(x) such that Pr[X <= x] ~ Phi(z(x)), for x > -1/2."""
    y = (x + 0.5) / lambda_
    t = peizer_pratt_t(y)
    return (x - lambda_ + 2 / 3 + EPS / (x + 1)) * jnp.sqrt((1 + t) / lambda_)


def peizer_pratt_t(y: Float[Array, '*shape']) -> Float[Array, '*shape']:
    """T(y) = (1 - y^2 + 2 y log y) / (1 - y)^2, stable around y = 1."""
    d = y - 1
    small = jnp.abs(d) < 0.25

    # series: T(1 + d) = sum_{k>=1} 2 (-1)^k d^k / ((k+1) (k+2)); 9 terms are
    # the fewest that keep the truncation error at the cutoff (~2e-8) below
    # float32 resolution; in float64 the Peizer-Pratt error dominates anyway
    series = jnp.zeros_like(d)
    for k in reversed(range(1, 10)):
        coef = 2 * (-1) ** k / ((k + 1) * (k + 2))
        series = d * (coef + series)

    y_safe = jnp.where(small, 2.0, y)
    d_safe = jnp.where(small, 1.0, d)
    direct = (1 - jnp.square(y_safe) + 2 * y_safe * jnp.log(y_safe)) / jnp.square(
        d_safe
    )

    return jnp.where(small, series, direct)
