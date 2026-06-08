# bartz/src/bartz/testing/_distr.py
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

"""Distribution families used as settings of the `gen_data` DGP."""

import math
from abc import abstractmethod

from equinox import Module, error_if
from jax import numpy as jnp
from jax import random
from jaxtyping import Array, Float, Integer, Key

from bartz._jaxext.random import loggamma

# The DGP identities documented in `Params` hold for arbitrary families given
# independence and the normalized moments declared by the base classes:
#   - x: mean 0, variance 1; its kurtosis enters the quadratic-budget
#     normalizer and var_v.
#   - beta/A draws: mean 0 and variance 1 suffice; every budget identity is at
#     most quadratic in these coefficients, so their higher moments never enter.
#   - gamma: mean 0, variance 1 *and* known kurtosis: var_v is quartic in gamma.
#   - s: E[s^2] = 1 exactly, E[s^4] known.
# Third moments never enter: every odd term carries a lone E[x] = 0 factor.


class Distr(Module):
    """Family of standardized distributions: mean 0, variance 1."""

    @property
    @abstractmethod
    def kurtosis(self) -> Float[Array, ''] | float:
        """Fourth moment ``E[z ** 4]`` of the standardized draws."""

    @abstractmethod
    def sample(
        self, key: Key[Array, ''], shape: tuple[int, ...]
    ) -> Float[Array, '*shape']:
        """Sample i.i.d. standardized values."""


class Uniform(Distr):
    """Continuous uniform distribution, standardized: U(-sqrt(3), sqrt(3))."""

    @property
    def kurtosis(self) -> float:
        """Fourth moment ``E[z ** 4] = 9/5``."""
        return 9 / 5

    def sample(
        self, key: Key[Array, ''], shape: tuple[int, ...]
    ) -> Float[Array, '*shape']:
        """Sample i.i.d. standardized continuous uniform values."""
        return random.uniform(key, shape, minval=-math.sqrt(3), maxval=math.sqrt(3))


class DiscreteUniform(Distr):
    """Uniform distribution on `m` equispaced levels, standardized.

    ``m=2`` gives random signs (levels -1 and 1), whose squares are constant
    (kurtosis 1).
    """

    m: Integer[Array, ''] | int
    """Number of levels, ``>= 2``."""

    @property
    def kurtosis(self) -> Float[Array, ''] | float:
        """Fourth moment ``E[z ** 4] = 3/5 (3m^2 - 7)/(m^2 - 1)``.

        1 at ``m=2``, increasing towards 9/5 (the continuous `Uniform`) as
        `m` grows.
        """
        mm = self.m * self.m
        # integer numerator and denominator are exact in any float precision,
        # so kurtosis == 1 identifies m == 2 reliably
        return (9 * mm - 21) / (5 * mm - 5)

    def sample(
        self, key: Key[Array, ''], shape: tuple[int, ...]
    ) -> Float[Array, '*shape']:
        """Sample i.i.d. standardized values on the `m` levels."""
        m = error_if(self.m, self.m < 2, 'm must be >= 2')
        levels = random.randint(key, shape, 0, m)
        mean = (m - 1) / 2
        var = (m * m - 1) / 12
        return (levels - mean) / jnp.sqrt(var)


class Normal(Distr):
    """Standard Normal distribution."""

    @property
    def kurtosis(self) -> float:
        """Fourth moment ``E[z ** 4] = 3``."""
        return 3.0

    def sample(
        self, key: Key[Array, ''], shape: tuple[int, ...]
    ) -> Float[Array, '*shape']:
        """Sample i.i.d. standard Normal values."""
        return random.normal(key, shape)


class ScaleDistr(Module):
    """Family of nonnegative scale distributions, normalized to ``E[s ** 2] = 1``."""

    @property
    @abstractmethod
    def fourth_moment(self) -> Float[Array, ''] | float:
        """Fourth moment ``E[s ** 4]`` of the normalized scales."""

    @abstractmethod
    def sample(
        self, key: Key[Array, ''], shape: tuple[int, ...]
    ) -> Float[Array, '*shape']:
        """Sample i.i.d. normalized scales."""


class Constant(ScaleDistr):
    """Scales concentrated at 1 (uniform predictor importance)."""

    @property
    def fourth_moment(self) -> float:
        """``E[s ** 4] = 1``."""
        return 1.0

    def sample(
        self,
        key: Key[Array, ''],  # noqa: ARG002
        shape: tuple[int, ...],
    ) -> Float[Array, '*shape']:
        """Return all-ones scales (`key` is unused)."""
        return jnp.ones(shape)


class Gamma(ScaleDistr):
    """Gamma(`alpha`) scales, rescaled to ``E[s ** 2] = 1``.

    Smaller `alpha` gives more dispersed scales; as ``alpha -> inf`` they
    concentrate at 1.
    """

    alpha: Float[Array, ''] | float
    """Shape parameter of the Gamma distribution, ``> 0``."""

    @property
    def fourth_moment(self) -> Float[Array, ''] | float:
        """``E[s ** 4] = (alpha + 2)(alpha + 3) / (alpha (alpha + 1))``."""
        a = self.alpha
        return (a + 2) * (a + 3) / (a * (a + 1))

    def sample(
        self, key: Key[Array, ''], shape: tuple[int, ...]
    ) -> Float[Array, '*shape']:
        """Sample i.i.d. normalized Gamma scales."""
        log_rate = jnp.log(self.alpha * (self.alpha + 1)) / 2
        return jnp.exp(loggamma(key, self.alpha, shape) - log_rate)


class SpikeSlab(ScaleDistr):
    """Two-point distribution over the scales 0 and ``1/sqrt(pi)``.

    The scale is ``1/sqrt(pi)`` with probability `pi` and 0 otherwise. This is
    hard variable selection: a fraction `pi` of the predictors is active, the
    others are exactly inert.
    """

    pi: Float[Array, ''] | float
    """Probability that a scale is nonzero, in ``(0, 1]``."""

    @property
    def fourth_moment(self) -> Float[Array, ''] | float:
        """``E[s ** 4] = 1 / pi``."""
        return 1 / self.pi

    def sample(
        self, key: Key[Array, ''], shape: tuple[int, ...]
    ) -> Float[Array, '*shape']:
        """Sample i.i.d. two-point scales."""
        return random.bernoulli(key, self.pi, shape) / jnp.sqrt(self.pi)
