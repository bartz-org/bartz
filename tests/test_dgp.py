# bartz/tests/test_dgp.py
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

"""Tests `bartz.testing.gen_data`."""

from collections.abc import Mapping
from functools import partial
from operator import attrgetter
from types import MappingProxyType
from typing import Literal

import pytest
from equinox import EquinoxRuntimeError, EquinoxTracetimeError
from jax import block_until_ready, jit, random, vmap
from jax import numpy as jnp
from jax.errors import JaxRuntimeError
from jaxtyping import Array, Bool, Float, Key
from numpy.testing import assert_array_less
from scipy.stats import norm

from bartz._jaxext import split
from bartz._typing import kwdict
from bartz.mcmcstep import OutcomeType
from bartz.testing import (
    DGP,
    Constant,
    DiscreteUniform,
    Distr,
    Gamma,
    Normal,
    Params,
    ScaleDistr,
    SpikeSlab,
    Uniform,
    gen_data,
)
from bartz.testing._dgp import (
    generate_partition,
    interaction_pattern,
    partitioned_interaction_pattern,
)
from tests.util import assert_allclose, assert_array_equal, assert_close_matrices, nnone

# Test parameters
ALPHA = 5e-7  # probability of false positive (aaaaapprox)
SIGMA_THRESHOLD = norm.isf(ALPHA / 2)  # threshold for z tests
KWARGS: Mapping = MappingProxyType(
    dict(n=100, p=20, k=3, q=4, sigma2_eps=0.1, sigma2_lin=0.4, sigma2_quad=0.5)
)
REPS: int = 10_000  # number of datasets, and of i.i.d. draws in moment tests
SPARSITY: float = 2.0  # Gamma shape for the sparse-DGP fixture (mu4 = 10 / 3)
# error_if surfaces eagerly (tracetime), under jit (runtime), or as a plain check
EQX_ERRORS = (EquinoxTracetimeError, EquinoxRuntimeError, JaxRuntimeError, ValueError)
HIGH_SPARSITY: float = 1e19  # largest before sparsity (sparsity + 1) overflows float32

# A substantial heteroskedasticity knob to exercise the het paths; the bounded
# quadratic multiplier keeps every marginal-variance estimator well-behaved.
HET_STRENGTH: float = 0.7


def assert_mean_z(
    samples: Float[Array, 'reps ...'],
    expected: Float[Array, ''] | float,
    se: Float[Array, '...'] | float | None = None,
) -> None:
    """Check via a z-test that the mean of `samples` along axis 0 is `expected`.

    The standard error is estimated from the samples unless `se` is given.
    """
    n_reps, *_ = samples.shape
    if se is None:
        se = jnp.std(samples, axis=0) / jnp.sqrt(n_reps)
    z_scores = jnp.abs((jnp.mean(samples, axis=0) - expected) / se)
    assert_array_less(z_scores, SIGMA_THRESHOLD)


def assert_uncorrelated_z(
    a: Float[Array, 'reps ...'], b: Float[Array, 'reps ...']
) -> None:
    """Check via a z-test that `a` and `b` have zero covariance along axis 0."""
    n_reps, *_ = a.shape
    cov = jnp.mean((a - jnp.mean(a, axis=0)) * (b - jnp.mean(b, axis=0)), axis=0)
    se = jnp.std(a, axis=0) * jnp.std(b, axis=0) / jnp.sqrt(n_reps)
    assert_array_less(jnp.abs(cov / se), SIGMA_THRESHOLD)


@partial(jit, static_argnames=('het_shape',))
def generate_dgps(
    keys: Key[Array, 'REPS'],
    lambda_: Float[Array, ''],
    s_distr: ScaleDistr,
    het_strength: float | None,
    het_shape: str | None,
    x_distr: Distr,
    gamma_distr: Distr,
) -> DGP:
    """Generate one dataset per random key (jitted, vmapped over keys)."""
    gen = partial(
        gen_data,
        lambda_=lambda_,
        s_distr=s_distr,
        het_strength=het_strength,
        het_shape=het_shape,
        x_distr=x_distr,
        gamma_distr=gamma_distr,
        **KWARGS,
    )
    return vmap(gen)(keys)


@pytest.fixture
def dgps(keys: split, request: pytest.FixtureRequest) -> DGP:
    """Generate DGP instances using vmap and jit.

    Indirectly parametrizable by a mapping with any of ``lambda_``,
    ``s_distr``, ``het_strength``, ``het_shape``, ``x_distr`` and
    ``gamma_distr``.
    """
    param: Mapping = getattr(request, 'param', None) or {}
    return generate_dgps(
        keys.pop(REPS),
        param.get('lambda_', 0.5),
        param.get('s_distr', Constant()),
        param.get('het_strength'),
        param.get('het_shape'),
        param.get('x_distr', Uniform()),
        param.get('gamma_distr', DiscreteUniform(2)),
    )


def test_shapes_and_dtypes(keys: split) -> None:
    """Test that all DGP attributes have correct shapes and dtypes."""
    dgp = gen_data(keys.pop(), lambda_=0.5, **KWARGS)
    n, p, k = KWARGS['n'], KWARGS['p'], KWARGS['k']

    floating_fields = {
        'x': (p, n),
        'y': (k, n),
        'z': (k, n),
        'params.beta_shared': (p,),
        'params.beta_separate': (k, p),
        'mulin_shared': (n,),
        'mulin_separate': (k, n),
        'mulin': (k, n),
        'params.A_shared': (p, p),
        'params.A_separate': (k, p, p),
        'params.s': (p,),
        'muquad_shared': (n,),
        'muquad_separate': (k, n),
        'muquad': (k, n),
        'mu': (k, n),
        'params.lambda_': (),
        'params.sigma2_lin': (),
        'params.sigma2_quad': (),
        'params.sigma2_eps': (),
        'params.offset': (),
    }
    for name, shape in floating_fields.items():
        field = attrgetter(name)(dgp)
        assert field.shape == shape, name
        assert jnp.issubdtype(field.dtype, jnp.floating), name

    assert nnone(dgp.params.partition).shape == (k, p)
    assert nnone(dgp.params.partition).dtype == jnp.bool_
    assert dgp.params.q.shape == ()
    assert jnp.issubdtype(dgp.params.q.dtype, jnp.integer)
    assert isinstance(dgp.params.x_distr, Uniform)
    assert isinstance(dgp.params.beta_distr, DiscreteUniform)
    assert isinstance(dgp.params.A_distr, DiscreteUniform)
    assert isinstance(dgp.params.gamma_distr, DiscreteUniform)
    assert isinstance(dgp.params.s_distr, Constant)


def discrete_levels(m: int) -> Float[Array, ' m']:
    """Compute the standardized levels of `DiscreteUniform`."""
    return (jnp.arange(m) - (m - 1) / 2) / jnp.sqrt((m * m - 1) / 12)


class TestDistr:
    """Test the standardized distribution families in isolation."""

    @pytest.fixture(
        params=[
            Uniform(),
            DiscreteUniform(2),
            DiscreteUniform(3),
            DiscreteUniform(5),
            DiscreteUniform(100),
            Normal(),
        ],
        ids=repr,
    )
    def distr(self, request: pytest.FixtureRequest) -> Distr:
        """Yield each distribution family."""
        return request.param

    def test_moments(self, keys: split, distr: Distr) -> None:
        """The draws have mean 0, variance 1 and fourth moment `kurtosis`."""
        z = distr.sample(keys.pop(), (REPS,))
        assert_mean_z(z, 0.0)
        if distr.kurtosis == 1:
            # degenerate z-tests: the squares are exactly constant
            assert_array_equal(z**2, jnp.ones_like(z))
        else:
            assert_mean_z(z**2, 1.0)
            assert_mean_z(z**4, distr.kurtosis)

    @pytest.mark.parametrize('m', [2, 3, 5, 100])
    def test_quantized_levels(self, keys: split, m: int) -> None:
        """`DiscreteUniform` takes exactly the `m` standardized levels."""
        z = DiscreteUniform(m).sample(keys.pop(), (REPS,))
        assert jnp.issubdtype(z.dtype, jnp.floating)
        assert_close_matrices(jnp.unique(z), discrete_levels(m), rtol=1e-6)

    @pytest.mark.parametrize('m', [2, 3, 5, 100])
    def test_levels_moments(self, m: int) -> None:
        """The (equiprobable) levels have mean 0, variance 1 and kurtosis `kurtosis`.

        Computing the moments over the levels checks the standardization and
        the `kurtosis` closed form exactly.
        """
        levels = discrete_levels(m)
        assert_allclose(jnp.mean(levels), 0.0, atol=1e-6)
        assert_allclose(jnp.var(levels), 1.0, rtol=1e-6)
        assert_allclose(jnp.mean(levels**4), DiscreteUniform(m).kurtosis, rtol=1e-6)

    def test_binary_levels(self, keys: split) -> None:
        """`DiscreteUniform(2)` samples exactly +-1 (squares constant, kurtosis 1)."""
        z = DiscreteUniform(2).sample(keys.pop(), (REPS,))
        assert_array_equal(jnp.unique(z), jnp.array([-1.0, 1.0]))
        assert DiscreteUniform(2).kurtosis == 1.0

    def test_quantize(self, keys: split, distr: Distr) -> None:
        """`quantize` is monotone and maps to equiprobable bins in [0, m)."""
        max_bins = 8
        z = distr.sample(keys.pop(), (REPS,))
        bins, m = distr.quantize(z, max_bins)
        assert jnp.issubdtype(bins.dtype, jnp.unsignedinteger)
        assert m <= max_bins
        assert jnp.all(bins < m)

        # quantization preserves the ordering of the values
        sorted_bins = bins.astype(jnp.int32)[jnp.argsort(z)]
        assert jnp.all(jnp.diff(sorted_bins) >= 0)

        # bins are equiprobable, up to the rounding of grouping `distr.m`
        # levels into `m` bins for DiscreteUniform
        if isinstance(distr, DiscreteUniform):
            level_bins = jnp.arange(distr.m) * m // distr.m
            prob = jnp.bincount(level_bins, length=max_bins) / distr.m
        else:
            prob = jnp.full(max_bins, 1 / max_bins)
        indicators = bins[:, None] == jnp.arange(max_bins)
        assert not jnp.any(indicators[:, prob == 0])
        assert_mean_z(indicators[:, prob > 0], prob[prob > 0])

    @pytest.mark.parametrize('m', [2, 3, 5, 100])
    def test_quantize_levels(self, m: int) -> None:
        """With enough bins, `quantize` recovers the level indices exactly."""
        bins, m_out = DiscreteUniform(m).quantize(discrete_levels(m), 128)
        assert m_out == m
        assert_array_equal(bins, jnp.arange(m), strict=False)

    def test_quantize_merges_levels(self) -> None:
        """With fewer bins than levels, the levels are spread evenly."""
        m, max_bins = 100, 8
        bins, m_out = DiscreteUniform(m).quantize(discrete_levels(m), max_bins)
        assert m_out == max_bins
        counts = jnp.bincount(bins, length=max_bins)
        assert jnp.all((counts == m // max_bins) | (counts == m // max_bins + 1))


class TestX:
    """Test the predictors generated by `gen_data`."""

    def test_x_support(self, dgps: DGP) -> None:
        """Test that continuous x lies in the standardized uniform support."""
        assert jnp.all(jnp.abs(dgps.x) <= jnp.sqrt(3.0))

    def test_x_mean(self, dgps: DGP) -> None:
        """Test that x has mean close to 0."""
        assert_mean_z(dgps.x, 0.0)

    def test_x_variance(self, dgps: DGP) -> None:
        """Test that x has variance close to 1."""
        n_reps, *_ = dgps.x.shape
        var = jnp.var(dgps.x, axis=0)  # Shape: (P, N)
        # conservative Gaussian approx (uniform x is light-tailed)
        std_of_var = jnp.sqrt(2 / (n_reps - 1))
        z_scores = jnp.abs((var - 1.0) / std_of_var)
        assert_array_less(z_scores, SIGMA_THRESHOLD)

    @pytest.mark.parametrize('m', [2, 3, 5])
    def test_quantized_support(self, keys: split, m: int) -> None:
        """Test that quantized x takes exactly the `m` standardized levels."""
        dgp = gen_data(keys.pop(), x_distr=DiscreteUniform(m), lambda_=0.5, **KWARGS)
        assert jnp.issubdtype(dgp.x.dtype, jnp.floating)
        assert_close_matrices(jnp.unique(dgp.x), discrete_levels(m), rtol=1e-6)
        assert isinstance(dgp.params.x_distr, DiscreteUniform)


class TestQuantize:
    """Test `DGP.quantize`."""

    def test_format(self, keys: split) -> None:
        """The output has the shapes, dtypes and bounds `mcmcstep.init` expects."""
        dgp = gen_data(keys.pop(), lambda_=0.5, **KWARGS)
        data = dgp.quantize()
        n, p = KWARGS['n'], KWARGS['p']
        assert data.x.shape == (p, n)
        assert data.x.dtype == jnp.uint8
        assert data.max_split.shape == (p,)
        assert data.max_split.dtype == jnp.uint8
        assert_array_equal(data.max_split, jnp.full(p, 255), strict=False)
        assert jnp.all(data.x <= data.max_split[:, None])
        assert_array_equal(data.y, dgp.y)

    @pytest.mark.parametrize(
        ('max_bins', 'dtype'), [(256, jnp.uint8), (257, jnp.uint16)]
    )
    def test_max_bins(self, keys: split, max_bins: int, dtype: type) -> None:
        """`max_bins` sets the number of levels; the dtype widens past 256."""
        dgp = gen_data(keys.pop(), lambda_=0.5, **KWARGS)
        data = dgp.quantize(max_bins=max_bins)
        assert data.x.dtype == dtype
        assert data.max_split.dtype == dtype
        assert_array_equal(
            data.max_split, jnp.full(KWARGS['p'], max_bins - 1), strict=False
        )
        assert jnp.all(data.x <= data.max_split[:, None])

    @pytest.mark.parametrize('m', [2, 5])
    def test_discrete_x(self, keys: split, m: int) -> None:
        """With discrete predictors, `max_split` counts the actual levels."""
        dgp = gen_data(keys.pop(), x_distr=DiscreteUniform(m), lambda_=0.5, **KWARGS)
        data = dgp.quantize()
        assert_array_equal(data.max_split, jnp.full(KWARGS['p'], m - 1), strict=False)
        assert jnp.all(data.x <= data.max_split[:, None])


class TestGeneratePartition:
    """Test the generate_partition function."""

    def test_partition_coverage(self, dgps: DGP) -> None:
        """Test that each predictor is assigned to exactly one component."""
        partitions = nnone(dgps.params.partition)  # Shape: (REPS, K, P)

        # Each column should sum to 1
        col_sums = jnp.sum(partitions, axis=1)  # Shape: (N_REPS, P)
        assert_array_equal(col_sums, 1, strict=False)

    def test_partition_counts(self, dgps: DGP) -> None:
        """Test that counts are either p//c or p//c + 1."""
        partitions = nnone(dgps.params.partition)  # Shape: (REPS, K, P)
        p, k = partitions.shape[2], partitions.shape[1]

        counts = jnp.sum(partitions, axis=2)  # Shape: (REPS, K)

        # Each count should be either P//K or P//K + 1
        floor_count = p // k
        valid = (counts == floor_count) | (counts == floor_count + 1)
        assert_array_equal(valid, True, strict=False)

    def test_partition_balance(self, dgps: DGP) -> None:
        """Test that predictors are roughly balanced across components."""
        partitions = nnone(dgps.params.partition)  # Shape: (REPS, K, P)
        _, k, p = partitions.shape
        counts = jnp.sum(partitions, axis=2)  # Shape: (REPS, K)
        assert_mean_z(counts, p / k)


def s_moment(alpha: float, power: int) -> float:
    """Analytic E[s ** power] for the scales drawn by `Gamma`.

    With ``s = Gamma(alpha) / sqrt(alpha (alpha + 1))`` one has
    ``E[s ** m] = prod_{i<m}(alpha + i) / (alpha (alpha + 1)) ** (m/2)``
    (``power`` must be even). In particular ``s_moment(a, 2) == 1``.
    """
    num = 1.0
    for i in range(power):
        num *= alpha + i
    return num / (alpha * (alpha + 1)) ** (power // 2)


class TestScaleDistr:
    """Test the scale families in isolation."""

    ALPHA_S: float = 5.0  # a moderate shape, light enough tails for the z tests

    def test_constant_is_ones(self, keys: split) -> None:
        """`Constant` returns all-ones scales (uniform importance)."""
        p = KWARGS['p']
        assert_array_equal(Constant().sample(keys.pop(), (p,)), jnp.ones(p))

    def test_shape_and_dtype(self, keys: split) -> None:
        """Scales have shape (p,) and a floating dtype."""
        p = KWARGS['p']
        s = Gamma(self.ALPHA_S).sample(keys.pop(), (p,))
        assert s.shape == (p,)
        assert jnp.issubdtype(s.dtype, jnp.floating)

    def test_second_moment(self, keys: split) -> None:
        """The scales are normalized to ``E[s ** 2] == 1``."""
        s2 = Gamma(self.ALPHA_S).sample(keys.pop(), (REPS,)) ** 2
        var_s2 = s_moment(self.ALPHA_S, 4) - 1.0  # Var[s^2] = E[s^4] - E[s^2]^2
        assert_mean_z(s2, 1.0, se=jnp.sqrt(var_s2 / REPS))

    def test_fourth_moment(self, keys: split) -> None:
        """``E[s ** 4]`` matches the analytic `fourth_moment` of the family."""
        gamma = Gamma(self.ALPHA_S)
        assert_allclose(gamma.fourth_moment, s_moment(self.ALPHA_S, 4), rtol=1e-6)
        s4 = gamma.sample(keys.pop(), (REPS,)) ** 4
        var_s4 = s_moment(self.ALPHA_S, 8) - gamma.fourth_moment**2
        assert_mean_z(s4, gamma.fourth_moment, se=jnp.sqrt(var_s4 / REPS))

    def test_dispersion_increases_with_sparsity(self, keys: split) -> None:
        """A smaller Gamma shape spreads the importances out more (larger Var[s^2])."""
        sparse = Gamma(1.0).sample(keys.pop(), (REPS,))
        dense = Gamma(10.0).sample(keys.pop(), (REPS,))
        assert jnp.var(sparse**2) > jnp.var(dense**2)


class TestSpikeSlab:
    """Test the `SpikeSlab` scale family."""

    PI: float = 0.3

    def test_support_and_moments(self, keys: split) -> None:
        """The scales take the two values 0 and 1/sqrt(pi), with the right moments."""
        distr = SpikeSlab(self.PI)
        s = distr.sample(keys.pop(), (REPS,))
        assert_close_matrices(
            jnp.unique(s), jnp.array([0.0, 1 / jnp.sqrt(self.PI)]), rtol=1e-6, atol=1e-6
        )
        assert_mean_z(s**2, 1.0)
        assert_mean_z(s**4, distr.fourth_moment)
        assert_allclose(distr.fourth_moment, 1 / self.PI, rtol=1e-6)

    def test_zeroes_inert_predictors(self, keys: split) -> None:
        """Predictors with a zero scale are exactly inert in every term."""
        dgp = gen_data(
            keys.pop(),
            lambda_=0.5,
            s_distr=SpikeSlab(0.5),
            het_strength=HET_STRENGTH,
            het_shape='vector',
            **KWARGS,
        )
        dead = dgp.params.s == 0
        # both outcomes near-certain at p = 20
        assert jnp.any(dead)
        assert not jnp.all(dead)
        assert_array_equal(dgp.params.beta_shared[dead], 0, strict=False)
        assert_array_equal(nnone(dgp.params.beta_separate)[:, dead], 0, strict=False)
        assert_array_equal(dgp.params.A_shared[dead, :], 0, strict=False)
        assert_array_equal(dgp.params.A_shared[:, dead], 0, strict=False)
        assert_array_equal(nnone(dgp.params.A_separate)[:, dead, :], 0, strict=False)
        assert_array_equal(nnone(dgp.params.gamma_shared)[dead], 0, strict=False)
        assert_array_equal(nnone(dgp.params.gamma_separate)[:, dead], 0, strict=False)


class TestFromPeff:
    """Test `ScaleDistr.from_peff`, the effective-active-predictors initializer."""

    P: int = 20

    @pytest.mark.parametrize('family', [Gamma, SpikeSlab])
    @pytest.mark.parametrize('peff', [1.0, 2.5, 7.0, 19.0])
    def test_fourth_moment_roundtrip(
        self, family: type[ScaleDistr], peff: float
    ) -> None:
        """`from_peff` builds the member with ``fourth_moment == p / peff``."""
        distr = family.from_peff(peff, self.P)
        assert_allclose(distr.fourth_moment, self.P / peff, rtol=1e-5)

    @pytest.mark.parametrize('peff', [1.0, 2.5, 7.0, 20.0])
    def test_spikeslab_pi_is_active_fraction(self, peff: float) -> None:
        """For `SpikeSlab`, ``pi`` is the active fraction ``peff / p``."""
        distr = SpikeSlab.from_peff(peff, self.P)
        assert isinstance(distr, SpikeSlab)
        assert_allclose(distr.pi, peff / self.P, rtol=1e-6)

    def test_constant_requires_peff_eq_p(self) -> None:
        """`Constant.from_peff` returns `Constant` when ``peff == p``."""
        assert isinstance(Constant.from_peff(float(self.P), self.P), Constant)

    def test_constant_rejects_peff_ne_p(self) -> None:
        """`Constant.from_peff` errors at ``peff != p`` (it has no free parameter)."""
        with pytest.raises(ValueError, match='Constant has peff == p only'):
            Constant.from_peff(3.0, self.P)

    @pytest.mark.parametrize('peff', [2.0, 8.0])
    def test_spikeslab_active_count(self, keys: split, peff: float) -> None:
        """`SpikeSlab.from_peff(peff)` activates `peff` predictors on average."""
        p = 200
        s = SpikeSlab.from_peff(peff, p).sample(keys.pop(), (REPS, p))
        assert_mean_z(jnp.sum(s > 0, axis=1).astype(float), peff)

    @pytest.mark.parametrize('family', [Gamma, SpikeSlab])
    def test_participation_ratio(self, keys: split, family: type[ScaleDistr]) -> None:
        """A large draw's participation ratio recovers the requested `peff`."""
        p = 100_000
        peff = p / 2
        s2 = family.from_peff(peff, p).sample(keys.pop(), (p,)) ** 2
        peff_hat = jnp.square(jnp.sum(s2)) / jnp.sum(jnp.square(s2))
        assert_allclose(peff_hat, peff, rtol=0.05)

    def test_gamma_rejects_peff_ge_p(self) -> None:
        """`Gamma.from_peff` errors at ``peff >= p`` (its alpha would diverge)."""
        with pytest.raises(EQX_ERRORS, match='Gamma needs peff'):
            block_until_ready(Gamma.from_peff(float(self.P), self.P))

    def test_spikeslab_rejects_peff_gt_p(self) -> None:
        """`SpikeSlab.from_peff` errors at ``peff > p`` (its pi would exceed 1)."""
        with pytest.raises(EQX_ERRORS, match='SpikeSlab needs peff'):
            block_until_ready(SpikeSlab.from_peff(self.P + 1.0, self.P))


def test_high_sparsity_matches_none(keys: split) -> None:
    """A very high Gamma shape reproduces the non-sparse (`None`) DGP.

    The importance scales concentrate at 1 as the Gamma shape goes to
    infinity, so the data matches the `Constant` run under the same key. The
    match is approximate because the scales are close to 1 but not exact.
    """
    kw: kwdict = dict(KWARGS, lambda_=0.5)
    key = keys.pop()
    none = gen_data(key, s_distr=Constant(), **kw)
    high = gen_data(random.clone(key), s_distr=Gamma(HIGH_SPARSITY), **kw)
    assert_close_matrices(high.mu, none.mu, rtol=1e-5)
    assert_close_matrices(high.y, none.y, rtol=1e-5)


def test_beta_shared_mean(dgps: DGP) -> None:
    """Test that beta_shared has mean close to 0."""
    assert_mean_z(dgps.params.beta_shared, 0.0)


def test_beta_separate_mean(dgps: DGP) -> None:
    """Test that beta_separate has mean close to 0."""
    assert_mean_z(nnone(dgps.params.beta_separate), 0.0)


def test_default_coefficients_have_equal_magnitude(keys: split) -> None:
    """The default random-sign draws give every predictor exactly equal weight.

    With ``s_distr=None`` and `DiscreteUniform` (m=2) coefficient families, the
    linear coefficients all have modulus sqrt(sigma2_lin / p) and the nonzero
    quadratic entries share a single modulus, so without scales the importance
    profile is exactly flat.
    """
    dgp = gen_data(keys.pop(), lambda_=0.5, **KWARGS)
    p = KWARGS['p']
    expected = jnp.sqrt(dgp.params.sigma2_lin / p)
    assert_close_matrices(
        jnp.abs(dgp.params.beta_shared), jnp.full(p, expected), rtol=1e-6
    )
    a_mags = jnp.abs(dgp.params.A_shared)[dgp.params.A_shared != 0]
    assert_close_matrices(a_mags, jnp.full_like(a_mags, a_mags[0]), rtol=1e-6)


WHICH_PARAMS = (
    'mulin_shared',
    'mulin_separate',
    'mulin',
    'muquad_shared',
    'muquad_separate',
    'muquad',
    'mu',
    'z',
    'y',
)

# Heteroskedastic DGP configurations: scalar het on the dense DGP, vector het
# on the sparse one (covering the het x sparsity interaction), and vector het
# with Normal projections (covering the kurt_gamma = 3 branch of ``var_v``).
HET_DGPS = (
    {'het_shape': 'scalar', 'het_strength': HET_STRENGTH},
    {'het_shape': 'vector', 'het_strength': HET_STRENGTH, 's_distr': Gamma(SPARSITY)},
    {'het_shape': 'vector', 'het_strength': HET_STRENGTH, 'gamma_distr': Normal()},
)
HET_DGPS_IDS = ('het_scalar', 'het_vector_sparse', 'het_vector_normal_gamma')

# Cases for the marginal-variance tests: homoskedastic (dense and sparse) over
# every field, plus the heteroskedastic DGPs, which must leave the ``z`` and
# ``y`` budgets unchanged since ``E[error_scale ** 2] == 1`` marginally. Het
# does not enter the latent mean fields at all (checked exactly by
# `test_stream_invariance_with_homoskedastic`), so only ``z`` and ``y`` are
# tested there. The binary-predictor DGP (kurtosis 1: the whole quadratic
# budget flows through the interactions) is tested on the quadratic field and
# the totals.
VARIANCE_CASES = tuple(
    pytest.param(dgp, which, id=f'{which}-{dgp_id}')
    for dgp, dgp_id, whiches in (
        ({}, 'dense', WHICH_PARAMS),
        ({'s_distr': Gamma(SPARSITY)}, 'sparse', WHICH_PARAMS),
        ({'x_distr': DiscreteUniform(2)}, 'binary_x', ('muquad', 'mu', 'z')),
        *((d, i, ('z', 'y')) for d, i in zip(HET_DGPS, HET_DGPS_IDS, strict=True)),
    )
    for which in whiches
)


def expected_variance(
    params: Params, which: str, *, prior: bool
) -> Float[Array, ' REPS']:
    """Return the prior (marginal) or expected population variance of `which`."""
    if which.startswith('mulin'):
        return params.sigma2_lin
    elif which.startswith('muquad'):
        quad = params.sigma2_quad
        return quad + params.sigma2_mean if prior else quad
    elif which == 'mu':
        total = params.sigma2_pri if prior else params.sigma2_pop
        return total - params.sigma2_eps
    elif which in ('z', 'y'):
        return params.sigma2_pri if prior else params.sigma2_pop
    else:  # pragma: no cover
        raise KeyError(which)


@pytest.mark.parametrize(('dgps', 'which'), VARIANCE_CASES, indirect=['dgps'])
def test_outcome_prior_variance(dgps: DGP, which: str) -> None:
    """Test that latent mean and outcome have the expected marginal variance.

    The budget holds with and without sparsity, and is preserved by
    heteroskedasticity (the noise multiplier is calibrated to unit mean).
    Sparsity and heteroskedasticity make ``y`` heavier-tailed, so the variance
    estimator spread is then set from the sample fourth moment instead of the
    Gaussian ``2 sigma^4`` (which would over-reject).
    """
    samples = getattr(dgps, which)  # Shape: (REPS, K?, N)
    n_reps, *_ = samples.shape

    var = jnp.var(samples, axis=0)  # Shape: (K?, N)
    expected_var = expected_variance(dgps.params, which, prior=True)[0].item()
    light_tailed = (
        isinstance(dgps.params.s_distr, Constant) and dgps.params.het_shape is None
    )
    if light_tailed:
        std_of_var = jnp.sqrt(2 * expected_var**2 / (n_reps - 1))
    else:
        fourth = jnp.mean((samples - jnp.mean(samples, axis=0)) ** 4, axis=0)
        std_of_var = jnp.sqrt((fourth - var**2) / n_reps)

    z_scores = jnp.abs((var - expected_var) / std_of_var)
    assert_array_less(z_scores, SIGMA_THRESHOLD)


@pytest.mark.parametrize(('dgps', 'which'), VARIANCE_CASES, indirect=['dgps'])
def test_outcome_pop_variance(dgps: DGP, which: str) -> None:
    """Test the expected population variance is on target.

    Holds with/without sparsity and under heteroskedasticity (since
    ``E[error_scale ** 2] == 1``). For the heteroskedastic cases the per-dataset
    variances are heavier-tailed, so the spread of their mean is estimated
    empirically rather than from the Gaussian ``2 sigma^4``.
    """
    samples = getattr(dgps, which)  # Shape: (REPS, K?, N)
    n_reps, *_ = samples.shape

    per_var = jnp.var(samples, axis=-1, ddof=1)  # Shape: (REPS, K?)
    expected_var = expected_variance(dgps.params, which, prior=False)[0].item()
    if dgps.params.het_shape is None:
        se = jnp.sqrt(2 * expected_var**2 / (n_reps - 1))
    else:
        se = None  # heavier-tailed per-dataset variances: estimate the spread
    assert_mean_z(per_var, expected_var, se=se)


@pytest.mark.parametrize('dgps', HET_DGPS, indirect=True, ids=HET_DGPS_IDS)
def test_error_scale_moments(dgps: DGP) -> None:
    """The noise multiplier has ``E[W ** 2] == 1`` and ``Var[W ** 2] == var_v``.

    The unit mean is the marginal relationship underlying the ``y`` cases of
    `test_outcome_pop_variance` / `test_outcome_prior_variance`: the het
    normalization is marginal, not per-instance, so the noise budget is
    preserved only in expectation over datasets. The dispersion matches the
    closed-form ``var_v``, which is identical across components, so the
    per-component estimates are compared to the single scalar.
    """
    w2 = nnone(dgps.error_scale) ** 2  # Shape: (REPS, K?, N)
    assert_mean_z(jnp.mean(w2, axis=-1), 1.0)
    assert_mean_z(jnp.mean((w2 - 1.0) ** 2, axis=-1), nnone(dgps.params.var_v)[0])


def test_variance_relationships(dgps: DGP) -> None:
    """Check some simple inequalities on variances."""
    assert jnp.all(dgps.params.sigma2_pri >= 0)
    assert jnp.all(dgps.params.sigma2_pop >= 0)
    assert jnp.all(dgps.params.sigma2_mean >= 0)
    assert jnp.all(dgps.params.sigma2_pri >= dgps.params.sigma2_pop)
    assert jnp.all(dgps.params.sigma2_pop >= dgps.params.sigma2_eps)


# A fixed positive-definite correlation matrix to exercise the error copula
# (k == 3 == KWARGS['k']); the off-diagonals are deliberately asymmetric in sign.
ERROR_CORR = jnp.array([[1.0, 0.5, -0.3], [0.5, 1.0, 0.2], [-0.3, 0.2, 1.0]])


@jit
def generate_corr_dgps(
    keys: Key[Array, ' REPS'], error_corr: Float[Array, 'k k']
) -> DGP:
    """Generate one dataset per key sharing a fixed across-component error correlation."""
    gen = partial(gen_data, lambda_=0.5, error_corr=error_corr, **KWARGS)
    return vmap(gen)(keys)


class TestErrorCorrelation:
    """Test the across-component error correlation (Gaussian copula)."""

    def test_default_is_independent(self, keys: split) -> None:
        """Without `error_corr` no error Cholesky factor is stored."""
        dgp = gen_data(keys.pop(), lambda_=0.5, **KWARGS)
        assert dgp.params.error_chol is None

    def test_cholesky_factors_correlation(self, keys: split) -> None:
        """`error_corr` yields a (k, k) lower-triangular factor with ``L Lᵀ == R``."""
        k = KWARGS['k']
        dgp = gen_data(keys.pop(), lambda_=0.5, error_corr=ERROR_CORR, **KWARGS)
        chol = nnone(dgp.params.error_chol)
        assert chol.shape == (k, k)
        assert jnp.all(jnp.triu(chol, 1) == 0)  # lower-triangular
        assert_close_matrices(chol @ chol.T, ERROR_CORR, rtol=1e-6)

    def test_identity_matches_independent(self, keys: split) -> None:
        """``error_corr = I`` reproduces the independent-error outcome."""
        k = KWARGS['k']
        key = keys.pop()
        indep = gen_data(key, lambda_=0.5, **KWARGS)
        ident = gen_data(
            random.clone(key), lambda_=0.5, error_corr=jnp.eye(k), **KWARGS
        )
        assert_close_matrices(ident.z, indep.z, rtol=1e-6)
        assert_close_matrices(ident.y, indep.y, rtol=1e-6)

    def test_normalized_unconditionally(self, keys: split) -> None:
        """A covariance and its correlation give the same errors (unit-diagonal norm)."""
        scale = jnp.array([2.0, 3.0, 4.0])
        cov = ERROR_CORR * scale[:, None] * scale[None, :]
        key = keys.pop()
        corr = gen_data(key, lambda_=0.5, error_corr=ERROR_CORR, **KWARGS)
        scaled = gen_data(random.clone(key), lambda_=0.5, error_corr=cov, **KWARGS)
        assert_close_matrices(scaled.z, corr.z, rtol=1e-5)

    def test_realized_correlation(self, keys: split) -> None:
        """The sampled error vectors have empirical correlation equal to `error_corr`."""
        k = KWARGS['k']
        dgps = generate_corr_dgps(keys.pop(REPS), ERROR_CORR)
        resid = (dgps.z - dgps.mu).transpose(1, 0, 2).reshape(k, -1)  # (k, REPS*N)
        assert_close_matrices(jnp.corrcoef(resid), ERROR_CORR, rtol=0.02)

    def test_preserves_marginal_variance(self, keys: split) -> None:
        """Correlating errors leaves each component's noise variance at ``sigma2_eps``."""
        k = KWARGS['k']
        dgps = generate_corr_dgps(keys.pop(REPS), ERROR_CORR)
        resid = (dgps.z - dgps.mu).transpose(1, 0, 2).reshape(k, -1)  # (k, REPS*N)
        var = jnp.var(resid, axis=1)
        assert_close_matrices(var, jnp.full(k, KWARGS['sigma2_eps']), rtol=0.02)


@pytest.mark.parametrize('dgps', [{'lambda_': 0.0}], indirect=True, ids=['lambda0'])
@pytest.mark.parametrize(
    'which',
    [
        'params.beta_separate',
        'mulin_separate',
        'mulin',
        'muquad_separate',
        'muquad',
        'mu',
        'y',
    ],
)
def test_rows_independent(dgps: DGP, which: str) -> None:
    """Test that rows are independent when lambda_=0.

    ``params.beta_separate`` does not depend on ``lambda_``, so its check
    covers every coupling.
    """
    samples = attrgetter(which)(dgps)  # Shape: (REPS, K, N or P)
    assert_uncorrelated_z(samples[:, 0, :], samples[:, 1, :])


@pytest.mark.parametrize('dgps', [{'lambda_': 1.0}], indirect=True, ids=['lambda1'])
@pytest.mark.parametrize('which', ['mulin', 'muquad', 'mu'])
def test_rows_identical(dgps: DGP, which: str) -> None:
    """Test that rows are identical when lambda_=1."""
    samples = getattr(dgps, which)  # Shape: (REPS, K, N)

    # Check that all rows are identical within each sample
    diffs = jnp.max(
        jnp.abs(samples[:, 0:1, :] - samples), axis=(1, 2)
    )  # Shape: (REPS,)
    assert_close_matrices(diffs, jnp.zeros_like(diffs), atol=1e-5)


class TestInteractionPattern:
    """Test the interaction_pattern function."""

    @pytest.fixture
    def pattern(self) -> Bool[Array, 'p p']:
        """Return the predictor interaction pattern."""
        return interaction_pattern(p=KWARGS['p'], q=KWARGS['q'])

    def test_symmetry(self, pattern: Bool[Array, 'p p']) -> None:
        """Test that interaction pattern is symmetric."""
        assert_array_equal(pattern, pattern.T)

    def test_diagonal(self, pattern: Bool[Array, 'p p']) -> None:
        """Test that diagonal is True."""
        assert_array_equal(jnp.diag(pattern), True, strict=False)

    def test_row_sums(self, pattern: Bool[Array, 'p p']) -> None:
        """Test that each row sums to q+1."""
        row_sums = jnp.sum(pattern, axis=1)
        assert_array_equal(row_sums, KWARGS['q'] + 1, strict=False)


class TestPartitionedInteractionPattern:
    """Test the partitioned_interaction_pattern function."""

    @pytest.fixture
    def partition(self, keys: split) -> Bool[Array, 'k p']:
        """Generate a partition of predictors."""
        return generate_partition(keys.pop(), p=KWARGS['p'], k=KWARGS['k'])

    @pytest.fixture
    def q(self) -> int:
        """Fix a value of the `q` parameter (must be even and < p // k)."""
        return KWARGS['q']

    @pytest.fixture
    def pattern(self, partition: Bool[Array, 'k p'], q: int) -> Bool[Array, 'k p p']:
        """Generate a multivariate interaction pattern that respects `partition`."""
        return partitioned_interaction_pattern(partition, q=q)

    def test_respects_partition(
        self, partition: Bool[Array, 'k p'], pattern: Bool[Array, 'k p p']
    ) -> None:
        """Test that pattern only has True values within partition blocks."""
        # For each component, check that True values only occur where partition is True
        # pattern[i, r, s] can only be True if partition[i, r] and partition[i, s] are True
        mask = partition[:, :, None] & partition[:, None, :]  # Shape: (k, p, p)
        assert_array_equal(pattern & ~mask, False, strict=False)

    def test_diagonal_within_partition(
        self, partition: Bool[Array, 'k p'], pattern: Bool[Array, 'k p p']
    ) -> None:
        """Test that the diagonal is True exactly on the partition."""
        assert_array_equal(jnp.diagonal(pattern, axis1=1, axis2=2), partition)

    def test_row_sums(
        self, pattern: Bool[Array, 'k p p'], partition: Bool[Array, 'k p'], q: int
    ) -> None:
        """Test that each row sums to q+1."""
        row_sums = jnp.sum(pattern, axis=2)
        target = jnp.where(partition, q + 1, 0)
        assert_array_equal(row_sums, target)

    def test_symmetry(self, pattern: Bool[Array, 'k p p']) -> None:
        """Test that interaction pattern is symmetric."""
        assert_array_equal(pattern, jnp.swapaxes(pattern, 1, 2))


def test_univariate(keys: split) -> None:
    """Check that k=None skips the separate path.

    At k=1/lambda_=1, the univariate output is the k=1 multivariate output
    with the leading axis squeezed away.
    """
    key = keys.pop()
    kw_mv: kwdict = dict(KWARGS, lambda_=1.0, k=1)
    kw_uv: kwdict = dict(KWARGS, lambda_=None, k=None)
    dgp_mv = gen_data(key, **kw_mv)
    dgp_uv = gen_data(random.clone(key), **kw_uv)

    assert dgp_uv.params.partition is None
    assert dgp_uv.params.beta_separate is None
    assert dgp_uv.params.A_separate is None
    assert dgp_uv.params.lambda_ is None
    assert dgp_uv.mulin_separate is None
    assert dgp_uv.muquad_separate is None
    assert dgp_mv.params.partition is not None

    assert_array_equal(dgp_uv.x, dgp_mv.x)
    assert_array_equal(dgp_uv.params.beta_shared, dgp_mv.params.beta_shared)
    assert_array_equal(dgp_uv.params.A_shared, dgp_mv.params.A_shared)
    assert_array_equal(dgp_uv.mulin_shared, dgp_mv.mulin_shared)
    assert_array_equal(dgp_uv.muquad_shared, dgp_mv.muquad_shared)
    assert_array_equal(dgp_uv.mulin, dgp_mv.mulin.squeeze(0))
    assert_array_equal(dgp_uv.muquad, dgp_mv.muquad.squeeze(0))
    assert_array_equal(dgp_uv.mu, dgp_mv.mu.squeeze(0))
    assert_array_equal(dgp_uv.z, dgp_mv.z.squeeze(0))
    assert_array_equal(dgp_uv.y, dgp_mv.y.squeeze(0))


def test_lambda_required_when_multivariate(keys: split) -> None:
    """`lambda_=None` with `k` set raises `ValueError`."""
    with pytest.raises(ValueError, match='lambda_ is required'):
        gen_data(keys.pop(), lambda_=None, **KWARGS)


def test_lambda_forbidden_when_univariate(keys: split) -> None:
    """`lambda_` not None with `k=None` raises `ValueError`."""
    kw: kwdict = dict(KWARGS, k=None)
    with pytest.raises(ValueError, match='lambda_ must be None'):
        gen_data(keys.pop(), lambda_=0.5, **kw)


def test_error_corr_forbidden_when_univariate(keys: split) -> None:
    """`error_corr` with `k=None` raises `ValueError`."""
    kw: kwdict = dict(KWARGS, k=None)
    with pytest.raises(ValueError, match='error_corr requires a multivariate'):
        gen_data(keys.pop(), lambda_=None, error_corr=jnp.eye(2), **kw)


def test_error_corr_wrong_shape(keys: split) -> None:
    """`error_corr` whose shape is not `(k, k)` raises `ValueError`."""
    with pytest.raises(ValueError, match='error_corr has shape'):
        gen_data(keys.pop(), lambda_=0.5, error_corr=jnp.eye(2), **KWARGS)


def test_m_too_small(keys: split) -> None:
    """``DiscreteUniform`` with ``m < 2`` raises.

    The runtime `error_if` surfaces only on synchronization (hence the
    `block_until_ready`), wrapped in an exception type that depends on the
    jit dispatch path (fresh compile vs. cache hit).
    """
    with pytest.raises((JaxRuntimeError, ValueError), match='m must be >= 2'):
        block_until_ready(
            gen_data(keys.pop(), x_distr=DiscreteUniform(1), lambda_=0.5, **KWARGS)
        )


def test_binary_x_requires_interactions(keys: split) -> None:
    """Binary x with ``q < 2`` raises: squares of binary predictors are constant.

    See `test_m_too_small` for the error handling details.
    """
    kw: kwdict = dict(KWARGS, q=0)
    with pytest.raises((JaxRuntimeError, ValueError), match='q must be >= 2'):
        block_until_ready(
            gen_data(keys.pop(), x_distr=DiscreteUniform(2), lambda_=0.5, **kw)
        )


def test_binary_x_min_interactions(keys: split) -> None:
    """Binary x with ``q=2`` is accepted and the quadratic term is not constant."""
    kw: kwdict = dict(KWARGS, q=2)
    dgp = gen_data(keys.pop(), x_distr=DiscreteUniform(2), lambda_=0.5, **kw)
    assert jnp.all(jnp.var(dgp.muquad, axis=-1) > 0)


@pytest.mark.parametrize(
    ('k', 'het_shape'),
    [(None, None), (None, 'scalar'), (3, None), (3, 'scalar'), (3, 'vector')],
)
def test_split(
    keys: split, k: int | None, het_shape: Literal['scalar', 'vector'] | None
) -> None:
    """`DGP.split()` slices every data field (incl. `error_scale`).

    Also checks the shared/blended fields are sliced consistently and that the
    univariate `None` separate fields are preserved.
    """
    lambda_ = None if k is None else 0.5
    het_strength = None if het_shape is None else HET_STRENGTH
    kw: kwdict = dict(KWARGS, k=k)
    dgp = gen_data(
        keys.pop(),
        lambda_=lambda_,
        het_strength=het_strength,
        het_shape=het_shape,
        **kw,
    )
    n_train = kw['n'] // 3
    train, test = dgp.split(n_train)

    for part, length in ((train, n_train), (test, kw['n'] - n_train)):
        assert part.x.shape == (kw['p'], length)
        core = (length,) if k is None else (k, length)
        assert part.y.shape == core
        assert part.z.shape == core
        assert part.mulin.shape == core
        assert part.muquad.shape == core
        assert part.mu.shape == core
        assert part.mulin_shared.shape == (length,)
        assert part.muquad_shared.shape == (length,)
        if k is None:
            assert part.params.partition is None
            assert part.params.beta_separate is None
            assert part.params.A_separate is None
            assert part.mulin_separate is None
            assert part.muquad_separate is None
        if het_shape is None:
            assert part.error_scale is None
        elif het_shape == 'scalar':
            assert nnone(part.error_scale).shape == (length,)
        else:
            assert nnone(part.error_scale).shape == (k, length)


def test_split_default_halves(keys: split) -> None:
    """`DGP.split()` without `n_train` splits the observations in half."""
    dgp = gen_data(keys.pop(), lambda_=0.5, **KWARGS)
    train, test = dgp.split()
    assert train.x.shape[1] == KWARGS['n'] // 2
    assert test.x.shape[1] == KWARGS['n'] - KWARGS['n'] // 2


class TestOutcomeType:
    """Tests for the `outcome_type` parameter of `gen_data`."""

    def test_default_is_continuous(self, keys: split) -> None:
        """Default `outcome_type` stores `OutcomeType.continuous` on Params."""
        dgp = gen_data(keys.pop(), lambda_=0.5, **KWARGS)
        assert dgp.params.outcome_type is OutcomeType.continuous

    def test_binary_univariate(self, keys: split) -> None:
        """Binary univariate output contains only 0.0/1.0 float values."""
        kw: kwdict = dict(KWARGS, k=None, outcome_type='binary')
        dgp = gen_data(keys.pop(), **kw)
        assert dgp.y.shape == (kw['n'],)
        assert dgp.y.dtype == jnp.float32
        assert_array_equal(jnp.unique(dgp.y), jnp.array([0.0, 1.0]))
        assert_array_equal(dgp.y, (dgp.z > 0).astype(dgp.y.dtype))
        assert dgp.params.outcome_type is OutcomeType.binary

    def test_binary_multivariate(self, keys: split) -> None:
        """Binary multivariate output contains only 0.0/1.0 in every row."""
        kw: kwdict = dict(KWARGS, lambda_=0.5, outcome_type='binary')
        dgp = gen_data(keys.pop(), **kw)
        assert dgp.y.shape == (kw['k'], kw['n'])
        assert_array_equal(jnp.unique(dgp.y), jnp.array([0.0, 1.0]))

    def test_binary_threshold_matches_latent(self, keys: split) -> None:
        """With the same key, binary `y` thresholds the same latent `z`.

        The key flow in `gen_data_from_params` is independent of outcome_type,
        so generating with `'continuous'` and `'binary'` under the same top
        level key must yield the same latent `z`, with `y` equal to `z` or to
        its threshold respectively.
        """
        kw: kwdict = dict(KWARGS, lambda_=0.5)
        key = keys.pop()
        dgp_cont = gen_data(key, **kw)
        dgp_bin = gen_data(random.clone(key), outcome_type='binary', **kw)
        assert_array_equal(dgp_cont.y, dgp_cont.z)
        assert_array_equal(dgp_bin.z, dgp_cont.z)
        assert_array_equal(dgp_bin.y, (dgp_bin.z > 0).astype(dgp_bin.z.dtype))

    def test_mixed(self, keys: split) -> None:
        """Mixed outcome_type: binary rows are 0/1, continuous rows match the baseline."""
        kw: kwdict = dict(KWARGS, lambda_=0.5)
        assert kw['k'] == 3
        key = keys.pop()
        dgp_cont = gen_data(key, **kw)
        dgp_mix = gen_data(
            random.clone(key), outcome_type=('continuous', 'binary', 'continuous'), **kw
        )
        assert dgp_mix.params.outcome_type == (
            OutcomeType.continuous,
            OutcomeType.binary,
            OutcomeType.continuous,
        )
        # the latent is independent of the outcome types
        assert_array_equal(dgp_mix.z, dgp_cont.z)
        # continuous rows unchanged
        assert_array_equal(dgp_mix.y[0], dgp_cont.y[0])
        assert_array_equal(dgp_mix.y[2], dgp_cont.y[2])
        # binary row is the threshold of the latent
        assert_array_equal(dgp_mix.y[1], (dgp_mix.z[1] > 0).astype(dgp_mix.z.dtype))

    def test_all_same_tuple_collapses_to_scalar(self, keys: split) -> None:
        """A tuple of identical types is stored as a scalar `OutcomeType`."""
        kw: kwdict = dict(
            KWARGS, lambda_=0.5, outcome_type=('binary', 'binary', 'binary')
        )
        dgp = gen_data(keys.pop(), **kw)
        assert dgp.params.outcome_type is OutcomeType.binary

    def test_validation_tuple_wrong_length(self, keys: split) -> None:
        """A tuple whose length does not match `k` raises `ValueError`."""
        kw: kwdict = dict(KWARGS, lambda_=0.5, outcome_type=('continuous', 'binary'))
        with pytest.raises(ValueError, match='outcome_type has length'):
            gen_data(keys.pop(), **kw)

    def test_validation_tuple_with_univariate(self, keys: split) -> None:
        """A tuple combined with the univariate path (`k is None`) raises."""
        kw: kwdict = dict(KWARGS, k=None, outcome_type=('continuous',))
        with pytest.raises(ValueError, match='tuple outcome_type requires'):
            gen_data(keys.pop(), **kw)


class TestOffset:
    """Tests for the `offset` parameter of `gen_data`."""

    def test_default_is_zero(self, keys: split) -> None:
        """Without `offset`, `params.offset` is 0."""
        dgp = gen_data(keys.pop(), lambda_=0.5, **KWARGS)
        assert dgp.params.offset == 0.0

    @pytest.mark.parametrize(
        'offset',
        [jnp.float32(0.7), jnp.array([-1.0, 0.0, 2.0])],
        ids=['scalar', 'vector'],
    )
    def test_shifts_latent_mean(
        self, keys: split, offset: Float[Array, ' k'] | Float[Array, '']
    ) -> None:
        """`offset` adds a constant to `mu` (hence `z`, `y`) after the lin/quad terms.

        A scalar shifts every component equally; a length-`k` vector shifts each
        one independently. Reusing the key, the only change from the zero-offset
        run is the shift; the linear and quadratic parts are untouched.
        """
        key = keys.pop()
        base = gen_data(key, lambda_=0.5, **KWARGS)
        shifted = gen_data(random.clone(key), lambda_=0.5, offset=offset, **KWARGS)
        assert_array_equal(shifted.params.offset, offset)
        assert_array_equal(shifted.mulin, base.mulin)
        assert_array_equal(shifted.muquad, base.muquad)
        offset = offset[..., None]  # broadcast like gen_data_from_params
        assert_array_equal(shifted.mu, base.mu + offset)
        assert_close_matrices(shifted.z, base.z + offset, rtol=1e-5)
        assert_close_matrices(shifted.y, base.y + offset, rtol=1e-5)

    def test_vector_requires_multivariate(self, keys: split) -> None:
        """A vector `offset` with `k=None` raises."""
        kw: kwdict = dict(KWARGS, k=None)
        with pytest.raises(ValueError, match='vector offset requires'):
            gen_data(keys.pop(), offset=jnp.zeros(3), **kw)

    def test_vector_wrong_length(self, keys: split) -> None:
        """A vector `offset` whose length is not `k` raises."""
        assert KWARGS['k'] != 2
        with pytest.raises(ValueError, match='offset has length 2 but k=3'):
            gen_data(keys.pop(), lambda_=0.5, offset=jnp.zeros(2), **KWARGS)

    def test_controls_binary_rate(self, keys: split) -> None:
        """A large `offset` drives the binary success probability to 0 or 1.

        This is the motivating use case: binary outcomes are thresholded, so the
        only way to move their base rate is to shift the latent mean.
        """
        kw: kwdict = dict(KWARGS, lambda_=0.5, outcome_type='binary')
        high = gen_data(keys.pop(), offset=10.0, **kw)
        low = gen_data(keys.pop(), offset=-10.0, **kw)
        assert jnp.mean(high.y) > 0.95
        assert jnp.mean(low.y) < 0.05


class TestHeteroskedasticity:
    """Tests for the heteroskedasticity feature of `gen_data`."""

    def test_homoskedastic_fields_are_none(self, keys: split) -> None:
        """Without heteroskedasticity every het field (incl. `error_scale`) is None."""
        dgp = gen_data(keys.pop(), lambda_=0.5, **KWARGS)
        assert dgp.error_scale is None
        assert dgp.params.gamma_shared is None
        assert dgp.params.gamma_separate is None
        assert dgp.params.var_v is None
        assert dgp.params.het_strength is None
        assert dgp.params.het_shape is None

    def test_vector_shapes_and_dtypes(self, keys: split) -> None:
        """Vector het exposes (k, n) scales and per-component coefficients."""
        n, p, k = KWARGS['n'], KWARGS['p'], KWARGS['k']
        dgp = gen_data(
            keys.pop(),
            lambda_=0.5,
            het_shape='vector',
            het_strength=HET_STRENGTH,
            **KWARGS,
        )
        error_scale = nnone(dgp.error_scale)
        assert error_scale.shape == (k, n)
        assert jnp.issubdtype(error_scale.dtype, jnp.floating)
        assert jnp.all(error_scale > 0)
        assert nnone(dgp.params.gamma_shared).shape == (p,)
        assert nnone(dgp.params.gamma_separate).shape == (k, p)
        assert nnone(dgp.params.var_v).shape == ()
        assert nnone(dgp.params.het_strength).shape == ()
        assert dgp.params.het_shape == 'vector'

    def test_scalar_shapes(self, keys: split) -> None:
        """Scalar het exposes one (n,) scale and no separate coefficients."""
        dgp = gen_data(
            keys.pop(),
            lambda_=0.5,
            het_shape='scalar',
            het_strength=HET_STRENGTH,
            **KWARGS,
        )
        assert nnone(dgp.error_scale).shape == (KWARGS['n'],)
        assert dgp.params.gamma_separate is None
        assert nnone(dgp.params.var_v).shape == ()
        assert dgp.params.het_shape == 'scalar'

    def test_univariate_scalar(self, keys: split) -> None:
        """Univariate (`k=None`) het produces an (n,) scale."""
        kw: kwdict = dict(KWARGS, k=None)
        dgp = gen_data(keys.pop(), het_shape='scalar', het_strength=HET_STRENGTH, **kw)
        assert nnone(dgp.error_scale).shape == (kw['n'],)
        assert dgp.y.shape == (kw['n'],)
        assert dgp.params.gamma_separate is None

    def test_heterogeneity_grows_with_strength(self, keys: split) -> None:
        """Larger ``het_strength`` spreads the noise-variance multiplier more.

        ``error_scale`` is flat at strength 0 and increasingly dispersed toward
        the maximally heterogeneous strength-1 limit (with the mean and predictor
        streams held fixed by reusing the same key).
        """
        key = keys.pop()
        kw: kwdict = dict(KWARGS, lambda_=0.5, het_shape='vector')
        spreads = jnp.array(
            [
                jnp.var(
                    nnone(
                        gen_data(random.clone(key), het_strength=rho, **kw).error_scale
                    )
                    ** 2
                )
                for rho in (0.0, 0.3, 0.6, 1.0)
            ]
        )
        assert_array_less(spreads[:-1], spreads[1:])

    @pytest.mark.parametrize('gamma_distr', [DiscreteUniform(2), Normal()])
    def test_var_v_matches_closed_form(self, keys: split, gamma_distr: Distr) -> None:
        """``var_v`` is the marginal ``Var[W ** 2]`` closed form of `Params`.

        Recomputes it from the public hyperparameters (vector het, with sparsity
        so ``E[s ** 4]`` and the partition factor both enter, and both the
        default and the Normal ``kappa_gamma``), catching wiring bugs.
        """
        rho, lambda_ = HET_STRENGTH, 0.5
        p, k = KWARGS['p'], KWARGS['k']
        dgp = gen_data(
            keys.pop(),
            lambda_=lambda_,
            s_distr=Gamma(SPARSITY),
            gamma_distr=gamma_distr,
            het_shape='vector',
            het_strength=rho,
            **KWARGS,
        )
        kurt_x_mu_4 = dgp.params.x_distr.kurtosis * dgp.params.s_distr.fourth_moment
        kurt_eta = dgp.params.gamma_distr.kurtosis * kurt_x_mu_4
        big_lambda = lambda_**2 + (1 - lambda_) ** 2 * k
        cross = 6 * lambda_ * (1 - lambda_) * (kurt_x_mu_4 - 1)
        r = p % k
        excess = ((kurt_eta - 3) * big_lambda + cross) / p + 3 * (
            1 - lambda_
        ) ** 2 * r * (k - r) / p**2
        assert_close_matrices(nnone(dgp.params.var_v), rho**2 * (2 + excess), rtol=1e-6)

    def test_stream_invariance_with_homoskedastic(self, keys: split) -> None:
        """Het leaves the mean/predictor streams intact and only rescales the noise."""
        kw: kwdict = dict(KWARGS, lambda_=0.5)
        key = keys.pop()
        homo = gen_data(key, **kw)
        het = gen_data(
            random.clone(key), het_shape='vector', het_strength=HET_STRENGTH, **kw
        )
        assert_array_equal(homo.x, het.x)
        assert_array_equal(homo.params.beta_shared, het.params.beta_shared)
        assert_array_equal(nnone(homo.params.A_separate), nnone(het.params.A_separate))
        assert_array_equal(nnone(homo.params.partition), nnone(het.params.partition))
        assert_array_equal(homo.params.s, het.params.s)
        assert_array_equal(homo.mu, het.mu)
        # the heteroskedastic noise is the homoskedastic noise scaled by error_scale
        recon = het.mu + (homo.y - homo.mu) * nnone(het.error_scale)
        assert_close_matrices(het.y, recon, rtol=1e-5)

    @pytest.mark.parametrize('het_shape', ['scalar', 'vector'])
    def test_zero_strength_is_homoskedastic(
        self, keys: split, het_shape: Literal['scalar', 'vector']
    ) -> None:
        """``het_strength=0`` collapses het back to the homoskedastic model.

        The variance multiplier is the flat floor ``1 - rho = 1`` (the
        projection coefficients are still drawn but do not enter), so
        ``error_scale`` is exactly 1 and the mean is untouched; the outcome then
        matches the ``het_shape=None`` run up to float32 reordering of the noise.
        """
        kw: kwdict = dict(KWARGS, lambda_=0.5)
        key = keys.pop()
        homo = gen_data(key, **kw)
        het = gen_data(random.clone(key), het_shape=het_shape, het_strength=0.0, **kw)

        error_scale = nnone(het.error_scale)
        assert_array_equal(error_scale, jnp.ones_like(error_scale))
        var_v = nnone(het.params.var_v)
        assert_array_equal(var_v, jnp.zeros_like(var_v))
        if het_shape == 'vector':
            assert het.params.gamma_separate is not None
        else:
            assert het.params.gamma_separate is None
        assert_array_equal(het.mu, homo.mu)
        assert_close_matrices(het.y, homo.y, rtol=1e-5)

    def test_scalar_and_vector_share_shared_stream(self, keys: split) -> None:
        """``'scalar'`` and ``'vector'`` het share the ``gamma_shared`` stream."""
        kw: kwdict = dict(KWARGS, lambda_=0.5, het_strength=HET_STRENGTH)
        key = keys.pop()
        scalar = gen_data(key, het_shape='scalar', **kw)
        vector = gen_data(random.clone(key), het_shape='vector', **kw)
        assert_array_equal(
            nnone(scalar.params.gamma_shared), nnone(vector.params.gamma_shared)
        )

    def test_vector_requires_multivariate(self, keys: split) -> None:
        """``het_shape='vector'`` with ``k=None`` raises."""
        kw: kwdict = dict(KWARGS, k=None)
        with pytest.raises(ValueError, match="het_shape='vector' requires"):
            gen_data(keys.pop(), het_shape='vector', het_strength=HET_STRENGTH, **kw)

    def test_strength_and_shape_must_agree(self, keys: split) -> None:
        """``het_strength`` and ``het_shape`` must be both set or both None."""
        with pytest.raises(ValueError, match='both set or both None'):
            gen_data(keys.pop(), lambda_=0.5, het_strength=HET_STRENGTH, **KWARGS)
        with pytest.raises(ValueError, match='both set or both None'):
            gen_data(keys.pop(), lambda_=0.5, het_shape='scalar', **KWARGS)

    def test_binary_thresholds_het_latent(self, keys: split) -> None:
        """Het works for binary outcomes: `y` thresholds the heteroskedastic latent.

        With a shared key the continuous and binary runs differ only by the final
        threshold, so the binary success probability is
        ``Phi(mu / (sqrt(sigma2_eps) * error_scale))``.
        """
        kw: kwdict = dict(
            KWARGS, lambda_=0.5, het_shape='vector', het_strength=HET_STRENGTH
        )
        key = keys.pop()
        cont = gen_data(key, **kw)
        binary = gen_data(random.clone(key), outcome_type='binary', **kw)
        assert_array_equal(jnp.unique(binary.y), jnp.array([0.0, 1.0]))
        assert_array_equal(binary.z, cont.z)
        assert_array_equal(binary.y, (cont.y > 0).astype(cont.y.dtype))
