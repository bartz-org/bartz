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
from types import MappingProxyType

import pytest
from jax import jit, random, vmap
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Key
from numpy.testing import assert_array_less
from scipy.stats import norm

from bartz._jaxext import split
from bartz.mcmcstep import OutcomeType
from bartz.testing import DGP, Params, gen_data
from bartz.testing._dgp import (
    generate_partition,
    generate_s,
    het_normalization,
    interaction_pattern,
    log_var_mgf,
    partitioned_interaction_pattern,
)
from tests.util import assert_allclose, assert_array_equal, assert_close_matrices

# Test parameters
ALPHA = 5e-7  # probability of false positive (aaaaapprox)
SIGMA_THRESHOLD = norm.isf(ALPHA / 2)  # threshold for z tests
KWARGS: Mapping = MappingProxyType(
    dict(n=100, p=20, k=3, q=4, sigma2_eps=0.1, sigma2_lin=0.4, sigma2_quad=0.5)
)
REPS: int = 10_000  # number of datasets
SPARSITY: float = 2.0  # Gamma shape for the sparse-DGP fixture (mu4 = 10 / 3)


@partial(jit, static_argnames=('het_shape',))
def generate_dgps(
    keys: Key[Array, 'REPS'],
    lambda_: Float[Array, ''],
    sparsity: float | None,
    sigma2_logscale: float | None = None,
    het_shape: str | None = None,
) -> DGP:
    """Generate one dataset per random key (jitted, vmapped over keys)."""
    gen = partial(
        gen_data,
        lambda_=lambda_,
        sparsity=sparsity,
        sigma2_logscale=sigma2_logscale,
        het_shape=het_shape,
        **KWARGS,
    )
    return vmap(gen)(keys)


@pytest.fixture
def dgps(keys: split, request: pytest.FixtureRequest) -> DGP:
    """Generate DGP instances using vmap and jit.

    Indirectly parametrizable either by `sparsity` (default `None`, i.e. dense)
    or by a mapping with any of ``sparsity``, ``lambda_``, ``sigma2_logscale``
    and ``het_shape`` to also exercise heteroskedasticity.
    """
    param = getattr(request, 'param', None)
    if isinstance(param, Mapping):
        return generate_dgps(
            keys.pop(REPS),
            param.get('lambda_', 0.5),
            param.get('sparsity'),
            param.get('sigma2_logscale'),
            param.get('het_shape'),
        )
    else:
        return generate_dgps(keys.pop(REPS), 0.5, param)


@pytest.fixture
def dgps_lambda_zero(keys: split) -> DGP:
    """Generate DGP instances with lambda_=0."""
    return generate_dgps(keys.pop(REPS), 0.0, None)


@pytest.fixture
def dgps_lambda_one(keys: split) -> DGP:
    """Generate DGP instances with lambda_=1."""
    return generate_dgps(keys.pop(REPS), 1.0, None)


def test_shapes_and_dtypes(keys: split) -> None:
    """Test that all DGP attributes have correct shapes and dtypes."""
    dgp = gen_data(keys.pop(), lambda_=0.5, **KWARGS)
    n, p, k = KWARGS['n'], KWARGS['p'], KWARGS['k']

    # Test shapes
    assert dgp.x.shape == (p, n)
    assert dgp.y.shape == (k, n)
    assert dgp.params.partition.shape == (k, p)
    assert dgp.params.beta_shared.shape == (p,)
    assert dgp.params.beta_separate.shape == (k, p)
    assert dgp.mulin_shared.shape == (n,)
    assert dgp.mulin_separate.shape == (k, n)
    assert dgp.mulin.shape == (k, n)
    assert dgp.params.A_shared.shape == (p, p)
    assert dgp.params.A_separate.shape == (k, p, p)
    assert dgp.params.s.shape == (p,)
    assert dgp.muquad_shared.shape == (n,)
    assert dgp.muquad_separate.shape == (k, n)
    assert dgp.muquad.shape == (k, n)
    assert dgp.mu.shape == (k, n)
    assert dgp.params.q.shape == ()
    assert dgp.params.lambda_.shape == ()
    assert dgp.params.sigma2_lin.shape == ()
    assert dgp.params.sigma2_quad.shape == ()
    assert dgp.params.sigma2_eps.shape == ()

    # Test dtypes
    assert jnp.issubdtype(dgp.x.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.y.dtype, jnp.floating)
    assert dgp.params.partition.dtype == jnp.bool_
    assert jnp.issubdtype(dgp.params.beta_shared.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.params.beta_separate.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.mulin_shared.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.mulin_separate.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.mulin.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.params.A_shared.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.params.A_separate.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.params.s.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.muquad_shared.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.muquad_separate.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.muquad.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.mu.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.params.q.dtype, jnp.integer)
    assert jnp.issubdtype(dgp.params.lambda_.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.params.sigma2_lin.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.params.sigma2_quad.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.params.sigma2_eps.dtype, jnp.floating)


class TestGenerateX:
    """Test the _generate_x method."""

    def test_x_mean(self, dgps: DGP) -> None:
        """Test that x has mean close to 0."""
        x_samples = dgps.x  # Shape: (REPS, P, N)
        n_reps = x_samples.shape[0]

        # Compute mean and std of mean for each element
        means = jnp.mean(x_samples, axis=0)  # Shape: (P, N)
        stds_of_mean = jnp.std(x_samples, axis=0) / jnp.sqrt(n_reps)  # Shape: (P, N)

        # All means should be within SIGMA_THRESHOLD standard deviations of 0
        z_scores = jnp.abs(means / stds_of_mean)
        assert_array_less(z_scores, SIGMA_THRESHOLD)

    def test_x_variance(self, dgps: DGP) -> None:
        """Test that x has variance close to 1."""
        x_samples = dgps.x  # Shape: (REPS, P, N)
        n_reps = x_samples.shape[0]

        # Compute variance for each element
        var = jnp.var(x_samples, axis=0)  # Shape: (P, N)
        expected_var = 1.0

        # Standard deviation of sample variance for each element
        std_of_var = jnp.sqrt(2 * expected_var**2 / (n_reps - 1))

        # All variances should be within SIGMA_THRESHOLD standard deviations of 1
        z_scores = jnp.abs((var - expected_var) / std_of_var)
        assert_array_less(z_scores, SIGMA_THRESHOLD)


class TestGeneratePartition:
    """Test the _generate_partition method."""

    def test_partition_coverage(self, dgps: DGP) -> None:
        """Test that each predictor is assigned to exactly one component."""
        partitions = dgps.params.partition  # Shape: (REPS, K, P)

        # Each column should sum to 1
        col_sums = jnp.sum(partitions, axis=1)  # Shape: (N_REPS, P)
        assert_array_equal(col_sums, 1, strict=False)

    def test_partition_counts(self, dgps: DGP) -> None:
        """Test that counts are either p//c or p//c + 1."""
        partitions = dgps.params.partition  # Shape: (REPS, K, P)
        p, k = partitions.shape[2], partitions.shape[1]

        counts = jnp.sum(partitions, axis=2)  # Shape: (REPS, K)

        # Each count should be either P//K or P//K + 1
        floor_count = p // k
        valid = (counts == floor_count) | (counts == floor_count + 1)
        assert_array_equal(valid, True, strict=False)

    def test_partition_balance(self, dgps: DGP) -> None:
        """Test that predictors are roughly balanced across components."""
        partitions = dgps.params.partition  # Shape: (REPS, K, P)
        n_reps, k, p = partitions.shape

        counts = jnp.sum(partitions, axis=2)  # Shape: (REPS, K)

        # Mean count per component should be P/K
        expected_mean = p / k
        means = jnp.mean(counts, axis=0)  # Shape: (K,)
        stds_of_mean = jnp.std(counts, axis=0) / jnp.sqrt(n_reps)  # Shape: (K,)

        z_scores = jnp.abs((means - expected_mean) / stds_of_mean)
        assert_array_less(z_scores, SIGMA_THRESHOLD)


def s_moment(sparsity: float, power: int) -> float:
    """Analytic E[s ** power] for the scales drawn by `generate_s`.

    With ``s = Gamma(sparsity) / sqrt(sparsity (sparsity + 1))`` one has
    ``E[s ** m] = prod_{i<m}(sparsity + i) / (sparsity (sparsity + 1)) ** (m/2)``
    (``power`` must be even). In particular ``s_moment(a, 2) == 1``.
    """
    num = 1.0
    for i in range(power):
        num *= sparsity + i
    return num / (sparsity * (sparsity + 1)) ** (power // 2)


class TestGenerateS:
    """Test `generate_s` in isolation."""

    N: int = 1_000_000  # number of i.i.d. scales for the moment estimates
    ALPHA_S: float = 5.0  # a moderate shape, light enough tails for the z tests

    def test_none_is_ones(self, keys: split) -> None:
        """`sparsity=None` returns all-ones scales (uniform importance)."""
        assert_array_equal(generate_s(keys.pop(), 10, None), jnp.ones(10))

    def test_shape_and_dtype(self, keys: split) -> None:
        """Scales have shape (p,) and a floating dtype."""
        s = generate_s(keys.pop(), 7, self.ALPHA_S)
        assert s.shape == (7,)
        assert jnp.issubdtype(s.dtype, jnp.floating)

    def test_second_moment(self, keys: split) -> None:
        """The scales are normalized to ``E[s ** 2] == 1``."""
        s2 = generate_s(keys.pop(), self.N, self.ALPHA_S) ** 2
        var_s2 = s_moment(self.ALPHA_S, 4) - 1.0  # Var[s^2] = E[s^4] - E[s^2]^2
        std_of_mean = jnp.sqrt(var_s2 / self.N)
        z = jnp.abs((jnp.mean(s2) - 1.0) / std_of_mean)
        assert_array_less(z, SIGMA_THRESHOLD)

    def test_fourth_moment(self, keys: split) -> None:
        """``E[s ** 4]`` matches the analytic ``mu4`` used in the normalizers."""
        s4 = generate_s(keys.pop(), self.N, self.ALPHA_S) ** 4
        mu4 = s_moment(self.ALPHA_S, 4)
        var_s4 = s_moment(self.ALPHA_S, 8) - mu4**2
        std_of_mean = jnp.sqrt(var_s4 / self.N)
        z = jnp.abs((jnp.mean(s4) - mu4) / std_of_mean)
        assert_array_less(z, SIGMA_THRESHOLD)

    def test_dispersion_increases_with_sparsity(self, keys: split) -> None:
        """Smaller `sparsity` spreads the importances out more (larger Var[s^2])."""
        sparse = generate_s(keys.pop(), self.N, 1.0)
        dense = generate_s(keys.pop(), self.N, 10.0)
        assert jnp.var(sparse**2) > jnp.var(dense**2)


class TestGenerateBetaShared:
    """Test the _generate_beta_shared method."""

    def test_beta_shared_mean(self, dgps: DGP) -> None:
        """Test that beta_shared has mean close to 0."""
        beta_samples = dgps.params.beta_shared  # Shape: (REPS, P)
        n_reps = beta_samples.shape[0]

        means = jnp.mean(beta_samples, axis=0)  # Shape: (P,)
        stds_of_mean = jnp.std(beta_samples, axis=0) / jnp.sqrt(n_reps)  # Shape: (P,)

        z_scores = jnp.abs(means / stds_of_mean)
        assert_array_less(z_scores, SIGMA_THRESHOLD)


class TestGenerateBetaSeparate:
    """Test the _generate_beta_separate method."""

    def test_beta_separate_mean(self, dgps: DGP) -> None:
        """Test that beta_separate has mean close to 0."""
        beta_samples = dgps.params.beta_separate  # Shape: (REPS, K, P)
        n_reps = beta_samples.shape[0]

        means = jnp.mean(beta_samples, axis=0)  # Shape: (K, P)
        stds_of_mean = jnp.std(beta_samples, axis=0) / jnp.sqrt(n_reps)  # Shape: (K, P)

        z_scores = jnp.abs(means / stds_of_mean)
        assert_array_less(z_scores, SIGMA_THRESHOLD)

    def test_beta_separate_independence(self, dgps: DGP) -> None:
        """Test that rows of beta_separate are independent."""
        beta_samples = dgps.params.beta_separate  # Shape: (REPS, K, P)
        n_reps = beta_samples.shape[0]

        beta0 = beta_samples[:, 0, :]  # Shape: (REPS, P)
        beta1 = beta_samples[:, 1, :]  # Shape: (REPS, P)

        # Compute covariance for each predictor position
        mean0 = jnp.mean(beta0, axis=0)  # Shape: (P,)
        mean1 = jnp.mean(beta1, axis=0)  # Shape: (P,)
        cov = jnp.mean((beta0 - mean0) * (beta1 - mean1), axis=0)  # Shape: (P,)

        # Standard deviation of covariance estimate
        std0 = jnp.std(beta0, axis=0)
        std1 = jnp.std(beta1, axis=0)
        std_of_cov = (std0 * std1) / jnp.sqrt(n_reps)

        z_scores = jnp.abs(cov / std_of_cov)
        assert_array_less(z_scores, SIGMA_THRESHOLD)


WHICH_PARAMS = (
    'mulin_shared',
    'mulin_separate',
    'mulin',
    'muquad_shared',
    'muquad_separate',
    'muquad',
    'mu',
    'y',
)

# A modest log-variance budget keeps the noise-variance multiplier light-tailed
# enough that the marginal-variance estimators below do not blow up.
HET_LOGSCALE: float = 0.1

# DGP configurations for the marginal-variance tests: homoskedastic (dense and
# sparse) plus scalar and vector heteroskedasticity, which must leave every
# marginal variance unchanged since ``E[error_scale ** 2] == 1``.
VARIANCE_DGPS = (
    None,
    SPARSITY,
    {'het_shape': 'scalar', 'sigma2_logscale': HET_LOGSCALE},
    {'het_shape': 'vector', 'sigma2_logscale': HET_LOGSCALE},
)
VARIANCE_DGPS_IDS = ('dense', 'sparse', 'het_scalar', 'het_vector')


def expected_pop_var(params: Params, which: str) -> Float[Array, ' REPS']:
    """Return the expected population variance of DGP attribute `which`."""
    if which.startswith('mulin'):
        return params.sigma2_lin
    elif which.startswith('muquad'):
        return params.sigma2_quad
    elif which == 'mu':
        return params.sigma2_pop - params.sigma2_eps
    elif which == 'y':
        return params.sigma2_pop
    else:  # pragma: no cover
        raise KeyError(which)


def expected_prior_var(params: Params, which: str) -> Float[Array, ' REPS']:
    """Return the marginal prior variance of DGP attribute `which`."""
    if which.startswith('mulin'):
        return params.sigma2_lin
    elif which.startswith('muquad'):
        return params.sigma2_quad + params.sigma2_mean
    elif which == 'mu':
        return params.sigma2_pri - params.sigma2_eps
    elif which == 'y':
        return params.sigma2_pri
    else:  # pragma: no cover
        raise KeyError(which)


@pytest.mark.parametrize('dgps', VARIANCE_DGPS, indirect=True, ids=VARIANCE_DGPS_IDS)
@pytest.mark.parametrize('which', WHICH_PARAMS)
def test_outcome_prior_variance(dgps: DGP, which: str) -> None:
    """Test that latent mean and outcome have the expected marginal variance.

    The budget holds with and without sparsity, and is preserved by
    heteroskedasticity (the noise multiplier is calibrated to unit mean).
    Sparsity and heteroskedasticity make ``y`` heavier-tailed, so the variance
    estimator spread is then set from the sample fourth moment instead of the
    Gaussian ``2 sigma^4`` (which would over-reject).
    """
    samples = getattr(dgps, which)  # Shape: (REPS, K?, N)
    n_reps = samples.shape[0]

    var = jnp.var(samples, axis=0)  # Shape: (K?, N)
    expected_var = expected_prior_var(dgps.params, which)[0].item()
    light_tailed = dgps.params.sparsity is None and dgps.params.het_shape is None
    if light_tailed:
        std_of_var = jnp.sqrt(2 * expected_var**2 / (n_reps - 1))
    else:
        fourth = jnp.mean((samples - jnp.mean(samples, axis=0)) ** 4, axis=0)
        std_of_var = jnp.sqrt((fourth - var**2) / n_reps)

    z_scores = jnp.abs((var - expected_var) / std_of_var)
    assert_array_less(z_scores, SIGMA_THRESHOLD)


@pytest.mark.parametrize('dgps', VARIANCE_DGPS, indirect=True, ids=VARIANCE_DGPS_IDS)
@pytest.mark.parametrize('which', WHICH_PARAMS)
def test_outcome_pop_variance(dgps: DGP, which: str) -> None:
    """Test the expected population variance is on target.

    Holds with/without sparsity and under heteroskedasticity (since
    ``E[error_scale ** 2] == 1``). For the heteroskedastic cases the per-dataset
    variances are heavier-tailed, so the spread of their mean is estimated
    empirically rather than from the Gaussian ``2 sigma^4``.
    """
    samples = getattr(dgps, which)  # Shape: (REPS, K?, N)
    n_reps = samples.shape[0]

    per_var = jnp.var(samples, axis=-1, ddof=1)  # Shape: (REPS, K?)
    var = jnp.mean(per_var, axis=0)  # Shape: (K?,)
    expected_var = expected_pop_var(dgps.params, which)[0].item()
    if dgps.params.het_shape is None:
        std_of_var = jnp.sqrt(2 * expected_var**2 / (n_reps - 1))
    else:
        std_of_var = jnp.std(per_var, axis=0) / jnp.sqrt(n_reps)

    z_scores = jnp.abs((var - expected_var) / std_of_var)
    assert_array_less(z_scores, SIGMA_THRESHOLD)


def test_variance_relationships(dgps: DGP) -> None:
    """Check some simple inequalities on variances."""
    assert jnp.all(dgps.params.sigma2_pri >= 0)
    assert jnp.all(dgps.params.sigma2_pop >= 0)
    assert jnp.all(dgps.params.sigma2_mean >= 0)
    assert jnp.all(dgps.params.sigma2_pri >= dgps.params.sigma2_pop)
    assert jnp.all(dgps.params.sigma2_pop >= dgps.params.sigma2_eps)


@pytest.mark.parametrize(
    'which',
    [
        'beta_separate',
        'mulin_separate',
        'mulin',
        'muquad_separate',
        'muquad',
        'mu',
        'y',
    ],
)
def test_rows_independent(dgps_lambda_zero: DGP, which: str) -> None:
    """Test that rows are independent when lambda_=0."""
    source = dgps_lambda_zero.params if which == 'beta_separate' else dgps_lambda_zero
    samples = getattr(source, which)  # Shape: (REPS, K, N or P)
    n_reps = samples.shape[0]

    samples0 = samples[:, 0, :]  # Shape: (REPS, N or P)
    samples1 = samples[:, 1, :]  # Shape: (REPS, N or P)

    # Compute covariance for each observation position
    mean0 = jnp.mean(samples0, axis=0)
    mean1 = jnp.mean(samples1, axis=0)
    cov = jnp.mean((samples0 - mean0) * (samples1 - mean1), axis=0)

    std0 = jnp.std(samples0, axis=0)
    std1 = jnp.std(samples1, axis=0)
    std_of_cov = (std0 * std1) / jnp.sqrt(n_reps)

    z_scores = jnp.abs(cov / std_of_cov)
    assert_array_less(z_scores, SIGMA_THRESHOLD)


@pytest.mark.parametrize('which', ['mulin', 'muquad', 'mu'])
def test_rows_identical(dgps_lambda_one: DGP, which: str) -> None:
    """Test that rows are identical when lambda_=1."""
    samples = getattr(dgps_lambda_one, which)  # Shape: (REPS, K, N)

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
        return interaction_pattern(p=10, q=4)

    def test_symmetry(self, pattern: Bool[Array, 'p p']) -> None:
        """Test that interaction pattern is symmetric."""
        assert_array_equal(pattern, pattern.T)

    def test_diagonal(self, pattern: Bool[Array, 'p p']) -> None:
        """Test that diagonal is True."""
        assert_array_equal(jnp.diag(pattern), True, strict=False)

    def test_row_sums(self, pattern: Bool[Array, 'p p']) -> None:
        """Test that each row sums to q+1."""
        row_sums = jnp.sum(pattern, axis=1)
        assert_array_equal(row_sums, 4 + 1, strict=False)


class TestPartitionedInteractionPattern:
    """Test the partitioned_interaction_pattern function."""

    @pytest.fixture
    def partition(self, keys: split) -> Bool[Array, 'k p']:
        """Generate a partition of predictors."""
        p, k = 20, 3
        return generate_partition(keys.pop(), p=p, k=k)

    @pytest.fixture
    def q(self) -> int:
        """Fix a value of the `q` parameter."""
        return 2

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
        """Test that diagonal elements within partition are True."""
        k, _, _ = pattern.shape
        for i in range(k):
            diagonal = pattern[i, partition[i], partition[i]]
            assert_array_equal(diagonal, True, strict=False)

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
    kw = dict(KWARGS)
    kw.update(lambda_=1.0, k=1)
    dgp_mv = gen_data(key, **kw)
    kw.update(lambda_=None, k=None)
    dgp_uv = gen_data(random.clone(key), **kw)

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
    assert_array_equal(dgp_uv.y, dgp_mv.y.squeeze(0))


def test_lambda_required_when_multivariate(keys: split) -> None:
    """`lambda_=None` with `k` set raises `ValueError`."""
    with pytest.raises(ValueError, match='lambda_ is required'):
        gen_data(keys.pop(), lambda_=None, **KWARGS)


def test_lambda_forbidden_when_univariate(keys: split) -> None:
    """`lambda_` not None with `k=None` raises `ValueError`."""
    kw = dict(KWARGS, k=None)
    with pytest.raises(ValueError, match='lambda_ must be None'):
        gen_data(keys.pop(), lambda_=0.5, **kw)


@pytest.mark.parametrize(
    ('k', 'het_shape'),
    [(None, None), (None, 'scalar'), (3, None), (3, 'scalar'), (3, 'vector')],
)
def test_split(keys: split, k: int | None, het_shape: str | None) -> None:
    """`DGP.split()` slices every data field (incl. `error_scale`).

    Also checks the shared/blended fields are sliced consistently and that the
    univariate `None` separate fields are preserved.
    """
    lambda_ = None if k is None else 0.5
    sigma2_logscale = None if het_shape is None else 0.2
    kw = dict(KWARGS, k=k)
    dgp = gen_data(
        keys.pop(),
        lambda_=lambda_,
        sigma2_logscale=sigma2_logscale,
        het_shape=het_shape,
        **kw,
    )
    n_train = kw['n'] // 3
    train, test = dgp.split(n_train)

    for part, length in ((train, n_train), (test, kw['n'] - n_train)):
        assert part.x.shape == (kw['p'], length)
        core = (length,) if k is None else (k, length)
        assert part.y.shape == core
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
            assert part.error_scale.shape == (length,)
        else:
            assert part.error_scale.shape == (k, length)


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
        kw = dict(KWARGS, k=None, outcome_type='binary')
        dgp = gen_data(keys.pop(), **kw)
        assert dgp.y.shape == (kw['n'],)
        assert dgp.y.dtype == jnp.float32
        assert_array_equal(jnp.unique(dgp.y), jnp.array([0.0, 1.0]))
        assert dgp.params.outcome_type is OutcomeType.binary

    def test_binary_multivariate(self, keys: split) -> None:
        """Binary multivariate output contains only 0.0/1.0 in every row."""
        kw = dict(KWARGS, lambda_=0.5, outcome_type='binary')
        dgp = gen_data(keys.pop(), **kw)
        assert dgp.y.shape == (kw['k'], kw['n'])
        assert_array_equal(jnp.unique(dgp.y), jnp.array([0.0, 1.0]))

    def test_binary_threshold_matches_latent(self, keys: split) -> None:
        """With the same key, binary `y` equals `(continuous_y > 0)`.

        The key flow in `gen_data_from_params` is independent of outcome_type,
        so generating with `'continuous'` and `'binary'` under the same top
        level key must yield latents that differ only by the final threshold.
        """
        kw = dict(KWARGS, lambda_=0.5)
        key = keys.pop()
        dgp_cont = gen_data(key, **kw)
        dgp_bin = gen_data(random.clone(key), outcome_type='binary', **kw)
        assert_array_equal(dgp_bin.y, (dgp_cont.y > 0).astype(jnp.float32))

    def test_mixed(self, keys: split) -> None:
        """Mixed outcome_type: binary rows are 0/1, continuous rows match the baseline."""
        kw = dict(KWARGS, lambda_=0.5)
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
        # continuous rows unchanged
        assert_array_equal(dgp_mix.y[0], dgp_cont.y[0])
        assert_array_equal(dgp_mix.y[2], dgp_cont.y[2])
        # binary row is the threshold of the latent
        assert_array_equal(dgp_mix.y[1], (dgp_cont.y[1] > 0).astype(jnp.float32))

    def test_all_same_tuple_collapses_to_scalar(self, keys: split) -> None:
        """A tuple of identical types is stored as a scalar `OutcomeType`."""
        kw = dict(KWARGS, lambda_=0.5, outcome_type=('binary', 'binary', 'binary'))
        dgp = gen_data(keys.pop(), **kw)
        assert dgp.params.outcome_type is OutcomeType.binary

    def test_validation_tuple_wrong_length(self, keys: split) -> None:
        """A tuple whose length does not match `k` raises `ValueError`."""
        kw = dict(KWARGS, lambda_=0.5, outcome_type=('continuous', 'binary'))
        with pytest.raises(ValueError, match='outcome_type has length'):
            gen_data(keys.pop(), **kw)

    def test_validation_tuple_with_univariate(self, keys: split) -> None:
        """A tuple combined with the univariate path (`k is None`) raises."""
        kw = dict(KWARGS, k=None, outcome_type=('continuous',))
        with pytest.raises(ValueError, match='tuple outcome_type requires'):
            gen_data(keys.pop(), **kw)


def uniform_x(key: Key[Array, ''], shape: tuple[int, ...]) -> Float[Array, ' *shape']:
    """Sample predictors from the same ``U(-sqrt3, sqrt3)`` as `generate_x`."""
    return random.uniform(key, shape, minval=-jnp.sqrt(3.0), maxval=jnp.sqrt(3.0))


class TestLogVarMGF:
    """Test `log_var_mgf` in isolation."""

    N: int = 8_000_000  # i.i.d. samples for the Monte Carlo estimates

    def test_zero(self) -> None:
        """The factor vanishes at ``v = 0`` (no heteroskedasticity)."""
        assert_allclose(log_var_mgf(jnp.zeros(())), jnp.zeros(()), atol=1e-7)

    def test_matches_integral(self) -> None:
        """It matches the defining integral ``int_0^1 exp(6 v t^2) dt`` on a grid."""
        v = jnp.linspace(0.0, 0.6, 50)
        t = jnp.linspace(0.0, 1.0, 20_001)
        integral = jnp.trapezoid(jnp.exp(6 * v[:, None] * t**2), t, axis=1)
        assert_close_matrices(log_var_mgf(v), jnp.log(integral), rtol=1e-4)

    @pytest.mark.parametrize('v', [0.02, 0.1, 0.3])
    def test_matches_montecarlo(self, keys: split, v: float) -> None:
        """It estimates ``log E[exp(2 g X)]`` for ``g ~ N(0, v)``, ``X ~ U``."""
        g = jnp.sqrt(v) * random.normal(keys.pop(), (self.N,))
        x = uniform_x(keys.pop(), (self.N,))
        mc = jnp.log(jnp.mean(jnp.exp(2 * g * x)))
        assert_allclose(log_var_mgf(jnp.asarray(v)), mc, rtol=1e-2)


class TestHetNormalization:
    """Test `het_normalization` in isolation."""

    N: int = 8_000_000  # i.i.d. coefficient/predictor rows for the MC estimates

    @pytest.mark.parametrize('batch_shape', [(), (3,), (2, 4)])
    def test_shapes(self, keys: split, batch_shape: tuple[int, ...]) -> None:
        """It reduces over the trailing predictor axis, keeping the batch shape."""
        var_coef = jnp.exp(random.normal(keys.pop(), (*batch_shape, 5)))
        offset, var_v = het_normalization(var_coef)
        assert offset.shape == batch_shape
        assert var_v.shape == batch_shape

    def montecarlo_multiplier(
        self, keys: split, var_coef: Float[Array, ' p'], offset: Float[Array, '']
    ) -> Float[Array, ' N']:
        """Draw ``W ** 2`` marginally over the coefficients ``g ~ N(0, var_coef)``."""
        g = jnp.sqrt(var_coef) * random.normal(keys.pop(), (self.N, var_coef.size))
        x = uniform_x(keys.pop(), (self.N, var_coef.size))
        return jnp.exp(2 * (jnp.sum(g * x, axis=1) + offset))

    def test_unit_second_moment(self, keys: split) -> None:
        """The offset makes ``E[W ** 2] = 1`` marginally over the coefficient draw."""
        var_coef = 0.04 * (0.5 + random.uniform(keys.pop(), (6,)))
        offset, _ = het_normalization(var_coef)
        w2 = self.montecarlo_multiplier(keys, var_coef, offset)
        se = jnp.std(w2) / jnp.sqrt(self.N)
        assert_array_less(jnp.abs(jnp.mean(w2) - 1.0) / se, SIGMA_THRESHOLD)

    def test_var_v(self, keys: split) -> None:
        """``var_v`` matches the marginal Monte Carlo variance of ``W ** 2``."""
        var_coef = 0.04 * (0.5 + random.uniform(keys.pop(), (6,)))
        offset, var_v = het_normalization(var_coef)
        w2 = self.montecarlo_multiplier(keys, var_coef, offset)
        assert_allclose(jnp.var(w2), var_v, rtol=3e-2)


class TestHeteroskedasticity:
    """Tests for the heteroskedasticity feature of `gen_data`."""

    def test_homoskedastic_fields_are_none(self, keys: split) -> None:
        """Without heteroskedasticity every het field (incl. `error_scale`) is None."""
        dgp = gen_data(keys.pop(), lambda_=0.5, **KWARGS)
        assert dgp.error_scale is None
        assert dgp.params.gamma_shared is None
        assert dgp.params.gamma_separate is None
        assert dgp.params.het_offset is None
        assert dgp.params.var_v is None
        assert dgp.params.sigma2_logscale is None
        assert dgp.params.het_shape is None

    def test_vector_shapes_and_dtypes(self, keys: split) -> None:
        """Vector het exposes (k, n) scales and per-component coefficients."""
        n, p, k = KWARGS['n'], KWARGS['p'], KWARGS['k']
        dgp = gen_data(
            keys.pop(), lambda_=0.5, het_shape='vector', sigma2_logscale=0.2, **KWARGS
        )
        assert dgp.error_scale.shape == (k, n)
        assert jnp.issubdtype(dgp.error_scale.dtype, jnp.floating)
        assert jnp.all(dgp.error_scale > 0)
        assert dgp.params.gamma_shared.shape == (p,)
        assert dgp.params.gamma_separate.shape == (k, p)
        assert dgp.params.het_offset.shape == (k,)
        assert dgp.params.var_v.shape == (k,)
        assert dgp.params.sigma2_logscale.shape == ()
        assert dgp.params.het_shape == 'vector'

    def test_scalar_shapes(self, keys: split) -> None:
        """Scalar het exposes one (n,) scale and no separate coefficients."""
        dgp = gen_data(
            keys.pop(), lambda_=0.5, het_shape='scalar', sigma2_logscale=0.2, **KWARGS
        )
        assert dgp.error_scale.shape == (KWARGS['n'],)
        assert dgp.params.gamma_separate is None
        assert dgp.params.het_offset.shape == ()
        assert dgp.params.var_v.shape == ()
        assert dgp.params.het_shape == 'scalar'

    def test_univariate_scalar(self, keys: split) -> None:
        """Univariate (`k=None`) het produces an (n,) scale."""
        kw = dict(KWARGS, k=None)
        dgp = gen_data(keys.pop(), het_shape='scalar', sigma2_logscale=0.2, **kw)
        assert dgp.error_scale.shape == (kw['n'],)
        assert dgp.y.shape == (kw['n'],)
        assert dgp.params.gamma_separate is None

    @pytest.mark.parametrize('het_shape', ['scalar', 'vector'])
    def test_marginal_noise_variance_is_one(self, keys: split, het_shape: str) -> None:
        """``E[error_scale ** 2] == 1`` per component, preserving the noise budget."""
        dgps = generate_dgps(keys.pop(REPS), 0.5, None, HET_LOGSCALE, het_shape)
        v = dgps.error_scale**2  # (REPS, K?, N)
        per_dataset = jnp.mean(v, axis=-1)  # (REPS, K?)
        mean = jnp.mean(per_dataset, axis=0)
        se = jnp.std(per_dataset, axis=0) / jnp.sqrt(REPS)
        assert_array_less(jnp.abs((mean - 1.0) / se), SIGMA_THRESHOLD)

    def test_offset_and_var_v_wiring(self, keys: split) -> None:
        """`gen_params` feeds ``het_normalization`` the coefficient prior variances.

        Rebuilds the prior variances from the public ``s``, ``lambda_`` and
        ``partition`` and checks the stored ``het_offset`` / ``var_v`` match,
        catching wiring bugs in the budget, ``lambda_`` weighting or partition.
        """
        tau2 = 0.2
        p, k = KWARGS['p'], KWARGS['k']
        dgp = gen_data(
            keys.pop(), lambda_=0.5, het_shape='vector', sigma2_logscale=tau2, **KWARGS
        )
        var_shared = tau2 / p * dgp.params.s**2
        var_separate = tau2 / (p / k) * dgp.params.s**2 * dgp.params.partition
        var_coef = 0.5 * var_shared + 0.5 * var_separate
        offset, var_v = het_normalization(var_coef)
        assert_close_matrices(dgp.params.het_offset, offset, rtol=1e-6)
        assert_close_matrices(dgp.params.var_v, var_v, rtol=1e-6)

    def test_stream_invariance_with_homoskedastic(self, keys: split) -> None:
        """Het leaves the mean/predictor streams intact and only rescales the noise."""
        kw = dict(KWARGS, lambda_=0.5)
        key = keys.pop()
        homo = gen_data(key, **kw)
        het = gen_data(random.clone(key), het_shape='vector', sigma2_logscale=0.3, **kw)
        assert_array_equal(homo.x, het.x)
        assert_array_equal(homo.params.beta_shared, het.params.beta_shared)
        assert_array_equal(homo.params.A_separate, het.params.A_separate)
        assert_array_equal(homo.params.partition, het.params.partition)
        assert_array_equal(homo.params.s, het.params.s)
        assert_array_equal(homo.mu, het.mu)
        # the heteroskedastic noise is the homoskedastic noise scaled by error_scale
        recon = het.mu + (homo.y - homo.mu) * het.error_scale
        assert_close_matrices(het.y, recon, rtol=1e-5)

    def test_scalar_and_vector_share_shared_stream(self, keys: split) -> None:
        """``'scalar'`` and ``'vector'`` het share the ``gamma_shared`` stream."""
        kw = dict(KWARGS, lambda_=0.5, sigma2_logscale=0.3)
        key = keys.pop()
        scalar = gen_data(key, het_shape='scalar', **kw)
        vector = gen_data(random.clone(key), het_shape='vector', **kw)
        assert_array_equal(scalar.params.gamma_shared, vector.params.gamma_shared)

    def test_vector_requires_multivariate(self, keys: split) -> None:
        """``het_shape='vector'`` with ``k=None`` raises."""
        kw = dict(KWARGS, k=None)
        with pytest.raises(ValueError, match="het_shape='vector' requires"):
            gen_data(keys.pop(), het_shape='vector', sigma2_logscale=0.2, **kw)

    def test_logscale_and_shape_must_agree(self, keys: split) -> None:
        """``sigma2_logscale`` and ``het_shape`` must be both set or both None."""
        with pytest.raises(ValueError, match='both set or both None'):
            gen_data(keys.pop(), lambda_=0.5, sigma2_logscale=0.2, **KWARGS)
        with pytest.raises(ValueError, match='both set or both None'):
            gen_data(keys.pop(), lambda_=0.5, het_shape='scalar', **KWARGS)

    def test_binary_thresholds_het_latent(self, keys: split) -> None:
        """Het works for binary outcomes: `y` thresholds the heteroskedastic latent.

        With a shared key the continuous and binary runs differ only by the final
        threshold, so the binary success probability is
        ``Phi(mu / (sqrt(sigma2_eps) * error_scale))``.
        """
        kw = dict(KWARGS, lambda_=0.5, het_shape='vector', sigma2_logscale=0.2)
        key = keys.pop()
        cont = gen_data(key, **kw)
        binary = gen_data(random.clone(key), outcome_type='binary', **kw)
        assert_array_equal(jnp.unique(binary.y), jnp.array([0.0, 1.0]))
        assert_array_equal(binary.y, (cont.y > 0).astype(jnp.float32))
