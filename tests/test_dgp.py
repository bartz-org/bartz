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
from numpy.testing import assert_allclose, assert_array_equal, assert_array_less
from scipy.stats import norm

from bartz.jaxext import split
from bartz.mcmcstep import OutcomeType
from bartz.testing import DGP, gen_data
from bartz.testing._dgp import (
    generate_partition,
    interaction_pattern,
    partitioned_interaction_pattern,
)

# Test parameters
ALPHA = 5e-7  # probability of false positive (aaaaapprox)
SIGMA_THRESHOLD = norm.isf(ALPHA / 2)  # threshold for z tests
KWARGS: Mapping = MappingProxyType(
    dict(n=100, p=20, k=3, q=4, sigma2_eps=0.1, sigma2_lin=0.4, sigma2_quad=0.5)
)
REPS: int = 10_000  # number of datasets


@jit
@partial(vmap, in_axes=(0, None))
def generate_dgps(key: Key[Array, 'REPS'], lam: Float[Array, '']) -> DGP:
    """Generate one dataset per random key."""
    return gen_data(key, lam=lam, **KWARGS)


@pytest.fixture
def dgps(keys: split) -> DGP:
    """Generate DGP instances using vmap and jit."""
    return generate_dgps(keys.pop(REPS), 0.5)


@pytest.fixture
def dgps_lambda_zero(keys: split) -> DGP:
    """Generate DGP instances with lambda=0."""
    return generate_dgps(keys.pop(REPS), 0.0)


@pytest.fixture
def dgps_lambda_one(keys: split) -> DGP:
    """Generate DGP instances with lambda=1."""
    return generate_dgps(keys.pop(REPS), 1.0)


def test_shapes_and_dtypes(keys: split) -> None:
    """Test that all DGP attributes have correct shapes and dtypes."""
    dgp = gen_data(keys.pop(), lam=0.5, **KWARGS)
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
    assert dgp.muquad_shared.shape == (n,)
    assert dgp.muquad_separate.shape == (k, n)
    assert dgp.muquad.shape == (k, n)
    assert dgp.mu.shape == (k, n)
    assert dgp.params.q.shape == ()
    assert dgp.params.lam.shape == ()
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
    assert jnp.issubdtype(dgp.muquad_shared.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.muquad_separate.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.muquad.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.mu.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.params.q.dtype, jnp.integer)
    assert jnp.issubdtype(dgp.params.lam.dtype, jnp.floating)
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
        assert_array_equal(col_sums, 1)

    def test_partition_counts(self, dgps: DGP) -> None:
        """Test that counts are either p//c or p//c + 1."""
        partitions = dgps.params.partition  # Shape: (REPS, K, P)
        p, k = partitions.shape[2], partitions.shape[1]

        counts = jnp.sum(partitions, axis=2)  # Shape: (REPS, K)

        # Each count should be either P//K or P//K + 1
        floor_count = p // k
        valid = (counts == floor_count) | (counts == floor_count + 1)
        assert_array_equal(valid, True)

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


@pytest.mark.parametrize(
    'which',
    [
        'mulin_shared',
        'mulin_separate',
        'mulin',
        'muquad_shared',
        'muquad_separate',
        'muquad',
        'mu',
        'y',
    ],
)
def test_outcome_prior_variance(dgps: DGP, which: str) -> None:
    """Test that latent mean and outcome have the expected elementwise variance."""
    samples = getattr(dgps, which)  # Shape: (REPS, K?, N)
    n_reps = samples.shape[0]

    var = jnp.var(samples, axis=0)  # Shape: (K?, N)

    if which.startswith('mulin'):
        expected_var = dgps.params.sigma2_lin
    elif which.startswith('muquad'):
        expected_var = dgps.params.sigma2_quad + dgps.params.sigma2_mean
    elif which == 'mu':
        expected_var = dgps.params.sigma2_pri - dgps.params.sigma2_eps
    elif which == 'y':
        expected_var = dgps.params.sigma2_pri
    else:  # pragma: no cover
        raise KeyError(which)

    expected_var = expected_var[0].item()
    std_of_var = jnp.sqrt(2 * expected_var**2 / (n_reps - 1))

    z_scores = jnp.abs((var - expected_var) / std_of_var)
    assert_array_less(z_scores, SIGMA_THRESHOLD)


@pytest.mark.parametrize(
    'which',
    [
        'mulin_shared',
        'mulin_separate',
        'mulin',
        'muquad_shared',
        'muquad_separate',
        'muquad',
        'mu',
        'y',
    ],
)
def test_outcome_pop_variance(dgps: DGP, which: str) -> None:
    """Test that latent mean and outcome have the expected elementwise variance."""
    samples = getattr(dgps, which)  # Shape: (REPS, K?, N)
    n_reps = samples.shape[0]

    var = jnp.var(samples, axis=-1, ddof=1)  # Shape: (REPS, K?)
    var = jnp.mean(var, axis=0)  # Shape: (K?,)

    if which.startswith('mulin'):
        expected_var = dgps.params.sigma2_lin
    elif which.startswith('muquad'):
        expected_var = dgps.params.sigma2_quad
    elif which == 'mu':
        expected_var = dgps.params.sigma2_pop - dgps.params.sigma2_eps
    elif which == 'y':
        expected_var = dgps.params.sigma2_pop
    else:  # pragma: no cover
        raise KeyError(which)

    expected_var = expected_var[0].item()
    std_of_var = jnp.sqrt(2 * expected_var**2 / (n_reps - 1))

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
    """Test that rows are independent when lambda=0."""
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
    """Test that rows are identical when lambda=1."""
    samples = getattr(dgps_lambda_one, which)  # Shape: (REPS, K, N)

    # Check that all rows are identical within each sample
    diffs = jnp.max(
        jnp.abs(samples[:, 0:1, :] - samples), axis=(1, 2)
    )  # Shape: (REPS,)
    assert_allclose(diffs, 0.0, atol=1e-5)


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
        assert_array_equal(jnp.diag(pattern), True)

    def test_row_sums(self, pattern: Bool[Array, 'p p']) -> None:
        """Test that each row sums to q+1."""
        row_sums = jnp.sum(pattern, axis=1)
        assert_array_equal(row_sums, 4 + 1)


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
        assert_array_equal(pattern & ~mask, False)

    def test_diagonal_within_partition(
        self, partition: Bool[Array, 'k p'], pattern: Bool[Array, 'k p p']
    ) -> None:
        """Test that diagonal elements within partition are True."""
        k, _, _ = pattern.shape
        for i in range(k):
            diagonal = pattern[i, partition[i], partition[i]]
            assert_array_equal(diagonal, True)

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

    At k=1/lam=1, the univariate output is the k=1 multivariate output
    with the leading axis squeezed away.
    """
    key = keys.pop()
    kw = dict(KWARGS)
    kw.update(lam=1.0, k=1)
    dgp_mv = gen_data(key, **kw)
    kw.update(lam=None, k=None)
    dgp_uv = gen_data(random.clone(key), **kw)

    assert dgp_uv.params.partition is None
    assert dgp_uv.params.beta_separate is None
    assert dgp_uv.params.A_separate is None
    assert dgp_uv.params.lam is None
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


def test_lam_required_when_multivariate(keys: split) -> None:
    """`lam=None` with `k` set raises `ValueError`."""
    with pytest.raises(ValueError, match='lam is required'):
        gen_data(keys.pop(), lam=None, **KWARGS)


def test_lam_forbidden_when_univariate(keys: split) -> None:
    """`lam` not None with `k=None` raises `ValueError`."""
    kw = dict(KWARGS, k=None)
    with pytest.raises(ValueError, match='lam must be None'):
        gen_data(keys.pop(), lam=0.5, **kw)


def test_univariate_split(keys: split) -> None:
    """`DGP.split()` on a univariate DGP preserves `None` separate fields.

    Also checks that the shared/blended fields are sliced consistently.
    """
    kw = dict(KWARGS, k=None)
    dgp = gen_data(keys.pop(), **kw)
    n_train = kw['n'] // 3
    train, test = dgp.split(n_train)

    for part, length in ((train, n_train), (test, kw['n'] - n_train)):
        assert part.params.partition is None
        assert part.params.beta_separate is None
        assert part.params.A_separate is None
        assert part.mulin_separate is None
        assert part.muquad_separate is None
        assert part.y.shape == (length,)
        assert part.mulin_shared.shape == (length,)
        assert part.muquad_shared.shape == (length,)
        assert part.mulin.shape == (length,)
        assert part.muquad.shape == (length,)
        assert part.mu.shape == (length,)
        assert part.x.shape == (kw['p'], length)


class TestOutcomeType:
    """Tests for the `outcome_type` parameter of `gen_data`."""

    def test_default_is_continuous(self, keys: split) -> None:
        """Default `outcome_type` stores `OutcomeType.continuous` on Params."""
        dgp = gen_data(keys.pop(), lam=0.5, **KWARGS)
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
        kw = dict(KWARGS, lam=0.5, outcome_type='binary')
        dgp = gen_data(keys.pop(), **kw)
        assert dgp.y.shape == (kw['k'], kw['n'])
        assert_array_equal(jnp.unique(dgp.y), jnp.array([0.0, 1.0]))

    def test_binary_threshold_matches_latent(self, keys: split) -> None:
        """With the same key, binary `y` equals `(continuous_y > 0)`.

        The key flow in `gen_data_from_params` is independent of outcome_type,
        so generating with `'continuous'` and `'binary'` under the same top
        level key must yield latents that differ only by the final threshold.
        """
        kw = dict(KWARGS, lam=0.5)
        key = keys.pop()
        dgp_cont = gen_data(key, **kw)
        dgp_bin = gen_data(random.clone(key), outcome_type='binary', **kw)
        assert_array_equal(dgp_bin.y, (dgp_cont.y > 0).astype(jnp.float32))

    def test_mixed(self, keys: split) -> None:
        """Mixed outcome_type: binary rows are 0/1, continuous rows match the baseline."""
        kw = dict(KWARGS, lam=0.5)
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
        kw = dict(KWARGS, lam=0.5, outcome_type=('binary', 'binary', 'binary'))
        dgp = gen_data(keys.pop(), **kw)
        assert dgp.params.outcome_type is OutcomeType.binary

    def test_validation_tuple_wrong_length(self, keys: split) -> None:
        """A tuple whose length does not match `k` raises `ValueError`."""
        kw = dict(KWARGS, lam=0.5, outcome_type=('continuous', 'binary'))
        with pytest.raises(ValueError, match='outcome_type has length'):
            gen_data(keys.pop(), **kw)

    def test_validation_tuple_with_univariate(self, keys: split) -> None:
        """A tuple combined with the univariate path (`k is None`) raises."""
        kw = dict(KWARGS, k=None, outcome_type=('continuous',))
        with pytest.raises(ValueError, match='tuple outcome_type requires'):
            gen_data(keys.pop(), **kw)
