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

"""Tests for the DGP data generating process."""

from collections.abc import Mapping
from functools import partial
from types import MappingProxyType

import pytest
from jax import jit, vmap
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Key
from numpy.testing import assert_allclose, assert_array_equal, assert_array_less
from scipy.stats import norm

from bartz.jaxext import split
from bartz.testing import DGP

# Test parameters
ALPHA = 5e-7  # probability of false positive
SIGMA_THRESHOLD = norm.isf(ALPHA / 2)  # threshold for z tests
KWARGS: Mapping = MappingProxyType(
    dict(n=100, p=20, c=3, q=4, sigma2_eps=0.1, sigma2_lin=0.4, sigma2_quad=0.5)
)
REPS: int = 10_000  # number of datasets


@jit
@partial(vmap, in_axes=(0, None))
def generate_dgps(key: Key[Array, 'REPS'], lam: Float[Array, '']) -> DGP:
    """Generate one dataset per random key."""
    return DGP(key, lam=lam, **KWARGS)


@pytest.fixture
def dgps(keys):
    """Generate DGP instances using vmap and jit."""
    return generate_dgps(keys.pop(REPS), 0.5)


@pytest.fixture
def dgps_lambda_zero(keys: split):
    """Generate DGP instances with lambda=0."""
    return generate_dgps(keys.pop(REPS), 0.0)


@pytest.fixture
def dgps_lambda_one(keys: split):
    """Generate DGP instances with lambda=1."""
    return generate_dgps(keys.pop(REPS), 1.0)


def test_shapes_and_dtypes(keys: split):
    """Test that all DGP attributes have correct shapes and dtypes."""
    dgp = DGP(keys.pop(), lam=0.5, **KWARGS)
    n, p, c = KWARGS['n'], KWARGS['p'], KWARGS['c']

    # Test shapes
    assert dgp.x.shape == (p, n)
    assert dgp.y.shape == (c, n)
    assert dgp.partition.shape == (c, p)
    assert dgp.beta_shared.shape == (p,)
    assert dgp.beta_separate.shape == (c, p)
    assert dgp.mulin_shared.shape == (n,)
    assert dgp.mulin_separate.shape == (c, n)
    assert dgp.mulin.shape == (c, n)
    assert dgp.A_shared.shape == (p, p)
    assert dgp.A_separate.shape == (c, p, p)
    assert dgp.muquad_shared.shape == (n,)
    assert dgp.muquad_separate.shape == (c, n)
    assert dgp.muquad.shape == (c, n)
    assert dgp.mu.shape == (c, n)
    assert dgp.q.shape == ()
    assert dgp.lam.shape == ()
    assert dgp.sigma2_lin.shape == ()
    assert dgp.sigma2_quad.shape == ()
    assert dgp.sigma2_eps.shape == ()

    # Test dtypes
    assert jnp.issubdtype(dgp.x.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.y.dtype, jnp.floating)
    assert dgp.partition.dtype == jnp.bool_
    assert jnp.issubdtype(dgp.beta_shared.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.beta_separate.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.mulin_shared.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.mulin_separate.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.mulin.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.A_shared.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.A_separate.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.muquad_shared.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.muquad_separate.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.muquad.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.mu.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.q.dtype, jnp.integer)
    assert jnp.issubdtype(dgp.lam.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.sigma2_lin.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.sigma2_quad.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.sigma2_eps.dtype, jnp.floating)


class TestGenerateX:
    """Test the _generate_x method."""

    def test_x_mean(self, dgps):
        """Test that x has mean close to 0."""
        x_samples = dgps.x  # Shape: (N_REPS, P, N)
        n_reps = x_samples.shape[0]

        # Compute mean and std of mean for each element
        means = jnp.mean(x_samples, axis=0)  # Shape: (P, N)
        stds_of_mean = jnp.std(x_samples, axis=0) / jnp.sqrt(n_reps)  # Shape: (P, N)

        # All means should be within SIGMA_THRESHOLD standard deviations of 0
        z_scores = jnp.abs(means / stds_of_mean)
        assert_array_less(z_scores, SIGMA_THRESHOLD)

    def test_x_variance(self, dgps):
        """Test that x has variance close to 1."""
        x_samples = dgps.x  # Shape: (N_REPS, P, N)
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

    def test_partition_coverage(self, dgps):
        """Test that each predictor is assigned to exactly one component."""
        partitions = dgps.partition  # Shape: (N_REPS, C, P)

        # Each column should sum to 1
        col_sums = jnp.sum(partitions, axis=1)  # Shape: (N_REPS, P)
        assert_array_equal(col_sums, 1)

    def test_partition_counts(self, dgps):
        """Test that counts are either p//c or p//c + 1."""
        partitions = dgps.partition  # Shape: (N_REPS, C, P)
        p, c = partitions.shape[2], partitions.shape[1]

        counts = jnp.sum(partitions, axis=2)  # Shape: (N_REPS, C)

        # Each count should be either P//C or P//C + 1
        floor_count = p // c
        valid = (counts == floor_count) | (counts == floor_count + 1)
        assert_array_equal(valid, True)

    def test_partition_balance(self, dgps):
        """Test that predictors are roughly balanced across components."""
        partitions = dgps.partition  # Shape: (N_REPS, C, P)
        n_reps, c, p = partitions.shape

        counts = jnp.sum(partitions, axis=2)  # Shape: (N_REPS, C)

        # Mean count per component should be P/C
        expected_mean = p / c
        means = jnp.mean(counts, axis=0)  # Shape: (C,)
        stds_of_mean = jnp.std(counts, axis=0) / jnp.sqrt(n_reps)  # Shape: (C,)

        z_scores = jnp.abs((means - expected_mean) / stds_of_mean)
        assert_array_less(z_scores, SIGMA_THRESHOLD)


class TestGenerateBetaShared:
    """Test the _generate_beta_shared method."""

    def test_beta_shared_mean(self, dgps):
        """Test that beta_shared has mean close to 0."""
        beta_samples = dgps.beta_shared  # Shape: (N_REPS, P)
        n_reps = beta_samples.shape[0]

        means = jnp.mean(beta_samples, axis=0)  # Shape: (P,)
        stds_of_mean = jnp.std(beta_samples, axis=0) / jnp.sqrt(n_reps)  # Shape: (P,)

        z_scores = jnp.abs(means / stds_of_mean)
        assert_array_less(z_scores, SIGMA_THRESHOLD)


class TestGenerateBetaSeparate:
    """Test the _generate_beta_separate method."""

    def test_beta_separate_mean(self, dgps):
        """Test that beta_separate has mean close to 0."""
        beta_samples = dgps.beta_separate  # Shape: (N_REPS, C, P)
        n_reps = beta_samples.shape[0]

        means = jnp.mean(beta_samples, axis=0)  # Shape: (C, P)
        stds_of_mean = jnp.std(beta_samples, axis=0) / jnp.sqrt(n_reps)  # Shape: (C, P)

        z_scores = jnp.abs(means / stds_of_mean)
        assert_array_less(z_scores, SIGMA_THRESHOLD)

    def test_beta_separate_independence(self, dgps):
        """Test that rows of beta_separate are independent."""
        beta_samples = dgps.beta_separate  # Shape: (N_REPS, C, P)
        n_reps = beta_samples.shape[0]

        beta0 = beta_samples[:, 0, :]  # Shape: (N_REPS, P)
        beta1 = beta_samples[:, 1, :]  # Shape: (N_REPS, P)

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
def test_outcome_prior_variance(dgps, which: str):
    """Test that latent mean and outcome have the expected elementwise variance."""
    samples = getattr(dgps, which)  # Shape: (N_REPS, C?, N)
    n_reps = samples.shape[0]

    var = jnp.var(samples, axis=0)  # Shape: (C?, N)

    if which.startswith('mulin'):
        expected_var = dgps.sigma2_lin
    elif which.startswith('muquad'):
        expected_var = dgps.sigma2_quad + dgps.sigma2_mean
    elif which == 'mu':
        expected_var = dgps.sigma2_pri - dgps.sigma2_eps
    elif which == 'y':
        expected_var = dgps.sigma2_pri
    else:
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
def test_outcome_pop_variance(dgps, which: str):
    """Test that latent mean and outcome have the expected elementwise variance."""
    samples = getattr(dgps, which)  # Shape: (N_REPS, C?, N)
    n_reps = samples.shape[0]

    var = jnp.var(samples, axis=-1, ddof=1)  # Shape: (N_REPS, C?)
    var = jnp.mean(var, axis=0)  # Shape: (C?,)

    if which.startswith('mulin'):
        expected_var = dgps.sigma2_lin
    elif which.startswith('muquad'):
        expected_var = dgps.sigma2_quad
    elif which == 'mu':
        expected_var = dgps.sigma2_pop - dgps.sigma2_eps
    elif which == 'y':
        expected_var = dgps.sigma2_pop
    else:
        raise KeyError(which)

    expected_var = expected_var[0].item()
    std_of_var = jnp.sqrt(2 * expected_var**2 / (n_reps - 1))

    z_scores = jnp.abs((var - expected_var) / std_of_var)
    assert_array_less(z_scores, SIGMA_THRESHOLD)


def test_variance_relationships(dgps):
    """Check some simple inequalities on variances."""
    assert jnp.all(dgps.sigma2_pri >= 0)
    assert jnp.all(dgps.sigma2_pop >= 0)
    assert jnp.all(dgps.sigma2_mean >= 0)
    assert jnp.all(dgps.sigma2_pri >= dgps.sigma2_pop)
    assert jnp.all(dgps.sigma2_pop >= dgps.sigma2_eps)


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
def test_rows_independent(dgps_lambda_zero, which):
    """Test that rows are independent when lambda=0."""
    samples = getattr(dgps_lambda_zero, which)  # Shape: (N_REPS, C, N or P)
    n_reps = samples.shape[0]

    samples0 = samples[:, 0, :]  # Shape: (N_REPS, N or P)
    samples1 = samples[:, 1, :]  # Shape: (N_REPS, N or P)

    # Compute covariance for each observation position
    mean0 = jnp.mean(samples0, axis=0)
    mean1 = jnp.mean(samples1, axis=0)
    cov = jnp.mean((samples0 - mean0) * (samples1 - mean1), axis=0)

    std0 = jnp.std(samples0, axis=0)
    std1 = jnp.std(samples1, axis=0)
    std_of_cov = (std0 * std1) / jnp.sqrt(n_reps)

    z_scores = jnp.abs(cov / std_of_cov)
    assert_array_less(z_scores, SIGMA_THRESHOLD)


def test_rows_identical(dgps_lambda_one):
    """Test that rows are identical when lambda=1."""
    mulins = dgps_lambda_one.mulin  # Shape: (N_REPS, C, N)

    # Check that all rows are identical within each sample
    diffs = jnp.max(
        jnp.abs(mulins[:, 0:1, :] - mulins), axis=(1, 2)
    )  # Shape: (N_REPS,)
    assert_allclose(diffs, 0.0, atol=1e-5)


class TestInteractionPattern:
    """Test the interaction pattern method."""

    @pytest.fixture
    def pattern(self) -> Bool[Array, 'p p']:
        """Return the predictor interaction pattern."""
        return DGP._interaction_pattern(p=10, q=4)

    def test_symmetry(self, pattern):
        """Test that interaction pattern is symmetric."""
        assert_array_equal(pattern, pattern.T)

    def test_diagonal(self, pattern):
        """Test that diagonal is True."""
        assert_array_equal(jnp.diag(pattern), True)

    def test_row_sums(self, pattern):
        """Test that each row sums to q+1."""
        row_sums = jnp.sum(pattern, axis=1)
        assert_array_equal(row_sums, 4 + 1)


class TestPartitionedInteractionPattern:
    """Test the partitioned interaction pattern method."""

    @pytest.fixture
    def partition(self, keys: split) -> Bool[Array, 'c p']:
        """Generate a partition of predictors."""
        p, c = 20, 3
        return DGP._generate_partition(keys.pop(), p=p, c=c)

    @pytest.fixture
    def q(self) -> int:
        """Fix a value of the `q` parameter."""
        return 2

    @pytest.fixture
    def pattern(self, partition: Bool[Array, 'c p'], q: int) -> Bool[Array, 'c p p']:
        """Generate a multivariate interaction pattern that respects `partition`."""
        return DGP._partitioned_interaction_pattern(partition, q=q)

    def test_respects_partition(self, partition, pattern):
        """Test that pattern only has True values within partition blocks."""
        # For each component, check that True values only occur where partition is True
        # pattern[i, k, l] can only be True if partition[i, k] and partition[i, l] are True
        mask = partition[:, :, None] & partition[:, None, :]  # Shape: (c, p, p)
        assert_array_equal(pattern & ~mask, False)

    def test_diagonal_within_partition(self, partition, pattern):
        """Test that diagonal elements within partition are True."""
        c, _, _ = pattern.shape
        for i in range(c):
            diagonal = pattern[i, partition[i], partition[i]]
            assert_array_equal(diagonal, True)

    def test_row_sums(self, pattern, partition, q):
        """Test that each row sums to q+1."""
        row_sums = jnp.sum(pattern, axis=2)
        target = jnp.where(partition, q + 1, 0)
        assert_array_equal(row_sums, target)

    def test_symmetry(self, pattern):
        """Test that interaction pattern is symmetric."""
        assert_array_equal(pattern, jnp.swapaxes(pattern, 1, 2))
