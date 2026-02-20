# bartz/src/bartz/testing/_dgp.py
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


"""Define `gen_data` that generates simulated data for testing."""

from dataclasses import replace

from equinox import Module, error_if
from jax import numpy as jnp
from jax import random
from jaxtyping import Array, Bool, Float, Int, Integer, Key

from bartz.jaxext import split


def generate_x(key: Key[Array, ''], n: int, p: int) -> Float[Array, 'p n']:
    """Generate predictors with mean 0 and variance 1.

    x_rj ~iid U(-√3, √3)
    """
    return random.uniform(key, (p, n), minval=-jnp.sqrt(3.0), maxval=jnp.sqrt(3.0))


def generate_partition(key: Key[Array, ''], p: int, k: int) -> Bool[Array, 'k p']:
    """Partition x components amongst y components.

    Each row i has either p // k or p // k + 1 non-zero entries.
    """
    keys = split(key)
    indices: Int[Array, 'p'] = jnp.linspace(0, k, p, endpoint=False)
    indices = jnp.trunc(indices).astype(jnp.int32)
    indices = random.permutation(keys.pop(), indices)
    assignments: Int[Array, 'k'] = random.permutation(keys.pop(), k)
    return indices == assignments[:, None]


def generate_beta_shared(
    key: Key[Array, ''], p: int, sigma2_lin: Float[Array, '']
) -> Float[Array, ' p']:
    """Generate shared linear coefficients for the lambda=1 case."""
    sigma2_beta = sigma2_lin / p
    return random.normal(key, (p,)) * jnp.sqrt(sigma2_beta)


def generate_beta_separate(
    key: Key[Array, ''], partition: Bool[Array, 'k p'], sigma2_lin: Float[Array, '']
) -> Float[Array, 'k p']:
    """Generate separate linear coefficients for the lambda=0 case."""
    k, p = partition.shape
    beta_separate: Float[Array, 'k p'] = random.normal(key, (k, p))
    sigma2_beta = sigma2_lin / (p / k)
    return jnp.where(partition, beta_separate, 0.0) * jnp.sqrt(sigma2_beta)


def compute_linear_mean_shared(
    beta_shared: Float[Array, ' p'], x: Float[Array, 'p n']
) -> Float[Array, ' n']:
    """mulin_ij = beta_r x_rj."""
    return beta_shared @ x


def compute_linear_mean_separate(
    beta_separate: Float[Array, 'k p'], x: Float[Array, 'p n']
) -> Float[Array, 'k n']:
    """mulin_ij = beta_ir x_rj."""
    return beta_separate @ x


def combine_mulin(
    mulin_shared: Float[Array, ' n'],
    mulin_separate: Float[Array, 'k n'],
    lam: Float[Array, ''],
) -> Float[Array, 'k n']:
    """Combine shared and separate linear means."""
    return jnp.sqrt(1.0 - lam) * mulin_separate + jnp.sqrt(lam) * mulin_shared


def interaction_pattern(p: int, q: Integer[Array, ''] | int) -> Bool[Array, 'p p']:
    """Create a symmetric interaction pattern for q interactions per variable.

    Parameters
    ----------
    p
        Number of predictors
    q
        Number of interactions per predictor (must be even)

    Returns
    -------
    Symmetric binary pattern of shape (p, p) where each row/col sums to q+1
    """
    q = error_if(q, q % 2 != 0, 'q must be even')
    q = error_if(q, q >= p, 'q must be less than p')

    i, j = jnp.ogrid[:p, :p]
    dist = jnp.minimum(jnp.abs(i - j), p - jnp.abs(i - j))
    return dist <= (q // 2)


def generate_A_shared(
    key: Key[Array, ''],
    p: int,
    q: Integer[Array, ''],
    sigma2_quad: Float[Array, ''],
    kurt_x: float,
) -> Float[Array, 'p p']:
    """Generate shared quadratic coefficients for the lambda=1 case."""
    pattern: Bool[Array, 'p p'] = interaction_pattern(p, q)
    A_shared: Float[Array, 'p p'] = random.normal(key, (p, p))
    A_shared = jnp.where(pattern, A_shared, 0.0)
    sigma2_A = sigma2_quad / (p * (kurt_x - 1 + q))
    return A_shared * jnp.sqrt(sigma2_A)


def partitioned_interaction_pattern(
    partition: Bool[Array, 'k p'], q: Integer[Array, ''] | int
) -> Bool[Array, 'k p p']:
    """Create k interaction patterns that use disjoint variable sets.

    Parameters
    ----------
    partition
        Binary partition of shape (k, p) indicating variable assignments
        to components
    q
        Number of interactions per predictor (must be even and < p // k)

    Returns
    -------
    Interaction patterns of shape (k, p, p)
    """
    k, p = partition.shape
    q = error_if(q, q % 2 != 0, 'q must be even')
    q = error_if(q, q >= p // k, 'q must be less than p // k')

    indices: Int[Array, 'k p'] = jnp.cumsum(partition, axis=1)
    linear_dist: Int[Array, 'k p p'] = jnp.abs(
        indices[:, :, None] - indices[:, None, :]
    )
    num_vars: Int[Array, 'k'] = jnp.max(indices, axis=1)
    wrapped_dist: Int[Array, 'k p p'] = jnp.minimum(
        linear_dist, num_vars[:, None, None] - linear_dist
    )
    interacts: Bool[Array, 'k p p'] = wrapped_dist <= (q // 2)
    interacts = jnp.where(partition[:, :, None], interacts, False)
    return jnp.where(partition[:, None, :], interacts, False)


def generate_A_separate(
    key: Key[Array, ''],
    partition: Bool[Array, 'k p'],
    q: Integer[Array, ''],
    sigma2_quad: Float[Array, ''],
    kurt_x: float,
) -> Float[Array, 'k p p']:
    """Generate separate quadratic coefficients for the lambda=0 case."""
    k, p = partition.shape
    A_separate: Float[Array, 'k p p'] = random.normal(key, (k, p, p))
    component_pattern: Bool[Array, 'k p p'] = partitioned_interaction_pattern(
        partition, q
    )
    A_separate = jnp.where(component_pattern, A_separate, 0.0)
    sigma2_A = sigma2_quad / (p / k * (kurt_x - 1 + q))
    return A_separate * jnp.sqrt(sigma2_A)


def compute_muquad_shared(
    A_shared: Float[Array, 'p p'], x: Float[Array, 'p n']
) -> Float[Array, ' n']:
    """Compute quadratic mean for the lambda=1 case.

    muquad_ij = A_rs x_rj x_sj
    Rows identical across components.
    """
    return jnp.einsum('rs,rj,sj->j', A_shared, x, x)


def compute_muquad_separate(
    A_separate: Float[Array, 'k p p'], x: Float[Array, 'p n']
) -> Float[Array, 'k n']:
    """Compute quadratic mean for the lambda=0 case.

    muquad_ij = A_irs x_rj x_sj
    Rows independent across components.
    """
    return jnp.einsum('irs,rj,sj->ij', A_separate, x, x)


def combine_muquad(
    muquad_shared: Float[Array, ' n'],
    muquad_separate: Float[Array, 'k n'],
    lam: Float[Array, ''],
) -> Float[Array, 'k n']:
    """Combine shared and separate quadratic means."""
    return jnp.sqrt(1.0 - lam) * muquad_separate + jnp.sqrt(lam) * muquad_shared


def compute_quadratic_mean(
    A: Float[Array, 'k p p'], x: Float[Array, 'p n']
) -> Float[Array, 'k n']:
    """Compute quadratic part of the latent mean."""
    return jnp.einsum('irs,rj,sj->ij', A, x, x)


def generate_outcome(
    key: Key[Array, ''], mu: Float[Array, 'k n'], sigma2_eps: Float[Array, '']
) -> Float[Array, 'k n']:
    """Generate noisy outcome."""
    eps: Float[Array, 'k n'] = random.normal(key, mu.shape)
    return mu + eps * jnp.sqrt(sigma2_eps)


class DGP(Module):
    """Quadratic multivariate DGP with n units, p predictors, k outcomes.

    Parameters
    ----------
    x
        Predictors of shape (p, n), variance 1
    y
        Noisy outcomes of shape (k, n) or (n,)
    partition
        Predictor-outcome assignment partition of shape (k, p)
    beta_shared
        Shared linear coefficients of shape (p,)
    beta_separate
        Separate linear coefficients of shape (k, p)
    mulin_shared
        Linear mean at lambda=1 (shared), shape (k, n), rows identical
    mulin_separate
        Linear mean at lambda=0 (separate), shape (k, n), rows independent
    mulin
        Linear part of latent mean of shape (k, n)
    A_shared
        Shared quadratic coefficients of shape (p, p)
    A_separate
        Separate quadratic coefficients of shape (k, p, p)
    muquad_shared
        Quadratic mean at lambda=1 (shared), shape (k, n), rows identical
    muquad_separate
        Quadratic mean at lambda=0 (separate), shape (k, n), rows independent
    muquad
        Quadratic part of latent mean of shape (k, n)
    mu
        True latent means of shape (k, n)
    q
        Number of interactions per predictor
    lam
        Coupling parameter in [0, 1]
    sigma2_lin
        Prior and expected population variance of mulin
    sigma2_quad
        Expected population variance of muquad
    sigma2_eps
        Variance of the error
    """

    # Main outputs
    x: Float[Array, 'p n']
    y: Float[Array, 'k n'] | Float[Array, ' n']

    # Intermediate results
    partition: Bool[Array, 'k p']
    beta_shared: Float[Array, ' p']
    beta_separate: Float[Array, 'k p']
    mulin_shared: Float[Array, ' n']
    mulin_separate: Float[Array, 'k n']
    mulin: Float[Array, 'k n']
    A_shared: Float[Array, 'p p']
    A_separate: Float[Array, 'k p p']
    muquad_shared: Float[Array, ' n']
    muquad_separate: Float[Array, 'k n']
    muquad: Float[Array, 'k n']
    mu: Float[Array, 'k n']

    # Params
    q: Integer[Array, '']
    lam: Float[Array, '']
    sigma2_lin: Float[Array, '']
    sigma2_quad: Float[Array, '']
    sigma2_eps: Float[Array, '']

    kurt_x: float = 9 / 5  # kurtosis of uniform distribution

    @property
    def sigma2_pri(self) -> Float[Array, '']:
        """Prior variance of y."""
        return self.sigma2_pop + self.sigma2_mean

    @property
    def sigma2_pop(self) -> Float[Array, '']:
        """Expected population variance of y."""
        return self.sigma2_lin + self.sigma2_quad + self.sigma2_eps

    @property
    def sigma2_mean(self) -> Float[Array, '']:
        """Variance of the mean function."""
        return self.sigma2_quad / (self.kurt_x - 1 + self.q)

    def split(self, n_train: int | None = None) -> tuple['DGP', 'DGP']:
        """Split the data into training and test sets."""
        if n_train is None:
            n_train = self.x.shape[1] // 2
        assert 0 < n_train < self.x.shape[1], 'n_train must be in (0, n)'
        train = replace(
            self,
            x=self.x[:, :n_train],
            y=self.y[:, :n_train],
            mulin_shared=self.mulin_shared[:n_train],
            mulin_separate=self.mulin_separate[:, :n_train],
            mulin=self.mulin[:, :n_train],
            muquad_shared=self.muquad_shared[:n_train],
            muquad_separate=self.muquad_separate[:, :n_train],
            muquad=self.muquad[:, :n_train],
            mu=self.mu[:, :n_train],
        )
        test = replace(
            self,
            x=self.x[:, n_train:],
            y=self.y[:, n_train:],
            mulin_shared=self.mulin_shared[n_train:],
            mulin_separate=self.mulin_separate[:, n_train:],
            mulin=self.mulin[:, n_train:],
            muquad_shared=self.muquad_shared[n_train:],
            muquad_separate=self.muquad_separate[:, n_train:],
            muquad=self.muquad[:, n_train:],
            mu=self.mu[:, n_train:],
        )
        return train, test


def gen_data(
    key: Key[Array, ''],
    *,
    n: int,
    p: int,
    k: int | None = None,
    q: Integer[Array, ''] | int,
    lam: Float[Array, ''] | float,
    sigma2_lin: Float[Array, ''] | float,
    sigma2_quad: Float[Array, ''] | float,
    sigma2_eps: Float[Array, ''] | float,
) -> DGP:
    """Generate data from a quadratic multivariate DGP.

    Parameters
    ----------
    key
        JAX random key
    n
        Number of observations
    p
        Number of predictors
    k
        Number of outcome components
    q
        Number of interactions per predictor (must be even and < p // k)
    lam
        Coupling parameter in [0, 1]. 0=independent, 1=identical components
    sigma2_lin
        Prior and expected population variance of the linear term
    sigma2_quad
        Expected population variance of the quadratic term
    sigma2_eps
        Variance of the error term

    Returns
    -------
    An object with all generated data and parameters.
    """
    squeeze = k is None
    if squeeze:
        k = 1

    assert p >= k, 'p must be at least k'

    # check q
    q = jnp.asarray(q)
    q = error_if(q, q % 2 != 0, 'q must be even')
    q = error_if(q, q >= p // k, 'q must be less than p // k')

    keys = split(key, 7)

    lam = jnp.asarray(lam)
    sigma2_lin = jnp.asarray(sigma2_lin)
    sigma2_quad = jnp.asarray(sigma2_quad)
    sigma2_eps = jnp.asarray(sigma2_eps)

    x = generate_x(keys.pop(), n, p)
    partition = generate_partition(keys.pop(), p, k)
    beta_shared = generate_beta_shared(keys.pop(), p, sigma2_lin)
    beta_separate = generate_beta_separate(keys.pop(), partition, sigma2_lin)
    mulin_shared = compute_linear_mean_shared(beta_shared, x)
    mulin_separate = compute_linear_mean_separate(beta_separate, x)
    mulin = combine_mulin(mulin_shared, mulin_separate, lam)
    A_shared = generate_A_shared(keys.pop(), p, q, sigma2_quad, DGP.kurt_x)
    A_separate = generate_A_separate(keys.pop(), partition, q, sigma2_quad, DGP.kurt_x)
    muquad_shared = compute_muquad_shared(A_shared, x)
    muquad_separate = compute_muquad_separate(A_separate, x)
    muquad = combine_muquad(muquad_shared, muquad_separate, lam)
    mu = mulin + muquad
    y = generate_outcome(keys.pop(), mu, sigma2_eps)
    if squeeze:
        y = y.squeeze(0)

    return DGP(
        x=x,
        y=y,
        partition=partition,
        beta_shared=beta_shared,
        beta_separate=beta_separate,
        mulin_shared=mulin_shared,
        mulin_separate=mulin_separate,
        mulin=mulin,
        A_shared=A_shared,
        A_separate=A_separate,
        muquad_shared=muquad_shared,
        muquad_separate=muquad_separate,
        muquad=muquad,
        mu=mu,
        q=q,
        lam=lam,
        sigma2_lin=sigma2_lin,
        sigma2_quad=sigma2_quad,
        sigma2_eps=sigma2_eps,
    )
