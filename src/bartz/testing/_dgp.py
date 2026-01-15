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


"""Data generating process for bivariate BART testing."""

from equinox import Module, error_if
from jax import numpy as jnp
from jax import random
from jaxtyping import Array, Bool, Float, Int, Integer, Key

from bartz.jaxext import split


class Split(Module):
    x_train: Float[Array, 'p n_train']
    y_train: Float[Array, 'c n_train']
    x_test: Float[Array, 'p n_test']
    y_test: Float[Array, 'c n_test']


class DGP(Module):
    """Quadratic multivariate DGP.

    Parameters
    ----------
    key
        JAX random key
    n
        Number of observations
    p
        Number of predictors
    c
        Number of outcome components
    q
        Number of interactions per predictor (must be even and < p // c)
    lam
        Coupling parameter in [0, 1]. 0=independent, 1=identical components
    sigma2_lin
        Prior and expected population variance of the linear term
    sigma2_quad
        Expected population variance of the quadratic term
    sigma2_eps
        Variance of the error term

    Attributes
    ----------
    x
        Predictors of shape (p, n), variance 1
    y
        Noisy outcomes of shape (c, n)
    partition
        Predictor-outcome assignment partition of shape (c, p)
    beta_shared
        Shared linear coefficients of shape (p,)
    beta_separate
        Separate linear coefficients of shape (c, p)
    mulin_shared
        Linear mean at lambda=1 (shared), shape (c, n), rows identical
    mulin_separate
        Linear mean at lambda=0 (separate), shape (c, n), rows independent
    mulin
        Linear part of latent mean of shape (c, n)
    A_shared
        Shared quadratic coefficients of shape (p, p)
    A_separate
        Separate quadratic coefficients of shape (c, p, p)
    muquad_shared
        Quadratic mean at lambda=1 (shared), shape (c, n), rows identical
    muquad_separate
        Quadratic mean at lambda=0 (separate), shape (c, n), rows independent
    muquad
        Quadratic part of latent mean of shape (c, n)
    mu
        True latent means of shape (c, n)
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
    y: Float[Array, 'c n']

    # Intermediate results
    partition: Bool[Array, 'c p']
    beta_shared: Float[Array, ' p']
    beta_separate: Float[Array, 'c p']
    mulin_shared: Float[Array, ' n']
    mulin_separate: Float[Array, 'c n']
    mulin: Float[Array, 'c n']
    A_shared: Float[Array, 'p p']
    A_separate: Float[Array, 'c p p']
    muquad_shared: Float[Array, ' n']
    muquad_separate: Float[Array, 'c n']
    muquad: Float[Array, 'c n']
    mu: Float[Array, 'c n']

    # Params
    q: Integer[Array, '']
    lam: Float[Array, '']
    sigma2_lin: Float[Array, '']
    sigma2_quad: Float[Array, '']
    sigma2_eps: Float[Array, '']

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

    @property
    def kurt_x(self) -> float:
        """Kurtosis of the predictors."""
        return 9 / 5  # uniform distribution

    def split(self, n_train: int | None = None) -> Split:
        """Split the data into training and test sets."""
        if n_train is None:
            n_train = self.x.shape[1] // 2
        assert 0 < n_train < self.x.shape[1], 'n_train must be in (0, n)'
        return Split(
            x_train=self.x[:, :n_train],
            y_train=self.y[:, :n_train],
            x_test=self.x[:, n_train:],
            y_test=self.y[:, n_train:],
        )

    def __init__(
        self,
        key: Key[Array, ''],
        *,
        n: int,
        p: int,
        c: int,
        q: Integer[Array, ''] | int,
        lam: Float[Array, ''] | float,
        sigma2_lin: Float[Array, ''] | float,
        sigma2_quad: Float[Array, ''] | float,
        sigma2_eps: Float[Array, ''] | float,
    ):
        assert p >= c, 'p must be at least c'
        assert q % 2 == 0, 'q must be even'
        assert q < p // c, 'q must be less than p // c'

        keys = split(key, 7)

        self.q = jnp.asarray(q)
        self.lam = jnp.asarray(lam)
        self.sigma2_lin = jnp.asarray(sigma2_lin)
        self.sigma2_quad = jnp.asarray(sigma2_quad)
        self.sigma2_eps = jnp.asarray(sigma2_eps)

        self.x = self._generate_x(keys.pop(), n, p)
        self.partition = self._generate_partition(keys.pop(), p, c)
        self.beta_shared = self._generate_beta_shared(keys.pop(), p, self.sigma2_lin)
        self.beta_separate = self._generate_beta_separate(
            keys.pop(), self.partition, self.sigma2_lin
        )
        self.mulin_shared = self._compute_linear_mean_shared(self.beta_shared, self.x)
        self.mulin_separate = self._compute_linear_mean_separate(
            self.beta_separate, self.x
        )
        self.mulin = self._combine_mulin(
            self.mulin_shared, self.mulin_separate, self.lam
        )
        self.A_shared = self._generate_A_shared(
            keys.pop(), p, self.q, self.sigma2_quad, self.kurt_x
        )
        self.A_separate = self._generate_A_separate(
            keys.pop(), self.partition, self.q, self.sigma2_quad, self.kurt_x
        )
        self.muquad_shared = self._compute_muquad_shared(self.A_shared, self.x)
        self.muquad_separate = self._compute_muquad_separate(self.A_separate, self.x)
        self.muquad = self._combine_muquad(
            self.muquad_shared, self.muquad_separate, self.lam
        )
        self.mu = self.mulin + self.muquad
        self.y = self._generate_outcome(keys.pop(), self.mu, self.sigma2_eps)

    @staticmethod
    def _generate_x(key: Key[Array, ''], n: int, p: int) -> Float[Array, 'p n']:
        """Generate predictors with mean 0 and variance 1.

        x_kj ~iid U(-√3, √3)
        """
        return random.uniform(key, (p, n), minval=-jnp.sqrt(3.0), maxval=jnp.sqrt(3.0))

    @staticmethod
    def _generate_partition(key: Key[Array, ''], p: int, c: int) -> Bool[Array, 'c p']:
        """Partition x components amongst y components.

        Each row i has either p // c or p // c + 1 non-zero entries.
        """
        keys = split(key)
        indices: Int[Array, 'p'] = jnp.linspace(0, c, p, endpoint=False)
        indices = jnp.trunc(indices).astype(jnp.int32)
        indices = random.permutation(keys.pop(), indices)
        assignments: Int[Array, 'c'] = random.permutation(keys.pop(), c)
        return indices == assignments[:, None]

    @staticmethod
    def _generate_beta_shared(
        key: Key[Array, ''], p: int, sigma2_lin: Float[Array, '']
    ) -> Float[Array, ' p']:
        """Generate shared linear coefficients for the lambda=1 case."""
        sigma2_beta = sigma2_lin / p
        return random.normal(key, (p,)) * jnp.sqrt(sigma2_beta)

    @staticmethod
    def _generate_beta_separate(
        key: Key[Array, ''], partition: Bool[Array, 'c p'], sigma2_lin: Float[Array, '']
    ) -> Float[Array, 'c p']:
        """Generate separate linear coefficients for the lambda=0 case."""
        c, p = partition.shape
        beta_separate: Float[Array, 'c p'] = random.normal(key, (c, p))
        sigma2_beta = sigma2_lin / (p / c)
        return jnp.where(partition, beta_separate, 0.0) * jnp.sqrt(sigma2_beta)

    @staticmethod
    def _compute_linear_mean_shared(
        beta_shared: Float[Array, ' p'], x: Float[Array, 'p n']
    ) -> Float[Array, ' n']:
        """mulin_ij = beta_k x_kj."""
        return beta_shared @ x

    @staticmethod
    def _compute_linear_mean_separate(
        beta_separate: Float[Array, 'c p'], x: Float[Array, 'p n']
    ) -> Float[Array, 'c n']:
        """mulin_ij = beta_ik x_kj."""
        return beta_separate @ x

    @staticmethod
    def _combine_mulin(
        mulin_shared: Float[Array, ' n'],
        mulin_separate: Float[Array, 'c n'],
        lam: Float[Array, ''],
    ) -> Float[Array, 'c n']:
        """Combine shared and separate linear means."""
        return jnp.sqrt(1.0 - lam) * mulin_separate + jnp.sqrt(lam) * mulin_shared

    @staticmethod
    def _interaction_pattern(p: int, q: Integer[Array, ''] | int) -> Bool[Array, 'p p']:
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

    @classmethod
    def _generate_A_shared(
        cls,
        key: Key[Array, ''],
        p: int,
        q: Integer[Array, ''],
        sigma2_quad: Float[Array, ''],
        kurt_x: float,
    ) -> Float[Array, 'p p']:
        """Generate shared quadratic coefficients for the lambda=1 case."""
        interaction_pattern: Bool[Array, 'p p'] = cls._interaction_pattern(p, q)
        A_shared: Float[Array, 'p p'] = random.normal(key, (p, p))
        A_shared = jnp.where(interaction_pattern, A_shared, 0.0)
        sigma2_A = sigma2_quad / (p * (kurt_x - 1 + q))
        return A_shared * jnp.sqrt(sigma2_A)

    @staticmethod
    def _partitioned_interaction_pattern(
        partition: Bool[Array, 'c p'], q: Integer[Array, ''] | int
    ) -> Bool[Array, 'c p p']:
        """Create c interaction patterns that use disjoint variable sets.

        Parameters
        ----------
        partition
            Binary partition of shape (c, p) indicating variable assignments
            to components
        q
            Number of interactions per predictor (must be even and < p // c)

        Returns
        -------
        Interaction patterns of shape (c, p, p)
        """
        c, p = partition.shape
        q = error_if(q, q % 2 != 0, 'q must be even')
        q = error_if(q, q >= p // c, 'q must be less than p // c')

        indices: Int[Array, 'c p'] = jnp.cumsum(partition, axis=1)
        linear_dist: Int[Array, 'c p p'] = jnp.abs(
            indices[:, :, None] - indices[:, None, :]
        )
        num_vars: Int[Array, 'c'] = jnp.max(indices, axis=1)
        wrapped_dist: Int[Array, 'c p p'] = jnp.minimum(
            linear_dist, num_vars[:, None, None] - linear_dist
        )
        interacts: Bool[Array, 'c p p'] = wrapped_dist <= (q // 2)
        interacts = jnp.where(partition[:, :, None], interacts, False)
        return jnp.where(partition[:, None, :], interacts, False)

    @classmethod
    def _generate_A_separate(
        cls,
        key: Key[Array, ''],
        partition: Bool[Array, 'c p'],
        q: Integer[Array, ''],
        sigma2_quad: Float[Array, ''],
        kurt_x: float,
    ) -> Float[Array, 'c p p']:
        """Generate separate quadratic coefficients for the lambda=0 case."""
        c, p = partition.shape
        A_separate: Float[Array, 'c p p'] = random.normal(key, (c, p, p))
        component_pattern: Bool[Array, 'c p p'] = cls._partitioned_interaction_pattern(
            partition, q
        )
        A_separate = jnp.where(component_pattern, A_separate, 0.0)
        sigma2_A = sigma2_quad / (p / c * (kurt_x - 1 + q))
        return A_separate * jnp.sqrt(sigma2_A)

    @staticmethod
    def _compute_muquad_shared(
        A_shared: Float[Array, 'p p'], x: Float[Array, 'p n']
    ) -> Float[Array, ' n']:
        """Compute quadratic mean for the lambda=1 case.

        muquad_ij = A_kl x_kj x_lj
        Rows identical across components.
        """
        return jnp.einsum('kl,kj,lj->j', A_shared, x, x)

    @staticmethod
    def _compute_muquad_separate(
        A_separate: Float[Array, 'c p p'], x: Float[Array, 'p n']
    ) -> Float[Array, 'c n']:
        """Compute quadratic mean for the lambda=0 case.

        muquad_ij = A_ikl x_kj x_lj
        Rows independent across components.
        """
        return jnp.einsum('ikl,kj,lj->ij', A_separate, x, x)

    @staticmethod
    def _combine_muquad(
        muquad_shared: Float[Array, ' n'],
        muquad_separate: Float[Array, 'c n'],
        lam: Float[Array, ''],
    ) -> Float[Array, 'c n']:
        """Combine shared and separate quadratic means."""
        return jnp.sqrt(1.0 - lam) * muquad_separate + jnp.sqrt(lam) * muquad_shared

    @staticmethod
    def _compute_quadratic_mean(
        A: Float[Array, 'c p p'], x: Float[Array, 'p n']
    ) -> Float[Array, 'c n']:
        """Compute quadratic part of the latent mean."""
        return jnp.einsum('ikl,kj,lj->ij', A, x, x)

    @staticmethod
    def _generate_outcome(
        key: Key[Array, ''], mu: Float[Array, 'c n'], sigma2_eps: Float[Array, '']
    ) -> Float[Array, 'c n']:
        """Generate noisy outcome."""
        eps: Float[Array, 'c n'] = random.normal(key, mu.shape)
        return mu + eps * jnp.sqrt(sigma2_eps)
