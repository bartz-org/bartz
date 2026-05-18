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
from functools import partial

from equinox import Module, error_if, field
from jax import jit, random
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Integer, Key

from bartz._jaxext import split
from bartz.mcmcstep import OutcomeType


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


def combine_shared_separate(
    shared: Float[Array, ' n'], separate: Float[Array, 'k n'], lambda_: Float[Array, '']
) -> Float[Array, 'k n']:
    """Combine shared and separate components via the lambda_ mixing weights."""
    return jnp.sqrt(1.0 - lambda_) * separate + jnp.sqrt(lambda_) * shared


def interaction_pattern(p: int, q: Integer[Array, ''] | int) -> Bool[Array, 'p p']:
    """Symmetric pattern where each row/col sums to q+1 (q must be even)."""
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
    """Interaction patterns over k disjoint variable sets (q even, < p // k)."""
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


def generate_outcome(
    key: Key[Array, ''],
    mu: Float[Array, 'k n'],
    sigma2_eps: Float[Array, ''],
    outcome_type: OutcomeType | tuple[OutcomeType, ...],
) -> Float[Array, 'k n']:
    """Sample y from mu and sigma2_eps (see `Params` for binary semantics)."""
    eps: Float[Array, 'k n'] = random.normal(key, mu.shape)
    latent = mu + eps * jnp.sqrt(sigma2_eps)
    if outcome_type is OutcomeType.continuous:
        return latent
    if outcome_type is OutcomeType.binary:
        return (latent > 0).astype(latent.dtype)
    binary_mask = jnp.array([t is OutcomeType.binary for t in outcome_type])
    return jnp.where(binary_mask[:, None], (latent > 0).astype(latent.dtype), latent)


class Params(Module):
    """Output of `gen_params`: all DGP quantities that do not depend on `n`.

    For multivariate outputs (``k is not None``) the latent mean for
    component ``i`` at observation ``j`` is

        mu_ij = sqrt(lambda_) * (beta_shared . x_j + x_j^T A_shared x_j)
              + sqrt(1 - lambda_) * (beta_separate_i . x_j + x_j^T A_separate_i x_j)

    with ``lambda_`` in ``[0, 1]`` interpolating between fully independent
    components (``lambda_=0``, each row uses its own coefficients restricted to
    the variables it owns via `partition`) and fully shared ones
    (``lambda_=1``, all rows share the same coefficients).

    For univariate outputs (``k is None``) the separate path is skipped and

        mu_j = beta_shared . x_j + x_j^T A_shared x_j;

    ``partition``, ``beta_separate``, ``A_separate`` and ``lambda_`` are all
    ``None``. The outcome is

        y_ij = mu_ij + eps_ij * sqrt(sigma2_eps),   eps_ij ~iid N(0, 1),

    possibly thresholded at 0 for binary components (see `outcome_type`).
    """

    partition: Bool[Array, 'k p'] | None
    """Predictor-outcome assignment partition of shape (k, p), used only at
    ``lambda_ < 1``. Row ``i`` is the binary mask of predictors assigned to
    component ``i``; rows are disjoint and each has either ``p // k`` or
    ``p // k + 1`` entries. ``None`` in univariate mode (``k is None``)."""

    beta_shared: Float[Array, ' p']
    """Shared linear coefficients of shape (p,), used at ``lambda_ > 0``."""

    beta_separate: Float[Array, 'k p'] | None
    """Separate linear coefficients of shape (k, p), used at ``lambda_ < 1``.
    Row ``i`` is supported on ``partition[i]``. ``None`` in univariate
    mode (``k is None``)."""

    A_shared: Float[Array, 'p p']
    """Shared quadratic coefficients of shape (p, p), used at ``lambda_ > 0``.
    Nonzero on a symmetric band of ``q + 1`` entries per row/col."""

    A_separate: Float[Array, 'k p p'] | None
    """Separate quadratic coefficients of shape (k, p, p), used at
    ``lambda_ < 1``. Slice ``i`` is supported on the outer product of
    ``partition[i]`` with itself. ``None`` in univariate mode
    (``k is None``)."""

    q: Integer[Array, '']
    """Number of quadratic interactions per predictor (even, ``< p // k``)."""

    lambda_: Float[Array, ''] | None
    """Coupling parameter in ``[0, 1]``: 0 = independent components,
    1 = identical components. ``None`` iff univariate (``partition is
    None``), in which case only the shared path contributes to ``mu``."""

    sigma2_lin: Float[Array, '']
    """Prior and expected population variance of the linear term of ``mu``."""

    sigma2_quad: Float[Array, '']
    """Expected population variance of the quadratic term of ``mu``."""

    sigma2_eps: Float[Array, '']
    """Variance of the additive error."""

    outcome_type: OutcomeType | tuple[OutcomeType, ...] = field(static=True)
    """Per-component outcome type, either a single `OutcomeType` applied to
    every row, or a tuple of length ``k`` for mixed outcomes. For binary
    components the continuous latent ``mu + eps * sqrt(sigma2_eps)`` is
    thresholded at 0, yielding 0.0/1.0 floats. Unlike the standard probit
    convention used by `bartz.mcmcstep.init` (which fixes the latent noise
    variance to 1), here the binary latents share the same ``sigma2_eps``
    as the continuous ones, so the marginal success probability is
    ``Phi(mu / sqrt(sigma2_eps))``."""

    kurt_x: float = 9 / 5  # kurtosis of uniform distribution
    """Kurtosis of the predictor distribution. Defaults to ``9 / 5``, the
    kurtosis of the uniform distribution used by `gen_data_from_params`."""

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


class DGP(Module):
    """Output of `gen_data` / `gen_data_from_params`: sampled data and parameters.

    See `Params` for the definition of the generative model. The ``_shared``
    fields are the ``lambda_=1`` limit (common across components), the
    ``_separate`` fields are the ``lambda_=0`` limit (independent across
    components), and the plain names are the realized mix at the sampled
    ``params.lambda_``.
    """

    x: Float[Array, 'p n']
    """Predictors of shape (p, n), marginally mean 0 and variance 1."""

    y: Float[Array, 'k n'] | Float[Array, ' n']
    """Noisy outcomes of shape (k, n), or (n,) if `gen_data` was called with
    ``k=None``."""

    mulin_shared: Float[Array, ' n']
    """Shared linear mean of shape (n,)."""

    mulin_separate: Float[Array, 'k n'] | None
    """Separate linear mean of shape (k, n), rows independent. ``None`` in
    univariate mode (``k is None``)."""

    mulin: Float[Array, 'k n'] | Float[Array, ' n']
    """Linear part of the latent mean of shape (k, n), or (n,) in univariate
    mode (``k is None``, equal to ``mulin_shared``)."""

    muquad_shared: Float[Array, ' n']
    """Shared quadratic mean of shape (n,)."""

    muquad_separate: Float[Array, 'k n'] | None
    """Separate quadratic mean of shape (k, n), rows independent. ``None`` in
    univariate mode (``k is None``)."""

    muquad: Float[Array, 'k n'] | Float[Array, ' n']
    """Quadratic part of the latent mean of shape (k, n), or (n,) in
    univariate mode (``k is None``, equal to ``muquad_shared``)."""

    mu: Float[Array, 'k n'] | Float[Array, ' n']
    """Latent mean ``mulin + muquad`` of shape (k, n), or (n,) in univariate
    mode (``k is None``)."""

    params: Params
    """DGP parameters, see `Params`."""

    def split(self, n_train: int | None = None) -> tuple['DGP', 'DGP']:
        """Split the data into training and test sets.

        Parameters
        ----------
        n_train
            Number of training observations. If None, split in half.

        Returns
        -------
        Two `DGP` object with the train and test splits.
        """
        if n_train is None:
            n_train = self.x.shape[1] // 2
        assert 0 < n_train < self.x.shape[1], 'n_train must be in (0, n)'
        train = replace(
            self,
            x=self.x[:, :n_train],
            y=self.y[..., :n_train],
            mulin_shared=self.mulin_shared[:n_train],
            mulin_separate=(
                None
                if self.mulin_separate is None
                else self.mulin_separate[:, :n_train]
            ),
            mulin=self.mulin[..., :n_train],
            muquad_shared=self.muquad_shared[:n_train],
            muquad_separate=(
                None
                if self.muquad_separate is None
                else self.muquad_separate[:, :n_train]
            ),
            muquad=self.muquad[..., :n_train],
            mu=self.mu[..., :n_train],
        )
        test = replace(
            self,
            x=self.x[:, n_train:],
            y=self.y[..., n_train:],
            mulin_shared=self.mulin_shared[n_train:],
            mulin_separate=(
                None
                if self.mulin_separate is None
                else self.mulin_separate[:, n_train:]
            ),
            mulin=self.mulin[..., n_train:],
            muquad_shared=self.muquad_shared[n_train:],
            muquad_separate=(
                None
                if self.muquad_separate is None
                else self.muquad_separate[:, n_train:]
            ),
            muquad=self.muquad[..., n_train:],
            mu=self.mu[..., n_train:],
        )
        return train, test


@partial(jit, static_argnames=('p', 'k', 'outcome_type'))
def gen_params(
    key: Key[Array, ''],
    *,
    p: int,
    k: int | None,
    q: Integer[Array, ''] | int,
    lambda_: Float[Array, ''] | float | None = None,
    sigma2_lin: Float[Array, ''] | float,
    sigma2_quad: Float[Array, ''] | float,
    sigma2_eps: Float[Array, ''] | float,
    outcome_type: OutcomeType | str | tuple[OutcomeType | str, ...] = 'continuous',
) -> Params:
    """Sample DGP coefficients and parameters (no dependence on `n`).

    See `Params` for the meaning of every parameter and the generative model
    they parametrize.

    Parameters
    ----------
    key
        JAX random key.
    p
        Number of predictors.
    k
        Number of outcome components. If `None`, generate a univariate DGP
        and skip the separate code path: ``partition``, ``beta_separate``,
        ``A_separate`` and ``lambda_`` are all set to ``None`` on the returned
        `Params`, and only the shared coefficients are drawn.
    q
        See `Params`.
    lambda_
        Coupling parameter; must be ``None`` iff ``k is None``. See `Params`.
    sigma2_lin
    sigma2_quad
    sigma2_eps
        See `Params`.
    outcome_type
        ``'continuous'``, ``'binary'``, an `OutcomeType`, or a tuple of length
        ``k`` for mixed outcomes. Tuples with all elements equal are collapsed
        to the scalar form. Tuples are not allowed when ``k is None``. See
        `Params` for the semantics.

    Returns
    -------
    A `Params` with the sampled coefficients and forwarded hyperparameters.

    Raises
    ------
    ValueError
        If ``outcome_type`` is a tuple whose length does not match ``k``, or
        if a tuple ``outcome_type`` is combined with ``k=None``, or if
        ``(lambda_ is None) != (k is None)``.
    """
    if (lambda_ is None) != (k is None):
        msg = (
            'lambda_ must be None when k is None'
            if k is None
            else 'lambda_ is required when k is not None'
        )
        raise ValueError(msg)

    if isinstance(outcome_type, tuple):
        if k is None:
            msg = 'tuple outcome_type requires a multivariate outcome (k != None)'
            raise ValueError(msg)
        types = tuple(OutcomeType(t) for t in outcome_type)
        if len(types) != k:
            msg = f'outcome_type has length {len(types)} but k={k}'
            raise ValueError(msg)
        outcome_type = types[0] if len(set(types)) == 1 else types
    else:
        outcome_type = OutcomeType(outcome_type)

    keys = split(key, 5)

    beta_shared = generate_beta_shared(keys.pop(), p, sigma2_lin)
    A_shared = generate_A_shared(keys.pop(), p, q, sigma2_quad, Params.kurt_x)

    if k is None:
        partition = None
        beta_separate = None
        A_separate = None
    else:
        assert p >= k, 'p must be at least k'
        partition = generate_partition(keys.pop(), p, k)
        beta_separate = generate_beta_separate(keys.pop(), partition, sigma2_lin)
        A_separate = generate_A_separate(
            keys.pop(), partition, q, sigma2_quad, Params.kurt_x
        )

    return Params(
        partition=partition,
        beta_shared=beta_shared,
        beta_separate=beta_separate,
        A_shared=A_shared,
        A_separate=A_separate,
        q=q,
        lambda_=lambda_,
        sigma2_lin=sigma2_lin,
        sigma2_quad=sigma2_quad,
        sigma2_eps=sigma2_eps,
        outcome_type=outcome_type,
    )


@partial(jit, static_argnames=('n',))
def gen_data_from_params(key: Key[Array, ''], params: Params, *, n: int) -> DGP:
    """Sample predictors and outcomes given fixed `params`.

    The output ``y`` always has shape ``(k, n)``; squeezing to ``(n,)`` for
    univariate outputs is done by `gen_data`.

    Parameters
    ----------
    key
        JAX random key.
    params
        DGP parameters from `gen_params`.
    n
        Number of observations.

    Returns
    -------
    A `DGP` object with `params` and the sampled data.
    """
    keys = split(key, 2)
    p = params.beta_shared.shape[0]

    x = generate_x(keys.pop(), n, p)
    mulin_shared = params.beta_shared @ x
    muquad_shared = jnp.einsum('rs,rj,sj->j', params.A_shared, x, x)

    if params.partition is None:
        mulin_separate = None
        muquad_separate = None
        mulin = mulin_shared
        muquad = muquad_shared
    else:
        mulin_separate = params.beta_separate @ x
        muquad_separate = jnp.einsum('irs,rj,sj->ij', params.A_separate, x, x)
        mulin = combine_shared_separate(mulin_shared, mulin_separate, params.lambda_)
        muquad = combine_shared_separate(muquad_shared, muquad_separate, params.lambda_)

    mu = mulin + muquad
    y = generate_outcome(keys.pop(), mu, params.sigma2_eps, params.outcome_type)

    return DGP(
        x=x,
        y=y,
        mulin_shared=mulin_shared,
        mulin_separate=mulin_separate,
        mulin=mulin,
        muquad_shared=muquad_shared,
        muquad_separate=muquad_separate,
        muquad=muquad,
        mu=mu,
        params=params,
    )


@partial(jit, static_argnames=('n', 'p', 'k', 'outcome_type'))
def gen_data(
    key: Key[Array, ''],
    *,
    n: int,
    p: int,
    k: int | None = None,
    q: Integer[Array, ''] | int,
    lambda_: Float[Array, ''] | float | None = None,
    sigma2_lin: Float[Array, ''] | float,
    sigma2_quad: Float[Array, ''] | float,
    sigma2_eps: Float[Array, ''] | float,
    outcome_type: OutcomeType | str | tuple[OutcomeType | str, ...] = 'continuous',
) -> DGP:
    """Generate data from a quadratic multivariate DGP.

    Thin wrapper around `gen_params` followed by `gen_data_from_params`. To
    batch across `n` (e.g. to fit memory), call `gen_params` once and then
    invoke `gen_data_from_params` per batch. See `Params` for the generative
    model and `DGP` for the returned fields.

    Parameters
    ----------
    key
        JAX random key.
    n
        Number of observations.
    p
        Number of predictors.
    k
        Number of outcome components. If `None`, produces a univariate output
        with ``y.shape == (n,)`` and skips the separate code path entirely.
    q
    lambda_
    sigma2_lin
    sigma2_quad
    sigma2_eps
    outcome_type
        Forwarded to `gen_params`; see `Params`.

    Returns
    -------
    A `DGP` object with the sampled data and parameters.
    """
    keys = split(key, 2)
    params = gen_params(
        keys.pop(),
        p=p,
        k=k,
        q=q,
        lambda_=lambda_,
        sigma2_lin=sigma2_lin,
        sigma2_quad=sigma2_quad,
        sigma2_eps=sigma2_eps,
        outcome_type=outcome_type,
    )
    return gen_data_from_params(keys.pop(), params, n=n)
