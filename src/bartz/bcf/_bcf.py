# bartz/src/bartz/bcf/_bcf.py
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

"""Bayesian Causal Forests (BCF) interface."""

import sys
from operator import attrgetter
from os import PathLike
from typing import Any, Literal, cast

import equinox as eqx
import jax.numpy as jnp
from equinox import error_if
from jax import lax, random
from jax.scipy.special import ndtr, ndtri
from jaxtyping import Array, Bool, Float32, Key, Real, Shaped, UInt

from bartz._interface import (
    ArrayLike,
    DataFrame,
    FloatLike,
    Series,
    _flatten_chain_sample,
    _guarded_response_variance,
    _process_error_variance_settings,
    _process_leaf_variance_settings,
    _process_offset_settings,
    _process_predictor_input,
    _process_response_input,
    _run_mcmc,
    predict_latent,
)
from bartz._jaxext import is_key, jit, jit_active, split
from bartz._npz import check_class, load_npz, save_npz, serializable
from bartz.bcf._state import init_bcf
from bartz.bcf._step import bcf_step
from bartz.bcf._trace import BCFBurninTrace, BCFMainTrace
from bartz.mcmcloop import MainTrace
from bartz.mcmcloop._trace import Trace
from bartz.mcmcstep import OutcomeType, Wishart
from bartz.mcmcstep._axes import chain_vmap_axes, trace_sample_axes
from bartz.mcmcstep._state import make_p_nonterminal
from bartz.prepcovars import UniqueQuantileBinner

if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:
    # WORKAROUND(python<3.11): typing.NotRequired was added in 3.11, and
    # before then typing.TypedDict ignores typing_extensions.NotRequired
    from typing_extensions import NotRequired, TypedDict


def process_bcf_predictor_input(
    x: Real[ArrayLike, 'n p'] | DataFrame,
) -> tuple[Shaped[Array, 'p n'], Any]:
    """
    Process predictors (one predictor per column) to bartz layout (p, n).

    Parameters
    ----------
    x
        The predictor data.

    Returns
    -------
    tuple[Shaped[Array, "p n"], Any]
        A tuple containing the predictors transposed to (p, n) shape and their
        original format metadata.
    """
    if not isinstance(x, DataFrame):
        x = jnp.asarray(x).T
    return _process_predictor_input(x)


def fold_chains(trace: Trace, path: str) -> Float32[Array, 'num_samples ...']:
    """Fold the chain axis of a trace field into its sample axis, like `predict_latent`."""
    get = attrgetter(path)
    return _flatten_chain_sample(
        get(trace), get(chain_vmap_axes(trace)), get(trace_sample_axes(trace))
    )


@jit
def any_not_binary(x: Float32[Array, ' n']) -> Bool[Array, '']:
    """Check whether any value of `x` is neither 0 nor 1."""
    return jnp.any((x != 0) & (x != 1))


def check_binary(x: Float32[Array, ' n'], name: str) -> Float32[Array, ' n']:
    """Check that the values of the variable `name` are all 0 or 1, outside of jit."""
    if not jit_active() and any_not_binary(x):
        msg = f'Values in `{name}` must be 0 or 1.'
        raise ValueError(msg)
    return x


def check_length(
    a: Shaped[Array, ' n'], a_name: str, x: Shaped[Array, 'p m'], x_name: str
) -> None:
    """Check that `a`, named `a_name`, has a value per row of `x`, named `x_name`."""
    (n,) = a.shape
    _, m = x.shape
    if n != m:
        msg = f'`{a_name}` has length {n}, but `{x_name}` has {m} rows.'
        raise ValueError(msg)


def stack_pihat(
    x: Shaped[Array, 'p n'],
    pihat: Float32[ArrayLike, ' n'] | Series,
    which: Literal['train', 'test'],
) -> Shaped[Array, 'p_plus_1 n']:
    """Append `pihat_train` or `pihat_test` to the predictors as the last one."""
    pihat = _process_response_input(pihat)
    check_length(pihat, f'pihat_{which}', x, f'x_{which}')
    return jnp.concatenate((x, pihat[None, :]))


def make_leaf_prior_cov_inv(
    value: FloatLike, sample: bool, shape: FloatLike, scale: FloatLike
) -> Wishart:
    """Build the leaf precision prior from the inverse-gamma prior on the variance."""
    if sample:
        return Wishart(nu=2 * shape, rate=2 * scale, value=value)
    else:
        return Wishart(nu=None, rate=None, value=value)


class BCFPrediction(TypedDict):
    """The posterior samples returned by `bcf.predict`, chains concatenated."""

    mu: Float32[Array, 'num_samples m']
    """The control mean, on the latent probit scale for binary outcomes."""

    tau: Float32[Array, 'num_samples m']
    """The treatment effect, on the latent probit scale for binary outcomes."""

    tau_prob: NotRequired[Float32[Array, 'num_samples m']]
    """The treatment effect on the probability scale, only for binary outcomes."""

    p1: NotRequired[Float32[Array, 'num_samples m']]
    """The probability of y being True if treated, only for binary outcomes."""

    p0: NotRequired[Float32[Array, 'num_samples m']]
    """The probability of y being True if untreated, only for binary outcomes."""


class BCFPotentialOutcomes(BCFPrediction):
    """
    The posterior predictive samples returned by `bcf.predict_potential_outcomes`.

    Also has all the fields of `BCFPrediction`.
    """

    y0: Float32[Array, 'num_samples m']
    """The outcome if untreated."""

    y1: Float32[Array, 'num_samples m']
    """The outcome if treated."""

    delta: Float32[Array, 'num_samples m']
    """The individual treatment effect ``y1 - y0``."""


@jit(static_argnums=(4,))
def predict(
    x_test: UInt[Array, 'p_or_p_plus_1 m'],
    trace: BCFMainTrace,
    y_mean: Float32[Array, ''],
    y_std: Float32[Array, ''],
    probabilities: bool,
    /,
) -> BCFPrediction:
    """Implement `bcf.predict` on the binned test predictors.

    Return the probit outputs of binary models only if `probabilities`.
    """
    # Evaluate the sum-of-trees (both forests walk the same unified test matrix)
    mu_latent = predict_latent(x_test, trace.mu, 'none')
    tau_latent = predict_latent(x_test, trace.tau, 'none')
    # fold the chains like `predict_latent` does to align the samples
    tau_latent += fold_chains(trace, 'tau_0')[:, None]

    b = fold_chains(trace, 'b')
    b0_expanded = b[:, 0, None]
    b1_expanded = b[:, 1, None]
    # Control mean: mu(X) + b_0 * (tau(X) + tau_0)
    mu_adjusted = mu_latent + b0_expanded * tau_latent
    # Compute CATE via adaptive coding difference
    cate = (b1_expanded - b0_expanded) * tau_latent

    if probabilities:
        p1 = ndtr(mu_latent + tau_latent * b1_expanded)
        p0 = ndtr(mu_latent + tau_latent * b0_expanded)
        return BCFPrediction(mu=mu_adjusted, tau=cate, tau_prob=p1 - p0, p1=p1, p0=p0)
    else:
        # y_mean and y_std are exactly 0 and 1 if the response is not
        # standardized, which is always the case for binary outcomes
        return BCFPrediction(mu=mu_adjusted * y_std + y_mean, tau=cate * y_std)


@jit(static_argnums=(3,))
def expected_outcome(
    mu: Float32[Array, 'num_samples m'],
    tau: Float32[Array, 'num_samples m'],
    z: Bool[Array, ' m'],
    probability: bool,
    /,
) -> Float32[Array, 'num_samples m']:
    """Implement `bcf.yhat_test`, or `bcf.prob_test` if `probability`."""
    yhat = mu + z * tau
    if probability:
        return ndtr(yhat)
    else:
        return yhat


@jit
def sigma_trace(
    trace: MainTrace, y_std: Float32[Array, ''], /
) -> Float32[Array, ' num_samples']:
    """Implement `bcf.sigma_trace`, jitted such that folding the chains does not copy."""
    # y_std is exactly 1 if the response is not standardized
    return lax.rsqrt(fold_chains(trace, 'error_cov_inv')) * y_std


@jit(static_argnums=(5,))
def sample_potential_outcomes(
    key: Key[Array, ''],
    mu: Float32[Array, 'num_samples m'],
    tau: Float32[Array, 'num_samples m'],
    sigma: Float32[Array, ' num_samples'],
    rho: Float32[Array, ''],
    binary: bool,
    /,
) -> tuple[
    Float32[Array, 'num_samples m'],
    Float32[Array, 'num_samples m'],
    Float32[Array, 'num_samples m'],
]:
    """Implement the sampling of `bcf.predict_potential_outcomes`."""
    u0, u1 = random.normal(key, (2, *mu.shape))

    eps0 = sigma[:, None] * u0
    # factored for accuracy at |rho| ~ 1
    eps1 = sigma[:, None] * (rho * u0 + jnp.sqrt((1 - rho) * (1 + rho)) * u1)

    y0 = mu + eps0
    y1 = mu + tau + eps1
    if binary:
        y0 = (y0 > 0.0).astype(jnp.float32)
        y1 = (y1 > 0.0).astype(jnp.float32)
        delta = y1 - y0
    else:
        delta = tau + (eps1 - eps0)
    return y0, y1, delta


@serializable
class bcf(eqx.Module):
    R"""
    Bayesian Causal Forests (BCF).

    Regress `y_train` on `x_train` and `z_train` (treatment) with two latent
    mean functions represented as sums of decision trees:
    Y = mu(X, pihat) + tau(X, pihat) * Z + error

    For continuous outcomes, the hyperparameters with units are on the scale
    of the standardized response, see `standardize`, while the outputs are on
    the scale of `y_train`.

    Parameters
    ----------
    x_train
        The training predictors (confounders/modifiers).
    y_train
        The training responses.
    z_train
        The binary treatment assignment (0 or 1).
    pihat_train
        The estimated propensity scores. If provided, appended to `x_train`.
    x_test
        The test predictors. If provided, predictions at these points are
        computed and stored in `mu_test` and `tau_test`.
    z_test
        The test treatment assignment (0 or 1). If provided together with
        `x_test`, the outcome prediction is stored in `yhat_test`.
    pihat_test
        The test propensity scores. Must be passed together with `pihat_train`.
    include_pihat_in_mu
        Whether to include propensity scores in the prognostic forest.
    include_pihat_in_tau
        Whether to include propensity scores in the treatment effect forest.
    num_trees_mu
        The number of trees used for the prognostic forest `mu`.
    num_trees_tau
        The number of trees used for the treatment effect forest `tau`.
    ndpost
        The number of MCMC samples to save, after burn-in, per chain. The
        posterior samples of all chains are concatenated, for a total of
        ``num_chains * ndpost``.
    nskip
        The number of initial MCMC samples to discard as burn-in, per chain.
    num_chains
        The number of independent MCMC chains. `None` for a single chain
        without an explicit chain axis, which is equivalent to 1.
    k_mu
        Prior parameter k for prognostic forest.
    k_tau
        Prior parameter k for treatment forest.
    sigma_df
        Prior degrees of freedom for error variance.
    sigma_scale
        Prior scale of the error standard deviation.
    sigma_init
        Initial value of the error standard deviation.
    leaf_prior_cov_inv_mu
        Custom leaf prior precision for the prognostic forest.
    leaf_prior_cov_inv_tau
        Custom leaf prior precision for the treatment effect forest.
    min_points_per_leaf_mu
        Minimum data points per leaf for prognostic forest.
    min_points_per_leaf_tau
        Minimum data points per leaf for treatment forest.
    tau_0_prior_var
        Prior variance for global intercept tau_0.
    sample_intercept
        Whether to sample a global treatment intercept `tau_0`.
    adaptive_coding
        Whether to use adaptive coding for the treatment effect.
    sample_sigma2_leaf_mu
        Whether to sample the leaf parameter variance for the prognostic forest.
    sigma2_leaf_shape_mu
        The shape parameter for the Inverse-Gamma prior on the prognostic forest leaf variance.
    sigma2_leaf_scale_mu
        The scale parameter for the Inverse-Gamma prior on the prognostic forest leaf variance.
    sample_sigma2_leaf_tau
        Whether to sample the leaf parameter variance for the treatment effect forest.
    sigma2_leaf_shape_tau
        The shape parameter for the Inverse-Gamma prior on the treatment effect forest leaf variance.
    sigma2_leaf_scale_tau
        The scale parameter for the Inverse-Gamma prior on the treatment effect forest leaf variance.
    standardize
        Whether to standardize `y_train` internally, ignored for binary
        outcomes. If `False`, the hyperparameters with units (`sigma_scale`,
        `sigma_init`, `leaf_prior_cov_inv_*`, `tau_0_prior_var`,
        `sigma2_leaf_scale_*`) are on the scale of `y_train` instead of the
        standardized one.
    outcome_type
        Either 'continuous' or 'binary' (probit link).
    delta_max
        Maximum plausible treatment effect on the probability scale for binary probit models.
    seed
        The seed for the random number generator.

    Raises
    ------
    ValueError
        If `z_train`, `z_test`, or `y_train` for binary outcomes, has values
        other than 0 or 1, or if the format of `x_test` does not match
        `x_train` format, or if `z_test` or `pihat_test` is passed without
        `x_test`, or if only one of `pihat_train` and `pihat_test` is passed,
        or if the length of `y_train`, `z_train` or `pihat_train` does not
        match `x_train`, or the length of `z_test` or `pihat_test` does not
        match `x_test`, or if `pihat_train` is passed but excluded from both
        forests.
    """

    _mcmc_state: Any
    _binner: Any
    _main_trace: BCFMainTrace
    _burnin_trace: BCFBurninTrace
    _x_train_fmt: Any = eqx.field(static=True)
    _has_pihat: bool = eqx.field(static=True)
    _y_mean: Float32[Array, '']
    _y_std: Float32[Array, '']
    _outcome_type: str = eqx.field(static=True)
    _mu_test: Float32[Array, 'num_samples m'] | None = None
    _tau_test: Float32[Array, 'num_samples m'] | None = None
    _z_test: Bool[Array, ' m'] | None = None

    def __init__(  # noqa: C901, PLR0915
        self,
        x_train: Real[ArrayLike, 'n p'] | DataFrame,
        y_train: Float32[ArrayLike, ' n'] | Series,
        z_train: Float32[ArrayLike, ' n'] | Series,
        *,
        pihat_train: Float32[ArrayLike, ' n'] | Series | None = None,
        x_test: Real[ArrayLike, 'm p'] | DataFrame | None = None,
        z_test: Float32[ArrayLike, ' m'] | Series | None = None,
        pihat_test: Float32[ArrayLike, ' m'] | Series | None = None,
        include_pihat_in_mu: bool = True,
        include_pihat_in_tau: bool = False,
        num_trees_mu: int = 250,
        num_trees_tau: int = 100,
        ndpost: int = 1000,
        nskip: int = 100,
        num_chains: int | None = None,
        k_mu: float = 2.0,
        k_tau: float = 10.0,
        sigma_df: float = 3.0,
        sigma_scale: float | Literal['auto'] = 'auto',
        sigma_init: float | Literal['auto'] = 'auto',
        leaf_prior_cov_inv_mu: FloatLike | None = None,
        leaf_prior_cov_inv_tau: FloatLike | None = None,
        min_points_per_leaf_mu: int = 5,
        min_points_per_leaf_tau: int = 5,
        tau_0_prior_var: float | None = None,
        sample_intercept: bool = True,
        adaptive_coding: bool = False,
        sample_sigma2_leaf_mu: bool = True,
        sigma2_leaf_shape_mu: float = 3.0,
        sigma2_leaf_scale_mu: FloatLike | None = None,
        sample_sigma2_leaf_tau: bool = False,
        sigma2_leaf_shape_tau: float = 3.0,
        sigma2_leaf_scale_tau: FloatLike | None = None,
        standardize: bool = True,
        outcome_type: Literal['continuous', 'binary'] = 'continuous',
        delta_max: float = 0.9,
        seed: int | Key[Array, ''] = 0,
    ) -> None:

        # 1. Pre-process the data (convert to arrays and transpose X to (p, n))
        x_train, self._x_train_fmt = process_bcf_predictor_input(x_train)
        y_train = _process_response_input(y_train)
        check_length(y_train, 'y_train', x_train, 'x_train')
        z_train = _process_response_input(z_train)
        check_length(z_train, 'z_train', x_train, 'x_train')
        z_train = check_binary(z_train, 'z_train').astype(bool)

        self._outcome_type = outcome_type

        if outcome_type == 'binary':
            y_train = check_binary(y_train, 'y_train')
            standardize = False

        if standardize:
            y_mean = jnp.mean(y_train)
            y_std = jnp.std(y_train)
            y_std = jnp.where(y_std == 0, 1.0, y_std)
            y_train_internal = (y_train - y_mean) / y_std
        else:
            y_mean = jnp.float32(0.0)
            y_std = jnp.float32(1.0)
            y_train_internal = y_train

        self._y_mean = y_mean
        self._y_std = y_std

        if pihat_train is not None:
            if not include_pihat_in_mu and not include_pihat_in_tau:
                msg = (
                    '`pihat_train` is unused if `include_pihat_in_mu` and'
                    ' `include_pihat_in_tau` are both False.'
                )
                raise ValueError(msg)
            x_train = stack_pihat(x_train, pihat_train, 'train')
        self._has_pihat = pihat_train is not None

        if x_test is None:
            if z_test is not None or pihat_test is not None:
                msg = '`z_test` and `pihat_test` require `x_test`.'
                raise ValueError(msg)
        else:
            x_test = self._process_x_test(x_test, pihat_test)
            if z_test is not None:
                z_test = _process_response_input(z_test)
                check_length(z_test, 'z_test', x_test, 'x_test')
                z_test = check_binary(z_test, 'z_test').astype(bool)

        # 3. Resolve priors for both mu and tau forests
        binary_mask = (
            jnp.ones((), dtype=bool)
            if outcome_type == 'binary'
            else jnp.zeros((), dtype=bool)
        )

        offset = _process_offset_settings(y_train_internal, binary_mask, None, None)

        if leaf_prior_cov_inv_mu is None:
            if outcome_type == 'binary':
                leaf_prior_cov_inv_mu = jnp.array(num_trees_mu, jnp.float32)
            else:
                leaf_prior_cov_inv_mu = _process_leaf_variance_settings(
                    y_train_internal,
                    binary_mask,
                    missing=None,
                    k=jnp.array(k_mu),
                    num_trees=num_trees_mu,
                    tau_num=None,
                    sigma_mu_df=None,
                ).value
        if leaf_prior_cov_inv_tau is None:
            if outcome_type == 'binary':
                p_val = 0.6827
                q_quantile = ndtri((p_val + 1) / 2.0)
                phi_0 = 1.0 / jnp.sqrt(2 * jnp.pi)
                sigma2_tau = ((delta_max / (q_quantile * phi_0)) ** 2) / num_trees_tau
                leaf_prior_cov_inv_tau = jnp.reciprocal(sigma2_tau)
            else:
                leaf_prior_cov_inv_tau = _process_leaf_variance_settings(
                    y_train_internal,
                    binary_mask,
                    missing=None,
                    k=jnp.array(k_tau),
                    num_trees=num_trees_tau,
                    tau_num=None,
                    sigma_mu_df=None,
                ).value

        error_cov_inv = _process_error_variance_settings(
            y_train_internal,
            OutcomeType(outcome_type),
            binary_mask,
            None,
            sigma_df,
            sigma_scale,
            sigma_init,
            None,
        )

        p_nonterminal_mu = make_p_nonterminal(d=6, alpha=0.95, beta=2.0)
        p_nonterminal_tau = make_p_nonterminal(d=6, alpha=0.25, beta=3.0)

        if outcome_type == 'binary':
            var_y = 1.0
        else:
            var_y = _guarded_response_variance(y_train_internal, None, None)
        if sigma2_leaf_scale_mu is None:
            sigma2_leaf_scale_mu = var_y / num_trees_mu
        if sigma2_leaf_scale_tau is None:
            sigma2_leaf_scale_tau = 0.5 * var_y / num_trees_tau

        # 3.5 Bin the unified data
        rng = seed if is_key(seed) else random.key(seed)
        keys = split(rng)

        binner = UniqueQuantileBinner(x_train, key=keys.pop())
        x_train_binned = binner.bin(x_train)
        # copies because `init_bcf` may donate them
        max_split_mu = jnp.copy(binner.max_split)
        max_split_tau = jnp.copy(binner.max_split)

        # pihat is the last predictor
        if self._has_pihat:
            if not include_pihat_in_mu:
                max_split_mu = max_split_mu.at[-1].set(0)
            if not include_pihat_in_tau:
                max_split_tau = max_split_tau.at[-1].set(0)

        # 4. Initialize BCFState
        initial_state = init_bcf(
            X_unified=x_train_binned,
            trt=z_train,
            y=y_train_internal,
            outcome_type=outcome_type,
            offset=offset,
            max_split_mu=max_split_mu,
            max_split_tau=max_split_tau,
            num_trees_mu=num_trees_mu,
            num_trees_tau=num_trees_tau,
            p_nonterminal_mu=p_nonterminal_mu,
            p_nonterminal_tau=p_nonterminal_tau,
            leaf_prior_cov_inv_mu=make_leaf_prior_cov_inv(
                leaf_prior_cov_inv_mu,
                sample_sigma2_leaf_mu,
                sigma2_leaf_shape_mu,
                sigma2_leaf_scale_mu,
            ),
            leaf_prior_cov_inv_tau=make_leaf_prior_cov_inv(
                leaf_prior_cov_inv_tau,
                sample_sigma2_leaf_tau,
                sigma2_leaf_shape_tau,
                sigma2_leaf_scale_tau,
            ),
            min_points_per_leaf_mu=min_points_per_leaf_mu,
            min_points_per_leaf_tau=min_points_per_leaf_tau,
            # ignore all predictors without splits, like `Bart(..., rm_const=True)`
            filter_splitless_vars_mu=jnp.sum(max_split_mu == 0).item(),
            filter_splitless_vars_tau=jnp.sum(max_split_tau == 0).item(),
            tau_0_prior_var=tau_0_prior_var,
            sample_intercept=sample_intercept,
            adaptive_coding=adaptive_coding,
            error_cov_inv=error_cov_inv,
            num_chains=num_chains,
        )

        # 5. Run the MCMC loop
        # WORKAROUND(python<3.12): once `run_mcmc` is generic over the state
        # subclass (PEP 695), the traces will come out typed, dropping the
        # casts.
        final_state, burnin_trace, main_trace = _run_mcmc(
            mcmc_state=initial_state,
            n_save=ndpost,
            n_burn=nskip,
            n_skip=1,
            printevery=100,
            pbar=True,
            key=keys.pop(),
            precompute_predict_train=False,
            run_mcmc_kw=dict(
                step=bcf_step,
                burnin_trace_type=BCFBurninTrace,
                main_trace_type=BCFMainTrace,
            ),
            check_platform=None,
        )
        self._mcmc_state = final_state
        self._binner = binner
        self._main_trace = cast(BCFMainTrace, main_trace)
        self._burnin_trace = cast(BCFBurninTrace, burnin_trace)

        # 6. Predict at the test points, now that the traces are available
        if x_test is not None:
            test_pred = self._predict_unified(x_test, probabilities=False)
            self._mu_test = test_pred['mu']
            self._tau_test = test_pred['tau']
            self._z_test = z_test

    def save_npz(self, path: str | PathLike) -> None:
        """
        Save the fitted model to a compressed npz archive.

        Parameters
        ----------
        path
            The file to write to.
        """
        save_npz(path, self)

    @classmethod
    def load_npz(cls, path: str | PathLike) -> 'bcf':
        """
        Load a model saved with `save_npz`.

        Parameters
        ----------
        path
            The file to read from.

        Returns
        -------
        bcf
            The loaded model, on the default device.
        """
        return check_class(load_npz(path), cls, path)

    def predict(
        self,
        x_test: Real[ArrayLike, 'm p'] | DataFrame,
        *,
        pihat_test: Float32[ArrayLike, ' m'] | Series | None = None,
    ) -> BCFPrediction:
        """
        Compute the control mean and the treatment effect at `x_test`.

        Parameters
        ----------
        x_test
            The test predictors.
        pihat_test
            The test propensity scores, required iff the model was fit with
            `pihat_train`.

        Returns
        -------
        The posterior samples at `x_test`.
        """
        return self._predict_unified(
            self._process_x_test(x_test, pihat_test),
            probabilities=self._outcome_type == 'binary',
        )

    def _process_x_test(
        self,
        x_test: Real[ArrayLike, 'm p'] | DataFrame,
        pihat_test: Float32[ArrayLike, ' m'] | Series | None,
    ) -> Shaped[Array, 'p m'] | Shaped[Array, 'p+1 m']:
        """Check the test inputs against the training ones and stack them."""
        x_test, x_test_fmt = process_bcf_predictor_input(x_test)
        if x_test_fmt != self._x_train_fmt:
            msg = (
                f'Format of x_test {x_test_fmt} does not match x_train'
                f' {self._x_train_fmt}'
            )
            raise ValueError(msg)

        if self._has_pihat and pihat_test is None:
            msg = '`pihat_test` is required, the model was fit with `pihat_train`.'
            raise ValueError(msg)
        elif not self._has_pihat and pihat_test is not None:
            msg = (
                '`pihat_test` is not allowed, the model was fit without `pihat_train`.'
            )
            raise ValueError(msg)
        elif pihat_test is None:
            return x_test
        else:
            return stack_pihat(x_test, pihat_test, 'test')

    def _predict_unified(
        self, x_test_unified: Shaped[Array, 'p_or_p_plus_1 m'], *, probabilities: bool
    ) -> BCFPrediction:
        """Implement `predict` on the test predictors stacked with pihat."""
        return predict(
            self._binner.bin(x_test_unified),
            self._main_trace,
            self._y_mean,
            self._y_std,
            probabilities,
        )

    @property
    def offset(self) -> Float32[Array, '']:
        """The prior mean of the prognostic function.

        On the latent probit scale for binary outcomes.
        """
        # y_mean and y_std are exactly 0 and 1 if the response is not standardized
        return self._mcmc_state.forest.offset * self._y_std + self._y_mean

    @property
    def sigma_trace(self) -> Float32[Array, ' num_samples']:
        """The posterior trace of residual standard deviation on the outcome scale, chains concatenated."""
        return sigma_trace(self._main_trace.mu, self._y_std)

    @property
    def mu_test(self) -> Float32[Array, 'num_samples m'] | None:
        """The control mean at `x_test` for each MCMC iteration.

        On the latent probit scale for binary outcomes.
        """
        return self._mu_test

    @property
    def tau_test(self) -> Float32[Array, 'num_samples m'] | None:
        """The treatment effect at `x_test` for each MCMC iteration.

        On the latent probit scale for binary outcomes.
        """
        return self._tau_test

    @property
    def yhat_test(self) -> Float32[Array, 'num_samples m'] | None:
        """The expected outcome at `x_test` under `z_test` for each MCMC iteration.

        On the latent probit scale for binary outcomes; see `prob_test`.
        """
        if self._z_test is None:
            return None
        else:
            assert self._mu_test is not None
            assert self._tau_test is not None
            return expected_outcome(self._mu_test, self._tau_test, self._z_test, False)

    @property
    def prob_test(self) -> Float32[Array, 'num_samples m'] | None:
        """The probability of y being True at `x_test` under `z_test`.

        `None` unless the outcome is binary and `x_test` and `z_test` were
        passed to the constructor.
        """
        if self._z_test is None or self._outcome_type != 'binary':
            return None
        else:
            assert self._mu_test is not None
            assert self._tau_test is not None
            return expected_outcome(self._mu_test, self._tau_test, self._z_test, True)

    def predict_potential_outcomes(
        self,
        x_test: Real[ArrayLike, 'm p'] | DataFrame,
        *,
        key: int | Key[Array, ''],
        pihat_test: Float32[ArrayLike, ' m'] | Series | None = None,
        rho: FloatLike = 0.0,
    ) -> BCFPotentialOutcomes:
        """
        Sample joint posterior predictive potential outcomes Y(0), Y(1), and "lift" Y(1) - Y(0).

        Parameters
        ----------
        x_test
            The test predictors.
        key
            A jax random key or an integer seed for sampling the errors.
        pihat_test
            The test propensity scores, required iff the model was fit with
            `pihat_train`.
        rho
            The correlation in [-1, 1] between the errors of `y0` and `y1`.
            The data carry no information on it, see [1]_. It affects only
            `delta` in `BCFPotentialOutcomes`, widening it as `rho` decreases.

        Returns
        -------
        The posterior predictive samples at `x_test`.

        References
        ----------
        .. [1] Imbens, Guido W., and Donald B. Rubin (2015). "Causal Inference
           for Statistics, Social, and Biomedical Sciences: An Introduction".
           Cambridge University Press, section 8.6.
        """
        rho = jnp.asarray(rho)
        # written to also catch nan
        rho = error_if(rho, ~(jnp.abs(rho) <= 1), 'rho must be in [-1, 1]')

        if not is_key(key):
            key = random.key(key)

        preds = self.predict(x_test=x_test, pihat_test=pihat_test)
        y0, y1, delta = sample_potential_outcomes(
            key,
            preds['mu'],
            preds['tau'],
            self.sigma_trace,
            rho,
            self._outcome_type == 'binary',
        )
        return BCFPotentialOutcomes(**preds, y0=y0, y1=y1, delta=delta)
