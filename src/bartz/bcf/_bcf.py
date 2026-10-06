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
from typing import Any, Literal, TypedDict, cast

import equinox as eqx
import jax.numpy as jnp
from equinox import error_if
from jax import lax, random
from jax.scipy import special
from jaxtyping import Array, Float32, Key, Real, Shaped

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
from bartz._jaxext import is_key, split
from bartz._npz import check_class, load_npz, save_npz, serializable
from bartz.bcf._loop import BCFBurninTrace, BCFMainTrace, bcf_step
from bartz.bcf._state import init_bcf
from bartz.mcmcloop._trace import Trace
from bartz.mcmcstep import OutcomeType, Wishart
from bartz.mcmcstep._axes import chain_vmap_axes, trace_sample_axes
from bartz.mcmcstep._state import make_p_nonterminal
from bartz.prepcovars import UniqueQuantileBinner

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:  # WORKAROUND(python<3.11): typing.NotRequired was added in 3.11
    from typing_extensions import NotRequired


def _process_bcf_predictor_input(
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


def _fold_chains(trace: Trace, path: str) -> Float32[Array, 'num_samples ...']:
    """Fold the chain axis of a trace field into its sample axis, like `predict_latent`."""
    get = attrgetter(path)
    return _flatten_chain_sample(
        get(trace), get(chain_vmap_axes(trace)), get(trace_sample_axes(trace))
    )


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
        If binary outcome is specified but `y_train` contains values other
        than 0 or 1, or if the format of `x_test` does not match `x_train`
        format, or if `z_test` or `pihat_test` is passed without `x_test`, or
        if only one of `pihat_train` and `pihat_test` is passed, or if
        `z_test` or `pihat_test` does not match the length of `x_test`.
    """

    _mcmc_state: Any
    _binner: Any
    _main_trace: Any
    _burnin_trace: Any
    _tau_0_trace: Any
    _b_trace: Any
    _leaf_prior_cov_inv_mu_trace: Any
    _leaf_prior_cov_inv_tau_trace: Any
    _x_train_fmt: Any = eqx.field(static=True, default=None)
    _has_pihat: bool = eqx.field(static=True, default=False)
    _standardize: bool = eqx.field(static=True, default=False)
    _y_mean: Float32[ArrayLike, ''] | float = eqx.field(default=0.0)
    _y_std: Float32[ArrayLike, ''] | float = eqx.field(default=1.0)
    _outcome_type: str = eqx.field(static=True, default='continuous')
    _offset: Float32[ArrayLike, ''] | float = eqx.field(default=0.0)
    _mu_test: Float32[Array, 'num_samples m'] | None = eqx.field(default=None)
    _tau_test: Float32[Array, 'num_samples m'] | None = eqx.field(default=None)
    _yhat_test: Float32[Array, 'num_samples m'] | None = eqx.field(default=None)

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
        x_train, self._x_train_fmt = _process_bcf_predictor_input(x_train)
        y_train = _process_response_input(y_train)
        z_train = _process_response_input(z_train)
        z_train = eqx.error_if(
            z_train,
            jnp.any((z_train != 0) & (z_train != 1)),
            'Values in `z_train` must be 0 or 1.',
        ).astype(bool)

        self._outcome_type = outcome_type

        if outcome_type == 'binary':
            if not jnp.all((y_train == 0) | (y_train == 1)):
                msg = 'Values in `y_train` must be strictly 0 or 1 for binary outcomes.'
                raise ValueError(msg)
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

        self._standardize = standardize
        self._y_mean = y_mean
        self._y_std = y_std

        if pihat_train is not None:
            pihat_train = _process_response_input(pihat_train)
        self._has_pihat = pihat_train is not None

        if x_test is None:
            if z_test is not None or pihat_test is not None:
                msg = '`z_test` and `pihat_test` require `x_test`.'
                raise ValueError(msg)
        else:
            x_test = self._process_x_test(x_test, pihat_test)
            _, m = x_test.shape
            if z_test is not None:
                z_test = _process_response_input(z_test)
                (len_z,) = z_test.shape
                if len_z != m:
                    msg = f'`z_test` has length {len_z}, but `x_test` has {m} rows.'
                    raise ValueError(msg)
                z_test = eqx.error_if(
                    z_test,
                    jnp.any((z_test != 0) & (z_test != 1)),
                    'Values in `z_test` must be 0 or 1.',
                ).astype(bool)

        # 2. Append pihat to X to create unified predictor matrix
        x_train_unified = x_train
        pihat_index = None

        if pihat_train is not None:
            # x_train is (p, n), pihat_train is (n,). Add a channel dim to pihat to
            # make it (1, n)
            pihat_row = pihat_train[jnp.newaxis, :]
            x_train_unified = jnp.concatenate([x_train_unified, pihat_row], axis=0)
            pihat_index = x_train_unified.shape[0] - 1

        # 3. Resolve priors for both mu and tau forests
        binary_mask = (
            jnp.ones((), dtype=bool)
            if outcome_type == 'binary'
            else jnp.zeros((), dtype=bool)
        )

        offset_val = _process_offset_settings(y_train_internal, binary_mask, None, None)
        self._offset = offset_val

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
                q_quantile = special.ndtri((p_val + 1) / 2.0)
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

        binner = UniqueQuantileBinner(x_train_unified, key=keys.pop())
        x_train_binned = binner.bin(x_train_unified)
        # copies because `init_bcf` may donate them
        max_split_mu = jnp.copy(binner.max_split)
        max_split_tau = jnp.copy(binner.max_split)

        if pihat_index is not None:
            if not include_pihat_in_mu:
                max_split_mu = max_split_mu.at[pihat_index].set(0)
            if not include_pihat_in_tau:
                # Block splits on propensity score for tau
                max_split_tau = max_split_tau.at[pihat_index].set(0)

        # 4. Initialize BCFState
        initial_state = init_bcf(
            X_unified=x_train_binned,
            trt=z_train,
            y=y_train_internal,
            outcome_type=outcome_type,
            offset=offset_val,
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
        burnin_trace = cast(BCFBurninTrace, burnin_trace)
        main_trace = cast(BCFMainTrace, main_trace)
        self._mcmc_state = final_state
        self._binner = binner
        # fold the chains like `predict_latent` does to align the samples
        self._tau_0_trace = _fold_chains(main_trace, 'tau_0')
        self._b_trace = _fold_chains(main_trace, 'b')
        self._leaf_prior_cov_inv_mu_trace = _fold_chains(
            main_trace, 'mu.leaf_prior_cov_inv'
        )
        self._leaf_prior_cov_inv_tau_trace = _fold_chains(
            main_trace, 'tau.leaf_prior_cov_inv'
        )
        self._main_trace = {'mu': main_trace.mu, 'tau': main_trace.tau}
        self._burnin_trace = {'mu': burnin_trace.mu, 'tau': burnin_trace.tau}

        # 6. Predict at the test points, now that the traces are available
        if x_test is not None:
            test_pred = self._predict_unified(x_test)
            self._mu_test = test_pred['mu']
            self._tau_test = test_pred['tau']
            if z_test is not None:
                self._yhat_test = self._mu_test + z_test * self._tau_test

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
        Compute predictions for both mu and tau forests at `x_test`.

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
        return self._predict_unified(self._process_x_test(x_test, pihat_test))

    def _process_x_test(
        self,
        x_test: Real[ArrayLike, 'm p'] | DataFrame,
        pihat_test: Float32[ArrayLike, ' m'] | Series | None,
    ) -> Shaped[Array, 'p m'] | Shaped[Array, 'p+1 m']:
        """Check the test inputs against the training ones and stack them."""
        x_test, x_test_fmt = _process_bcf_predictor_input(x_test)
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
            pihat_test = _process_response_input(pihat_test)
            _, m = x_test.shape
            (len_pihat,) = pihat_test.shape
            if len_pihat != m:
                msg = f'`pihat_test` has length {len_pihat}, but `x_test` has {m} rows.'
                raise ValueError(msg)
            return jnp.concatenate([x_test, pihat_test[None, :]], axis=0)

    def _predict_unified(
        self, x_test_unified: Shaped[Array, 'p m'] | Shaped[Array, 'p+1 m']
    ) -> BCFPrediction:
        """Implement `predict` on the test predictors stacked with pihat."""
        # Bin the test data
        x_test_binned = self._binner.bin(x_test_unified)

        # Evaluate the sum-of-trees (both forests walk the same unified test matrix)
        mu_latent = predict_latent(x_test_binned, self._main_trace['mu'], 'none')
        tau_latent = predict_latent(x_test_binned, self._main_trace['tau'], 'none')

        # Add the global tau_0 intercept
        tau_latent = tau_latent + self._tau_0_trace[:, jnp.newaxis]

        b0_expanded = self._b_trace[:, 0, jnp.newaxis]
        b1_expanded = self._b_trace[:, 1, jnp.newaxis]
        # Control mean: mu(X) + b_0 * (tau(X) + tau_0)
        mu_adjusted = mu_latent + b0_expanded * tau_latent
        # Compute CATE via adaptive coding difference
        cate = (b1_expanded - b0_expanded) * tau_latent

        if getattr(self, '_outcome_type', 'continuous') == 'binary':
            p1 = special.ndtr(mu_latent + tau_latent * b1_expanded)
            p0 = special.ndtr(mu_latent + tau_latent * b0_expanded)
            cate_prob = p1 - p0
            return BCFPrediction(
                mu=mu_adjusted, tau=cate, tau_prob=cate_prob, p1=p1, p0=p0
            )

        if self._standardize:
            mu_adjusted = mu_adjusted * self._y_std + self._y_mean
            cate = cate * self._y_std

        return BCFPrediction(mu=mu_adjusted, tau=cate)

    @property
    def sigma_trace(self) -> Float32[Array, ' num_samples']:
        """The posterior trace of residual standard deviation on the outcome scale, chains concatenated."""
        error_cov_inv = _fold_chains(self._main_trace['mu'], 'error_cov_inv')
        sigma_internal = lax.rsqrt(error_cov_inv)
        if self._standardize:
            return sigma_internal * self._y_std
        return sigma_internal

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
        """The outcome at `x_test` under `z_test` for each MCMC iteration.

        On the latent probit scale for binary outcomes; see `prob_test`.
        """
        return self._yhat_test

    @property
    def prob_test(self) -> Float32[Array, 'num_samples m'] | None:
        """The probability of y being True at `x_test` under `z_test`.

        `None` unless the outcome is binary and `x_test` and `z_test` were
        passed to the constructor.
        """
        if self._yhat_test is None or self._outcome_type != 'binary':
            return None
        return special.ndtr(self._yhat_test)

    def predict_potential_outcomes(
        self,
        x_test: Real[ArrayLike, 'm p'] | DataFrame,
        *,
        key: int | Key[Array, ''],
        pihat_test: Float32[ArrayLike, ' m'] | Series | None = None,
        rho: FloatLike = 0.0,
    ) -> BCFPotentialOutcomes:
        """
        Sample joint posterior predictive potential outcomes Y(0), Y(1), and lift.

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
        mu = preds['mu']
        tau = preds['tau']
        num_samples, m = mu.shape

        sigma = self.sigma_trace[:, jnp.newaxis]

        keys = split(key)
        u0 = random.normal(keys.pop(), shape=(num_samples, m), dtype=jnp.float32)
        u1 = random.normal(keys.pop(), shape=(num_samples, m), dtype=jnp.float32)

        eps0 = sigma * u0
        # factored for accuracy at |rho| ~ 1
        eps1 = sigma * (rho * u0 + jnp.sqrt((1 - rho) * (1 + rho)) * u1)

        y0_latent = mu + eps0
        y1_latent = mu + tau + eps1

        if getattr(self, '_outcome_type', 'continuous') == 'binary':
            y0 = (y0_latent > 0.0).astype(jnp.float32)
            y1 = (y1_latent > 0.0).astype(jnp.float32)
        else:
            y0 = y0_latent
            y1 = y1_latent

        return BCFPotentialOutcomes(**preds, y0=y0, y1=y1, delta=y1 - y0)
