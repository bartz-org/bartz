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

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import device_put, random, tree
from jax.scipy import special
from jaxtyping import Array, Float32, Key, Real, Shaped

from bartz._interface import (
    ArrayLike,
    DataFrame,
    FloatLike,
    Series,
    _process_error_variance_settings,
    _process_leaf_variance_settings,
    _process_offset_settings,
    _process_predictor_input,
    _process_response_input,
    predict_latent,
)
from bartz.bcf._loop import run_bcf_mcmc
from bartz.bcf._state import init_bcf
from bartz.mcmcloop import MainTrace
from bartz.mcmcstep import OutcomeType
from bartz.mcmcstep._state import make_p_nonterminal
from bartz.prepcovars import RangeEvenBinner, UniqueQuantileBinner


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


def _serialize_binner(binner: Any, max_split: Any = None) -> dict[str, Any]:  # noqa: ANN401
    """
    Serialize binner state to a dictionary of NumPy arrays.

    Parameters
    ----------
    binner
        The binner object to serialize.
    max_split
        Optional pre-extracted max_split array.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the serialized binner attributes.

    Raises
    ------
    RuntimeError
        If max_split was donated to JAX and not provided explicitly.
    """
    binner_dict = {}
    binner_dict['class'] = binner.__class__.__name__

    if max_split is None:  # pragma: no cover
        try:
            max_split = np.asarray(binner.max_split)
        except RuntimeError as exc:
            msg = 'max_split array was donated to JAX. Pass max_split explicitly.'
            raise RuntimeError(msg) from exc
    binner_dict['max_split'] = np.asarray(max_split)

    # Pylint protected-access (W0212) is explicitly bypassed here because
    # _serialize_binner serves as a dedicated external adapter extracting
    # private trace arrays for NPZ persistence.
    # pylint: disable=protected-access
    if hasattr(binner, '_splits'):
        binner_dict['_splits'] = np.asarray(binner._splits)  # noqa: SLF001

    if hasattr(binner, '_low'):  # pragma: no cover
        binner_dict['_low'] = np.asarray(binner._low)  # noqa: SLF001
        binner_dict['_high'] = np.asarray(binner._high)  # noqa: SLF001
        binner_dict['_max_bins'] = np.asarray(binner._max_bins)  # noqa: SLF001
    # pylint: enable=protected-access

    return binner_dict


def _deserialize_binner(data: Any) -> Any:  # noqa: ANN401
    """
    Reconstruct binner instance from serialized NPZ data dictionary.

    Parameters
    ----------
    data
        A mapping containing the serialized binner fields from NPZ archive.

    Returns
    -------
    binner : Any
        A reconstituted binner instance.
    """
    binner_cls_name = str(data.get('binner.class', 'UniqueQuantileBinner'))
    if binner_cls_name == 'UniqueQuantileBinner':
        binner = object.__new__(UniqueQuantileBinner)
        object.__setattr__(binner, '_splits', jnp.asarray(data['binner._splits']))
        object.__setattr__(binner, 'max_split', jnp.asarray(data['binner.max_split']))
    else:  # pragma: no cover
        binner = object.__new__(RangeEvenBinner)
        object.__setattr__(binner, '_low', jnp.asarray(data['binner._low']))
        object.__setattr__(binner, '_high', jnp.asarray(data['binner._high']))
        object.__setattr__(binner, '_max_bins', int(data['binner._max_bins']))
        object.__setattr__(binner, 'max_split', jnp.asarray(data['binner.max_split']))
    return binner


class bcf(eqx.Module):  # pylint: disable=invalid-name
    R"""
    Bayesian Causal Forests (BCF).

    Regress `y_train` on `x_train` and `z_train` (treatment) with two latent
    mean functions represented as sums of decision trees:
    Y = mu(X, pihat) + tau(X, pihat) * Z + error

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
        The test predictors.
    z_test
        The test treatment assignment.
    pihat_test
        The test propensity scores.
    include_pihat_in_mu
        Whether to include propensity scores in the prognostic forest.
    include_pihat_in_tau
        Whether to include propensity scores in the treatment effect forest.
    num_trees_mu
        The number of trees used for the prognostic forest `mu`.
    num_trees_tau
        The number of trees used for the treatment effect forest `tau`.
    ndpost
        The number of MCMC samples to save, after burn-in.
    nskip
        The number of initial MCMC samples to discard as burn-in.
    k_mu
        Prior parameter k for prognostic forest.
    k_tau
        Prior parameter k for treatment forest.
    sigma_df
        Prior degrees of freedom for error variance.
    sigma_scale
        Prior scale for error variance.
    sigma_init
        Initial value for error variance.
    leaf_prior_cov_inv_mu
        Custom inverse covariance matrix for the prognostic forest leaf prior.
    leaf_prior_cov_inv_tau
        Custom inverse covariance matrix for the treatment effect forest leaf prior.
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
        Whether to standardize the response `y_train` internally during model training.
    outcome_type
        Either 'continuous' or 'binary' (probit link).
    delta_max
        Maximum plausible treatment effect on the probability scale for binary probit models.
    seed
        The seed for the random number generator.

    Raises
    ------
    ValueError
        If binary outcome is specified but `y_train` contains values other than 0 or 1.
        If the format of `x_test` does not match `x_train` format.
    """

    _mcmc_state: Any
    _binner: Any
    _main_trace: Any
    _burnin_trace: Any
    _tau_0_trace: Any
    _b0_trace: Any
    _b1_trace: Any
    _leaf_prior_cov_inv_mu_trace: Any
    _leaf_prior_cov_inv_tau_trace: Any
    _x_train_fmt: Any = eqx.field(static=True, default=None)
    _standardize: bool = eqx.field(static=True, default=False)
    _y_mean: Float32[ArrayLike, ''] | float = eqx.field(default=0.0)
    _y_std: Float32[ArrayLike, ''] | float = eqx.field(default=1.0)
    _outcome_type: str = eqx.field(static=True, default='continuous')
    _offset: Float32[ArrayLike, ''] | float = eqx.field(default=0.0)

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
        k_mu: float = 2.0,
        k_tau: float = 10.0,
        sigma_df: float = 3.0,
        sigma_scale: float | Literal['auto'] = 'auto',
        sigma_init: float | Literal['auto'] = 'auto',
        leaf_prior_cov_inv_mu: FloatLike | Float32[ArrayLike, '*shape'] | None = None,
        leaf_prior_cov_inv_tau: FloatLike | Float32[ArrayLike, '*shape'] | None = None,
        min_points_per_leaf_mu: int = 5,
        min_points_per_leaf_tau: int = 5,
        tau_0_prior_var: float | None = None,
        sample_intercept: bool = True,
        adaptive_coding: bool = False,
        sample_sigma2_leaf_mu: bool = True,
        sigma2_leaf_shape_mu: float = 3.0,
        sigma2_leaf_scale_mu: float | None = None,
        sample_sigma2_leaf_tau: bool = False,
        sigma2_leaf_shape_tau: float = 3.0,
        sigma2_leaf_scale_tau: float | None = None,
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
            y_std_safe = jnp.where(y_std == 0, 1.0, y_std)
            y_train_internal = (y_train - y_mean) / y_std_safe
        else:
            y_mean = jnp.float32(0.0)
            y_std = jnp.float32(1.0)
            y_train_internal = y_train

        self._standardize = standardize
        self._y_mean = y_mean
        self._y_std = y_std

        if pihat_train is not None:
            pihat_train = _process_response_input(pihat_train)

        if x_test is not None:
            x_test, x_test_fmt = _process_bcf_predictor_input(x_test)
            if x_test_fmt != self._x_train_fmt:
                msg = (
                    f'Format of x_test {x_test_fmt} does not match x_train'
                    f' {self._x_train_fmt}'
                )
                raise ValueError(msg)
            if z_test is not None:
                z_test = _process_response_input(z_test)
            if pihat_test is not None:
                pihat_test = _process_response_input(pihat_test)

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
                leaf_prior_cov_inv_mu = jnp.array(num_trees_mu / 1.0, dtype=jnp.float32)
            else:
                leaf_prior_cov_inv_mu = _process_leaf_variance_settings(
                    y_train_internal,
                    binary_mask,
                    missing=None,
                    k=jnp.asarray(k_mu, dtype=jnp.float32),
                    num_trees=num_trees_mu,
                    tau_num=None,
                )
        if leaf_prior_cov_inv_tau is None:
            if outcome_type == 'binary':
                p_val = 0.6827
                q_quantile = special.ndtri((p_val + 1) / 2.0)
                phi_0 = 1.0 / jnp.sqrt(2 * jnp.pi)
                sigma2_tau = ((delta_max / (q_quantile * phi_0)) ** 2) / num_trees_tau
                leaf_prior_cov_inv_tau = jnp.array(1.0 / sigma2_tau, dtype=jnp.float32)
            else:
                leaf_prior_cov_inv_tau = _process_leaf_variance_settings(
                    y_train_internal,
                    binary_mask,
                    missing=None,
                    k=jnp.asarray(k_tau, dtype=jnp.float32),
                    num_trees=num_trees_tau,
                    tau_num=None,
                )

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

        p_nonterminal_mu = make_p_nonterminal(d=10, alpha=0.95, beta=2.0)
        p_nonterminal_tau = make_p_nonterminal(d=5, alpha=0.25, beta=3.0)

        if sigma2_leaf_scale_mu is None:
            sigma2_leaf_scale_mu = 1.0 / num_trees_mu
        if sigma2_leaf_scale_tau is None:
            sigma2_leaf_scale_tau = 0.5 / num_trees_tau

        # 3.5 Bin the unified data
        rng = random.key(seed) if not isinstance(seed, jax.Array) else seed
        rng, key_binner = random.split(rng)

        binner = UniqueQuantileBinner(x_train_unified, key=key_binner)
        x_train_binned = binner.bin(x_train_unified)
        max_split_mu = jnp.array(binner.max_split)
        max_split_tau = jnp.array(binner.max_split)

        if pihat_index is not None:
            if not include_pihat_in_mu:
                max_split_mu = max_split_mu.at[pihat_index].set(0)
            if not include_pihat_in_tau:
                # Block splits on propensity score for tau
                max_split_tau = max_split_tau.at[pihat_index].set(0)

        # 4. Initialize BCFState (single subclass)
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
            leaf_prior_cov_inv_mu=leaf_prior_cov_inv_mu,
            leaf_prior_cov_inv_tau=leaf_prior_cov_inv_tau,
            min_points_per_leaf_mu=min_points_per_leaf_mu,
            min_points_per_leaf_tau=min_points_per_leaf_tau,
            tau_0_prior_var=tau_0_prior_var,
            sample_intercept=sample_intercept,
            adaptive_coding=adaptive_coding,
            sample_leaf_prior_cov_inv_mu=sample_sigma2_leaf_mu,
            leaf_prior_cov_inv_shape_mu=sigma2_leaf_shape_mu,
            leaf_prior_cov_inv_rate_mu=sigma2_leaf_scale_mu,
            sample_leaf_prior_cov_inv_tau=sample_sigma2_leaf_tau,
            leaf_prior_cov_inv_shape_tau=sigma2_leaf_shape_tau,
            leaf_prior_cov_inv_rate_tau=sigma2_leaf_scale_tau,
            error_cov_inv=error_cov_inv,
        )

        # 5. Run the MCMC loop
        rng, loop_key = random.split(rng)

        final_state, final_carry = run_bcf_mcmc(
            key=loop_key, state=initial_state, n_save=ndpost, n_burn=nskip, n_skip=0
        )
        self._mcmc_state = final_state
        self._binner = binner
        self._tau_0_trace = final_carry.tau_0_main_trace
        self._b0_trace = final_carry.b0_main_trace
        self._b1_trace = final_carry.b1_main_trace
        self._leaf_prior_cov_inv_mu_trace = final_carry.leaf_prior_cov_inv_mu_main_trace
        self._leaf_prior_cov_inv_tau_trace = (
            final_carry.leaf_prior_cov_inv_tau_main_trace
        )

        main_trace_mu = final_carry.mu_main_trace
        main_trace_mu = eqx.tree_at(
            lambda t: t.offset, main_trace_mu, initial_state.forest.offset
        )

        main_trace_tau = final_carry.tau_main_trace
        main_trace_tau = eqx.tree_at(
            lambda t: t.offset,
            main_trace_tau,
            jnp.zeros_like(initial_state.forest.offset),
        )

        self._main_trace = {'mu': main_trace_mu, 'tau': main_trace_tau}
        self._burnin_trace = {
            'mu': final_carry.mu_burnin_trace,
            'tau': final_carry.tau_burnin_trace,
        }

    @classmethod
    def _from_saved_state(
        cls,
        binner: Any,  # noqa: ANN401
        tau_0_trace: Any,  # noqa: ANN401
        b0_trace: Any,  # noqa: ANN401
        b1_trace: Any,  # noqa: ANN401
        main_trace: Any,  # noqa: ANN401
        burnin_trace: Any = None,  # noqa: ANN401
        mcmc_state: Any = None,  # noqa: ANN401
        leaf_prior_cov_inv_mu_trace: Any = None,  # noqa: ANN401
        leaf_prior_cov_inv_tau_trace: Any = None,  # noqa: ANN401
        x_train_fmt: Any = None,  # noqa: ANN401
        standardize: bool = False,
        y_mean: Float32[ArrayLike, ''] | float = 0.0,
        y_std: Float32[ArrayLike, ''] | float = 1.0,
        outcome_type: str = 'continuous',
        offset: Float32[ArrayLike, ''] | float = 0.0,
    ) -> bcf:
        """
        Private factory constructor to initialize bcf instance from restored state.

        Parameters
        ----------
        binner
            The binner instance for continuous predictor transforms.
        tau_0_trace
            Posterior trace of the tau_0 intercept.
        b0_trace
            Posterior trace of the b0 control scaling factor.
        b1_trace
            Posterior trace of the b1 treatment scaling factor.
        main_trace
            Posterior traces for mu and tau forests.
        burnin_trace
            Optional burn-in trace data.
        mcmc_state
            Optional final MCMC state.
        leaf_prior_cov_inv_mu_trace
            Optional prior variance trace for mu forest.
        leaf_prior_cov_inv_tau_trace
            Optional prior variance trace for tau forest.
        x_train_fmt
            Formatting metadata of training predictors.
        standardize
            Whether predictions should be unscaled back to original outcome units.
        y_mean
            Original training outcome mean for unscaling.
        y_std
            Original training outcome standard deviation for unscaling.
        outcome_type
            The regression target type ('continuous' or 'binary').
        offset
            Probit latent scale offset (0.0 for continuous).

        Returns
        -------
        bcf
            A reconstituted `bcf` model ready for prediction.
        """
        model = object.__new__(cls)
        object.__setattr__(model, '_mcmc_state', mcmc_state)
        object.__setattr__(model, '_binner', binner)
        object.__setattr__(model, '_main_trace', main_trace)
        object.__setattr__(model, '_burnin_trace', burnin_trace)
        object.__setattr__(model, '_tau_0_trace', tau_0_trace)
        object.__setattr__(model, '_b0_trace', b0_trace)
        object.__setattr__(model, '_b1_trace', b1_trace)
        object.__setattr__(
            model, '_leaf_prior_cov_inv_mu_trace', leaf_prior_cov_inv_mu_trace
        )
        object.__setattr__(
            model, '_leaf_prior_cov_inv_tau_trace', leaf_prior_cov_inv_tau_trace
        )
        object.__setattr__(model, '_x_train_fmt', x_train_fmt)
        object.__setattr__(model, '_standardize', standardize)
        object.__setattr__(model, '_y_mean', jnp.asarray(y_mean, dtype=jnp.float32))
        object.__setattr__(model, '_y_std', jnp.asarray(y_std, dtype=jnp.float32))
        object.__setattr__(model, '_outcome_type', outcome_type)
        object.__setattr__(model, '_offset', jnp.asarray(offset, dtype=jnp.float32))
        return model

    def save_npz(self, path: str | Path) -> None:
        """
        Save the loaded BCF traces to an NPZ archive.

        Parameters
        ----------
        path
            The file path to save the NPZ archive.
        """
        state = {}

        # Explicit schema versioning
        state['schema_version'] = np.array(1)

        # Serialize binner attributes
        try:
            max_split = np.asarray(self._binner.max_split)
        except RuntimeError:  # pragma: no cover
            # Array was donated to JAX during fit(), recover from state
            max_split = (
                np.asarray(self._mcmc_state.forest_tau.max_split)
                if self._mcmc_state is not None
                else None
            )

        for k, v in _serialize_binner(self._binner, max_split=max_split).items():
            state[f'binner.{k}'] = v

        # Save scalar traces
        state['tau_0_trace'] = np.asarray(self._tau_0_trace)
        state['b0_trace'] = np.asarray(self._b0_trace)
        state['b1_trace'] = np.asarray(self._b1_trace)

        # Save standardization metadata
        state['standardize'] = np.array(self._standardize)
        state['_y_mean'] = np.asarray(self._y_mean)
        state['_y_std'] = np.asarray(self._y_std)

        # Save binary / probit metadata
        state['_outcome_type'] = np.array(self._outcome_type)
        state['_offset'] = np.asarray(self._offset)

        # Save main_trace (mu and tau forests) based on dataclass fields
        for forest_name, trace in self._main_trace.items():
            state[f'main_trace.{forest_name}.class'] = trace.__class__.__name__
            for field_name in trace.__dataclass_fields__:
                # Skip mesh because we cannot easily serialize JAX mesh obj
                if field_name == 'mesh':
                    continue
                val = getattr(trace, field_name)
                if val is not None:
                    state[f'main_trace.{forest_name}.{field_name}'] = np.asarray(val)

        # Save format string if any
        if self._x_train_fmt is not None:
            state['x_train_fmt'] = json.dumps(self._x_train_fmt)

        np.savez_compressed(path, allow_pickle=True, **state)

    @classmethod
    def load_npz(cls, path: str | Path) -> bcf:
        """
        Load BCF traces from an NPZ archive, bypassing __init__ MCMC.

        Parameters
        ----------
        path
            The file path to the saved NPZ archive.

        Returns
        -------
        bcf
            A reconstituted `bcf` object ready for prediction.

        Raises
        ------
        ValueError
            If the schema version in the archive is unsupported.
        """
        with np.load(path, allow_pickle=False) as data:
            # Inspect schema version
            schema_version = int(data.get('schema_version', 1))
            if schema_version != 1:
                msg = f'Unsupported schema version: {schema_version}'
                raise ValueError(msg)

            # Reconstruct binner
            binner = _deserialize_binner(data)

            # Scalar traces
            tau_0_trace = jnp.asarray(data['tau_0_trace'])
            b0_trace = jnp.asarray(data['b0_trace'])
            b1_trace = jnp.asarray(data['b1_trace'])

            # Standardization metadata
            standardize = bool(data.get('standardize', False))
            y_mean = data.get('_y_mean', 0.0)
            y_std = data.get('_y_std', 1.0)
            outcome_type = str(data.get('_outcome_type', 'continuous'))
            offset = data.get('_offset', 0.0)

            # Reconstruct main_trace respecting dataclass field defaults
            main_trace = {}
            for forest_name in ['mu', 'tau']:
                trace = object.__new__(MainTrace)
                for field_name, field_def in MainTrace.__dataclass_fields__.items():
                    key = f'main_trace.{forest_name}.{field_name}'
                    if key in data:
                        val = data[key]
                        if field_name == 'has_chains':
                            val = bool(val)
                        else:
                            val = jnp.asarray(val)
                        object.__setattr__(trace, field_name, val)
                    else:
                        # Respect dataclass default or default_factory if defined
                        if field_def.default is not dataclasses.MISSING:
                            default_val = field_def.default  # pragma: no cover
                        elif field_def.default_factory is not dataclasses.MISSING:
                            default_val = (
                                field_def.default_factory()
                            )  # pragma: no cover
                        else:
                            default_val = None
                        object.__setattr__(trace, field_name, default_val)
                main_trace[forest_name] = trace

            fmt_str = str(data.get('x_train_fmt', 'None'))
            x_train_fmt = None if fmt_str == 'None' else json.loads(fmt_str)

            model = cls._from_saved_state(
                binner=binner,
                tau_0_trace=tau_0_trace,
                b0_trace=b0_trace,
                b1_trace=b1_trace,
                main_trace=main_trace,
                x_train_fmt=x_train_fmt,
                standardize=standardize,
                y_mean=y_mean,
                y_std=y_std,
                outcome_type=outcome_type,
                offset=offset,
            )

            # Push loaded dictionary of arrays back into accelerator memory
            return tree.map(device_put, model)

    def predict(
        self,
        x_test: Real[ArrayLike, 'm p'] | DataFrame,
        *,
        pihat_test: Float32[ArrayLike, ' m'] | Series | None = None,
        include_pihat_in_mu: bool = True,  # noqa: ARG002
        include_pihat_in_tau: bool = False,  # noqa: ARG002
    ) -> dict[str, Float32[Array, 'ndpost m']]:
        """
        Compute predictions for both mu and tau forests at `x_test`.

        Parameters
        ----------
        x_test
            The test predictors.
        pihat_test
            The test propensity scores.
        include_pihat_in_mu
            Whether to include propensity scores in prognostic forest prediction.
        include_pihat_in_tau
            Whether to include propensity scores in treatment effect forest prediction.

        Returns
        -------
        dict
            A dictionary with "mu" and "tau" containing the posterior samples
            of the respective forests evaluated at x_test. Shapes are (ndpost, m).

        Raises
        ------
        ValueError
            If the format of `x_test` does not match `x_train` format.
        """
        x_test, x_test_fmt = _process_bcf_predictor_input(x_test)
        if x_test_fmt != self._x_train_fmt:
            msg = (
                f'Format of x_test {x_test_fmt} does not match x_train'
                f' {self._x_train_fmt}'
            )
            raise ValueError(msg)

        if pihat_test is not None:
            pihat_test = _process_response_input(pihat_test)

        x_test_unified = x_test
        if pihat_test is not None:
            pihat_row = pihat_test[jnp.newaxis, :]
            x_test_unified = jnp.concatenate([x_test_unified, pihat_row], axis=0)

        # Bin the test data
        x_test_binned = self._binner.bin(x_test_unified)

        # Evaluate the sum-of-trees (both forests walk the same unified test matrix)
        mu_latent = predict_latent(x_test_binned, self._main_trace['mu'], 'none')
        tau_latent = predict_latent(x_test_binned, self._main_trace['tau'], 'none')

        # Add the global tau_0 intercept
        tau_latent = tau_latent + self._tau_0_trace[:, jnp.newaxis]

        b0_expanded = self._b0_trace[:, jnp.newaxis]
        b1_expanded = self._b1_trace[:, jnp.newaxis]
        # Control mean: mu(X) + b_0 * (tau(X) + tau_0)
        mu_adjusted = mu_latent + b0_expanded * tau_latent
        # Compute CATE via adaptive coding difference
        cate = (b1_expanded - b0_expanded) * tau_latent

        if getattr(self, '_outcome_type', 'continuous') == 'binary':
            p1 = special.ndtr(mu_latent + tau_latent * b1_expanded)
            p0 = special.ndtr(mu_latent + tau_latent * b0_expanded)
            cate_prob = p1 - p0
            return {
                'mu': mu_adjusted,
                'tau': cate,
                'tau_prob': cate_prob,
                'p1': p1,
                'p0': p0,
            }

        if self._standardize:
            mu_adjusted = mu_adjusted * self._y_std + self._y_mean
            cate = cate * self._y_std

        return {'mu': mu_adjusted, 'tau': cate}

    @property
    def sigma_trace(self) -> Float32[Array, ' ndpost']:
        """The posterior trace of residual standard deviation on the outcome scale."""
        error_cov_inv = self._main_trace['mu'].error_cov_inv
        sigma_internal = 1.0 / jnp.sqrt(error_cov_inv)
        if self._standardize:
            return sigma_internal * self._y_std
        return sigma_internal

    def predict_potential_outcomes(
        self,
        x_test: Real[ArrayLike, 'm p'] | DataFrame,
        *,
        pihat_test: Float32[ArrayLike, ' m'] | Series | None = None,
        rho: float = 0.5,
        key: Key[Array, ''] | int | None = None,
        include_pihat_in_mu: bool = True,
        include_pihat_in_tau: bool = False,
    ) -> dict[str, Float32[Array, 'ndpost m']]:
        """
        Sample joint posterior predictive potential outcomes Y(0), Y(1), and lift.

        Parameters
        ----------
        x_test
            The test predictors.
        pihat_test
            Optional test propensity scores.
        rho
            Cross-world counterfactual noise correlation in [0, 1].
        key
            JAX PRNG key or integer seed for stochastic noise sampling.
        include_pihat_in_mu
            Whether to include propensity scores in prognostic forest prediction.
        include_pihat_in_tau
            Whether to include propensity scores in treatment effect forest prediction.

        Returns
        -------
        dict
            A dictionary mapping outcome names to posterior predictive arrays.

        Raises
        ------
        ValueError
            If rho is not within [0, 1].
        """
        if not 0.0 <= rho <= 1.0:
            msg = f'rho must be in [0, 1], got {rho}'
            raise ValueError(msg)

        if key is None:
            key = random.key(0)
        elif isinstance(key, int):
            key = random.key(key)

        preds = self.predict(
            x_test=x_test,
            pihat_test=pihat_test,
            include_pihat_in_mu=include_pihat_in_mu,
            include_pihat_in_tau=include_pihat_in_tau,
        )
        mu = preds['mu']
        tau = preds['tau']
        ndpost, m = mu.shape

        sigma = self.sigma_trace[:, jnp.newaxis]

        k0, k1 = random.split(key)
        u0 = random.normal(k0, shape=(ndpost, m), dtype=jnp.float32)
        u1 = random.normal(k1, shape=(ndpost, m), dtype=jnp.float32)

        rho_f = jnp.float32(rho)
        eps0 = sigma * u0
        eps1 = sigma * (rho_f * u0 + jnp.sqrt(jnp.maximum(0.0, 1.0 - rho_f**2)) * u1)

        y0_latent = mu + eps0
        y1_latent = mu + tau + eps1

        if getattr(self, '_outcome_type', 'continuous') == 'binary':
            y0 = (y0_latent > 0.0).astype(jnp.float32)
            y1 = (y1_latent > 0.0).astype(jnp.float32)
            delta = y1 - y0
            return {
                'y0': y0,
                'y1': y1,
                'delta': delta,
                'mu': mu,
                'tau': tau,
                'p0': preds['p0'],
                'p1': preds['p1'],
                'tau_prob': preds['tau_prob'],
            }

        y0 = y0_latent
        y1 = y1_latent
        delta = y1 - y0

        return {'y0': y0, 'y1': y1, 'delta': delta, 'mu': mu, 'tau': tau}
