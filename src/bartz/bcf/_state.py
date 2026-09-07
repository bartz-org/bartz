# bartz/src/bartz/bcf/_state.py
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

"""Module defining the BCF State and initialization."""

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Float32, UInt

from bartz._jaxext import field
from bartz.mcmcstep._state import (
    CHAIN_AXIS,
    ArrayLike,
    FloatLike,
    Forest,
    State,
    Wishart,
    chain_vmap_axes,
    init,
)


class BCFState(State):
    """The full MCMC state for a Bayesian Causal Forest.

    Attributes
    ----------
      forest_tau: The treatment forest (tau).
      trt: The treatment variable.
      resid_tau: The residuals of the tau forest.
      prec_scale_tau: Scale on the error precision for the tau forest.
      inv_sdev_scale_tau: Reciprocal of standard deviation scale for the tau
        forest.
      leaf_prior_cov_inv_tau: Prior precision for tau leaf values.
      tau_0: Global intercept for the treatment effect.
      tau_0_prior_var: Prior variance of tau_0.
    """

    forest_tau: Forest
    trt: Float32[Array, ' n'] = field(data=-1)
    resid_tau: Float32[Array, '*chains n'] | Float32[Array, '*chains k n'] = field(
        chains=CHAIN_AXIS, data=-1
    )
    prec_scale_tau: Float32[Array, ' n'] | Float32[Array, 'k k n'] | None = field(
        data=-1
    )
    inv_sdev_scale_tau: Float32[Array, ' n'] | Float32[Array, 'k n'] | None = field(
        data=-1
    )
    tau_X: Float32[Array, ' n'] = field(data=-1)
    tau_0: Float32[Array, '*chains']
    b0: Float32[Array, '*chains']
    b1: Float32[Array, '*chains']

    leaf_prior_cov_inv_tau: Float32[Array, ''] | Float32[Array, 'k k']
    tau_0_prior_var: Float32[Array, '']
    sigma2_leaf_shape_mu: Float32[Array, '']
    sigma2_leaf_scale_mu: Float32[Array, '']
    sigma2_leaf_shape_tau: Float32[Array, '']
    sigma2_leaf_scale_tau: Float32[Array, '']
    sample_intercept: bool = field(static=True)
    adaptive_coding: bool = field(static=True)
    sample_sigma2_leaf_mu: bool = field(static=True)
    sample_sigma2_leaf_tau: bool = field(static=True)

    @property
    def has_chains(self) -> bool:
        """Whether the state is multichain (i.e. has a chain axis)."""
        return self.forest.has_chains

    def num_chains(self) -> int | None:
        """Return the number of chains, or `None` if the state is single-chain."""
        if not self.has_chains:
            return None

        c = chain_vmap_axes(self.forest).var_tree  # pragma: no cover
        return self.forest.var_tree.shape[c]  # pragma: no cover


def init_bcf(
    *,
    X_unified: UInt[ArrayLike, 'p n'],
    trt: Float32[ArrayLike, ' n'],
    y: Float32[ArrayLike, ' n'] | Float32[ArrayLike, ' k n'],
    outcome_type: Literal['continuous', 'binary'] = 'continuous',
    offset: FloatLike | Float[ArrayLike, ' k'],
    max_split_mu: UInt[ArrayLike, ' p'],
    max_split_tau: UInt[ArrayLike, ' p'],
    num_trees_mu: int,
    num_trees_tau: int,
    p_nonterminal_mu: Float32[ArrayLike, ' d_mu_minus_1'],
    p_nonterminal_tau: Float32[ArrayLike, ' d_tau_minus_1'],
    leaf_prior_cov_inv_mu: FloatLike | Float[ArrayLike, 'k k'],
    leaf_prior_cov_inv_tau: FloatLike | Float[ArrayLike, 'k k'],
    min_points_per_leaf_mu: int = 10,
    min_points_per_leaf_tau: int = 10,
    tau_0_prior_var: FloatLike | None = None,
    sample_intercept: bool = True,
    adaptive_coding: bool = False,
    sample_sigma2_leaf_mu: bool = True,
    sigma2_leaf_shape_mu: FloatLike = 3.0,
    sigma2_leaf_scale_mu: FloatLike = 1.0,
    sample_sigma2_leaf_tau: bool = False,
    sigma2_leaf_shape_tau: FloatLike = 3.0,
    sigma2_leaf_scale_tau: FloatLike = 1.0,
    **kwargs: Any,
) -> BCFState:
    """
    Initialize a BCFState as a subclass of State.

    Parameters
    ----------
    X_unified
        The unified binned predictors matrix [X, pihat].
    trt
        The treatment assignment array.
    y
        The response array.
    outcome_type
        The regression target type ('continuous' or 'binary').
    offset
        The response offset.
    max_split_mu
        Maximum splits for prognostic forest.
    max_split_tau
        Maximum splits for treatment forest.
    num_trees_mu
        Number of trees in prognostic forest.
    num_trees_tau
        Number of trees in treatment forest.
    p_nonterminal_mu
        Split prior for prognostic.
    p_nonterminal_tau
        Split prior for treatment.
    leaf_prior_cov_inv_mu
        Leaf variance prior for prognostic.
    leaf_prior_cov_inv_tau
        Leaf variance prior for treatment.
    min_points_per_leaf_mu
        Minimum data points per leaf for prognostic forest.
    min_points_per_leaf_tau
        Minimum data points per leaf for treatment forest.
    tau_0_prior_var
        Prior variance for the global treatment intercept `tau_0`.
    sample_intercept
        Whether to sample a global treatment intercept `tau_0`.
    adaptive_coding
        Whether to use adaptive coding for the treatment effect.
    sample_sigma2_leaf_mu
        Whether to sample leaf variance for prognostic forest.
    sigma2_leaf_shape_mu
        Shape parameter for prior on leaf variance of prognostic forest.
    sigma2_leaf_scale_mu
        Scale parameter for prior on leaf variance of prognostic forest.
    sample_sigma2_leaf_tau
        Whether to sample leaf variance for treatment forest.
    sigma2_leaf_shape_tau
        Shape parameter for prior on leaf variance of treatment forest.
    sigma2_leaf_scale_tau
        Scale parameter for prior on leaf variance of treatment forest.
    **kwargs
        Additional kwargs for the base BART initializer.

    Returns
    -------
    BCFState
        The initialized BCFState.
    """
    trt_array = jnp.asarray(trt, jnp.float32)

    user_filter_splitless = kwargs.pop('filter_splitless_vars', 0)
    filter_splitless_vars_mu = max(
        user_filter_splitless, int(jnp.sum(max_split_mu == 0))
    )
    filter_splitless_vars_tau = max(
        user_filter_splitless, int(jnp.sum(max_split_tau == 0))
    )

    if tau_0_prior_var is None:
        if outcome_type == 'binary':
            tau_0_prior_var_val = jnp.asarray(1.0, jnp.float32)
        else:
            tau_0_prior_var_val = jnp.var(jnp.asarray(y, jnp.float32))
    else:
        tau_0_prior_var_val = jnp.asarray(tau_0_prior_var, jnp.float32)

    y_mu = jnp.copy(y)
    kwargs_mu = jax.tree.map(
        lambda x: jnp.copy(x) if isinstance(x, jax.Array) else x, kwargs
    )

    # We copy X_unified because the first init() call will donate and delete it.
    x_unified_copy = jnp.copy(X_unified)

    # 1. Initialize prognostic state (contains base variables, X, offset, resid)
    state_mu = init(
        X=x_unified_copy,
        y=y_mu,
        outcome_type=outcome_type,
        offset=offset,
        max_split=max_split_mu,
        num_trees=num_trees_mu,
        p_nonterminal=p_nonterminal_mu,
        leaf_prior_cov_inv=leaf_prior_cov_inv_mu,
        filter_splitless_vars=filter_splitless_vars_mu,
        min_points_per_leaf=min_points_per_leaf_mu,
        **kwargs_mu,
    )

    # 2. Initialize treatment state (used to extract tau components, e.g. weights)
    safe_trt = jnp.where(trt_array == 0, 1.0, trt_array)
    error_scale = 1.0 / jnp.abs(safe_trt)
    missing = trt_array == 0

    kwargs_tau = dict(kwargs)
    if outcome_type == 'binary' and kwargs_tau.get('error_cov_inv') is None:
        kwargs_tau['error_cov_inv'] = Wishart(
            nu=0.0, rate=0.0, value=jnp.array(1.0, dtype=jnp.float32)
        )

    state_tau = init(
        X=X_unified,
        y=y,
        outcome_type='continuous',
        offset=0.0,
        max_split=max_split_tau,
        num_trees=num_trees_tau,
        p_nonterminal=p_nonterminal_tau,
        leaf_prior_cov_inv=leaf_prior_cov_inv_tau,
        error_scale=error_scale,
        missing=missing,
        filter_splitless_vars=filter_splitless_vars_tau,
        min_points_per_leaf=min_points_per_leaf_tau,
        **kwargs_tau,
    )

    fixed_error_cov_inv = eqx.tree_at(
        lambda w: (w.nu, w.rate),
        state_tau.error_cov_inv,
        (None, None),
        is_leaf=lambda x: x is None,
    )
    state_tau = eqx.tree_at(lambda s: s.error_cov_inv, state_tau, fixed_error_cov_inv)

    if adaptive_coding:
        b0_init = -0.5
        b1_init = 0.5
    else:
        b0_init = 0.0
        b1_init = 1.0

    # Assemble everything into the BCFState subclass
    return BCFState(
        # Inherited fields from State (populated from state_mu)
        _chain_anchor=state_mu._chain_anchor,  # pylint: disable=protected-access # noqa: SLF001
        X=state_mu.X,
        y=state_mu.y,
        z=state_mu.z,
        binary_indices=state_mu.binary_indices,
        resid=state_mu.resid,  # mu residuals
        resid_unit=state_mu.resid_unit,
        resid_eff_scale=state_mu.resid_eff_scale,
        resid_inexact_integral=state_mu.resid_inexact_integral,
        error_cov_inv=state_mu.error_cov_inv,
        error_scale=state_mu.error_scale,
        prec_scale=state_mu.prec_scale,
        inv_sdev_scale=state_mu.inv_sdev_scale,
        inv_sdev_unit=state_mu.inv_sdev_unit,
        n_non_missing=state_mu.n_non_missing,
        sum_diag_prec_scale=state_mu.sum_diag_prec_scale,
        forest=state_mu.forest,  # mu forest
        config=state_mu.config,
        # Subclass additions
        forest_tau=state_tau.forest,  # tau forest
        resid_tau=state_tau.resid,  # tau residuals
        prec_scale_tau=state_tau.prec_scale,
        inv_sdev_scale_tau=state_tau.inv_sdev_scale,
        trt=trt_array,
        tau_X=jnp.zeros(len(trt_array), dtype=jnp.float32),
        tau_0=jnp.zeros((), dtype=jnp.float32),
        b0=jnp.array(b0_init, dtype=jnp.float32),
        b1=jnp.array(b1_init, dtype=jnp.float32),
        tau_0_prior_var=tau_0_prior_var_val,
        leaf_prior_cov_inv_tau=state_tau.forest.leaf_prior_cov_inv,
        sample_intercept=sample_intercept,
        adaptive_coding=adaptive_coding,
        sample_sigma2_leaf_mu=sample_sigma2_leaf_mu,
        sigma2_leaf_shape_mu=jnp.asarray(sigma2_leaf_shape_mu, jnp.float32),
        sigma2_leaf_scale_mu=jnp.asarray(sigma2_leaf_scale_mu, jnp.float32),
        sample_sigma2_leaf_tau=sample_sigma2_leaf_tau,
        sigma2_leaf_shape_tau=jnp.asarray(sigma2_leaf_shape_tau, jnp.float32),
        sigma2_leaf_scale_tau=jnp.asarray(sigma2_leaf_scale_tau, jnp.float32),
    )
