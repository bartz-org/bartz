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

"""Define `BCFState` and `init_bcf`."""

from dataclasses import replace
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float32, UInt

from bartz._jaxext import field
from bartz.mcmcstep._axes import CHAIN_AXIS
from bartz.mcmcstep._state import ArrayLike, FloatLike, Forest, State, Wishart, init


class BCFState(State):
    """The full MCMC state for a Bayesian Causal Forest.

    The fields inherited from `State` refer to the prognostic (mu) forest.
    """

    forest_tau: Forest
    """The treatment forest (tau)."""

    trt: Bool[Array, ' n'] = field(data=-1)
    """Whether each unit is treated."""

    tau_X: Float32[Array, '*chains n'] | None = field(chains=CHAIN_AXIS, data=-1)
    """The treatment effect predicted by the tau forest at each datapoint,
    `None` if not needed because `b_prior_cov_inv` is `None`."""

    tau_0: Float32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    """Global intercept for the treatment effect."""

    b0: Float32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    """Adaptive coding weight for untreated units."""

    b1: Float32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    """Adaptive coding weight for treated units."""

    b_prior_cov_inv: Float32[Array, ''] | None
    """Prior precision of `b0` and `b1`, `None` to leave them unchanged."""

    tau_0_prior_cov_inv: Float32[Array, ''] | None
    """Prior precision of `tau_0`, `None` to hold `tau_0` at zero."""

    leaf_prior_cov_inv_shape_mu: Float32[Array, ''] | None
    """Shape of the Gamma prior on the mu leaf precision
    `forest.leaf_prior_cov_inv`. Set it and the rate to `None` to hold the
    precision fixed."""

    leaf_prior_cov_inv_rate_mu: Float32[Array, ''] | None
    """Rate of the Gamma prior on the mu leaf precision."""

    leaf_prior_cov_inv_shape_tau: Float32[Array, ''] | None
    """Shape of the Gamma prior on the tau leaf precision
    `forest_tau.leaf_prior_cov_inv`. Set it and the rate to `None` to hold the
    precision fixed."""

    leaf_prior_cov_inv_rate_tau: Float32[Array, ''] | None
    """Rate of the Gamma prior on the tau leaf precision."""


def init_bcf(
    *,
    X_unified: UInt[ArrayLike, 'p n'],
    trt: Bool[ArrayLike, ' n'],
    y: Float32[ArrayLike, ' n'],
    outcome_type: Literal['continuous', 'binary'] = 'continuous',
    offset: FloatLike,
    max_split_mu: UInt[ArrayLike, ' p'],
    max_split_tau: UInt[ArrayLike, ' p'],
    num_trees_mu: int,
    num_trees_tau: int,
    p_nonterminal_mu: Float32[ArrayLike, ' d_mu_minus_1'],
    p_nonterminal_tau: Float32[ArrayLike, ' d_tau_minus_1'],
    leaf_prior_cov_inv_mu: FloatLike,
    leaf_prior_cov_inv_tau: FloatLike,
    min_points_per_leaf_mu: int = 10,
    min_points_per_leaf_tau: int = 10,
    filter_splitless_vars_mu: int = 0,
    filter_splitless_vars_tau: int = 0,
    tau_0_prior_var: FloatLike | None = None,
    sample_intercept: bool = True,
    adaptive_coding: bool = False,
    sample_leaf_prior_cov_inv_mu: bool = True,
    sample_leaf_prior_cov_inv_tau: bool = False,
    leaf_prior_cov_inv_shape_mu: FloatLike = 3.0,
    leaf_prior_cov_inv_shape_tau: FloatLike = 3.0,
    leaf_prior_cov_inv_rate_mu: FloatLike = 1.0,
    leaf_prior_cov_inv_rate_tau: FloatLike = 1.0,
    error_cov_inv: Wishart | None = None,
) -> BCFState:
    """
    Initialize a BCFState as a subclass of State.

    Parameters
    ----------
    X_unified
        The unified binned predictors matrix [X, pihat].
    trt
        The binary treatment assignment.
    y
        The response array.
    outcome_type
        The regression target type ('continuous' or 'binary').
    offset
        The response offset.
    max_split_mu
    max_split_tau
        Maximum splits for the prognostic and treatment forests.
    num_trees_mu
    num_trees_tau
        Number of trees in the prognostic and treatment forests.
    p_nonterminal_mu
    p_nonterminal_tau
        Split priors for the prognostic and treatment forests.
    leaf_prior_cov_inv_mu
    leaf_prior_cov_inv_tau
        Leaf prior precisions of the prognostic and treatment forests.
    min_points_per_leaf_mu
    min_points_per_leaf_tau
        Minimum data points per leaf for the prognostic and treatment forests.
    filter_splitless_vars_mu
    filter_splitless_vars_tau
        The maximum number of predictors without splits that each forest can
        ignore, see `bartz.mcmcstep.init`. Must be known at trace time.
    tau_0_prior_var
        Prior variance for the global treatment intercept `tau_0`.
    sample_intercept
        Whether to sample a global treatment intercept `tau_0`.
    adaptive_coding
        Whether to use adaptive coding for the treatment effect.
    sample_leaf_prior_cov_inv_mu
    sample_leaf_prior_cov_inv_tau
        Whether to sample the leaf prior precisions of the prognostic and
        treatment forests.
    leaf_prior_cov_inv_shape_mu
    leaf_prior_cov_inv_shape_tau
        Shapes of the Gamma priors on the prognostic and treatment leaf
        precisions.
    leaf_prior_cov_inv_rate_mu
    leaf_prior_cov_inv_rate_tau
        Rates of the Gamma priors on the prognostic and treatment leaf
        precisions.
    error_cov_inv
        The Wishart prior on the error precision and its initial value, `None`
        for binary outcomes. See `bartz.mcmcstep.init`.

    Returns
    -------
    BCFState
        The initialized BCFState.
    """
    trt_array = jnp.asarray(trt)

    if not sample_intercept:
        tau_0_prior_cov_inv = None
    elif tau_0_prior_var is not None:
        tau_0_prior_cov_inv = jnp.reciprocal(jnp.asarray(tau_0_prior_var, jnp.float32))
    elif outcome_type == 'binary':
        tau_0_prior_cov_inv = jnp.array(1.0, jnp.float32)
    else:
        tau_0_prior_cov_inv = jnp.reciprocal(jnp.var(jnp.asarray(y)))

    if sample_leaf_prior_cov_inv_mu:
        shape_mu = jnp.asarray(leaf_prior_cov_inv_shape_mu, jnp.float32)
        rate_mu = jnp.asarray(leaf_prior_cov_inv_rate_mu, jnp.float32)
    else:
        shape_mu = None
        rate_mu = None

    if sample_leaf_prior_cov_inv_tau:
        shape_tau = jnp.asarray(leaf_prior_cov_inv_shape_tau, jnp.float32)
        rate_tau = jnp.asarray(leaf_prior_cov_inv_rate_tau, jnp.float32)
    else:
        shape_tau = None
        rate_tau = None

    # 1. Initialize prognostic state (contains base variables, X, offset, resid)
    state_mu = init(
        X=X_unified,
        y=y,
        outcome_type=outcome_type,
        offset=offset,
        max_split=max_split_mu,
        num_trees=num_trees_mu,
        p_nonterminal=p_nonterminal_mu,
        leaf_prior_cov_inv=leaf_prior_cov_inv_mu,
        filter_splitless_vars=filter_splitless_vars_mu,
        min_points_per_leaf=min_points_per_leaf_mu,
        error_cov_inv=error_cov_inv,
    )

    # 2. Initialize treatment state, only its forest is kept
    assert state_mu.resid.dtype == jnp.float32  # to use it as `error_scale`
    state_tau = init(
        X=state_mu.X,
        y=state_mu.y,
        error_scale=state_mu.resid,
        # `error_scale` is stored unchanged by init(), and the bcf step does
        # not need any initial precision scale value to be correct, so we pass
        # `resid` through to to make `init` set up heteroskedasticity without
        # allocating a new buffer
        outcome_type='continuous',
        offset=0.0,
        max_split=max_split_tau,
        num_trees=num_trees_tau,
        p_nonterminal=p_nonterminal_tau,
        leaf_prior_cov_inv=leaf_prior_cov_inv_tau,
        filter_splitless_vars=filter_splitless_vars_tau,
        min_points_per_leaf=min_points_per_leaf_tau,
        # tau is pretend-initialized as continuous outcome, so pass a dummy error_cov_inv
        error_cov_inv=Wishart(nu=0.0, rate=0.0, value=1.0),
    )

    # reclaim the mu buffers that rode through the tau init untouched
    state_mu = replace(
        state_mu, X=state_tau.X, y=state_tau.y, resid=state_tau.error_scale
    )

    if adaptive_coding:
        b0_init = -0.5
        b1_init = 0.5
        b_prior_cov_inv = jnp.array(2.0, jnp.float32)
    else:
        b0_init = 0.0
        b1_init = 1.0
        b_prior_cov_inv = None

    # Assemble everything into the BCFState subclass
    return BCFState(
        # Inherited fields from State (populated from state_mu)
        _chain_anchor=state_mu._chain_anchor,  # noqa: SLF001
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
        trt=trt_array,
        tau_X=jnp.zeros(len(trt_array), dtype=jnp.float32) if adaptive_coding else None,
        tau_0=jnp.zeros((), dtype=jnp.float32),
        b0=jnp.array(b0_init, jnp.float32),
        b1=jnp.array(b1_init, jnp.float32),
        tau_0_prior_cov_inv=tau_0_prior_cov_inv,
        b_prior_cov_inv=b_prior_cov_inv,
        leaf_prior_cov_inv_shape_mu=shape_mu,
        leaf_prior_cov_inv_rate_mu=rate_mu,
        leaf_prior_cov_inv_shape_tau=shape_tau,
        leaf_prior_cov_inv_rate_tau=rate_tau,
    )
