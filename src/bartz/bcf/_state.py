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
from bartz._npz import serializable
from bartz.mcmcstep._axes import CHAIN_AXIS
from bartz.mcmcstep._state import (
    ArrayLike,
    FloatLike,
    Forest,
    State,
    Wishart,
    init,
    initial_prec_tree,
)


@serializable
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

    b: Float32[Array, '*chains 2'] = field(chains=CHAIN_AXIS)
    """Adaptive coding weights for untreated and treated units."""

    b_prior_cov_inv: Float32[Array, ''] | None
    """Prior precision of `b`, `None` to leave it unchanged."""

    tau_0_prior_cov_inv: Float32[Array, ''] | None
    """Prior precision of `tau_0`, `None` to leave it unchanged."""


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
    leaf_prior_cov_inv_mu: Wishart,
    leaf_prior_cov_inv_tau: Wishart,
    min_points_per_leaf_mu: int,
    min_points_per_leaf_tau: int,
    filter_splitless_vars_mu: int = 0,
    filter_splitless_vars_tau: int = 0,
    tau_0_prior_var: FloatLike | None = None,
    sample_intercept: bool = True,
    adaptive_coding: bool = False,
    error_cov_inv: Wishart | None = None,
    num_chains: int | None = None,
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
        The Wishart priors on the leaf precisions of the prognostic and
        treatment forests and their initial values, see `bartz.mcmcstep.init`.
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
    error_cov_inv
        The Wishart prior on the error precision and its initial value, `None`
        for binary outcomes. See `bartz.mcmcstep.init`.
    num_chains
        The number of independent MCMC chains, `None` for a single chain
        without a chain axis. See `bartz.mcmcstep.init`.

    Returns
    -------
    The initial BCF MCMC state.

    Notes
    -----
    The arrays passed to this function as arguments may be donated,
    invalidating them, see `bartz.mcmcstep.init`.
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
        num_chains=num_chains,
    )

    # 2. Initialize treatment state, only its forest is kept
    state_tau = init(
        X=state_mu.X,
        y=jnp.copy(state_mu.y),
        missing=~trt_array,
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
        num_chains=num_chains,
    )

    # reclaim X, which rode through the tau init untouched
    state_mu = replace(state_mu, X=state_tau.X)

    if adaptive_coding:
        b_init = jnp.array([-0.5, 0.5])
        b_prior_cov_inv = jnp.array(2.0, jnp.float32)
    else:
        b_init = jnp.array([0.0, 1.0])
        b_prior_cov_inv = None

    # the fields built here get a leading chain axis, if any
    chain_shape = () if num_chains is None else (num_chains,)
    (n,) = trt_array.shape

    # the tau likelihood precision of each datapoint is b_z^2 (see `bcf_step`),
    # so seed the tau forest's per-leaf precision cache from the coding weights
    # rather than from the missingness mask used by the tau init
    forest_tau = state_tau.forest
    assert forest_tau.prec_tree is not None
    *_, tree_size = forest_tau.prec_tree.shape
    b_z = coding_basis(b_init, trt_array)
    prec_tree = initial_prec_tree((num_trees_tau, tree_size), jnp.square(b_z))
    forest_tau = replace(
        forest_tau,
        prec_tree=jnp.broadcast_to(prec_tree, (*chain_shape, *prec_tree.shape)),
    )

    return BCFState(
        **vars(state_mu),
        forest_tau=forest_tau,
        trt=trt_array,
        tau_X=jnp.zeros((*chain_shape, n)) if adaptive_coding else None,
        tau_0=jnp.zeros(chain_shape),
        b=jnp.broadcast_to(b_init, (*chain_shape, 2)),
        tau_0_prior_cov_inv=tau_0_prior_cov_inv,
        b_prior_cov_inv=b_prior_cov_inv,
    )


def swap_mu_tau_forests(state: BCFState) -> BCFState:
    """Swap the prognostic and treatment forests."""
    return replace(state, forest=state.forest_tau, forest_tau=state.forest)


def coding_basis(
    b: Float32[Array, ' 2'], trt: Bool[Array, ' n']
) -> Float32[Array, ' n']:
    """Return the coding weight of each unit, ``b[trt]``."""
    return b[trt.astype(int)]
