# bartz/src/bartz/bcf/_loop.py
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

"""Module implementing the BCF MCMC loop."""

from dataclasses import replace
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random, vmap
from jaxtyping import Array, Bool, Float, Float32, Int32, Key, UInt

from bartz._jaxext import split
from bartz._jaxext.random import loggamma
from bartz.bcf._state import BCFState
from bartz.grove._grove import is_actual_leaf
from bartz.mcmcloop._loop import _empty_trace, _set
from bartz.mcmcloop._trace import BurninTrace, MainTrace
from bartz.mcmcstep._state import State
from bartz.mcmcstep._step import step


class _BCFCarry(eqx.Module):
    """Carry used in the BCF loop."""

    state: BCFState
    key: Key[Array, '']
    i_total: Int32[Array, '']

    mu_burnin_trace: BurninTrace
    tau_burnin_trace: BurninTrace
    mu_main_trace: MainTrace
    tau_main_trace: MainTrace

    tau_0_main_trace: Float32[Array, ' n_save']
    b0_main_trace: Float32[Array, ' n_save']
    b1_main_trace: Float32[Array, ' n_save']
    leaf_prior_cov_inv_mu_main_trace: Float32[Array, ' n_save']
    leaf_prior_cov_inv_tau_main_trace: Float32[Array, ' n_save']


def _compute_leaf_prior_stats(
    st: UInt[Array, '*chains num_trees half_tree_size'],
    lt: Float[Array, '*chains num_trees 2*half_tree_size'],
) -> tuple[Int32[Array, '*chains'], Float[Array, '*chains']]:
    """
    Compute the number of active leaves and their sum of squares.

    Parameters
    ----------
    st
        The split tree array of shape (*chains, num_trees, half_tree_size).
    lt
        The leaf tree array of shape (*chains, num_trees, 2*half_tree_size).

    Returns
    -------
    num_active
        The number of active leaves of shape (*chains,).
    sum_sq
        The sum of squares of leaf values of shape (*chains,).
    """
    st_flat = st.reshape(-1, st.shape[-1])
    is_leaf_flat = vmap(lambda s: is_actual_leaf(s, add_bottom_level=True))(st_flat)
    is_leaf = is_leaf_flat.reshape((*st.shape[:-1], is_leaf_flat.shape[-1]))
    num_active = jnp.sum(is_leaf, axis=(-2, -1))
    sum_sq = jnp.sum(jnp.square(lt) * is_leaf, axis=(-2, -1))
    return num_active, sum_sq


def _sample_leaf_prior_cov_inv(
    key: Key[Array, ''],
    state: State,
    shape: Float32[Array, ''],
    rate: Float32[Array, ''],
) -> Float32[Array, '']:
    """
    Draw the leaf prior precision of a forest from its Gamma conditional.

    Parameters
    ----------
    key
        A JAX PRNG key.
    state
        The state holding the forest, with the leaves already updated.
    shape
    rate
        The parameters of the Gamma prior on the precision.

    Returns
    -------
    The sampled leaf prior precision.
    """
    num_active, sum_sq = _compute_leaf_prior_stats(
        state.forest.split_tree, state.forest.leaf_tree
    )
    a = shape + num_active / 2.0
    # leaves are stored in `leaf_unit` units; convert their sum of squares to
    # data units so the Gamma update matches the data-scale prior rate
    b = rate + sum_sq * jnp.square(state.forest.leaf_unit) / 2.0
    return jnp.exp(loggamma(key, a)) / b


@jax.named_call
def bcf_step(key: Key[Array, ''], state: BCFState) -> BCFState:
    """
    Do one BCF MCMC step.

    Parameters
    ----------
    key
        A JAX PRNG key to ensure deterministic sampling.
    state
        The current iteration's BCF state.

    Returns
    -------
    BCFState
        The updated BCF state after a single Gibbs sweep across parameters.
    """
    keys = split(key, 7)

    # 1. Update prognostic forest (mu)
    # `step` rebuilds the state with `replace`, so it preserves the subclass.
    # WORKAROUND(python<3.12): type `step` as generic over the state subclass
    # (PEP 695) instead of casting here, since a TypeVar renders badly in the
    # html documentation.
    state = cast(BCFState, step(keys.pop(), state))

    if state.leaf_prior_cov_inv_shape_mu is not None:
        assert state.leaf_prior_cov_inv_rate_mu is not None
        state = eqx.tree_at(
            lambda s: s.forest.leaf_prior_cov_inv,
            state,
            _sample_leaf_prior_cov_inv(
                keys.pop(),
                state,
                state.leaf_prior_cov_inv_shape_mu,
                state.leaf_prior_cov_inv_rate_mu,
            ),
        )

    resid_val = state.resid  # updated global residual R
    latest_error_cov_inv = state.error_cov_inv

    # `resid` is stored scaled: ``resid_unit * resid = data residual``, whereas
    # `tau_0`, `tau_X`, `b0`, `b1`, `sigma2` and `tau_0_prior_cov_inv` are on the
    # data scale (matching `_bcf.predict`). Convert `resid` in and out of data
    # units so the scalar Gibbs updates below are unit-consistent for any
    # `resid_unit` (no-op when it is 1). Copy it out because the tau `step`
    # below donates the state that shares this buffer.
    resid_unit = jnp.copy(state.resid_unit)

    # 2. Update tau_0 intercept
    trt_val = jnp.copy(state.trt)
    tau_0 = state.tau_0
    sigma2 = 1.0 / latest_error_cov_inv.value

    # Adaptive coding basis
    b_z = jnp.where(trt_val, state.b1, state.b0)

    if state.tau_0_prior_cov_inv is not None:
        # partial residual removing current tau_0 effect, on the data scale
        partial = resid_val * resid_unit + tau_0 * b_z

        prec = jnp.sum(jnp.square(b_z)) / sigma2 + state.tau_0_prior_cov_inv
        mean = jnp.sum(b_z * partial) / sigma2 / prec

        tau_0_new = mean + random.normal(keys.pop(), shape=mean.shape) * jax.lax.rsqrt(
            prec
        )
    else:
        tau_0_new = jnp.zeros_like(tau_0)

    # Update R to reflect new tau_0 (back into scaled storage units)
    resid_val = resid_val - b_z * (tau_0_new - tau_0) / resid_unit

    # 3. Update treatment effect forest (tau)
    # Target for tau is (Y - mu - b_z * tau_0) / b_z.
    # Its residual is target - tau = (Y - mu - b_z*tau_0 - b_z*tau) / b_z
    # We disable sampling of the error variance in the tau step
    # by setting nu=None (the variance was already sampled in the mu step).
    # We copy latest_error_cov_inv.value because the tau step will donate and delete it.
    b_z_safe = jnp.where(jnp.abs(b_z) < 1e-10, 1.0, b_z)
    initial_resid_tau = jnp.where(jnp.abs(b_z) < 1e-10, 0.0, resid_val / b_z_safe)
    inv_sdev_scale_tau = jnp.where(jnp.abs(b_z) < 1e-10, 0.0, jnp.abs(b_z_safe))

    fixed_error_cov_inv = eqx.tree_at(
        lambda w: w.value, latest_error_cov_inv, jnp.copy(latest_error_cov_inv.value)
    )
    fixed_error_cov_inv = eqx.tree_at(
        lambda w: (w.nu, w.rate),
        fixed_error_cov_inv,
        (None, None),
        is_leaf=lambda x: x is None,
    )

    # Swap the tau forest into the forest slot so `step` runs on it; the mu
    # forest rides along in `forest_tau` and is swapped back afterwards. Keep
    # the mu-phase state to restore the fields overwritten by the swap.
    mu_state = state
    state = replace(
        state,
        forest=state.forest_tau,
        forest_tau=state.forest,
        z=None,
        binary_indices=None,
        resid=jnp.copy(initial_resid_tau),
        error_cov_inv=fixed_error_cov_inv,
        error_scale=None,
        prec_scale=jnp.square(inv_sdev_scale_tau),
        inv_sdev_scale=inv_sdev_scale_tau,
    )

    state = cast(BCFState, step(keys.pop(), state))

    if state.leaf_prior_cov_inv_shape_tau is not None:
        assert state.leaf_prior_cov_inv_rate_tau is not None
        state = eqx.tree_at(
            lambda s: s.forest.leaf_prior_cov_inv,
            state,
            _sample_leaf_prior_cov_inv(
                keys.pop(),
                state,
                state.leaf_prior_cov_inv_shape_tau,
                state.leaf_prior_cov_inv_rate_tau,
            ),
        )

    # Update tau_X! (the residual difference is scaled, bring it to data units)
    tau_X_new = (
        None
        if state.tau_X is None
        else state.tau_X + (initial_resid_tau - state.resid) * resid_unit
    )

    resid_val = jnp.where(jnp.abs(b_z) < 1e-10, resid_val, state.resid * b_z_safe)

    # 4. Update adaptive coding weights (b0, b1)
    if state.b_prior_cov_inv is not None:
        assert tau_X_new is not None
        tau_full = tau_0_new + tau_X_new
        resid_partial = resid_val * resid_unit + tau_full * b_z

        b0_prec = (
            jnp.sum(jnp.square(tau_full) * ~trt_val) / sigma2 + state.b_prior_cov_inv
        )
        b0_mean = jnp.sum(tau_full * resid_partial * ~trt_val) / sigma2 / b0_prec
        b0_new = b0_mean + random.normal(
            keys.pop(), shape=b0_mean.shape
        ) * jax.lax.rsqrt(b0_prec)

        b1_prec = (
            jnp.sum(jnp.square(tau_full) * trt_val) / sigma2 + state.b_prior_cov_inv
        )
        b1_mean = jnp.sum(tau_full * resid_partial * trt_val) / sigma2 / b1_prec
        b1_new = b1_mean + random.normal(
            keys.pop(), shape=b1_mean.shape
        ) * jax.lax.rsqrt(b1_prec)

        b_z_new = jnp.where(trt_val, b1_new, b0_new)
        resid_val = (resid_partial - tau_full * b_z_new) / resid_unit
    else:
        b0_new = state.b0
        b1_new = state.b1

    # 5. Swap the forests back and restore the mu-side fields
    return replace(
        state,
        forest=state.forest_tau,
        forest_tau=state.forest,
        z=mu_state.z,
        binary_indices=mu_state.binary_indices,
        resid=resid_val,  # updated global residual R
        resid_eff_scale=mu_state.resid_eff_scale,
        resid_inexact_integral=mu_state.resid_inexact_integral,
        error_cov_inv=latest_error_cov_inv,
        error_scale=mu_state.error_scale,
        prec_scale=mu_state.prec_scale,
        inv_sdev_scale=mu_state.inv_sdev_scale,
        tau_X=tau_X_new,
        trt=trt_val,
        tau_0=tau_0_new,
        b0=b0_new,
        b1=b1_new,
    )


def _tau_view(state: BCFState) -> BCFState:
    """Return the state with the tau forest in the mu forest slot."""
    return replace(state, forest=state.forest_tau, forest_tau=state.forest)


def run_bcf_mcmc(
    key: Key[Array, ''], state: BCFState, n_save: int, n_burn: int, n_skip: int
) -> tuple[BCFState, _BCFCarry]:
    """
    Run the BCF MCMC loop.

    Parameters
    ----------
    key
        The PRNG key for the loop.
    state
        The initial BCF state.
    n_save
        The number of iterations to save.
    n_burn
        The number of iterations to discard as burn-in.
    n_skip
        The number of iterations to skip between saves.

    Returns
    -------
    final_state
        The state at the final iteration.
    final_carry
        The final _BCFCarry containing the populated traces.
    """
    step_fn = bcf_step

    tau_state = _tau_view(state)

    # Pre-allocate empty traces
    mu_b_empty = _empty_trace(n_burn, state, BurninTrace)
    tau_b_empty = _empty_trace(n_burn, tau_state, BurninTrace)

    mu_m_empty = _empty_trace(n_save, state, MainTrace)
    tau_m_empty = _empty_trace(n_save, tau_state, MainTrace)

    tau_0_m_empty = jnp.zeros((n_save,))
    b0_m_empty = jnp.zeros((n_save,))
    b1_m_empty = jnp.zeros((n_save,))
    leaf_prior_cov_inv_mu_m_empty = jnp.zeros((n_save,))
    leaf_prior_cov_inv_tau_m_empty = jnp.zeros((n_save,))

    carry = _BCFCarry(
        state=state,
        key=key,
        i_total=jnp.int32(0),
        mu_burnin_trace=mu_b_empty,
        tau_burnin_trace=tau_b_empty,
        mu_main_trace=mu_m_empty,
        tau_main_trace=tau_m_empty,
        tau_0_main_trace=tau_0_m_empty,
        b0_main_trace=b0_m_empty,
        b1_main_trace=b1_m_empty,
        leaf_prior_cov_inv_mu_main_trace=leaf_prior_cov_inv_mu_m_empty,
        leaf_prior_cov_inv_tau_main_trace=leaf_prior_cov_inv_tau_m_empty,
    )

    n_iters = n_burn + (1 + n_skip) * n_save

    def cond_fn(carry: _BCFCarry) -> Bool[Array, '']:
        return carry.i_total < n_iters

    def body_fn(carry: _BCFCarry) -> _BCFCarry:
        keys = split(carry.key)

        new_state = step_fn(keys.pop(), carry.state)
        i = carry.i_total

        # Calculate trace update indices
        noop_idx = jnp.iinfo(jnp.int32).max
        burnin_idx = jnp.where(i < n_burn, i, noop_idx)
        main_idx = jnp.where(i >= n_burn, (i - n_burn) // (1 + n_skip), noop_idx)

        # Convert state to trace representations
        mu_b = BurninTrace.from_state(new_state)

        tau_state_new = _tau_view(new_state)
        tau_b = BurninTrace.from_state(tau_state_new)

        mu_m = MainTrace.from_state(new_state)
        tau_m = MainTrace.from_state(tau_state_new)

        # Write trace data using mode='drop'
        new_mu_b_trace = _set(carry.mu_burnin_trace, burnin_idx, mu_b)
        new_tau_b_trace = _set(carry.tau_burnin_trace, burnin_idx, tau_b)

        new_mu_m_trace = _set(carry.mu_main_trace, main_idx, mu_m)
        new_tau_m_trace = _set(carry.tau_main_trace, main_idx, tau_m)

        new_tau_0_m_trace = carry.tau_0_main_trace.at[main_idx].set(
            new_state.tau_0, mode='drop'
        )
        new_b0_m_trace = carry.b0_main_trace.at[main_idx].set(new_state.b0, mode='drop')
        new_b1_m_trace = carry.b1_main_trace.at[main_idx].set(new_state.b1, mode='drop')
        new_leaf_prior_cov_inv_mu_m_trace = carry.leaf_prior_cov_inv_mu_main_trace.at[
            main_idx
        ].set(new_state.forest.leaf_prior_cov_inv, mode='drop')
        new_leaf_prior_cov_inv_tau_m_trace = carry.leaf_prior_cov_inv_tau_main_trace.at[
            main_idx
        ].set(new_state.forest_tau.leaf_prior_cov_inv, mode='drop')

        return _BCFCarry(
            state=new_state,
            key=keys.pop(),
            i_total=i + 1,
            mu_burnin_trace=new_mu_b_trace,
            tau_burnin_trace=new_tau_b_trace,
            mu_main_trace=new_mu_m_trace,
            tau_main_trace=new_tau_m_trace,
            tau_0_main_trace=new_tau_0_m_trace,
            b0_main_trace=new_b0_m_trace,
            b1_main_trace=new_b1_m_trace,
            leaf_prior_cov_inv_mu_main_trace=new_leaf_prior_cov_inv_mu_m_trace,
            leaf_prior_cov_inv_tau_main_trace=new_leaf_prior_cov_inv_tau_m_trace,
        )

    final_carry = jax.lax.while_loop(cond_fn, body_fn, carry)
    return final_carry.state, final_carry
