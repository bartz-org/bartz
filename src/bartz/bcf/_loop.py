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

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random, vmap
from jaxtyping import Array, Bool, Float, Float32, Int32, Key, UInt

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
    leaf_prior_cov_inv_mu_main_trace: Float32[Array, '...']
    leaf_prior_cov_inv_tau_main_trace: Float32[Array, '...']


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
    keys = random.split(key, 7)

    # 1. Update prognostic forest (mu)
    # Construct a temporary standard State for the mu step.
    # This prevents JAX from donating and deleting the BCF-specific fields in
    # 'state'.
    temp_mu_state = State(
        _chain_anchor=state._chain_anchor,  # pylint: disable=protected-access # noqa: SLF001
        X=state.X,
        y=state.y,
        z=state.z,
        binary_indices=state.binary_indices,
        resid=state.resid,  # mu residuals
        resid_unit=state.resid_unit,
        resid_eff_scale=state.resid_eff_scale,
        resid_inexact_integral=state.resid_inexact_integral,
        error_cov_inv=state.error_cov_inv,
        error_scale=state.error_scale,
        prec_scale=state.prec_scale,
        inv_sdev_scale=state.inv_sdev_scale,
        inv_sdev_unit=state.inv_sdev_unit,
        n_non_missing=state.n_non_missing,
        sum_diag_prec_scale=state.sum_diag_prec_scale,
        forest=state.forest,  # mu forest
        config=state.config,
    )

    temp_mu_state = step(keys[0], temp_mu_state)

    def sample_leaf_prior_cov_inv_mu() -> Float32[Array, '...']:
        num_active, sum_sq = _compute_leaf_prior_stats(
            temp_mu_state.forest.split_tree, temp_mu_state.forest.leaf_tree
        )
        a = jnp.asarray(
            state.sigma2_leaf_shape_mu + num_active / 2.0, dtype=jnp.float32
        )
        # leaves are stored in `leaf_unit` units; convert their sum of squares to
        # data units so the Gamma update matches the data-scale prior scale
        b = jnp.asarray(
            state.sigma2_leaf_scale_mu
            + sum_sq * jnp.square(temp_mu_state.forest.leaf_unit) / 2.0,
            dtype=jnp.float32,
        )
        return jnp.exp(loggamma(keys[5], a)) / b

    leaf_prior_cov_inv_mu = jax.lax.cond(
        state.sample_sigma2_leaf_mu,
        sample_leaf_prior_cov_inv_mu,
        lambda: temp_mu_state.forest.leaf_prior_cov_inv,
    )

    temp_mu_state = eqx.tree_at(
        lambda s: s.forest.leaf_prior_cov_inv, temp_mu_state, leaf_prior_cov_inv_mu
    )

    resid_val = temp_mu_state.resid  # updated global residual R
    latest_error_cov_inv = temp_mu_state.error_cov_inv

    # `resid` is stored scaled: ``resid_unit * resid = data residual``, whereas
    # `tau_0`, `tau_X`, `b0`, `b1`, `sigma2` and `tau_0_prior_var` are on the
    # data scale (matching `_bcf.predict`). Convert `resid` in and out of data
    # units so the scalar Gibbs updates below are unit-consistent for any
    # `resid_unit` (no-op when it is 1). Copy it out because the tau `step`
    # below donates the state that shares this buffer.
    resid_unit = jnp.copy(temp_mu_state.resid_unit)

    # 2. Update tau_0 intercept
    trt_val = jnp.copy(state.trt)
    tau_0 = state.tau_0
    sigma2 = 1.0 / latest_error_cov_inv.value

    # Adaptive coding basis
    b_z = jnp.where(trt_val == 1, state.b1, state.b0)

    # partial residual removing current tau_0 effect, on the data scale
    partial = resid_val * resid_unit + tau_0 * b_z

    prec = jnp.sum(jnp.square(b_z)) / sigma2 + 1.0 / state.tau_0_prior_var
    mean = jnp.sum(b_z * partial) / sigma2 / prec

    def sample_tau_0_fn() -> Float32[Array, '']:
        return mean + random.normal(keys[1], shape=mean.shape) * jax.lax.rsqrt(prec)

    tau_0_new = jax.lax.cond(
        state.sample_intercept, sample_tau_0_fn, lambda: jnp.zeros_like(mean)
    )

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

    # Construct a temporary standard State for the tau step
    temp_tau_state = State(
        _chain_anchor=temp_mu_state._chain_anchor,  # pylint: disable=protected-access # noqa: SLF001
        X=temp_mu_state.X,
        y=temp_mu_state.y,
        z=None,
        binary_indices=None,
        resid=jnp.copy(initial_resid_tau),
        resid_unit=temp_mu_state.resid_unit,
        resid_eff_scale=temp_mu_state.resid_eff_scale,
        resid_inexact_integral=temp_mu_state.resid_inexact_integral,
        error_cov_inv=fixed_error_cov_inv,
        error_scale=None,
        prec_scale=jnp.square(inv_sdev_scale_tau),
        inv_sdev_scale=inv_sdev_scale_tau,
        inv_sdev_unit=temp_mu_state.inv_sdev_unit,
        n_non_missing=temp_mu_state.n_non_missing,
        sum_diag_prec_scale=temp_mu_state.sum_diag_prec_scale,
        forest=state.forest_tau,
        config=temp_mu_state.config,
    )

    temp_tau_state = step(keys[2], temp_tau_state)

    def sample_leaf_prior_cov_inv_tau() -> Float32[Array, '...']:
        num_active, sum_sq = _compute_leaf_prior_stats(
            temp_tau_state.forest.split_tree, temp_tau_state.forest.leaf_tree
        )
        a = jnp.asarray(
            state.sigma2_leaf_shape_tau + num_active / 2.0, dtype=jnp.float32
        )
        # leaves are stored in `leaf_unit` units; convert their sum of squares to
        # data units so the Gamma update matches the data-scale prior scale
        b = jnp.asarray(
            state.sigma2_leaf_scale_tau
            + sum_sq * jnp.square(temp_tau_state.forest.leaf_unit) / 2.0,
            dtype=jnp.float32,
        )
        return jnp.exp(loggamma(keys[6], a)) / b

    leaf_prior_cov_inv_tau = jax.lax.cond(
        state.sample_sigma2_leaf_tau,
        sample_leaf_prior_cov_inv_tau,
        lambda: temp_tau_state.forest.leaf_prior_cov_inv,
    )

    temp_tau_state = eqx.tree_at(
        lambda s: s.forest.leaf_prior_cov_inv, temp_tau_state, leaf_prior_cov_inv_tau
    )

    # Update tau_X! (the residual difference is scaled, bring it to data units)
    tau_X_new = state.tau_X + (initial_resid_tau - temp_tau_state.resid) * resid_unit

    resid_val = jnp.where(
        jnp.abs(b_z) < 1e-10, resid_val, temp_tau_state.resid * b_z_safe
    )

    # 4. Update adaptive coding weights (b0, b1)
    def sample_b0_b1_fn() -> tuple[jax.Array, jax.Array, jax.Array]:
        tau_full = tau_0_new + tau_X_new
        resid_partial = resid_val * resid_unit + tau_full * b_z

        b0_prec = jnp.sum(jnp.square(tau_full) * (trt_val == 0)) / sigma2 + 2.0
        b0_mean = jnp.sum(tau_full * resid_partial * (trt_val == 0)) / sigma2 / b0_prec
        b0_new_val = b0_mean + random.normal(
            keys[3], shape=b0_mean.shape
        ) * jax.lax.rsqrt(b0_prec)

        b1_prec = jnp.sum(jnp.square(tau_full) * (trt_val == 1)) / sigma2 + 2.0
        b1_mean = jnp.sum(tau_full * resid_partial * (trt_val == 1)) / sigma2 / b1_prec
        b1_new_val = b1_mean + random.normal(
            keys[4], shape=b1_mean.shape
        ) * jax.lax.rsqrt(b1_prec)

        b_z_new = jnp.where(trt_val == 1, b1_new_val, b0_new_val)
        resid_val_new = (resid_partial - tau_full * b_z_new) / resid_unit

        return b0_new_val, b1_new_val, resid_val_new

    b0_new, b1_new, resid_val = jax.lax.cond(
        state.adaptive_coding, sample_b0_b1_fn, lambda: (state.b0, state.b1, resid_val)
    )

    # 5. Reconstruct and return the updated BCFState
    return BCFState(
        _chain_anchor=temp_tau_state._chain_anchor,  # pylint: disable=protected-access # noqa: SLF001
        X=temp_tau_state.X,
        y=temp_mu_state.y,
        z=temp_mu_state.z,
        binary_indices=temp_mu_state.binary_indices,
        resid=resid_val,  # updated global residual R
        resid_unit=temp_mu_state.resid_unit,
        resid_eff_scale=temp_mu_state.resid_eff_scale,
        resid_inexact_integral=temp_mu_state.resid_inexact_integral,
        error_cov_inv=latest_error_cov_inv,
        error_scale=temp_mu_state.error_scale,
        prec_scale=temp_mu_state.prec_scale,
        inv_sdev_scale=temp_mu_state.inv_sdev_scale,
        inv_sdev_unit=temp_mu_state.inv_sdev_unit,
        n_non_missing=temp_mu_state.n_non_missing,
        sum_diag_prec_scale=temp_mu_state.sum_diag_prec_scale,
        forest=temp_mu_state.forest,
        config=temp_tau_state.config,
        forest_tau=temp_tau_state.forest,
        resid_tau=temp_tau_state.resid,
        prec_scale_tau=temp_tau_state.prec_scale,
        inv_sdev_scale_tau=temp_tau_state.inv_sdev_scale,
        tau_X=tau_X_new,
        trt=trt_val,
        tau_0=tau_0_new,
        b0=b0_new,
        b1=b1_new,
        tau_0_prior_var=state.tau_0_prior_var,
        leaf_prior_cov_inv_tau=temp_tau_state.forest.leaf_prior_cov_inv,
        sample_intercept=state.sample_intercept,
        adaptive_coding=state.adaptive_coding,
        sample_sigma2_leaf_mu=state.sample_sigma2_leaf_mu,
        sigma2_leaf_shape_mu=state.sigma2_leaf_shape_mu,
        sigma2_leaf_scale_mu=state.sigma2_leaf_scale_mu,
        sample_sigma2_leaf_tau=state.sample_sigma2_leaf_tau,
        sigma2_leaf_shape_tau=state.sigma2_leaf_shape_tau,
        sigma2_leaf_scale_tau=state.sigma2_leaf_scale_tau,
    )


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

    # Helper to represent standard State for tau trace pre-allocation
    temp_tau_state = State(
        _chain_anchor=state._chain_anchor,  # pylint: disable=protected-access # noqa: SLF001
        X=state.X,
        y=state.y,
        z=state.z,
        binary_indices=state.binary_indices,
        resid=state.resid_tau,
        resid_unit=state.resid_unit,
        resid_eff_scale=state.resid_eff_scale,
        resid_inexact_integral=state.resid_inexact_integral,
        error_cov_inv=state.error_cov_inv,
        error_scale=state.error_scale,
        prec_scale=state.prec_scale_tau,
        inv_sdev_scale=state.inv_sdev_scale_tau,
        inv_sdev_unit=state.inv_sdev_unit,
        n_non_missing=state.n_non_missing,
        sum_diag_prec_scale=state.sum_diag_prec_scale,
        forest=state.forest_tau,
        config=state.config,
    )

    # Pre-allocate empty traces
    mu_b_empty = _empty_trace(n_burn, state, BurninTrace)
    tau_b_empty = _empty_trace(n_burn, temp_tau_state, BurninTrace)

    mu_m_empty = _empty_trace(n_save, state, MainTrace)
    tau_m_empty = _empty_trace(n_save, temp_tau_state, MainTrace)

    tau_0_m_empty = jnp.zeros((n_save,))
    b0_m_empty = jnp.zeros((n_save,))
    b1_m_empty = jnp.zeros((n_save,))
    leaf_prior_cov_inv_mu_shape = (
        state.forest.leaf_prior_cov_inv.shape
        if state.forest.leaf_prior_cov_inv is not None
        else ()
    )
    leaf_prior_cov_inv_mu_m_empty = jnp.zeros((n_save, *leaf_prior_cov_inv_mu_shape))
    leaf_prior_cov_inv_tau_shape = (
        state.forest_tau.leaf_prior_cov_inv.shape
        if state.forest_tau.leaf_prior_cov_inv is not None
        else ()
    )
    leaf_prior_cov_inv_tau_m_empty = jnp.zeros((n_save, *leaf_prior_cov_inv_tau_shape))

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
        key, step_key = random.split(carry.key)

        new_state = step_fn(step_key, carry.state)
        i = carry.i_total

        # Calculate trace update indices
        noop_idx = jnp.iinfo(jnp.int32).max
        burnin_idx = jnp.where(i < n_burn, i, noop_idx)
        main_idx = jnp.where(i >= n_burn, (i - n_burn) // (1 + n_skip), noop_idx)

        # Convert state to trace representations
        mu_b = BurninTrace.from_state(new_state)

        temp_tau_state_new = State(
            _chain_anchor=new_state._chain_anchor,  # pylint: disable=protected-access # noqa: SLF001
            X=new_state.X,
            y=new_state.y,
            z=new_state.z,
            binary_indices=new_state.binary_indices,
            resid=new_state.resid_tau,
            resid_unit=new_state.resid_unit,
            resid_eff_scale=new_state.resid_eff_scale,
            resid_inexact_integral=new_state.resid_inexact_integral,
            error_cov_inv=new_state.error_cov_inv,
            error_scale=new_state.error_scale,
            prec_scale=new_state.prec_scale_tau,
            inv_sdev_scale=new_state.inv_sdev_scale_tau,
            inv_sdev_unit=new_state.inv_sdev_unit,
            n_non_missing=new_state.n_non_missing,
            sum_diag_prec_scale=new_state.sum_diag_prec_scale,
            forest=new_state.forest_tau,
            config=new_state.config,
        )
        tau_b = BurninTrace.from_state(temp_tau_state_new)

        mu_m = MainTrace.from_state(new_state)
        tau_m = MainTrace.from_state(temp_tau_state_new)

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
            key=key,
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
