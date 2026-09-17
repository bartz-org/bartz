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

import jax.numpy as jnp
from equinox import Module, tree_at
from jax import lax, random, vmap
from jaxtyping import Array, Bool, Float, Float32, Int32, Key, UInt

from bartz._jaxext import float32_matmuls, jit, sliced_map, split
from bartz._jaxext.random import loggamma
from bartz.bcf._state import BCFState
from bartz.grove._grove import is_actual_leaf
from bartz.mcmcloop._loop import _empty_trace, _set
from bartz.mcmcloop._trace import BurninTrace, MainTrace
from bartz.mcmcstep._state import Forest, State, StepConfig
from bartz.mcmcstep._step import step, step_trees, sum_resid


class _BCFCarry(Module):
    """Carry used in the BCF loop."""

    state: BCFState
    key: Key[Array, '']
    i_total: Int32[Array, '']

    mu_burnin_trace: BurninTrace
    tau_burnin_trace: BurninTrace
    mu_main_trace: MainTrace
    tau_main_trace: MainTrace

    tau_0_main_trace: Float32[Array, ' n_save']
    b_main_trace: Float32[Array, 'n_save 2']
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


def recompute_prec_trees(
    forest: Forest, prec_scale: Float[Array, ' n'], config: StepConfig
) -> Float32[Array, 'num_trees tree_size']:
    """
    Rebuild `Forest.prec_tree` from scratch for a new `prec_scale`.

    Like the incremental update in `step`, the result is valid at the leaves of
    the largest version of each tree and at the node of the last move.

    Parameters
    ----------
    forest
        The forest whose per-leaf precision cache is stale.
    prec_scale
        The new per-datapoint precision scale.
    config
        The MCMC configuration, for the reduction settings.

    Returns
    -------
    The per-leaf sums of `prec_scale`.
    """
    _, tree_size = forest.leaf_tree.shape

    def one_tree(
        args: tuple[UInt[Array, ' n'], Int32[Array, '']],
    ) -> Float32[Array, ' tree_size']:
        leaf_indices, move_node = args
        # the sum over all bins has the shape of the residual reduction, so use
        # its settings rather than the ones tuned for the two-bin `prec_tree` update
        tree = sum_resid(
            prec_scale,
            leaf_indices,
            tree_size,
            config.resid_reduction_config,
            config.data_sharded,
        )
        children = 2 * move_node + jnp.arange(2)
        return tree.at[move_node].set(tree[children].sum())

    xs = (forest.leaf_indices, forest.move_node)

    def all_trees() -> Float32[Array, 'num_trees tree_size']:
        return vmap(one_tree)(xs)

    if config.prec_count_num_trees is None:
        return all_trees()

    else:

        def tree_batches(
            batch_size: int = config.prec_count_num_trees,
        ) -> Float32[Array, 'num_trees tree_size']:
            return sliced_map(one_tree, xs, batch_size=batch_size)

        # like `compute_prec_trees`, batch the trees on cpu to bound the reduction
        # temporaries
        return lax.platform_dependent(cpu=tree_batches, cuda=all_trees)


@jit(donate_argnums=(1,))
@float32_matmuls
def bcf_step(key: Key[Array, ''], state: BCFState) -> BCFState:
    """
    Do one BCF MCMC step.

    Parameters
    ----------
    key
        A jax random key.
    state
        A BCF mcmc state, as created by `init_bcf`.

    Returns
    -------
    The new BCF mcmc state.

    Notes
    -----
    The memory of the input state is re-used for the output state, so the input
    state can not be used any more after calling `bcf_step`. All this applies
    outside of `jax.jit`.
    """
    keys = split(key, 6)

    # 1. Update prognostic forest (mu)
    # `step` rebuilds the state with `replace`, so it preserves the subclass.
    # WORKAROUND(python<3.12): type `step` as generic over the state subclass
    # (PEP 695) instead of casting here, since a TypeVar renders badly in the
    # html documentation.
    state = cast(BCFState, step(keys.pop(), state))

    if state.leaf_prior_cov_inv_shape_mu is not None:
        assert state.leaf_prior_cov_inv_rate_mu is not None
        state = tree_at(
            lambda s: s.forest.leaf_prior_cov_inv,
            state,
            _sample_leaf_prior_cov_inv(
                keys.pop(),
                state,
                state.leaf_prior_cov_inv_shape_mu,
                state.leaf_prior_cov_inv_rate_mu,
            ),
        )

    # `resid` is stored scaled: ``resid_unit * resid = data residual``, whereas
    # `tau_0`, `tau_X`, `b`, `error_cov_inv` and `tau_0_prior_cov_inv` are
    # on the data scale (matching `_bcf.predict`). Convert `resid` in and out of data
    # units so the scalar Gibbs updates below are unit-consistent for any
    # `resid_unit` (no-op when it is 1).

    # 2. Update tau_0 intercept

    # get coding basis, possibly adaptive so not just 0 and 1
    b_z = state.b[state.trt.astype(int)]

    if state.tau_0_prior_cov_inv is not None:
        # partial residual removing current tau_0 effect, on the data scale
        partial_resid = state.resid * state.resid_unit + state.tau_0 * b_z

        # determine full conditional of tau_0
        prec = (
            jnp.sum(jnp.square(b_z)) * state.error_cov_inv.value
            + state.tau_0_prior_cov_inv
        )
        mean = jnp.sum(b_z * partial_resid) * state.error_cov_inv.value / prec

        # sample tau_0 from full conditional
        tau_0_new = mean + random.normal(keys.pop()) * lax.rsqrt(prec)

        # update state to reflect new tau_0
        state = replace(
            state,
            tau_0=tau_0_new,
            resid=state.resid - b_z * (tau_0_new - state.tau_0) / state.resid_unit,
        )

    # 3. Update treatment effect forest (tau)
    # Target for tau is (Y - mu - b_z * tau_0) / b_z.
    # Its residual is target - tau = (Y - mu - b_z*tau_0 - b_z*tau) / b_z
    b_z_zero = jnp.abs(b_z) < 1e-10
    b_z_safe = jnp.where(b_z_zero, 1.0, b_z)
    mu_resid = state.resid
    initial_resid_tau = jnp.where(b_z_zero, 0.0, mu_resid / b_z_safe)

    # Swap the tau forest into the forest slot and run only the tree step on
    # it; the mu forest rides along in `forest_tau` and is swapped back at the
    # end of this section. The other sub-steps of `step` (latent outcome,
    # error precision, sparsity, step counter) belong to the mu phase alone.
    mu_prec_scale = state.prec_scale
    state = replace(
        state,
        forest=state.forest_tau,
        forest_tau=state.forest,
        resid=initial_resid_tau,
        prec_scale=jnp.square(b_z),
    )

    state = cast(BCFState, step_trees(keys.pop(), state))

    if state.leaf_prior_cov_inv_shape_tau is not None:
        assert state.leaf_prior_cov_inv_rate_tau is not None
        state = tree_at(
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
    if state.tau_X is not None:
        state = replace(
            state,
            tau_X=state.tau_X + (initial_resid_tau - state.resid) * state.resid_unit,
        )

    # Swap the forests back and restore the mu-side fields
    state = replace(
        state,
        forest=state.forest_tau,
        forest_tau=state.forest,
        resid=jnp.where(b_z_zero, mu_resid, state.resid * b_z_safe),
        prec_scale=mu_prec_scale,
    )

    # 4. Update adaptive coding weights b
    if state.b_prior_cov_inv is not None:
        assert state.tau_X is not None
        tau_full = state.tau_0 + state.tau_X
        resid_partial = state.resid * state.resid_unit + tau_full * b_z

        # one Gibbs update per group, control (b0) and treated (b1)
        groups = jnp.stack([~state.trt, state.trt])
        b_prec = (
            jnp.sum(jnp.square(tau_full) * groups, axis=1) * state.error_cov_inv.value
            + state.b_prior_cov_inv
        )
        b_mean = (
            jnp.sum(tau_full * resid_partial * groups, axis=1)
            * state.error_cov_inv.value
            / b_prec
        )
        state = replace(
            state, b=b_mean + random.normal(keys.pop(), (2,)) * lax.rsqrt(b_prec)
        )

        b_z = state.b[state.trt.astype(int)]
        state = replace(
            state, resid=(resid_partial - tau_full * b_z) / state.resid_unit
        )

        # the tau precision scale b_z^2 changed on every datapoint, so the
        # incrementally maintained per-leaf cache of the tau forest is stale
        # everywhere
        state = tree_at(
            lambda s: s.forest_tau.prec_tree,
            state,
            recompute_prec_trees(state.forest_tau, jnp.square(b_z), state.config),
        )

    return state


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
    b_m_empty = jnp.zeros((n_save, 2))
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
        b_main_trace=b_m_empty,
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
        new_b_m_trace = carry.b_main_trace.at[main_idx, :].set(new_state.b, mode='drop')
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
            b_main_trace=new_b_m_trace,
            leaf_prior_cov_inv_mu_main_trace=new_leaf_prior_cov_inv_mu_m_trace,
            leaf_prior_cov_inv_tau_main_trace=new_leaf_prior_cov_inv_tau_m_trace,
        )

    final_carry = lax.while_loop(cond_fn, body_fn, carry)
    return final_carry.state, final_carry
