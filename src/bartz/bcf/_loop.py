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

"""Implement the BCF MCMC step and traces, to be run with `run_mcmc`."""

from dataclasses import replace
from typing import cast

import jax.numpy as jnp
from equinox import tree_at
from jax import lax, random, vmap
from jaxtyping import Array, Float, Float32, Int32, Key, UInt

from bartz._jaxext import field, float32_matmuls, jit, sliced_map, split
from bartz.bcf._state import BCFState
from bartz.mcmcloop._trace import BurninTrace, MainTrace, Trace
from bartz.mcmcstep._axes import CHAIN_AXIS
from bartz.mcmcstep._state import (
    Forest,
    State,
    StepConfig,
    split_key_for_chains,
    vmap_chains,
)
from bartz.mcmcstep._step import step, step_leaf_prior_cov_inv, step_trees, sum_resid


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


def bcf_step_mu(key: Key[Array, ''], state: BCFState) -> BCFState:
    """Update the prognostic forest and its leaf prior precision."""
    # `step` rebuilds the state with `replace`, so it preserves the subclass.
    # WORKAROUND(python<3.12): type `step` as generic over the state subclass
    # (PEP 695) instead of casting here, since a TypeVar renders badly in the
    # html documentation.
    return cast(BCFState, step(key, state))


def bcf_step_tau_0(key: Key[Array, ''], state: BCFState) -> BCFState:
    """Update the treatment effect intercept."""
    # `resid` is stored scaled: ``resid_unit * resid = data residual``, whereas
    # `tau_0`, `tau_X`, `b`, `error_cov_inv` and `tau_0_prior_cov_inv` are
    # on the data scale (matching `_bcf.predict`). Convert `resid` in and out of data
    # units so the scalar Gibbs updates here and in `bcf_step_b` are
    # unit-consistent for any `resid_unit` (no-op when it is 1).

    if state.tau_0_prior_cov_inv is None:
        return state

    else:
        # get coding basis, possibly adaptive so not just 0 and 1
        b_z = state.b[state.trt.astype(int)]

        # partial residual removing current tau_0 effect, on the data scale
        partial_resid = state.resid * state.resid_unit + state.tau_0 * b_z

        # determine full conditional of tau_0
        prec = (
            jnp.sum(jnp.square(b_z)) * state.error_cov_inv.value
            + state.tau_0_prior_cov_inv
        )
        mean = jnp.sum(b_z * partial_resid) * state.error_cov_inv.value / prec

        # sample tau_0 from full conditional
        tau_0_new = mean + random.normal(key) * lax.rsqrt(prec)

        # update state to reflect new tau_0
        return replace(
            state,
            tau_0=tau_0_new,
            resid=state.resid - b_z * (tau_0_new - state.tau_0) / state.resid_unit,
        )


def bcf_step_tau(key: Key[Array, ''], state: BCFState) -> BCFState:
    """Update the treatment effect forest and its leaf prior precision."""
    keys = split(key, 2)

    # get coding basis, possibly adaptive so not just 0 and 1
    b_z = state.b[state.trt.astype(int)]

    # Target for tau is (Y - mu - b_z * tau_0) / b_z.
    # Its residual is target - tau = (Y - mu - b_z*tau_0 - b_z*tau) / b_z
    b_z_zero = jnp.abs(b_z) < 1e-10
    b_z_safe = jnp.where(b_z_zero, 1.0, b_z)
    mu_resid = state.resid
    initial_resid_tau = jnp.where(b_z_zero, 0.0, mu_resid / b_z_safe)

    # Swap the tau forest into the forest slot and run only the tree and leaf
    # prior precision steps on it; the mu forest rides along in `forest_tau`
    # and is swapped back at the end. The other sub-steps of `step` (latent
    # outcome, error precision, sparsity, step counter) belong to the mu phase
    # alone.
    mu_prec_scale = state.prec_scale
    state = replace(
        state,
        forest=state.forest_tau,
        forest_tau=state.forest,
        resid=initial_resid_tau,
        prec_scale=jnp.square(b_z),
    )

    state = cast(BCFState, step_trees(keys.pop(), state))
    state = cast(BCFState, step_leaf_prior_cov_inv(keys.pop(), state))

    # Update tau_X! (the residual difference is scaled, bring it to data units)
    if state.tau_X is not None:
        state = replace(
            state,
            tau_X=state.tau_X + (initial_resid_tau - state.resid) * state.resid_unit,
        )

    # Swap the forests back and restore the mu-side fields
    return replace(
        state,
        forest=state.forest_tau,
        forest_tau=state.forest,
        resid=jnp.where(b_z_zero, mu_resid, state.resid * b_z_safe),
        prec_scale=mu_prec_scale,
    )


def bcf_step_b(key: Key[Array, ''], state: BCFState) -> BCFState:
    """Update the adaptive coding weights."""
    if state.b_prior_cov_inv is None:
        return state

    else:
        assert state.tau_X is not None

        # get coding basis
        b_z = state.b[state.trt.astype(int)]

        # partial residual removing current b effect, on the data scale (see
        # `bcf_step_tau_0` about units)
        tau_full = state.tau_0 + state.tau_X
        partial_resid = state.resid * state.resid_unit + tau_full * b_z

        # determine full conditional of b, one Gibbs update per treatment group
        groups = jnp.stack([~state.trt, state.trt])
        prec = (
            jnp.sum(jnp.square(tau_full) * groups, axis=1) * state.error_cov_inv.value
            + state.b_prior_cov_inv
        )
        mean = (
            jnp.sum(tau_full * partial_resid * groups, axis=1)
            * state.error_cov_inv.value
            / prec
        )

        # sample b from full conditional
        b_new = mean + random.normal(key, (2,)) * lax.rsqrt(prec)
        b_z_new = b_new[state.trt.astype(int)]

        # update state to reflect new b
        state = replace(
            state,
            b=b_new,
            resid=state.resid - tau_full * (b_z_new - b_z) / state.resid_unit,
        )

        # the tau precision scale b_z^2 changed on every datapoint, so the
        # incrementally maintained per-leaf cache of the tau forest is stale
        # everywhere
        return tree_at(
            lambda s: s.forest_tau.prec_tree,
            state,
            recompute_prec_trees(state.forest_tau, jnp.square(b_z_new), state.config),
        )


@jit(donate_argnums=(1,))
@split_key_for_chains
@vmap_chains
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
    keys = split(key, 4)
    state = bcf_step_mu(keys.pop(), state)
    state = bcf_step_tau_0(keys.pop(), state)
    state = bcf_step_tau(keys.pop(), state)
    return bcf_step_b(keys.pop(), state)


def _tau_view(state: BCFState) -> BCFState:
    """Return the state with the tau forest in the mu forest slot."""
    return replace(state, forest=state.forest_tau, forest_tau=state.forest)


class BCFBurninTrace(Trace):
    """Burn-in trace of the BCF MCMC, the per-forest diagnostics and the scalar parameters."""

    mu: BurninTrace
    """The trace of the prognostic forest."""

    tau: BurninTrace
    """The trace of the treatment forest."""

    tau_0: Float32[Array, '*chains_and_samples'] = field(chains=CHAIN_AXIS, samples=0)
    """The treatment effect intercept."""

    b: Float32[Array, '*chains_and_samples 2'] = field(chains=CHAIN_AXIS, samples=0)
    """The adaptive coding weights for untreated and treated units."""

    @classmethod
    def from_state(cls, state: State) -> 'BCFBurninTrace':
        """Create a single-item burn-in trace from a BCF state."""
        assert isinstance(state, BCFState)
        return cls(
            mu=BurninTrace.from_state(state),
            tau=BurninTrace.from_state(_tau_view(state)),
            tau_0=state.tau_0,
            b=state.b,
        )


class BCFMainTrace(BCFBurninTrace):
    """Main trace of the BCF MCMC, with the trees of both forests."""

    mu: MainTrace
    """The trace of the prognostic forest."""

    tau: MainTrace
    """The trace of the treatment forest."""

    @classmethod
    def from_state(cls, state: State) -> 'BCFMainTrace':
        """Create a single-item main trace from a BCF state."""
        assert isinstance(state, BCFState)
        kw: dict = dict(
            vars(BCFBurninTrace.from_state(state)),
            mu=MainTrace.from_state(state),
            tau=MainTrace.from_state(_tau_view(state)),
        )
        return cls(**kw)
