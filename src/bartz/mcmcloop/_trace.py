# bartz/src/bartz/mcmcloop/_trace.py
#
# Copyright (c) 2024-2026, The Bartz Contributors
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

"""Trace dataclasses returned by `run_mcmc`."""

from equinox import Module
from jax import numpy as jnp
from jax.nn import softmax
from jax.sharding import Mesh
from jaxtyping import Array, Float32, Int32, UInt

from bartz.grove import HeapArrays
from bartz.mcmcstep import State
from bartz.mcmcstep._axes import CHAIN_AXIS, chain_vmap_axes, chainful_axis, field


class BurninTrace(Module):
    """MCMC trace with only diagnostic values."""

    has_chains: bool = field(static=True)
    """Whether the trace carries an explicit chain axis."""

    mesh: Mesh | None = field(static=True)
    """The device mesh the trace arrays are sharded on, or `None`."""

    # The union-free count diagnostics are declared before `error_cov_inv` (and
    # before `MainTrace.leaf_tree`) so they bind the variadic
    # `*chains_and_samples` axis first. Otherwise the runtime typechecker,
    # evaluating the `... | ... k k` / `... | ... k tree_size` unions in a
    # hash-randomized order, can mis-bind `*chains_and_samples` against the `k`
    # axis for a multivariate-without-chains trace (the layouts are
    # rank-ambiguous).
    grow_prop_count: Int32[Array, '*chains_and_samples'] = field(
        chains=CHAIN_AXIS, samples=0
    )
    """The number of grow proposals made during one full MCMC cycle."""

    grow_acc_count: Int32[Array, '*chains_and_samples'] = field(
        chains=CHAIN_AXIS, samples=0
    )
    """The number of grow moves accepted during one full MCMC cycle."""

    prune_prop_count: Int32[Array, '*chains_and_samples'] = field(
        chains=CHAIN_AXIS, samples=0
    )
    """The number of prune proposals made during one full MCMC cycle."""

    prune_acc_count: Int32[Array, '*chains_and_samples'] = field(
        chains=CHAIN_AXIS, samples=0
    )
    """The number of prune moves accepted during one full MCMC cycle."""

    error_cov_inv: (
        Float32[Array, '*chains_and_samples']
        | Float32[Array, '*chains_and_samples k k']
    ) = field(chains=CHAIN_AXIS, samples=0)
    """The inverse error covariance (scalar for univariate, matrix for
    multivariate). Identity in binary regression."""

    theta: Float32[Array, '*chains_and_samples'] | None = field(
        chains=CHAIN_AXIS, samples=0
    )
    """The concentration parameter of the Dirichlet prior on the variable
    split probabilities, or `None` if it was not sampled."""

    log_likelihood: Float32[Array, '*chains_and_samples num_trees'] | None = field(
        chains=CHAIN_AXIS, samples=0
    )
    """The log likelihood ratio of the proposed move on each tree, or `None`."""

    log_trans_prior: Float32[Array, '*chains_and_samples num_trees'] | None = field(
        chains=CHAIN_AXIS, samples=0
    )
    """The log transition and prior Metropolis-Hastings ratio of the proposed
    move on each tree, or `None`."""

    @classmethod
    def from_state(cls, state: State) -> 'BurninTrace':
        """Create a single-item burn-in trace from a MCMC state."""
        return cls(
            has_chains=state.has_chains,
            mesh=state.config.mesh,
            error_cov_inv=state.error_cov_inv,
            theta=state.forest.theta,
            grow_prop_count=state.forest.grow_prop_count,
            grow_acc_count=state.forest.grow_acc_count,
            prune_prop_count=state.forest.prune_prop_count,
            prune_acc_count=state.forest.prune_acc_count,
            log_likelihood=state.forest.log_likelihood,
            log_trans_prior=state.forest.log_trans_prior,
        )


class MainTrace(BurninTrace, HeapArrays):
    """MCMC trace with trees and diagnostic values."""

    leaf_tree: (
        Float32[Array, '*chains_and_samples num_trees tree_size']
        | Float32[Array, '*chains_and_samples num_trees k tree_size']
    ) = field(chains=CHAIN_AXIS, samples=0)
    """The leaf values."""

    var_tree: UInt[Array, '*chains_and_samples num_trees tree_size//2'] = field(
        chains=CHAIN_AXIS, samples=0
    )
    """The decision axes."""

    split_tree: UInt[Array, '*chains_and_samples num_trees tree_size//2'] = field(
        chains=CHAIN_AXIS, samples=0
    )
    """The decision boundaries."""

    offset: Float32[Array, ''] | Float32[Array, ' k']
    """Constant shift added to the sum of trees."""

    varprob: Float32[Array, '*chains_and_samples p'] | None = field(
        chains=CHAIN_AXIS, samples=0
    )
    """The probability of choosing each variable for a decision rule,
    normalized over variables, or `None` when variable selection is off."""

    @classmethod
    def from_state(cls, state: State) -> 'MainTrace':
        """Create a single-item main trace from a MCMC state."""
        # compute varprob
        log_s = state.forest.log_s
        if log_s is None:
            varprob = None
        else:
            chain_axis = chain_vmap_axes(state.forest).log_s
            p_axis = chainful_axis(0, chain_axis)  # (p,)
            where = state.forest.max_split.astype(bool)
            if chain_axis is not None:
                where = jnp.expand_dims(where, chain_axis)
            varprob = softmax(log_s, axis=p_axis, where=where)

        return cls(
            leaf_tree=state.forest.leaf_tree,
            var_tree=state.forest.var_tree,
            split_tree=state.forest.split_tree,
            offset=state.offset,
            varprob=varprob,
            **vars(BurninTrace.from_state(state)),
        )
