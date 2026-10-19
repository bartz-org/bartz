# bartz/src/bartz/debug/_simulate.py
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

"""Simulate the parameters and the data of the BART model of an MCMC state."""

from dataclasses import fields, replace

from jax import lax, random
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, Float, Float32, Key, Shaped

from bartz._jaxext import field, jit, split
from bartz.debug._prior import (
    MinPointsCriterion,
    sample_diag_wishart,
    sample_forest_structure,
    sample_leaves,
    sample_s,
    sample_theta,
    sample_wishart,
)
from bartz.grove import evaluate_forest
from bartz.mcmcstep import DiagWishart, State, Wishart
from bartz.mcmcstep._axes import CHAIN_AXIS, chain_to_axis, chain_vmap_axes
from bartz.mcmcstep._rebuild import outcome_target, rebuild_derived_chain
from bartz.mcmcstep._state import (
    StateT,
    chol_with_gersh,
    split_key_for_chains,
    vmap_chains,
)


class ChainDataState(State):
    """A `State` with separate data for each chain.

    The MCMC runs on it like on a `State`, with each chain fitting its own
    outcome `y`. Use `from_state` to create one.
    """

    y: Float32[Array, '*chains n'] | Float32[Array, '*chains k n'] = field(
        chains=CHAIN_AXIS, data=-1
    )
    """The outcome of each chain, see `State.y`."""

    @classmethod
    def from_state(cls, state: State) -> 'ChainDataState':
        """Copy the data of `state` to each of its chains.

        Parameters
        ----------
        state
            A BART MCMC state, as created by `bartz.mcmcstep.init`.

        Returns
        -------
        The same state, with a copy of the outcome for each chain.

        Notes
        -----
        The arrays other than `y` are shared with `state`, so functions that
        re-use the memory of their input, like `bartz.mcmcstep.step`, invalidate
        `state` when applied to the output.
        """
        y = state.y
        if state.has_chains:
            # `resid` has the same layout `y` gets
            chain_axis = chain_vmap_axes(state).resid
            y = jnp.broadcast_to(jnp.expand_dims(y, chain_axis), state.resid.shape)
        kw: dict = dict({f.name: getattr(state, f.name) for f in fields(state)}, y=y)
        return cls(**kw)


@jit(donate_argnums=(1,))
def sample_prior(key: Key[Array, ''], state: StateT) -> StateT:
    """Draw the parameters of the model from their prior.

    The prior is the one defined by the hyperparameters in `state`. Replaces
    the trees and the leaves, and, where they have a prior, the leaf and error
    precisions and the variable selection parameters. The data is left
    unchanged, and the fields derived from the parameters are updated to match.

    Parameters
    ----------
    key
        A jax random key.
    state
        A BART MCMC state, as created by `bartz.mcmcstep.init`.

    Returns
    -------
    The state with new parameters.

    Raises
    ------
    NotImplementedError
        If the minimum number of points per node is set together with a prior
        on the variable selection probabilities, or if the data is sharded.

    Notes
    -----
    The minimum number of points per node restricts the prior support to the
    trees that satisfy the constraints on the data `State.X`. The trees are
    sampled exactly by rejection. This does not work with a random
    `Sparsity.log_s`, because then the constraints also change its marginal
    distribution.

    The memory of the input state is re-used for the output state, so the input
    state can not be used any more, like with `bartz.mcmcstep.step`.
    """
    if state.config.data_sharded:
        msg = 'sampling with the data sharded is not supported'
        raise NotImplementedError(msg)

    forest = state.forest
    constrained = (
        forest.min_points_per_decision_node is not None
        or forest.min_points_per_leaf is not None
    )
    if constrained and forest.sparsity.theta is not None:
        msg = 'variable selection with a minimum number of points per node is not supported'
        raise NotImplementedError(msg)

    keys = split(key, 4)
    state = sample_theta_chains(keys.pop(), state)
    state = split_key_for_chains(vmap_chains(sample_sparsity))(keys.pop(), state)
    state = sample_trees(keys.pop(), state)
    return split_key_for_chains(vmap_chains(sample_values))(keys.pop(), state)


def sample_theta_chains(key: Key[Array, ''], state: StateT) -> StateT:
    """Draw `Sparsity.theta`, if it has a prior, of all chains at once."""
    forest = state.forest
    if forest.sparsity.rho is None:
        return state
    else:
        assert forest.sparsity.a is not None
        assert forest.sparsity.b is not None
        num_chains = state.num_chains()
        theta = sample_theta(
            key,
            forest.sparsity.rho,
            forest.sparsity.a,
            forest.sparsity.b,
            () if num_chains is None else (num_chains,),
            log_range=forest.sparsity.log_theta_range,
        )
        return replace(
            state,
            forest=replace(forest, sparsity=replace(forest.sparsity, theta=theta)),
        )


def sample_sparsity(key: Key[Array, ''], state: StateT) -> StateT:
    """Draw `Sparsity.log_s`, if it has a prior."""
    forest = state.forest
    if forest.sparsity.theta is None:
        return state
    else:
        (p,) = forest.max_split.shape
        log_s = sample_s(key, forest.sparsity.theta, p)
        return replace(
            state,
            forest=replace(forest, sparsity=replace(forest.sparsity, log_s=log_s)),
        )


def sample_trees(key: Key[Array, ''], state: StateT) -> StateT:
    """Draw the decision rules of the trees, of all chains at once."""
    forest = state.forest

    # put the chain axis first
    axes = chain_vmap_axes(state).forest
    log_s = forest.sparsity.log_s
    if log_s is not None:
        log_s = chain_to_axis(log_s, axes.sparsity.log_s)
    *shape, half_tree_size = chain_to_axis(forest.split_tree, axes.split_tree).shape

    depths = half_tree_size.bit_length() - 1
    p_nonterminal = forest.p_nonterminal[2 ** jnp.arange(depths)]

    if (
        forest.min_points_per_decision_node is None
        and forest.min_points_per_leaf is None
    ):
        criterion = None
    else:
        criterion = MinPointsCriterion(
            X=state.X,
            min_points_per_decision_node=forest.min_points_per_decision_node,
            min_points_per_leaf=forest.min_points_per_leaf,
        )

    trees = sample_forest_structure(
        key,
        tuple(shape),
        max_split=forest.max_split,
        p_nonterminal=p_nonterminal,
        log_s=log_s,
        criterion=criterion,
    )

    forest = replace(
        forest,
        var_tree=chain_from_first(trees.var_tree, axes.var_tree),
        split_tree=chain_from_first(trees.split_tree, axes.split_tree),
    )
    return replace(state, forest=forest)


def chain_from_first(
    arr: Shaped[Array, '...'], chain_axis: int | None
) -> Shaped[Array, '...']:
    """Move the leading chain axis of `arr` to `chain_axis`, the inverse of `chain_to_axis`."""
    return arr if chain_axis is None else jnp.moveaxis(arr, 0, chain_axis)


def sample_values(key: Key[Array, ''], state: StateT) -> StateT:
    """Draw the leaves and the precisions, then rebuild the derived fields."""
    keys = split(key, 3)
    forest = state.forest

    # precisions
    leaf_prior_cov_inv, leaf_prec_chol = sample_precision(
        keys.pop(), forest.leaf_prior_cov_inv
    )
    error_cov_inv, _ = sample_precision(keys.pop(), state.error_cov_inv)
    if state.binary_indices is not None:
        # the binary components have unit error variance, they have no prior
        bi = state.binary_indices
        error_cov_inv = replace(
            error_cov_inv, value=error_cov_inv.value.at[bi, bi].set(1.0)
        )

    # leaves
    leaves = sample_leaves(keys.pop(), forest.leaf_tree.shape, leaf_prec_chol)

    forest = replace(
        forest,
        leaf_tree=to_leaf_storage(leaves, state),
        leaf_prior_cov_inv=leaf_prior_cov_inv,
    )
    state = replace(state, forest=forest, error_cov_inv=error_cov_inv)
    return rebuild_derived_chain(state)


def sample_precision(
    key: Key[Array, ''], prior: Wishart
) -> tuple[Wishart, Float32[Array, ''] | Float32[Array, 'k k']]:
    """Draw the precision from its prior, if it has one, and get its Cholesky factor."""
    if prior.nu is None:
        value = prior.value
        chol = jnp.sqrt(value) if value.ndim == 0 else chol_with_gersh(value)
        return prior, chol
    else:
        assert prior.rate is not None
        if isinstance(prior, DiagWishart):
            nu = prior.inv_wishart_marginal_nu
            assert nu is not None
            value, chol = sample_diag_wishart(key, nu, prior.rate)
        else:
            value, chol = sample_wishart(key, prior.nu, prior.rate)
        return replace(prior, value=value), chol


def to_leaf_storage(
    leaves: Float32[Array, 'num_trees *k tree_size'], state: State
) -> Float[Array, 'num_trees *k tree_size']:
    """Convert the leaves to the storage units and dtype of `Forest.leaf_tree`."""
    forest = state.forest
    leaves /= forest.leaf_unit[..., None]

    q = state.config.leaf_quantization
    if q is not None:
        # same grid as in `bartz.mcmcstep.step`
        nmant = jnp.finfo(state.resid.dtype).nmant
        quantum = (state.resid_eff_scale / forest.leaf_unit) * 2.0 ** (q - nmant)
        leaves = jnp.round(leaves / quantum[..., None]) * quantum[..., None]

    finfo = jnp.finfo(forest.leaf_tree.dtype)
    leaves = lax.reduce_precision(leaves, finfo.nexp, finfo.nmant)
    return leaves.astype(forest.leaf_tree.dtype)


@jit(donate_argnums=(1,))
def sample_data(key: Key[Array, ''], state: StateT) -> StateT:
    """Draw the data from the model given the parameters.

    Replaces the outcome `State.y`, and the latent outcome `State.z` in binary
    regression. The predictors and the missing datapoints are left unchanged.
    `State.resid` is updated incrementally rather than recomputed, so any
    inconsistency it had with the other fields persists.

    Parameters
    ----------
    key
        A jax random key.
    state
        A BART MCMC state. If it has multiple chains, it must be a
        `ChainDataState`, to draw separate data for each chain.

    Returns
    -------
    The state with new data.

    Raises
    ------
    ValueError
        If `state` has multiple chains which share the data.
    NotImplementedError
        If the data is sharded.

    Notes
    -----
    The memory of the input state is re-used for the output state, so the input
    state can not be used any more, like with `bartz.mcmcstep.step`.
    """
    if state.has_chains and not isinstance(state, ChainDataState):
        msg = 'the chains share the data, use `ChainDataState.from_state`'
        raise ValueError(msg)
    if state.config.data_sharded:
        msg = 'sampling with the data sharded is not supported'
        raise NotImplementedError(msg)
    return split_key_for_chains(vmap_chains(sample_data_chain))(key, state)


def sample_data_chain(key: Key[Array, ''], state: StateT) -> StateT:
    """Implement `sample_data` on a state without a chain axis."""
    # evaluate the forest
    eta = (
        evaluate_forest(state.X, state.forest, sum_batch_axis=-1)
        + state.forest.offset[..., None]
    )

    # draw the error, scaled per datapoint
    error = random.normal(key, eta.shape)
    prec = state.error_cov_inv.value
    if prec.ndim == 0:
        error *= lax.rsqrt(prec)
    else:
        chol = chol_with_gersh(prec)
        error = solve_triangular(chol, error, lower=True, trans='T')
    if state.error_scale is not None:
        error *= state.error_scale
    latent = eta + error

    # split the latent outcome into the observed and latent parts
    if state.binary_indices is not None:
        z = latent[state.binary_indices, :]
        y = latent.at[state.binary_indices, :].set(z > 0)
    elif state.z is not None:
        z = latent
        y = (z > 0).astype(latent.dtype)
    else:
        z = None
        y = latent

    # keep the missing datapoints as they are
    if state.inv_sdev_scale is not None:
        missing = state.inv_sdev_scale == 0
        y = jnp.where(missing, state.y, y)
        if z is not None:
            assert state.z is not None
            if state.binary_indices is not None and missing.ndim > 1:
                missing = missing[state.binary_indices, :]
            z = jnp.where(missing, state.z, z)

    new_state = replace(state, y=y, z=z)
    delta = outcome_target(new_state) - outcome_target(state)
    resid = state.resid + (delta / state.resid_unit[..., None]).astype(
        state.resid.dtype
    )
    return replace(new_state, resid=resid)
