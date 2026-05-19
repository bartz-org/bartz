# bartz/src/bartz/mcmcstep/_step.py
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

"""Implement `step`, `step_trees`, and the accept-reject logic."""

from dataclasses import replace
from functools import partial

import jax
from equinox import Module, tree_at
from jax import jit, lax, named_call, random, vmap
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln, logsumexp
from jaxtyping import Array, Bool, Float32, Int32, Integer, Key, Shaped, UInt, UInt32

from bartz._jaxext import split, truncated_normal_onesided, vmap_nodoc
from bartz.grove import var_histogram
from bartz.mcmcstep._moves import Moves, propose_moves
from bartz.mcmcstep._state import (
    State,
    StepConfig,
    chol_with_gersh,
    field,
    get_axis_size,
    shard_map_state,
    split_key_for_chains,
    vmap_chains,
)


@partial(jit, donate_argnums=(1,))
@split_key_for_chains
@shard_map_state
@vmap_chains
def step(key: Key[Array, ''], bart: State) -> State:
    """
    Do one MCMC step.

    Parameters
    ----------
    key
        A jax random key.
    bart
        A BART mcmc state, as created by `init`.

    Returns
    -------
    The new BART mcmc state.

    Notes
    -----
    The memory of the input state is re-used for the output state, so the input
    state can not be used any more after calling `step`. All this applies
    outside of `jax.jit`.
    """
    keys = split(key, 4)

    bart = step_trees(keys.pop(), bart)

    if bart.z is not None:
        bart = step_z(keys.pop(), bart)

    if bart.error_cov_df is not None:
        bart = step_error_cov_inv(keys.pop(), bart)

    bart = step_sparse(keys.pop(), bart)
    return step_config(bart)


@named_call
def step_trees(key: Key[Array, ''], bart: State) -> State:
    """
    Forest sampling step of BART MCMC.

    Parameters
    ----------
    key
        A jax random key.
    bart
        A BART mcmc state, as created by `init`.

    Returns
    -------
    The new BART mcmc state.

    Notes
    -----
    This function zeroes the proposal counters.
    """
    keys = split(key)
    moves = propose_moves(keys.pop(), bart.forest)
    return accept_moves_and_sample_leaves(keys.pop(), bart, moves)


@named_call
def accept_moves_and_sample_leaves(
    key: Key[Array, ''], bart: State, moves: Moves
) -> State:
    """
    Accept or reject the proposed moves and sample the new leaf values.

    Parameters
    ----------
    key
        A jax random key.
    bart
        A valid BART mcmc state.
    moves
        The proposed moves, see `propose_moves`.

    Returns
    -------
    A new (valid) BART mcmc state.
    """
    pso = accept_moves_parallel_stage(key, bart, moves)
    bart, moves = accept_moves_sequential_stage(pso)
    return accept_moves_final_stage(bart, moves)


class Counts(Module):
    """Number of datapoints in the nodes involved in proposed moves for each tree."""

    left: UInt[Array, '*chains num_trees'] = field(chains=0)
    """Number of datapoints in the left child."""

    right: UInt[Array, '*chains num_trees'] = field(chains=0)
    """Number of datapoints in the right child."""

    total: UInt[Array, '*chains num_trees'] = field(chains=0)
    """Number of datapoints in the parent (``= left + right``)."""


class Precs(Module):
    """Likelihood precision scale in the nodes involved in proposed moves for each tree.

    The "likelihood precision scale" of a tree node is the sum of the inverse
    squared error scales of the datapoints selected by the node.
    """

    left: (
        Float32[Array, '*chains num_trees'] | Float32[Array, '*chains num_trees k k']
    ) = field(chains=0)
    """Likelihood precision scale in the left child."""

    right: (
        Float32[Array, '*chains num_trees'] | Float32[Array, '*chains num_trees k k']
    ) = field(chains=0)
    """Likelihood precision scale in the right child."""

    total: (
        Float32[Array, '*chains num_trees'] | Float32[Array, '*chains num_trees k k']
    ) = field(chains=0)
    """Likelihood precision scale in the parent (``= left + right``)."""


class PreLkV(Module):
    """Non-sequential terms of the likelihood ratio for each tree.

    These terms can be computed in parallel across trees. Each one of the
    left/right/total terms is, in the univariate case, the scalar

        ``1 / error_cov_inv + n_left/right/total / leaf_prior_cov_inv``.

    In the multivariate homoskedastic or scalar weight case, this is the matrix term

        ``error_cov_inv @ inv(leaf_prior_cov_inv + n_left/right/total * error_cov_inv) @ error_cov_inv``.

    In the multivariate vector-weight case, this is instead

        ``chol(leaf_prior_cov_inv + n_left/right/total * error_cov_inv)``

    ``n_left`` is the number of datapoints in the left child, or the
    likelihood precision scale in the heteroskedastic case. Similarly for
    right, total.
    """

    left: (
        Float32[Array, '*chains num_trees'] | Float32[Array, '*chains num_trees k k']
    ) = field(chains=0)
    """Full conditional variance, scaled covariance, or precision cholesky, for
    the left leaf."""

    right: (
        Float32[Array, '*chains num_trees'] | Float32[Array, '*chains num_trees k k']
    ) = field(chains=0)
    """Full conditional variance, scaled covariance, or precision cholesky, for
    the right leaf."""

    total: (
        Float32[Array, '*chains num_trees'] | Float32[Array, '*chains num_trees k k']
    ) = field(chains=0)
    """Full conditional variance, scaled covariance, or precision cholesky, for
    the the join of the left and right leaves."""

    log_sqrt_term: Float32[Array, '*chains num_trees'] = field(chains=0)
    """The logarithm of the square root term of the likelihood ratio."""


class PreLk(Module):
    """Non-sequential terms of the likelihood ratio shared by all trees."""

    exp_factor: Float32[Array, '*chains'] | None = field(chains=0)
    """The factor to multiply the likelihood ratio by, shared by all trees.
    Set only in the univariate path."""

    error_cov_inv: Float32[Array, '*chains k k'] | None = field(chains=0)
    """The global error precision scale. Set only in the multivariate
    heteroskedastic vector-weight case."""


class PreLf(Module):
    """Pre-computed terms used to sample leaves from their posterior.

    These terms can be computed in parallel across trees.

    For each tree and leaf, the terms are scalars in the univariate case, and
    matrices/vectors in the multivariate case.
    """

    mean_factor: (
        Float32[Array, '*chains num_trees 2**d']
        | Float32[Array, '*chains num_trees k k 2**d']
    ) = field(chains=0)
    """The factor to be right-multiplied by the sum of the scaled residuals to
    obtain the posterior mean."""

    centered_leaves: (
        Float32[Array, '*chains num_trees 2**d']
        | Float32[Array, '*chains num_trees k 2**d']
    ) = field(chains=0)
    """The mean-zero normal values to be added to the posterior mean to
    obtain the posterior leaf samples."""


class ParallelStageOut(Module):
    """The output of `accept_moves_parallel_stage`."""

    bart: State
    """A partially updated BART mcmc state."""

    moves: Moves
    """The proposed moves, with `partial_ratio` set to `None` and
    `log_trans_prior_ratio` set to its final value."""

    prec_trees: (
        Float32[Array, '*chains num_trees 2**d']
        | Int32[Array, '*chains num_trees 2**d']
        | Float32[Array, '*chains num_trees k k 2**d']
    ) = field(chains=0)
    """The likelihood precision scale in each potential or actual leaf node."""

    move_precs: Precs | Counts
    """The likelihood precision scale in each node modified by the moves. If
    `bart.prec_scale` is not set, this is set to `move_counts`."""

    prelkv: PreLkV
    """Object with pre-computed terms of the likelihood ratios."""

    prelk: PreLk | None
    """Object with pre-computed terms of the likelihood ratios."""

    prelf: PreLf
    """Object with pre-computed terms of the leaf samples."""


@named_call
def accept_moves_parallel_stage(
    key: Key[Array, ''], bart: State, moves: Moves
) -> ParallelStageOut:
    """
    Pre-compute quantities used to accept moves, in parallel across trees.

    Parameters
    ----------
    key
        A jax random key.
    bart
        A BART mcmc state.
    moves
        The proposed moves, see `propose_moves`.

    Returns
    -------
    An object with all that could be done in parallel.
    """
    # where the move is grow, modify the state like the move was accepted
    bart = replace(
        bart,
        forest=replace(
            bart.forest,
            var_tree=moves.var_tree,
            leaf_indices=apply_grow_to_indices(moves, bart.forest.leaf_indices, bart.X),
            leaf_tree=adapt_leaf_trees_to_grow_indices(bart.forest.leaf_tree, moves),
        ),
    )

    # count number of datapoints per leaf
    if (
        bart.forest.min_points_per_decision_node is not None
        or bart.forest.min_points_per_leaf is not None
        or bart.prec_scale is None
    ):
        count_trees, move_counts = compute_count_trees(
            bart.forest.leaf_indices, moves, bart.config
        )

    # mark which leaves & potential leaves have enough points to be grown
    if bart.forest.min_points_per_decision_node is not None:
        count_half_trees = count_trees[:, : bart.forest.var_tree.shape[1]]
        moves = replace(
            moves,
            affluence_tree=moves.affluence_tree
            & (count_half_trees >= bart.forest.min_points_per_decision_node),
        )

    # copy updated affluence_tree to state
    bart = tree_at(lambda bart: bart.forest.affluence_tree, bart, moves.affluence_tree)

    # veto grove move if new leaves don't have enough datapoints
    if bart.forest.min_points_per_leaf is not None:
        moves = replace(
            moves,
            allowed=moves.allowed
            & (move_counts.left >= bart.forest.min_points_per_leaf)
            & (move_counts.right >= bart.forest.min_points_per_leaf),
        )

    # count number of datapoints per leaf, weighted by error precision scale
    if bart.prec_scale is None:
        prec_trees = count_trees
        move_precs = move_counts
    else:
        prec_trees, move_precs = compute_prec_trees(
            bart.prec_scale, bart.forest.leaf_indices, moves, bart.config
        )
    assert move_precs is not None

    # compute some missing information about moves
    moves = complete_ratio(moves, bart.forest.p_nonterminal)
    save_ratios = bart.forest.log_likelihood is not None
    bart = replace(
        bart,
        forest=replace(
            bart.forest,
            grow_prop_count=jnp.sum(moves.grow),
            prune_prop_count=jnp.sum(moves.allowed & ~moves.grow),
            log_trans_prior=moves.log_trans_prior_ratio if save_ratios else None,
        ),
    )

    prelkv, prelk = precompute_likelihood_terms(
        bart.error_cov_inv, bart.forest.leaf_prior_cov_inv, move_precs
    )
    prelf = precompute_leaf_terms(
        key, prec_trees, bart.error_cov_inv, bart.forest.leaf_prior_cov_inv
    )

    return ParallelStageOut(
        bart=bart,
        moves=moves,
        prec_trees=prec_trees,
        move_precs=move_precs,
        prelkv=prelkv,
        prelk=prelk,
        prelf=prelf,
    )


@named_call
@partial(vmap_nodoc, in_axes=(0, 0, None))
def apply_grow_to_indices(
    moves: Moves, leaf_indices: UInt[Array, 'num_trees n'], X: UInt[Array, 'p n']
) -> UInt[Array, 'num_trees n']:
    """
    Update the leaf indices to apply a grow move.

    Parameters
    ----------
    moves
        The proposed moves, see `propose_moves`.
    leaf_indices
        The index of the leaf each datapoint falls into.
    X
        The predictors matrix.

    Returns
    -------
    The updated leaf indices.
    """
    left_child = moves.node.astype(leaf_indices.dtype) << 1
    x: UInt[Array, ' n'] = X[moves.grow_var, :]
    go_right = x >= moves.grow_split
    tree_size = jnp.array(2 * moves.var_tree.size)
    node_to_update = jnp.where(moves.grow, moves.node, tree_size)
    return jnp.where(
        leaf_indices == node_to_update, left_child + go_right, leaf_indices
    )


def _compute_count_or_prec_trees(
    prec_scale: Float32[Array, ' n'] | Float32[Array, 'k k n'] | None,
    leaf_indices: UInt[Array, 'num_trees n'],
    moves: Moves,
    config: StepConfig,
) -> (
    tuple[UInt32[Array, 'num_trees 2**d'], Counts]
    | tuple[Float32[Array, 'num_trees 2**d'], Precs]
    | tuple[Float32[Array, 'num_trees k k 2**d'], Precs]
):
    """Implement `compute_count_trees` and `compute_prec_trees`."""
    if config.prec_count_num_trees is None:
        compute = vmap(_compute_count_or_prec_tree, in_axes=(None, 0, 0, None))
        return compute(prec_scale, leaf_indices, moves, config)

    def compute(
        args: tuple[UInt[Array, ' n'], Moves],
    ) -> (
        tuple[UInt32[Array, ' 2**d'], Counts]
        | tuple[Float32[Array, ' 2**d'], Precs]
        | tuple[Float32[Array, 'k k 2**d'], Precs]
    ):
        leaf_indices, moves = args
        return _compute_count_or_prec_tree(prec_scale, leaf_indices, moves, config)

    return lax.map(
        compute, (leaf_indices, moves), batch_size=config.prec_count_num_trees
    )


def _compute_count_or_prec_tree(
    prec_scale: Float32[Array, ' n'] | Float32[Array, 'k k n'] | None,
    leaf_indices: UInt[Array, ' n'],
    moves: Moves,
    config: StepConfig,
) -> (
    tuple[UInt32[Array, ' 2**d'], Counts]
    | tuple[Float32[Array, ' 2**d'], Precs]
    | tuple[Float32[Array, 'k k 2**d'], Precs]
):
    """Compute count or precision tree for a single tree."""
    (tree_size,) = moves.var_tree.shape
    tree_size *= 2

    if prec_scale is None:
        value = 1
        cls = Counts
        dtype = jnp.uint32
        num_batches = config.count_num_batches
    else:
        value = prec_scale
        cls = Precs
        dtype = jnp.float32
        num_batches = config.prec_num_batches

    trees = _scatter_add(
        value, leaf_indices, tree_size, dtype, num_batches, config.data_sharded
    )

    # count datapoints in nodes modified by move
    left = trees[..., moves.left]
    right = trees[..., moves.right]
    counts = cls(left=left, right=right, total=left + right)

    # write count into non-leaf node
    trees = trees.at[..., moves.node].set(counts.total)

    return trees, counts


@named_call
def compute_count_trees(
    leaf_indices: UInt[Array, 'num_trees n'], moves: Moves, config: StepConfig
) -> tuple[UInt32[Array, 'num_trees 2**d'], Counts]:
    """
    Count the number of datapoints in each leaf.

    Parameters
    ----------
    leaf_indices
        The index of the leaf each datapoint falls into, with the deeper version
        of the tree (post-GROW, pre-PRUNE).
    moves
        The proposed moves, see `propose_moves`.
    config
        The MCMC configuration.

    Returns
    -------
    count_trees : Int32[Array, 'num_trees 2**d']
        The number of points in each potential or actual leaf node.
    counts : Counts
        The counts of the number of points in the leaves grown or pruned by the
        moves.
    """
    return _compute_count_or_prec_trees(None, leaf_indices, moves, config)


@named_call
def compute_prec_trees(
    prec_scale: Float32[Array, ' n'] | Float32[Array, 'k k n'],
    leaf_indices: UInt[Array, 'num_trees n'],
    moves: Moves,
    config: StepConfig,
) -> tuple[
    Float32[Array, 'num_trees 2**d'] | Float32[Array, 'num_trees k k 2**d'], Precs
]:
    """
    Compute the likelihood precision scale in each leaf.

    Parameters
    ----------
    prec_scale
        The scale of the precision of the error on each datapoint.
    leaf_indices
        The index of the leaf each datapoint falls into, with the deeper version
        of the tree (post-GROW, pre-PRUNE).
    moves
        The proposed moves, see `propose_moves`.
    config
        The MCMC configuration.

    Returns
    -------
    prec_trees : Float32[Array, 'num_trees 2**d'] | Float32[Array, 'num_trees k k 2**d']
        The likelihood precision scale in each potential or actual leaf node.
    precs : Precs
        The likelihood precision scale in the nodes involved in the moves.
    """
    return _compute_count_or_prec_trees(prec_scale, leaf_indices, moves, config)


@partial(vmap_nodoc, in_axes=(0, None))
def complete_ratio(moves: Moves, p_nonterminal: Float32[Array, ' 2**d']) -> Moves:
    """
    Complete non-likelihood MH ratio calculation.

    This function adds the probability of choosing a prune move over the grow
    move in the inverse transition, and the a priori probability that the
    children nodes are leaves.

    Parameters
    ----------
    moves
        The proposed moves. Must have already been updated to keep into account
        the thresholds on the number of datapoints per node, this happens in
        `accept_moves_parallel_stage`.
    p_nonterminal
        The a priori probability of each node being nonterminal conditional on
        its ancestors, including at the maximum depth where it should be zero.

    Returns
    -------
    The updated moves, with `partial_ratio=None` and `log_trans_prior_ratio` set.
    """
    # can the leaves be grown?
    left_growable = moves.affluence_tree.at[moves.left].get(
        mode='fill', fill_value=False
    )
    right_growable = moves.affluence_tree.at[moves.right].get(
        mode='fill', fill_value=False
    )

    # p_prune if grow
    other_growable_leaves = moves.num_growable >= 2
    grow_again_allowed = other_growable_leaves | left_growable | right_growable
    grow_p_prune = jnp.where(grow_again_allowed, 0.5, 1.0)

    # p_prune if prune
    prune_p_prune = jnp.where(moves.num_growable, 0.5, 1)

    # select p_prune
    p_prune = jnp.where(moves.grow, grow_p_prune, prune_p_prune)

    # prior probability of both children being terminal
    pt_left = 1 - p_nonterminal[moves.left] * left_growable
    pt_right = 1 - p_nonterminal[moves.right] * right_growable
    pt_children = pt_left * pt_right

    assert moves.partial_ratio is not None
    return replace(
        moves,
        log_trans_prior_ratio=jnp.log(moves.partial_ratio * pt_children * p_prune),
        partial_ratio=None,
    )


@named_call
@vmap_nodoc
def adapt_leaf_trees_to_grow_indices(
    leaf_trees: Float32[Array, 'num_trees 2**d'], moves: Moves
) -> Float32[Array, 'num_trees 2**d']:
    """
    Modify leaves such that post-grow indices work on the original tree.

    The value of the leaf to grow is copied to what would be its children if the
    grow move was accepted.

    Parameters
    ----------
    leaf_trees
        The leaf values.
    moves
        The proposed moves, see `propose_moves`.

    Returns
    -------
    The modified leaf values.
    """
    values_at_node = leaf_trees[..., moves.node]
    return (
        leaf_trees.at[..., jnp.where(moves.grow, moves.left, leaf_trees.size)]
        .set(values_at_node)
        .at[..., jnp.where(moves.grow, moves.right, leaf_trees.size)]
        .set(values_at_node)
    )


def _logdet_from_chol(L: Float32[Array, '... k k']) -> Float32[Array, '...']:
    """Compute logdet of A = LL' via Cholesky (sum of log of diag^2)."""
    diags: Float32[Array, '... k'] = jnp.diagonal(L, axis1=-2, axis2=-1)
    return 2.0 * jnp.sum(jnp.log(diags), axis=-1)


def _precompute_likelihood_terms_uv(
    error_cov_inv: Float32[Array, ''],
    leaf_prior_cov_inv: Float32[Array, ''],
    move_precs: Precs | Counts,
) -> tuple[PreLkV, PreLk]:
    sigma2 = jnp.reciprocal(error_cov_inv)
    sigma_mu2 = jnp.reciprocal(leaf_prior_cov_inv)
    left = sigma2 + move_precs.left * sigma_mu2
    right = sigma2 + move_precs.right * sigma_mu2
    total = sigma2 + move_precs.total * sigma_mu2
    prelkv = PreLkV(
        left=left,
        right=right,
        total=total,
        log_sqrt_term=jnp.log(sigma2 * total / (left * right)) / 2,
    )
    return prelkv, PreLk(
        exp_factor=error_cov_inv / leaf_prior_cov_inv / 2, error_cov_inv=None
    )


def compute_B(
    error_cov_inv: Float32[Array, 'k k'], resid: Float32[Array, 'k k *tree_size']
) -> Float32[Array, ' k *tree_size']:
    """Compute the leaf score from the leaf weighted sum of residuals."""
    return jnp.einsum('ab,ab...->a...', error_cov_inv, resid)


def _precompute_likelihood_terms_mv_het(
    error_cov_inv: Float32[Array, 'k k'],
    leaf_prior_cov_inv: Float32[Array, 'k k'],
    move_precs: Precs,
) -> tuple[PreLkV, PreLk]:
    L_left: Float32[Array, 'num_trees k k'] = chol_with_gersh(
        error_cov_inv * move_precs.left + leaf_prior_cov_inv
    )
    L_right: Float32[Array, 'num_trees k k'] = chol_with_gersh(
        error_cov_inv * move_precs.right + leaf_prior_cov_inv
    )
    L_total: Float32[Array, 'num_trees k k'] = chol_with_gersh(
        error_cov_inv * move_precs.total + leaf_prior_cov_inv
    )

    log_sqrt_term: Float32[Array, ' num_trees'] = 0.5 * (
        _logdet_from_chol(chol_with_gersh(leaf_prior_cov_inv))
        + _logdet_from_chol(L_total)
        - _logdet_from_chol(L_left)
        - _logdet_from_chol(L_right)
    )

    return PreLkV(
        left=L_left, right=L_right, total=L_total, log_sqrt_term=log_sqrt_term
    ), PreLk(exp_factor=None, error_cov_inv=error_cov_inv)


def _precompute_likelihood_terms_mv(
    error_cov_inv: Float32[Array, 'k k'],
    leaf_prior_cov_inv: Float32[Array, 'k k'],
    move_precs: Precs | Counts,
) -> tuple[PreLkV, None]:
    nL: Shaped[Array, 'num_trees 1 1'] = move_precs.left[..., None, None]
    nR: Shaped[Array, 'num_trees 1 1'] = move_precs.right[..., None, None]
    nT: Shaped[Array, 'num_trees 1 1'] = move_precs.total[..., None, None]

    L_left: Float32[Array, 'num_trees k k'] = chol_with_gersh(
        error_cov_inv * nL + leaf_prior_cov_inv
    )
    L_right: Float32[Array, 'num_trees k k'] = chol_with_gersh(
        error_cov_inv * nR + leaf_prior_cov_inv
    )
    L_total: Float32[Array, 'num_trees k k'] = chol_with_gersh(
        error_cov_inv * nT + leaf_prior_cov_inv
    )

    log_sqrt_term: Float32[Array, ' num_trees'] = 0.5 * (
        _logdet_from_chol(chol_with_gersh(leaf_prior_cov_inv))
        + _logdet_from_chol(L_total)
        - _logdet_from_chol(L_left)
        - _logdet_from_chol(L_right)
    )

    def _term_from_chol(
        L: Float32[Array, 'num_trees k k'],
    ) -> Float32[Array, 'num_trees k k']:
        rhs: Float32[Array, 'num_trees k k'] = jnp.broadcast_to(error_cov_inv, L.shape)
        Y: Float32[Array, 'num_trees k k'] = solve_triangular(L, rhs, lower=True)
        return Y.mT @ Y

    prelkv = PreLkV(
        left=_term_from_chol(L_left),
        right=_term_from_chol(L_right),
        total=_term_from_chol(L_total),
        log_sqrt_term=log_sqrt_term,
    )

    return prelkv, None


@named_call
def precompute_likelihood_terms(
    error_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'],
    leaf_prior_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'],
    move_precs: Precs | Counts,
) -> tuple[PreLkV, PreLk | None]:
    """
    Pre-compute terms used in the likelihood ratio of the acceptance step.

    Handles both univariate and multivariate cases based on the shape of the
    input arrays. In the multivariate case, per-datapoint error precision
    scales (if any) are isotropic across components, so they enter the
    likelihood only through the per-leaf scalar precision sum carried by
    `move_precs` (a `Precs` instead of a `Counts`).

    Parameters
    ----------
    error_cov_inv
        The inverse error variance (univariate) or the inverse of the error
        covariance matrix (multivariate). This is the inverse global error
        variance factor if `prec_scale` is set.
    leaf_prior_cov_inv
        The inverse prior variance of each leaf (univariate) or the inverse of
        prior covariance matrix of each leaf (multivariate).
    move_precs
        The likelihood precision scale in the leaves grown or pruned by the
        moves, under keys 'left', 'right', and 'total' (left + right).

    Returns
    -------
    prelkv : PreLkV
        Pre-computed terms of the likelihood ratio, one per tree.
    prelk : PreLk | None
        Pre-computed terms of the likelihood ratio, shared by all trees.
    """
    if error_cov_inv.ndim == 0:
        return _precompute_likelihood_terms_uv(
            error_cov_inv, leaf_prior_cov_inv, move_precs
        )
    elif move_precs.left.ndim == 3:
        assert isinstance(move_precs, Precs)
        return _precompute_likelihood_terms_mv_het(
            error_cov_inv, leaf_prior_cov_inv, move_precs
        )
    else:
        return _precompute_likelihood_terms_mv(
            error_cov_inv, leaf_prior_cov_inv, move_precs
        )


def _precompute_leaf_terms_uv(
    key: Key[Array, ''],
    prec_trees: Float32[Array, 'num_trees 2**d'],
    error_cov_inv: Float32[Array, ''],
    leaf_prior_cov_inv: Float32[Array, ''],
    z: Float32[Array, 'num_trees 2**d'] | None = None,
) -> PreLf:
    prec_lk = prec_trees * error_cov_inv
    var_post = jnp.reciprocal(prec_lk + leaf_prior_cov_inv)
    if z is None:
        z = random.normal(key, prec_trees.shape, error_cov_inv.dtype)
    return PreLf(
        mean_factor=var_post * error_cov_inv,
        # | mean = mean_lk * prec_lk * var_post
        # | resid_tree = mean_lk * prec_tree  -->
        # |    -->  mean_lk = resid_tree / prec_tree  (kind of)
        # | mean_factor =
        # |    = mean / resid_tree =
        # |    = resid_tree / prec_tree * prec_lk * var_post / resid_tree =
        # |    = 1 / prec_tree * prec_tree / sigma2 * var_post =
        # |    = var_post / sigma2
        centered_leaves=z * jnp.sqrt(var_post),
    )


def _precompute_leaf_terms_mv(
    key: Key[Array, ''],
    prec_trees: Float32[Array, 'num_trees 2**d'],
    error_cov_inv: Float32[Array, 'k k'],
    leaf_prior_cov_inv: Float32[Array, 'k k'],
    z: Float32[Array, 'num_trees 2**d k'] | None = None,
) -> PreLf:
    num_trees, tree_size = prec_trees.shape
    k, _ = error_cov_inv.shape
    n_k: Float32[Array, 'num_trees tree_size 1 1'] = prec_trees[..., None, None]

    # Only broadcast the inverse of error covariance matrix to satisfy JAX's
    # batching rules for `lax.linalg.solve_triangular`, which does not support
    # implicit broadcasting.
    error_cov_inv_batched = jnp.broadcast_to(
        error_cov_inv, (num_trees, tree_size, k, k)
    )

    posterior_precision: Float32[Array, 'num_trees tree_size k k'] = (
        leaf_prior_cov_inv + n_k * error_cov_inv_batched
    )

    L_prec: Float32[Array, 'num_trees tree_size k k'] = chol_with_gersh(
        posterior_precision
    )
    Y: Float32[Array, 'num_trees tree_size k k'] = solve_triangular(
        L_prec, error_cov_inv_batched, lower=True
    )
    mean_factor: Float32[Array, 'num_trees tree_size k k'] = solve_triangular(
        L_prec, Y, trans='T', lower=True
    )
    mean_factor = mean_factor.mT
    mean_factor_out: Float32[Array, 'num_trees k k tree_size'] = jnp.moveaxis(
        mean_factor, 1, -1
    )

    if z is None:
        z = random.normal(key, (num_trees, tree_size, k))
    centered_leaves: Float32[Array, 'num_trees tree_size k'] = solve_triangular(
        L_prec, z[..., None], trans='T'
    ).squeeze(-1)
    centered_leaves_out: Float32[Array, 'num_trees k tree_size'] = jnp.swapaxes(
        centered_leaves, -1, -2
    )

    return PreLf(mean_factor=mean_factor_out, centered_leaves=centered_leaves_out)


def _precompute_leaf_terms_mv_het(
    key: Key[Array, ''],
    prec_trees: Float32[Array, 'num_trees k k 2**d'],
    error_cov_inv: Float32[Array, 'k k'],
    leaf_prior_cov_inv: Float32[Array, 'k k'],
    z: Float32[Array, 'num_trees 2**d k'] | None = None,
) -> PreLf:
    num_trees, k, _, tree_size = prec_trees.shape

    # bring the leaf axis to position 1 so chol/solve see (..., k, k)
    prec: Float32[Array, 'num_trees tree_size k k'] = jnp.moveaxis(prec_trees, -1, 1)

    posterior_precision: Float32[Array, 'num_trees tree_size k k'] = (
        leaf_prior_cov_inv + error_cov_inv * prec
    )

    L_prec: Float32[Array, 'num_trees tree_size k k'] = chol_with_gersh(
        posterior_precision
    )

    if z is None:
        z = random.normal(key, (num_trees, tree_size, k))
    centered_leaves: Float32[Array, 'num_trees tree_size k'] = solve_triangular(
        L_prec, z[:, :, :, None], trans='T'
    ).squeeze(-1)

    # restore the leaf axis to the trailing position for storage
    mean_factor_out: Float32[Array, 'num_trees k k tree_size'] = jnp.moveaxis(
        L_prec, 1, -1
    )
    centered_leaves_out: Float32[Array, 'num_trees k tree_size'] = jnp.swapaxes(
        centered_leaves, -1, -2
    )

    return PreLf(mean_factor=mean_factor_out, centered_leaves=centered_leaves_out)


@named_call
def precompute_leaf_terms(
    key: Key[Array, ''],
    prec_trees: Float32[Array, 'num_trees 2**d'] | Float32[Array, 'num_trees k k 2**d'],
    error_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'],
    leaf_prior_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'],
    z: Float32[Array, 'num_trees 2**d']
    | Float32[Array, 'num_trees 2**d k']
    | None = None,
) -> PreLf:
    """
    Pre-compute terms used to sample leaves from their posterior.

    Handles both univariate and multivariate cases based on the shape of the
    input arrays.

    Parameters
    ----------
    key
        A jax random key.
    prec_trees
        The likelihood precision scale in each potential or actual leaf node.
    error_cov_inv
        The inverse error variance (univariate) or the inverse of error
        covariance matrix (multivariate). For univariate case, this is the
        inverse global error variance factor if `prec_scale` is set.
    leaf_prior_cov_inv
        The inverse prior variance of each leaf (univariate) or the inverse of
        prior covariance matrix of each leaf (multivariate).
    z
        Optional standard normal noise to use for sampling the centered leaves.
        This is intended for testing purposes only.

    Returns
    -------
    Pre-computed terms for leaf sampling.
    """
    if error_cov_inv.ndim == 0:
        return _precompute_leaf_terms_uv(
            key, prec_trees, error_cov_inv, leaf_prior_cov_inv, z
        )
    elif prec_trees.ndim == 4:
        return _precompute_leaf_terms_mv_het(
            key, prec_trees, error_cov_inv, leaf_prior_cov_inv, z
        )
    else:
        return _precompute_leaf_terms_mv(
            key, prec_trees, error_cov_inv, leaf_prior_cov_inv, z
        )


@named_call
def accept_moves_sequential_stage(pso: ParallelStageOut) -> tuple[State, Moves]:
    """
    Accept/reject the moves one tree at a time.

    This is the most performance-sensitive function because it contains all and
    only the parts of the algorithm that can not be parallelized across trees.

    Parameters
    ----------
    pso
        The output of `accept_moves_parallel_stage`.

    Returns
    -------
    bart : State
        A partially updated BART mcmc state.
    moves : Moves
        The accepted/rejected moves, with `acc` and `to_prune` set.
    """

    def loop(
        resid: Float32[Array, ' n'] | Float32[Array, ' k n'], pt: SeqStageInPerTree
    ) -> tuple[
        Float32[Array, ' n'] | Float32[Array, ' k n'],
        tuple[
            Float32[Array, ' 2**d'] | Float32[Array, ' k 2**d'],
            Bool[Array, ''],
            Bool[Array, ''],
            Float32[Array, ''] | None,
        ],
    ]:
        resid, leaf_tree, acc, to_prune, lkratio = accept_move_and_sample_leaves(
            resid,
            SeqStageInAllTrees(
                pso.bart.X,
                pso.bart.config.resid_num_batches,
                pso.bart.config.data_sharded,
                pso.bart.prec_scale,
                pso.bart.forest.log_likelihood is not None,
                pso.prelk,
            ),
            pt,
        )
        return resid, (leaf_tree, acc, to_prune, lkratio)

    pts = SeqStageInPerTree(
        pso.bart.forest.leaf_tree,
        pso.prec_trees,
        pso.moves,
        pso.move_precs,
        pso.bart.forest.leaf_indices,
        pso.prelkv,
        pso.prelf,
    )
    resid, (leaf_trees, acc, to_prune, lkratio) = lax.scan(loop, pso.bart.resid, pts)

    bart = replace(
        pso.bart,
        resid=resid,
        forest=replace(pso.bart.forest, leaf_tree=leaf_trees, log_likelihood=lkratio),
    )
    moves = replace(pso.moves, acc=acc, to_prune=to_prune)

    return bart, moves


class SeqStageInAllTrees(Module):
    """The inputs to `accept_move_and_sample_leaves` that are shared by all trees."""

    X: UInt[Array, 'p n']
    """The predictors."""

    resid_num_batches: int | None = field(static=True)
    """The number of batches for computing the sum of residuals in each leaf."""

    data_sharded: bool = field(static=True)
    """Whether the data axis is sharded across devices."""

    prec_scale: Float32[Array, ' n'] | Float32[Array, 'k k n'] | None
    """The scale of the precision of the error on each datapoint. If None, it
    is assumed to be 1."""

    save_ratios: bool = field(static=True)
    """Whether to save the acceptance ratios."""

    prelk: PreLk | None
    """The pre-computed terms of the likelihood ratio which are shared across
    trees."""


class SeqStageInPerTree(Module):
    """The inputs to `accept_move_and_sample_leaves` that are separate for each tree."""

    leaf_tree: Float32[Array, ' 2**d'] | Float32[Array, ' k 2**d']
    """The leaf values of the tree."""

    prec_tree: Float32[Array, ' 2**d'] | Float32[Array, ' k k 2**d']
    """The likelihood precision scale in each potential or actual leaf node."""

    move: Moves
    """The proposed move, see `propose_moves`."""

    move_precs: Precs | Counts
    """The likelihood precision scale in each node modified by the moves."""

    leaf_indices: UInt[Array, ' n']
    """The leaf indices for the largest version of the tree compatible with
    the move."""

    prelkv: PreLkV
    """The pre-computed terms of the likelihood ratio which are specific to the tree."""

    prelf: PreLf
    """The pre-computed terms of the leaf sampling which are specific to the tree."""


@named_call
def accept_move_and_sample_leaves(
    resid: Float32[Array, ' n'] | Float32[Array, ' k n'],
    at: SeqStageInAllTrees,
    pt: SeqStageInPerTree,
) -> tuple[
    Float32[Array, ' n'] | Float32[Array, ' k n'],
    Float32[Array, ' 2**d'] | Float32[Array, ' k 2**d'],
    Bool[Array, ''],
    Bool[Array, ''],
    Float32[Array, ''] | None,
]:
    """
    Accept or reject a proposed move and sample the new leaf values.

    Parameters
    ----------
    resid
        The residuals (data minus forest value).
    at
        The inputs that are the same for all trees.
    pt
        The inputs that are separate for each tree.

    Returns
    -------
    resid : Float32[Array, 'n'] | Float32[Array, ' k n']
        The updated residuals (data minus forest value).
    leaf_tree : Float32[Array, '2**d'] | Float32[Array, ' k 2**d']
        The new leaf values of the tree.
    acc : Bool[Array, '']
        Whether the move was accepted.
    to_prune : Bool[Array, '']
        Whether, to reflect the acceptance status of the move, the state should
        be updated by pruning the leaves involved in the move.
    log_lk_ratio : Float32[Array, ''] | None
        The logarithm of the likelihood ratio for the move. `None` if not to be
        saved.
    """
    # sum residuals in each leaf, in tree proposed by grow move
    if at.prec_scale is None:
        scaled_resid = resid
    else:
        scaled_resid = resid * at.prec_scale

    tree_size = pt.leaf_tree.shape[-1]  # 2**d

    resid_tree = sum_resid(
        scaled_resid, pt.leaf_indices, tree_size, at.resid_num_batches, at.data_sharded
    )

    # subtract starting tree from function
    resid_tree += pt.prec_tree * pt.leaf_tree

    # sum residuals in parent node modified by move and compute likelihood
    resid_left = resid_tree[..., pt.move.left]
    resid_right = resid_tree[..., pt.move.right]
    resid_total = resid_left + resid_right
    assert pt.move.node.dtype == jnp.int32
    resid_tree = resid_tree.at[..., pt.move.node].set(resid_total)

    log_lk_ratio = compute_likelihood_ratio(
        resid_total, resid_left, resid_right, pt.prelkv, at.prelk
    )

    # calculate accept/reject ratio
    log_ratio = pt.move.log_trans_prior_ratio + log_lk_ratio
    log_ratio = jnp.where(pt.move.grow, log_ratio, -log_ratio)
    if not at.save_ratios:
        log_lk_ratio = None

    # determine whether to accept the move
    acc = pt.move.allowed & (pt.move.logu <= log_ratio)

    # compute leaves posterior and sample leaves
    if at.prelk is not None and at.prelk.error_cov_inv is not None:
        # multivariate w/ vector weights
        b_tree = compute_B(at.prelk.error_cov_inv, resid_tree)  # (k, 2**d)
        l_lead = jnp.moveaxis(pt.prelf.mean_factor, -1, 0)  # (2**d, k, k)
        b_lead = b_tree.T[:, :, None]  # (2**d, k, 1)
        y = solve_triangular(l_lead, b_lead, lower=True)
        mu = solve_triangular(l_lead, y, lower=True, trans='T').squeeze(-1)
        mean_post = mu.T  # (k, 2**d)
    elif resid.ndim > 1:
        # multivariate homoskedastic or scalar weights
        mean_post = jnp.einsum('kil,kl->il', pt.prelf.mean_factor, resid_tree)
    else:
        # univariate
        mean_post = resid_tree * pt.prelf.mean_factor
    leaf_tree = mean_post + pt.prelf.centered_leaves

    # copy leaves around such that the leaf indices point to the correct leaf
    to_prune = acc ^ pt.move.grow
    leaf_tree = (
        leaf_tree.at[..., jnp.where(to_prune, pt.move.left, tree_size)]
        .set(leaf_tree[..., pt.move.node])
        .at[..., jnp.where(to_prune, pt.move.right, tree_size)]
        .set(leaf_tree[..., pt.move.node])
    )
    # replace old tree with new tree in function values
    resid += (pt.leaf_tree - leaf_tree)[..., pt.leaf_indices]

    return resid, leaf_tree, acc, to_prune, log_lk_ratio


@named_call
def sum_resid(
    scaled_resid: (
        Float32[Array, ' n'] | Float32[Array, 'k n'] | Float32[Array, 'k k n']
    ),
    leaf_indices: UInt[Array, ' n'],
    tree_size: int,
    resid_num_batches: int | None,
    data_sharded: bool,
) -> (
    Float32[Array, ' {tree_size}']
    | Float32[Array, 'k {tree_size}']
    | Float32[Array, 'k k {tree_size}']
):
    """
    Sum the residuals in each leaf.

    Parameters
    ----------
    scaled_resid
        The residuals (data minus forest value) multiplied by the error
        precision scale.
    leaf_indices
        The leaf indices of the tree (in which leaf each data point falls into).
    tree_size
        The size of the tree array (2 ** d).
    resid_num_batches
        The number of batches for computing the sum of residuals in each leaf.
    data_sharded
        Whether the data axis is sharded; if true, the result is psum-reduced
        across the ``'data'`` axis of the enclosing `shard_map`.

    Returns
    -------
    The per-leaf sum, with the same leading dimensions as ``scaled_resid`` and
    a trailing axis over the leaves.
    """
    return _scatter_add(
        scaled_resid,
        leaf_indices,
        tree_size,
        jnp.float32,
        resid_num_batches,
        data_sharded,
    )


def _scatter_add(
    values: Float32[Array, '*batch_shape n'] | int,
    indices: Integer[Array, ' n'],
    size: int,
    dtype: jnp.dtype,
    batch_size: int | None,
    data_sharded: bool,
) -> Shaped[Array, '*batch_shape {size}']:
    """Indexed reduce along the last axis of `values`, with optional batching.

    When `data_sharded` is True the result is psum-reduced across the
    ``'data'`` axis of the current `shard_map` region.
    """
    values = jnp.asarray(values)
    assert values.ndim == 0 or values.shape[-1:] == indices.shape
    return _scatter_add_impl(
        values,
        indices,
        size=size,
        dtype=dtype,
        num_batches=batch_size,
        final_psum=data_sharded,
    )


def _scatter_add_impl(
    values: Float32[Array, '*batch_shape n'] | Int32[Array, ''],
    indices: Integer[Array, ' n'],
    /,
    *,
    size: int,
    dtype: jnp.dtype,
    num_batches: int | None,
    final_psum: bool = False,
) -> Shaped[Array, '*batch_shape {size}']:
    batch_shape = values.shape[:-1]
    if num_batches is None:
        out = jnp.zeros((*batch_shape, size), dtype).at[..., indices].add(values)

    else:
        # in the sharded case, n is the size of the local shard, not the full size
        (n,) = indices.shape
        batch_indices = jnp.arange(n) % num_batches
        out = (
            jnp.zeros((*batch_shape, size, num_batches), dtype)
            .at[..., indices, batch_indices]
            .add(values)
            .sum(axis=-1)
        )

    if final_psum:
        out = lax.psum(out, 'data')
    return out


def _compute_likelihood_ratio_uv(
    total_resid: Float32[Array, ''],
    left_resid: Float32[Array, ''],
    right_resid: Float32[Array, ''],
    prelkv: PreLkV,
    prelk: PreLk,
) -> Float32[Array, '']:
    exp_term = prelk.exp_factor * (
        left_resid * left_resid / prelkv.left
        + right_resid * right_resid / prelkv.right
        - total_resid * total_resid / prelkv.total
    )
    return prelkv.log_sqrt_term + exp_term


def _compute_likelihood_ratio_mv(
    total_resid: Float32[Array, ' k'],
    left_resid: Float32[Array, ' k'],
    right_resid: Float32[Array, ' k'],
    prelkv: PreLkV,
) -> Float32[Array, '']:
    def _quadratic_form(
        r: Float32[Array, ' k'], mat: Float32[Array, 'k k']
    ) -> Float32[Array, '']:
        return r @ mat @ r

    qf_left = _quadratic_form(left_resid, prelkv.left)
    qf_right = _quadratic_form(right_resid, prelkv.right)
    qf_total = _quadratic_form(total_resid, prelkv.total)
    exp_term = 0.5 * (qf_left + qf_right - qf_total)
    return prelkv.log_sqrt_term + exp_term


def _compute_likelihood_ratio_mv_het(
    total_resid: Float32[Array, 'k k'],
    left_resid: Float32[Array, 'k k'],
    right_resid: Float32[Array, 'k k'],
    error_cov_inv: Float32[Array, 'k k'],
    prelkv: PreLkV,
) -> Float32[Array, '']:
    def quad(
        L: Float32[Array, 'k k'], resid: Float32[Array, 'k k']
    ) -> Float32[Array, '']:
        b = compute_B(error_cov_inv, resid)
        y = solve_triangular(L, b[:, None], lower=True).squeeze(-1)
        return y @ y

    qf_left = quad(prelkv.left, left_resid)
    qf_right = quad(prelkv.right, right_resid)
    qf_total = quad(prelkv.total, total_resid)
    exp_term = 0.5 * (qf_left + qf_right - qf_total)
    return prelkv.log_sqrt_term + exp_term


@named_call
def compute_likelihood_ratio(
    total_resid: (Float32[Array, ''] | Float32[Array, ' k'] | Float32[Array, 'k k']),
    left_resid: (Float32[Array, ''] | Float32[Array, ' k'] | Float32[Array, 'k k']),
    right_resid: (Float32[Array, ''] | Float32[Array, ' k'] | Float32[Array, 'k k']),
    prelkv: PreLkV,
    prelk: PreLk | None,
) -> Float32[Array, '']:
    """
    Compute the likelihood ratio of a grow move.

    Parameters
    ----------
    total_resid
    left_resid
    right_resid
        The sum of the residuals (scaled by error precision scale) of the
        datapoints falling in the nodes involved in the moves.
    prelkv
    prelk
        The pre-computed terms of the likelihood ratio, see
        `precompute_likelihood_terms`.

    Returns
    -------
    The log-likelihood ratio log P(data | new tree) - log P(data | old tree).
    """
    if prelk is not None and prelk.error_cov_inv is not None:
        return _compute_likelihood_ratio_mv_het(
            total_resid, left_resid, right_resid, prelk.error_cov_inv, prelkv
        )
    elif total_resid.ndim > 0:
        return _compute_likelihood_ratio_mv(
            total_resid, left_resid, right_resid, prelkv
        )
    else:
        assert prelk is not None
        return _compute_likelihood_ratio_uv(
            total_resid, left_resid, right_resid, prelkv, prelk
        )


@named_call
def accept_moves_final_stage(bart: State, moves: Moves) -> State:
    """
    Post-process the mcmc state after accepting/rejecting the moves.

    This function is separate from `accept_moves_sequential_stage` to signal it
    can work in parallel across trees.

    Parameters
    ----------
    bart
        A partially updated BART mcmc state.
    moves
        The proposed moves (see `propose_moves`) as updated by
        `accept_moves_sequential_stage`.

    Returns
    -------
    The fully updated BART mcmc state.
    """
    return replace(
        bart,
        forest=replace(
            bart.forest,
            grow_acc_count=jnp.sum(moves.acc & moves.grow),
            prune_acc_count=jnp.sum(moves.acc & ~moves.grow),
            leaf_indices=apply_moves_to_leaf_indices(bart.forest.leaf_indices, moves),
            split_tree=apply_moves_to_split_trees(bart.forest.split_tree, moves),
        ),
    )


@named_call
@vmap_nodoc
def apply_moves_to_leaf_indices(
    leaf_indices: UInt[Array, 'num_trees n'], moves: Moves
) -> UInt[Array, 'num_trees n']:
    """
    Update the leaf indices to match the accepted move.

    Parameters
    ----------
    leaf_indices
        The index of the leaf each datapoint falls into, if the grow move was
        accepted.
    moves
        The proposed moves (see `propose_moves`), as updated by
        `accept_moves_sequential_stage`.

    Returns
    -------
    The updated leaf indices.
    """
    mask = ~jnp.array(1, leaf_indices.dtype)  # ...1111111110
    is_child = (leaf_indices & mask) == moves.left
    assert moves.to_prune is not None
    return jnp.where(
        is_child & moves.to_prune, moves.node.astype(leaf_indices.dtype), leaf_indices
    )


@named_call
@vmap_nodoc
def apply_moves_to_split_trees(
    split_tree: UInt[Array, 'num_trees 2**(d-1)'], moves: Moves
) -> UInt[Array, 'num_trees 2**(d-1)']:
    """
    Update the split trees to match the accepted move.

    Parameters
    ----------
    split_tree
        The cutpoints of the decision nodes in the initial trees.
    moves
        The proposed moves (see `propose_moves`), as updated by
        `accept_moves_sequential_stage`.

    Returns
    -------
    The updated split trees.
    """
    assert moves.to_prune is not None
    return (
        split_tree.at[jnp.where(moves.grow, moves.node, split_tree.size)]
        .set(moves.grow_split.astype(split_tree.dtype))
        .at[jnp.where(moves.to_prune, moves.node, split_tree.size)]
        .set(0)
    )


@jax.jit
def _sample_wishart_bartlett(
    key: Key[Array, ''], df: Float32[Array, ''], scale_inv: Float32[Array, 'k k']
) -> Float32[Array, 'k k']:
    """
    Sample a precision matrix W ~ Wishart(df, scale_inv^-1) using Bartlett decomposition.

    Parameters
    ----------
    key
        A JAX random key
    df
        Degrees of freedom
    scale_inv
        Scale matrix of the corresponding Inverse Wishart distribution

    Returns
    -------
    A sample from Wishart(df, scale)
    """
    keys = split(key)

    # Diagonal elements: A_ii ~ sqrt(chi^2(df - i))
    # chi^2(k) = Gamma(k/2, scale=2)
    k, _ = scale_inv.shape
    df_vector = df - jnp.arange(k)
    chi2_samples = random.gamma(keys.pop(), df_vector / 2.0) * 2.0
    diag_A = jnp.sqrt(chi2_samples)

    off_diag_A = random.normal(keys.pop(), (k, k))
    A = jnp.tril(off_diag_A, -1) + jnp.diag(diag_A)
    L = chol_with_gersh(scale_inv, absolute_eps=True)
    T = solve_triangular(L, A, lower=True, trans='T')

    return T @ T.T


def _step_error_cov_inv_mv(key: Key[Array, ''], bart: State) -> State:
    assert bart.error_cov_df is not None
    assert bart.error_cov_scale is not None

    resid = bart.resid
    if bart.inv_sdev_scale is None:
        _, n_eff = resid.shape
        n_eff *= get_axis_size(bart.config.mesh, 'data')
    else:
        # 2-D inv_sdev_scale dispatches to the diagonal path, so here it is 1-D
        n_eff = jnp.sum(bart.inv_sdev_scale != 0, axis=-1)
        if bart.config.data_sharded:
            n_eff = lax.psum(n_eff, 'data')
        resid *= bart.inv_sdev_scale
    df_post = bart.error_cov_df + n_eff
    rrt = resid @ resid.T
    if bart.config.data_sharded:
        rrt = lax.psum(rrt, 'data')
    scale_post = bart.error_cov_scale + rrt

    prec = _sample_wishart_bartlett(key, df_post, scale_post)
    return replace(bart, error_cov_inv=prec)


def _step_error_cov_inv_diag(key: Key[Array, ''], bart: State) -> State:
    """Per-component inverse-gamma update for univariate, mixed, and partial-missing paths."""
    assert bart.error_cov_scale is not None
    assert bart.error_cov_df is not None

    resid = bart.resid
    if bart.inv_sdev_scale is not None:
        resid *= bart.inv_sdev_scale

    # alpha
    if bart.inv_sdev_scale is None:
        *_, n_eff = resid.shape
        n_eff *= get_axis_size(bart.config.mesh, 'data')
    else:
        n_eff = jnp.sum(bart.inv_sdev_scale != 0, axis=-1)
        if bart.config.data_sharded:
            n_eff = lax.psum(n_eff, 'data')
    alpha = bart.error_cov_df / 2 + n_eff / 2

    # beta
    norm2 = jnp.einsum('...n,...n->...', resid, resid)
    if bart.config.data_sharded:
        norm2 = lax.psum(norm2, 'data')
    scale = bart.error_cov_scale
    kshape = resid.shape[:-1]
    if kshape:
        scale = jnp.diag(scale)
    beta = scale / 2 + norm2 / 2

    samples = random.gamma(key, alpha, kshape)
    prec = samples / beta
    if bart.binary_indices is not None:
        prec = prec.at[bart.binary_indices].set(1.0)
    if kshape:
        prec = jnp.diag(prec)
    return replace(bart, error_cov_inv=prec)


@named_call
def step_error_cov_inv(key: Key[Array, ''], bart: State) -> State:
    """MCMC-update the inverse error covariance."""
    if (
        bart.error_cov_inv.ndim == 2
        and bart.binary_indices is None
        and (bart.inv_sdev_scale is None or bart.inv_sdev_scale.ndim == 1)
    ):
        return _step_error_cov_inv_mv(key, bart)
    else:
        return _step_error_cov_inv_diag(key, bart)


@named_call
def step_z(key: Key[Array, ''], bart: State) -> State:
    """
    MCMC-update the latent variable for binary regression.

    Parameters
    ----------
    key
        A jax random key.
    bart
        A BART MCMC state.

    Returns
    -------
    The updated BART MCMC state.
    """
    assert bart.z is not None
    assert bart.binary_y is not None

    if bart.binary_indices is not None:
        resid = bart.resid[..., bart.binary_indices, :]
    else:
        resid = bart.resid

    trees_plus_offset = bart.z - resid
    if bart.config.data_sharded:
        # decorrelate the seed across data shards; the seed is replicated
        # because the trees and most of the algorithm are replicated
        key = random.fold_in(key, lax.axis_index('data'))
    resid = truncated_normal_onesided(key, (), ~bart.binary_y, -trees_plus_offset)
    z = trees_plus_offset + resid

    if bart.binary_indices is not None:
        resid = bart.resid.at[..., bart.binary_indices, :].set(resid)

    return replace(bart, z=z, resid=resid)


@named_call
def step_s(key: Key[Array, ''], bart: State) -> State:
    """
    Update `log_s` using Dirichlet sampling.

    The prior is s ~ Dirichlet(theta/p, ..., theta/p), and the posterior
    is s ~ Dirichlet(theta/p + varcount, ..., theta/p + varcount), where
    varcount is the count of how many times each variable is used in the
    current forest.

    Parameters
    ----------
    key
        Random key for sampling.
    bart
        The current BART state.

    Returns
    -------
    Updated BART state with re-sampled `log_s`.

    Notes
    -----
    This full conditional is approximated, because it does not take into account
    that there are forbidden decision rules.
    """
    assert bart.forest.theta is not None

    # histogram current variable usage
    p = bart.forest.max_split.size
    varcount = var_histogram(
        p, bart.forest.var_tree, bart.forest.split_tree, sum_batch_axis=-1
    )

    # sample from Dirichlet posterior
    alpha = bart.forest.theta / p + varcount
    log_s = random.loggamma(key, alpha)

    # update forest with new s
    return replace(bart, forest=replace(bart.forest, log_s=log_s))


@named_call
def step_theta(key: Key[Array, ''], bart: State, *, num_grid: int = 1000) -> State:
    """
    Update `theta`.

    The prior is theta / (theta + rho) ~ Beta(a, b).

    Parameters
    ----------
    key
        Random key for sampling.
    bart
        The current BART state.
    num_grid
        The number of points in the evenly-spaced grid used to sample
        theta / (theta + rho).

    Returns
    -------
    Updated BART state with re-sampled `theta`.
    """
    assert bart.forest.log_s is not None
    assert bart.forest.rho is not None
    assert bart.forest.a is not None
    assert bart.forest.b is not None

    # the grid points are the midpoints of num_grid bins in (0, 1)
    padding = 1 / (2 * num_grid)
    lambda_grid = jnp.linspace(padding, 1 - padding, num_grid)

    # normalize s
    log_s = bart.forest.log_s - logsumexp(bart.forest.log_s)

    # sample lambda
    logp, theta_grid = _log_p_lambda(
        lambda_grid, log_s, bart.forest.rho, bart.forest.a, bart.forest.b
    )
    i = random.categorical(key, logp)
    theta = theta_grid[i]

    return replace(bart, forest=replace(bart.forest, theta=theta))


def _log_p_lambda(
    lambda_: Float32[Array, ' num_grid'],
    log_s: Float32[Array, ' p'],
    rho: Float32[Array, ''],
    a: Float32[Array, ''],
    b: Float32[Array, ''],
) -> tuple[Float32[Array, ' num_grid'], Float32[Array, ' num_grid']]:
    # in the following I use lambda_[::-1] == 1 - lambda_
    theta = rho * lambda_ / lambda_[::-1]
    p = log_s.size
    return (
        (a - 1) * jnp.log1p(-lambda_[::-1])  # log(lambda)
        + (b - 1) * jnp.log1p(-lambda_)  # log(1 - lambda)
        + gammaln(theta)
        - p * gammaln(theta / p)
        + theta / p * jnp.sum(log_s)
    ), theta


@named_call
def step_sparse(key: Key[Array, ''], bart: State) -> State:
    """
    Update the sparsity parameters.

    This invokes `step_s`, and then `step_theta` only if the parameters of
    the theta prior are defined.

    Parameters
    ----------
    key
        Random key for sampling.
    bart
        The current BART state.

    Returns
    -------
    Updated BART state with re-sampled `log_s` and `theta`.
    """
    if bart.config.sparse_on_at is not None:
        bart = lax.cond(
            bart.config.steps_done < bart.config.sparse_on_at,
            lambda _key, bart: bart,
            _step_sparse,
            key,
            bart,
        )
    return bart


def _step_sparse(key: Key[Array, ''], bart: State) -> State:
    keys = split(key)
    bart = step_s(keys.pop(), bart)
    if bart.forest.rho is not None:
        bart = step_theta(keys.pop(), bart)
    return bart


@named_call
def step_config(bart: State) -> State:
    config = bart.config
    config = replace(config, steps_done=config.steps_done + 1)
    return replace(bart, config=config)
