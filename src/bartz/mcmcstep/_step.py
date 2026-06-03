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
from typing import Literal

from equinox import AbstractVar
from jax import lax, named_call, random, vmap
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln, logsumexp
from jaxtyping import Array, Bool, Float32, Key, Shaped, UInt, UInt32

from bartz._jaxext import (
    Module,
    field,
    jit,
    split,
    truncated_normal_onesided,
    vmap_nodoc,
)
from bartz._jaxext.random import loggamma
from bartz.grove import var_histogram
from bartz.mcmcstep._moves import Moves, propose_moves
from bartz.mcmcstep._scatter import _scatter_add
from bartz.mcmcstep._state import (
    State,
    StepConfig,
    chol_with_gersh,
    get_axis_size,
    shard_map_state,
    split_key_for_chains,
    vmap_chains,
)


@jit(donate_argnums=(1,))
@split_key_for_chains
@shard_map_state
@vmap_chains
def step(key: Key[Array, ''], state: State) -> State:
    """
    Do one MCMC step.

    Parameters
    ----------
    key
        A jax random key.
    state
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

    state = step_trees(keys.pop(), state)

    if state.z is not None:
        state = step_z(keys.pop(), state)

    if state.error_cov_df is not None:
        state = step_error_cov_inv(keys.pop(), state)

    state = step_sparse(keys.pop(), state)
    return step_config(state)


@named_call
def step_trees(key: Key[Array, ''], state: State) -> State:
    """
    Forest sampling step of BART MCMC.

    Parameters
    ----------
    key
        A jax random key.
    state
        A BART mcmc state, as created by `init`.

    Returns
    -------
    The new BART mcmc state.

    Notes
    -----
    This function zeroes the proposal counters.
    """
    keys = split(key)
    moves = propose_moves(keys.pop(), state.forest)
    return accept_moves_and_sample_leaves(keys.pop(), state, moves)


@named_call
def accept_moves_and_sample_leaves(
    key: Key[Array, ''], state: State, moves: Moves
) -> State:
    """
    Accept or reject the proposed moves and sample the new leaf values.

    Parameters
    ----------
    key
        A jax random key.
    state
        A valid BART mcmc state.
    moves
        The proposed moves, see `propose_moves`.

    Returns
    -------
    A new (valid) BART mcmc state.
    """
    pso = accept_moves_parallel_stage(key, state, moves)
    state, moves = accept_moves_sequential_stage(pso)
    return accept_moves_final_stage(state, moves)


class Counts(Module):
    """Number of datapoints in the nodes involved in proposed moves for each tree."""

    left: UInt[Array, '*num_trees']
    """Number of datapoints in the left child."""

    right: UInt[Array, '*num_trees']
    """Number of datapoints in the right child."""

    total: UInt[Array, '*num_trees']
    """Number of datapoints in the parent (``= left + right``)."""


class Precs(Module):
    """Likelihood precision scale in the nodes involved in proposed moves for each tree.

    The "likelihood precision scale" of a tree node is the sum of the inverse
    squared error scales of the datapoints selected by the node. It is a scalar
    per node (`PrecsScalar`) with scalar error weights, or a ``k k`` matrix per
    node (`PrecsMatrix`) with vector error weights.

    Abstract base: it is built one tree at a time under `vmap`/`lax.map`
    (`_compute_count_or_prec_tree`), so the ``num_trees`` axis is variadic
    (``*num_trees``, absent per element, present once batched). The scalar and
    matrix layouts differ in rank, so they live in two concrete subclasses with
    union-free annotations; a single class carrying a ``... | ... k k`` union
    would make the greedy variadic mis-bind against the ``k`` axis under the
    runtime typechecker.
    """

    left: AbstractVar[Float32[Array, '*num_trees'] | Float32[Array, '*num_trees k k']]
    """Likelihood precision scale in the left child."""

    right: AbstractVar[Float32[Array, '*num_trees'] | Float32[Array, '*num_trees k k']]
    """Likelihood precision scale in the right child."""

    total: AbstractVar[Float32[Array, '*num_trees'] | Float32[Array, '*num_trees k k']]
    """Likelihood precision scale in the parent (``= left + right``)."""


class PrecsScalar(Precs):
    """`Precs` with a scalar precision per node (univariate or scalar-weight case)."""

    left: Float32[Array, '*num_trees']
    right: Float32[Array, '*num_trees']
    total: Float32[Array, '*num_trees']


class PrecsMatrix(Precs):
    """`Precs` with a ``k k`` precision matrix per node (vector-weight case)."""

    left: Float32[Array, '*num_trees k k']
    right: Float32[Array, '*num_trees k k']
    total: Float32[Array, '*num_trees k k']


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

    # `log_sqrt_term` is declared before `left`/`right`/`total` so its single
    # (union-free) annotation binds the variadic `*num_trees` axis first;
    # otherwise the runtime typechecker can greedily mis-bind `*num_trees`
    # against the `k` axis of the `... | ... k k` unions (the multivariate and
    # univariate layouts are rank-ambiguous).
    log_sqrt_term: Float32[Array, '*num_trees']
    """The logarithm of the square root term of the likelihood ratio."""

    left: Float32[Array, '*num_trees'] | Float32[Array, '*num_trees k k']
    """Full conditional variance, scaled covariance, or precision cholesky, for
    the left leaf."""

    right: Float32[Array, '*num_trees'] | Float32[Array, '*num_trees k k']
    """Full conditional variance, scaled covariance, or precision cholesky, for
    the right leaf."""

    total: Float32[Array, '*num_trees'] | Float32[Array, '*num_trees k k']
    """Full conditional variance, scaled covariance, or precision cholesky, for
    the the join of the left and right leaves."""


class PreLk(Module):
    """Non-sequential terms of the likelihood ratio shared by all trees."""

    exp_factor: Float32[Array, ''] | None
    """The factor to multiply the likelihood ratio by, shared by all trees.
    Set only in the univariate path."""

    error_cov_inv: Float32[Array, 'k k'] | None
    """The global error precision scale. Set only in the multivariate
    heteroskedastic vector-weight case."""


class PreLf(Module):
    """Pre-computed terms used to sample leaves from their posterior.

    These terms can be computed in parallel across trees.

    For each tree and leaf, the terms are scalars in the univariate case
    (`PreLfUV`), and matrices/vectors in the multivariate case (`PreLfMV`).

    Abstract base: the univariate and multivariate layouts differ in rank, so
    they live in two concrete subclasses with union-free annotations (see `Precs`
    for the runtime-typechecker rationale). The ``num_trees`` axis is variadic so
    the same annotations also match a per-element layout if vmapped over trees.
    """

    mean_factor: AbstractVar[
        Float32[Array, '*num_trees tree_size']
        | Float32[Array, '*num_trees k k tree_size']
    ]
    """The factor to be right-multiplied by the sum of the scaled residuals to
    obtain the posterior mean."""

    centered_leaves: AbstractVar[
        Float32[Array, '*num_trees tree_size']
        | Float32[Array, '*num_trees k tree_size']
    ]
    """The mean-zero normal values to be added to the posterior mean to
    obtain the posterior leaf samples."""


class PreLfUV(PreLf):
    """`PreLf` for the univariate case."""

    mean_factor: Float32[Array, '*num_trees tree_size']
    centered_leaves: Float32[Array, '*num_trees tree_size']


class PreLfMV(PreLf):
    """`PreLf` for the multivariate case."""

    mean_factor: Float32[Array, '*num_trees k k tree_size']
    centered_leaves: Float32[Array, '*num_trees k tree_size']


class ParallelStageOut(Module):
    """The output of `accept_moves_parallel_stage`."""

    state: State
    """A partially updated BART mcmc state."""

    moves: Moves
    """The proposed moves, with `partial_ratio` set to `None` and
    `log_trans_prior_ratio` set to its final value."""

    # `num_trees` stays a fixed (non-variadic) axis: `ParallelStageOut` is always
    # built with the tree axis present (never per tree under vmap), so the union
    # is disambiguated by rank/dtype and needs no anchor (cf. `Precs`/`PreLf`).
    prec_trees: (
        Float32[Array, 'num_trees tree_size']
        | UInt32[Array, 'num_trees tree_size']
        | Float32[Array, 'num_trees k k tree_size']
    )
    """The likelihood precision scale in each potential or actual leaf node."""

    move_precs: Precs | Counts
    """The likelihood precision scale in each node modified by the moves. If
    `state.prec_scale` is not set, this is set to `move_counts`."""

    prelkv: PreLkV
    """Object with pre-computed terms of the likelihood ratios."""

    prelk: PreLk | None
    """Object with pre-computed terms of the likelihood ratios."""

    prelf: PreLf
    """Object with pre-computed terms of the leaf samples."""


@named_call
def accept_moves_parallel_stage(
    key: Key[Array, ''], state: State, moves: Moves
) -> ParallelStageOut:
    """
    Pre-compute quantities used to accept moves, in parallel across trees.

    Parameters
    ----------
    key
        A jax random key.
    state
        A BART mcmc state.
    moves
        The proposed moves, see `propose_moves`.

    Returns
    -------
    An object with all that could be done in parallel.
    """
    # where the move is grow, modify the state like the move was accepted
    state = replace(
        state,
        forest=replace(
            state.forest,
            var_tree=moves.var_tree,
            leaf_indices=apply_grow_to_indices(
                moves, state.forest.leaf_indices, state.X
            ),
            leaf_tree=adapt_leaf_trees_to_grow_indices(state.forest.leaf_tree, moves),
        ),
    )

    # count number of datapoints per leaf
    if (
        state.forest.min_points_per_decision_node is not None
        or state.forest.min_points_per_leaf is not None
        or state.prec_scale is None
    ):
        count_trees, move_counts = compute_count_trees(
            state.forest.leaf_indices, moves, state.config
        )

    # affluence of the nodes touched by each move: whether `node`/`left`/`right`,
    # as leaves, would be growable (admissible rule + enough datapoints). The
    # children must also lie within the heap, i.e. not be at the bottom level;
    # `node` always does and is admissible as a leaf, so it only needs the count.
    # These feed the transition ratio and the final `affluence_tree` update.
    _, half = state.forest.var_tree.shape
    left_in_heap = moves.left < half
    right_in_heap = moves.right < half
    if state.forest.min_points_per_decision_node is not None:
        min_dn = state.forest.min_points_per_decision_node
        moves = replace(
            moves,
            node_affluent=move_counts.total >= min_dn,
            left_affluent=left_in_heap
            & moves.left_growable
            & (move_counts.left >= min_dn),
            right_affluent=right_in_heap
            & moves.right_growable
            & (move_counts.right >= min_dn),
        )
    else:
        moves = replace(
            moves,
            node_affluent=jnp.ones_like(moves.grow),
            left_affluent=left_in_heap & moves.left_growable,
            right_affluent=right_in_heap & moves.right_growable,
        )

    # veto grove move if new leaves don't have enough datapoints
    if state.forest.min_points_per_leaf is not None:
        moves = replace(
            moves,
            allowed=moves.allowed
            & (move_counts.left >= state.forest.min_points_per_leaf)
            & (move_counts.right >= state.forest.min_points_per_leaf),
        )

    # count number of datapoints per leaf, weighted by error precision scale
    if state.prec_scale is None:
        prec_trees = count_trees
        move_precs = move_counts
    else:
        prec_trees, move_precs = compute_prec_trees(
            state.prec_scale, state.forest.leaf_indices, moves, state.config
        )
    assert move_precs is not None

    # compute some missing information about moves
    moves = complete_ratio(moves, state.forest.p_nonterminal)
    save_ratios = state.forest.log_likelihood is not None
    state = replace(
        state,
        forest=replace(
            state.forest,
            grow_prop_count=jnp.sum(moves.grow),
            prune_prop_count=jnp.sum(moves.allowed & ~moves.grow),
            log_trans_prior=moves.log_trans_prior_ratio if save_ratios else None,
        ),
    )

    prelkv, prelk = precompute_likelihood_terms(
        state.error_cov_inv, state.forest.leaf_prior_cov_inv, move_precs
    )
    prelf = precompute_leaf_terms(
        key, prec_trees, state.error_cov_inv, state.forest.leaf_prior_cov_inv
    )

    return ParallelStageOut(
        state=state,
        moves=moves,
        prec_trees=prec_trees,
        move_precs=move_precs,
        prelkv=prelkv,
        prelk=prelk,
        prelf=prelf,
    )


@named_call
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
    return _apply_grow_to_indices(moves, leaf_indices, X)


@partial(vmap_nodoc, in_axes=(0, 0, None))
def _apply_grow_to_indices(
    moves: Moves, leaf_indices: UInt[Array, ' n'], X: UInt[Array, 'p n']
) -> UInt[Array, ' n']:
    """Implement `apply_grow_to_indices`."""
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
    tuple[UInt32[Array, 'num_trees tree_size'], Counts]
    | tuple[Float32[Array, 'num_trees tree_size'], Precs]
    | tuple[Float32[Array, 'num_trees k k tree_size'], Precs]
):
    """Implement `compute_count_trees` and `compute_prec_trees`."""
    if config.prec_count_num_trees is None:
        compute = vmap(_compute_count_or_prec_tree, in_axes=(None, 0, 0, None))
        return compute(prec_scale, leaf_indices, moves, config)

    def compute(
        args: tuple[UInt[Array, ' n'], Moves],
    ) -> (
        tuple[UInt32[Array, ' tree_size'], Counts]
        | tuple[Float32[Array, ' tree_size'], Precs]
        | tuple[Float32[Array, 'k k tree_size'], Precs]
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
    tuple[UInt32[Array, ' tree_size'], Counts]
    | tuple[Float32[Array, ' tree_size'], Precs]
    | tuple[Float32[Array, 'k k tree_size'], Precs]
):
    """Compute count or precision tree for a single tree."""
    (tree_size,) = moves.var_tree.shape
    tree_size *= 2

    if prec_scale is None:
        value = 1
        cls = Counts
        dtype = jnp.uint32
        num_batches = config.count_num_batches
        which = 'count'
    else:
        value = prec_scale
        # scalar weights -> scalar precision per node; vector weights (k k n) ->
        # k by k precision matrix per node.
        cls = PrecsMatrix if prec_scale.ndim == 3 else PrecsScalar
        dtype = jnp.float32
        num_batches = config.prec_num_batches
        which = 'prec'

    trees = _scatter_add(
        value, leaf_indices, tree_size, dtype, num_batches, which, config.data_sharded
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
) -> tuple[UInt32[Array, 'num_trees tree_size'], Counts]:
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
    count_trees : UInt32[Array, 'num_trees tree_size']
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
    Float32[Array, 'num_trees tree_size'] | Float32[Array, 'num_trees k k tree_size'],
    Precs,
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
    prec_trees : Float32[Array, 'num_trees tree_size'] | Float32[Array, 'num_trees k k tree_size']
        The likelihood precision scale in each potential or actual leaf node.
    precs : Precs
        The likelihood precision scale in the nodes involved in the moves.
    """
    return _compute_count_or_prec_trees(prec_scale, leaf_indices, moves, config)


@partial(vmap_nodoc, in_axes=(0, None))
def complete_ratio(moves: Moves, p_nonterminal: Float32[Array, ' tree_size']) -> Moves:
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
    # can the children be grown by the proposal? `left_affluent`/`right_affluent`
    # already fold in the `min_points_per_decision_node` threshold, because the
    # grow proposal draws from the pool of leaves that pass it. This enters only
    # the transition probability.
    left_affluent = moves.left_affluent
    right_affluent = moves.right_affluent

    # p_prune if grow
    other_growable_leaves = moves.num_growable >= 2
    grow_again_allowed = other_growable_leaves | left_affluent | right_affluent
    grow_p_prune = jnp.where(grow_again_allowed, 0.5, 1.0)

    # p_prune if prune
    prune_p_prune = jnp.where(moves.num_growable, 0.5, 1)

    # select p_prune
    p_prune = jnp.where(moves.grow, grow_p_prune, prune_p_prune)

    # prior probability of both children being terminal. This uses the
    # admissibility ignoring counts, because the standard BART prior conditions
    # the non-terminal probability only on the existence of available decision
    # rules, not on the count thresholds (which are a bartz proposal-efficiency
    # device, not part of the target distribution).
    pt_left = 1 - p_nonterminal[moves.left] * moves.left_growable
    pt_right = 1 - p_nonterminal[moves.right] * moves.right_growable
    pt_children = pt_left * pt_right

    assert moves.partial_ratio is not None
    return replace(
        moves,
        log_trans_prior_ratio=jnp.log(moves.partial_ratio * pt_children * p_prune),
        partial_ratio=None,
    )


@named_call
def adapt_leaf_trees_to_grow_indices(
    leaf_trees: Float32[Array, 'num_trees tree_size']
    | Float32[Array, 'num_trees k tree_size'],
    moves: Moves,
) -> Float32[Array, 'num_trees tree_size'] | Float32[Array, 'num_trees k tree_size']:
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
    return _adapt_leaf_trees_to_grow_indices(leaf_trees, moves)


@vmap_nodoc
def _adapt_leaf_trees_to_grow_indices(
    leaf_trees: Float32[Array, ' tree_size'] | Float32[Array, ' k tree_size'],
    moves: Moves,
) -> Float32[Array, ' tree_size'] | Float32[Array, ' k tree_size']:
    """Implement `adapt_leaf_trees_to_grow_indices`."""
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
    move_precs: PrecsScalar | Counts,
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
    move_precs: PrecsMatrix,
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
    move_precs: PrecsScalar | Counts,
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
    elif isinstance(move_precs, PrecsMatrix):
        return _precompute_likelihood_terms_mv_het(
            error_cov_inv, leaf_prior_cov_inv, move_precs
        )
    else:
        return _precompute_likelihood_terms_mv(
            error_cov_inv, leaf_prior_cov_inv, move_precs
        )


def _precompute_leaf_terms_uv(
    key: Key[Array, ''],
    prec_trees: Float32[Array, 'num_trees tree_size']
    | UInt32[Array, 'num_trees tree_size'],
    error_cov_inv: Float32[Array, ''],
    leaf_prior_cov_inv: Float32[Array, ''],
    z: Float32[Array, 'num_trees tree_size'] | None = None,
) -> PreLfUV:
    prec_lk = prec_trees * error_cov_inv
    var_post = jnp.reciprocal(prec_lk + leaf_prior_cov_inv)
    if z is None:
        z = random.normal(key, prec_trees.shape, error_cov_inv.dtype)
    return PreLfUV(
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
    prec_trees: Float32[Array, 'num_trees tree_size']
    | UInt32[Array, 'num_trees tree_size'],
    error_cov_inv: Float32[Array, 'k k'],
    leaf_prior_cov_inv: Float32[Array, 'k k'],
    z: Float32[Array, 'num_trees tree_size k'] | None = None,
) -> PreLfMV:
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

    return PreLfMV(mean_factor=mean_factor_out, centered_leaves=centered_leaves_out)


def _precompute_leaf_terms_mv_het(
    key: Key[Array, ''],
    prec_trees: Float32[Array, 'num_trees k k tree_size'],
    error_cov_inv: Float32[Array, 'k k'],
    leaf_prior_cov_inv: Float32[Array, 'k k'],
    z: Float32[Array, 'num_trees tree_size k'] | None = None,
) -> PreLfMV:
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

    return PreLfMV(mean_factor=mean_factor_out, centered_leaves=centered_leaves_out)


@named_call
def precompute_leaf_terms(
    key: Key[Array, ''],
    prec_trees: Float32[Array, 'num_trees tree_size']
    | UInt32[Array, 'num_trees tree_size']
    | Float32[Array, 'num_trees k k tree_size'],
    error_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'],
    leaf_prior_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'],
    z: Float32[Array, 'num_trees tree_size']
    | Float32[Array, 'num_trees tree_size k']
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
    state : State
        A partially updated BART mcmc state.
    moves : Moves
        The accepted/rejected moves, with `acc` and `to_prune` set.
    """

    def loop(
        resid: Float32[Array, ' n'] | Float32[Array, ' k n'], pt: SeqStageInPerTree
    ) -> tuple[
        Float32[Array, ' n'] | Float32[Array, ' k n'],
        tuple[
            Float32[Array, ' tree_size'] | Float32[Array, ' k tree_size'],
            Bool[Array, ''],
            Bool[Array, ''],
            Float32[Array, ''] | None,
        ],
    ]:
        resid, leaf_tree, acc, to_prune, lkratio = accept_move_and_sample_leaves(
            resid,
            SeqStageInAllTrees(
                pso.state.X,
                pso.state.config.resid_num_batches,
                pso.state.config.data_sharded,
                pso.state.prec_scale,
                pso.state.forest.log_likelihood is not None,
                pso.prelk,
            ),
            pt,
        )
        return resid, (leaf_tree, acc, to_prune, lkratio)

    pts = SeqStageInPerTree(
        pso.state.forest.leaf_tree,
        pso.prec_trees,
        pso.moves,
        pso.move_precs,
        pso.state.forest.leaf_indices,
        pso.prelkv,
        pso.prelf,
    )
    resid, (leaf_trees, acc, to_prune, lkratio) = lax.scan(
        loop, pso.state.resid, pts, unroll=pso.state.config.sequential_unroll
    )

    state = replace(
        pso.state,
        resid=resid,
        forest=replace(pso.state.forest, leaf_tree=leaf_trees, log_likelihood=lkratio),
    )
    moves = replace(pso.moves, acc=acc, to_prune=to_prune)

    return state, moves


class SeqStageInAllTrees(Module):
    """The inputs to `accept_move_and_sample_leaves` that are shared by all trees."""

    X: UInt[Array, 'p n']
    """The predictors."""

    resid_num_batches: int | None | Literal['auto'] = field(static=True)
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

    # Although consumed one tree at a time by `lax.scan`, this object is only
    # ever constructed in the stacked (batched) form fed to the scan, so
    # `num_trees` stays a fixed (non-variadic) leading axis disambiguated by
    # rank/dtype (cf. `ParallelStageOut`); the per-tree slices reach `loop` via
    # scan, which does not re-run `__init__`.
    leaf_tree: (
        Float32[Array, 'num_trees tree_size'] | Float32[Array, 'num_trees k tree_size']
    )
    """The leaf values of the trees."""

    prec_tree: (
        Float32[Array, 'num_trees tree_size']
        | UInt32[Array, 'num_trees tree_size']
        | Float32[Array, 'num_trees k k tree_size']
    )
    """The likelihood precision scale in each potential or actual leaf node."""

    move: Moves
    """The proposed move, see `propose_moves`."""

    move_precs: Precs | Counts
    """The likelihood precision scale in each node modified by the moves."""

    leaf_indices: UInt[Array, 'num_trees n']
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
    Float32[Array, ' tree_size'] | Float32[Array, ' k tree_size'],
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
    leaf_tree : Float32[Array, 'tree_size'] | Float32[Array, ' k tree_size']
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
    resid_num_batches: int | None | Literal['auto'],
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
    The per-leaf sum, with the same leading dimensions as ``scaled_resid`` and a trailing axis over the leaves.
    """
    return _scatter_add(
        scaled_resid,
        leaf_indices,
        tree_size,
        jnp.float32,
        resid_num_batches,
        'resid',
        data_sharded,
    )


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
def accept_moves_final_stage(state: State, moves: Moves) -> State:
    """
    Post-process the mcmc state after accepting/rejecting the moves.

    This function is separate from `accept_moves_sequential_stage` to signal it
    can work in parallel across trees.

    Parameters
    ----------
    state
        A partially updated BART mcmc state.
    moves
        The proposed moves (see `propose_moves`) as updated by
        `accept_moves_sequential_stage`.

    Returns
    -------
    The fully updated BART mcmc state.
    """
    return replace(
        state,
        forest=replace(
            state.forest,
            grow_acc_count=jnp.sum(moves.acc & moves.grow),
            prune_acc_count=jnp.sum(moves.acc & ~moves.grow),
            leaf_indices=apply_moves_to_leaf_indices(state.forest.leaf_indices, moves),
            split_tree=apply_moves_to_split_trees(state.forest.split_tree, moves),
            affluence_tree=apply_moves_to_affluence_trees(
                state.forest.affluence_tree, moves
            ),
        ),
    )


@named_call
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
    return _apply_moves_to_leaf_indices(leaf_indices, moves)


@vmap_nodoc
def _apply_moves_to_leaf_indices(
    leaf_indices: UInt[Array, ' n'], moves: Moves
) -> UInt[Array, ' n']:
    """Implement `apply_moves_to_leaf_indices`."""
    mask = ~jnp.array(1, leaf_indices.dtype)  # ...1111111110
    is_child = (leaf_indices & mask) == moves.left
    assert moves.to_prune is not None
    return jnp.where(
        is_child & moves.to_prune, moves.node.astype(leaf_indices.dtype), leaf_indices
    )


@named_call
def apply_moves_to_split_trees(
    split_tree: UInt[Array, 'num_trees half_tree_size'], moves: Moves
) -> UInt[Array, 'num_trees half_tree_size']:
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
    return _apply_moves_to_split_trees(split_tree, moves)


@vmap_nodoc
def _apply_moves_to_split_trees(
    split_tree: UInt[Array, ' half_tree_size'], moves: Moves
) -> UInt[Array, ' half_tree_size']:
    """Implement `apply_moves_to_split_trees`."""
    assert moves.to_prune is not None
    return (
        split_tree.at[jnp.where(moves.grow, moves.node, split_tree.size)]
        .set(moves.grow_split.astype(split_tree.dtype))
        .at[jnp.where(moves.to_prune, moves.node, split_tree.size)]
        .set(0)
    )


@named_call
def apply_moves_to_affluence_trees(
    affluence_tree: Bool[Array, 'num_trees half_tree_size'], moves: Moves
) -> Bool[Array, 'num_trees half_tree_size']:
    """
    Update the affluence trees to match the accepted move.

    The affluence tree marks the growable leaves; this restores that invariant
    after the move by re-marking only the nodes it touched, starting from the
    clean pre-move mask.

    Parameters
    ----------
    affluence_tree
        The mask of the growable leaves in the initial trees.
    moves
        The proposed moves (see `propose_moves`), as updated by
        `accept_moves_sequential_stage`.

    Returns
    -------
    The updated affluence trees.
    """
    return _apply_moves_to_affluence_trees(affluence_tree, moves)


@vmap_nodoc
def _apply_moves_to_affluence_trees(
    affluence_tree: Bool[Array, ' half_tree_size'], moves: Moves
) -> Bool[Array, ' half_tree_size']:
    """Implement `apply_moves_to_affluence_trees`."""
    assert moves.to_prune is not None
    assert moves.node_affluent is not None
    size = affluence_tree.size
    # GROW: node becomes internal, children become leaves with their affluence.
    # PRUNE (accepted prune or rejected grow): node becomes a leaf, children are
    # deleted. Indices of the move not taken resolve to `size` and drop.
    return (
        affluence_tree.at[jnp.where(moves.grow, moves.node, size)]
        .set(False)
        .at[jnp.where(moves.grow, moves.left, size)]
        .set(moves.left_affluent)
        .at[jnp.where(moves.grow, moves.right, size)]
        .set(moves.right_affluent)
        .at[jnp.where(moves.to_prune, moves.node, size)]
        .set(moves.node_affluent)
        .at[jnp.where(moves.to_prune, moves.left, size)]
        .set(False)
        .at[jnp.where(moves.to_prune, moves.right, size)]
        .set(False)
    )


@jit
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

    # Diagonal elements: A_ii ~ sqrt(chi^2(df - i)), with chi^2(k) = Gamma(k/2, scale=2).
    # sqrt(2 * Gamma) = sqrt(2) * exp(loggamma / 2), folding the sqrt into the exp.
    k, _ = scale_inv.shape
    df_vector = df - jnp.arange(k)
    diag_A = jnp.sqrt(2.0) * jnp.exp(loggamma(keys.pop(), df_vector / 2.0) / 2.0)

    off_diag_A = random.normal(keys.pop(), (k, k))
    A = jnp.tril(off_diag_A, -1) + jnp.diag(diag_A)
    L = chol_with_gersh(scale_inv, absolute_eps=True)
    T = solve_triangular(L, A, lower=True, trans='T')

    return T @ T.T


def _step_error_cov_inv_mv(key: Key[Array, ''], state: State) -> State:
    assert state.error_cov_df is not None
    assert state.error_cov_scale is not None

    resid = state.resid
    if state.inv_sdev_scale is None:
        _, n_eff = resid.shape
        n_eff *= get_axis_size(state.config.mesh, 'data')
    else:
        # 2-D inv_sdev_scale dispatches to the diagonal path, so here it is 1-D
        n_eff = jnp.sum(state.inv_sdev_scale != 0, axis=-1)
        if state.config.data_sharded:
            n_eff = lax.psum(n_eff, 'data')
        resid *= state.inv_sdev_scale
    df_post = state.error_cov_df + n_eff
    rrt = resid @ resid.T
    if state.config.data_sharded:
        rrt = lax.psum(rrt, 'data')
    scale_post = state.error_cov_scale + rrt

    prec = _sample_wishart_bartlett(key, df_post, scale_post)
    return replace(state, error_cov_inv=prec)


def _step_error_cov_inv_diag(key: Key[Array, ''], state: State) -> State:
    """Per-component inverse-gamma update for univariate, mixed, and partial-missing paths."""
    assert state.error_cov_scale is not None
    assert state.error_cov_df is not None

    resid = state.resid
    if state.inv_sdev_scale is not None:
        resid *= state.inv_sdev_scale

    # alpha
    if state.inv_sdev_scale is None:
        *_, n_eff = resid.shape
        n_eff *= get_axis_size(state.config.mesh, 'data')
    else:
        n_eff = jnp.sum(state.inv_sdev_scale != 0, axis=-1)
        if state.config.data_sharded:
            n_eff = lax.psum(n_eff, 'data')
    alpha = state.error_cov_df / 2 + n_eff / 2

    # beta
    norm2 = jnp.einsum('...n,...n->...', resid, resid)
    if state.config.data_sharded:
        norm2 = lax.psum(norm2, 'data')
    scale = state.error_cov_scale
    kshape = resid.shape[:-1]
    if kshape:
        scale = jnp.diag(scale)
    beta = scale / 2 + norm2 / 2

    # draw the gamma from the first of a split, mirroring the Bartlett sampler
    # in the multivariate path so the two branches coincide at k=1
    keys = split(key)
    samples = jnp.exp(loggamma(keys.pop(), alpha, kshape))
    prec = samples / beta
    if state.binary_indices is not None:
        prec = prec.at[state.binary_indices].set(1.0)
    if kshape:
        prec = jnp.diag(prec)
    return replace(state, error_cov_inv=prec)


@named_call
def step_error_cov_inv(key: Key[Array, ''], state: State) -> State:
    """MCMC-update the inverse error covariance."""
    if (
        state.error_cov_inv.ndim == 2
        and state.binary_indices is None
        and (state.inv_sdev_scale is None or state.inv_sdev_scale.ndim == 1)
    ):
        return _step_error_cov_inv_mv(key, state)
    else:
        return _step_error_cov_inv_diag(key, state)


@named_call
def step_z(key: Key[Array, ''], state: State) -> State:
    """
    MCMC-update the latent variable for binary regression.

    Parameters
    ----------
    key
        A jax random key.
    state
        A BART MCMC state.

    Returns
    -------
    The updated BART MCMC state.
    """
    assert state.z is not None
    assert state.binary_y is not None

    if state.binary_indices is not None:
        resid = state.resid[..., state.binary_indices, :]
    else:
        resid = state.resid

    trees_plus_offset = state.z - resid
    if state.config.data_sharded:
        # decorrelate the seed across data shards; the seed is replicated
        # because the trees and most of the algorithm are replicated
        key = random.fold_in(key, lax.axis_index('data'))
    resid = truncated_normal_onesided(key, (), ~state.binary_y, -trees_plus_offset)
    z = trees_plus_offset + resid

    if state.binary_indices is not None:
        resid = state.resid.at[..., state.binary_indices, :].set(resid)

    return replace(state, z=z, resid=resid)


@named_call
def step_s(key: Key[Array, ''], state: State) -> State:
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
    state
        The current BART state.

    Returns
    -------
    Updated BART state with re-sampled `log_s`.

    Notes
    -----
    This full conditional is approximated, because it does not take into account
    that there are forbidden decision rules.
    """
    assert state.forest.theta is not None

    # histogram current variable usage
    p = state.forest.max_split.size
    varcount = var_histogram(
        p, state.forest.var_tree, state.forest.split_tree, sum_batch_axis=-1
    )

    # sample from Dirichlet posterior
    alpha = state.forest.theta / p + varcount
    log_s = loggamma(key, alpha)

    # update forest with new s
    return replace(state, forest=replace(state.forest, log_s=log_s))


@named_call
def step_theta(key: Key[Array, ''], state: State, *, num_grid: int = 1000) -> State:
    """
    Update `theta`.

    The prior is theta / (theta + rho) ~ Beta(a, b).

    Parameters
    ----------
    key
        Random key for sampling.
    state
        The current BART state.
    num_grid
        The number of points in the evenly-spaced grid used to sample
        theta / (theta + rho).

    Returns
    -------
    Updated BART state with re-sampled `theta`.
    """
    assert state.forest.log_s is not None
    assert state.forest.rho is not None
    assert state.forest.a is not None
    assert state.forest.b is not None

    # the grid points are the midpoints of num_grid bins in (0, 1)
    padding = 1 / (2 * num_grid)
    lambda_grid = jnp.linspace(padding, 1 - padding, num_grid)

    # normalize s
    log_s = state.forest.log_s - logsumexp(state.forest.log_s)

    # sample lambda
    logp, theta_grid = _log_p_lambda(
        lambda_grid, log_s, state.forest.rho, state.forest.a, state.forest.b
    )
    i = random.categorical(key, logp)
    theta = theta_grid[i]

    return replace(state, forest=replace(state.forest, theta=theta))


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
def step_sparse(key: Key[Array, ''], state: State) -> State:
    """
    Update the sparsity parameters.

    This invokes `step_s`, and then `step_theta` only if the parameters of
    the theta prior are defined.

    Parameters
    ----------
    key
        Random key for sampling.
    state
        The current BART state.

    Returns
    -------
    Updated BART state with re-sampled `log_s` and `theta`.
    """
    if state.config.sparse_on_at is not None:
        state = lax.cond(
            state.config.steps_done < state.config.sparse_on_at,
            lambda _key, state: state,
            _step_sparse,
            key,
            state,
        )
    return state


def _step_sparse(key: Key[Array, ''], state: State) -> State:
    keys = split(key)
    state = step_s(keys.pop(), state)
    if state.forest.rho is not None:
        state = step_theta(keys.pop(), state)
    return state


@named_call
def step_config(state: State) -> State:
    config = state.config
    config = replace(config, steps_done=config.steps_done + 1)
    return replace(state, config=config)
