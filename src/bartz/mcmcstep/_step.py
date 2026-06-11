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
from typing import overload

from equinox import AbstractVar
from jax import lax, named_call, random, vmap
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln, logsumexp
from jaxtyping import Array, Bool, Float, Float32, Int32, Key, Shaped, UInt, UInt32

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
from bartz.mcmcstep._reduction import ReductionConfig
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

    lrt: UInt[Array, '*num_trees 3']
    """Number of datapoints in the left child, right child, and parent
    (``= left + right``), stacked along the trailing axis."""


class PreLkV(Module):
    """Non-sequential terms of the likelihood ratio for each tree.

    These terms are derived from the leaf precompute terms (`PreLf`) gathered
    at the nodes involved in each move. The terms for the left child, right
    child, and their join (the parent node) are stacked along the axis right
    after the tree axis. Each term is, in the univariate case, the scalar

        ``error_cov_inv^2 / (leaf_prior_cov_inv + n * error_cov_inv)``.

    In the multivariate homoskedastic or scalar weight case, this is the matrix term

        ``error_cov_inv @ inv(leaf_prior_cov_inv + n * error_cov_inv) @ error_cov_inv``.

    In the multivariate vector-weight case, this is instead

        ``chol(leaf_prior_cov_inv + n * error_cov_inv)``

    ``n`` is the number of datapoints in the node, or the likelihood precision
    scale in the heteroskedastic case.
    """

    # `log_sqrt_term` is declared before `lrt` so its single (union-free)
    # annotation binds the variadic `*num_trees` axis first; otherwise the
    # runtime typechecker can greedily mis-bind `*num_trees` against the `k`
    # axis of the `... | ... k k` union (the multivariate and univariate
    # layouts are rank-ambiguous).
    log_sqrt_term: Float32[Array, '*num_trees']
    """The logarithm of the square root term of the likelihood ratio."""

    lrt: Float32[Array, '*num_trees 3'] | Float32[Array, '*num_trees 3 k k']
    """Scaled full conditional variance, scaled covariance, or precision
    cholesky, for the left child, right child, and their join."""


class PreLf(Module):
    """Pre-computed terms used to sample leaves from their posterior.

    These terms can be computed in parallel across trees.

    For each tree and leaf, the terms are scalars in the univariate case
    (`PreLfUV`), and matrices/vectors in the multivariate case (`PreLfMV`,
    `PreLfMVHet`).

    Abstract base: the layouts differ in rank, so they live in concrete
    subclasses with union-free annotations; a single class carrying a shape
    union would make the greedy variadic mis-bind against the ``k`` axes under
    the runtime typechecker. The concrete class also tags the meaning of
    `mean_factor`, which drives the dispatch in `precompute_likelihood_terms`
    and in the sequential stage. The ``num_trees`` axis is variadic so the same
    annotations also match a per-element layout if vmapped over trees.
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
    """``error_cov_inv / prec``, where ``prec`` is the posterior precision of
    the leaf."""

    centered_leaves: Float32[Array, '*num_trees tree_size']
    """Zero-mean normal draws with the posterior variance of each leaf."""


class PreLfMV(PreLf):
    """`PreLf` for the multivariate homoskedastic or scalar-weight case."""

    mean_factor: Float32[Array, '*num_trees k k tree_size']
    """``error_cov_inv @ inv(prec)``, where ``prec`` is the posterior precision
    of the leaf."""

    centered_leaves: Float32[Array, '*num_trees k tree_size']
    """Zero-mean normal draws with the posterior covariance of each leaf."""

    logdet_prec: Float32[Array, '*num_trees tree_size']
    """The log-determinant of the posterior precision of each leaf."""


class PreLfMVHet(PreLf):
    """`PreLf` for the multivariate vector-weight case."""

    mean_factor: Float32[Array, '*num_trees k k tree_size']
    """The lower Cholesky factor of the posterior precision of each leaf; the
    mean solve happens downstream in the sequential stage."""

    centered_leaves: Float32[Array, '*num_trees k tree_size']
    """Zero-mean normal draws with the posterior covariance of each leaf."""


class ParallelStageOut(Module):
    """The output of `accept_moves_parallel_stage`."""

    state: State
    """A partially updated BART mcmc state."""

    moves: Moves
    """The proposed moves, with `partial_ratio` set to `None` and
    `log_trans_prior_ratio` set to its final value."""

    # `num_trees` stays a fixed (non-variadic) axis: `ParallelStageOut` is always
    # built with the tree axis present (never per tree under vmap), so the union
    # is disambiguated by rank/dtype and needs no anchor (cf. `PreLf`).
    prec_trees: (
        Float32[Array, 'num_trees tree_size']
        | UInt32[Array, 'num_trees tree_size']
        | Float32[Array, 'num_trees k k tree_size']
    )
    """The likelihood precision scale in each potential or actual leaf node."""

    prelkv: PreLkV
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

    # update the cached number of datapoints per leaf at the nodes involved
    # in the moves
    if (
        state.forest.min_points_per_decision_node is not None
        or state.forest.min_points_per_leaf is not None
        or state.prec_scale is None
    ):
        assert state.forest.count_tree is not None
        count_trees, move_counts = compute_count_trees(
            state.forest.count_tree, state.forest.leaf_indices, moves, state.config
        )
        state = replace(state, forest=replace(state.forest, count_tree=count_trees))

    # affluence of the nodes touched by each move: whether they would be
    # growable as leaves (admissible rule + enough datapoints). The children
    # must also lie within the heap, i.e. not be at the bottom level; the
    # parent always does. These feed the transition ratio and the final
    # `affluence_tree` update.
    _, half = state.forest.var_tree.shape
    lrt_affluent = (moves.lrt_nodes < half) & moves.lrt_growable
    if state.forest.min_points_per_decision_node is not None:
        lrt_affluent &= move_counts.lrt >= state.forest.min_points_per_decision_node
    moves = replace(moves, lrt_affluent=lrt_affluent)

    # veto grove move if new leaves don't have enough datapoints
    if state.forest.min_points_per_leaf is not None:
        moves = replace(
            moves,
            allowed=moves.allowed
            & jnp.all(
                move_counts.lrt[..., :2] >= state.forest.min_points_per_leaf, axis=-1
            ),
        )

    # update the cached number of datapoints per leaf, weighted by error
    # precision scale, at the nodes involved in the moves
    if state.prec_scale is None:
        prec_trees = count_trees
    else:
        assert state.forest.prec_tree is not None
        prec_trees = compute_prec_trees(
            state.forest.prec_tree,
            state.prec_scale,
            state.forest.leaf_indices,
            moves,
            state.config,
        )
        state = replace(state, forest=replace(state.forest, prec_tree=prec_trees))

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

    prelf = precompute_leaf_terms(
        key, prec_trees, state.error_cov_inv, state.forest.leaf_prior_cov_inv
    )
    prelkv = precompute_likelihood_terms(
        state.error_cov_inv, state.forest.leaf_prior_cov_inv, prelf, moves
    )

    return ParallelStageOut(
        state=state, moves=moves, prec_trees=prec_trees, prelkv=prelkv, prelf=prelf
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
    left_child = moves.lrt_nodes[0].astype(leaf_indices.dtype)
    x: UInt[Array, ' n'] = X[moves.grow_var, :]
    go_right = x >= moves.grow_split
    tree_size = jnp.array(2 * moves.var_tree.size)
    node_to_update = jnp.where(moves.grow, moves.lrt_nodes[2], tree_size)
    return jnp.where(
        leaf_indices == node_to_update, left_child + go_right, leaf_indices
    )


def _fill_lrt_total(lrt: Shaped[Array, '*k_k 3']) -> Shaped[Array, '*k_k 3']:
    """Set the total slot of stacked (left, right, total) values to left + right.

    The left and right slots pass through unchanged, the stale value in the
    total slot is ignored. Implemented with fusable elementwise operations.
    """
    total = lrt[..., 0] + lrt[..., 1]
    return jnp.where(jnp.arange(3) == 2, total[..., None], lrt)


@overload
def _compute_count_or_prec_trees(
    prec_scale: None,
    trees: UInt32[Array, 'num_trees tree_size'],
    leaf_indices: UInt[Array, 'num_trees n'],
    moves: Moves,
    config: StepConfig,
) -> tuple[UInt32[Array, 'num_trees tree_size'], Counts]: ...


@overload
def _compute_count_or_prec_trees(
    prec_scale: Float[Array, ' n'] | Float[Array, 'k k n'],
    trees: Float32[Array, 'num_trees tree_size']
    | Float32[Array, 'num_trees k k tree_size'],
    leaf_indices: UInt[Array, 'num_trees n'],
    moves: Moves,
    config: StepConfig,
) -> (
    tuple[Float32[Array, 'num_trees tree_size'], None]
    | tuple[Float32[Array, 'num_trees k k tree_size'], None]
): ...


def _compute_count_or_prec_trees(
    prec_scale: Float[Array, ' n'] | Float[Array, 'k k n'] | None,
    trees: UInt32[Array, 'num_trees tree_size']
    | Float32[Array, 'num_trees tree_size']
    | Float32[Array, 'num_trees k k tree_size'],
    leaf_indices: UInt[Array, 'num_trees n'],
    moves: Moves,
    config: StepConfig,
) -> (
    tuple[UInt32[Array, 'num_trees tree_size'], Counts]
    | tuple[Float32[Array, 'num_trees tree_size'], None]
    | tuple[Float32[Array, 'num_trees k k tree_size'], None]
):
    """Implement `compute_count_trees` and `compute_prec_trees`."""
    if config.prec_count_num_trees is None:
        compute = vmap(_compute_count_or_prec_tree, in_axes=(None, 0, 0, 0, None))
        return compute(prec_scale, trees, leaf_indices, moves, config)

    def compute(
        args: tuple[
            UInt32[Array, ' tree_size']
            | Float32[Array, ' tree_size']
            | Float32[Array, 'k k tree_size'],
            UInt[Array, ' n'],
            Moves,
        ],
    ) -> (
        tuple[UInt32[Array, ' tree_size'], Counts]
        | tuple[Float32[Array, ' tree_size'], None]
        | tuple[Float32[Array, 'k k tree_size'], None]
    ):
        tree, leaf_indices, moves = args
        return _compute_count_or_prec_tree(
            prec_scale, tree, leaf_indices, moves, config
        )

    return lax.map(
        compute, (trees, leaf_indices, moves), batch_size=config.prec_count_num_trees
    )


def _compute_count_or_prec_tree(
    prec_scale: Float[Array, ' n'] | Float[Array, 'k k n'] | None,
    tree: UInt32[Array, ' tree_size']
    | Float32[Array, ' tree_size']
    | Float32[Array, 'k k tree_size'],
    leaf_indices: UInt[Array, ' n'],
    moves: Moves,
    config: StepConfig,
) -> (
    tuple[UInt32[Array, ' tree_size'], Counts]
    | tuple[Float32[Array, ' tree_size'], None]
    | tuple[Float32[Array, 'k k tree_size'], None]
):
    """Update the cached count or precision tree for a single tree."""
    (tree_size,) = moves.var_tree.shape
    tree_size *= 2

    if prec_scale is None:
        value = 1
        dtype = jnp.uint32
        reduction_config = config.count_reduction_config
    else:
        value = prec_scale
        dtype = jnp.float32
        reduction_config = config.prec_reduction_config

    # the cached tree is valid at the leaves, and the move only changes the
    # values at the nodes it involves, so reduce into the move's children alone:
    # the contiguous pair (left, right) = (2 * node, 2 * node + 1) = lrt_nodes[:2]
    lr = reduction_config._reduce(  # noqa: SLF001
        value,
        leaf_indices,
        size=tree_size,
        subset_start=moves.lrt_nodes[0],
        subset_length=2,
        dtype=dtype,
        data_sharded=config.data_sharded,
    )

    # write the children sums into the cache along with their total at the
    # parent node (a non-leaf in the post-grow indexing the reduce runs on);
    # the weighted version of the counts is not needed because the likelihood
    # terms are derived from the leaf terms
    total = lr[..., 0] + lr[..., 1]
    lrt = jnp.concatenate([lr, total[..., None]], axis=-1)
    tree = tree.at[..., moves.lrt_nodes].set(lrt)

    if prec_scale is None:
        return tree, Counts(lrt=lrt)
    else:
        return tree, None


@named_call
def compute_count_trees(
    count_trees: UInt32[Array, 'num_trees tree_size'],
    leaf_indices: UInt[Array, 'num_trees n'],
    moves: Moves,
    config: StepConfig,
) -> tuple[UInt32[Array, 'num_trees tree_size'], Counts]:
    """
    Update the cached number of datapoints per leaf at the moves' nodes.

    Parameters
    ----------
    count_trees
        The cached number of points in each leaf; valid at the leaves of the
        pre-move trees.
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
        The updated cache, valid in each potential or actual leaf node.
    counts : Counts
        The counts of the number of points in the leaves grown or pruned by the
        moves.
    """
    return _compute_count_or_prec_trees(None, count_trees, leaf_indices, moves, config)


@named_call
def compute_prec_trees(
    prec_trees: Float32[Array, 'num_trees tree_size']
    | Float32[Array, 'num_trees k k tree_size'],
    prec_scale: Float[Array, ' n'] | Float[Array, 'k k n'],
    leaf_indices: UInt[Array, 'num_trees n'],
    moves: Moves,
    config: StepConfig,
) -> Float32[Array, 'num_trees tree_size'] | Float32[Array, 'num_trees k k tree_size']:
    """
    Update the cached per-leaf likelihood precision scale at the moves' nodes.

    Parameters
    ----------
    prec_trees
        The cached likelihood precision scale in each leaf; valid at the leaves
        of the pre-move trees.
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
    The updated cache, valid in each potential or actual leaf node.
    """
    trees, _ = _compute_count_or_prec_trees(
        prec_scale, prec_trees, leaf_indices, moves, config
    )
    return trees


@partial(vmap_nodoc, in_axes=(0, None))
def complete_ratio(moves: Moves, p_nonterminal: Float32[Array, ' tree_size']) -> Moves:
    """
    Complete non-likelihood MH ratio calculation.

    This function adds the probability of choosing a prune move over the grow
    move in the inverse transition, and the prior odds that the modified node
    is nonterminal with terminal children.

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
    assert moves.lrt_affluent is not None

    # can the children be grown by the proposal? `lrt_affluent` already folds
    # in the `min_points_per_decision_node` threshold, because the grow
    # proposal draws from the pool of leaves that pass it. This enters only the
    # transition probability.

    # p_prune if grow
    other_growable_leaves = moves.num_growable >= 2
    grow_again_allowed = other_growable_leaves | jnp.any(moves.lrt_affluent[:2])
    grow_p_prune = jnp.where(grow_again_allowed, 0.5, 1.0)

    # p_prune if prune
    prune_p_prune = jnp.where(moves.num_growable, 0.5, 1)

    # select p_prune
    p_prune = jnp.where(moves.grow, grow_p_prune, prune_p_prune)

    # prior odds of the node being nonterminal, times the prior probability of
    # both children being terminal. The children terminality uses the
    # admissibility ignoring counts, because the standard BART prior conditions
    # the non-terminal probability only on the existence of available decision
    # rules, not on the count thresholds (which are a bartz proposal-efficiency
    # device, not part of the target distribution). The fill value avoids a 0
    # and then an inf in the log if the move is not allowed and the indices are
    # out of bounds.
    pnt = p_nonterminal.at[moves.lrt_nodes].get(mode='fill', fill_value=0.5)
    prior_ratio = pnt[2] / (1 - pnt[2]) * jnp.prod(1 - pnt[:2] * moves.lrt_growable[:2])

    assert moves.partial_ratio is not None
    return replace(
        moves,
        log_trans_prior_ratio=jnp.log(moves.partial_ratio * prior_ratio * p_prune),
        partial_ratio=None,
    )


@named_call
def adapt_leaf_trees_to_grow_indices(
    leaf_trees: Float[Array, 'num_trees tree_size']
    | Float[Array, 'num_trees k tree_size'],
    moves: Moves,
) -> Float[Array, 'num_trees tree_size'] | Float[Array, 'num_trees k tree_size']:
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
    leaf_trees: Float[Array, ' tree_size'] | Float[Array, ' k tree_size'], moves: Moves
) -> Float[Array, ' tree_size'] | Float[Array, ' k tree_size']:
    """Implement `adapt_leaf_trees_to_grow_indices`."""
    # the parent slot is written back unchanged to share a single scatter
    values_at_node = leaf_trees[..., moves.lrt_nodes[2]]
    return leaf_trees.at[
        ..., jnp.where(moves.grow, moves.lrt_nodes, leaf_trees.size)
    ].set(values_at_node[..., None])


def _logdet_from_chol(L: Float32[Array, '... k k']) -> Float32[Array, '...']:
    """Compute logdet of A = LL' via Cholesky (sum of log of diag^2)."""
    diags: Float32[Array, '... k'] = jnp.diagonal(L, axis1=-2, axis2=-1)
    return 2.0 * jnp.sum(jnp.log(diags), axis=-1)


def compute_B(
    error_cov_inv: Float32[Array, 'k k'], resid: Float32[Array, 'k k *tree_size']
) -> Float32[Array, ' k *tree_size']:
    """Compute the leaf score from the leaf weighted sum of residuals."""
    return jnp.einsum('ab,ab...->a...', error_cov_inv, resid)


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
    z: Float32[Array, 'num_trees k tree_size'] | None = None,
) -> PreLfMV:
    num_trees, tree_size = prec_trees.shape
    k, _ = error_cov_inv.shape
    if z is None:
        z = random.normal(key, (num_trees, k, tree_size))

    def per_leaf(
        prec: Float32[Array, ''] | UInt32[Array, ''], z: Float32[Array, ' k']
    ) -> tuple[Float32[Array, 'k k'], Float32[Array, ' k'], Float32[Array, '']]:
        L_prec = chol_with_gersh(leaf_prior_cov_inv + prec * error_cov_inv)
        Y = solve_triangular(L_prec, error_cov_inv, lower=True)
        mean_factor = solve_triangular(L_prec, Y, trans='T', lower=True).mT
        centered = solve_triangular(L_prec, z[:, None], trans='T', lower=True).squeeze(
            -1
        )
        # only a few leaves per tree end up using their logdet, but reducing
        # right away is lighter on memory than storing diagonals for later
        return mean_factor, centered, _logdet_from_chol(L_prec)

    # vmap over trees then over leaves; the leaf axis is trailing in both
    # `prec_trees`/`z` (in_axes) and the stored output (out_axes=-1)
    return PreLfMV(*vmap(vmap(per_leaf, in_axes=(0, -1), out_axes=-1))(prec_trees, z))


def _precompute_leaf_terms_mv_het(
    key: Key[Array, ''],
    prec_trees: Float32[Array, 'num_trees k k tree_size'],
    error_cov_inv: Float32[Array, 'k k'],
    leaf_prior_cov_inv: Float32[Array, 'k k'],
    z: Float32[Array, 'num_trees k tree_size'] | None = None,
) -> PreLfMVHet:
    num_trees, k, _, tree_size = prec_trees.shape
    if z is None:
        z = random.normal(key, (num_trees, k, tree_size))

    def per_leaf(
        prec: Float32[Array, 'k k'], z: Float32[Array, ' k']
    ) -> tuple[Float32[Array, 'k k'], Float32[Array, ' k']]:
        # mean_factor stores the precision cholesky itself; the mean solve happens
        # downstream in `accept_move_and_sample_leaves`
        L_prec = chol_with_gersh(leaf_prior_cov_inv + error_cov_inv * prec)
        centered = solve_triangular(L_prec, z[:, None], trans='T', lower=True).squeeze(
            -1
        )
        return L_prec, centered

    # vmap over trees then over leaves; the leaf axis is trailing in both
    # `prec_trees`/`z` (in_axes=-1) and the stored output (out_axes=-1)
    return PreLfMVHet(
        *vmap(vmap(per_leaf, in_axes=(-1, -1), out_axes=-1))(prec_trees, z)
    )


@named_call
def precompute_leaf_terms(
    key: Key[Array, ''],
    prec_trees: Float32[Array, 'num_trees tree_size']
    | UInt32[Array, 'num_trees tree_size']
    | Float32[Array, 'num_trees k k tree_size'],
    error_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'],
    leaf_prior_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'],
    z: Float32[Array, 'num_trees tree_size']
    | Float32[Array, 'num_trees k tree_size']
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


@vmap_nodoc
def _gather_lrt(
    leaf_values: Float32[Array, '*k_k tree_size'], lrt_nodes: Int32[Array, ' 3']
) -> Float32[Array, ' 3 *k_k']:
    """Gather per-tree leaf values at the left child, right child, and parent."""
    return jnp.moveaxis(leaf_values[..., lrt_nodes], -1, 0)


def _precompute_likelihood_terms_uv(
    error_cov_inv: Float32[Array, ''],
    leaf_prior_cov_inv: Float32[Array, ''],
    prelf: PreLfUV,
    lrt_nodes: Int32[Array, 'num_trees 3'],
) -> PreLkV:
    # mean_factor is error_cov_inv / prec, complete the sandwich
    lrt = error_cov_inv * _gather_lrt(prelf.mean_factor, lrt_nodes)
    # the same value with the prior-only precision, computed with the same
    # operations as in `_precompute_leaf_terms_uv` such that it matches `lrt`
    # bitwise on empty nodes and the ratio is exactly 1 without data
    prior_lrt = error_cov_inv * (jnp.reciprocal(leaf_prior_cov_inv) * error_cov_inv)
    log_sqrt_term = jnp.log(lrt[..., 0] * lrt[..., 1] / (prior_lrt * lrt[..., 2])) / 2
    return PreLkV(lrt=lrt, log_sqrt_term=log_sqrt_term)


def _precompute_likelihood_terms_mv(
    error_cov_inv: Float32[Array, 'k k'],
    leaf_prior_cov_inv: Float32[Array, 'k k'],
    prelf: PreLfMV,
    lrt_nodes: Int32[Array, 'num_trees 3'],
) -> PreLkV:
    logdet_prior = _logdet_from_chol(chol_with_gersh(leaf_prior_cov_inv))
    logdet_prec = _gather_lrt(prelf.logdet_prec, lrt_nodes)
    log_sqrt_term = (logdet_prior + logdet_prec @ jnp.array([-1.0, -1.0, 1.0])) / 2

    # mean_factor is error_cov_inv @ inv(prec), complete the sandwich
    mean_factor = _gather_lrt(prelf.mean_factor, lrt_nodes)  # (num_trees, 3, k, k)
    return PreLkV(lrt=mean_factor @ error_cov_inv, log_sqrt_term=log_sqrt_term)


def _precompute_likelihood_terms_mv_het(
    leaf_prior_cov_inv: Float32[Array, 'k k'],
    prelf: PreLfMVHet,
    lrt_nodes: Int32[Array, 'num_trees 3'],
) -> PreLkV:
    logdet_prior = _logdet_from_chol(chol_with_gersh(leaf_prior_cov_inv))

    # mean_factor is the precision cholesky itself
    L = _gather_lrt(prelf.mean_factor, lrt_nodes)  # (num_trees, 3, k, k)
    log_sqrt_term = (
        logdet_prior + _logdet_from_chol(L) @ jnp.array([-1.0, -1.0, 1.0])
    ) / 2
    return PreLkV(lrt=L, log_sqrt_term=log_sqrt_term)


@named_call
def precompute_likelihood_terms(
    error_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'],
    leaf_prior_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'],
    prelf: PreLf,
    moves: Moves,
) -> PreLkV:
    """
    Pre-compute terms used in the likelihood ratio of the acceptance step.

    The likelihood ratio terms are mostly a subset of the leaf sampling terms,
    so they are derived from `prelf`, gathered at the nodes involved in the
    moves.

    Parameters
    ----------
    error_cov_inv
        The inverse error variance (univariate) or the inverse of the error
        covariance matrix (multivariate). This is the inverse global error
        variance factor if `prec_scale` is set.
    leaf_prior_cov_inv
        The inverse prior variance of each leaf (univariate) or the inverse of
        prior covariance matrix of each leaf (multivariate).
    prelf
        The pre-computed terms of the leaf sampling, see `precompute_leaf_terms`.
    moves
        The proposed moves, see `propose_moves`.

    Returns
    -------
    Pre-computed terms of the likelihood ratio, one per tree.
    """
    if isinstance(prelf, PreLfUV):
        return _precompute_likelihood_terms_uv(
            error_cov_inv, leaf_prior_cov_inv, prelf, moves.lrt_nodes
        )
    elif isinstance(prelf, PreLfMVHet):
        return _precompute_likelihood_terms_mv_het(
            leaf_prior_cov_inv, prelf, moves.lrt_nodes
        )
    else:
        assert isinstance(prelf, PreLfMV)
        return _precompute_likelihood_terms_mv(
            error_cov_inv, leaf_prior_cov_inv, prelf, moves.lrt_nodes
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
            Float[Array, ' tree_size'] | Float[Array, ' k tree_size'],
            Bool[Array, ''],
            Bool[Array, ''],
            Float32[Array, ''] | None,
        ],
    ]:
        resid, leaf_tree, acc, to_prune, lkratio = accept_move_and_sample_leaves(
            resid,
            SeqStageInAllTrees(
                pso.state.X,
                pso.state.config.resid_reduction_config,
                pso.state.config.data_sharded,
                pso.state.prec_scale,
                pso.state.forest.log_likelihood is not None,
                pso.state.error_cov_inv if isinstance(pso.prelf, PreLfMVHet) else None,
                pso.state.forest.leaf_scale,
            ),
            pt,
        )
        return resid, (leaf_tree, acc, to_prune, lkratio)

    pts = SeqStageInPerTree(
        pso.state.forest.leaf_tree,
        pso.prec_trees,
        pso.moves,
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

    resid_reduction_config: ReductionConfig
    """How to sum the residuals in each leaf."""

    data_sharded: bool = field(static=True)
    """Whether the data axis is sharded across devices."""

    prec_scale: Float[Array, ' n'] | Float[Array, 'k k n'] | None
    """The scale of the precision of the error on each datapoint. If None, it
    is assumed to be 1."""

    save_ratios: bool = field(static=True)
    """Whether to save the acceptance ratios."""

    error_cov_inv: Float32[Array, 'k k'] | None
    """The global error precision scale. Set only in the multivariate
    vector-weight case, where the sequential stage needs it to compute the
    leaf scores."""

    leaf_scale: Float32[Array, ''] | Float32[Array, ' k']
    """The scale of the stored leaf values, see `bartz.mcmcstep.Forest.leaf_scale`."""


class SeqStageInPerTree(Module):
    """The inputs to `accept_move_and_sample_leaves` that are separate for each tree."""

    # Although consumed one tree at a time by `lax.scan`, this object is only
    # ever constructed in the stacked (batched) form fed to the scan, so
    # `num_trees` stays a fixed (non-variadic) leading axis disambiguated by
    # rank/dtype (cf. `ParallelStageOut`); the per-tree slices reach `loop` via
    # scan, which does not re-run `__init__`.
    leaf_tree: (
        Float[Array, 'num_trees tree_size'] | Float[Array, 'num_trees k tree_size']
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
    Float[Array, ' tree_size'] | Float[Array, ' k tree_size'],
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
    leaf_tree : Float[Array, 'tree_size'] | Float[Array, ' k tree_size']
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
        scaled_resid,
        pt.leaf_indices,
        tree_size,
        at.resid_reduction_config,
        at.data_sharded,
    )

    # convert the starting tree to data units; multiplying by the float32
    # scale also takes care of upcasting narrow leaf dtypes
    prev_leaf_tree = at.leaf_scale[..., None] * pt.leaf_tree

    # subtract starting tree from function
    resid_tree += pt.prec_tree * prev_leaf_tree

    # sum residuals in parent node modified by move and compute likelihood;
    # the children slots are written back unchanged to share a single scatter
    assert pt.move.lrt_nodes.dtype == jnp.int32
    resid_lrt = _fill_lrt_total(resid_tree[..., pt.move.lrt_nodes])
    resid_tree = resid_tree.at[..., pt.move.lrt_nodes].set(resid_lrt)

    log_lk_ratio = compute_likelihood_ratio(resid_lrt, pt.prelkv, at.error_cov_inv)

    # calculate accept/reject ratio
    log_ratio = pt.move.log_trans_prior_ratio + log_lk_ratio
    log_ratio = jnp.where(pt.move.grow, log_ratio, -log_ratio)
    if not at.save_ratios:
        log_lk_ratio = None

    # determine whether to accept the move
    acc = pt.move.allowed & (pt.move.logu <= log_ratio)

    # compute leaves posterior and sample leaves
    if at.error_cov_inv is not None:
        # multivariate w/ vector weights
        b_tree = compute_B(at.error_cov_inv, resid_tree)  # (k, 2**d)
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

    # copy leaves around such that the leaf indices point to the correct leaf;
    # the parent slot is written back unchanged to share a single scatter
    to_prune = acc ^ pt.move.grow
    leaf_tree = leaf_tree.at[
        ..., jnp.where(to_prune, pt.move.lrt_nodes, tree_size)
    ].set(leaf_tree[..., pt.move.lrt_nodes[2], None])

    # round the new leaves to the storage units and dtype; the residuals are
    # then updated with the rounded values to stay consistent with the trees
    leaf_tree = (leaf_tree / at.leaf_scale[..., None]).astype(pt.leaf_tree.dtype)

    # replace old tree with new tree in function values
    resid += (prev_leaf_tree - at.leaf_scale[..., None] * leaf_tree)[
        ..., pt.leaf_indices
    ]

    return resid, leaf_tree, acc, to_prune, log_lk_ratio


@named_call
def sum_resid(
    scaled_resid: (
        Float32[Array, ' n'] | Float32[Array, 'k n'] | Float32[Array, 'k k n']
    ),
    leaf_indices: UInt[Array, ' n'],
    tree_size: int,
    reduction_config: ReductionConfig,
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
    reduction_config
        How to sum the residuals in each leaf.
    data_sharded
        Whether the data axis is sharded; if true, the result is psum-reduced
        across the ``'data'`` axis of the enclosing `shard_map`.

    Returns
    -------
    The per-leaf sum, with the same leading dimensions as ``scaled_resid`` and a trailing axis over the leaves.
    """
    return reduction_config._reduce(  # noqa: SLF001
        scaled_resid,
        leaf_indices,
        size=tree_size,
        dtype=jnp.float32,
        data_sharded=data_sharded,
    )


def _compute_likelihood_ratio_uv(
    resid_lrt: Float32[Array, ' 3'], prelkv: PreLkV
) -> Float32[Array, '']:
    # quadratic form r * v * r for each of the (left, right, total) terms
    qf = resid_lrt * resid_lrt * prelkv.lrt
    exp_term = 0.5 * (qf @ jnp.array([1.0, 1.0, -1.0]))
    return prelkv.log_sqrt_term + exp_term


def _compute_likelihood_ratio_mv(
    resid_lrt: Float32[Array, 'k 3'], prelkv: PreLkV
) -> Float32[Array, '']:
    # quadratic form r' M r for each of the (left, right, total) terms
    qf = jnp.einsum('it,tij,jt->t', resid_lrt, prelkv.lrt, resid_lrt)
    exp_term = 0.5 * (qf @ jnp.array([1.0, 1.0, -1.0]))
    return prelkv.log_sqrt_term + exp_term


def _compute_likelihood_ratio_mv_het(
    resid_lrt: Float32[Array, 'k k 3'],
    error_cov_inv: Float32[Array, 'k k'],
    prelkv: PreLkV,
) -> Float32[Array, '']:
    b = compute_B(error_cov_inv, resid_lrt)  # (k, 3)
    y = solve_triangular(prelkv.lrt, b.T[..., None], lower=True).squeeze(-1)  # (3, k)
    qf = jnp.einsum('ti,ti->t', y, y)
    exp_term = 0.5 * (qf @ jnp.array([1.0, 1.0, -1.0]))
    return prelkv.log_sqrt_term + exp_term


@named_call
def compute_likelihood_ratio(
    resid_lrt: (Float32[Array, ' 3'] | Float32[Array, 'k 3'] | Float32[Array, 'k k 3']),
    prelkv: PreLkV,
    error_cov_inv: Float32[Array, 'k k'] | None,
) -> Float32[Array, '']:
    """
    Compute the likelihood ratio of a grow move.

    Parameters
    ----------
    resid_lrt
        The sum of the residuals (scaled by error precision scale) of the
        datapoints falling in the left child, right child, and parent node
        involved in the move, stacked along the trailing axis.
    prelkv
        The pre-computed terms of the likelihood ratio, see
        `precompute_likelihood_terms`.
    error_cov_inv
        The global error precision scale. Set only in the multivariate
        vector-weight case.

    Returns
    -------
    The log-likelihood ratio log P(data | new tree) - log P(data | old tree).
    """
    if error_cov_inv is not None:
        return _compute_likelihood_ratio_mv_het(resid_lrt, error_cov_inv, prelkv)
    elif resid_lrt.ndim > 1:
        return _compute_likelihood_ratio_mv(resid_lrt, prelkv)
    else:
        return _compute_likelihood_ratio_uv(resid_lrt, prelkv)


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
    assert moves.acc is not None
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
    is_child = (leaf_indices & mask) == moves.lrt_nodes[0]
    assert moves.to_prune is not None
    return jnp.where(
        is_child & moves.to_prune,
        moves.lrt_nodes[2].astype(leaf_indices.dtype),
        leaf_indices,
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
    # a single scatter serves both cases: an accepted grow writes the new
    # cutpoint, while pruning (accepted prune or rejected grow) zeroes the node
    return split_tree.at[
        jnp.where(moves.grow | moves.to_prune, moves.lrt_nodes[2], split_tree.size)
    ].set(jnp.where(moves.to_prune, 0, moves.grow_split).astype(split_tree.dtype))


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
    assert moves.lrt_affluent is not None
    # GROW: node becomes internal, children become leaves with their affluence.
    # PRUNE (accepted prune or rejected grow): node becomes a leaf with its
    # affluence, children are deleted. Either way all three nodes are written:
    # the mask keeps the affluence of the nodes that become leaves and zeroes
    # the rest. If no move is applied (a rejected prune), the indices resolve
    # to `size` and the writes drop.
    becomes_leaf = moves.to_prune ^ jnp.array([True, True, False])
    return affluence_tree.at[
        jnp.where(moves.grow | moves.to_prune, moves.lrt_nodes, affluence_tree.size)
    ].set(moves.lrt_affluent & becomes_leaf)


@jit
def _sample_wishart_bartlett(
    key: Key[Array, ''],
    df: Float32[Array, ''] | float,
    scale_inv: Float32[Array, 'k k'],
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
