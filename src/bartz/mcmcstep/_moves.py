# bartz/src/bartz/mcmcstep/_moves.py
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

"""Implement `propose_moves` and associated dataclasses."""

from functools import partial

import jax
from equinox import Module
from jax import numpy as jnp
from jax import random
from jaxtyping import Array, Bool, Float32, Int32, Integer, Key, UInt

from bartz import grove
from bartz._profiler import jit_and_block_if_profiling
from bartz.jaxext import minimal_unsigned_dtype, split, vmap_nodoc
from bartz.mcmcstep._state import Forest, field, vmap_chains


class Moves(Module):
    """
    Moves proposed to modify each tree.

    Parameters
    ----------
    allowed
        Whether there is a possible move. If `False`, the other values may not
        make sense. The only case in which a move is marked as allowed but is
        then vetoed is if it does not satisfy `min_points_per_leaf`, which for
        efficiency is implemented post-hoc without changing the rest of the
        MCMC logic.
    grow
        Whether the move is a grow move or a prune move.
    num_growable
        The number of growable leaves in the original tree.
    node
        The index of the leaf to grow or node to prune.
    left
    right
        The indices of the children of 'node'.
    partial_ratio
        A factor of the Metropolis-Hastings ratio of the move. It lacks the
        likelihood ratio, the probability of proposing the prune move, and the
        probability that the children of the modified node are terminal. If the
        move is PRUNE, the ratio is inverted. `None` once
        `log_trans_prior_ratio` has been computed.
    log_trans_prior_ratio
        The logarithm of the product of the transition and prior terms of the
        Metropolis-Hastings ratio for the acceptance of the proposed move.
        `None` if not yet computed. If PRUNE, the log-ratio is negated.
    grow_var
        The decision axes of the new rules.
    grow_split
        The decision boundaries of the new rules.
    var_tree
        The updated decision axes of the trees, valid whatever move.
    affluence_tree
        A partially updated `affluence_tree`, marking non-leaf nodes that would
        become leaves if the move was accepted. This mark initially (out of
        `propose_moves`) takes into account if there would be available decision
        rules to grow the leaf, and whether there are enough datapoints in the
        node is instead checked later in `accept_moves_parallel_stage`.
    logu
        The logarithm of a uniform (0, 1] random variable to be used to
        accept the move. It's in (-oo, 0].
    acc
        Whether the move was accepted. `None` if not yet computed.
    to_prune
        Whether the final operation to apply the move is pruning. This indicates
        an accepted prune move or a rejected grow move. `None` if not yet
        computed.
    """

    allowed: Bool[Array, '*chains num_trees'] = field(chains=True)
    grow: Bool[Array, '*chains num_trees'] = field(chains=True)
    num_growable: UInt[Array, '*chains num_trees'] = field(chains=True)
    node: UInt[Array, '*chains num_trees'] = field(chains=True)
    left: UInt[Array, '*chains num_trees'] = field(chains=True)
    right: UInt[Array, '*chains num_trees'] = field(chains=True)
    partial_ratio: Float32[Array, '*chains num_trees'] | None = field(chains=True)
    log_trans_prior_ratio: None | Float32[Array, '*chains num_trees'] = field(
        chains=True
    )
    grow_var: UInt[Array, '*chains num_trees'] = field(chains=True)
    grow_split: UInt[Array, '*chains num_trees'] = field(chains=True)
    var_tree: UInt[Array, '*chains num_trees 2**(d-1)'] = field(chains=True)
    affluence_tree: Bool[Array, '*chains num_trees 2**(d-1)'] = field(chains=True)
    logu: Float32[Array, '*chains num_trees'] = field(chains=True)
    acc: None | Bool[Array, '*chains num_trees'] = field(chains=True)
    to_prune: None | Bool[Array, '*chains num_trees'] = field(chains=True)


@partial(jit_and_block_if_profiling, donate_argnums=(1,))
@vmap_chains
def propose_moves(key: Key[Array, ''], forest: Forest) -> Moves:
    """
    Propose moves for all the trees.

    There are two types of moves: GROW (convert a leaf to a decision node and
    add two leaves beneath it) and PRUNE (convert the parent of two leaves to a
    leaf, deleting its children).

    Parameters
    ----------
    key
        A jax random key.
    forest
        The `forest` field of a BART MCMC state.

    Returns
    -------
    The proposed move for each tree.
    """
    num_trees = forest.leaf_tree.shape[0]
    keys = split(key, 2)
    grow_keys, prune_keys = keys.pop((2, num_trees))

    # compute moves
    grow_moves = propose_grow_moves(
        grow_keys,
        forest.var_tree,
        forest.split_tree,
        forest.affluence_tree,
        forest.max_split,
        forest.blocked_vars,
        forest.p_nonterminal,
        forest.p_propose_grow,
        forest.log_s,
    )
    prune_moves = propose_prune_moves(
        prune_keys,
        forest.split_tree,
        grow_moves.affluence_tree,
        forest.p_nonterminal,
        forest.p_propose_grow,
    )

    u, exp1mlogu = random.uniform(keys.pop(), (2, num_trees))

    # choose between grow or prune
    p_grow = jnp.where(
        grow_moves.allowed & prune_moves.allowed, 0.5, grow_moves.allowed
    )
    grow = u < p_grow  # use < instead of <= because u is in [0, 1)

    # compute children indices
    node = jnp.where(grow, grow_moves.node, prune_moves.node)
    left, right = (node << 1) | jnp.arange(2)[:, None]

    return Moves(
        allowed=grow_moves.allowed | prune_moves.allowed,
        grow=grow,
        num_growable=grow_moves.num_growable,
        node=node,
        left=left,
        right=right,
        partial_ratio=jnp.where(
            grow, grow_moves.partial_ratio, prune_moves.partial_ratio
        ),
        log_trans_prior_ratio=None,  # will be set in complete_ratio
        grow_var=grow_moves.var,
        grow_split=grow_moves.split,
        # var_tree does not need to be updated if prune
        var_tree=grow_moves.var_tree,
        # affluence_tree is updated for both moves unconditionally, prune last
        affluence_tree=prune_moves.affluence_tree,
        logu=jnp.log1p(-exp1mlogu),
        acc=None,  # will be set in accept_moves_sequential_stage
        to_prune=None,  # will be set in accept_moves_sequential_stage
    )


class GrowMoves(Module):
    """
    Represent a proposed grow move for each tree.

    Parameters
    ----------
    allowed
        Whether the move is allowed for proposal.
    num_growable
        The number of leaves that can be proposed for grow.
    node
        The index of the leaf to grow. ``2 ** d`` if there are no growable
        leaves.
    var
    split
        The decision axis and boundary of the new rule.
    partial_ratio
        A factor of the Metropolis-Hastings ratio of the move. It lacks
        the likelihood ratio and the probability of proposing the prune
        move.
    var_tree
        The updated decision axes of the tree.
    affluence_tree
        A partially updated `affluence_tree` that marks each new leaf that
        would be produced as `True` if it would have available decision rules.
    """

    allowed: Bool[Array, ' num_trees']
    num_growable: UInt[Array, ' num_trees']
    node: UInt[Array, ' num_trees']
    var: UInt[Array, ' num_trees']
    split: UInt[Array, ' num_trees']
    partial_ratio: Float32[Array, ' num_trees']
    var_tree: UInt[Array, 'num_trees 2**(d-1)']
    affluence_tree: Bool[Array, 'num_trees 2**(d-1)']


@partial(vmap_nodoc, in_axes=(0, 0, 0, 0, None, None, None, None, None))
def propose_grow_moves(
    key: Key[Array, ' num_trees'],
    var_tree: UInt[Array, 'num_trees 2**(d-1)'],
    split_tree: UInt[Array, 'num_trees 2**(d-1)'],
    affluence_tree: Bool[Array, 'num_trees 2**(d-1)'],
    max_split: UInt[Array, ' p'],
    blocked_vars: Int32[Array, ' k'] | None,
    p_nonterminal: Float32[Array, ' 2**d'],
    p_propose_grow: Float32[Array, ' 2**(d-1)'],
    log_s: Float32[Array, ' p'] | None,
) -> GrowMoves:
    """
    Propose a GROW move for each tree.

    A GROW move picks a leaf node and converts it to a non-terminal node with
    two leaf children.

    Parameters
    ----------
    key
        A jax random key.
    var_tree
        The splitting axes of the tree.
    split_tree
        The splitting points of the tree.
    affluence_tree
        Whether each leaf has enough points to be grown.
    max_split
        The maximum split index for each variable.
    blocked_vars
        The indices of the variables that have no available cutpoints.
    p_nonterminal
        The a priori probability of a node to be nonterminal conditional on the
        ancestors, including at the maximum depth where it should be zero.
    p_propose_grow
        The unnormalized probability of choosing a leaf to grow.
    log_s
        Unnormalized log-probability used to choose a variable to split on
        amongst the available ones.

    Returns
    -------
    An object representing the proposed move.

    Notes
    -----
    The move is not proposed if each leaf is already at maximum depth, or has
    less datapoints than the requested threshold `min_points_per_decision_node`,
    or it does not have any available decision rules given its ancestors. This
    is marked by setting `allowed` to `False` and `num_growable` to 0.
    """
    keys = split(key, 3)

    leaf_to_grow, num_growable, prob_choose, num_prunable = choose_leaf(
        keys.pop(), split_tree, affluence_tree, p_propose_grow
    )

    # sample a decision rule
    var, num_available_var = choose_variable(
        keys.pop(), var_tree, split_tree, max_split, leaf_to_grow, blocked_vars, log_s
    )
    split_idx, l, r = choose_split(
        keys.pop(), var, var_tree, split_tree, max_split, leaf_to_grow
    )

    # determine if the new leaves would have available decision rules; if the
    # move is blocked, these values may not make sense
    leftright_growable = (num_available_var > 1) | jnp.stack(
        [l < split_idx, split_idx + 1 < r]
    )
    leftright = (leaf_to_grow << 1) | jnp.arange(2)
    affluence_tree = affluence_tree.at[leftright].set(leftright_growable)

    ratio = compute_partial_ratio(
        prob_choose, num_prunable, p_nonterminal, leaf_to_grow
    )

    return GrowMoves(
        allowed=num_growable > 0,
        num_growable=num_growable,
        node=leaf_to_grow,
        var=var,
        split=split_idx,
        partial_ratio=ratio,
        var_tree=var_tree.at[leaf_to_grow].set(var.astype(var_tree.dtype)),
        affluence_tree=affluence_tree,
    )


def choose_leaf(
    key: Key[Array, ''],
    split_tree: UInt[Array, ' 2**(d-1)'],
    affluence_tree: Bool[Array, ' 2**(d-1)'],
    p_propose_grow: Float32[Array, ' 2**(d-1)'],
) -> tuple[Int32[Array, ''], Int32[Array, ''], Float32[Array, ''], Int32[Array, '']]:
    """
    Choose a leaf node to grow in a tree.

    Parameters
    ----------
    key
        A jax random key.
    split_tree
        The splitting points of the tree.
    affluence_tree
        Whether a leaf has enough points that it could be split into two leaves
        satisfying the `min_points_per_decision_node` requirement.
    p_propose_grow
        The unnormalized probability of choosing a leaf to grow.

    Returns
    -------
    leaf_to_grow : Int32[Array, '']
        The index of the leaf to grow. If ``num_growable == 0``, return
        ``2 ** d``.
    num_growable : Int32[Array, '']
        The number of leaf nodes that can be grown, i.e., are nonterminal
        and have at least twice `min_points_per_decision_node`.
    prob_choose : Float32[Array, '']
        The (normalized) probability that this function had to choose that
        specific leaf, given the arguments.
    num_prunable : Int32[Array, '']
        The number of leaf parents that could be pruned, after converting the
        selected leaf to a non-terminal node.
    """
    is_growable = growable_leaves(split_tree, affluence_tree)
    num_growable = jnp.count_nonzero(is_growable)
    distr = jnp.where(is_growable, p_propose_grow, 0)
    leaf_to_grow, distr_norm = categorical(key, distr)
    leaf_to_grow = jnp.where(num_growable, leaf_to_grow, 2 * split_tree.size)
    prob_choose = distr[leaf_to_grow] / jnp.where(distr_norm, distr_norm, 1)
    is_parent = grove.is_leaves_parent(split_tree.at[leaf_to_grow].set(1))
    num_prunable = jnp.count_nonzero(is_parent)
    return leaf_to_grow, num_growable, prob_choose, num_prunable


def growable_leaves(
    split_tree: UInt[Array, ' 2**(d-1)'], affluence_tree: Bool[Array, ' 2**(d-1)']
) -> Bool[Array, ' 2**(d-1)']:
    """
    Return a mask indicating the leaf nodes that can be proposed for growth.

    The condition is that a leaf is not at the bottom level, has available
    decision rules given its ancestors, and has at least
    `min_points_per_decision_node` points.

    Parameters
    ----------
    split_tree
        The splitting points of the tree.
    affluence_tree
        Marks leaves that can be grown.

    Returns
    -------
    The mask indicating the leaf nodes that can be proposed to grow.

    Notes
    -----
    This function needs `split_tree` and not just `affluence_tree` because
    `affluence_tree` can be "dirty", i.e., mark unused nodes as `True`.
    """
    return grove.is_actual_leaf(split_tree) & affluence_tree


def categorical(
    key: Key[Array, ''], distr: Float32[Array, ' n']
) -> tuple[Int32[Array, ''], Float32[Array, '']]:
    """
    Return a random integer from an arbitrary distribution.

    Parameters
    ----------
    key
        A jax random key.
    distr
        An unnormalized probability distribution.

    Returns
    -------
    u : Int32[Array, '']
        A random integer in the range ``[0, n)``. If all probabilities are zero,
        return ``n``.
    norm : Float32[Array, '']
        The sum of `distr`.

    Notes
    -----
    This function uses a cumsum instead of the Gumbel trick, so it's ok only
    for small ranges with probabilities well greater than 0.
    """
    ecdf = jnp.cumsum(distr)
    u = random.uniform(key, (), ecdf.dtype, 0, ecdf[-1])
    return jnp.searchsorted(ecdf, u, 'right', method='compare_all'), ecdf[-1]


def choose_variable(
    key: Key[Array, ''],
    var_tree: UInt[Array, ' 2**(d-1)'],
    split_tree: UInt[Array, ' 2**(d-1)'],
    max_split: UInt[Array, ' p'],
    leaf_index: Int32[Array, ''],
    blocked_vars: Int32[Array, ' k'] | None,
    log_s: Float32[Array, ' p'] | None,
) -> tuple[Int32[Array, ''], Int32[Array, '']]:
    """
    Choose a variable to split on for a new non-terminal node.

    Parameters
    ----------
    key
        A jax random key.
    var_tree
        The variable indices of the tree.
    split_tree
        The splitting points of the tree.
    max_split
        The maximum split index for each variable.
    leaf_index
        The index of the leaf to grow.
    blocked_vars
        The indices of the variables that have no available cutpoints. If
        `None`, all variables are assumed unblocked.
    log_s
        The logarithm of the prior probability for choosing a variable. If
        `None`, use a uniform distribution.

    Returns
    -------
    var : Int32[Array, '']
        The index of the variable to split on.
    num_available_var : Int32[Array, '']
        The number of variables with available decision rules `var` was chosen
        from.
    """
    var_to_ignore = fully_used_variables(var_tree, split_tree, max_split, leaf_index)
    if blocked_vars is not None:
        var_to_ignore = jnp.concatenate([var_to_ignore, blocked_vars])

    if log_s is None:
        return randint_exclude(key, max_split.size, var_to_ignore)
    else:
        return categorical_exclude(key, log_s, var_to_ignore)


def fully_used_variables(
    var_tree: UInt[Array, ' 2**(d-1)'],
    split_tree: UInt[Array, ' 2**(d-1)'],
    max_split: UInt[Array, ' p'],
    leaf_index: Int32[Array, ''],
) -> UInt[Array, ' d-2']:
    """
    Find variables in the ancestors of a node that have an empty split range.

    Parameters
    ----------
    var_tree
        The variable indices of the tree.
    split_tree
        The splitting points of the tree.
    max_split
        The maximum split index for each variable.
    leaf_index
        The index of the node, assumed to be valid for `var_tree`.

    Returns
    -------
    The indices of the variables that have an empty split range.

    Notes
    -----
    The number of unused variables is not known in advance. Unused values in the
    array are filled with `p`. The fill values are not guaranteed to be placed
    in any particular order, and variables may appear more than once.
    """
    var_to_ignore = ancestor_variables(var_tree, max_split, leaf_index)
    split_range_vec = jax.vmap(split_range, in_axes=(None, None, None, None, 0))
    l, r = split_range_vec(var_tree, split_tree, max_split, leaf_index, var_to_ignore)
    num_split = r - l
    return jnp.where(num_split == 0, var_to_ignore, max_split.size)
    # the type of var_to_ignore is already sufficient to hold max_split.size,
    # see ancestor_variables()


def ancestor_variables(
    var_tree: UInt[Array, ' 2**(d-1)'],
    max_split: UInt[Array, ' p'],
    node_index: Int32[Array, ''],
) -> UInt[Array, ' d-2']:
    """
    Return the list of variables in the ancestors of a node.

    Parameters
    ----------
    var_tree
        The variable indices of the tree.
    max_split
        The maximum split index for each variable. Used only to get `p`.
    node_index
        The index of the node, assumed to be valid for `var_tree`.

    Returns
    -------
    The variable indices of the ancestors of the node.

    Notes
    -----
    The ancestors are the nodes going from the root to the parent of the node.
    The number of ancestors is not known at tracing time; unused spots in the
    output array are filled with `p`.
    """
    max_num_ancestors = grove.tree_depth(var_tree) - 1
    index = node_index >> jnp.arange(max_num_ancestors, 0, -1)
    var = var_tree[index]
    var_type = minimal_unsigned_dtype(max_split.size)
    p = jnp.array(max_split.size, var_type)
    return jnp.where(index, var, p)


def split_range(
    var_tree: UInt[Array, ' 2**(d-1)'],
    split_tree: UInt[Array, ' 2**(d-1)'],
    max_split: UInt[Array, ' p'],
    node_index: Int32[Array, ''],
    ref_var: Int32[Array, ''],
) -> tuple[Int32[Array, ''], Int32[Array, '']]:
    """
    Return the range of allowed splits for a variable at a given node.

    Parameters
    ----------
    var_tree
        The variable indices of the tree.
    split_tree
        The splitting points of the tree.
    max_split
        The maximum split index for each variable.
    node_index
        The index of the node, assumed to be valid for `var_tree`.
    ref_var
        The variable for which to measure the split range.

    Returns
    -------
    The range of allowed splits as [l, r). If `ref_var` is out of bounds, l=r=1.
    """
    max_num_ancestors = grove.tree_depth(var_tree) - 1
    index = node_index >> jnp.arange(max_num_ancestors)
    right_child = (index & 1).astype(bool)
    index >>= 1
    split = split_tree[index].astype(jnp.int32)
    cond = (var_tree[index] == ref_var) & index.astype(bool)
    l = jnp.max(split, initial=0, where=cond & right_child)
    initial_r = 1 + max_split.at[ref_var].get(mode='fill', fill_value=0).astype(
        jnp.int32
    )
    r = jnp.min(split, initial=initial_r, where=cond & ~right_child)

    return l + 1, r


def randint_exclude(
    key: Key[Array, ''], sup: int | Integer[Array, ''], exclude: Integer[Array, ' n']
) -> tuple[Int32[Array, ''], Int32[Array, '']]:
    """
    Return a random integer in a range, excluding some values.

    Parameters
    ----------
    key
        A jax random key.
    sup
        The exclusive upper bound of the range, must be >= 1.
    exclude
        The values to exclude from the range. Values greater than or equal to
        `sup` are ignored. Values can appear more than once.

    Returns
    -------
    u : Int32[Array, '']
        A random integer `u` in the range ``[0, sup)`` such that ``u not in
        exclude``.
    num_allowed : Int32[Array, '']
        The number of integers in the range that were not excluded.

    Notes
    -----
    If all values in the range are excluded, return `sup`.
    """
    exclude, num_allowed = _process_exclude(sup, exclude)
    u = random.randint(key, (), 0, num_allowed)
    u_shifted = u + jnp.arange(exclude.size)
    u_shifted = jnp.minimum(u_shifted, sup - 1)
    u += jnp.sum(u_shifted >= exclude)
    return u, num_allowed


def _process_exclude(sup, exclude):
    exclude = jnp.unique(exclude, size=exclude.size, fill_value=sup)
    num_allowed = sup - jnp.sum(exclude < sup)
    return exclude, num_allowed


def categorical_exclude(
    key: Key[Array, ''], logits: Float32[Array, ' k'], exclude: Integer[Array, ' n']
) -> tuple[Int32[Array, ''], Int32[Array, '']]:
    """
    Draw from a categorical distribution, excluding a set of values.

    Parameters
    ----------
    key
        A jax random key.
    logits
        The unnormalized log-probabilities of each category.
    exclude
        The values to exclude from the range [0, k). Values greater than or
        equal to `logits.size` are ignored. Values can appear more than once.

    Returns
    -------
    u : Int32[Array, '']
        A random integer in the range ``[0, k)`` such that ``u not in exclude``.
    num_allowed : Int32[Array, '']
        The number of integers in the range that were not excluded.

    Notes
    -----
    If all values in the range are excluded, the result is unspecified.
    """
    exclude, num_allowed = _process_exclude(logits.size, exclude)
    kinda_neg_inf = jnp.finfo(logits.dtype).min
    logits = logits.at[exclude].set(kinda_neg_inf)
    u = random.categorical(key, logits)
    return u, num_allowed


def choose_split(
    key: Key[Array, ''],
    var: Int32[Array, ''],
    var_tree: UInt[Array, ' 2**(d-1)'],
    split_tree: UInt[Array, ' 2**(d-1)'],
    max_split: UInt[Array, ' p'],
    leaf_index: Int32[Array, ''],
) -> tuple[Int32[Array, ''], Int32[Array, ''], Int32[Array, '']]:
    """
    Choose a split point for a new non-terminal node.

    Parameters
    ----------
    key
        A jax random key.
    var
        The variable to split on.
    var_tree
        The splitting axes of the tree. Does not need to already contain `var`
        at `leaf_index`.
    split_tree
        The splitting points of the tree.
    max_split
        The maximum split index for each variable.
    leaf_index
        The index of the leaf to grow.

    Returns
    -------
    split : Int32[Array, '']
        The cutpoint.
    l : Int32[Array, '']
    r : Int32[Array, '']
        The integer range `split` was drawn from is [l, r).

    Notes
    -----
    If `var` is out of bounds, or if the available split range on that variable
    is empty, return 0.
    """
    l, r = split_range(var_tree, split_tree, max_split, leaf_index, var)
    return jnp.where(l < r, random.randint(key, (), l, r), 0), l, r


def compute_partial_ratio(
    prob_choose: Float32[Array, ''],
    num_prunable: Int32[Array, ''],
    p_nonterminal: Float32[Array, ' 2**d'],
    leaf_to_grow: Int32[Array, ''],
) -> Float32[Array, '']:
    """
    Compute the product of the transition and prior ratios of a grow move.

    Parameters
    ----------
    prob_choose
        The probability that the leaf had to be chosen amongst the growable
        leaves.
    num_prunable
        The number of leaf parents that could be pruned, after converting the
        leaf to be grown to a non-terminal node.
    p_nonterminal
        The a priori probability of each node being nonterminal conditional on
        its ancestors.
    leaf_to_grow
        The index of the leaf to grow.

    Returns
    -------
    The partial transition ratio times the prior ratio.

    Notes
    -----
    The transition ratio is P(new tree => old tree) / P(old tree => new tree).
    The "partial" transition ratio returned is missing the factor P(propose
    prune) in the numerator. The prior ratio is P(new tree) / P(old tree). The
    "partial" prior ratio is missing the factor P(children are leaves).
    """
    # the two ratios also contain factors num_available_split *
    # num_available_var * s[var], but they cancel out

    # p_prune and 1 - p_nonterminal[child] * I(is the child growable) can't be
    # computed here because they need the count trees, which are computed in the
    # acceptance phase

    prune_allowed = leaf_to_grow != 1
    # prune allowed  <--->  the initial tree is not a root
    # leaf to grow is root  -->  the tree can only be a root
    # tree is a root  -->  the only leaf I can grow is root
    p_grow = jnp.where(prune_allowed, 0.5, 1)
    inv_trans_ratio = p_grow * prob_choose * num_prunable

    # .at.get because if leaf_to_grow is out of bounds (move not allowed), this
    # would produce a 0 and then an inf when `complete_ratio` takes the log
    pnt = p_nonterminal.at[leaf_to_grow].get(mode='fill', fill_value=0.5)
    tree_ratio = pnt / (1 - pnt)

    return tree_ratio / jnp.where(inv_trans_ratio, inv_trans_ratio, 1)


class PruneMoves(Module):
    """
    Represent a proposed prune move for each tree.

    Parameters
    ----------
    allowed
        Whether the move is possible.
    node
        The index of the node to prune. ``2 ** d`` if no node can be pruned.
    partial_ratio
        A factor of the Metropolis-Hastings ratio of the move. It lacks the
        likelihood ratio, the probability of proposing the prune move, and the
        prior probability that the children of the node to prune are leaves.
        This ratio is inverted, and is meant to be inverted back in
        `accept_move_and_sample_leaves`.
    affluence_tree
        A partially updated `affluence_tree`, marking the node to prune as
        growable.
    """

    allowed: Bool[Array, ' num_trees']
    node: UInt[Array, ' num_trees']
    partial_ratio: Float32[Array, ' num_trees']
    affluence_tree: Bool[Array, 'num_trees 2**(d-1)']


@partial(vmap_nodoc, in_axes=(0, 0, 0, None, None))
def propose_prune_moves(
    key: Key[Array, ''],
    split_tree: UInt[Array, ' 2**(d-1)'],
    affluence_tree: Bool[Array, ' 2**(d-1)'],
    p_nonterminal: Float32[Array, ' 2**d'],
    p_propose_grow: Float32[Array, ' 2**(d-1)'],
) -> PruneMoves:
    """
    Tree structure prune move proposal of BART MCMC.

    Parameters
    ----------
    key
        A jax random key.
    split_tree
        The splitting points of the tree.
    affluence_tree
        Whether each leaf can be grown.
    p_nonterminal
        The a priori probability of a node to be nonterminal conditional on
        the ancestors, including at the maximum depth where it should be zero.
    p_propose_grow
        The unnormalized probability of choosing a leaf to grow.

    Returns
    -------
    An object representing the proposed moves.
    """
    node_to_prune, num_prunable, prob_choose, affluence_tree = choose_leaf_parent(
        key, split_tree, affluence_tree, p_propose_grow
    )

    ratio = compute_partial_ratio(
        prob_choose, num_prunable, p_nonterminal, node_to_prune
    )

    return PruneMoves(
        allowed=split_tree[1].astype(bool),  # allowed iff the tree is not a root
        node=node_to_prune,
        partial_ratio=ratio,
        affluence_tree=affluence_tree,
    )


def choose_leaf_parent(
    key: Key[Array, ''],
    split_tree: UInt[Array, ' 2**(d-1)'],
    affluence_tree: Bool[Array, ' 2**(d-1)'],
    p_propose_grow: Float32[Array, ' 2**(d-1)'],
) -> tuple[
    Int32[Array, ''],
    Int32[Array, ''],
    Float32[Array, ''],
    Bool[Array, 'num_trees 2**(d-1)'],
]:
    """
    Pick a non-terminal node with leaf children to prune in a tree.

    Parameters
    ----------
    key
        A jax random key.
    split_tree
        The splitting points of the tree.
    affluence_tree
        Whether a leaf has enough points to be grown.
    p_propose_grow
        The unnormalized probability of choosing a leaf to grow.

    Returns
    -------
    node_to_prune : Int32[Array, '']
        The index of the node to prune. If ``num_prunable == 0``, return
        ``2 ** d``.
    num_prunable : Int32[Array, '']
        The number of leaf parents that could be pruned.
    prob_choose : Float32[Array, '']
        The (normalized) probability that `choose_leaf` would chose
        `node_to_prune` as leaf to grow, if passed the tree where
        `node_to_prune` had been pruned.
    affluence_tree : Bool[Array, 'num_trees 2**(d-1)']
        A partially updated `affluence_tree`, marking the node to prune as
        growable.
    """
    # sample a node to prune
    is_prunable = grove.is_leaves_parent(split_tree)
    num_prunable = jnp.count_nonzero(is_prunable)
    node_to_prune = randint_masked(key, is_prunable)
    node_to_prune = jnp.where(num_prunable, node_to_prune, 2 * split_tree.size)

    # compute stuff for reverse move
    split_tree = split_tree.at[node_to_prune].set(0)
    affluence_tree = affluence_tree.at[node_to_prune].set(True)
    is_growable_leaf = growable_leaves(split_tree, affluence_tree)
    distr_norm = jnp.sum(p_propose_grow, where=is_growable_leaf)
    prob_choose = p_propose_grow.at[node_to_prune].get(mode='fill', fill_value=0)
    prob_choose = prob_choose / jnp.where(distr_norm, distr_norm, 1)

    return node_to_prune, num_prunable, prob_choose, affluence_tree


def randint_masked(key: Key[Array, ''], mask: Bool[Array, ' n']) -> Int32[Array, '']:
    """
    Return a random integer in a range, including only some values.

    Parameters
    ----------
    key
        A jax random key.
    mask
        The mask indicating the allowed values.

    Returns
    -------
    A random integer in the range ``[0, n)`` such that ``mask[u] == True``.

    Notes
    -----
    If all values in the mask are `False`, return `n`. This function is
    optimized for small `n`.
    """
    ecdf = jnp.cumsum(mask)
    u = random.randint(key, (), 0, ecdf[-1])
    return jnp.searchsorted(ecdf, u, 'right', method='compare_all')
