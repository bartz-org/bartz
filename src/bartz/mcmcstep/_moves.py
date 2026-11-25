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

from equinox import Module, field
from jax import named_call, random, vmap
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float32, Int32, Integer, Key, UInt

from bartz import grove
from bartz._jaxext import minimal_unsigned_dtype, split
from bartz.mcmcstep._state import Forest, used_trees_mask


class Moves(Module):
    """Moves proposed to modify each tree."""

    allowed: Bool[Array, '*num_trees']
    """Whether there is a possible move. If `False`, the other values may not
    make sense. The only case in which a move is marked as allowed but is
    then vetoed is if it does not satisfy `min_points_per_leaf`, which for
    efficiency is implemented post-hoc without changing the rest of the
    MCMC logic."""

    grow: Bool[Array, '*num_trees']
    """Whether the move is a grow move or a prune move."""

    num_growable: Int32[Array, '*num_trees']
    """The number of growable leaves in the original tree."""

    lrt_nodes: Int32[Array, '*num_trees 3']
    """The indices of the left child, right child, and parent node the move
    operates on, stacked along the trailing axis. The parent is the leaf to
    grow or the node to prune."""

    lr_leaves: Int32[Array, '*num_trees 2']
    """The indices of the left and right child in the leaf arrays, see
    `bartz.grove.node_to_leaf_index`. The parent, as a leaf, has the index of
    the left child. Out of bounds if there is no node to operate on."""

    partial_ratio: Float32[Array, '*num_trees'] | None
    """A factor of the Metropolis-Hastings ratio of the move. It lacks the
    likelihood ratio, the probability of proposing the prune move, and the
    prior probabilities that the modified node is nonterminal and that its
    children are terminal. If the move is PRUNE, the ratio is inverted. `None`
    once `log_trans_prior_ratio` has been computed."""

    log_trans_prior_ratio: None | Float32[Array, '*num_trees']
    """The logarithm of the product of the transition and prior terms of the
    Metropolis-Hastings ratio for the acceptance of the proposed move.
    `None` if not yet computed. If PRUNE, the log-ratio is negated."""

    grow_var: Int32[Array, '*num_trees']
    """The decision axes of the new rules."""

    grow_split: Int32[Array, '*num_trees']
    """The decision boundaries of the new rules."""

    var_tree: UInt[Array, '*num_trees half_tree_size']
    """The updated decision axes of the trees, valid whatever move."""

    lrt_growable: Bool[Array, '*num_trees 3']
    """Whether the nodes in `lrt_nodes`, as leaves, have available decision
    rules, *not* counting the datapoint thresholds. The parent slot is `True`
    by construction. This is the admissibility used in the prior term of the
    Metropolis-Hastings ratio, matching the standard BART prior which ignores
    the count constraints."""

    lrt_affluent: None | Bool[Array, '*num_trees 3']
    """Whether the nodes in `lrt_nodes`, as leaves within the heap, would be
    growable leaves, counting the datapoint thresholds. The children slots
    matter when the move grows, the parent slot when it prunes or a grow is
    rejected. `None` until set in `accept_moves_parallel_stage`."""

    logu: Float32[Array, '*num_trees']
    """The logarithm of a uniform (0, 1] random variable to be used to
    accept the move. It's in (-oo, 0]."""

    acc: None | Bool[Array, '*num_trees']
    """Whether the move was accepted. `None` if not yet computed."""

    to_prune: None | Bool[Array, '*num_trees']
    """Whether the final operation to apply the move is pruning. This indicates
    an accepted prune move or a rejected grow move. `None` if not yet
    computed."""


class CategoricalTable(Module):
    """A categorical distribution laid out in rows for `categorical_exclude`."""

    logits: Float32[Array, 'rows cols']
    """The log-weights of the categories in row-major order, -inf for the
    excluded categories and for the padding."""

    row_max: Float32[Array, ' rows']
    """The maximum of each row of `logits`."""

    row_sum: Float32[Array, ' rows']
    """The total weight of each row, relative to its largest weight."""

    num_excluded: Int32[Array, '']
    """The number of excluded categories."""

    size: int = field(static=True)
    """The number of categories."""


@named_call
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
    keys = split(key)

    # tabulate the split probabilities once for all the trees
    if forest.sparsity.log_s is None:
        s_table = None
    else:
        s_table = categorical_table(forest.sparsity.log_s, forest.blocked_vars)

    propose = partial(
        _propose_moves,
        max_split=forest.max_split,
        blocked_vars=forest.blocked_vars,
        p_propose_grow=forest.p_propose_grow,
        s_table=s_table,
    )
    return vmap(propose)(
        keys.pop(num_trees),
        forest.var_tree,
        forest.split_tree,
        forest.affluence_tree,
        used_trees_mask(forest),
    )


@named_call
def _propose_moves(
    key: Key[Array, ''],
    var_tree: UInt[Array, ' half_tree_size'],
    split_tree: UInt[Array, ' half_tree_size'],
    affluence_tree: Bool[Array, ' half_tree_size'],
    used: Bool[Array, ''] | None,
    *,
    max_split: UInt[Array, ' p'],
    blocked_vars: UInt[Array, ' k'] | None,
    p_propose_grow: Float32[Array, ' half_tree_size'],
    s_table: CategoricalTable | None,
) -> Moves:
    """
    Propose a GROW or PRUNE move for a tree.

    A GROW move picks a leaf node and converts it to a non-terminal node with
    two leaf children. A PRUNE move picks a non-terminal node with two leaf
    children and converts it back to a leaf.

    The grow leaf and the prune node are both sampled, then one of the two
    moves is selected, and the decision-rule machinery is run once on the
    selected node. For GROW this samples a new rule and the admissibility of
    the new leaves; for PRUNE it re-derives, from the rule already stored at
    the node, the admissibility of the leaves that would be deleted, reusing
    the same criterion without a second tree traversal.

    Parameters
    ----------
    key
        A jax random key.
    var_tree
        The splitting axes of the tree.
    split_tree
        The splitting points of the tree.
    affluence_tree
        Marks the leaves that can be grown.
    used
        Whether the tree is in use, see `bartz.mcmcstep.Forest.used_trees`;
        `None` if all the trees are.
    max_split
        The maximum split index for each variable.
    blocked_vars
        The indices of the variables that have no available cutpoints.
    p_propose_grow
        The unnormalized probability of choosing a leaf to grow.
    s_table
        The distribution used to choose a variable to split on amongst the
        available ones, see `choose_variable`.

    Returns
    -------
    An object representing the proposed move.

    Notes
    -----
    A GROW move is not possible if the tree is not in use, or each leaf is
    already at maximum depth, or has less datapoints than the requested
    threshold `min_points_per_decision_node`, or it does not have any available
    decision rules given its ancestors. A PRUNE move is not possible if the tree
    is a single root. If neither move is possible, `allowed` is set to `False`.
    """
    keys = split(key, 5)

    # sample a leaf to grow and a node to prune
    leaf_to_grow, num_growable, grow_prob_choose, grow_num_prunable = choose_leaf(
        keys.pop(), split_tree, affluence_tree, p_propose_grow
    )
    node_to_prune, prune_num_prunable, prune_prob_choose = choose_leaf_parent(
        keys.pop(), split_tree, affluence_tree, p_propose_grow
    )

    # choose between grow and prune
    grow_allowed = num_growable > 0
    prune_allowed = split_tree[1].astype(bool)  # the tree is not a single root
    if used is not None:
        grow_allowed &= used
    u, exp1mlogu = random.uniform(keys.pop(), (2,))
    p_grow = jnp.where(grow_allowed & prune_allowed, 0.5, grow_allowed)
    grow = u < p_grow  # use < instead of <= because u is in [0, 1)

    # merge the node the move operates on, and stack its children indices with
    # it: [left child, right child, node]
    node = jnp.where(grow, leaf_to_grow, node_to_prune)
    lrt_nodes = (node << jnp.array([1, 1, 0])) | jnp.array([0, 1, 0])
    left_leaf = jnp.where(node < split_tree.size, grove.node_to_leaf_index(node), node)
    lr_leaves = jnp.stack([left_leaf, node])

    # sample a decision rule for GROW; for PRUNE recover the rule already stored
    # at the node. `num_available_var` is the count of admissible variables at
    # the node, which is the same quantity needed for either move.
    sampled_var, num_available_var = choose_variable(
        keys.pop(),
        var_tree,
        split_tree,
        max_split,
        node,
        blocked_vars=blocked_vars,
        s_table=s_table,
    )
    existing_var = var_tree.at[node].get(mode='fill', fill_value=0).astype(jnp.int32)
    var = jnp.where(grow, sampled_var, existing_var)
    sampled_split, l, r = choose_split(
        keys.pop(), var_tree, split_tree, max_split, node, var=var
    )
    existing_split = (
        split_tree.at[node].get(mode='fill', fill_value=0).astype(jnp.int32)
    )
    split_idx = jnp.where(grow, sampled_split, existing_split)

    # admissibility of the nodes involved in the move as leaves, ignoring
    # counts; for GROW the children are the new leaves, for PRUNE the
    # (existing) leaves that would be deleted, while `node` is admissible by
    # construction whatever the move. The criterion for a child is the same as
    # for grow: it is admissible if there is another available variable, or the
    # used variable still has a nonempty split range on the child's side. If
    # the move is blocked these values may not make sense.
    lrt_growable = (num_available_var > 1) | jnp.stack(
        [l < split_idx, split_idx + 1 < r, jnp.array(True)]
    )

    # partial Metropolis-Hastings ratio; the formula is shared, treating the
    # node as the leaf being grown (for PRUNE this is the reverse grow move,
    # whose ratio is inverted later)
    prob_choose = jnp.where(grow, grow_prob_choose, prune_prob_choose)
    num_prunable = jnp.where(grow, grow_num_prunable, prune_num_prunable)
    ratio = compute_partial_ratio(prob_choose, num_prunable, node)

    return Moves(
        allowed=grow_allowed | prune_allowed,
        grow=grow,
        num_growable=num_growable,
        lrt_nodes=lrt_nodes,
        lr_leaves=lr_leaves,
        partial_ratio=ratio,
        log_trans_prior_ratio=None,  # will be set in complete_ratio
        grow_var=sampled_var,
        grow_split=sampled_split,
        # var_tree only changes at `node` for GROW; for PRUNE this is a no-op
        # since `var` equals the existing variable there
        var_tree=var_tree.at[node].set(var.astype(var_tree.dtype)),
        lrt_growable=lrt_growable,
        lrt_affluent=None,  # set in accept_moves_parallel_stage
        logu=jnp.log1p(-exp1mlogu),
        acc=None,  # will be set in accept_moves_sequential_stage
        to_prune=None,  # will be set in accept_moves_sequential_stage
    )


def choose_leaf(
    key: Key[Array, ''],
    split_tree: UInt[Array, ' half_tree_size'],
    affluence_tree: Bool[Array, ' half_tree_size'],
    p_propose_grow: Float32[Array, ' half_tree_size'],
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
        Marks the leaves that can be grown.
    p_propose_grow
        The unnormalized probability of choosing a leaf to grow.

    Returns
    -------
    leaf_to_grow : Int32[Array, '']
        The index of the leaf to grow. If ``num_growable == 0``, return
        ``2 ** d``.
    num_growable : Int32[Array, '']
        The number of growable leaves in the tree.
    prob_choose : Float32[Array, '']
        The (normalized) probability that this function had to choose that
        specific leaf, given the arguments.
    num_prunable : Int32[Array, '']
        The number of leaf parents that could be pruned, after converting the
        selected leaf to a non-terminal node.
    """
    is_growable = affluence_tree  # `affluence_tree` is the clean growable mask
    num_growable = jnp.count_nonzero(is_growable)
    distr = jnp.where(is_growable, p_propose_grow, 0)
    leaf_to_grow = categorical_cumsum(random_bits30(key), distr)
    leaf_to_grow = jnp.where(num_growable, leaf_to_grow, 2 * split_tree.size)
    distr_norm = jnp.sum(distr)
    prob_choose = distr[leaf_to_grow] / jnp.where(distr_norm, distr_norm, 1)
    is_parent = grove.is_leaves_parent(split_tree.at[leaf_to_grow].set(1))
    num_prunable = jnp.count_nonzero(is_parent)
    return leaf_to_grow, num_growable, prob_choose, num_prunable


def choose_variable(
    key: Key[Array, ''],
    var_tree: UInt[Array, ' half_tree_size'],
    split_tree: UInt[Array, ' half_tree_size'],
    max_split: UInt[Array, ' p'],
    leaf_index: Int32[Array, ''],
    *,
    blocked_vars: UInt[Array, ' k'] | None,
    s_table: CategoricalTable | None,
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
    s_table
        The prior probability for choosing a variable, tabulated by
        `categorical_table` with `blocked_vars` excluded. If `None`, use a
        uniform distribution.

    Returns
    -------
    var : Int32[Array, '']
        The index of the variable to split on.
    num_available_var : Int32[Array, '']
        The number of variables with available decision rules `var` was chosen
        from.
    """
    var_to_ignore = fully_used_variables(var_tree, split_tree, max_split, leaf_index)

    if s_table is None:
        if blocked_vars is not None:
            var_to_ignore = jnp.concatenate([var_to_ignore, blocked_vars])
        return randint_exclude(key, max_split.size, var_to_ignore)
    else:
        # the blocked variables have no splits, so they are never in
        # `var_to_ignore`, which lists variables used by the ancestors
        return categorical_exclude(key, s_table, var_to_ignore)


def fully_used_variables(
    var_tree: UInt[Array, ' half_tree_size'],
    split_tree: UInt[Array, ' half_tree_size'],
    max_split: UInt[Array, ' p'],
    leaf_index: Int32[Array, ''],
) -> UInt[Array, ' d_minus_2']:
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
    split_range_vec = vmap(split_range, in_axes=(None, None, None, None, 0))
    l, r = split_range_vec(var_tree, split_tree, max_split, leaf_index, var_to_ignore)
    num_split = r - l
    return jnp.where(num_split == 0, var_to_ignore, max_split.size)
    # the type of var_to_ignore is already sufficient to hold max_split.size,
    # see ancestor_variables()


def ancestor_variables(
    var_tree: UInt[Array, ' half_tree_size'],
    max_split: UInt[Array, ' p'],
    node_index: Int32[Array, ''],
) -> UInt[Array, ' d_minus_2']:
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
    var_tree: UInt[Array, ' half_tree_size'],
    split_tree: UInt[Array, ' half_tree_size'],
    max_split: UInt[Array, ' p'],
    node_index: Int32[Array, ''],
    ref_var: Integer[Array, ''],
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


def _process_exclude(
    sup: int | Integer[Array, ''], exclude: Integer[Array, ' n']
) -> tuple[Integer[Array, ' n'], Integer[Array, '']]:
    exclude = jnp.unique(exclude, size=exclude.size, fill_value=sup)
    num_allowed = sup - jnp.sum(exclude < sup)
    return exclude, num_allowed


def categorical_table(
    logits: Float32[Array, ' k'], exclude: Integer[Array, ' n'] | None
) -> CategoricalTable:
    """
    Tabulate a categorical distribution for `categorical_exclude`.

    Parameters
    ----------
    logits
        The unnormalized log-probabilities of each category.
    exclude
        The values to exclude from the range [0, k), without repetitions.
        Values greater than or equal to `k` are ignored. If `None`, no value
        is excluded.

    Returns
    -------
    The distribution, with the weights arranged in about sqrt(k) rows.
    """
    (k,) = logits.shape
    cols = 1 << (k.bit_length() // 2)
    rows = k // cols + 1  # this always pads, so `k` indexes a padding slot

    if exclude is None:
        num_excluded = jnp.zeros((), jnp.int32)
    else:
        logits = logits.at[exclude].set(-jnp.inf)
        num_excluded = jnp.count_nonzero(exclude < k)
    logits = jnp.pad(logits, (0, rows * cols - k), constant_values=-jnp.inf)
    logits = logits.reshape(rows, cols)

    weights, row_max = exp_minus_max(logits)
    return CategoricalTable(
        logits=logits,
        row_max=row_max,
        row_sum=jnp.sum(weights, axis=1),
        num_excluded=num_excluded,
        size=k,
    )


def exp_minus_max(
    x: Float32[Array, '*batch n'],
) -> tuple[Float32[Array, '*batch n'], Float32[Array, '*batch']]:
    """Return ``exp(x - max(x))`` and ``max(x)`` along the last axis.

    If `x` is all -inf, the exponential is 0 instead of nan.
    """
    m = jnp.max(x, axis=-1, keepdims=True)
    return jnp.exp(x - jnp.where(jnp.isneginf(m), 0, m)), m.squeeze(-1)


def categorical_exclude(
    key: Key[Array, ''], table: CategoricalTable, exclude: Integer[Array, ' n']
) -> tuple[Int32[Array, ''], Int32[Array, '']]:
    """
    Draw from a categorical distribution, excluding a set of values.

    Parameters
    ----------
    key
        A jax random key.
    table
        The distribution over the range [0, k), see `categorical_table`.
    exclude
        The values to exclude from the range, in addition to and distinct from
        those already excluded in `table`. Values greater than or equal to `k`
        are ignored. Values can appear more than once.

    Returns
    -------
    u : Int32[Array, '']
        A random integer in the range ``[0, k)`` such that ``u not in exclude``.
    num_allowed : Int32[Array, '']
        The number of integers in the range that were not excluded.

    Notes
    -----
    If the values that are not excluded have total probability zero, return
    `k`.

    The draw is by inversion in two steps, first the row then the entry within
    it, in O(sqrt(k)) instead of O(k). Each step has its own random word and
    normalizes the weights to their maximum after the exclusions, so it is
    accurate even if the excluded values have far larger weights than the
    others.
    """
    bits30_row, bits30_col = random_bits30(key, (2,))
    _, cols = table.logits.shape
    exclude = jnp.minimum(exclude.astype(jnp.int32), table.size)
    exclude, num_allowed = _process_exclude(table.size, exclude)

    def masked_row(row: Int32[Array, '']) -> Float32[Array, ' cols']:
        index = row * cols + jnp.arange(cols)
        excluded = jnp.any(index == exclude[:, None], axis=0)
        return jnp.where(excluded, -jnp.inf, table.logits[row, :])

    # recompute from scratch the totals of the rows with excluded values,
    # instead of subtracting weights, such that a row has total 0 if and only
    # if no value can be drawn from it; repeated rows write the same totals
    touched = exclude // cols
    weights, touched_max = exp_minus_max(vmap(masked_row)(touched))
    row_max = table.row_max.at[touched].set(touched_max)
    row_sum = table.row_sum.at[touched].set(jnp.sum(weights, axis=1))

    # draw the row
    scale, _ = exp_minus_max(row_max)
    row = categorical_cumsum(bits30_row, scale * row_sum)

    # draw the entry within the row; if there was no row to draw, the index
    # overflows the table
    weights, _ = exp_minus_max(masked_row(row))
    col = categorical_cumsum(bits30_col, weights)
    u = jnp.minimum(row * cols + col, table.size)

    return u, num_allowed - table.num_excluded


def random_bits30(
    key: Key[Array, ''], shape: tuple[int, ...] = ()
) -> Int32[Array, '*shape']:
    """Return random integers uniform in ``[0, 2^30)``, for `categorical_cumsum`."""
    return (random.bits(key, shape, jnp.uint32) >> 2).astype(jnp.int32)


def categorical_cumsum(
    bits30: Int32[Array, ''], distr: Float32[Array, ' n']
) -> Int32[Array, '']:
    """
    Return a random integer from an arbitrary distribution.

    Parameters
    ----------
    bits30
        A random integer, uniform in ``[0, 2^30)``, see `random_bits30`.
    distr
        An unnormalized probability distribution.

    Returns
    -------
    A random integer in the range ``[0, n)``, or `n` if all probabilities are zero.

    Notes
    -----
    This never returns a value with probability zero. It accumulates integers,
    because xla does not sum sequentially in a float cumsum, which then may
    increase or decrease across a zero.

    The probabilities are rounded to multiples of 2^-30. Due to rounding they
    may not add up to 1; the last nonzero values absorb the difference.
    """
    total = jnp.sum(distr)
    quanta = jnp.round(distr * (2**30 / jnp.where(total, total, 1))).astype(jnp.int32)
    ecdf = jnp.cumsum(quanta)
    # keep u below the total, or at 0 if the total is 0 such that the search
    # returns n; on gpu, taking the total from `ecdf`, or handling the zero
    # total with a `where`, makes the compilation up to 20% slower
    u = jnp.minimum(bits30, jnp.maximum(jnp.sum(quanta) - 1, 0))
    return jnp.searchsorted(ecdf, u, 'right', method='compare_all')


def choose_split(
    key: Key[Array, ''],
    var_tree: UInt[Array, ' half_tree_size'],
    split_tree: UInt[Array, ' half_tree_size'],
    max_split: UInt[Array, ' p'],
    leaf_index: Int32[Array, ''],
    *,
    var: Int32[Array, ''],
) -> tuple[Int32[Array, ''], Int32[Array, ''], Int32[Array, '']]:
    """
    Choose a split point for a new non-terminal node.

    Parameters
    ----------
    key
        A jax random key.
    var_tree
        The splitting axes of the tree. Does not need to already contain `var`
        at `leaf_index`.
    split_tree
        The splitting points of the tree.
    max_split
        The maximum split index for each variable.
    leaf_index
        The index of the leaf to grow.
    var
        The variable to split on.

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
    leaf_to_grow: Int32[Array, ''],
) -> Float32[Array, '']:
    """
    Compute the partial transition ratio of a grow move.

    Parameters
    ----------
    prob_choose
        The probability that the leaf had to be chosen amongst the growable
        leaves.
    num_prunable
        The number of leaf parents that could be pruned, after converting the
        leaf to be grown to a non-terminal node.
    leaf_to_grow
        The index of the leaf to grow.

    Returns
    -------
    The partial transition ratio.

    Notes
    -----
    The transition ratio is P(new tree => old tree) / P(old tree => new tree).
    The "partial" transition ratio returned is missing the factor P(propose
    prune) in the numerator. The prior ratio P(new tree) / P(old tree) is not
    included either: its decision rule factors cancel against the transition
    ratio, and its node terminality factors are applied in `complete_ratio`.
    """
    # the transition and prior ratios also contain factors num_available_split
    # * num_available_var * s[var], but they cancel out

    # p_prune can't be computed here because it needs the count trees, which
    # are computed in the acceptance phase; the prior terminality factors are
    # computed alongside it in `complete_ratio` to share a single gather of
    # p_nonterminal

    prune_allowed = leaf_to_grow != 1
    # prune allowed  <--->  the initial tree is not a root
    # leaf to grow is root  -->  the tree can only be a root
    # tree is a root  -->  the only leaf I can grow is root
    p_grow = jnp.where(prune_allowed, 0.5, 1)
    inv_trans_ratio = p_grow * prob_choose * num_prunable

    return jnp.reciprocal(jnp.where(inv_trans_ratio, inv_trans_ratio, 1))


def choose_leaf_parent(
    key: Key[Array, ''],
    split_tree: UInt[Array, ' half_tree_size'],
    affluence_tree: Bool[Array, ' half_tree_size'],
    p_propose_grow: Float32[Array, ' half_tree_size'],
) -> tuple[Int32[Array, ''], Int32[Array, ''], Float32[Array, '']]:
    """
    Pick a non-terminal node with leaf children to prune in a tree.

    Parameters
    ----------
    key
        A jax random key.
    split_tree
        The splitting points of the tree.
    affluence_tree
        Marks the leaves that can be grown.
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
    """
    # sample a node to prune
    is_prunable = grove.is_leaves_parent(split_tree)
    num_prunable = jnp.count_nonzero(is_prunable)
    node_to_prune = randint_masked(key, is_prunable)
    node_to_prune = jnp.where(num_prunable, node_to_prune, 2 * split_tree.size)

    # growable leaves of the tree pruned at `node_to_prune`: start from the
    # current growable leaves, drop the two children that would be deleted, and
    # add `node_to_prune`, which becomes a growable leaf (it was a decision
    # node, so it has an admissible rule), with a single scatter on the stacked
    # [left, right, node] indices. Out-of-heap indices drop harmlessly.
    lrt_nodes = (node_to_prune << jnp.array([1, 1, 0])) | jnp.array([0, 1, 0])
    is_growable_leaf = affluence_tree.at[lrt_nodes].set(jnp.array([False, False, True]))
    distr_norm = jnp.sum(p_propose_grow, where=is_growable_leaf)
    prob_choose = p_propose_grow.at[node_to_prune].get(mode='fill', fill_value=0)
    prob_choose = prob_choose / jnp.where(distr_norm, distr_norm, 1)

    return node_to_prune, num_prunable, prob_choose


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
