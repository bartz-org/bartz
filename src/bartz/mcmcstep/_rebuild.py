# bartz/src/bartz/mcmcstep/_rebuild.py
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

"""Recompute the fields of the MCMC state that are derived from the others."""

from dataclasses import replace
from functools import partial

from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, Bool, Float, Int32, Shaped, UInt, UInt32

from bartz._jaxext import jit
from bartz.grove import evaluate_forest, is_actual_leaf, traverse_forest
from bartz.mcmcstep._axes import chain_vmap_axes
from bartz.mcmcstep._moves import _process_exclude, fully_used_variables
from bartz.mcmcstep._reduction import AutoBatchedReduction
from bartz.mcmcstep._state import Forest, State, StateT, outcome_target


@jit(donate_argnums=(0,))
def rebuild_derived(state: StateT) -> StateT:
    """Recompute the fields of `state` derived from its parameters and data.

    The parameters are the trees, the precisions, `Sparsity.log_s`,
    `Sparsity.theta`, and `State.z`; the data are `State.y` and the other fields
    fixed by `init`. The caches are rebuilt from scratch, any pending prune is
    dropped, and the per-step statistics are zeroed. The run history
    (`State.resid_eff_scale`, `State.resid_inexact_integral`,
    `StepConfig.steps_done`) is left untouched. The memory of the input state
    is re-used for the output state.
    """
    if state.config.data_sharded:
        msg = 'rebuilding a state with the data sharded is not supported'
        raise NotImplementedError(msg)
    if state.has_chains:
        axes = chain_vmap_axes(state)
        return vmap(rebuild_derived_chain, in_axes=(axes,), out_axes=axes)(state)
    else:
        return rebuild_derived_chain(state)


def rebuild_derived_chain(state: StateT) -> StateT:
    """Implement `rebuild_derived` on a state without a chain axis."""
    forest = state.forest
    num_trees, *_, tree_size = forest.leaf_tree.shape
    leaf_indices = traverse_forest(state.X, forest.var_tree, forest.split_tree)
    assert leaf_indices.dtype == forest.leaf_indices.dtype

    reduce = partial(
        AutoBatchedReduction()._reduce,  # noqa: SLF001
        size=tree_size,
        data_sharded=False,
    )
    if forest.count_tree is None:
        count_tree = None
    else:
        count_tree = vmap(partial(reduce, 1, dtype=jnp.uint32))(leaf_indices)
    if state.prec_scale is None:
        prec_tree = None
    else:
        prec_tree = vmap(partial(reduce, state.prec_scale, dtype=jnp.float32))(
            leaf_indices
        )

    affluence_tree = vmap(
        partial(
            affluent_leaves,
            max_split=forest.max_split,
            blocked_vars=forest.blocked_vars,
            min_points_per_decision_node=forest.min_points_per_decision_node,
        )
    )(forest.var_tree, forest.split_tree, count_tree)

    forest = replace(
        forest,
        leaf_indices=leaf_indices,
        affluence_tree=affluence_tree,
        count_tree=count_tree,
        prec_tree=prec_tree,
        to_prune=jnp.zeros(num_trees, bool),
        move_node=jnp.zeros(num_trees, jnp.int32),
        grow_prop_count=jnp.zeros_like(forest.grow_prop_count),
        prune_prop_count=jnp.zeros_like(forest.prune_prop_count),
        grow_acc_count=jnp.zeros_like(forest.grow_acc_count),
        prune_acc_count=jnp.zeros_like(forest.prune_acc_count),
        log_trans_prior=zeros_like_or_none(forest.log_trans_prior),
        log_likelihood=zeros_like_or_none(forest.log_likelihood),
    )
    return replace(state, forest=forest, resid=compute_resid(state, forest))


def zeros_like_or_none(
    x: Shaped[Array, '*shape'] | None,
) -> Shaped[Array, '*shape'] | None:
    """Return zeros shaped like `x`, or `None` if `x` is `None`."""
    return None if x is None else jnp.zeros_like(x)


def affluent_leaves(
    var_tree: UInt[Array, ' half_tree_size'],
    split_tree: UInt[Array, ' half_tree_size'],
    count_tree: UInt32[Array, ' 2*half_tree_size'] | None,
    *,
    max_split: UInt[Array, ' p'],
    blocked_vars: UInt[Array, ' q'] | None,
    min_points_per_decision_node: Int32[Array, ''] | None,
) -> Bool[Array, ' half_tree_size']:
    """Mark the leaves that can be grown, see `Forest.affluence_tree`."""
    (half_tree_size,) = split_tree.shape
    (p,) = max_split.shape

    def has_available_rule(node: Int32[Array, '']) -> Bool[Array, '']:
        exclude = fully_used_variables(var_tree, split_tree, max_split, node)
        if blocked_vars is not None:
            exclude = jnp.concatenate([exclude, blocked_vars])
        _, num_allowed = _process_exclude(p, exclude)
        return num_allowed > 0

    nodes = jnp.arange(1, half_tree_size)
    available = (
        jnp.zeros(half_tree_size, bool).at[1:].set(vmap(has_available_rule)(nodes))
    )
    out = is_actual_leaf(split_tree) & available
    if min_points_per_decision_node is not None:
        assert count_tree is not None
        out &= count_tree[:half_tree_size] >= min_points_per_decision_node
    return out


def compute_resid(
    state: State, forest: Forest
) -> Float[Array, ' n'] | Float[Array, ' k n']:
    """Compute `State.resid` from the data and the forest."""
    fit = evaluate_forest(state.X, forest, sum_batch_axis=-1)
    resid = outcome_target(state) - fit - forest.offset[..., None]
    return (resid / state.resid_unit[..., None]).astype(state.resid.dtype)
