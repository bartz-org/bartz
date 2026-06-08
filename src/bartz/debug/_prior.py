# bartz/src/bartz/debug/_prior.py
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

"""Sampling from the BART prior."""

from dataclasses import replace
from functools import partial

from equinox import Module
from jax import lax, random
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float32, Int32, Key, UInt

from bartz._jaxext import jit, minimal_unsigned_dtype, vmap_nodoc
from bartz._jaxext import split as split_key
from bartz.grove import TreesTrace
from bartz.mcmcstep._moves import randint_masked


class SamplePriorStack(Module):
    """Represent the manually managed stack used in `sample_prior`.

    Each level of the stack represents a recursion into a child node in a
    binary tree of maximum depth `d`.
    """

    nonterminal: Bool[Array, ' d_minus_1']
    """Whether the node is valid or the recursion is into unused node slots."""

    lower: UInt[Array, 'd_minus_1 p']
    """The available cutpoints along ``var`` are in the integer range
    ``[1 + lower[var], 1 + upper[var])``."""

    upper: UInt[Array, 'd_minus_1 p']
    """The available cutpoints along ``var`` are in the integer range
    ``[1 + lower[var], 1 + upper[var])``."""

    var: UInt[Array, ' d_minus_1']
    """The variable of a decision node."""

    split: UInt[Array, ' d_minus_1']
    """The cutpoint of a decision node."""

    @classmethod
    def initial(
        cls, p_nonterminal: Float32[Array, ' d_minus_1'], max_split: UInt[Array, ' p']
    ) -> 'SamplePriorStack':
        """Initialize the stack.

        Parameters
        ----------
        p_nonterminal
            The prior probability of a node being non-terminal conditional on
            its ancestors and on having available decision rules, at each depth.
        max_split
            The number of cutpoints along each variable.

        Returns
        -------
        A `SamplePriorStack` initialized to start the recursion.
        """
        var_dtype = minimal_unsigned_dtype(max_split.size - 1)
        return cls(
            nonterminal=jnp.ones(p_nonterminal.size, bool),
            lower=jnp.zeros((p_nonterminal.size, max_split.size), max_split.dtype),
            upper=jnp.broadcast_to(max_split, (p_nonterminal.size, max_split.size)),
            var=jnp.zeros(p_nonterminal.size, var_dtype),
            split=jnp.zeros(p_nonterminal.size, max_split.dtype),
        )


def _initial_trees(
    key: Key[Array, ''],
    sigma_mu: Float32[Array, ''],
    p_nonterminal: Float32[Array, ' d_minus_1'],
    max_split: UInt[Array, ' p'],
) -> TreesTrace:
    """Create a single tree with random leaves and empty structure.

    The leaves are already correct and do not need to be changed.

    Parameters
    ----------
    key
        A jax random key.
    sigma_mu
        The prior standard deviation of each leaf.
    p_nonterminal
        The prior probability of a node being non-terminal conditional on
        its ancestors and on having available decision rules, at each depth.
    max_split
        The number of cutpoints along each variable.

    Returns
    -------
    Trees initialized with random leaves and stub tree structures.
    """
    heap_size = 2 ** (p_nonterminal.size + 1)
    return TreesTrace(
        leaf_tree=sigma_mu * random.normal(key, (heap_size,)),
        var_tree=jnp.zeros(
            heap_size // 2, dtype=minimal_unsigned_dtype(max_split.size - 1)
        ),
        split_tree=jnp.zeros(heap_size // 2, dtype=max_split.dtype),
    )


class SamplePriorCarry(Module):
    """Object holding values carried along the recursion in `sample_prior`."""

    key: Key[Array, '']
    """A jax random key used to sample decision rules."""

    stack: SamplePriorStack
    """The stack used to manage the recursion."""

    trees: TreesTrace
    """The output arrays."""

    @classmethod
    def initial(
        cls,
        key: Key[Array, ''],
        sigma_mu: Float32[Array, ''],
        p_nonterminal: Float32[Array, ' d_minus_1'],
        max_split: UInt[Array, ' p'],
    ) -> 'SamplePriorCarry':
        """Initialize the carry object.

        Parameters
        ----------
        key
            A jax random key.
        sigma_mu
            The prior standard deviation of each leaf.
        p_nonterminal
            The prior probability of a node being non-terminal conditional on
            its ancestors and on having available decision rules, at each depth.
        max_split
            The number of cutpoints along each variable.

        Returns
        -------
        A `SamplePriorCarry` initialized to start the recursion.
        """
        keys = split_key(key)
        return cls(
            keys.pop(),
            SamplePriorStack.initial(p_nonterminal, max_split),
            _initial_trees(keys.pop(), sigma_mu, p_nonterminal, max_split),
        )


class SamplePriorX(Module):
    """Object representing the recursion scan in `sample_prior`.

    The sequence of nodes to visit is pre-computed recursively once, unrolling
    the recursion schedule.
    """

    node: Int32[Array, ' half_tree_size_minus_1']
    """The heap index of the node to visit."""

    depth: Int32[Array, ' half_tree_size_minus_1']
    """The depth of the node."""

    next_depth: Int32[Array, ' half_tree_size_minus_1']
    """The depth of the next node to visit, either the left child or the right
    sibling of the node or of an ancestor."""

    @classmethod
    def initial(cls, p_nonterminal: Float32[Array, ' d_minus_1']) -> 'SamplePriorX':
        """Initialize the sequence of nodes to visit.

        Parameters
        ----------
        p_nonterminal
            The prior probability of a node being non-terminal conditional on
            its ancestors and on having available decision rules, at each depth.

        Returns
        -------
        A `SamplePriorX` initialized with the sequence of nodes to visit.
        """
        seq = cls._sequence(p_nonterminal.size)
        assert len(seq) == 2**p_nonterminal.size - 1
        node = [node for node, depth in seq]
        depth = [depth for node, depth in seq]
        next_depth = [*depth[1:], p_nonterminal.size]
        return cls(
            node=jnp.array(node),
            depth=jnp.array(depth),
            next_depth=jnp.array(next_depth),
        )

    @classmethod
    def _sequence(
        cls, max_depth: int, depth: int = 0, node: int = 1
    ) -> tuple[tuple[int, int], ...]:
        """Recursively generate a sequence [(node, depth), ...]."""
        if depth < max_depth:
            out = ((node, depth),)
            out += cls._sequence(max_depth, depth + 1, 2 * node)
            out += cls._sequence(max_depth, depth + 1, 2 * node + 1)
            return out
        return ()


def sample_prior_onetree(
    key: Key[Array, ''],
    max_split: UInt[Array, ' p'],
    p_nonterminal: Float32[Array, ' d_minus_1'],
    sigma_mu: Float32[Array, ''],
) -> TreesTrace:
    """Sample a tree from the BART prior.

    Parameters
    ----------
    key
        A jax random key.
    max_split
        The maximum split value for each variable.
    p_nonterminal
        The prior probability of a node being non-terminal conditional on
        its ancestors and on having available decision rules, at each depth.
    sigma_mu
        The prior standard deviation of each leaf.

    Returns
    -------
    An object containing a generated tree.
    """
    carry = SamplePriorCarry.initial(key, sigma_mu, p_nonterminal, max_split)
    xs = SamplePriorX.initial(p_nonterminal)

    def loop(carry: SamplePriorCarry, x: SamplePriorX) -> tuple[SamplePriorCarry, None]:
        keys = split_key(carry.key, 4)

        # get variables at current stack level
        stack = carry.stack
        nonterminal = stack.nonterminal[x.depth]
        lower = stack.lower[x.depth, :]
        upper = stack.upper[x.depth, :]

        # sample a random decision rule
        available: Bool[Array, ' p'] = lower < upper
        allowed = jnp.any(available)
        var = randint_masked(keys.pop(), available)
        split = 1 + random.randint(keys.pop(), (), lower[var], upper[var])

        # cast to shorter integer types
        var = var.astype(carry.trees.var_tree.dtype)
        split = split.astype(carry.trees.split_tree.dtype)

        # decide whether to try to grow the node if it is growable
        pnt = p_nonterminal[x.depth]
        try_nonterminal: Bool[Array, ''] = random.bernoulli(keys.pop(), pnt)
        nonterminal &= try_nonterminal & allowed

        # update trees
        trees = carry.trees
        trees = replace(
            trees,
            var_tree=trees.var_tree.at[x.node].set(var),
            split_tree=trees.split_tree.at[x.node].set(
                jnp.where(nonterminal, split, 0)
            ),
        )

        def write_push_stack() -> SamplePriorStack:
            """Update the stack to go to the left child."""
            return replace(
                stack,
                nonterminal=stack.nonterminal.at[x.next_depth].set(nonterminal),
                lower=stack.lower.at[x.next_depth, :].set(lower),
                upper=stack.upper.at[x.next_depth, :].set(upper.at[var].set(split - 1)),
                var=stack.var.at[x.depth].set(var),
                split=stack.split.at[x.depth].set(split),
            )

        def pop_push_stack() -> SamplePriorStack:
            """Update the stack to go to the right sibling, possibly at lower depth."""
            var = stack.var[x.next_depth - 1]
            split = stack.split[x.next_depth - 1]
            lower = stack.lower[x.next_depth - 1, :]
            upper = stack.upper[x.next_depth - 1, :]
            return replace(
                stack,
                lower=stack.lower.at[x.next_depth, :].set(lower.at[var].set(split)),
                upper=stack.upper.at[x.next_depth, :].set(upper),
            )

        # update stack
        stack = lax.cond(x.next_depth > x.depth, write_push_stack, pop_push_stack)

        # update carry
        carry = replace(carry, key=keys.pop(), stack=stack, trees=trees)
        return carry, None

    carry, _ = lax.scan(loop, carry, xs)
    return carry.trees


def sample_prior_forest(
    keys: Key[Array, ' num_trees'],
    max_split: UInt[Array, ' p'],
    p_nonterminal: Float32[Array, ' d_minus_1'],
    sigma_mu: Float32[Array, ''],
) -> TreesTrace:
    """Sample a set of independent trees from the BART prior.

    Parameters
    ----------
    keys
        A sequence of jax random keys, one for each tree. This determined the
        number of trees sampled.
    max_split
        The maximum split value for each variable.
    p_nonterminal
        The prior probability of a node being non-terminal conditional on
        its ancestors and on having available decision rules, at each depth.
    sigma_mu
        The prior standard deviation of each leaf.

    Returns
    -------
    An object containing the generated trees.
    """
    var_tree, split_tree, leaf_tree = _sample_prior_forest(
        keys, max_split, p_nonterminal, sigma_mu
    )
    return TreesTrace(var_tree=var_tree, split_tree=split_tree, leaf_tree=leaf_tree)


@partial(vmap_nodoc, in_axes=(0, None, None, None))
def _sample_prior_forest(
    key: Key[Array, ''],
    max_split: UInt[Array, ' p'],
    p_nonterminal: Float32[Array, ' d_minus_1'],
    sigma_mu: Float32[Array, ''],
) -> tuple[
    UInt[Array, ' half_tree_size'],
    UInt[Array, ' half_tree_size'],
    Float32[Array, ' 2*half_tree_size'],
]:
    """Implement `sample_prior_forest` for a single tree.

    The heaps are returned as a bare tuple to keep the constants `leaf_scale`
    and `offset` out of the vmapped outputs.
    """
    trees = sample_prior_onetree(key, max_split, p_nonterminal, sigma_mu)
    return trees.var_tree, trees.split_tree, trees.leaf_tree


@jit(static_argnums=(1, 2))
def sample_prior(
    key: Key[Array, ''],
    trace_length: int,
    num_trees: int,
    max_split: UInt[Array, ' p'],
    p_nonterminal: Float32[Array, ' d_minus_1'],
    sigma_mu: Float32[Array, ''],
) -> TreesTrace:
    """Sample independent trees from the BART prior.

    Parameters
    ----------
    key
        A jax random key.
    trace_length
        The number of iterations.
    num_trees
        The number of trees for each iteration.
    max_split
        The number of cutpoints along each variable.
    p_nonterminal
        The prior probability of a node being non-terminal conditional on
        its ancestors and on having available decision rules, at each depth.
        This determines the maximum depth of the trees.
    sigma_mu
        The prior standard deviation of each leaf.

    Returns
    -------
    An object containing the generated trees, with batch shape (trace_length, num_trees).
    """
    keys = random.split(key, trace_length * num_trees)
    trees = sample_prior_forest(keys, max_split, p_nonterminal, sigma_mu)
    reshape = lambda x: x.reshape(trace_length, num_trees, -1)
    return TreesTrace(
        var_tree=reshape(trees.var_tree),
        split_tree=reshape(trees.split_tree),
        leaf_tree=reshape(trees.leaf_tree),
    )
