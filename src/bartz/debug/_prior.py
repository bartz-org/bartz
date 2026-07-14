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

from equinox import Module
from jax import lax, random
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float32, Int32, Key, UInt

from bartz._jaxext import jit, minimal_unsigned_dtype
from bartz._jaxext import split as split_key
from bartz._jaxext.random import loggamma
from bartz.grove import TreesTrace
from bartz.mcmcstep._moves import randint_masked

FloatLike = float | Float32[Array, '']


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
    log_s: Float32[Array, ' p'] | None = None,
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
    log_s
        The logarithm of the unnormalized prior probability of splitting on each
        variable. If `None`, variables are chosen uniformly at random.

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

        # sample a random decision rule, drawing the variable from `s`
        # (uniformly if `log_s` is None) among those with an available cutpoint
        available: Bool[Array, ' p'] = lower < upper
        allowed = jnp.any(available)
        if log_s is None:
            var = randint_masked(keys.pop(), available)
        else:
            logits = jnp.where(available, log_s, jnp.finfo(log_s.dtype).min)
            var = random.categorical(keys.pop(), logits)
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
    keys: Key[Array, '*batch num_trees'],
    max_split: UInt[Array, ' p'],
    p_nonterminal: Float32[Array, ' d_minus_1'],
    sigma_mu: Float32[Array, ''],
    log_s: Float32[Array, ' p'] | Float32[Array, '*batch p'] | None = None,
) -> TreesTrace:
    """Sample forests of trees from the BART prior, batched over extra axes.

    Parameters
    ----------
    keys
        Random keys with shape ``(*batch, num_trees)``. The last axis indexes the
        trees of a forest; the leading axes are mapped over independently.
    max_split
        The maximum split value for each variable.
    p_nonterminal
        The prior probability of a node being non-terminal conditional on
        its ancestors and on having available decision rules, at each depth.
    sigma_mu
        The prior standard deviation of each leaf.
    log_s
        The logarithm of the unnormalized prior probability of splitting on each
        variable, shape ``(*batch, p)``: one vector per forest, shared across its
        trees. If `None`, variables are chosen uniformly at random.

    Returns
    -------
    An object containing the generated trees, with batch shape ``(*batch, num_trees)``.
    """

    def onetree(
        key: Key[Array, ''],
        max_split: UInt[Array, ' p'],
        p_nonterminal: Float32[Array, ' d_minus_1'],
        sigma_mu: Float32[Array, ''],
        log_s: Float32[Array, ' p'] | None,
    ) -> tuple[
        UInt[Array, ' half_tree_size'],
        UInt[Array, ' half_tree_size'],
        Float32[Array, ' tree_size'],
    ]:
        """Convert result to tuple to use jnp.vectorize."""
        trees = sample_prior_onetree(key, max_split, p_nonterminal, sigma_mu, log_s)
        return trees.var_tree, trees.split_tree, trees.leaf_tree

    # vectorize over trees
    over_trees = jnp.vectorize(
        onetree, excluded={1, 2, 3, 4}, signature='()->(h),(h),(l)'
    )

    # vectorize over batching dims
    if log_s is None:
        excluded = {1, 2, 3, 4}
        signature = '(t)->(t,h),(t,h),(t,l)'
    else:
        excluded = {1, 2, 3}
        signature = '(t),(p)->(t,h),(t,h),(t,l)'
    batched = jnp.vectorize(over_trees, excluded=excluded, signature=signature)

    # invoke and pack result
    var_tree, split_tree, leaf_tree = batched(
        keys, max_split, p_nonterminal, sigma_mu, log_s
    )
    return TreesTrace(var_tree=var_tree, split_tree=split_tree, leaf_tree=leaf_tree)


class PriorSample(TreesTrace):
    """Output of `sample_prior`."""

    # defaults because the inherited `leaf_unit`/`offset` fields have them
    log_s: Float32[Array, 'trace_length p'] | None = None
    """The per-iteration log unnormalized pmf for choosing variables to split on,
    `None` means uniform distribution."""

    theta: Float32[Array, ' trace_length'] | None = None
    """The per-iteration Dirichlet concentration, `None` if `s` is not drawn from
    a Dirichlet prior."""


@jit(static_argnums=(1, 2))
def sample_prior(
    key: Key[Array, ''],
    trace_length: int,
    num_trees: int,
    max_split: UInt[Array, ' p'],
    p_nonterminal: Float32[Array, ' d_minus_1'],
    sigma_mu: Float32[Array, ''],
    log_s: Float32[Array, ' p'] | Float32[Array, 'trace_length p'] | None = None,
    theta: FloatLike | None = None,
    a: FloatLike | None = None,
    b: FloatLike | None = None,
    rho: FloatLike | None = None,
) -> PriorSample:
    r"""Sample independent trees from the BART prior.

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
    log_s
        The logarithm of the unnormalized prior probability of splitting on each
        variable, either shared across all trees (shape ``(p,)``) or specified
        per iteration (shape ``(trace_length, p)``). If `None`, variables are
        chosen uniformly at random. Mutually exclusive with `theta`, `a`, `b`,
        `rho`, which instead sample it from its prior.
    theta
        The Dirichlet concentration parameter. If set, and `rho`, `a`, `b` are
        not, `log_s` is sampled with this fixed concentration.
    a
    b
    rho
        Parameters of the prior :math:`\theta/(\theta+\rho) \sim
        \mathrm{Beta}(a, b)`. If all set, both `theta` and `log_s` are sampled.

    Returns
    -------
    A trace of trees sampled from the prior.
    """
    keys = split_key(key)
    p = max_split.size

    log_s, theta = sample_sparsity(keys.pop(), trace_length, p, log_s, theta, a, b, rho)

    tree_keys = random.split(keys.pop(), trace_length * num_trees)
    tree_keys = tree_keys.reshape(trace_length, num_trees)
    trees = sample_prior_forest(tree_keys, max_split, p_nonterminal, sigma_mu, log_s)

    return PriorSample(
        leaf_tree=trees.leaf_tree,
        var_tree=trees.var_tree,
        split_tree=trees.split_tree,
        log_s=log_s,
        theta=theta,
    )


def sample_sparsity(
    key: Key[Array, ''],
    trace_length: int,
    p: int,
    log_s: Float32[Array, ' p'] | Float32[Array, 'trace_length p'] | None,
    theta: FloatLike | None,
    a: FloatLike | None,
    b: FloatLike | None,
    rho: FloatLike | None,
) -> tuple[
    Float32[Array, 'trace_length p'] | None, Float32[Array, ' trace_length'] | None
]:
    """Resolve the per-iteration `log_s` and `theta` of the sparsity prior of `sample_prior`."""
    if rho is not None or a is not None or b is not None:
        # 2-level prior: draw theta from its Beta prior, then log_s | theta
        if rho is None or a is None or b is None:
            msg = 'rho, a, b must be either all set or all None'
            raise ValueError(msg)
        if theta is not None or log_s is not None:
            msg = 'theta and log_s are sampled from the prior, do not also set them'
            raise ValueError(msg)
        keys = split_key(key)
        theta = sample_theta(keys.pop(), rho, a, b, (trace_length,))
        log_s = sample_s(keys.pop(), theta, p)
    elif theta is not None:
        # 1-level prior: draw log_s from the Dirichlet with a fixed concentration
        if log_s is not None:
            msg = 'log_s is sampled from the prior, do not also set it'
            raise ValueError(msg)
        theta = jnp.broadcast_to(jnp.asarray(theta), (trace_length,))
        log_s = sample_s(key, theta, p)
    elif log_s is not None:
        # fixed split probabilities, shared across the iterations
        theta = None
        log_s = jnp.broadcast_to(log_s, (trace_length, p))
    else:
        # uniform variable selection
        theta = None
        log_s = None
    return log_s, theta


def sample_theta(
    key: Key[Array, ''],
    rho: FloatLike,
    a: FloatLike,
    b: FloatLike,
    shape: tuple[int, ...] = (),
) -> Float32[Array, ' *shape']:
    r"""Sample the Dirichlet concentration `theta` from its prior.

    The prior is :math:`\theta/(\theta+\rho) \sim \mathrm{Beta}(a, b)`.

    Parameters
    ----------
    key
        A jax random key.
    rho
        The scale of `theta`.
    a
    b
        The shape parameters of the Beta prior on :math:`\theta/(\theta+\rho)`.
    shape
        The shape of the sample. A scalar by default.

    Returns
    -------
    The sampled concentration parameter(s).
    """
    lambda_ = random.beta(key, a, b, shape)
    return rho * lambda_ / (1 - lambda_)


def sample_s(
    key: Key[Array, ''], theta: Float32[Array, ' *batch'] | FloatLike, p: int
) -> Float32[Array, '*batch {p}']:
    r"""Sample the log variable selection probabilities from their prior.

    The prior is :math:`s \sim \mathrm{Dirichlet}(\theta/p, \ldots, \theta/p)`.
    The result is `log_s`, the log of unnormalized weights whose `softmax` is
    `s`, the representation expected by `sample_prior`.

    Parameters
    ----------
    key
        A jax random key.
    theta
        The Dirichlet concentration parameter. Batched values draw an
        independent `s` for each entry.
    p
        The number of variables.

    Returns
    -------
    The log unnormalized variable selection probabilities.
    """
    theta = jnp.asarray(theta)
    alpha = jnp.broadcast_to(theta[..., None] / p, (*theta.shape, p))
    return loggamma(key, alpha)
