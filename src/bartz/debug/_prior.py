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

import math
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import TypeVar

from equinox import Module
from jax import eval_shape, lax, random, tree, vmap
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, Bool, Float32, Int32, Key, PyTree, Shaped, UInt

from bartz._jaxext import jit, minimal_unsigned_dtype
from bartz._jaxext import split as split_key
from bartz._jaxext.random import loggamma
from bartz.grove import TreesTrace, is_actual_leaf, is_leaves_parent, traverse_forest
from bartz.mcmcstep._moves import randint_masked
from bartz.mcmcstep._state import chol_with_gersh

FloatLike = float | Float32[Array, '']


class SamplePriorStack(Module):
    """Represent the manually managed stack used in `sample_prior_trees`.

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


class TreeStructure(Module):
    """The decision rules of a tree, or of a batch of trees."""

    var_tree: UInt[Array, '*batch_shape half_tree_size']
    """The variables of the decision rules."""

    split_tree: UInt[Array, '*batch_shape half_tree_size']
    """The cutpoints of the decision rules, 0 at the leaves."""

    @classmethod
    def stump(
        cls, p_nonterminal: Float32[Array, ' d_minus_1'], max_split: UInt[Array, ' p']
    ) -> 'TreeStructure':
        """Create a tree with only the root."""
        half_tree_size = 2**p_nonterminal.size
        return cls(
            var_tree=jnp.zeros(
                half_tree_size, minimal_unsigned_dtype(max_split.size - 1)
            ),
            split_tree=jnp.zeros(half_tree_size, max_split.dtype),
        )


class SamplePriorCarry(Module):
    """Object holding values carried along the recursion in `sample_prior_trees`."""

    key: Key[Array, '']
    """A jax random key used to sample decision rules."""

    stack: SamplePriorStack
    """The stack used to manage the recursion."""

    trees: TreeStructure
    """The output arrays."""

    @classmethod
    def initial(
        cls,
        key: Key[Array, ''],
        p_nonterminal: Float32[Array, ' d_minus_1'],
        max_split: UInt[Array, ' p'],
    ) -> 'SamplePriorCarry':
        """Initialize the carry object.

        Parameters
        ----------
        key
            A jax random key.
        p_nonterminal
            The prior probability of a node being non-terminal conditional on
            its ancestors and on having available decision rules, at each depth.
        max_split
            The number of cutpoints along each variable.

        Returns
        -------
        A `SamplePriorCarry` initialized to start the recursion.
        """
        return cls(
            key,
            SamplePriorStack.initial(p_nonterminal, max_split),
            TreeStructure.stump(p_nonterminal, max_split),
        )


class SamplePriorX(Module):
    """Object representing the recursion scan in `sample_prior_trees`.

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


def sample_tree_structure(
    key: Key[Array, ''],
    max_split: UInt[Array, ' p'],
    p_nonterminal: Float32[Array, ' d_minus_1'],
    log_s: Float32[Array, ' p'] | None = None,
) -> TreeStructure:
    """Sample the decision rules of a tree from the BART prior.

    Parameters
    ----------
    key
        A jax random key.
    max_split
        The maximum split value for each variable.
    p_nonterminal
        The prior probability of a node being non-terminal conditional on
        its ancestors and on having available decision rules, at each depth.
    log_s
        The logarithm of the unnormalized prior probability of splitting on each
        variable. If `None`, variables are chosen uniformly at random.

    Returns
    -------
    The decision rules of the tree.
    """
    carry = SamplePriorCarry.initial(key, p_nonterminal, max_split)
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


T = TypeVar('T')


class RejectionCriterion(Module):
    """A criterion to accept or reject the samples in `rejection_sample`."""

    @abstractmethod
    def accept(self, sample: PyTree[Array]) -> Bool[Array, '']:
        """Whether to accept a single sample."""


class RejectionCarry(Module):
    """The loop state of `rejection_sample`."""

    samples: PyTree[Array]
    """The accepted samples, stacked along a leading axis."""

    missing: Bool[Array, ' num']
    """Which slots of `samples` are still empty."""

    iteration: Int32[Array, '']
    """The number of iterations done."""


def rejection_sample(
    key: Key[Array, ''],
    draw: Callable[[Key[Array, ''], int], T],
    criterion: RejectionCriterion,
    num: int,
) -> T:
    """Draw independent samples conditional on a criterion.

    Parameters
    ----------
    key
        A jax random key.
    draw
        A function taking a key and a number of samples, that draws that many
        independent samples stacked along a leading axis.
    criterion
        The criterion the samples must satisfy.
    num
        The number of samples.

    Returns
    -------
    The accepted samples, stacked along a leading axis.

    Notes
    -----
    Each iteration draws `num` candidates and fills the empty slots with the
    accepted ones. The candidates are independent and identically
    distributed, so the slots can take them in any order that depends only on
    which are accepted.
    """

    def body(carry: RejectionCarry) -> RejectionCarry:
        candidates = draw(random.fold_in(key, carry.iteration), num)
        accepted = vmap(criterion.accept)(candidates)

        # the i-th empty slot takes the i-th accepted candidate
        (source,) = jnp.nonzero(accepted, size=num, fill_value=0)
        rank = jnp.cumsum(carry.missing) - 1
        fill = carry.missing & (rank < jnp.count_nonzero(accepted))
        source = source[jnp.clip(rank, 0, num - 1)]

        def update(
            old: Shaped[Array, ' num *shape'], new: Shaped[Array, ' num *shape']
        ) -> Shaped[Array, ' num *shape']:
            mask = fill.reshape(num, *(1,) * (old.ndim - 1))
            return jnp.where(mask, new[source], old)

        return RejectionCarry(
            samples=tree.map(update, carry.samples, candidates),
            missing=carry.missing & ~fill,
            iteration=carry.iteration + 1,
        )

    shapes = eval_shape(lambda key: draw(key, num), key)
    carry = RejectionCarry(
        samples=tree.map(lambda s: jnp.zeros(s.shape, s.dtype), shapes),
        missing=jnp.ones(num, bool),
        iteration=jnp.int32(0),
    )
    carry = lax.while_loop(lambda carry: jnp.any(carry.missing), body, carry)
    return carry.samples


class MinPointsCriterion(RejectionCriterion):
    """Accept the trees with enough datapoints in their nodes."""

    X: UInt[Array, 'p n']
    """The predictors of the datapoints."""

    min_points_per_decision_node: Int32[Array, ''] | None
    """The minimum number of datapoints in a decision node."""

    min_points_per_leaf: Int32[Array, ''] | None
    """The minimum number of datapoints in a leaf, but the root."""

    def accept(self, sample: TreeStructure) -> Bool[Array, '']:
        """Check the tree has enough datapoints in its nodes."""
        (half_tree_size,) = sample.split_tree.shape
        leaf_indices = traverse_forest(self.X, sample.var_tree, sample.split_tree)
        count = jnp.zeros(2 * half_tree_size, jnp.int32).at[leaf_indices].add(1)

        ok = jnp.bool_(True)
        if self.min_points_per_decision_node is not None:
            # any decision node has at least as many datapoints as the
            # decision nodes below it, so check only the parents of two leaves
            parent_count = count.reshape(half_tree_size, 2).sum(axis=1)
            enough = parent_count >= self.min_points_per_decision_node
            ok &= jnp.all(enough | ~is_leaves_parent(sample.split_tree))
        if self.min_points_per_leaf is not None:
            is_leaf = is_actual_leaf(sample.split_tree, add_bottom_level=True)
            is_child = is_leaf.at[1].set(False)
            ok &= jnp.all((count >= self.min_points_per_leaf) | ~is_child)
        return ok


def sample_forest_structure(
    key: Key[Array, ''],
    shape: tuple[int, ...],
    *,
    max_split: UInt[Array, ' p'],
    p_nonterminal: Float32[Array, ' d_minus_1'],
    log_s: Float32[Array, ' p'] | Float32[Array, '*batch p'] | None = None,
    criterion: RejectionCriterion | None = None,
) -> TreeStructure:
    """Sample the decision rules of independent trees from the BART prior.

    Parameters
    ----------
    key
        A jax random key.
    shape
        The shape ``(*batch, num_trees)`` of the batch of trees.
    max_split
        The maximum split value for each variable.
    p_nonterminal
        The prior probability of a node being non-terminal conditional on
        its ancestors and on having available decision rules, at each depth.
    log_s
        The logarithm of the unnormalized prior probability of splitting on each
        variable, either shared or with shape ``(*batch, p)``. If `None`,
        variables are chosen uniformly at random.
    criterion
        If set, the trees are conditioned on satisfying it. This requires them
        to be identically distributed, so `log_s` must be shared.

    Returns
    -------
    The decision rules of the trees, with batch shape `shape`.

    Raises
    ------
    ValueError
        If `criterion` is set and `log_s` is not shared.
    """
    sample_one = partial(
        sample_tree_structure, max_split=max_split, p_nonterminal=p_nonterminal
    )

    if criterion is None:

        def onetree(
            key: Key[Array, ''], log_s: Float32[Array, ' p'] | None
        ) -> tuple[UInt[Array, ' half_tree_size'], UInt[Array, ' half_tree_size']]:
            tree = sample_one(key, log_s=log_s)
            return tree.var_tree, tree.split_tree

        if log_s is None:
            signature = '()->(h),(h)'
        else:
            signature = '(),(p)->(h),(h)'
            log_s = log_s[..., None, :]  # broadcast over the trees

        keys = random.split(key, shape)
        var_tree, split_tree = jnp.vectorize(
            onetree, excluded=set() if log_s is not None else {1}, signature=signature
        )(keys, log_s)
        return TreeStructure(var_tree=var_tree, split_tree=split_tree)

    elif log_s is not None and log_s.ndim > 1:
        msg = 'conditioning on a criterion requires a shared `log_s`'
        raise ValueError(msg)

    else:

        def draw(key: Key[Array, ''], num: int) -> TreeStructure:
            return vmap(partial(sample_one, log_s=log_s))(random.split(key, num))

        trees = rejection_sample(key, draw, criterion, math.prod(shape))
        return tree.map(lambda x: x.reshape(*shape, *x.shape[1:]), trees)


def sample_leaves(
    key: Key[Array, ''],
    shape: tuple[int, ...],
    prec_chol: Float32[Array, ''] | Float32[Array, 'k k'],
) -> Float32[Array, '*shape']:
    """Sample leaves from their prior, a centered Normal.

    Parameters
    ----------
    key
        A jax random key.
    shape
        The shape of the leaves, ``(*batch, tree_size)``, or ``(*batch, k,
        tree_size)`` for vector-valued leaves.
    prec_chol
        The lower Cholesky factor of the prior precision of a leaf, a scalar
        square root for scalar leaves.

    Returns
    -------
    The leaves.
    """
    eps = random.normal(key, shape)
    if prec_chol.ndim == 0:
        return eps / prec_chol
    else:
        # solve L^T x = eps, such that cov(x) = (L L^T)^-1
        solve = jnp.vectorize(
            partial(solve_triangular, lower=True, trans='T'),
            signature='(k,k),(k,m)->(k,m)',
        )
        return solve(prec_chol, eps)


def sample_wishart(
    key: Key[Array, ''],
    nu: Float32[Array, ''],
    rate: Float32[Array, ''] | Float32[Array, 'k k'],
) -> tuple[
    Float32[Array, ''] | Float32[Array, 'k k'],
    Float32[Array, ''] | Float32[Array, 'k k'],
]:
    """Sample a precision matrix from its Wishart prior.

    Parameters
    ----------
    key
        A jax random key.
    nu
        The degrees of freedom.
    rate
        The rate matrix, such that the mean is ``nu * rate^-1``. A scalar for
        a scalar precision, which then is Gamma distributed.

    Returns
    -------
    value : Float32[Array, ''] | Float32[Array, 'k k']
        The precision.
    chol : Float32[Array, ''] | Float32[Array, 'k k']
        Its lower Cholesky factor, a square root for a scalar precision.
    """
    if rate.ndim == 0:
        value = 2 * jnp.exp(loggamma(key, nu / 2)) / rate
        return value, jnp.sqrt(value)
    else:
        # Bartlett decomposition: value = (L A) (L A)^T, with L L^T = rate^-1
        keys = split_key(key)
        k, _ = rate.shape
        diag = jnp.sqrt(2 * jnp.exp(loggamma(keys.pop(), (nu - jnp.arange(k)) / 2)))
        A = jnp.tril(random.normal(keys.pop(), (k, k)), -1) + jnp.diag(diag)
        # with rate = U U^T, U upper triangular, L = U^-T is lower triangular;
        # U is the lower Cholesky factor with the axes reversed
        U = chol_with_gersh(rate[::-1, ::-1])[::-1, ::-1]
        chol = solve_triangular(U, A, lower=False, trans='T')
        return chol @ chol.T, chol


def sample_diag_wishart(
    key: Key[Array, ''], nu: Float32[Array, ''], rate: Float32[Array, 'k k']
) -> tuple[Float32[Array, 'k k'], Float32[Array, 'k k']]:
    """Sample a diagonal precision matrix with independent Gamma components.

    Parameters
    ----------
    key
        A jax random key.
    nu
        Twice the shape parameter of the Gamma distributions.
    rate
        Twice the rate parameters of the Gamma distributions on the diagonal.

    Returns
    -------
    value : Float32[Array, 'k k']
        The precision.
    chol : Float32[Array, 'k k']
        Its lower Cholesky factor.
    """
    rate = jnp.diag(rate)
    value = 2 * jnp.exp(loggamma(key, nu / 2, rate.shape)) / rate
    return jnp.diag(value), jnp.diag(jnp.sqrt(value))


class PriorSample(TreesTrace):
    """Output of `sample_prior_trees`."""

    # defaults because the inherited `leaf_unit`/`offset` fields have them
    log_s: Float32[Array, 'trace_length p'] | None = None
    """The per-iteration log unnormalized pmf for choosing variables to split on,
    `None` means uniform distribution."""

    theta: Float32[Array, ' trace_length'] | None = None
    """The per-iteration Dirichlet concentration, `None` if `s` is not drawn from
    a Dirichlet prior."""


@jit(static_argnames=('trace_length', 'num_trees'))
def sample_prior_trees(
    key: Key[Array, ''],
    trace_length: int,
    num_trees: int,
    *,
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
    keys = split_key(key, 3)
    p = max_split.size

    log_s, theta = sample_sparsity(
        keys.pop(), trace_length, p, log_s=log_s, theta=theta, a=a, b=b, rho=rho
    )

    trees = sample_forest_structure(
        keys.pop(),
        (trace_length, num_trees),
        max_split=max_split,
        p_nonterminal=p_nonterminal,
        log_s=log_s,
    )
    tree_size = 2 ** (p_nonterminal.size + 1)
    leaf_tree = sample_leaves(
        keys.pop(), (trace_length, num_trees, tree_size), jnp.reciprocal(sigma_mu)
    )

    return PriorSample(
        leaf_tree=leaf_tree,
        var_tree=trees.var_tree,
        split_tree=trees.split_tree,
        log_s=log_s,
        theta=theta,
    )


def sample_sparsity(
    key: Key[Array, ''],
    trace_length: int,
    p: int,
    *,
    log_s: Float32[Array, ' p'] | Float32[Array, 'trace_length p'] | None,
    theta: FloatLike | None,
    a: FloatLike | None,
    b: FloatLike | None,
    rho: FloatLike | None,
) -> tuple[
    Float32[Array, 'trace_length p'] | None, Float32[Array, ' trace_length'] | None
]:
    """Resolve the per-iteration `log_s` and `theta` of the sparsity prior of `sample_prior_trees`."""
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
    *,
    log_range: FloatLike | None = None,
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
    log_range
        If set, the prior is restricted to :math:`|\log(\theta/\rho)| \le`
        `log_range`.

    Returns
    -------
    The sampled concentration parameter(s).
    """

    # log(theta / rho) is the log ratio of two Gamma variates, drawn in log
    # space such that theta does not underflow
    def draw(key: Key[Array, ''], num: int) -> Float32[Array, ' num']:
        ab = jnp.stack([jnp.asarray(a), jnp.asarray(b)])
        log_ga, log_gb = loggamma(key, ab[:, None], (2, num))
        return log_ga - log_gb

    num = math.prod(shape)
    if log_range is None:
        v = draw(key, num)
    else:
        v = rejection_sample(key, draw, WithinRange(jnp.asarray(log_range)), num)
    return rho * jnp.exp(v.reshape(shape))


class WithinRange(RejectionCriterion):
    """Accept x in [-bound, bound]."""

    bound: Float32[Array, '']
    """The bound (inclusive) on the absolute value."""

    def accept(self, sample: Float32[Array, '']) -> Bool[Array, '']:
        """Check the sample is within the range."""
        return jnp.abs(sample) <= self.bound


def sample_s(
    key: Key[Array, ''], theta: Float32[Array, ' *batch'] | FloatLike, p: int
) -> Float32[Array, '*batch {p}']:
    r"""Sample the log variable selection probabilities from their prior.

    The prior is :math:`s \sim \mathrm{Dirichlet}(\theta/p, \ldots, \theta/p)`.
    The result is `log_s`, the log of unnormalized weights whose `softmax` is
    `s`, the representation expected by `sample_prior_trees`.

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
