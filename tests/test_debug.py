# bartz/tests/test_debug.py
#
# Copyright (c) 2025-2026, The Bartz Contributors
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

"""Test `bartz.debug`."""

from collections import namedtuple
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy
import pytest
from equinox import tree_at
from jax import numpy as jnp
from jax import random, vmap
from jax.nn import softmax
from jax.scipy.special import xlogy
from jax.scipy.stats import chi2
from jaxtyping import Array, Bool, Float, Float32, Int32, Key, Shaped, UInt8
from scipy import stats
from scipy.stats import ks_1samp, ks_2samp

from bartz._jaxext import get_device_count, minimal_unsigned_dtype, split
from bartz.debug import (
    BARTTraceMeta,
    MinimalTrace,
    sample_data,
    sample_prior,
    trees_BART_to_bartz,
)
from bartz.debug._prior import (
    MinPointsCriterion,
    RejectionCriterion,
    TreeStructure,
    rejection_sample,
    sample_diag_wishart,
    sample_forest_structure,
    sample_leaves,
    sample_s,
    sample_theta,
    sample_wishart,
)
from bartz.grove import TreesTrace, check_trace, describe_error, format_tree
from bartz.grove._check import check_tree
from bartz.mcmcloop import compute_varcount, evaluate_trace
from bartz.mcmcstep import DiagWishart, State, Wishart, init, make_p_nonterminal
from bartz.mcmcstep._rebuild import rebuild_derived
from bartz.prepcovars import GivenSplitsBinner
from tests.util import (
    assert_array_equal,
    assert_close_matrices,
    jaxtyping_disabled,
    manual_tree,
    nnone,
)


def max_lr_test(
    counts: Int32[Array, ' k'], prob: Float32[Array, ' k']
) -> Float32[Array, '']:
    """Return the p-value of the maximal likelihood-ratio goodness-of-fit test.

    Tests the null hypothesis that `counts` is multinomial with category
    probabilities `prob`. The statistic is twice the log-likelihood ratio of the
    saturated model to the null, asymptotically chi-squared with ``k - 1``
    degrees of freedom under the null. This is the G-test, equivalent to
    ``scipy.stats.power_divergence(lambda_='log-likelihood')``, recomputed in
    jax to keep float32 and skip scipy's sum check.
    """
    (k,) = counts.shape
    expected = counts.sum() * prob
    statistic = 2 * jnp.sum(xlogy(counts, counts / expected))
    return chi2.sf(statistic, k - 1)


def test_format_tree() -> None:
    """Check the output of `format_tree` on a single example."""
    tree = manual_tree(
        [[1.0], [2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], [[4], [1, 2]], [[15], [0, 3]]
    )
    s = format_tree(tree)
    print(s)
    ref_s = """\
 1 ┐x4 < 15
 2 ├── 1.00000 * 2.00000
 3 └──┐x2 < 3
 6    ├──╢1.00000 * 6.00000
 7    └──╢1.00000 * 7.00000"""
    assert s == ref_s


def test_format_tree_leaf_unit() -> None:
    """Check `format_tree` renders the leaf scale and the leaf dtype precision."""
    tree = manual_tree(
        [[1.0], [2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], [[4], [1, 2]], [[15], [0, 3]]
    )
    tree = tree_at(
        lambda t: (t.leaf_tree, t.leaf_unit),
        tree,
        (tree.leaf_tree.astype(jnp.float16), jnp.float32(0.5)),
    )
    s = format_tree(tree)
    print(s)
    ref_s = """\
 1 ┐x4 < 15
 2 ├── 0.500000 * 2.00
 3 └──┐x2 < 3
 6    ├──╢0.500000 * 6.00
 7    └──╢0.500000 * 7.00"""
    assert s == ref_s


def test_format_tree_print_all() -> None:
    """Check `format_tree` renders every heap node when ``print_all`` is set."""
    tree = manual_tree(
        [[1.0], [2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], [[4], [1, 2]], [[15], [0, 3]]
    )
    s = format_tree(tree, print_all=True)
    print(s)
    ref_s = """\
 1 ┐decision(4, 15, 1.00000 * 1.00000)
 2 ├──┐leaf(1, 0, 1.00000 * 2.00000)
 4 │  ├── unused(0, 0, 1.00000 * 4.00000)
 5 │  └── unused(0, 0, 1.00000 * 5.00000)
 3 └──┐decision(2, 3, 1.00000 * 3.00000)
 6    ├── leaf(0, 0, 1.00000 * 6.00000)
 7    └── leaf(0, 0, 1.00000 * 7.00000)"""
    assert s == ref_s


def test_format_tree_multivariate() -> None:
    """Check the output of `format_tree` on a tree with multivariate leaves."""
    tree = manual_tree(
        [[1.0], [2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], [[4], [1, 2]], [[15], [0, 3]]
    )
    # add a leaf axis to turn the univariate tree into a multivariate one
    tree = tree_at(
        lambda t: t.leaf_tree, tree, jnp.stack([tree.leaf_tree, -tree.leaf_tree])
    )
    assert tree.leaf_tree.ndim == tree.var_tree.ndim + 1
    s = format_tree(tree)
    print(s)
    ref_s = """\
 1 ┐x4 < 15
 2 ├── 1.00000 * [2.00000, -2.00000]
 3 └──┐x2 < 3
 6    ├──╢1.00000 * [6.00000, -6.00000]
 7    └──╢1.00000 * [7.00000, -7.00000]"""
    assert s == ref_s


class TestSampleForest:
    """Test `debug._prior.sample_forest_structure` with `sample_leaves`."""

    Args = namedtuple(
        'Args',
        [
            'key',
            'trace_length',
            'num_trees',
            'max_split',
            'p_nonterminal',
            'sigma_mu',
            'log_s',
        ],
    )

    @pytest.fixture(params=['uniform', 'nonuniform'])
    def args(self, request: pytest.FixtureRequest, keys: split) -> Args:
        """Prepare arguments for `sample`, with and without `log_s`."""
        # config
        trace_length = 1000
        num_trees = 200
        maxdepth = 6
        alpha = 0.95
        beta = 2
        max_split = 5

        # prepare arguments
        d = jnp.arange(maxdepth - 1)
        p_nonterminal = alpha / (1 + d).astype(float) ** beta
        p = maxdepth - 1
        max_split = jnp.full(p, jnp.array(max_split, minimal_unsigned_dtype(max_split)))
        sigma_mu = 1 / jnp.sqrt(num_trees)
        log_s = None if request.param == 'uniform' else jnp.log(1 + jnp.arange(p))

        return self.Args(
            keys.pop(),
            trace_length,
            num_trees,
            max_split,
            p_nonterminal,
            sigma_mu,
            log_s,
        )

    @staticmethod
    def sample(args: Args) -> TreesTrace:
        """Sample a trace of independent trees from the prior."""
        keys = split(args.key)
        shape = (args.trace_length, args.num_trees)
        trees = sample_forest_structure(
            keys.pop(),
            shape,
            max_split=args.max_split,
            p_nonterminal=args.p_nonterminal,
            log_s=args.log_s,
        )
        tree_size = 2 ** (args.p_nonterminal.size + 1)
        leaf_tree = sample_leaves(
            keys.pop(), (*shape, tree_size), jnp.reciprocal(args.sigma_mu)
        )
        return TreesTrace(
            leaf_tree=leaf_tree, var_tree=trees.var_tree, split_tree=trees.split_tree
        )

    def test_valid_trees(self, args: Args) -> None:
        """Check all sampled trees are valid."""
        trees = self.sample(args)
        batch_shape = (args.trace_length, args.num_trees)
        heap_size = 2 ** (args.p_nonterminal.size + 1)
        assert trees.leaf_tree.shape == (*batch_shape, heap_size)
        assert trees.var_tree.shape == (*batch_shape, heap_size // 2)
        assert trees.split_tree.shape == (*batch_shape, heap_size // 2)
        bad = check_trace(trees, args.max_split)
        num_bad = jnp.count_nonzero(bad).item()
        assert num_bad == 0

    def test_max_depth(self, keys: split, args: Args) -> None:
        """Check that trees stop growing when p_nonterminal = 0."""
        for max_depth in range(args.p_nonterminal.size + 1):
            p_nonterminal = jnp.zeros_like(args.p_nonterminal)
            p_nonterminal = p_nonterminal.at[:max_depth].set(1.0)
            args = tree_at(lambda args: args.p_nonterminal, args, p_nonterminal)
            args = tree_at(lambda args: args.key, args, keys.pop())
            trees = self.sample(args)
            assert jnp.all(trees.split_tree[:, :, 1 : 2**max_depth])
            assert not jnp.any(trees.split_tree[:, :, 2**max_depth :])

    def test_forest_sdev(self, keys: split, args: Args) -> None:
        """Check that the sum of trees is standard Normal."""
        trees = self.sample(args)
        leaf_indices = random.randint(
            keys.pop(), trees.leaf_tree.shape[:2], 0, trees.leaf_tree.shape[-1]
        )
        batch_indices = jnp.ogrid[
            : trees.leaf_tree.shape[0], : trees.leaf_tree.shape[1]
        ]
        leaves = trees.leaf_tree[(*batch_indices, leaf_indices)]
        sum_of_trees = jnp.sum(leaves, axis=1)

        test = ks_1samp(sum_of_trees, stats.norm.cdf)
        assert test.pvalue > 0.1

    def test_trees_differ(self, args: Args) -> None:
        """Check that trees are different across iterations."""
        trees = self.sample(args)
        for attr in ('leaf_tree', 'var_tree', 'split_tree'):
            heap = getattr(trees, attr)
            diff_trace = jnp.diff(heap, axis=0)
            diff_forest = jnp.diff(heap, axis=1)
            assert jnp.any(diff_trace)
            assert jnp.any(diff_forest)

    def test_root_distribution(self, args: Args) -> None:
        """Check the root split variable follows the prior split probabilities.

        At the root every variable is available, so the chosen variable is
        distributed exactly as `softmax(log_s)` (uniform when `log_s` is None).
        The ~200k root samples make the test's power against any realistic
        sampler bug essentially 1 (e.g. ignoring `log_s` gives p well below
        1e-10), while the 1e-3 threshold keeps false positives rare.
        """
        p = args.max_split.size
        trees = self.sample(args)
        counts = jnp.bincount(trees.var_tree[:, :, 1].ravel(), length=p)
        log_s = jnp.zeros(p) if args.log_s is None else args.log_s
        assert max_lr_test(counts, softmax(log_s)) > 1e-3

    def test_log_s_broadcast_equiv(self, args: Args) -> None:
        """A per-iteration `log_s` with equal rows matches the shared `log_s`.

        Passing one `log_s` row per iteration that happens to be constant must
        give exactly the same trees as passing that single row shared across all
        trees, since both reduce to the same per-tree split probabilities.
        """
        p = args.max_split.size
        log_s = jnp.log(1 + jnp.arange(p)).astype(jnp.float32)
        args = args._replace(log_s=log_s)
        shared = self.sample(args)
        per_iter = self.sample(
            args._replace(log_s=jnp.broadcast_to(log_s, (args.trace_length, p)))
        )
        for attr in ('leaf_tree', 'var_tree', 'split_tree'):
            assert_array_equal(getattr(shared, attr), getattr(per_iter, attr))

    def test_per_iteration_log_s(self, args: Args) -> None:
        """Each iteration's root variable follows that iteration's `log_s`.

        Alternate iterations between two different split-probability vectors and
        check that, splitting the root variables by parity, each group follows
        the `softmax` of its own `log_s`. This would fail if `sample_forest_structure`
        ignored the per-iteration axis or misaligned it with the trees.
        """
        p = args.max_split.size
        v_even = jnp.log(1 + jnp.arange(p)).astype(jnp.float32)
        v_odd = v_even[::-1]
        log_s = jnp.where(
            (jnp.arange(args.trace_length) % 2)[:, None] == 0, v_even, v_odd
        )

        trees = self.sample(args._replace(log_s=log_s))
        root = trees.var_tree[:, :, 1]
        for parity, v in ((0, v_even), (1, v_odd)):
            counts = jnp.bincount(root[parity::2].ravel(), length=p)
            assert max_lr_test(counts, softmax(v)) > 1e-3


class TestSampleTheta:
    """Test `_prior.sample_theta`."""

    def test_prior_distribution(self, keys: split) -> None:
        """``theta / (theta + rho)`` follows the ``Beta(a, b)`` prior."""
        rho, a, b = 7.0, 0.5, 2.0
        theta = sample_theta(keys.pop(), rho, a, b, (200_000,))
        lambda_ = theta / (theta + rho)
        # the ~200k draws give power ~1 against a wrong Beta, while the 1e-3
        # threshold keeps false positives rare
        assert ks_1samp(lambda_, stats.beta(a, b).cdf).pvalue > 1e-3

    def test_scalar_shape(self, keys: split) -> None:
        """The default shape draws a scalar `theta`."""
        assert sample_theta(keys.pop(), 1.0, 0.5, 2.0).shape == ()

    def test_restricted(self, keys: split) -> None:
        """With a range, ``logit(theta / (theta + rho))`` is a truncated Beta."""
        rho, a, b, log_range = 7.0, 0.05, 1.0, 40.0
        theta_samp = sample_theta(
            keys.pop(), rho, a, b, (200_000,), log_range=log_range
        )
        v_samp = numpy.log(numpy.asarray(theta_samp, numpy.float64) / rho)
        assert numpy.all(numpy.abs(v_samp) <= log_range)

        # the cdf of v, computed from the Beta cdf in log space for accuracy
        def cdf(v: Float[numpy.ndarray, '*shape']) -> Float[numpy.ndarray, '*shape']:
            log_lambda = -numpy.logaddexp(0, -v)
            return stats.beta(a, b).cdf(numpy.exp(log_lambda))

        lo, hi = cdf(numpy.array(-log_range)), cdf(numpy.array(log_range))
        assert ks_1samp(v_samp, lambda v: (cdf(v) - lo) / (hi - lo)).pvalue > 1e-3


class TestSampleS:
    """Test `_prior.sample_s`."""

    def test_prior_distribution(self, keys: split) -> None:
        """``softmax(log_s)`` is Dirichlet, so each component is the right Beta.

        For a symmetric ``Dirichlet(theta/p, ..., theta/p)`` the marginal of
        each component is ``Beta(theta/p, theta (p-1)/p)``.
        """
        p, theta = 4, 3.0
        log_s = sample_s(keys.pop(), jnp.full(200_000, theta), p)
        s = softmax(log_s, axis=1)
        marginal = stats.beta(theta / p, theta * (p - 1) / p)
        for j in range(p):
            assert ks_1samp(s[:, j], marginal.cdf).pvalue > 1e-3

    def test_scalar_theta_shape(self, keys: split) -> None:
        """A scalar `theta` yields a single ``(p,)`` `log_s` vector."""
        assert sample_s(keys.pop(), 2.0, 5).shape == (5,)


class PositiveCriterion(RejectionCriterion):
    """Accept the scalars above a threshold."""

    threshold: float

    def accept(self, sample: Float32[Array, '']) -> Bool[Array, '']:
        """Check the sample is above the threshold."""
        return sample > self.threshold


class TestRejectionSample:
    """Test `_prior.rejection_sample`."""

    @pytest.mark.parametrize('threshold', [0.0, 2.0])
    def test_truncated_normal(self, keys: split, threshold: float) -> None:
        """Normal samples conditioned on a lower bound are truncated Normal.

        With the higher threshold, about 2% of the candidates are accepted, so
        filling the samples takes many iterations.
        """
        x = rejection_sample(
            keys.pop(),
            lambda key, num: random.normal(key, (num,)),
            PositiveCriterion(threshold),
            10_000,
        )
        assert jnp.all(x > threshold)
        cdf = stats.truncnorm(threshold, numpy.inf).cdf
        assert ks_1samp(x, cdf).pvalue > 1e-3


class TestMinPointsCriterion:
    """Test `_prior.MinPointsCriterion`."""

    @pytest.fixture
    def structure(self) -> TreeStructure:
        """Make a tree splitting the root at 1 and its right child at 2."""
        tree = manual_tree(
            [[0.0], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0], [0, 0]], [[1], [0, 2]]
        )
        return TreeStructure(var_tree=tree.var_tree, split_tree=tree.split_tree)

    @pytest.fixture
    def X(self) -> UInt8[Array, '1 6']:
        """Make datapoints that fill the leaves of `structure` with 2, 3, 1 points.

        The decision nodes then have 6 and 4 datapoints.
        """
        return jnp.array([[0, 0, 1, 1, 1, 2]], jnp.uint8)

    @pytest.mark.parametrize(
        ('decision', 'leaf', 'expected'),
        [
            (None, None, True),
            (4, None, True),
            (5, None, False),
            (None, 1, True),
            (None, 2, False),
            (4, 1, True),
        ],
    )
    def test_accept(
        self,
        structure: TreeStructure,
        X: UInt8[Array, '1 6'],
        decision: int | None,
        leaf: int | None,
        expected: bool,
    ) -> None:
        """The criterion checks the datapoints in decision nodes and in leaves."""
        criterion = MinPointsCriterion(
            X=X,
            min_points_per_decision_node=None
            if decision is None
            else jnp.int32(decision),
            min_points_per_leaf=None if leaf is None else jnp.int32(leaf),
        )
        assert criterion.accept(structure) == expected

    def test_stump(self, structure: TreeStructure, X: UInt8[Array, '1 6']) -> None:
        """The stump is always accepted, as the root counts as no leaf."""
        criterion = MinPointsCriterion(
            X=X,
            min_points_per_decision_node=jnp.int32(100),
            min_points_per_leaf=jnp.int32(100),
        )
        stump = TreeStructure(
            var_tree=jnp.zeros_like(structure.var_tree),
            split_tree=jnp.zeros_like(structure.split_tree),
        )
        assert criterion.accept(stump)


class TestConstrainedForestStructure:
    """Test `_prior.sample_forest_structure` with a criterion."""

    @pytest.fixture
    def criterion(self, keys: split) -> MinPointsCriterion:
        """Make a criterion on random datapoints."""
        return MinPointsCriterion(
            X=random.randint(keys.pop(), (2, 20), 0, 4, jnp.uint8),
            min_points_per_decision_node=jnp.int32(6),
            min_points_per_leaf=jnp.int32(3),
        )

    def kw(self) -> dict:
        """Prepare the prior parameters."""
        return dict(max_split=jnp.full(2, 3, jnp.uint8), p_nonterminal=jnp.full(4, 0.7))

    def test_matches_filtering(
        self, keys: split, criterion: MinPointsCriterion
    ) -> None:
        """Conditioning on a criterion matches filtering unconditioned trees."""
        constrained = sample_forest_structure(
            keys.pop(), (5_000,), criterion=criterion, **self.kw()
        )
        free = sample_forest_structure(keys.pop(), (50_000,), **self.kw())
        accept = vmap(criterion.accept)
        assert jnp.all(accept(constrained))
        accepted = accept(free)

        # compare the number of nodes, and the variable of the root
        num_nodes = jnp.count_nonzero(constrained.split_tree, axis=-1)
        reference = jnp.count_nonzero(free.split_tree[accepted], axis=-1)
        assert ks_2samp(num_nodes, reference).pvalue > 1e-3
        reference = free.var_tree[accepted, 1]
        assert ks_2samp(constrained.var_tree[:, 1], reference).pvalue > 1e-3

    def test_per_forest_log_s(self, keys: split, criterion: MinPointsCriterion) -> None:
        """A `log_s` that varies across forests is rejected."""
        log_s = jnp.zeros((3, 2))
        with pytest.raises(ValueError, match='shared `log_s`'):
            sample_forest_structure(
                keys.pop(), (3, 10), criterion=criterion, log_s=log_s, **self.kw()
            )


class TestSamplePriorConstraintsChains:
    """Test `sample_prior` with constraints and a fixed `log_s` in multiple chains."""

    def init(self, keys: split) -> State:
        """Create a multichain state with constraints and a fixed `log_s`."""
        p, n = 2, 20
        return init(
            X=random.randint(keys.pop(), (p, n), 0, 4, jnp.uint8),
            y=jnp.zeros(n),
            offset=0.0,
            max_split=jnp.full(p, 3, jnp.uint8),
            num_trees=5,
            p_nonterminal=make_p_nonterminal(4),
            leaf_prior_cov_inv=Wishart(None, None, 5.0),
            error_cov_inv=Wishart(4.0, 4.0, 1.0),
            min_points_per_decision_node=6,
            min_points_per_leaf=3,
            log_s=jnp.array([0.0, -1.0]),
            num_chains=3,
        )

    def test_shared(self, keys: split) -> None:
        """The fixed `log_s` has no chain axis, and `sample_prior` works with it."""
        state = self.init(keys)
        assert nnone(state.forest.sparsity.log_s).shape == (2,)
        state = sample_prior(keys.pop(), state)
        assert state.forest.split_tree.any()


class TestSampleModel:
    """Test `sample_prior` and `sample_data` on small states."""

    n = 10

    def init(
        self,
        key: Key[Array, ''],
        outcome: Literal['continuous', 'binary', 'mixed'] = 'continuous',
        **kwargs: Any,
    ) -> State:
        """Create a small state, with `kwargs` passed to `init`."""
        p = 2
        k = 2 if outcome == 'mixed' else None
        kshape = () if k is None else (k,)
        eye = jnp.float32(1.0) if k is None else jnp.eye(k)

        error_kw: dict
        if outcome == 'binary':
            error_kw = dict(outcome_type='binary')
        elif outcome == 'mixed':
            error_kw = dict(
                outcome_type=['continuous', 'binary'],
                error_cov_inv=DiagWishart(4.0, 4.0 * eye, eye),
            )
        else:
            error_kw = dict(error_cov_inv=Wishart(4.0, 4.0, 1.0))

        return init(
            X=random.randint(key, (p, self.n), 0, 4, jnp.uint8),
            y=jnp.zeros((*kshape, self.n)),
            offset=jnp.zeros(kshape),
            max_split=jnp.full(p, 3, jnp.uint8),
            num_trees=5,
            p_nonterminal=make_p_nonterminal(4),
            leaf_prior_cov_inv=Wishart(None, None, 5.0 * eye),
            **error_kw,
            **kwargs,
        )

    def test_shared_chains(self, keys: split) -> None:
        """`sample_data` rejects chains that share the data."""
        state = self.init(keys.pop(), num_chains=2)
        with pytest.raises(ValueError, match='the chains share the data'):
            sample_data(keys.pop(), state)

    def test_constraints_with_theta_prior(self, keys: split) -> None:
        """`sample_prior` rejects a minimum number of points with a prior on `theta`."""
        state = self.init(
            keys.pop(), min_points_per_leaf=3, a=0.5, b=1.0, rho=2.0, sparse_on_at=0
        )
        with pytest.raises(NotImplementedError, match='minimum number of points'):
            sample_prior(keys.pop(), state)

    def test_data_sharded(self, keys: split) -> None:
        """Sampling and rebuilding reject a state with the data sharded."""
        if get_device_count() < 2:
            pytest.skip('need at least 2 devices for the data axis')
        state = self.init(keys.pop(), mesh={'data': 2})
        match = 'with the data sharded is not supported'
        with pytest.raises(NotImplementedError, match=match):
            sample_prior(keys.pop(), state)
        with pytest.raises(NotImplementedError, match=match):
            sample_data(keys.pop(), state)
        with pytest.raises(NotImplementedError, match=match):
            rebuild_derived(state)

    @pytest.mark.parametrize('outcome', ['continuous', 'binary', 'mixed'])
    def test_missing_kept(
        self, keys: split, outcome: Literal['continuous', 'binary', 'mixed']
    ) -> None:
        """`sample_data` leaves the missing datapoints unchanged."""
        # sample missingness indicator
        missing_shape = (2, self.n) if outcome == 'mixed' else (self.n,)
        missing = random.bernoulli(keys.pop(), 0.5, missing_shape)

        # initialize a state with missing data
        state = self.init(keys.pop(), outcome, missing=missing)

        # extract outcome and latent outcome
        y = jnp.array(state.y)
        z = nnone(state.z) if outcome != 'continuous' else None
        z = None if z is None else jnp.array(z)

        # slice `missing` to match the shape of `z`
        z_missing = missing
        if state.binary_indices is not None:
            z_missing = missing[state.binary_indices, :]

        # resample the data on the state with missing data
        new = sample_data(keys.pop(), state)

        # check the values marked as missing remained the ones at initialization
        assert_array_equal(jnp.where(missing, new.y, y), y)
        assert jnp.any(new.y != y)
        if z is not None:
            assert_array_equal(jnp.where(z_missing, nnone(new.z), z), z)

    def test_leaf_quantization(self, keys: split) -> None:
        """`sample_prior` rounds the leaves to the quantization grid of `step`."""
        # store keys to sync the random streams of the two cases
        init_key, prior_key = keys.pop(2)

        # sample state from prior w/o leaf quantization
        plain = sample_prior(prior_key, self.init(init_key))

        # sample state from prior w/ leaf quantization
        q = 20
        quantized_init = self.init(random.clone(init_key), leaf_quantization=q)

        # infer quantization unit; get it from the state before sampling the
        # prior to implicitly check `sample_prior` does not change quantization
        # settings (and before `sample_prior` donates the state)
        nmant = jnp.finfo(quantized_init.resid.dtype).nmant
        quantum = (
            quantized_init.resid_eff_scale
            / quantized_init.forest.leaf_unit
            * 2.0 ** (q - nmant)
        )

        quantized = sample_prior(random.clone(prior_key), quantized_init)

        # check the leaf tree of the quantized state is indeed quantized
        leaf_tree_q = quantized.forest.leaf_tree / quantum
        assert_array_equal(leaf_tree_q, jnp.round(leaf_tree_q))

        # check the rounding used to quantize was correct and non-trivial
        diff = quantized.forest.leaf_tree - plain.forest.leaf_tree
        assert jnp.all(jnp.abs(diff) <= quantum / 2)
        assert jnp.any(diff != 0)


class TestSampleWishart:
    """Test `_prior.sample_wishart` and `_prior.sample_diag_wishart`."""

    def test_scalar(self, keys: split) -> None:
        """A scalar precision is Gamma, and its factor is the square root."""
        nu, rate = 5.0, 2.0
        value, chol = vmap(sample_wishart, in_axes=(0, None, None))(
            keys.pop(100_000), jnp.float32(nu), jnp.float32(rate)
        )
        assert_close_matrices(jnp.square(chol), value, rtol=1e-6)
        cdf = stats.gamma(nu / 2, scale=2 / rate).cdf
        assert ks_1samp(value, cdf).pvalue > 1e-3

    def test_matrix(self, keys: split) -> None:
        """The diagonal of a Wishart matrix is Gamma, with the right mean."""
        nu = 5.0
        rate = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        value, chol = vmap(sample_wishart, in_axes=(0, None, None))(
            keys.pop(100_000), jnp.float32(nu), rate
        )
        assert_close_matrices(chol @ chol.mT, value, rtol=1e-5, reduce_rank=True)
        assert_array_equal(jnp.triu(chol, 1), jnp.zeros_like(chol))
        scale = jnp.linalg.inv(rate)
        assert_close_matrices(value.mean(axis=0), nu * scale, rtol=0.02)
        for i in range(2):
            cdf = stats.gamma(nu / 2, scale=2 * scale[i, i].item()).cdf
            assert ks_1samp(value[:, i, i], cdf).pvalue > 1e-3

    def test_diag(self, keys: split) -> None:
        """The diagonal precision has independent Gamma components."""
        nu = 5.0
        rate = jnp.diag(jnp.array([2.0, 0.5]))
        value, chol = vmap(sample_diag_wishart, in_axes=(0, None, None))(
            keys.pop(100_000), jnp.float32(nu), rate
        )
        assert_close_matrices(chol @ chol.mT, value, rtol=1e-5, reduce_rank=True)
        for i in range(2):
            cdf = stats.gamma(nu / 2, scale=2 / rate[i, i].item()).cdf
            assert ks_1samp(value[:, i, i], cdf).pvalue > 1e-3


def test_sample_leaves_covariance(keys: split) -> None:
    """Vector-valued leaves have the inverse of the prior precision as covariance."""
    prec = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    chol = jnp.linalg.cholesky(prec)
    leaves = sample_leaves(keys.pop(), (100_000, 2, 4), chol)
    flat = jnp.moveaxis(leaves, 1, -1).reshape(-1, 2)
    # cov = prec^-1 = (L L^T)^-1, so L^T cov L = I
    whitened = chol.T @ jnp.cov(flat.T) @ chol
    assert_close_matrices(whitened, jnp.eye(2), rtol=0.02)


class TestCheckTree:
    """Trigger each subcheck of `bartz.grove._check.check_tree` in isolation.

    Each test crafts a tree that violates one specific invariant, then asserts
    that `describe_error` reports the expected check (and only the expected
    checks). When violating one invariant also forces another (e.g.
    `check_stray_nodes` always breaks `check_num_nodes`), the test asserts the
    exact set of co-failures.
    """

    # Two-variable problem with max_split[v] = 4 used by the "easy" cases.
    _MS = jnp.array([4, 4], jnp.uint8)
    # Wider range used by the rule/bounds tests.
    _MS_WIDE = jnp.array([10, 10], jnp.uint8)

    @staticmethod
    def _zeros_tree() -> TreesTrace:
        """Return an all-zeros (single-leaf) depth-3 tree, valid by construction."""
        return TreesTrace(
            leaf_tree=jnp.zeros(8),
            var_tree=jnp.zeros(4, jnp.uint8),
            split_tree=jnp.zeros(4, jnp.uint8),
        )

    @staticmethod
    def _describe(tree: TreesTrace, max_split: Shaped[jnp.ndarray, '...']) -> list[str]:
        """Run `check_tree` and return the names of the failing checks."""
        # `check_tree` is a runtime validator deliberately fed malformed trees
        # here; disable jaxtyping so it isn't pre-empted by the import-hook
        # type-checker (which would reject the bad dtypes/shapes first).
        with jaxtyping_disabled():
            return describe_error(check_tree(tree, max_split))

    def test_types_var_dtype(self) -> None:
        """Wrong `var_tree` dtype trips `check_types`."""
        tree = tree_at(
            lambda t: t.var_tree, self._zeros_tree(), jnp.zeros(4, jnp.uint16)
        )
        assert self._describe(tree, self._MS) == ['check_types']

    def test_types_split_dtype(self) -> None:
        """Wrong `split_tree` dtype trips `check_types`."""
        tree = tree_at(
            lambda t: t.split_tree, self._zeros_tree(), jnp.zeros(4, jnp.uint16)
        )
        assert self._describe(tree, self._MS) == ['check_types']

    def test_types_max_split_signed(self) -> None:
        """Signed `max_split` dtype trips `check_types`."""
        # also cast split_tree to match the signed dtype, otherwise the dtype
        # mismatch between max_split and split_tree would itself trip check_types
        signed_max_split = jnp.array([4, 4], jnp.int32)
        tree = tree_at(
            lambda t: t.split_tree, self._zeros_tree(), jnp.zeros(4, jnp.int32)
        )
        assert self._describe(tree, signed_max_split) == ['check_types']

    def test_shapes_leaf_3d(self) -> None:
        """A 3D `leaf_tree` trips `check_shapes`."""
        tree = tree_at(lambda t: t.leaf_tree, self._zeros_tree(), jnp.zeros((1, 1, 8)))
        assert self._describe(tree, self._MS) == ['check_shapes']

    def test_shapes_leaf_size_mismatch(self) -> None:
        """`leaf_tree.shape[-1] != 2 * var_tree.size` trips `check_shapes`."""
        tree = tree_at(lambda t: t.leaf_tree, self._zeros_tree(), jnp.zeros(16))
        assert self._describe(tree, self._MS) == ['check_shapes']

    def test_unused_node_var(self) -> None:
        """A dirty `var_tree[0]` trips `check_unused_node`."""
        tree = tree_at(
            lambda t: t.var_tree, self._zeros_tree(), jnp.array([1, 0, 0, 0], jnp.uint8)
        )
        assert self._describe(tree, self._MS) == ['check_unused_node']

    def test_unused_node_split(self) -> None:
        """A dirty `split_tree[0]` trips `check_unused_node` and `check_num_nodes`.

        Setting `split_tree[0] != 0` bumps the internal-node count without
        adding a corresponding leaf, so the leaf-vs-internal balance check
        also fails.
        """
        tree = tree_at(
            lambda t: t.split_tree,
            self._zeros_tree(),
            jnp.array([1, 0, 0, 0], jnp.uint8),
        )
        assert self._describe(tree, self._MS) == [
            'check_unused_node',
            'check_num_nodes',
        ]

    def test_leaf_values_nan(self) -> None:
        """A NaN leaf trips `check_leaf_values`."""
        leaf = jnp.zeros(8).at[1].set(jnp.nan)
        tree = tree_at(lambda t: t.leaf_tree, self._zeros_tree(), leaf)
        assert self._describe(tree, self._MS) == ['check_leaf_values']

    def test_leaf_values_inf(self) -> None:
        """An infinite leaf trips `check_leaf_values`."""
        leaf = jnp.zeros(8).at[3].set(jnp.inf)
        tree = tree_at(lambda t: t.leaf_tree, self._zeros_tree(), leaf)
        assert self._describe(tree, self._MS) == ['check_leaf_values']

    def test_stray_nodes(self) -> None:
        """A non-leaf node with a leaf parent trips `check_stray_nodes`.

        Such a "stray" is also counted as internal but contributes no leaf, so
        `check_num_nodes` necessarily fails alongside.
        """
        tree = tree_at(
            lambda t: t.split_tree,
            self._zeros_tree(),
            jnp.array([0, 0, 1, 0], jnp.uint8),
        )
        assert self._describe(tree, self._MS) == [
            'check_stray_nodes',
            'check_num_nodes',
        ]

    def test_rule_consistency(self) -> None:
        """A descendant split outside its ancestor's range trips `check_rule_consistency`.

        Root splits var 0 at 5, so the left child's upper bound on var 0 is 4;
        giving the left child a split of 8 on the same variable violates the
        ancestor constraint without breaking any other invariant.
        """
        tree = TreesTrace(
            leaf_tree=jnp.zeros(8),
            var_tree=jnp.zeros(4, jnp.uint8),
            split_tree=jnp.array([0, 5, 8, 0], jnp.uint8),
        )
        assert self._describe(tree, self._MS_WIDE) == ['check_rule_consistency']

    def test_var_in_bounds(self) -> None:
        """A variable index >= `max_split.size` on a decision node trips `check_var_in_bounds`."""
        tree = TreesTrace(
            leaf_tree=jnp.zeros(8),
            var_tree=jnp.array([0, 2, 0, 0], jnp.uint8),
            split_tree=jnp.array([0, 5, 0, 0], jnp.uint8),
        )
        assert self._describe(tree, self._MS_WIDE) == ['check_var_in_bounds']

    def test_split_in_bounds(self) -> None:
        """A split above `max_split[var]` trips `check_split_in_bounds`."""
        tree = TreesTrace(
            leaf_tree=jnp.zeros(8),
            var_tree=jnp.zeros(4, jnp.uint8),
            split_tree=jnp.array([0, 20, 0, 0], jnp.uint8),
        )
        assert self._describe(tree, self._MS_WIDE) == ['check_split_in_bounds']

    def test_valid_tree_passes(self) -> None:
        """Sanity: the base all-zeros tree triggers no checks."""
        assert self._describe(self._zeros_tree(), self._MS) == []


BART3_DATA = Path(__file__).parent / 'data' / 'bart3'


class BART3Fit(NamedTuple):
    """A fit of R BART3 saved by ``data/bart3/generate.R``."""

    trees: str
    x_train: Float32[Array, 'p n']
    xinfo: Float32[Array, 'p m']
    offset: float
    yhat_train: Float32[Array, 'ndpost n']
    varcount: Int32[Array, 'ndpost p']


@pytest.fixture(scope='module')
def bart3_fit() -> BART3Fit:
    """Load the saved R BART3 fit."""

    def load(name: str, dtype: type = numpy.float32) -> Shaped[Array, '...']:
        return jnp.asarray(
            numpy.loadtxt(BART3_DATA / f'{name}.txt', dtype=dtype, ndmin=2)
        )

    return BART3Fit(
        trees=(BART3_DATA / 'trees.txt').read_text(),
        x_train=load('x_train'),
        xinfo=load('xinfo'),
        offset=load('offset').item(),
        yhat_train=load('yhat_train'),
        varcount=load('varcount', numpy.int32),
    )


class TestTreesBARTToBartz:
    """Test `trees_BART_to_bartz` against a trace saved from R BART3."""

    def check(self, fit: BART3Fit, trace: MinimalTrace, meta: BARTTraceMeta) -> None:
        """Check the converted trace reproduces the R BART3 outputs."""
        ndpost, _ = fit.yhat_train.shape
        p, m = fit.xinfo.shape
        assert meta.ndpost == ndpost
        assert meta.numcut.shape == (p,)
        assert jnp.all(meta.numcut <= m)

        bad = check_trace(trace, meta.numcut)
        assert_array_equal(bad, jnp.zeros_like(bad))

        varcount = compute_varcount(p, trace)
        assert_array_equal(varcount, fit.varcount)

        binner = GivenSplitsBinner(fit.x_train, xinfo=fit.xinfo)
        yhat_train = evaluate_trace(binner.bin(fit.x_train), trace)
        assert_close_matrices(yhat_train, fit.yhat_train, rtol=1e-6)

    @pytest.mark.parametrize('trailing_newline', [True, False])
    def test_convert(self, bart3_fit: BART3Fit, trailing_newline: bool) -> None:
        """Check the conversion of the trace."""
        trees = bart3_fit.trees
        assert trees.endswith('\n')
        if not trailing_newline:
            trees = trees.removesuffix('\n')
        trace, meta = trees_BART_to_bartz(trees, offset=bart3_fit.offset)
        self.check(bart3_fit, trace, meta)

    def test_min_maxdepth(self, bart3_fit: BART3Fit) -> None:
        """Check `min_maxdepth` enlarges the heap without changing the trees."""
        maxdepth = 10
        trace, meta = trees_BART_to_bartz(
            bart3_fit.trees, offset=bart3_fit.offset, min_maxdepth=maxdepth
        )
        assert meta.heap_size < 2**maxdepth
        *_, heap_size = trace.leaf_tree.shape
        assert heap_size == 2**maxdepth
        self.check(bart3_fit, trace, meta)

    @pytest.mark.parametrize(
        ('line', 'replacement', 'match'),
        [
            (0, '20 10', 'Malformed header'),
            (1, 'five', 'Malformed tree header'),
            (2, '1 1 4', 'Malformed node definition'),
            (None, 'leftover', 'Leftover'),
        ],
    )
    def test_malformed(
        self, bart3_fit: BART3Fit, line: int | None, replacement: str, match: str
    ) -> None:
        """Check malformed traces are rejected."""
        lines = bart3_fit.trees.split('\n')
        if line is None:
            lines.append(replacement)
        else:
            lines[line] = replacement
        with pytest.raises(ValueError, match=match):
            trees_BART_to_bartz('\n'.join(lines))
