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

import pytest
from equinox import tree_at
from jax import numpy as jnp
from jax import random
from jax.nn import softmax
from jax.scipy.special import xlogy
from jax.scipy.stats import chi2
from jaxtyping import Array, Float32, Int32, Shaped
from scipy import stats
from scipy.stats import ks_1samp

from bartz._jaxext import minimal_unsigned_dtype, split
from bartz.debug import sample_prior
from bartz.debug._prior import sample_s, sample_theta
from bartz.grove import TreesTrace, check_trace, describe_error, format_tree
from bartz.grove._check import check_tree
from tests.util import assert_array_equal, jaxtyping_disabled, manual_tree


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


class TestSamplePrior:
    """Test `debug.sample_prior`."""

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
        """Prepare arguments for `sample_prior`, with and without `log_s`."""
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

    def test_valid_trees(self, args: Args) -> None:
        """Check all sampled trees are valid."""
        trees = sample_prior(*args)
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
            trees = sample_prior(*args)
            assert jnp.all(trees.split_tree[:, :, 1 : 2**max_depth])
            assert not jnp.any(trees.split_tree[:, :, 2**max_depth :])

    def test_forest_sdev(self, keys: split, args: Args) -> None:
        """Check that the sum of trees is standard Normal."""
        trees = sample_prior(*args)
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
        trees = sample_prior(*args)
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
        trees = sample_prior(*args)
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
        shared = sample_prior(*args)
        per_iter = sample_prior(
            *args._replace(log_s=jnp.broadcast_to(log_s, (args.trace_length, p)))
        )
        for attr in ('leaf_tree', 'var_tree', 'split_tree'):
            assert_array_equal(getattr(shared, attr), getattr(per_iter, attr))

    def test_per_iteration_log_s(self, args: Args) -> None:
        """Each iteration's root variable follows that iteration's `log_s`.

        Alternate iterations between two different split-probability vectors and
        check that, splitting the root variables by parity, each group follows
        the `softmax` of its own `log_s`. This would fail if `sample_prior`
        ignored the per-iteration axis or misaligned it with the trees.
        """
        p = args.max_split.size
        v_even = jnp.log(1 + jnp.arange(p)).astype(jnp.float32)
        v_odd = v_even[::-1]
        log_s = jnp.where(
            (jnp.arange(args.trace_length) % 2)[:, None] == 0, v_even, v_odd
        )

        trees = sample_prior(*args._replace(log_s=log_s))
        root = trees.var_tree[:, :, 1]
        for parity, v in ((0, v_even), (1, v_odd)):
            counts = jnp.bincount(root[parity::2].ravel(), length=p)
            assert max_lr_test(counts, softmax(v)) > 1e-3

    def test_sparse_two_level(self, args: Args) -> None:
        """The 2-level prior returns per-iteration latents that drive the trees.

        Sampling `theta` and `s` from their priors must yield latents of the
        right shape and valid trees, and the trees must actually use the sampled
        `log_s`: averaged over all root splits, the chosen variable's `log_s`
        exceeds the per-iteration mean `log_s`, since the root variable is drawn
        with probability `softmax(log_s)`.
        """
        p = args.max_split.size
        result = sample_prior(*args._replace(log_s=None), a=0.5, b=1.0, rho=float(p))
        theta = result.theta
        log_s = result.log_s
        assert theta is not None
        assert log_s is not None
        assert theta.shape == (args.trace_length,)
        assert log_s.shape == (args.trace_length, p)
        assert jnp.all(theta > 0)
        assert jnp.all(jnp.isfinite(log_s))
        bad = check_trace(result, args.max_split)
        assert jnp.count_nonzero(bad).item() == 0

        root = result.var_tree[:, :, 1]
        chosen = jnp.take_along_axis(log_s, root, axis=1)
        assert chosen.mean() > log_s.mean()

    def test_sparse_one_level(self, args: Args) -> None:
        """The 1-level prior draws `log_s` from a fixed-concentration Dirichlet.

        Setting `theta` alone (no `rho`, `a`, `b`) must yield a per-iteration
        `log_s` of the right shape, broadcasting the fixed `theta` to one value
        per iteration, and valid trees.
        """
        p = args.max_split.size
        result = sample_prior(*args._replace(log_s=None), theta=float(p))
        assert result.theta is not None
        assert result.log_s is not None
        assert jnp.all(result.theta == p)
        assert result.log_s.shape == (args.trace_length, p)
        assert jnp.all(jnp.isfinite(result.log_s))
        bad = check_trace(result, args.max_split)
        assert jnp.count_nonzero(bad).item() == 0

    @pytest.mark.parametrize(
        'kwargs', [dict(rho=1.0, a=0.5), dict(rho=1.0, b=1.0), dict(a=0.5, b=1.0)]
    )
    def test_partial_rho_a_b(self, args: Args, kwargs: dict) -> None:
        """Setting only some of `rho`, `a`, `b` raises."""
        with pytest.raises(ValueError, match='rho, a, b must be either'):
            sample_prior(*args._replace(log_s=None), **kwargs)

    def test_theta_with_rho_a_b(self, args: Args) -> None:
        """Setting `theta` together with `rho`, `a`, `b` raises."""
        with pytest.raises(ValueError, match='sampled from the prior'):
            sample_prior(*args._replace(log_s=None), theta=1.0, a=0.5, b=1.0, rho=1.0)

    def test_log_s_with_rho_a_b(self, args: Args) -> None:
        """Setting `log_s` together with `rho`, `a`, `b` raises."""
        log_s = jnp.zeros(args.max_split.size)
        with pytest.raises(ValueError, match='sampled from the prior'):
            sample_prior(*args._replace(log_s=log_s), a=0.5, b=1.0, rho=1.0)

    def test_log_s_with_theta(self, args: Args) -> None:
        """Setting `log_s` together with `theta` raises."""
        log_s = jnp.zeros(args.max_split.size)
        with pytest.raises(ValueError, match='sampled from the prior'):
            sample_prior(*args._replace(log_s=log_s), theta=1.0)


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
