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
from scipy import stats
from scipy.stats import ks_1samp

from bartz._jaxext import minimal_unsigned_dtype, split
from bartz.debug import sample_prior
from bartz.grove import TreesTrace, check_trace, describe_error, format_tree
from bartz.grove._check import check_tree
from tests.util import manual_tree


def test_format_tree() -> None:
    """Check the output of `format_tree` on a single example."""
    tree = manual_tree(
        [[1.0], [2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], [[4], [1, 2]], [[15], [0, 3]]
    )
    s = format_tree(tree)
    print(s)
    ref_s = """\
 1 ┐x4 < 15
 2 ├── 2.0
 3 └──┐x2 < 3
 6    ├──╢6.0
 7    └──╢7.0"""
    assert s == ref_s


class TestSamplePrior:
    """Test `debug.sample_prior`."""

    Args = namedtuple(
        'Args',
        ['key', 'trace_length', 'num_trees', 'max_split', 'p_nonterminal', 'sigma_mu'],
    )

    @pytest.fixture
    def args(self, keys: split) -> Args:
        """Prepare arguments for `sample_prior`."""
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

        return self.Args(
            keys.pop(), trace_length, num_trees, max_split, p_nonterminal, sigma_mu
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
    def _describe(tree: TreesTrace, max_split: jnp.ndarray) -> list[str]:
        """Run `check_tree` and return the names of the failing checks."""
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
