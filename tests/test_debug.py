# bartz/tests/test_debug.py
#
# Copyright (c) 2025, The Bartz Contributors
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

from bartz.debug import check_trace, format_tree, sample_prior
from bartz.jaxext import minimal_unsigned_dtype
from tests.util import manual_tree


def test_format_tree():
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
    def args(self, keys):
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

    def test_valid_trees(self, args: Args):
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

    def test_max_depth(self, keys, args: Args):
        """Check that trees stop growing when p_nonterminal = 0."""
        for max_depth in range(args.p_nonterminal.size + 1):
            p_nonterminal = jnp.zeros_like(args.p_nonterminal)
            p_nonterminal = p_nonterminal.at[:max_depth].set(1.0)
            args = tree_at(lambda args: args.p_nonterminal, args, p_nonterminal)
            args = tree_at(lambda args: args.key, args, keys.pop())
            trees = sample_prior(*args)
            assert jnp.all(trees.split_tree[:, :, 1 : 2**max_depth])
            assert not jnp.any(trees.split_tree[:, :, 2**max_depth :])

    def test_forest_sdev(self, keys, args: Args):
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

    def test_trees_differ(self, args: Args):
        """Check that trees are different across iterations."""
        trees = sample_prior(*args)
        for attr in ('leaf_tree', 'var_tree', 'split_tree'):
            heap = getattr(trees, attr)
            diff_trace = jnp.diff(heap, axis=0)
            diff_forest = jnp.diff(heap, axis=1)
            assert jnp.any(diff_trace)
            assert jnp.any(diff_forest)
