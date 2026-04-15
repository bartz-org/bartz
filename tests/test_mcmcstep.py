# bartz/tests/test_mcmcstep.py
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

"""Test `bartz.mcmcstep`."""

from collections.abc import Sequence
from dataclasses import replace
from functools import partial, wraps
from math import prod
from typing import Literal, NamedTuple

import jax
import pytest
from beartype import beartype
from jax import debug_key_reuse, make_mesh, random, tree, vmap
from jax import numpy as jnp
from jax.sharding import AxisType, Mesh, PartitionSpec, SingleDeviceSharding
from jax.tree_util import KeyPath, keystr
from jaxtyping import (
    Array,
    Bool,
    Float,
    Float32,
    Int32,
    Key,
    PyTree,
    UInt8,
    UInt32,
    jaxtyped,
)
from numpy.testing import assert_allclose, assert_array_equal
from pytest import FixtureRequest  # noqa: PT013
from pytest_subtests import SubTests
from scipy import stats
from scipy.stats import chi2, ks_1samp, ks_2samp

from bartz.jaxext import get_device_count, minimal_unsigned_dtype, split
from bartz.mcmcstep import State, init, step
from bartz.mcmcstep._moves import (
    ancestor_variables,
    randint_exclude,
    randint_masked,
    split_range,
)
from bartz.mcmcstep._state import chain_vmap_axes, data_vmap_axes
from bartz.mcmcstep._step import (
    Counts,
    _compute_likelihood_ratio_mv,
    _compute_likelihood_ratio_uv,
    _precompute_leaf_terms_mv,
    _precompute_leaf_terms_uv,
    _precompute_likelihood_terms_mv,
    _precompute_likelihood_terms_uv,
    _sample_wishart_bartlett,
    _step_error_cov_inv_mv,
    _step_error_cov_inv_uv,
    step_error_cov_inv,
    step_trees,
    step_z,
)
from tests.util import assert_close_matrices, manual_tree


class VarTreeData(NamedTuple):
    """Fixture data pairing a variable tree with its max-split array."""

    var_tree: UInt8[Array, ' nodes']
    max_split: UInt8[Array, ' p']


class SplitRangeData(NamedTuple):
    """Fixture data pairing variable/split trees with a max-split array."""

    var_tree: UInt8[Array, ' nodes']
    split_tree: UInt8[Array, ' nodes']
    max_split: UInt8[Array, ' p']


def vmap_randint_masked(
    key: Key[Array, ''], mask: Bool[Array, ' n'], size: int
) -> Int32[Array, '* n']:
    """Vectorized version of `randint_masked`."""
    vrm = vmap(randint_masked, in_axes=(0, None))
    keys = split(key, 1)
    return vrm(keys.pop(size), mask)


class TestRandintMasked:
    """Test `mcmcstep.randint_masked`."""

    def test_all_false(self, keys: split) -> None:
        """Check what happens when no value is allowed."""
        for size in range(1, 10):
            u = randint_masked(keys.pop(), jnp.zeros(size, bool))
            assert u == size

    def test_all_true(self, keys: split) -> None:
        """Check it's equivalent to `randint` when all values are allowed."""
        key = keys.pop()
        size = 10_000
        u1 = randint_masked(key, jnp.ones(size, bool))
        u2 = random.randint(random.clone(key), (), 0, size)
        assert u1 == u2

    def test_no_disallowed_values(self, keys: split) -> None:
        """Check disallowed values are never selected."""
        key = keys.pop()
        for _ in range(100):
            keys = split(key, 3)
            mask = random.bernoulli(keys.pop(), 0.5, (10,))
            if not jnp.any(mask):  # pragma: no cover, rarely happens
                continue
            u = randint_masked(keys.pop(), mask)
            assert 0 <= u < mask.size
            assert mask[u]
            key = keys.pop()

    def test_correct_distribution(self, keys: split) -> None:
        """Check the distribution of values is uniform."""
        # create mask
        num_allowed = 10
        mask = jnp.zeros(2 * num_allowed, bool)
        mask = mask.at[:num_allowed].set(True)
        indices = jnp.arange(mask.size)
        indices = random.permutation(keys.pop(), indices)
        mask = mask[indices]

        # sample values
        n = 10_000
        u: Int32[Array, '{n}'] = vmap_randint_masked(keys.pop(), mask, n)
        u = indices[u]
        assert jnp.all(u < num_allowed)

        # check that the distribution is uniform
        # likelihood ratio test for multinomial with free p vs. constant p
        k = jnp.bincount(u, length=num_allowed)
        llr = jnp.sum(jnp.where(k, k * jnp.log(k / n * num_allowed), 0))
        lamda = 2 * llr
        pvalue = stats.chi2.sf(lamda, num_allowed - 1)
        assert pvalue > 0.1


class TestAncestorVariables:
    """Test `mcmcstep._moves.ancestor_variables`."""

    @pytest.fixture
    def depth2_tree(self) -> VarTreeData:
        R"""
        Tree with var_tree of size 4 (tree_depth=2, max_num_ancestors=1).

        Structure (heap indices):
              1 (root, var=2)
             / \
            2   3 (vars=0, 1)
           /\   /\
          4 5  6  7 (leaves, in leaf_tree only)

        Note: var_tree indices are 1-3, leaf indices 4-7 are beyond var_tree.
        """
        tree = manual_tree(
            [[0.0], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[2], [0, 1]], [[5], [3, 4]]
        )
        var_tree = tree.var_tree.astype(jnp.uint8)
        max_split = jnp.full(5, 10, jnp.uint8)
        return VarTreeData(var_tree, max_split)

    @pytest.fixture
    def depth3_tree(self) -> VarTreeData:
        """
        Tree with var_tree of size 8 (tree_depth=3, max_num_ancestors=2).

        Heap indices 1-7 in var_tree, 8-15 leaves.
        """
        tree = manual_tree(
            [[0.0], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0] * 8],
            [[3], [2, 1], [0, 4, 5, 6]],
            [[1], [2, 3], [4, 5, 6, 7]],
        )
        var_tree = tree.var_tree.astype(jnp.uint8)
        max_split = jnp.full(10, 10, jnp.uint8)
        return VarTreeData(var_tree, max_split)

    def test_root_node(self, depth2_tree: VarTreeData) -> None:
        """Check that root node has no ancestors (all slots filled with p)."""
        var_tree, max_split = depth2_tree

        # Root node (index 1) has no ancestors
        result = ancestor_variables(var_tree, max_split, jnp.int32(1))
        # var_tree size=4 -> tree_depth=2 -> max_num_ancestors=1
        # All slots should be p (sentinel) since root has no ancestors
        assert_array_equal(result, [max_split.size])

    def test_child_of_root(self, depth2_tree: VarTreeData) -> None:
        """Check that children of root have one ancestor (the root's variable)."""
        var_tree, max_split = depth2_tree

        # Left child of root (index 2): ancestor is root (var=2)
        result = ancestor_variables(var_tree, max_split, jnp.int32(2))
        assert result.shape == (1,)
        assert_array_equal(result, [2])

        # Right child of root (index 3): ancestor is root (var=2)
        result = ancestor_variables(var_tree, max_split, jnp.int32(3))
        assert_array_equal(result, [2])

    def test_deep_node(self, depth3_tree: VarTreeData) -> None:
        """Check ancestors for nodes at depth 3."""
        var_tree, max_split = depth3_tree

        # Node 4: parent is 2 (var=2), grandparent is 1 (var=3)
        result = ancestor_variables(var_tree, max_split, jnp.int32(4))
        assert result.shape == (2,)
        assert_array_equal(result, [3, 2])

        # Node 5: parent is 2 (var=2), grandparent is 1 (var=3)
        result = ancestor_variables(var_tree, max_split, jnp.int32(5))
        assert_array_equal(result, [3, 2])

        # Node 6: parent is 3 (var=1), grandparent is 1 (var=3)
        result = ancestor_variables(var_tree, max_split, jnp.int32(6))
        assert_array_equal(result, [3, 1])

        # Node 7: parent is 3 (var=1), grandparent is 1 (var=3)
        result = ancestor_variables(var_tree, max_split, jnp.int32(7))
        assert_array_equal(result, [3, 1])

    def test_intermediate_node(self, depth3_tree: VarTreeData) -> None:
        """Check ancestors for an intermediate (non-leaf) node."""
        var_tree, max_split = depth3_tree

        # Node 2: parent is 1 (var=3), one slot filled, one sentinel
        result = ancestor_variables(var_tree, max_split, jnp.int32(2))
        assert_array_equal(result, [max_split.size, 3])

        # Node 3: parent is 1 (var=3), one slot filled, one sentinel
        result = ancestor_variables(var_tree, max_split, jnp.int32(3))
        assert_array_equal(result, [max_split.size, 3])

    def test_single_variable(self) -> None:
        """Check with only one variable (p=1)."""
        tree = manual_tree(
            [[0.0], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0], [0, 0]], [[4], [3, 5]]
        )
        var_tree = tree.var_tree.astype(jnp.uint8)
        max_split = jnp.ones(1, minimal_unsigned_dtype(10))

        # Node 2: ancestor is root (var=0)
        result = ancestor_variables(var_tree, max_split, jnp.int32(2))
        assert_array_equal(result, [0])

        # Root has no ancestors
        result = ancestor_variables(var_tree, max_split, jnp.int32(1))
        assert_array_equal(result, [max_split.size])

    def test_type_edge(self, depth3_tree: VarTreeData) -> None:
        """Check that types are handled correctly when using uint8 and uint16 together."""
        var_tree, max_split = depth3_tree
        var_tree = var_tree.astype(jnp.uint8)
        max_split = jnp.full(256, 10, jnp.uint8)
        assert minimal_unsigned_dtype(max_split.size) == jnp.uint16

        # Node 2: parent is 1 (var=3), one slot filled, one sentinel
        result = ancestor_variables(var_tree, max_split, jnp.int32(2))
        assert_array_equal(result, [max_split.size, 3])

        # Node 3: parent is 1 (var=3), one slot filled, one sentinel
        result = ancestor_variables(var_tree, max_split, jnp.int32(3))
        assert_array_equal(result, [max_split.size, 3])


class TestRandintExclude:
    """Test `mcmcstep._moves.randint_exclude`."""

    def test_empty_exclude(self, keys: split) -> None:
        """If exclude is empty, it's equivalent to randint(key, (), 0, sup)."""
        key = keys.pop()
        sup = 10_000
        u1, num_allowed = randint_exclude(key, sup, jnp.array([], jnp.int32))
        u2 = random.randint(random.clone(key), (), 0, sup)
        assert num_allowed == sup
        assert u1 == u2

    def test_exclude_out_of_range_is_ignored(self, keys: split) -> None:
        """Values >= sup are ignored for both u and num_allowed."""
        key = keys.pop()
        sup = 7
        exclude = jnp.array([7, 8, 100, 7, 999])
        u, num_allowed = randint_exclude(key, sup, exclude)
        assert num_allowed == sup
        assert 0 <= u < sup

    def test_duplicate_excludes_ignored(self, keys: split) -> None:
        """Duplicates should be de-duplicated (set semantics for allowed count)."""
        sup = 10
        exclude_with_dupes = jnp.array([1, 1, 1, 3, 3, 9])
        exclude_unique = jnp.array([1, 3, 9])

        key = keys.pop()
        u1, n1 = randint_exclude(key, sup, exclude_with_dupes)
        u2, n2 = randint_exclude(random.clone(key), sup, exclude_unique)
        assert u1 == u2
        assert n1 == n2 == (sup - 3)

    def test_all_values_excluded_returns_sup(self, keys: split) -> None:
        """If all values are excluded, u must be sup and num_allowed=0."""
        for sup in range(1, 30, 5):
            exclude = jnp.arange(sup)
            u, num_allowed = randint_exclude(keys.pop(), sup, exclude)
            assert num_allowed == 0
            assert u == sup

    def test_never_returns_excluded_values(self, keys: split) -> None:
        """Across repeated sampling, u is always in [0,sup) and not excluded, unless num_allowed=0."""
        sup = 20
        reps = 200

        # Use a fixed-length exclude array; include invalid values so masking paths are hit.
        exclude = random.randint(keys.pop(), (reps, 30), 0, sup + 10)
        randint_exclude_v = vmap(randint_exclude, in_axes=(0, None, 0))
        keys_v = keys.pop(reps)
        u, num_allowed = randint_exclude_v(keys_v, sup, exclude)
        assert jnp.all(jnp.where(num_allowed == 0, u == sup, True))
        assert jnp.all(jnp.where(num_allowed == 0, True, u >= 0))
        assert jnp.all(jnp.where(num_allowed == 0, True, u < sup))
        # "not in exclude" should be understood modulo "exclude values >= sup are ignored"
        assert jnp.all(
            jnp.where(
                num_allowed == 0,
                True,
                ~jnp.any((exclude < sup) & (exclude == u[:, None]), axis=1),
            )
        )

    def test_num_allowed_matches_count(self, keys: split) -> None:
        """num_allowed must match sup - |unique(exclude ∩ [0,sup))|."""
        sup = 50
        reps = 50

        exclude = random.randint(
            keys.pop(), (reps, 80), 0, sup + 25
        )  # includes some >= sup

        randint_exclude_v = vmap(randint_exclude, in_axes=(0, None, 0))
        keys_v = keys.pop(reps)
        _, num_allowed = randint_exclude_v(keys_v, sup, exclude)

        # Expected count computed via set semantics on valid excluded values.
        # For each row, we replace invalid excluded values with `sup` (sentinel),
        # then count how many unique values are < sup.
        unique_v = vmap(
            lambda e: jnp.unique(jnp.minimum(e, sup), size=e.size, fill_value=sup)
        )
        valid_excluded = unique_v(exclude)
        expected_num_allowed = sup - jnp.sum(valid_excluded < sup, axis=1)

        assert jnp.all(num_allowed == expected_num_allowed)

    def test_correct_distribution_single_excluded(self, keys: split) -> None:
        """
        With one excluded value, u should be uniform over the remaining sup-1 values.

        We map u into a compact index in [0, sup-1) and run a chi-square GOF test.
        """
        sup = 8
        excluded = jnp.int32(3)
        exclude = jnp.array([excluded])

        n = 20_000
        keys_v = keys.pop(n)
        randint_exclude_v = vmap(randint_exclude, in_axes=(0, None, None))
        u, num_allowed = randint_exclude_v(keys_v, sup, exclude)

        assert jnp.all(num_allowed == (sup - 1))
        assert jnp.all(u != excluded)
        assert jnp.all((u >= 0) & (u < sup))

        # Map allowed values to 0..sup-2 by "closing the gap" at excluded.
        u_mapped = jnp.where(u < excluded, u, u - 1)
        k = jnp.bincount(u_mapped, length=sup - 1)

        # Chi-square GOF against uniform over sup-1 categories.
        expected = n / (sup - 1)
        chi2 = jnp.sum((k - expected) ** 2 / expected)
        pvalue = stats.chi2.sf(chi2, sup - 2)
        assert pvalue > 0.01


class TestSplitRange:
    """Test `mcmcstep._moves.split_range`."""

    @pytest.fixture
    def max_split(self) -> UInt8[Array, ' p']:
        """Maximum split indices for 3 variables."""
        # max_split[v] = maximum split index for variable v
        # split_range returns [l, r) in *1-based* split indices, so initial r = 1 + max_split[v]
        return jnp.array([10, 10, 10], dtype=jnp.uint8)

    @pytest.fixture
    def depth3_tree(self, max_split: UInt8[Array, ' p']) -> SplitRangeData:
        R"""
        Small depth-3 tree (var_tree size 8 => nodes 1..7 exist).

        Structure (heap indices):
              1 (var=0, split=5)
             / \
            2   3 (var=1, split=7; var=0, split=8)
           / \ / \
          4  5 6  7 (leaves or internal, but valid node indices for queries)

        This shape allows testing constraints from different ancestors (root + parent).
        """
        tree = manual_tree(
            [[0.0], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0] * 8],
            [[0], [1, 0], [0, 2, 2, 2]],
            [[5], [7, 8], [1, 1, 1, 1]],
        )
        var_tree = tree.var_tree.astype(jnp.uint8)
        split_tree = tree.split_tree.astype(jnp.uint8)
        return SplitRangeData(var_tree, split_tree, max_split)

    def test_dtypes(self, depth3_tree: SplitRangeData) -> None:
        """Check the output types."""
        var_tree, split_tree, max_split = depth3_tree
        l, r = split_range(
            var_tree, split_tree, max_split, jnp.int32(2), jnp.int32(max_split.size)
        )
        assert l.dtype == jnp.int32
        assert r.dtype == jnp.int32

    def test_ref_var_out_of_bounds(self, depth3_tree: SplitRangeData) -> None:
        """If ref_var is out of bounds, l=r=1."""
        var_tree, split_tree, max_split = depth3_tree
        l, r = split_range(
            var_tree, split_tree, max_split, jnp.int32(2), jnp.int32(max_split.size)
        )
        assert l == 1
        assert r == 1

    def test_root_node_no_constraints(self, depth3_tree: SplitRangeData) -> None:
        """Root has no ancestors => range should be the full [1, 1+max_split[var])."""
        var_tree, split_tree, max_split = depth3_tree

        # root is node_index=1, variable is var_tree[1]==0
        l, r = split_range(var_tree, split_tree, max_split, jnp.int32(1), jnp.int32(0))
        assert l == 1
        assert r == 1 + max_split[0]

    def test_unrelated_variable_no_constraints(
        self, depth3_tree: SplitRangeData
    ) -> None:
        """If ancestors don't use ref_var, range should be full [1, 1+max_split[ref_var])."""
        var_tree, split_tree, max_split = depth3_tree

        # node 6 path: 1 -> 3 -> 6, ancestors vars are [0 at node 1, 0 at node 3]
        # ref_var=2 never appears => no tightening
        l, r = split_range(var_tree, split_tree, max_split, jnp.int32(6), jnp.int32(2))
        assert l == 1
        assert r == 1 + max_split[2]

    def test_left_child_sets_upper_bound(self, depth3_tree: SplitRangeData) -> None:
        """For left subtree of an ancestor split on ref_var, r should be tightened to that split."""
        var_tree, split_tree, max_split = depth3_tree

        # node 2 is left child of root (root var=0, split=5)
        # For ref_var=0, being in left subtree implies x < 5 => r=min(r, 5)
        l, r = split_range(var_tree, split_tree, max_split, jnp.int32(2), jnp.int32(0))
        assert l == 1
        assert r == 5

    def test_right_child_sets_lower_bound(self, depth3_tree: SplitRangeData) -> None:
        """For right subtree of an ancestor split on ref_var, l should be raised to that split+1."""
        var_tree, split_tree, max_split = depth3_tree

        # node 3 is right child of root (root var=0, split=5)
        # For ref_var=0, being in right subtree implies x >= 5 => l becomes 5+1
        l, r = split_range(var_tree, split_tree, max_split, jnp.int32(3), jnp.int32(0))
        assert l == 6
        assert r == 1 + max_split[0]

    def test_two_ancestors_combine_bounds(self, depth3_tree: SplitRangeData) -> None:
        """Bounds from multiple ancestors on the same variable should combine (max lower, min upper)."""
        var_tree, split_tree, max_split = depth3_tree

        # node 6 path: 1 -> 3 -> 6
        # ancestor 1: var=0 split=5, node 6 is in right subtree => l>=6
        # ancestor 3: var=0 split=8, node 6 is in left subtree of node 3 => r<=8
        l, r = split_range(var_tree, split_tree, max_split, jnp.int32(6), jnp.int32(0))
        assert l == 6
        assert r == 8

    def test_ref_var_constraints_from_parent_only(
        self, depth3_tree: SplitRangeData
    ) -> None:
        """If only a deeper ancestor matches ref_var, constraints should come only from those matches."""
        var_tree, split_tree, max_split = depth3_tree

        # node 4 path: 1 -> 2 -> 4
        # root var=0 split=5 does not constrain ref_var=1
        # parent node 2 var=1 split=7, node 4 is left child => r<=7
        l, r = split_range(var_tree, split_tree, max_split, jnp.int32(4), jnp.int32(1))
        assert l == 1
        assert r == 7

    def test_no_allowed_splits_when_bounds_cross(
        self, max_split: UInt8[Array, ' p']
    ) -> None:
        """
        If constraints make the interval empty, l can become >= r.

        (The function does not clamp; consumers should handle it.)
        """
        # Build a minimal tree where:
        # - root splits var 0 at 8
        # - node 3 (right child) splits var 0 at 3
        # Query node 6 (left child of node 3):
        # - from root (right subtree): l = 8+1 = 9
        # - from node 3 (left subtree): r = 3
        tree = manual_tree(
            [[0.0], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0] * 8],
            [[0], [2, 0], [0, 2, 2, 2]],
            [[8], [1, 3], [1, 1, 1, 1]],
            ignore_errors=['check_rule_consistency'],
        )
        var_tree = tree.var_tree.astype(jnp.uint8)
        split_tree = tree.split_tree.astype(jnp.uint8)

        l, r = split_range(var_tree, split_tree, max_split, jnp.int32(6), jnp.int32(0))
        assert l == 9
        assert r == 3

    def test_minimal_tree(self) -> None:
        """Test the minimal tree."""
        # We want the shortest possible `var_tree`/`split_tree` arrays that still
        # represent a valid tree for the function:
        # - tree_depth(var_tree)=1  -> max_num_ancestors=0
        # - arrays therefore only need to include the unused 0 slot + root at index 1
        #   (size 2, indices 0..1).
        var_tree = jnp.array([0, 0], dtype=jnp.uint8)  # index 1 is root, var=0
        split_tree = jnp.array([0, 0], dtype=jnp.uint8)
        max_split = jnp.array([3], dtype=jnp.uint8)  # allow splits 1..3 (r should be 4)

        l, r = split_range(var_tree, split_tree, max_split, jnp.int32(1), jnp.int32(0))
        assert l == 1
        assert r == 4


@jaxtyped(typechecker=beartype)
@wraps(step)
def typechecking_step(key: Key[Array, ''], state: State) -> State:
    """Wrap `bartz.mcmcstep.step` because `jaxtyping.jaxtyped` can not be applied to a jitted function."""
    return step(key, state)


class TestMultichain:
    """Basic tests of the multichain functionality."""

    n = 100

    @pytest.fixture(
        params=['uv-binary', 'uv-continuous', 'mv-binary', 'mv-continuous', 'mv-mixed']
    )
    def init_kwargs(self, keys: split, request: pytest.FixtureRequest) -> dict:
        """Return arguments for `init`."""
        kind = request.param
        mv = kind.startswith('mv-')
        binary = kind.endswith('-binary')
        mixed = kind == 'mv-mixed'

        p = 10
        k = 2
        d = 6
        numcut = 10
        num_trees = 5
        X = random.randint(keys.pop(), (p, self.n), 0, numcut + 1, jnp.uint32)
        max_split = jnp.full(p, numcut + 1, jnp.uint32)

        if mixed:
            y = jnp.zeros((k, self.n), jnp.float32)
            y = y.at[0].set(
                random.bernoulli(keys.pop(), 0.5, (self.n,)).astype(jnp.float32)
            )
            y = y.at[1].set(random.normal(keys.pop(), (self.n,)))
            offset = random.normal(keys.pop(), (k,))
            leaf_prior_cov_inv = jnp.eye(k) * num_trees
        else:
            if mv:
                y_shape = (k, self.n)
                offset = random.normal(keys.pop(), (k,))
                leaf_prior_cov_inv = jnp.eye(k) * num_trees
            else:
                y_shape = (self.n,)
                offset = random.normal(keys.pop(), ())
                leaf_prior_cov_inv = jnp.float32(num_trees)

            if binary:
                y = random.bernoulli(keys.pop(), 0.5, y_shape).astype(jnp.float32)
            else:
                y = random.normal(keys.pop(), y_shape)

        kw = dict(
            X=X,
            y=y,
            offset=offset,
            max_split=max_split,
            num_trees=num_trees,
            p_nonterminal=jnp.full(d - 1, 0.9),
            leaf_prior_cov_inv=leaf_prior_cov_inv,
        )

        if mixed:
            kw.update(
                outcome_type=['binary', 'continuous'],
                error_cov_df=2.0,  # keep this a weak type
                error_cov_scale=jnp.diag(jnp.array([0.0, 2.0])),
            )
        elif binary:
            kw.update(outcome_type='binary')
        else:
            kw.update(
                error_cov_df=2.0,  # keep this a weak type
                error_cov_scale=2 * jnp.eye(k) if mv else 2.0,
            )

        return kw

    @pytest.mark.parametrize('num_chains', [None, 0, 1, -1, 4, -4])
    @pytest.mark.parametrize('shard_data', [False, True])
    def test_basic(
        self,
        init_kwargs: dict,
        num_chains: int | None,
        shard_data: bool,
        subtests: SubTests,
        keys: split,
    ) -> None:
        """Create a multichain `State` with `init` and step it once."""
        mesh = {}

        if num_chains is not None and num_chains < 0:
            num_chains = -num_chains
            mesh.update(chains=min(2, num_chains) if num_chains else 2)

        if shard_data:
            mesh.update(data=5)

        if not mesh:
            mesh = None
        else:
            targets = dict(chains=num_chains, data=self.n)
            while prod(mesh.values()) > get_device_count():
                for key in mesh:
                    if mesh[key] > 1:
                        mesh[key] -= 1
                        while targets[key] % mesh[key] != 0:
                            mesh[key] -= 1
                        break

        with subtests.test('init'):
            typechecking_init = jaxtyped(init, typechecker=beartype)
            state = typechecking_init(**init_kwargs, num_chains=num_chains, mesh=mesh)
            assert state.forest.num_chains() == num_chains
            check_strong_types(state)
            check_sharding(state, state.config.mesh)

        with subtests.test('step'):
            with debug_key_reuse(False):
                # key reuse checks trigger with empty key array apparently
                new_state = typechecking_step(keys.pop(), state)
            assert new_state.forest.num_chains() == num_chains
            check_strong_types(new_state)
            check_sharding(new_state, state.config.mesh)
            check_same_structure(state, new_state)

    def test_multichain_equiv_stack(self, init_kwargs: dict, keys: split) -> None:
        """Check that stacking multiple chains is equivalent to a multichain trace."""
        num_chains = 4
        num_iters = 10

        copy_args = partial(copy_arrays, init_kwargs)

        # create initial states
        mc_state = init(**copy_args(), num_chains=num_chains)
        sc_states = [
            init(
                **copy_args(),
                num_chains=None,
                resid_num_batches=mc_state.config.resid_num_batches,
                count_num_batches=mc_state.config.count_num_batches,
                prec_num_batches=mc_state.config.prec_num_batches,
            )
            for _ in range(num_chains)
        ]

        # run a few mcmc steps with the same random keys
        for _ in range(num_iters):
            mc_key = keys.pop()
            sc_keys = random.split(random.clone(mc_key), num_chains)

            mc_state = step(mc_key, mc_state)
            sc_states = [
                step(key, state) for key, state in zip(sc_keys, sc_states, strict=True)
            ]

        # stack single-chain states
        def stack_leaf(
            _path: KeyPath,
            chain_axis: int | None,
            mc_x: Array | None,
            *sc_xs: Array | None,
        ) -> Array | None:
            if chain_axis is None or mc_x is None:
                return mc_x
            else:
                return jnp.stack(sc_xs, axis=chain_axis)

        chain_axes = chain_vmap_axes(mc_state)
        stacked_state = tree.map_with_path(
            stack_leaf, chain_axes, mc_state, *sc_states, is_leaf=lambda x: x is None
        )

        # check the mc state is equal to the stacked state
        def check_equal(path: KeyPath, mc: Array, stacked: Array) -> None:
            str_path = keystr(path)
            exact = mc.platform() == 'cpu' or jnp.issubdtype(mc.dtype, jnp.integer)
            assert_close_matrices(
                mc,
                stacked,
                err_msg=f'{str_path}: ',
                rtol=0 if exact else 1e-5,
                reduce_rank=True,
            )

        tree.map_with_path(check_equal, mc_state, stacked_state)

    def chain_vmap_axes(self, state: State) -> State:
        """Old manual version of `chain_vmap_axes(_: State)`."""

        def choose_vmap_index(path: KeyPath, _: Array) -> Literal[0, None]:
            no_vmap_attrs = (
                '.X',
                '.binary_y',
                '.binary_indices',
                '.offset',
                '.prec_scale',
                '.error_cov_df',
                '.error_cov_scale',
                '.forest.max_split',
                '.forest.blocked_vars',
                '.forest.p_nonterminal',
                '.forest.p_propose_grow',
                '.forest.min_points_per_decision_node',
                '.forest.min_points_per_leaf',
                '.forest.leaf_prior_cov_inv',
                '.forest.a',
                '.forest.b',
                '.forest.rho',
                '.config.sparse_on_at',
                '.config.steps_done',
            )
            if keystr(path) in no_vmap_attrs:
                return None
            else:
                return 0

        return tree.map_with_path(choose_vmap_index, state)

    def data_vmap_axes(self, state: State) -> State:
        """Hardcoded version of `data_vmap_axes(_: State)`."""

        def choose_vmap_index(path: KeyPath, _: Array) -> Literal[-1, None]:
            vmap_attrs = (
                '.X',
                '.binary_y',
                '.z',
                '.resid',
                '.prec_scale',
                '.forest.leaf_indices',
            )
            if keystr(path) in vmap_attrs:
                return -1
            else:
                return None

        return tree.map_with_path(choose_vmap_index, state)

    def test_vmap_axes(self, init_kwargs: dict) -> None:
        """Check `data_vmap_axes` and `chain_vmap_axes` on a `State`."""
        state = init(**init_kwargs)

        chain_axes = chain_vmap_axes(state)
        data_axes = data_vmap_axes(state)

        ref_chain_axes = self.chain_vmap_axes(state)
        ref_data_axes = self.data_vmap_axes(state)

        def assert_equal(
            _path: KeyPath, axis: int | None, ref_axis: int | None
        ) -> None:
            assert axis == ref_axis

        tree.map_with_path(assert_equal, chain_axes, ref_chain_axes)
        tree.map_with_path(assert_equal, data_axes, ref_data_axes)

    def test_normalize_spec(self) -> None:
        """Test `normalize_spec`."""
        devices = jax.devices('cpu')[:3]
        mesh = make_mesh(
            (len(devices), 1),
            ('ciao', 'bau'),
            axis_types=(AxisType.Auto, AxisType.Auto),
            devices=devices,
        )
        assert normalize_spec(['ciao'], mesh, (1, 1, 1)) == PartitionSpec(
            'ciao' if len(devices) > 1 else None, None, None
        )
        assert normalize_spec([None, 'bau'], mesh, (1, 1)) == PartitionSpec(None, None)
        assert normalize_spec(['ciao'], mesh, (0,)) == PartitionSpec(None)
        assert normalize_spec([None, 'ciao'], mesh, (0, 1)) == PartitionSpec(None, None)


def check_sharding(x: PyTree, mesh: Mesh | None) -> None:
    """Check that chains and data are sharded as expected."""
    chain_axes = chain_vmap_axes(x)
    data_axes = data_vmap_axes(x)

    def check_leaf(
        _path: KeyPath, x: Array | None, chain_axis: int | None, data_axis: int | None
    ) -> None:
        if x is None:
            return
        elif mesh is None:
            assert isinstance(x.sharding, SingleDeviceSharding)
        else:
            spec = get_normal_spec(x)

            expected_spec = [None] * x.ndim
            if 'chains' in mesh.axis_names and chain_axis is not None:
                expected_spec[chain_axis] = 'chains'
            if 'data' in mesh.axis_names and data_axis is not None:
                expected_spec[data_axis] = 'data'
            expected_spec = normalize_spec(expected_spec, mesh, x.shape)

            assert spec == expected_spec

    tree.map_with_path(
        check_leaf, x, chain_axes, data_axes, is_leaf=lambda x: x is None
    )


def get_normal_spec(x: Array) -> PartitionSpec:
    """Get the partition spec of `x` and apply `normalize_spec`."""
    spec = x.sharding.spec
    mesh = x.sharding.mesh
    return normalize_spec(spec, mesh, x.shape)


def normalize_spec(
    spec: Sequence[str | None], mesh: Mesh, shape: tuple[int, ...]
) -> PartitionSpec:
    """Put a spec in standard form, i.e., fill with `None` until length `ndim` and put `None` on axes with mesh size 1 or if array size is 0."""
    s = list(spec)
    ndim = len(shape)
    assert len(s) <= ndim
    s.extend([None] * (ndim - len(s)))

    array_size = prod(shape)
    for i in range(ndim):
        if s[i] is not None:
            j = mesh.axis_names.index(s[i])
            mesh_size = mesh.axis_sizes[j]
            if mesh_size == 1 or array_size == 0:
                s[i] = None

    assert len(s) == ndim
    return PartitionSpec(*s)


def check_strong_types(x: PyTree[Array]) -> None:
    """Check all arrays in `x` have strong types."""

    def check_leaf(path: KeyPath, x: Array) -> None:
        assert not x.weak_type, f'{keystr(path)} has weak type'

    tree.map_with_path(check_leaf, x)


def check_same_structure(x: PyTree, y: PyTree) -> None:
    """Check that two PyTrees have the same structure, incl. shape and type of the arrays."""

    def check(_path: KeyPath, x: Array, y: Array) -> None:
        assert x.shape == y.shape
        assert x.dtype == y.dtype
        if jax.__version_info__ >= (0, 6, 0):
            # not actually know if this will work properly in 0.6.0
            assert x.sharding.is_equivalent_to(y.sharding, x.ndim)

    tree.map_with_path(check, x, y)


class TestMixedBinaryContinuous:
    """Tests for mixed binary-continuous multivariate outcome support."""

    n = 100
    p = 10
    k = 3
    numcut = 10
    num_trees = 5
    d = 6

    @pytest.fixture
    def init_kwargs(self, keys: split) -> dict:
        """Return arguments for `init` with mixed binary-continuous outcomes."""
        X = random.randint(keys.pop(), (self.p, self.n), 0, self.numcut + 1, jnp.uint32)
        max_split = jnp.full(self.p, self.numcut + 1, jnp.uint32)

        y = jnp.zeros((self.k, self.n), jnp.float32)
        y = y.at[0].set(random.bernoulli(keys.pop(), 0.5, (self.n,)))
        y = y.at[1].set(random.normal(keys.pop(), (self.n,)))
        y = y.at[2].set(random.bernoulli(keys.pop(), 0.3, (self.n,)))

        return dict(
            X=X,
            y=y,
            outcome_type=['binary', 'continuous', 'binary'],
            offset=random.normal(keys.pop(), (self.k,)),
            max_split=max_split,
            num_trees=self.num_trees,
            p_nonterminal=jnp.full(self.d - 1, 0.9),
            leaf_prior_cov_inv=jnp.eye(self.k) * self.num_trees,
            error_cov_df=2.0,
            error_cov_scale=jnp.diag(jnp.array([0.0, 2.0, 0.0])),
        )

    def test_init_shapes(self, init_kwargs: dict) -> None:
        """Check that init produces correct shapes for mixed outcomes."""
        state = init(**init_kwargs)

        # binary_indices should contain indices of binary components
        assert state.binary_indices is not None
        assert_array_equal(state.binary_indices, jnp.array([0, 2], jnp.int32))

        # binary_y should have only binary rows (kb=2)
        assert state.binary_y is not None
        assert state.binary_y.shape == (2, self.n)
        assert state.binary_y.dtype == jnp.bool_

        # z should have only binary rows (kb=2)
        assert state.z is not None
        assert state.z.shape == (2, self.n)

        # resid should have all k rows
        assert state.resid.shape == (self.k, self.n)

        # error_cov_inv should be a (k, k) diagonal matrix
        assert state.error_cov_inv is not None
        assert state.error_cov_inv.shape == (self.k, self.k)
        # off-diagonal should be zero
        assert_array_equal(state.error_cov_inv, jnp.diag(jnp.diag(state.error_cov_inv)))

        # binary diagonal entries should be 1.0
        assert state.error_cov_inv[0, 0] == 1.0
        assert state.error_cov_inv[2, 2] == 1.0

        # error_cov_df and error_cov_scale should be set
        assert state.error_cov_df is not None
        assert state.error_cov_scale is not None

    def test_init_binary_y_values(self, init_kwargs: dict) -> None:
        """Check that binary_y correctly extracts binary components from y."""
        y = init_kwargs['y']
        state = init(**init_kwargs)

        assert state.binary_y is not None
        # binary_y[0] should correspond to y[0] (first binary component)
        assert_array_equal(state.binary_y[0], y[0] != 0)
        # binary_y[1] should correspond to y[2] (second binary component)
        assert_array_equal(state.binary_y[1], y[2] != 0)

    def test_init_resid_binary_rows_zero(self, init_kwargs: dict) -> None:
        """Check that the binary rows of resid are initialized to zero."""
        state = init(**init_kwargs)

        # binary rows (0 and 2) should be zero
        assert_array_equal(state.resid[0], jnp.zeros(self.n))
        assert_array_equal(state.resid[2], jnp.zeros(self.n))

        # continuous row (1) should be y[1] - offset[1]
        y = init_kwargs['y']
        offset = init_kwargs['offset']
        expected = y[1] - offset[1]
        assert_array_equal(state.resid[1], expected)

    def test_init_z_values(self, init_kwargs: dict) -> None:
        """Check that z is initialized to offset for binary components."""
        state = init(**init_kwargs)

        assert state.z is not None
        offset = init_kwargs['offset']
        # z[0] should be offset[0] (first binary component)
        assert_array_equal(state.z[0], jnp.full(self.n, offset[0]))
        # z[1] should be offset[2] (second binary component, index 2 in y)
        assert_array_equal(state.z[1], jnp.full(self.n, offset[2]))

    def test_init_rejects_nondiagonal_scale(self, init_kwargs: dict) -> None:
        """Check that init rejects non-diagonal error_cov_scale."""
        init_kwargs['error_cov_scale'] += 0.1 * jnp.ones((self.k, self.k))
        with pytest.raises(Exception, match='diagonal'):
            _state = init(**init_kwargs)

    def test_init_rejects_error_scale(self, init_kwargs: dict) -> None:
        """Check that init rejects error_scale for mixed outcomes."""
        with pytest.raises(AssertionError, match='error_scale'):
            init(**init_kwargs, error_scale=jnp.ones(self.n))

    def test_step_z_updates_only_binary_resid(
        self, init_kwargs: dict, keys: split
    ) -> None:
        """Check that step_z modifies only the binary rows of resid."""
        state = init(**init_kwargs)

        # run a few tree steps first so resid is nonzero
        state = step_trees(keys.pop(), state)

        new_state = step_z(keys.pop(), state)

        # continuous row (index 1) should be unchanged
        assert_array_equal(new_state.resid[1], state.resid[1])

        # binary rows should generally change (could be same by extreme
        # coincidence, but practically never for 100 points)
        assert not jnp.array_equal(new_state.resid[0], state.resid[0])

    def test_step_error_cov_inv_updates_only_continuous(
        self, init_kwargs: dict, keys: split
    ) -> None:
        """Check that step_error_cov_inv updates only continuous diagonal entries."""
        state = init(**init_kwargs)
        prec = state.error_cov_inv[1, 1]

        # replace resid because the default initial resid is 0 for binary
        # outcomes, which triggers a division by zero in step_error_cov_inv
        state = replace(state, resid=jnp.full_like(state.resid, 1.0))

        new_state = step_error_cov_inv(keys.pop(), state)

        # binary diagonal entries (indices 0, 2) should stay 1.0
        assert new_state.error_cov_inv[0, 0] == 1.0
        assert new_state.error_cov_inv[2, 2] == 1.0

        # continuous diagonal entry (index 1) should be updated (not the init value)
        assert new_state.error_cov_inv[1, 1] != prec

        # off-diagonal should remain zero
        assert_array_equal(
            new_state.error_cov_inv, jnp.diag(jnp.diag(new_state.error_cov_inv))
        )

    @pytest.mark.parametrize('outcome_type', ['binary', 'continuous'])
    def test_all_same_outcome_sequence(
        self, outcome_type: str, keys: split, init_kwargs: dict
    ) -> None:
        """Check that uniform sequence outcome_type matches the scalar form."""
        if outcome_type == 'binary':
            init_kwargs.update(
                y=random.bernoulli(keys.pop(), 0.5, (self.k, self.n)).astype(
                    jnp.float32
                ),
                error_cov_df=None,
                error_cov_scale=None,
            )
        else:
            init_kwargs.update(
                y=random.normal(keys.pop(), (self.k, self.n)),
                error_cov_df=2.0,
                error_cov_scale=2 * jnp.eye(self.k),
            )

        copy_args = partial(copy_arrays, init_kwargs)

        init_kwargs.update(outcome_type=outcome_type)
        scalar_state = init(**copy_args())

        init_kwargs.update(outcome_type=[outcome_type] * self.k)
        sequence_state = init(**copy_args())

        def check_equal(path: KeyPath, scalar: Array, sequence: Array) -> None:
            assert_array_equal(scalar, sequence, err_msg=f'{keystr(path)}: ')

        tree.map_with_path(check_equal, scalar_state, sequence_state)

    def test_outcome_type_length_mismatch(self, init_kwargs: dict) -> None:
        """Check that mismatched outcome_type length raises."""
        init_kwargs.update(outcome_type=['binary'] * (self.k - 1))
        with pytest.raises(AssertionError):
            init(**init_kwargs)


class MCMCStepData(NamedTuple):
    """Toy dataset for testing."""

    X: Int32[Array, 'p n']
    y: Float32[Array, ' n']
    max_split: UInt32[Array, ' p']


def random_pd_matrix(key: Key[Array, ''], k: int) -> Float[Array, '{k} {k}']:
    """Generate a random positive definite matrix."""
    A = random.normal(key, (k, k))
    return A @ A.T + jnp.eye(k)


@pytest.fixture(params=[(10, 2), (20, 5), (3, 100), (50, 50)])
def mcmcstep_data_shape(request: FixtureRequest) -> tuple[int, int]:
    """Provide (n, p) pairs for testing."""
    return request.param


@pytest.fixture
def mcmcstep_data(mcmcstep_data_shape: tuple[int, int]) -> MCMCStepData:
    """Generate a toy dataset."""
    n, p = mcmcstep_data_shape
    X = jnp.arange(n * p).reshape(p, n)
    y = jnp.linspace(-1, 1, n)
    max_split = jnp.full(p, 5, dtype=jnp.uint32)
    return MCMCStepData(X, y, max_split)


class TestWishart:
    """Test the basic properties of the wishart sampler output."""

    # Parameterize with (k, df) pairs
    @pytest.fixture(params=[(1, 3), (3, 3), (3, 5), (3, 100), (100, 102)])
    def wishart_params(self, request: FixtureRequest) -> tuple[int, int]:
        """Provide (k, df) pairs for testing."""
        k, df = request.param
        return k, df

    def ill_conditioned_matrix(
        self, key: Key[Array, ''], k: int, condition_number: float = 1e6
    ) -> Float[Array, '{k} {k}']:
        """Generate a ill conditioned random positive semi-definite matrix."""
        A = random.normal(key, (k, k))
        U, _ = jnp.linalg.qr(A)

        if k == 1:
            eigs = jnp.zeros(1)
        else:
            smalls = jnp.geomspace(1.0, 1.0 / condition_number, num=k - 1)
            eigs = jnp.concatenate([smalls, jnp.array([0.0])])
        return (U * eigs) @ U.T

    def test_size(self, keys: split, wishart_params: tuple[int, int]) -> None:
        """Check that the sample generated by wishart sampler is of shape k*k."""
        k, df = wishart_params
        scale = random_pd_matrix(keys.pop(), k)
        sample = _sample_wishart_bartlett(keys.pop(), df, scale)
        assert sample.shape == (k, k)

    def test_symmetric(self, keys: split, wishart_params: tuple[int, int]) -> None:
        """Check that the sample generated by wishart sampler is symmetric."""
        k, df = wishart_params
        scale = random_pd_matrix(keys.pop(), k)
        sample = _sample_wishart_bartlett(keys.pop(), df, scale)
        assert_close_matrices(sample, sample.T, rtol=1e-6)

    def test_pos_def(self, keys: split, wishart_params: tuple[int, int]) -> None:
        """Check that the sample generated by wishart sampler is positive definite."""
        k, df = wishart_params
        scale = random_pd_matrix(keys.pop(), k)
        sample = _sample_wishart_bartlett(keys.pop(), df, scale)
        eigs = jnp.linalg.eigvalsh(sample)
        assert jnp.all(eigs > 0)

    def test_near_singular_scale(
        self, keys: split, wishart_params: tuple[int, int]
    ) -> None:
        """Check that the wishart sampler still works with singular or near singular matrix."""
        k, df = wishart_params
        ill_conditioned_scale = self.ill_conditioned_matrix(keys.pop(), k)
        sample = _sample_wishart_bartlett(keys.pop(), df, ill_conditioned_scale)
        assert jnp.all(jnp.isfinite(sample))

    def test_wishart_dist(self, keys: split, wishart_params: tuple[int, int]) -> None:
        """Check that the sample generated by wishart sampler follows a wishart distribution."""
        k, df = wishart_params
        sigma = random_pd_matrix(keys.pop(), k)
        scale_inv = jnp.linalg.inv(sigma)

        a = random.normal(keys.pop(), (k,))
        denominator = a.T @ sigma @ a

        sampler = vmap(_sample_wishart_bartlett, in_axes=(0, None, None))
        W = sampler(keys.pop(1000), float(df), scale_inv)
        t = jnp.einsum('ijk,j,k->i', W, a, a) / denominator

        test = ks_1samp(t, chi2(df).cdf)
        assert test.pvalue > 0.01


class TestPrecomputeTerms:
    """Test _precompute_likelihood_terms_mv and _precompute_leaf_terms_mv correctness and stability."""

    @pytest.fixture(params=[1, 2, 5, 10])
    def k(self, request: FixtureRequest) -> int:
        """Provide different ks for testing."""
        return request.param

    def test_shapes_leaf(self, keys: split, k: int) -> None:
        """Check that shapes of outputs are correct."""
        num_trees, tree_size = 3, 4
        prec_trees = jnp.ones((num_trees, tree_size))
        error_cov_inv = random_pd_matrix(keys.pop(), k)
        leaf_prior_cov_inv = random_pd_matrix(keys.pop(), k)

        result = _precompute_leaf_terms_mv(
            keys.pop(), prec_trees, error_cov_inv, leaf_prior_cov_inv
        )
        assert result.mean_factor.shape == (num_trees, k, k, tree_size)
        assert result.centered_leaves.shape == (num_trees, k, tree_size)

    def test_likelihood_equiv(self, keys: split) -> None:
        """Check that _compute_likelihood_ratio_uv and _compute_likelihood_ratio_mv agree when k = 1."""
        inv_sigma2 = random.uniform(keys.pop(), (), minval=0.1, maxval=5.0)
        leaf_prior_cov_inv_uv = random.uniform(keys.pop(), (), minval=0.1, maxval=5.0)
        error_cov_inv = jnp.array([[inv_sigma2]])
        leaf_prior_cov_inv = jnp.array([[leaf_prior_cov_inv_uv]])

        precs = Counts(left=jnp.array(3.0), right=jnp.array(4.0), total=jnp.array(7.0))

        total_resid = random.normal(keys.pop(), (1,))
        left_resid = random.normal(keys.pop(), (1,))
        right_resid = random.normal(keys.pop(), (1,))

        prelkv_mv, _ = _precompute_likelihood_terms_mv(
            error_cov_inv, leaf_prior_cov_inv, precs
        )
        likelihood_mv = _compute_likelihood_ratio_mv(
            total_resid, left_resid, right_resid, prelkv_mv
        )

        prelkv_uv, prelk_uv = _precompute_likelihood_terms_uv(
            inv_sigma2, leaf_prior_cov_inv_uv, precs
        )
        likelihood_uv = _compute_likelihood_ratio_uv(
            total_resid.item(),
            left_resid.item(),
            right_resid.item(),
            prelkv_uv,
            prelk_uv,
        )

        assert_allclose(
            prelkv_mv.log_sqrt_term, prelkv_uv.log_sqrt_term, rtol=1e-6, atol=1e-6
        )
        assert_allclose(likelihood_mv, likelihood_uv, rtol=1e-6, atol=1e-6)

    def test_leaf_terms_equiv(self, keys: split) -> None:
        """Check that _precompute_leaf_terms_uv and _precompute_leaf_terms_mv agree when k = 1."""
        num_trees, tree_size = 2, 3
        inv_sigma2 = random.uniform(keys.pop(), (), minval=0.1, maxval=5.0)
        leaf_prior_cov_inv_uv = random.uniform(keys.pop(), (), minval=0.1, maxval=5.0)

        error_cov_inv = jnp.array([[inv_sigma2]])
        leaf_prior_cov_inv = jnp.array([[leaf_prior_cov_inv_uv]])
        prec_trees = random.uniform(keys.pop(), (num_trees, tree_size)) * 5.0
        z_mv = random.normal(keys.pop(), (num_trees, tree_size, 1))
        z_uv = z_mv.squeeze(axis=-1)

        result_uv = _precompute_leaf_terms_uv(
            keys.pop(), prec_trees, inv_sigma2, leaf_prior_cov_inv_uv, z_uv
        )
        result_mv = _precompute_leaf_terms_mv(
            keys.pop(), prec_trees, error_cov_inv, leaf_prior_cov_inv, z_mv
        )

        assert_allclose(
            result_uv.mean_factor,
            result_mv.mean_factor.squeeze((1, 2)),
            rtol=1e-6,
            atol=1e-6,
        )
        assert_allclose(
            result_uv.centered_leaves,
            result_mv.centered_leaves.squeeze(1),
            rtol=1e-6,
            atol=1e-6,
        )


class TestMVBartIntegration:
    """Test equivalence between Univariate and Multivariate (k=1) modes."""

    @pytest.mark.parametrize('binary', [False, True])
    def test_init_equivalence(self, mcmcstep_data: MCMCStepData, binary: bool) -> None:
        """Test that init produces compatible structures for UV and MV(k=1)."""
        X, y, max_split = mcmcstep_data
        p_nonterminal = jnp.array([0.9, 0.5])

        if binary:
            y = (y > 0).astype(jnp.float32)

        common = partial(
            copy_arrays,
            dict(
                X=X,
                max_split=max_split,
                num_trees=10,
                p_nonterminal=p_nonterminal,
                resid_num_batches=None,
                count_num_batches=None,
            ),
        )

        uv_kw: dict = dict(y=y, offset=0.0, leaf_prior_cov_inv=1.0)
        mv_kw: dict = dict(
            y=y[None, :], offset=jnp.zeros(1), leaf_prior_cov_inv=jnp.array([[1.0]])
        )

        if binary:
            uv_kw.update(outcome_type='binary')
            mv_kw.update(outcome_type='binary')
        else:
            uv_kw.update(error_cov_df=6.0, error_cov_scale=4.0)
            mv_kw.update(error_cov_df=jnp.array(6.0), error_cov_scale=4.0 * jnp.eye(1))

        bart_uv = init(**uv_kw, **common())
        bart_mv = init(**mv_kw, **common())

        assert bart_uv.resid.ndim == 1
        assert bart_mv.resid.ndim == 2
        assert bart_mv.resid.shape[0] == 1
        assert bart_mv.resid.shape[1] == bart_uv.resid.shape[0]

        assert jnp.ndim(bart_uv.error_cov_inv) == 0
        assert bart_mv.error_cov_inv.shape == (1, 1)

        if binary:
            assert bart_uv.binary_y is not None
            assert bart_mv.binary_y is not None
            assert bart_uv.binary_y.ndim == 1
            assert bart_mv.binary_y.ndim == 2
            assert_array_equal(bart_uv.binary_y, bart_mv.binary_y.squeeze(0))
            assert bart_uv.z is not None
            assert bart_mv.z is not None
            assert bart_uv.z.ndim == 1
            assert bart_mv.z.ndim == 2
            assert_array_equal(bart_uv.z, bart_mv.z.squeeze(0))

        assert_array_equal(bart_uv.resid, bart_mv.resid.squeeze(0))
        assert_array_equal(bart_uv.forest.var_tree, bart_mv.forest.var_tree)
        assert_array_equal(bart_uv.forest.split_tree, bart_mv.forest.split_tree)
        assert_array_equal(
            bart_uv.forest.leaf_tree, bart_mv.forest.leaf_tree.squeeze(1)
        )
        assert_array_equal(bart_uv.forest.leaf_indices, bart_mv.forest.leaf_indices)
        assert_array_equal(bart_uv.forest.p_nonterminal, bart_mv.forest.p_nonterminal)
        assert_array_equal(bart_uv.forest.p_propose_grow, bart_mv.forest.p_propose_grow)
        assert_array_equal(bart_uv.forest.affluence_tree, bart_mv.forest.affluence_tree)

    def test_step_sigma_distribution_match(
        self, keys: split, mcmcstep_data: MCMCStepData
    ) -> None:
        """
        Test that _step_error_cov_inv_uv and _step_error_cov_inv_mv (k = 1) sample from the same posterior.

        UV: 1/sigma2 ~ Gamma(alpha_post, beta_post)
        MV: error_cov_inv ~ Wishart(df_post, scale_post)
        """
        X, y, _ = mcmcstep_data
        resid = random.normal(keys.pop(), (y.size,))

        # inverse gamma prior: alpha = df / 2, beta = scale / 2
        df_prior = jnp.float32(20.0)
        scale_prior = jnp.float32(10.0)

        common: dict = dict(
            X=X,
            binary_y=None,
            binary_indices=None,
            error_cov_df=df_prior,
            z=None,
            offset=0.0,
            prec_scale=None,
            forest=None,
            config=None,
        )

        st_uv = State(
            **common, resid=resid, error_cov_scale=scale_prior, error_cov_inv=1.0
        )

        st_mv = State(
            **common,
            resid=resid[None, :],
            error_cov_scale=jnp.array([[scale_prior]]),
            error_cov_inv=jnp.eye(1),
        )

        def sample_uv(k: Key[Array, '']) -> Float32[Array, '']:
            return _step_error_cov_inv_uv(k, st_uv).error_cov_inv

        def sample_mv(k: Key[Array, '']) -> Float32[Array, '']:
            return _step_error_cov_inv_mv(k, st_mv).error_cov_inv.reshape(())

        n_samples = 10000
        samples_uv = vmap(sample_uv)(keys.pop(n_samples))
        samples_mv = vmap(sample_mv)(keys.pop(n_samples))

        _, p_value = ks_2samp(samples_uv, samples_mv)

        assert jnp.abs(jnp.mean(samples_uv) - jnp.mean(samples_mv)) < 0.01
        assert p_value > 0.01


class TestMVBartSteps:
    """Test the full MCMC step trajectory (init + multiple steps)."""

    @pytest.mark.parametrize('binary', [False, True])
    def test_step_trees_exact_match(
        self, keys: split, mcmcstep_data: MCMCStepData, binary: bool
    ) -> None:
        """Test that MV tree logic is Identical to UV logic."""
        X, y, max_split = mcmcstep_data
        n_trees = 100

        if binary:
            y = (y > 0).astype(jnp.float32)

        params = partial(
            copy_arrays,
            dict(
                X=X,
                max_split=max_split,
                num_trees=n_trees,
                p_nonterminal=jnp.array([0.9, 0.5]),
                resid_num_batches=None,
                count_num_batches=None,
            ),
        )

        uv_kw: dict = dict(y=y, offset=0.0, leaf_prior_cov_inv=jnp.float32(n_trees))
        mv_kw: dict = dict(
            y=y[None, :], offset=jnp.zeros(1), leaf_prior_cov_inv=n_trees * jnp.eye(1)
        )

        if binary:
            uv_kw.update(outcome_type='binary')
            mv_kw.update(outcome_type='binary')
        else:
            uv_kw.update(error_cov_df=4.0, error_cov_scale=2.0)
            mv_kw.update(error_cov_df=jnp.array(4.0), error_cov_scale=2 * jnp.eye(1))

        uv_state = init(**uv_kw, **params())
        mv_state = init(**mv_kw, **params())

        mv_state = replace(
            mv_state,
            resid=uv_state.resid[None, :],
            error_cov_inv=jnp.array([[uv_state.error_cov_inv]]),
            forest=replace(
                mv_state.forest,
                var_tree=uv_state.forest.var_tree,
                split_tree=uv_state.forest.split_tree,
                leaf_tree=uv_state.forest.leaf_tree[:, None, :],
                leaf_indices=uv_state.forest.leaf_indices,
                affluence_tree=uv_state.forest.affluence_tree,
            ),
        )

        key = keys.pop()
        uv_next = step_trees(key, uv_state)
        mv_next = step_trees(random.clone(key), mv_state)

        assert_close_matrices(
            uv_next.resid, mv_next.resid.squeeze(0), atol=1e-6, rtol=1e-6
        )
        assert_close_matrices(
            uv_state.forest.leaf_tree,
            mv_state.forest.leaf_tree.squeeze(1),
            atol=1e-6,
            rtol=1e-6,
        )

        assert_array_equal(uv_state.forest.var_tree, mv_state.forest.var_tree)
        assert_array_equal(uv_state.forest.split_tree, mv_state.forest.split_tree)
        assert_array_equal(uv_state.forest.leaf_indices, mv_state.forest.leaf_indices)
        assert_array_equal(
            uv_state.forest.affluence_tree, mv_state.forest.affluence_tree
        )

        assert_array_equal(
            uv_state.forest.grow_prop_count, mv_state.forest.grow_prop_count
        )
        assert_array_equal(
            uv_state.forest.grow_acc_count, mv_state.forest.grow_acc_count
        )
        assert_array_equal(
            uv_state.forest.prune_prop_count, mv_state.forest.prune_prop_count
        )
        assert_array_equal(
            uv_state.forest.prune_acc_count, mv_state.forest.prune_acc_count
        )

    @pytest.mark.parametrize('binary', [False, True])
    def test_mv_steps(
        self, keys: split, mcmcstep_data: MCMCStepData, binary: bool
    ) -> None:
        """Test that mv mode can run without crashing."""
        X, y_uv, max_split = mcmcstep_data
        k = 3

        if binary:
            y = random.bernoulli(keys.pop(), 0.5, (k, y_uv.size)).astype(jnp.float32)
        else:
            y = jnp.tile(y_uv, (k, 1))
            y = y + random.normal(keys.pop(), y.shape) * 0.1

        kw: dict = dict(
            X=X,
            y=y,
            offset=jnp.zeros(k),
            max_split=max_split,
            num_trees=5,
            p_nonterminal=jnp.array([0.9, 0.5]),
            leaf_prior_cov_inv=jnp.eye(k),
            resid_num_batches=None,
            count_num_batches=None,
        )

        if binary:
            kw.update(outcome_type='binary')
        else:
            kw.update(error_cov_df=jnp.array(10.0), error_cov_scale=jnp.eye(k))

        mv_state = init(**kw)

        for key in keys.pop(10):
            mv_state = step(key, mv_state)

            assert jnp.all(jnp.isfinite(mv_state.resid))
            assert jnp.all(jnp.isfinite(mv_state.forest.leaf_tree))

            assert mv_state.resid.shape == (k, y.shape[1])

            assert jnp.all(jnp.isfinite(mv_state.error_cov_inv))
            assert mv_state.error_cov_inv.shape == (k, k)

            if binary:
                assert mv_state.z is not None
                assert jnp.all(jnp.isfinite(mv_state.z))
                assert mv_state.z.shape == (k, y.shape[1])


def copy_arrays(x: PyTree) -> PyTree:
    """Make a copy of the arrays in `x`, intended for buffer donation."""
    return tree.map(lambda x: jnp.array(x) if isinstance(x, jnp.ndarray) else x, x)
