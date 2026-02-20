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
from math import prod
from typing import Literal, NamedTuple

import jax
import pytest
from beartype import beartype
from jax import debug_key_reuse, make_mesh, random, tree, vmap
from jax import numpy as jnp
from jax.sharding import AxisType, Mesh, PartitionSpec, SingleDeviceSharding
from jax.tree_util import KeyPath, keystr
from jaxtyping import Array, Bool, Int32, Key, PyTree, UInt8, jaxtyped
from numpy.testing import assert_array_equal
from pytest_subtests import SubTests
from scipy import stats

from bartz import profile_mode
from bartz.jaxext import get_device_count, minimal_unsigned_dtype, split
from bartz.mcmcstep import State, init, step
from bartz.mcmcstep._moves import (
    ancestor_variables,
    randint_exclude,
    randint_masked,
    split_range,
)
from bartz.mcmcstep._state import chain_vmap_axes, data_vmap_axes
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


class TestMultichain:
    """Basic tests of the multichain functionality."""

    n = 100

    @pytest.fixture
    def init_kwargs(self, keys: split) -> dict:
        """Return arguments for `init`."""
        p = 10
        k = 2
        d = 6
        numcut = 10
        num_trees = 5
        return dict(
            X=random.randint(keys.pop(), (p, self.n), 0, numcut + 1, jnp.uint32),
            y=random.normal(keys.pop(), (k, self.n)),
            offset=random.normal(keys.pop(), (k,)),
            max_split=jnp.full(p, numcut + 1, jnp.uint32),
            num_trees=num_trees,
            p_nonterminal=jnp.full(d - 1, 0.9),
            leaf_prior_cov_inv=jnp.eye(k) * num_trees,
            error_cov_df=2.0,  # keep this a weak type
            error_cov_scale=2 * jnp.eye(k),
        )

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
            typechecking_step = jaxtyped(step, typechecker=beartype)
            with debug_key_reuse(num_chains != 0):
                # key reuse checks trigger with empty key array apparently
                new_state = typechecking_step(keys.pop(), state)
            assert new_state.forest.num_chains() == num_chains
            check_strong_types(new_state)
            check_sharding(new_state, state.config.mesh)

    @pytest.mark.parametrize('profile', [False, True])
    def test_multichain_equiv_stack(
        self, init_kwargs: dict, keys: split, profile: bool
    ) -> None:
        """Check that stacking multiple chains is equivalent to a multichain trace."""
        num_chains = 4
        num_iters = 10

        # create initial states
        mc_state = init(**init_kwargs, num_chains=num_chains)
        sc_states = [
            init(
                **init_kwargs,
                num_chains=None,
                resid_num_batches=mc_state.config.resid_num_batches,
                count_num_batches=mc_state.config.count_num_batches,
                prec_num_batches=mc_state.config.prec_num_batches,
            )
            for _ in range(num_chains)
        ]

        # run a few mcmc steps with the same random keys
        with profile_mode(profile):
            for _ in range(num_iters):
                mc_key = keys.pop()
                sc_keys = random.split(random.clone(mc_key), num_chains)

                mc_state = step(mc_key, mc_state)
                sc_states = [
                    step(key, state)
                    for key, state in zip(sc_keys, sc_states, strict=True)
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
                '.y',
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
                '.y',
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
