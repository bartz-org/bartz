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

import math
from collections.abc import Callable, Sequence
from dataclasses import fields, replace
from functools import partial, wraps
from typing import Any, Literal, NamedTuple

import jax
import numpy
import pytest
from beartype import beartype
from jax import (
    debug_key_reuse,
    device_put,
    jit,
    lax,
    make_mesh,
    random,
    shard_map,
    tree,
    vmap,
)
from jax import numpy as jnp
from jax.nn import logsumexp, softmax
from jax.ops import segment_sum
from jax.sharding import (
    AxisType,
    Mesh,
    NamedSharding,
    PartitionSpec,
    SingleDeviceSharding,
)
from jax.tree_util import KeyPath, keystr
from jax.typing import DTypeLike
from jaxtyping import (
    Array,
    Bool,
    Float,
    Float32,
    Int32,
    Integer,
    Key,
    PyTree,
    Shaped,
    UInt,
    UInt8,
    UInt32,
    jaxtyped,
)
from pytest import FixtureRequest  # noqa: PT013
from pytest_subtests import SubTests
from scipy import special, stats
from scipy.stats import chi2, ks_1samp, ks_2samp

from bartz._jaxext import (
    Module,
    field,
    float32_matmuls,
    get_default_device,
    get_default_devices,
    get_device_count,
    minimal_unsigned_dtype,
    split,
)
from bartz._typing import kwdict
from bartz.debug import ChainDataState, sample_data, sample_prior
from bartz.debug._prior import sample_s
from bartz.grove import (
    evaluate_forest,
    is_actual_leaf,
    tree_actual_depth,
    tree_depths,
    var_histogram,
)
from bartz.mcmcstep import (
    AutoBatchedReduction,
    AutoOneHotOrPopcountReduction,
    AutoOneHotReduction,
    BatchedReduction,
    DiagWishart,
    OneHotReduction,
    PopcountReduction,
    ReductionConfig,
    State,
    Wishart,
    init,
    make_p_nonterminal,
    step,
)
from bartz.mcmcstep._axes import (
    chain_to_axis,
    chain_vmap_axes,
    data_vmap_axes,
    trace_sample_axes,
)
from bartz.mcmcstep._moves import (
    ancestor_variables,
    fully_used_variables,
    randint_exclude,
    randint_masked,
    split_range,
)
from bartz.mcmcstep._rebuild import rebuild_derived
from bartz.mcmcstep._reduction import _gpu_sm_count
from bartz.mcmcstep._state import (
    Forest,
    Sparsity,
    StepConfig,
    chol_with_gersh,
    inv_via_chol_with_gersh,
    outcome_target,
    search_divisor,
)
from bartz.mcmcstep._step import (
    S,
    _compute_likelihood_ratio_mv,
    _compute_likelihood_ratio_uv,
    _log_p_v,
    _precompute_leaf_terms_mv,
    _precompute_leaf_terms_uv,
    _precompute_likelihood_terms_mv,
    _precompute_likelihood_terms_uv,
    _sample_wishart_bartlett,
    _step_error_cov_inv_diag,
    _step_error_cov_inv_mv,
    apply_moves_to_leaf_indices,
    leaf_scatter,
    log_blocked_mass_tree,
    sample_s_augmentation,
    sample_wishart_posterior,
    step_error_cov_inv,
    step_leaf_prior_cov_inv,
    step_s,
    step_theta,
    step_trees,
    step_z,
    stirling_remainder,
)
from bartz.testing import gen_data
from tests.util import (
    assert_allclose,
    assert_array_equal,
    assert_close_matrices,
    assert_different_matrices,
    condf,
    manual_tree,
    nnone,
    skip_empty_shard_map_bug,
)

# Forest size shared across the module. jax recompiles whenever an array shape
# changes, so every test that has no reason to need a specific number of trees
# uses this, letting jax reuse compiled code.
NUM_TREES = 5


class VarTreeData(NamedTuple):
    """Fixture data pairing a variable tree with its max-split array."""

    var_tree: UInt8[Array, ' nodes']
    max_split: UInt8[Array, ' p']


class SplitRangeData(NamedTuple):
    """Fixture data pairing variable/split trees with a max-split array."""

    var_tree: UInt8[Array, ' nodes']
    split_tree: UInt8[Array, ' nodes']
    max_split: UInt8[Array, ' p']


def _minimal_step_config() -> StepConfig:
    """Single-device `StepConfig` with all reduction settings disabled."""
    return StepConfig(
        steps_done=jnp.int32(0),
        sparse_on_at=None,
        resid_reduction_config=BatchedReduction(num_batches=None),
        count_reduction_config=BatchedReduction(num_batches=None),
        prec_reduction_config=BatchedReduction(num_batches=None),
        prec_count_num_trees=None,
        sequential_unroll=1,
        augment=False,
        mesh=None,
    )


class _EmptyForest(Forest):
    """Placeholder `Forest` for error-covariance sampler tests, which never read it.

    Its no-arg `__init__` bypasses `Forest`'s type-checked initializer and leaves
    every field as `None`, so the instance satisfies `State`'s `forest: Forest`
    type while contributing no pytree leaves and no self-consistent arrays.
    """

    def __init__(self) -> None:
        for f in fields(Forest):
            object.__setattr__(self, f.name, None)


def _bare_state_kw(
    kshape: tuple[int, ...], n_non_missing: int | Integer[Array, '*k'], **kw: Any
) -> dict:
    """`State` fields with unit scales, no error scales, and no binary components.

    The error-covariance samplers read only these, `resid`, `inv_sdev_scale`,
    `inv_sdev_unit`, and `error_cov_inv`; `kw` adds or overrides fields.
    """
    base = dict(
        _chain_anchor=jnp.zeros(()),
        binary_indices=None,
        z=None,
        prec_scale=None,
        inv_sdev_scale=None,
        inv_sdev_unit=jnp.ones(kshape),
        resid_unit=jnp.ones(kshape),  # unit scale: resid is in data units
        resid_eff_scale=jnp.ones(kshape),
        resid_inexact_integral=jnp.zeros(kshape),
        error_scale=None,
        n_non_missing=jnp.asarray(n_non_missing),
        sum_diag_prec_scale=jnp.asarray(n_non_missing, jnp.float32),
        forest=_EmptyForest(),
        config=_minimal_step_config(),
    )
    return dict(base, **kw)


class _HasChainsBase(Module):
    """Base for test Modules that declares `has_chains=True`."""

    @property
    def has_chains(self) -> bool:
        return True


class _MarkerLeaf(_HasChainsBase):
    """Module used by `TestVmapAxesMarkers` covering all marker combinations."""

    chained: Shaped[Array, '...'] = field(
        chains=0, default_factory=lambda: jnp.zeros((2, 3))
    )
    datad: Shaped[Array, '...'] = field(
        data=-1, default_factory=lambda: jnp.zeros((2, 3))
    )
    both: Shaped[Array, '...'] = field(
        chains=0, data=-1, default_factory=lambda: jnp.zeros((2, 3))
    )
    plain: Shaped[Array, '...'] = field(default_factory=lambda: jnp.zeros((2, 3)))
    static_thing: int = field(static=True, default=42)


class TestVmapAxesMarkers:
    """Test the `field` / `chain_vmap_axes` machinery."""

    @pytest.fixture
    def x(self) -> _MarkerLeaf:
        """Build a module instance exercising every marker combination."""
        return _MarkerLeaf()

    @pytest.mark.parametrize('inner_has_chains', [False, True])
    def test_nested_has_chains(self, inner_has_chains: bool) -> None:
        """A nested Module's `has_chains` overrides the outer one in its subtree."""

        class _Inner(Module):
            flag: bool = field(static=True)
            arr: Shaped[Array, '...'] = field(
                chains=0, default_factory=lambda: jnp.zeros((2, 3))
            )

            @property
            def has_chains(self) -> bool:
                return self.flag

        class _Outer(_HasChainsBase):
            inner: _Inner
            arr: Shaped[Array, '...'] = field(
                chains=0, default_factory=lambda: jnp.zeros((2, 3))
            )

        axes = chain_vmap_axes(_Outer(inner=_Inner(flag=inner_has_chains)))
        assert axes.arr == 0
        assert axes.inner.arr == (0 if inner_has_chains else None)

    def test_default_none(self, x: _MarkerLeaf) -> None:
        """An unmarked field becomes None."""
        assert chain_vmap_axes(x).plain is None

    def test_chain_marker_value(self, x: _MarkerLeaf) -> None:
        """A chains-marked field reports its integer axis under chains."""
        assert chain_vmap_axes(x).chained == 0

    def test_data_marker_no_chain(self, x: _MarkerLeaf) -> None:
        """A data-only marked field reports None under chains."""
        assert chain_vmap_axes(x).datad is None

    def test_both_markers(self, x: _MarkerLeaf) -> None:
        """A field marked under both keys reports its chain axis under chains."""
        assert chain_vmap_axes(x).both == 0

    def test_static_passthrough(self, x: _MarkerLeaf) -> None:
        """Static fields keep their original (non-axis) value."""
        assert chain_vmap_axes(x).static_thing == 42

    def test_normalization(self) -> None:
        """Negative chain markers are normalized per-leaf against the leaf's ndim."""

        class _NonCanonicalMarkers(_HasChainsBase):
            """Module that uses non-default integer axis indices."""

            a: Shaped[Array, '...'] = field(chains=2)
            b: Shaped[Array, '...'] = field(data=-1)
            c: Shaped[Array, '...'] = field(chains=-2, data=0)

        arr = jnp.zeros((2, 3, 4))
        x = _NonCanonicalMarkers(a=arr, b=arr, c=arr)
        cax = chain_vmap_axes(x)
        # 3-D leaves: chains=2 -> 2, chains=-2 -> 1
        assert cax.a == 2
        assert cax.b is None
        assert cax.c == 1

    def test_negative_marker_normalization(self) -> None:
        """A non-trivial negative chain marker on a 3-D leaf normalizes correctly."""

        class _NegativeChainMarker(_HasChainsBase):
            arr: Shaped[Array, '...'] = field(chains=-2)

        x = _NegativeChainMarker(arr=jnp.zeros((2, 3, 4)))
        # chains=-2 on a 3-D leaf -> normalized to 1
        assert chain_vmap_axes(x).arr == 1

    def test_marker_out_of_bounds_raises(self) -> None:
        """A chain marker whose absolute value exceeds the leaf ndim raises an axis error."""

        class _ChainScalar(_HasChainsBase):
            scalar: Shaped[Array, '...'] = field(chains=0)

        # 0-D leaf can't have axis 0; strict normalization rejects it
        x = _ChainScalar(scalar=jnp.zeros(()))
        with pytest.raises(numpy.exceptions.AxisError):
            chain_vmap_axes(x)

    def test_no_chains_short_circuit(self) -> None:
        """When `has_chains` reports False, marked leaves get None even if they'd be valid."""

        class _NoChains(_HasChainsBase):
            arr: Shaped[Array, '...'] = field(
                chains=0, default_factory=lambda: jnp.zeros((2, 3))
            )

            @property
            def has_chains(self) -> bool:
                return False

        x = _NoChains()
        # `has_chains=False` short-circuits the marker normalization
        assert chain_vmap_axes(x).arr is None

    def test_nested_module(self) -> None:
        """Recursion descends into Module-valued fields."""

        class _InnerMarker(_HasChainsBase):
            chain_arr: Shaped[Array, '...'] = field(chains=0)
            data_arr: Shaped[Array, '...'] = field(data=-1)

        class _OuterMarker(_HasChainsBase):
            inner: _InnerMarker
            outer_chain: Shaped[Array, '...'] = field(chains=0)

        x = _OuterMarker(
            inner=_InnerMarker(chain_arr=jnp.zeros(2), data_arr=jnp.zeros(2)),
            outer_chain=jnp.zeros(2),
        )
        cax = chain_vmap_axes(x)
        assert cax.outer_chain == 0
        assert cax.inner.chain_arr == 0
        assert cax.inner.data_arr is None

    def test_subtree_broadcast(self) -> None:
        """A marked pytree-valued field gets the marker on every leaf."""

        class _ContainerMarker(_HasChainsBase):
            tup: tuple = field(chains=0)
            dct: dict = field(data=-1)

        x = _ContainerMarker(
            tup=(jnp.zeros(2), jnp.zeros(2)), dct={'p': jnp.zeros(2), 'q': jnp.zeros(2)}
        )
        cax = chain_vmap_axes(x)
        assert cax.tup == (0, 0)
        assert cax.dct == {'p': None, 'q': None}

    def test_none_valued_field(self) -> None:
        """A marked field holding None yields None (empty pytree)."""

        class _OptionalMarker(_HasChainsBase):
            maybe: None | Shaped[Array, '...'] = field(chains=0)

        x = _OptionalMarker(maybe=None)
        assert chain_vmap_axes(x).maybe is None

    def test_bool_rejected(self) -> None:
        """field() rejects bool to avoid the int-subclass footgun."""
        with pytest.raises(AssertionError):
            field(chains=True)
        with pytest.raises(AssertionError):
            field(data=False)
        with pytest.raises(AssertionError):
            field(chains=True, data=-1)
        with pytest.raises(AssertionError):
            field(samples=True)

    def test_pytree_container_of_modules(self) -> None:
        """A list of Modules at the top level is walked per-element."""
        x = [_MarkerLeaf(), _MarkerLeaf()]
        cax = chain_vmap_axes(x)
        assert isinstance(cax, list)
        assert len(cax) == 2
        for el in cax:
            assert el.chained == 0
            assert el.both == 0
            assert el.plain is None
            assert el.datad is None


CoreAxisView = tuple[str, Callable[[PyTree], PyTree]]


class TestCoreAxisMarkers:
    """Test marker resolution under the chain-less core convention.

    Both `data_vmap_axes` and `trace_sample_axes` resolve a marker declared
    in the chain-less "core" layout; the chain axis, if any, is treated as
    inserted afterward. These tests are parametrized over the ``data`` and
    ``samples`` markers because their resolution semantics are identical
    modulo the metadata key they read.
    """

    @pytest.fixture(
        params=[
            pytest.param(('data', data_vmap_axes), id='data'),
            pytest.param(('samples', trace_sample_axes), id='samples'),
        ]
    )
    def axis_view(self, request: FixtureRequest) -> CoreAxisView:
        """Return a (marker name, axis-resolution function) pair."""
        return request.param

    def test_no_chains_axis_zero(self, axis_view: CoreAxisView) -> None:
        """Without a chain axis, marker=0 resolves to 0."""
        kind, fn = axis_view

        class _M(Module):
            arr: Shaped[Array, '...'] = field(
                **{kind: 0}, default_factory=lambda: jnp.zeros((5, 3))
            )

            @property
            def has_chains(self) -> bool:
                return False

        assert fn(_M()).arr == 0

    def test_chains_zero_axis_zero(self, axis_view: CoreAxisView) -> None:
        """``chains=0`` shifts marker=0 to trace position 1."""
        kind, fn = axis_view

        class _M(_HasChainsBase):
            arr: Shaped[Array, '...'] = field(
                chains=0, **{kind: 0}, default_factory=lambda: jnp.zeros((4, 5, 3))
            )

        assert fn(_M()).arr == 1

    def test_chains_after_axis_no_shift(self, axis_view: CoreAxisView) -> None:
        """A chain axis past the marker leaves the marker unchanged."""
        kind, fn = axis_view

        class _M(_HasChainsBase):
            arr: Shaped[Array, '...'] = field(
                chains=2, **{kind: 0}, default_factory=lambda: jnp.zeros((5, 3, 4))
            )

        # 3-D leaf, chains=2 -> 2; marker=0 in core (ndim-1=2).
        # chain_axis (2) > axis (0) -> stays at 0.
        assert fn(_M()).arr == 0

    def test_axis_at_nonzero_position(self, axis_view: CoreAxisView) -> None:
        """A non-zero marker in core layout is preserved (shifted by chain if needed)."""
        kind, fn = axis_view

        class _M(_HasChainsBase):
            arr: Shaped[Array, '...'] = field(
                chains=0, **{kind: 2}, default_factory=lambda: jnp.zeros((4, 5, 6, 7))
            )

        # 4-D leaf with chain at 0 -> core ndim 3, marker=2 -> 2.
        # chain_axis (0) <= axis (2) -> shifted to 3.
        assert fn(_M()).arr == 3

    def test_negative_marker(self, axis_view: CoreAxisView) -> None:
        """Negative markers are normalized against the core ndim."""
        kind, fn = axis_view

        class _M(_HasChainsBase):
            arr: Shaped[Array, '...'] = field(
                chains=0, **{kind: -1}, default_factory=lambda: jnp.zeros((4, 5, 6))
            )

        # 3-D leaf with chain at 0 -> core ndim 2, marker=-1 -> 1.
        # chain_axis (0) <= axis (1) -> shifted to 2.
        assert fn(_M()).arr == 2

    def test_no_marker(self, axis_view: CoreAxisView) -> None:
        """A field without the marker resolves to None."""
        _, fn = axis_view

        class _M(_HasChainsBase):
            arr: Shaped[Array, '...'] = field(
                chains=0, default_factory=lambda: jnp.zeros((4, 5))
            )

        assert fn(_M()).arr is None

    def test_has_chains_false_marker_still_resolved(
        self, axis_view: CoreAxisView
    ) -> None:
        """``has_chains=False`` does not affect marker resolution."""
        kind, fn = axis_view

        class _M(Module):
            arr: Shaped[Array, '...'] = field(
                chains=0, **{kind: 0}, default_factory=lambda: jnp.zeros((5, 3))
            )

            @property
            def has_chains(self) -> bool:
                return False

        # `has_chains=False` strips the chain offset; marker stays at core 0.
        assert fn(_M()).arr == 0

    def test_optional_field_is_none(self, axis_view: CoreAxisView) -> None:
        """A marked field holding ``None`` yields ``None``."""
        kind, fn = axis_view

        class _M(_HasChainsBase):
            maybe: None | Shaped[Array, '...'] = field(chains=0, **{kind: 0})

        assert fn(_M(maybe=None)).maybe is None

    def test_mixed_fields(self, axis_view: CoreAxisView) -> None:
        """A module with both marked and unmarked fields handles each correctly."""
        kind, fn = axis_view

        class _M(_HasChainsBase):
            marked: Shaped[Array, '...'] = field(
                chains=0, **{kind: 0}, default_factory=lambda: jnp.zeros((4, 5, 3))
            )
            unmarked: Shaped[Array, '...'] = field(
                default_factory=lambda: jnp.zeros((4, 3))
            )
            no_chain: Shaped[Array, '...'] = field(
                **{kind: 0}, default_factory=lambda: jnp.zeros((5, 3))
            )

        out = fn(_M())
        assert out.marked == 1
        assert out.unmarked is None
        assert out.no_chain == 0

    def test_no_chain_marker_uses_leaf_ndim(self, axis_view: CoreAxisView) -> None:
        """A field with no chain marker normalizes the marker against the leaf ndim."""
        kind, fn = axis_view

        class _M(_HasChainsBase):
            arr: Shaped[Array, '...'] = field(
                **{kind: -1}, default_factory=lambda: jnp.zeros((2, 3))
            )

        # No chain on this leaf -> core ndim = leaf.ndim = 2, marker=-1 -> 1.
        assert fn(_M()).arr == 1

    def test_marker_out_of_bounds_raises(self, axis_view: CoreAxisView) -> None:
        """A marker out of bounds for the core ndim raises an axis error."""
        kind, fn = axis_view

        class _M(_HasChainsBase):
            arr: Shaped[Array, '...'] = field(
                chains=0, **{kind: 5}, default_factory=lambda: jnp.zeros((4, 5, 3))
            )

        # core ndim is 2; raw=5 is out of bounds.
        with pytest.raises(numpy.exceptions.AxisError):
            fn(_M())

    def test_nested_module(self, axis_view: CoreAxisView) -> None:
        """Recursion descends into Module-valued fields."""
        kind, fn = axis_view

        class _Inner(_HasChainsBase):
            marked: Shaped[Array, '...'] = field(**{kind: -1})

        class _Outer(_HasChainsBase):
            inner: _Inner
            outer_marked: Shaped[Array, '...'] = field(**{kind: 0})

        x = _Outer(inner=_Inner(marked=jnp.zeros(2)), outer_marked=jnp.zeros((3, 4)))
        out = fn(x)
        # 1-D leaf: marker=-1 -> 0
        assert out.inner.marked == 0
        # 2-D leaf: marker=0 -> 0
        assert out.outer_marked == 0

    def test_subtree_broadcast(self, axis_view: CoreAxisView) -> None:
        """A marked pytree-valued field gets the marker on every leaf."""
        kind, fn = axis_view

        class _Container(_HasChainsBase):
            dct: dict = field(**{kind: -1})

        x = _Container(dct={'p': jnp.zeros(2), 'q': jnp.zeros(2)})
        # 1-D leaves: marker=-1 -> 0
        assert fn(x).dct == {'p': 0, 'q': 0}

    def test_pytree_container_of_modules(self, axis_view: CoreAxisView) -> None:
        """A list of Modules at the top level is walked per-element."""
        kind, fn = axis_view

        class _M(_HasChainsBase):
            arr: Shaped[Array, '...'] = field(
                **{kind: -1}, default_factory=lambda: jnp.zeros((2, 3))
            )

        out = fn([_M(), _M()])
        assert isinstance(out, list)
        assert len(out) == 2
        # 2-D leaves, no chain on this leaf -> core ndim 2, marker=-1 -> 1
        for el in out:
            assert el.arr == 1


class TestSearchDivisor:
    """Test `search_divisor`."""

    def test_target_already_divides(self) -> None:
        """If `target_divisor` divides `dividend`, return it unchanged via the fast path."""
        assert search_divisor(5, 100, 1, 100) == 5
        # target_divisor outside [low, up] is still returned when it divides
        assert search_divisor(10, 100, 50, 80) == 10
        assert search_divisor(1, 10, 5, 5) == 1

    def test_no_divisors_in_range_returns_target(self) -> None:
        """Fall back to `target_divisor` when no value in `[low, up]` divides `dividend`."""
        # 11 is prime; range [2, 10] contains no divisors of 11
        assert search_divisor(3, 11, 2, 10) == 3
        # 17 is prime; range [2, 16] contains no divisors of 17
        assert search_divisor(7, 17, 2, 16) == 7

    def test_finds_closest_divisor(self) -> None:
        """Return the divisor in `[low, up]` closest to `target_divisor`."""
        # divisors of 24: 1, 2, 3, 4, 6, 8, 12, 24
        # target=5 -> 4 and 6 tie at distance 1; argmin tie-break picks 4
        assert search_divisor(5, 24, 1, 24) == 4
        # target=10 -> 8 and 12 tie at distance 2; tie-break picks 8
        assert search_divisor(10, 24, 1, 24) == 8
        # target=20 -> 24 closer than 12
        assert search_divisor(20, 24, 1, 24) == 24

    def test_range_restricts_candidates(self) -> None:
        """Only divisors within `[low, up]` are considered."""
        # divisors of 24 in [5, 7] -> only 6
        assert search_divisor(10, 24, 5, 7) == 6
        # divisors of 24 in [7, 11] -> only 8
        assert search_divisor(10, 24, 7, 11) == 8

    def test_single_point_range(self) -> None:
        """A `low == up` range either yields that value or the fallback."""
        # 6 divides 24
        assert search_divisor(10, 24, 6, 6) == 6
        # 5 does not divide 24; no other candidates -> fallback to target
        assert search_divisor(10, 24, 5, 5) == 10

    def test_target_equals_one(self) -> None:
        """`target_divisor == 1` always divides any dividend."""
        assert search_divisor(1, 7, 2, 7) == 1
        assert search_divisor(1, 100, 1, 100) == 1

    def test_target_equals_dividend(self) -> None:
        """`target_divisor == dividend` divides exactly."""
        assert search_divisor(7, 7, 1, 7) == 7

    def test_ties_pick_lower(self) -> None:
        """Equidistant divisors are broken by picking the lower one (argmin convention)."""
        # divisors of 12 in [1, 12]: 1, 2, 3, 4, 6, 12
        # target=5 -> 4 and 6 tie at distance 1; tie-break picks 4
        assert search_divisor(5, 12, 1, 12) == 4

    def test_return_type_is_python_int(self) -> None:
        """Returned values are Python ints, not numpy scalars."""
        assert isinstance(search_divisor(5, 100, 1, 100), int)
        assert isinstance(search_divisor(5, 24, 1, 24), int)
        assert isinstance(search_divisor(3, 11, 2, 10), int)

    @pytest.mark.parametrize(
        ('target', 'dividend', 'low', 'up'),
        [
            (0, 10, 1, 10),  # target_divisor < 1
            (1, 10, 0, 10),  # low < 1
            (1, 10, 5, 4),  # low > up
            (1, 10, 1, 11),  # up > dividend
        ],
    )
    def test_assertion_failures(
        self, target: int, dividend: int, low: int, up: int
    ) -> None:
        """Preconditions on the arguments are enforced via assertions."""
        with pytest.raises(AssertionError):
            search_divisor(target, dividend, low, up)


def reduce_reference(
    values: Float[Array, '*batch_shape n'] | int,
    indices: UInt[Array, ' n'],
    /,
    *,
    size: int,
    subset_start: Integer[Array, ''] | None = None,
    subset_length: int | None = None,
    dtype: DTypeLike,
    data_sharded: bool,
) -> Shaped[Array, '*batch_shape {(size,subset_length)[bool(subset_length)]}']:
    """External baseline mirroring `_reduce` via `jax.ops.segment_sum`."""
    values = jnp.asarray(values)
    # `segment_sum` reduces the leading axis, `_reduce` the last one; a
    # scalar value is the count case, weighting each datapoint equally
    if values.ndim == 0:
        data = jnp.broadcast_to(values.astype(dtype), indices.shape)
    else:
        data = jnp.moveaxis(values, -1, 0).astype(dtype)
    out = segment_sum(data, indices, num_segments=size)
    if subset_length is not None:
        assert subset_start is not None  # set together with subset_length
        # select the range's bins from the full reduction; 'fill' makes
        # out-of-domain bins read as zero instead of clamping
        bins = subset_start.astype(jnp.uint32) + jnp.arange(
            subset_length, dtype=jnp.uint32
        )
        out = out.at[bins].get(mode='fill', fill_value=0)
    out = jnp.moveaxis(out, 0, -1)
    if data_sharded:
        out = lax.psum(out, 'data')
    return out


class TestReduction:
    """Check every `ReductionConfig` matches an unbatched segment-sum baseline."""

    @pytest.fixture
    def configs(self) -> tuple[ReductionConfig, ...]:
        """Configs covering every subclass and setting."""
        return (
            # BatchedReduction: unbatched and explicit batch counts (a divisor of
            # `n` and a non-divisor, which leaves an uneven final batch), each batch
            # axis layout, strided vs contiguous batch assignment; plus the
            # per-platform automatic count of AutoBatchedReduction
            BatchedReduction(num_batches=None),
            AutoBatchedReduction(),
            BatchedReduction(num_batches=4),
            BatchedReduction(num_batches=7),
            BatchedReduction(num_batches=4, batches_inner=False),
            BatchedReduction(num_batches=4, contiguous=True),
            BatchedReduction(num_batches=7, contiguous=True),
            BatchedReduction(num_batches=7, batches_inner=False, contiguous=True),
            # OneHotReduction: every contraction method in both memory layouts
            OneHotReduction(method='matmul', n_inner=True),
            OneHotReduction(method='matmul', n_inner=False),
            OneHotReduction(method='multiply', n_inner=True),
            OneHotReduction(method='multiply', n_inner=False),
            OneHotReduction(method='scatter_set', n_inner=True),
            OneHotReduction(method='scatter_set', n_inner=False),
            OneHotReduction(method='chunked'),
            # AutoOneHotReduction: per-site, per-platform method and layout
            AutoOneHotReduction(),
        )

    def test_matches_reference(
        self, configs: tuple[ReductionConfig, ...], keys: split, subtests: SubTests
    ) -> None:
        """Every config matches the reference on a battery of invocations.

        The cases cover scalar count weights (exact integer match), float
        values with and without batch dimensions, float16 values whose per-bin
        sums overflow float16 (catching accumulation in the values' dtype
        rather than in the requested `dtype`), composition with an external
        `vmap` that batches only the indices (the layout `step` uses to
        count/sum over many trees at once), reductions over a contiguous bin
        range (with a non-power-of-2 length and running past the index domain),
        and a data axis sharded with `shard_map`.
        """
        n, size = 512, 8
        indices = random.randint(keys.pop(), (n,), 0, size).astype(jnp.uint32)
        tree_indices = random.randint(keys.pop(), (NUM_TREES, n), 0, size).astype(
            jnp.uint32
        )

        # float values: univariate, and with the 1d/2d batch shapes of the
        # multivariate leaf-sum and precision paths
        vector = random.normal(keys.pop(), (n,))
        batched = random.normal(keys.pop(), (3, n))
        matrix = random.normal(keys.pop(), (2, 2, n))

        # float16 values: each per-bin sum overflows the float16 range, so a
        # reduction accumulating in the values' dtype returns inf
        overflowing = jnp.full(n, 4096.0, jnp.float16)

        count_kw = dict(size=size, dtype=jnp.uint32, data_sharded=False)
        float_kw = dict(size=size, dtype=jnp.float32, data_sharded=False)

        # bin range: starts mid-domain, has a non-power-of-2 length, and runs one
        # past `size`, so its last bin is out of the index domain and its sum
        # must come out zero
        range_kw = dict(float_kw, subset_start=jnp.uint32(6), subset_length=3)

        # comparison function for integer calculations
        exact = assert_array_equal

        # comparison function for float calculations; on gpu the matmul method
        # contracts via tf32 (~1e-3 relative error); cpu keeps the tight
        # tolerance, so accuracy is still fully checked there
        rtol = 1e-3 if indices.platform() != 'cpu' else 1e-5
        close = partial(assert_close_matrices, rtol=rtol, atol=1e-6, reduce_rank=True)

        # each case invokes a reduce function `f` (a config's `_reduce` or the
        # reference) in one pattern, paired with the appropriate comparison
        cases = {
            'count': (lambda f: f(1, indices, **count_kw), exact),
            'float': (lambda f: f(vector, indices, **float_kw), close),
            'float batched': (lambda f: f(batched, indices, **float_kw), close),
            'float matrix': (lambda f: f(matrix, indices, **float_kw), close),
            # a length that does not divide into chunks of datapoints
            'count odd n': (lambda f: f(1, indices[:509], **count_kw), exact),
            'float odd n': (
                lambda f: f(vector[:509], indices[:509], **float_kw),
                close,
            ),
            'float16': (lambda f: f(overflowing, indices, **float_kw), close),
            'vmap count': (
                lambda f: vmap(partial(f, 1, **count_kw))(tree_indices),
                exact,
            ),
            'vmap float': (
                lambda f: vmap(partial(f, vector, **float_kw))(tree_indices),
                close,
            ),
            'range count': (
                lambda f: f(1, indices, **dict(range_kw, dtype=jnp.uint32)),
                exact,
            ),
            'range float': (lambda f: f(vector, indices, **range_kw), close),
            'range float matrix': (lambda f: f(matrix, indices, **range_kw), close),
            'vmap range float': (
                lambda f: vmap(partial(f, vector, **range_kw))(tree_indices),
                close,
            ),
        }

        # `data_sharded=True` runs under `shard_map` and sums the per-shard
        # reductions; on a single device `psum` is the identity, so a missing
        # sum could not be caught and the sharded cases are not worth running
        num_shards = min(4, get_device_count())
        if num_shards >= 2:
            mesh = make_mesh(
                (num_shards,), ('data',), devices=get_default_devices()[:num_shards]
            )
            data, replicated = PartitionSpec('data'), PartitionSpec()
            sharded_indices = device_put(indices, NamedSharding(mesh, data))
            sharded_vector = device_put(vector, NamedSharding(mesh, data))
            cases['sharded count'] = (
                lambda f: shard_map(
                    partial(f, 1, **dict(count_kw, data_sharded=True)),
                    mesh=mesh,
                    in_specs=(data,),
                    out_specs=replicated,
                )(sharded_indices),
                exact,
            )
            cases['sharded float'] = (
                lambda f: shard_map(
                    partial(f, **dict(float_kw, data_sharded=True)),
                    mesh=mesh,
                    in_specs=(data, data),
                    out_specs=replicated,
                )(sharded_vector, sharded_indices),
                close,
            )
            # the range bounds are closed over, so they are replicated across shards
            cases['sharded range float'] = (
                lambda f: shard_map(
                    partial(f, **dict(range_kw, data_sharded=True)),
                    mesh=mesh,
                    in_specs=(data, data),
                    out_specs=replicated,
                )(sharded_vector, sharded_indices),
                close,
            )

        expected = {name: run(reduce_reference) for name, (run, _) in cases.items()}

        for config in configs:
            for name, (run, compare) in cases.items():
                with subtests.test(config=repr(config), case=name):
                    compare(run(config._reduce), expected[name])

    def test_count_only_matches_reference(
        self, keys: split, subtests: SubTests
    ) -> None:
        """The count-only configs match the reference on counts into a bin range.

        The cases cover every unsigned index width, lengths that leave word
        chunks, words, and indices over from the packing, bin ranges inside,
        past the index domain, and past the lane width, all indices in the
        counted bin, composition with an external `vmap` over trees, and a
        sharded data axis.
        """
        configs = (PopcountReduction(), AutoOneHotOrPopcountReduction())
        size = 8
        count_kw = dict(size=size, dtype=jnp.uint32, data_sharded=False)
        # (start, length): in the domain, two bins, past `size`, past uint8
        ranges = ((2, 1), (3, 2), (7, 2), (300, 1))

        def count(
            indices: UInt[Array, ' n'], start: int, length: int
        ) -> Callable[[Callable[..., Array]], Array]:
            kw = dict(count_kw, subset_start=jnp.uint32(start), subset_length=length)
            return lambda f: f(1, indices, **kw)

        cases = {}
        for dtype in (jnp.uint8, jnp.uint16, jnp.uint32):
            # a multiple of the chunk length in every width, then lengths that
            # leave chunks, words and indices over, and one without any chunk
            for n in (512, 1031, 7):
                indices = random.randint(keys.pop(), (n,), 0, size).astype(dtype)
                for start, length in ranges:
                    name = f'{jnp.dtype(dtype).name} n={n} range={start},{length}'
                    cases[name] = count(indices, start, length)
        cases['all in bin'] = count(jnp.full(1031, 3, jnp.uint8), 3, 1)
        tree_indices = random.randint(keys.pop(), (NUM_TREES, 1031), 0, size).astype(
            jnp.uint8
        )
        range_kw = dict(count_kw, subset_start=jnp.uint32(2), subset_length=2)
        cases['vmap'] = lambda f: vmap(partial(f, 1, **range_kw))(tree_indices)

        num_shards = min(4, get_device_count())
        if num_shards >= 2:
            mesh = make_mesh(
                (num_shards,), ('data',), devices=get_default_devices()[:num_shards]
            )
            data, replicated = PartitionSpec('data'), PartitionSpec()
            sharded = random.randint(keys.pop(), (num_shards * 1031,), 0, size)
            sharded = device_put(sharded.astype(jnp.uint8), NamedSharding(mesh, data))
            cases['sharded'] = lambda f: shard_map(
                partial(f, 1, **dict(range_kw, data_sharded=True)),
                mesh=mesh,
                in_specs=(data,),
                out_specs=replicated,
            )(sharded)

        for name, run in cases.items():
            expected = run(reduce_reference)
            for config in configs:
                with subtests.test(config=repr(config), case=name):
                    assert_array_equal(run(config._reduce), expected)

    @pytest.mark.parametrize('num_bins', [1, 2])
    def test_chunked_does_not_materialize_product(self, num_bins: int) -> None:
        """On cpu, 'chunked' avoids materializing the one-hot product, 'multiply' not.

        This tracks the behavior of xla:cpu that motivates using 'chunked' in
        `AutoOneHotReduction`: if it changes, that choice should be revisited.
        """
        if get_default_device().platform != 'cpu':
            pytest.skip('xla:cpu behavior')
        num_trees = 8
        n = 2**16
        values = jnp.ones(n)
        indices = jnp.zeros((num_trees, n), jnp.uint8)
        kw: dict = dict(
            size=4,
            subset_start=jnp.uint32(1),
            subset_length=num_bins,
            dtype=jnp.float32,
            data_sharded=False,
        )

        def temp_bytes(config: OneHotReduction) -> int:
            reduce = vmap(partial(config._reduce, **kw), in_axes=(None, 0))
            compiled = jit(reduce).lower(values, indices).compile()
            analysis = compiled.memory_analysis()
            assert analysis is not None
            return analysis.temp_size_in_bytes

        # the float32 product of the values with a one-hot over a single bin
        product_bytes = num_trees * n * 4
        assert (
            temp_bytes(OneHotReduction(method='multiply')) >= num_bins * product_bytes
        )
        assert temp_bytes(OneHotReduction(method='chunked')) < product_bytes / 2

    def test_chunked_rejects_outer_layout(self) -> None:
        """The 'chunked' one-hot method supports only the datapoints inner."""
        config = OneHotReduction(method='chunked', n_inner=False)
        with pytest.raises(NotImplementedError, match='n_inner=True'):
            config._reduce(
                1, jnp.zeros(8, jnp.uint8), size=4, dtype=jnp.uint32, data_sharded=False
            )

    def test_count_only_rejects_other_uses(self) -> None:
        """The count-only configs reject values other than 1 and a missing range."""
        indices = jnp.zeros(8, jnp.uint8)
        range_kw: dict = dict(size=4, subset_start=jnp.uint32(0), subset_length=1)
        for config in (PopcountReduction(), AutoOneHotOrPopcountReduction()):
            with pytest.raises(AssertionError):
                config._reduce(
                    jnp.ones(8),
                    indices,
                    dtype=jnp.float32,
                    data_sharded=False,
                    **range_kw,
                )
            with pytest.raises(AssertionError):
                config._reduce(
                    2, indices, dtype=jnp.uint32, data_sharded=False, **range_kw
                )
            with pytest.raises(AssertionError):
                config._reduce(1, indices, size=4, dtype=jnp.uint32, data_sharded=False)

    def test_gpu_sm_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The cuda branch reads the shared SM count and rejects mixed gpus."""

        class FakeDevice(NamedTuple):
            core_count: int

        monkeypatch.setattr(
            'bartz.mcmcstep._reduction.backends', lambda: {'cuda': None}
        )

        monkeypatch.setattr(jax, 'devices', lambda _: [FakeDevice(84), FakeDevice(84)])
        assert _gpu_sm_count() == 84

        monkeypatch.setattr(jax, 'devices', lambda _: [FakeDevice(84), FakeDevice(80)])
        with pytest.raises(ValueError, match='differing SM counts'):
            _gpu_sm_count()


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
        lambda_ = 2 * llr
        pvalue = stats.chi2.sf(lambda_, num_allowed - 1)
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
        assert_array_equal(result, [max_split.size], strict=False)

    def test_child_of_root(self, depth2_tree: VarTreeData) -> None:
        """Check that children of root have one ancestor (the root's variable)."""
        var_tree, max_split = depth2_tree

        # Left child of root (index 2): ancestor is root (var=2)
        result = ancestor_variables(var_tree, max_split, jnp.int32(2))
        assert result.shape == (1,)
        assert_array_equal(result, [2], strict=False)

        # Right child of root (index 3): ancestor is root (var=2)
        result = ancestor_variables(var_tree, max_split, jnp.int32(3))
        assert_array_equal(result, [2], strict=False)

    def test_deep_node(self, depth3_tree: VarTreeData) -> None:
        """Check ancestors for nodes at depth 3."""
        var_tree, max_split = depth3_tree

        # Node 4: parent is 2 (var=2), grandparent is 1 (var=3)
        result = ancestor_variables(var_tree, max_split, jnp.int32(4))
        assert result.shape == (2,)
        assert_array_equal(result, [3, 2], strict=False)

        # Node 5: parent is 2 (var=2), grandparent is 1 (var=3)
        result = ancestor_variables(var_tree, max_split, jnp.int32(5))
        assert_array_equal(result, [3, 2], strict=False)

        # Node 6: parent is 3 (var=1), grandparent is 1 (var=3)
        result = ancestor_variables(var_tree, max_split, jnp.int32(6))
        assert_array_equal(result, [3, 1], strict=False)

        # Node 7: parent is 3 (var=1), grandparent is 1 (var=3)
        result = ancestor_variables(var_tree, max_split, jnp.int32(7))
        assert_array_equal(result, [3, 1], strict=False)

    def test_intermediate_node(self, depth3_tree: VarTreeData) -> None:
        """Check ancestors for an intermediate (non-leaf) node."""
        var_tree, max_split = depth3_tree

        # Node 2: parent is 1 (var=3), one slot filled, one sentinel
        result = ancestor_variables(var_tree, max_split, jnp.int32(2))
        assert_array_equal(result, [max_split.size, 3], strict=False)

        # Node 3: parent is 1 (var=3), one slot filled, one sentinel
        result = ancestor_variables(var_tree, max_split, jnp.int32(3))
        assert_array_equal(result, [max_split.size, 3], strict=False)

    def test_single_variable(self) -> None:
        """Check with only one variable (p=1)."""
        tree = manual_tree(
            [[0.0], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0], [0, 0]], [[4], [3, 5]]
        )
        var_tree = tree.var_tree.astype(jnp.uint8)
        max_split = jnp.ones(1, minimal_unsigned_dtype(10))

        # Node 2: ancestor is root (var=0)
        result = ancestor_variables(var_tree, max_split, jnp.int32(2))
        assert_array_equal(result, [0], strict=False)

        # Root has no ancestors
        result = ancestor_variables(var_tree, max_split, jnp.int32(1))
        assert_array_equal(result, [max_split.size], strict=False)

    def test_type_edge(self, depth3_tree: VarTreeData) -> None:
        """Check that types are handled correctly when using uint8 and uint16 together."""
        var_tree, max_split = depth3_tree
        var_tree = var_tree.astype(jnp.uint8)
        max_split = jnp.full(256, 10, jnp.uint8)
        assert minimal_unsigned_dtype(max_split.size) == jnp.uint16

        # Node 2: parent is 1 (var=3), one slot filled, one sentinel
        result = ancestor_variables(var_tree, max_split, jnp.int32(2))
        assert_array_equal(result, [max_split.size, 3], strict=False)

        # Node 3: parent is 1 (var=3), one slot filled, one sentinel
        result = ancestor_variables(var_tree, max_split, jnp.int32(3))
        assert_array_equal(result, [max_split.size, 3], strict=False)


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

    n = 60  # 3 * 4 * 5, maximize divisibility for sharding tests

    @pytest.fixture(
        params=[
            'uv-binary',
            'uv-continuous',
            'mv-binary',
            'mv-continuous',
            'mv-continuous-vec-scales',
            'mv-mixed',
        ]
    )
    def init_kwargs(self, keys: split, request: pytest.FixtureRequest) -> dict:
        """Return arguments for `init`."""
        kind = request.param
        mv = kind.startswith('mv-')
        binary = kind.endswith('-binary')
        mixed = kind == 'mv-mixed'
        vec_scales = kind == 'mv-continuous-vec-scales'

        p = 10
        k = 2
        d = 6
        numcut = 10
        X = random.randint(keys.pop(), (p, self.n), 0, numcut + 1, jnp.uint32)
        max_split = jnp.full(p, numcut + 1, jnp.uint32)

        if mixed:
            y = jnp.zeros((k, self.n), jnp.float32)
            y = y.at[0].set(
                random.bernoulli(keys.pop(), 0.5, (self.n,)).astype(jnp.float32)
            )
            y = y.at[1].set(random.normal(keys.pop(), (self.n,)))
            offset = random.normal(keys.pop(), (k,))
            leaf_prior_cov_inv = Wishart(None, None, jnp.eye(k) * NUM_TREES)
        else:
            if mv:
                y_shape = (k, self.n)
                offset = random.normal(keys.pop(), (k,))
                leaf_prior_cov_inv = Wishart(None, None, jnp.eye(k) * NUM_TREES)
            else:
                y_shape = (self.n,)
                offset = random.normal(keys.pop(), ())
                leaf_prior_cov_inv = Wishart(None, None, jnp.float32(NUM_TREES))

            if binary:
                y = random.bernoulli(keys.pop(), 0.5, y_shape).astype(jnp.float32)
            else:
                y = random.normal(keys.pop(), y_shape)

        kw: dict = dict(
            X=X,
            y=y,
            offset=offset,
            max_split=max_split,
            num_trees=NUM_TREES,
            p_nonterminal=jnp.full(d - 1, 0.9),
            leaf_prior_cov_inv=leaf_prior_cov_inv,
        )

        if mixed:
            kw.update(
                outcome_type=['binary', 'continuous'],
                error_cov_inv=DiagWishart(
                    nu=2.0, rate=jnp.diag(jnp.array([0.0, 2.0])), value=jnp.eye(2)
                ),
            )
        elif binary:
            kw.update(outcome_type='binary')
        else:
            kw.update(
                error_cov_inv=Wishart(
                    nu=2.0,
                    rate=2 * jnp.eye(k) if mv else 2.0,
                    value=jnp.eye(k) if mv else 1.0,
                )
            )

        if vec_scales:
            kw.update(
                error_scale=jnp.exp(
                    random.uniform(keys.pop(), (k, self.n), float, -0.5, 0.5)
                )
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
            targets = dict(data=self.n)
            if num_chains is not None:
                targets = dict(targets, chains=num_chains)
            while math.prod(mesh.values()) > get_device_count():
                for key in mesh:
                    if mesh[key] > 1:
                        mesh[key] -= 1
                        while targets[key] % mesh[key] != 0:
                            mesh[key] -= 1
                        break

        with subtests.test('init'):
            typechecking_init = jaxtyped(init, typechecker=beartype)
            state = typechecking_init(**init_kwargs, num_chains=num_chains, mesh=mesh)
            assert state.num_chains() == num_chains
            check_strong_types(state)
            check_sharding(state, state.config.mesh)

        with subtests.test('step'):
            skip_empty_shard_map_bug(num_chains == 0 and mesh is not None)
            with debug_key_reuse(False):
                # key reuse checks trigger with empty key array apparently
                new_state = typechecking_step(keys.pop(), state)
            assert new_state.num_chains() == num_chains
            check_strong_types(new_state)
            check_sharding(new_state, state.config.mesh)
            check_same_structure(state, new_state)

    @pytest.mark.parametrize(
        ('num_chains', 'chains_axis', 'match'),
        [
            (None, 2, 'num_chains is None'),  # 'chains' axis but scalar chains
            (3, 2, 'does not divide'),  # 2 does not divide 3 chains
        ],
    )
    def test_init_rejects_inconsistent_chains_mesh(
        self, init_kwargs: dict, num_chains: int | None, chains_axis: int, match: str
    ) -> None:
        """`init` rejects a 'chains' mesh axis inconsistent with `num_chains`."""
        if get_device_count() < chains_axis:
            pytest.skip(f'Need at least {chains_axis} devices for this mesh.')
        with pytest.raises(ValueError, match=match):
            init(**init_kwargs, num_chains=num_chains, mesh=dict(chains=chains_axis))

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
                resid_reduction_config=mc_state.config.resid_reduction_config,
                count_reduction_config=mc_state.config.count_reduction_config,
                prec_reduction_config=mc_state.config.prec_reduction_config,
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
            mc_x: Shaped[Array, '*shape'] | None,
            *sc_xs: Shaped[Array, '...'] | None,
        ) -> Shaped[Array, '*shape'] | None:
            if chain_axis is None or mc_x is None:
                return mc_x
            else:
                return jnp.stack([nnone(x) for x in sc_xs], axis=chain_axis)

        chain_axes = chain_vmap_axes(mc_state)
        stacked_state = tree.map_with_path(
            stack_leaf, chain_axes, mc_state, *sc_states, is_leaf=lambda x: x is None
        )

        # check the mc state is equal to the stacked state
        # reduced-precision leaves quantize the slightly different float32
        # reductions of the multichain vs stacked single-chain runs, so
        # equivalence holds only to the rounding floor; the leaves and everything
        # derived from them carry this loss
        inexact_rtol = condf(mc_state.forest.leaf_tree, 1e-5, 1e-3)

        def check_equal(
            path: KeyPath, mc: Shaped[Array, '*shape'], stacked: Shaped[Array, '*shape']
        ) -> None:
            str_path = keystr(path)
            exact = jnp.issubdtype(mc.dtype, jnp.integer)
            assert_close_matrices(
                mc,
                stacked,
                err_msg=f'{str_path}: ',
                rtol=0 if exact else inexact_rtol,
                reduce_rank=True,
            )

        tree.map_with_path(check_equal, mc_state, stacked_state)

    def chain_vmap_axes(self, state: State) -> State:
        """Manual reference for `chain_vmap_axes(_: State)`.

        Mirrors the production semantics: when ``state.has_chains`` is False
        every leaf maps to None; otherwise chain-bearing fields normalize to 0
        (their first axis), unmarked fields and `None` leaves give None.
        """
        has_chains = state.has_chains

        def choose_vmap_index(
            path: KeyPath, leaf: Shaped[Array, '...'] | None
        ) -> int | None:
            no_vmap_attrs = (
                '.X',
                '.y',
                '.binary_indices',
                '.prec_scale',
                '.inv_sdev_scale',
                '.error_scale',
                '.error_cov_inv.nu',
                '.error_cov_inv.rate',
                '.forest.max_split',
                '.forest.blocked_vars',
                '.forest.p_nonterminal',
                '.forest.p_propose_grow',
                '.forest.min_points_per_decision_node',
                '.forest.min_points_per_leaf',
                '.forest.leaf_prior_cov_inv.nu',
                '.forest.leaf_prior_cov_inv.rate',
                '.forest.sparsity.a',
                '.forest.sparsity.b',
                '.forest.sparsity.rho',
                '.config.sparse_on_at',
                '.config.steps_done',
            )
            if not has_chains:
                return None
            if keystr(path) in no_vmap_attrs or leaf is None:
                return None
            return 0

        return tree.map_with_path(choose_vmap_index, state)

    def data_vmap_axes(self, state: State) -> State:
        """Manual reference for `data_vmap_axes(_: State)`.

        Each data field is marked with ``data=-1``; the marker is normalized
        per-leaf, so the result is ``leaf.ndim - 1`` (or None if the leaf has
        no axes / no marker).
        """

        def choose_vmap_index(
            path: KeyPath, leaf: Shaped[Array, '...'] | None
        ) -> int | None:
            vmap_attrs = (
                '.X',
                '.y',
                '.z',
                '.resid',
                '.prec_scale',
                '.inv_sdev_scale',
                '.error_scale',
                '.forest.leaf_indices',
            )
            if keystr(path) not in vmap_attrs or leaf is None:
                return None
            if leaf.ndim == 0:
                return None
            return leaf.ndim - 1

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
        devices = get_default_devices()[:3]
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
        _path: KeyPath,
        x: Shaped[Array, '...'] | None,
        chain_axis: int | None,
        data_axis: int | None,
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


def get_normal_spec(x: Shaped[Array, '...']) -> PartitionSpec:
    """Get the partition spec of `x` and apply `normalize_spec`."""
    sharding = x.sharding
    assert isinstance(sharding, NamedSharding)
    mesh = sharding.mesh
    assert isinstance(mesh, Mesh)
    return normalize_spec(sharding.spec, mesh, x.shape)


def normalize_spec(
    spec: PartitionSpec | Sequence[str | None], mesh: Mesh, shape: tuple[int, ...]
) -> PartitionSpec:
    """Put a spec in standard form, i.e., fill with `None` until length `ndim` and put `None` on axes with mesh size 1 or if array size is 0."""
    s = list(spec)
    ndim = len(shape)
    assert len(s) <= ndim
    s.extend([None] * (ndim - len(s)))

    array_size = math.prod(shape)
    for i in range(ndim):
        if s[i] is not None:
            mesh_size = mesh.shape[s[i]]
            if mesh_size == 1 or array_size == 0:
                s[i] = None

    assert len(s) == ndim
    return PartitionSpec(*s)


def check_strong_types(x: PyTree[Array]) -> None:
    """Check all arrays in `x` have strong types."""

    def check_leaf(path: KeyPath, x: Shaped[Array, '...']) -> None:
        assert not x.weak_type, f'{keystr(path)} has weak type'

    tree.map_with_path(check_leaf, x)


def check_same_structure(x: PyTree, y: PyTree) -> None:
    """Check that two PyTrees have the same structure, incl. shape and type of the arrays."""

    def check(
        _path: KeyPath, x: Shaped[Array, '*shape'], y: Shaped[Array, '*shape']
    ) -> None:
        assert x.shape == y.shape
        assert x.dtype == y.dtype
        # WORKAROUND(jax<0.7): empty arrays get their sharding spec canonicalized
        # inconsistently across code paths, so skip the check in that case.
        assert x.sharding.is_equivalent_to(y.sharding, x.ndim) or (
            x.size == 0 and jax.__version_info__ < (0, 7, 0)
        )

    tree.map_with_path(check, x, y)


def test_z_differs_across_data_shards(keys: split) -> None:
    """Check `step_z` produces independent z per data shard.

    With (X, y) replicated across data shards, the only source of variation
    between shards is the random key. Guards the
    `random.fold_in(key, axis_index('data'))` call in `step_z`: without it
    the same key reaches every shard and, since the local data is
    identical, the truncated normal draw would be the same on every shard.
    """
    num_data_shards = min(4, get_device_count())
    if num_data_shards < 2:
        pytest.skip('need at least 2 devices for the data axis')

    n_per_shard = 20
    p = 5
    numcut = 10

    X_one = random.randint(keys.pop(), (p, n_per_shard), 0, numcut + 1, jnp.uint32)
    y_one = random.bernoulli(keys.pop(), 0.5, (n_per_shard,)).astype(jnp.float32)

    X = jnp.tile(X_one, (1, num_data_shards))
    y = jnp.tile(y_one, num_data_shards)
    max_split = jnp.full(p, numcut + 1, jnp.uint32)

    state = init(
        X=X,
        y=y,
        outcome_type='binary',
        offset=jnp.float32(0.0),
        max_split=max_split,
        num_trees=NUM_TREES,
        p_nonterminal=jnp.full(5, 0.9),
        leaf_prior_cov_inv=Wishart(None, None, jnp.float32(NUM_TREES)),
        mesh={'data': num_data_shards},
    )

    new_state = step(keys.pop(), state)

    assert new_state.z is not None
    z_per_shard = new_state.z.reshape(num_data_shards, n_per_shard)
    for i in range(num_data_shards):
        for j in range(i + 1, num_data_shards):
            assert_different_matrices(
                z_per_shard[i],
                z_per_shard[j],
                rtol=0.5,
                atol=0,
                err_msg=f'shards {i} and {j} produced similar z\n',
            )


@pytest.mark.parametrize(
    ('min_points_per_decision_node', 'min_points_per_leaf'),
    [(None, None), (10, None), (10, 5)],
)
def test_affluence_tree_stays_clean(
    keys: split,
    min_points_per_decision_node: int | None,
    min_points_per_leaf: int | None,
) -> None:
    """`affluence_tree` marks only actual growable leaves, with no dirty bits.

    The MCMC keeps `affluence_tree` clean (a `True` bit only on a node that is
    really a leaf), instead of relying on a downstream `is_actual_leaf` mask. A
    bit left on a grown-away or pruned-away node would be a regression.
    """
    p, n, num_steps = 5, 200, 50
    data = gen_data(
        keys.pop(), n=n, p=p, q=0, sigma2_lin=1.0, sigma2_quad=1.0, sigma2_eps=1.0
    ).quantize()
    state = init(
        X=data.x,
        y=data.y,
        offset=0.0,
        max_split=data.max_split,
        num_trees=NUM_TREES,
        p_nonterminal=make_p_nonterminal(6),
        leaf_prior_cov_inv=Wishart(None, None, 1.0),
        error_cov_inv=Wishart(nu=2.0, rate=2.0, value=1.0),
        min_points_per_decision_node=min_points_per_decision_node,
        min_points_per_leaf=min_points_per_leaf,
    )

    Trees = tuple[Bool[Array, 'trees half'], UInt8[Array, 'trees half']]

    @jit
    def run_chain(
        state: State, step_keys: Key[Array, ' steps']
    ) -> tuple[Bool[Array, 'steps trees half'], UInt8[Array, 'steps trees half']]:
        def body(state: State, key: Key[Array, '']) -> tuple[State, Trees]:
            state = step(key, state)
            return state, (state.forest.affluence_tree, state.forest.split_tree)

        _, out = lax.scan(body, state, step_keys)
        return out

    affluence, split_tree = run_chain(state, keys.pop(num_steps))

    # an affluent node must be an actual leaf, in every tree at every step
    is_leaf = vmap(vmap(is_actual_leaf))(split_tree)
    assert_array_equal(affluence & ~is_leaf, jnp.zeros_like(affluence))
    # sanity: the mask is non-trivially populated (else the check is vacuous)
    assert jnp.any(affluence)


@pytest.mark.parametrize(
    'case',
    [
        'plain',
        'min_points',
        'error_scale',
        'chains',
        'binary',
        'multivariate',
        'blocked',
    ],
)
def test_rebuild_derived(keys: split, case: str) -> None:
    """`rebuild_derived` reproduces the caches maintained by `step`."""
    # generate data
    n = 100
    k = 2 if case == 'multivariate' else None
    outcome_type = 'binary' if case == 'binary' else 'continuous'
    data = gen_data(
        keys.pop(),
        n=n,
        p=5,
        k=k,
        q=0,
        lambda_=None if k is None else 0.5,
        sigma2_lin=1.0,
        sigma2_quad=1.0,
        sigma2_eps=1.0,
        outcome_type=outcome_type,
    ).quantize()

    # prepare arguments to `init()`
    eye = jnp.float32(1.0) if k is None else jnp.eye(k)
    kw: dict = dict(
        X=data.x,
        y=data.y,
        outcome_type=outcome_type,
        offset=jnp.zeros(() if k is None else (k,)),
        max_split=data.max_split,
        num_trees=NUM_TREES,
        p_nonterminal=make_p_nonterminal(6),
        leaf_prior_cov_inv=Wishart(None, None, NUM_TREES * eye),
        error_cov_inv=None if case == 'binary' else Wishart(4.0, 4.0 * eye, eye),
    )
    if case == 'min_points':
        kw = dict(kw, min_points_per_decision_node=10, min_points_per_leaf=5)
    elif case == 'error_scale':
        kw = dict(kw, error_scale=jnp.linspace(0.5, 2.0, n))
    elif case == 'chains':
        kw = dict(kw, num_chains=2)
    elif case == 'blocked':
        # a predictor without splits, excluded with `blocked_vars`
        kw = dict(
            kw,
            X=kw['X'].at[0, :].set(0),
            max_split=kw['max_split'].at[0].set(0),
            filter_splitless_vars=1,
        )

    # create initial state
    state = init(**kw)

    # run some mcmc steps
    for key in keys.pop(20):
        state = step(key, state)

    # recalculate derived quantities in the state from scratch
    old_forest = state.forest
    rebuilt = rebuild_derived(copy_arrays(state))
    new_forest = rebuilt.forest

    # compare leaf_indices
    apply = apply_moves_to_leaf_indices
    if state.has_chains:
        apply = vmap(apply)
    assert_array_equal(
        apply(old_forest.leaf_indices, old_forest.to_prune, old_forest.move_node),
        new_forest.leaf_indices,
    )

    # compare affluence tree
    assert_array_equal(old_forest.affluence_tree, new_forest.affluence_tree)

    # compare count and prec tree
    is_leaf = jnp.vectorize(
        partial(is_actual_leaf, add_bottom_level=True), signature='(h)->(t)'
    )(old_forest.split_tree)
    if old_forest.count_tree is not None:
        assert new_forest.count_tree is not None
        assert_array_equal(
            jnp.where(is_leaf, old_forest.count_tree, 0), new_forest.count_tree
        )
    if old_forest.prec_tree is not None:
        assert new_forest.prec_tree is not None
        assert_close_matrices(
            jnp.where(is_leaf, old_forest.prec_tree, 0), new_forest.prec_tree, rtol=1e-5
        )

    # compare residuals
    assert_close_matrices(state.resid, rebuilt.resid, rtol=1e-5)


# settings of `test_joint_distribution`
GEWEKE_NUM_CHAINS = 4000
GEWEKE_NUM_STEPS = 30
GEWEKE_NUM_TREES = 20
GEWEKE_N = 20
GEWEKE_NUM_TEST = 3
GEWEKE_MAXDEPTH = 6
GEWEKE_FLAT_P_NONTERMINAL = 0.6
GEWEKE_NU = 10.0
GEWEKE_ERROR_PREC = 4.0


class GewekeCase(NamedTuple):
    """Model configuration of a `test_joint_distribution` case."""

    p: int
    nsplits: int
    sparsity: Literal['fixed', 'free', 'supersparse'] | None = None
    flat: bool = False
    min_points: tuple[int, int] | None = None
    leaf_prior: bool = False
    outcome: Literal['continuous', 'binary', 'multivariate', 'mixed'] = 'continuous'
    error_scale: bool = False

    def __str__(self) -> str:
        defaults = GewekeCase(self.p, self.nsplits)
        parts = [f'p{self.p}', f'nsplits{self.nsplits}']
        for name, value, default in zip(self._fields, self, defaults, strict=True):
            if value == default:
                pass
            elif value is True:
                parts.append(name)
            elif isinstance(value, tuple):
                parts.append(name + '_'.join(map(str, value)))
            else:
                parts.append(f'{name}_{value}')
        return '-'.join(parts)


def geweke_sparse_kw(
    sparsity: Literal['fixed', 'free', 'supersparse'] | None, p: int
) -> kwdict:
    """Sparsity-prior hyperparameters for `init`."""
    # 'free' is the default of `Bart`, 'supersparse' puts more prior mass on a
    # small theta, which makes `s` very sparse and stresses the augmentation of
    # `log_s` with tiny eligible split probability masses
    params = {
        'fixed': dict(theta=float(p)),
        'free': dict(a=0.5, b=1.0, rho=float(p)),
        'supersparse': dict(a=0.1, b=1.0, rho=float(p)),
    }
    return {} if sparsity is None else params[sparsity]


def geweke_state(key: Key[Array, ''], case: GewekeCase) -> ChainDataState:
    """Create a state with the model of `case` and many chains, each with its own data."""
    k = 2 if case.outcome in ('multivariate', 'mixed') else None
    kshape = () if k is None else (k,)
    eye = jnp.float32(1.0) if k is None else jnp.eye(k)

    leaf_prec = GEWEKE_NUM_TREES * eye
    if case.leaf_prior:
        rate = GEWEKE_NU / GEWEKE_NUM_TREES * eye
        leaf_prior_cov_inv = Wishart(GEWEKE_NU, rate, leaf_prec)
    else:
        leaf_prior_cov_inv = Wishart(None, None, leaf_prec)

    error_kw: dict
    error_rate = GEWEKE_NU / GEWEKE_ERROR_PREC * eye
    if case.outcome == 'binary':
        error_kw = dict(outcome_type='binary')
    elif case.outcome == 'mixed':
        error_kw = dict(
            outcome_type=['continuous', 'binary'],
            error_cov_inv=DiagWishart(GEWEKE_NU, error_rate, eye),
        )
    else:
        error_kw = dict(error_cov_inv=Wishart(GEWEKE_NU, error_rate, eye))

    if case.flat:
        p_nonterminal = jnp.full(GEWEKE_MAXDEPTH - 1, GEWEKE_FLAT_P_NONTERMINAL)
    else:
        p_nonterminal = make_p_nonterminal(GEWEKE_MAXDEPTH)

    min_points_kw: dict = {}
    if case.min_points is not None:
        decision, leaf = case.min_points
        min_points_kw = dict(
            min_points_per_decision_node=decision, min_points_per_leaf=leaf
        )

    sparse_kw = geweke_sparse_kw(case.sparsity, case.p)
    if case.sparsity is not None:
        sparse_kw = dict(**sparse_kw, sparse_on_at=0, augment=True)

    state = init(
        X=random.randint(key, (case.p, GEWEKE_N), 0, case.nsplits + 1, jnp.uint8),
        y=jnp.zeros((*kshape, GEWEKE_N)),
        offset=jnp.zeros(kshape),
        max_split=jnp.full(case.p, case.nsplits, jnp.uint8),
        num_trees=GEWEKE_NUM_TREES,
        p_nonterminal=p_nonterminal,
        leaf_prior_cov_inv=leaf_prior_cov_inv,
        error_scale=jnp.linspace(0.5, 2.0, GEWEKE_N) if case.error_scale else None,
        num_chains=GEWEKE_NUM_CHAINS,
        **error_kw,
        **min_points_kw,
        **sparse_kw,
    )
    return ChainDataState.from_state(state)


def sample_joint(key: Key[Array, ''], state: ChainDataState) -> ChainDataState:
    """Draw parameters and data from the model, independently in each chain."""
    keys = split(key)
    return sample_data(keys.pop(), sample_prior(keys.pop(), state))


def geweke_step(key: Key[Array, ''], state: State) -> State:
    """Draw new data given the parameters, then the parameters given the data."""
    keys = split(key)
    return step(keys.pop(), sample_data(keys.pop(), state))


@jit
def run_geweke_chains(step_keys: Key[Array, ' steps'], state: State) -> State:
    """Alternate `step` and `sample_data` in many independent chains."""

    def body(state: State, key: Key[Array, '']) -> tuple[State, None]:
        return geweke_step(key, state), None

    state, _ = lax.scan(body, state, step_keys)
    return state


def summaries(
    state: State, X_test: UInt8[Array, 'p m']
) -> dict[str, Shaped[Array, ' chains']]:
    """Compute scalar summaries of the parameters and data of each chain."""
    in_axes = (chain_vmap_axes(state),)
    return vmap(partial(chain_summaries, X_test=X_test), in_axes=in_axes)(state)


def chain_summaries(
    state: State, X_test: UInt8[Array, 'p m']
) -> dict[str, Shaped[Array, '']]:
    """Compute scalar summaries of the parameters and data of a single chain."""
    forest = state.forest
    p, _ = X_test.shape
    num_splits = jnp.count_nonzero(forest.split_tree, axis=-1)
    varcount = var_histogram(p, forest.var_tree, forest.split_tree, sum_batch_axis=-1)
    yhat = evaluate_forest(X_test, forest, sum_batch_axis=-1).ravel()
    (num_yhat,) = yhat.shape
    out = {
        'number of stub trees': jnp.count_nonzero(num_splits == 0),
        'number of simple trees': jnp.count_nonzero(num_splits == 1),
        'number of nodes': num_splits.sum(),
        'imbalance index': avg_imbalance_index(forest.split_tree),
        'average max tree depth': avg_max_tree_depth(forest.split_tree),
        **{f'varcount[{i}]': varcount[i] for i in range(p)},
        **{f'y_test[{i}]': yhat[i] for i in range(num_yhat)},
    }

    if forest.sparsity.theta is not None:
        assert forest.sparsity.log_s is not None
        if p > 1:  # with p=1, s is always [1]
            varprob = softmax(forest.sparsity.log_s)
            out['effective predictors'] = effective_predictors(varprob)
        if forest.sparsity.rho is not None:
            out['theta'] = forest.sparsity.theta

    for name, prec in (
        ('leaf precision', forest.leaf_prior_cov_inv),
        ('error precision', state.error_cov_inv),
    ):
        if prec.nu is not None:
            value = prec.value.ravel()
            (size,) = value.shape
            out |= {f'{name}[{i}]': value[i] for i in range(size)}

    errors = standardized_errors(state)
    out['error sum of squares'] = jnp.sum(jnp.square(errors))
    out['first error'] = errors[0, 0]
    return out


def standardized_errors(state: State) -> Float32[Array, 'k n']:
    """Compute the errors of the model of a single chain, whitened to iid standard Normal."""
    fit = evaluate_forest(state.X, state.forest, sum_batch_axis=-1)
    errors = outcome_target(state) - fit - state.forest.offset[..., None]
    if state.error_scale is not None:
        errors /= state.error_scale
    prec = state.error_cov_inv.value
    if prec.ndim == 0:
        return (errors * jnp.sqrt(prec))[None, :]
    else:
        return jnp.linalg.cholesky(prec).T @ errors


def effective_predictors(
    varprob: Float32[Array, '*batch p'],
) -> Float32[Array, ' *batch']:
    """Effective number of predictors, the exp of the split-probability entropy."""
    entropy = -jnp.sum(jnp.where(varprob > 0, varprob * jnp.log(varprob), 0.0), axis=-1)
    return jnp.exp(entropy)


@partial(jnp.vectorize, signature='(n,m)->()')
def avg_imbalance_index(
    split_tree: UInt[Array, '*batch_shape ntree 2**(d-1)'],
) -> Float32[Array, '*batch_shape']:
    """Measure average tree imbalance in the forest."""
    is_leaf = vmap(partial(is_actual_leaf, add_bottom_level=True))(split_tree)
    depths = tree_depths(is_leaf.shape[-1])
    depths = jnp.broadcast_to(depths, is_leaf.shape)
    index = jnp.std(depths, where=is_leaf, axis=-1)
    return index.mean(-1)


@partial(jnp.vectorize, signature='(n,m)->()')
def avg_max_tree_depth(
    split_tree: UInt[Array, '*batch_shape ntree 2**(d-1)'],
) -> Float32[Array, '*batch_shape']:
    """Measure average maximum tree depth in the forest."""
    depth = vmap(tree_actual_depth)(split_tree)
    return depth.mean(-1)


@pytest.mark.parametrize(
    'case',
    [
        GewekeCase(1, 1),
        GewekeCase(3, 2),
        GewekeCase(10, 1),
        GewekeCase(10, 255),
        GewekeCase(10, 1, sparsity='fixed'),
        GewekeCase(10, 1, sparsity='free'),
        GewekeCase(10, 255, sparsity='free'),
        GewekeCase(1, 1, sparsity='free'),
        GewekeCase(3, 2, sparsity='free'),
        GewekeCase(3, 2, sparsity='supersparse'),
        GewekeCase(10, 1, sparsity='supersparse'),
        # with a flatter tree prior, fewer moves are accepted almost surely and
        # the trees run out of decision rules, so the proposal ratios matter
        GewekeCase(1, 1, flat=True),
        GewekeCase(1, 3, flat=True),
        GewekeCase(3, 2, flat=True),
        GewekeCase(10, 255, flat=True),
        GewekeCase(3, 2, flat=True, min_points=(6, 3)),
        GewekeCase(10, 255, min_points=(6, 3)),
        GewekeCase(10, 255, leaf_prior=True),
        GewekeCase(10, 255, outcome='binary'),
        GewekeCase(10, 255, outcome='multivariate', leaf_prior=True),
        GewekeCase(10, 255, outcome='mixed'),
        GewekeCase(10, 255, error_scale=True),
    ],
    ids=str,
)
def test_joint_distribution(keys: split, case: GewekeCase, subtests: SubTests) -> None:
    """Check `step` keeps parameters and data jointly distributed as the model.

    Many independent chains start from draws of the model, and alternate
    `step` with drawing new data given the parameters (Geweke 2004). If `step`
    leaves the posterior invariant, the chains stay distributed as the model,
    which is compared against independent draws. The blocked low-p cases with
    variable selection stress-test the augmentation of `log_s`.
    """
    state = geweke_state(keys.pop(), case)
    initial = sample_joint(keys.pop(), copy_arrays(state))
    final = run_geweke_chains(keys.pop(GEWEKE_NUM_STEPS), initial)
    ref = sample_joint(keys.pop(), state)

    # a sampler stuck at its initial state would pass the comparison
    with subtests.test('chains move'):
        changed = initial.forest.split_tree != final.forest.split_tree
        changed = chain_to_axis(changed, chain_vmap_axes(final).forest.split_tree)
        moved = jnp.any(changed.reshape(GEWEKE_NUM_CHAINS, -1), axis=1)
        assert jnp.mean(moved) > 0.5

    X_test = random.randint(
        keys.pop(), (case.p, GEWEKE_NUM_TEST), 0, case.nsplits + 1, jnp.uint8
    )
    mcmc_stats = summaries(final, X_test)
    ref_stats = summaries(ref, X_test)

    # where the distribution is known exactly, compare with it, which has more
    # power than comparing with independent draws
    errors = vmap(standardized_errors, in_axes=(chain_vmap_axes(final),))(final)
    exact = {'errors': (errors.ravel(), stats.norm.cdf)}
    for name, prec in (
        ('leaf precision', final.forest.leaf_prior_cov_inv),
        ('error precision', final.error_cov_inv),
    ):
        if prec.nu is not None and prec.value.ndim == 1:
            assert prec.rate is not None
            cdf = stats.gamma(prec.nu.item() / 2, scale=2 / prec.rate.item()).cdf
            exact[name] = (prec.value, cdf)

    # keep the joint false alarm rate low
    threshold = 1e-3 / (len(mcmc_stats) + len(exact))

    for name, x in mcmc_stats.items():
        with subtests.test(name):
            pvalue = ks_2samp(x, ref_stats[name], method='asymp').pvalue
            assert pvalue > threshold

    for name, (x, cdf) in exact.items():
        with subtests.test(f'{name} (exact)'):
            assert ks_1samp(x, cdf).pvalue > threshold


class TestMixedBinaryContinuous:
    """Tests for mixed binary-continuous multivariate outcome support."""

    n = 100
    p = 10
    k = 3
    numcut = 10
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
            num_trees=NUM_TREES,
            p_nonterminal=jnp.full(self.d - 1, 0.9),
            leaf_prior_cov_inv=Wishart(None, None, jnp.eye(self.k) * NUM_TREES),
            error_cov_inv=DiagWishart(
                nu=2.0, rate=jnp.diag(jnp.array([0.0, 2.0, 0.0])), value=jnp.eye(self.k)
            ),
        )

    def test_init_shapes(self, init_kwargs: dict) -> None:
        """Check that init produces correct shapes for mixed outcomes."""
        state = init(**init_kwargs)

        # binary_indices should contain indices of binary components
        assert state.binary_indices is not None
        assert_array_equal(state.binary_indices, jnp.array([0, 2], jnp.int32))

        # y should hold all k rows, whatever the outcome type
        assert state.y.shape == (self.k, self.n)
        assert state.y.dtype == jnp.float32

        # z should have only binary rows (kb=2)
        assert state.z is not None
        assert state.z.shape == (2, self.n)

        # resid should have all k rows
        assert state.resid.shape == (self.k, self.n)

        # error_cov_inv.value should be a (k, k) diagonal matrix
        value = state.error_cov_inv.value
        assert value.shape == (self.k, self.k)
        # off-diagonal should be zero
        assert_array_equal(value, jnp.diag(jnp.diag(value)))

        # binary diagonal entries should be 1.0
        assert value[0, 0] == 1.0
        assert value[2, 2] == 1.0

        # the Wishart prior parameters should be set
        assert state.error_cov_inv.nu is not None
        assert state.error_cov_inv.rate is not None

    def test_init_resid_binary_rows_zero(self, init_kwargs: dict) -> None:
        """Check that the binary rows of resid are initialized to zero."""
        # copy the inputs because init may donate them
        state = init(**copy_arrays(init_kwargs))

        # binary rows (0 and 2) should be zero
        assert_array_equal(state.resid[0], jnp.zeros(self.n))
        assert_array_equal(state.resid[2], jnp.zeros(self.n))

        # continuous row (1) should be y[1] - offset[1] in data units (resid is
        # stored in units of resid_unit)
        y = init_kwargs['y']
        offset = init_kwargs['offset']
        expected = y[1] - offset[1]
        data_units = state.resid[1] * state.resid_unit[1]
        assert_close_matrices(data_units, expected, rtol=1e-6)

    def test_init_z_values(self, init_kwargs: dict) -> None:
        """Check that z is initialized to offset for binary components."""
        state = init(**init_kwargs)

        assert state.z is not None
        offset = init_kwargs['offset']
        # z[0] should be offset[0] (first binary component)
        assert_array_equal(state.z[0], jnp.full(self.n, offset[0]))
        # z[1] should be offset[2] (second binary component, index 2 in y)
        assert_array_equal(state.z[1], jnp.full(self.n, offset[2]))

    @pytest.mark.parametrize(
        ('outcome_type', 'with_missing'),
        [
            (['binary', 'continuous', 'binary'], False),
            (['continuous', 'continuous', 'continuous'], True),
        ],
        ids=['mixed', 'partial_missing'],
    )
    def test_init_rejects_nondiagonal_scale(
        self, init_kwargs: dict, outcome_type: Sequence[str], with_missing: bool
    ) -> None:
        """Check that init rejects non-diagonal error_cov_scale."""
        init_kwargs['outcome_type'] = outcome_type
        if with_missing:
            init_kwargs['missing'] = jnp.zeros((self.k, self.n), jnp.bool_)
        prior = init_kwargs['error_cov_inv']
        init_kwargs['error_cov_inv'] = DiagWishart(
            nu=prior.nu,
            rate=prior.rate + 0.1 * jnp.ones((self.k, self.k)),
            value=prior.value,
        )
        with pytest.raises(Exception, match='diagonal'):
            _state = init(**init_kwargs)

    @pytest.mark.parametrize(
        ('outcome_type', 'with_missing'),
        [
            (['binary', 'continuous', 'binary'], False),
            (['continuous', 'continuous', 'continuous'], True),
        ],
        ids=['mixed', 'partial_missing'],
    )
    def test_init_rejects_nondiagonal_value(
        self, init_kwargs: dict, outcome_type: Sequence[str], with_missing: bool
    ) -> None:
        """Check that init rejects a non-diagonal initial precision value."""
        init_kwargs['outcome_type'] = outcome_type
        if with_missing:
            init_kwargs['missing'] = jnp.zeros((self.k, self.n), jnp.bool_)
        prior = init_kwargs['error_cov_inv']
        init_kwargs['error_cov_inv'] = DiagWishart(
            nu=prior.nu, rate=prior.rate, value=prior.value.at[0, 1].set(0.5)
        )
        with pytest.raises(Exception, match='diagonal'):
            _state = init(**init_kwargs)

    def test_init_rejects_binary_nonunit_value(self, init_kwargs: dict) -> None:
        """Check that init rejects a binary initial precision other than 1."""
        prior = init_kwargs['error_cov_inv']
        # component 0 is binary, so its precision must stay at 1
        init_kwargs['error_cov_inv'] = DiagWishart(
            nu=prior.nu, rate=prior.rate, value=prior.value.at[0, 0].set(2.0)
        )
        with pytest.raises(Exception, match='binary error precision must be 1'):
            _state = init(**init_kwargs)

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
        prec = state.error_cov_inv.value[1, 1]

        # replace resid because the default initial resid is 0 for binary
        # outcomes, which triggers a division by zero in step_error_cov_inv
        state = replace(state, resid=jnp.full_like(state.resid, 1.0))

        new_state = step_error_cov_inv(keys.pop(), state)
        new_value = new_state.error_cov_inv.value

        # binary diagonal entries (indices 0, 2) should stay 1.0
        assert new_value[0, 0] == 1.0
        assert new_value[2, 2] == 1.0

        # continuous diagonal entry (index 1) should be updated (not the init value)
        assert new_value[1, 1] != prec

        # off-diagonal should remain zero
        assert_array_equal(new_value, jnp.diag(jnp.diag(new_value)))

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
                error_cov_inv=None,
            )
        else:
            init_kwargs.update(
                y=random.normal(keys.pop(), (self.k, self.n)),
                error_cov_inv=Wishart(
                    nu=2.0, rate=2 * jnp.eye(self.k), value=jnp.eye(self.k)
                ),
            )

        copy_args = partial(copy_arrays, init_kwargs)

        init_kwargs.update(outcome_type=outcome_type)
        scalar_state = init(**copy_args())

        init_kwargs.update(outcome_type=[outcome_type] * self.k)
        sequence_state = init(**copy_args())

        def check_equal(
            path: KeyPath,
            scalar: Shaped[Array, '*shape'],
            sequence: Shaped[Array, '*shape'],
        ) -> None:
            assert_array_equal(scalar, sequence, err_msg=f'{keystr(path)}: ')

        tree.map_with_path(check_equal, scalar_state, sequence_state)

    def test_outcome_type_length_mismatch(self, init_kwargs: dict) -> None:
        """Check that mismatched outcome_type length raises."""
        init_kwargs.update(outcome_type=['binary'] * (self.k - 1))
        with pytest.raises(AssertionError):
            init(**init_kwargs)


class MCMCStepData(NamedTuple):
    """Toy dataset for testing."""

    X: UInt32[Array, 'p n']
    y: Float32[Array, ' n']
    max_split: UInt32[Array, ' p']


def random_pd_matrix(key: Key[Array, ''], k: int) -> Float[Array, '{k} {k}']:
    """Generate a random positive definite matrix."""
    A = random.normal(key, (k, k))
    return A @ A.T + jnp.eye(k)


@pytest.fixture(params=[(10, 100), (50, 5)])
def mcmcstep_data_shape(request: FixtureRequest) -> tuple[int, int]:
    """Provide (n, p) pairs for testing."""
    return request.param


@pytest.fixture
def mcmcstep_data(mcmcstep_data_shape: tuple[int, int]) -> MCMCStepData:
    """Generate a toy dataset."""
    n, p = mcmcstep_data_shape
    numcut = 5
    # bin the datapoint index into [0, numcut], rotating the phase per variable.
    # Binning keeps every variable splittable, since values above the cutpoint
    # range send all datapoints to the same child, and the rotation makes the
    # variables differ while each stays correlated with `y`.
    index = (jnp.arange(n) + jnp.arange(p)[:, None]) % n
    X = (index * (numcut + 1) // n).astype(jnp.uint32)
    y = jnp.linspace(-1, 1, n)
    max_split = jnp.full(p, numcut, jnp.uint32)
    return MCMCStepData(X, y, max_split)


class TestStepLeafPriorCovInv:
    """Test the Gibbs update of the leaf prior precision."""

    @pytest.fixture(params=['uv', 'mv'])
    def init_kwargs(self, request: FixtureRequest, mcmcstep_data: MCMCStepData) -> dict:
        """Arguments to `init` for a univariate or bivariate regression."""
        X, y, max_split = mcmcstep_data
        kw: dict = dict(
            X=X,
            max_split=max_split,
            num_trees=NUM_TREES,
            p_nonterminal=jnp.array([0.9, 0.5]),
        )
        # a precision of 4 gives a leaf unit of 1/2, which exercises the
        # conversion of the leaves to data units
        if request.param == 'mv':
            k = 2
            eye = jnp.eye(k)
            return dict(
                kw,
                y=jnp.stack([y, -y]),
                offset=jnp.zeros(k),
                leaf_prior_cov_inv=Wishart(None, None, 4.0 * eye),
                error_cov_inv=Wishart(nu=2.0, rate=2.0 * eye, value=eye),
            )
        else:
            return dict(
                kw,
                y=y,
                offset=0.0,
                leaf_prior_cov_inv=Wishart(None, None, 4.0),
                error_cov_inv=Wishart(nu=2.0, rate=2.0, value=1.0),
            )

    @pytest.fixture
    def prior(self, init_kwargs: dict) -> Wishart:
        """Put a Wishart prior on the leaf precision, centered on the fixed value."""
        value = init_kwargs['leaf_prior_cov_inv'].value
        return Wishart(nu=3.0 + value.ndim, rate=0.5 * value, value=value)

    def test_reject_diag_wishart(self, init_kwargs: dict, prior: Wishart) -> None:
        """`DiagWishart` is not allowed as leaf prior."""
        if prior.value.ndim == 0:
            pytest.skip('DiagWishart is multivariate only')
        diag = DiagWishart(nu=prior.nu, rate=prior.rate, value=prior.value)
        kw: dict = dict(init_kwargs, leaf_prior_cov_inv=diag)
        with pytest.raises(AssertionError, match='dense Wishart'):
            init(**kw)

    def test_reject_shape_mismatch(self, init_kwargs: dict, prior: Wishart) -> None:
        """The prior rate must match the outcome shape."""
        bad = Wishart(nu=prior.nu, rate=jnp.ones((3, 3)), value=prior.value)
        kw: dict = dict(init_kwargs, leaf_prior_cov_inv=bad)
        with pytest.raises(AssertionError):
            init(**kw)

    # `step` runs under float32 matmuls; without it, gpu contracts via tf32
    @float32_matmuls
    def test_closed_form(self, keys: split, init_kwargs: dict, prior: Wishart) -> None:
        """The update on a hand-built forest matches the conjugate formula."""
        kw: dict = dict(init_kwargs, leaf_prior_cov_inv=prior)
        state = init(**kw)
        prior = state.forest.leaf_prior_cov_inv  # `init` donates the input

        # tree 0: root split, leaves 2, 3
        # tree 2: nodes 1 and 2 split, leaves 3, 4, 5 (4, 5 at the bottom level)
        # other trees: root leaf
        split_tree = jnp.zeros_like(state.forest.split_tree)
        split_tree = split_tree.at[0, 1].set(1)
        split_tree = split_tree.at[2, 1:3].set(1)
        leaves = [(0, 2), (0, 3), (1, 1), (2, 3), (2, 4), (2, 5), (3, 1), (4, 1)]

        # fill all nodes, non-leaves included, to check they are masked out
        leaf_tree = random.normal(keys.pop(), state.forest.leaf_tree.shape)

        state = replace(
            state,
            forest=replace(state.forest, split_tree=split_tree, leaf_tree=leaf_tree),
        )

        values = jnp.stack([leaf_tree[t, ..., i] for t, i in leaves])
        values *= state.forest.leaf_unit
        if values.ndim == 2:
            expected_scatter = values.T @ values
        else:
            expected_scatter = jnp.sum(jnp.square(values))

        count, scatter = leaf_scatter(state.forest)
        assert count == len(leaves)
        assert_close_matrices(scatter, expected_scatter, rtol=1e-6)

        key = keys.pop()
        actual = step_leaf_prior_cov_inv(key, state).forest.leaf_prior_cov_inv.value
        expected = sample_wishart_posterior(
            random.clone(key), prior, jnp.int32(len(leaves)), expected_scatter
        )
        assert_close_matrices(actual, expected, rtol=1e-6)


@pytest.mark.parametrize('offset', [-30.0, 30.0])
def test_z_valid_with_confident_misclassification(
    keys: split, mcmcstep_data: MCMCStepData, offset: float
) -> None:
    """Check `step_z` when the latent is deep on the wrong side.

    An offset which contradicts the outcome truncates the latent normal far into
    a tail, where the inverse normal cdf overflows if not guarded, and where the
    sample saturates short of the sign implied by the outcome if not clipped.
    """
    X, y, max_split = mcmcstep_data
    state = init(
        X=X,
        y=(y > 0).astype(jnp.float32),
        outcome_type='binary',
        offset=offset,
        max_split=max_split,
        num_trees=NUM_TREES,
        p_nonterminal=jnp.array([0.9, 0.5]),
        leaf_prior_cov_inv=Wishart(None, None, 1.0),
    )

    # `step_z` and not `step` because `step` is jitted, and whether xla flushes
    # the subnormals which cause the overflow depends on how it fuses the ops
    new_state = step_z(keys.pop(), state)

    z = nnone(new_state.z)
    assert jnp.all(jnp.isfinite(z))
    assert jnp.all(jnp.isfinite(new_state.resid))
    assert jnp.all(jnp.where(new_state.y != 0, z >= 0, z <= 0))


def test_chol_with_gersh_disparate_scales() -> None:
    """Gershgorin stabilization is per-component, not a single global shift.

    A global shift set by the largest component would swamp the precision of
    much smaller ones, as when a mixed model heavily scales a continuous
    outcome alongside O(1) binary ones, corrupting the Cholesky and the
    inverse (and hence `leaf_unit`).
    """
    # diagonal precisions spanning 14 orders of magnitude
    precisions = jnp.array([1e-8, 1.0, 1e6])
    mat = jnp.diag(precisions)
    assert_close_matrices(
        chol_with_gersh(mat), jnp.diag(jnp.sqrt(precisions)), rtol=1e-5
    )
    assert_close_matrices(
        inv_via_chol_with_gersh(mat), jnp.diag(1 / precisions), rtol=1e-5
    )


class TestWishart:
    """Test the basic properties of the wishart sampler output."""

    # Parameterize with (k, df) pairs
    @pytest.fixture(params=[(1, 1.0), (3, 100.0), (100, 102.0)])
    def wishart_params(self, request: FixtureRequest) -> tuple[int, float]:
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

    def test_inv_wishart_marginal_nu(self) -> None:
        """The marginal df is `nu` for scalar priors and ``nu - k + 1`` for matrices."""
        assert Wishart(nu=None, rate=None, value=1.0).inv_wishart_marginal_nu is None
        for cls, rate, k in (
            (Wishart, 2.0, 1),
            (Wishart, jnp.eye(1), 1),
            (Wishart, jnp.eye(3), 3),
            (DiagWishart, jnp.eye(3), 3),
        ):
            prior = cls(nu=5.0, rate=rate, value=rate)
            assert prior.inv_wishart_marginal_nu == 5.0 - (k - 1)
            # round-trip through the classmethod
            prior = cls.from_inv_wishart_marginal_nu(3.0, rate=rate, value=rate)
            assert type(prior) is cls
            assert prior.nu == 3.0 + (k - 1)
            assert prior.inv_wishart_marginal_nu == 3.0

    def test_size(self, keys: split, wishart_params: tuple[int, float]) -> None:
        """Check that the sample generated by wishart sampler is of shape k*k."""
        k, df = wishart_params
        scale = random_pd_matrix(keys.pop(), k)
        sample = _sample_wishart_bartlett(keys.pop(), df, scale)
        assert sample.shape == (k, k)

    def test_symmetric(self, keys: split, wishart_params: tuple[int, float]) -> None:
        """Check that the sample generated by wishart sampler is symmetric."""
        k, df = wishart_params
        scale = random_pd_matrix(keys.pop(), k)
        sample = _sample_wishart_bartlett(keys.pop(), df, scale)
        assert_close_matrices(sample, sample.T, rtol=1e-6)

    def test_pos_def(self, keys: split, wishart_params: tuple[int, float]) -> None:
        """Check that the sample generated by wishart sampler is positive definite."""
        k, df = wishart_params
        scale = random_pd_matrix(keys.pop(), k)
        sample = _sample_wishart_bartlett(keys.pop(), df, scale)
        eigs = jnp.linalg.eigvalsh(sample)
        assert jnp.all(eigs > 0)

    def test_near_singular_scale(
        self, keys: split, wishart_params: tuple[int, float]
    ) -> None:
        """Check that the wishart sampler still works with singular or near singular matrix."""
        k, df = wishart_params
        ill_conditioned_scale = self.ill_conditioned_matrix(keys.pop(), k)
        sample = _sample_wishart_bartlett(keys.pop(), df, ill_conditioned_scale)
        assert jnp.all(jnp.isfinite(sample))

    def test_wishart_dist(self, keys: split, wishart_params: tuple[int, float]) -> None:
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

    @pytest.fixture(params=[1, 3])
    def k(self, request: FixtureRequest) -> int:
        """Provide different ks for testing."""
        return request.param

    def test_shapes_leaf(self, keys: split, k: int) -> None:
        """Check that shapes of outputs are correct."""
        tree_size = 4
        prec_trees = jnp.ones((NUM_TREES, tree_size))
        error_cov_inv = random_pd_matrix(keys.pop(), k)
        leaf_prior_cov_inv = random_pd_matrix(keys.pop(), k)

        result = _precompute_leaf_terms_mv(
            keys.pop(), prec_trees, error_cov_inv, leaf_prior_cov_inv
        )
        assert result.mean_factor.shape == (NUM_TREES, k, k, tree_size)
        assert result.centered_leaves.shape == (NUM_TREES, k, tree_size)
        assert result.logdet_prec.shape == (NUM_TREES, tree_size)

    def test_likelihood_equiv(self, keys: split) -> None:
        """Check that _compute_likelihood_ratio_uv and _compute_likelihood_ratio_mv agree when k = 1."""
        inv_sigma2 = random.uniform(keys.pop(), (), minval=0.1, maxval=5.0)
        leaf_prior_cov_inv_uv = random.uniform(keys.pop(), (), minval=0.1, maxval=5.0)
        error_cov_inv = jnp.array([[inv_sigma2]])
        leaf_prior_cov_inv = jnp.array([[leaf_prior_cov_inv_uv]])

        # precision of the parent node (= left + right) at heap position 1,
        # of its left and right children at positions 2 and 3
        prec_trees = jnp.array([[0.0, 7.0, 3.0, 4.0]])
        lrt_nodes = jnp.array([[2, 3, 1]])

        # sum of scaled residuals in the left, right, and parent node; k = 1
        resid_lrt = random.normal(keys.pop(), (1, 3))

        prelf_mv = _precompute_leaf_terms_mv(
            keys.pop(), prec_trees, error_cov_inv, leaf_prior_cov_inv
        )
        prelkv_mv = _precompute_likelihood_terms_mv(
            error_cov_inv, leaf_prior_cov_inv, prelf_mv, lrt_nodes
        )
        # the precompute terms are batched over trees, while the ratio is
        # computed one tree at a time; strip the singleton num_trees axis
        prelkv_mv = tree.map(lambda x: x.squeeze(0), prelkv_mv)
        likelihood_mv = _compute_likelihood_ratio_mv(resid_lrt, prelkv_mv)

        prelf_uv = _precompute_leaf_terms_uv(
            keys.pop(), prec_trees, inv_sigma2, leaf_prior_cov_inv_uv
        )
        prelkv_uv = _precompute_likelihood_terms_uv(
            inv_sigma2, leaf_prior_cov_inv_uv, prelf_uv, lrt_nodes
        )
        prelkv_uv = tree.map(lambda x: x.squeeze(0), prelkv_uv)
        likelihood_uv = _compute_likelihood_ratio_uv(resid_lrt[0, :], prelkv_uv)

        assert_allclose(
            prelkv_mv.log_sqrt_term, prelkv_uv.log_sqrt_term, rtol=1e-6, atol=1e-6
        )
        assert_allclose(likelihood_mv, likelihood_uv, rtol=1e-6, atol=1e-6)

    def test_leaf_terms_equiv(self, keys: split) -> None:
        """Check that _precompute_leaf_terms_uv and _precompute_leaf_terms_mv agree when k = 1."""
        tree_size = 3
        inv_sigma2 = random.uniform(keys.pop(), (), minval=0.1, maxval=5.0)
        leaf_prior_cov_inv_uv = random.uniform(keys.pop(), (), minval=0.1, maxval=5.0)

        error_cov_inv = jnp.array([[inv_sigma2]])
        leaf_prior_cov_inv = jnp.array([[leaf_prior_cov_inv_uv]])
        prec_trees = random.uniform(keys.pop(), (NUM_TREES, tree_size)) * 5.0
        z_uv = random.normal(keys.pop(), (NUM_TREES, tree_size))
        z_mv = z_uv[:, None, :]  # (num_trees, k=1, tree_size), leaf axis trailing

        result_uv = _precompute_leaf_terms_uv(
            keys.pop(), prec_trees, inv_sigma2, leaf_prior_cov_inv_uv, z_uv
        )
        result_mv = _precompute_leaf_terms_mv(
            keys.pop(), prec_trees, error_cov_inv, leaf_prior_cov_inv, z_mv
        )

        assert_close_matrices(
            result_uv.mean_factor,
            result_mv.mean_factor.squeeze((1, 2)),
            rtol=1e-6,
            atol=1e-7,
        )
        assert_close_matrices(
            result_uv.centered_leaves,
            result_mv.centered_leaves.squeeze(1),
            rtol=1e-6,
            atol=1e-7,
        )
        # the posterior precision is error_cov_inv / mean_factor when k = 1
        assert_close_matrices(
            jnp.log(inv_sigma2 / result_uv.mean_factor),
            result_mv.logdet_prec,
            rtol=1e-6,
            atol=1e-7,
        )


class TestMVBartIntegration:
    """Test equivalence between Univariate and Multivariate (k=1) modes."""

    def test_step_sigma_distribution_match(
        self, keys: split, mcmcstep_data: MCMCStepData
    ) -> None:
        """
        Test that the univariate (diag) and multivariate (k = 1) samplers draw from the same posterior.

        UV: 1/sigma2 ~ Gamma(alpha_post, beta_post)
        MV: error_cov_inv ~ Wishart(df_post, scale_post)
        """
        X, y, _ = mcmcstep_data
        resid = random.normal(keys.pop(), (y.size,))

        # inverse gamma prior: alpha = df / 2, beta = scale / 2
        df_prior = jnp.float32(20.0)
        scale_prior = jnp.float32(10.0)

        common = _bare_state_kw((), y.size, X=X, y=y)

        st_uv = State(
            **common,
            resid=resid,
            error_cov_inv=Wishart(
                nu=df_prior, rate=scale_prior, value=jnp.float32(1.0)
            ),
        )

        st_mv = State(
            **common,
            resid=resid[None, :],
            error_cov_inv=Wishart(
                nu=df_prior, rate=jnp.array([[scale_prior]]), value=jnp.eye(1)
            ),
        )

        def sample_uv(k: Key[Array, '']) -> Float32[Array, '']:
            return _step_error_cov_inv_diag(k, st_uv).error_cov_inv.value

        def sample_mv(k: Key[Array, '']) -> Float32[Array, '']:
            return _step_error_cov_inv_mv(k, st_mv).error_cov_inv.value.reshape(())

        n_samples = 10000
        samples_uv = vmap(sample_uv)(keys.pop(n_samples))
        samples_mv = vmap(sample_mv)(keys.pop(n_samples))

        _, p_value = ks_2samp(samples_uv, samples_mv)

        # The UV and MV draws are independent, so their sample means differ by
        # Monte Carlo error. For the smallest `n` this standard error is ~0.008,
        # so a 0.01 bound is only ~1.3 sigma and trips ~20% of the time; 0.05 is
        # ~6 sigma. Distribution equality is checked robustly by the KS test.
        assert jnp.abs(jnp.mean(samples_uv) - jnp.mean(samples_mv)) < 0.05
        # `samples_uv` and `samples_mv` are independent draws from the same
        # distribution, so the KS gate has a per-shape false-positive rate equal
        # to its threshold; keep it low to avoid tripping on benign realizations.
        assert p_value > 0.001

    def test_diag_matches_dense_marginal_variance(
        self, keys: split, mcmcstep_data: MCMCStepData
    ) -> None:
        """A `DiagWishart` draws each variance from the dense Wishart's marginal.

        With the same `nu` and `rate`, the posterior of a component's variance
        under the diagonal sampler must match the marginal of the inverse of
        the dense Wishart draw, whose degrees of freedom are ``nu - k + 1``.
        """
        X, y, _ = mcmcstep_data
        k = 2
        resid = random.normal(keys.pop(), (k, y.size))

        df_prior = jnp.float32(20.0)
        rate_prior = jnp.diag(jnp.array([10.0, 5.0]))

        common = _bare_state_kw((k,), y.size, X=X, y=resid, resid=resid)

        st_dense = State(
            **common,
            error_cov_inv=Wishart(nu=df_prior, rate=rate_prior, value=jnp.eye(k)),
        )
        st_diag = State(
            **common,
            error_cov_inv=DiagWishart(nu=df_prior, rate=rate_prior, value=jnp.eye(k)),
        )
        assert st_diag.error_cov_inv.inv_wishart_marginal_nu == df_prior - (k - 1)

        def sample_dense(key: Key[Array, '']) -> Float32[Array, ' k']:
            prec = _step_error_cov_inv_mv(key, st_dense).error_cov_inv.value
            return jnp.diagonal(jnp.linalg.inv(prec))

        def sample_diag(key: Key[Array, '']) -> Float32[Array, ' k']:
            prec = _step_error_cov_inv_diag(key, st_diag).error_cov_inv.value
            return jnp.reciprocal(jnp.diagonal(prec))

        n_samples = 10000
        var_dense = vmap(sample_dense)(keys.pop(n_samples))
        var_diag = vmap(sample_diag)(keys.pop(n_samples))

        for i in range(k):
            _, p_value = ks_2samp(var_dense[:, i], var_diag[:, i])
            assert p_value > 0.001

    def test_error_cov_inv_missing_equals_drop(
        self, keys: split, mcmcstep_data: MCMCStepData
    ) -> None:
        """1-D ``inv_sdev_scale`` zeros give the same sample as dropping those positions.

        Covers both the univariate diagonal path and the multivariate Wishart path
        (``inv_sdev_scale`` 1-D, no partial missingness), which is the dof fix.
        """
        X, y, _ = mcmcstep_data
        n = y.size
        resid_1d = random.normal(keys.pop(), (n,))
        mask = random.bernoulli(keys.pop(), 0.3, (n,))
        keep = ~mask
        # guard against a silently vacuous test: if nothing is masked, or
        # everything is, the two states compared below coincide
        assert jnp.any(mask)
        assert jnp.any(keep)
        inv_sdev = jnp.where(mask, 0.0, 1.0)

        df_prior = jnp.float32(20.0)
        scale_prior = jnp.float32(10.0)
        # `X` is not read by the samplers, but its `n` axis is cross-checked
        # against `resid`, so the dropped states carry the subset `X[:, keep]`.
        # both the masked and dropped states see the same kept-point count and
        # (unscaled) precision sum
        n_kept = jnp.sum(keep)
        uv_prior = Wishart(nu=df_prior, rate=scale_prior, value=jnp.float32(1.0))
        mv_prior = Wishart(nu=df_prior, rate=scale_prior[None, None], value=jnp.eye(1))

        st_uv_with = State(
            **_bare_state_kw(
                (),
                n_kept,
                X=X,
                y=resid_1d,
                resid=resid_1d,
                inv_sdev_scale=inv_sdev,
                error_cov_inv=uv_prior,
            )
        )
        st_uv_drop = State(
            **_bare_state_kw(
                (),
                n_kept,
                X=X[:, keep],
                y=resid_1d[keep],
                resid=resid_1d[keep],
                inv_sdev_scale=None,
                error_cov_inv=uv_prior,
            )
        )

        key = keys.pop()
        sample_with = _step_error_cov_inv_diag(key, st_uv_with).error_cov_inv.value
        sample_drop = _step_error_cov_inv_diag(key, st_uv_drop).error_cov_inv.value
        assert_allclose(sample_with, sample_drop, rtol=1e-6)

        # multivariate Wishart path (k=1) with 1-D inv_sdev_scale and zeros
        st_mv_with = State(
            **_bare_state_kw(
                (),
                n_kept,
                X=X,
                y=resid_1d,
                resid=resid_1d[None, :],
                inv_sdev_scale=inv_sdev,
                error_cov_inv=mv_prior,
            )
        )
        st_mv_drop = State(
            **_bare_state_kw(
                (),
                n_kept,
                X=X[:, keep],
                y=resid_1d[keep],
                resid=resid_1d[None, keep],
                inv_sdev_scale=None,
                error_cov_inv=mv_prior,
            )
        )
        sample_mv_with = _step_error_cov_inv_mv(key, st_mv_with).error_cov_inv.value
        sample_mv_drop = _step_error_cov_inv_mv(key, st_mv_drop).error_cov_inv.value
        assert_allclose(sample_mv_with, sample_mv_drop, rtol=1e-6)

    def test_error_cov_inv_2d_scale_units(
        self, keys: split, mcmcstep_data: MCMCStepData
    ) -> None:
        """The dense sampler applies per-component `inv_sdev_unit` to 2-D scales.

        A 2-D `inv_sdev_scale` stored in distinct per-component units must draw
        the same precision as the same scales with the units multiplied in.
        """
        X, y, _ = mcmcstep_data
        k = 2
        n = y.size
        resid = random.normal(keys.pop(), (k, n))
        # powers of two so that factoring the units out is exact
        inv_sdev_unit = jnp.array([0.5, 4.0])
        inv_sdev_scale = 2.0 ** random.randint(keys.pop(), (k, n), -2, 3)

        def make_state(
            inv_sdev_scale: Float32[Array, 'k n'], inv_sdev_unit: Float32[Array, ' k']
        ) -> State:
            return State(
                **_bare_state_kw(
                    (k,),
                    jnp.full(k, n),
                    X=X,
                    y=resid,
                    resid=resid,
                    inv_sdev_scale=inv_sdev_scale,
                    inv_sdev_unit=inv_sdev_unit,
                    sum_diag_prec_scale=jnp.sum(
                        jnp.square(inv_sdev_scale * inv_sdev_unit[:, None]), axis=-1
                    ),
                    error_cov_inv=Wishart(nu=20.0, rate=jnp.eye(k), value=jnp.eye(k)),
                )
            )

        st_units = make_state(inv_sdev_scale, inv_sdev_unit)
        st_plain = make_state(inv_sdev_scale * inv_sdev_unit[:, None], jnp.ones(k))

        key = keys.pop()
        prec_units = step_error_cov_inv(key, st_units).error_cov_inv.value
        prec_plain = step_error_cov_inv(key, st_plain).error_cov_inv.value
        # the 2-D scales must keep the dense path
        assert jnp.all(prec_units[~jnp.eye(k, dtype=bool)] != 0)
        assert_close_matrices(prec_units, prec_plain, rtol=1e-6)


class TestMultivariate:
    """Test for multivariate outcomes specifically."""

    @pytest.mark.parametrize('kind', ['binary', 'homo', 'het'])
    def test_1d_mv_matches_uv(
        self, keys: split, mcmcstep_data: MCMCStepData, kind: str
    ) -> None:
        """Check that multivariate with k=1 is equivalent to univariate."""
        X, y, max_split = mcmcstep_data

        if kind == 'binary':
            y = (y > 0).astype(jnp.float32)

        params = partial(
            copy_arrays,
            dict(
                X=X,
                max_split=max_split,
                num_trees=NUM_TREES,
                p_nonterminal=jnp.array([0.9, 0.5]),
                resid_reduction_config=BatchedReduction(num_batches=None),
                count_reduction_config=BatchedReduction(num_batches=None),
            ),
        )

        uv_kw: dict = dict(
            y=y,
            offset=0.0,
            leaf_prior_cov_inv=Wishart(None, None, jnp.float32(NUM_TREES)),
        )
        mv_kw: dict = dict(
            y=y[None, :],
            offset=jnp.zeros(1),
            leaf_prior_cov_inv=Wishart(None, None, NUM_TREES * jnp.eye(1)),
        )

        if kind == 'binary':
            uv_kw.update(outcome_type='binary')
            mv_kw.update(outcome_type='binary')
        else:
            uv_kw.update(error_cov_inv=Wishart(nu=4.0, rate=2.0, value=2.0))
            mv_kw.update(
                error_cov_inv=Wishart(
                    nu=jnp.array(4.0), rate=2 * jnp.eye(1), value=2 * jnp.eye(1)
                )
            )

        if kind == 'het':
            w = jnp.exp(random.uniform(keys.pop(), (y.size,), float, -0.5, 0.5))
            uv_kw.update(error_scale=w)
            mv_kw.update(error_scale=w[None, :])  # (1, n) — triggers vector-het path

        uv_state = init(**uv_kw, **params())
        mv_state = init(**mv_kw, **params())

        mv_state = replace(
            mv_state,
            resid=uv_state.resid[None, :],
            error_cov_inv=replace(
                mv_state.error_cov_inv,
                value=jnp.array([[uv_state.error_cov_inv.value]]),
            ),
            forest=replace(
                mv_state.forest,
                **copy_arrays(
                    dict(
                        var_tree=uv_state.forest.var_tree,
                        split_tree=uv_state.forest.split_tree,
                        leaf_tree=uv_state.forest.leaf_tree[:, None, :],
                        leaf_indices=uv_state.forest.leaf_indices,
                        affluence_tree=uv_state.forest.affluence_tree,
                    )
                ),
            ),
        )

        for key in keys.pop(3):
            # the uv and mv (k=1) reductions round the stored leaves slightly
            # differently, so with reduced leaf precision these continuous
            # quantities agree only to the rounding floor
            rtol = condf(uv_state.forest.leaf_tree, 1e-5, 1e-3)
            assert_close_matrices(
                uv_state.resid, mv_state.resid.squeeze(0), rtol=rtol, atol=1e-6
            )
            assert_close_matrices(
                uv_state.forest.leaf_tree,
                mv_state.forest.leaf_tree.squeeze(1),
                rtol=rtol,
            )

            # the full `step` resamples error_cov_inv: the diagonal (uv) and
            # Wishart (mv, k=1) paths must agree, up to the resid difference fed
            # into the denominator and the Gershgorin jitter of the mv Cholesky
            assert_close_matrices(
                uv_state.error_cov_inv.value.reshape(1, 1),
                mv_state.error_cov_inv.value,
                rtol=condf(uv_state.forest.leaf_tree, 1e-5, 1e-3),
            )

            assert_array_equal(uv_state.forest.var_tree, mv_state.forest.var_tree)
            assert_array_equal(uv_state.forest.split_tree, mv_state.forest.split_tree)
            assert_array_equal(
                uv_state.forest.leaf_indices, mv_state.forest.leaf_indices
            )
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

            uv_state = step(key, uv_state)
            mv_state = step(random.clone(key), mv_state)

    @pytest.mark.parametrize('kind', ['binary', 'homo', 'het'])
    def test_smoke(self, keys: split, mcmcstep_data: MCMCStepData, kind: str) -> None:
        """Run a few steps, check shapes and valid values."""
        X, y_uv, max_split = mcmcstep_data
        k = 3
        n = y_uv.size

        if kind == 'binary':
            y = random.bernoulli(keys.pop(), 0.5, (k, n)).astype(jnp.float32)
        else:
            y = jnp.tile(y_uv, (k, 1))
            y = y + random.normal(keys.pop(), y.shape) * 0.1

        kw: dict = dict(
            X=X,
            y=y,
            offset=jnp.zeros(k),
            max_split=max_split,
            num_trees=NUM_TREES,
            p_nonterminal=jnp.array([0.9, 0.5]),
            leaf_prior_cov_inv=Wishart(None, None, jnp.eye(k)),
            resid_reduction_config=BatchedReduction(num_batches=None),
            count_reduction_config=BatchedReduction(num_batches=None),
        )

        if kind == 'binary':
            kw.update(outcome_type='binary')
        else:
            kw.update(
                error_cov_inv=Wishart(
                    nu=jnp.array(10.0), rate=jnp.eye(k), value=10 * jnp.eye(k)
                )
            )

        if kind == 'het':
            w = jnp.exp(random.uniform(keys.pop(), (k, n), float, -0.5, 0.5))
            kw.update(error_scale=w)

        mv_state = init(**kw)

        if kind == 'het':
            assert mv_state.prec_scale is not None
            assert mv_state.prec_scale.shape == (k, k, n)
            assert mv_state.inv_sdev_scale is not None
            assert mv_state.inv_sdev_scale.shape == (k, n)

        for key in keys.pop(10):
            mv_state = step(key, mv_state)

            assert jnp.all(jnp.isfinite(mv_state.resid))
            assert jnp.all(jnp.isfinite(mv_state.forest.leaf_tree))

            assert mv_state.resid.shape == y.shape

            assert jnp.all(jnp.isfinite(mv_state.error_cov_inv.value))
            assert mv_state.error_cov_inv.value.shape == (k, k)

            if kind == 'binary':
                assert mv_state.z is not None
                assert jnp.all(jnp.isfinite(mv_state.z))
                assert mv_state.z.shape == y.shape


def copy_arrays(x: PyTree) -> PyTree:
    """Make a copy of the arrays in `x`, intended for buffer donation."""
    return tree.map(lambda x: jnp.array(x) if isinstance(x, jnp.ndarray) else x, x)


class TestStepTheta:
    """Test `step_theta`."""

    def state(
        self, keys: split, a: float, b: float, theta: float, p: int = 10
    ) -> State:
        """Create a state with variable selection and a fixed `log_s`.

        `theta` is the concentration `log_s` is drawn with, and `rho` is set to `p`.
        """
        n = 5
        state = init(
            X=jnp.zeros((p, n), jnp.uint8),
            y=jnp.zeros(n),
            offset=0.0,
            max_split=jnp.ones(p, jnp.uint8),
            num_trees=2,
            p_nonterminal=make_p_nonterminal(3),
            leaf_prior_cov_inv=Wishart(None, None, 2.0),
            error_cov_inv=Wishart(4.0, 4.0, 1.0),
            a=a,
            b=b,
            rho=float(p),
            sparse_on_at=0,
        )
        log_s = sample_s(keys.pop(), theta, p)
        return replace(
            state,
            forest=replace(
                state.forest, sparsity=replace(state.forest.sparsity, log_s=log_s)
            ),
        )

    @staticmethod
    def log_density(
        state: State, v: Float[numpy.ndarray, ' n']
    ) -> Float[numpy.ndarray, ' n']:
        """Compute the unnormalized log density of ``log(theta / rho)``, in float64."""
        forest = state.forest
        log_s = numpy.asarray(nnone(forest.sparsity.log_s), numpy.float64)
        log_s -= special.logsumexp(log_s)
        a, b = nnone(forest.sparsity.a).item(), nnone(forest.sparsity.b).item()
        rho = nnone(forest.sparsity.rho).item()
        (p,) = log_s.shape
        theta = rho * numpy.exp(v)
        return (
            -a * numpy.logaddexp(0, -v)
            - b * numpy.logaddexp(0, v)
            + special.gammaln(theta)
            - p * special.gammaln(theta / p)
            + theta / p * log_s.sum()
        )

    def posterior_cdf(self, state: State) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Compute the cdf of ``log(theta / rho)`` on a fine grid, in float64."""
        v = numpy.linspace(-80, 60, 2_000_001)
        log_density = self.log_density(state, v)
        cdf = numpy.cumsum(numpy.exp(log_density - log_density.max()))
        return v, cdf / cdf[-1]

    @staticmethod
    @jit
    def run(
        keys: Key[Array, ' draws'], state: State, v0: Float32[Array, '']
    ) -> Float32[Array, ' draws']:
        """Draw ``log(theta / rho)`` with `step_theta`, from ``theta = rho exp(v0)``."""
        forest = state.forest
        theta = nnone(forest.sparsity.rho) * jnp.exp(v0)
        state = replace(
            state,
            forest=replace(forest, sparsity=replace(forest.sparsity, theta=theta)),
        )

        def draw(key: Key[Array, '']) -> Float32[Array, '']:
            new_theta = nnone(step_theta(key, state).forest.sparsity.theta)
            return jnp.log(new_theta / nnone(forest.sparsity.rho))

        return vmap(draw)(keys)

    @pytest.mark.parametrize(('a', 'b'), [(0.5, 1.0), (20.0, 20.0), (0.2, 0.2)])
    @pytest.mark.parametrize('theta', [300.0, 3.0, 0.01])
    @pytest.mark.parametrize('p', [10, 1000])
    def test_posterior(
        self, keys: split, a: float, b: float, theta: float, p: int
    ) -> None:
        """Check the draws are distributed as the full conditional.

        The approximation of the full conditional is accurate only to within a
        bin of the grid, which is not detectable here with up to ~1000
        variables.
        """
        state = self.state(keys, a, b, theta, p)
        v, cdf = self.posterior_cdf(state)
        v1 = self.run(keys.pop(20_000), state, jnp.float32(0.0))
        cdf_func = partial(numpy.interp, xp=v, fp=cdf)
        assert ks_1samp(numpy.asarray(v1, numpy.float64), cdf_func).pvalue > 1e-3

    @pytest.mark.parametrize('c', [0.01, 5.0, 300.0])
    @pytest.mark.parametrize('p', [1000, 100_000])
    def test_log_density_precision(self, keys: split, c: float, p: int) -> None:
        """The float32 log density is accurate on the support of the posterior.

        The tolerance is about the float32 resolution of a log density of
        magnitude `p`. `c` is the concentration of each component of `s`.
        """
        state = self.state(keys, 0.5, 1.0, c * p, p)
        v, cdf = self.posterior_cdf(state)
        lo, hi = numpy.interp([1e-6, 1 - 1e-6], cdf, v)
        v = numpy.linspace(lo, hi, 1001)
        forest = state.forest
        log_density = _log_p_v(
            jnp.asarray(v, jnp.float32),
            log_s=nnone(forest.sparsity.log_s),
            a=nnone(forest.sparsity.a),
            b=nnone(forest.sparsity.b),
            rho=nnone(forest.sparsity.rho),
        )
        error = numpy.asarray(log_density, numpy.float64) - self.log_density(state, v)
        assert_allclose(error, error.mean(), atol=2e-6 * p, allow_non_scalar=True)

    def test_many_variables(self, keys: split) -> None:
        """With very many variables, `theta` keeps moving near its full conditional.

        The full conditional is narrower than a bin of the grid, which made a
        Metropolis-Hastings correction of the draw reject almost always.
        """
        state = self.state(keys, 1.0, 1.0, 3.0, 100_000)
        v, cdf = self.posterior_cdf(state)
        median = numpy.interp(0.5, cdf, v)
        v0 = jnp.float32(median)
        v1 = self.run(keys.pop(1000), state, v0)
        assert jnp.mean(v1 != v0) > 0.99
        width = 2 * 40.0 / 10_000  # the grid has the maximum number of bins
        assert abs(numpy.median(numpy.asarray(v1)) - median) < width

    def test_extreme_sparsity(self, keys: split) -> None:
        """A very sparse `s` makes the log density overflow, but not `theta`."""
        state = self.state(keys, 1.0, 1.0, 1.0)
        p = nnone(state.forest.sparsity.log_s).size
        log_s = jnp.full(p, -1e21).at[0].set(0.0)
        state = replace(
            state,
            forest=replace(
                state.forest, sparsity=replace(state.forest.sparsity, log_s=log_s)
            ),
        )
        v = self.run(keys.pop(100), state, jnp.float32(0.0))
        assert jnp.all(jnp.isfinite(v))


def test_stirling_remainder() -> None:
    """`stirling_remainder` is accurate in float32 over the whole range."""
    z = numpy.logspace(-30, 30, 1001).astype(numpy.float32)
    r = stirling_remainder(jnp.asarray(z))
    z = z.astype(numpy.float64)
    expected = (
        special.gammaln(z) - (z - 0.5) * numpy.log(z) + z - numpy.log(2 * numpy.pi) / 2
    )
    # the formula cancels catastrophically for large z also in float64
    large = z > 1e3
    expected[large] = 1 / (12 * z[large]) - 1 / (360 * z[large] ** 3)
    assert_allclose(r, expected, rtol=2e-6, allow_non_scalar=True)


class TestSampleSAugmentation:
    """Test `mcmcstep._step.sample_s_augmentation`.

    The sampler draws, for each variable, the number of attempted-but-discarded
    splits used to make the `log_s` full conditional exact in the presence of
    forbidden decision rules.
    """

    @staticmethod
    def _forest(
        var_tree: UInt[Array, ' num_trees half_tree_size'],
        split_tree: UInt[Array, ' num_trees half_tree_size'],
        max_split: UInt[Array, ' p'],
        log_s: Float32[Array, ' p'],
    ) -> Forest:
        """Build a forest exposing only the fields the sampler reads."""
        forest = _EmptyForest()
        object.__setattr__(forest, 'var_tree', var_tree)
        object.__setattr__(forest, 'split_tree', split_tree)
        object.__setattr__(forest, 'max_split', max_split)
        sparsity = Sparsity(
            log_s=log_s, theta=None, a=None, b=None, rho=None, log_theta_range=None
        )
        object.__setattr__(forest, 'sparsity', sparsity)
        return forest

    @staticmethod
    def _replicate(
        tree_heap: UInt[Array, ' half_tree_size'], num_trees: int
    ) -> UInt[Array, ' num_trees half_tree_size']:
        """Stack `num_trees` identical copies of a tree heap."""
        return jnp.broadcast_to(tree_heap, (num_trees, tree_heap.size))

    def test_no_selectable_variables(self, keys: split) -> None:
        """With no variable to split on, the counts are zero, not nan."""
        stumps = jnp.zeros((NUM_TREES, 4), jnp.uint8)
        forest = self._forest(
            stumps, stumps, jnp.zeros(3, jnp.uint8), jnp.array([0.0, -1.0, 1.0])
        )
        log_draws = sample_s_augmentation(keys.pop(), forest)
        assert_array_equal(jnp.exp(log_draws), jnp.zeros(3))

    def test_mean_matches_closed_form(self, keys: split) -> None:
        """A tree blocking one variable at one node gives the known Poisson mean."""
        # the root splits variable 0 at its lowest cutpoint, so its left child
        # can no longer split variable 0; variables 1 and 2 are never blocked
        tree_heap = manual_tree(
            [[0.0], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0], [1, 1]], [[1], [5, 5]]
        )
        s = jnp.array([0.5, 0.3, 0.2])
        forest = self._forest(
            self._replicate(tree_heap.var_tree, NUM_TREES),
            self._replicate(tree_heap.split_tree, NUM_TREES),
            jnp.array([5, 5, 5], jnp.uint8),
            jnp.log(s),
        )
        # each tree blocks variable 0 at one node with eligible mass 1 - s[0], so
        # the augmentation count of variable 0 is Poisson with mean below
        expected = NUM_TREES * s[0] / (1 - s[0])

        log_draws = jit(vmap(lambda key: sample_s_augmentation(key, forest)))(
            keys.pop(20_000)
        )
        draws = jnp.exp(log_draws)
        # variables 1 and 2 are never blocked, so their counts are exactly zero
        assert_array_equal(draws[:, 1:], jnp.zeros_like(draws[:, 1:]))
        mean = jnp.mean(draws[:, 0])
        sem = jnp.std(draws[:, 0]) / jnp.sqrt(draws.shape[0])
        assert abs(float(mean) - float(expected)) < 5 * float(sem)

    @staticmethod
    def _reference_blocked_mass_tree(
        key: Key[Array, ''],
        var_tree: UInt[Array, ' half_tree_size'],
        split_tree: UInt[Array, ' half_tree_size'],
        max_split: UInt[Array, ' p'],
        log_s: Float32[Array, ' p'],
    ) -> Float32[Array, ' p']:
        """Ancestor-list reference for `_blocked_mass_tree`.

        This is the straightforward per-node implementation: list the ineligible
        variables at each node, deduplicate them, sum the eligible split
        probabilities with a mask over all the variables, and accumulate the
        augmentation weights per variable in log space. Given the same key it draws the same
        per-node weights as `_blocked_mass_tree`, so the two must agree up to
        summation reassociation.
        """
        p = max_split.size
        (half_tree_size,) = split_tree.shape
        nodes = jnp.arange(half_tree_size)
        ineligible = vmap(fully_used_variables, in_axes=(None, None, None, 0))(
            var_tree, split_tree, max_split, nodes
        )
        *_, ncol = ineligible.shape
        equal = ineligible[:, :, None] == ineligible[:, None, :]
        earlier = jnp.tril(jnp.ones((ncol, ncol), bool), -1)
        seen_before = jnp.any(equal & earlier, axis=-1)
        is_first = (ineligible != p) & ~seen_before
        mask = jnp.any(ineligible[:, :, None] == jnp.arange(p), axis=1)
        log_eligible = logsumexp(jnp.where(mask, -jnp.inf, log_s), axis=-1)
        is_internal = split_tree.astype(bool)
        log_weight = jnp.log(random.exponential(key, (half_tree_size,))) - log_eligible
        log_weight = jnp.where(is_internal, log_weight, -jnp.inf)
        log_weights = jnp.where(is_first, log_weight[:, None], -jnp.inf).ravel()
        at_var = ineligible.ravel() == jnp.arange(p)[:, None]
        return logsumexp(jnp.where(at_var, log_weights, -jnp.inf), axis=1)

    @pytest.mark.parametrize('sparse', [False, True])
    def test_blocked_mass_matches_reference(self, keys: split, sparse: bool) -> None:
        """`_blocked_mass_tree` matches the ancestor-list reference to float.

        With a very sparse `s`, the eligible mass is tiny where the dominant
        variable is ineligible, and computing it as one minus the ineligible
        mass would cancel.
        """
        # Grow a forest tuned for heavy blocking: few cutpoints so variables
        # exhaust quickly, but enough trees and variables and a high grow
        # probability (low beta) so the trees stay deep below the exhausted
        # variables, which is what makes the blocked mass nonzero. Hence the
        # tree count is not `NUM_TREES`. This reliably blocks both child
        # sides across seeds (checked over 40 seeds: >= 5 blocked entries each).
        p, n = 6, 120
        state = init(
            X=random.randint(keys.pop(), (p, n), 0, 3, jnp.uint8),
            y=random.normal(keys.pop(), (n,)),
            offset=0.0,
            max_split=jnp.full(p, 2, jnp.uint8),
            num_trees=16,
            p_nonterminal=make_p_nonterminal(6, alpha=0.99, beta=0.5),
            leaf_prior_cov_inv=Wishart(None, None, 1.0),
            error_cov_inv=Wishart(nu=3.0, rate=1.0, value=3.0),
            theta=float(p),
            a=0.5,
            b=1.0,
            rho=float(p),
            sparse_on_at=0,
            augment=True,
        )
        for _ in range(40):
            state = step(keys.pop(), state)
        forest = state.forest
        (num_trees, half_tree_size) = forest.var_tree.shape
        d_minus_1 = half_tree_size.bit_length() - 1

        # get log_s
        log_s = nnone(forest.sparsity.log_s)
        if sparse:
            log_s = sample_s(keys.pop(), 0.01, p)

        # normalize log_s
        log_s -= logsumexp(log_s)  # all selectable

        # prepare arguments shared by the actual and reference implementations
        tree_keys = keys.pop(num_trees)
        kw: dict = dict(in_axes=(0, 0, 0, None, None))
        args = (tree_keys, forest.var_tree, forest.split_tree, forest.max_split)

        # invoke actual implementation
        s = S.from_log_s(log_s, jnp.ones(p, bool), 2 * d_minus_1)
        new = vmap(log_blocked_mass_tree, **kw)(*args, s)

        # invoke reference implementation
        ref = vmap(self._reference_blocked_mass_tree, **kw)(*args, log_s)

        # guard against a silently vacuous test: many variables must be blocked
        assert int(jnp.isfinite(new).sum()) >= 5
        assert_array_equal(jnp.isneginf(new), jnp.isneginf(ref))
        finite = jnp.isfinite(ref)
        assert_close_matrices(
            jnp.where(finite, new, 0.0), jnp.where(finite, ref, 0.0), atol=1e-4
        )

    def test_augment_noop_without_forbidden_rules(self, keys: split) -> None:
        """Without forbidden rules, `step_s` draws the same with or without it."""
        p, n = 4, 50
        state = init(
            X=random.randint(keys.pop(), (p, n), 0, 5, jnp.uint8),
            y=random.normal(keys.pop(), (n,)),
            offset=0.0,
            max_split=jnp.full(p, 4, jnp.uint8),
            num_trees=NUM_TREES,
            p_nonterminal=make_p_nonterminal(4),
            leaf_prior_cov_inv=Wishart(None, None, 1.0),
            error_cov_inv=Wishart(nu=3.0, rate=1.0, value=3.0),
            theta=float(p),
            a=0.5,
            b=1.0,
            rho=float(p),
            sparse_on_at=0,
            augment=True,
        )
        assert state.config.augment is True  # the flag flows into the step config

        # plant trees whose only internal node is the root, so no decision rule
        # is ever forbidden, whatever the cutpoint
        forest = state.forest
        (num_trees, half_tree_size) = forest.var_tree.shape
        root = jnp.zeros((num_trees, half_tree_size)).at[:, 1]
        forest = replace(
            forest,
            var_tree=root.set(0).astype(forest.var_tree.dtype),
            split_tree=root.set(2).astype(forest.split_tree.dtype),
        )
        state = replace(state, forest=forest)

        # the augmentation is then exactly zero
        assert jnp.all(jnp.isneginf(sample_s_augmentation(keys.pop(), forest)))

        # so the Dirichlet draw is identical with and without augmentation
        key = keys.pop()
        on = step_s(key, replace(state, config=replace(state.config, augment=True)))
        off = step_s(key, replace(state, config=replace(state.config, augment=False)))
        assert_array_equal(
            nnone(on.forest.sparsity.log_s), nnone(off.forest.sparsity.log_s)
        )
