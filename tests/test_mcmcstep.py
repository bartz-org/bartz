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
from typing import NamedTuple

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
from jax.nn import softmax
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
from scipy import stats
from scipy.stats import chi2, ks_1samp, ks_2samp

from bartz._jaxext import (
    Module,
    field,
    get_default_device,
    get_default_devices,
    get_device_count,
    minimal_unsigned_dtype,
    split,
)
from bartz.grove import is_actual_leaf
from bartz.mcmcstep import (
    AutoBatchedReduction,
    AutoOneHotReduction,
    BatchedReduction,
    DiagWishart,
    OneHotReduction,
    PallasReduction,
    ReductionConfig,
    State,
    Wishart,
    init,
    make_p_nonterminal,
    step,
)
from bartz.mcmcstep._axes import chain_vmap_axes, data_vmap_axes, trace_sample_axes
from bartz.mcmcstep._moves import (
    ancestor_variables,
    fully_used_variables,
    randint_exclude,
    randint_masked,
    split_range,
)
from bartz.mcmcstep._reduction import _gpu_sm_count, _resolve_pallas_backend
from bartz.mcmcstep._state import (
    Forest,
    StepConfig,
    _inv_via_chol_with_gersh,
    _search_divisor,
    chol_with_gersh,
)
from bartz.mcmcstep._step import (
    _blocked_mass_tree,
    _compute_likelihood_ratio_mv,
    _compute_likelihood_ratio_uv,
    _precompute_leaf_terms_mv,
    _precompute_leaf_terms_uv,
    _precompute_likelihood_terms_mv,
    _precompute_likelihood_terms_uv,
    _sample_wishart_bartlett,
    _step_error_cov_inv_diag,
    _step_error_cov_inv_mv,
    sample_s_augmentation,
    step_error_cov_inv,
    step_s,
    step_trees,
    step_z,
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
)


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
    """Test `_search_divisor`."""

    def test_target_already_divides(self) -> None:
        """If `target_divisor` divides `dividend`, return it unchanged via the fast path."""
        assert _search_divisor(5, 100, 1, 100) == 5
        # target_divisor outside [low, up] is still returned when it divides
        assert _search_divisor(10, 100, 50, 80) == 10
        assert _search_divisor(1, 10, 5, 5) == 1

    def test_no_divisors_in_range_returns_target(self) -> None:
        """Fall back to `target_divisor` when no value in `[low, up]` divides `dividend`."""
        # 11 is prime; range [2, 10] contains no divisors of 11
        assert _search_divisor(3, 11, 2, 10) == 3
        # 17 is prime; range [2, 16] contains no divisors of 17
        assert _search_divisor(7, 17, 2, 16) == 7

    def test_finds_closest_divisor(self) -> None:
        """Return the divisor in `[low, up]` closest to `target_divisor`."""
        # divisors of 24: 1, 2, 3, 4, 6, 8, 12, 24
        # target=5 -> 4 and 6 tie at distance 1; argmin tie-break picks 4
        assert _search_divisor(5, 24, 1, 24) == 4
        # target=10 -> 8 and 12 tie at distance 2; tie-break picks 8
        assert _search_divisor(10, 24, 1, 24) == 8
        # target=20 -> 24 closer than 12
        assert _search_divisor(20, 24, 1, 24) == 24

    def test_range_restricts_candidates(self) -> None:
        """Only divisors within `[low, up]` are considered."""
        # divisors of 24 in [5, 7] -> only 6
        assert _search_divisor(10, 24, 5, 7) == 6
        # divisors of 24 in [7, 11] -> only 8
        assert _search_divisor(10, 24, 7, 11) == 8

    def test_single_point_range(self) -> None:
        """A `low == up` range either yields that value or the fallback."""
        # 6 divides 24
        assert _search_divisor(10, 24, 6, 6) == 6
        # 5 does not divide 24; no other candidates -> fallback to target
        assert _search_divisor(10, 24, 5, 5) == 10

    def test_target_equals_one(self) -> None:
        """`target_divisor == 1` always divides any dividend."""
        assert _search_divisor(1, 7, 2, 7) == 1
        assert _search_divisor(1, 100, 1, 100) == 1

    def test_target_equals_dividend(self) -> None:
        """`target_divisor == dividend` divides exactly."""
        assert _search_divisor(7, 7, 1, 7) == 7

    def test_ties_pick_lower(self) -> None:
        """Equidistant divisors are broken by picking the lower one (argmin convention)."""
        # divisors of 12 in [1, 12]: 1, 2, 3, 4, 6, 12
        # target=5 -> 4 and 6 tie at distance 1; tie-break picks 4
        assert _search_divisor(5, 12, 1, 12) == 4

    def test_return_type_is_python_int(self) -> None:
        """Returned values are Python ints, not numpy scalars."""
        assert isinstance(_search_divisor(5, 100, 1, 100), int)
        assert isinstance(_search_divisor(5, 24, 1, 24), int)
        assert isinstance(_search_divisor(3, 11, 2, 10), int)

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
            _search_divisor(target, dividend, low, up)


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
        """Configs covering every subclass and setting.

        Built in a fixture rather than at class-body (import) time so that
        `get_default_device` reads the platform after jax is configured (e.g.
        by the ``--platform`` option).
        """
        # PallasReduction backend: Triton on gpu, interpret mode on cpu (the
        # only mode that runs there)
        pallas_backend = 'triton' if get_default_device().platform == 'gpu' else 'cpu'
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
            # AutoOneHotReduction: per-site, per-platform method and layout
            AutoOneHotReduction(),
            # PallasReduction: fully automatic grid and tile, then explicit ones
            PallasReduction(backend=pallas_backend),
            PallasReduction(backend=pallas_backend, num_blocks=1, block_size=64),
            PallasReduction(backend=pallas_backend, num_blocks=8, block_size=16),
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
        and a data axis sharded with `shard_map`, which `PallasReduction` must
        reject.
        """
        n, size, num_trees = 512, 8, 4
        indices = random.randint(keys.pop(), (n,), 0, size).astype(jnp.uint32)
        tree_indices = random.randint(keys.pop(), (num_trees, n), 0, size).astype(
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
        # bin range: starts mid-domain, has a non-power-of-2 length (exercising
        # the bins padding `PallasReduction` needs for Triton), and runs one
        # past `size`, so its last bin is out of the index domain and its sum
        # must come out zero
        range_kw = dict(float_kw, subset_start=jnp.uint32(6), subset_length=3)
        exact = assert_array_equal  # the count path must match exactly
        # on gpu the matmul method contracts via tf32 (~1e-3 relative error); cpu
        # keeps the tight tolerance, so accuracy is still fully checked there
        rtol = 1e-3 if indices.platform() != 'cpu' else 1e-5  # ty: ignore[unresolved-attribute]
        close = partial(assert_close_matrices, rtol=rtol, atol=1e-6, reduce_rank=True)

        # each case invokes a reduce function `f` (a config's `_reduce` or the
        # reference) in one pattern, paired with the appropriate comparison
        cases = {
            'count': (lambda f: f(1, indices, **count_kw), exact),
            'float': (lambda f: f(vector, indices, **float_kw), close),
            'float batched': (lambda f: f(batched, indices, **float_kw), close),
            'float matrix': (lambda f: f(matrix, indices, **float_kw), close),
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
                    if name.startswith('sharded') and isinstance(
                        config, PallasReduction
                    ):
                        with pytest.raises(NotImplementedError):
                            run(config._reduce)
                    else:
                        compare(run(config._reduce), expected[name])

    def test_resolve_pallas_backend(self) -> None:
        """Only the 'triton' backend may yield compiler params."""
        assert _resolve_pallas_backend('cpu') is None
        assert _resolve_pallas_backend('default') is None
        assert _resolve_pallas_backend('triton') is not None

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
            'mv-continuous-vec-weights',
            'mv-mixed',
        ]
    )
    def init_kwargs(self, keys: split, request: pytest.FixtureRequest) -> dict:
        """Return arguments for `init`."""
        kind = request.param
        mv = kind.startswith('mv-')
        binary = kind.endswith('-binary')
        mixed = kind == 'mv-mixed'
        vec_weights = kind == 'mv-continuous-vec-weights'

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

        if vec_weights:
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
                '.forest.leaf_prior_cov_inv',
                '.forest.a',
                '.forest.b',
                '.forest.rho',
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
    num_trees = 5

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
        num_trees=num_trees,
        p_nonterminal=jnp.full(5, 0.9),
        leaf_prior_cov_inv=jnp.float32(num_trees),
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
    p, n, num_trees, num_steps = 5, 200, 20, 50
    data = gen_data(
        keys.pop(), n=n, p=p, q=0, sigma2_lin=1.0, sigma2_quad=1.0, sigma2_eps=1.0
    ).quantize()
    state = init(
        X=data.x,
        y=data.y,
        offset=0.0,
        max_split=data.max_split,
        num_trees=num_trees,
        p_nonterminal=make_p_nonterminal(6),
        leaf_prior_cov_inv=1.0,
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
        # stored in units of resid_scale)
        y = init_kwargs['y']
        offset = init_kwargs['offset']
        expected = y[1] - offset[1]
        data_units = state.resid[1] * state.resid_scale[1]
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


@pytest.fixture(params=[(10, 2), (20, 5), (3, 100), (50, 50)])
def mcmcstep_data_shape(request: FixtureRequest) -> tuple[int, int]:
    """Provide (n, p) pairs for testing."""
    return request.param


@pytest.fixture
def mcmcstep_data(mcmcstep_data_shape: tuple[int, int]) -> MCMCStepData:
    """Generate a toy dataset."""
    n, p = mcmcstep_data_shape
    X = jnp.arange(n * p, dtype=jnp.uint32).reshape(p, n)
    y = jnp.linspace(-1, 1, n)
    max_split = jnp.full(p, 5, dtype=jnp.uint32)
    return MCMCStepData(X, y, max_split)


def test_chol_with_gersh_disparate_scales() -> None:
    """Gershgorin stabilization is per-component, not a single global shift.

    A global shift set by the largest component would swamp the precision of
    much smaller ones, as when a mixed model heavily scales a continuous
    outcome alongside O(1) binary ones, corrupting the Cholesky and the
    inverse (and hence `leaf_scale`).
    """
    # diagonal precisions spanning 14 orders of magnitude
    precisions = jnp.array([1e-8, 1.0, 1e6])
    mat = jnp.diag(precisions)
    assert_close_matrices(
        chol_with_gersh(mat), jnp.diag(jnp.sqrt(precisions)), rtol=1e-5
    )
    assert_close_matrices(
        _inv_via_chol_with_gersh(mat), jnp.diag(1 / precisions), rtol=1e-5
    )


def test_weights_scale_invariance(keys: split, mcmcstep_data: MCMCStepData) -> None:
    """Rescaling `error_scale` by a constant only rescales the error variance.

    Rescaling all the weights by ``c`` is equivalent to rescaling the error
    variance by ``c ** 2``, so with the error precision prior and initial
    value adjusted accordingly, and ``c`` a power of two, the two MCMCs
    coincide exactly. This exercises the `State.inv_sdev_unit` normalization
    end-to-end: with float16 weight storage, an unnormalized
    ``1 / error_scale ** 2`` would overflow (small ``c``) or underflow (large
    ``c``) and the runs would diverge.
    """
    X, y, max_split = mcmcstep_data
    (n,) = y.shape
    w = jnp.exp(random.uniform(keys.pop(), (n,), float, -0.5, 0.5))

    def make_state(c: float) -> State:
        kw: dict = dict(
            X=X,
            y=y,
            offset=jnp.zeros(()),
            max_split=max_split,
            num_trees=10,
            p_nonterminal=jnp.array([0.9, 0.5]),
            leaf_prior_cov_inv=jnp.float32(10.0),
            error_cov_inv=Wishart(
                nu=jnp.float32(2.0),
                rate=jnp.float32(2.0 / c**2),
                value=jnp.float32(c**2),
            ),
            error_scale=c * w,
            prec_scale_dtype=jnp.float16,
            resid_reduction_config=BatchedReduction(num_batches=None),
            count_reduction_config=BatchedReduction(num_batches=None),
            prec_reduction_config=BatchedReduction(num_batches=None),
        )
        return init(**copy_arrays(kw))

    for c in (2.0**-9, 2.0**9):
        state_ref = make_state(1.0)
        state_scaled = make_state(c)

        # the unit absorbs the rescaling exactly, being a power of two, so the
        # stored weights are identical and O(1) in both runs
        assert state_scaled.prec_scale is not None
        assert state_scaled.prec_scale.dtype == jnp.float16
        assert jnp.all(jnp.isfinite(state_scaled.prec_scale))
        assert jnp.all(state_scaled.prec_scale > 0)
        assert_array_equal(nnone(state_ref.prec_scale), state_scaled.prec_scale)
        assert_array_equal(
            nnone(state_ref.inv_sdev_scale), nnone(state_scaled.inv_sdev_scale)
        )
        assert_array_equal(state_scaled.inv_sdev_unit, state_ref.inv_sdev_unit / c)

        for key in keys.pop(3):
            state_ref = step(key, state_ref)
            state_scaled = step(random.clone(key), state_scaled)

            assert_array_equal(
                state_ref.forest.split_tree, state_scaled.forest.split_tree
            )
            assert_array_equal(state_ref.forest.var_tree, state_scaled.forest.var_tree)
            assert_array_equal(
                state_ref.forest.leaf_tree, state_scaled.forest.leaf_tree
            )
            assert_array_equal(state_ref.resid, state_scaled.resid)
            assert_array_equal(
                state_scaled.error_cov_inv.value, c**2 * state_ref.error_cov_inv.value
            )


class TestWishart:
    """Test the basic properties of the wishart sampler output."""

    # Parameterize with (k, df) pairs
    @pytest.fixture(params=[(1, 3.0), (3, 3.0), (3, 5.0), (3, 100.0), (100, 102.0)])
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
        assert result.logdet_prec.shape == (num_trees, tree_size)

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
        num_trees, tree_size = 2, 3
        inv_sigma2 = random.uniform(keys.pop(), (), minval=0.1, maxval=5.0)
        leaf_prior_cov_inv_uv = random.uniform(keys.pop(), (), minval=0.1, maxval=5.0)

        error_cov_inv = jnp.array([[inv_sigma2]])
        leaf_prior_cov_inv = jnp.array([[leaf_prior_cov_inv_uv]])
        prec_trees = random.uniform(keys.pop(), (num_trees, tree_size)) * 5.0
        z_uv = random.normal(keys.pop(), (num_trees, tree_size))
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
            atol=1e-6,
        )
        assert_close_matrices(
            result_uv.centered_leaves,
            result_mv.centered_leaves.squeeze(1),
            rtol=1e-6,
            atol=1e-6,
        )
        # the posterior precision is error_cov_inv / mean_factor when k = 1
        assert_close_matrices(
            jnp.log(inv_sigma2 / result_uv.mean_factor),
            result_mv.logdet_prec,
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
                resid_reduction_config=BatchedReduction(num_batches=None),
                count_reduction_config=BatchedReduction(num_batches=None),
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
            uv_kw.update(error_cov_inv=Wishart(nu=6.0, rate=4.0, value=1.5))
            mv_kw.update(
                error_cov_inv=Wishart(
                    nu=jnp.array(6.0), rate=4.0 * jnp.eye(1), value=1.5 * jnp.eye(1)
                )
            )

        bart_uv = init(**uv_kw, **common())
        bart_mv = init(**mv_kw, **common())

        assert bart_uv.resid.ndim == 1
        assert bart_mv.resid.ndim == 2
        assert bart_mv.resid.shape[0] == 1
        assert bart_mv.resid.shape[1] == bart_uv.resid.shape[0]

        assert jnp.ndim(bart_uv.error_cov_inv.value) == 0
        assert bart_mv.error_cov_inv.value.shape == (1, 1)

        assert bart_uv.y.ndim == 1
        assert bart_mv.y.ndim == 2
        assert_array_equal(bart_uv.y, bart_mv.y.squeeze(0))

        if binary:
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
        Test that the univariate (diag) and multivariate (k = 1) samplers draw from the same posterior.

        UV: 1/sigma2 ~ Gamma(alpha_post, beta_post)
        MV: error_cov_inv ~ Wishart(df_post, scale_post)
        """
        X, y, _ = mcmcstep_data
        resid = random.normal(keys.pop(), (y.size,))

        # inverse gamma prior: alpha = df / 2, beta = scale / 2
        df_prior = jnp.float32(20.0)
        scale_prior = jnp.float32(10.0)

        common: dict = dict(
            _chain_anchor=jnp.zeros(()),
            X=X,
            y=y,
            binary_indices=None,
            z=None,
            prec_scale=None,
            inv_sdev_scale=None,
            inv_sdev_unit=jnp.ones(()),
            resid_scale=jnp.ones(()),  # unit scale: resid is in data units
            resid_eff_scale=jnp.ones(()),
            resid_inexact_integral=jnp.zeros(()),
            error_scale=None,
            n_non_missing=jnp.asarray(y.size),
            sum_diag_prec_scale=jnp.asarray(float(y.size)),
            forest=_EmptyForest(),
            config=_minimal_step_config(),
        )

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
        inv_sdev = jnp.where(mask, 0.0, 1.0)

        df_prior = jnp.float32(20.0)
        scale_prior = jnp.float32(10.0)
        # `X` is not read by the samplers, but its `n` axis is cross-checked
        # against `resid`, so the dropped states carry the subset `X[:, keep]`.
        # both the masked and dropped states see the same kept-point count and
        # (unweighted) precision sum
        n_kept = jnp.sum(keep)
        common: dict = dict(
            _chain_anchor=jnp.zeros(()),
            binary_indices=None,
            z=None,
            prec_scale=None,
            inv_sdev_unit=jnp.ones(()),
            resid_scale=jnp.ones(()),  # unit scale: resid is in data units
            resid_eff_scale=jnp.ones(()),
            resid_inexact_integral=jnp.zeros(()),
            error_scale=None,
            n_non_missing=n_kept,
            sum_diag_prec_scale=n_kept.astype(jnp.float32),
            forest=_EmptyForest(),
            config=_minimal_step_config(),
        )
        uv_prior = Wishart(nu=df_prior, rate=scale_prior, value=jnp.float32(1.0))
        mv_prior = Wishart(nu=df_prior, rate=scale_prior[None, None], value=jnp.eye(1))

        st_uv_with = State(
            **common,
            X=X,
            y=resid_1d,
            resid=resid_1d,
            inv_sdev_scale=inv_sdev,
            error_cov_inv=uv_prior,
        )
        st_uv_drop = State(
            **common,
            X=X[:, keep],
            y=resid_1d[keep],
            resid=resid_1d[keep],
            inv_sdev_scale=None,
            error_cov_inv=uv_prior,
        )

        key = keys.pop()
        sample_with = _step_error_cov_inv_diag(key, st_uv_with).error_cov_inv.value
        sample_drop = _step_error_cov_inv_diag(key, st_uv_drop).error_cov_inv.value
        assert_allclose(sample_with, sample_drop, rtol=1e-6)

        # multivariate Wishart path (k=1) with 1-D inv_sdev_scale and zeros
        st_mv_with = State(
            **common,
            X=X,
            y=resid_1d,
            resid=resid_1d[None, :],
            inv_sdev_scale=inv_sdev,
            error_cov_inv=mv_prior,
        )
        st_mv_drop = State(
            **common,
            X=X[:, keep],
            y=resid_1d[keep],
            resid=resid_1d[None, keep],
            inv_sdev_scale=None,
            error_cov_inv=mv_prior,
        )
        sample_mv_with = _step_error_cov_inv_mv(key, st_mv_with).error_cov_inv.value
        sample_mv_drop = _step_error_cov_inv_mv(key, st_mv_drop).error_cov_inv.value
        assert_allclose(sample_mv_with, sample_mv_drop, rtol=1e-6)


class TestMultivariate:
    """Test for multivariate outcomes specifically."""

    @pytest.mark.parametrize('kind', ['binary', 'homo', 'het'])
    def test_1d_mv_matches_uv(
        self, keys: split, mcmcstep_data: MCMCStepData, kind: str
    ) -> None:
        """Check that multivariate with k=1 is equivalent to univariate."""
        X, y, max_split = mcmcstep_data
        n_trees = 100

        if kind == 'binary':
            y = (y > 0).astype(jnp.float32)

        params = partial(
            copy_arrays,
            dict(
                X=X,
                max_split=max_split,
                num_trees=n_trees,
                p_nonterminal=jnp.array([0.9, 0.5]),
                resid_reduction_config=BatchedReduction(num_batches=None),
                count_reduction_config=BatchedReduction(num_batches=None),
            ),
        )

        uv_kw: dict = dict(y=y, offset=0.0, leaf_prior_cov_inv=jnp.float32(n_trees))
        mv_kw: dict = dict(
            y=y[None, :], offset=jnp.zeros(1), leaf_prior_cov_inv=n_trees * jnp.eye(1)
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
            rtol = condf(uv_state.forest.leaf_tree, 1e-6, 1e-3)
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
            num_trees=5,
            p_nonterminal=jnp.array([0.9, 0.5]),
            leaf_prior_cov_inv=jnp.eye(k),
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

    @pytest.mark.parametrize('k', [1, 3])
    def test_mv_het_vector_equiv_scalar(
        self, keys: split, mcmcstep_data: MCMCStepData, k: int
    ) -> None:
        """Constant vector weights equal scalar weights."""
        X, y_uv, max_split = mcmcstep_data
        n = y_uv.size

        y = jnp.tile(y_uv, (k, 1))
        y = y + random.normal(keys.pop(), y.shape) * 0.1

        w_scalar = jnp.exp(random.uniform(keys.pop(), (n,), float, -0.5, 0.5))
        w_vector = jnp.broadcast_to(w_scalar, (k, n))

        params = partial(
            copy_arrays,
            dict(
                X=X,
                y=y,
                offset=jnp.zeros(k),
                max_split=max_split,
                num_trees=10,
                p_nonterminal=jnp.array([0.9, 0.5]),
                leaf_prior_cov_inv=jnp.eye(k),
                error_cov_inv=Wishart(
                    nu=jnp.array(4.0 + k), rate=jnp.eye(k), value=(4.0 + k) * jnp.eye(k)
                ),
                resid_reduction_config=BatchedReduction(num_batches=None),
                count_reduction_config=BatchedReduction(num_batches=None),
                # this checks the scalar- and vector-weight code paths agree; the
                # scalar (n,) and vector (k, k, n) prec_scale carry the same values
                # but float16 storage perturbs the two paths' linear algebra
                # differently, so store in float32 to test the path equivalence alone
                prec_scale_dtype=jnp.float32,
            ),
        )

        scalar_state = init(error_scale=w_scalar, **params())
        vector_state = init(error_scale=w_vector, **params())

        assert scalar_state.prec_scale is not None
        assert scalar_state.prec_scale.shape == (n,)
        assert vector_state.prec_scale is not None
        assert vector_state.prec_scale.shape == (k, k, n)

        # scalar weights produce 1-D ``inv_sdev_scale`` (Wishart error cov update)
        # while vector weights produce 2-D ``inv_sdev_scale`` (diagonal update),
        # so the error_cov_inv updates are not equivalent, we can't use the full `step`.
        for _ in range(3):
            key = keys.pop()
            scalar_state = step_trees(key, scalar_state)
            vector_state = step_trees(random.clone(key), vector_state)

            assert_close_matrices(scalar_state.resid, vector_state.resid, rtol=1e-5)
            assert_close_matrices(
                scalar_state.forest.leaf_tree,
                vector_state.forest.leaf_tree,
                rtol=3e-5,
                reduce_rank=True,
            )


def copy_arrays(x: PyTree) -> PyTree:
    """Make a copy of the arrays in `x`, intended for buffer donation."""
    return tree.map(lambda x: jnp.array(x) if isinstance(x, jnp.ndarray) else x, x)


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
        object.__setattr__(forest, 'log_s', log_s)
        return forest

    @staticmethod
    def _replicate(
        tree_heap: UInt[Array, ' half_tree_size'], num_trees: int
    ) -> UInt[Array, ' num_trees half_tree_size']:
        """Stack `num_trees` identical copies of a tree heap."""
        return jnp.broadcast_to(tree_heap, (num_trees, tree_heap.size))

    def test_mean_matches_closed_form(self, keys: split) -> None:
        """A tree blocking one variable at one node gives the known Poisson mean."""
        # the root splits variable 0 at its lowest cutpoint, so its left child
        # can no longer split variable 0; variables 1 and 2 are never blocked
        tree_heap = manual_tree(
            [[0.0], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0], [1, 1]], [[1], [5, 5]]
        )
        num_trees = 4
        s = jnp.array([0.5, 0.3, 0.2])
        forest = self._forest(
            self._replicate(tree_heap.var_tree, num_trees),
            self._replicate(tree_heap.split_tree, num_trees),
            jnp.array([5, 5, 5], jnp.uint8),
            jnp.log(s),
        )
        # each tree blocks variable 0 at one node with eligible mass 1 - s[0], so
        # the augmentation count of variable 0 is Poisson with mean below
        expected = num_trees * s[0] / (1 - s[0])

        draws = jit(vmap(lambda key: sample_s_augmentation(key, forest)))(
            keys.pop(20_000)
        )
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
        s: Float32[Array, ' p'],
    ) -> Float32[Array, ' p']:
        """Ancestor-list reference for `_blocked_mass_tree`.

        This is the straightforward per-node implementation: list the ineligible
        variables at each node, deduplicate them, and scatter the augmentation
        weight directly. Given the same key it draws the same per-node weights as
        `_blocked_mass_tree`, so the two must agree up to summation reassociation.
        """
        p = max_split.size
        (half_tree_size,) = split_tree.shape
        s_padded = jnp.append(s, 0.0)
        nodes = jnp.arange(half_tree_size)
        ineligible = vmap(fully_used_variables, in_axes=(None, None, None, 0))(
            var_tree, split_tree, max_split, nodes
        )
        *_, ncol = ineligible.shape
        equal = ineligible[:, :, None] == ineligible[:, None, :]
        earlier = jnp.tril(jnp.ones((ncol, ncol), bool), -1)
        seen_before = jnp.any(equal & earlier, axis=-1)
        is_first = (ineligible != p) & ~seen_before
        ineligible_mass = jnp.sum(s_padded[ineligible] * is_first, axis=-1)
        eligible_mass = jnp.maximum(1.0 - ineligible_mass, jnp.finfo(jnp.float32).eps)
        is_internal = split_tree.astype(bool)
        weight = jnp.where(is_internal, random.exponential(key, (half_tree_size,)), 0.0)
        weight /= eligible_mass
        weights = is_first * weight[:, None]
        return jnp.zeros(p + 1).at[ineligible.ravel()].add(weights.ravel())[:p]

    def test_blocked_mass_matches_reference(self, keys: split) -> None:
        """`_blocked_mass_tree` matches the ancestor-list reference to float."""
        # Grow a forest tuned for heavy blocking: few cutpoints so variables
        # exhaust quickly, but enough variables and a high grow probability (low
        # beta) so the trees stay deep below the exhausted variables, which is
        # what makes the blocked mass nonzero. This reliably blocks both child
        # sides across seeds (checked over 40 seeds: >= 5 blocked entries each).
        p, n = 6, 120
        state = init(
            X=random.randint(keys.pop(), (p, n), 0, 3, jnp.uint8),
            y=random.normal(keys.pop(), (n,)),
            offset=0.0,
            max_split=jnp.full(p, 2, jnp.uint8),
            num_trees=16,
            p_nonterminal=make_p_nonterminal(6, alpha=0.99, beta=0.5),
            leaf_prior_cov_inv=1.0,
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
        (num_trees, _) = forest.var_tree.shape

        s = softmax(nnone(forest.log_s))  # all selectable
        tree_keys = keys.pop(num_trees)
        kw = dict(in_axes=(0, 0, 0, None, None))
        args = (tree_keys, forest.var_tree, forest.split_tree, forest.max_split, s)
        new = vmap(_blocked_mass_tree, **kw)(*args)
        ref = vmap(self._reference_blocked_mass_tree, **kw)(*args)

        # guard against a silently vacuous test: many variables must be blocked
        assert int((new > 0).sum()) >= 5
        assert_close_matrices(new, ref, rtol=1e-5)

    def test_augment_noop_without_forbidden_rules(self, keys: split) -> None:
        """Without forbidden rules, `step_s` draws the same with or without it."""
        p, n = 4, 50
        state = init(
            X=random.randint(keys.pop(), (p, n), 0, 5, jnp.uint8),
            y=random.normal(keys.pop(), (n,)),
            offset=0.0,
            max_split=jnp.full(p, 4, jnp.uint8),
            num_trees=6,
            p_nonterminal=make_p_nonterminal(4),
            leaf_prior_cov_inv=1.0,
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
        assert jnp.all(sample_s_augmentation(keys.pop(), forest) == 0)

        # so the Dirichlet draw is identical with and without augmentation
        key = keys.pop()
        on = step_s(key, replace(state, config=replace(state.config, augment=True)))
        off = step_s(key, replace(state, config=replace(state.config, augment=False)))
        assert_array_equal(nnone(on.forest.log_s), nnone(off.forest.log_s))
