# bartz/tests/test_prepcovars.py
#
# Copyright (c) 2024-2026, The Bartz Contributors
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

"""Test the `bartz.prepcovars` module."""

import pytest
from jax import debug_infs, random
from jax import numpy as jnp
from numpy.testing import assert_array_equal

from bartz.jaxext import split
from bartz.prepcovars import (
    Binner,
    GivenSplitsBinner,
    RangeEvenBinner,
    UniqueQuantileBinner,
    bin_predictors,
    parse_xinfo,
    quantilized_splits_from_matrix,
    subsample,
    uniform_splits_from_matrix,
)


class TestQuantilizer:
    """Test `prepcovars.quantilized_splits_from_matrix`."""

    @pytest.mark.parametrize(
        'fill_value', [jnp.finfo(jnp.float32).max, jnp.iinfo(jnp.int32).max]
    )
    def test_splits_fill(self, fill_value: float | int) -> None:
        """Check how predictors with less unique values are right-padded."""
        with debug_infs(not jnp.isinf(fill_value)):
            fill_value = jnp.array(fill_value)
            x = jnp.array([[1, 1, 3, 3], [1, 3, 3, 5], [1, 3, 5, 7]], fill_value.dtype)
            splits, _ = quantilized_splits_from_matrix(x, 100)
        expected_splits = [[2, fill_value, fill_value], [2, 4, fill_value], [2, 4, 6]]
        assert_array_equal(splits, expected_splits)

    def test_max_splits(self) -> None:
        """Check that the number of splits per predictor is counted correctly."""
        x = jnp.array([[1, 1, 1, 1], [4, 4, 1, 1], [2, 1, 3, 2], [1, 4, 2, 3]])
        _, max_split = quantilized_splits_from_matrix(x, 100)
        assert_array_equal(max_split, jnp.arange(4))

    def test_integer_splits_overflow(self) -> None:
        """Check that the splits are computed correctly at the limit of overflow."""
        x = jnp.array([[-(2**31), 2**31 - 2]])
        splits, _ = quantilized_splits_from_matrix(x, 100)
        expected_splits = [[-1]]
        assert_array_equal(splits, expected_splits)

    @pytest.mark.parametrize('dtype', [int, float])
    def test_splits_type(self, dtype: type) -> None:
        """Check that the input type is preserved."""
        x = jnp.arange(10, dtype=dtype)[None, :]
        splits, _ = quantilized_splits_from_matrix(x, 100)
        assert splits.dtype == x.dtype

    def test_splits_length(self) -> None:
        """Check that the correct number of splits is returned in corner cases."""
        x = jnp.linspace(0, 1, 10)[None, :]

        short_splits, _ = quantilized_splits_from_matrix(x, 2)
        assert short_splits.shape == (1, 1)

        long_splits, _ = quantilized_splits_from_matrix(x, 100)
        assert long_splits.shape == (1, 9)

        just_right_splits, _ = quantilized_splits_from_matrix(x, 10)
        assert just_right_splits.shape == (1, 9)

        no_splits, _ = quantilized_splits_from_matrix(x, 1)
        assert no_splits.shape == (1, 0)

    def test_round_trip(self) -> None:
        """Check that `bin_predictors` is the ~inverse of `quantilized_splits_from_matrix`."""
        x = jnp.arange(10)[None, :]
        splits, _ = quantilized_splits_from_matrix(x, 100)
        b = bin_predictors(x, splits)
        assert_array_equal(x, b)

    def test_one_value(self) -> None:
        """Check there's only 1 bin (0 splits) if there is 1 datapoint."""
        x = jnp.arange(10)[:, None]
        _, max_split = quantilized_splits_from_matrix(x, 100)
        assert_array_equal(max_split, jnp.full(len(x), 0))

    def test_zero_values(self) -> None:
        """Check what happens when no binning is possible."""
        x = jnp.empty((1, 0))
        with pytest.raises(ValueError, match='at least 1'):
            quantilized_splits_from_matrix(x, 100)

    def test_zero_bins(self) -> None:
        """Check what happens when no binning is possible."""
        x = jnp.arange(10)[None, :]
        with pytest.raises(ValueError, match='at least 1'):
            quantilized_splits_from_matrix(x, 0)


class TestSubsample:
    """Test `prepcovars.subsample`."""

    def test_shape_and_subset(self, keys: split) -> None:
        """When n > max_samples, output has shape (p, max_samples) and values come from the matching row."""
        p, n, max_samples = 3, 100, 10
        x = jnp.arange(p * n).reshape(p, n)
        out = subsample(keys.pop(), x, max_samples)
        assert out.shape == (p, max_samples)
        for i in range(p):
            assert jnp.all(jnp.isin(out[i], x[i]))

    def test_no_replacement(self, keys: split) -> None:
        """Each output row contains distinct values (no replacement) when input row is distinct."""
        p, n, max_samples = 4, 50, 20
        x = jnp.arange(p * n).reshape(p, n)
        out = subsample(keys.pop(), x, max_samples)
        for i in range(p):
            assert jnp.unique(out[i]).size == max_samples

    def test_n_equals_max_samples(self, keys: split) -> None:
        """When n == max_samples, X is returned unchanged."""
        x = jnp.arange(12.0).reshape(3, 4)
        out = subsample(keys.pop(), x, 4)
        assert_array_equal(out, x, strict=True)

    def test_n_less_than_max_samples(self, keys: split) -> None:
        """When n < max_samples, X is returned unchanged."""
        x = jnp.arange(12.0).reshape(3, 4)
        out = subsample(keys.pop(), x, 100)
        assert out.shape == x.shape
        assert_array_equal(out, x, strict=True)

    @pytest.mark.parametrize('dtype', [jnp.float32, jnp.int32])
    def test_dtype_preservation(self, keys: split, dtype: jnp.dtype) -> None:
        """Output dtype matches input dtype."""
        x = jnp.arange(60, dtype=dtype).reshape(3, 20)
        out = subsample(keys.pop(), x, 5)
        assert out.dtype == x.dtype

    def test_per_row_independence(self, keys: split) -> None:
        """Rows are sampled independently (output values stay within the originating row)."""
        # Use disjoint value ranges per row so we can detect cross-row leakage.
        n, max_samples = 40, 8
        row0 = jnp.arange(0, n)
        row1 = jnp.arange(1000, 1000 + n)
        row2 = jnp.arange(2000, 2000 + n)
        x = jnp.stack([row0, row1, row2])
        out = subsample(keys.pop(), x, max_samples)
        assert jnp.all((out[0] >= 0) & (out[0] < n))
        assert jnp.all((out[1] >= 1000) & (out[1] < 1000 + n))
        assert jnp.all((out[2] >= 2000) & (out[2] < 2000 + n))
        # Different rows should not all pick the same column indices: convert
        # back to indices within each row and check they differ across rows.
        idx0 = out[0]
        idx1 = out[1] - 1000
        assert not jnp.array_equal(jnp.sort(idx0), jnp.sort(idx1))

    def test_determinism(self, keys: split) -> None:
        """Calling subsample twice with the same key gives identical output."""
        key = keys.pop()
        x = jnp.arange(60.0).reshape(3, 20)
        out1 = subsample(key, x, 5)
        out2 = subsample(random.clone(key), x, 5)
        assert_array_equal(out1, out2, strict=True)

    def test_different_keys_give_different_output(self, keys: split) -> None:
        """Two distinct keys produce different subsamples (collision negligible at this size)."""
        x = jnp.arange(200)[None, :]
        out1 = subsample(keys.pop(), x, 20)
        out2 = subsample(keys.pop(), x, 20)
        assert not jnp.array_equal(out1, out2)

    def test_max_samples_one(self, keys: split) -> None:
        """max_samples == 1 yields shape (p, 1) with each value drawn from its row."""
        p, n = 3, 7
        x = jnp.arange(p * n).reshape(p, n)
        out = subsample(keys.pop(), x, 1)
        assert out.shape == (p, 1)
        for i in range(p):
            assert jnp.isin(out[i, 0], x[i])

    def test_max_samples_zero_raises(self, keys: split) -> None:
        """max_samples < 1 raises ValueError."""
        x = jnp.arange(20.0).reshape(2, 10)
        with pytest.raises(ValueError, match='at least 1'):
            subsample(keys.pop(), x, 0)

    def test_n_zero(self, keys: split) -> None:
        """An (p, 0) input is returned unchanged regardless of max_samples."""
        x = jnp.empty((3, 0))
        out = subsample(keys.pop(), x, 5)
        assert out.shape == (3, 0)
        assert_array_equal(out, x, strict=True)

    def test_p_one(self, keys: split) -> None:
        """Single-row matrix works."""
        x = jnp.arange(50)[None, :]
        out = subsample(keys.pop(), x, 5)
        assert out.shape == (1, 5)
        assert jnp.all(jnp.isin(out[0], x[0]))


class TestBinners:
    """Test the `Binner` subclasses."""

    def test_base_class_is_abstract(self) -> None:
        """The base `Binner` class cannot be instantiated directly."""
        x = jnp.arange(8.0).reshape(2, 4)
        with pytest.raises(TypeError, match='abstract'):
            Binner(x)  # type: ignore[abstract]

    def test_range_even_matches_underlying(self) -> None:
        """`RangeEvenBinner` is consistent with `uniform_splits_from_matrix`."""
        x = jnp.linspace(-1, 1, 24).reshape(3, 8)
        binner = RangeEvenBinner(x, max_bins=8)
        ref_splits, ref_max_split = uniform_splits_from_matrix(x, 8)
        assert_array_equal(binner._splits, ref_splits)
        assert_array_equal(binner.max_split, ref_max_split)
        assert_array_equal(binner.bin(x), bin_predictors(x, ref_splits))

    def test_range_even_ignores_key(self, keys: split) -> None:
        """`RangeEvenBinner` accepts a `key` and produces the same output."""
        x = jnp.linspace(-1, 1, 24).reshape(3, 8)
        without = RangeEvenBinner(x, max_bins=8)
        withk = RangeEvenBinner(x, max_bins=8, key=keys.pop())
        assert_array_equal(withk._splits, without._splits)
        assert_array_equal(withk.max_split, without.max_split)

    def test_unique_quantile_no_subsample(self) -> None:
        """With `max_subsample=None`, output matches `quantilized_splits_from_matrix`."""
        x = jnp.tile(jnp.arange(10.0), (3, 1))
        binner = UniqueQuantileBinner(x, max_bins=8, max_subsample=None)
        ref_splits, ref_max_split = quantilized_splits_from_matrix(x, 8)
        assert_array_equal(binner._splits, ref_splits)
        assert_array_equal(binner.max_split, ref_max_split)
        assert_array_equal(binner.bin(x), bin_predictors(x, ref_splits))

    def test_unique_quantile_no_subsample_does_not_need_key(self) -> None:
        """When `n <= max_subsample`, no `key` is required."""
        x = jnp.tile(jnp.arange(10.0), (3, 1))
        # n=10, max_subsample=100 -> no subsampling, key=None is fine
        binner = UniqueQuantileBinner(x, max_bins=8, max_subsample=100)
        assert binner.max_split.shape == (3,)

    def test_unique_quantile_requires_key_when_subsampling(self) -> None:
        """`UniqueQuantileBinner` raises if `n > max_subsample` and `key=None`."""
        x = jnp.tile(jnp.arange(10.0), (3, 1))
        with pytest.raises(ValueError, match='requires a `key`'):
            UniqueQuantileBinner(x, max_bins=8, max_subsample=4)

    def test_unique_quantile_subsamples_with_key(self, keys: split) -> None:
        """When subsampling triggers, the binner caps `max_split` and bins data correctly."""
        x = jnp.tile(jnp.arange(20.0), (2, 1))
        max_sub = 5
        binner = UniqueQuantileBinner(
            x, max_bins=64, max_subsample=max_sub, key=keys.pop()
        )
        # at most max_sub - 1 cutpoints once subsampling has taken place
        assert jnp.all(binner.max_split <= max_sub - 1)
        # the binner can still bin its training input without falling off the
        # end of the splits array (no out-of-range bin indices)
        bins = binner.bin(x)
        assert bins.shape == x.shape
        assert jnp.all(bins <= binner.max_split[:, None])

    def test_given_splits_matches_parse_xinfo(self) -> None:
        """`GivenSplitsBinner` matches `parse_xinfo` semantics."""
        xinfo = jnp.array([[1.1, 2.3, jnp.nan], [-50.0, 10.0, 20.0]], dtype=jnp.float32)
        x = jnp.zeros((2, 5), dtype=jnp.float32)
        binner = GivenSplitsBinner(x, xinfo=xinfo)
        ref_splits, ref_max_split = parse_xinfo(xinfo)
        assert_array_equal(binner._splits, ref_splits)
        assert_array_equal(binner.max_split, ref_max_split)

    def test_given_splits_wrong_p_raises(self) -> None:
        """Mismatched `xinfo.shape[0]` vs `X.shape[0]` raises ValueError."""
        xinfo = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        x = jnp.zeros((5, 0), dtype=jnp.float32)
        with pytest.raises(ValueError, match=r'xinfo\.shape'):
            GivenSplitsBinner(x, xinfo=xinfo)


def test_binner_left_boundary() -> None:
    """Check that the first bin is right-closed."""
    splits = jnp.array([[1, 2, 3]])

    x = jnp.array([[0, 1]])
    b = bin_predictors(x, splits)
    assert_array_equal(b, [[0, 0]])


def test_binner_right_boundary() -> None:
    """Check that the next-to-last bin is right-closed."""
    splits = jnp.array([[1, 2, 3, 2**31 - 1]])

    x = jnp.array([[2**31 - 1]])
    b = bin_predictors(x, splits)
    assert_array_equal(b, [[3]])
