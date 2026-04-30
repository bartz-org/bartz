# bartz/src/bartz/prepcovars.py
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

"""Functions and classes to preprocess data."""

from abc import abstractmethod
from functools import partial
from typing import Any, Protocol

from equinox import AbstractVar, Module
from jax import jit, random, vmap
from jax import numpy as jnp
from jaxtyping import Array, Float, Integer, Key, Real, UInt

from bartz.jaxext import autobatch, minimal_unsigned_dtype, unique


def parse_xinfo(
    xinfo: Float[Array, 'p m'],
) -> tuple[Float[Array, 'p m'], UInt[Array, ' p']]:
    """Parse pre-defined splits in the format of the R package BART.

    Parameters
    ----------
    xinfo
        A matrix with the cutpoins to use to bin each predictor. Each row shall
        contain a sorted list of cutpoints for a predictor. If there are less
        cutpoints than the number of columns in the matrix, fill the remaining
        cells with NaN.

        `xinfo` shall be a matrix even if `x_train` is a dataframe.

    Returns
    -------
    splits : Float[Array, 'p m']
        `xinfo` modified by replacing nan with a large value.
    max_split : UInt[Array, 'p']
        The number of non-nan elements in each row of `xinfo`.
    """
    is_not_nan = ~jnp.isnan(xinfo)
    max_split = jnp.sum(is_not_nan, axis=1)
    max_split = max_split.astype(minimal_unsigned_dtype(xinfo.shape[1]))
    huge = _huge_value(xinfo)
    splits = jnp.where(is_not_nan, xinfo, huge)
    return splits, max_split


@partial(jit, static_argnums=(2,))
def subsample(
    key: Key[Array, ''], X: Real[Array, 'p n'], max_samples: int
) -> Real[Array, 'p m']:
    """Randomly thin each predictor row to at most `max_samples` elements.

    Parameters
    ----------
    key
        A jax random key.
    X
        A matrix with `p` predictors and `n` observations.
    max_samples
        The target maximum number of samples per row.

    Returns
    -------
    A matrix with `p` rows and ``min(n, max_samples)`` columns. If
    ``n <= max_samples``, `X` is returned unchanged. Otherwise each row contains
    `max_samples` distinct values drawn without replacement from the
    corresponding row of `X`, with rows sampled independently. The order of
    values within each row is unspecified.

    Raises
    ------
    ValueError
        If `max_samples` is less than 1.
    """
    if max_samples < 1:
        msg = f'{max_samples=}, must be at least 1.'
        raise ValueError(msg)

    p, n = X.shape
    if n <= max_samples:
        return X

    keys = random.split(key, p)

    @partial(autobatch, max_io_nbytes=2**29)
    @vmap
    def per_row(k: Key[Array, ''], x: Real[Array, ' n']) -> Real[Array, ' m']:
        return random.choice(k, x, shape=(max_samples,), replace=False)

    return per_row(keys, X)


@partial(jit, static_argnums=(1,))
def quantilized_splits_from_matrix(
    X: Real[Array, 'p n'], max_bins: int
) -> tuple[Real[Array, 'p m'], UInt[Array, ' p']]:
    """
    Determine bins that make the distribution of each predictor uniform.

    Parameters
    ----------
    X
        A matrix with `p` predictors and `n` observations.
    max_bins
        The maximum number of bins to produce.

    Returns
    -------
    splits : Real[Array, 'p m']
        A matrix containing, for each predictor, the boundaries between bins.
        `m` is ``min(max_bins, n) - 1``, which is an upper bound on the number
        of splits. Each predictor may have a different number of splits; unused
        values at the end of each row are filled with the maximum value
        representable in the type of `X`.
    max_split : UInt[Array, ' p']
        The number of actually used values in each row of `splits`.

    Raises
    ------
    ValueError
        If `X` has no columns or if `max_bins` is less than 1.
    """
    out_length = min(max_bins, X.shape[1]) - 1

    if out_length < 0:
        msg = f'{X.shape[1]=} and {max_bins=}, they should be both at least 1.'
        raise ValueError(msg)

    @partial(autobatch, max_io_nbytes=2**29)
    def quantilize(
        X: Real[Array, 'p n'],
    ) -> tuple[Real[Array, 'p m'], UInt[Array, ' p']]:
        # wrap this function because autobatch needs traceable args
        return _quantilized_splits_from_matrix(X, out_length)

    return quantilize(X)


@partial(vmap, in_axes=(0, None))
def _quantilized_splits_from_matrix(
    x: Real[Array, 'p n'], out_length: int
) -> tuple[Real[Array, 'p m'], UInt[Array, ' p']]:
    # find the sorted unique values in x
    huge = _huge_value(x)
    u, actual_length = unique(x, size=x.size, fill_value=huge)

    # compute the midpoints between each unique value
    if jnp.issubdtype(x.dtype, jnp.integer):
        midpoints = u[:-1] + _ensure_unsigned(u[1:] - u[:-1]) // 2
    else:
        midpoints = u[:-1] + (u[1:] - u[:-1]) / 2
        # using x_i + (x_i+1 - x_i) / 2 instead of (x_i + x_i+1) / 2 is to
        # avoid overflow
    actual_length -= 1
    if midpoints.size:
        midpoints = midpoints.at[actual_length].set(huge)

    # take a subset of the midpoints if there are more than the requested maximum
    indices = jnp.linspace(-1, actual_length, out_length + 2)[1:-1]
    indices = jnp.around(indices).astype(minimal_unsigned_dtype(midpoints.size - 1))
    # indices calculation with float rather than int to avoid potential
    # overflow with int32, and to round to nearest instead of rounding down
    decimated_midpoints = midpoints[indices]
    truncated_midpoints = midpoints[:out_length]
    splits = jnp.where(
        actual_length > out_length, decimated_midpoints, truncated_midpoints
    )
    max_split = jnp.minimum(actual_length, out_length)
    max_split = max_split.astype(minimal_unsigned_dtype(out_length))
    return splits, max_split


def _huge_value(x: Array) -> int | float:
    """
    Return the maximum value that can be stored in `x`.

    Parameters
    ----------
    x
        A numerical numpy or jax array.

    Returns
    -------
    The maximum value allowed by `x`'s type (finite for floats).
    """
    if jnp.issubdtype(x.dtype, jnp.integer):
        return jnp.iinfo(x.dtype).max
    else:
        return float(jnp.finfo(x.dtype).max)


def _ensure_unsigned(x: Integer[Array, '*shape']) -> UInt[Array, '*shape']:
    """If x has signed integer type, cast it to the unsigned dtype of the same size."""
    return x.astype(_signed_to_unsigned(x.dtype))


def _signed_to_unsigned(int_dtype: jnp.dtype) -> jnp.dtype:
    """
    Map a signed integer type to its unsigned counterpart.

    Unsigned types are passed through.
    """
    assert jnp.issubdtype(int_dtype, jnp.integer)
    if jnp.issubdtype(int_dtype, jnp.unsignedinteger):
        return int_dtype
    match int_dtype:
        case jnp.int8:
            return jnp.uint8
        case jnp.int16:
            return jnp.uint16
        case jnp.int32:
            return jnp.uint32
        case jnp.int64:
            return jnp.uint64
        case _:
            msg = f'unexpected integer type {int_dtype}'
            raise TypeError(msg)


@partial(jit, static_argnums=(1,))
def uniform_splits_from_matrix(
    X: Real[Array, 'p n'], num_bins: int
) -> tuple[Real[Array, 'p m'], UInt[Array, ' p']]:
    """
    Make an evenly spaced binning grid.

    Parameters
    ----------
    X
        A matrix with `p` predictors and `n` observations.
    num_bins
        The number of bins to produce.

    Returns
    -------
    splits : Real[Array, 'p m']
        A matrix containing, for each predictor, the boundaries between bins.
        The excluded endpoints are the minimum and maximum value in each row of
        `X`.
    max_split : UInt[Array, ' p']
        The number of cutpoints in each row of `splits`, i.e., ``num_bins - 1``.
    """
    low = jnp.min(X, axis=1)
    high = jnp.max(X, axis=1)
    splits = jnp.linspace(low, high, num_bins + 1, axis=1)[:, 1:-1]
    assert splits.shape == (X.shape[0], num_bins - 1)
    max_split = jnp.full(*splits.shape, minimal_unsigned_dtype(num_bins - 1))
    return splits, max_split


@partial(jit, static_argnames=('method',))
def bin_predictors(
    X: Real[Array, 'p n'], splits: Real[Array, 'p m'], **kw: Any
) -> UInt[Array, 'p n']:
    """
    Bin the predictors according to the given splits.

    A value ``x`` is mapped to bin ``i`` iff ``splits[i - 1] < x <= splits[i]``.

    Parameters
    ----------
    X
        A matrix with `p` predictors and `n` observations.
    splits
        A matrix containing, for each predictor, the boundaries between bins.
        `m` is the maximum number of splits; each row may have shorter
        actual length, marked by padding unused locations at the end of the
        row with the maximum value allowed by the type.
    **kw
        Additional arguments are passed to `jax.numpy.searchsorted`.

    Returns
    -------
    `X` but with each value replaced by the index of the bin it falls into.
    """

    @partial(autobatch, max_io_nbytes=2**29)
    @vmap
    def bin_predictors(
        x: Real[Array, 'p n'], splits: Real[Array, 'p m']
    ) -> UInt[Array, 'p n']:
        dtype = minimal_unsigned_dtype(splits.size)
        return jnp.searchsorted(splits, x, **kw).astype(dtype)

    return bin_predictors(X, splits)


class Binner(Module):
    """Abstract base class for predictor binners.

    A binner inspects the training predictors at construction time and
    encapsulates the logic that maps any predictor matrix (training or
    test) to bin indices via `bin`. The number of cutpoints used per
    predictor is exposed as `max_split`.

    The constructor takes the training predictors and an optional random
    key. Concrete subclasses may add their own keyword arguments. Binners
    that do not use the key still accept it for protocol uniformity and
    silently ignore it. Binners that need it raise `ValueError` if it is
    not provided.
    """

    max_split: AbstractVar[UInt[Array, ' p']]

    @abstractmethod
    def __init__(
        self, X: Real[Array, 'p n'], *, key: Key[Array, ''] | None = None
    ) -> None: ...

    @abstractmethod
    def bin(self, X: Real[Array, 'p n']) -> UInt[Array, 'p n']:
        """Bin `X` according to the cutpoints determined at construction."""
        ...


class BinnerFactory(Protocol):
    """Callable that constructs a `Binner` from training predictors.

    This is the type of the `binner` argument of `bartz.Bart`. A bare
    `Binner` subclass satisfies this protocol, as does
    ``functools.partial(BinnerSubclass, **subclass_kwargs)``.
    """

    def __call__(
        self, X: Real[Array, 'p n'], *, key: Key[Array, ''] | None = None
    ) -> Binner:
        """Construct a `Binner` from `X` and an optional random key."""
        ...


class RangeEvenBinner(Binner):
    """Binner with cutpoints evenly spaced over the observed range.

    Parameters
    ----------
    X
        Training predictors with `p` predictors and `n` observations.
    max_bins
        The number of bins per predictor; ``max_bins - 1`` cutpoints are
        produced per predictor.
    key
        Accepted for protocol uniformity; unused.
    """

    _splits: Real[Array, 'p m']
    max_split: UInt[Array, ' p']

    def __init__(
        self,
        X: Real[Array, 'p n'],
        *,
        max_bins: int = 256,
        key: Key[Array, ''] | None = None,
    ) -> None:
        del key
        self._splits, self.max_split = uniform_splits_from_matrix(X, max_bins)

    def bin(self, X: Real[Array, 'p n']) -> UInt[Array, 'p n']:
        """Bin `X` with `bin_predictors` using the cutpoints chosen at construction."""
        return bin_predictors(X, self._splits)


class UniqueQuantileBinner(Binner):
    """Binner with quantile-based cutpoints from observed unique values.

    For each predictor, cutpoints are placed between sorted unique values
    so that the empirical distribution is approximately uniform across
    bins. The number of cutpoints is at most ``max_bins - 1`` and at most
    one less than the number of unique values.

    When ``n > max_subsample``, the predictor matrix is randomly thinned
    along the observation axis to ``max_subsample`` columns before
    quantilization, which keeps quantilization tractable on very large
    datasets at the cost of approximate quantiles.

    Note: the quantiles are on the unique values, not the original distribution.

    Parameters
    ----------
    X
        Training predictors with `p` predictors and `n` observations.
    max_bins
        The maximum number of bins per predictor.
    max_subsample
        The maximum number of observations to use when computing
        quantiles. If `None`, no subsampling is performed. If `n` exceeds
        this, `key` is required.
    key
        Random key for subsampling. Required when ``X.shape[1] >
        max_subsample``; otherwise unused.

    Raises
    ------
    ValueError
        If subsampling would trigger but `key` is `None`.
    """

    _splits: Real[Array, 'p m']
    max_split: UInt[Array, ' p']

    def __init__(
        self,
        X: Real[Array, 'p n'],
        *,
        max_bins: int = 256,
        max_subsample: int | None = 100_000,
        key: Key[Array, ''] | None = None,
    ) -> None:
        if max_subsample is not None and X.shape[1] > max_subsample:
            if key is None:
                msg = (
                    'UniqueQuantileBinner requires a `key` because '
                    f'n={X.shape[1]} exceeds max_subsample={max_subsample}.'
                )
                raise ValueError(msg)
            X = subsample(key, X, max_subsample)
        self._splits, self.max_split = quantilized_splits_from_matrix(X, max_bins)

    def bin(self, X: Real[Array, 'p n']) -> UInt[Array, 'p n']:
        """Bin `X` with `bin_predictors` using the cutpoints chosen at construction."""
        return bin_predictors(X, self._splits)


class GivenSplitsBinner(Binner):
    """Binner with cutpoints supplied directly in R BART `xinfo` format.

    Parameters
    ----------
    X
        Training predictors. Used only to validate the shape of `xinfo`.
    xinfo
        A `(p, m)` matrix of cutpoints; see `parse_xinfo`. NaN-padded
        entries are treated as unused.
    key
        Accepted for protocol uniformity; unused.

    Raises
    ------
    ValueError
        If `xinfo` is not 2D, or if its first dimension does not match
        ``X.shape[0]``.
    """

    _splits: Float[Array, 'p m']
    max_split: UInt[Array, ' p']

    def __init__(
        self,
        X: Real[Array, 'p n'],
        *,
        xinfo: Float[Array, 'p m'],
        key: Key[Array, ''] | None = None,
    ) -> None:
        del key
        if xinfo.ndim != 2 or xinfo.shape[0] != X.shape[0]:
            msg = f'{xinfo.shape=} different from expected ({X.shape[0]}, *)'
            raise ValueError(msg)
        self._splits, self.max_split = parse_xinfo(xinfo)

    def bin(self, X: Real[Array, 'p n']) -> UInt[Array, 'p n']:
        """Bin `X` with `bin_predictors` using the cutpoints chosen at construction."""
        return bin_predictors(X, self._splits)
