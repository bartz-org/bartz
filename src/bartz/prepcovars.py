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

"""Functions to preprocess data."""

from functools import partial
from typing import Any

from jax import jit, vmap
from jax import numpy as jnp
from jaxtyping import Array, Float, Integer, Real, UInt

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
