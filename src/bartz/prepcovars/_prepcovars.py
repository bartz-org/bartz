# bartz/src/bartz/prepcovars/_prepcovars.py
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

"""Implementation of the predictor preprocessing utilities."""

from abc import abstractmethod
from functools import partial
from typing import Any, Protocol, runtime_checkable

from equinox import AbstractVar, Module, field
from jax import numpy as jnp
from jax import random, vmap
from jax.scipy.sparse.linalg import cg
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Float32, Integer, Key, Real, Shaped, UInt

from bartz._jaxext import autobatch, jit, minimal_unsigned_dtype, unique


def _parse_xinfo(
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


@jit(static_argnums=(2,))
def _subsample(
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
    A matrix with `p` rows and ``min(n, max_samples)`` columns. If ``n <= max_samples``, `X` is returned unchanged. Otherwise each row contains `max_samples` distinct values drawn without replacement from the corresponding row of `X`, with rows sampled independently. The order of values within each row is unspecified.

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


@jit(static_argnums=(1,))
def _quantilized_splits_from_matrix(
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
        return _quantilized_splits_from_vector(X, out_length)

    return quantilize(X)


@partial(vmap, in_axes=(0, None))
def _quantilized_splits_from_vector(
    x: Real[Array, ' n'], out_length: int
) -> tuple[Real[Array, ' m'], UInt[Array, '']]:
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


def _signed_to_unsigned(int_dtype: DTypeLike) -> DTypeLike:
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


@jit(static_argnums=(1,))
def _uniform_splits_from_matrix(
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
    splits = _uniform_splits_from_range(low, high, num_bins)
    assert splits.shape == (X.shape[0], num_bins - 1)
    max_split = jnp.full(*splits.shape, minimal_unsigned_dtype(num_bins - 1))
    return splits, max_split


@jit(static_argnums=(2,))
def _uniform_splits_from_range(
    low: Real[Array, ' p'], high: Real[Array, ' p'], num_bins: int
) -> Real[Array, 'p m']:
    """
    Make an evenly spaced binning grid from per-predictor ranges.

    Parameters
    ----------
    low
        The lower endpoint of the grid for each predictor.
    high
        The upper endpoint of the grid for each predictor.
    num_bins
        The number of bins to produce.

    Returns
    -------
    A `(p, num_bins - 1)` matrix of cutpoints, with `low` and `high` excluded.
    """
    splits = jnp.linspace(low, high, num_bins + 1, axis=1)[:, 1:-1]
    (p,) = low.shape
    assert splits.shape == (p, num_bins - 1)
    return splits


@jit(static_argnums=(3,))
def _bin_predictors_uniform(
    X: Real[Array, 'p n'],
    low: Real[Array, ' p'],
    high: Real[Array, ' p'],
    num_bins: int,
) -> UInt[Array, 'p n']:
    """
    Bin predictors onto an evenly spaced grid without materializing the cutpoints.

    This is the arithmetic equivalent of binning with the splits from
    `_uniform_splits_from_range`: cutpoint ``j`` is ``low + (j + 1) * step``
    with ``step = (high - low) / num_bins``, and ``x`` falls in bin ``i`` iff
    ``cutpoint[i - 1] < x <= cutpoint[i]``.

    Parameters
    ----------
    X
        A matrix with `p` predictors and `n` observations.
    low
        The minimum value of each predictor's grid.
    high
        The maximum value of each predictor's grid.
    num_bins
        The number of bins per predictor.

    Returns
    -------
    `X` with each value replaced by the index of the bin it falls into.
    """
    step = (high - low) / num_bins
    safe_step = jnp.where(step > 0, step, 1)
    # bin = #{cutpoints < x}; right-closed bins make this ceil(t) - 1 (= floor(t)
    # away from cutpoints), matching `searchsorted(..., side='left')`
    t = (X - low[:, None]) / safe_step[:, None]
    bins = jnp.ceil(t) - 1
    # constant predictors (step == 0) have coincident cutpoints at `low`
    bins = jnp.where(
        step[:, None] > 0, bins, jnp.where(low[:, None] < X, num_bins - 1, 0)
    )
    bins = jnp.clip(bins, 0, num_bins - 1)
    return bins.astype(minimal_unsigned_dtype(num_bins - 1))


@jit(static_argnames=('method',))
def _bin_predictors(
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
        x: Real[Array, ' n'], splits: Real[Array, ' m']
    ) -> UInt[Array, ' n']:
        dtype = minimal_unsigned_dtype(splits.size)
        return jnp.searchsorted(splits, x, **kw).astype(dtype)

    return bin_predictors(X, splits)


class Binner(Module):
    """Abstract base class for predictor binners.

    A binner inspects the training predictors at construction time,
    chooses cutpoints for each predictor, and encapsulates the logic
    that maps any predictor matrix (training or test) to bin indices via
    `bin`.

    A predictor value ``x`` is mapped to bin ``i`` iff
    ``c[i - 1] < x <= c[i]``, where ``c`` are the cutpoints chosen for
    that predictor at construction. A predictor with ``k`` cutpoints
    therefore has ``k + 1`` bins indexed from ``0`` to ``k``. The number
    of cutpoints actually used per predictor is exposed as `max_split`
    and may differ across predictors; the remaining capacity, if any, is
    padded internally with the maximum value representable in the dtype
    of the cutpoints, so binning still produces a valid in-range index.

    The constructor takes the training predictors and an optional random
    key. Concrete subclasses may add their own keyword arguments. Binners
    that do not use the key still accept it for protocol uniformity and
    silently ignore it. Binners that need it raise `ValueError` if it is
    not provided.
    """

    max_split: AbstractVar[UInt[Array, ' p']]
    """The number of cutpoints actually used for each of the `p` predictors."""

    @abstractmethod
    def __init__(
        self, X: Real[Array, 'p n'], *, key: Key[Array, ''] | None = None
    ) -> None: ...

    @abstractmethod
    def bin(self, X: Real[Array, 'p n']) -> UInt[Array, 'p n']:
        """Map predictors to bin indices using the cutpoints chosen at construction.

        Parameters
        ----------
        X
            A matrix with `p` predictors and `n` observations. Must have
            the same number of predictors as the training matrix passed
            to the constructor.

        Returns
        -------
        Quantized `X` with minimal data type.
        """
        ...


@runtime_checkable
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

    For each predictor, ``max_bins - 1`` cutpoints are placed at
    equally spaced positions strictly between the minimum and the
    maximum value observed in the training matrix. All predictors use
    the same number of cutpoints.

    Parameters
    ----------
    X
        Training predictors with `p` predictors and `n` observations.
    max_bins
        The number of bins per predictor; ``max_bins - 1`` cutpoints
        are produced per predictor.
    key
        Accepted for protocol uniformity; unused.
    """

    _low: Real[Array, ' p']
    """Minimum observed value per predictor."""

    _high: Real[Array, ' p']
    """Maximum observed value per predictor."""

    # WORKAROUND(jax<0.9.1): use `jax.tree.static` instead of `field(static=True)`
    _max_bins: int = field(static=True)
    """Number of bins per predictor."""

    max_split: UInt[Array, ' p']

    def __init__(
        self,
        X: Real[Array, 'p n'],
        *,
        max_bins: int = 256,
        key: Key[Array, ''] | None = None,
    ) -> None:
        del key
        self._low = jnp.min(X, axis=1)
        self._high = jnp.max(X, axis=1)
        self._max_bins = max_bins
        self.max_split = jnp.full(
            X.shape[0], max_bins - 1, minimal_unsigned_dtype(max_bins - 1)
        )

    @property
    def _splits(self) -> Real[Array, 'p m']:
        """Materialize the cutpoints. Intended for testing only, not library use.

        The cutpoints are not stored: `bin` works arithmetically from the
        observed range, since they are evenly spaced. This property reconstructs
        them only to expose them; the library should rely on `bin` and
        `max_split` instead.
        """
        return _uniform_splits_from_range(self._low, self._high, self._max_bins)

    def bin(self, X: Real[Array, 'p n']) -> UInt[Array, 'p n']:
        return _bin_predictors_uniform(X, self._low, self._high, self._max_bins)


class UniqueQuantileBinner(Binner):
    """Binner with quantile-based cutpoints from observed unique values.

    For each predictor, cutpoints are placed between sorted unique
    values so that the empirical distribution is approximately uniform
    across bins. The number of cutpoints is at most ``max_bins - 1``
    and at most one less than the number of unique values, so different
    predictors may end up with different effective cutpoint counts.
    Trailing unused entries of the cutpoint matrix are padded with the
    maximum value representable in the dtype of `X`.

    Note: the quantiles are over the *unique* values, not over the
    original distribution.

    When ``n > max_subsample``, the predictor matrix is randomly thinned
    along the observation axis to ``max_subsample`` columns before
    quantilization. Each predictor row is thinned independently and
    without replacement. This keeps quantilization tractable on very
    large datasets at the cost of approximate quantiles.

    Parameters
    ----------
    X
        Training predictors with `p` predictors and `n` observations.
    max_bins
        The maximum number of bins per predictor.
    max_subsample
        The maximum number of observations to use when computing
        quantiles. If `None`, no subsampling is performed. If `n`
        exceeds this, `key` is required.
    key
        Random key for subsampling. Required when ``X.shape[1] >
        max_subsample``; otherwise unused.

    Raises
    ------
    ValueError
        If subsampling would trigger but `key` is `None`.
    """

    _splits: Real[Array, 'p m']
    """Cutpoints per predictor, padded on the right with the dtype's maximum value."""

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
            X = _subsample(key, X, max_subsample)
        self._splits, self.max_split = _quantilized_splits_from_matrix(X, max_bins)

    def bin(self, X: Real[Array, 'p n']) -> UInt[Array, 'p n']:
        return _bin_predictors(X, self._splits)


class GivenSplitsBinner(Binner):
    """Binner with cutpoints supplied directly in R BART `xinfo` format.

    The cutpoints are taken verbatim from `xinfo`: a `(p, m)` matrix
    whose rows hold per-predictor sorted cutpoints, with NaN-padded
    trailing entries marking unused capacity. Internally NaNs are
    replaced by the maximum representable value in the dtype of
    `xinfo`, and `max_split` is set to the count of non-NaN entries
    per row, so binning behaves as if the row had been declared with
    only its non-NaN cutpoints.

    Parameters
    ----------
    X
        Training predictors. Used only to validate the shape of `xinfo`.
    xinfo
        A `(p, m)` matrix of cutpoints. Each row holds a sorted list of
        cutpoints for one predictor, optionally padded on the right with
        NaN.
    key
        Accepted for protocol uniformity; unused.

    Raises
    ------
    ValueError
        If `xinfo` is not 2D, or if its first dimension does not match
        ``X.shape[0]``.
    """

    _splits: Float[Array, 'p m']
    """Cutpoints per predictor, with NaNs replaced by the dtype's maximum value."""

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
        self._splits, self.max_split = _parse_xinfo(xinfo)

    def bin(self, X: Real[Array, 'p n']) -> UInt[Array, 'p n']:
        return _bin_predictors(X, self._splits)


@jit
def _sigma2_from_ols(
    x_train: Shaped[Array, 'p n'], y_train: Float32[Array, ' n'] | Float32[Array, 'k n']
) -> Float32[Array, ''] | Float32[Array, ' k']:
    """Return the error variance estimated with OLS with intercept."""
    x_centered = x_train.T - x_train.mean(axis=1)
    y_centered = y_train.T - y_train.mean(axis=-1)
    # centering is equivalent to adding an intercept column
    _, chisq, rank, _ = jnp.linalg.lstsq(x_centered, y_centered)
    chisq = chisq.reshape(y_train.shape[:-1])
    dof = y_train.shape[-1] - rank
    return chisq / dof


@jit
def _sigma2_from_cg(
    x: Shaped[Array, 'p n'], y: Float32[Array, ' *k n'], maxiter: Integer[Array, '']
) -> Float32[Array, ' *k']:
    """Estimate the error variance using approximate OLS with cg for large-scale problems."""
    # center both variables, and rescale x. centering is equivalent to adding
    # an intercept term, while rescaling x does not change the result since we
    # are returning something that depends on the residuals only
    y -= y.mean(axis=-1, keepdims=True)
    x -= x.mean(axis=-1, keepdims=True)
    scale = x.std(axis=1, keepdims=True)
    x /= jnp.where(scale, scale, 1.0)

    # compute residuals
    p, n = x.shape
    if p <= n:
        rhs: Float32[Array, ' p *k'] = x @ y.T
        beta: Float32[Array, ' p *k'] = _cg_x_x_t(x, rhs, maxiter)
        r: Float32[Array, ' n *k'] = y.T - x.T @ beta
    else:
        z: Float32[Array, ' n *k'] = _cg_x_x_t(x.T, y.T, maxiter)
        r: Float32[Array, ' n *k'] = y.T - x.T @ (x @ z)

    # estimate residual variance
    return jnp.sum(jnp.square(r), axis=0) / (n - maxiter)


def _cg_x_x_t(
    x: Shaped[Array, 'p n'] | Shaped[Array, 'n p'],
    rhs: Float32[Array, ' p *k'] | Float32[Array, ' n *k'],
    maxiter: Integer[Array, ''],
) -> Float32[Array, ' p *k'] | Float32[Array, ' n *k']:
    """Solve (XX' + eps I) u = rhs."""
    # check x is transposed to be a short matrix, and other properties
    p, n = x.shape
    assert p <= n
    r1, *_ = rhs.shape
    assert p == r1

    # upper bound the max eigenvalue of XX' to determine epsilon
    abs_x = jnp.abs(x)
    max_eigv = jnp.max(abs_x @ (abs_x.T @ jnp.ones(p)))
    eps = jnp.finfo(max_eigv.dtype).eps * p * max_eigv

    # solve with vmapped cg
    xxt = lambda rhs: x @ (x.T @ rhs) + eps * rhs
    func = lambda rhs: cg(xxt, rhs, tol=0, maxiter=maxiter)
    if rhs.ndim == 2:
        func = vmap(func, in_axes=1, out_axes=1)
    out, _ = func(rhs)
    return out
