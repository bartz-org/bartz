# bartz/src/bartz/stochtree/_preprocess.py
#
# Copyright (c) 2026, The Bartz Contributors
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

"""Auto-preprocessing of covariates for the stochtree-compatible BART interface.

`Preprocessor` handles any eager dataframe supported by narwhals (pandas,
polars, pyarrow, modin, cudf), wrapped with `to_narwhals`::

    pp = Preprocessor()
    df_train = to_narwhals(X_train)  # None if not a dataframe
    varprob = pp.fit(df_train, variable_weights=w)
    x_train = pp.transform(df_train)
    x_new = pp.transform(to_narwhals(X_new))    # at prediction time

`fit` records the per-column encoding and returns the variable weights expanded
to match the new column count (or `None`); `transform` returns the
post-processing covariate matrix as a 2-D numpy float32 array (rows=observations,
columns=expanded features).

Per-column handling:

- ordered categorical (pandas ordered `Categorical`, ordered pyarrow
  dictionary): ordinal encoded into a single integer-valued column, with the
  declared category order giving the integer mapping. polars has no ordered
  categorical dtype; pass an integer column for ordinal encoding.
- unordered categorical (pandas unordered `Categorical`, polars `Enum`, pyarrow
  dictionary): one-hot encoded into one binary column per declared category. A
  polars `Enum` round-trips to a pandas *unordered* `Categorical`, so the two
  are treated identically.
- boolean: cast to ``{0.0, 1.0}``, single column.
- numeric (integer, unsigned, float, decimal): pass-through as float.
- pandas columns with a pyarrow dtype (`pandas.ArrowDtype`): handled like the
  corresponding pyarrow column.
- anything else (strings, ``object``, datetime, polars `Categorical`, etc.):
  raises `ValueError`. polars `Categorical` has no per-column category list,
  so it must be cast to an `Enum` (one-hot), or recoded as integers (ordinal).

When a single original column expands into ``k`` output columns (one-hot), the
original `variable_weights` entry for that column is split evenly across the
``k`` expansions, preserving each original variable's total splitting budget
(matching stochtree's `bart.py` behavior).

Null values in categorical columns, and category values unknown at `transform`
time, raise `ValueError`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, overload, runtime_checkable

import narwhals.stable.v2 as nw
import numpy as np
from jaxtyping import Float32, Int, Shaped

# Not `numpy.typing.ArrayLike`: that is a PEP 695 type alias since numpy 2.5,
# which jaxtyping cannot subscript.
from bartz.mcmcstep._state import ArrayLike


@runtime_checkable
class DataFrame(Protocol):
    """Any eager dataframe supported by narwhals, e.g., pandas, polars, or pyarrow."""

    @property
    def columns(self) -> object:
        """The columns."""
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """The number of rows and columns."""
        ...


_UNSEEN_PREVIEW = 10

ColumnKind: TypeAlias = Literal['numeric', 'bool', 'ordered_cat', 'unordered_cat']


@dataclass(frozen=True)
class _ColumnSpec:
    """Per-original-column fitted state."""

    kind: ColumnKind
    """Encoding to apply to the column."""

    name: str
    """Original column name (for error messages)."""

    categories: tuple[Any, ...] = ()
    """Declared category list for ordered_cat / unordered_cat."""

    @property
    def width(self) -> int:
        """Number of output columns this spec produces."""
        if self.kind == 'unordered_cat':
            return len(self.categories)
        else:
            return 1


def _unseen_error(name: str, unseen: Sequence[Any], known: Sequence[Any]) -> ValueError:
    """Build the error for category values absent from the fitted list."""
    msg = (
        f'column {name!r}: {len(unseen)} distinct value(s) at transform time are'
        ' not in the fitted category list; unseen sample:'
        f' {list(unseen)[:_UNSEEN_PREVIEW]}; known categories:'
        f' {list(known)[:_UNSEEN_PREVIEW]}'
    )
    return ValueError(msg)


def _unsupported_dtype_error(name: str, dtype: object) -> ValueError:
    """Build the error for a column whose dtype has no supported encoding."""
    msg = (
        f'column {name!r} has unsupported dtype {dtype!r}; supported types are'
        ' numeric, boolean, pandas ordered/unordered Categorical, polars Enum,'
        ' and pyarrow dictionary. Convert strings, objects, datetimes, etc. to'
        ' one of these (e.g. an explicit Categorical / Enum) before fitting.'
    )
    return ValueError(msg)


def _polars_categorical_error(name: str) -> ValueError:
    """Build the error rejecting a polars `Categorical` column."""
    msg = (
        f'column {name!r} is a polars Categorical, which has no per-column'
        ' category list. Cast it to a polars Enum with an explicit category list'
        ' (pl.Enum([...])) for one-hot encoding, or replace it with an integer'
        ' column giving the rank of each level for ordinal encoding.'
    )
    return ValueError(msg)


def _fit_column(series: nw.Series, name: str) -> _ColumnSpec:
    """Inspect a column's dtype and return its encoding spec."""
    dtype = series.dtype
    impl = series.implementation
    if isinstance(dtype, nw.Enum):
        # narwhals maps ordered pandas categoricals to `Enum`, unordered ones
        # to `Categorical`
        kind: ColumnKind = 'ordered_cat' if impl.is_pandas_like() else 'unordered_cat'
        return _ColumnSpec(kind, name, tuple(dtype.categories))
    elif isinstance(dtype, nw.Categorical):
        if impl.is_polars():
            # narwhals can only list the observed values
            raise _polars_categorical_error(name)
        # narwhals drops the ordered flag of pyarrow dictionaries
        ordered = impl.is_pyarrow() and series.to_native().type.ordered
        kind = 'ordered_cat' if ordered else 'unordered_cat'
        return _ColumnSpec(kind, name, tuple(series.cat.get_categories().to_list()))
    elif dtype == nw.Boolean:
        return _ColumnSpec('bool', name)
    elif dtype.is_numeric():
        return _ColumnSpec('numeric', name)
    else:
        raise _unsupported_dtype_error(name, dtype)


def _get_column(df: nw.DataFrame[Any], name: str) -> nw.Series:
    """Get a column, moving it to pyarrow if it has a pyarrow dtype in pandas."""
    series = df.get_column(name)
    if series.implementation.is_pandas_like() and hasattr(
        series.to_native().dtype, 'pyarrow_dtype'
    ):
        # narwhals can't use `.cat` on `pandas.ArrowDtype` dictionaries
        arrow = series.to_frame().to_arrow().column(0)
        return nw.from_native(arrow, series_only=True)
    else:
        return series


def _category_codes(series: nw.Series, spec: _ColumnSpec) -> Int[np.ndarray, ' n']:
    """Map values to their position in the fitted categories; raise on null or unseen."""
    if series.null_count():
        msg = (
            f'column {spec.name!r}: null values are not supported in categorical'
            ' columns'
        )
        raise ValueError(msg)
    # look up only the distinct values, then map them natively
    table = {c: i for i, c in enumerate(spec.categories)}
    uniq = series.unique().to_list()
    uniq_codes = [table.get(u, -1) for u in uniq]
    unseen = [u for u, code in zip(uniq, uniq_codes, strict=True) if code < 0]
    if unseen:
        raise _unseen_error(spec.name, unseen, spec.categories)
    return series.replace_strict(uniq, uniq_codes, return_dtype=nw.Int64).to_numpy()


def _transform_column(
    series: nw.Series, spec: _ColumnSpec
) -> Float32[np.ndarray, 'n _']:
    """Encode a single column according to its fitted spec."""
    if spec.kind == 'ordered_cat':
        return _category_codes(series, spec)[:, None].astype(np.float32)
    elif spec.kind == 'unordered_cat':
        return np.eye(spec.width, dtype=np.float32)[_category_codes(series, spec)]
    else:
        return series.cast(nw.Float32).to_numpy()[:, None]


def _expand_variable_weights(
    weights: Shaped[ArrayLike, '...'], original_var_indices: Sequence[int], n_orig: int
) -> Float32[np.ndarray, ' p']:
    """Split each original weight evenly across its one-hot expansions."""
    w = np.asarray(weights, dtype=np.float32)
    if w.shape != (n_orig,):
        msg = (
            f'variable_weights must have shape ({n_orig},) matching the number'
            f' of original columns; got {w.shape}'
        )
        raise ValueError(msg)
    if not original_var_indices:
        return np.empty((0,), dtype=np.float32)
    counts = np.bincount(np.asarray(original_var_indices), minlength=n_orig)
    return np.array([w[j] / counts[j] for j in original_var_indices], dtype=np.float32)


class Preprocessor:
    """Stochtree-style covariate preprocessor for dataframes."""

    _fitted: bool = False
    _specs: Sequence[_ColumnSpec] = ()
    _original_var_indices: Sequence[int] = ()

    @property
    def fitted(self) -> bool:
        """Whether `fit` has been called."""
        return self._fitted

    @property
    def n_original_columns(self) -> int:
        """Number of columns in the dataframe given to `fit`."""
        return len(self._specs)

    @property
    def n_processed_columns(self) -> int:
        """Number of columns in the matrix returned by `transform`."""
        return len(self._original_var_indices)

    @property
    def original_var_indices(self) -> tuple[int, ...]:
        """For each output column, the index of the original column it came from."""
        return tuple(self._original_var_indices)

    @overload
    def fit(
        self, df: nw.DataFrame[Any], *, variable_weights: Shaped[ArrayLike, '...']
    ) -> Float32[np.ndarray, ' p']: ...

    @overload
    def fit(
        self, df: nw.DataFrame[Any], *, variable_weights: None = None
    ) -> Float32[np.ndarray, ' p'] | None: ...

    def fit(
        self,
        df: nw.DataFrame[Any],
        *,
        variable_weights: Shaped[ArrayLike, '...'] | None = None,
    ) -> Float32[np.ndarray, ' p'] | None:
        """Record the per-column encoding and return the expanded variable weights.

        Returns `None` when no weights are supplied and no column expands into
        several output columns, so the caller can fall back to the native
        uniform-weights path; otherwise returns the weights split across each
        original column's one-hot expansion.
        """
        self._specs = tuple(_fit_column(_get_column(df, c), str(c)) for c in df.columns)
        self._original_var_indices = tuple(
            i for i, spec in enumerate(self._specs) for _ in range(spec.width)
        )
        self._fitted = True
        expanded = len(self._original_var_indices) != len(self._specs)
        if variable_weights is None:
            if not expanded:
                return None
            variable_weights = np.full(len(self._specs), 1.0 / len(self._specs))
        return _expand_variable_weights(
            variable_weights, self._original_var_indices, len(self._specs)
        )

    def transform(self, df: nw.DataFrame[Any]) -> Float32[np.ndarray, 'n p']:
        """Apply the fitted transformation to a new dataframe, from any library."""
        if not self.fitted:
            msg = 'preprocessor has not been fitted yet; call fit first'
            raise RuntimeError(msg)
        n, p = df.shape
        if p != len(self._specs):
            msg = (
                f'transform input has {p} columns; preprocessor was fitted'
                f' on {len(self._specs)} columns'
            )
            raise ValueError(msg)
        cols = [
            _transform_column(_get_column(df, c), spec)
            for c, spec in zip(df.columns, self._specs, strict=True)
        ]
        if cols:
            return np.concatenate(cols, axis=1)
        else:
            return np.empty((n, 0), dtype=np.float32)


def to_narwhals(X: object) -> nw.DataFrame[Any] | None:
    """Wrap `X` with narwhals if it is an eager dataframe it supports, else return `None`; reject lazy frames."""
    df = nw.from_native(X, pass_through=True)
    if isinstance(df, nw.DataFrame):
        return df
    elif isinstance(df, nw.LazyFrame):
        msg = (
            f'lazy dataframes are not supported, got {type(X).__name__}; collect it'
            ' into an eager dataframe first, e.g., with polars `LazyFrame.collect()`'
        )
        raise TypeError(msg)
    else:
        return None
