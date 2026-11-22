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

- ordered categorical (pandas ordered `Categorical`, polars `Enum`, ordered
  pyarrow dictionary): ordinal encoded into a single integer-valued column, with
  the declared category order giving the integer mapping.
- unordered categorical (pandas unordered `Categorical`, polars `Categorical`,
  unordered pyarrow dictionary): one-hot encoded into one binary column per
  declared category. A polars `Categorical` has no per-column category list, so
  its categories are the values observed at `fit`, sorted.
- boolean: cast to ``{0.0, 1.0}``, single column.
- numeric (integer, unsigned, float, decimal): pass-through as float.
- pandas columns with a pyarrow dtype (`pandas.ArrowDtype`): handled like the
  corresponding pyarrow column.
- anything else (strings, ``object``, datetime, etc.): raises `ValueError`.

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
from warnings import warn

import narwhals.stable.v2 as nw
import numpy as np
from jaxtyping import Float32, Shaped
from narwhals.stable.v2.dependencies import get_polars

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
        ' numeric, boolean, pandas ordered/unordered Categorical, polars Enum or'
        ' Categorical, and pyarrow dictionary. Convert strings, objects,'
        ' datetimes, etc. to one of these (e.g. an explicit Categorical / Enum)'
        ' before fitting.'
    )
    return ValueError(msg)


def _polars_enum_message(name: str) -> str:
    """Build the warning about the encoding of a polars `Enum` column."""
    msg = (
        f'column {name!r} is a polars Enum, which is ordered, so it is encoded as'
        ' a single ordinal column; bartz <= 0.14 one-hot encoded it. For one-hot'
        ' encoding, use a polars Categorical.'
    )
    # WORKAROUND(polars<2): drop this note
    if int(get_polars().__version__.split('.')[0]) < 2:
        msg += (
            ' Note that polars < 2 converts an Enum to an unordered categorical in'
            ' to_pandas() and to_arrow(), which is one-hot encoded.'
        )
    return msg


def _fit_column(series: nw.Series, name: str) -> _ColumnSpec:
    """Inspect a column's dtype and return its encoding spec."""
    dtype = series.dtype
    impl = series.implementation
    if isinstance(dtype, nw.Enum):
        if impl.is_polars():
            warn(_polars_enum_message(name))
        return _ColumnSpec('ordered_cat', name, tuple(dtype.categories))
    elif isinstance(dtype, nw.Categorical) and impl.is_polars():
        # sort to not depend on the order of appearance
        categories = tuple(sorted(series.drop_nulls().unique().to_list()))
        return _ColumnSpec('unordered_cat', name, categories)
    elif isinstance(dtype, nw.Categorical):
        # narwhals drops the ordered flag of pyarrow dictionaries
        ordered = impl.is_pyarrow() and series.to_native().type.ordered
        kind: ColumnKind = 'ordered_cat' if ordered else 'unordered_cat'
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


def _category_lookup(
    series: nw.Series, spec: _ColumnSpec
) -> tuple[list[Any], list[int]]:
    """Return the distinct values and their fitted category codes; raise on null or unseen."""
    if series.null_count():
        msg = (
            f'column {spec.name!r}: null values are not supported in categorical'
            ' columns'
        )
        raise ValueError(msg)
    table = {c: i for i, c in enumerate(spec.categories)}
    uniq = series.unique().to_list()
    codes = [table.get(u, -1) for u in uniq]
    unseen = [u for u, code in zip(uniq, codes, strict=True) if code < 0]
    if unseen:
        raise _unseen_error(spec.name, unseen, spec.categories)
    return uniq, codes


def _column_exprs(series: nw.Series, spec: _ColumnSpec) -> list[nw.Expr]:
    """Build the expressions encoding a column according to its fitted spec."""
    col = nw.col(series.name)
    if spec.kind == 'ordered_cat':
        # keyed on the observed values, which are valid for any column dtype
        uniq, codes = _category_lookup(series, spec)
        return [col.replace_strict(uniq, codes).cast(nw.Float32)]
    elif spec.kind == 'unordered_cat':
        _category_lookup(series, spec)
        return [(col == c).cast(nw.Float32) for c in spec.categories]
    else:
        return [col.cast(nw.Float32)]


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
        exprs = [
            expr
            for c, spec in zip(df.columns, self._specs, strict=True)
            for expr in _column_exprs(df.get_column(c), spec)
        ]
        if exprs:
            out = df.select(*(e.alias(str(i)) for i, e in enumerate(exprs)))
            return out.to_numpy().astype(np.float32, copy=False)
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
