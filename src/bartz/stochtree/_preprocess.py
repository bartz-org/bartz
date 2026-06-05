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

Two parallel implementations are provided, `PandasPreprocessor` and
`PolarsPreprocessor`, each handling the corresponding dataframe library. Both
classes have the same interface::

    pp = PandasPreprocessor()      # or PolarsPreprocessor()
    varprob = pp.fit(X_train, variable_weights=w)
    x_train = pp.transform(X_train)
    x_new = pp.transform(X_new)    # at prediction time

`fit` records the per-column encoding and returns the variable weights expanded
to match the new column count (or `None`); `transform` returns the
post-processing covariate matrix as a 2-D numpy float32 array (rows=observations,
columns=expanded features).

Per-column handling:

- ordered categorical (pandas ordered `Categorical`): ordinal encoded into a
  single integer-valued column, with the declared category order giving the
  integer mapping. polars has no ordered categorical dtype; pass an integer
  column for ordinal encoding.
- unordered categorical (pandas unordered `Categorical`, polars `Enum`): one-hot
  encoded into one binary column per declared category. A polars `Enum`
  round-trips to a pandas *unordered* `Categorical`, so the two are treated
  identically.
- boolean: cast to ``{0.0, 1.0}``, single column.
- numeric (integer, unsigned, float): pass-through as float.
- anything else (strings, ``object``, datetime, polars `Categorical`, etc.):
  raises `ValueError`. polars `Categorical` has no reliable per-column category
  list (the categories live in a process-wide string cache shared across
  columns), so it must be cast to an `Enum` (one-hot) or an integer (ordinal)
  first.

When a single original column expands into ``k`` output columns (one-hot), the
original `variable_weights` entry for that column is split evenly across the
``k`` expansions, preserving each original variable's total splitting budget
(matching stochtree's `bart.py` behavior).

Unknown category values encountered during `transform` raise `ValueError`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from jaxtyping import Float32, Shaped
from numpy.typing import ArrayLike

# Duck-typed stand-ins for the optional dataframe libraries. bartz does not
# depend on pandas or polars at runtime, so we cannot reference their real
# classes here; these aliases resolve to `Any` but give the signatures below
# legible names.
DataFrame: TypeAlias = Any  # a pandas or polars DataFrame
Series: TypeAlias = Any  # a pandas or polars Series
PolarsModule: TypeAlias = Any  # the polars top-level module

_UNSEEN_PREVIEW = 10

ColumnKind: TypeAlias = Literal['numeric', 'bool', 'ordered_cat', 'unordered_cat']


@dataclass(frozen=True)
class _ColumnSpec:
    """Per-original-column fitted state."""

    kind: ColumnKind
    """Encoding to apply to the column."""

    name: str
    """Original column name (for error messages)."""

    categories: tuple[Any, ...] | None = None
    """Declared category list for ordered_cat / unordered_cat."""

    @property
    def width(self) -> int:
        """Number of output columns this spec produces."""
        if self.kind == 'unordered_cat':
            assert self.categories is not None
            return len(self.categories)
        return 1


def _unseen_error(name: str, unseen: Sequence[Any], known: Sequence[Any]) -> ValueError:
    """Build the error for category values absent from the fitted list."""
    uniq = sorted({repr(v) for v in unseen})
    msg = (
        f'column {name!r}: {len(unseen)} value(s) at transform time are not in'
        f' the fitted category list; unseen sample: {uniq[:_UNSEEN_PREVIEW]};'
        f' known categories: {list(known)[:_UNSEEN_PREVIEW]}'
    )
    return ValueError(msg)


def _unsupported_dtype_error(name: str, dtype: object) -> ValueError:
    """Build the error for a column whose dtype has no supported encoding."""
    msg = (
        f'column {name!r} has unsupported dtype {dtype!r}; supported types are'
        ' numeric, boolean, pandas ordered/unordered Categorical, and polars'
        ' Enum. Convert strings, objects, datetimes, etc. to one of these (e.g.'
        ' an explicit Categorical / Enum) before fitting.'
    )
    return ValueError(msg)


def _polars_categorical_error(name: str) -> ValueError:
    """Build the error rejecting a polars `Categorical` column."""
    msg = (
        f'column {name!r} is a polars Categorical, which has no reliable'
        ' per-column category list (the categories live in a process-wide'
        ' string cache shared across columns). Cast it to a polars Enum with an'
        ' explicit category list (pl.Enum([...])) for one-hot encoding, or to an'
        ' integer column for ordinal encoding.'
    )
    return ValueError(msg)


def _ordinal_encode(
    values: Shaped[np.ndarray, ' n'], categories: Sequence[Any], name: str
) -> Float32[np.ndarray, 'n 1']:
    """Map `values` to integer positions in `categories`; raise on unseen."""
    table = {c: i for i, c in enumerate(categories)}
    out = np.empty(len(values), dtype=np.float32)
    unseen: list[Any] = []
    for i, v in enumerate(values):
        code = table.get(v, -1)
        if code < 0:
            unseen.append(v)
        else:
            out[i] = code
    if unseen:
        raise _unseen_error(name, unseen, categories)
    return out[:, None]


def _one_hot_encode(
    values: Shaped[np.ndarray, ' n'], categories: Sequence[Any], name: str
) -> Float32[np.ndarray, 'n k']:
    """Build a ``(n, k)`` one-hot matrix using `categories` order; raise on unseen."""
    table = {c: i for i, c in enumerate(categories)}
    n = len(values)
    k = len(categories)
    out = np.zeros((n, k), dtype=np.float32)
    unseen: list[Any] = []
    for i, v in enumerate(values):
        code = table.get(v, -1)
        if code < 0:
            unseen.append(v)
        else:
            out[i, code] = 1.0
    if unseen:
        raise _unseen_error(name, unseen, categories)
    return out


def _polars_one_hot(
    pl: PolarsModule, series: Series, categories: Sequence[Any], name: str
) -> Float32[np.ndarray, 'n k']:
    """Validate via cast to `pl.Enum(categories)` and one-hot via polars APIs.

    Polars's `Enum` cast natively raises on any value not in `categories`, and
    `to_physical` returns the integer codes in the declared-category order. The
    `np.eye` index is the only numpy bit and is just an identity-matrix lookup;
    the categorical bookkeeping itself stays inside polars.
    """
    cats = list(categories)
    try:
        coded = series.cast(pl.Enum(cats))
    except pl.exceptions.InvalidOperationError as exc:
        # Identify the actual unseen values for a friendly error. Cast to String
        # first: the input column may itself be an Enum with different categories,
        # which would make a direct is_in(cats) fail trying to coerce the list.
        known = set(cats)
        unseen = sorted(
            {
                v
                for v in series.cast(pl.String).to_list()
                if v not in known and v is not None
            }
        )
        raise _unseen_error(name, unseen, cats) from exc
    if coded.null_count():
        msg = f'column {name!r}: null values are not supported in categorical columns'
        raise ValueError(msg)
    codes = coded.to_physical().to_numpy()
    return np.eye(len(cats), dtype=np.float32)[codes]


def _expand_variable_weights(
    weights: ArrayLike, original_var_indices: Sequence[int], n_orig: int
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


def _stack(
    cols: Sequence[Float32[np.ndarray, 'n _']], n_rows: int
) -> Float32[np.ndarray, 'n p']:
    if not cols:
        return np.empty((n_rows, 0), dtype=np.float32)
    return np.concatenate(cols, axis=1)


class _PreprocessorBase:
    """Common state for `PandasPreprocessor` and `PolarsPreprocessor`."""

    _library: str = ''
    """Top-level module prefix of the supported dataframe library."""

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

    def fit(
        self, X: DataFrame, *, variable_weights: ArrayLike | None = None
    ) -> Float32[np.ndarray, ' p'] | None:
        """Record the per-column encoding and return the expanded variable weights.

        Returns `None` when no weights are supplied and no column expands into
        several output columns, so the caller can fall back to the native
        uniform-weights path; otherwise returns the weights split across each
        original column's one-hot expansion.
        """
        self._check_library(X)
        specs: list[_ColumnSpec] = []
        original_var_indices: list[int] = []
        for orig_idx in range(X.shape[1]):
            name, series = self._get_column(X, orig_idx)
            spec = self._fit_column(series, str(name))
            specs.append(spec)
            original_var_indices.extend([orig_idx] * spec.width)
        self._specs = tuple(specs)
        self._original_var_indices = tuple(original_var_indices)
        self._fitted = True
        expanded = len(set(original_var_indices)) != len(original_var_indices)
        if variable_weights is None:
            if not expanded:
                return None
            variable_weights = np.full(len(specs), 1.0 / len(specs))
        return _expand_variable_weights(
            variable_weights, self._original_var_indices, len(self._specs)
        )

    def transform(self, X: DataFrame) -> Float32[np.ndarray, 'n p']:
        """Apply the fitted transformation to a new dataframe."""
        self._check_fitted()
        self._check_library(X)
        self._check_n_columns(X.shape[1])
        cols = [
            self._transform_column(self._get_column(X, orig_idx)[1], spec)
            for orig_idx, spec in enumerate(self._specs)
        ]
        return _stack(cols, X.shape[0])

    def _check_fitted(self) -> None:
        if not self._fitted:
            msg = 'preprocessor has not been fitted yet; call fit first'
            raise RuntimeError(msg)

    def _check_n_columns(self, n_cols: int) -> None:
        if n_cols != len(self._specs):
            msg = (
                f'transform input has {n_cols} columns; preprocessor was fitted'
                f' on {len(self._specs)} columns'
            )
            raise ValueError(msg)

    def _check_library(self, X: DataFrame) -> None:
        module = type(X).__module__
        if not module.startswith(self._library):
            msg = (
                f'this preprocessor handles {self._library} dataframes, but got'
                f' an object from {module!r}; fit and transform must use the same'
                ' dataframe library'
            )
            raise TypeError(msg)

    @staticmethod
    def _get_column(X: DataFrame, orig_idx: int) -> tuple[Any, Series]:
        """Return the ``(name, series)`` of the column at position `orig_idx`."""
        raise NotImplementedError

    @staticmethod
    def _fit_column(series: Series, name: str) -> _ColumnSpec:
        """Inspect a column's dtype and return its encoding spec."""
        raise NotImplementedError

    @staticmethod
    def _transform_column(
        series: Series, spec: _ColumnSpec
    ) -> Float32[np.ndarray, 'n _']:
        """Encode a single column according to its fitted spec."""
        raise NotImplementedError


class PandasPreprocessor(_PreprocessorBase):
    """Stochtree-style covariate preprocessor for `pandas.DataFrame` inputs."""

    _library = 'pandas'

    @staticmethod
    def _get_column(X: DataFrame, orig_idx: int) -> tuple[Any, Series]:
        return X.columns[orig_idx], X.iloc[:, orig_idx]

    @staticmethod
    def _fit_column(series: Series, name: str) -> _ColumnSpec:
        import pandas as pd  # noqa: PLC0415  # optional runtime dependency

        dt = series.dtype
        if isinstance(dt, pd.CategoricalDtype):
            cats = tuple(dt.categories)
            kind: ColumnKind = 'ordered_cat' if dt.ordered else 'unordered_cat'
            return _ColumnSpec(kind, name, categories=cats)
        if pd.api.types.is_bool_dtype(dt):
            return _ColumnSpec('bool', name)
        if pd.api.types.is_numeric_dtype(dt):
            return _ColumnSpec('numeric', name)
        raise _unsupported_dtype_error(name, dt)

    @staticmethod
    def _transform_column(
        series: Series, spec: _ColumnSpec
    ) -> Float32[np.ndarray, 'n _']:
        if spec.kind == 'ordered_cat':
            assert spec.categories is not None
            return _ordinal_encode(series.to_numpy(), spec.categories, spec.name)
        if spec.kind == 'unordered_cat':
            assert spec.categories is not None
            return _one_hot_encode(series.to_numpy(), spec.categories, spec.name)
        return series.to_numpy(dtype=np.float32)[:, None]


class PolarsPreprocessor(_PreprocessorBase):
    """Stochtree-style covariate preprocessor for `polars.DataFrame` inputs."""

    _library = 'polars'

    @staticmethod
    def _get_column(X: DataFrame, orig_idx: int) -> tuple[Any, Series]:
        name = X.columns[orig_idx]
        return name, X[name]

    @staticmethod
    def _fit_column(series: Series, name: str) -> _ColumnSpec:
        import polars as pl  # noqa: PLC0415  # optional runtime dependency

        dt = series.dtype
        if isinstance(dt, pl.Enum):
            # A polars Enum round-trips to a pandas *unordered* Categorical, so
            # we treat it as unordered (one-hot). For ordinal encoding, pass an
            # integer column.
            return _ColumnSpec(
                'unordered_cat', name, categories=tuple(dt.categories.to_list())
            )
        if isinstance(dt, pl.Categorical):
            raise _polars_categorical_error(name)
        if dt == pl.Boolean:
            return _ColumnSpec('bool', name)
        if dt.is_numeric():
            return _ColumnSpec('numeric', name)
        raise _unsupported_dtype_error(name, dt)

    @staticmethod
    def _transform_column(
        series: Series, spec: _ColumnSpec
    ) -> Float32[np.ndarray, 'n _']:
        import polars as pl  # noqa: PLC0415  # optional runtime dependency

        if spec.kind == 'unordered_cat':
            assert spec.categories is not None
            return _polars_one_hot(pl, series, spec.categories, spec.name)
        return series.cast(pl.Float32).to_numpy()[:, None]


def make_preprocessor(X: object) -> _PreprocessorBase | None:
    """Return a preprocessor matched to `X`'s library, or `None` if `X` is not a DataFrame.

    Dispatches by inspecting ``type(X).__module__`` to avoid hard imports of
    pandas/polars.
    """
    mod = type(X).__module__
    if mod.startswith('polars'):
        return PolarsPreprocessor()
    if mod.startswith('pandas'):
        return PandasPreprocessor()
    return None
