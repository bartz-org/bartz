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
    x_train, varprob = pp.fit_transform(X_train, variable_weights=w)
    x_test = pp.transform(X_test)  # if X_test is provided
    x_new  = pp.transform(X_new)   # at prediction time

`fit_transform` returns the post-processing covariate matrix as a 2-D numpy
float array (rows=observations, columns=expanded features) plus the variable
weights vector expanded to match the new column count (or `None`).

Per-column handling, matching stochtree's `CovariatePreprocessor`:

- ordered categorical (pandas ordered `Categorical`, polars `Enum`): ordinal
  encoded into a single integer-valued column, with the declared category order
  giving the integer mapping.
- unordered categorical / string (pandas unordered `Categorical`, pandas
  ``string`` / ``object``, polars ``Categorical`` / ``String``): one-hot
  encoded into one binary column per category, using the categories observed
  at fit time.
- boolean: cast to ``{0.0, 1.0}``, single column.
- numeric (integer, unsigned, float): pass-through as float.
- anything else (datetime, timedelta, etc.): warned and dropped.

When a single original column expands into ``k`` output columns (one-hot), the
original `variable_weights` entry for that column is split evenly across the
``k`` expansions, preserving each original variable's total splitting budget
(matching stochtree's `bart.py` behavior).

Unknown values encountered during `transform` raise `ValueError`.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np

# Duck-typed stand-ins for the optional dataframe libraries. bartz does not
# depend on pandas or polars at runtime, so we cannot reference their real
# classes here; these aliases resolve to `Any` but give the signatures below
# legible names.
DataFrame: TypeAlias = Any  # a pandas or polars DataFrame
Series: TypeAlias = Any  # a pandas or polars Series
PandasModule: TypeAlias = Any  # the pandas top-level module
PolarsModule: TypeAlias = Any  # the polars top-level module

_UNSEEN_PREVIEW = 10


@dataclass
class _ColumnSpec:
    """Per-original-column fitted state."""

    kind: str
    """One of 'numeric', 'bool', 'ordered_cat', 'unordered_cat', 'dropped'."""

    name: str
    """Original column name (for error messages)."""

    categories: list[Any] | None = None
    """Declared/observed category list for ordered_cat / unordered_cat."""

    dropped_dtype: str | None = None
    """Stringified dtype that caused the column to be dropped."""


def _ordinal_encode(values: np.ndarray, categories: list, name: str) -> np.ndarray:
    """Map `values` to integer positions in `categories`; raise on unseen."""
    table = {c: i for i, c in enumerate(categories)}
    out = np.empty(len(values), dtype=np.float64)
    unseen: list = []
    for i, v in enumerate(values):
        code = table.get(v, -1)
        if code < 0:
            unseen.append(v)
        else:
            out[i] = code
    if unseen:
        _raise_unseen(name, unseen, categories)
    return out.reshape(-1, 1)


def _one_hot_encode(values: np.ndarray, categories: list, name: str) -> np.ndarray:
    """Build a ``(n, k)`` one-hot matrix using `categories` order; raise on unseen."""
    table = {c: i for i, c in enumerate(categories)}
    n = len(values)
    k = len(categories)
    out = np.zeros((n, k), dtype=np.float64)
    unseen: list = []
    for i, v in enumerate(values):
        code = table.get(v, -1)
        if code < 0:
            unseen.append(v)
        else:
            out[i, code] = 1.0
    if unseen:
        _raise_unseen(name, unseen, categories)
    return out


def _raise_unseen(name: str, unseen: list, known: list) -> None:
    uniq = sorted({repr(v) for v in unseen})
    msg = (
        f'column {name!r}: {len(unseen)} value(s) at transform time are not in'
        f' the fitted category list; unseen sample: {uniq[:_UNSEEN_PREVIEW]};'
        f' known categories: {list(known)[:_UNSEEN_PREVIEW]}'
    )
    raise ValueError(msg)


def _polars_encode(
    pl: PolarsModule,
    series: Series,
    categories: list,
    name: str,
    *,
    mode: Literal['ordinal', 'one_hot'],
) -> np.ndarray:
    """Validate via cast to `pl.Enum(categories)` and encode via polars APIs.

    Polars's `Enum` cast natively raises on any value not in `categories`, and
    `to_physical` returns the integer codes in the declared-category order. The
    `np.eye` index is the only numpy bit and is just an identity-matrix lookup;
    the categorical bookkeeping itself stays inside polars.
    """
    cats = list(categories)
    try:
        coded = series.cast(pl.Enum(cats))
    except pl.exceptions.InvalidOperationError:
        # Identify the actual unseen values for a friendly error. is_in works
        # against a Python list and is short-circuited by polars.
        is_known = series.is_in(cats).fill_null(value=False)
        unseen = series.filter(~is_known).drop_nulls().unique().sort().to_list()
        _raise_unseen(name, unseen, cats)
        raise  # unreachable, kept to satisfy the type checker
    if coded.null_count():
        msg = f'column {name!r}: null values are not supported in categorical columns'
        raise ValueError(msg)
    codes = coded.to_physical().to_numpy()
    if mode == 'ordinal':
        return codes.astype(np.float64).reshape(-1, 1)
    return np.eye(len(cats), dtype=np.float64)[codes]


def _expand_variable_weights(
    weights: np.ndarray | None, original_var_indices: list[int], n_orig: int
) -> np.ndarray | None:
    """Split each original weight evenly across its one-hot expansions."""
    if weights is None:
        return None
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n_orig,):
        msg = (
            f'variable_weights must have shape ({n_orig},) matching the number'
            f' of original columns; got {w.shape}'
        )
        raise ValueError(msg)
    if not original_var_indices:
        return np.empty((0,), dtype=np.float64)
    counts = np.bincount(np.asarray(original_var_indices), minlength=n_orig)
    return np.array([w[j] / counts[j] for j in original_var_indices], dtype=np.float64)


def _stack(cols: list[np.ndarray], n_rows: int) -> np.ndarray:
    if not cols:
        return np.empty((n_rows, 0), dtype=np.float64)
    return np.concatenate(cols, axis=1)


class _PreprocessorBase:
    """Common state for `PandasPreprocessor` and `PolarsPreprocessor`."""

    def __init__(self) -> None:
        self._fitted: bool = False
        self._specs: list[_ColumnSpec] = []
        self._original_var_indices: list[int] = []

    @property
    def fitted(self) -> bool:
        """Whether `fit_transform` has been called."""
        return self._fitted

    @property
    def n_original_columns(self) -> int:
        """Number of columns in the dataframe given to `fit_transform`."""
        return len(self._specs)

    @property
    def n_processed_columns(self) -> int:
        """Number of columns in the matrix returned by `fit_transform` / `transform`."""
        return len(self._original_var_indices)

    @property
    def original_var_indices(self) -> list[int]:
        """For each output column, the index of the original column it came from."""
        return list(self._original_var_indices)

    def _check_fitted(self) -> None:
        if not self._fitted:
            msg = 'preprocessor has not been fitted yet; call fit_transform first'
            raise RuntimeError(msg)

    def _check_n_columns(self, n_cols: int) -> None:
        if n_cols != len(self._specs):
            msg = (
                f'transform input has {n_cols} columns; preprocessor was fitted'
                f' on {len(self._specs)} columns'
            )
            raise ValueError(msg)

    def _warn_dropped(self, dropped: list[tuple[str, str]]) -> None:
        if not dropped:
            return
        pretty = ', '.join(f'{n!r} ({d})' for n, d in dropped)
        warnings.warn(
            f'unsupported column dtypes (will be ignored): {pretty}', stacklevel=3
        )


class PandasPreprocessor(_PreprocessorBase):
    """Stochtree-style covariate preprocessor for `pandas.DataFrame` inputs."""

    def fit_transform(
        self, X: DataFrame, *, variable_weights: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Fit on `X` and return ``(X_processed, variable_weights_expanded)``."""
        import pandas as pd  # noqa: PLC0415  # optional runtime dependency

        self._specs = []
        self._original_var_indices = []
        cols_out: list[np.ndarray] = []
        dropped: list[tuple[str, str]] = []
        for orig_idx, name in enumerate(X.columns):
            spec, arr = self._fit_column(pd, X[name], str(name))
            self._specs.append(spec)
            if arr is None:
                dropped.append((spec.name, spec.dropped_dtype or '?'))
                continue
            cols_out.append(arr)
            self._original_var_indices.extend([orig_idx] * arr.shape[1])
        self._warn_dropped(dropped)
        self._fitted = True
        return (
            _stack(cols_out, X.shape[0]),
            _expand_variable_weights(
                variable_weights, self._original_var_indices, len(self._specs)
            ),
        )

    def transform(self, X: DataFrame) -> np.ndarray:
        """Apply the fitted transformation to a new `pandas.DataFrame`."""
        self._check_fitted()
        self._check_n_columns(X.shape[1])
        cols_out: list[np.ndarray] = []
        for orig_idx, spec in enumerate(self._specs):
            if spec.kind == 'dropped':
                continue
            series = X.iloc[:, orig_idx]
            cols_out.append(self._transform_column(series, spec))
        return _stack(cols_out, X.shape[0])

    @staticmethod
    def _fit_column(
        pd: PandasModule, series: Series, name: str
    ) -> tuple[_ColumnSpec, np.ndarray | None]:
        dt = series.dtype
        if isinstance(dt, pd.CategoricalDtype):
            cats = list(dt.categories)
            values = series.to_numpy()
            if dt.ordered:
                return (
                    _ColumnSpec('ordered_cat', name, categories=cats),
                    _ordinal_encode(values, cats, name),
                )
            return (
                _ColumnSpec('unordered_cat', name, categories=cats),
                _one_hot_encode(values, cats, name),
            )
        if pd.api.types.is_bool_dtype(dt):
            return (
                _ColumnSpec('bool', name),
                series.to_numpy().astype(np.float64).reshape(-1, 1),
            )
        if pd.api.types.is_numeric_dtype(dt):
            return (
                _ColumnSpec('numeric', name),
                series.to_numpy().astype(np.float64).reshape(-1, 1),
            )
        if dt.kind == 'O' or pd.api.types.is_string_dtype(dt):
            values = series.to_numpy()
            cats = sorted(set(values))
            return (
                _ColumnSpec('unordered_cat', name, categories=cats),
                _one_hot_encode(values, cats, name),
            )
        return (_ColumnSpec('dropped', name, dropped_dtype=str(dt)), None)

    @staticmethod
    def _transform_column(series: Series, spec: _ColumnSpec) -> np.ndarray:
        values = series.to_numpy()
        if spec.kind == 'ordered_cat':
            return _ordinal_encode(values, spec.categories or [], spec.name)
        if spec.kind == 'unordered_cat':
            return _one_hot_encode(values, spec.categories or [], spec.name)
        return values.astype(np.float64).reshape(-1, 1)


class PolarsPreprocessor(_PreprocessorBase):
    """Stochtree-style covariate preprocessor for `polars.DataFrame` inputs."""

    def fit_transform(
        self, X: DataFrame, *, variable_weights: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Fit on `X` and return ``(X_processed, variable_weights_expanded)``."""
        import polars as pl  # noqa: PLC0415  # optional runtime dependency

        self._specs = []
        self._original_var_indices = []
        cols_out: list[np.ndarray] = []
        dropped: list[tuple[str, str]] = []
        for orig_idx, name in enumerate(X.columns):
            spec, arr = self._fit_column(pl, X[name], str(name))
            self._specs.append(spec)
            if arr is None:
                dropped.append((spec.name, spec.dropped_dtype or '?'))
                continue
            cols_out.append(arr)
            self._original_var_indices.extend([orig_idx] * arr.shape[1])
        self._warn_dropped(dropped)
        self._fitted = True
        return (
            _stack(cols_out, X.shape[0]),
            _expand_variable_weights(
                variable_weights, self._original_var_indices, len(self._specs)
            ),
        )

    def transform(self, X: DataFrame) -> np.ndarray:
        """Apply the fitted transformation to a new `polars.DataFrame`."""
        import polars as pl  # noqa: PLC0415  # optional runtime dependency

        self._check_fitted()
        self._check_n_columns(X.shape[1])
        cols_out: list[np.ndarray] = []
        for orig_idx, spec in enumerate(self._specs):
            if spec.kind == 'dropped':
                continue
            series = X[X.columns[orig_idx]]
            cols_out.append(self._transform_column(pl, series, spec))
        return _stack(cols_out, X.shape[0])

    @staticmethod
    def _fit_column(
        pl: PolarsModule, series: Series, name: str
    ) -> tuple[_ColumnSpec, np.ndarray | None]:
        dt = series.dtype
        if isinstance(dt, pl.Enum):
            cats = dt.categories.to_list()
            return (
                _ColumnSpec('ordered_cat', name, categories=cats),
                _polars_encode(pl, series, cats, name, mode='ordinal'),
            )
        if isinstance(dt, pl.Categorical) or dt == pl.String:
            # NOTE: post the polars Categorical refactor (PR #23016),
            # `series.cat.get_categories()` returns the shared global
            # `Categories` object — not the per-column observed values — so we
            # extract observed values via `unique()`. Sorted for deterministic
            # column ordering across runs (`unique()` order is non-deterministic).
            cats = sorted(series.drop_nulls().unique().to_list())
            return (
                _ColumnSpec('unordered_cat', name, categories=cats),
                _polars_encode(pl, series, cats, name, mode='one_hot'),
            )
        if dt == pl.Boolean:
            return (
                _ColumnSpec('bool', name),
                series.to_numpy().astype(np.float64).reshape(-1, 1),
            )
        if dt.is_numeric():
            return (
                _ColumnSpec('numeric', name),
                series.to_numpy().astype(np.float64).reshape(-1, 1),
            )
        return (_ColumnSpec('dropped', name, dropped_dtype=str(dt)), None)

    @staticmethod
    def _transform_column(
        pl: PolarsModule, series: Series, spec: _ColumnSpec
    ) -> np.ndarray:
        if spec.kind == 'ordered_cat':
            return _polars_encode(
                pl, series, spec.categories or [], spec.name, mode='ordinal'
            )
        if spec.kind == 'unordered_cat':
            return _polars_encode(
                pl, series, spec.categories or [], spec.name, mode='one_hot'
            )
        return series.to_numpy().astype(np.float64).reshape(-1, 1)


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
