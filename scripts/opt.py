# bartz/scripts/opt.py
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

"""Clock `mcmcstep.step` over a grid of hyperparameters defined in a config file.

The config file is a JSONC document like:

.. code-block:: jsonc

    {
        "plot": {                                   // plot-only metadata
            "scan":   "n",                          // x-axis param
            "reduce": "resid_reduction"             // min'd over (the "optimal" curve)
        },
        "values": {                                 // value list for every param
            "n":                  [1024, 4096, 16384],
            "k":                  [null, 2],
            "maxdepth":           [6],
            "weights":            [false],
            "num_trees":          [5, 50, 200],
            "resid_reduction": [                    // reduction slot: one entry per kind
                {"kind": "batched", "num_batches": [null, 1, 8, 64]},
                {"kind": "onehot", "method": ["matmul", "multiply"]},
                "pallas"                            // short for {"kind": "pallas"}
            ],
            "sequential_unroll":  [1, 2, 4],
            "num_chains":         [null, 4]
        }
    }

``values`` may list any subset of the fields of `ConfigParams`; fields whose
name matches a parameter of `init` with a default are filled in with that
default if omitted.

The reduction slots (`ConfigParams.reduction_field_names`) hold `ReductionConfig`
instances and use a dedicated syntax: a list of entries, each selecting a
reduction algorithm through its ``kind`` tag (the `ReductionConfig` subclass
name, lowercased and stripped of the 'Reduction' suffix) plus per-knob value
lists (a scalar stands for a singleton list, omitted knobs take the class
defaults). Each entry expands to the grid over its knob lists; the slot's
values are the concatenation across entries, deduplicated.

``plot.scan`` and ``plot.reduce`` name the x-axis and the minimized-over axis.
Each must be a field of `ConfigParams` or, for a reduction slot with a single
entry, a knob of that entry addressed as ``"<slot>.<knob>"``. The legend /
multi-line dimensions (the "matrix") are derived automatically as the
multi-valued axes other than scan and reduce, where a single-entry slot
contributes its multi-valued knobs as dotted axes while a multi-entry slot
counts as one categorical axis. The matrix list is written into the output
config under ``plot.matrix``, together with ``plot.drop`` (result columns
redundant for plotting) and ``plot.sentinels`` (integer columns where -1
encodes `None` and -2 encodes ``'auto'``), for downstream tools.
"""

import inspect
import json
import os
import subprocess
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Callable, Iterator
from dataclasses import MISSING, dataclass, field, fields, replace

# WORKAROUND(python<3.11): use datetime.UTC and enum.StrEnum
from datetime import datetime, timezone
from enum import Enum
from gc import collect
from itertools import product
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any, Literal, get_args

import json5
from jax import Array, block_until_ready, random
from jax import config as jax_config
from jax import numpy as jnp
from jax.errors import JaxRuntimeError
from jaxtyping import Shaped
from polars import DataFrame, concat, read_parquet
from tqdm import tqdm

from bartz._jaxext import get_default_device
from bartz.mcmcstep import (
    BatchedReduction,
    OneHotReduction,
    PallasReduction,
    ReductionConfig,
    State,
    init,
    make_p_nonterminal,
    step,
)
from bartz.mcmcstep._reduction import _ceil_pow2
from bartz.testing import gen_data

# Rough element-count budgets used by `ConfigParams.is_valid` to skip
# combinations that would materialize huge intermediates.
MAX_LEAF_INDICES_SIZE = 2**30
MAX_ONEHOT_SIZE = 2**30


def init_default(name: str) -> Any:  # noqa: ANN401
    """Default value of parameter ``name`` in `init`'s signature."""
    default = inspect.signature(init).parameters[name].default
    if default is inspect.Parameter.empty:
        msg = f'`init` parameter {name!r} has no default'
        raise ValueError(msg)
    return default


def read_jax_config(name: str) -> Any:  # noqa: ANN401
    """Read a dynamically-defined `jax.config` option."""
    return getattr(jax_config, name)


# WORKAROUND(python<3.11): subclass StrEnum and use auto() values
class Logging(str, Enum):
    """Logging mode for benchmark runs."""

    no = 'no'
    pbar = 'pbar'
    results = 'results'

    __str__ = str.__str__  # what StrEnum does, keeps argparse --help readable


def _kind_name(cls: type[ReductionConfig]) -> str:
    """Config-file tag of a `ReductionConfig` subclass, derived from its name."""
    return cls.__name__.removesuffix('Reduction').lower()


# Map config-file kind tag to ReductionConfig subclass; new subclasses exported
# by bartz.mcmcstep become available in config files automatically.
REDUCTION_KINDS: dict[str, type[ReductionConfig]] = {
    _kind_name(cls): cls for cls in ReductionConfig.__subclasses__()
}


def _reduction_label(cfg: ReductionConfig) -> str:
    """Compact label of `cfg`: kind tag plus the non-default knobs."""
    knobs = ', '.join(
        f'{f.name}={getattr(cfg, f.name)}'
        for f in fields(cfg)
        if getattr(cfg, f.name) != f.default
    )
    return f'{_kind_name(type(cfg))}({knobs})'


def _spec_from_reduction(cfg: ReductionConfig) -> list[dict[str, Any]]:
    """Slot spec equivalent to the single config `cfg`, listing the non-default knobs."""
    entry: dict[str, Any] = {'kind': _kind_name(type(cfg))}
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        if value != f.default:
            entry[f.name] = [value]
    return [entry]


def _normalize_slot_spec(
    path: Path, name: str, spec: list[Any]
) -> list[dict[str, Any]]:
    """Validate a reduction-slot spec and normalize knob values to lists."""
    out: list[dict[str, Any]] = []
    for raw_entry in spec:
        entry = {'kind': raw_entry} if isinstance(raw_entry, str) else raw_entry
        if not isinstance(entry, dict) or 'kind' not in entry:
            msg = (
                f'config {path}: "values[{name!r}]" entries must be kind strings '
                f'or objects with a "kind" key'
            )
            raise TypeError(msg)
        cls = REDUCTION_KINDS.get(entry['kind'])
        if cls is None:
            msg = (
                f'config {path}: "values[{name!r}]": unknown kind '
                f'{entry["kind"]!r}. Allowed: {sorted(REDUCTION_KINDS)}.'
            )
            raise ValueError(msg)
        allowed = {f.name for f in fields(cls)}
        extra = entry.keys() - allowed - {'kind'}
        if extra:
            msg = (
                f'config {path}: "values[{name!r}]" {entry["kind"]} entry has '
                f'unknown knobs: {sorted(extra)}. Allowed: {sorted(allowed)}.'
            )
            raise ValueError(msg)
        norm: dict[str, Any] = {'kind': entry['kind']}
        for f in fields(cls):
            if f.name in entry:
                value = entry[f.name]
                values = value if isinstance(value, list) else [value]
                if not values:
                    msg = (
                        f'config {path}: "values[{name!r}]" {entry["kind"]} knob '
                        f'{f.name!r} must be a non-empty list'
                    )
                    raise ValueError(msg)
                norm[f.name] = values
        out.append(norm)
    return out


def _expand_slot_spec(spec: list[dict[str, Any]]) -> list[ReductionConfig]:
    """Expand a normalized slot spec into the list of distinct config instances."""
    variants: list[ReductionConfig] = []
    for entry in spec:
        cls = REDUCTION_KINDS[entry['kind']]
        knobs = {k: v for k, v in entry.items() if k != 'kind'}
        variants.extend(
            cls(**dict(zip(knobs, combo, strict=True)))
            for combo in product(*knobs.values())
        )
    return list(dict.fromkeys(variants))


def _slot_kinds(spec: list[dict[str, Any]]) -> tuple[type[ReductionConfig], ...]:
    """Distinct config classes appearing in a slot's expansion, in order."""
    return tuple(dict.fromkeys(type(cfg) for cfg in _expand_slot_spec(spec)))


def _slot_knobs(spec: list[dict[str, Any]]) -> tuple[str, ...]:
    """Union of knob names across the kinds in a slot's expansion, in order."""
    return tuple(
        dict.fromkeys(f.name for cls in _slot_kinds(spec) for f in fields(cls))
    )


def _knob_has_sentinels(tp: Any) -> bool:  # noqa: ANN401
    """Whether a knob type admits `None` or ``'auto'``, which are sentinel-encoded."""
    args = get_args(tp)
    return (
        type(None) in args or 'auto' in args or any('auto' in get_args(a) for a in args)
    )


# ConfigParams fields (other than the reduction slots, handled by their type)
# whose values are sentinel-encoded when saved.
_SENTINEL_FIELDS = ('num_chains', 'prec_count_num_trees')


def _encode_knob(value: Any) -> Any:  # noqa: ANN401
    """Encode a value for the results DataFrame: `None` -> -1, ``'auto'`` -> -2.

    Keeps the columns purely numeric (no mixed int/str, no nulls colliding with
    missing data) so downstream tooling sees a stable Int64 dtype.
    """
    if value is None:
        return -1
    elif value == 'auto':
        return -2
    else:
        return value


def _pallas_is_valid(cfg: PallasReduction, n: int) -> bool:
    """Whether `cfg` runs on the default device with `n` datapoints."""
    device = get_default_device()
    if cfg.backend == 'cpu':
        if device.platform != 'cpu':
            return False
    elif device.platform != 'gpu':
        return False
    if cfg.backend == 'default':
        # Mosaic GPU lowers only on Hopper+ (compute capability 9.0)
        cc = getattr(device, 'compute_capability', None)
        if cc is None or float(cc) < 9.0:
            return False
    if isinstance(cfg.block_size, int):
        if cfg.block_size > _ceil_pow2(n):
            return False
        if device.platform == 'gpu' and cfg.block_size & (cfg.block_size - 1):
            return False
    return not (isinstance(cfg.num_blocks, int) and not 1 <= cfg.num_blocks <= n)


@dataclass(frozen=True)
class ConfigParams:
    """One resolved combination of parameters, as named in the config file.

    A field declared with ``dataclasses.field(metadata={'restart': True})`` is
    "restart-tier": changing its value requires restarting Python (e.g. it
    affects env vars or import-time configuration). The benchmark loop is
    two-tiered: the outer loop iterates over restart-tier values in order, and
    the inner loop randomizes only over the remaining fields.

    A field declared with ``metadata={'reduction': True}`` is a reduction slot:
    it holds a `ReductionConfig` instance and uses the kind-tagged slot syntax
    in config files.
    """

    n: int
    """Number of data points."""

    k: int | None
    """Multivariate outcome dimension; ``None`` for univariate."""

    maxdepth: int
    """Maximum tree depth; passed through `make_p_nonterminal`."""

    weights: bool
    """Whether to set `init`'s ``error_scale`` to ``jnp.ones(n)``."""

    num_trees: int
    """`init`'s ``num_trees`` kwarg."""

    resid_reduction: ReductionConfig = field(
        default=init_default('resid_reduction_config'), metadata={'reduction': True}
    )
    """`init`'s ``resid_reduction_config`` kwarg."""

    count_reduction: ReductionConfig = field(
        default=init_default('count_reduction_config'), metadata={'reduction': True}
    )
    """`init`'s ``count_reduction_config`` kwarg."""

    prec_reduction: ReductionConfig = field(
        default=init_default('prec_reduction_config'), metadata={'reduction': True}
    )
    """`init`'s ``prec_reduction_config`` kwarg."""

    prec_count_num_trees: int | None | Literal['auto'] = init_default(
        'prec_count_num_trees'
    )
    """`init`'s ``prec_count_num_trees`` kwarg."""

    sequential_unroll: int | bool = init_default('sequential_unroll')
    """`init`'s ``sequential_unroll`` kwarg."""

    num_chains: int | None = init_default('num_chains')
    """`init`'s ``num_chains`` kwarg."""

    optimization_level: str | None = field(
        default=read_jax_config('jax_optimization_level'), metadata={'jax_config': True}
    )
    """JAX ``jax_optimization_level``; one of ``'O0'``..``'O3'``. ``None`` in the config resolves to the JAX default (typically ``'UNKNOWN'``) at load time."""

    exec_time_optimization_effort: float | None = field(
        default=read_jax_config('jax_exec_time_optimization_effort'),
        metadata={'jax_config': True},
    )
    """JAX ``jax_exec_time_optimization_effort``; float in ``[-1.0, 1.0]``. ``None`` in the config resolves to the JAX default (typically ``0.0``) at load time."""

    memory_fitting_level: str | None = field(
        default=read_jax_config('jax_memory_fitting_level'),
        metadata={'jax_config': True},
    )
    """JAX ``jax_memory_fitting_level``; one of ``'O0'``..``'O3'``. ``None`` in the config resolves to the JAX default (typically ``'O2'``) at load time."""

    memory_fitting_effort: float | None = field(
        default=read_jax_config('jax_memory_fitting_effort'),
        metadata={'jax_config': True},
    )
    """JAX ``jax_memory_fitting_effort``; float in ``[-1.0, 1.0]``. ``None`` in the config resolves to the JAX default (typically ``0.0``) at load time."""

    enable_pgle: bool | None = field(
        default=read_jax_config('jax_enable_pgle'), metadata={'jax_config': True}
    )
    """JAX ``jax_enable_pgle``; whether to enable Profile-Guided Latency Estimation. ``None`` in the config resolves to the JAX default (typically ``False``) at load time."""

    gpu_autotune_level: int | None = field(default=None, metadata={'restart': True})
    """XLA ``--xla_gpu_autotune_level`` (integer); ``None`` to leave the env var untouched."""

    cpu_use_thunk_runtime: bool | None = field(default=None, metadata={'restart': True})
    """XLA ``--xla_cpu_use_thunk_runtime`` (bool); ``None`` to leave the env var untouched."""

    cpu_enable_fast_math: bool | None = field(default=None, metadata={'restart': True})
    """XLA ``--xla_cpu_enable_fast_math`` (bool); ``None`` to leave the env var untouched."""

    chain_axis: int | None = field(default=None, metadata={'restart': True})
    """bartz ``CHAIN_AXIS`` env var (int); ``None`` to leave the env var untouched."""

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Names of every field, in declaration order."""
        return tuple(f.name for f in fields(cls))

    @classmethod
    def restart_field_names(cls) -> tuple[str, ...]:
        """Fields marked ``restart=True``; changing one requires restarting Python."""
        return tuple(f.name for f in fields(cls) if f.metadata.get('restart'))

    @classmethod
    def jax_config_field_names(cls) -> tuple[str, ...]:
        """Fields marked ``jax_config=True``; applied via `jax_config.update`."""
        return tuple(f.name for f in fields(cls) if f.metadata.get('jax_config'))

    @classmethod
    def reduction_field_names(cls) -> tuple[str, ...]:
        """Fields holding a `ReductionConfig`; they use the slot syntax in configs."""
        return tuple(f.name for f in fields(cls) if f.metadata.get('reduction'))

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        """Map of field name to default value, for fields that have one."""
        return {f.name: f.default for f in fields(cls) if f.default is not MISSING}

    def is_valid(self) -> bool:
        """Whether this combination of values is admissible."""
        chains = 1 if self.num_chains is None else self.num_chains
        k = 1 if self.k is None else self.k

        # number of trees processed at once in the count/prec pass
        if self.prec_count_num_trees == 'auto':
            # mirrors the self-limiting budget of `init`'s auto resolution
            trees_at_once = max(
                1, min(self.num_trees, 2**27 // max(1, self.n * chains))
            )
        elif self.prec_count_num_trees is None:
            trees_at_once = self.num_trees
        elif self.prec_count_num_trees > self.num_trees:
            return False
        else:
            trees_at_once = self.prec_count_num_trees

        # the count/prec pass materializes the leaf indices of a batch of trees
        if chains * trees_at_once * self.n > MAX_LEAF_INDICES_SIZE:
            return False

        # rough sizings of the reduce inputs: the number of one-hot matrices
        # (one per array of indices) and of value rows contracted against them
        reductions = (
            (self.resid_reduction, chains, chains * k),
            (self.count_reduction, chains * trees_at_once, chains * trees_at_once),
            (self.prec_reduction, chains * trees_at_once, chains * trees_at_once),
        )
        return all(
            self._reduction_is_valid(cfg, idx_rows, val_rows)
            for cfg, idx_rows, val_rows in reductions
        )

    def _reduction_is_valid(
        self, cfg: ReductionConfig, idx_rows: int, val_rows: int
    ) -> bool:
        """Whether `cfg` is admissible for this combination of values."""
        if isinstance(cfg, BatchedReduction):
            return not (isinstance(cfg.num_batches, int) and cfg.num_batches > self.n)
        elif isinstance(cfg, OneHotReduction):
            # 'matmul' materializes one (size, n) one-hot per indices array; the
            # other methods may materialize the full (rows, size, n) buffer
            rows = idx_rows if cfg.method == 'matmul' else val_rows
            return rows * 2**self.maxdepth * self.n <= MAX_ONEHOT_SIZE
        elif isinstance(cfg, PallasReduction):
            return _pallas_is_valid(cfg, self.n)
        else:
            return True

    def to_init_kwargs(self) -> 'InitKwargs':
        """Translate this combination into kwargs for `init`."""
        # gen_data requires p >= k, but the benchmarks use a single predictor:
        # generate with p = k and keep only the first predictor
        data = gen_data(
            random.key(2026_06_07),
            n=self.n,
            p=1 if self.k is None else self.k,
            k=self.k,
            q=0,
            lambda_=None if self.k is None else 0.5,
            sigma2_lin=1.0,
            sigma2_quad=1.0,
            sigma2_eps=1.0,
        ).quantize()
        eye = 1.0 if self.k is None else jnp.eye(self.k)
        return InitKwargs(
            X=data.x[:1, :],
            y=data.y,
            offset=jnp.zeros(data.y.shape[:-1]),
            max_split=data.max_split[:1],
            num_trees=self.num_trees,
            p_nonterminal=make_p_nonterminal(self.maxdepth, 0.95, 2),
            leaf_prior_cov_inv=self.num_trees * eye,
            error_cov_df=2.0,
            error_cov_scale=2 * eye,
            error_scale=jnp.ones(self.n) if self.weights else None,
            resid_reduction_config=self.resid_reduction,
            count_reduction_config=self.count_reduction,
            prec_reduction_config=self.prec_reduction,
            prec_count_num_trees=self.prec_count_num_trees,
            sequential_unroll=self.sequential_unroll,
            num_chains=self.num_chains,
        )

    def apply_jax_config(self) -> None:
        """Apply non-restart JAX config flags from this combo."""
        for name in self.jax_config_field_names():
            jax_config.update(f'jax_{name}', getattr(self, name))

    def __str__(self) -> str:
        parts = []
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, ReductionConfig):
                value = _reduction_label(value)
            parts.append(f'{f.name}={value}')
        return ' '.join(parts)


@dataclass(frozen=True)
class InitKwargs:
    """Keyword arguments for `bartz.mcmcstep.init`, in init's signature order."""

    X: Shaped[Array, '...']
    y: Shaped[Array, '...']
    offset: Shaped[Array, '...']
    max_split: Shaped[Array, '...']
    num_trees: int
    p_nonterminal: Shaped[Array, '...']
    leaf_prior_cov_inv: float | Shaped[Array, '...']
    error_cov_df: float
    error_cov_scale: float | Shaped[Array, '...']
    error_scale: Shaped[Array, '...'] | None
    resid_reduction_config: ReductionConfig
    count_reduction_config: ReductionConfig
    prec_reduction_config: ReductionConfig
    prec_count_num_trees: int | None | Literal['auto']
    sequential_unroll: int | bool
    num_chains: int | None

    def init(self) -> State:
        """Call `mcmcstep.init` with these arguments."""
        return init(**{f.name: getattr(self, f.name) for f in fields(self)})


def _xla_flags_for_restart(restart_values: dict[str, Any]) -> str:
    """Build the `XLA_FLAGS` contribution for the XLA-related restart-tier values."""
    flags: list[str] = []
    v = restart_values.get('gpu_autotune_level')
    if v is not None:
        flags.append(f'--xla_gpu_autotune_level={v}')
    v = restart_values.get('cpu_use_thunk_runtime')
    if v is not None:
        flags.append(f'--xla_cpu_use_thunk_runtime={"true" if v else "false"}')
    v = restart_values.get('cpu_enable_fast_math')
    if v is not None:
        flags.append(f'--xla_cpu_enable_fast_math={"true" if v else "false"}')
    return ' '.join(flags)


class _InlineArraysJSON5Encoder(json5.JSON5Encoder):
    """JSON5 encoder that renders arrays of scalars on a single line."""

    def _encode_array(self, obj: Any, seen: set, level: int) -> str:  # noqa: ANN401
        if any(isinstance(el, (dict, list)) for el in obj):
            return super()._encode_array(obj, seen, level)
        if not obj:
            return '[]'
        return (
            '['
            + ', '.join(self.encode(el, seen, level, as_key=False) for el in obj)
            + ']'
        )


@dataclass(frozen=True)
class PlotConfig:
    """Plot-only metadata: which params take on x-axis / legend / reduce roles."""

    scan: str
    """Axis plotted on the x-axis."""

    reduce: str
    """Axis minimised over to build the 'optimal' curve."""

    matrix: tuple[str, ...]
    """Axes used as legend / multi-line dimensions.

    Derived from ``values``: axes (other than ``scan`` and ``reduce``) that
    take on more than one value, where a single-entry reduction slot
    contributes its multi-valued knobs as dotted axes and a multi-entry slot
    counts as one categorical axis.
    """

    drop: tuple[str, ...]
    """Result columns redundant for plotting, to be dropped by downstream tools.

    Per-knob columns of a slot whose label column plays a role, or label
    columns superseded by their per-knob columns.
    """

    sentinels: tuple[str, ...]
    """Integer result columns where -1 encodes `None` and -2 encodes ``'auto'``."""


def _role_axes(values: dict[str, list[Any]]) -> set[str]:
    """Names usable as ``plot.scan``/``plot.reduce`` given normalized ``values``."""
    axes = set(values)
    for name in ConfigParams.reduction_field_names():
        spec = values[name]
        if len(spec) == 1:
            axes |= {f'{name}.{knob}' for knob in spec[0] if knob != 'kind'}
    return axes


def _derive_matrix(
    values: dict[str, list[Any]], scan: str, reduce: str
) -> tuple[str, ...]:
    """Derive the legend / multi-line axes from normalized ``values``."""
    matrix: list[str] = []
    for name, vals in values.items():
        if name in (scan, reduce):
            continue
        if name in ConfigParams.reduction_field_names():
            if len(vals) == 1:
                matrix += [
                    f'{name}.{knob}'
                    for knob, kvals in vals[0].items()
                    if knob != 'kind'
                    and f'{name}.{knob}' not in (scan, reduce)
                    and len(set(kvals)) > 1
                ]
            elif len(_expand_slot_spec(vals)) > 1:
                matrix.append(name)
        elif len(vals) > 1:
            matrix.append(name)
    return tuple(matrix)


def _derive_drop(values: dict[str, list[Any]], roles: set[str]) -> tuple[str, ...]:
    """List the result columns redundant for plotting.

    For each reduction slot: if its label column plays a role, the per-knob
    columns are redundant; if instead some per-knob column plays a role, the
    label column is; otherwise the slot is fixed to a single variant and the
    label column summarizes the per-knob ones.
    """
    drop: list[str] = []
    for name in ConfigParams.reduction_field_names():
        children = [
            f'{name}.kind',
            *(f'{name}.{knob}' for knob in _slot_knobs(values[name])),
        ]
        if name in roles:
            drop += [c for c in children if c not in roles]
        elif any(c in roles for c in children):
            drop.append(name)
        else:
            drop += children
    return tuple(drop)


def _derive_sentinels(values: dict[str, list[Any]]) -> tuple[str, ...]:
    """List the sentinel-encoded result columns."""
    cols = list(_SENTINEL_FIELDS)
    for name in ConfigParams.reduction_field_names():
        for cls in _slot_kinds(values[name]):
            for f in fields(cls):
                col = f'{name}.{f.name}'
                if col not in cols and _knob_has_sentinels(f.type):
                    cols.append(col)
    return tuple(cols)


@dataclass(frozen=True)
class Config:
    """Parsed and validated config file content."""

    plot: PlotConfig
    """Plot-only metadata (x-axis, reduce, matrix roles)."""

    values: dict[str, list[Any]]
    """Per-field list of values to enumerate; keys = `ConfigParams.field_names()`.

    Reduction slots hold the normalized spec (list of kind-tagged entries with
    per-knob value lists) rather than the expanded `ReductionConfig` instances;
    see `axis_values`.
    """

    @classmethod
    def load(cls, path: Path) -> 'Config':
        """Load and validate a JSONC config file."""
        with path.open() as f:
            raw = json5.load(f)
        cls._check_schema(path, raw)
        values = dict(raw['values'])
        cls._check_values_keys(path, values)

        # normalize the reduction slots to spec form, defaulting absent ones
        defaults = ConfigParams.defaults()
        for name in ConfigParams.reduction_field_names():
            if name in values:
                values[name] = _normalize_slot_spec(path, name, values[name])
            else:
                values[name] = _spec_from_reduction(defaults[name])

        plot_raw = raw['plot']
        cls._check_roles(
            path, [plot_raw['scan'], plot_raw['reduce']], _role_axes(values)
        )
        matrix = _derive_matrix(values, plot_raw['scan'], plot_raw['reduce'])
        roles = {plot_raw['scan'], plot_raw['reduce'], *matrix}
        # For JAX-config fields, an explicit ``null`` in the file means "use the
        # JAX default"; substitute it so ConfigParams carries concrete values.
        for name in ConfigParams.jax_config_field_names():
            if name in values:
                values[name] = [
                    defaults[name] if v is None else v for v in values[name]
                ]
        for name, default in defaults.items():
            if name not in ConfigParams.reduction_field_names():
                values.setdefault(name, [default])
        plot = PlotConfig(
            scan=plot_raw['scan'],
            reduce=plot_raw['reduce'],
            matrix=matrix,
            drop=_derive_drop(values, roles),
            sentinels=_derive_sentinels(values),
        )
        return cls(plot=plot, values=values)

    @staticmethod
    def _check_schema(path: Path, raw: dict[str, Any]) -> None:
        required = {'plot', 'values'}
        missing = required - raw.keys()
        if missing:
            msg = f'config {path}: missing required keys: {sorted(missing)}'
            raise ValueError(msg)
        extra = raw.keys() - required
        if extra:
            msg = f'config {path}: unknown top-level keys: {sorted(extra)}'
            raise ValueError(msg)
        Config._check_plot_schema(path, raw['plot'])
        if not isinstance(raw['values'], dict):
            msg = f'config {path}: "values" must be an object'
            raise TypeError(msg)
        for k, v in raw['values'].items():
            if not isinstance(v, list) or len(v) == 0:
                msg = f'config {path}: "values[{k!r}]" must be a non-empty list'
                raise TypeError(msg)

    @staticmethod
    def _check_plot_schema(path: Path, plot: object) -> None:
        if not isinstance(plot, dict):
            msg = f'config {path}: "plot" must be an object'
            raise TypeError(msg)
        required = {'scan', 'reduce'}
        missing = required - plot.keys()
        if missing:
            msg = f'config {path}: "plot" missing keys: {sorted(missing)}'
            raise ValueError(msg)
        extra = plot.keys() - required
        if extra:
            msg = f'config {path}: "plot" has unknown keys: {sorted(extra)}'
            raise ValueError(msg)
        if not isinstance(plot['scan'], str):  # ty: ignore[invalid-argument-type]
            msg = f'config {path}: "plot.scan" must be a string'
            raise TypeError(msg)
        if not isinstance(plot['reduce'], str):  # ty: ignore[invalid-argument-type]
            msg = f'config {path}: "plot.reduce" must be a string'
            raise TypeError(msg)

    @staticmethod
    def _check_values_keys(path: Path, values: dict[str, list[Any]]) -> None:
        allowed = set(ConfigParams.field_names())
        optional = set(ConfigParams.defaults())
        required = allowed - optional
        missing = required - values.keys()
        if missing:
            msg = f'config {path}: "values" missing params: {sorted(missing)}'
            raise ValueError(msg)
        extra = values.keys() - allowed
        if extra:
            msg = (
                f'config {path}: "values" has unknown params: {sorted(extra)}. '
                f'Allowed: {sorted(allowed)}.'
            )
            raise ValueError(msg)

    @staticmethod
    def _check_roles(path: Path, named: list[str], known: set[str]) -> None:
        missing_role = [n for n in named if n not in known]
        if missing_role:
            msg = (
                f'config {path}: plot.scan/reduce names are not valid axes: '
                f'{missing_role}'
            )
            raise ValueError(msg)
        if len(named) != len(set(named)):
            counts: dict[str, int] = {}
            for n in named:
                counts[n] = counts.get(n, 0) + 1
            dups = sorted(n for n, c in counts.items() if c > 1)
            msg = f'config {path}: parameter(s) in multiple roles: {dups}'
            raise ValueError(msg)

    def minimal(self) -> 'Config':
        """Return a Config with each multi-element range truncated to (first, last).

        In reduction slots, every entry is kept and its knob ranges are
        truncated.
        """

        def trunc(vals: list[Any]) -> list[Any]:
            return [vals[0], vals[-1]] if len(vals) > 1 else list(vals)

        values: dict[str, list[Any]] = {}
        for name, vals in self.values.items():
            if name in ConfigParams.reduction_field_names():
                values[name] = [
                    {
                        knob: (kvals if knob == 'kind' else trunc(kvals))
                        for knob, kvals in entry.items()
                    }
                    for entry in vals
                ]
            else:
                values[name] = trunc(vals)
        return replace(self, values=values)

    def axis_values(self, name: str) -> list[Any]:
        """Values of axis ``name``, with reduction slots expanded to instances."""
        if name in ConfigParams.reduction_field_names():
            return _expand_slot_spec(self.values[name])
        else:
            return self.values[name]

    def result_columns(self) -> tuple[str, ...]:
        """Column names of the results DataFrame, in a deterministic order."""
        cols: list[str] = []
        for f in fields(ConfigParams):
            cols.append(f.name)
            if f.metadata.get('reduction'):
                cols.append(f'{f.name}.kind')
                cols.extend(
                    f'{f.name}.{knob}' for knob in _slot_knobs(self.values[f.name])
                )
        return (*cols, 'time_est', 'time_lo', 'time_up')

    def restart_iter(self) -> Iterator[dict[str, Any]]:
        """Yield restart-tier value dicts, in deterministic field-declaration order.

        Always yields at least once: an empty dict if no restart-tier fields exist.
        """
        names = [n for n in ConfigParams.restart_field_names() if n in self.values]
        if not names:
            yield {}
            return
        ranges = [self.values[n] for n in names]
        for vals in product(*ranges):
            yield dict(zip(names, vals, strict=True))

    def inner_iter(self, restart_values: dict[str, Any]) -> Iterator[ConfigParams]:
        """Yield valid `ConfigParams` for every non-restart combination with ``restart_values`` fixed."""
        names = [n for n in self.values if n not in restart_values]
        ranges = [self.axis_values(n) for n in names]
        for vals in product(*ranges):
            params = ConfigParams(
                **restart_values, **dict(zip(names, vals, strict=True))
            )
            if params.is_valid():
                yield params


class Benchmark:
    """Benchmark harness for one full MCMC step."""

    def setup(self, init_kwargs: InitKwargs) -> None:
        """Initialize BART state and warm up the MCMC step."""
        self.state: State = init_kwargs.init()
        self.key = random.key(2026_01_20_13_31)
        # PGLE profiles `jax_pgle_profiling_runs` executions then recompiles once
        # (slowly); warm up past that so every timed run is at steady state.
        n_warmup = 1
        if read_jax_config('jax_enable_pgle'):
            n_warmup += read_jax_config('jax_pgle_profiling_runs') + 1
        for _ in range(n_warmup):
            self.task()

    def task(self) -> None:
        """Run one full MCMC step."""
        self.state = block_until_ready(step(self.key, self.state))

    def teardown(self) -> None:
        """Drop state and clear compiled `step` cache."""
        del self.state
        step.clear_cache()
        # don't use jax.clear_caches() because it makes everything 3x slower
        collect()


def clock(func: Callable[[], None]) -> float:
    """Return elapsed seconds for a single function call."""
    start = perf_counter()
    func()
    return perf_counter() - start


def benchmark_loop(config: Config, args: Namespace, out_dir: Path) -> DataFrame:
    """Run the two-tiered benchmark scan.

    Outer loop: iterates restart-tier values in order, spawning a fresh Python
    subprocess for each non-empty restart-tier combination so that env vars and
    import-time configuration are picked up cleanly. The empty combination
    (i.e. no restart-tier fields in the config) runs inline. Inner loop
    (delegated to `inner_benchmark_loop` either inline or in a worker):
    randomizes the remaining combinations.
    """
    frames: list[DataFrame] = []
    restart_combos = list(config.restart_iter())
    n_restart = len(restart_combos)
    for i, restart_values in enumerate(restart_combos):
        if restart_values and args.logging != Logging.no:
            print(f'=== process set {i + 1}/{n_restart} ===', flush=True)
        if not restart_values:
            frames.append(inner_benchmark_loop(config, args, restart_values))
        else:
            partial = out_dir / f'_part_{i:03d}.parquet'
            cmd = [
                sys.executable,
                '-m',
                'scripts.opt',
                str(args.config),
                '--logging',
                args.logging.value,
                '--num',
                str(args.num),
                '--worker',
                json.dumps(restart_values),
                '--worker-out',
                str(partial),
            ]
            if args.minimal:
                cmd.append('--minimal')
            extra_flags = _xla_flags_for_restart(restart_values)
            chain_axis = restart_values.get('chain_axis')
            env = None
            if extra_flags or chain_axis is not None:
                env = dict(os.environ)
                if extra_flags:
                    existing = env.get('XLA_FLAGS', '')
                    env['XLA_FLAGS'] = f'{existing} {extra_flags}'.strip()
                if chain_axis is not None:
                    env['CHAIN_AXIS'] = str(chain_axis)
            subprocess.run(cmd, check=True, env=env)  # noqa: S603
            frames.append(read_parquet(partial))
    return concat(frames, how='vertical')


def inner_benchmark_loop(
    config: Config, args: Namespace, restart_values: dict[str, Any]
) -> DataFrame:
    """Run timing benchmarks over non-restart combinations, with ``restart_values`` fixed."""
    combinations = list(config.inner_iter(restart_values))
    Random(2026_05_20_10_19).shuffle(combinations)  # noqa: S311
    if args.logging == Logging.pbar:
        combinations = tqdm(combinations)

    slot_knobs = {
        name: _slot_knobs(config.values[name])
        for name in ConfigParams.reduction_field_names()
    }
    results: dict[str, list[Any]] = {name: [] for name in config.result_columns()}
    for params in combinations:
        if args.logging == Logging.results:
            print(f'{params}...', end='', flush=True)

        params.apply_jax_config()

        bench = Benchmark()
        try:
            bench.setup(params.to_init_kwargs())
        except JaxRuntimeError as e:
            if 'RESOURCE_EXHAUSTED: Out of memory while trying to allocate' in str(e):
                time_est = float('nan')
                time_lo = float('nan')
                time_up = float('nan')
            else:
                raise
        else:
            times = [clock(bench.task) for _ in range(args.num)]
            time_est = min(times)
            time_lo = min(times)
            time_up = min(times) + (max(times) - min(times)) / args.num

        bench.teardown()

        if args.logging == Logging.results:
            print(f' {time_est:#.2g} s')

        _save_params(results, params, slot_knobs)
        results['time_est'].append(time_est)
        results['time_lo'].append(time_lo)
        results['time_up'].append(time_up)

    return DataFrame(results)


def _save_params(
    results: dict[str, list[Any]],
    params: ConfigParams,
    slot_knobs: dict[str, tuple[str, ...]],
) -> None:
    """Append the values of `params` to the per-column `results` lists."""
    for f in fields(params):
        value = getattr(params, f.name)
        if f.metadata.get('reduction'):
            results[f.name].append(_reduction_label(value))
            results[f'{f.name}.kind'].append(_kind_name(type(value)))
            active = {kf.name for kf in fields(type(value))}
            for knob in slot_knobs[f.name]:
                col = f'{f.name}.{knob}'
                if knob in active:
                    results[col].append(_encode_knob(getattr(value, knob)))
                else:
                    results[col].append(None)
        elif f.name in _SENTINEL_FIELDS:
            results[f.name].append(_encode_knob(value))
        else:
            results[f.name].append(value)


def enable_compilation_cache() -> None:
    """Enable JAX compilation caching to speed repeated runs."""
    jax_config.update('jax_compilation_cache_dir', 'config/jax_cache')
    jax_config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax_config.update('jax_persistent_cache_min_compile_time_secs', 0.1)


def make_output_dir() -> Path:
    """Create a dated output directory next to this script."""
    stamp = datetime.now(tz=timezone.utc).astimezone().strftime('%Y-%m-%dT%H-%M-%S')
    device = get_default_device().device_kind.replace(' ', '_')
    out_dir = Path(__file__).with_name(f'opt-{stamp}-{device}')
    out_dir.mkdir()
    return out_dir


def parse_args() -> Namespace:
    """Parse CLI arguments."""
    parser = ArgumentParser(
        description='Clock mcmcstep.step over a config-defined hyperparameter grid.',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'config',
        type=Path,
        help='Path to a JSONC config file defining roles and value ranges.',
    )
    parser.add_argument(
        '--logging',
        type=Logging,
        choices=list(Logging),
        default=Logging.pbar,
        help='Logging mode for benchmark runs.',
    )
    parser.add_argument(
        '--num',
        type=int,
        default=10,
        help='Number of timing repetitions for each configuration.',
    )
    parser.add_argument(
        '--minimal',
        action='store_true',
        default=False,
        help='Use a minimal hyperparameter grid (first + last value per range).',
    )
    parser.add_argument(
        '--worker',
        default=None,
        help='Internal: JSON-encoded restart_values for worker mode.',
    )
    parser.add_argument(
        '--worker-out',
        type=Path,
        default=None,
        help='Internal: output parquet path for worker mode.',
    )
    return parser.parse_args()


def main() -> None:
    """Entry point of the script."""
    enable_compilation_cache()
    args = parse_args()
    config = Config.load(args.config)
    if args.minimal:
        config = config.minimal()

    if args.worker is not None:
        restart_values = json.loads(args.worker)
        df = inner_benchmark_loop(config, args, restart_values)
        df.write_parquet(args.worker_out)
        return

    out_dir = make_output_dir()
    with (out_dir / 'config.jsonc').open('w') as f:
        json5.dump(
            {
                'plot': {
                    'scan': config.plot.scan,
                    'reduce': config.plot.reduce,
                    'matrix': list(config.plot.matrix),
                    'drop': list(config.plot.drop),
                    'sentinels': list(config.plot.sentinels),
                },
                'values': config.values,
            },
            f,
            indent=4,
            quote_keys=True,
            cls=_InlineArraysJSON5Encoder,
        )

    results = benchmark_loop(config, args, out_dir)

    results_path = out_dir / 'results.parquet'
    print(f'write {results_path}...')
    results.write_parquet(results_path)
    print(out_dir)


if __name__ == '__main__':
    main()
