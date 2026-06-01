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
            "reduce": "resid_num_batches"           // min'd over (the "optimal" curve)
        },
        "values": {                                 // value list for every param
            "n":                  [1024, 4096, 16384],
            "k":                  [null, 2],
            "maxdepth":           [6],
            "weights":            [false],
            "num_trees":          [5, 50, 200],
            "resid_num_batches":  [null, 1, 2, 4, 8],
            "count_num_batches":  [null],
            "prec_num_batches":   [null],
            "num_chains":         [null, 4]
        }
    }

``values`` may list any subset of the fields of `ConfigParams`; fields whose
name matches a parameter of `init` with a default are filled in with that
default if omitted. ``plot.scan`` and ``plot.reduce`` must be one of
`ConfigParams`'s fields. The legend / multi-line dimensions (the "matrix") are
derived automatically as the params in ``values`` with more than one value,
other than ``plot.scan`` and ``plot.reduce``; the resulting list is written
into the output config under ``plot.matrix`` for downstream tools.
"""

import inspect
import json
import os
import subprocess
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Callable, Iterator
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from enum import StrEnum, auto
from gc import collect
from itertools import product
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any, Literal

import json5
from jax import Array, block_until_ready, random
from jax import config as jax_config
from jax import numpy as jnp
from jax.errors import JaxRuntimeError
from polars import DataFrame, concat, read_parquet
from tqdm import tqdm

from bartz._jaxext import get_default_device
from bartz.mcmcstep import State, init, make_p_nonterminal, step
from bartz.testing import gen_nonsense_data

MAX_LEAF_INDICES_SIZE = 2**30  # 1 GiB


def init_default(name: str) -> Any:  # noqa: ANN401
    """Default value of parameter ``name`` in `init`'s signature."""
    default = inspect.signature(init).parameters[name].default
    if default is inspect.Parameter.empty:
        msg = f'`init` parameter {name!r} has no default'
        raise ValueError(msg)
    return default


class Logging(StrEnum):
    """Logging mode for benchmark runs."""

    no = auto()
    pbar = auto()
    results = auto()


@dataclass(frozen=True)
class ConfigParams:
    """One resolved combination of parameters, as named in the config file.

    A field declared with ``dataclasses.field(metadata={'restart': True})`` is
    "restart-tier": changing its value requires restarting Python (e.g. it
    affects env vars or import-time configuration). The benchmark loop is
    two-tiered: the outer loop iterates over restart-tier values in order, and
    the inner loop randomizes only over the remaining fields.
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

    resid_num_batches: int | None | Literal['auto'] = init_default('resid_num_batches')
    """`init`'s ``resid_num_batches`` kwarg."""

    count_num_batches: int | None | Literal['auto'] = init_default('count_num_batches')
    """`init`'s ``count_num_batches`` kwarg."""

    prec_num_batches: int | None | Literal['auto'] = init_default('prec_num_batches')
    """`init`'s ``prec_num_batches`` kwarg."""

    num_chains: int | None = init_default('num_chains')
    """`init`'s ``num_chains`` kwarg."""

    optimization_level: str | None = field(
        default=jax_config.jax_optimization_level, metadata={'jax_config': True}
    )
    """JAX ``jax_optimization_level``; one of ``'O0'``..``'O3'``. ``None`` in the config resolves to the JAX default (typically ``'UNKNOWN'``) at load time."""

    exec_time_optimization_effort: float | None = field(
        default=jax_config.jax_exec_time_optimization_effort,
        metadata={'jax_config': True},
    )
    """JAX ``jax_exec_time_optimization_effort``; float in ``[-1.0, 1.0]``. ``None`` in the config resolves to the JAX default (typically ``0.0``) at load time."""

    memory_fitting_level: str | None = field(
        default=jax_config.jax_memory_fitting_level, metadata={'jax_config': True}
    )
    """JAX ``jax_memory_fitting_level``; one of ``'O0'``..``'O3'``. ``None`` in the config resolves to the JAX default (typically ``'O2'``) at load time."""

    memory_fitting_effort: float | None = field(
        default=jax_config.jax_memory_fitting_effort, metadata={'jax_config': True}
    )
    """JAX ``jax_memory_fitting_effort``; float in ``[-1.0, 1.0]``. ``None`` in the config resolves to the JAX default (typically ``0.0``) at load time."""

    enable_pgle: bool | None = field(
        default=jax_config.jax_enable_pgle, metadata={'jax_config': True}
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
    def defaults(cls) -> dict[str, Any]:
        """Map of field name to default value, for fields that have one."""
        return {f.name: f.default for f in fields(cls) if f.default is not MISSING}

    def is_valid(self) -> bool:
        """Whether this combination of values is admissible."""
        for nb in (
            self.resid_num_batches,
            self.count_num_batches,
            self.prec_num_batches,
        ):
            if isinstance(nb, int) and nb > self.n:
                return False
        uses_count = self.count_num_batches != 'auto' or self.prec_num_batches != 'auto'
        return not (uses_count and self.num_trees * self.n > MAX_LEAF_INDICES_SIZE)

    def to_init_kwargs(self) -> 'InitKwargs':
        """Translate this combination into kwargs for `init`."""
        X, y, max_split = gen_nonsense_data(1, self.n, self.k)
        eye = 1.0 if self.k is None else jnp.eye(self.k)
        return InitKwargs(
            X=X,
            y=y,
            offset=jnp.zeros(y.shape[:-1]),
            max_split=max_split,
            num_trees=self.num_trees,
            p_nonterminal=make_p_nonterminal(self.maxdepth, 0.95, 2),
            leaf_prior_cov_inv=self.num_trees * eye,
            error_cov_df=2.0,
            error_cov_scale=2 * eye,
            error_scale=jnp.ones(self.n) if self.weights else None,
            resid_num_batches=self.resid_num_batches,
            count_num_batches=self.count_num_batches,
            prec_num_batches=self.prec_num_batches,
            num_chains=self.num_chains,
        )

    def apply_jax_config(self) -> None:
        """Apply non-restart JAX config flags from this combo."""
        for name in self.jax_config_field_names():
            jax_config.update(f'jax_{name}', getattr(self, name))

    def __str__(self) -> str:
        return ' '.join(f'{f.name}={getattr(self, f.name)}' for f in fields(self))


@dataclass(frozen=True)
class InitKwargs:
    """Keyword arguments for `bartz.mcmcstep.init`, in init's signature order."""

    X: Array
    y: Array
    offset: Array
    max_split: Array
    num_trees: int
    p_nonterminal: Array
    leaf_prior_cov_inv: float | Array
    error_cov_df: float
    error_cov_scale: float | Array
    error_scale: Array | None
    resid_num_batches: int | None | Literal['auto']
    count_num_batches: int | None | Literal['auto']
    prec_num_batches: int | None | Literal['auto']
    num_chains: int | None

    def init(self) -> State:
        """Call `mcmcstep.init` with these arguments."""
        return init(**{f.name: getattr(self, f.name) for f in fields(self)})


# Map ConfigParams field name -> function applied to its value before storing
# it in the results DataFrame. The goal is to keep the columns purely numeric
# (no mixed int/str, no nulls colliding with missing data) so downstream
# tooling sees a stable Int64 dtype.
def _num_batches_save(v: int | None | Literal['auto']) -> int:
    if v is None:
        return -1
    if v == 'auto':
        return -2
    return v


_SAVE_PREPROCESSORS: dict[str, Callable[[Any], Any]] = {
    'num_chains': lambda v: -1 if v is None else v,
    'resid_num_batches': _num_batches_save,
    'count_num_batches': _num_batches_save,
    'prec_num_batches': _num_batches_save,
}


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
    """JSON5 encoder that renders arrays on a single line."""

    def _encode_array(self, obj: Any, seen: set, level: int) -> str:  # noqa: ANN401
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
    """Field plotted on the x-axis."""

    reduce: str
    """Field minimised over to build the 'optimal' curve."""

    matrix: tuple[str, ...]
    """Fields used as legend / multi-line dimensions.

    Derived from ``values``: params (other than ``scan`` and ``reduce``) that
    were given more than one value in the input config.
    """


@dataclass(frozen=True)
class Config:
    """Parsed and validated config file content."""

    plot: PlotConfig
    """Plot-only metadata (x-axis, reduce, matrix roles)."""

    values: dict[str, list[Any]]
    """Per-field list of values to enumerate; keys = `ConfigParams.field_names()`."""

    @classmethod
    def load(cls, path: Path) -> 'Config':
        """Load and validate a JSONC config file."""
        with path.open() as f:
            raw = json5.load(f)
        cls._check_schema(path, raw)
        values = dict(raw['values'])
        cls._check_values_keys(path, values)
        plot_raw = raw['plot']
        cls._check_roles(path, [plot_raw['scan'], plot_raw['reduce']], set(values))
        matrix = tuple(
            name
            for name, vals in values.items()
            if name not in (plot_raw['scan'], plot_raw['reduce']) and len(vals) > 1
        )
        # For JAX-config fields, an explicit ``null`` in the file means "use the
        # JAX default"; substitute it so ConfigParams carries concrete values.
        defaults = ConfigParams.defaults()
        for name in ConfigParams.jax_config_field_names():
            if name in values:
                values[name] = [
                    defaults[name] if v is None else v for v in values[name]
                ]
        for name, default in defaults.items():
            values.setdefault(name, [default])
        plot = PlotConfig(
            scan=plot_raw['scan'], reduce=plot_raw['reduce'], matrix=matrix
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
        if not isinstance(plot['scan'], str):
            msg = f'config {path}: "plot.scan" must be a string'
            raise TypeError(msg)
        if not isinstance(plot['reduce'], str):
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
            msg = f'config {path}: plot.scan/reduce names not in values: {missing_role}'
            raise ValueError(msg)
        if len(named) != len(set(named)):
            counts: dict[str, int] = {}
            for n in named:
                counts[n] = counts.get(n, 0) + 1
            dups = sorted(n for n, c in counts.items() if c > 1)
            msg = f'config {path}: parameter(s) in multiple roles: {dups}'
            raise ValueError(msg)

    def minimal(self) -> 'Config':
        """Return a Config with each multi-element range truncated to (first, last)."""
        return replace(
            self,
            values={
                k: ([v[0], v[-1]] if len(v) > 1 else list(v))
                for k, v in self.values.items()
            },
        )

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
        ranges = [self.values[n] for n in names]
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
        if jax_config.jax_enable_pgle:
            n_warmup += jax_config.jax_pgle_profiling_runs + 1
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

    results: dict[str, list[Any]] = {name: [] for name in ConfigParams.field_names()}
    for col in ('time_est', 'time_lo', 'time_up'):
        results[col] = []
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

        for name, value in asdict(params).items():
            preprocess = _SAVE_PREPROCESSORS.get(name)
            results[name].append(preprocess(value) if preprocess else value)
        results['time_est'].append(time_est)
        results['time_lo'].append(time_lo)
        results['time_up'].append(time_up)

    return DataFrame(results)


def enable_compilation_cache() -> None:
    """Enable JAX compilation caching to speed repeated runs."""
    jax_config.update('jax_compilation_cache_dir', 'config/jax_cache')
    jax_config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax_config.update('jax_persistent_cache_min_compile_time_secs', 0.1)


def make_output_dir() -> Path:
    """Create a dated output directory next to this script."""
    stamp = datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%dT%H-%M-%S')
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
