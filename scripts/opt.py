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

Run from the project root as a module so ``benchmarks`` is importable, e.g.
``uv run python -m scripts.opt config.jsonc --minimal``.

The config file is a JSONC document like:

.. code-block:: jsonc

    {
        "scan":   "n",                              // x-axis param
        "reduce": "resid_num_batches",              // min'd over (the "optimal" curve)
        "values": {                                 // value list for every param
            "n":                  [1024, 4096, 16384],
            "k":                  [null, 2],
            "maxdepth":           [6],
            "weights":            [false],
            "num_trees":          [5, 50, 200],
            "resid_num_batches":  [null, 1, 2, 4, 8],
            "count_num_batches":  [null],
            "prec_num_batches":   [null]
        }
    }

``values`` may list any subset of the fields of `ConfigParams`; fields whose
name matches a parameter of `init` with a default are filled in with that
default if omitted. ``scan`` and ``reduce`` must be one of `ConfigParams`'s
fields. The legend / multi-line dimensions (the "matrix") are derived
automatically as the params in ``values`` with more than one value, other than
``scan`` and ``reduce``; the resulting list is written into the output config
under the key ``matrix`` for downstream tools.

`ConfigParams` covers two kinds of params:

- ``n`` and ``k`` set the data shape: ``X``, ``y``, ``max_split`` come from
  `benchmarks.speed.gen_nonsense_data(1, n, k)`.
- ``maxdepth`` and ``weights`` map to `init` args without being literal init
  kwargs: ``maxdepth`` -> ``p_nonterminal = make_p_nonterminal(maxdepth, 0.95,
  2)``; ``weights = True`` -> ``error_scale = jnp.ones(n)`` (else ``None``).
- The remaining fields are forwarded to `init` verbatim.

``leaf_prior_cov_inv``, ``error_cov_df`` and ``error_cov_scale`` are hardcoded
in `ConfigParams.to_init_kwargs` and not configurable from the file.
"""

import inspect
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, fields, replace
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
from polars import DataFrame, concat
from tqdm import tqdm

from bartz._jaxext import get_default_device
from bartz.mcmcstep import State, init, make_p_nonterminal, step
from benchmarks.speed import gen_nonsense_data

MAX_LEAF_INDICES_SIZE = 2**30  # 1 GiB


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

    resid_num_batches: int | None | Literal['auto']
    """`init`'s ``resid_num_batches`` kwarg."""

    count_num_batches: int | None | Literal['auto']
    """`init`'s ``count_num_batches`` kwarg."""

    prec_num_batches: int | None | Literal['auto']
    """`init`'s ``prec_num_batches`` kwarg."""

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Names of every field, in declaration order."""
        return tuple(f.name for f in fields(cls))

    @classmethod
    def restart_field_names(cls) -> tuple[str, ...]:
        """Fields marked ``restart=True``; changing one requires restarting Python."""
        return tuple(f.name for f in fields(cls) if f.metadata.get('restart'))

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
        )

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

    def init(self) -> State:
        """Call `mcmcstep.init` with these arguments."""
        return init(**{f.name: getattr(self, f.name) for f in fields(self)})


def _init_param_defaults() -> dict[str, Any]:
    """Default values pulled from `init`'s signature, restricted to `ConfigParams` fields."""
    sig = inspect.signature(init)
    field_names = set(ConfigParams.field_names())
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if name in field_names and param.default is not inspect.Parameter.empty
    }


@dataclass(frozen=True)
class Config:
    """Parsed and validated config file content."""

    scan: str
    """Field plotted on the x-axis."""

    reduce: str
    """Field minimised over to build the 'optimal' curve."""

    matrix: tuple[str, ...]
    """Fields used as legend / multi-line dimensions.

    Derived from ``values``: params (other than ``scan`` and ``reduce``) that
    were given more than one value in the input config.
    """

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
        cls._check_roles(path, [raw['scan'], raw['reduce']], set(values))
        matrix = tuple(
            name
            for name, vals in values.items()
            if name not in (raw['scan'], raw['reduce']) and len(vals) > 1
        )
        for name, default in _init_param_defaults().items():
            values.setdefault(name, [default])
        return cls(scan=raw['scan'], reduce=raw['reduce'], matrix=matrix, values=values)

    @staticmethod
    def _check_schema(path: Path, raw: dict[str, Any]) -> None:
        required = {'scan', 'reduce', 'values'}
        missing = required - raw.keys()
        if missing:
            msg = f'config {path}: missing required keys: {sorted(missing)}'
            raise ValueError(msg)
        extra = raw.keys() - required
        if extra:
            msg = f'config {path}: unknown top-level keys: {sorted(extra)}'
            raise ValueError(msg)
        if not isinstance(raw['scan'], str):
            msg = f'config {path}: "scan" must be a string'
            raise TypeError(msg)
        if not isinstance(raw['reduce'], str):
            msg = f'config {path}: "reduce" must be a string'
            raise TypeError(msg)
        if not isinstance(raw['values'], dict):
            msg = f'config {path}: "values" must be an object'
            raise TypeError(msg)
        for k, v in raw['values'].items():
            if not isinstance(v, list) or len(v) == 0:
                msg = f'config {path}: "values[{k!r}]" must be a non-empty list'
                raise TypeError(msg)

    @staticmethod
    def _check_values_keys(path: Path, values: dict[str, list[Any]]) -> None:
        allowed = set(ConfigParams.field_names())
        optional = set(_init_param_defaults())
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
            msg = f'config {path}: scan/reduce names not in values: {missing_role}'
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


def benchmark_loop(config: Config, args: Namespace) -> DataFrame:
    """Run the two-tiered benchmark scan.

    Outer loop: iterates restart-tier values in order. Inner loop: randomizes
    the remaining combinations. Once restart-tier params drive a real Python
    restart, the outer loop will spawn a subprocess per restart-tier value
    instead of calling `inner_benchmark_loop` directly.
    """
    if args.minimal:
        config = config.minimal()
    frames: list[DataFrame] = []
    for restart_values in config.restart_iter():
        if restart_values and args.logging != Logging.no:
            label = ' '.join(f'{k}={v}' for k, v in restart_values.items())
            print(f'=== restart: {label} ===', flush=True)
        frames.append(inner_benchmark_loop(config, args, restart_values))
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
            results[name].append(value)
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
    return parser.parse_args()


def main() -> None:
    """Entry point of the script."""
    enable_compilation_cache()
    args = parse_args()
    config = Config.load(args.config)

    out_dir = make_output_dir()
    with (out_dir / 'config.jsonc').open('w') as f:
        json5.dump(
            {
                'scan': config.scan,
                'reduce': config.reduce,
                'matrix': list(config.matrix),
                'values': config.values,
            },
            f,
            indent=4,
        )

    results = benchmark_loop(config, args)

    results_path = out_dir / 'results.parquet'
    print(f'write {results_path}...')
    results.write_parquet(results_path)
    print(out_dir)


if __name__ == '__main__':
    main()
