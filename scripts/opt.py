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

The config file is a JSONC document that assigns roles to hyperparameters:

.. code-block:: jsonc

    {
        "scan":   "n",                            // x-axis param
        "reduce": "num_batches",                  // min'd over (the "optimal" curve)
        "matrix": ["reduction", "k", "weights"], // legend / multi-line dimensions
        "defaults": {"num_trees": 16}            // override per-param "rest" value
    }

Parameters not mentioned in any role keep a single "rest" value (see `Defaults`).
The optional ``defaults`` map overrides those values per config.
"""

import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from enum import StrEnum, auto
from gc import collect
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import json5
from jax import block_until_ready, random
from jax import config as jax_config
from jax import numpy as jnp
from jax.errors import JaxRuntimeError
from polars import DataFrame
from tqdm import tqdm

from bartz._jaxext import get_default_device
from bartz.mcmcstep import State, init, make_p_nonterminal, step
from benchmarks.speed import gen_nonsense_data

MAX_LEAF_INDICES_SIZE = 2**30  # 1 GiB

# Sentinel: param has no role assigned, fill in from Defaults.
UNSET = object()


class Reduction(StrEnum):
    """Which `init` reduction kwarg `num_batches` configures."""

    resid = auto()
    count = auto()


class Logging(StrEnum):
    """Logging mode for benchmark runs."""

    no = auto()
    pbar = auto()
    results = auto()


@dataclass(frozen=True)
class Defaults:
    """Single-valued rest values for every hyperparameter."""

    reduction: Reduction = Reduction.resid
    weights: bool = False
    n: int = 1024
    num_batches: int | None = None
    k: int | None = None
    num_trees: int = 5


@dataclass(frozen=True)
class Params:
    """Value ranges for every hyperparameter plus the matching `Defaults`."""

    reduction: Sequence[Any] = (Reduction.resid, Reduction.count)
    weights: Sequence[Any] = (False, True)
    n: Sequence[Any] = tuple(4**i for i in range(12 + 1))
    num_batches: Sequence[Any] = (None, *(2**i for i in range(13 + 1)))
    k: Sequence[Any] = (None, 1, 2, 4)
    num_trees: Sequence[Any] = tuple(4**i for i in range(5 + 1))

    defaults: Defaults = field(default_factory=Defaults)

    @classmethod
    def range_field_names(cls) -> tuple[str, ...]:
        """Names of every value-range field (everything except `defaults`)."""
        return tuple(f.name for f in fields(cls) if f.name != 'defaults')

    def valid(
        self,
        *,
        reduction: Reduction,
        weights: bool,  # noqa: ARG002
        n: int,
        num_batches: int | None,
        k: int | None,  # noqa: ARG002
        num_trees: int,
    ) -> bool:
        """Whether a fully-resolved combination of values is admissible."""
        num_batches_ok = num_batches is None or num_batches <= n
        memory_ok = (
            reduction != Reduction.count or num_trees * n <= MAX_LEAF_INDICES_SIZE
        )
        return num_batches_ok and memory_ok

    def product_iter(self) -> Iterator[dict[str, Any]]:
        """Yield resolved (UNSET-free) dicts for every valid combination."""
        names = self.range_field_names()
        ranges = [getattr(self, name) for name in names]
        for vals in product(*ranges):
            resolved = {
                name: getattr(self.defaults, name) if v is UNSET else v
                for name, v in zip(names, vals, strict=True)
            }
            if self.valid(**resolved):
                yield resolved

    def minimal(self) -> 'Params':
        """Truncate every multi-element range to (first, last); keep singletons."""
        kwargs: dict[str, Any] = {}
        for name in self.range_field_names():
            v = getattr(self, name)
            kwargs[name] = (v[0], v[-1]) if len(v) > 1 else v
        return Params(**kwargs, defaults=self.defaults)


class Benchmark:
    """Benchmark harness for one full MCMC step."""

    def setup(
        self,
        *,
        reduction: Reduction,
        weights: bool,
        n: int,
        num_batches: int | None,
        k: int | None,
        num_trees: int,
    ) -> None:
        """Initialize BART state and warm up the MCMC step."""
        X, y, max_split = gen_nonsense_data(1, n, k)

        if reduction == Reduction.resid:
            nb_kwargs = dict(
                resid_num_batches=num_batches,
                count_num_batches=None,
                prec_num_batches=None,
            )
        elif weights:  # count + weights -> prec
            nb_kwargs = dict(
                resid_num_batches=None,
                count_num_batches=None,
                prec_num_batches=num_batches,
            )
        else:  # count, no weights -> count
            nb_kwargs = dict(
                resid_num_batches=None,
                count_num_batches=num_batches,
                prec_num_batches=None,
            )

        self.state: State = init(
            X=X,
            y=y,
            offset=jnp.zeros(y.shape[:-1]),
            max_split=max_split,
            num_trees=num_trees,
            p_nonterminal=make_p_nonterminal(6, 0.95, 2),
            leaf_prior_cov_inv=num_trees * (1.0 if k is None else jnp.eye(k)),
            error_cov_df=2.0,
            error_cov_scale=2 * (1.0 if k is None else jnp.eye(k)),
            error_scale=jnp.ones(n) if weights else None,
            **nb_kwargs,
        )
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


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the JSONC role-mapping config file."""
    with path.open() as f:
        config = json5.load(f)

    required = {'scan', 'reduce', 'matrix'}
    optional = {'defaults'}
    missing = required - config.keys()
    if missing:
        msg = f'config {path}: missing required keys: {sorted(missing)}'
        raise ValueError(msg)
    extra = config.keys() - required - optional
    if extra:
        msg = f'config {path}: unknown top-level keys: {sorted(extra)}'
        raise ValueError(msg)
    if not isinstance(config['scan'], str):
        msg = f'config {path}: "scan" must be a string'
        raise TypeError(msg)
    if not isinstance(config['reduce'], str):
        msg = f'config {path}: "reduce" must be a string'
        raise TypeError(msg)
    if not isinstance(config['matrix'], list) or not all(
        isinstance(x, str) for x in config['matrix']
    ):
        msg = f'config {path}: "matrix" must be a list of strings'
        raise TypeError(msg)

    known = set(Params.range_field_names())
    named = [config['scan'], config['reduce'], *config['matrix']]
    unknown = [name for name in named if name not in known]
    if unknown:
        msg = f'config {path}: unknown parameter names: {unknown}'
        raise ValueError(msg)
    if len(named) != len(set(named)):
        counts: dict[str, int] = {}
        for name in named:
            counts[name] = counts.get(name, 0) + 1
        dups = sorted(name for name, c in counts.items() if c > 1)
        msg = f'config {path}: parameter(s) in multiple roles: {dups}'
        raise ValueError(msg)

    _validate_defaults(path, config.get('defaults', {}), known, set(named))

    return config


def _validate_defaults(
    path: Path, defaults: object, known: set[str], in_roles: set[str]
) -> None:
    """Validate the optional ``defaults`` config map."""
    if not isinstance(defaults, dict):
        msg = f'config {path}: "defaults" must be an object'
        raise TypeError(msg)
    unknown_defaults = defaults.keys() - known
    if unknown_defaults:
        msg = f'config {path}: unknown defaults: {sorted(unknown_defaults)}'
        raise ValueError(msg)
    overlap = defaults.keys() & in_roles
    if overlap:
        msg = f'config {path}: defaults overlap with roles: {sorted(overlap)}'
        raise ValueError(msg)


def params_from_config(config: dict[str, Any], *, minimal: bool) -> Params:
    """Build a `Params` whose ranges include only role-assigned parameters."""
    in_roles = {config['scan'], config['reduce'], *config['matrix']}
    overrides = dict(config.get('defaults', {}))
    if 'reduction' in overrides:
        overrides['reduction'] = Reduction(overrides['reduction'])
    defaults = Defaults(**overrides)
    full = Params()
    kwargs = {
        name: getattr(full, name) if name in in_roles else (UNSET,)
        for name in Params.range_field_names()
    }
    params = Params(**kwargs, defaults=defaults)
    if minimal:
        params = params.minimal()
    return params


def benchmark_loop(params: Params, args: Namespace) -> DataFrame:
    """Run timing benchmarks over every valid combination of `params`."""
    iterator: Iterable[dict[str, Any]] = list(params.product_iter())
    if args.logging == Logging.pbar:
        iterator = tqdm(iterator)

    results: dict[str, list[Any]] = {}
    for resolved in iterator:
        if args.logging == Logging.results:
            print(
                ' '.join(f'{k}={v}' for k, v in resolved.items()) + '...',
                end='',
                flush=True,
            )

        bench = Benchmark()
        try:
            bench.setup(**resolved)
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

        for k, v in resolved.items():
            results.setdefault(k, []).append(v)
        results.setdefault('time_est', []).append(time_est)
        results.setdefault('time_lo', []).append(time_lo)
        results.setdefault('time_up', []).append(time_up)

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
        help='Path to a JSONC config file mapping roles to parameter names.',
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
    config = load_config(args.config)
    params = params_from_config(config, minimal=args.minimal)

    out_dir = make_output_dir()
    shutil.copy(args.config, out_dir / 'config.jsonc')

    results = benchmark_loop(params, args)

    results_path = out_dir / 'results.parquet'
    print(f'write {results_path}...')
    results.write_parquet(results_path)
    print(out_dir)


if __name__ == '__main__':
    main()
