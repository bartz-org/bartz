# bartz/scripts/optnumbatches.py
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

"""Optimize number of batches, bartz circa v0.8.0."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from gc import collect
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

from jax import block_until_ready, config, jit, random
from jax import numpy as jnp
from jaxtyping import Array, Key
from polars import DataFrame
from tqdm import tqdm

from bartz.mcmcstep import State, init, make_p_nonterminal
from bartz.mcmcstep._step import step_trees
from benchmarks.speed import gen_nonsense_data


@dataclass(frozen=True)
class Params:
    """Benchmark parameter grid for batch optimization tests."""

    weights: Sequence[bool] = (False, True)
    k: Sequence[int | None] = (None, 1, 2, 4)
    n: Sequence[int] = tuple(4**i for i in range(12 + 1))
    num_batches: Sequence[None | int] = (None, *(2**i for i in range(10 + 1)))

    def product_iter(self) -> Iterator[dict[str, Any]]:
        """Yield parameter dictionaries for all valid combinations."""
        stuff = vars(self)
        for vals in product(*stuff.values()):
            # skip heteroskedastic multivariate, it's not implemented
            weights, k, _, _ = vals
            if weights and k is not None:
                continue

            yield dict(zip(stuff, vals, strict=True))

    def minimal(self) -> 'Params':
        """Return a Params instance with minimal value ranges."""
        return Params(
            *((v[0], v[-1]) if len(v) > 1 else v for v in vars(self).values())
        )


@partial(jit, donate_argnums=(1,))
def step_func(key: Key[Array, ''], state: State) -> State:
    """Perform one JIT-compiled tree update step."""
    return step_trees(key, state)


class Benchmark:
    """Benchmark harness for JAX tree stepping."""

    def setup(self, weights: bool, k: None | int, n: int, num_batches: None | int):
        """Initialize benchmark state for the given parameters."""
        X, y, max_split = gen_nonsense_data(1, n, k)
        num_trees = 10
        self.state = init(
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
            resid_num_batches=num_batches,
        )
        self.key = random.key(2026_01_20_13_31)
        self.task()  # warmup

    def task(self):
        """Run a single benchmark task step."""
        self.state = block_until_ready(step_func(self.key, self.state))

    def teardown(self):
        """Clean up benchmark state and caches."""
        del self.state
        step_func.clear_cache()
        collect()


class Logging(Enum):
    """Logging mode for benchmark runs."""

    no = auto()
    pbar = auto()
    results = auto()


def clock(func: Callable) -> float:
    """Return elapsed seconds for a single function call."""
    start = perf_counter()
    func()
    return perf_counter() - start


def benchmark_loop(
    *, logging: Logging = Logging.pbar, num: int = 10, minimal: bool = False
) -> DataFrame:
    """Run timing benchmarks over parameter combinations."""
    params = Params()
    if minimal:
        params = params.minimal()
    results: dict[str, list[Any]] = {}
    iterator = list(params.product_iter())

    if logging == Logging.pbar:
        iterator = tqdm(iterator)

    for param in iterator:
        if logging == Logging.results:
            print(
                ' '.join(f'{k}={v}' for k, v in param.items()) + '...',
                end='',
                flush=True,
            )

        bench = Benchmark()
        bench.setup(**param)

        times = [clock(bench.task) for _ in range(num)]
        time_est = min(times)
        time_lo = min(times)
        time_up = min(times) + (max(times) - min(times)) / num

        bench.teardown()

        if logging == Logging.results:
            print(f' {time_est:#.2g} s')

        for k, v in param.items():
            results.setdefault(k, []).append(v)
        results.setdefault('time_est', []).append(time_est)
        results.setdefault('time_lo', []).append(time_lo)
        results.setdefault('time_up', []).append(time_up)

    return DataFrame(results)


def enable_compilation_cache():
    """Enable JAX compilation caching to speed repeated runs."""
    config.update('jax_compilation_cache_dir', 'config/jax_cache')
    config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    config.update('jax_persistent_cache_min_compile_time_secs', 0.1)


def save_results(results: DataFrame) -> None:
    """Write benchmark results to a parquet file."""
    file = Path(__file__).with_suffix('.parquet')
    print(f'write {file}...')
    results.write_parquet(file)


if __name__ == '__main__':
    enable_compilation_cache()
    result = benchmark_loop()
    save_results(result)
