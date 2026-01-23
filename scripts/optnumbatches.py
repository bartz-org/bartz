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

"""Clock `step_trees` to find the optimal number of batches for indexed reductions."""

from abc import ABC, abstractmethod
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from functools import partial
from gc import collect
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

from jax import block_until_ready, config, jit, random
from jax import numpy as jnp
from jax.errors import JaxRuntimeError
from jaxtyping import Array, Key
from polars import DataFrame
from tqdm import tqdm

from bartz.jaxext import get_default_device
from bartz.mcmcstep import State, init, make_p_nonterminal
from bartz.mcmcstep._step import step_trees
from benchmarks.speed import gen_nonsense_data

T = TypeVar('T')

MAX_LEAF_INDICES_SIZE = 2**30  # 1 GiB


@dataclass(frozen=True)
class ParamsBase(ABC):
    """Base class to define hyperparameter grids."""

    weights: Sequence[bool] = (False, True)
    n: Sequence[int] = tuple(4**i for i in range(12 + 1))
    num_batches: Sequence[None | int] = (None, *(2**i for i in range(13 + 1)))

    @abstractmethod
    def valid(self, *, n: int, num_batches: int | None, **_) -> bool:
        """Check if a set of parameter values is valid."""
        return num_batches is None or num_batches <= n

    def product_iter(self) -> Iterator[dict[str, Any]]:
        """Yield parameter dictionaries for all valid combinations."""
        stuff = vars(self)
        for vals in product(*stuff.values()):
            result = dict(zip(stuff, vals, strict=True))
            if self.valid(**result):
                yield result

    def minimal(self: T) -> T:
        """Drop all values but the first and last in each range."""
        return self.__class__(
            *((v[0], v[-1]) if len(v) > 1 else v for v in vars(self).values())
        )


@dataclass(frozen=True)
class ParamsResid(ParamsBase):
    """Hyperparamer grid to clock summing residuals."""

    k: Sequence[int | None] = (None, 1, 2, 4)

    def valid(self, *, weights: bool, k: int | None, **values) -> bool:  # ty:ignore[invalid-method-override]
        """Skip heteroskedastic multivariate."""
        return (not weights or k is None) and super().valid(**values)


@dataclass(frozen=True)
class ParamsCount(ParamsBase):
    """Hyperparamer grid to clock summing precisions or counting datapoints."""

    num_trees: Sequence[int] = tuple(4**i for i in range(5 + 1))

    def valid(self, *, n: int, num_trees: int, **values) -> bool:  # ty:ignore[invalid-method-override]
        """Skip if it would use too much memory."""
        return num_trees * n <= MAX_LEAF_INDICES_SIZE and super().valid(n=n, **values)


@partial(jit, donate_argnums=(1,))
def step_func(key: Key[Array, ''], state: State) -> State:
    """Perform one JIT-compiled tree update step."""
    return step_trees(key, state)


class Benchmark:
    """Benchmark harness for JAX tree stepping."""

    def setup(
        self,
        paramcls: type[ParamsBase],
        *,
        weights: bool,
        n: int,
        num_batches: None | int,
        k: int | None = None,
        num_trees: int = 5,
    ):
        """Initialize BART state and warmup MCMC step."""
        # generate data
        X, y, max_split = gen_nonsense_data(1, n, k)

        # determine which reduction to configure
        if paramcls is ParamsResid:
            nb_kwargs: dict = dict(
                resid_num_batches=num_batches,
                count_num_batches=None,
                prec_num_batches=None,
            )
        elif paramcls is ParamsCount and weights:
            nb_kwargs: dict = dict(
                resid_num_batches=None,
                count_num_batches=None,
                prec_num_batches=num_batches,
            )
        elif paramcls is ParamsCount and not weights:
            nb_kwargs: dict = dict(
                resid_num_batches=None,
                count_num_batches=num_batches,
                prec_num_batches=None,
            )
        else:
            raise TypeError

        # initialize state
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
            **nb_kwargs,
        )
        self.key = random.key(2026_01_20_13_31)

        # warm up MCMC
        self.task()

    def task(self):
        """Run `step_trees` once."""
        self.state = block_until_ready(step_func(self.key, self.state))

    def teardown(self):
        """Delete the state and the compiled function."""
        del self.state
        step_func.clear_cache()
        # don't use jax.clear_caches() because it makes everything 3x slower
        collect()


class Logging(StrEnum):
    """Logging mode for benchmark runs."""

    no = auto()
    pbar = auto()
    results = auto()


class Reduction(StrEnum):
    """Which reduction to benchmark."""

    resid = auto()
    count = auto()


def clock(func: Callable) -> float:
    """Return elapsed seconds for a single function call."""
    start = perf_counter()
    func()
    return perf_counter() - start


def get_params(args: Namespace) -> ParamsBase:
    """Choose the grid of hyperparameters."""
    if args.reduction == Reduction.resid:
        params = ParamsResid()
    else:
        params = ParamsCount()
    if args.minimal:
        params = params.minimal()
    return params


def benchmark_loop(args: Namespace) -> DataFrame:
    """Run timing benchmarks over parameter combinations."""
    # get grid of hyperparameters
    params = get_params(args)
    iterator = list(params.product_iter())
    if args.logging == Logging.pbar:
        iterator = tqdm(iterator)

    # loop over hyperparameter combinations
    results: dict[str, list[Any]] = {}
    for param in iterator:
        # print hypers
        if args.logging == Logging.results:
            print(
                ' '.join(f'{k}={v}' for k, v in param.items()) + '...',
                end='',
                flush=True,
            )

        # create benchmark harness object
        bench = Benchmark()

        # handle out-of-memory errors
        try:
            # create bart state
            bench.setup(type(params), **param)

        except JaxRuntimeError as e:
            # out of memory
            if 'RESOURCE_EXHAUSTED: Out of memory while trying to allocate' in str(e):
                time_est = float('nan')
                time_lo = float('nan')
                time_up = float('nan')

            # some other error
            else:
                raise
        else:
            # clock MCMC step
            times = [clock(bench.task) for _ in range(args.num)]
            time_est = min(times)
            time_lo = min(times)
            time_up = min(times) + (max(times) - min(times)) / args.num

        # free memory
        bench.teardown()

        # print timing results
        if args.logging == Logging.results:
            print(f' {time_est:#.2g} s')

        # save results
        for k, v in param.items():
            results.setdefault(k, []).append(v)
        results.setdefault('time_est', []).append(time_est)
        results.setdefault('time_lo', []).append(time_lo)
        results.setdefault('time_up', []).append(time_up)

    # convert to polars dataframe
    return DataFrame(results)


def enable_compilation_cache():
    """Enable JAX compilation caching to speed repeated runs."""
    config.update('jax_compilation_cache_dir', 'config/jax_cache')
    config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    config.update('jax_persistent_cache_min_compile_time_secs', 0.1)


def save_results(results: DataFrame, args: Namespace) -> None:
    """Write benchmark results to a parquet file."""
    base = Path(__file__).stem
    device = get_default_device().device_kind.replace(' ', '_')
    filename = f'{base}_{args.reduction}_{args.num}_{args.minimal}_{device}.parquet'
    file = Path(__file__).with_name(filename)
    print(f'write {file}...')
    results.write_parquet(file)


def parse_args() -> Namespace:
    """Parse CLI arguments."""
    parser = ArgumentParser(
        description='Clock step_trees to find the optimal number of batches.',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'reduction',
        type=Reduction,
        choices=list(Reduction),
        help='Which reduction to benchmark.',
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
        help='Use a minimal hyperparameter grid.',
    )
    return parser.parse_args()


def main():
    """Entry point of the script."""
    enable_compilation_cache()
    args = parse_args()
    result = benchmark_loop(args)
    save_results(result, args)


if __name__ == '__main__':
    main()
