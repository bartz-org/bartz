# bartz/benchmarks/speed.py
#
# Copyright (c) 2025-2026, The Bartz Contributors
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

"""Measure the speed of the MCMC and its interfaces."""

from collections.abc import Mapping
from contextlib import nullcontext, redirect_stdout
from dataclasses import replace
from functools import partial
from inspect import signature
from io import StringIO
from types import MappingProxyType
from typing import Any, Literal

from equinox import Module, error_if
from jax import (
    block_until_ready,
    clear_caches,
    debug,
    ensure_compile_time_eval,
    eval_shape,
    jit,
    random,
    vmap,
)
from jax import numpy as jnp
from jax.errors import JaxRuntimeError
from jax.sharding import Mesh
from jax.tree_util import tree_map
from jaxtyping import Array, Float32, Integer, Key, UInt8

import bartz
from bartz import mcmcloop, mcmcstep
from bartz.mcmcloop import run_mcmc
from benchmarks.latest_bartz.jaxext import get_device_count, split

try:
    from bartz.mcmcstep import State
except ImportError:
    # old versions use a dictionary to store the mcmc state
    State: type = dict

try:
    from bartz.BART import mc_gbart as gbart
except ImportError:
    from bartz.BART import gbart

from bartz.mcmcstep import init, step
from benchmarks.latest_bartz.mcmcstep import make_p_nonterminal
from benchmarks.latest_bartz.testing import gen_data

# asv config
timeout = 30.0

# config
P = 100
N = 10000
NTREE = 50
NITERS = 10


@partial(jit, static_argnums=(0, 1, 2))
def gen_nonsense_data(
    p: int, n: int, k: int | None
) -> tuple[
    UInt8[Array, '{p} {n}'],
    Float32[Array, ' {n}'] | Float32[Array, '{k} {n}'],
    UInt8[Array, ' {p}'],
]:
    """Generate pretty nonsensical data."""
    X = jnp.arange(p * n, dtype=jnp.uint8).reshape(p, n)
    X = vmap(jnp.roll)(X, jnp.arange(p))
    max_split = jnp.full(p, 255, jnp.uint8)
    shift = 0 if k is None else jnp.linspace(0, 2 * jnp.pi, k, endpoint=False)[:, None]
    y = jnp.cos(jnp.linspace(0, 2 * jnp.pi / 32 * n, n) + shift)
    return X, y, max_split


Kind = Literal['plain', 'weights', 'binary', 'sparse']


def get_default_platform() -> str:
    """Get the default JAX platform (cpu, gpu)."""
    with ensure_compile_time_eval():
        return jnp.zeros(()).platform()


def simple_init(  # noqa: C901, PLR0915
    p: int,
    n: int,
    num_trees: int,
    kind: Kind = 'plain',
    *,
    k: int | None = None,
    num_chains: int | None = None,
    mesh: dict[str, int] | Mesh | None = None,
    **kwargs: Any,
) -> State:
    """Glue code to support `mcmcstep.init` across API changes."""
    X, y, max_split = gen_nonsense_data(p, n, k)

    kw: dict = dict(
        X=X,
        y=y,
        offset=0.0,
        max_split=max_split,
        num_trees=num_trees,
        p_nonterminal=make_p_nonterminal(6, 0.95, 2),
        leaf_prior_cov_inv=jnp.float32(num_trees),
        error_cov_df=2.0,
        error_cov_scale=2.0,
        min_points_per_decision_node=10,
        num_chains=num_chains,
        mesh=mesh,
    )

    # adapt arguments for old versions
    sig = signature(init)
    if 'offset' not in sig.parameters:
        kw.pop('offset')
    if 'sigma2_alpha' in sig.parameters:
        # old version: convert error_cov_df/scale to sigma2_alpha/beta
        # inverse gamma prior: alpha = df / 2, beta = scale / 2
        kw['sigma2_alpha'] = kw.pop('error_cov_df') / 2
        kw['sigma2_beta'] = kw.pop('error_cov_scale') / 2
    if 'leaf_prior_cov_inv' not in sig.parameters:
        if 'sigma_mu2' in sig.parameters:
            kw['sigma_mu2'] = 1 / kw.pop('leaf_prior_cov_inv')
        else:
            kw.pop('leaf_prior_cov_inv')
    if 'min_points_per_decision_node' not in sig.parameters:
        kw.pop('min_points_per_decision_node')
        kw.update(min_points_per_leaf=5)
    if 'suffstat_batch_size' in sig.parameters:
        # bypass the tracing bug fixed in v0.2.1
        kw.update(suffstat_batch_size=None)
    if 'mesh' not in sig.parameters:
        if mesh is None:
            kw.pop('mesh')
        else:
            msg = 'mesh not supported.'
            raise NotImplementedError(msg)
    if 'num_chains' not in sig.parameters:
        if num_chains is None:
            kw.pop('num_chains')
        else:
            msg = 'multichain not supported'
            raise NotImplementedError(msg)

    match kind:
        case 'weights':
            if 'error_scale' not in sig.parameters:
                msg = 'weights not supported'
                raise NotImplementedError(msg)
            kw['error_scale'] = jnp.ones(n)

        case 'binary':
            sig = signature(gbart)
            if 'type' not in sig.parameters:
                msg = 'binary not supported'
                raise NotImplementedError(msg)
            kw['y'] = y > 0
            kw.pop('sigma2_alpha', None)
            kw.pop('sigma2_beta', None)
            kw.pop('error_cov_df', None)
            kw.pop('error_cov_scale', None)

        case 'sparse':
            if (
                not hasattr(mcmcstep, 'step_sparse')
                and 'sparse_on_at' not in sig.parameters
            ):
                msg = 'sparse not supported'
                raise NotImplementedError(msg)
            kw.update(a=0.5, b=1.0, rho=float(p), sparse_on_at=999999)
            if 'sparse_on_at' not in sig.parameters:
                kw.pop('sparse_on_at')

    kw.update(kwargs)

    return init(**kw)


Mode = Literal['compile', 'run']
Cache = Literal['cold', 'warm']


class AutoParamNames:
    """Superclass that automatically sets `param_names` on subclasses."""

    def __init_subclass__(cls, **_: Any) -> None:
        method = cls.setup
        sig = signature(method)
        params = list(sig.parameters)
        assert params[0] == 'self'
        cls.param_names = tuple(params[1:])


class StepGeneric(AutoParamNames):
    """Benchmarks of `mcmcstep.step`."""

    params: tuple[tuple[Mode, ...], tuple[Kind, ...], tuple[int | None, ...]] = (
        ('compile', 'run'),
        ('plain', 'binary', 'weights', 'sparse'),
        (None, 1, 2),
    )

    def setup(self, mode: Mode, kind: Kind, chains: int | None, **kwargs: Any) -> None:
        """Create an initial MCMC state and random seed, compile & warm-up."""
        keys = list(random.split(random.key(2025_06_24_12_07)))

        kw: dict = dict(p=P, n=N, num_trees=NTREE, kind=kind, num_chains=chains)
        kw.update(kwargs)

        self.args = (keys, simple_init(**kw))

        def func(keys: list[Key[Array, '']], bart: State) -> State:
            sparse_inside_step = not hasattr(mcmcloop, 'sparse_callback')
            if kind == 'sparse' and sparse_inside_step:
                bart = replace(bart, config=replace(bart.config, sparse_on_at=0))
            bart = step(key=keys.pop(), bart=bart)
            if kind == 'sparse' and not sparse_inside_step:
                bart = mcmcstep.step_sparse(keys.pop(), bart)  # ty:ignore[unresolved-attribute] in this case it's an old version that has that attribute
            return bart

        self.jitted_func = jit(func)
        self.compiled_func = self.jitted_func.lower(*self.args).compile()
        if mode == 'run':
            block_until_ready(self.compiled_func(*self.args))
        self.mode = mode

    def time_step(self, *_: Any) -> None:
        """Time compiling `step` or running it."""
        match self.mode:
            case 'compile':
                self.jitted_func.clear_cache()
                self.jitted_func.lower(*self.args).compile()
            case 'run':
                block_until_ready(self.compiled_func(*self.args))
            case _:
                raise KeyError(self.mode)


class StepSharded(StepGeneric):
    """Benchmark `mcmcstep.step` with sharded data."""

    params = ((False, True),)

    def setup(self, sharded: bool) -> None:  # ty:ignore[invalid-method-override]
        """Set up with settings that make the effect of sharding salient."""
        sig = signature(init)
        if 'mesh' not in sig.parameters:
            msg = 'data sharding not supported'
            raise NotImplementedError(msg)

        if get_device_count() < 2:
            msg = 'Only one device, can not shard'
            raise NotImplementedError(msg)

        super().setup(
            'run',
            'plain',
            None,
            p=1,
            n=2000_000,
            num_trees=1,
            mesh=dict(data=2) if sharded else None,
        )


class BaseGbart(AutoParamNames):
    """Base class to benchmark `mc_gbart`."""

    # asv config
    # the param list is empty in this class, this makes asv skip this class
    # instead of considering it a benchmark to run
    params = ((),)
    warmup_time = 0.0
    number = 1

    def setup(
        self,
        niters: int = NITERS,
        nchains: int = 1,
        cache: Cache = 'warm',
        profile: bool = False,
        kwargs: Mapping[str, Any] = MappingProxyType({}),
    ) -> None:
        """Prepare the arguments and run once to warm-up."""
        # check support for multiple chains
        sig = signature(gbart)
        support_multichain = 'mc_cores' in sig.parameters
        if nchains != 1 and not support_multichain:
            msg = 'multi-chain not supported'
            raise NotImplementedError(msg)

        # random seed
        keys = split(random.key(2025_06_24_14_55))

        # generate simulated data
        dgp = gen_data(
            keys.pop(),
            n=N,
            p=P,
            k=1,
            q=2,
            lam=0,
            sigma2_lin=0.4,
            sigma2_quad=0.4,
            sigma2_eps=0.2,
        )

        # arguments
        self.kw = dict(
            x_train=dgp.x,
            y_train=dgp.y.squeeze(0),
            ntree=NTREE,
            nskip=niters // 2,
            ndpost=(niters - niters // 2) * nchains,
            seed=keys.pop(),
        )
        if support_multichain:
            self.kw.update(mc_cores=nchains)
        self.kw.update(kwargs)

        # set profile mode
        if not profile:
            self.context = nullcontext
        elif hasattr(bartz, 'profile_mode'):
            self.context = lambda: bartz.profile_mode(True)
        else:
            msg = 'Profile mode not supported.'
            raise NotImplementedError(msg)

        # decide how much to cold-start
        match cache:
            case 'cold':
                clear_caches()
            case 'warm':
                self.time_gbart()
            case _:
                raise KeyError(cache)

    def time_gbart(self, *_: Any) -> None:
        """Time instantiating the class."""
        with redirect_stdout(StringIO()), self.context():
            bart = gbart(**self.kw)
            if isinstance(bart, Module):
                block_until_ready(bart)
            else:
                block_until_ready((bart._mcmc_state, bart._main_trace))


class GbartIters(BaseGbart):
    """Time `mc_gbart` vs. the number of iterations.

    This is useful to distinguish the startup time from the time per iteration.
    """

    params = ((0, NITERS, 2 * NITERS, 3 * NITERS, 4 * NITERS, 5 * NITERS),)


class GbartChains(BaseGbart):
    """Time `mc_gbart` vs. the number of chains."""

    params = ((1, 2, 4, 8, 16, 32), (False, True))

    def setup(self, nchains: int, shard: bool) -> None:  # ty:ignore[invalid-method-override]
        """Set up to use or not multiple cpus."""
        # check there is support for multichain
        if 'mc_cores' not in signature(gbart).parameters:
            msg = 'multichain not supported'
            raise NotImplementedError(msg)

        # check there are enough devices to shard
        if shard and get_device_count() < 2:
            msg = 'Only one device, can not shard'
            raise NotImplementedError(msg)

        # determine the arguments to set up sharding
        if not shard:
            # disable sharding unconditionally
            kwargs = dict(num_chain_devices=None)
        elif get_default_platform() == 'cpu':
            # on cpu sharding is automatical, no arguments needed
            kwargs = {}
        else:
            # on gpu shard explicitly
            kwargs = dict(num_chain_devices=min(nchains, get_device_count()))

        super().setup(NITERS, nchains, 'warm', False, dict(bart_kwargs=kwargs))


class GbartGeneric(BaseGbart):
    """General timing of `mc_gbart` with many settings."""

    params = ((0, NITERS), (1, 6), ('warm', 'cold'), (False, True))


class BaseRunMcmc(AutoParamNames):
    """Base class to benchmark `run_mcmc`."""

    # asv config
    params = ((),)  # empty to make asv skip this class
    warmup_time = 0.0
    number = 1

    # other config
    kill_canary = 'kill-canary happy-chinese-voiceover'

    def setup(
        self,
        kill_niters: int | None = None,
        mode: Mode = 'run',
        cache: Cache = 'warm',
        kwargs: Mapping[str, Any] = MappingProxyType({}),
        n: int = N,
    ) -> None:
        """Prepare the arguments, compile the function, and run to warm-up."""
        kw: dict = dict(
            key=random.key(2025_04_25_15_57),
            bart=simple_init(P, n, NTREE),
            n_save=NITERS,
            n_burn=0,
            n_skip=1,
            callback=partial(
                kill_callback, canary=self.kill_canary, kill_niters=kill_niters
            ),
        )
        kw.update(kwargs)

        # handle different callback name in v0.6.0
        params = signature(run_mcmc).parameters
        if 'callback' not in params:
            kw['inner_callback'] = kw.pop('callback')

        # catch bug and skip if found
        detect_zero_division_error_bug(kw)

        # prepare task to run in benchmark
        match mode:
            case 'compile':
                static_argnames = ('n_save', 'n_skip', 'n_burn')
                if 'callback' in params:
                    static_argnames += ('callback',)
                else:
                    static_argnames += ('inner_callback',)
                f = jit(run_mcmc, static_argnames=static_argnames)

                def task() -> None:
                    f.clear_cache()
                    f.lower(**kw).compile()
            case 'run':

                def task() -> None:
                    block_until_ready(run_mcmc(**kw))
            case _:
                raise KeyError(mode)
        self.task = task
        self.kill_niters = kill_niters

        # decide how much to cold-start
        match cache:
            case 'cold':
                clear_caches()
            case 'warm':
                # prepare copies of the args because of buffer donation
                key = jnp.copy(kw['key'])
                bart = tree_map(jnp.copy, kw['bart'])
                self.time_run_mcmc()
                # put copies in place of donated buffers
                kw.update(key=key, bart=bart)
            case _:
                raise KeyError(cache)

    def time_run_mcmc(self, *_: Any) -> None:
        """Time running or compiling the function."""
        try:
            self.task()
        except JaxRuntimeError as e:
            is_expected = self.kill_canary in str(e)
            if not is_expected:
                raise
        else:
            if self.kill_niters is not None:
                msg = 'expected JaxRuntimeError with canary not raised'
                raise RuntimeError(msg)


def kill_callback(
    *,
    canary: str,
    kill_niters: int | None,
    bart: State,
    i_total: Integer[Array, ''],
    **_: Any,
) -> None:
    """Throw error `canary` after `kill_niters` in `run_mcmc`.

    Partially evaluate `kill_callback` on the first two arguments before
    passing it to `run_mcmc`.
    """
    if kill_niters is None:
        return
    # error_cov_inv (or sigma2 in old versions) is one of the last things
    # modified in the mcmc loop, so using it as token ensures ordering,
    # also it does not have n in the dimensionality
    if isinstance(bart, dict):
        token = bart['sigma2']
    elif hasattr(bart, 'sigma2'):
        token = bart.sigma2
    else:
        token = bart.error_cov_inv
    stop = i_total + 1 == kill_niters  # i_total is updated after callback
    token = error_if(token, stop, canary)
    debug.callback(lambda _token: None, token)  # to avoid DCE


def detect_zero_division_error_bug(kw: dict) -> None:
    """Detect a division by zero error with 0 iterations in v0.6.0."""
    try:
        array_kw = {k: v for k, v in kw.items() if isinstance(v, jnp.ndarray)}
        nonarray_kw = {k: v for k, v in kw.items() if not isinstance(v, jnp.ndarray)}
        partial_run_mcmc = partial(run_mcmc, **nonarray_kw)
        eval_shape(partial_run_mcmc, **array_kw)

    except ZeroDivisionError:
        if kw['n_save'] + kw['n_burn']:
            raise
        else:
            msg = 'skipping due to division by zero bug with zero iterations'
            raise NotImplementedError(msg) from None


class RunMcmcVsTraceLength(BaseRunMcmc):
    """Timings of `run_mcmc` parametrized by length of the trace to save.

    This benchmark is intended to pin a bug where the whole trace is duplicated
    on every mcmc iteration.
    """

    # asv config
    params = ((2**6, 2**8, 2**10, 2**12, 2**14, 2**16),)

    def setup(self, n_save: int) -> None:  # ty:ignore[invalid-method-override]
        """Set up to kill after a certain number of iterations."""
        kill_niters = min(self.params[0])
        super().setup(kill_niters, kwargs=dict(n_save=n_save), n=0)


class RunMcmc(BaseRunMcmc):
    """Timings of `run_mcmc`."""

    # asv config
    params = (('compile', 'run'), (0, NITERS), ('cold', 'warm'))

    def setup(self, mode: Mode, niters: int, cache: Cache) -> None:  # ty:ignore[invalid-method-override]
        """Prepare the arguments, compile the function, and run to warm-up."""
        super().setup(
            None, mode, cache, dict(n_save=niters // 2, n_burn=(niters - niters // 2))
        )
