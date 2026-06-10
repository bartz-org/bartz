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

import sys
from collections.abc import Callable, Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from functools import partial
from inspect import signature
from io import StringIO
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from equinox import Module, error_if
from jax import (
    block_until_ready,
    clear_caches,
    debug,
    ensure_compile_time_eval,
    eval_shape,
    random,
    tree,
)
from jax import numpy as jnp
from jax.errors import JaxRuntimeError
from jax.sharding import Mesh
from jaxtyping import Array, Integer, Key

from bartz import mcmcloop, mcmcstep
from bartz.mcmcloop import run_mcmc
from benchmarks.latest_bartz._jaxext import get_device_count, jit, split

if TYPE_CHECKING:
    from bartz.mcmcstep import State
else:
    try:
        from bartz.mcmcstep import State
    except ImportError:
        # WORKAROUND(bartz<0.6.0): old versions use a dictionary for the mcmc state
        State = dict

try:
    from bartz.BART import mc_gbart as gbart
except ImportError:
    # WORKAROUND(bartz<0.8.0): mc_gbart was introduced in 0.8.0; fall back to gbart
    from bartz.BART import gbart

from bartz.mcmcstep import init, step
from benchmarks.latest_bartz.mcmcstep import make_p_nonterminal
from benchmarks.latest_bartz.testing import gen_data

# asv config
timeout = 120.0

# config
P = 100
N = 10000
NTREE = 50
NITERS = 10


Kind = Literal['plain', 'weights', 'binary', 'sparse', 'multivariate']


def get_default_platform() -> str:
    """Get the default JAX platform (cpu, gpu)."""
    with ensure_compile_time_eval():
        return jnp.zeros(0).platform()  # ty: ignore[unresolved-attribute]


def simple_init(  # noqa: C901, PLR0915
    p: int,
    n: int,
    num_trees: int,
    kind: Kind = 'plain',
    *,
    num_chains: int | None = None,
    mesh: dict[str, int] | Mesh | None = None,
    **kwargs: Any,
) -> State:
    """Glue code to support `mcmcstep.init` across API changes."""
    # generate data
    if kind == 'multivariate':
        k = 2
    else:
        k = None
    # q must be even and < p (< p // k for multivariate); the step timing does
    # not depend on the DGP, so clamp to a valid value for small p
    q = min(2, (p - 1 if k is None else p // k - 1))
    q -= q % 2
    data = gen_data(
        random.key(2026_06_07),
        n=n,
        p=p,
        k=k,
        q=q,
        lambda_=None if k is None else 0.5,
        sigma2_lin=1.0,
        sigma2_quad=1.0,
        sigma2_eps=1.0,
    ).quantize()

    kw: dict = dict(
        X=data.x,
        y=data.y,
        offset=0.0 if k is None else jnp.zeros(k),
        max_split=data.max_split,
        num_trees=num_trees,
        p_nonterminal=make_p_nonterminal(6, 0.95, 2),
        leaf_prior_cov_inv=jnp.float32(num_trees) * (1.0 if k is None else jnp.eye(k)),
        error_cov_df=2.0,
        error_cov_scale=2.0 * (1.0 if k is None else jnp.eye(k)),
        min_points_per_decision_node=10,
        num_chains=num_chains,
        mesh=mesh,
    )

    # adapt arguments for old versions
    sig = signature(init)
    if 'offset' not in sig.parameters:
        # WORKAROUND(bartz<0.6.0): offset was added to init in 0.6.0
        kw.pop('offset')
    if 'sigma2_alpha' in sig.parameters:
        # WORKAROUND(bartz<0.8.0): pre-0.8.0 used sigma2_alpha/beta instead of
        # error_cov_df/scale. Inverse gamma prior: alpha = df/2, beta = scale/2.
        kw['sigma2_alpha'] = kw.pop('error_cov_df') / 2
        kw['sigma2_beta'] = kw.pop('error_cov_scale') / 2
    if 'leaf_prior_cov_inv' not in sig.parameters:
        # WORKAROUND(bartz<0.8.0): pre-0.8.0 used sigma_mu2 (0.6.0-0.7.0) or had
        # no equivalent (0.4.1-0.5.0)
        if 'sigma_mu2' in sig.parameters:
            kw['sigma_mu2'] = 1 / kw.pop('leaf_prior_cov_inv')
        else:
            kw.pop('leaf_prior_cov_inv')
    if 'min_points_per_decision_node' not in sig.parameters:
        # WORKAROUND(bartz<0.7.0): use min_points_per_leaf instead
        kw.pop('min_points_per_decision_node')
        kw.update(min_points_per_leaf=5)
    if 'mesh' not in sig.parameters:
        # WORKAROUND(bartz<0.8.0): no device sharding
        if mesh is None:
            kw.pop('mesh')
        else:
            msg = 'mesh not supported.'
            raise NotImplementedError(msg)
    if 'num_chains' not in sig.parameters:
        # WORKAROUND(bartz<0.8.0): no built-in multichain support
        if num_chains is None:
            kw.pop('num_chains')
        else:
            msg = 'multichain not supported'
            raise NotImplementedError(msg)

    match kind:
        case 'weights':
            if 'error_scale' not in sig.parameters:
                # WORKAROUND(bartz<0.5.0): no heteroskedastic weights
                msg = 'weights not supported'
                raise NotImplementedError(msg)
            kw['error_scale'] = jnp.ones(n)

        case 'binary':
            sig = signature(gbart)
            if 'type' not in sig.parameters:
                # WORKAROUND(bartz<0.6.0): no probit-link binary regression
                msg = 'binary not supported'
                raise NotImplementedError(msg)
            kw['y'] = data.y > 0
            kw.pop('sigma2_alpha', None)
            kw.pop('sigma2_beta', None)
            kw.pop('error_cov_df', None)
            kw.pop('error_cov_scale', None)

            sig = signature(init)
            if 'outcome_type' in sig.parameters:
                # WORKAROUND(bartz<0.9.0): from 0.9.0 binary y is float and the
                # outcome type is passed separately via outcome_type=
                kw['outcome_type'] = 'binary'
                kw['y'] = kw['y'].astype(jnp.float32)

        case 'sparse':
            if (
                not hasattr(mcmcstep, 'step_sparse')
                and 'sparse_on_at' not in sig.parameters
            ):
                # WORKAROUND(bartz<0.7.0): variable selection added in 0.7.0
                msg = 'sparse not supported'
                raise NotImplementedError(msg)
            kw.update(a=0.5, b=1.0, rho=float(p), sparse_on_at=999999)
            if 'sparse_on_at' not in sig.parameters:
                # WORKAROUND(bartz<0.8.0): pre-0.8.0 sparse step was external
                kw.pop('sparse_on_at')

        case 'multivariate':
            if 'leaf_prior_cov_inv' not in sig.parameters:
                # WORKAROUND(bartz<0.8.0): multivariate outcomes added in 0.8.0
                msg = 'multivariate not supported'
                raise NotImplementedError(msg)

    kw.update(kwargs)

    return init(**kw)


Mode = Literal['compile', 'run']
Cache = Literal['cold', 'warm']


class AutoParamNames:
    """Superclass that automatically sets `param_names` on subclasses."""

    param_names: ClassVar[tuple[str, ...]]

    setup: ClassVar[Callable[..., None]]
    """Defined by subclasses, introspected to set `param_names`."""

    def __init_subclass__(cls, **_: Any) -> None:
        method = cls.setup
        sig = signature(method)
        params = list(sig.parameters)
        assert params[0] == 'self'
        cls.param_names = tuple(params[1:])


class StepBase(AutoParamNames):
    """Shared setup for benchmarks of `mcmcstep.step`."""

    def make_state(self, **kwargs: Any) -> State:
        """Build the initial MCMC state, allocated on device.

        Subclasses that only compile (never run) `step` may override this to
        return an abstract state.
        """
        return simple_init(**kwargs)

    def setup(self, mode: Mode, kind: Kind, chains: int | None, **kwargs: Any) -> None:
        """Create an initial MCMC state and random seed, compile & warm-up."""
        key = random.key(2025_06_24_12_07)

        kw: dict = dict(p=P, n=N, num_trees=NTREE, kind=kind, num_chains=chains)
        kw.update(kwargs)
        state = self.make_state(**kw)

        # WORKAROUND(bartz<0.9.0): from 0.9.0, `step` is itself jitted with
        # state donation and handles variable selection internally, so it can
        # be benchmarked directly, exercising its own compilation setup. Wrap
        # older versions to emulate that setup: jit with state donation, plus
        # the separate sparse step or config adjustment where needed.
        step_handles_sparse = 'sparse_on_at' in signature(init).parameters
        step_is_jitted = all(hasattr(step, attr) for attr in ('clear_cache', 'lower'))
        if step_handles_sparse and step_is_jitted:
            if kind == 'sparse':
                # turn variable selection on from the first iteration; int32 to
                # match the dtype set by init in the lowered signature
                config = replace(state.config, sparse_on_at=jnp.int32(0))
                state = replace(state, config=config)
            self.args = (key, state)
            self.jitted_func = step
        else:
            keys = list(random.split(key))
            self.args = (keys, state)

            # WORKAROUND(bartz<0.5.0): v0.4.1 step had signature (bart, key);
            # v0.5.0+ uses (key, bart). Dispatch positionally on first param
            # name, since the decorator-wrapped step on modern bartz does not
            # accept `bart` as a keyword argument.
            step_bart_first = next(iter(signature(step).parameters)) == 'bart'

            def func(keys: list[Key[Array, '']], bart: State) -> State:
                # WORKAROUND(bartz<0.8.0): pre-0.8.0 sparse step is done by a
                # separate `mcmcstep.step_sparse` call; from 0.8.0 it's inside
                # `step` via `bart.config.sparse_on_at`.
                sparse_inside_step = not hasattr(mcmcloop, 'sparse_callback')
                if kind == 'sparse' and sparse_inside_step:
                    bart = replace(bart, config=replace(bart.config, sparse_on_at=0))
                if step_bart_first:
                    bart = step(bart, keys.pop())  # ty: ignore[invalid-argument-type]
                else:
                    bart = step(keys.pop(), bart)
                if kind == 'sparse' and not sparse_inside_step:
                    bart = mcmcstep.step_sparse(keys.pop(), bart)  # ty:ignore[unresolved-attribute]
                return bart

            self.jitted_func = jit(func, donate_argnums=(1,))

        # the two dispatch branches produce different (jitted_func, args)
        # signatures, which ty can not match up across attributes
        self.compiled_func = self.jitted_func.lower(*self.args).compile()  # ty:ignore[invalid-argument-type]
        if mode == 'run':
            self.run_step()
        self.mode = mode

    def run_step(self) -> None:
        """Run the compiled step, swapping the donated state for its output."""
        key, state = self.args
        state = block_until_ready(self.compiled_func(key, state))
        self.args = (key, state)


class StepGeneric(StepBase):
    """Time compiling or running `mcmcstep.step`."""

    params: tuple[tuple[Mode, ...], tuple[Kind, ...], tuple[int | None, ...]] = (
        ('compile', 'run'),
        ('plain', 'binary', 'weights', 'sparse', 'multivariate'),
        (None, 1, 2),
    )

    def time_step(self, *_: Any) -> None:
        """Time compiling `step` or running it."""
        match self.mode:
            case 'compile':
                self.jitted_func.clear_cache()
                self.jitted_func.lower(*self.args).compile()  # ty:ignore[invalid-argument-type]
            case 'run':
                self.run_step()
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


MemStat = Literal['state', 'peak']


class CompiledMemoryStats(Protocol):
    """The fields of jax's `memory_analysis()` result used here."""

    argument_size_in_bytes: int
    output_size_in_bytes: int
    alias_size_in_bytes: int
    temp_size_in_bytes: int
    peak_memory_in_bytes: int


def standardize_memory_analysis(analysis: CompiledMemoryStats) -> dict[MemStat, int]:
    """Map a compiled function's memory analysis to backend-independent bytes.

    The raw fields of `jax.stages.Compiled.memory_analysis()` carry
    backend-dependent meanings. Determined empirically with jax and jaxlib
    0.10.1 on the cpu and cuda sm_86 (RTX 3060) backends:

    - `argument`, `output`, `alias` and `temp` mean the same on both backends:
      persistent inputs, outputs, donated-in-place buffers, and the
      peak-simultaneous scratch arena. `temp` reuses freed space, so it is a
      high-water mark, not a sum over the run.
    - `peak_memory_in_bytes` differs: on cpu it approximately equals `argument
      + output - alias` and *excludes* `temp`; on gpu it is the true high-water
      mark, the same quantity *plus* `temp` (minus small liveness overlaps).

    We report `state = argument + output - alias` (persistent footprint) and
    `peak = state + temp` (high-water mark, the gpu meaning), reading the raw
    peak directly on gpu and reconstructing it on cpu.
    """
    state = (
        analysis.argument_size_in_bytes
        + analysis.output_size_in_bytes
        - analysis.alias_size_in_bytes
    )
    if get_default_platform() == 'cpu':
        # the cpu peak field omits temp, so reconstruct the high-water mark
        peak = state + analysis.temp_size_in_bytes
    else:
        # gpu (and presumably tpu) already report the liveness-aware peak
        peak = analysis.peak_memory_in_bytes
    return {'state': state, 'peak': peak}


class StepMemory(StepBase):
    """Device memory footprint of the compiled `mcmcstep.step` at large scale.

    Reports, in bytes, the persistent `state` buffers and the `peak` high-water
    mark (normalized to a backend-independent meaning by
    `standardize_memory_analysis`) at a scale that far exceeds device memory.
    The state is built abstractly, so `step` is compiled and analyzed but never
    allocated nor run.
    """

    params: tuple[tuple[Kind, ...], tuple[int | None, ...], tuple[MemStat, ...]] = (
        ('plain', 'binary', 'weights', 'sparse', 'multivariate'),
        (None, 1, 2),
        ('state', 'peak'),
    )

    def make_state(self, **kwargs: Any) -> State:
        """Build the state abstractly, to analyze a scale too large to allocate."""
        return eval_shape(lambda: simple_init(**kwargs))

    def setup(self, kind: Kind, chains: int | None, stat: MemStat) -> None:  # ty:ignore[invalid-method-override]
        """Compile `step` at large scale and extract its memory analysis."""
        # n=16Mi, num_trees=1Ki: only compiled, never run, so it may exceed
        # device memory. 16Mi = 16 * 2**20.
        super().setup('compile', kind, chains, n=16 * 1024**2, num_trees=1024)
        analysis = self.compiled_func.memory_analysis()
        if analysis is None:
            # the active backend does not expose a memory analysis
            msg = 'memory analysis not available'
            raise NotImplementedError(msg)
        self.memory = standardize_memory_analysis(analysis)
        self.stat = stat

    def track_memory(self, *_: Any) -> int:
        """Report the selected standardized memory statistic of `step`."""
        return self.memory[self.stat]

    track_memory.unit = 'bytes'  # ty: ignore[unresolved-attribute]


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
        predict: bool = False,
        kwargs: Mapping[str, Any] = MappingProxyType({}),
    ) -> None:
        """Prepare the arguments and run once to warm-up."""
        # check support for multiple chains
        sig = signature(gbart)
        support_multichain = 'mc_cores' in sig.parameters
        if nchains != 1 and not support_multichain:
            # WORKAROUND(bartz<0.8.0): mc_gbart (with mc_cores) was added in 0.8.0
            msg = 'multi-chain not supported'
            raise NotImplementedError(msg)

        # random seed
        keys = split(random.key(2025_06_24_14_55))

        # generate simulated data
        dgp = gen_data(
            keys.pop(),
            n=2 * N,
            p=P,
            k=1,
            q=2,
            lambda_=0,
            sigma2_lin=0.4,
            sigma2_quad=0.4,
            sigma2_eps=0.2,
        )
        train, test = dgp.split()
        block_until_ready((train, test))

        # arguments
        self.kw: dict = dict(
            x_train=train.x,
            y_train=train.y.squeeze(0),
            ntree=NTREE,
            nskip=niters // 2,
            ndpost=(niters - niters // 2) * nchains,
            seed=keys.pop(),
        )
        if support_multichain:
            self.kw.update(mc_cores=nchains)
        self.kw.update(kwargs)
        block_until_ready(self.kw)

        # save information used to run predictions
        self.predict = predict
        if predict:
            self.test = test
            self.bart = gbart(**self.kw)
            block_bart(self.bart)

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
        with redirect_stdout(StringIO()):
            if self.predict:
                ypred = self.bart.predict(self.test.x)
                block_until_ready(ypred)
            else:
                bart = gbart(**self.kw)
                block_bart(bart)


def block_bart(bart: gbart) -> None:
    """Block a bart object until ready, adapting for old versions."""
    if isinstance(bart, Module):
        block_until_ready(bart)
    else:
        # WORKAROUND(bartz<0.7.0): pre-0.7.0 gbart was not an equinox Module
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

        # WORKAROUND(bartz<0.7.0): v0.6.0 used `inner_callback` instead of `callback`
        params = signature(run_mcmc).parameters
        if 'callback' not in params:
            kw['inner_callback'] = kw.pop('callback')

        # WORKAROUND(bartz<0.11.0): `run_mcmc`'s `bart` argument was renamed `state`
        state_arg = 'bart' if 'bart' in params else 'state'
        if state_arg != 'bart':
            kw[state_arg] = kw.pop('bart')

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
                bart = tree.map(jnp.copy, kw[state_arg])
                self.time_run_mcmc()
                # put copies in place of donated buffers
                kw.update({'key': key, state_arg: bart})
            case _:
                raise KeyError(cache)

    def time_run_mcmc(self, *_: Any) -> None:
        """Time running or compiling the function."""
        # capture stderr to suppress equinox's noisy traceback when the
        # kill-canary error is raised as expected
        captured = StringIO()
        try:
            with redirect_stderr(captured):
                self.task()
        except JaxRuntimeError as e:
            is_expected = self.kill_canary in str(e)
            if not is_expected:
                sys.stderr.write(captured.getvalue())
                raise
        else:
            sys.stderr.write(captured.getvalue())
            if self.kill_niters is not None:
                msg = 'expected JaxRuntimeError with canary not raised'
                raise RuntimeError(msg)


def kill_callback(
    *,
    canary: str,
    kill_niters: int | None,
    i_total: Integer[Array, ''],
    bart: State | None = None,
    state: State | None = None,
    **_: Any,
) -> None:
    """Throw error `canary` after `kill_niters` in `run_mcmc`.

    Partially evaluate `kill_callback` on the first two arguments before
    passing it to `run_mcmc`.
    """
    if kill_niters is None:
        return
    # WORKAROUND(bartz<0.11.0): the callback's `bart` argument was renamed `state`
    state = bart if state is None else state
    assert state is not None  # run_mcmc always passes one of bart/state
    # error_cov_inv is one of the last things modified in the mcmc loop, so
    # using it as token ensures ordering; also it does not have n in the
    # dimensionality.
    if isinstance(state, dict):
        # WORKAROUND(bartz<0.6.0): pre-0.6.0 state was a dict keyed by 'sigma2'
        token = state['sigma2']  # ty: ignore[invalid-argument-type]
    elif hasattr(state, 'sigma2'):
        # WORKAROUND(bartz<0.8.0): State.sigma2 was renamed to error_cov_inv in 0.8.0
        token = state.sigma2
    else:
        token = state.error_cov_inv
    stop = i_total + 1 == kill_niters  # i_total is updated after callback
    token = error_if(token, stop, canary)
    debug.callback(lambda _token: None, token)  # to avoid DCE


def detect_zero_division_error_bug(kw: dict) -> None:
    """Detect a division by zero error with 0 iterations in v0.6.0.

    WORKAROUND(bartz<0.7.0): the bug only exists in v0.6.0.
    """
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
