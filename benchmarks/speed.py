# bartz/benchmarks/speed.py
#
# Copyright (c) 2025, The Bartz Contributors
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

from contextlib import redirect_stdout
from functools import partial
from inspect import signature
from io import StringIO
from itertools import product
from re import escape, match
from typing import Literal

import jax
from asv_runner.benchmarks.mark import skip_for_params
from equinox import error_if
from jax import block_until_ready, clear_caches, eval_shape, jit, random, vmap
from jax import numpy as jnp
from jax.errors import JaxRuntimeError
from jax.tree import map_with_path
from jax.tree_util import tree_map

from bartz import mcmcstep
from bartz.mcmcloop import run_mcmc

try:
    from bartz.BART import mc_gbart as gbart
except ImportError:
    from bartz.BART import gbart

from bartz.mcmcstep import init, step

# asv config
timeout = 30.0

# config
P = 100
N = 10000
NTREE = 50
NITERS = 10


def gen_data(p: int, n: int):
    """Generate pretty nonsensical data."""
    X = jnp.arange(p * n, dtype=jnp.uint8).reshape(p, n)
    X = vmap(jnp.roll)(X, jnp.arange(p))
    max_split = jnp.full(p, 255, jnp.uint8)
    y = jnp.cos(jnp.linspace(0, 2 * jnp.pi / 32 * n, n))
    return X, y, max_split


def make_p_nonterminal(maxdepth: int):
    """Prepare the p_nonterminal argument to `mcmcstep.init`."""
    depth = jnp.arange(maxdepth - 1)
    base = 0.95
    power = 2
    return base / (1 + depth).astype(float) ** power


Kind = Literal['plain', 'weights', 'binary', 'sparse', 'vmap-1', 'vmap-2']


@partial(jit, static_argnums=(0, 1, 2, 3))
def simple_init(p: int, n: int, ntree: int, kind: Kind = 'plain', /, **kwargs):  # noqa: C901
    """Simplified version of `bartz.mcmcstep.init` with data pre-filled."""
    X, y, max_split = gen_data(p, n)

    kw: dict = dict(
        X=X,
        y=y,
        offset=0.0,
        max_split=max_split,
        num_trees=ntree,
        p_nonterminal=make_p_nonterminal(6),
        leaf_prior_cov_inv=jnp.float32(ntree),
        error_cov_df=2.0,
        error_cov_scale=2.0,
        min_points_per_decision_node=10,
        filter_splitless_vars=False,
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
    if 'filter_splitless_vars' not in sig.parameters:
        kw.pop('filter_splitless_vars')
    if 'suffstat_batch_size' in sig.parameters:
        # bypass the tracing bug fixed in v0.2.1
        kw.update(suffstat_batch_size=None)

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
            if not hasattr(mcmcstep, 'step_sparse'):
                msg = 'sparse not supported'
                raise NotImplementedError(msg)
            kw.update(a=0.5, b=1.0, rho=float(p))

    kw.update(kwargs)

    state = init(**kw)

    if kind.startswith('vmap-'):
        axes = vmap_axes_for_state(state)
        length = int(kind.split('-')[1])
        state = vmap(lambda x: x, in_axes=None, out_axes=axes, axis_size=length)(state)

    return state


def vmap_axes_for_state(state):
    """Get vmap axes for the MCMC state."""

    def choose_vmap_index(path, _) -> Literal[0, None]:
        no_vmap_attrs = (
            'X',
            'y',
            'offset',
            'prec_scale',
            'sigma2_alpha',
            'sigma2_beta',
            'error_cov_df',
            'error_cov_scale',
            'max_split',
            'blocked_vars',
            'p_nonterminal',
            'p_propose_grow',
            'min_points_per_decision_node',
            'min_points_per_leaf',
            'sigma_mu2',
            'leaf_prior_cov_inv',
            'a',
            'b',
            'rho',
        )
        str_path = ''.join(map(str, path))
        if any(match(rf'\b{escape(attr)}\b', str_path) for attr in no_vmap_attrs):
            return None
        else:
            return 0

    return map_with_path(choose_vmap_index, state)


Mode = Literal['compile', 'run']
Cache = Literal['cold', 'warm']


class TimeRunMcmc:
    """Timings of `run_mcmc`."""

    # asv config
    params: tuple[tuple[Mode, ...], tuple[int, ...], tuple[Cache, ...]] = (
        ('compile', 'run'),
        (0, NITERS),
        ('cold', 'warm'),
    )
    param_names = ('mode', 'niters', 'cache')
    warmup_time = 0.0
    number = 1

    def setup(self, mode: Mode, niters: int, cache: Cache):
        """Prepare the arguments, compile the function, and run to warm-up."""
        self.kw = dict(
            key=random.key(2025_04_25_15_57),
            bart=simple_init(P, N, NTREE),
            n_save=niters // 2,
            n_burn=niters // 2,
            n_skip=1,
            callback=lambda **_: None,
        )

        # catch bug and skip if found
        try:
            array_kw = {k: v for k, v in self.kw.items() if isinstance(v, jnp.ndarray)}
            nonarray_kw = {
                k: v for k, v in self.kw.items() if not isinstance(v, jnp.ndarray)
            }
            partial_run_mcmc = partial(run_mcmc, **nonarray_kw)
            eval_shape(partial_run_mcmc, **array_kw)
        except ZeroDivisionError:
            if niters:
                raise
            else:
                msg = 'skipping due to division by zero bug with zero iterations'
                raise NotImplementedError(msg) from None

        # decide how much to cold-start
        match cache:
            case 'cold':
                clear_caches()
            case 'warm':
                # prepare copies of the args because of buffer donation
                key = jnp.copy(self.kw['key'])
                bart = tree_map(jnp.copy, self.kw['bart'])
                self.time_run_mcmc(mode)
                # put copies in place of donated buffers
                self.kw.update(key=key, bart=bart)

    # skip compile with 0 iters bc in past versions 0 iters had a noop code path
    @skip_for_params(list(product(['compile'], [0], params[2])))
    def time_run_mcmc(self, mode: Mode, *_):
        """Time running or compiling the function."""
        match mode:
            case 'compile':
                # re-wrap and jit the function in the benchmark case because otherwise
                # the compiled function gets cached even if I call `compile` explicitly
                @partial(
                    jit, static_argnames=('n_save', 'n_skip', 'n_burn', 'callback')
                )
                def f(**kw):
                    return run_mcmc(**kw)

                f.lower(**self.kw).compile()

            case 'run':
                block_until_ready(run_mcmc(**self.kw))


class TimeStep:
    """Benchmarks of `mcmcstep.step`."""

    params: tuple[tuple[Mode, ...], tuple[Kind, ...]] = (
        ('compile', 'run'),
        ('plain', 'binary', 'weights', 'sparse', 'vmap-1', 'vmap-2'),
    )
    param_names = ('mode', 'kind')

    def setup(self, mode: Mode, kind: Kind):
        """Create an initial MCMC state and random seed, compile & warm-up."""
        key = random.key(2025_06_24_12_07)
        if kind.startswith('vmap-'):
            length = int(kind.split('-')[1])
            keys = list(random.split(key, (2, length)))
        else:
            keys = list(random.split(key))

        self.args = (keys, simple_init(P, N, NTREE, kind))

        def func(keys, bart):
            bart = step(key=keys.pop(), bart=bart)
            if kind == 'sparse':
                bart = mcmcstep.step_sparse(keys.pop(), bart)
            return bart

        if kind.startswith('vmap-'):
            axes = vmap_axes_for_state(self.args[1])
            func = vmap(func, in_axes=(0, axes), out_axes=axes)

        self.func = func
        self.compiled_func = jit(func).lower(*self.args).compile()
        if mode == 'run':
            block_until_ready(self.compiled_func(*self.args))

    def time_step(self, mode: Mode, _):
        """Time running compiling `step` or running it a few times."""
        match mode:
            case 'compile':

                @jit
                def f(*args):
                    return self.func(*args)

                f.lower(*self.args).compile()

            case 'run':
                block_until_ready(self.compiled_func(*self.args))


class TimeGbart:
    """Benchmarks of `BART.mc_gbart`."""

    # asv config
    params: tuple[tuple[int, ...], tuple[Cache, ...], tuple[int, ...]] = (
        (0, NITERS),
        ('cold', 'warm'),
        (1, 2, 8, 32),
    )
    param_names = ('niters', 'cache', 'nchains')
    warmup_time = 0.0
    number = 1

    def setup(self, niters: int, cache: Cache, nchains: int):
        """Prepare the arguments and run once to warm-up."""
        # check support for multiple chains
        if (niters == 0 or cache == 'cold') and nchains > 1:
            msg = 'skip multi-chain with 0 iterations or cold cache'
            raise NotImplementedError(msg)

        sig = signature(gbart)
        support_multichain = 'mc_cores' in sig.parameters
        if nchains != 1 and not support_multichain:
            msg = 'multi-chain not supported'
            raise NotImplementedError(msg)

        # random seed
        key = random.key(2025_06_24_14_55)
        keys = list(random.split(key, 3))

        # generate simulated data
        sigma = 0.1
        T = 2
        X = random.uniform(keys.pop(), (P, N), float, -2, 2)
        f = lambda X: jnp.sum(jnp.cos(2 * jnp.pi / T * X), axis=0)
        y = f(X) + sigma * random.normal(keys.pop(), (N,))

        # arguments
        self.kw = dict(
            x_train=X,
            y_train=y,
            nskip=niters // 2,
            ndpost=(niters - niters // 2) * nchains,
            seed=keys.pop(),
        )
        if support_multichain:
            self.kw.update(mc_cores=nchains)

        # decide how much to cold-start
        match cache:
            case 'cold':
                clear_caches()
            case 'warm':
                self.time_gbart()

    def time_gbart(self, *_):
        """Time instantiating the class."""
        with redirect_stdout(StringIO()):
            bart = gbart(**self.kw)
            block_until_ready((bart._mcmc_state, bart._main_trace))


class TimeRunMcmcVsTraceLength:
    """Timings of `run_mcmc` parametrized by length of the trace to save.

    This benchmark is intended to pin a bug where the whole trace is duplicated
    on every mcmc iteration.
    """

    # asv config
    params: tuple[tuple[int, ...]] = ((2**6, 2**8, 2**10, 2**12, 2**14, 2**16),)
    param_names = ('n_save',)
    warmup_time = 0.0
    number = 1

    # other config
    canary = 'canary happy-chinese-voiceover'

    def setup(self, n_save: int):
        """Prepare the arguments, compile the function, and run to warm-up."""
        n_iters = min(self.params[0])

        def callback(*, bart, i_total, **_):
            # error_cov_inv (or sigma2 in old versions) is one of the last things
            # modified in the mcmc loop, so using it as token ensures ordering,
            # also it does not have n in the dimensionality
            if hasattr(bart, 'sigma2'):
                token = bart.sigma2
            else:
                token = bart.error_cov_inv
            stop = i_total + 1 == n_iters  # i_total is updated after callback
            token = error_if(token, stop, self.canary)
            jax.debug.print('{}', token)  # prevent dead code elimination

        self.kw: dict = dict(
            key=random.key(2025_04_25_15_57),
            bart=simple_init(P, 0, NTREE),
            n_save=n_save,
            n_burn=0,
            n_skip=1,
            callback=callback,
        )

        # prepare copies of the args because of buffer donation
        key = jnp.copy(self.kw['key'])
        bart = tree_map(jnp.copy, self.kw['bart'])
        self.time_run_mcmc()
        # put copies in place of donated buffers
        self.kw.update(key=key, bart=bart)

    def time_run_mcmc(self, *_):
        """Time running the function."""
        try:
            run_mcmc(**self.kw)
        except JaxRuntimeError as e:
            is_expected = self.canary in str(e)
            if not is_expected:
                raise
        else:
            msg = 'expected JaxRuntimeError with canary not raised'
            raise RuntimeError(msg)
