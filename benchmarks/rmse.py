# bartz/benchmarks/rmse.py
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

"""Measure the predictive performance on test sets."""

from contextlib import redirect_stdout
from functools import partial
from inspect import signature
from io import StringIO

import jax
from jax import jit, lax, random
from jax import numpy as jnp
from jaxtyping import Array, Float32, Key

try:
    from bartz.BART import mc_gbart
except ImportError:
    from bartz.BART import gbart as mc_gbart  # pre v0.8.0

from benchmarks.latest_bartz.jaxext import split
from benchmarks.latest_bartz.testing import DGP, gen_data
from benchmarks.speed import get_default_platform


def make_data(
    key: Key[Array, ''], n_train: int, n_test: int, p: int
) -> tuple[DGP, DGP]:
    """Simulate data and split in train-test set."""
    return gen_data(
        key,
        n=n_train + n_test,
        p=p,
        k=1,
        q=2,
        lam=0,
        sigma2_lin=0.5,
        sigma2_quad=0.5,
        sigma2_eps=0.2,
        # the function `run_sim_impl` below compares the prediction with the
        # true latent mean, so the mse lower bound is 0 rather than sigma2_eps,
        # so the total latent variance is 1 to have a nice reference.
    ).split(n_train)


@partial(jit, static_argnums=(1, 2, 3))
def run_sim(
    keys: Key[Array, ' nreps'], n_train: int, n_test: int, p: int
) -> Float32[Array, '']:
    """Run simulation experiment for each key in `keys`, return avg mse."""
    run_sim_loop = lambda key: run_sim_impl(key, n_train, n_test, p)
    mses = lax.map(run_sim_loop, keys)
    return jnp.mean(mses)


def run_sim_impl(
    key: Key[Array, ' nreps'], n_train: int, n_test: int, p: int
) -> Float32[Array, '']:
    """Simulate data, run gbart, return mse."""
    keys = split(key)
    train, test = make_data(keys.pop(), n_train, n_test, p)

    kw: dict = dict(
        x_train=train.x,
        y_train=train.y.squeeze(0),
        x_test=test.x,
        nskip=1000,
        ndpost=1000,
        seed=keys.pop(),
        rm_const=None,  # needed to jit everything
        # None is needed for old versions, still works in place of False in new
        # ones
        printevery=2001,  # in old versions it can't be set to None
        bart_kwargs=dict(devices=jax.devices(get_default_platform())),
        mc_cores=1,
    )

    # adapt for older versions
    sig = signature(mc_gbart)

    def drop_if_missing(arg: str) -> None:
        if arg not in sig.parameters:
            kw.pop(arg)

    drop_if_missing('rm_const')
    drop_if_missing('bart_kwargs')
    drop_if_missing('mc_cores')

    bart = mc_gbart(**kw)

    return jnp.mean(jnp.square(bart.yhat_test_mean - test.mu.squeeze(0)))


class EvalGbart:
    """Out-of-sample evaluation of gbart."""

    # asv config
    timeout = 1000
    unit = 'latent_var'

    def track_mse(self) -> float:
        """Return the RMSE for predictions on a test set."""
        key = random.key(2025_06_26_21_02)
        keys = random.split(key, 30)
        with redirect_stdout(StringIO()):  # bc we can't set printevery=None
            return run_sim(keys, 50, 30, 5).item()
