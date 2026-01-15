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
from io import StringIO

from jax import jit, random
from jax import numpy as jnp
from jaxtyping import Array, Key

from bartz.BART import gbart
from benchmarks.latest_bartz.testing import DGP, gen_data


@partial(jit, static_argnums=(1, 2, 3))
def make_data(
    key: Key[Array, ''], n_train: int, n_test: int, p: int
) -> tuple[DGP, DGP]:
    """Simulate data and split in train-test set."""
    return gen_data(
        key,
        n=n_train + n_test,
        p=p,
        k=1,
        q=4,
        lam=0,
        sigma2_lin=0.4,
        sigma2_quad=0.4,
        sigma2_eps=0.2,
    ).split(n_train)


class EvalGbart:
    """Out-of-sample evaluation of gbart."""

    # asv config
    timeout = 120.0
    unit = 'latent_sdev'

    def track_rmse(self) -> float:
        """Return the RMSE for predictions on a test set."""
        key = random.key(2025_06_26_21_02)
        train, test = make_data(key, 100, 1000, 20)
        with redirect_stdout(StringIO()):
            bart = gbart(
                train.x,
                train.y.squeeze(0),
                x_test=test.x,
                nskip=1000,
                ndpost=1000,
                seed=key,
            )
        return jnp.sqrt(
            jnp.mean(jnp.square(bart.yhat_test_mean - test.mu.squeeze(0)))
        ).item()
