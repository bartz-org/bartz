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
from dataclasses import dataclass
from functools import partial
from io import StringIO

from jax import jit, random, vmap
from jax import numpy as jnp

from bartz.BART import gbart


@partial(jit, static_argnums=(1, 2))
def simulate_data(key, n: int, p: int, max_interactions):
    """Simulate data for regression.

    This uses data-based standardization, so you have to generate train &
    test at once.
    """
    # split random key
    keys = list(random.split(key, 4))

    # generate matrices
    X = random.uniform(keys.pop(), (p, n))
    beta = random.normal(keys.pop(), (p,))
    A = random.normal(keys.pop(), (p, p))
    error = random.normal(keys.pop(), (n,))

    # make A banded to limit the number of interactions
    num_nonzero = 1 + (max_interactions - 1) // 2
    num_nonzero = jnp.clip(num_nonzero, 0, p)
    interaction_pattern = jnp.arange(p) < num_nonzero
    multi_roll = vmap(jnp.roll, in_axes=(None, 0))
    nonzero = multi_roll(interaction_pattern, jnp.arange(p))
    A *= nonzero

    # compute terms
    linear = beta @ X
    quadratic = jnp.einsum('ai,bi,ab->i', X, X, A)

    # equalize the terms
    mu = linear / jnp.std(linear) + quadratic / jnp.std(quadratic)
    mu /= jnp.std(mu)  # because linear and quadratic are correlated

    return X, mu, error


@dataclass(frozen=True)
class Data:
    """Data for regression."""

    X_train: jnp.ndarray
    mu_train: jnp.ndarray
    error_train: jnp.ndarray
    X_test: jnp.ndarray
    mu_test: jnp.ndarray
    error_test: jnp.ndarray

    @property
    def y_train(self):
        """Return the training targets."""
        return self.mu_train + self.error_train

    @property
    def y_test(self):
        """Return the test targets."""
        return self.mu_test + self.error_test


def make_data(key, n_train: int, n_test: int, p: int) -> Data:
    """Simulate data and split in train-test set."""
    X, mu, error = simulate_data(key, n_train + n_test, p, 5)
    return Data(
        X[:, :n_train],
        mu[:n_train],
        error[:n_train],
        X[:, n_train:],
        mu[n_train:],
        error[n_train:],
    )


class EvalGbart:
    """Out-of-sample evaluation of gbart."""

    # asv config
    timeout = 120.0
    unit = 'latent_sdev'

    def track_rmse(self) -> float:
        """Return the RMSE for predictions on a test set."""
        key = random.key(2025_06_26_21_02)
        data = make_data(key, 100, 1000, 20)
        with redirect_stdout(StringIO()):
            bart = gbart(
                data.X_train,
                data.y_train,
                x_test=data.X_test,
                nskip=1000,
                ndpost=1000,
                seed=key,
            )
        return jnp.sqrt(jnp.mean(jnp.square(bart.yhat_test_mean - data.mu_test))).item()
