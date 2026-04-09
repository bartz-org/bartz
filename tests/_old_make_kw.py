# bartz/tests/_old_make_kw.py
#
# Copyright (c) 2024-2026, The Bartz Contributors
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

"""Temporary verbatim copy of the old make_kw machinery for regression testing."""

from typing import Any, Literal

import jax
from jax import numpy as jnp
from jax import random
from jax.scipy.special import ndtr
from jaxtyping import Array, Bool, Float32, Key, Real

from bartz.jaxext import get_device_count, split


def gen_X(
    key: Key[Array, ''], p: int, n: int, kind: Literal['continuous', 'binary']
) -> Real[Array, 'p n']:
    """Generate a matrix of predictors."""
    match kind:
        case 'continuous':
            return random.uniform(key, (p, n), float, -2, 2)
        case 'binary':  # pragma: no branch
            return random.bernoulli(key, 0.5, (p, n)).astype(float)


def f(x: Real[Array, 'p n'], s: Real[Array, ' p']) -> Float32[Array, ' n']:
    """Conditional mean of the DGP."""
    T = 2
    return s @ jnp.cos(2 * jnp.pi / T * x) / jnp.sqrt(s @ s)


def gen_w(key: Key[Array, ''], n: int) -> Float32[Array, ' n']:
    """Generate a vector of error weights."""
    return jnp.exp(random.uniform(key, (n,), float, -1, 1))


def gen_y(
    key: Key[Array, ''],
    X: Real[Array, 'p n'],
    w: Float32[Array, ' n'] | None,
    kind: Literal['continuous', 'probit'],
    *,
    s: Real[Array, ' p'] | Literal['uniform', 'random'] = 'uniform',
) -> Float32[Array, ' n'] | Bool[Array, ' n']:
    """Generate responses given predictors."""
    keys = split(key, 3)

    p, n = X.shape
    if isinstance(s, jax.Array):
        pass
    elif s == 'random':
        s = jnp.exp(random.uniform(keys.pop(), (p,), float, -1, 1))
    elif s == 'uniform':  # pragma: no branch
        s = jnp.ones(p)

    match kind:
        case 'continuous':
            sigma = 0.1
            error = sigma * random.normal(keys.pop(), (n,))
            if w is not None:
                error *= w
            return f(X, s) + error

        case 'probit':  # pragma: no branch
            assert w is None
            _, n = X.shape
            error = random.normal(keys.pop(), (n,))
            prob = ndtr(f(X, s) + error)
            return random.bernoulli(keys.pop(), prob, (n,))


def make_kw(key: Key[Array, ''], variant: int) -> dict[str, Any]:
    """Return a dictionary of keyword arguments for BART."""
    keys = split(key, 5)

    match variant:
        # continuous regression with some settings that induce large types,
        # sparsity with free theta
        case 1:
            X = gen_X(keys.pop(), 2, 30, 'continuous')
            Xt = gen_X(keys.pop(), 2, 31, 'continuous')
            y = gen_y(keys.pop(), X, None, 'continuous', s='random')
            return dict(
                x_train=X,
                y_train=y,
                x_test=Xt,
                sparse=True,
                ntree=20,
                ndpost=100,
                nskip=50,
                printevery=50,
                usequants=False,
                numcut=256,  # > 255 to use uint16 for X and split_trees
                mc_cores=1,
                seed=keys.pop(),
                bart_kwargs=dict(
                    maxdepth=9,  # > 8 to use uint16 for leaf_indices
                    num_data_devices=min(2, get_device_count()),
                    init_kw=dict(
                        resid_num_batches=None,
                        count_num_batches=None,
                        prec_num_batches=None,
                        prec_count_num_trees=5,
                        target_platform=None,
                        save_ratios=True,
                    ),
                ),
            )

        # binary regression with binary X and high p
        case 2:
            p = 257  # > 256 to use uint16 for var_trees.
            X = gen_X(keys.pop(), p, 30, 'binary')
            Xt = gen_X(keys.pop(), p, 31, 'binary')
            y = gen_y(keys.pop(), X, None, 'probit')
            return dict(
                x_train=X,
                y_train=y,
                x_test=Xt,
                type='pbart',
                ntree=20,
                ndpost=100,
                nskip=50,
                keepevery=1,  # the default with binary would be 10
                printevery=None,
                usequants=True,
                # usequants=True with binary X to check the case in which the
                # splits are less than the statically known maximum
                numcut=255,
                seed=keys.pop(),
                mc_cores=2,  # keep this 2 on binary so continuous gets 1 and 2
                bart_kwargs=dict(
                    maxdepth=6,
                    num_chain_devices=None,
                    init_kw=dict(
                        save_ratios=False,
                        min_points_per_decision_node=None,
                        min_points_per_leaf=None,
                    ),
                ),
            )

        # continuous regression with error weights and sparsity with fixed theta
        case 3:  # pragma: no branch
            X = gen_X(keys.pop(), 2, 30, 'continuous')
            Xt = gen_X(keys.pop(), 2, 31, 'continuous')
            w = gen_w(keys.pop(), X.shape[1])
            y = gen_y(keys.pop(), X, w, 'continuous', s='random')
            return dict(
                x_train=X,
                x_test=Xt,
                y_train=y,
                w=w,
                sparse=True,
                theta=2,
                varprob=jnp.array([0.2, 0.8]),
                ntree=20,
                ndpost=100,
                nskip=50,
                printevery=50,
                usequants=True,
                numcut=10,
                seed=keys.pop(),
                mc_cores=2,
                bart_kwargs=dict(
                    maxdepth=8,  # 8 to check if leaf_indices changes type too soon
                    init_kw=dict(
                        save_ratios=True,
                        resid_num_batches=16,
                        count_num_batches=16,
                        prec_num_batches=16,
                        target_platform=None,
                        prec_count_num_trees=7,
                    ),
                ),
            )

        case _:  # pragma: no cover
            msg = f'Unknown variant {variant}'
            raise ValueError(msg)
