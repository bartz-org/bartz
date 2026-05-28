# bartz/tests/test_interface.py
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

"""Test `bartz.Bart`.

This is the main suite of tests.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager, redirect_stderr
from dataclasses import dataclass, replace
from functools import partial
from gc import collect
from inspect import signature
from io import StringIO
from typing import Any, Literal, NamedTuple
from weakref import ReferenceType, ref

import jax
import numpy
import polars as pl
import pytest
from equinox import EquinoxRuntimeError, tree_at
from jax import (
    block_until_ready,
    config,
    debug_infs,
    debug_key_reuse,
    debug_nans,
    lax,
    random,
    tree,
    vmap,
)
from jax import numpy as jnp
from jax.sharding import Mesh, SingleDeviceSharding
from jax.tree_util import KeyPath, keystr
from jaxtyping import Array, Bool, Float32, Int32, Key, PyTree, Real, Shaped, UInt
from numpy.testing import assert_array_less
from pytest import CaptureFixture, FixtureRequest  # noqa: PT013
from pytest_subtests import SubTests

from bartz import Bart as OriginalBart
from bartz import PredictKind
from bartz._interface import predict_latent
from bartz._jaxext import get_default_device, get_device_count, is_key, split
from bartz.debug import TraceWithOffset, sample_prior
from bartz.grove import (
    check_trace,
    forest_depth_distr,
    is_actual_leaf,
    tree_actual_depth,
    tree_depth,
    tree_depths,
)
from bartz.mcmcloop import compute_varcount, evaluate_trace
from bartz.mcmcstep import State
from bartz.mcmcstep._state import chain_to_axis, chain_vmap_axes
from bartz.prepcovars import GivenSplitsBinner, RangeEvenBinner, UniqueQuantileBinner
from tests.test_mcmcstep import check_sharding, get_normal_spec, normalize_spec
from tests.util import (
    assert_array_equal,
    assert_close_matrices,
    assert_different_matrices,
    clipped_logit,
    periodic_sigint,
    rhat_rank,
)


class Bart(OriginalBart):
    """Wrapper that enables debug checks by default."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._check_trees(error=True)
        self._check_replicated_trees()


def gen_X(
    key: Key[Array, ''], p: int, n: int, kind: Literal['continuous', 'binary']
) -> Real[Array, 'p n']:
    """Generate a matrix of predictors."""
    match kind:
        case 'continuous':
            return random.uniform(key, (p, n), float, -2, 2)
        case 'binary':  # pragma: no branch
            return random.bernoulli(key, 0.5, (p, n)).astype(float)


def f(x: Real[Array, 'p n'], s: Real[Array, ' p'], k: int) -> Float32[Array, 'k n']:
    """Conditional mean of the DGP."""
    T = 2
    shifts = jnp.linspace(0, T, k, endpoint=False)
    arg = 2 * jnp.pi / T * x + shifts[:, None, None]  # (k, p, n)
    norm = jnp.sqrt(s @ s)
    return jnp.einsum('p,kpn->kn', s, jnp.cos(arg)) / norm


def gen_w(
    key: Key[Array, ''], shape: int | Sequence[int]
) -> Float32[Array, ' {shape}']:
    """Generate a vector of error weights."""
    if isinstance(shape, int):
        shape = (shape,)
    return jnp.exp(random.uniform(key, shape, float, -1, 1))


def gen_missing(
    key: Key[Array, ''], shape: int | Sequence[int], prob: float = 0.2
) -> Bool[Array, ' {shape}']:
    """Generate a boolean missingness mask with about ``prob`` true entries."""
    if isinstance(shape, int):
        shape = (shape,)
    return random.bernoulli(key, prob, shape)


def gen_y(
    key: Key[Array, ''],
    X: Real[Array, 'p n'],
    w: Float32[Array, ' n'] | Float32[Array, 'k n'] | None,
    outcome_type: str | Sequence[str] = 'continuous',
    *,
    k: int | None = None,
    s: Real[Array, ' p'] | Literal['uniform', 'random'] = 'uniform',
) -> Float32[Array, 'k n'] | Float32[Array, ' n']:
    """Generate responses given predictors."""
    keys = split(key, 2)

    p, n = X.shape
    if isinstance(s, Array):
        pass
    elif s == 'random':
        s = jnp.exp(random.uniform(keys.pop(), (p,), float, -1, 1))
    elif s == 'uniform':  # pragma: no branch
        s = jnp.ones(p)

    # normalize outcome_type to list
    effective_k = 1 if k is None else k
    if isinstance(outcome_type, str):
        outcome_type = [outcome_type] * effective_k
    assert len(outcome_type) == effective_k

    # mean and noise — always (effective_k, n)
    mu = f(X, s, effective_k)
    sigma = 0.1
    error = sigma * random.normal(keys.pop(), (effective_k, n))
    if w is not None:
        error = error * w
    y = mu + error

    # binarize
    for i, ot in enumerate(outcome_type):
        if ot == 'binary':
            y = y.at[i].set((y[i] > 0).astype(float))

    # squeeze for univariate
    if k is None:
        y = y.squeeze(axis=0)

    return y


def _bart_default(kw: dict[str, Any], param_name: str) -> Any:  # noqa: ANN401
    """Do `kw.get(param_name, <default in Bart>)`."""
    param = signature(OriginalBart).parameters[param_name]
    if param.default is param.empty:
        return kw[param_name]
    return kw.get(param_name, param.default)


class BartKW(NamedTuple):
    """Keyword arguments for `Bart` plus associated test data."""

    kw: dict[str, Any]
    x_test: Real[Array, 'p m']
    w_test: Float32[Array, ' m'] | Float32[Array, 'k m'] | None = None

    @property
    def uses_quantile_binner(self) -> bool:
        """Whether `kw['binner']` (a class or `partial`) is `UniqueQuantileBinner`."""
        binner = self.kw.get('binner')
        return getattr(binner, 'func', binner) is UniqueQuantileBinner

    @property
    def any_binary(self) -> bool:
        """Whether `kw['outcome_type']` includes any binary component."""
        outcome_type = _bart_default(self.kw, 'outcome_type')
        if isinstance(outcome_type, str):
            return outcome_type == 'binary'
        return 'binary' in outcome_type

    @property
    def all_binary(self) -> bool:
        """Whether all components of `kw['outcome_type']` are binary."""
        outcome_type = _bart_default(self.kw, 'outcome_type')
        if isinstance(outcome_type, str):
            return outcome_type == 'binary'
        return all(t == 'binary' for t in outcome_type)

    @property
    def is_mixed(self) -> bool:
        """Whether `kw['outcome_type']` has both binary and continuous components."""
        outcome_type = _bart_default(self.kw, 'outcome_type')
        if isinstance(outcome_type, str):
            return False
        return any(t == 'binary' for t in outcome_type) and any(
            t != 'binary' for t in outcome_type
        )

    @property
    def binary_mask(self) -> Bool[Array, '*k']:
        """Bool mask of binary components; shape matches ``y_train.shape[:-1]``."""
        outcome_type = _bart_default(self.kw, 'outcome_type')
        if isinstance(outcome_type, str):
            mask = jnp.bool_(outcome_type == 'binary')
        else:
            mask = jnp.array([t == 'binary' for t in outcome_type])
        return jnp.broadcast_to(mask, self.kshape)

    @property
    def p(self) -> int:
        """Number of predictors (``kw['x_train'].shape[0]``)."""
        p, _ = self.kw['x_train'].shape
        return p

    @property
    def n(self) -> int:
        """Number of training observations (``kw['x_train'].shape[1]``)."""
        _, n = self.kw['x_train'].shape
        return n

    @property
    def kshape(self) -> tuple[int, ...]:
        """Outer dimensions of `kw['y_train']`: ``()`` UV, ``(k,)`` MV."""
        return self.kw['y_train'].shape[:-1]

    @property
    def k(self) -> int | None:
        """Number of outcomes; ``None`` if univariate."""
        shape = self.kshape
        if not shape:
            return None
        (k,) = shape
        return k

    @property
    def num_chains(self) -> int | None:
        """Value of ``kw['num_chains']`` (or its `Bart` default)."""
        return _bart_default(self.kw, 'num_chains')

    @property
    def n_save(self) -> int:
        """Value of ``kw['n_save']`` (or its `Bart` default)."""
        return _bart_default(self.kw, 'n_save')

    @property
    def ndpost(self) -> int:
        """Total number of posterior samples across chains."""
        return self.n_save * (self.num_chains or 1)

    @property
    def num_trees(self) -> int:
        """Value of ``kw['num_trees']`` (or its `Bart` default)."""
        return _bart_default(self.kw, 'num_trees')

    @property
    def maxdepth(self) -> int:
        """Value of ``kw['maxdepth']`` (or its `Bart` default)."""
        return _bart_default(self.kw, 'maxdepth')

    @property
    def sparse(self) -> bool:
        """Value of ``kw['sparse']`` (or its `Bart` default)."""
        return _bart_default(self.kw, 'sparse')

    @property
    def max_bins(self) -> int:
        """Upper bound on the number of bins implied by `kw['binner']`."""
        binner = _bart_default(self.kw, 'binner')
        if isinstance(binner, partial):
            subcls = binner.func
            partial_kwargs = binner.keywords
        else:
            subcls = binner
            partial_kwargs = {}
        bound = signature(subcls).bind_partial(**partial_kwargs)
        bound.apply_defaults()
        defaults = dict(bound.arguments)
        if subcls is GivenSplitsBinner:
            return defaults['xinfo'].shape[1] + 1
        elif subcls in (UniqueQuantileBinner, RangeEvenBinner):
            return defaults['max_bins']
        else:
            msg = f'Cannot deduce max_bins for binner of type {subcls.__name__}'
            raise NotImplementedError(msg)


def make_kw(key: Key[Array, ''], variant: int) -> BartKW:
    """Return keyword arguments for `Bart` and test predictors."""
    keys = split(key, 10)  # 10 is just some high number
    n = 20
    nt = 21
    p = 2
    high_p = 257  # > 256 to use uint16 for var_trees.
    common = dict(num_trees=20, n_save=50, n_burn=50, seed=keys.pop())

    match variant:
        # continuous regression with some settings that induce large types,
        # sparsity with free theta
        case 1:
            X = gen_X(keys.pop(), p, n, 'continuous')
            bkw = BartKW(
                kw=dict(
                    x_train=X,
                    y_train=gen_y(keys.pop(), X, None, 'continuous', s='random'),
                    outcome_type='continuous',
                    sparse=True,
                    **common,
                    printevery=50,
                    binner=partial(RangeEvenBinner, max_bins=257),
                    # > 256 to use uint16 for X and split_trees
                    num_chains=None,
                    maxdepth=9,  # > 8 to use uint16 for leaf_indices
                    init_kw=dict(
                        resid_num_batches=None,
                        count_num_batches=None,
                        prec_num_batches=None,
                        prec_count_num_trees=5,
                        target_platform=None,
                        save_ratios=True,
                        min_points_per_leaf=5,
                    ),
                ),
                x_test=gen_X(keys.pop(), p, nt, 'continuous'),
            )

        # binary regression with binary X and high p
        case 2:
            X = gen_X(keys.pop(), high_p, n, 'binary')
            bkw = BartKW(
                kw=dict(
                    x_train=X,
                    y_train=gen_y(keys.pop(), X, None, 'binary'),
                    outcome_type='binary',
                    **common,
                    n_skip=1,  # the mc_gbart default with binary would be 10
                    printevery=None,
                    # quantile-binned, with binary X, to check the case in
                    # which the splits are less than the statically known
                    # maximum; full-sample quantilization for determinism
                    binner=partial(
                        UniqueQuantileBinner, max_bins=256, max_subsample=None
                    ),
                    num_chains=2,
                    maxdepth=6,
                    num_data_devices=min(2, get_device_count()),
                    num_chain_devices=None,  # for mc_gbart, turn autoshard off
                    init_kw=dict(
                        save_ratios=False,
                        min_points_per_decision_node=None,
                        min_points_per_leaf=None,
                    ),
                ),
                x_test=gen_X(keys.pop(), high_p, nt, 'binary'),
            )

        # continuous regression with error weights and sparsity with fixed theta
        case 3:
            X = gen_X(keys.pop(), p, n, 'continuous')
            w = gen_w(keys.pop(), X.shape[1])
            bkw = BartKW(
                kw=dict(
                    x_train=X,
                    y_train=gen_y(keys.pop(), X, w, 'continuous', s='random'),
                    outcome_type='continuous',
                    w=w,
                    sparse=True,
                    theta=2,
                    varprob=jnp.array([0.2, 0.8]),
                    **common,
                    printevery=50,
                    binner=partial(
                        UniqueQuantileBinner, max_bins=11, max_subsample=None
                    ),
                    num_chains=2,
                    num_chain_devices=min(2, get_device_count()),
                    maxdepth=8,  # 8 to check if leaf_indices changes type too soon
                    init_kw=dict(
                        save_ratios=True,
                        resid_num_batches=16,
                        count_num_batches=16,
                        prec_num_batches=16,
                        target_platform=None,
                        prec_count_num_trees=7,
                        min_points_per_leaf=5,
                    ),
                ),
                x_test=gen_X(keys.pop(), p, nt, 'continuous'),
                w_test=gen_w(keys.pop(), nt),
            )

        # multivariate continuous regression with error weights and some
        # settings that induce large types, sparsity with free theta
        case 4:
            X = gen_X(keys.pop(), p, n, 'continuous')
            w = gen_w(keys.pop(), X.shape[1])
            bkw = BartKW(
                kw=dict(
                    x_train=X,
                    y_train=gen_y(keys.pop(), X, w, 'continuous', k=2, s='random'),
                    outcome_type='continuous',
                    w=w,
                    sparse=True,
                    **common,
                    printevery=50,
                    binner=partial(RangeEvenBinner, max_bins=257),
                    # > 256 to use uint16 for X and split_trees
                    num_chains=None,
                    maxdepth=9,  # > 8 to use uint16 for leaf_indices
                    init_kw=dict(
                        resid_num_batches=None,
                        count_num_batches=None,
                        prec_num_batches=None,
                        prec_count_num_trees=5,
                        target_platform=None,
                        save_ratios=True,
                        min_points_per_leaf=5,
                    ),
                ),
                x_test=gen_X(keys.pop(), p, nt, 'continuous'),
                w_test=gen_w(keys.pop(), nt),
            )

        # multivariate binary regression with binary X and high p
        case 5:
            X = gen_X(keys.pop(), high_p, n, 'binary')
            bkw = BartKW(
                kw=dict(
                    x_train=X,
                    y_train=gen_y(keys.pop(), X, None, 'binary', k=2),
                    outcome_type='binary',
                    **common,
                    printevery=None,
                    # quantile-binned with binary X, deterministic via
                    # full-sample quantilization
                    binner=partial(
                        UniqueQuantileBinner, max_bins=256, max_subsample=None
                    ),
                    num_chains=2,
                    maxdepth=6,
                    num_data_devices=min(2, get_device_count()),
                    init_kw=dict(
                        save_ratios=False,
                        min_points_per_decision_node=None,
                        min_points_per_leaf=None,
                    ),
                ),
                x_test=gen_X(keys.pop(), high_p, nt, 'binary'),
            )

        # multivariate mixed binary-continuous regression with sparsity with
        # fixed theta and missing subunits
        case 6:
            X = gen_X(keys.pop(), p, n, 'continuous')
            outcome_type = ['continuous', 'binary', 'binary']
            bkw = BartKW(
                kw=dict(
                    x_train=X,
                    y_train=gen_y(keys.pop(), X, None, outcome_type, s='random', k=3),
                    outcome_type=outcome_type,
                    missing=gen_missing(keys.pop(), (len(outcome_type), n)),
                    sparse=True,
                    theta=2,
                    varprob=jnp.array([0.2, 0.8]),
                    **common,
                    printevery=50,
                    binner=partial(
                        UniqueQuantileBinner, max_bins=11, max_subsample=None
                    ),
                    num_chains=2,
                    num_chain_devices=min(2, get_device_count()),
                    maxdepth=8,  # 8 to check if leaf_indices changes type too soon
                    init_kw=dict(
                        save_ratios=True,
                        resid_num_batches=16,
                        count_num_batches=16,
                        prec_num_batches=16,
                        target_platform=None,
                        prec_count_num_trees=7,
                        min_points_per_leaf=5,
                    ),
                ),
                x_test=gen_X(keys.pop(), p, nt, 'continuous'),
            )

        # multivariate continuous regression with vector weights
        case 7:  # pragma: no branch
            k = 2
            X = gen_X(keys.pop(), p, n, 'continuous')
            w = gen_w(keys.pop(), (k, n))
            bkw = BartKW(
                kw=dict(
                    x_train=X,
                    y_train=gen_y(keys.pop(), X, w, 'continuous', k=k, s='random'),
                    outcome_type='continuous',
                    w=w,
                    **common,
                    num_chains=None,
                ),
                x_test=gen_X(keys.pop(), p, nt, 'continuous'),
                w_test=gen_w(keys.pop(), (k, nt)),
            )

        case _:  # pragma: no cover
            msg = f'Unknown variant {variant}'
            raise ValueError(msg)

    return bkw


# test only the multivariate variants, because the other ones are tested in
# test_BART.py
@pytest.fixture(
    params=(
        pytest.param(4, id='v4'),
        pytest.param(5, id='v5'),
        pytest.param(6, id='v6'),
        pytest.param(7, id='v7'),
    ),
    scope='module',
)
def variant(request: FixtureRequest) -> int:
    """Return a parametrized indicator to select different BART configurations."""
    return request.param


@pytest.fixture
def bkw(keys: split, variant: int) -> BartKW:
    """Return keyword arguments for Bart and test predictors."""
    return make_kw(keys.pop(), variant)


def set_num_datapoints(kw: dict, n: int) -> dict:
    """Set the number of datapoints in the kw dictionary."""
    assert n <= kw['y_train'].shape[-1]
    kw = kw.copy()
    kw['x_train'] = kw['x_train'][:, :n]
    kw['y_train'] = kw['y_train'][..., :n]
    if kw.get('w') is not None:
        kw['w'] = kw['w'][..., :n]
    if kw.get('missing') is not None:
        kw['missing'] = kw['missing'][..., :n]
    return kw


def test_meta_bkw_is_not_shared(bkw: BartKW) -> None:
    """Check that the kw dictionary is not shared across tests."""
    bkw.kw.clear()
    # this will make other tests fail if it's a shared dictionary


@dataclass(frozen=True)
class CachedBart:
    """Pre-computed BART run shared between multiple tests that do not change the arguments."""

    bkw: BartKW
    bart: Bart


class TestWithCachedBart:
    """Group of slow tests that check the same BART run, for efficiency."""

    @pytest.fixture(scope='class')
    def cachedbart(self, variant: int) -> CachedBart:
        """Return a pre-computed Bart."""
        key = random.key(0x139CD0C0)
        keys = random.split(key, 10)  # 10 is just some high number
        key = keys[variant]
        bkw = make_kw(key, variant)
        kw = bkw.kw

        nchains = 4
        kw.update(
            num_trees=max(2 * bkw.n, bkw.p),
            n_burn=3000,
            n_save=1000,
            n_skip=1,
            num_chains=nchains,
        )
        init_kw = dict(kw.get('init_kw', {}))
        init_kw.update(min_points_per_decision_node=10, min_points_per_leaf=5)
        kw['init_kw'] = init_kw

        return CachedBart(bkw=bkw, bart=Bart(**kw))

    def test_residuals_accuracy(self, cachedbart: CachedBart) -> None:
        """Check that running residuals are close to the recomputed final residuals."""
        accum_resid, actual_resid = cachedbart.bart._compare_resid(
            y=cachedbart.bkw.kw['y_train']
        )
        assert_close_matrices(accum_resid, actual_resid, rtol=1e-4, reduce_rank=True)

    def test_convergence(self, cachedbart: CachedBart, subtests: SubTests) -> None:
        """Run multiple chains and check convergence with rhat."""
        bart = cachedbart.bart
        bkw = cachedbart.bkw
        num_chains = bkw.num_chains
        nsamples = bkw.n_save

        with subtests.test('yhat_train'):
            yhat_train = bart.predict('train', kind='latent_samples')
            yhat_train_chains = yhat_train.reshape(num_chains, nsamples, -1)
            rhat_yhat_train = rhat_rank(yhat_train_chains, split=True)
            assert_array_less(rhat_yhat_train, 1.1)

        if bkw.all_binary:
            with subtests.test('prob_train'):
                prob_train = bart.predict('train', kind='mean_samples')
                prob_train_chains = prob_train.reshape(num_chains, nsamples, -1)
                rhat_prob_train = rhat_rank(
                    clipped_logit(prob_train_chains, 1e-5), split=True
                )
                assert_array_less(rhat_prob_train, 1.005)

        elif bkw.is_mixed:
            with subtests.test('sigma'):
                # mixed regression: check get_error_sdev, dropping binary
                # components (NaN sdev)
                with debug_nans(False):
                    sigma = bart.get_error_sdev().reshape(num_chains, nsamples, -1)
                    binary_mask = bkw.binary_mask
                    if (
                        binary_mask.ndim > 0
                    ):  # pragma: no branch, always on in mv variants
                        sigma = sigma[:, :, ~binary_mask]
                rhat_sigma = rhat_rank(sigma, split=True)
                assert_array_less(rhat_sigma, 1.05)

        else:
            with subtests.test('error_cov_inv'):
                # all continuous: check full precision matrix convergence
                # using upper triangular elements (matrix is symmetric).
                # When the error covariance is constrained diagonal (2-D
                # inv_sdev_scale), off-diagonal entries are deterministically
                # zero, so rhat is undefined there; check only the diagonal.
                error_cov_inv = chain_to_axis(
                    bart._main_trace.error_cov_inv,
                    chain_vmap_axes(bart._main_trace).error_cov_inv,
                )
                if error_cov_inv.ndim == 2:  # pragma: no cover, only mv by default
                    error_cov_inv = error_cov_inv[:, :, None, None]
                _, _, k, _ = error_cov_inv.shape
                inv_sdev = bart._mcmc_state.inv_sdev_scale
                if inv_sdev is not None and inv_sdev.ndim == 2:
                    ti = tj = jnp.arange(k)
                else:
                    ti, tj = jnp.triu_indices(k)
                error_cov_inv = error_cov_inv[:, :, ti, tj]
                rhat_prec = rhat_rank(error_cov_inv, split=True)
                assert_array_less(rhat_prec, 1.01)

        if bkw.p < bkw.n:
            with subtests.test('varcount'):
                varcount_vals = bart.varcount.reshape(num_chains, nsamples, bkw.p)
                rhat_varcount = rhat_rank(varcount_vals, split=True)
                assert_array_less(rhat_varcount, 1.4)

            if bkw.sparse:  # pragma: no branch
                with subtests.test('varprob'):
                    varprob_vals = bart.varprob.reshape(num_chains, nsamples, bkw.p)
                    rhat_varprob = rhat_rank(varprob_vals[:, :, 1:], split=True)
                    assert_array_less(rhat_varprob, 1.3)

    def test_different_chains(self, cachedbart: CachedBart) -> None:
        """Check that different chains give different results."""
        bart = cachedbart.bart

        step_theta = bart._mcmc_state.forest.rho is not None

        def assert_different(x: PyTree[Array], **kwargs: Any) -> None:
            def assert_different(
                path: KeyPath, x: Array | None, chain_axis: int | None
            ) -> None:
                str_path = keystr(path)
                if (
                    (str_path.endswith('.theta') and not step_theta)
                    or (
                        str_path.endswith('.error_cov_inv')
                        and bart._mcmc_state.error_cov_df is None
                        # fixed covariance matrix, all chains equal
                    )
                    or (
                        x is not None
                        and jnp.issubdtype(x.dtype, jnp.integer)
                        and x.ndim == 1
                        # too few integers, may collide
                    )
                ):
                    return
                if x is not None and chain_axis is not None:
                    ref = jnp.broadcast_to(x.mean(chain_axis, keepdims=True), x.shape)
                    assert_different_matrices(
                        x.astype(jnp.float32),
                        ref,
                        reduce_rank=True,
                        ord='fro' if x.ndim >= 2 else 2,
                        atol=0,
                        err_msg=f'chain samples are not different for {str_path}\n',
                        **kwargs,
                    )

            axes = chain_vmap_axes(x)
            tree.map_with_path(assert_different, x, axes, is_leaf=lambda x: x is None)

        assert_different(bart._mcmc_state, rtol=0.01)
        assert_different(bart._main_trace, rtol=0.01)
        assert_different(bart._burnin_trace, rtol=0.01)


def test_sequential_guarantee(bkw: BartKW, subtests: SubTests) -> None:
    """Check that the way iterations are saved does not influence the result."""
    kw = bkw.kw
    kw['n_skip'] = 1
    bart1 = Bart(**kw)

    num_chains = bkw.num_chains or 1
    y_shape = kw['y_train'].shape

    # run moving some samples from burn-in to main
    kw2 = dict(kw)
    kw2['seed'] = random.clone(kw2['seed'])
    if bkw.sparse:
        init_kw = dict(kw2.get('init_kw', {}))
        init_kw.setdefault('sparse_on_at', kw2['n_burn'] // 2)
        kw2['init_kw'] = init_kw
    delta = 1
    kw2['n_burn'] -= delta
    kw2['n_save'] += delta
    bart2 = Bart(**kw2)
    bart2_yhat_train = (
        bart2.predict('train', kind='latent_samples')
        .reshape(num_chains, kw2['n_save'], *y_shape)[:, delta:]
        .reshape(num_chains * bkw.n_save, *y_shape)
    )

    with subtests.test('shift burn-in'):
        rtol = (
            0
            if bart1.predict('train', kind='latent_samples').platform() == 'cpu'
            else 2e-6
        )
        assert_close_matrices(
            bart1.predict('train', kind='latent_samples'),
            bart2_yhat_train,
            rtol=rtol,
            reduce_rank=True,
        )

    # run keeping 1 every 2 samples
    kw3 = dict(kw)
    kw3['seed'] = random.clone(kw3['seed'])
    kw3['n_skip'] = 2
    bart3 = Bart(**kw3)
    bart1_yhat_train = bart1.predict('train', kind='latent_samples').reshape(
        num_chains, kw3['n_save'], *y_shape
    )[:, 1::2, :, ...]
    bart3_yhat_train = bart3.predict('train', kind='latent_samples').reshape(
        num_chains, kw3['n_save'], *y_shape
    )[:, : bart1_yhat_train.shape[1], :, ...]

    with subtests.test('change thinning'):
        rtol = (
            0
            if bart1.predict('train', kind='latent_samples').platform() == 'cpu'
            else 2e-6
        )
        assert_close_matrices(
            bart1_yhat_train, bart3_yhat_train, rtol=rtol, reduce_rank=True
        )


def test_missing_ignored(bkw: BartKW, keys: split) -> None:
    """Garbage finite y values at missing positions don't affect the fit."""
    kw = bkw.kw
    y_train = kw['y_train']
    missing = gen_missing(keys.pop(), y_train.shape)
    kw['missing'] = missing

    # Pin y-dependent priors otherwise they are influenced by garbage values
    kw['offset'] = 0.0
    kw['tau_num'] = 2.0
    if not bkw.all_binary:
        kw['lambda_'] = 1.0

    bart1 = Bart(**kw)

    garbage = random.normal(keys.pop(), y_train.shape) * 1e3
    kw2 = dict(
        kw, seed=random.clone(kw['seed']), y_train=jnp.where(missing, garbage, y_train)
    )
    bart2 = Bart(**kw2)

    yhat1 = bart1.predict('train', kind='latent_samples')
    yhat2 = bart2.predict('train', kind='latent_samples')
    rtol = 0 if yhat1.platform() == 'cpu' else 1e-5
    assert_close_matrices(yhat1, yhat2, rtol=rtol, reduce_rank=True)


def test_output_shapes(bkw: BartKW, keys: split) -> None:
    """Check the output shapes of the Bart predictions and attributes."""
    kw = bkw.kw
    bart = Bart(**kw)

    ndpost = bkw.ndpost
    _, m = bkw.x_test.shape
    k = bkw.kshape

    assert bart.offset.shape == k

    assert bart.predict('train', kind='mean_samples').shape == (ndpost, *k, bkw.n)
    assert bart.predict('train', kind='mean').shape == (*k, bkw.n)
    assert bart.predict('train', kind='latent_samples').shape == (ndpost, *k, bkw.n)
    assert bart.predict('train', kind='outcome_samples', key=keys.pop()).shape == (
        ndpost,
        *k,
        bkw.n,
    )

    assert bart.predict(bkw.x_test, kind='mean_samples').shape == (ndpost, *k, m)
    assert bart.predict(bkw.x_test, kind='mean').shape == (*k, m)
    assert bart.predict(bkw.x_test, kind='latent_samples').shape == (ndpost, *k, m)
    assert bart.predict(
        bkw.x_test, kind='outcome_samples', w=bkw.w_test, key=keys.pop()
    ).shape == (ndpost, *k, m)

    with debug_nans(False):
        assert bart.get_error_sdev().shape == (ndpost, *k)
        assert bart.get_error_sdev(mean=True).shape == k

    if bkw.all_binary:
        assert bart.sigest is None
    else:
        assert bart.sigest.shape == k

    assert bart.varcount.shape == (ndpost, bkw.p)
    assert bart.varcount_mean.shape == (bkw.p,)
    assert bart.varprob.shape == (ndpost, bkw.p)
    assert bart.varprob_mean.shape == (bkw.p,)

    # get_latent_prec shape
    n_burn = kw['n_burn']
    prec = bart.get_latent_prec()
    if bkw.num_chains is not None:
        assert prec.shape == (bkw.num_chains, n_burn + bkw.n_save, *k, *k)
    else:
        assert prec.shape == (n_burn + bkw.n_save, *k, *k)


def test_output_types(bkw: BartKW, keys: split) -> None:
    """Check the output types of all the attributes of Bart."""
    kw = bkw.kw
    bart = Bart(**kw)

    if not bkw.all_binary:
        assert bart.sigest.dtype == jnp.float32
    assert bart.offset.dtype == jnp.float32
    assert isinstance(bart.n_save, int)
    assert bart.predict('train', kind='mean').dtype == jnp.float32
    assert bart.predict('train', kind='mean_samples').dtype == jnp.float32
    assert bart.predict('train', kind='latent_samples').dtype == jnp.float32
    assert (
        bart.predict('train', kind='outcome_samples', key=keys.pop()).dtype
        == jnp.float32
    )
    assert bart.get_latent_prec().dtype == jnp.float32
    with debug_nans(False):
        assert bart.get_error_sdev().dtype == jnp.float32
        assert bart.get_error_sdev(mean=True).dtype == jnp.float32
    assert bart.varcount.dtype == jnp.int32
    assert bart.varcount_mean.dtype == jnp.float32
    assert bart.varprob.dtype == jnp.float32
    assert bart.varprob_mean.dtype == jnp.float32


def test_output_ranges(bkw: BartKW, keys: split) -> None:
    """Check value constraints on Bart outputs."""
    kw = bkw.kw
    bart = Bart(**kw)

    binary_mask = bkw.binary_mask

    mean = bart.predict('train', kind='mean')
    bmean = mean[binary_mask, ...]
    assert jnp.all((bmean >= 0) & (bmean <= 1))

    outcomes = bart.predict('train', kind='outcome_samples', key=keys.pop())
    boutcomes = outcomes[:, binary_mask, ...]
    assert jnp.all((boutcomes == 0) | (boutcomes == 1))

    with debug_nans(False):
        sdev = bart.get_error_sdev()
        assert jnp.all(jnp.isnan(sdev[..., binary_mask]))
        assert jnp.all(jnp.isfinite(sdev[..., ~binary_mask]))
        assert jnp.all(sdev[..., ~binary_mask] > 0)

        sdev_mean = bart.get_error_sdev(mean=True)
        assert jnp.all(jnp.isnan(sdev_mean[binary_mask]))
        assert jnp.all(jnp.isfinite(sdev_mean[~binary_mask]))
        assert jnp.all(sdev_mean[~binary_mask] > 0)

    # sigest values for mixed
    if bkw.all_binary:
        assert bart.sigest is None
    else:
        assert jnp.all(bart.sigest[binary_mask] == 0.0)
        assert jnp.all(bart.sigest[~binary_mask] > 0)

    # get_latent_prec: symmetry and positive definiteness
    if bkw.k is not None:  # pragma: no branch, always mv with defaults
        prec = bart.get_latent_prec()
        assert_close_matrices(prec, prec.mT, rtol=1e-6, reduce_rank=True)
        eigvals = jnp.linalg.eigvalsh(prec)
        assert jnp.all(eigvals > 0)


def test_predict_means(bkw: BartKW, keys: split, subtests: SubTests) -> None:
    """Check that the various outputs of `Bart.predict` are consistent."""
    bart = Bart(**bkw.kw)

    mean = bart.predict('train', kind='mean')  # (*k, n)
    mean_samples = bart.predict('train', kind='mean_samples')  # (ndpost, *k, n)
    latent_samples = bart.predict('train', kind='latent_samples')  # (ndpost, *k, n)
    outcome_samples = bart.predict('train', kind='outcome_samples', key=keys.pop())
    # outcome_samples has shape (num_chains*n_save, *k, n)

    with subtests.test('mean_samples vs mean'):
        mean_from_mean_samples = mean_samples.mean(0)
        assert_close_matrices(mean, mean_from_mean_samples, rtol=1e-7)

    with subtests.test('latent_samples vs mean_samples'):
        binary_mask = bkw.binary_mask
        cont_mean_samples = mean_samples[:, ~binary_mask, ...]
        cont_latent_samples = latent_samples[:, ~binary_mask, ...]
        assert_close_matrices(
            cont_mean_samples.T, cont_latent_samples.T, reduce_rank=True
        )

    with subtests.test('outcome_samples vs mean'):
        mean_from_os = outcome_samples.mean(0)
        sdev = outcome_samples.std(0) / jnp.sqrt(outcome_samples.shape[0])
        t = jnp.abs(mean - mean_from_os) / sdev
        assert jnp.all(t < 5)


def test_predict(bkw: BartKW) -> None:
    """Check that predict with x_train matches predict with 'train'."""
    kw = bkw.kw
    bart = Bart(**kw)
    yhat_train = bart.predict(kw['x_train'], kind='latent_samples')
    assert_close_matrices(
        bart.predict('train', kind='latent_samples'),
        yhat_train,
        rtol=1e-6,
        reduce_rank=True,
    )


class TestVarprobAttr:
    """Test the `Bart.varprob` attribute."""

    def test_basic_properties(self, bkw: BartKW) -> None:
        """Basic checks of the `varprob` attribute."""
        kw = bkw.kw
        bart = Bart(**kw)

        assert jnp.all(bart.varprob >= 0)
        assert jnp.all(bart.varprob <= 1)
        varprob_sum = bart.varprob.sum(axis=1)
        assert_close_matrices(varprob_sum, jnp.ones_like(varprob_sum), rtol=1e-6)

        if not bkw.sparse:
            unique = jnp.unique(bart.varprob)
            assert unique.size in (1, 2)
            if unique.size == 2:  # pragma: no cover
                assert unique[0] == 0

        assert_array_equal(bart.varprob_mean, bart.varprob.mean(axis=0))

    def test_blocked_vars(self, keys: split) -> None:
        """Check that varprob = 0 on predictors blocked a priori."""
        X = gen_X(keys.pop(), 2, 30, 'continuous')
        y = gen_y(keys.pop(), X, None, 'continuous')
        with debug_nans(False):
            xinfo = jnp.array([[jnp.nan], [0]])
        bart = Bart(
            x_train=X,
            y_train=y,
            binner=partial(GivenSplitsBinner, xinfo=xinfo),
            seed=keys.pop(),
        )
        assert_array_equal(bart._mcmc_state.forest.max_split, [0, 1], strict=False)
        assert_array_equal(bart.varprob_mean, [0, 1], strict=False)
        assert jnp.all(bart.varprob_mean == bart.varprob)


@pytest.mark.parametrize('theta', ['fixed', 'free'])
def test_variable_selection(keys: split, theta: Literal['fixed', 'free']) -> None:
    """Check that variable selection works."""
    p = 100
    peff = 5
    n = 1000

    mask = jnp.zeros(p, bool).at[:peff].set(True)
    mask = random.permutation(keys.pop(), mask)
    s = mask.astype(float)

    X = gen_X(keys.pop(), p, n, 'continuous')
    y = gen_y(keys.pop(), X, None, 'continuous', s=s)

    bart = Bart(
        x_train=X,
        y_train=y,
        n_burn=1000,
        sparse=True,
        theta=peff if theta == 'fixed' else None,
        seed=keys.pop(),
    )

    assert bart.varprob_mean[mask].sum() >= 0.9
    assert bart.varprob_mean[mask].min().item() > 0.5 / peff
    assert bart.varprob_mean[~mask].max().item() < 1 / (p - peff)


def test_scale_shift(bkw: BartKW) -> None:
    """Check self-consistency of rescaling the inputs.

    For mixed binary-continuous outcomes, only the continuous components
    of `y_train` are rescaled, and binary components are matched between
    `bart1` and `bart2` via `bkw.binary_mask`.
    """
    kw = bkw.kw

    if bkw.all_binary:
        pytest.skip('Cannot rescale binary responses.')

    bart1 = Bart(**kw)
    mask = bkw.binary_mask

    offset = 0.4703189
    scale = 0.5294714

    y = kw['y_train']
    y = jnp.where(mask[..., None], y, offset + y * scale)

    kw2 = dict(kw)
    kw2.update(y_train=y, seed=random.clone(kw['seed']))
    bart2 = Bart(**kw2)

    assert_close_matrices(
        bart1.offset,
        jnp.where(mask, bart2.offset, (bart2.offset - offset) / scale),
        rtol=1e-5,
    )
    assert_close_matrices(
        bart1.sigest, jnp.where(mask, bart2.sigest, bart2.sigest / scale), rtol=1e-6
    )
    assert_array_equal(bart1._mcmc_state.error_cov_df, bart2._mcmc_state.error_cov_df)

    masked_scale = jnp.where(mask, 1.0, scale)
    if masked_scale.ndim:
        masked_scale = masked_scale[:, None]
    cov_scale = masked_scale * masked_scale.T

    assert_close_matrices(
        bart1._mcmc_state.forest.leaf_prior_cov_inv,
        bart2._mcmc_state.forest.leaf_prior_cov_inv * cov_scale,
        rtol=1e-6,
    )

    assert_close_matrices(
        bart1._mcmc_state.error_cov_scale * cov_scale,
        bart2._mcmc_state.error_cov_scale,
        rtol=1e-6,
    )

    mask_pred = mask[..., None]

    yhat1 = bart1.predict('train', kind='latent_samples')
    yhat2 = bart2.predict('train', kind='latent_samples')
    assert_close_matrices(
        yhat1,
        jnp.where(mask_pred, yhat2, (yhat2 - offset) / scale),
        rtol=1e-5,
        reduce_rank=True,
    )

    mean1 = bart1.predict('train', kind='mean')
    mean2 = bart2.predict('train', kind='mean')
    assert_close_matrices(
        mean1,
        jnp.where(mask_pred, mean2, (mean2 - offset) / scale),
        rtol=1e-5,
        reduce_rank=True,
    )

    yhat_test1 = bart1.predict(kw['x_train'], kind='latent_samples')
    yhat_test2 = bart2.predict(kw['x_train'], kind='latent_samples')
    assert_close_matrices(
        yhat_test1,
        jnp.where(mask_pred, yhat_test2, (yhat_test2 - offset) / scale),
        rtol=1e-5,
        reduce_rank=True,
    )

    yhat_test_mean1 = bart1.predict(kw['x_train'], kind='mean')
    yhat_test_mean2 = bart2.predict(kw['x_train'], kind='mean')
    assert_close_matrices(
        yhat_test_mean1,
        jnp.where(mask_pred, yhat_test_mean2, (yhat_test_mean2 - offset) / scale),
        rtol=1e-5,
        reduce_rank=True,
    )

    # binary positions of get_error_sdev are NaN; replace with 0 to compare.
    with debug_nans(False):
        sdev_actual = jnp.where(mask, 0.0, bart1.get_error_sdev())
        sdev_expected = jnp.where(mask, 0.0, bart2.get_error_sdev() / scale)
        sdev_mean_actual = jnp.where(mask, 0.0, bart1.get_error_sdev(mean=True))
        sdev_mean_expected = jnp.where(
            mask, 0.0, bart2.get_error_sdev(mean=True) / scale
        )
    assert_close_matrices(sdev_actual, sdev_expected, rtol=1e-5, reduce_rank=True)
    assert_close_matrices(sdev_mean_actual, sdev_mean_expected, rtol=1e-5)


def test_min_points_per_decision_node(bkw: BartKW) -> None:
    """Check that the limit of at least 10 datapoints per decision node is respected."""
    kw = bkw.kw
    init_kw = dict(kw.get('init_kw', {}))
    init_kw['min_points_per_leaf'] = None
    kw['init_kw'] = init_kw
    bart = Bart(**kw)
    distr = bart._points_per_decision_node_distr()
    distr_marg = distr.sum(axis=(0, 1))

    min_points = init_kw.get('min_points_per_decision_node', 10)

    if min_points is None:
        assert distr_marg[9] > 0
    else:
        assert jnp.all(distr_marg[:min_points] == 0)
        assert jnp.any(distr_marg[min_points:] > 0)


def test_min_points_per_leaf(bkw: BartKW) -> None:
    """Check that the limit of at least 5 datapoints per leaf is respected."""
    kw = bkw.kw
    init_kw = dict(kw.get('init_kw', {}))
    init_kw['min_points_per_decision_node'] = None
    kw['init_kw'] = init_kw
    bart = Bart(**kw)
    distr = bart._points_per_leaf_distr()
    distr_marg = distr.sum(axis=(0, 1))

    min_points = init_kw.get('min_points_per_leaf')  # default None

    if min_points is None:
        assert distr_marg[4] > 0
    else:
        assert jnp.all(distr_marg[:min_points] == 0)
        assert distr_marg[min_points] > 0


@pytest.mark.parametrize('num_datapoints', [0, 1])
def test_zero_or_one_datapoint(bkw: BartKW, num_datapoints: int) -> None:
    """Check automatic data scaling with 0 or 1 datapoints."""
    kw = set_num_datapoints(bkw.kw, num_datapoints)

    if num_datapoints == 0 or bkw.uses_quantile_binner:
        nsplits = 10
        xinfo = jnp.broadcast_to(
            jnp.arange(nsplits, dtype=jnp.float32), (bkw.p, nsplits)
        )
        kw.update(binner=partial(GivenSplitsBinner, xinfo=xinfo))

    kw.update(num_data_devices=None)

    init_kw = dict(kw.get('init_kw', {}))
    init_kw.update(
        save_ratios=True, min_points_per_decision_node=None, min_points_per_leaf=None
    )
    kw['init_kw'] = init_kw

    bart = Bart(**kw)

    assert bart.predict('train', kind='latent_samples').shape == (
        bkw.ndpost,
        *bkw.kshape,
        num_datapoints,
    )

    # check bart.offset
    mask = bkw.binary_mask
    if num_datapoints == 0 or bkw.all_binary:
        assert_array_equal(bart.offset, jnp.zeros_like(bart.offset))
    else:
        assert_close_matrices(
            bart.offset[..., None],
            jnp.where(mask[..., None], 0.0, kw['y_train']),
            rtol=1e-6,
        )

    # check bart.sigest and set expected tau_num
    if bkw.all_binary:
        tau_num = 3
        assert bart.sigest is None
    else:
        tau_num = jnp.where(mask, 3.0, 1.0)
        expected_sigest = jnp.where(mask, 0.0, 1.0)
        assert_array_equal(bart.sigest, expected_sigest)

    # check leaf_prior_cov_inv
    expected_cov_inv = (2**2 * bkw.num_trees) / tau_num**2
    leaf_prior_cov_inv = bart._mcmc_state.forest.leaf_prior_cov_inv
    if leaf_prior_cov_inv.ndim == 2:  # pragma: no branch, always mv with defaults
        expected_cov_inv = jnp.eye(leaf_prior_cov_inv.shape[0]) * expected_cov_inv
    assert_close_matrices(leaf_prior_cov_inv, expected_cov_inv, rtol=1e-6)

    # with 1 datapoint in the multivariate case, log_likelihood is not exactly 0
    # because matrix inversion adds an epsilon to handle ill-conditioned matrices
    atol = 0.0 if num_datapoints == 0 else 1e-4
    assert_close_matrices(
        bart._burnin_trace.log_likelihood,
        jnp.zeros_like(bart._burnin_trace.log_likelihood),
        atol=atol,
        reduce_rank=True,
    )
    assert_close_matrices(
        bart._main_trace.log_likelihood,
        jnp.zeros_like(bart._main_trace.log_likelihood),
        atol=atol,
        reduce_rank=True,
    )


def test_two_datapoints(bkw: BartKW) -> None:
    """Check automatic data scaling with 2 datapoints."""
    kw = set_num_datapoints(bkw.kw, 2)
    init_kw = dict(kw.get('init_kw', {}))
    init_kw.update(
        save_ratios=True, min_points_per_decision_node=None, min_points_per_leaf=None
    )
    kw['init_kw'] = init_kw
    bart = Bart(**kw)
    if not bkw.all_binary:
        ref_sigest = jnp.where(bkw.binary_mask, 0.0, kw['y_train'].std(axis=-1))
        assert_close_matrices(bart.sigest, ref_sigest, rtol=1e-6)
    if bkw.uses_quantile_binner:
        assert jnp.all(bart._mcmc_state.forest.max_split <= 1)
    assert not jnp.all(bart._burnin_trace.log_likelihood == 0.0)
    assert not jnp.all(bart._main_trace.log_likelihood == 0.0)


def test_few_datapoints(bkw: BartKW) -> None:
    """Check that the trees cannot grow if there are not enough datapoints."""
    kw = bkw.kw
    init_kw = dict(kw.get('init_kw', {}))
    init_kw.update(min_points_per_decision_node=10, min_points_per_leaf=None)
    kw1 = dict(set_num_datapoints(kw, 8), init_kw=init_kw)
    bart = Bart(**kw1)
    yhat = bart.predict('train', kind='latent_samples')
    assert jnp.all(yhat == yhat[..., :, :1])

    init_kw2 = dict(kw.get('init_kw', {}))
    init_kw2.update(min_points_per_decision_node=None, min_points_per_leaf=5)
    kw2 = dict(
        set_num_datapoints(kw, 8), init_kw=init_kw2, seed=random.clone(kw['seed'])
    )
    bart = Bart(**kw2)
    yhat = bart.predict('train', kind='latent_samples')
    assert jnp.all(yhat == yhat[..., :, :1])


def test_xinfo() -> None:
    """Simple check that `GivenSplitsBinner` works as the `binner`."""
    with debug_nans(False):
        xinfo = jnp.array(
            [[1.1, 2.3, jnp.nan], [-50, 10, 20], [jnp.nan, jnp.nan, jnp.nan]]
        )
    kw = dict(
        x_train=jnp.empty((3, 0)),
        y_train=jnp.empty(0),
        n_save=0,
        n_burn=0,
        binner=partial(GivenSplitsBinner, xinfo=xinfo),
    )
    bart = Bart(**kw)

    xinfo_wo_nan = jnp.where(jnp.isnan(xinfo), jnp.finfo(jnp.float32).max, xinfo)
    assert_array_equal(bart._binner._splits, xinfo_wo_nan)
    assert_array_equal(bart._mcmc_state.forest.max_split, [2, 3, 0], strict=False)


def test_xinfo_wrong_p() -> None:
    """Check that `GivenSplitsBinner` rejects mismatched shapes."""
    with debug_nans(False):
        xinfo = jnp.array(
            [[1.1, 2.3, jnp.nan], [-50, 10, 20], [jnp.nan, jnp.nan, jnp.nan]]
        )
    kw = dict(
        x_train=jnp.empty((5, 0)),
        y_train=jnp.empty(0),
        n_save=0,
        n_burn=0,
        binner=partial(GivenSplitsBinner, xinfo=xinfo),
    )
    with pytest.raises(ValueError, match=r'xinfo\.shape'):
        Bart(**kw)


@pytest.mark.parametrize(('p', 'nsplits'), [(1, 1), (3, 2), (10, 1), (10, 255)])
def test_prior(keys: split, p: int, nsplits: int, subtests: SubTests) -> None:
    """Check that the posterior without data is equivalent to the prior."""
    bart = run_bart_like_prior(keys.pop(), p, nsplits, subtests)

    prior_trace = sample_prior_like(keys.pop(), bart, subtests)

    with subtests.test('number of stub trees'):
        nstub_mcmc = count_stub_trees(bart._main_trace.split_tree)
        nstub_prior = count_stub_trees(prior_trace.split_tree)
        rhat_nstub = rhat_rank([nstub_mcmc, nstub_prior], split=False)
        assert_array_less(rhat_nstub, 1.01)

    if (p, nsplits) != (1, 1):
        with subtests.test('number of simple trees'):
            nsimple_mcmc = count_simple_trees(bart._main_trace.split_tree)
            nsimple_prior = count_simple_trees(prior_trace.split_tree)
            rhat_nsimple = rhat_rank([nsimple_mcmc, nsimple_prior], split=False)
            assert_array_less(rhat_nsimple, 1.01)

        varcount_prior = compute_varcount(
            bart._mcmc_state.forest.max_split.size, prior_trace
        )

        with subtests.test('varcount'):
            rhat_varcount = rhat_rank([bart.varcount, varcount_prior], split=False)
            if p == 10:
                assert_array_less(rhat_varcount, 1.05)
            else:
                assert_array_less(rhat_varcount, 1.02)

        with subtests.test('number of nodes'):
            sum_varcount_mcmc = bart.varcount.sum(axis=1)
            sum_varcount_prior = varcount_prior.sum(axis=1)
            rhat_sum_varcount = rhat_rank(
                [sum_varcount_mcmc, sum_varcount_prior], split=False
            )
            assert_array_less(rhat_sum_varcount, 1.02)

        with subtests.test('imbalance index'):
            imb_mcmc = avg_imbalance_index(bart._main_trace.split_tree)
            imb_prior = avg_imbalance_index(prior_trace.split_tree)
            rhat_imb = rhat_rank([imb_mcmc, imb_prior], split=False)
            assert_array_less(rhat_imb, 1.02)

        with subtests.test('average max tree depth'):
            maxd_mcmc = avg_max_tree_depth(bart._main_trace.split_tree)
            maxd_prior = avg_max_tree_depth(prior_trace.split_tree)
            rhat_maxd = rhat_rank([maxd_mcmc, maxd_prior], split=False)
            assert_array_less(rhat_maxd, 1.02)

        with subtests.test('max tree depth distribution'):
            dd_mcmc = bart._depth_distr()
            dd_prior = forest_depth_distr(prior_trace.split_tree)
            rhat_dd = rhat_rank([dd_mcmc.squeeze(0), dd_prior], split=False)
            assert_array_less(rhat_dd, 1.02)

    with subtests.test('y_test'):
        X = random.randint(keys.pop(), (p, 30), 0, nsplits + 1)
        yhat_mcmc = predict_latent(X, bart._main_trace)
        yhat_prior = evaluate_trace(X, prior_trace)
        rhat_yhat = rhat_rank([yhat_mcmc, yhat_prior], split=False)
        assert_array_less(rhat_yhat, 1.01)


def run_bart_like_prior(
    key: Key[Array, ''], p: int, nsplits: int, subtests: SubTests
) -> Bart:
    """Run `Bart` without datapoints to sample the prior distribution."""
    xinfo = jnp.broadcast_to(jnp.arange(nsplits, dtype=jnp.float32), (p, nsplits))

    bart = Bart(
        x_train=jnp.empty((p, 0)),
        y_train=jnp.empty(0),
        num_trees=20,
        n_save=1000,
        n_burn=3000,
        printevery=None,
        binner=partial(GivenSplitsBinner, xinfo=xinfo),
        seed=key,
        num_chains=None,
        init_kw=dict(
            min_points_per_decision_node=None,
            min_points_per_leaf=None,
            save_ratios=True,
        ),
    )

    with subtests.test('likelihood ratio = 1'):
        assert_close_matrices(
            bart._burnin_trace.log_likelihood,
            jnp.zeros_like(bart._burnin_trace.log_likelihood),
            atol=1e-5,
            reduce_rank=True,
        )
        assert_close_matrices(
            bart._main_trace.log_likelihood,
            jnp.zeros_like(bart._main_trace.log_likelihood),
            atol=1e-5,
            reduce_rank=True,
        )

    return bart


def sample_prior_like(
    key: Key[Array, ''], bart: Bart, subtests: SubTests
) -> TraceWithOffset:
    """Sample from the prior with the same settings used in `bart`."""
    p_nonterminal = bart._mcmc_state.forest.p_nonterminal
    max_depth = tree_depth(p_nonterminal)
    indices = 2 ** jnp.arange(max_depth - 1)
    p_nonterminal = p_nonterminal[indices]

    prior_trees = sample_prior(
        key,
        bart.n_save,
        len(bart._mcmc_state.forest.leaf_tree),
        bart._mcmc_state.forest.max_split,
        p_nonterminal,
        jnp.sqrt(jnp.reciprocal(bart._mcmc_state.forest.leaf_prior_cov_inv)),
    )

    with subtests.test('check prior trees'):
        bad = check_trace(prior_trees, bart._mcmc_state.forest.max_split)
        bad_count = jnp.count_nonzero(bad)
        assert bad_count == 0

    return TraceWithOffset.from_trees_trace(prior_trees, bart.offset)


def count_stub_trees(
    split_tree: UInt[Array, '*batch_shape ntree 2**(d-1)'],
) -> Int32[Array, '*batch_shape']:
    """Count the number of trees with only a root node."""
    return (~split_tree.any(-1)).sum(-1)


def count_simple_trees(
    split_tree: UInt[Array, '*batch_shape ntree 2**(d-1)'],
) -> Int32[Array, '*batch_shape']:
    """Count the number of trees with 2 layers."""
    return (split_tree.astype(bool).sum(-1) == 1).sum(-1)


@partial(jnp.vectorize, signature='(n,m)->()')
def avg_imbalance_index(
    split_tree: UInt[Array, '*batch_shape ntree 2**(d-1)'],
) -> Float32[Array, '*batch_shape']:
    """Measure average tree imbalance in the forest."""
    is_leaf = vmap(partial(is_actual_leaf, add_bottom_level=True))(split_tree)
    depths = tree_depths(is_leaf.shape[-1])
    depths = jnp.broadcast_to(depths, is_leaf.shape)
    index = jnp.std(depths, where=is_leaf, axis=-1)
    return index.mean(-1)


@partial(jnp.vectorize, signature='(n,m)->()')
def avg_max_tree_depth(
    split_tree: UInt[Array, '*batch_shape ntree 2**(d-1)'],
) -> Float32[Array, '*batch_shape']:
    """Measure average maximum tree depth in the forest."""
    depth = vmap(tree_actual_depth)(split_tree)
    return depth.mean(-1)


def test_jit(bkw: BartKW) -> None:
    """Test that jitting around the whole interface works."""
    kw = bkw.kw
    kw.update(printevery=None, rm_const=False)

    platform = kw['y_train'].platform()
    kw.update(devices=jax.devices(platform))

    X = kw.pop('x_train')
    y = kw.pop('y_train')
    w = kw.pop('w', None)
    key = kw.pop('seed')

    def task(
        X: Shaped[Array, 'p n'],
        y: Shaped[Array, ' n'] | Shaped[Array, 'k n'],
        w: Float32[Array, ' n'] | None,
        key: Key[Array, ''],
    ) -> tuple[State, Shaped[Array, 'ndpost n'] | Shaped[Array, 'ndpost k n']]:
        bart = OriginalBart(X, y, w=w, **kw, seed=key)
        return bart._mcmc_state, bart.predict('train', kind='latent_samples')

    task_compiled = jax.jit(task)

    _state1, pred1 = task(X, y, w, key)
    _state2, pred2 = task_compiled(X, y, w, random.clone(key))

    assert_close_matrices(pred1, pred2, rtol=1e-5, reduce_rank=True)


def test_print_callback_terminates_dot_line(
    bkw: BartKW, capsys: CaptureFixture[str]
) -> None:
    """MCMC logging ends with a newline when the last iteration only prints a dot."""
    kw = bkw.kw
    n_iters = kw['n_burn'] + kw['n_save'] * _bart_default(kw, 'n_skip')
    printevery = _bart_default(kw, 'printevery')
    # ensure the last iteration falls outside a report boundary so it prints
    # only a dot, exercising the previously buggy path
    if printevery is None or n_iters % printevery == 0:
        kw['printevery'] = n_iters - 1
    block_until_ready(Bart(**kw))
    captured = capsys.readouterr()
    assert captured.out.endswith('.\n'), repr(captured.out[-200:])


@pytest.mark.flaky
# it's flaky because the interrupt may be caught and converted by jax internals (#33054)
@pytest.mark.timeout(32)
def test_interrupt(bkw: BartKW) -> None:
    """Test that the MCMC can be interrupted with ^C."""
    kw = bkw.kw
    kw.update(printevery=1, n_save=0, n_burn=10000)

    # Send the first ^C after 3 s, if the time was too short, it would interrupt
    # a first interruptible phase of jax compilation. Then send ^C every second,
    # in case the first ^C landed during a second non-interruptible compilation
    # phase that eats ^C and ignores it.
    with (
        pytest.raises(KeyboardInterrupt),
        periodic_sigint(first_after=3.0, interval=1.0),
    ):
        block_until_ready(Bart(**kw))


def test_polars(bkw: BartKW) -> None:
    """Test passing data as DataFrame and Series."""
    kw = bkw.kw

    bart = Bart(**kw)
    pred = bart.predict(bkw.x_test, kind='latent_samples')

    def to_polars(a: Array | None) -> pl.Series | pl.DataFrame | None:
        if a is None:
            return None
        arr = numpy.array(a)
        if arr.ndim == 1:
            return pl.Series(arr)
        return pl.DataFrame(arr.T)

    kw2 = dict(kw)
    kw2.update(
        seed=random.clone(kw2['seed']),
        x_train=to_polars(kw['x_train']),
        y_train=to_polars(kw['y_train']),
        w=to_polars(kw.get('w')),
        missing=to_polars(kw.get('missing')),
    )
    bart2 = Bart(**kw2)
    x_test_pl = pl.DataFrame(numpy.array(bkw.x_test).T)
    pred2 = bart2.predict(x_test_pl, kind='latent_samples')

    rtol = 0 if pred.platform() == 'cpu' else 2e-6

    assert_close_matrices(
        bart.predict('train', kind='latent_samples'),
        bart2.predict('train', kind='latent_samples'),
        rtol=rtol,
        reduce_rank=True,
    )
    if not bkw.any_binary:
        assert_close_matrices(bart.get_error_sdev(), bart2.get_error_sdev(), rtol=rtol)
    assert_close_matrices(pred, pred2, rtol=rtol, reduce_rank=True)


def test_data_format_mismatch(bkw: BartKW) -> None:
    """Test that passing predictors with mismatched formats raises an error."""
    kw = bkw.kw
    kw.update(
        x_train=pl.DataFrame(numpy.array(kw['x_train']).T),
        w=None if kw.get('w') is None else pl.Series(numpy.array(kw['w'])),
    )
    bart = Bart(**kw)
    with pytest.raises(ValueError, match='format mismatch'):
        bart.predict(numpy.array(bkw.x_test))


def test_automatic_integer_types(bkw: BartKW) -> None:
    """Test that integer variables in the MCMC state have the correct type."""
    kw = bkw.kw
    bart = Bart(**kw)

    def select_type(cond: bool) -> type:
        return jnp.uint8 if cond else jnp.uint16

    leaf_indices_type = select_type(bkw.maxdepth <= 8)
    split_trees_type = X_type = select_type(bkw.max_bins <= 256)
    var_trees_type = select_type(bkw.p <= 256)

    assert bart._mcmc_state.forest.var_tree.dtype == var_trees_type
    assert bart._mcmc_state.forest.split_tree.dtype == split_trees_type
    assert bart._mcmc_state.forest.leaf_indices.dtype == leaf_indices_type
    assert bart._mcmc_state.X.dtype == X_type
    assert bart._mcmc_state.forest.max_split.dtype == split_trees_type


def check_chains_data_sharding(
    x: Array, chains: bool, data: bool, mesh: Mesh | None
) -> None:
    """Check `x` is sharded as indicated by the boolean flags."""
    if mesh is None:
        assert isinstance(x.sharding, SingleDeviceSharding)
        return
    expected_spec = [None] * x.ndim
    if data and 'data' in mesh.axis_names:
        expected_spec[-1] = 'data'
    if chains and 'chains' in mesh.axis_names:
        expected_spec[0] = 'chains'
    expected_spec = normalize_spec(expected_spec, mesh, x.shape)
    assert get_normal_spec(x) == expected_spec


def get_expect_sharded(kw: dict) -> bool:
    """Check whether we expect sharding to be set up based on the arguments."""
    num_chain_devices = kw.get('num_chain_devices', 'auto')
    num_chains = kw.get('num_chains', 4)
    return (
        hasattr(num_chain_devices, '__index__')
        or kw.get('num_data_devices') is not None
        or (
            num_chain_devices == 'auto'
            and num_chains is not None
            and num_chains > 1
            and get_device_count() > 1
            and get_default_device().platform == 'cpu'
        )
    )


def test_sharding(bkw: BartKW, variant: int, keys: split) -> None:
    """Check that chains and data shards live on their own devices throughout the interface."""
    # WORKAROUND(jax<0.7): sharding bug, no time to fix
    if jax.__version_info__ < (0, 7, 0) and variant in (2, 5):
        pytest.xfail('Sharding bug in bartz with jax<0.7.')
    kw = bkw.kw
    bart = Bart(**kw)

    expect_sharded = get_expect_sharded(kw)
    mesh = bart._mcmc_state.config.mesh
    assert expect_sharded == (mesh is not None)

    # check internal attributes that have chains/data metadata, so their
    # expected sharding can be automatically inferred by `check_sharding`
    check = partial(check_sharding, mesh=mesh)
    check(bart._mcmc_state)
    check(bart._burnin_trace)
    check(bart._main_trace)

    check = partial(check_chains_data_sharding, mesh=mesh)

    for kind in PredictKind:
        extra: dict = {'key': keys.pop()} if kind is PredictKind.outcome_samples else {}
        yhat_train = bart.predict('train', kind=kind, **extra)
        if kind is PredictKind.mean:
            check(yhat_train, chains=False, data=True)
        else:
            check(yhat_train, chains=True, data=True)

            if kind is PredictKind.outcome_samples:
                extra['w'] = bkw.w_test
            yhat_test = bart.predict(bkw.x_test, kind=kind, **extra)
            check(yhat_test, chains=True, data=False)

    assert bart.offset.is_fully_replicated
    if bart.sigest is not None:
        assert bart.sigest.is_fully_replicated
    assert bart.varcount_mean.is_fully_replicated
    assert bart.varprob_mean.is_fully_replicated


class TestVarprobParam:
    """Test the `varprob` parameter."""

    def test_biased_predictor_choice(self, keys: split, bkw: BartKW) -> None:
        """Check that if `varprob[i]` is high then predictor `i` is used more than others."""
        kw = bkw.kw
        i = random.randint(keys.pop(), (), 0, bkw.p)
        vp = jnp.full(bkw.p, 0.001).at[i].set(1)
        vp /= vp.sum()
        kw.update(sparse=False, varprob=vp)
        bart = Bart(**kw)
        vc = bart.varcount_mean
        vc /= vc.sum()
        assert vc[i] > vp[i] * 0.6

    def test_positive(self, bkw: BartKW, subtests: SubTests) -> None:
        """Check that an error is raised if varprob is not > 0."""
        kw = bkw.kw

        with subtests.test('not negative'):
            assert bkw.p > 1
            varprob = jnp.ones(bkw.p).at[0].set(-1.0)
            kw_neg = dict(kw, varprob=varprob)
            with pytest.raises(EquinoxRuntimeError, match='varprob must be > 0'):
                Bart(**kw_neg)

        with subtests.test('not 0'):
            varprob = jnp.zeros(bkw.p).at[0].set(1.0)
            kw_zero = dict(kw, varprob=varprob)
            with pytest.raises(EquinoxRuntimeError, match='varprob must be > 0'):
                Bart(**kw_zero)


def run_bart_and_block(bkw: BartKW, keys: split) -> None:
    """Run bart and block until all outputs are ready."""
    bart = Bart(**bkw.kw)
    stuff = (
        bart,
        bart.predict('train', kind='latent_samples'),
        bart.predict('train', kind='mean'),
        bart.predict('train', kind='mean_samples'),
        bart.predict('train', kind='outcome_samples', key=keys.pop()),
        bart.predict(bkw.x_test, kind='latent_samples'),
        bart.predict(bkw.x_test, kind='mean'),
        bart.predict(bkw.x_test, kind='mean_samples'),
        bart.predict(bkw.x_test, kind='outcome_samples', key=keys.pop(), w=bkw.w_test),
        bart.get_error_sdev(),
        bart.get_latent_prec(),
        bart.varcount,
        bart.varprob,
    )
    block_until_ready(stuff)


@contextmanager
def array_garbage_collection_guard(value: str) -> Iterator[None]:
    """Implement `jax.array_garbage_collection_guard`, added in jax v0.9.1."""
    # WORKAROUND(jax<0.9.1): replace with `jax.array_garbage_collection_guard`
    setting = 'jax_array_garbage_collection_guard'
    prev = getattr(config, setting)
    config.update(setting, value)
    try:
        yield
    finally:
        config.update(setting, prev)


@contextmanager
def catch_array_gc_guard() -> Iterator[None]:
    """Catch array GC guard log messages and raise as exceptions."""
    # we need this because array_garbage_collection_guard('fatal')
    # terminates the process, and the 'log' messages are emitted from C++
    # code that swallows Python exceptions, so we can only check after the
    # context exits
    buf = StringIO()
    with redirect_stderr(buf), array_garbage_collection_guard('log'):
        yield
    captured = buf.getvalue()
    errmsg = '`jax.Array` was deleted by the Python garbage collector'
    if errmsg in captured:
        raise RuntimeError(captured)


def create_array_cycle() -> ReferenceType[Array]:
    """Create a reference cycle of two jax.Arrays."""
    n1 = jnp.ones((2, 2))
    n2 = jnp.zeros((2, 2))
    n1.next = n2
    n2.next = n1
    return ref(n1)


def test_catch_array_gc_guard() -> None:
    """Test `catch_array_gc_guard`."""
    collect()
    with (  # noqa: PT012, in case a gc collection happened before the explicit one
        pytest.raises(RuntimeError, match='deleted by the Python garbage collector'),
        catch_array_gc_guard(),
    ):
        weak = create_array_cycle()
        assert weak() is not None
        collect()
    assert weak() is None


def test_debug_checks(keys: split, bkw: BartKW) -> None:
    """Run with invasive jax debug options active."""
    with debug_nans(True), debug_infs(True), debug_key_reuse(True):
        run_bart_and_block(bkw, keys)


def test_no_array_gc(keys: split, bkw: BartKW) -> None:
    """Check that no reference cycles cause `jax.Array`s to be deleted by gc.

    Kept separate from `test_debug_checks` because `debug_nans` and `debug_infs`
    force jax through a slow Python dispatch path that itself creates cycles,
    yielding spurious failures unrelated to bartz code.
    """
    collect()
    with catch_array_gc_guard():
        run_bart_and_block(bkw, keys)
        collect()


def test_equiv_sharding(bkw: BartKW, subtests: SubTests) -> None:
    """Check that the result is the same with/without sharding."""
    if len(jax.devices()) < 2:  # this branch is covered in single cpu tests config
        pytest.skip('Need at least 2 devices for this test')
    if bkw.any_binary:
        # Binary regression uses `step_z`, which on data sharding folds the
        # shard index into the key to decorrelate per-datapoint draws — this
        # intentionally breaks bit-equivalence with the unsharded execution.
        pytest.skip('step_z breaks sharding equivalence on binary outcomes')

    baseline_kw = tree.map(lambda x: x, bkw.kw)
    baseline_kw.update(
        num_chain_devices=None, num_data_devices=None, n_burn=0, n_save=10, num_chains=2
    )
    bart = Bart(**baseline_kw)

    def check_equal(path: KeyPath, xb: Array, xs: Array) -> None:
        assert_close_matrices(
            xs, xb, err_msg=f'{keystr(path)}: ', rtol=1e-5, reduce_rank=True
        )

    def remove_mesh(bart: Bart) -> Bart:
        cfg = bart._mcmc_state.config
        cfg = replace(cfg, mesh=None)
        return tree_at(lambda b: b._mcmc_state.config, bart, cfg)

    with subtests.test('shard chains'):
        chains_kw = tree.map(lambda x: x, baseline_kw)
        chains_kw.update(num_chain_devices=2)
        bart_chains = Bart(**chains_kw)
        bart_chains = remove_mesh(bart_chains)
        tree.map_with_path(check_equal, bart, bart_chains)

    with subtests.test('shard data'):
        data_kw = tree.map(lambda x: x, baseline_kw)
        data_kw.update(num_data_devices=2)
        bart_data = Bart(**data_kw)
        bart_data = remove_mesh(bart_data)
        tree.map_with_path(check_equal, bart, bart_data)

    if len(jax.devices()) >= 4:  # pragma: no branch
        with subtests.test('shard data and chains'):
            both_kw = tree.map(lambda x: x, baseline_kw)
            both_kw.update(num_chain_devices=2, num_data_devices=2)
            bart_both = Bart(**both_kw)
            bart_both = remove_mesh(bart_both)
            tree.map_with_path(check_equal, bart, bart_both)


def test_num_trees(bkw: BartKW, subtests: SubTests) -> None:
    """Test the number of trees."""
    kw = bkw.kw
    kw.update(n_burn=0, n_save=0)

    with subtests.test('given num_trees'):
        bart = Bart(**kw)
        assert bart.num_trees == kw['num_trees']

    with subtests.test('default num_trees'):
        kw2 = {k: v for k, v in kw.items() if k != 'num_trees'}
        bart = Bart(**kw2)
        assert bart.num_trees == 200


class ExampleData(NamedTuple):
    """Small nonsense MV dataset for TestMVBartInterface."""

    x: Float32[Array, 'p n']
    y: Float32[Array, 'k n']
    w: Float32[Array, ' n']
    kwargs: dict[str, Any]


class TestMVBartInterface:
    """Some specific multivariate tests with their own data in and parametrization."""

    @pytest.fixture
    def example_data(self, keys: split) -> ExampleData:
        """Return a small nonsense MV dataset (x, y, w)."""
        return ExampleData(
            x=random.normal(keys.pop(), (2, 5)),
            y=random.normal(keys.pop(), (3, 5)),
            w=random.normal(keys.pop(), (5,)),
            kwargs=dict(num_trees=5, n_save=0, n_burn=0, num_chains=None),
        )

    @pytest.mark.parametrize('outcome_mode', ['continuous', 'mixed'])
    def test_scalar_params(
        self, example_data: ExampleData, subtests: SubTests, outcome_mode: str
    ) -> None:
        """Test that scalar configuration params are broadcasted."""
        x, y, _, kw = example_data
        k, _ = y.shape
        if outcome_mode == 'mixed':
            outcome_type = ['binary'] + ['continuous'] * (k - 1)
        else:
            outcome_type = 'continuous'
        kw.update(x_train=x, y_train=y, outcome_type=outcome_type)

        with subtests.test('offset'):
            bart = Bart(offset=0.0, **kw)
            assert bart.offset.shape == (k,)

        with subtests.test('sigest'):
            bart = Bart(sigest=1.0, **kw)
            assert bart.sigest.shape == (k,)

        with subtests.test('lambda_'):
            bart = Bart(lambda_=1.0, **kw)
            assert bart.sigest is None
            assert bart._mcmc_state.error_cov_scale.shape == (k, k)

    def test_mixed_rejects_weights(self, example_data: ExampleData) -> None:
        """Mixed outcome_type + weights should raise."""
        x, y, w, kw = example_data
        k, _ = y.shape
        with pytest.raises(ValueError, match='binary'):
            Bart(
                x_train=x,
                y_train=y,
                w=w,
                **kw,
                outcome_type=['binary'] + ['continuous'] * (k - 1),
            )

    def test_outcome_type_length_mismatch(self, example_data: ExampleData) -> None:
        """Sequence outcome_type with wrong length should raise."""
        x, y, _, kw = example_data
        with pytest.raises(ValueError, match='length'):
            Bart(x_train=x, y_train=y, **kw, outcome_type=['continuous', 'continuous'])

    def test_sequence_outcome_type_requires_2d(self, example_data: ExampleData) -> None:
        """Sequence outcome_type with 1D y should raise."""
        x, y, _, kw = example_data
        with pytest.raises(ValueError, match=r'y_train\.shape=\(1, n\)'):
            Bart(x_train=x, y_train=y[0, :], **kw, outcome_type=['continuous'])


def test_uv_mv_k1_equivalence(bkw: BartKW) -> None:
    """Test that Bart initializes equivalent states for UV and MV(k=1)."""
    outcome_type = bkw.kw['outcome_type']
    if isinstance(outcome_type, Sequence) and not isinstance(outcome_type, str):
        outcome_type = outcome_type[0]

    y_mv = bkw.kw['y_train']
    if bkw.k is None:  # pragma: no cover, bc defaults are mv only
        y_mv = y_mv[None, :]
    y_mv = y_mv[:1, :]
    y_uv = y_mv.squeeze(0)

    bkw.kw.update(outcome_type=outcome_type, n_burn=0, n_save=0, w=None, missing=None)
    del bkw.kw['y_train']
    bart_uv = Bart(y_train=y_uv, **bkw.kw)
    bart_mv = Bart(y_train=y_mv, **bkw.kw)

    state_uv = bart_uv._mcmc_state
    state_mv = bart_mv._mcmc_state

    # Move chain axis to position 0 so squeezes target the k axis at -2 / -1.
    uv_axes = chain_vmap_axes(state_uv)
    mv_axes = chain_vmap_axes(state_mv)

    # Residuals and error covariance
    assert_close_matrices(
        chain_to_axis(state_uv.resid, uv_axes.resid),
        chain_to_axis(state_mv.resid, mv_axes.resid).squeeze(-2),
    )
    assert_close_matrices(
        chain_to_axis(state_uv.error_cov_inv, uv_axes.error_cov_inv),
        chain_to_axis(state_mv.error_cov_inv, mv_axes.error_cov_inv).squeeze((-2, -1)),
        rtol=1e-6,
    )

    # Prior parameters
    assert_array_equal(bart_uv.offset, bart_mv.offset.squeeze(0))
    assert_array_equal(
        state_uv.forest.leaf_prior_cov_inv,
        state_mv.forest.leaf_prior_cov_inv.reshape(()),
    )
    if outcome_type == 'continuous':
        assert_array_equal(state_uv.error_cov_df, state_mv.error_cov_df)
        assert_array_equal(
            state_uv.error_cov_scale, state_mv.error_cov_scale.reshape(())
        )
        assert_array_equal(bart_uv.sigest, bart_mv.sigest.squeeze(0))

    # Forest structure
    uv_forest_axes = chain_vmap_axes(state_uv.forest)
    mv_forest_axes = chain_vmap_axes(state_mv.forest)
    assert_array_equal(
        chain_to_axis(state_uv.forest.var_tree, uv_forest_axes.var_tree),
        chain_to_axis(state_mv.forest.var_tree, mv_forest_axes.var_tree),
    )
    assert_array_equal(
        chain_to_axis(state_uv.forest.split_tree, uv_forest_axes.split_tree),
        chain_to_axis(state_mv.forest.split_tree, mv_forest_axes.split_tree),
    )
    assert_array_equal(
        chain_to_axis(state_uv.forest.leaf_tree, uv_forest_axes.leaf_tree),
        chain_to_axis(state_mv.forest.leaf_tree, mv_forest_axes.leaf_tree).squeeze(-2),
    )
    assert_array_equal(
        chain_to_axis(state_uv.forest.leaf_indices, uv_forest_axes.leaf_indices),
        chain_to_axis(state_mv.forest.leaf_indices, mv_forest_axes.leaf_indices),
    )


def test_get_latent_prec_only_continuous(bkw: BartKW) -> None:
    """get_latent_prec(only_continuous=True) removes binary components."""
    kw = bkw.kw
    if bkw.k is None:  # pragma: no cover, bc defaults are mv only
        pytest.skip('UV variant')

    bart = Bart(**kw)
    if bkw.all_binary:
        with pytest.raises(ValueError, match='only binary'):
            bart.get_latent_prec(only_continuous=True)
        return

    prec = bart.get_latent_prec(only_continuous=True)
    kc = int(jnp.sum(~bkw.binary_mask))

    n_burn = kw['n_burn']
    if bkw.num_chains is not None:
        assert prec.shape == (bkw.num_chains, n_burn + bkw.n_save, kc, kc)
    else:
        assert prec.shape == (n_burn + bkw.n_save, kc, kc)


def test_get_error_sdev_values(bkw: BartKW) -> None:
    """get_error_sdev matches manual computation from precision matrices."""
    kw = bkw.kw
    if bkw.all_binary:
        pytest.skip('binary variant')
    bart = Bart(**kw)
    n_burn = kw['n_burn']

    with debug_nans(False):
        sdev = bart.get_error_sdev()
        if sdev.ndim == 1:  # pragma: no cover, bc defaults are mv only
            sdev = sdev[:, None]  # reshape as vector of length 1

    # manual: invert each precision matrix, take sqrt of diagonal
    prec = bart.get_latent_prec()
    if prec.ndim < 3:  # pragma: no cover, bc defaults are mv only
        prec = prec[..., :, None, None]  # reshape as 1x1 matrix
    prec = prec[..., n_burn:, :, :]  # skip burnin
    prec = lax.collapse(prec, 0, -2)  # flatten chains

    cov = jnp.linalg.inv(prec)
    sdev_ref = jnp.sqrt(jnp.diagonal(cov, axis1=-2, axis2=-1))

    # for mixed, compare only continuous components (binary have NaN sdev)
    mask = jnp.atleast_1d(~bkw.binary_mask)
    sdev = sdev[:, mask]
    sdev_ref = sdev_ref[:, mask]

    assert_close_matrices(sdev, sdev_ref, rtol=1e-5)


def test_devices_platform(bkw: BartKW) -> None:
    """Check that passing `devices='cpu'/'gpu'` ends up on the expected device."""
    bart1 = Bart(**bkw.kw)
    platform = bart1._main_trace.grow_prop_count.platform()
    kw2 = dict(bkw.kw, devices=platform)
    bart2 = Bart(**kw2)
    assert_identical_bart(bart1, bart2)


def assert_identical_bart(bart1: Bart, bart2: Bart) -> None:
    """Check that two `Bart` objects are equal."""

    def check_same(path: KeyPath, x1: Array, x2: Array) -> None:
        assert x1.shape == x2.shape
        assert x1.dtype == x2.dtype
        assert x1.sharding.is_equivalent_to(x2.sharding, x1.ndim)
        if jnp.issubdtype(x1.dtype, jnp.floating):
            assert_close_matrices(
                x1, x2, rtol=1e-5, err_msg=keystr(path), reduce_rank=True
            )
        else:
            assert_array_equal(x1, x2, strict=True, err_msg=keystr(path))

    tree.map_with_path(check_same, bart1, bart2)

    treedef1 = tree.structure(bart1)
    treedef2 = tree.structure(bart2)
    assert treedef1 == treedef2


def test_numpy_input(bkw: BartKW) -> None:
    """Check if all numerical inputs are numpy arrays, everything works as usual."""
    bart1 = Bart(**bkw.kw)

    def to_numpy_array(x: Array | object) -> numpy.ndarray | object:
        if isinstance(x, (Array, float)) and not is_key(x):
            return numpy.asarray(x)
        else:
            return x

    kw2 = tree.map(to_numpy_array, bkw.kw)
    bart2 = Bart(**kw2)

    assert_identical_bart(bart1, bart2)


def test_sigest_wrong_special_value(bkw: BartKW) -> None:
    """Trigger error on unrecognized `sigest` value."""
    value = 'ohohoh'
    if bkw.all_binary:
        pytest.skip('Parameter ignored with binary outcomes.')
    kw = dict(bkw.kw, sigest=value)
    with pytest.raises(ValueError, match=value):
        Bart(**kw)


def test_sigest_cg(bkw: BartKW) -> None:
    """Check the `sigest='cg'` is an approximation of `sigest='ols-or-variance'`."""
    if bkw.p >= bkw.n or bkw.all_binary:
        pytest.skip('Requires p < n and continuous outcomes.')
    bart_ols = Bart(**dict(bkw.kw, sigest='ols-or-variance'))
    bart_cg = Bart(**dict(bkw.kw, sigest='cg'))
    mask = ~bkw.binary_mask
    assert_close_matrices(bart_cg.sigest[mask], bart_ols.sigest[mask], rtol=1e-6)


def test_sigest_auto_cg(keys: split) -> None:
    """Check the `sigest='auto'` branch that switches to 'cg'."""
    n = 110
    p = 1000
    assert n * p * p > 10_000 * 100 * 100

    x = random.normal(keys.pop(), (p, n))
    y = random.normal(keys.pop(), (n,))
    # y is random so we can check 'cg' is regularizing in this high-p problem:
    # the correct answer is sigest = std(y), but since p > n ordinary linear
    # regression would always overfit perfectly and return sigest = 0.

    bart = Bart(x, y, seed=keys.pop(), n_save=0, n_burn=0)
    stdy = jnp.std(y)
    assert bart.sigest <= stdy
    assert bart.sigest >= stdy * 1e-3  # not that much regularization...
