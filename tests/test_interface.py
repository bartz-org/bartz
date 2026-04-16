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
from io import StringIO
from typing import Any, Literal, NamedTuple

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
from jax.sharding import SingleDeviceSharding
from jax.tree_util import KeyPath, keystr
from jaxtyping import Array, Float32, Int32, Key, PyTree, Real, Shaped, UInt
from numpy.testing import assert_allclose, assert_array_equal
from pytest import FixtureRequest  # noqa: PT013
from pytest_subtests import SubTests

from bartz import Bart as OriginalBart
from bartz.debug import TraceWithOffset, sample_prior
from bartz.grove import (
    check_trace,
    forest_depth_distr,
    is_actual_leaf,
    tree_actual_depth,
    tree_depth,
    tree_depths,
)
from bartz.jaxext import get_device_count, split
from bartz.mcmcloop import compute_varcount, evaluate_trace
from bartz.mcmcstep import State
from bartz.mcmcstep._state import chain_vmap_axes
from tests.conftest import get_disable_problematic_sharding
from tests.test_mcmcstep import check_sharding, get_normal_spec, normalize_spec
from tests.util import (
    assert_close_matrices,
    assert_different_matrices,
    multivariate_rhat,
    periodic_sigint,
    rhat,
)


class Bart(OriginalBart):
    """Wrapper that enables debug checks by default."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.check_trees(error=True)
        self.check_replicated_trees()


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


def gen_w(key: Key[Array, ''], n: int) -> Float32[Array, ' n']:
    """Generate a vector of error weights."""
    return jnp.exp(random.uniform(key, (n,), float, -1, 1))


def gen_y(
    key: Key[Array, ''],
    X: Real[Array, 'p n'],
    w: Float32[Array, ' n'] | None,
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


class BartKW(NamedTuple):
    """Keyword arguments for `Bart` plus associated test data."""

    kw: dict[str, Any]
    x_test: Real[Array, 'p m']
    w_test: Float32[Array, ' m'] | None = None


def make_kw(key: Key[Array, ''], variant: int) -> BartKW:
    """Return keyword arguments for `Bart` and test predictors."""
    keys = split(key, 10)  # 10 is just some high number
    n = 20
    nt = 21
    p = 2
    high_p = 257  # > 256 to use uint16 for var_trees.

    match variant:
        # continuous regression with some settings that induce large types,
        # sparsity with free theta
        case 1:
            X = gen_X(keys.pop(), p, n, 'continuous')
            Xt = gen_X(keys.pop(), p, nt, 'continuous')
            y = gen_y(keys.pop(), X, None, 'continuous', s='random')
            bkw = BartKW(
                kw=dict(
                    x_train=X,
                    y_train=y,
                    outcome_type='continuous',
                    sparse=True,
                    num_trees=20,
                    ndpost=100,
                    nskip=50,
                    printevery=50,
                    usequants=False,
                    numcut=256,  # > 255 to use uint16 for X and split_trees
                    num_chains=None,
                    seed=keys.pop(),
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
                x_test=Xt,
            )

        # binary regression with binary X and high p
        case 2:
            X = gen_X(keys.pop(), high_p, n, 'binary')
            Xt = gen_X(keys.pop(), high_p, nt, 'binary')
            y = gen_y(keys.pop(), X, None, 'binary')
            bkw = BartKW(
                kw=dict(
                    x_train=X,
                    y_train=y,
                    outcome_type='binary',
                    num_trees=20,
                    ndpost=100,
                    nskip=50,
                    keepevery=1,  # the default with binary would be 10
                    printevery=None,
                    usequants=True,
                    # usequants=True with binary X to check the case in which the
                    # splits are less than the statically known maximum
                    numcut=255,
                    seed=keys.pop(),
                    num_chains=2,
                    maxdepth=6,
                    num_data_devices=min(2, get_device_count()),
                    num_chain_devices=None,
                    init_kw=dict(
                        save_ratios=False,
                        min_points_per_decision_node=None,
                        min_points_per_leaf=None,
                    ),
                ),
                x_test=Xt,
            )

        # continuous regression with error weights and sparsity with fixed theta
        case 3:
            X = gen_X(keys.pop(), p, n, 'continuous')
            Xt = gen_X(keys.pop(), p, nt, 'continuous')
            w = gen_w(keys.pop(), X.shape[1])
            wt = gen_w(keys.pop(), Xt.shape[1])
            y = gen_y(keys.pop(), X, w, 'continuous', s='random')
            bkw = BartKW(
                kw=dict(
                    x_train=X,
                    y_train=y,
                    outcome_type='continuous',
                    w=w,
                    sparse=True,
                    theta=2,
                    varprob=jnp.array([0.2, 0.8]),
                    num_trees=20,
                    ndpost=100,
                    nskip=50,
                    printevery=50,
                    usequants=True,
                    numcut=10,
                    seed=keys.pop(),
                    num_chains=2,
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
                x_test=Xt,
                w_test=wt,
            )

        # multivariate continuous regression with some settings that induce
        # large types, sparsity with free theta
        case 4:
            X = gen_X(keys.pop(), p, n, 'continuous')
            Xt = gen_X(keys.pop(), p, nt, 'continuous')
            y = gen_y(keys.pop(), X, None, 'continuous', k=1, s='random')
            bkw = BartKW(
                kw=dict(
                    x_train=X,
                    y_train=y,
                    outcome_type='continuous',
                    sparse=True,
                    num_trees=20,
                    ndpost=100,
                    nskip=50,
                    printevery=50,
                    usequants=False,
                    numcut=256,  # > 255 to use uint16 for X and split_trees
                    num_chains=None,
                    seed=keys.pop(),
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
                x_test=Xt,
            )

        # multivariate binary regression with binary X and high p
        case 5:
            X = gen_X(keys.pop(), high_p, n, 'binary')
            Xt = gen_X(keys.pop(), high_p, nt, 'binary')
            y = gen_y(keys.pop(), X, None, 'binary', k=2)
            bkw = BartKW(
                kw=dict(
                    x_train=X,
                    y_train=y,
                    outcome_type='binary',
                    num_trees=20,
                    ndpost=100,
                    nskip=50,
                    printevery=None,
                    usequants=True,
                    # usequants=True with binary X to check the case in which the
                    # splits are less than the statically known maximum
                    numcut=255,
                    seed=keys.pop(),
                    num_chains=2,
                    maxdepth=6,
                    num_data_devices=min(2, get_device_count()),
                    num_chain_devices=None,
                    init_kw=dict(
                        save_ratios=False,
                        min_points_per_decision_node=None,
                        min_points_per_leaf=None,
                    ),
                ),
                x_test=Xt,
            )

        # multivariate mixed binary-continuous regression with sparsity with
        # fixed theta
        case 6:  # pragma: no branch
            X = gen_X(keys.pop(), p, n, 'continuous')
            Xt = gen_X(keys.pop(), p, nt, 'continuous')
            outcome_type = ['continuous', 'binary', 'binary']
            y = gen_y(
                keys.pop(), X, None, outcome_type, s='random', k=len(outcome_type)
            )
            bkw = BartKW(
                kw=dict(
                    x_train=X,
                    y_train=y,
                    outcome_type=outcome_type,
                    sparse=True,
                    theta=2,
                    varprob=jnp.array([0.2, 0.8]),
                    num_trees=20,
                    ndpost=100,
                    nskip=50,
                    printevery=50,
                    usequants=True,
                    numcut=10,
                    seed=keys.pop(),
                    num_chains=2,
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
                x_test=Xt,
            )

        case _:  # pragma: no cover
            msg = f'Unknown variant {variant}'
            raise ValueError(msg)

    if get_disable_problematic_sharding():
        bkw.kw['num_data_devices'] = None
        bkw.kw['num_chain_devices'] = None

    return bkw


# test only the multivariate variants, because the other ones are tested in
# test_BART.py
@pytest.fixture(params=(4, 5, 6), scope='module')
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
        kw['w'] = kw['w'][:n]
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

        p, n = kw['x_train'].shape
        nchains = 4
        kw.update(
            num_trees=max(2 * n, p),
            nskip=3000,
            ndpost=nchains * 1000,
            keepevery=1,
            num_chains=nchains,
        )
        init_kw = dict(kw.get('init_kw', {}))
        init_kw.update(min_points_per_decision_node=10, min_points_per_leaf=5)
        kw['init_kw'] = init_kw

        return CachedBart(bkw=bkw, bart=Bart(**kw))

    def test_residuals_accuracy(self, cachedbart: CachedBart) -> None:
        """Check that running residuals are close to the recomputed final residuals."""
        accum_resid, actual_resid = cachedbart.bart.compare_resid(
            y=cachedbart.bkw.kw['y_train']
        )
        assert_close_matrices(accum_resid, actual_resid, rtol=1e-4, reduce_rank=True)

    def test_convergence(self, cachedbart: CachedBart) -> None:
        """Run multiple chains and check convergence with rhat."""
        bart = cachedbart.bart
        bkw = cachedbart.bkw
        p, n = bkw.kw['x_train'].shape
        num_chains = 4
        nsamples = bart.ndpost // num_chains
        binary = bkw.kw['outcome_type'] == 'binary'

        yhat_train = bart.predict('train', kind='latent_samples')
        yhat_train_chains = yhat_train.reshape(num_chains, nsamples, -1)
        rhat_yhat_train = multivariate_rhat(yhat_train_chains)
        assert rhat_yhat_train < 6
        print(f'{rhat_yhat_train.item()=}')

        mixed = isinstance(bkw.kw['outcome_type'], list)
        if binary:
            prob_train = bart.predict('train', kind='mean_samples')
            prob_train_chains = prob_train.reshape(num_chains, nsamples, -1)
            rhat_prob_train = multivariate_rhat(prob_train_chains)
            assert rhat_prob_train < 1.2
            print(f'{rhat_prob_train.item()=}')
        elif mixed:
            # mixed regression: check get_error_sdev, dropping binary
            # components (NaN sdev)
            with debug_nans(False):
                sigma = bart.get_error_sdev().reshape(num_chains, nsamples, -1)
                binary_mask = bart._binary_mask
                if binary_mask.ndim > 0:
                    sigma = sigma[:, :, ~binary_mask]
            rhat_sigma = multivariate_rhat(sigma)
            assert rhat_sigma < 1.2
            print(f'{rhat_sigma.item()=}')
        else:
            # all continuous: check full precision matrix convergence
            # using upper triangular elements (matrix is symmetric)
            error_cov_inv = bart._main_trace.error_cov_inv
            if error_cov_inv.ndim == 2:
                error_cov_inv = error_cov_inv[:, :, None, None]
            _, _, k, _ = error_cov_inv.shape
            ti, tj = jnp.triu_indices(k)
            error_cov_inv = error_cov_inv[:, :, ti, tj]
            rhat_prec = multivariate_rhat(error_cov_inv)
            assert rhat_prec < 1.2
            print(f'{rhat_prec.item()=}')

        if p < n:
            varcount_vals = bart.varcount.reshape(num_chains, nsamples, p)
            rhat_varcount = multivariate_rhat(varcount_vals)
            assert rhat_varcount < 7
            print(f'{rhat_varcount.item()=}')

            if bkw.kw.get('sparse', False):  # pragma: no branch
                varprob_vals = bart.varprob.reshape(num_chains, nsamples, p)
                rhat_varprob = multivariate_rhat(varprob_vals[:, :, 1:])
                assert rhat_varprob < 7
                print(f'{rhat_varprob.item()=}')

    def test_different_chains(self, cachedbart: CachedBart) -> None:
        """Check that different chains give different results."""
        bart = cachedbart.bart

        step_theta = bart._mcmc_state.forest.rho is not None

        def assert_different(x: PyTree[Array], **kwargs: Any) -> None:
            def assert_different(
                path: KeyPath, x: Array | None, chain_axis: int | None
            ) -> None:
                str_path = keystr(path)
                if str_path.endswith('.theta') and not step_theta:
                    return
                if (
                    str_path.endswith('.error_cov_inv')
                    and bart._mcmc_state.error_cov_df is None
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
    kw['keepevery'] = 1
    bart1 = Bart(**kw)

    num_chains = kw.get('num_chains')
    mc_cores = 1 if num_chains is None else num_chains
    y_shape = kw['y_train'].shape

    # run moving some samples from burn-in to main
    kw2 = dict(kw)
    kw2['seed'] = random.clone(kw2['seed'])
    if kw2.get('sparse', False):
        init_kw = dict(kw2.get('init_kw', {}))
        init_kw.setdefault('sparse_on_at', kw2['nskip'] // 2)
        kw2['init_kw'] = init_kw
    delta = 1
    kw2['nskip'] -= delta
    kw2['ndpost'] += delta * mc_cores
    bart2 = Bart(**kw2)
    bart2_yhat_train = (
        bart2.predict('train', kind='latent_samples')
        .reshape(mc_cores, kw2['ndpost'] // mc_cores, *y_shape)[:, delta:]
        .reshape(bart1.ndpost, *y_shape)
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
    kw3['keepevery'] = 2
    bart3 = Bart(**kw3)
    bart1_yhat_train = bart1.predict('train', kind='latent_samples').reshape(
        mc_cores, kw3['ndpost'] // mc_cores, *y_shape
    )[:, 1::2, :, ...]
    bart3_yhat_train = bart3.predict('train', kind='latent_samples').reshape(
        mc_cores, kw3['ndpost'] // mc_cores, *y_shape
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


def test_output_shapes(bkw: BartKW, keys: split) -> None:
    """Check the output shapes of the Bart predictions and attributes."""
    kw = bkw.kw
    bart = Bart(**kw)

    ndpost = bart.ndpost
    p, n = kw['x_train'].shape
    _, m = bkw.x_test.shape
    k = kw['y_train'].shape[:-1]  # () or (k,)

    assert bart.offset.shape == k

    assert bart.predict('train', kind='mean_samples').shape == (ndpost, *k, n)
    assert bart.predict('train', kind='mean').shape == (*k, n)
    assert bart.predict('train', kind='latent_samples').shape == (ndpost, *k, n)
    assert bart.predict('train', kind='outcome_samples', key=keys.pop()).shape == (
        ndpost,
        *k,
        n,
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

    if kw['outcome_type'] == 'binary':
        assert bart.sigest is None
    else:
        assert bart.sigest.shape == k

    assert bart.varcount.shape == (ndpost, p)
    assert bart.varcount_mean.shape == (p,)
    assert bart.varprob.shape == (ndpost, p)
    assert bart.varprob_mean.shape == (p,)

    # get_latent_prec shape
    num_chains = kw.get('num_chains')
    nskip = kw['nskip']
    prec = bart.get_latent_prec()
    if num_chains is not None:
        assert prec.shape == (num_chains, nskip + ndpost // num_chains, *k, *k)
    else:
        assert prec.shape == (nskip + ndpost, *k, *k)


def test_output_types(bkw: BartKW, keys: split) -> None:
    """Check the output types of all the attributes of Bart."""
    kw = bkw.kw
    bart = Bart(**kw)

    if kw['outcome_type'] != 'binary':
        assert bart.sigest.dtype == jnp.float32
    assert bart.offset.dtype == jnp.float32
    assert isinstance(bart.ndpost, int)
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

    binary_mask = bart._binary_mask

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
    if binary_mask.all():
        assert bart.sigest is None
    else:
        assert jnp.all(bart.sigest[binary_mask] == 0.0)
        assert jnp.all(bart.sigest[~binary_mask] > 0)

    # get_latent_prec: symmetry and positive definiteness
    if kw['y_train'].ndim == 2:
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
    # outcome_samples has shape (ndpost, *k, n)

    with subtests.test('mean_samples vs mean'):
        mean_from_mean_samples = mean_samples.mean(0)
        assert_close_matrices(mean, mean_from_mean_samples)

    with subtests.test('latent_samples vs mean_samples'):
        binary_mask = bart._binary_mask
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
        assert_allclose(bart.varprob.sum(axis=1), 1, rtol=1e-6)

        sparse = kw.get('sparse', False)
        if not sparse:
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
        bart = Bart(x_train=X, y_train=y, xinfo=xinfo, seed=keys.pop())
        assert_array_equal(bart._mcmc_state.forest.max_split, [0, 1])
        assert_array_equal(bart.varprob_mean, [0, 1])
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
        nskip=1000,
        sparse=True,
        theta=peff if theta == 'fixed' else None,
        seed=keys.pop(),
    )

    assert bart.varprob_mean[mask].sum() >= 0.9
    assert bart.varprob_mean[mask].min().item() > 0.5 / peff
    assert bart.varprob_mean[~mask].max().item() < 1 / (p - peff)


def test_scale_shift(bkw: BartKW) -> None:
    """Check self-consistency of rescaling the inputs."""
    kw = bkw.kw
    outcome_type = kw['outcome_type']
    if outcome_type == 'binary' or (
        isinstance(outcome_type, Sequence)
        and not isinstance(outcome_type, str)
        and any(t == 'binary' for t in outcome_type)
    ):
        pytest.skip('Cannot rescale binary responses.')

    bart1 = Bart(**kw)

    offset = 0.4703189
    scale = 0.5294714
    kw2 = dict(kw)
    kw2.update(y_train=offset + kw['y_train'] * scale, seed=random.clone(kw['seed']))
    bart2 = Bart(**kw2)

    assert_allclose(bart1.offset, (bart2.offset - offset) / scale, rtol=1e-6, atol=1e-6)
    assert_allclose(
        bart1._mcmc_state.forest.leaf_prior_cov_inv,
        bart2._mcmc_state.forest.leaf_prior_cov_inv * scale**2,
        rtol=1e-6,
        atol=0,
    )
    assert_allclose(bart1.sigest, bart2.sigest / scale, rtol=1e-6)
    assert_array_equal(bart1._mcmc_state.error_cov_df, bart2._mcmc_state.error_cov_df)
    assert_allclose(
        bart1._mcmc_state.error_cov_scale,
        bart2._mcmc_state.error_cov_scale / scale**2,
        rtol=1e-6,
    )

    yhat1 = bart1.predict('train', kind='latent_samples')
    yhat2 = bart2.predict('train', kind='latent_samples')
    assert_close_matrices(yhat1, (yhat2 - offset) / scale, rtol=1e-5, reduce_rank=True)

    mean1 = bart1.predict('train', kind='mean')
    mean2 = bart2.predict('train', kind='mean')
    assert_close_matrices(mean1, (mean2 - offset) / scale, rtol=1e-5, reduce_rank=True)

    yhat_test1 = bart1.predict(kw['x_train'], kind='latent_samples')
    yhat_test2 = bart2.predict(kw['x_train'], kind='latent_samples')
    assert_close_matrices(
        yhat_test1, (yhat_test2 - offset) / scale, rtol=1e-5, reduce_rank=True
    )

    yhat_test_mean1 = bart1.predict(kw['x_train'], kind='mean')
    yhat_test_mean2 = bart2.predict(kw['x_train'], kind='mean')
    assert_close_matrices(
        yhat_test_mean1, (yhat_test_mean2 - offset) / scale, rtol=1e-5, reduce_rank=True
    )

    sdev1 = bart1.get_error_sdev()
    sdev2 = bart2.get_error_sdev()
    assert_close_matrices(sdev1, sdev2 / scale, rtol=1e-5, reduce_rank=True)
    assert_allclose(
        bart1.get_error_sdev(mean=True),
        bart2.get_error_sdev(mean=True) / scale,
        rtol=1e-6,
        atol=1e-6,
    )


def test_min_points_per_decision_node(bkw: BartKW) -> None:
    """Check that the limit of at least 10 datapoints per decision node is respected."""
    kw = bkw.kw
    init_kw = dict(kw.get('init_kw', {}))
    init_kw['min_points_per_leaf'] = None
    kw['init_kw'] = init_kw
    bart = Bart(**kw)
    distr = bart.points_per_decision_node_distr()
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
    distr = bart.points_per_leaf_distr()
    distr_marg = distr.sum(axis=(0, 1))

    min_points = init_kw.get('min_points_per_leaf', 5)

    if min_points is None:
        assert distr_marg[4] > 0
    else:
        assert jnp.all(distr_marg[:min_points] == 0)
        assert distr_marg[min_points] > 0


def test_no_datapoints(bkw: BartKW) -> None:
    """Check automatic data scaling with 0 datapoints."""
    kw = set_num_datapoints(bkw.kw, 0)

    p, _ = kw['x_train'].shape
    nsplits = 10
    xinfo = jnp.broadcast_to(jnp.arange(nsplits, dtype=jnp.float32), (p, nsplits))
    kw.update(xinfo=xinfo)

    kw.update(num_data_devices=None)

    init_kw = dict(kw.get('init_kw', {}))
    init_kw.update(
        save_ratios=True, min_points_per_decision_node=None, min_points_per_leaf=None
    )
    kw['init_kw'] = init_kw

    bart = Bart(**kw)

    ndpost = bart.ndpost
    k = kw['y_train'].shape[:-1]  # () or (k,)
    assert bart.predict('train', kind='latent_samples').shape == (ndpost, *k, 0)

    assert_array_equal(bart.offset, 0)  # compare against jnp.zeros with strict=True
    outcome_type = kw['outcome_type']
    if outcome_type == 'binary':
        tau_num = 3
        assert bart.sigest is None
    elif isinstance(outcome_type, Sequence) and not isinstance(outcome_type, str):
        binary_mask = jnp.array([t == 'binary' for t in outcome_type])
        tau_num = jnp.where(binary_mask, 3.0, 1.0)
        expected_sigest = jnp.where(binary_mask, 0.0, 1.0)
        assert_array_equal(bart.sigest, expected_sigest)
    else:
        tau_num = 1
        assert_array_equal(bart.sigest, 1)  # compare against jnp.ones with strict=True
    expected_cov_inv = jnp.float32((2**2 * kw['num_trees']) / tau_num**2)
    leaf_prior_cov_inv = bart._mcmc_state.forest.leaf_prior_cov_inv
    if leaf_prior_cov_inv.ndim == 2:
        expected_cov_inv = jnp.eye(leaf_prior_cov_inv.shape[0]) * expected_cov_inv
    assert_close_matrices(leaf_prior_cov_inv, expected_cov_inv, rtol=1e-6)

    assert_array_equal(bart._burnin_trace.log_likelihood, 0.0)
    assert_array_equal(bart._main_trace.log_likelihood, 0.0)


def test_one_datapoint(bkw: BartKW) -> None:
    """Check automatic data scaling with 1 datapoint."""
    kw = set_num_datapoints(bkw.kw, 1)

    if kw.get('usequants', False):
        p, _ = kw['x_train'].shape
        nsplits = 10
        xinfo = jnp.broadcast_to(jnp.arange(nsplits, dtype=jnp.float32), (p, nsplits))
        kw.update(xinfo=xinfo)

    kw.update(num_data_devices=None)

    init_kw = dict(kw.get('init_kw', {}))
    init_kw.update(
        save_ratios=True, min_points_per_decision_node=None, min_points_per_leaf=None
    )
    kw['init_kw'] = init_kw

    bart = Bart(**kw)
    outcome_type = kw['outcome_type']
    k = kw['y_train'].shape[:-1]  # () or (k,)
    if outcome_type == 'binary':
        tau_num = 3
        assert bart.sigest is None
        assert_array_equal(bart.offset, jnp.zeros(k), strict=True)
    elif isinstance(outcome_type, Sequence) and not isinstance(outcome_type, str):
        binary_mask = jnp.array([t == 'binary' for t in outcome_type])
        tau_num = jnp.where(binary_mask, 3.0, 1.0)
        expected_sigest = jnp.where(binary_mask, 0.0, 1.0)
        assert_array_equal(bart.sigest, expected_sigest)
    else:
        tau_num = 1
        assert_array_equal(bart.sigest, jnp.ones(k), strict=True)
        assert_close_matrices(bart.offset[..., None], kw['y_train'], rtol=1e-6)
    expected_cov_inv = (2**2 * kw['num_trees']) / tau_num**2
    leaf_prior_cov_inv = bart._mcmc_state.forest.leaf_prior_cov_inv
    if leaf_prior_cov_inv.ndim == 2:
        expected_cov_inv = jnp.eye(leaf_prior_cov_inv.shape[0]) * expected_cov_inv
    assert_allclose(leaf_prior_cov_inv, expected_cov_inv, rtol=1e-6)

    # in the multivariate case, it's not exactly 0 because matrix inversion
    # adds an epsilon to handle ill-conditioned matrices
    assert_allclose(bart._burnin_trace.log_likelihood, 0.0, atol=1e-6)
    assert_allclose(bart._main_trace.log_likelihood, 0.0, atol=1e-6)


def test_two_datapoints(bkw: BartKW) -> None:
    """Check automatic data scaling with 2 datapoints."""
    kw = set_num_datapoints(bkw.kw, 2)
    init_kw = dict(kw.get('init_kw', {}))
    init_kw.update(
        save_ratios=True, min_points_per_decision_node=None, min_points_per_leaf=None
    )
    kw['init_kw'] = init_kw
    bart = Bart(**kw)
    if kw['outcome_type'] != 'binary':
        ref_sigest = jnp.where(bart._binary_mask, 0.0, kw['y_train'].std(axis=-1))
        assert_close_matrices(bart.sigest, ref_sigest, rtol=1e-6)
    if kw.get('usequants', False):
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
    """Simple check that the `xinfo` parameter works."""
    with debug_nans(False):
        xinfo = jnp.array(
            [[1.1, 2.3, jnp.nan], [-50, 10, 20], [jnp.nan, jnp.nan, jnp.nan]]
        )
    kw = dict(
        x_train=jnp.empty((3, 0)),
        y_train=jnp.empty(0),
        ndpost=0,
        nskip=0,
        usequants=True,
        numcut=0,
        xinfo=xinfo,
    )
    bart = Bart(**kw)

    xinfo_wo_nan = jnp.where(jnp.isnan(xinfo), jnp.finfo(jnp.float32).max, xinfo)
    assert_array_equal(bart._splits, xinfo_wo_nan)
    assert_array_equal(bart._mcmc_state.forest.max_split, [2, 3, 0])


def test_xinfo_wrong_p() -> None:
    """Check that `xinfo` must have the same number of rows as `X`."""
    with debug_nans(False):
        xinfo = jnp.array(
            [[1.1, 2.3, jnp.nan], [-50, 10, 20], [jnp.nan, jnp.nan, jnp.nan]]
        )
    kw = dict(
        x_train=jnp.empty((5, 0)), y_train=jnp.empty(0), ndpost=0, nskip=0, xinfo=xinfo
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
        rhat_nstub = rhat([nstub_mcmc, nstub_prior])
        assert rhat_nstub < 1.01

    if (p, nsplits) != (1, 1):
        with subtests.test('number of simple trees'):
            nsimple_mcmc = count_simple_trees(bart._main_trace.split_tree)
            nsimple_prior = count_simple_trees(prior_trace.split_tree)
            rhat_nsimple = rhat([nsimple_mcmc, nsimple_prior])
            assert rhat_nsimple < 1.01

        varcount_prior = compute_varcount(
            bart._mcmc_state.forest.max_split.size, prior_trace
        )

        with subtests.test('varcount'):
            rhat_varcount = multivariate_rhat([bart.varcount, varcount_prior])
            if p == 10:
                assert rhat_varcount < 1.4
            else:
                assert rhat_varcount < 1.05

        with subtests.test('number of nodes'):
            sum_varcount_mcmc = bart.varcount.sum(axis=1)
            sum_varcount_prior = varcount_prior.sum(axis=1)
            rhat_sum_varcount = rhat([sum_varcount_mcmc, sum_varcount_prior])
            assert rhat_sum_varcount < 1.05

        with subtests.test('imbalance index'):
            imb_mcmc = avg_imbalance_index(bart._main_trace.split_tree)
            imb_prior = avg_imbalance_index(prior_trace.split_tree)
            rhat_imb = rhat([imb_mcmc, imb_prior])
            assert rhat_imb < 1.02

        with subtests.test('average max tree depth'):
            maxd_mcmc = avg_max_tree_depth(bart._main_trace.split_tree)
            maxd_prior = avg_max_tree_depth(prior_trace.split_tree)
            rhat_maxd = rhat([maxd_mcmc, maxd_prior])
            assert rhat_maxd < 1.02

        with subtests.test('max tree depth distribution'):
            dd_mcmc = bart.depth_distr()
            dd_prior = forest_depth_distr(prior_trace.split_tree)
            rhat_dd = multivariate_rhat([dd_mcmc.squeeze(0), dd_prior])
            assert rhat_dd < 1.05

    with subtests.test('y_test'):
        X = random.randint(keys.pop(), (p, 30), 0, nsplits + 1)
        yhat_mcmc = bart._predict(X)
        yhat_prior = evaluate_trace(X, prior_trace)
        rhat_yhat = multivariate_rhat([yhat_mcmc, yhat_prior])
        assert rhat_yhat < 1.1


def run_bart_like_prior(
    key: Key[Array, ''], p: int, nsplits: int, subtests: SubTests
) -> Bart:
    """Run `Bart` without datapoints to sample the prior distribution."""
    xinfo = jnp.broadcast_to(jnp.arange(nsplits, dtype=jnp.float32), (p, nsplits))

    bart = Bart(
        x_train=jnp.empty((p, 0)),
        y_train=jnp.empty(0),
        num_trees=20,
        ndpost=1000,
        nskip=3000,
        printevery=None,
        xinfo=xinfo,
        seed=key,
        num_chains=None,
        init_kw=dict(
            min_points_per_decision_node=None,
            min_points_per_leaf=None,
            save_ratios=True,
        ),
    )

    with subtests.test('likelihood ratio = 1'):
        assert_array_equal(bart._burnin_trace.log_likelihood, 0.0)
        assert_array_equal(bart._main_trace.log_likelihood, 0.0)

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
        bart.ndpost,
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


@pytest.mark.flaky
@pytest.mark.timeout(32)
def test_interrupt(bkw: BartKW) -> None:
    """Test that the MCMC can be interrupted with ^C."""
    kw = bkw.kw
    kw.update(printevery=1, ndpost=0, nskip=10000)

    with (
        pytest.raises(KeyboardInterrupt),
        periodic_sigint(first_after=3.0, interval=1.0),
    ):
        Bart(**kw)


def test_polars(bkw: BartKW) -> None:
    """Test passing data as DataFrame and Series."""
    kw = bkw.kw
    if kw['y_train'].ndim == 2:
        pytest.skip('Dataframe input for y_train not supported.')

    bart = Bart(**kw)
    pred = bart.predict(bkw.x_test, kind='latent_samples')

    kw2 = dict(kw)
    kw2.update(
        seed=random.clone(kw2['seed']),
        x_train=pl.DataFrame(numpy.array(kw['x_train']).T),
        y_train=pl.Series(numpy.array(kw['y_train'])),
        w=None if kw.get('w') is None else pl.Series(numpy.array(kw['w'])),
    )
    bart2 = Bart(**kw2)
    x_test_pl = pl.DataFrame(numpy.array(bkw.x_test).T)
    pred2 = bart2.predict(x_test_pl, kind='latent_samples')

    rtol = 0 if pred.platform() == 'cpu' else 2e-6

    assert_close_matrices(
        bart.predict('train', kind='latent_samples'),
        bart2.predict('train', kind='latent_samples'),
        rtol=rtol,
    )
    sdev1 = bart.get_error_sdev() if kw['outcome_type'] != 'binary' else None
    sdev2 = bart2.get_error_sdev() if kw['outcome_type'] != 'binary' else None
    if sdev1 is not None:
        assert_close_matrices(sdev1, sdev2, rtol=rtol)
    assert_close_matrices(pred, pred2, rtol=rtol)


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

    maxdepth = kw.get('maxdepth', 6)
    leaf_indices_type = select_type(maxdepth <= 8)
    split_trees_type = X_type = select_type(kw['numcut'] <= 255)
    var_trees_type = select_type(kw['x_train'].shape[0] <= 256)

    assert bart._mcmc_state.forest.var_tree.dtype == var_trees_type
    assert bart._mcmc_state.forest.split_tree.dtype == split_trees_type
    assert bart._mcmc_state.forest.leaf_indices.dtype == leaf_indices_type
    assert bart._mcmc_state.X.dtype == X_type
    assert bart._mcmc_state.forest.max_split.dtype == split_trees_type


def check_data_sharding(x: Array | None, mesh: jax.sharding.Mesh) -> None:
    """Check the sharding of `x` assuming it may be sharded only along the last 'data' axis."""
    if x is None:
        return
    elif mesh is None:
        assert isinstance(x.sharding, SingleDeviceSharding)
    elif 'data' in mesh.axis_names:
        expected_num_devices = min(2, get_device_count())
        assert x.sharding.num_devices == expected_num_devices
        expected_spec = (None,) * (x.ndim - 1) + ('data',)
        assert get_normal_spec(x) == normalize_spec(expected_spec, mesh, x.shape)


def check_chain_sharding(x: Array | None, mesh: jax.sharding.Mesh) -> None:
    """Check the sharding of `x` assuming it may be sharded only along the first 'chains' axis."""
    if x is None:
        return
    elif mesh is None:
        assert isinstance(x.sharding, SingleDeviceSharding)
    elif 'chains' in mesh.axis_names:
        expected_num_devices = min(2, get_device_count())
        assert x.sharding.num_devices == expected_num_devices
        assert get_normal_spec(x) == ('chains',) + (None,) * (x.ndim - 1)


def get_expect_sharded(kw: dict) -> bool:
    """Check whether we expect sharding to be set up based on the arguments."""
    return (
        kw.get('num_chain_devices') is not None
        or kw.get('num_data_devices') is not None
    )


def test_sharding(bkw: BartKW) -> None:
    """Check that chains live on their own devices throughout the interface."""
    kw = bkw.kw
    bart = Bart(**kw)

    expect_sharded = get_expect_sharded(kw)
    mesh = bart._mcmc_state.config.mesh
    assert expect_sharded == (mesh is not None)

    check = partial(check_sharding, mesh=mesh)
    check(bart._mcmc_state)
    check(bart._burnin_trace)
    check(bart._main_trace)

    check_chain = partial(check_chain_sharding, mesh=mesh)

    yhat_train = bart.predict('train', kind='latent_samples')
    check_chain(yhat_train)
    check_data_sharding(yhat_train, mesh)

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
        p, _ = kw['x_train'].shape
        i = random.randint(keys.pop(), (), 0, p)
        vp = jnp.full(p, 0.001).at[i].set(1)
        vp /= vp.sum()
        kw.update(sparse=False, varprob=vp)
        bart = Bart(**kw)
        vc = bart.varcount_mean
        vc /= vc.sum()
        assert vc[i] > vp[i] * 0.6

    def test_positive(self, bkw: BartKW, subtests: SubTests) -> None:
        """Check that an error is raised if varprob is not > 0."""
        kw = bkw.kw
        p, _ = kw['x_train'].shape

        with subtests.test('not negative'):
            assert p > 1
            varprob = jnp.ones(p).at[0].set(-1.0)
            kw_neg = dict(kw, varprob=varprob)
            with pytest.raises(EquinoxRuntimeError, match='varprob must be > 0'):
                Bart(**kw_neg)

        with subtests.test('not 0'):
            varprob = jnp.zeros(p).at[0].set(1.0)
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
    # terminates the process
    buf = StringIO()
    with array_garbage_collection_guard('log'), redirect_stderr(buf):
        yield
    captured = buf.getvalue()
    if '`jax.Array` was deleted by the Python garbage collector' in captured:
        raise RuntimeError(captured)


def test_debug_checks(keys: split, bkw: BartKW) -> None:
    """Run with invasive jax debug options active."""
    collect()
    with (
        debug_nans(True),
        debug_infs(True),
        debug_key_reuse(True),
        catch_array_gc_guard(),
    ):
        run_bart_and_block(bkw, keys)
        collect()


def test_equiv_sharding(bkw: BartKW, subtests: SubTests) -> None:
    """Check that the result is the same with/without sharding."""
    if get_disable_problematic_sharding():
        pytest.skip('Sharding disabled by --disable-problematic-sharding')
    if len(jax.devices()) < 2:
        pytest.skip('Need at least 2 devices for this test')

    baseline_kw = tree.map(lambda x: x, bkw.kw)
    baseline_kw.update(
        num_chain_devices=None, num_data_devices=None, nskip=0, ndpost=20, num_chains=2
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

    if len(jax.devices()) >= 4:
        with subtests.test('shard data and chains'):
            both_kw = tree.map(lambda x: x, baseline_kw)
            both_kw.update(num_chain_devices=2, num_data_devices=2)
            bart_both = Bart(**both_kw)
            bart_both = remove_mesh(bart_both)
            tree.map_with_path(check_equal, bart, bart_both)


def test_num_trees(bkw: BartKW, subtests: SubTests) -> None:
    """Test the number of trees."""
    kw = bkw.kw
    kw.update(nskip=0, ndpost=0)

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
            kwargs=dict(num_trees=5, ndpost=0, nskip=0, num_chains=None),
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

        with subtests.test('lamda'):
            bart = Bart(lamda=1.0, **kw)
            assert bart.sigest is None
            assert bart._mcmc_state.error_cov_scale.shape == (k, k)

    def test_mv_rejects_weights(self, example_data: ExampleData) -> None:
        """MV + weights should raise."""
        x, y, w, kw = example_data
        with pytest.raises(ValueError, match='Weights'):
            Bart(x_train=x, y_train=y, w=w, **kw)

    def test_mixed_rejects_weights(self, example_data: ExampleData) -> None:
        """Mixed outcome_type + weights should raise."""
        x, y, w, kw = example_data
        k, _ = y.shape
        with pytest.raises(ValueError, match='univariate continuous'):
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
    if y_mv.ndim == 1:
        y_mv = y_mv[None, :]
    y_mv = y_mv[:1, :]
    y_uv = y_mv.squeeze(0)

    bkw.kw.update(outcome_type=outcome_type, nskip=0, ndpost=0, w=None)
    del bkw.kw['y_train']
    bart_uv = Bart(y_train=y_uv, **bkw.kw)
    bart_mv = Bart(y_train=y_mv, **bkw.kw)

    state_uv = bart_uv._mcmc_state
    state_mv = bart_mv._mcmc_state

    # Residuals and error covariance
    assert_close_matrices(state_uv.resid, state_mv.resid.squeeze(-2))
    assert_allclose(
        state_uv.error_cov_inv, state_mv.error_cov_inv.squeeze((-2, -1)), rtol=1e-6
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
    assert_array_equal(state_uv.forest.var_tree, state_mv.forest.var_tree)
    assert_array_equal(state_uv.forest.split_tree, state_mv.forest.split_tree)
    assert_array_equal(state_uv.forest.leaf_tree, state_mv.forest.leaf_tree.squeeze(-2))
    assert_array_equal(state_uv.forest.leaf_indices, state_mv.forest.leaf_indices)


def test_get_latent_prec_only_continuous(bkw: BartKW) -> None:
    """get_latent_prec(only_continuous=True) removes binary components."""
    kw = bkw.kw
    if kw['y_train'].ndim < 2:
        pytest.skip('UV variant')

    bart = Bart(**kw)
    outcome_type = kw['outcome_type']
    if outcome_type == 'binary':
        with pytest.raises(ValueError, match='only binary'):
            bart.get_latent_prec(only_continuous=True)
        return

    k, _ = kw['y_train'].shape

    prec = bart.get_latent_prec(only_continuous=True)
    if isinstance(outcome_type, list):
        kc = sum(1 for t in outcome_type if t != 'binary')
    else:
        kc = k

    ndpost = bart.ndpost
    nskip = kw['nskip']
    num_chains = kw.get('num_chains')
    if num_chains is not None:
        assert prec.shape == (num_chains, nskip + ndpost // num_chains, kc, kc)
    else:
        assert prec.shape == (nskip + ndpost, kc, kc)


def test_get_error_sdev_values(bkw: BartKW) -> None:
    """get_error_sdev matches manual computation from precision matrices."""
    kw = bkw.kw
    outcome_type = kw['outcome_type']
    if outcome_type == 'binary':
        pytest.skip('binary variant')
    bart = Bart(**kw)
    nskip = kw['nskip']

    with debug_nans(False):
        sdev = bart.get_error_sdev()
        if sdev.ndim == 1:  # univariate
            sdev = sdev[:, None]  # reshape as vector of length 1

    # manual: invert each precision matrix, take sqrt of diagonal
    prec = bart.get_latent_prec()
    if prec.ndim < 3:  # univariate
        prec = prec[..., :, None, None]  # reshape as 1x1 matrix
    prec = prec[..., nskip:, :, :]  # skip burnin
    prec = lax.collapse(prec, 0, -2)  # flatten chains

    cov = jnp.linalg.inv(prec)
    sdev_ref = jnp.sqrt(jnp.diagonal(cov, axis1=-2, axis2=-1))

    # for mixed, compare only continuous components (binary have NaN sdev)
    mask = jnp.atleast_1d(~bart._binary_mask)
    sdev = sdev[:, mask]
    sdev_ref = sdev_ref[:, mask]

    assert_close_matrices(sdev, sdev_ref, rtol=1e-5)
