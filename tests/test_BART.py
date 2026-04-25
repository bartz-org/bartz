# bartz/tests/test_BART.py
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

"""Test `bartz.BART`.

This is the main suite of tests.
"""

from dataclasses import dataclass, replace
from functools import partial
from sys import version_info
from typing import Any, Literal

import jax
import numpy
import polars as pl
import pytest
from equinox import EquinoxRuntimeError, tree_at
from jax import block_until_ready, debug_nans, devices, random, tree, vmap
from jax import numpy as jnp
from jax.scipy.special import logit, ndtr
from jax.sharding import Mesh, SingleDeviceSharding
from jax.tree_util import KeyPath, keystr
from jaxtyping import Array, Float32, Int32, Key, PyTree, Shaped, UInt
from numpy.testing import assert_allclose, assert_array_equal
from pytest_subtests import SubTests

from bartz.debug import TraceWithOffset, sample_prior, trees_BART_to_bartz
from bartz.debug import debug_gbart as gbart
from bartz.debug import debug_mc_gbart as mc_gbart
from bartz.grove import (
    check_trace,
    forest_depth_distr,
    is_actual_leaf,
    tree_actual_depth,
    tree_depth,
    tree_depths,
)
from bartz.jaxext import get_default_device, get_device_count, split
from bartz.mcmcloop import compute_varcount, evaluate_trace
from bartz.mcmcstep import State
from bartz.mcmcstep._state import chain_vmap_axes
from tests.conftest import get_disable_problematic_sharding
from tests.test_interface import BartKW, gen_X, gen_y, make_kw
from tests.test_mcmcstep import check_sharding, get_normal_spec, normalize_spec
from tests.util import (
    assert_close_matrices,
    assert_different_matrices,
    get_old_python_tuple,
    multivariate_rhat,
    periodic_sigint,
    rhat,
)

try:
    from tests.rbartpackages import BART3
except ValueError as exc:
    # on PR ci, R is not installed because the tests using it are skipped
    if 'r_home' not in str(exc):
        raise


def bart_kw_to_mc_gbart(bkw: BartKW) -> dict[str, Any]:
    """Convert `Bart` keyword arguments to `mc_gbart` keyword arguments."""
    kw = dict(bkw.kw)

    # outcome_type -> type
    outcome_type = kw.pop('outcome_type', 'continuous')
    kw['type'] = 'pbart' if outcome_type == 'binary' else 'wbart'

    # num_trees -> ntree
    kw['ntree'] = kw.pop('num_trees')  # must be present bc different defaults

    # num_chains -> mc_cores
    num_chains = kw.pop('num_chains')  # must be present bc different defaults
    assert num_chains != 1  # because mc_gbart does not have an equivalent option
    mc_cores = 1 if num_chains is None else num_chains
    kw['mc_cores'] = mc_cores

    # n_save (per-chain) -> ndpost (across-chains total)
    kw['ndpost'] = kw.pop('n_save') * mc_cores

    # collect bart_kwargs from top-level Bart params
    bart_kwargs: dict[str, Any] = {}

    # drop num_chain_devices on cpu because mc_gbart has automatic sharding on
    # cpu, unless it's None which disables the automatism
    if 'num_chain_devices' in kw:
        num_chain_devices = kw.pop('num_chain_devices')
        if get_default_device().platform != 'cpu' or num_chain_devices is None:
            bart_kwargs['num_chain_devices'] = num_chain_devices

    if 'num_data_devices' in kw:
        bart_kwargs['num_data_devices'] = kw.pop('num_data_devices')

    # maxdepth has the same default
    if 'maxdepth' in kw:
        bart_kwargs['maxdepth'] = kw.pop('maxdepth')

    # init_kw must be present bc it contains min_points_per_leaf which has
    # different defaults
    init_kw = dict(kw.pop('init_kw'))
    # min_points_per_leaf: remove if equal to mc_gbart default (5) so mc_gbart's
    # default-setting code is tested
    if init_kw['min_points_per_leaf'] == 5:
        del init_kw['min_points_per_leaf']
    bart_kwargs['init_kw'] = init_kw
    kw['bart_kwargs'] = bart_kwargs

    # re-add x_test
    kw['x_test'] = bkw.x_test

    return kw


def make_gbart_kw(key: Key[Array, ''], variant: int) -> dict[str, Any]:
    """Return a dictionary of keyword arguments for `mc_gbart`."""
    return bart_kw_to_mc_gbart(make_kw(key, variant))


N_VARIANTS = 3


@pytest.fixture(
    params=[
        1,
        pytest.param(2, marks=pytest.mark.slow),
        pytest.param(3, marks=pytest.mark.slow),
    ],
    scope='module',
)
def variant(request: pytest.FixtureRequest) -> int:
    """Return a parametrized indicator to select different BART configurations."""
    return request.param


@pytest.fixture
def kw(keys: split, variant: int) -> dict[str, Any]:
    """Return a dictionary of keyword arguments for BART."""
    return make_gbart_kw(keys.pop(), variant)


@dataclass(frozen=True)
class CachedBart:
    """Pre-computed BART run shared between multiple tests that do not change the arguments."""

    kwargs: dict[str, Any]
    bart: mc_gbart


@pytest.mark.slow
class TestWithCachedBart:  # pragma: slow
    """Group of slow tests that check the same BART run, for efficiency."""

    @pytest.fixture(scope='class')
    def cachedbart(self, variant: int) -> CachedBart:
        """Return a pre-computed BART."""
        # create a random seed that depends only on the variant, since this
        # fixture is shared between multiple tests
        key = random.key(0x139CD0C0)
        keys = random.split(key, N_VARIANTS)
        key = keys[variant - 1]
        kw = make_gbart_kw(key, variant)

        # modify configs to make them appropriate for convergence checks and R
        p, n = kw['x_train'].shape
        nchains = 4
        kw.update(
            ntree=max(2 * n, p),
            nskip=3000,
            ndpost=nchains * 1002,
            # 1002 instead of 1000 because it is divisible by 3. R's BART3 caps
            # the number of chains at the number of cores on the machine. On CI
            # we have 2 or 3 cores, so this is going to happen. If it wasn't
            # divisible, ndpost would be rounded up to the next multiple for
            # BART3 but not for bartz that just honors the user request, the
            # lengths would not match, and so stacking to compute rhat would
            # fail.
            keepevery=1,
            mc_cores=nchains,
        )
        # R BART can't change the min_points_per_leaf setting
        kw.setdefault('bart_kwargs', {}).setdefault('init_kw', {}).update(
            min_points_per_decision_node=10, min_points_per_leaf=5
        )

        bart = mc_gbart(**kw)

        return CachedBart(kwargs=kw, bart=bart)

    def test_residuals_accuracy(self, cachedbart: CachedBart) -> None:
        """Check that running residuals are close to the recomputed final residuals."""
        accum_resid, actual_resid = cachedbart.bart.compare_resid(
            y=cachedbart.kwargs['y_train']
        )
        assert_close_matrices(accum_resid, actual_resid, rtol=1e-4)

    def test_convergence(self, cachedbart: CachedBart) -> None:
        """Run multiple chains and check convergence with rhat."""
        bart = cachedbart.bart
        nchains, _ = bart._mcmc_state.resid.shape
        nsamples = bart.ndpost // nchains
        kw = cachedbart.kwargs
        p, n = kw['x_train'].shape

        yhat_train = bart.yhat_train.reshape(nchains, nsamples, n)
        rhat_yhat_train = multivariate_rhat(yhat_train)
        assert rhat_yhat_train < 6
        print(f'{rhat_yhat_train.item()=}')

        if kw['type'] == 'pbart':  # binary regression
            prob_train = bart.prob_train.reshape(nchains, nsamples, n)
            rhat_prob_train = multivariate_rhat(prob_train)
            assert rhat_prob_train < 1.2
            print(f'{rhat_prob_train.item()=}')

        else:  # continuous regression
            sigma = bart.sigma[nsamples:, :].T
            rhat_sigma = rhat(sigma)
            assert rhat_sigma < 1.2
            print(f'{rhat_sigma.item()=}')

        if p < n:
            varcount = bart.varcount.reshape(nchains, nsamples, p)
            rhat_varcount = multivariate_rhat(varcount)
            assert rhat_varcount < 7
            print(f'{rhat_varcount.item()=}')

            if kw.get('sparse', False):  # pragma: no branch
                varprob = bart.varprob.reshape(nchains, nsamples, p)
                rhat_varprob = multivariate_rhat(varprob[:, :, 1:])
                # drop one component because varprob sums to 1
                assert rhat_varprob < 7
                print(f'{rhat_varprob.item()=}')

    def kw_bartz_to_BART3(self, key: Key[Array, ''], kw: dict, bart: mc_gbart) -> dict:
        """Convert bartz keyword arguments to R BART3 keyword arguments."""
        kw_BART: dict = dict(**kw, rm_const=False)
        kw_BART.pop('bart_kwargs')
        kw_BART.pop('maxdepth', None)
        for arg in 'w', 'printevery':
            if arg in kw_BART and kw_BART[arg] is None:
                kw_BART.pop(arg)
        kw_BART['seed'] = random.randint(key, (), 0, jnp.uint32(2**31)).item()

        # Set BART cutpoints manually. This means I am not checking that the
        # automatic cutpoint determination of BART is the same of my package. They
        # are similar but have some differences, and having exactly the same
        # cutpoints is more important for the test.
        kw_BART['transposed'] = True  # this disables predictors pre-processing
        kw_BART['numcut'] = bart._mcmc_state.forest.max_split
        kw_BART['xinfo'] = bart._splits

        return kw_BART

    def check_rbart(
        self, kw: dict[str, Any], bart: mc_gbart, rbart: BART3.mc_gbart
    ) -> None:
        """Subroutine for `test_comparison_BART3`, check that the R BART output is self-consistent."""
        # convert the trees to bartz format
        trees = rbart.treedraws['trees']
        trace, meta = trees_BART_to_bartz(trees, offset=rbart.offset)

        # check the trees are valid
        assert jnp.all(meta.numcut <= bart._mcmc_state.forest.max_split)
        bad = check_trace(trace, meta.numcut)
        num_bad = jnp.count_nonzero(bad)
        assert num_bad == 0

        # check varcount
        varcount = compute_varcount(meta.numcut.size, trace)
        assert jnp.all(varcount == rbart.varcount)

        # check yhat_train
        yhat_train = evaluate_trace(bart._mcmc_state.X, trace)
        assert_close_matrices(
            yhat_train, rbart.yhat_train.astype(numpy.float32), rtol=1e-6
        )

        # check yhat_test
        Xt = bart._bart._bin_predictors(kw['x_test'], bart._splits)
        yhat_test = evaluate_trace(Xt, trace)
        assert_close_matrices(
            yhat_test, rbart.yhat_test.astype(numpy.float32), rtol=1e-6
        )

        if kw['type'] == 'pbart':
            # check prob_train
            prob_train = ndtr(yhat_train)
            assert_close_matrices(
                prob_train, rbart.prob_train.astype(numpy.float32), rtol=1e-7
            )

            # check prob_test
            prob_test = ndtr(yhat_test)
            assert_close_matrices(
                prob_test, rbart.prob_test.astype(numpy.float32), rtol=1e-7
            )

    def test_comparison_BART3(
        self, cachedbart: CachedBart, keys: split, subtests: SubTests
    ) -> None:
        """Check `bartz.BART` gives results similar to the R package BART3."""
        bart = cachedbart.bart
        kw = cachedbart.kwargs
        p, n = kw['x_train'].shape

        # run R bart
        kw_BART = self.kw_bartz_to_BART3(keys.pop(), kw, bart)
        rbart = BART3.mc_gbart(**kw_BART)
        # use mc_gbart instead of gbart because gbart does not use the seed

        # first cross-check the outputs of R BART alone
        self.check_rbart(kw, bart, rbart)

        # compare results of bartz and BART

        with subtests.test('offset'):
            assert_allclose(bart.offset, rbart.offset, rtol=1e-6, atol=1e-7)
            # I would check sigest as well, but it's not in the R object despite what
            # the documentation says

        with subtests.test('yhat_train'):
            rhat_yhat_train = multivariate_rhat([bart.yhat_train, rbart.yhat_train])
            assert rhat_yhat_train < 3.5

        with subtests.test('yhat_test'):
            rhat_yhat_test = multivariate_rhat([bart.yhat_test, rbart.yhat_test])
            assert rhat_yhat_test < 3.5

        if kw['type'] == 'pbart':  # binary regression
            with subtests.test('prob_train'):
                rhat_prob_train = multivariate_rhat([bart.prob_train, rbart.prob_train])
                assert rhat_prob_train < 1.2

            with subtests.test('prob_test'):
                rhat_prob_test = multivariate_rhat([bart.prob_test, rbart.prob_test])
                assert rhat_prob_test < 1.2

        else:  # continuous regression
            with subtests.test('yhat_train_mean'):
                assert_close_matrices(
                    bart.yhat_train_mean,
                    rbart.yhat_train_mean.astype(numpy.float32),
                    rtol=0.8,
                )

            with subtests.test('yhat_test_mean'):
                assert_close_matrices(
                    bart.yhat_test_mean,
                    rbart.yhat_test_mean.astype(numpy.float32),
                    rtol=0.9,
                )

            with subtests.test('sigma'):
                rhat_sigma = rhat(
                    [bart.sigma_[-bart.ndpost :], rbart.sigma_[-rbart.ndpost :]]
                )
                assert rhat_sigma < 1.3

            with subtests.test('sigma_mean'):
                assert_allclose(bart.sigma_mean, rbart.sigma_mean, rtol=0.1)

        with subtests.test('tree_node_count'):
            # check number of tree nodes in forest
            bart_count = bart.varcount.sum(axis=1)
            rbart_count = rbart.varcount.sum(axis=1)
            rhat_count = rhat([bart_count, rbart_count])
            assert rhat_count < 30  # genuinely bad, see below
            assert_allclose(bart_count.mean(), rbart_count.mean(), rtol=0.2)

        if p < n:
            # skip if p is large because it would be difficult for the MCMC to get
            # stuff about predictors right

            with subtests.test('varcount'):
                rhat_varcount = multivariate_rhat([bart.varcount, rbart.varcount])
                # there is a visible discrepancy on the number of nodes, with bartz
                # having deeper trees, this 6 is not just "not good to sampling
                # accuracy but close in practice."
                assert rhat_varcount < 7

            with subtests.test('varcount_mean'):
                assert_close_matrices(
                    bart.varcount_mean,
                    rbart.varcount_mean.astype(numpy.float32),
                    rtol=0.7,
                    atol=9,
                )

            if kw.get('sparse', False):  # pragma: no branch
                with subtests.test('varprob'):
                    rhat_varprob = multivariate_rhat(
                        clipped_logit(
                            jnp.stack([bart.varprob, rbart.varprob])[:, :, 1:], 1e-5
                        )
                    )
                    # drop one component because varprob sums to 1
                    assert rhat_varprob < 4

                with subtests.test('varprob_mean'):
                    assert_close_matrices(
                        logit(bart.varprob_mean[1:]),
                        logit(rbart.varprob_mean[1:]),
                        atol=2.1 * (p - 1) ** 0.5,
                    )

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

        assert_different(bart._mcmc_state, rtol=0.02)
        assert_different(bart._main_trace, rtol=0.03)
        assert_different(bart._burnin_trace, rtol=0.03)


def clipped_logit(x: Array, eps: float) -> Array:
    """Compute the logit of x, clipping x to [eps, 1-eps] to avoid infinities."""
    return logit(jnp.clip(x, eps, 1 - eps))


def test_sequential_guarantee(kw: dict, subtests: SubTests) -> None:
    """Check that the way iterations are saved does not influence the result."""
    # reference run
    kw['keepevery'] = 1
    bart1 = mc_gbart(**kw)

    # run moving some samples form burn-in to main
    kw2 = kw.copy()
    kw2['seed'] = random.clone(kw2['seed'])
    if kw2.get('sparse', False):
        kw2.setdefault('bart_kwargs', {}).setdefault('init_kw', {}).setdefault(
            'sparse_on_at', kw2['nskip'] // 2
        )
    delta = 1
    kw2['nskip'] -= delta
    kw2['ndpost'] += delta * kw2['mc_cores']
    bart2 = mc_gbart(**kw2)
    n = kw2['y_train'].size
    bart2_yhat_train = bart2.yhat_train.reshape(
        kw2['mc_cores'], kw2['ndpost'] // kw2['mc_cores'], n
    )[:, delta:, :].reshape(bart1.ndpost, n)

    with subtests.test('shift burn-in'):
        rtol = 0 if bart1.yhat_train.platform() == 'cpu' else 2e-6
        # on gpu typically it works fine, but in one case there was a small
        # numerical difference in one of two chains
        assert_close_matrices(bart1.yhat_train, bart2_yhat_train, rtol=rtol)

    # run keeping 1 every 2 samples
    kw3 = kw.copy()
    kw3['seed'] = random.clone(kw3['seed'])
    kw3['keepevery'] = 2
    bart3 = mc_gbart(**kw3)
    bart1_yhat_train = bart1.yhat_train.reshape(
        kw3['mc_cores'], kw3['ndpost'] // kw3['mc_cores'], n
    )[:, 1::2, :]
    bart3_yhat_train = bart3.yhat_train.reshape(
        kw3['mc_cores'], kw3['ndpost'] // kw3['mc_cores'], n
    )[:, : bart1_yhat_train.shape[1], :]

    with subtests.test('change thinning'):
        rtol = 0 if bart1.yhat_train.platform() == 'cpu' else 2e-6
        # on gpu typically it works fine, but in one case there was a small
        # numerical difference in one of two chains
        assert_close_matrices(
            bart1_yhat_train, bart3_yhat_train, rtol=rtol, reduce_rank=True
        )


def test_output_shapes(kw: dict[str, Any]) -> None:
    """Check the output shapes of all the array attributes of `bartz.BART.mc_gbart`."""
    bart = mc_gbart(**kw)

    ndpost = kw['ndpost']
    nskip = kw['nskip']
    mc_cores = kw['mc_cores']
    p, n = kw['x_train'].shape
    _, m = kw['x_test'].shape

    binary = kw['type'] == 'pbart'

    assert ndpost == bart.ndpost
    assert bart.offset.shape == ()
    if binary:
        assert bart.prob_test.shape == (ndpost, m)
        assert bart.prob_test_mean.shape == (m,)
        assert bart.prob_train.shape == (ndpost, n)
        assert bart.prob_train_mean.shape == (n,)
        assert bart.sigma is None
        assert bart.sigma_ is None
        assert bart.sigma_mean is None
    else:
        assert bart.prob_test is None
        assert bart.prob_test_mean is None
        assert bart.prob_train is None
        assert bart.prob_train_mean is None
        if mc_cores == 1:
            assert bart.sigma.shape == (nskip + ndpost,)
        else:
            assert bart.sigma.shape == (nskip + ndpost // mc_cores, mc_cores)
        assert bart.sigma_.shape == (ndpost,)
        assert bart.sigma_mean.shape == ()
    assert bart.varcount.shape == (ndpost, p)
    assert bart.varcount_mean.shape == (p,)
    assert bart.varprob.shape == (ndpost, p)
    assert bart.varprob_mean.shape == (p,)
    assert bart.yhat_test.shape == (ndpost, m)
    if binary:
        assert bart.yhat_test_mean is None
    else:
        assert bart.yhat_test_mean.shape == (m,)
    assert bart.yhat_train.shape == (ndpost, n)
    if binary:
        assert bart.yhat_train_mean is None
    else:
        assert bart.yhat_train_mean.shape == (n,)


def test_output_types(kw: dict[str, Any]) -> None:
    """Check the output types of all the attributes of BART.gbart."""
    bart = mc_gbart(**kw)

    binary = kw['type'] == 'pbart'

    assert bart.offset.dtype == jnp.float32
    assert isinstance(bart.ndpost, int)
    if binary:
        assert bart.prob_test.dtype == jnp.float32
        assert bart.prob_test_mean.dtype == jnp.float32
        assert bart.prob_train.dtype == jnp.float32
        assert bart.prob_train_mean.dtype == jnp.float32
    else:
        assert bart.sigma.dtype == jnp.float32
        assert bart.sigma_.dtype == jnp.float32
        assert bart.sigma_mean.dtype == jnp.float32
    assert bart.varcount.dtype == jnp.int32
    assert bart.varcount_mean.dtype == jnp.float32
    assert bart.varprob.dtype == jnp.float32
    assert bart.varprob_mean.dtype == jnp.float32
    assert bart.yhat_test.dtype == jnp.float32
    if not binary:
        assert bart.yhat_test_mean.dtype == jnp.float32
    assert bart.yhat_train.dtype == jnp.float32
    if not binary:
        assert bart.yhat_train_mean.dtype == jnp.float32


def test_predict(kw: dict[str, Any]) -> None:
    """Check that the public BART.gbart.predict method works."""
    bart = mc_gbart(**kw)
    yhat_train = bart.predict(kw['x_train'])
    assert_close_matrices(bart.yhat_train, yhat_train, rtol=1e-6)
    # the need for this approximate comparison is surprising; exact comparison
    # fails on cpu ci on linux


class TestVarprobAttr:
    """Test the `mc_gbart.varprob` attribute."""

    def test_basic_properties(self, kw: dict[str, Any]) -> None:
        """Basic checks of the `varprob` attribute."""
        bart = mc_gbart(**kw)

        # basic properties of probabilities
        assert jnp.all(bart.varprob >= 0)
        assert jnp.all(bart.varprob <= 1)
        assert_allclose(bart.varprob.sum(axis=1), 1, rtol=1e-6)

        # probabilities are either 0 or 1/peff if sparsity is disabled
        sparse = kw.get('sparse', False)
        if not sparse:
            unique = jnp.unique(bart.varprob)
            assert unique.size in (1, 2)
            if unique.size == 2:  # pragma: no cover
                assert unique[0] == 0

        # the mean is the mean
        assert_array_equal(bart.varprob_mean, bart.varprob.mean(axis=0))

    def test_blocked_vars(self, keys: split) -> None:
        """Check that varprob = 0 on predictors blocked a priori."""
        X = gen_X(keys.pop(), 2, 30, 'continuous')
        y = gen_y(keys.pop(), X, None, 'continuous')
        with debug_nans(False):
            xinfo = jnp.array([[jnp.nan], [0]])
        bart = mc_gbart(x_train=X, y_train=y, xinfo=xinfo, seed=keys.pop())
        assert_array_equal(bart._mcmc_state.forest.max_split, [0, 1])
        assert_array_equal(bart.varprob_mean, [0, 1])
        assert jnp.all(bart.varprob_mean == bart.varprob)


@pytest.mark.parametrize('theta', ['fixed', 'free'])
def test_variable_selection(keys: split, theta: Literal['fixed', 'free']) -> None:
    """Check that variable selection works."""
    # data config
    p = 100  # number of predictors
    peff = 5  # number of actually used predictors
    n = 1000

    # generate sparsity pattern
    mask = jnp.zeros(p, bool).at[:peff].set(True)
    mask = random.permutation(keys.pop(), mask)
    s = mask.astype(float)

    # generate data
    X = gen_X(keys.pop(), p, n, 'continuous')
    y = gen_y(keys.pop(), X, None, 'continuous', s=s)

    # run bart
    bart = mc_gbart(
        x_train=X,
        y_train=y,
        nskip=1000,
        sparse=True,
        theta=peff if theta == 'fixed' else None,
        seed=keys.pop(),
    )

    # check that the variables have been identified
    assert bart.varprob_mean[mask].sum() >= 0.9
    assert bart.varprob_mean[mask].min().item() > 0.5 / peff
    assert bart.varprob_mean[~mask].max().item() < 1 / (p - peff)


def test_scale_shift(kw: dict[str, Any]) -> None:
    """Check self-consistency of rescaling the inputs."""
    if kw['type'] == 'pbart':
        pytest.skip('Cannot rescale binary responses.')

    bart1 = mc_gbart(**kw)

    offset = 0.4703189
    scale = 0.5294714
    kw.update(y_train=offset + kw['y_train'] * scale, seed=random.clone(kw['seed']))
    # note: using the same seed does not guarantee stable error because the mcmc
    # makes discrete choices based on thresholds on floats, so numerical error
    # can be amplified.
    bart2 = mc_gbart(**kw)

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
    assert_close_matrices(
        bart1.yhat_train, (bart2.yhat_train - offset) / scale, rtol=1e-5
    )
    assert_close_matrices(
        bart1.yhat_train_mean, (bart2.yhat_train_mean - offset) / scale, rtol=1e-5
    )
    assert_close_matrices(
        bart1.yhat_test, (bart2.yhat_test - offset) / scale, rtol=1e-5
    )
    assert_close_matrices(
        bart1.yhat_test_mean, (bart2.yhat_test_mean - offset) / scale, rtol=1e-5
    )
    assert_close_matrices(bart1.sigma, bart2.sigma / scale, rtol=1e-5)
    assert_allclose(bart1.sigma_mean, bart2.sigma_mean / scale, rtol=1e-6, atol=1e-6)


def test_min_points_per_decision_node(kw: dict[str, Any]) -> None:
    """Check that the limit of at least 10 datapoints per decision node is respected."""
    kw.setdefault('bart_kwargs', {}).setdefault('init_kw', {}).update(
        min_points_per_leaf=None
    )
    bart = mc_gbart(**kw)
    distr = bart.points_per_decision_node_distr()
    distr_marg = distr.sum(axis=(0, 1))

    min_points = (
        kw.get('bart_kwargs', {})
        .get('init_kw', {})
        .get('min_points_per_decision_node', 10)
    )

    if min_points is None:
        assert distr_marg[9] > 0
    else:
        assert jnp.all(distr_marg[:min_points] == 0)
        assert jnp.any(distr_marg[min_points:] > 0)


def test_min_points_per_leaf(kw: dict[str, Any]) -> None:
    """Check that the limit of at least 5 datapoints per leaf is respected."""
    kw.setdefault('bart_kwargs', {}).setdefault('init_kw', {}).update(
        min_points_per_decision_node=None
    )
    bart = mc_gbart(**kw)
    distr = bart.points_per_leaf_distr()
    distr_marg = distr.sum(axis=(0, 1))

    min_points = (
        kw.get('bart_kwargs', {}).get('init_kw', {}).get('min_points_per_leaf', 5)
    )

    if min_points is None:
        assert distr_marg[4] > 0
    else:
        assert jnp.all(distr_marg[:min_points] == 0)
        assert distr_marg[min_points] > 0


def set_num_datapoints(kw: dict, n: int) -> dict:
    """Set the number of datapoints in the kw dictionary."""
    assert n <= kw['y_train'].size
    kw = kw.copy()
    kw['x_train'] = kw['x_train'][:, :n]
    kw['y_train'] = kw['y_train'][:n]
    if kw.get('w') is not None:
        kw['w'] = kw['w'][:n]
    return kw


def test_no_datapoints(kw: dict[str, Any]) -> None:
    """Check automatic data scaling with 0 datapoints."""
    # remove all datapoints
    kw = set_num_datapoints(kw, 0)

    # set the split grid manually because automatic setting relies on datapoints
    p, _ = kw['x_train'].shape
    nsplits = 10
    xinfo = jnp.broadcast_to(jnp.arange(nsplits, dtype=jnp.float32), (p, nsplits))
    kw.update(xinfo=xinfo)

    # disable data sharding
    kw.setdefault('bart_kwargs', {}).update(num_data_devices=None)

    # enable saving the likelihood ratio to check it's always 1
    kw.setdefault('bart_kwargs', {}).setdefault('init_kw', {}).update(
        save_ratios=True, min_points_per_decision_node=None, min_points_per_leaf=None
    )

    # run bart
    bart = mc_gbart(**kw)

    # check there are indeed 0 datapoints in the output
    ndpost = kw['ndpost']
    assert bart.yhat_train.shape == (ndpost, 0)

    # check default values that may be set in a special way if there are 0 datapoints
    assert bart.offset == 0
    if kw['type'] == 'pbart':
        tau_num = 3
        assert bart.sigest is None
    else:
        tau_num = 1
        assert bart.sigest == 1
    assert_allclose(
        bart._mcmc_state.forest.leaf_prior_cov_inv,
        (2**2 * kw['ntree']) / tau_num**2,
        rtol=1e-6,
    )

    # check the likelihood ratio is always 1
    assert_array_equal(bart._burnin_trace.log_likelihood, 0.0)
    assert_array_equal(bart._main_trace.log_likelihood, 0.0)


def test_one_datapoint(kw: dict[str, Any]) -> None:
    """Check automatic data scaling with 1 datapoint."""
    kw = set_num_datapoints(kw, 1)

    # set the split grid manually because otherwise there would be 0 cutpoints
    # when usequants=True, and computing varprob produces nans in that case
    if kw.get('usequants', False):
        p, _ = kw['x_train'].shape
        nsplits = 10
        xinfo = jnp.broadcast_to(jnp.arange(nsplits, dtype=jnp.float32), (p, nsplits))
        kw.update(xinfo=xinfo)

    # disable data sharding
    kw.setdefault('bart_kwargs', {}).update(num_data_devices=None)

    # enable saving the likelihood ratio to check it's always 1
    kw.setdefault('bart_kwargs', {}).setdefault('init_kw', {}).update(
        save_ratios=True, min_points_per_decision_node=None, min_points_per_leaf=None
    )

    bart = mc_gbart(**kw)
    if kw['type'] == 'pbart':
        tau_num = 3
        assert bart.sigest is None
        assert bart.offset == 0
    else:
        tau_num = 1
        assert bart.sigest == 1
        assert bart.offset == kw['y_train'].item()
    assert_allclose(
        bart._mcmc_state.forest.leaf_prior_cov_inv,
        (2**2 * kw['ntree']) / tau_num**2,
        rtol=1e-6,
    )

    # check the likelihood ratio is always 1
    assert_array_equal(bart._burnin_trace.log_likelihood, 0.0)
    assert_array_equal(bart._main_trace.log_likelihood, 0.0)


def test_two_datapoints(kw: dict[str, Any]) -> None:
    """Check automatic data scaling with 2 datapoints."""
    kw = set_num_datapoints(kw, 2)
    kw.setdefault('bart_kwargs', {}).setdefault('init_kw', {}).update(
        save_ratios=True, min_points_per_decision_node=None, min_points_per_leaf=None
    )
    bart = mc_gbart(**kw)
    if kw['type'] != 'pbart':
        assert_allclose(bart.sigest, kw['y_train'].std(), rtol=1e-6)
    if kw['usequants']:
        assert jnp.all(bart._mcmc_state.forest.max_split <= 1)
    assert not jnp.all(bart._burnin_trace.log_likelihood == 0.0)
    assert not jnp.all(bart._main_trace.log_likelihood == 0.0)


def test_few_datapoints(kw: dict[str, Any]) -> None:
    """Check that the trees cannot grow if there are not enough datapoints.

    If there are less than 10 datapoints, it is not possible to satisfy the 10
    points per decision node requirement, neither the 5 datapoints per leaf
    constraint.
    """
    kw.setdefault('bart_kwargs', {}).setdefault('init_kw', {}).update(
        min_points_per_decision_node=10, min_points_per_leaf=None
    )
    kw = set_num_datapoints(kw, 8)  # < 10 = 2 * 5, multiple of 2 shards
    bart = mc_gbart(**kw)
    assert jnp.all(bart.yhat_train == bart.yhat_train[:, :1])

    kw.setdefault('bart_kwargs', {}).setdefault('init_kw', {}).update(
        min_points_per_decision_node=None, min_points_per_leaf=5
    )
    kw['seed'] = random.clone(kw['seed'])
    bart = mc_gbart(**kw)
    assert jnp.all(bart.yhat_train == bart.yhat_train[:, :1])


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
        # these `usequants` and `numcut` values would lead to an error, so this
        # checks they are ignored if `xinfo` is specified
        usequants=True,
        numcut=0,
        xinfo=xinfo,
    )
    bart = mc_gbart(**kw)

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
        mc_gbart(**kw)


@pytest.mark.parametrize(
    ('p', 'nsplits'),
    [
        (1, 1),  # sure that trees do not grow beyond depth 2
        (3, 2),  # likely to have no available decision rules on some nodes
        (10, 1),  # always available decision rules, but never on the same variable
        (10, 255),  # likely always available decision rules for all variables
    ],
)
def test_prior(keys: split, p: int, nsplits: int, subtests: SubTests) -> None:
    """Check that the posterior without data is equivalent to the prior."""
    # run bart without data
    bart = run_bart_like_prior(keys.pop(), p, nsplits, subtests)

    # sample from prior
    prior_trace = sample_prior_like(keys.pop(), bart, subtests)

    with subtests.test('number of stub trees'):
        nstub_mcmc = count_stub_trees(bart._main_trace.split_tree)
        nstub_prior = count_stub_trees(prior_trace.split_tree)
        rhat_nstub = rhat([nstub_mcmc, nstub_prior])
        assert rhat_nstub < 1.01

    if (p, nsplits) != (1, 1):
        # all the following are equivalent to nstub in the 1-1 case

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
                # varcount is p-dimensional
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
        yhat_mcmc = bart._bart._predict(X)
        yhat_prior = evaluate_trace(X, prior_trace)
        rhat_yhat = multivariate_rhat([yhat_mcmc, yhat_prior])
        assert rhat_yhat < 1.1


def run_bart_like_prior(
    key: Key[Array, ''], p: int, nsplits: int, subtests: SubTests
) -> mc_gbart:
    """Run `mc_gbart` without datapoints to sample the prior distribution."""
    # set the split grid manually because automatic setting relies on datapoints
    xinfo = jnp.broadcast_to(jnp.arange(nsplits, dtype=jnp.float32), (p, nsplits))

    # configure bart to run many mcmc iterations, without data
    kw: dict = dict(
        x_train=jnp.empty((p, 0)),
        y_train=jnp.empty(0),
        ntree=20,
        ndpost=1000,
        nskip=3000,
        printevery=None,
        xinfo=xinfo,
        seed=key,
        mc_cores=1,
        bart_kwargs=dict(
            init_kw=dict(
                # unset limits on datapoints per node because there's no data
                min_points_per_decision_node=None,
                min_points_per_leaf=None,
                # save likelihood ratio to check it's 1
                save_ratios=True,
            )
        ),
    )

    bart = mc_gbart(**kw)

    with subtests.test('likelihood ratio = 1'):
        assert_array_equal(bart._burnin_trace.log_likelihood, 0.0)
        assert_array_equal(bart._main_trace.log_likelihood, 0.0)

    return bart


def sample_prior_like(
    key: Key[Array, ''], bart: mc_gbart, subtests: SubTests
) -> TraceWithOffset:
    """Sample from the prior with the same settings used in `bart`."""
    # extract p_nonterminal in original format from mcmc state
    p_nonterminal = bart._mcmc_state.forest.p_nonterminal
    max_depth = tree_depth(p_nonterminal)
    indices = 2 ** jnp.arange(max_depth - 1)
    p_nonterminal = p_nonterminal[indices]

    # sample from prior
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

    # pack up trees together with offset
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
    """Measure average tree imbalance in the forest.

    The imbalance is measured as the standard deviation of the depth of the
    leaves.
    """
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


def test_rhat(keys: split) -> None:
    """Test the multivariate R-hat implementation."""
    chains, divergent_chains = random.normal(keys.pop(), (2, 2, 1000, 10))
    mean_offset = jnp.arange(len(chains))
    divergent_chains += mean_offset[:, None, None]
    rhat = multivariate_rhat(chains)
    rhat_divergent = multivariate_rhat(divergent_chains)
    assert rhat < 1.02
    assert rhat_divergent > 5


def test_jit(kw: dict[str, Any]) -> None:
    """Test that jitting around the whole interface works."""
    # set printevery to None to move all iterations to the inner loop and avoid
    # multiple compilation
    kw.update(printevery=None)

    # do not count splitless variables because it breaks tracing
    kw.update(rm_const=False)

    # do not check trees because it breaks tracing
    kw.update(check_trees=False, check_replicated_trees=False)

    # set device as under jit it can not be inferred from the array
    platform = kw['y_train'].platform()
    kw.setdefault('bart_kwargs', {}).update(devices=jax.devices(platform))

    # negate mc_cores to silence error about device not possible to infer
    kw.update(mc_cores=-kw['mc_cores'])

    # remove arguments passed through the jit call
    X = kw.pop('x_train')
    y = kw.pop('y_train')
    w = kw.pop('w', None)
    key = kw.pop('seed')

    def task(
        X: Shaped[Array, 'p n'],
        y: Shaped[Array, ' n'],
        w: Float32[Array, ' n'] | None,
        key: Key[Array, ''],
    ) -> tuple[State, Shaped[Array, 'ndpost n']]:
        bart = mc_gbart(X, y, w=w, **kw, seed=key)
        return bart._mcmc_state, bart.yhat_train

    task_compiled = jax.jit(task)

    _state1, pred1 = task(X, y, w, key)
    _state2, pred2 = task_compiled(X, y, w, random.clone(key))

    assert_close_matrices(pred1, pred2, rtol=1e-5)


@pytest.mark.flaky
# it's flaky because the interrupt may be caught and converted by jax internals (#33054)
@pytest.mark.timeout(32)
def test_interrupt(kw: dict[str, Any]) -> None:
    """Test that the MCMC can be interrupted with ^C."""
    kw.update(printevery=1, ndpost=0, nskip=10000)

    # Send the first ^C after 3 s, if the time was too short, it would interrupt
    # a first interruptible phase of jax compilation. Then send ^C every second,
    # in case the first ^C landed during a second non-interruptible compilation
    # phase that eats ^C and ignores it.
    with (
        pytest.raises(KeyboardInterrupt),
        periodic_sigint(first_after=3.0, interval=1.0),
    ):
        block_until_ready(mc_gbart(**kw))


def test_polars(kw: dict[str, Any]) -> None:
    """Test passing data as DataFrame and Series."""
    bart = mc_gbart(**kw)
    pred = bart.predict(kw['x_test'])

    kw.update(
        seed=random.clone(kw['seed']),
        x_train=pl.DataFrame(numpy.array(kw['x_train']).T),
        x_test=pl.DataFrame(numpy.array(kw['x_test']).T),
        y_train=pl.Series(numpy.array(kw['y_train'])),
        w=None if kw.get('w') is None else pl.Series(numpy.array(kw['w'])),
    )
    bart2 = mc_gbart(**kw)
    pred2 = bart2.predict(kw['x_test'])

    rtol = 0 if pred.platform() == 'cpu' else 2e-6

    assert_close_matrices(bart.yhat_train, bart2.yhat_train, rtol=rtol)
    if bart.sigma is not None:
        assert_close_matrices(bart.sigma, bart2.sigma, rtol=rtol)
    assert_close_matrices(pred, pred2, rtol=rtol)


def test_data_format_mismatch(kw: dict[str, Any]) -> None:
    """Test that passing predictors with mismatched formats raises an error."""
    kw.update(
        x_train=pl.DataFrame(numpy.array(kw['x_train']).T),
        x_test=pl.DataFrame(numpy.array(kw['x_test']).T),
        w=None if kw.get('w') is None else pl.Series(numpy.array(kw['w'])),
    )
    bart = mc_gbart(**kw)
    with pytest.raises(ValueError, match='format mismatch'):
        bart.predict(kw['x_test'].to_numpy().T)


def test_automatic_integer_types(kw: dict[str, Any]) -> None:
    """Test that integer variables in the MCMC state have the correct type.

    Some integer variables change type automatically to be as small as possible.
    """
    bart = mc_gbart(**kw)

    def select_type(cond: bool) -> type:
        return jnp.uint8 if cond else jnp.uint16

    leaf_indices_type = select_type(kw.get('bart_kwargs', {}).get('maxdepth', 6) <= 8)
    split_trees_type = X_type = select_type(kw['numcut'] <= 255)
    var_trees_type = select_type(kw['x_train'].shape[0] <= 256)

    assert bart._mcmc_state.forest.var_tree.dtype == var_trees_type
    assert bart._mcmc_state.forest.split_tree.dtype == split_trees_type
    assert bart._mcmc_state.forest.leaf_indices.dtype == leaf_indices_type
    assert bart._mcmc_state.X.dtype == X_type
    assert bart._mcmc_state.forest.max_split.dtype == split_trees_type


def test_gbart_multichain_error(keys: split) -> None:
    """Check that `bartz.BART.gbart` does not support `mc_cores`."""
    X = gen_X(keys.pop(), 10, 100, 'continuous')
    y = gen_y(keys.pop(), X, None, 'continuous')
    with pytest.raises(TypeError, match=r'mc_cores'):
        gbart(X, y, mc_cores=1)
    with pytest.raises(TypeError, match=r'mc_cores'):
        gbart(X, y, mc_cores=2)
    with pytest.raises(TypeError, match=r'mc_cores'):
        gbart(X, y, mc_cores='gatto')


def get_expect_sharded(kw: dict) -> bool:
    """Check whether we expect sharding to be set up based on the arguments."""
    bart_kwargs = kw.get('bart_kwargs', {})
    num_chain_devices = bart_kwargs.get('num_chain_devices')
    num_data_devices = bart_kwargs.get('num_data_devices')
    return (
        num_chain_devices is not None
        or num_data_devices is not None
        or (
            kw.get('mc_cores', 2) > 1
            and get_device_count() > 1
            and get_default_device().platform == 'cpu'
            and 'num_chain_devices' not in bart_kwargs
        )
    )


def check_data_sharding(x: Array | None, mesh: Mesh) -> None:
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


def check_chain_sharding(x: Array | None, mesh: Mesh) -> None:
    """Check the sharding of `x` assuming it may be sharded only along the first 'chains' axis."""
    if x is None:
        return
    elif mesh is None:
        assert isinstance(x.sharding, SingleDeviceSharding)
    elif 'chains' in mesh.axis_names:
        expected_num_devices = min(2, get_device_count())
        assert x.sharding.num_devices == expected_num_devices
        assert get_normal_spec(x) == normalize_spec(('chains',), mesh, x.shape)


def test_sharding(kw: dict, variant: int) -> None:
    """Check that chains live on their own devices throughout the interface."""
    if version_info[:2] == get_old_python_tuple() and variant in (2, 5):
        pytest.xfail('Actual sharding bug in bartz with old jax, no time to fix.')
    bart = mc_gbart(**kw)

    # check the mesh is set up iff we expect sharding
    expect_sharded = get_expect_sharded(kw)
    mesh = bart._mcmc_state.config.mesh
    assert expect_sharded == (mesh is not None)

    check = partial(check_sharding, mesh=mesh)
    check(bart._mcmc_state)
    check(bart._burnin_trace)
    check(bart._main_trace)

    check_chain = partial(check_chain_sharding, mesh=mesh)

    check_chain(bart.yhat_test)
    check_chain(bart.prob_test)
    check_chain(bart.prob_train)
    if bart.sigma is not None:
        check_chain(bart.sigma.T)
    check_chain(bart.sigma_)
    check_chain(bart.varcount)
    check_chain(bart.varprob)
    check_chain(bart.yhat_train)

    check_data = partial(check_data_sharding, mesh=mesh)

    check_data(bart.prob_train)
    check_data(bart.prob_train_mean)
    check_data(bart.yhat_train)
    check_data(bart.yhat_train_mean)

    assert bart.offset.is_fully_replicated
    if bart.sigest is not None:
        assert bart.sigest.is_fully_replicated
    if bart.sigma_mean is not None:
        assert bart.sigma_mean.is_fully_replicated
    assert bart.varcount_mean.is_fully_replicated
    assert bart.varprob_mean.is_fully_replicated
    if bart.yhat_test_mean is not None:
        assert bart.yhat_test_mean.is_fully_replicated


class TestVarprobParam:
    """Test the `varprob` parameter."""

    def test_biased_predictor_choice(self, keys: split, kw: dict) -> None:
        """Check that if `varprob[i]` is high then predictor `i` is used more than others."""
        p, _ = kw['x_train'].shape
        i = random.randint(keys.pop(), (), 0, p)
        vp = jnp.full(p, 0.001).at[i].set(1)
        vp /= vp.sum()
        kw.update(sparse=False, varprob=vp)
        bart = mc_gbart(**kw)
        vc = bart.varcount_mean
        vc /= vc.sum()
        assert vc[i] > vp[i] * 0.6

    def test_positive(self, kw: dict, subtests: SubTests) -> None:
        """Check that an error is raised if varprob is not > 0."""
        p, _ = kw['x_train'].shape

        with subtests.test('not negative'):
            assert p > 1
            varprob = jnp.ones(p).at[0].set(-1.0)
            kw.update(varprob=varprob)
            with pytest.raises(EquinoxRuntimeError, match='varprob must be > 0'):
                mc_gbart(**kw)

        with subtests.test('not 0'):
            varprob = jnp.zeros(p).at[0].set(1.0)
            kw.update(varprob=varprob)
            with pytest.raises(EquinoxRuntimeError, match='varprob must be > 0'):
                mc_gbart(**kw)


@pytest.mark.slow
def test_equiv_sharding(kw: dict, subtests: SubTests) -> None:  # pragma: slow
    """Check that the result is the same with/without sharding."""
    if get_disable_problematic_sharding():  # pragma: no cover
        pytest.skip('Sharding disabled by --disable-problematic-sharding')
    if len(devices()) < 2:  # this branch is covered in the single cpu test config
        pytest.skip('Need at least 2 devices for this test')

    # baseline without sharding
    baseline_kw = tree.map(lambda x: x, kw)  # deep copy of structure
    baseline_kw.setdefault('bart_kwargs', {}).update(
        num_chain_devices=None, num_data_devices=None
    )
    baseline_kw.update(nskip=0, ndpost=20, mc_cores=2)
    bart = mc_gbart(**baseline_kw)

    def check_equal(path: KeyPath, xb: Array, xs: Array) -> None:
        assert_close_matrices(
            xs, xb, err_msg=f'{keystr(path)}: ', rtol=1e-5, reduce_rank=True
        )

    def remove_mesh(bart: mc_gbart) -> mc_gbart:
        config = bart._mcmc_state.config
        config = replace(config, mesh=None)
        return tree_at(lambda bart: bart._mcmc_state.config, bart, config)

    with subtests.test('shard chains'):
        chains_kw = tree.map(lambda x: x, baseline_kw)
        chains_kw.setdefault('bart_kwargs', {}).update(num_chain_devices=2)
        bart_chains = mc_gbart(**chains_kw)
        bart_chains = remove_mesh(bart_chains)
        tree.map_with_path(check_equal, bart, bart_chains)

    with subtests.test('shard data'):
        data_kw = tree.map(lambda x: x, baseline_kw)
        data_kw.setdefault('bart_kwargs', {}).update(num_data_devices=2)
        bart_data = mc_gbart(**data_kw)
        bart_data = remove_mesh(bart_data)
        tree.map_with_path(check_equal, bart, bart_data)

    if len(devices()) >= 4:  # pragma: no branch
        with subtests.test('shard data and chains'):
            both_kw = tree.map(lambda x: x, baseline_kw)
            both_kw.setdefault('bart_kwargs', {}).update(
                num_chain_devices=2, num_data_devices=2
            )
            bart_both = mc_gbart(**both_kw)
            bart_both = remove_mesh(bart_both)
            tree.map_with_path(check_equal, bart, bart_both)


def test_num_trees(kw: dict, subtests: SubTests) -> None:
    """Test the number of trees."""
    kw.update(nskip=0, ndpost=0)

    with subtests.test('given ntree'):
        bart = mc_gbart(**kw)
        assert bart._bart.num_trees == kw['ntree']

    with subtests.test('default ntree'):
        if kw['type'] == 'pbart':
            default_ntree = 50
        else:
            default_ntree = 200
        kw.pop('ntree')
        bart = mc_gbart(**kw)
        assert bart._bart.num_trees == default_ntree
