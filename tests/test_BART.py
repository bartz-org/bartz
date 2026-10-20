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


"""Test `bartz.BART`, the wrapper mimicking the R package BART3.

`Bart` itself is tested in `test_interface.py`, so these tests target the
wrapper: the translation of the arguments, defaults, and outputs.
"""

from functools import partial
from typing import Any, NamedTuple

import jax
import numpy
import polars as pl
import pytest
from jax import jit, random, tree
from jax import numpy as jnp
from jax.scipy.special import ndtr
from jaxtyping import Array, Float32, Key, Shaped
from scipy import stats

from bartz import Bart, SparseConfig
from bartz._jaxext import is_key, split
from bartz._typing import kwdict
from bartz.BART import gbart as original_gbart
from bartz.BART import mc_gbart as original_mc_gbart
from bartz.prepcovars import GivenSplitsBinner, RangeEvenBinner, UniqueQuantileBinner
from bartz.testing import gen_data
from tests.test_interface import GEN_KW, assert_identical_bart
from tests.util import (
    assert_allclose,
    assert_array_equal,
    assert_close_matrices,
    condf,
    nnone,
    rerun_on_gpu,
)


class mc_gbart(original_mc_gbart):
    """Wrapper that enables debug checks by default."""

    def __init__(
        self,
        *args: Any,
        check_trees: bool = True,
        check_replicated_trees: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if check_trees:
            self._bart._check_trees(error=True)
        if check_replicated_trees:
            self._bart._check_replicated_trees()


class gbart(mc_gbart, original_gbart):
    """Wrapper of `gbart` that enables debug checks."""

    # passthrough __init__: without it, ty synthesizes a dataclass __init__
    # from the Module fields instead of inheriting the wrappers' ones
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


# common sizes, so that jax reuses compiled code across tests
N_TRAIN = 30
N_TEST = 10
P = 3
NSKIP = 20
NDPOST = 41  # odd, to exercise the rounding up to a multiple of `mc_cores`


class Case(NamedTuple):
    """An `mc_gbart` configuration and the `Bart` call it should translate to."""

    cls: type[mc_gbart]
    gkw: dict[str, Any]
    """Arguments to `cls`."""
    bkw: dict[str, Any]
    """Arguments to `Bart` equivalent to `gkw`."""
    x_test: Shaped[Array, 'p m']
    """The test predictors as array in the layout of `Bart`."""
    sigest: float | None
    """Expected value of `mc_gbart.sigest`."""


def bart3_sigma_kw(
    sigest: float, sigdf: float = 3.0, sigquant: float = 0.9
) -> dict[str, float]:
    """Return the `Bart` error prior equivalent to R BART3's settings."""
    # same formula as in R BART3
    lambda_ = sigest**2 * stats.chi2.ppf(1 - sigquant, sigdf) / sigdf
    scale = numpy.sqrt(lambda_)
    return dict(sigma_df=sigdf, sigma_scale=scale, sigma_init=scale)


def ols_sigest(x: Shaped[Array, 'p n'], y: Shaped[Array, ' n']) -> float:
    """Estimate the error sdev by OLS with intercept, like `summary(lm(y ~ x))$sigma` in R."""
    _, n = x.shape
    design = numpy.column_stack([numpy.ones(n), numpy.asarray(x, numpy.float64).T])
    response = numpy.asarray(y, numpy.float64)
    coef, _, rank, _ = numpy.linalg.lstsq(design, response)
    rss = numpy.sum(numpy.square(response - design @ coef))
    return numpy.sqrt(rss / (n - rank)).item()


# the `Bart` equivalent of `mc_gbart(x_train, y_train)`, apart from the data
# and the error prior, which depends on the data
BART3_DEFAULTS: kwdict = dict(
    outcome_type='continuous',
    sparse=SparseConfig(enabled=False, augment=False),
    varprob=None,
    binner=partial(RangeEvenBinner, max_bins=101),
    rm_const=True,
    k=2.0,
    power=2.0,
    base=0.95,
    tau_num=None,
    offset=None,
    error_scale=None,
    num_trees=200,
    n_save=500,
    n_burn=100,
    n_skip=1,
    printevery=100,
    num_chains=2,
    seed=0,
    maxdepth=6,
    precompute_predict_train=True,
    init_kw=dict(min_points_per_leaf=5),
)


def make_case(key: Key[Array, ''], name: str) -> Case:
    """Return the configuration `name`."""
    keys = split(key)
    match name:
        # continuous regression, data as arrays, mostly defaults
        case 'wbart':
            train, test = gen_data(keys.pop(), n=N_TRAIN + N_TEST, p=P, **GEN_KW).split(
                N_TRAIN
            )
            sigest = ols_sigest(train.x, train.y)
            seed = keys.pop()
            return Case(
                cls=mc_gbart,
                gkw=dict(
                    x_train=train.x.T,
                    y_train=train.y,
                    x_test=test.x.T,
                    ndpost=NDPOST,
                    nskip=NSKIP,
                    printevery=None,
                    seed=seed,
                ),
                bkw=dict(
                    BART3_DEFAULTS,
                    x_train=train.x,
                    y_train=train.y,
                    **bart3_sigma_kw(sigest),
                    n_save=21,
                    n_burn=NSKIP,
                    printevery=None,
                    seed=random.clone(seed),
                ),
                x_test=test.x,
                sigest=sigest,
            )

        # binary regression with `gbart`, quantile binning, sparsity
        case 'pbart':
            train, test = gen_data(
                keys.pop(), n=N_TRAIN + N_TEST, p=P, outcome_type='binary', **GEN_KW
            ).split(N_TRAIN)
            varprob = jnp.array([0.2, 0.3, 0.5])
            seed = keys.pop()
            return Case(
                cls=gbart,
                gkw=dict(
                    x_train=train.x.T,
                    y_train=train.y,
                    x_test=test.x.T,
                    type='pbart',
                    sparse=True,
                    theta=2.0,
                    varprob=varprob,
                    usequants=True,
                    numcut=10,
                    ndpost=NDPOST,
                    nskip=NSKIP,
                    printevery=None,
                    seed=seed,
                ),
                bkw=dict(
                    BART3_DEFAULTS,
                    x_train=train.x,
                    y_train=train.y,
                    # type-dependent defaults
                    outcome_type='binary',
                    num_trees=50,
                    n_skip=10,
                    # gbart is single-chain
                    num_chains=None,
                    sparse=SparseConfig(theta=2.0, augment=False),
                    varprob=varprob,
                    binner=partial(
                        UniqueQuantileBinner, max_bins=11, max_subsample=None
                    ),
                    n_save=NDPOST,
                    n_burn=NSKIP,
                    printevery=None,
                    seed=random.clone(seed),
                ),
                x_test=test.x,
                sigest=None,
            )

        # continuous regression, data as dataframes, most options set
        case 'custom':  # pragma: no branch
            train, test = gen_data(
                keys.pop(),
                n=N_TRAIN + N_TEST,
                p=P,
                het_strength=0.5,
                het_shape='scalar',
                **GEN_KW,
            ).split(N_TRAIN)
            nan = numpy.nan
            xinfo = numpy.array(
                [[-1.0, 0.0, 1.0], [0.0, nan, nan], [-0.5, 0.5, nan]], numpy.float32
            )
            bart_kwargs = dict(maxdepth=5, init_kw=dict(min_points_per_leaf=3))
            seed = keys.pop()
            return Case(
                cls=mc_gbart,
                gkw=dict(
                    x_train=pl.DataFrame(numpy.asarray(train.x.T)),
                    y_train=pl.Series(numpy.asarray(train.y)),
                    x_test=pl.DataFrame(numpy.asarray(test.x.T)),
                    sparse=True,
                    a=1.0,
                    b=2.0,
                    rho=2.0,
                    augment=True,
                    xinfo=xinfo,
                    rm_const=False,
                    sigest=0.5,
                    sigdf=5.0,
                    sigquant=0.75,
                    k=1.5,
                    power=1.5,
                    base=0.9,
                    tau_num=2.0,
                    offset=0.3,
                    w=pl.Series(numpy.asarray(train.error_scale)),
                    ntree=30,
                    ndpost=40,
                    nskip=NSKIP,
                    keepevery=2,
                    printevery=None,
                    mc_cores=3,
                    seed=seed,
                    bart_kwargs=bart_kwargs,
                ),
                bkw=dict(
                    BART3_DEFAULTS,
                    # arrays instead of dataframes, which `Bart` would record
                    # as the data format
                    x_train=train.x,
                    y_train=train.y,
                    sparse=SparseConfig(a=1.0, b=2.0, rho=2.0, augment=True),
                    binner=partial(GivenSplitsBinner, xinfo=jnp.asarray(xinfo)),
                    rm_const=False,
                    **bart3_sigma_kw(0.5, 5.0, 0.75),
                    k=1.5,
                    power=1.5,
                    base=0.9,
                    tau_num=2.0,
                    offset=0.3,
                    error_scale=train.error_scale,
                    num_trees=30,
                    n_save=14,
                    n_burn=NSKIP,
                    n_skip=2,
                    printevery=None,
                    num_chains=3,
                    seed=random.clone(seed),
                    **bart_kwargs,
                ),
                x_test=test.x,
                sigest=0.5,
            )

        case _:  # pragma: no cover
            msg = f'Unknown case {name}'
            raise ValueError(msg)


class Fit(NamedTuple):
    """A configuration fitted with `mc_gbart` and with the equivalent `Bart`."""

    case: Case
    bart: mc_gbart
    ref: Bart


CASE_NAMES = ('wbart', 'pbart', 'custom')


@pytest.fixture(scope='module', params=CASE_NAMES)
def fit(request: pytest.FixtureRequest) -> Fit:
    """Return a fitted configuration, shared by the tests that do not modify it."""
    # module-scoped, so the seed can't come from the `keys` fixture
    key = random.fold_in(random.key(0x2B6E1F07), CASE_NAMES.index(request.param))
    case = make_case(key, request.param)
    return Fit(case=case, bart=case.cls(**case.gkw), ref=Bart(**case.bkw))


@pytest.mark.flaky(max_runs=5, rerun_filter=rerun_on_gpu)
# see `assert_identical_bart`
def test_translation(fit: Fit) -> None:
    """Check that `mc_gbart` runs the same MCMC as the equivalent `Bart`."""
    assert_identical_bart(fit.bart._bart, fit.ref)


def to_numpy_array(x: object) -> object:
    """Convert jax arrays to numpy, leave anything else as is."""
    if isinstance(x, Array) and not is_key(x):
        return numpy.asarray(x)
    else:
        return x


@pytest.mark.flaky(max_runs=5, rerun_filter=rerun_on_gpu)
# see `assert_identical_bart`
def test_numpy_input(fit: Fit) -> None:
    """Check that numpy inputs give the same result as jax inputs."""
    gkw = tree.map(to_numpy_array, fit.case.gkw)
    gkw = dict(gkw, seed=random.clone(fit.case.gkw['seed']))
    bart = fit.case.cls(**gkw)
    assert_identical_bart(bart._bart, fit.bart._bart)
    assert_array_equal(bart.yhat_test, fit.bart.yhat_test)


def test_jit(fit: Fit) -> None:
    """Check that jitting around the whole wrapper works."""

    def to_array(x: object) -> object:
        if isinstance(x, pl.DataFrame | pl.Series):
            return jnp.asarray(x.to_numpy())
        else:
            return x

    kw: dict = {k: to_array(v) for k, v in fit.case.gkw.items()}
    bart_kwargs: dict = kw.get('bart_kwargs', {})
    platform = fit.bart.yhat_train.platform()
    kw.update(
        # counting splitless variables breaks tracing
        rm_const=False,
        # the debug checks break tracing
        check_trees=False,
        check_replicated_trees=False,
        # under jit the device can't be inferred from the arrays
        bart_kwargs=dict(bart_kwargs, devices=jax.devices(platform)),
    )
    traced = {
        k: kw.pop(k) for k in ('x_train', 'y_train', 'x_test', 'w', 'seed') if k in kw
    }

    def task(traced: dict[str, Any]) -> tuple[Array, Array | None, Array | None]:
        bart = fit.case.cls(**traced, **kw)
        return bart.yhat_train, bart.yhat_test, bart.sigest

    eager = task(traced)
    jitted = jit(task)(dict(traced, seed=random.clone(traced['seed'])))

    rtol = condf(fit.bart._mcmc_state.forest.leaf_tree, 1e-5, 1e-3)
    for e, j in zip(eager, jitted, strict=True):
        if e is None:
            assert j is None
        else:
            assert_close_matrices(j, e, rtol=rtol)


def test_sigest(fit: Fit) -> None:
    """Check `mc_gbart.sigest`."""
    if fit.case.sigest is None:
        assert fit.bart.sigest is None
    else:
        assert_allclose(fit.bart.sigest, fit.case.sigest, rtol=1e-5)


def test_lambda_explicit(fit: Fit) -> None:
    """An explicit `lambda_` sets the error prior and leaves `sigest` unset."""
    if fit.case.sigest is None:
        pytest.skip('sigma prior is fixed for binary regression')
    lambda_ = 0.7
    gkw: kwdict = dict(fit.case.gkw, lambda_=lambda_)
    gkw.pop('sigest', None)
    bart = fit.case.cls(**gkw)
    assert bart.sigest is None

    scale = numpy.sqrt(lambda_)
    ref = Bart(**dict(fit.case.bkw, sigma_scale=scale, sigma_init=scale))
    actual = bart._mcmc_state.error_cov_inv
    desired = ref._mcmc_state.error_cov_inv
    assert_array_equal(nnone(actual.nu), nnone(desired.nu))
    assert_allclose(nnone(actual.rate), nnone(desired.rate), rtol=1e-6)


def is_binary(case: Case) -> bool:
    """Whether the configuration is a binary regression."""
    return case.gkw.get('type') == 'pbart'


def get_mc_cores(case: Case) -> int:
    """Return the number of chains of the configuration."""
    return case.gkw.get('mc_cores', 1 if case.cls is gbart else 2)


def test_ndpost_offset(fit: Fit) -> None:
    """Check `ndpost` is rounded up to a multiple of `mc_cores`, and `offset`."""
    mc_cores = get_mc_cores(fit.case)
    ndpost = mc_cores * -(-fit.case.gkw['ndpost'] // mc_cores)
    assert fit.bart.ndpost == ndpost
    assert fit.bart._bart.ndpost == ndpost
    assert_array_equal(fit.bart.offset, fit.bart._bart.offset)


def test_yhat(fit: Fit) -> None:
    """Check the `yhat_*` attributes and `predict`."""
    bart = fit.bart
    inner = bart._bart

    yhat_train = inner.predict('train', kind='latent_samples')
    assert yhat_train.shape == (bart.ndpost, N_TRAIN)
    assert_array_equal(bart.yhat_train, yhat_train)

    yhat_test = inner.predict(fit.case.x_test, kind='latent_samples')
    assert yhat_test.shape == (bart.ndpost, N_TEST)
    assert_array_equal(bart.yhat_test, yhat_test)
    assert_array_equal(bart.predict(fit.case.gkw['x_test']), yhat_test)

    if is_binary(fit.case):
        assert bart.yhat_train_mean is None
        assert bart.yhat_test_mean is None
    else:
        assert_close_matrices(bart.yhat_train_mean, yhat_train.mean(0), rtol=1e-6)
        assert_close_matrices(bart.yhat_test_mean, yhat_test.mean(0), rtol=1e-6)


def test_prob(fit: Fit) -> None:
    """Check the `prob_*` attributes."""
    bart = fit.bart
    if is_binary(fit.case):
        prob_train = ndtr(bart.yhat_train)
        prob_test = ndtr(nnone(bart.yhat_test))
        assert_close_matrices(bart.prob_train, prob_train, rtol=1e-6)
        assert_close_matrices(bart.prob_test, prob_test, rtol=1e-6)
        assert_close_matrices(bart.prob_train_mean, prob_train.mean(0), rtol=1e-6)
        assert_close_matrices(bart.prob_test_mean, prob_test.mean(0), rtol=1e-6)
    else:
        assert bart.prob_train is None
        assert bart.prob_test is None
        assert bart.prob_train_mean is None
        assert bart.prob_test_mean is None


def trace_shape(fit: Fit) -> tuple[int, ...]:
    """Shape of the attributes with burn-in: (samples, chains), or (samples,)."""
    mc_cores = get_mc_cores(fit.case)
    num_samples = fit.case.gkw['nskip'] + fit.bart.ndpost // mc_cores
    return (num_samples,) if mc_cores == 1 else (num_samples, mc_cores)


def test_sigma(fit: Fit) -> None:
    """Check the `sigma*` attributes."""
    bart = fit.bart
    if is_binary(fit.case):
        assert bart.sigma is None
        assert bart.sigma_ is None
        assert bart.sigma_mean is None
    else:
        sigma = nnone(bart.sigma)
        assert sigma.shape == trace_shape(fit)
        assert sigma.dtype == jnp.float32

        sigma_ = bart._bart.get_error_sdev()
        assert_close_matrices(bart.sigma_, sigma_, rtol=1e-6)
        assert_allclose(bart.sigma_mean, sigma_.mean(), rtol=1e-6)

        # `sigma` adds the burn-in and uses the (samples, chains) layout, while
        # `sigma_` concatenates the chains
        nskip = fit.case.gkw['nskip']
        num_samples, *_ = sigma.shape
        main = sigma[nskip:, ...].reshape(num_samples - nskip, -1)
        assert_array_equal(main.T.reshape(-1), nnone(bart.sigma_))


def test_accept(fit: Fit) -> None:
    """Check the `accept` attribute."""
    bart = fit.bart
    assert bart.accept.shape == trace_shape(fit)
    assert_array_equal(bart.accept, bart._bart.accept.T)


def test_varcount_varprob(fit: Fit) -> None:
    """Check the `varcount*` and `varprob*` attributes."""
    bart = fit.bart
    inner = bart._bart
    assert bart.varcount.shape == (bart.ndpost, P)
    assert bart.varcount.dtype == jnp.int32
    assert_array_equal(bart.varcount, inner.varcount)
    assert_array_equal(bart.varcount_mean, inner.varcount_mean)
    assert bart.varprob.shape == (bart.ndpost, P)
    assert_array_equal(bart.varprob, inner.varprob)
    assert_array_equal(bart.varprob_mean, inner.varprob_mean)


def test_data_format_mismatch(fit: Fit) -> None:
    """Check `predict` rejects predictors in a format different from `x_train`."""
    x_test = fit.case.gkw['x_test']
    if isinstance(x_test, pl.DataFrame):
        x_test = x_test.to_numpy()
    else:
        x_test = pl.DataFrame(numpy.asarray(x_test))
    with pytest.raises(ValueError, match='format mismatch'):
        fit.bart.predict(x_test)


@pytest.fixture
def small_data(keys: split) -> tuple[Shaped[Array, 'n p'], Shaped[Array, ' n']]:
    """Return predictors in R layout and a continuous response."""
    dgp = gen_data(keys.pop(), n=10, p=P, **GEN_KW)
    return dgp.x.T, dgp.y


def binarize(y: Float32[Array, ' n']) -> Float32[Array, ' n']:
    """Convert a continuous response to binary."""
    return (y > 0).astype(jnp.float32)


def test_w_with_binary_raises(small_data: tuple[Array, Array]) -> None:
    """Reject `w` with binary regression (heteroskedastic probit)."""
    x, y = small_data
    with pytest.raises(ValueError, match='not supported with binary regression'):
        mc_gbart(x, binarize(y), type='pbart', w=jnp.ones_like(y))


@pytest.mark.parametrize('param', ['sigest', 'lambda_'])
def test_binary_rejects_sigma_settings(
    small_data: tuple[Array, Array], param: str
) -> None:
    """Binary regression rejects `sigest`/`lambda_` (they are ignored)."""
    x, y = small_data
    kw: kwdict = {param: 1.0}
    with pytest.raises(ValueError, match='Do not set `sigest` or `lambda_` for binary'):
        mc_gbart(x, binarize(y), type='pbart', **kw)


def test_sigest_and_lambda_raises(small_data: tuple[Array, Array]) -> None:
    """`sigest` is ignored if `lambda_` is set, so setting both is rejected."""
    x, y = small_data
    with pytest.raises(ValueError, match='Do not set `sigest` if `lambda_`'):
        mc_gbart(x, y, sigest=1.0, lambda_=1.0)


def test_sigest_ols_needs_more_data(small_data: tuple[Array, Array]) -> None:
    """Estimating `sigest` by OLS requires more datapoints than parameters."""
    x, y = small_data
    n = P + 1  # one parameter per predictor, plus the intercept
    with pytest.raises(ValueError, match='cannot estimate `sigest` by OLS'):
        mc_gbart(x[:n, :], y[:n])


def test_gbart_rejects_mc_cores(small_data: tuple[Array, Array]) -> None:
    """Check that `bartz.BART.gbart` does not support `mc_cores`."""
    x, y = small_data
    for mc_cores in (1, 2, 'gatto'):
        with pytest.raises(TypeError, match=r'mc_cores'):
            gbart(x, y, mc_cores=mc_cores)
