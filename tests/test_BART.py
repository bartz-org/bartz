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

from dataclasses import dataclass
from functools import partial
from inspect import signature
from typing import Any, NamedTuple

import jax
import numpy
import polars as pl
import pytest
from jax import jit, random, tree
from jax import numpy as jnp
from jax.scipy.special import logit, ndtr
from jaxtyping import Array, Float32, Key, Shaped
from numpy.testing import assert_array_less
from pytest_subtests import SubTests
from scipy import stats

from bartz import Bart, SparseConfig
from bartz._jaxext import is_key, split
from bartz._typing import kwdict
from bartz.BART import gbart as original_gbart
from bartz.BART import mc_gbart as original_mc_gbart
from bartz.debug import trees_BART_to_bartz
from bartz.grove import check_trace
from bartz.mcmcloop import compute_varcount, evaluate_trace
from bartz.prepcovars import (
    BinnerFactory,
    GivenSplitsBinner,
    RangeEvenBinner,
    UniqueQuantileBinner,
)
from bartz.testing import gen_data
from tests.test_interface import GEN_KW, BartKW, assert_identical_bart, make_kw
from tests.util import (
    assert_allclose,
    assert_array_equal,
    assert_close_matrices,
    clipped_logit,
    condf,
    int_seed,
    nnone,
    rerun_on_gpu,
    rhat_rank,
)

try:
    from rbartpackages import BART3
except ValueError as exc:
    # allow collection in environments without R installed
    if 'r_home' not in str(exc):
        raise


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


# --- wrapper tests ---

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


# --- comparison with R BART3 ---


def get_with_default(kw: dict, param_name: str) -> Any:  # noqa: ANN401
    """Do `kw.get(param_name, <default in mc_gbart>)`."""
    sig = signature(original_mc_gbart)
    param = sig.parameters[param_name]
    if param.default is param.empty:
        return kw[param_name]
    else:
        return kw.get(param_name, param.default)


def bart_kw_to_mc_gbart(bkw: BartKW) -> dict[str, Any]:
    """Convert `Bart` keyword arguments to `mc_gbart` keyword arguments."""
    kw = dict(bkw.kw)

    def pop(param_name: str) -> Any:  # noqa: ANN401
        """Remove `param_name` from kw, return the default value of the parameter in `Bart` if missing."""
        sig = signature(Bart)
        param = sig.parameters[param_name]
        if param.default is param.empty:
            return kw.pop(param_name)
        else:
            return kw.pop(param_name, param.default)

    def push(param_name: str, value: object) -> None:
        """Set `kw[param_name] = value`, unless value is equal to default value in `mc_gbart`."""
        sig = signature(original_mc_gbart)
        param = sig.parameters[param_name]
        if param.default == value:
            kw.pop(param_name, None)
        else:
            kw[param_name] = value

    # outcome_type -> type
    outcome_type = pop('outcome_type')
    push('type', 'pbart' if outcome_type == 'binary' else 'wbart')

    # error_scale -> w
    push('w', pop('error_scale'))

    # num_trees -> ntree
    push('ntree', pop('num_trees'))

    # num_chains -> mc_cores
    num_chains = pop('num_chains')
    assert num_chains != 1  # because mc_gbart does not have an equivalent option
    mc_cores = 1 if num_chains is None else num_chains
    push('mc_cores', mc_cores)

    # n_save (per-chain) -> ndpost (across-chains total)
    push('ndpost', pop('n_save') * mc_cores)

    # n_burn -> nskip, n_skip -> keepevery
    push('nskip', pop('n_burn'))
    push('keepevery', pop('n_skip'))

    # sparse: expand Bart's SparseConfig into mc_gbart's flat prior params.
    # augment is deliberately not forwarded to the comparison: bartz's augment
    # (exact full conditional) and BART3's augment (approximate expected counts)
    # are different algorithms; worse, BART3 folds its augmentation pseudo-counts
    # into the reported varcount, so it stops matching the trees (which breaks
    # the `check_rbart` self-consistency check). So compare on the shared,
    # un-augmented baseline; augmentation is exercised directly elsewhere
    # (test_variable_selection, and the sample_s_augmentation tests).
    sparse = pop('sparse')
    push('sparse', sparse.enabled)
    push('theta', sparse.theta)
    push('a', sparse.a)
    push('b', sparse.b)
    push('rho', sparse.rho)

    # binner -> xinfo, usequants, numcut
    binner = pop('binner')
    binargs = convert_binner(binner)
    for k, v in binargs.items():
        push(k, v)

    # collect bart_kwargs from top-level Bart params
    bart_kwargs: dict[str, Any] = {}

    # move Bart-only keys that mc_gbart does not change the default of
    for key in ('num_chain_devices', 'num_data_devices', 'maxdepth', 'sigma_mu_df'):
        if key in kw:
            bart_kwargs[key] = kw.pop(key)

    # precompute_predict_train: mc_gbart defaults it on, Bart off
    precompute_predict_train = pop('precompute_predict_train')
    if precompute_predict_train is not True:
        bart_kwargs['precompute_predict_train'] = precompute_predict_train

    # init_kw must be present bc it contains min_points_per_leaf which has
    # different defaults
    init_kw = dict(kw.pop('init_kw'))
    # min_points_per_leaf: remove if equal to mc_gbart default (5) so mc_gbart's
    # default-setting code is tested. min_points_per_decision_node is passed
    # through: both Bart and mc_gbart default it to 10.
    if init_kw['min_points_per_leaf'] == 5:
        del init_kw['min_points_per_leaf']
    bart_kwargs['init_kw'] = init_kw
    kw['bart_kwargs'] = bart_kwargs

    # re-add x_test; mc_gbart follows BART3's (n, p) array layout, so transpose
    # the (p, n) predictor matrices used internally by Bart
    kw['x_train'] = kw['x_train'].T
    kw['x_test'] = bkw.x_test.T

    return kw


def convert_binner(binner: BinnerFactory) -> dict[str, Any]:
    """Convert the `binner` argument to `Bart` to the corresponding arguments for `mc_gbart`."""
    # standardize input as (subcls, defaults), where defaults contains the
    # constructor defaults of subcls with the partial keywords applied on top
    if isinstance(binner, partial):
        subcls = binner.func
        partial_kwargs = binner.keywords
    else:
        subcls = binner
        partial_kwargs = {}
    bound = signature(subcls).bind_partial(**partial_kwargs)
    bound.apply_defaults()
    defaults = dict(bound.arguments)

    # convert to mc_gbart parameters
    if subcls is GivenSplitsBinner:
        return {'xinfo': defaults['xinfo']}
    elif subcls is UniqueQuantileBinner:
        assert defaults['max_subsample'] is None
        return {'usequants': True, 'numcut': defaults['max_bins'] - 1}
    elif subcls is RangeEvenBinner:
        return {'usequants': False, 'numcut': defaults['max_bins'] - 1}
    else:  # pragma: no cover
        msg = f'Cannot convert binner of type {subcls!r}'
        raise NotImplementedError(msg)


def make_gbart_kw(key: Key[Array, ''], variant: int) -> dict[str, Any]:
    """Return a dictionary of keyword arguments for `mc_gbart`."""
    return bart_kw_to_mc_gbart(make_kw(key, variant))


@pytest.fixture(
    params=[
        pytest.param(1, id='v1'),
        pytest.param(2, id='v2'),
        pytest.param(3, id='v3'),
    ],
    scope='module',
)
def variant(request: pytest.FixtureRequest) -> int:
    """Return a parametrized indicator to select different BART configurations."""
    return request.param


@dataclass(frozen=True)
class CachedBart:
    """Pre-computed BART run shared between multiple tests that do not change the arguments."""

    kwargs: dict[str, Any]
    bart: mc_gbart


class TestWithCachedBart:
    """Group of slow tests that check the same BART run, for efficiency."""

    @pytest.fixture(scope='class')
    def cachedbart(self, variant: int) -> CachedBart:
        """Return a pre-computed BART."""
        # create a random seed that depends only on the variant, since this
        # fixture is shared between multiple tests
        key = random.key(0x139CD0C0)
        keys = random.split(key, 10)  # 10 is just some high number
        key = keys[variant]
        kw = make_gbart_kw(key, variant)

        # modify configs to make them appropriate for convergence checks and R
        n, p = kw['x_train'].shape
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
        # R BART hard-codes nl>=5 && nr>=5; force the matching bartz constraint.
        # min_points_per_decision_node=10 is bartz's efficient proposal (skip
        # leaves too small to split), which targets the same posterior as BART3.
        kw.setdefault('bart_kwargs', {}).setdefault('init_kw', {}).update(
            min_points_per_decision_node=10, min_points_per_leaf=5
        )

        bart = mc_gbart(**kw)

        return CachedBart(kwargs=kw, bart=bart)

    def kw_bartz_to_BART3(self, key: Key[Array, ''], kw: dict, bart: mc_gbart) -> dict:
        """Convert bartz keyword arguments to R BART3 keyword arguments."""
        kw_BART: dict = dict(**kw, rm_const=False)
        kw_BART.pop('bart_kwargs')
        kw_BART.pop('maxdepth', None)
        for arg in 'w', 'printevery':
            if arg in kw_BART and kw_BART[arg] is None:
                kw_BART.pop(arg)
        kw_BART['seed'] = int_seed(key)

        # Set BART cutpoints manually. This means I am not checking that the
        # automatic cutpoint determination of BART is the same of my package. They
        # are similar but have some differences, and having exactly the same
        # cutpoints is more important for the test.
        # transposed=True disables predictors pre-processing, so BART3 expects
        # the (p, n) layout; mc_gbart uses (n, p), so transpose back
        kw_BART['transposed'] = True
        kw_BART['x_train'] = kw['x_train'].T
        kw_BART['x_test'] = kw['x_test'].T
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

        # check yhat_test; the binner uses the (p, m) layout
        Xt = bart._bart._binner.bin(kw['x_test'].T)
        yhat_test = evaluate_trace(Xt, trace)
        assert_close_matrices(
            yhat_test, nnone(rbart.yhat_test).astype(numpy.float32), rtol=1e-6
        )

        if get_with_default(kw, 'type') == 'pbart':
            # check prob_train
            prob_train = ndtr(yhat_train)
            assert_close_matrices(
                prob_train, nnone(rbart.prob_train).astype(numpy.float32), rtol=1e-7
            )

            # check prob_test
            prob_test = ndtr(yhat_test)
            assert_close_matrices(
                prob_test, nnone(rbart.prob_test).astype(numpy.float32), rtol=1e-7
            )

    def test_comparison_BART3(
        self, cachedbart: CachedBart, keys: split, subtests: SubTests
    ) -> None:
        """Check `bartz.BART` gives results similar to the R package BART3."""
        bart = cachedbart.bart
        kw = cachedbart.kwargs
        n, p = kw['x_train'].shape

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
            rhat_yhat_train = rhat_rank(
                jnp.stack([bart.yhat_train, rbart.yhat_train]), split=False
            )
            assert_array_less(rhat_yhat_train, 1.08)

        with subtests.test('yhat_test'):
            rhat_yhat_test = rhat_rank(
                jnp.stack([nnone(bart.yhat_test), nnone(rbart.yhat_test)]), split=False
            )
            assert_array_less(rhat_yhat_test, 1.08)

        if get_with_default(kw, 'type') == 'pbart':  # binary regression
            with subtests.test('prob_train'):
                rhat_prob_train = rhat_rank(
                    clipped_logit(
                        jnp.stack([nnone(bart.prob_train), nnone(rbart.prob_train)]),
                        1e-5,
                    ),
                    split=False,
                )
                assert_array_less(rhat_prob_train, 1.005)

            with subtests.test('prob_test'):
                rhat_prob_test = rhat_rank(
                    clipped_logit(
                        jnp.stack([nnone(bart.prob_test), nnone(rbart.prob_test)]), 1e-5
                    ),
                    split=False,
                )
                assert_array_less(rhat_prob_test, 1.005)

        else:  # continuous regression
            with subtests.test('yhat_train_mean'):
                assert_close_matrices(
                    nnone(bart.yhat_train_mean)
                    - nnone(rbart.yhat_train_mean).astype(numpy.float32),
                    jnp.concatenate([bart.yhat_train, rbart.yhat_train]).std(axis=0),
                    tozero=True,
                    rtol=0.4,
                )

            with subtests.test('yhat_test_mean'):
                assert_close_matrices(
                    nnone(bart.yhat_test_mean)
                    - nnone(rbart.yhat_test_mean).astype(numpy.float32),
                    jnp.concatenate(
                        [nnone(bart.yhat_test), nnone(rbart.yhat_test)]
                    ).std(axis=0),
                    tozero=True,
                    rtol=0.4,
                )

            with subtests.test('sigma'):
                rhat_sigma = rhat_rank(
                    [
                        nnone(bart.sigma_)[-bart.ndpost :],
                        nnone(rbart.sigma_)[-rbart.ndpost :],
                    ],
                    split=False,
                )
                assert_array_less(rhat_sigma, 1.06)

            with subtests.test('sigma_mean'):
                assert_allclose(
                    nnone(bart.sigma_mean), nnone(rbart.sigma_mean), rtol=0.1
                )

        with subtests.test('tree_node_count'):
            # check number of tree nodes in forest
            bart_count = bart.varcount.sum(axis=1)
            rbart_count = rbart.varcount.sum(axis=1)
            rhat_count = rhat_rank([bart_count, rbart_count], split=False)
            # threshold is set to accommodate the high-p variant where MCMC
            # noise dominates; low-p variants give rhat ~ 1.05.
            assert_array_less(rhat_count, 2.3)
            assert_allclose(bart_count.mean(), rbart_count.mean(), rtol=0.1)

        if p < n:
            # skip if p is large because it would be difficult for the MCMC to get
            # stuff about predictors right

            with subtests.test('varcount'):
                rhat_varcount = rhat_rank([bart.varcount, rbart.varcount], split=False)
                assert_array_less(rhat_varcount, 1.25)

            with subtests.test('varcount_mean'):
                assert_close_matrices(
                    bart.varcount_mean,
                    rbart.varcount_mean.astype(numpy.float32),
                    rtol=0.7,
                    atol=6.3,
                )

            if kw.get('sparse', False):  # pragma: no branch
                with subtests.test('varprob'):
                    rhat_varprob = rhat_rank(
                        clipped_logit(
                            jnp.stack([bart.varprob, rbart.varprob])[:, :, 1:], 1e-5
                        ),
                        split=False,
                    )
                    # drop one component because varprob sums to 1
                    assert_array_less(rhat_varprob, 1.3)

                with subtests.test('varprob_mean'):
                    assert_close_matrices(
                        logit(bart.varprob_mean[1:]),
                        logit(rbart.varprob_mean[1:]),
                        atol=2.1,
                    )
