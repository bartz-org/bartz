# bartz/tests/test_stochtree.py
#
# Copyright (c) 2026, The Bartz Contributors
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

"""Test `bartz.stochtree`, the shim mimicking the stochtree Python package."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

import jax
import numpy as np
import pandas as pd
import polars as pl
import pytest
from equinox import EquinoxRuntimeError
from jax import Array, random
from jax import numpy as jnp
from jax.scipy.special import ndtr
from numpy.testing import assert_array_less
from numpy.typing import ArrayLike
from pytest_subtests import SubTests

import bartz.stochtree as bst
import stochtree
from bartz._jaxext import split
from bartz.stochtree._preprocess import (
    PandasPreprocessor,
    PolarsPreprocessor,
    _PreprocessorBase,
)
from bartz.testing import gen_data
from tests.util import (
    assert_allclose,
    assert_array_equal,
    assert_close_matrices,
    clipped_logit,
    rhat_rank,
)

_GEN_KW = dict(p=4, q=2, sigma2_lin=0.5, sigma2_quad=0.5, sigma2_eps=0.5)
# `mean_forest_params` override that disables the bartz-incompatible
# leaf-variance sampler. All sample()-using tests start from this base.
_MFP_BASE: Mapping = MappingProxyType({'sample_sigma2_leaf': False})


class _Data(NamedTuple):
    X_train: Array  # (n, p)
    y_train: Array  # (n,) for continuous, (n,) int for binary
    X_test: Array  # (m, p)


def _make_continuous(keys: split, n: int = 200, n_test: int = 60) -> _Data:
    train = gen_data(keys.pop(), n=n, outcome_type='continuous', **_GEN_KW)
    test = gen_data(keys.pop(), n=n_test, outcome_type='continuous', **_GEN_KW)
    return _Data(X_train=train.x.T, y_train=train.y, X_test=test.x.T)


def _make_binary(keys: split, n: int = 200, n_test: int = 60) -> _Data:
    train = gen_data(keys.pop(), n=n, outcome_type='binary', **_GEN_KW)
    test = gen_data(keys.pop(), n=n_test, outcome_type='binary', **_GEN_KW)
    return _Data(X_train=train.x.T, y_train=train.y.astype(jnp.int32), X_test=test.x.T)


def _rhat_two_chains(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute rank-normalized Rhat between two ``(n, num_samples)`` matrices.

    Treats each of the two arrays as one MCMC chain of ``num_samples`` draws
    for each of the ``n`` outputs.
    """
    stacked = np.stack(
        [np.asarray(a, dtype=float).T, np.asarray(b, dtype=float).T], axis=0
    )
    return rhat_rank(stacked, split=False)


def test_continuous_smoke(keys: split) -> None:
    """Sample a small continuous model and check output shapes."""
    data = _make_continuous(keys, n=80, n_test=20)
    m = bst.BARTModel()
    m.sample(
        X_train=data.X_train,
        y_train=data.y_train,
        X_test=data.X_test,
        num_gfr=0,
        num_burnin=30,
        num_mcmc=80,
        mean_forest_params=_MFP_BASE,
    )
    assert m.is_sampled()
    assert m.y_hat_train.shape == (data.X_train.shape[0], m.num_samples)
    assert m.y_hat_test.shape == (data.X_test.shape[0], m.num_samples)
    assert m.global_var_samples.shape == (m.num_samples,)
    assert m.outcome_model.outcome == 'continuous'


def test_predict_shapes(keys: split) -> None:
    """Check the various combinations of `type`, `terms`, and `scale`."""
    data = _make_continuous(keys, n=50, n_test=15)
    m = bst.BARTModel()
    m.sample(
        X_train=data.X_train,
        y_train=data.y_train,
        num_gfr=0,
        num_burnin=10,
        num_mcmc=30,
        mean_forest_params=_MFP_BASE,
    )
    assert m.predict(data.X_test, terms='y_hat').shape == (
        data.X_test.shape[0],
        m.num_samples,
    )
    assert m.predict(data.X_test, terms='y_hat', type='mean').shape == (
        data.X_test.shape[0],
    )
    full = m.predict(data.X_test, terms='all')
    assert isinstance(full, dict)
    assert full['y_hat'].shape == (data.X_test.shape[0], m.num_samples)


def test_binary_smoke(keys: split) -> None:
    """Sample a probit model and check probability-scale predictions."""
    data = _make_binary(keys, n=80, n_test=20)
    m = bst.BARTModel()
    m.sample(
        X_train=data.X_train,
        y_train=data.y_train,
        X_test=data.X_test,
        num_gfr=0,
        num_burnin=30,
        num_mcmc=80,
        general_params={
            'outcome_model': bst.OutcomeModel(outcome='binary', link='probit')
        },
        mean_forest_params=_MFP_BASE,
    )
    assert m.outcome_model.outcome == 'binary'
    prob = np.asarray(
        m.predict(data.X_test, terms='y_hat', type='mean', scale='probability')
    )
    assert prob.shape == (data.X_test.shape[0],)
    assert ((prob >= 0) & (prob <= 1)).all()
    # global_var_samples is fixed to 1 for probit
    assert np.allclose(np.asarray(m.global_var_samples), 1.0)


def test_multi_chain(keys: split) -> None:
    """`num_chains > 1` concatenates samples across chains."""
    data = _make_continuous(keys, n=50, n_test=10)
    num_chains = 3
    num_mcmc = 40
    m = bst.BARTModel()
    m.sample(
        X_train=data.X_train,
        y_train=data.y_train,
        num_gfr=0,
        num_burnin=20,
        num_mcmc=num_mcmc,
        general_params={'num_chains': num_chains},
        mean_forest_params=_MFP_BASE,
    )
    assert m.num_samples == num_chains * num_mcmc
    assert m.y_hat_train.shape[1] == num_chains * num_mcmc


def test_not_sampled_error(keys: split) -> None:
    """`predict` before `sample` raises `NotSampledError`."""
    data = _make_continuous(keys, n=10, n_test=5)
    m = bst.BARTModel()
    assert not m.is_sampled()
    with pytest.raises(bst.NotSampledError):
        m.predict(data.X_test)


def test_missing_num_gfr_raises(keys: split) -> None:
    """`num_gfr` is keyword-only with no default; omitting it is a TypeError."""
    data = _make_continuous(keys, n=10, n_test=5)
    m = bst.BARTModel()
    with pytest.raises(TypeError, match='num_gfr'):
        m.sample(  # type: ignore[call-arg]
            X_train=data.X_train,
            y_train=data.y_train,
            num_burnin=2,
            num_mcmc=2,
            mean_forest_params=_MFP_BASE,
        )


def test_num_gfr_nonzero_raises(keys: split) -> None:
    """The grow-from-root sampler is not supported."""
    data = _make_continuous(keys, n=10, n_test=5)
    m = bst.BARTModel()
    with pytest.raises(NotImplementedError, match='grow-from-root'):
        m.sample(
            X_train=data.X_train,
            y_train=data.y_train,
            num_gfr=1,
            num_burnin=2,
            num_mcmc=2,
            mean_forest_params=_MFP_BASE,
        )


def test_sample_sigma2_leaf_true_raises(keys: split) -> None:
    """Stochtree's default of `sample_sigma2_leaf=True` must be explicitly disabled."""
    data = _make_continuous(keys, n=10, n_test=5)
    m = bst.BARTModel()
    with pytest.raises(NotImplementedError, match='sample_sigma2_leaf'):
        m.sample(
            X_train=data.X_train,
            y_train=data.y_train,
            num_gfr=0,
            num_burnin=2,
            num_mcmc=2,
        )


def test_sigma2_init_with_proper_prior_raises(keys: split) -> None:
    """`sigma2_init` is allowed only when the variance prior is improper."""
    data = _make_continuous(keys, n=10, n_test=5)
    m = bst.BARTModel()
    with pytest.raises(NotImplementedError, match='sigma2_init'):
        m.sample(
            X_train=data.X_train,
            y_train=data.y_train,
            num_gfr=0,
            num_burnin=2,
            num_mcmc=2,
            general_params={
                'sigma2_global_shape': 1.5,
                'sigma2_global_scale': 0.5,
                'sigma2_init': 1.0,
            },
            mean_forest_params=_MFP_BASE,
        )


def test_unknown_dict_keys_rejected(keys: split) -> None:
    """Unknown keys in `general_params` / `mean_forest_params` raise."""
    data = _make_continuous(keys, n=10, n_test=5)
    m = bst.BARTModel()
    with pytest.raises(ValueError, match='general_params contains unsupported key'):
        m.sample(
            X_train=data.X_train,
            y_train=data.y_train,
            num_gfr=0,
            num_burnin=2,
            num_mcmc=2,
            general_params={'bogus': 1},
            mean_forest_params=_MFP_BASE,
        )
    m = bst.BARTModel()
    with pytest.raises(ValueError, match='mean_forest_params contains unsupported key'):
        m.sample(
            X_train=data.X_train,
            y_train=data.y_train,
            num_gfr=0,
            num_burnin=2,
            num_mcmc=2,
            mean_forest_params={**_MFP_BASE, 'keep_vars': []},
        )


def test_unsupported_outcome_model_raises(keys: split) -> None:
    """Cloglog and other unsupported link functions are rejected."""
    data = _make_binary(keys, n=20, n_test=5)
    m = bst.BARTModel()
    with pytest.raises(NotImplementedError, match='unsupported outcome_model'):
        m.sample(
            X_train=data.X_train,
            y_train=data.y_train,
            num_gfr=0,
            num_burnin=2,
            num_mcmc=2,
            general_params={
                'outcome_model': bst.OutcomeModel(outcome='binary', link='cloglog')
            },
            mean_forest_params=_MFP_BASE,
        )


def test_variable_weights_validation(keys: split) -> None:
    """`variable_weights` must be strictly positive."""
    data = _make_continuous(keys, n=40, n_test=5)
    p = data.X_train.shape[1]
    m = bst.BARTModel()
    with pytest.raises(EquinoxRuntimeError, match='varprob must be > 0'):
        m.sample(
            X_train=data.X_train,
            y_train=data.y_train,
            num_gfr=0,
            num_burnin=2,
            num_mcmc=5,
            general_params={'variable_weights': np.zeros(p)},
            mean_forest_params=_MFP_BASE,
        )
    m2 = bst.BARTModel()
    m2.sample(
        X_train=data.X_train,
        y_train=data.y_train,
        num_gfr=0,
        num_burnin=2,
        num_mcmc=5,
        general_params={'variable_weights': np.full(p, 1e-3)},
        mean_forest_params=_MFP_BASE,
    )
    assert m2.is_sampled()


def test_standardization_matches(keys: split) -> None:
    """``y_bar`` / ``y_std`` are computed identically to stochtree (modulo float32)."""
    data = _make_continuous(keys, n=100, n_test=10)
    st_model = stochtree.BARTModel()
    st_model.sample(
        X_train=np.asarray(data.X_train, dtype=np.float64),
        y_train=np.asarray(data.y_train, dtype=np.float64),
        num_gfr=0,
        num_burnin=5,
        num_mcmc=10,
        mean_forest_params={'sample_sigma2_leaf': False},
    )
    bz_model = bst.BARTModel()
    bz_model.sample(
        X_train=data.X_train,
        y_train=data.y_train,
        num_gfr=0,
        num_burnin=5,
        num_mcmc=10,
        mean_forest_params=_MFP_BASE,
    )
    # bartz computes the standardization in float32; stochtree in float64.
    assert_allclose(bz_model.y_bar, st_model.y_bar, rtol=1e-6)
    assert_allclose(bz_model.y_std, st_model.y_std, rtol=1e-6)


@pytest.fixture
def comparison(
    request: pytest.FixtureRequest, keys: split
) -> tuple[str, stochtree.BARTModel, bst.BARTModel]:
    """Sample matching models from stochtree and bartz.stochtree.

    Parametrized indirectly on the outcome type ('continuous' or 'binary').
    Returns the outcome type alongside the two fitted models.

    The *same* keyword-argument dict is fed to both `stochtree.BARTModel` and
    `bartz.stochtree.BARTModel`, exercising the interface compatibility: the
    inputs are float64 (which bartz silently downcasts); `num_gfr=0` and
    `sample_sigma2_leaf=False` are set explicitly because bartz rejects their
    stochtree defaults while stochtree happily accepts the overrides; and a
    single `OutcomeModel` instance is duck-typed identically by both packages.
    """
    outcome = request.param
    num_burnin = 1000
    num_mcmc = 1000

    if outcome == 'continuous':
        data = _make_continuous(keys, n=50, n_test=80)
        seed = 13
        y = np.asarray(data.y_train, dtype=np.float64)
        extra: dict = {}
    else:
        data = _make_binary(keys, n=50, n_test=80)
        seed = 29
        y = np.asarray(data.y_train, dtype=np.int64)
        extra = {'outcome_model': bst.OutcomeModel(outcome='binary', link='probit')}

    kwargs: dict = dict(
        X_train=np.asarray(data.X_train, dtype=np.float64),
        y_train=y,
        X_test=np.asarray(data.X_test, dtype=np.float64),
        num_gfr=0,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params={'random_seed': seed, 'num_chains': 2, **extra},
        mean_forest_params={**_MFP_BASE, 'num_trees': 50},
    )

    st_model = stochtree.BARTModel()
    st_model.sample(**kwargs)
    bz_model = bst.BARTModel()
    bz_model.sample(**kwargs)
    return outcome, st_model, bz_model


@pytest.mark.parametrize('comparison', ['continuous', 'binary'], indirect=True)
def test_compare_with_stochtree(
    comparison: tuple[str, stochtree.BARTModel, bst.BARTModel], subtests: SubTests
) -> None:
    """bartz.stochtree mixes with stochtree; per-output Rhat stays near 1.

    bartz reproduces stochtree's MCMC, so the two target the same posterior and
    differ only by Monte Carlo noise. The fixture picks an easy-to-converge
    regime (small ``n``, ``num_trees >= max(n, p)``) so that noise stays small;
    a single threshold just above 1 bounds both outcome types.
    """
    outcome, st_model, bz_model = comparison

    # y_bar is deterministic and must match modulo float32 precision; for
    # binary it is ndtri(mean(y)).
    with subtests.test('y_bar'):
        assert_allclose(bz_model.y_bar, st_model.y_bar, rtol=1e-5)

    if outcome == 'continuous':
        with subtests.test('y_std'):
            assert_allclose(bz_model.y_std, st_model.y_std, rtol=1e-6)

    with subtests.test('rhat_y_hat_train'):
        rhat = _rhat_two_chains(bz_model.y_hat_train, st_model.y_hat_train)
        assert_array_less(rhat, 1.02)

    with subtests.test('rhat_y_hat_test'):
        rhat = _rhat_two_chains(bz_model.y_hat_test, st_model.y_hat_test)
        assert_array_less(rhat, 1.02)

    if outcome == 'continuous':
        with subtests.test('rhat_sigma'):
            bz_sigma = np.sqrt(np.asarray(bz_model.global_var_samples))
            st_sigma = np.sqrt(np.asarray(st_model.global_var_samples))
            # shape (1, num_samples) so rhat collapses to a scalar
            rhat = _rhat_two_chains(bz_sigma[None, :], st_sigma[None, :])
            assert_array_less(rhat, 1.02)
    else:
        with subtests.test('rhat_prob_train'):
            bz_prob = np.asarray(ndtr(bz_model.y_hat_train))
            st_prob = np.asarray(ndtr(st_model.y_hat_train))
            # logit-transform to spread the tails before computing Rhat
            bz_l = np.asarray(clipped_logit(bz_prob, 1e-5))
            st_l = np.asarray(clipped_logit(st_prob, 1e-5))
            rhat = _rhat_two_chains(bz_l, st_l)
            assert_array_less(rhat, 1.02)


def test_jit(keys: split) -> None:
    """Test that jitting around BARTModel.sample + predict works.

    All values that aren't used for shape / Python-level control flow are
    passed as jit arguments, so any accidentally non-traceable use of them
    surfaces as a ConcretizationTypeError.
    """
    data = _make_continuous(keys, n=80, n_test=20)
    p = data.X_train.shape[1]
    args: dict = {
        'key': random.key(0),
        'X': data.X_train,
        'y': data.y_train,
        'X_test': data.X_test,
        'X_predict': data.X_test[:5],
        'w': jnp.ones(data.y_train.shape),
        'sigma2_init': 1.0,
        'alpha': 0.95,
        'beta': 2.0,
        'sigma2_leaf_init': 1.0 / 200,
        'variable_weights': jnp.ones(p),
    }
    # devices can't be inferred from a jit tracer, so pre-determine the platform;
    # rm_const requires concrete max_split values, so disable it for tracing.
    bart_kwargs: dict = {
        'devices': jax.devices(args['y'].platform()),
        'rm_const': False,
    }

    def task(
        key: Array,
        X: Array,
        y: Array,
        X_test: Array,
        X_predict: Array,
        w: Array,
        sigma2_init: float,
        alpha: float,
        beta: float,
        sigma2_leaf_init: float,
        variable_weights: Array,
    ) -> tuple[Array, ...]:
        m = bst.BARTModel()
        m.sample(
            X_train=X,
            y_train=y,
            X_test=X_test,
            observation_weights=w,
            num_gfr=0,
            num_burnin=5,
            num_mcmc=10,
            general_params={
                'random_seed': key,
                'sigma2_init': sigma2_init,
                'variable_weights': variable_weights,
            },
            mean_forest_params={
                **_MFP_BASE,
                'alpha': alpha,
                'beta': beta,
                'sigma2_leaf_init': sigma2_leaf_init,
            },
            bart_kwargs=bart_kwargs,
        )
        pred_mean = m.predict(X_predict, type='mean', terms='y_hat')
        pred_post = m.predict(X_predict, type='posterior', terms='y_hat')
        return (
            m.y_hat_train,
            m.y_hat_test,
            m.global_var_samples,
            m.y_bar,
            m.y_std,
            pred_mean,
            pred_post,
        )

    args_cloned = dict(args, key=random.clone(args['key']))

    out1 = task(**args)
    out2 = jax.jit(task)(**args_cloned)

    for a, b in zip(out1, out2, strict=True):
        assert_close_matrices(a, b, rtol=1e-5)


class TestPreprocessing:
    """Tests for the pandas / polars auto-preprocessing pipeline.

    Each test is parametrized on ``flavor in {'pandas', 'polars'}`` to exercise
    both backends. Helpers at the top build identically-structured frames in
    both libraries from a shared "spec" so the test body stays library-agnostic.
    """

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _make_pandas_column(spec: dict) -> object:
        vals = spec['values']
        k = spec['kind']
        if k == 'numeric':
            return pd.array(vals, dtype='Float64')
        if k == 'bool':
            return pd.array(vals, dtype='boolean')
        if k == 'ordered_cat':
            return pd.Categorical(vals, categories=spec['categories'], ordered=True)
        if k == 'unordered_cat':
            return pd.Categorical(vals, categories=spec['categories'], ordered=False)
        if k == 'string':
            return pd.array(vals, dtype='string')
        if k == 'datetime':
            return pd.to_datetime(vals)
        msg = f'unknown kind {k!r}'
        raise ValueError(msg)

    @staticmethod
    def _make_polars_column(name: str, spec: dict) -> pl.Series:
        vals = spec['values']
        k = spec['kind']
        if k == 'numeric':
            return pl.Series(name, vals, dtype=pl.Float64)
        if k == 'bool':
            return pl.Series(name, vals, dtype=pl.Boolean)
        if k == 'unordered_cat':
            # polars Enum is the cross-library twin of a pandas unordered
            # Categorical (it round-trips to one), and bartz one-hot encodes it.
            return pl.Series(name, vals, dtype=pl.Enum(spec['categories']))
        if k == 'polars_categorical':
            return pl.Series(name, vals, dtype=pl.Categorical)
        if k == 'string':
            return pl.Series(name, vals, dtype=pl.String)
        if k == 'datetime':
            return pl.Series(name, vals).str.strptime(pl.Datetime)
        msg = f'unknown or pandas-only kind {k!r}'
        raise ValueError(msg)

    @classmethod
    def _make_df(
        cls, flavor: str, columns: dict[str, dict]
    ) -> pd.DataFrame | pl.DataFrame:
        """Build a DataFrame from a spec.

        Each entry of `columns` maps a column name to a dict with a 'kind' key
        plus a 'values' key (and 'categories' for the categorical kinds). Kinds:
        'numeric', 'bool', 'unordered_cat', 'string', 'datetime' work in both
        flavors; 'ordered_cat' is pandas-only (polars has no ordered categorical)
        and 'polars_categorical' is polars-only.
        """
        if flavor == 'pandas':
            return pd.DataFrame(
                {n: cls._make_pandas_column(s) for n, s in columns.items()}
            )
        if flavor == 'polars':
            return pl.DataFrame(
                {n: cls._make_polars_column(n, s) for n, s in columns.items()}
            )
        msg = f'unknown flavor {flavor!r}'
        raise ValueError(msg)

    @staticmethod
    def _make_preprocessor(flavor: str) -> _PreprocessorBase:
        return PandasPreprocessor() if flavor == 'pandas' else PolarsPreprocessor()

    @staticmethod
    def _sample(
        X_train: ArrayLike | pd.DataFrame | pl.DataFrame,
        y: ArrayLike,
        X_test: ArrayLike | pd.DataFrame | pl.DataFrame | None = None,
        seed: int = 0,
        **extras: object,
    ) -> bst.BARTModel:
        m = bst.BARTModel()
        m.sample(
            X_train=X_train,
            y_train=y,
            X_test=X_test,
            num_gfr=0,
            num_burnin=20,
            num_mcmc=40,
            general_params=dict(extras.pop('general_params', {}), random_seed=seed),
            mean_forest_params=_MFP_BASE,
            **extras,
        )
        return m

    # --------------------------------------------------- preprocessor unit tests

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_numeric_passthrough(self, flavor: str) -> None:
        """Pure-numeric columns are cast to float and forwarded unchanged."""
        df = self._make_df(
            flavor,
            {
                'a': {'kind': 'numeric', 'values': [0.1, 0.2, 0.3]},
                'b': {'kind': 'numeric', 'values': [1.0, 2.0, 3.0]},
            },
        )
        pp = self._make_preprocessor(flavor)
        w = pp.fit(df)
        X = pp.transform(df)
        assert_array_equal(
            X, np.array([[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]], np.float32)
        )
        assert w is None
        assert pp.n_processed_columns == 2
        assert pp.original_var_indices == (0, 1)

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_bool_passthrough(self, flavor: str) -> None:
        """Boolean columns become a single 0/1 column."""
        df = self._make_df(
            flavor, {'b': {'kind': 'bool', 'values': [True, False, True, False]}}
        )
        pp = self._make_preprocessor(flavor)
        pp.fit(df)
        X = pp.transform(df)
        assert_array_equal(X, np.array([[1.0], [0.0], [1.0], [0.0]], np.float32))

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_unordered_categorical_one_hot(self, flavor: str) -> None:
        """Unordered cat with k levels becomes a (n, k) binary matrix."""
        df = self._make_df(
            flavor,
            {
                'c': {
                    'kind': 'unordered_cat',
                    'values': ['a', 'b', 'a', 'c'],
                    'categories': ['a', 'b', 'c'],
                }
            },
        )
        pp = self._make_preprocessor(flavor)
        pp.fit(df)
        X = pp.transform(df)
        # Each row has exactly one 1; mass conserved
        assert X.shape == (4, 3)
        assert_array_equal(X.sum(axis=1), np.ones(4, np.float32))
        # The row values 'a','b','a','c' map to columns in the categories' order
        cats = pp._specs[0].categories
        for i, v in enumerate(['a', 'b', 'a', 'c']):
            assert X[i, cats.index(v)] == 1.0

    def test_ordered_categorical_ordinal(self) -> None:
        """Ordered cat becomes a single integer-coded column honoring the declared order.

        Pandas-only: polars has no ordered categorical dtype (`Enum` round-trips
        to a pandas *unordered* Categorical), so ordinal encoding is exposed only
        through pandas; polars users pass an integer column instead.
        """
        df = self._make_df(
            'pandas',
            {
                'c': {
                    'kind': 'ordered_cat',
                    'values': ['low', 'hi', 'mid', 'low'],
                    'categories': ['low', 'mid', 'hi'],
                }
            },
        )
        pp = self._make_preprocessor('pandas')
        pp.fit(df)
        X = pp.transform(df)
        assert_array_equal(X, np.array([[0.0], [2.0], [1.0], [0.0]], np.float32))

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_string_column_raises(self, flavor: str) -> None:
        """String/object columns are unsupported and raise at fit time."""
        df = self._make_df(
            flavor,
            {
                'a': {'kind': 'numeric', 'values': [0.1, 0.2, 0.3, 0.4]},
                's': {'kind': 'string', 'values': ['y', 'x', 'y', 'z']},
            },
        )
        pp = self._make_preprocessor(flavor)
        with pytest.raises(ValueError, match='unsupported dtype'):
            pp.fit(df)

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_datetime_column_raises(self, flavor: str) -> None:
        """Datetime columns are unsupported and raise at fit time."""
        df = self._make_df(
            flavor,
            {
                'a': {'kind': 'numeric', 'values': [0.1, 0.2]},
                'dt': {'kind': 'datetime', 'values': ['2024-01-01', '2024-01-02']},
            },
        )
        pp = self._make_preprocessor(flavor)
        with pytest.raises(ValueError, match='unsupported dtype'):
            pp.fit(df)

    def test_polars_categorical_raises(self) -> None:
        """A polars `Categorical` column is rejected (no per-column category list)."""
        df = self._make_df(
            'polars', {'c': {'kind': 'polars_categorical', 'values': ['a', 'b', 'a']}}
        )
        pp = self._make_preprocessor('polars')
        with pytest.raises(ValueError, match='polars Categorical'):
            pp.fit(df)

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_variable_weights_split_across_one_hot(self, flavor: str) -> None:
        """`variable_weights[i]` is divided evenly across an expanded column's k outputs."""
        df = self._make_df(
            flavor,
            {
                'a': {'kind': 'numeric', 'values': [0.1, 0.2, 0.3, 0.4]},
                'c': {
                    'kind': 'unordered_cat',
                    'values': ['a', 'b', 'a', 'c'],
                    'categories': ['a', 'b', 'c', 'd'],
                },
            },
        )
        pp = self._make_preprocessor(flavor)
        weights_in = np.array([2.0, 6.0])
        w_out = pp.fit(df, variable_weights=weights_in)
        # numeric stays at 2.0; the cat splits its budget evenly across its
        # k one-hot columns (k=4 from the declared category list in both flavors)
        k_cat = pp.n_processed_columns - 1
        expected = np.concatenate([[2.0], np.full(k_cat, 6.0 / k_cat)]).astype(
            np.float32
        )
        assert_close_matrices(w_out, expected)
        # Per-original-variable total preserved
        per_orig_sum = np.zeros(2)
        for j, w in zip(pp.original_var_indices, w_out, strict=True):
            per_orig_sum[j] += w
        assert_close_matrices(per_orig_sum, weights_in)

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_variable_weights_shape_mismatch_raises(self, flavor: str) -> None:
        """A mis-shaped `variable_weights` is caught up front."""
        df = self._make_df(flavor, {'a': {'kind': 'numeric', 'values': [0.1, 0.2]}})
        pp = self._make_preprocessor(flavor)
        with pytest.raises(ValueError, match='variable_weights must have shape'):
            pp.fit(df, variable_weights=np.array([1.0, 2.0]))

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_unseen_category_at_transform_raises(self, flavor: str) -> None:
        """A category absent from the fit-time data raises at transform time."""
        df_train = self._make_df(
            flavor,
            {
                'c': {
                    'kind': 'unordered_cat',
                    'values': ['a', 'b', 'a'],
                    'categories': ['a', 'b'],
                }
            },
        )
        df_test = self._make_df(
            flavor,
            {
                'c': {
                    'kind': 'unordered_cat',
                    'values': ['a', 'b', 'a'],
                    'categories': ['a', 'b'],
                }
            },
        )
        pp = self._make_preprocessor(flavor)
        pp.fit(df_train)
        # Sanity: equivalent test frame transforms fine
        pp.transform(df_test)
        # Now introduce an unseen value
        df_bad = self._make_df(
            flavor,
            {
                'c': {
                    'kind': 'unordered_cat',
                    'values': ['a', 'z'],
                    'categories': ['a', 'z'],
                }
            },
        )
        with pytest.raises(ValueError, match='not in the fitted category list'):
            pp.transform(df_bad)

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_disjoint_train_predict_categories_raises(self, flavor: str) -> None:
        """Completely disjoint training / prediction category sets raise at predict."""
        df_train = self._make_df(
            flavor,
            {
                'c': {
                    'kind': 'unordered_cat',
                    'values': ['a', 'b', 'a', 'b'],
                    'categories': ['a', 'b'],
                }
            },
        )
        df_predict = self._make_df(
            flavor,
            {
                'c': {
                    'kind': 'unordered_cat',
                    'values': ['x', 'y', 'z', 'x'],
                    'categories': ['x', 'y', 'z'],
                }
            },
        )
        pp = self._make_preprocessor(flavor)
        pp.fit(df_train)
        with pytest.raises(ValueError, match='not in the fitted category list'):
            pp.transform(df_predict)

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_n_columns_mismatch_at_transform_raises(self, flavor: str) -> None:
        """Transforming a frame with a different column count is rejected."""
        df_train = self._make_df(
            flavor,
            {
                'a': {'kind': 'numeric', 'values': [0.1, 0.2]},
                'b': {'kind': 'numeric', 'values': [1.0, 2.0]},
            },
        )
        df_test = self._make_df(
            flavor, {'a': {'kind': 'numeric', 'values': [0.1, 0.2]}}
        )
        pp = self._make_preprocessor(flavor)
        pp.fit(df_train)
        with pytest.raises(ValueError, match='preprocessor was fitted on 2 columns'):
            pp.transform(df_test)

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_transform_other_library_raises(self, flavor: str) -> None:
        """Transforming with a different dataframe library than fit raises."""
        spec = {'a': {'kind': 'numeric', 'values': [0.1, 0.2]}}
        other = 'polars' if flavor == 'pandas' else 'pandas'
        pp = self._make_preprocessor(flavor)
        pp.fit(self._make_df(flavor, spec))
        with pytest.raises(TypeError, match='same dataframe library'):
            pp.transform(self._make_df(other, spec))

    # ----------------------------------------------------- end-to-end integration

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_end_to_end_numeric_matches_array(self, keys: split, flavor: str) -> None:
        """A numeric DataFrame produces bit-identical posteriors to the same array."""
        data = _make_continuous(keys, n=60, n_test=15)
        X_arr = np.asarray(data.X_train)
        Xte_arr = np.asarray(data.X_test)
        names = [f'x{i}' for i in range(X_arr.shape[1])]
        df_train = self._make_df(
            flavor,
            {
                n: {'kind': 'numeric', 'values': X_arr[:, j].tolist()}
                for j, n in enumerate(names)
            },
        )
        df_test = self._make_df(
            flavor,
            {
                n: {'kind': 'numeric', 'values': Xte_arr[:, j].tolist()}
                for j, n in enumerate(names)
            },
        )
        m_arr = self._sample(X_arr, data.y_train, X_test=Xte_arr, seed=11)
        m_df = self._sample(df_train, data.y_train, X_test=df_test, seed=11)
        assert_close_matrices(m_arr.y_hat_train, m_df.y_hat_train)
        assert_close_matrices(m_arr.y_hat_test, m_df.y_hat_test)

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_end_to_end_one_hot_matches_manual(self, keys: split, flavor: str) -> None:
        """A DataFrame with an unordered cat matches a manually-one-hot-encoded array."""
        data = _make_continuous(keys, n=60, n_test=15)
        X_arr = np.asarray(data.X_train)
        Xte_arr = np.asarray(data.X_test)
        rng = np.random.default_rng(0)
        cat_train = rng.choice(['a', 'b', 'c'], size=X_arr.shape[0])
        cat_test = rng.choice(['a', 'b', 'c'], size=Xte_arr.shape[0])
        # Manual one-hot via numpy (categories order ['a','b','c'])
        cats = ['a', 'b', 'c']
        one_hot_train = np.eye(3)[[cats.index(v) for v in cat_train]]
        one_hot_test = np.eye(3)[[cats.index(v) for v in cat_test]]
        X_manual = np.concatenate([X_arr, one_hot_train], axis=1)
        Xte_manual = np.concatenate([Xte_arr, one_hot_test], axis=1)
        # DataFrame with the cat column
        names = [f'x{i}' for i in range(X_arr.shape[1])]
        df_train = self._make_df(
            flavor,
            {
                **{
                    n: {'kind': 'numeric', 'values': X_arr[:, j].tolist()}
                    for j, n in enumerate(names)
                },
                'c': {
                    'kind': 'unordered_cat',
                    'values': cat_train.tolist(),
                    'categories': cats,
                },
            },
        )
        df_test = self._make_df(
            flavor,
            {
                **{
                    n: {'kind': 'numeric', 'values': Xte_arr[:, j].tolist()}
                    for j, n in enumerate(names)
                },
                'c': {
                    'kind': 'unordered_cat',
                    'values': cat_test.tolist(),
                    'categories': cats,
                },
            },
        )
        # Use matching, exactly-representable weights on both sides so the test
        # isolates the *encoding*: the DataFrame's cat budget (k) splits evenly
        # to 1.0 per one-hot column, matching the manual array's per-column 1.0.
        p_num = X_arr.shape[1]
        k = len(cats)
        w_manual = np.ones(p_num + k)
        w_df = np.array([1.0] * p_num + [float(k)])
        m_manual = self._sample(
            X_manual,
            data.y_train,
            X_test=Xte_manual,
            seed=23,
            general_params={'variable_weights': w_manual},
        )
        m_df = self._sample(
            df_train,
            data.y_train,
            X_test=df_test,
            seed=23,
            general_params={'variable_weights': w_df},
        )
        assert_close_matrices(m_manual.y_hat_train, m_df.y_hat_train)
        assert_close_matrices(m_manual.y_hat_test, m_df.y_hat_test)

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_end_to_end_default_weights_split_like_stochtree(
        self, keys: split, flavor: str
    ) -> None:
        """Default weights follow stochtree's split, not uniform-over-columns.

        The categorical's budget is split evenly across its one-hot columns
        (stochtree semantics), not shared uniformly with the numeric columns.
        Sizes are chosen so every weight is exactly representable in float32
        (3 numeric cols -> 1/4 each; a 2-level cat -> 1/4/2 = 1/8 each), letting
        us compare the default-weights DataFrame model bit-for-bit against a
        manual one-hot array fed the equivalent expanded weights. If the default
        reverted to uniform-over-expanded-columns (1/5 each) this would fail.
        """
        data = _make_continuous(keys, n=60, n_test=15)
        X_arr = np.asarray(data.X_train)[:, :3]
        Xte_arr = np.asarray(data.X_test)[:, :3]
        rng = np.random.default_rng(1)
        cats = ['a', 'b']
        cat_train = rng.choice(cats, size=X_arr.shape[0])
        cat_test = rng.choice(cats, size=Xte_arr.shape[0])
        one_hot_train = np.eye(2)[[cats.index(v) for v in cat_train]]
        one_hot_test = np.eye(2)[[cats.index(v) for v in cat_test]]
        X_manual = np.concatenate([X_arr, one_hot_train], axis=1)
        Xte_manual = np.concatenate([Xte_arr, one_hot_test], axis=1)
        names = [f'x{i}' for i in range(X_arr.shape[1])]

        def _spec(arr: np.ndarray, cat: np.ndarray) -> dict:
            return {
                **{
                    n: {'kind': 'numeric', 'values': arr[:, j].tolist()}
                    for j, n in enumerate(names)
                },
                'c': {
                    'kind': 'unordered_cat',
                    'values': cat.tolist(),
                    'categories': cats,
                },
            }

        df_train = self._make_df(flavor, _spec(X_arr, cat_train))
        df_test = self._make_df(flavor, _spec(Xte_arr, cat_test))
        # stochtree's default split: each original variable keeps budget 1/4.
        w_split = np.array([0.25, 0.25, 0.25, 0.125, 0.125])
        m_df = self._sample(df_train, data.y_train, X_test=df_test, seed=7)
        m_manual = self._sample(
            X_manual,
            data.y_train,
            X_test=Xte_manual,
            seed=7,
            general_params={'variable_weights': w_split},
        )
        assert_close_matrices(m_manual.y_hat_train, m_df.y_hat_train)
        assert_close_matrices(m_manual.y_hat_test, m_df.y_hat_test)

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_predict_with_array_after_dataframe_fit_raises(
        self, keys: split, flavor: str
    ) -> None:
        """Predicting on a raw array after a DataFrame fit raises a clear error."""
        data = _make_continuous(keys, n=40, n_test=10)
        X_arr = np.asarray(data.X_train)
        names = [f'x{i}' for i in range(X_arr.shape[1])]
        df_train = self._make_df(
            flavor,
            {
                n: {'kind': 'numeric', 'values': X_arr[:, j].tolist()}
                for j, n in enumerate(names)
            },
        )
        m = self._sample(df_train, data.y_train, seed=5)
        with pytest.raises(TypeError, match='must also be a pandas/polars DataFrame'):
            m.predict(np.asarray(data.X_test), terms='y_hat')

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_end_to_end_unseen_category_in_predict_raises(
        self, keys: split, flavor: str
    ) -> None:
        """`predict()` with a brand-new category value raises a clear error."""
        data = _make_continuous(keys, n=40, n_test=10)
        X_arr = np.asarray(data.X_train)
        names = [f'x{i}' for i in range(X_arr.shape[1])]
        cat_train = ['a', 'b'] * (X_arr.shape[0] // 2)
        df_train = self._make_df(
            flavor,
            {
                **{
                    n: {'kind': 'numeric', 'values': X_arr[:, j].tolist()}
                    for j, n in enumerate(names)
                },
                'c': {
                    'kind': 'unordered_cat',
                    'values': cat_train,
                    'categories': ['a', 'b'],
                },
            },
        )
        m = self._sample(df_train, data.y_train, seed=5)
        # X to predict at: completely disjoint category set
        df_predict = self._make_df(
            flavor,
            {
                **{n: {'kind': 'numeric', 'values': [0.0, 0.0]} for n in names},
                'c': {
                    'kind': 'unordered_cat',
                    'values': ['x', 'y'],
                    'categories': ['x', 'y'],
                },
            },
        )
        with pytest.raises(ValueError, match='not in the fitted category list'):
            m.predict(df_predict, terms='y_hat')

    @pytest.mark.parametrize('flavor', ['pandas', 'polars'])
    def test_end_to_end_unsupported_column_raises(self, flavor: str) -> None:
        """An unsupported column dtype makes the model refuse to fit."""
        df = self._make_df(
            flavor, {'dt': {'kind': 'datetime', 'values': ['2024-01-01'] * 10}}
        )
        y = np.zeros(10, dtype=np.float32)
        m = bst.BARTModel()
        with pytest.raises(ValueError, match='unsupported dtype'):
            m.sample(
                X_train=df,
                y_train=y,
                num_gfr=0,
                num_burnin=2,
                num_mcmc=2,
                mean_forest_params=_MFP_BASE,
            )
