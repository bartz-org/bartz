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

from typing import NamedTuple

import jax
import numpy as np
import pytest
import stochtree
from equinox import EquinoxRuntimeError
from jax import numpy as jnp
from jax import random
from jax.scipy.special import ndtr
from numpy.testing import assert_array_less
from pytest_subtests import SubTests

import bartz.stochtree as bst
from bartz._jaxext import split
from bartz.testing import gen_data
from tests.util import assert_allclose, assert_close_matrices, clipped_logit, rhat_rank

_GEN_KW = dict(p=4, q=2, sigma2_lin=0.5, sigma2_quad=0.5, sigma2_eps=0.5)
# `mean_forest_params` override that disables the bartz-incompatible
# leaf-variance sampler. All sample()-using tests start from this base.
_MFP_BASE: dict = {'sample_sigma2_leaf': False}
# Force both packages to share the same proper variance prior and sigma^2 init.
# Without this, stochtree falls back to its improper-prior data calibration and
# bartz falls back to ``sigest='auto'``, so their posteriors disagree by
# more than the Rhat tolerance below.
_ALIGNED_PRIOR: dict = {
    'sigma2_global_shape': 1.5,
    'sigma2_global_scale': 0.5,
    'sigma2_init': 1.0,
}


class _Data(NamedTuple):
    X_train: np.ndarray  # (n, p)
    y_train: np.ndarray  # (n,) for continuous, (n,) int for binary
    X_test: np.ndarray  # (m, p)


def _make_continuous(keys: split, n: int = 200, n_test: int = 60) -> _Data:
    train = gen_data(keys.pop(), n=n, outcome_type='continuous', **_GEN_KW)
    test = gen_data(keys.pop(), n=n_test, outcome_type='continuous', **_GEN_KW)
    return _Data(
        X_train=np.asarray(train.x.T, dtype=np.float64),
        y_train=np.asarray(train.y, dtype=np.float64),
        X_test=np.asarray(test.x.T, dtype=np.float64),
    )


def _make_binary(keys: split, n: int = 200, n_test: int = 60) -> _Data:
    train = gen_data(keys.pop(), n=n, outcome_type='binary', **_GEN_KW)
    test = gen_data(keys.pop(), n=n_test, outcome_type='binary', **_GEN_KW)
    return _Data(
        X_train=np.asarray(train.x.T, dtype=np.float64),
        y_train=np.asarray(train.y, dtype=np.int64),
        X_test=np.asarray(test.x.T, dtype=np.float64),
    )


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
        X_train=data.X_train,
        y_train=data.y_train,
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
def comparison_continuous(keys: split) -> tuple[stochtree.BARTModel, bst.BARTModel]:
    """Sample matching continuous models from stochtree and bartz.stochtree."""
    data = _make_continuous(keys, n=300, n_test=80)
    num_burnin = 300
    num_mcmc = 800
    common = {'random_seed': 13, 'num_chains': 2, **_ALIGNED_PRIOR}

    st_model = stochtree.BARTModel()
    st_model.sample(
        X_train=data.X_train,
        y_train=data.y_train,
        X_test=data.X_test,
        num_gfr=0,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params=common,
        mean_forest_params={'sample_sigma2_leaf': False},
    )
    bz_model = bst.BARTModel()
    bz_model.sample(
        X_train=data.X_train,
        y_train=data.y_train,
        X_test=data.X_test,
        num_gfr=0,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params=common,
        mean_forest_params=_MFP_BASE,
    )
    return st_model, bz_model


def test_compare_continuous_with_stochtree(
    comparison_continuous: tuple[stochtree.BARTModel, bst.BARTModel], subtests: SubTests
) -> None:
    """bartz.stochtree mixes with stochtree, per-output Rhat stays bounded.

    The threshold is well above 1 because bartz and stochtree implement BART
    with different internal MCMC schemes (proposal distributions, residual
    bookkeeping) and so target slightly different finite-sample posteriors
    even when given the same priors.
    """
    st_model, bz_model = comparison_continuous

    with subtests.test('y_bar'):
        assert_allclose(bz_model.y_bar, st_model.y_bar, rtol=1e-6)
    with subtests.test('y_std'):
        assert_allclose(bz_model.y_std, st_model.y_std, rtol=1e-6)

    with subtests.test('rhat_y_hat_train'):
        rhat = _rhat_two_chains(bz_model.y_hat_train, st_model.y_hat_train)
        assert_array_less(rhat, 1.15)

    with subtests.test('rhat_y_hat_test'):
        rhat = _rhat_two_chains(bz_model.y_hat_test, st_model.y_hat_test)
        assert_array_less(rhat, 1.15)

    with subtests.test('rhat_sigma'):
        bz_sigma = np.sqrt(np.asarray(bz_model.global_var_samples))
        st_sigma = np.sqrt(np.asarray(st_model.global_var_samples))
        # shape (1, num_samples) so rhat collapses to a scalar
        rhat = _rhat_two_chains(bz_sigma[None, :], st_sigma[None, :])
        assert_array_less(rhat, 1.15)


@pytest.fixture
def comparison_binary(keys: split) -> tuple[stochtree.BARTModel, bst.BARTModel]:
    """Sample matching probit binary models from stochtree and bartz.stochtree."""
    data = _make_binary(keys, n=300, n_test=80)
    num_burnin = 300
    num_mcmc = 800
    seed = 29

    st_model = stochtree.BARTModel()
    st_model.sample(
        X_train=data.X_train,
        y_train=data.y_train,
        X_test=data.X_test,
        num_gfr=0,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params={
            'random_seed': seed,
            'num_chains': 2,
            'outcome_model': stochtree.OutcomeModel(outcome='binary', link='probit'),
        },
        mean_forest_params={'sample_sigma2_leaf': False},
    )
    bz_model = bst.BARTModel()
    bz_model.sample(
        X_train=data.X_train,
        y_train=data.y_train,
        X_test=data.X_test,
        num_gfr=0,
        num_burnin=num_burnin,
        num_mcmc=num_mcmc,
        general_params={
            'random_seed': seed,
            'num_chains': 2,
            'outcome_model': bst.OutcomeModel(outcome='binary', link='probit'),
        },
        mean_forest_params=_MFP_BASE,
    )
    return st_model, bz_model


def test_compare_binary_with_stochtree(
    comparison_binary: tuple[stochtree.BARTModel, bst.BARTModel], subtests: SubTests
) -> None:
    """bartz.stochtree probit outputs mix with stochtree without lifting Rhat too much.

    Probit binary regression has a higher tolerance than the continuous case
    because the latent-variable data augmentation amplifies the per-sampler
    algorithmic differences.
    """
    st_model, bz_model = comparison_binary

    # y_bar = ndtri(mean(y)) is deterministic and must match modulo float32 precision
    with subtests.test('y_bar'):
        assert_allclose(bz_model.y_bar, st_model.y_bar, rtol=1e-5)

    with subtests.test('rhat_y_hat_train'):
        rhat = _rhat_two_chains(bz_model.y_hat_train, st_model.y_hat_train)
        assert_array_less(rhat, 1.30)

    with subtests.test('rhat_y_hat_test'):
        rhat = _rhat_two_chains(bz_model.y_hat_test, st_model.y_hat_test)
        assert_array_less(rhat, 1.30)

    with subtests.test('rhat_prob_train'):
        bz_prob = np.asarray(ndtr(bz_model.y_hat_train))
        st_prob = np.asarray(ndtr(st_model.y_hat_train))
        # logit-transform to spread the tails before computing Rhat
        bz_l = np.asarray(clipped_logit(bz_prob, 1e-5))
        st_l = np.asarray(clipped_logit(st_prob, 1e-5))
        rhat = _rhat_two_chains(bz_l, st_l)
        assert_array_less(rhat, 1.30)


def test_jit(keys: split) -> None:
    """Test that jitting around BARTModel.sample + predict works.

    All values that aren't used for shape / Python-level control flow are
    passed as jit arguments, so any accidentally non-traceable use of them
    surfaces as a ConcretizationTypeError.
    """
    data = _make_continuous(keys, n=80, n_test=20)
    X_train = jnp.asarray(data.X_train)
    y_train = jnp.asarray(data.y_train)
    X_test = jnp.asarray(data.X_test)
    X_predict = jnp.asarray(data.X_test[:5])
    w = jnp.ones(y_train.shape)
    key = random.key(0)
    p = X_train.shape[1]
    # traceable scalars / arrays
    sigma2_init = jnp.float32(1.0)
    alpha = jnp.float32(0.95)
    beta = jnp.float32(2.0)
    sigma2_leaf_init = jnp.float32(1.0 / 200)
    variable_weights = jnp.ones(p)
    # devices can't be inferred from a jit tracer, so pre-determine the platform;
    # rm_const requires concrete max_split values, so disable it for tracing.
    bart_kwargs: dict = {'devices': jax.devices(y_train.platform()), 'rm_const': False}

    def task(
        X: jax.Array,
        y: jax.Array,
        X_test: jax.Array,
        X_predict: jax.Array,
        w: jax.Array,
        key: jax.Array,
        sigma2_init: jax.Array,
        alpha: jax.Array,
        beta: jax.Array,
        sigma2_leaf_init: jax.Array,
        variable_weights: jax.Array,
    ) -> tuple[jax.Array, ...]:
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

    args = (
        X_train,
        y_train,
        X_test,
        X_predict,
        w,
        key,
        sigma2_init,
        alpha,
        beta,
        sigma2_leaf_init,
        variable_weights,
    )
    args_cloned = (*args[:5], random.clone(key), *args[6:])

    out1 = task(*args)
    out2 = jax.jit(task)(*args_cloned)

    for a, b in zip(out1, out2, strict=True):
        assert_close_matrices(a, b, rtol=1e-5)
