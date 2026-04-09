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

"""Test the multivariate interface of `bartz.Bart`."""

from typing import Any, Literal, NamedTuple

import jax
import pytest
from jax import debug_nans, random
from jax import numpy as jnp
from jax.scipy.special import ndtr
from jaxtyping import Array, Bool, Float32, Key, Real
from numpy.testing import assert_allclose, assert_array_equal
from pytest import FixtureRequest  # noqa: PT013
from pytest_subtests import SubTests

from bartz import Bart
from bartz.jaxext import get_device_count, split
from bartz.testing import gen_data
from tests.util import assert_close_matrices, multivariate_rhat


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


class BartKW(NamedTuple):
    """Keyword arguments for `Bart` plus associated test data."""

    kw: dict[str, Any]
    x_test: Real[Array, 'p m']


def make_kw(key: Key[Array, ''], variant: int) -> BartKW:
    """Return keyword arguments for `Bart` and test predictors."""
    keys = split(key, 5)

    match variant:
        # continuous regression with some settings that induce large types,
        # sparsity with free theta
        case 1:
            X = gen_X(keys.pop(), 2, 30, 'continuous')
            Xt = gen_X(keys.pop(), 2, 31, 'continuous')
            y = gen_y(keys.pop(), X, None, 'continuous', s='random')
            return BartKW(
                kw=dict(
                    x_train=X,
                    y_train=y,
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
                    num_data_devices=min(2, get_device_count()),
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
            p = 257  # > 256 to use uint16 for var_trees.
            X = gen_X(keys.pop(), p, 30, 'binary')
            Xt = gen_X(keys.pop(), p, 31, 'binary')
            y = gen_y(keys.pop(), X, None, 'probit')
            return BartKW(
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
        case 3:  # pragma: no branch
            X = gen_X(keys.pop(), 2, 30, 'continuous')
            Xt = gen_X(keys.pop(), 2, 31, 'continuous')
            w = gen_w(keys.pop(), X.shape[1])
            y = gen_y(keys.pop(), X, w, 'continuous', s='random')
            return BartKW(
                kw=dict(
                    x_train=X,
                    y_train=y,
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
            )

        case _:  # pragma: no cover
            msg = f'Unknown variant {variant}'
            raise ValueError(msg)


class MVData(NamedTuple):
    """Dataset for testing in `TestMVBartInterface`."""

    x: Float32[Array, 'p n']
    y: Float32[Array, 'k n']
    w: Float32[Array, ' n'] | None = None
    outcome_type: str | list[str] = 'continuous'


class TestMVBartInterface:
    """Tests for the high-level Bart class with multivariate y."""

    @pytest.fixture
    def kwargs(self) -> dict:
        """Provide base keyword arguments for Bart initialization."""
        return dict(num_trees=5, ndpost=10, nskip=5, num_chains=None)

    @pytest.fixture(params=[None, 2])
    def num_chains(self, request: FixtureRequest) -> int | None:
        """Provide the number of chains to test with."""
        return request.param

    @pytest.fixture(params=[(20, 6, 2), (20, 10, 3), (50, 100, 4), (50, 50, 5)])
    def mv_data_shape(self, request: FixtureRequest) -> tuple[int, int, int]:
        """Provide (n, p, k) triples for testing."""
        return request.param

    @pytest.fixture
    def mv_data(self, keys: split, mv_data_shape: tuple[int, int, int]) -> MVData:
        """Generate a toy multivariate dataset."""
        n, p, k = mv_data_shape
        dgp = gen_data(
            keys.pop(),
            n=n,
            p=p,
            k=k,
            q=2,
            lam=0.5,
            sigma2_lin=0.4,
            sigma2_quad=0.5,
            sigma2_eps=0.1,
        )
        return MVData(dgp.x, dgp.y)

    @pytest.fixture
    def example_data(self, keys: split) -> MVData:
        """Return a small nonsense dataset for tests that don't need realistic data."""
        return MVData(
            random.normal(keys.pop(), (2, 5)),
            random.normal(keys.pop(), (3, 5)),
            random.normal(keys.pop(), (5,)),
        )

    @pytest.fixture(params=['continuous', 'binary', 'mixed'])
    def mv_outcome_data(self, request: FixtureRequest, mv_data: MVData) -> MVData:
        """Apply outcome_mode postprocessing to mv_data."""
        outcome_mode = request.param
        k = mv_data.y.shape[0]
        if outcome_mode == 'binary':
            y = (mv_data.y > 0).astype(jnp.float32)
            outcome_type = 'binary'
        elif outcome_mode == 'mixed':
            y = mv_data.y.at[0].set((mv_data.y[0] > 0).astype(jnp.float32))
            outcome_type = ['binary'] + ['continuous'] * (k - 1)
        else:
            y = mv_data.y
            outcome_type = 'continuous'
        return MVData(x=mv_data.x, y=y, outcome_type=outcome_type)

    def test_initialization_and_shapes(
        self, keys: split, mv_outcome_data: MVData, kwargs: dict, num_chains: int | None
    ) -> None:
        """Test that MV Bart predicts with correct shapes."""
        k, n = mv_outcome_data.y.shape
        n_test = 40
        p = mv_outcome_data.x.shape[0]
        outcome_type = mv_outcome_data.outcome_type

        kwargs.update(num_chains=num_chains, outcome_type=outcome_type)
        bart = Bart(x_train=mv_outcome_data.x, y_train=mv_outcome_data.y, **kwargs)

        # test predict shape
        x_test = random.normal(keys.pop(), (p, n_test))
        for x, m in [('train', n), (x_test, n_test)]:
            y_pred = bart.predict(x, kind='latent_samples')
            assert y_pred.shape == (bart.ndpost, k, m)

            y_pred = bart.predict(x, kind='mean_samples')
            assert y_pred.shape == (bart.ndpost, k, m)

            y_pred = bart.predict(x, kind='mean')
            assert y_pred.shape == (k, m)

            y_pred = bart.predict(x, kind='outcome_samples', key=keys.pop())
            assert y_pred.shape == (bart.ndpost, k, m)

            mean = bart.predict(x, kind='mean')
            outcomes = bart.predict(x, kind='outcome_samples', key=keys.pop())
            if outcome_type == 'binary':
                assert jnp.all((mean >= 0) & (mean <= 1))
                assert jnp.all((outcomes == 0) | (outcomes == 1))
            elif isinstance(outcome_type, list):
                assert jnp.all((mean[0] >= 0) & (mean[0] <= 1))
                assert jnp.all((outcomes[..., 0, :] == 0) | (outcomes[..., 0, :] == 1))

        # config params shape
        assert bart.offset.shape == (k,)
        if outcome_type == 'binary':
            assert bart.sigest is None
        elif isinstance(outcome_type, list):
            sigest = bart.sigest
            assert sigest.shape == (k,)
            assert sigest[0] == 0.0  # binary component
            assert jnp.all(sigest[1:] > 0)  # continuous components
        else:
            assert bart.sigest.shape == (k,)

    @pytest.mark.parametrize('outcome_mode', ['continuous', 'mixed'])
    def test_scalar_params(
        self, example_data: MVData, subtests: SubTests, outcome_mode: str, kwargs: dict
    ) -> None:
        """Test that scalar configuration params are broadcasted."""
        k, _ = example_data.y.shape
        if outcome_mode == 'mixed':
            outcome_type = ['binary'] + ['continuous'] * (k - 1)
        else:
            outcome_type = 'continuous'
        kwargs.update(ndpost=0, nskip=0, outcome_type=outcome_type)

        with subtests.test('offset'):
            bart = Bart(example_data.x, example_data.y, offset=0.0, **kwargs)
            assert bart.offset.shape == (k,)

        with subtests.test('sigest'):
            bart = Bart(example_data.x, example_data.y, sigest=1.0, **kwargs)
            assert bart.sigest.shape == (k,)

        with subtests.test('lamda'):
            bart = Bart(example_data.x, example_data.y, lamda=1.0, **kwargs)
            assert bart.sigest is None
            assert bart._mcmc_state.error_cov_scale.shape == (k, k)

    def test_mv_rejects_weights(self, example_data: MVData, kwargs: dict) -> None:
        """MV + weights should raise."""
        with pytest.raises(ValueError, match='Weights'):
            Bart(
                x_train=example_data.x,
                y_train=example_data.y,
                w=example_data.w,
                **kwargs,
            )

    def test_get_latent_prec(
        self, mv_outcome_data: MVData, kwargs: dict, num_chains: int | None
    ) -> None:
        """get_latent_prec returns correct shape, dtype, and is symmetric PD."""
        k = mv_outcome_data.y.shape[0]
        kwargs.update(num_chains=num_chains, outcome_type=mv_outcome_data.outcome_type)
        bart = Bart(x_train=mv_outcome_data.x, y_train=mv_outcome_data.y, **kwargs)
        ndpost = kwargs['ndpost']
        nskip = kwargs['nskip']

        prec = bart.get_latent_prec()
        if num_chains is not None:
            assert prec.shape == (num_chains, nskip + ndpost // num_chains, k, k)
        else:
            assert prec.shape == (nskip + ndpost, k, k)
        assert prec.dtype == jnp.float32

        # check symmetry and positive definiteness
        assert_close_matrices(prec, prec.mT, rtol=1e-6, reduce_rank=True)
        eigvals = jnp.linalg.eigvalsh(prec)
        assert jnp.all(eigvals > 0)

    def test_get_latent_prec_only_continuous(
        self, mv_outcome_data: MVData, kwargs: dict, num_chains: int | None
    ) -> None:
        """get_latent_prec(only_continuous=True) removes binary components."""
        k = mv_outcome_data.y.shape[0]
        outcome_type = mv_outcome_data.outcome_type
        kwargs.update(num_chains=num_chains, outcome_type=outcome_type)
        bart = Bart(x_train=mv_outcome_data.x, y_train=mv_outcome_data.y, **kwargs)
        ndpost = kwargs['ndpost']
        nskip = kwargs['nskip']

        if outcome_type == 'binary':
            with pytest.raises(ValueError, match='only binary'):
                bart.get_latent_prec(only_continuous=True)
            return

        prec = bart.get_latent_prec(only_continuous=True)
        if isinstance(outcome_type, list):
            kb = sum(1 for t in outcome_type if t == 'binary')
            kc = k - kb
        else:
            kc = k
        if num_chains is not None:
            assert prec.shape == (num_chains, nskip + ndpost // num_chains, kc, kc)
        else:
            assert prec.shape == (nskip + ndpost, kc, kc)

    def test_get_error_sdev_shape(
        self, mv_outcome_data: MVData, kwargs: dict, num_chains: int | None
    ) -> None:
        """get_error_sdev returns correct shape and NaN pattern."""
        k = mv_outcome_data.y.shape[0]
        outcome_type = mv_outcome_data.outcome_type
        kwargs.update(num_chains=num_chains, outcome_type=outcome_type)
        bart = Bart(x_train=mv_outcome_data.x, y_train=mv_outcome_data.y, **kwargs)
        ndpost = kwargs['ndpost']

        with debug_nans(False):
            sdev = bart.get_error_sdev()
            assert sdev.shape == (ndpost, k)

            if outcome_type == 'binary':
                assert jnp.all(jnp.isnan(sdev))
            elif isinstance(outcome_type, list):
                binary_mask = jnp.array([t == 'binary' for t in outcome_type])
                assert jnp.all(jnp.isnan(sdev[:, binary_mask]))
                assert jnp.all(jnp.isfinite(sdev[:, ~binary_mask]))
            else:
                assert jnp.all(jnp.isfinite(sdev))
                assert jnp.all(sdev > 0)

    def test_get_error_sdev_mean(
        self, mv_outcome_data: MVData, kwargs: dict, num_chains: int | None
    ) -> None:
        """get_error_sdev(mean=True) returns correct shape and NaN pattern."""
        k = mv_outcome_data.y.shape[0]
        outcome_type = mv_outcome_data.outcome_type
        kwargs.update(num_chains=num_chains, outcome_type=outcome_type)
        bart = Bart(x_train=mv_outcome_data.x, y_train=mv_outcome_data.y, **kwargs)

        with debug_nans(False):
            sdev_mean = bart.get_error_sdev(mean=True)
            assert sdev_mean.shape == (k,)

            if outcome_type == 'binary':
                assert jnp.all(jnp.isnan(sdev_mean))
            elif isinstance(outcome_type, list):
                binary_mask = jnp.array([t == 'binary' for t in outcome_type])
                assert jnp.all(jnp.isnan(sdev_mean[binary_mask]))
                assert jnp.all(jnp.isfinite(sdev_mean[~binary_mask]))
            else:
                assert jnp.all(jnp.isfinite(sdev_mean))
                assert jnp.all(sdev_mean > 0)

    def test_get_error_sdev_values(
        self, mv_data: MVData, kwargs: dict, num_chains: int | None
    ) -> None:
        """get_error_sdev matches manual computation from precision matrices."""
        kwargs.update(num_chains=num_chains)
        bart = Bart(x_train=mv_data.x, y_train=mv_data.y, **kwargs)
        nskip = kwargs['nskip']
        sdev = bart.get_error_sdev()

        # manual: invert each precision matrix, take sqrt of diagonal
        prec = bart.get_latent_prec()
        if num_chains is not None:
            prec_post = prec[:, nskip:]  # skip burnin per chain
            prec_post = prec_post.reshape(-1, *prec_post.shape[2:])  # flatten chains
        else:
            prec_post = prec[nskip:]  # skip burnin
        cov = jnp.linalg.inv(prec_post)
        expected = jnp.sqrt(jnp.diagonal(cov, axis1=-2, axis2=-1))
        assert_close_matrices(sdev, expected, rtol=1e-5)

    def test_mixed_rejects_weights(self, example_data: MVData, kwargs: dict) -> None:
        """Mixed outcome_type + weights should raise."""
        k, _ = example_data.y.shape
        kwargs.update(outcome_type=['binary'] + ['continuous'] * (k - 1))
        with pytest.raises(ValueError, match='univariate continuous'):
            Bart(
                x_train=example_data.x,
                y_train=example_data.y,
                w=example_data.w,
                **kwargs,
            )

    def test_outcome_type_length_mismatch(
        self, example_data: MVData, kwargs: dict
    ) -> None:
        """Sequence outcome_type with wrong length should raise."""
        kwargs.update(outcome_type=['continuous', 'continuous'])
        with pytest.raises(ValueError, match='length'):
            Bart(x_train=example_data.x, y_train=example_data.y, **kwargs)

    def test_sequence_outcome_type_requires_2d(self, keys: split, kwargs: dict) -> None:
        """Sequence outcome_type with 1D y should raise."""
        x = random.normal(keys.pop(), (2, 5))
        y = random.normal(keys.pop(), (5,))
        kwargs.update(outcome_type=['continuous'])
        with pytest.raises(ValueError, match=r'y_train\.shape=\(1, n\)'):
            Bart(x_train=x, y_train=y, **kwargs)

    def test_uv_mv_k1_equivalence(self, keys: split) -> None:
        """Test that Bart class initializes equivalent states for UV and MV (k=1)."""
        n, p = 20, 5
        X = random.normal(keys.pop(), (p, n))
        y_uv = random.normal(keys.pop(), (n,))
        y_mv = y_uv[None, :]  # shape (1, n)

        common = dict(x_train=X, num_trees=10, ndpost=0, nskip=0, num_chains=1, seed=42)
        bart_uv = Bart(y_train=y_uv, **common)
        bart_mv = Bart(y_train=y_mv, **common)

        state_uv = bart_uv._mcmc_state
        state_mv = bart_mv._mcmc_state

        # Residuals and error covariance
        assert_allclose(state_uv.resid, state_mv.resid.squeeze(0), atol=1e-6, rtol=1e-6)
        assert_allclose(
            state_uv.error_cov_inv,
            state_mv.error_cov_inv.reshape(()),
            atol=1e-6,
            rtol=1e-6,
        )

        # Prior parameters
        assert_array_equal(
            state_uv.forest.leaf_prior_cov_inv,
            state_mv.forest.leaf_prior_cov_inv.reshape(()),
        )
        assert_array_equal(state_uv.error_cov_df, state_mv.error_cov_df)
        assert_array_equal(
            state_uv.error_cov_scale, state_mv.error_cov_scale.reshape(())
        )

        # Forest structure
        assert_array_equal(state_uv.forest.var_tree, state_mv.forest.var_tree)
        assert_array_equal(state_uv.forest.split_tree, state_mv.forest.split_tree)
        assert_array_equal(
            state_uv.forest.leaf_tree, state_mv.forest.leaf_tree.squeeze(-2)
        )
        assert_array_equal(state_uv.forest.leaf_indices, state_mv.forest.leaf_indices)

        # Offset
        assert_allclose(bart_uv.offset, bart_mv.offset.squeeze(0), atol=1e-6, rtol=1e-6)

        # Sigest
        assert_allclose(bart_uv.sigest, bart_mv.sigest.squeeze(0), atol=1e-6, rtol=1e-6)

    def test_mvbart_convergence(self, mv_data: MVData, keys: split) -> None:
        """Test that MV Bart chains converge using R-hat."""
        _, n_train = mv_data.x.shape
        k_dim = mv_data.y.shape[0]

        num_chains = 4
        ndpost = 2000
        nsamples_per_chain = ndpost // num_chains
        nskip = 4000
        keepevery = 5
        num_trees = 100

        bart = Bart(
            x_train=mv_data.x,
            y_train=mv_data.y,
            num_trees=num_trees,
            ndpost=ndpost,
            nskip=nskip,
            keepevery=keepevery,
            num_chains=num_chains,
            seed=keys.pop(),
        )

        # Check yhat convergence
        yhat_train = bart.predict('train', kind='latent_samples')
        yhat_train = yhat_train.reshape(num_chains, nsamples_per_chain, k_dim * n_train)
        rhat_yhat_train = multivariate_rhat(yhat_train)
        assert rhat_yhat_train < 1.6
        print(f'{rhat_yhat_train.item()=}')

        # Check covariance matrix convergence
        prec_trace = bart._main_trace.error_cov_inv
        if prec_trace.ndim == 3:
            prec_trace = prec_trace.reshape(
                num_chains, nsamples_per_chain, k_dim, k_dim
            )
        prec_flat = prec_trace.reshape(num_chains, nsamples_per_chain, -1)
        assert jnp.all(jnp.std(prec_flat, axis=1) > 1e-8), 'Sigma is not updating!'
        rhat_prec = multivariate_rhat(prec_flat)
        assert rhat_prec < 1.1
        print(f'{rhat_prec.item()=}')
