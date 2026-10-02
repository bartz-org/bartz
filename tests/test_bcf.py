# bartz/tests/test_bcf.py
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

"""Tests for Bayesian Causal Forests (BCF)."""

import math
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import stochtree
from equinox import EquinoxRuntimeError, Module
from jax import random, tree, vmap
from jax.scipy.special import ndtr
from jax.tree_util import KeyPath, keystr
from jaxtyping import Array, ArrayLike, Float32, Key, Shaped
from pytest_subtests import SubTests
from scipy import stats

from bartz._jaxext import split
from bartz.bcf._bcf import UniqueQuantileBinner, bcf
from bartz.bcf._loop import BCFBurninTrace, BCFMainTrace, bcf_step
from bartz.bcf._state import BCFState, init_bcf
from bartz.grove import evaluate_forest, is_actual_leaf
from bartz.mcmcloop import run_mcmc
from bartz.mcmcstep import Forest, Wishart
from bartz.mcmcstep._axes import chain_vmap_axes
from bartz.mcmcstep._step import apply_moves_to_leaf_indices
from bartz.testing import gen_data
from tests.test_mcmcloop import assert_trace_close, cat_traces
from tests.util import (
    assert_allclose,
    assert_array_equal,
    assert_close_matrices,
    assert_different_matrices,
    int_seed,
    jaxtyping_disabled,
    rhat_rank,
)


def _prec_tree_from_scratch(
    forest: Forest, prec_scale: Shaped[ArrayLike, ' n']
) -> Shaped[ArrayLike, 'num_trees 2*half_tree_size']:
    """Sum `prec_scale` over the datapoints in each leaf of each tree."""
    leaf_indices = apply_moves_to_leaf_indices(
        forest.leaf_indices, forest.to_prune, forest.move_node
    )
    _, tree_size = forest.leaf_tree.shape

    def scatter(idx: Shaped[ArrayLike, ' n']) -> Shaped[ArrayLike, ' 2*half_tree_size']:
        return jnp.zeros(tree_size).at[idx].add(prec_scale)

    return vmap(scatter)(leaf_indices)


def _check_chains_match(
    multi: BCFState, singles: Sequence[BCFState], err_msg: str
) -> None:
    """Check each chain of `multi` matches the corresponding single-chain state."""

    def check_leaf(
        path: KeyPath,
        chain_axis: int | None,
        m: Shaped[Array, '*shape'] | None,
        *singles: Shaped[Array, '...'] | None,
    ) -> None:
        if m is None:
            return
        for i, s in enumerate(singles):
            mi = m if chain_axis is None else jnp.take(m, i, axis=chain_axis)
            msg = f'{err_msg}{keystr(path)}, chain {i}: '
            if jnp.issubdtype(m.dtype, jnp.inexact):
                assert_close_matrices(mi, s, rtol=1e-5, err_msg=msg, reduce_rank=True)
            else:
                assert_array_equal(mi, s, err_msg=msg)

    tree.map_with_path(
        check_leaf, chain_vmap_axes(multi), multi, *singles, is_leaf=lambda x: x is None
    )


def _assert_chains_differ(model: bcf) -> None:
    """Check the chains of a two-chain `bcf` differ in all the traced values that vary."""

    def check(
        path: KeyPath, x: Shaped[Array, '*shape'] | None, chain_axis: int | None
    ) -> None:
        if x is None or chain_axis is None:
            return
        chains = np.moveaxis(np.asarray(x), chain_axis, 0)
        # skip the values held fixed, e.g., unsampled leaf prior precisions
        if np.all(chains == chains.flat[0]):
            return
        # flatten to compare with the vector norm, the matrix 2-norm would
        # need an expensive svd on the big tree arrays
        assert_different_matrices(
            chains[0, ...].reshape(-1),
            chains[1, ...].reshape(-1),
            rtol=1e-3,
            atol=0,
            err_msg=f'{keystr(path)}: ',
        )

    traces = dict(model._main_trace, tau_0=model._tau_0_trace.reshape(2, -1))
    axes = dict({k: chain_vmap_axes(v) for k, v in model._main_trace.items()}, tau_0=0)
    tree.map_with_path(check, traces, axes, is_leaf=lambda x: x is None)


class BCFData(Module):
    """Synthetic BCF data with the true prognostic and treatment effect functions."""

    x: Float32[Array, 'n p']
    z: Float32[Array, ' n']
    """Treatment indicator, 0 or 1."""
    pihat: Float32[Array, ' n']
    """True propensity score."""
    y: Float32[Array, ' n']
    mu: Float32[Array, ' n']
    tau: Float32[Array, ' n']


def gen_bcf_data(
    key: Key[Array, ''],
    *,
    n: int,
    p: int = 5,
    mu_loc: float = 0.0,
    mu_scale: float = 1.0,
    tau_loc: float = 1.0,
    tau_scale: float = 0.5,
    noise_scale: float = 0.2,
) -> BCFData:
    """Generate confounded data with heterogeneous treatment effect.

    The prognostic function, the treatment effect and the treatment latent
    are the three components of a `gen_data` DGP with partially shared
    predictors. Each is linear with unit variance before rescaling.
    """
    dgp = gen_data(
        key,
        n=n,
        p=p,
        k=3,
        q=0,
        lambda_=0.5,
        sigma2_lin=1.0,
        sigma2_quad=0.0,
        sigma2_eps=1.0,
        outcome_type=('continuous', 'continuous', 'binary'),
    )
    mu = mu_loc + mu_scale * dgp.mu[0, :]
    tau = tau_loc + tau_scale * dgp.mu[1, :]
    z = dgp.y[2, :]
    noise = dgp.y[0, :] - dgp.mu[0, :]
    return BCFData(
        x=dgp.x.T,
        z=z,
        # with unit error variance, P(z = 1) = Phi(latent mean)
        pihat=ndtr(dgp.mu[2, :]),
        y=mu + tau * z + noise_scale * noise,
        mu=mu,
        tau=tau,
    )


def split_bcf_data(data: BCFData, n_train: int) -> tuple[BCFData, BCFData]:
    """Split `data` into training and test sets."""
    train = tree.map(lambda a: a[:n_train, ...], data)
    test = tree.map(lambda a: a[n_train:, ...], data)
    return train, test


class TestBcf:
    """Tests for the BCF wrapper module."""

    def test_bcf_save_load_npz(self, keys: split) -> None:
        """Tests saving and loading a multichain BCF model via NPZ preserves prediction equality."""
        train = gen_bcf_data(keys.pop(), n=200)

        model = bcf(
            x_train=train.x,
            y_train=train.y,
            z_train=train.z,
            pihat_train=train.pihat,
            num_trees_mu=2,
            num_trees_tau=2,
            ndpost=3,
            nskip=2,
            num_chains=2,
            standardize=False,
            seed=keys.pop(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / 'test_bcf.npz'
            model.save_npz(npz_path)

            # Verify schema_version is present in archive
            with np.load(npz_path) as archive:
                assert 'schema_version' in archive
                assert int(archive['schema_version']) == 1

            loaded_model = bcf.load_npz(npz_path)

            preds_orig = model.predict(train.x, pihat_test=train.pihat)
            preds_loaded = loaded_model.predict(train.x, pihat_test=train.pihat)

            assert_allclose(preds_loaded['mu'], preds_orig['mu'], allow_non_scalar=True)
            assert_allclose(
                preds_loaded['tau'], preds_orig['tau'], allow_non_scalar=True
            )
            assert_array_equal(loaded_model.sigma_trace, model.sigma_trace)

            # no x_test at construction, so no test predictions to restore
            assert loaded_model.mu_test is None
            assert loaded_model.tau_test is None
            assert loaded_model.yhat_test is None

    def test_bcf_save_load_npz_standardized(self, keys: split) -> None:
        """Tests that saving and loading an auto-standardized model preserves scale metadata and unscaling."""
        train = gen_bcf_data(
            keys.pop(), n=200, mu_loc=10.0, mu_scale=10.0, tau_loc=4.0, tau_scale=2.0
        )

        model = bcf(
            x_train=train.x,
            y_train=train.y,
            z_train=train.z,
            pihat_train=train.pihat,
            x_test=train.x,
            z_test=train.z,
            pihat_test=train.pihat,
            num_trees_mu=2,
            num_trees_tau=2,
            ndpost=3,
            nskip=2,
            standardize=True,
            seed=keys.pop(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / 'test_bcf_std.npz'
            model.save_npz(npz_path)

            with np.load(npz_path) as archive:
                assert 'standardize' in archive
                assert bool(archive['standardize'])
                assert '_y_mean' in archive
                assert '_y_std' in archive

            loaded_model = bcf.load_npz(npz_path)

            preds_orig = model.predict(train.x, pihat_test=train.pihat)
            preds_loaded = loaded_model.predict(train.x, pihat_test=train.pihat)

            assert_allclose(preds_loaded['mu'], preds_orig['mu'], allow_non_scalar=True)
            assert_allclose(
                preds_loaded['tau'], preds_orig['tau'], allow_non_scalar=True
            )

            # the outcome scale carries into the stored test predictions, which
            # round-trip through the archive
            assert_array_equal(model.mu_test, preds_orig['mu'])
            assert_array_equal(
                model.yhat_test, preds_orig['mu'] + train.z * preds_orig['tau']
            )
            assert_array_equal(loaded_model.mu_test, model.mu_test)
            assert_array_equal(loaded_model.tau_test, model.tau_test)
            assert_array_equal(loaded_model.yhat_test, model.yhat_test)
            assert loaded_model.prob_test is None

    def test_bcf_standardization_equivalence(self, keys: split) -> None:
        """Tests that automatic standardization is numerically equivalent to manual pre-scaling."""
        train = gen_bcf_data(
            keys.pop(), n=150, p=4, mu_loc=5.0, mu_scale=5.0, tau_loc=2.0, tau_scale=1.0
        )

        y_mean = np.mean(train.y)
        y_std = np.std(train.y)
        y_scaled = (train.y - y_mean) / y_std
        key = keys.pop()

        # Model 1: Auto standardization on raw y_train
        model_auto = bcf(
            x_train=train.x,
            y_train=train.y,
            z_train=train.z,
            pihat_train=train.pihat,
            num_trees_mu=5,
            num_trees_tau=5,
            ndpost=10,
            nskip=5,
            standardize=True,
            seed=key,
        )

        # Model 2: Manual pre-scaling with standardize=False
        model_manual = bcf(
            x_train=train.x,
            y_train=y_scaled,
            z_train=train.z,
            pihat_train=train.pihat,
            num_trees_mu=5,
            num_trees_tau=5,
            ndpost=10,
            nskip=5,
            standardize=False,
            seed=random.clone(key),
        )

        preds_auto = model_auto.predict(train.x, pihat_test=train.pihat)
        preds_manual = model_manual.predict(train.x, pihat_test=train.pihat)

        # Manual unscaling
        manual_mu_unscaled = preds_manual['mu'] * y_std + y_mean
        manual_tau_unscaled = preds_manual['tau'] * y_std

        assert_allclose(
            preds_auto['mu'],
            manual_mu_unscaled,
            rtol=1e-4,
            atol=1e-4,
            allow_non_scalar=True,
        )
        assert_allclose(
            preds_auto['tau'],
            manual_tau_unscaled,
            rtol=1e-4,
            atol=1e-4,
            allow_non_scalar=True,
        )

    def test_bcf_load_npz_unsupported_schema_version(self) -> None:
        """Tests that loading an NPZ file with a future schema version raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / 'invalid_schema.npz'
            np.savez(npz_path, schema_version=999)
            with pytest.raises(ValueError, match='Unsupported schema version: 999'):
                bcf.load_npz(npz_path)

    def test_bcf_statistical_convergence(self, keys: split, subtests: SubTests) -> None:
        """Multichain convergence and out-of-sample DGP recovery.

        Two chains must agree (Rhat near 1) without being identical, and a
        prior-matched model must recover the known treatment and prognostic
        effects on held-out data.
        """
        n = 100
        train, test = split_bcf_data(gen_bcf_data(keys.pop(), n=n + 300), n)

        y_mean = np.mean(train.y)
        y_std = np.std(train.y)
        y_scaled = (train.y - y_mean) / y_std

        ndpost = 2500
        nskip = 1500

        # 1. Internal Stability Model (JAX defaults), with two chains
        num_chains = 2
        model_jax = bcf(
            x_train=train.x,
            y_train=y_scaled,
            z_train=train.z,
            pihat_train=train.pihat,
            num_trees_mu=50,
            num_trees_tau=20,
            ndpost=ndpost,
            nskip=nskip,
            num_chains=num_chains,
            sample_sigma2_leaf_mu=False,
            sample_sigma2_leaf_tau=False,
            seed=keys.pop(),
        )

        # 2. Structural Alignment Models (Forced Prior Matching)
        leaf_prior_cov_inv_mu = 50.0
        leaf_prior_cov_inv_tau = 40.0

        model_jax_matched = bcf(
            x_train=train.x,
            y_train=y_scaled,
            z_train=train.z,
            pihat_train=train.pihat,
            num_trees_mu=50,
            num_trees_tau=20,
            ndpost=ndpost,
            nskip=nskip,
            leaf_prior_cov_inv_mu=leaf_prior_cov_inv_mu,
            leaf_prior_cov_inv_tau=leaf_prior_cov_inv_tau,
            sigma_df=0.0,
            sigma_scale=0.0,
            sample_sigma2_leaf_mu=False,
            sample_sigma2_leaf_tau=False,
            seed=keys.pop(),
        )

        # 1. Internal reproducibility: the two chains agree, but not trivially
        with subtests.test('chains agree'):
            preds_train = model_jax.predict(x_test=train.x, pihat_test=train.pihat)
            # the chains are concatenated along the sample axis
            yhat = preds_train['mu'] + preds_train['tau'] * train.z
            rhat_yhat = rhat_rank(yhat.reshape(num_chains, ndpost, n), split=False)
            tau = preds_train['tau'].reshape(num_chains, ndpost, n)
            rhat_tau_jax = rhat_rank(tau, split=True)
            assert np.max(rhat_yhat) < 1.06
            assert np.percentile(rhat_tau_jax, 95) < 1.10

        with subtests.test('chains differ'):
            _assert_chains_differ(model_jax)

        # 2. Out-of-sample recovery of the known DGP, on held-out data. RMSE
        # (not correlation) catches magnitude/offset errors; a constant tau
        # predictor scores ~0.5, so this requires capturing the heterogeneity.
        preds = model_jax_matched.predict(x_test=test.x, pihat_test=test.pihat)
        tau_hat = np.mean(np.array(preds['tau']) * y_std, axis=0)
        mu_hat = np.mean(np.array(preds['mu']) * y_std + y_mean, axis=0)
        assert np.sqrt(np.mean((tau_hat - test.tau) ** 2)) < 0.35
        assert np.sqrt(np.mean((mu_hat - test.mu) ** 2)) < 0.45

    def test_bcf_null_treatment_effect(self, keys: split) -> None:
        """Verifies that BCF does not find a treatment effect when tau=0."""
        data = gen_bcf_data(
            keys.pop(), n=220, tau_loc=0.0, tau_scale=0.0, noise_scale=0.5
        )
        train, test = split_bcf_data(data, 200)

        y_mean = np.mean(train.y)
        y_std = np.std(train.y)
        y_scaled = (train.y - y_mean) / y_std

        model = bcf(
            x_train=train.x,
            y_train=y_scaled,
            z_train=train.z,
            pihat_train=train.pihat,
            num_trees_mu=50,
            num_trees_tau=20,
            ndpost=400,
            nskip=400,
            sample_sigma2_leaf_mu=False,
            sample_sigma2_leaf_tau=False,
            seed=keys.pop(),
        )

        preds = model.predict(x_test=test.x, pihat_test=test.pihat)

        tau_samples = preds['tau'] * y_std
        cate_mean = np.mean(tau_samples, axis=0)

        # Point estimates should be close to 0
        assert np.mean(np.abs(cate_mean)) < 0.2

        # 95% Credible interval should cover 0 for >= 90% of units
        lower_bounds = np.percentile(tau_samples, 2.5, axis=0)
        upper_bounds = np.percentile(tau_samples, 97.5, axis=0)
        contains_zero = (lower_bounds <= 0.0) & (upper_bounds >= 0.0)
        assert np.mean(contains_zero) >= 0.90

    def test_bcf_noise_variance_recovery(self, keys: split) -> None:
        """Verifies that the BCF model recovers the true residual noise variance."""
        noise_scale = 0.5
        train = gen_bcf_data(keys.pop(), n=300, tau_loc=1.5, noise_scale=noise_scale)

        model = bcf(
            x_train=train.x,
            y_train=train.y,
            z_train=train.z,
            pihat_train=train.pihat,
            num_trees_mu=100,
            num_trees_tau=30,
            ndpost=600,
            nskip=400,
            seed=keys.pop(),
        )

        # Recovered noise variance is (1.0 / precision) * y_std^2
        recovered_sigma2 = (1.0 / model._main_trace['mu'].error_cov_inv) * (
            model._y_std**2
        )
        posterior_mean_sigma2 = np.mean(recovered_sigma2)

        assert np.isclose(posterior_mean_sigma2, np.square(noise_scale), atol=0.10)

    def test_bcf_one_step_residual_invariant(self, keys: split) -> None:
        """Verifies that R == y - offset - mu_fit - (tau_0 + tau_fit) * Z."""
        train = gen_bcf_data(keys.pop(), n=100)

        x_train_t = train.x.T
        binner = UniqueQuantileBinner(x_train_t, key=keys.pop())
        x_binned = binner.bin(x_train_t)
        max_split = binner.max_split

        init_state = init_bcf(
            X_unified=x_binned,
            trt=train.z.astype(bool),
            # `init_bcf` may donate its arguments
            y=jnp.copy(train.y),
            offset=0.0,
            max_split_mu=jnp.array(max_split),
            max_split_tau=jnp.array(max_split),
            num_trees_mu=5,
            num_trees_tau=5,
            p_nonterminal_mu=np.ones(5, dtype=np.float32) * 0.95,
            p_nonterminal_tau=np.ones(5, dtype=np.float32) * 0.95,
            leaf_prior_cov_inv_mu=Wishart(nu=6.0, rate=2.0, value=1.0),
            leaf_prior_cov_inv_tau=Wishart(nu=None, rate=None, value=1.0),
            error_cov_inv=Wishart(
                nu=jnp.float32(1.0),
                rate=jnp.array(1.0, dtype=jnp.float32),
                value=jnp.array(1.0, dtype=jnp.float32),
            ),
        )

        new_state = bcf_step(keys.pop(), init_state)

        mu_fit_raw = evaluate_forest(new_state.X, new_state.forest).sum(axis=0)
        mu_fit = (
            mu_fit_raw
            if new_state.inv_sdev_scale is None
            else mu_fit_raw / new_state.inv_sdev_scale
        )

        tau_fit_raw = evaluate_forest(new_state.X, new_state.forest_tau).sum(axis=0)

        b_z = new_state.b[train.z.astype(int)]
        expected_resid = (
            train.y
            - new_state.forest.offset
            - mu_fit
            - b_z * (new_state.tau_0 + tau_fit_raw)
        )

        # `resid` is stored scaled (``resid_unit * resid = data residual``)
        assert_allclose(
            new_state.resid * new_state.resid_unit,
            expected_resid,
            atol=1e-5,
            allow_non_scalar=True,
        )

    @pytest.mark.parametrize(
        ('adaptive_coding', 'prec_count_num_trees'),
        [(False, None), (True, None), (True, 1)],
    )
    def test_bcf_step_tau_prec_tree_cache(
        self, keys: split, adaptive_coding: bool, prec_count_num_trees: int | None
    ) -> None:
        """
        Check `bcf_step` keeps the tau forest's `prec_tree` cache consistent.

        The tau likelihood precision of each datapoint is ``b_z**2``, so after a
        step that resamples the coding weights, the cached per-leaf sums must
        match the new weights. Setting `prec_count_num_trees` exercises the
        batched rebuild of the cache.
        """
        train = gen_bcf_data(keys.pop(), n=100)

        x_train_t = train.x.T
        binner = UniqueQuantileBinner(x_train_t, key=keys.pop())
        x_binned = binner.bin(x_train_t)
        max_split = binner.max_split

        state = init_bcf(
            X_unified=x_binned,
            trt=train.z.astype(bool),
            y=train.y,
            offset=0.0,
            max_split_mu=jnp.array(max_split),
            max_split_tau=jnp.array(max_split),
            num_trees_mu=2,
            num_trees_tau=3,
            p_nonterminal_mu=np.ones(4, dtype=np.float32) * 0.95,
            p_nonterminal_tau=np.ones(4, dtype=np.float32) * 0.95,
            min_points_per_leaf_tau=1,
            leaf_prior_cov_inv_mu=Wishart(nu=6.0, rate=2.0, value=1.0),
            leaf_prior_cov_inv_tau=Wishart(nu=None, rate=None, value=1.0),
            adaptive_coding=adaptive_coding,
            error_cov_inv=Wishart(
                nu=jnp.float32(1.0),
                rate=jnp.array(1.0, dtype=jnp.float32),
                value=jnp.array(1.0, dtype=jnp.float32),
            ),
        )
        state = replace(
            state,
            config=replace(state.config, prec_count_num_trees=prec_count_num_trees),
        )

        def check_tau_prec_tree(state: BCFState, err_msg: str) -> None:
            forest = state.forest_tau
            assert forest.prec_tree is not None
            b_z = state.b[state.trt.astype(int)]
            expected = _prec_tree_from_scratch(forest, jnp.square(b_z))
            is_leaf = vmap(partial(is_actual_leaf, add_bottom_level=True))(
                forest.split_tree
            )
            assert_close_matrices(
                jnp.where(is_leaf, forest.prec_tree, 0.0),
                jnp.where(is_leaf, expected, 0.0),
                rtol=1e-5,
                err_msg=err_msg,
            )

        check_tau_prec_tree(state, 'at init: ')

        for i in range(4):
            state = bcf_step(keys.pop(), state)
            check_tau_prec_tree(state, f'after step {i + 1}: ')

    def test_bcf_multichain(self, keys: split) -> None:
        """Check each chain of a multichain BCF matches a single-chain one."""
        train = gen_bcf_data(keys.pop(), n=100)

        x_train_t = train.x.T
        binner = UniqueQuantileBinner(x_train_t, key=keys.pop())
        x_binned = binner.bin(x_train_t)

        def make_state(num_chains: int | None) -> BCFState:
            # `init_bcf` may donate its arguments, so pass fresh copies
            return init_bcf(
                X_unified=jnp.copy(x_binned),
                trt=train.z.astype(bool),
                y=jnp.copy(train.y),
                offset=0.0,
                max_split_mu=jnp.copy(binner.max_split),
                max_split_tau=jnp.copy(binner.max_split),
                num_trees_mu=2,
                num_trees_tau=3,
                p_nonterminal_mu=jnp.full(4, 0.95),
                p_nonterminal_tau=jnp.full(4, 0.95),
                leaf_prior_cov_inv_mu=Wishart(nu=6.0, rate=2.0, value=1.0),
                leaf_prior_cov_inv_tau=Wishart(nu=6.0, rate=2.0, value=1.0),
                adaptive_coding=True,
                error_cov_inv=Wishart(
                    nu=jnp.array(1.0), rate=jnp.array(1.0), value=jnp.array(1.0)
                ),
                num_chains=num_chains,
            )

        num_chains = 3
        multi = make_state(num_chains)
        assert multi.num_chains() == num_chains

        # the reduction configs depend on `num_chains`, share them to get the
        # same sums
        singles = [
            replace(make_state(None), config=tree.map(jnp.copy, multi.config))
            for _ in range(num_chains)
        ]
        assert singles[0].num_chains() is None
        _check_chains_match(multi, singles, 'init: ')

        # step the multichain state and the single-chain ones with the same
        # per-chain keys
        for i in range(3):
            key = keys.pop()
            multi = bcf_step(key, multi)
            single_keys = random.split(random.clone(key), num_chains)
            singles = [
                bcf_step(k, s) for k, s in zip(single_keys, singles, strict=True)
            ]
            _check_chains_match(multi, singles, f'step {i + 1}: ')

    @pytest.mark.parametrize('num_chains', [None, 2])
    def test_bcf_run_mcmc_restartable(
        self, keys: split, subtests: SubTests, num_chains: int | None
    ) -> None:
        """Check splitting a BCF `run_mcmc` run and chunking it do not matter."""
        train = gen_bcf_data(keys.pop(), n=100)

        x_train_t = train.x.T
        binner = UniqueQuantileBinner(x_train_t, key=keys.pop())
        x_binned = binner.bin(x_train_t)

        state = init_bcf(
            X_unified=x_binned,
            trt=train.z.astype(bool),
            y=train.y,
            offset=0.0,
            max_split_mu=binner.max_split,
            # `init_bcf` may donate its arguments, so don't pass the same array twice
            max_split_tau=jnp.copy(binner.max_split),
            num_trees_mu=2,
            num_trees_tau=3,
            p_nonterminal_mu=jnp.full(4, 0.95),
            p_nonterminal_tau=jnp.full(4, 0.95),
            leaf_prior_cov_inv_mu=Wishart(nu=6.0, rate=2.0, value=1.0),
            leaf_prior_cov_inv_tau=Wishart(nu=None, rate=None, value=1.0),
            adaptive_coding=True,
            error_cov_inv=Wishart(
                nu=jnp.array(1.0), rate=jnp.array(1.0), value=jnp.array(1.0)
            ),
            num_chains=num_chains,
        )

        key = keys.pop()
        kw: dict = dict(
            step=bcf_step,
            burnin_trace_type=BCFBurninTrace,
            main_trace_type=BCFMainTrace,
        )

        # one run of 2 burn-in + 3 saved iterations; `run_mcmc` donates the
        # state, so pass a copy
        final_single, burnin_single, main_single = run_mcmc(
            key, tree.map(jnp.copy, state), 3, n_burn=2, **kw
        )
        # the same run split as 2 + 1 and 0 + 2, in inner loops of 2 iterations
        mid, burnin_a, main_a = run_mcmc(
            random.clone(key),
            tree.map(jnp.copy, state),
            1,
            n_burn=2,
            inner_loop_length=2,
            **kw,
        )
        final_split, _, main_b = run_mcmc(
            random.clone(key), mid, 2, n_burn=0, inner_loop_length=2, **kw
        )

        tree.map(assert_trace_close, final_single, final_split)
        tree.map(assert_trace_close, burnin_single, burnin_a)
        tree.map(assert_trace_close, main_single, cat_traces(main_a, main_b))

        # piggyback on the runs above to check the trace layout, and that the
        # last sample is the final state
        final_single = cast(BCFState, final_single)
        burnin_single = cast(BCFBurninTrace, burnin_single)
        main_single = cast(BCFMainTrace, main_single)
        with subtests.test('trace layout'):
            chain_shape = () if num_chains is None else (num_chains,)
            assert burnin_single.tau_0.shape == (*chain_shape, 2)
            assert main_single.tau_0.shape == (*chain_shape, 3)
            assert main_single.b.shape == (*chain_shape, 3, 2)
            assert main_single.mu.var_tree.shape[:-1] == (*chain_shape, 3, 2)
            assert main_single.tau.var_tree.shape[:-1] == (*chain_shape, 3, 3)
            assert_array_equal(main_single.tau_0[..., -1], final_single.tau_0)
            assert_array_equal(main_single.b[..., -1, :], final_single.b)
            assert_array_equal(
                main_single.tau.leaf_tree[..., -1, :, :],
                final_single.forest_tau.leaf_tree,
            )

    def test_bcf_unsplittable_x_reduction(self, keys: split) -> None:
        """Verifies BCF degenerates to Bayesian linear regression when max_split is 0."""
        rng = np.random.default_rng(int_seed(keys.pop()))
        n = 300
        p = 1
        x_train = np.ones((n, p), dtype=np.float32)

        pi = 0.5
        z_train = rng.binomial(1, pi, size=n).astype(np.float32)

        mu_true = 5.0
        tau_true = -3.0
        y_train = (mu_true + tau_true * z_train + rng.normal(size=n) * 1.0).astype(
            np.float32
        )

        model = bcf(
            x_train=x_train,
            y_train=y_train,
            z_train=z_train,
            pihat_train=np.zeros(n, dtype=np.float32),
            num_trees_mu=1,
            num_trees_tau=1,
            ndpost=1000,
            nskip=1000,
            sigma_df=0.0,
            sigma_scale=0.0,
            sample_sigma2_leaf_mu=False,
            sample_sigma2_leaf_tau=False,
            seed=keys.pop(),
        )

        preds = model.predict(
            x_test=np.ones((1, p), dtype=np.float32),
            pihat_test=np.zeros(1, dtype=np.float32),
        )
        mu_mcmc = preds['mu'][:, 0]
        tau_mcmc = preds['tau'][:, 0]

        slope, intercept, _, _, _ = stats.linregress(z_train, y_train)

        assert np.isclose(np.mean(mu_mcmc), intercept, atol=2.50)
        assert np.isclose(np.mean(tau_mcmc), slope, atol=2.50)

    def test_bcf_adaptive_coding(self, keys: split) -> None:
        """Adaptive coding recovers the known treatment effect out of sample."""
        n = 100
        train, test = split_bcf_data(gen_bcf_data(keys.pop(), n=n + 300), n)

        y_mean = np.mean(train.y)
        y_std = np.std(train.y)
        y_scaled = (train.y - y_mean) / y_std

        ndpost = 1000
        nskip = 500

        leaf_prior_cov_inv_mu = 50.0
        leaf_prior_cov_inv_tau = 40.0

        model_jax = bcf(
            x_train=train.x,
            y_train=y_scaled,
            z_train=train.z,
            pihat_train=train.pihat,
            num_trees_mu=50,
            num_trees_tau=20,
            ndpost=ndpost,
            nskip=nskip,
            leaf_prior_cov_inv_mu=leaf_prior_cov_inv_mu,
            leaf_prior_cov_inv_tau=leaf_prior_cov_inv_tau,
            sigma_df=0.0,
            sigma_scale=0.0,
            adaptive_coding=True,
            seed=keys.pop(),
        )

        preds = model_jax.predict(x_test=test.x, pihat_test=test.pihat)
        cate_hat = np.mean(np.array(preds['tau']) * y_std, axis=0)
        mu_hat = np.mean(np.array(preds['mu']) * y_std + y_mean, axis=0)
        assert np.sqrt(np.mean((cate_hat - test.tau) ** 2)) < 0.4
        assert np.sqrt(np.mean((mu_hat - test.mu) ** 2)) < 0.4

    def test_bcf_leaf_variance_prior_inactive(self, keys: split) -> None:
        """Verifies that sample_sigma2_leaf=False keeps the prior variance fixed."""
        n = 50
        p = 3
        train = gen_bcf_data(keys.pop(), n=n, p=p)

        y_scaled = (train.y - np.mean(train.y)) / np.std(train.y)

        model = bcf(
            x_train=train.x,
            y_train=y_scaled,
            z_train=train.z,
            pihat_train=train.pihat,
            num_trees_mu=10,
            num_trees_tau=5,
            ndpost=10,
            nskip=0,
            sample_sigma2_leaf_mu=False,
            sample_sigma2_leaf_tau=False,
            seed=keys.pop(),
        )

        # Check that trace variances are constant
        mu_prior_vars = np.array(model._leaf_prior_cov_inv_mu_trace)
        tau_prior_vars = np.array(model._leaf_prior_cov_inv_tau_trace)

        # Assert variance across the chain (axis 0) is 0
        assert_allclose(np.var(mu_prior_vars, axis=0), 0.0, atol=1e-7)
        assert_allclose(np.var(tau_prior_vars, axis=0), 0.0, atol=1e-7)

        # Compare to active
        model_active = bcf(
            x_train=train.x,
            y_train=y_scaled,
            z_train=train.z,
            pihat_train=train.pihat,
            num_trees_mu=10,
            num_trees_tau=5,
            ndpost=10,
            nskip=0,
            sample_sigma2_leaf_mu=True,
            sample_sigma2_leaf_tau=True,
            seed=keys.pop(),
        )

        mu_prior_vars_active = np.array(model_active._leaf_prior_cov_inv_mu_trace)
        tau_prior_vars_active = np.array(model_active._leaf_prior_cov_inv_tau_trace)

        assert np.var(mu_prior_vars_active, axis=0).mean() > 1e-4
        assert np.var(tau_prior_vars_active, axis=0).mean() > 1e-4

    def test_bcf_leaf_variance_prior_active_equivalence(self, keys: split) -> None:
        """Adaptive leaf variance matches StochTree's scale and recovers the DGP."""
        n = 500
        train, test = split_bcf_data(gen_bcf_data(keys.pop(), n=n + 300), n)

        y_mean = np.mean(train.y)
        y_std = np.std(train.y)
        y_scaled = (train.y - y_mean) / y_std

        ndpost = 1500
        nskip = 1000

        model_jax = bcf(
            x_train=train.x,
            y_train=y_scaled,
            z_train=train.z,
            pihat_train=train.pihat,
            num_trees_mu=200,
            num_trees_tau=50,
            ndpost=ndpost,
            nskip=nskip,
            sample_sigma2_leaf_mu=True,
            sample_sigma2_leaf_tau=True,
            sigma2_leaf_shape_mu=3.0,
            sigma2_leaf_shape_tau=3.0,
            sigma2_leaf_scale_mu=4.0 / 200.0,
            sigma2_leaf_scale_tau=1.0 / 50.0,
            sigma_df=0.0,
            sigma_scale=0.0,
            adaptive_coding=False,
            min_points_per_leaf_mu=3,
            min_points_per_leaf_tau=3,
            seed=keys.pop(),
        )

        # stochtree's sampler uses IG(shape / 2, scale / 2) instead of the
        # documented IG(shape, scale), so double its parameters to match
        model_st = stochtree.BCFModel()
        model_st.sample(
            X_train=np.asarray(train.x),
            Z_train=np.asarray(train.z),
            y_train=np.asarray(train.y, np.float64),
            propensity_train=np.asarray(train.pihat),
            num_mcmc=ndpost,
            num_gfr=0,
            num_burnin=nskip,
            prognostic_forest_params={
                'num_trees': 200,
                'sample_sigma2_leaf': True,
                'sigma2_leaf_shape': 2 * 3.0,
                'sigma2_leaf_scale': 2 * 4.0 / 200.0,
                'min_samples_leaf': 3,
            },
            treatment_effect_forest_params={
                'num_trees': 50,
                'sample_sigma2_leaf': True,
                'sigma2_leaf_shape': 2 * 3.0,
                'sigma2_leaf_scale': 2 * 1.0 / 50.0,
                'sample_intercept': True,
                'min_samples_leaf': 3,
            },
            general_params={
                'adaptive_coding': False,
                'random_seed': int_seed(keys.pop()),
                'control_coding_init': 0.0,
                'treated_coding_init': 1.0,
            },
        )

        y_var = np.var(train.y)

        leaf_var_mu_jax = np.mean(1.0 / model_jax._leaf_prior_cov_inv_mu_trace) * y_var
        leaf_var_mu_st = np.mean(model_st.leaf_scale_mu_samples) * y_var
        assert_allclose(leaf_var_mu_jax, leaf_var_mu_st, rtol=0.3)

        leaf_var_tau_jax = (
            np.mean(1.0 / model_jax._leaf_prior_cov_inv_tau_trace) * y_var
        )
        leaf_var_tau_st = np.mean(model_st.leaf_scale_tau_samples) * y_var
        # single-chain estimates of the tau leaf variance vary by a factor >2
        # across MCMC seeds in both implementations, hence the loose tolerance
        assert_allclose(leaf_var_tau_jax, leaf_var_tau_st, rtol=1.5)

        preds = model_jax.predict(x_test=test.x, pihat_test=test.pihat)
        tau_hat = np.mean(np.array(preds['tau']) * y_std, axis=0)
        mu_hat = np.mean(np.array(preds['mu']) * y_std + y_mean, axis=0)
        assert np.sqrt(np.mean((tau_hat - test.tau) ** 2)) < 0.15
        assert np.sqrt(np.mean((mu_hat - test.mu) ** 2)) < 0.15

    def test_predict_potential_outcomes(self, keys: split) -> None:
        """Tests posterior predictive potential outcome sampling in BCF."""
        train = gen_bcf_data(keys.pop(), n=150)

        model = bcf(
            x_train=train.x,
            y_train=train.y,
            z_train=train.z,
            pihat_train=train.pihat,
            num_trees_mu=5,
            num_trees_tau=5,
            ndpost=20,
            nskip=10,
            standardize=True,
            seed=keys.pop(),
        )

        # 1. Verify sigma_trace
        # with self.subTest(name='sigma_trace'):
        sigma = model.sigma_trace
        assert sigma.shape == (20,)
        assert bool(jnp.all(sigma > 0.0))

        x_test = train.x[:30]
        pihat_test = train.pihat[:30]

        # 2. Test shapes, keys, and realized lift consistency
        # with self.subTest(name='shapes_and_lift_consistency'):
        res = model.predict_potential_outcomes(
            x_test=x_test, pihat_test=pihat_test, rho=0.5, key=keys.pop()
        )
        for key in ['y0', 'y1', 'delta', 'mu', 'tau']:
            assert key in res
            assert res[key].shape == (20, 30)

        assert_allclose(
            np.array(res['delta']),
            np.array(res['y1'] - res['y0']),
            rtol=1e-5,
            atol=1e-5,
            allow_non_scalar=True,
        )

        # 3. Test rho = 1.0 (Rank preservation -> delta == tau)
        # with self.subTest(name='rank_preservation_rho_1'):
        res_rho1 = model.predict_potential_outcomes(
            x_test=x_test, pihat_test=pihat_test, rho=1.0, key=keys.pop()
        )
        assert_allclose(
            np.array(res_rho1['delta']),
            np.array(res_rho1['tau']),
            rtol=1e-5,
            atol=1e-5,
            allow_non_scalar=True,
        )

        # 4. Test rho = 0.0 (Independent shocks -> delta != tau)
        # with self.subTest(name='independent_shocks_rho_0'):
        res_rho0 = model.predict_potential_outcomes(
            x_test=x_test, pihat_test=pihat_test, rho=0.0, key=keys.pop()
        )
        diff = np.abs(np.array(res_rho0['delta'] - res_rho0['tau']))
        assert np.any(diff > 1e-3)

        # 5. Test invalid rho validation
        # with self.subTest(name='invalid_rho_validation'):
        with pytest.raises(ValueError, match='rho must be in'):
            model.predict_potential_outcomes(x_test=x_test, rho=-0.1)
        with pytest.raises(ValueError, match='rho must be in'):
            model.predict_potential_outcomes(x_test=x_test, rho=1.5)

        # 6. key=None (default RNG) and integer-seed keys are both accepted
        res_key_none = model.predict_potential_outcomes(
            x_test=x_test, pihat_test=pihat_test, key=None
        )
        res_key_int = model.predict_potential_outcomes(
            x_test=x_test, pihat_test=pihat_test, key=int_seed(keys.pop())
        )
        assert res_key_none['y0'].shape == res_key_int['y0'].shape

    def test_bcf_binary_model(self, keys: split) -> None:
        """Tests binary BCF end-to-end: initialization, offset, and predictions."""
        # Generate data with non-trivial positive rate (~70% positive)
        train = gen_bcf_data(keys.pop(), n=200)
        y_train = (train.y > np.percentile(train.y, 30)).astype(np.float32)

        model = bcf(
            x_train=train.x,
            y_train=y_train,
            z_train=train.z,
            pihat_train=train.pihat,
            x_test=train.x,
            z_test=train.z,
            pihat_test=train.pihat,
            num_trees_mu=5,
            num_trees_tau=5,
            ndpost=10,
            nskip=5,
            outcome_type='binary',
            delta_max=0.9,
            seed=keys.pop(),
        )

        # Sub-test 1: Verify initialization and offset correctness
        # with self.subTest(name='offset_and_scaling_invariants'):
        assert model._y_std == 1.0
        assert model._y_mean == 0.0
        expected_offset = stats.norm.ppf(np.mean(y_train))
        assert np.isclose(model._offset, expected_offset, atol=0.0001)

        # Sub-test 2: Verify prediction keys and valid probability bounds
        # with self.subTest(name='prediction_probabilities'):
        preds = model.predict(train.x, pihat_test=train.pihat)
        assert 'mu' in preds
        assert 'tau' in preds
        assert 'tau_prob' in preds
        assert 'p1' in preds
        assert 'p0' in preds

        # Check that raw probabilities are within [0, 1]
        assert np.all((preds['p0'] >= 0.0) & (preds['p0'] <= 1.0))
        assert np.all((preds['p1'] >= 0.0) & (preds['p1'] <= 1.0))

        # Sub-test 3: Constructor test predictions are on the latent scale,
        # with prob_test carrying the probability
        assert_array_equal(model.mu_test, preds['mu'])
        assert_array_equal(model.tau_test, preds['tau'])
        prob_test = model.prob_test
        assert prob_test is not None
        assert_close_matrices(
            prob_test, np.where(train.z, preds['p1'], preds['p0']), rtol=1e-5
        )
        assert np.all((prob_test >= 0.0) & (prob_test <= 1.0))

        # prob_test survives a save/load round-trip
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / 'test_bcf_binary.npz'
            model.save_npz(npz_path)
            assert_array_equal(bcf.load_npz(npz_path).prob_test, prob_test)

        # Sub-test 4: potential outcomes on a binary model return 0/1 labels
        po = model.predict_potential_outcomes(
            train.x, pihat_test=train.pihat, key=keys.pop()
        )
        assert np.isin(np.array(po['y0']), (0.0, 1.0)).all()
        assert np.isin(np.array(po['y1']), (0.0, 1.0)).all()

    def test_bcf_binary_requires_0_1(self, keys: split) -> None:
        """Binary BCF rejects outcomes that are not 0/1."""
        train = gen_bcf_data(keys.pop(), n=20)
        with pytest.raises(ValueError, match='strictly 0 or 1'):
            bcf(
                x_train=train.x,
                y_train=np.full(20, 2.0, np.float32),
                z_train=train.z,
                pihat_train=train.pihat,
                outcome_type='binary',
                num_trees_mu=2,
                num_trees_tau=2,
                ndpost=1,
                nskip=0,
                seed=keys.pop(),
            )

    def test_bcf_treatment_requires_0_1(self, keys: split) -> None:
        """BCF rejects treatments that are not 0/1."""
        train = gen_bcf_data(keys.pop(), n=20)
        with pytest.raises(EquinoxRuntimeError, match='must be 0 or 1'):
            bcf(
                x_train=train.x,
                y_train=train.y,
                z_train=np.full(20, 0.5, np.float32),
                pihat_train=train.pihat,
                num_trees_mu=2,
                num_trees_tau=2,
                ndpost=1,
                nskip=0,
                seed=keys.pop(),
            )

    @pytest.mark.parametrize(
        ('sample_intercept', 'num_chains'), [(True, None), (False, 2)]
    )
    def test_bcf_constructor_options(
        self, keys: split, sample_intercept: bool, num_chains: int | None
    ) -> None:
        """Constructor x_test/z_test, pihat toggle, tau_0 prior/toggle, chains, sigma_trace."""
        train, test = split_bcf_data(gen_bcf_data(keys.pop(), n=45), 30)
        ndpost = 3
        model = bcf(
            x_train=train.x,
            y_train=train.y,
            z_train=train.z,
            pihat_train=train.pihat,
            x_test=test.x,
            z_test=test.z,
            pihat_test=test.pihat,
            include_pihat_in_mu=False,
            tau_0_prior_var=0.5,
            sample_intercept=sample_intercept,
            standardize=False,
            num_trees_mu=2,
            num_trees_tau=2,
            ndpost=ndpost,
            nskip=1,
            num_chains=num_chains,
            seed=keys.pop(),
        )
        assert model._mcmc_state.num_chains() == num_chains
        chain_shape = () if num_chains is None else (num_chains,)
        num_samples = math.prod(chain_shape) * ndpost
        assert model._tau_0_trace.shape == (num_samples,)
        assert model._b_trace.shape == (num_samples, 2)
        assert model.mu_test is not None
        assert model.mu_test.shape == (num_samples, len(test.x))
        tau_0_is_zero = model._tau_0_trace == 0
        assert_array_equal(tau_0_is_zero, jnp.full(num_samples, not sample_intercept))

        # the chains are concatenated one after the other
        error_cov_inv = model._main_trace['mu'].error_cov_inv
        assert error_cov_inv.shape == (*chain_shape, ndpost)
        assert_array_equal(
            model.sigma_trace, jnp.reciprocal(jnp.sqrt(error_cov_inv)).reshape(-1)
        )

        # test predictions computed at construction match predict()
        preds = model.predict(test.x, pihat_test=test.pihat)
        assert_array_equal(model.mu_test, preds['mu'])
        assert_array_equal(model.tau_test, preds['tau'])
        assert_array_equal(model.yhat_test, preds['mu'] + test.z * preds['tau'])
        assert model.prob_test is None

    def test_bcf_test_predictions_absent(self, keys: split) -> None:
        """Without x_test, the test prediction attributes stay None."""
        train = gen_bcf_data(keys.pop(), n=30)
        model = bcf(
            x_train=train.x,
            y_train=train.y,
            z_train=train.z,
            pihat_train=train.pihat,
            num_trees_mu=2,
            num_trees_tau=2,
            ndpost=3,
            nskip=1,
            seed=int_seed(keys.pop()),  # covers integer seeds
        )
        assert model.mu_test is None
        assert model.tau_test is None
        assert model.yhat_test is None
        assert model.prob_test is None

    def test_bcf_test_input_errors(self, keys: split) -> None:
        """Inconsistent test inputs are rejected before running the MCMC."""
        train, test = split_bcf_data(gen_bcf_data(keys.pop(), n=45), 30)
        kwargs: dict = dict(
            x_train=train.x,
            y_train=train.y,
            z_train=train.z,
            num_trees_mu=2,
            num_trees_tau=2,
            ndpost=2,
            nskip=1,
            seed=keys.pop(),
        )
        with pytest.raises(ValueError, match='require `x_test`'):
            bcf(**kwargs, z_test=test.z)
        with pytest.raises(ValueError, match='require `x_test`'):
            bcf(**kwargs, pihat_test=test.pihat)
        with pytest.raises(ValueError, match='must be passed together'):
            bcf(**kwargs, pihat_train=train.pihat, x_test=test.x)
        with pytest.raises(ValueError, match='must be passed together'):
            bcf(**kwargs, x_test=test.x, pihat_test=test.pihat)
        # jaxtyping binds `m` across x_test/z_test/pihat_test, so disable it to
        # reach the explicit length checks (users run without the import hook)
        with (
            jaxtyping_disabled(),
            pytest.raises(ValueError, match='`z_test` has length'),
        ):
            bcf(**kwargs, x_test=test.x, z_test=train.z)
        with (
            jaxtyping_disabled(),
            pytest.raises(ValueError, match='`pihat_test` has length'),
        ):
            bcf(
                **kwargs, pihat_train=train.pihat, x_test=test.x, pihat_test=train.pihat
            )
        with pytest.raises(EquinoxRuntimeError, match='must be 0 or 1'):
            bcf(**kwargs, x_test=test.x, z_test=np.full(15, 2.0, np.float32))

    def test_bcf_x_test_format_mismatch(self, keys: split) -> None:
        """x_test format must match x_train, at construction and at predict."""
        train = gen_bcf_data(keys.pop(), n=30)
        x_test_df = pd.DataFrame(np.asarray(train.x)[:15])
        with pytest.raises(ValueError, match='does not match x_train'):
            bcf(
                x_train=train.x,
                y_train=train.y,
                z_train=train.z,
                pihat_train=train.pihat,
                x_test=x_test_df,
                num_trees_mu=2,
                num_trees_tau=2,
                ndpost=2,
                nskip=1,
                seed=keys.pop(),
            )
        model = bcf(
            x_train=train.x,
            y_train=train.y,
            z_train=train.z,
            pihat_train=train.pihat,
            num_trees_mu=2,
            num_trees_tau=2,
            ndpost=3,
            nskip=1,
            seed=keys.pop(),
        )
        with pytest.raises(ValueError, match='does not match x_train'):
            model.predict(x_test_df)
