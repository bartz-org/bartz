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

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import stochtree
from equinox import EquinoxRuntimeError
from jax import random
from jaxtyping import ArrayLike, Shaped
from scipy import stats

from bartz.bcf._bcf import UniqueQuantileBinner, bcf
from bartz.bcf._loop import bcf_step
from bartz.bcf._state import init_bcf
from bartz.grove import evaluate_forest
from bartz.mcmcstep import Wishart
from tests.util import assert_allclose, rhat_rank


def _rhat_two_chains(
    a: Shaped[ArrayLike, '*shape'], b: Shaped[ArrayLike, '*shape']
) -> Shaped[np.ndarray, '...']:
    """
    Compute rank-normalized Rhat between two (num_samples, n) matrices.

    Parameters
    ----------
    a
        First MCMC chain samples of shape (num_samples, n).
    b
        Second MCMC chain samples of shape (num_samples, n).

    Returns
    -------
    Shaped[np.ndarray, '...']
        An array of Rhat values for each of the n outputs.
    """
    stacked = np.stack([a, b], axis=0)  # shape (2, num_samples, n)
    return rhat_rank(stacked, split=False)


# pylint: disable=protected-access
class TestBcf:
    """Tests for the BCF wrapper module."""

    @staticmethod
    def _generate_bcf_data(
        n: int,
        p: int = 5,
        mu_coefs: tuple[float, float] = (1.0, 1.0),
        tau_coefs: tuple[float, float] = (2.0, 1.0),
        noise_scale: float = 0.5,
        seed: int = 42,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.random.Generator,
    ]:
        """
        Generate synthetic data for BCF testing.

        Parameters
        ----------
        n
            The number of observations.
        p
            The number of covariates (default 5).
        mu_coefs
            The coefficients for the mean model.
        tau_coefs
            The coefficients for the treatment effect.
        noise_scale
            The standard deviation of the error term.
        seed
            The random seed constraint.

        Returns
        -------
        tuple
            A tuple containing (x_train, pi, z_train, y_train, mu, tau, rng).
        """
        rng = np.random.default_rng(seed)
        x_train = rng.normal(size=(n, p)).astype(np.float32)
        pi = 1.0 / (1.0 + np.exp(-x_train[:, 0]))
        z_train = rng.binomial(1, pi).astype(np.float32)

        mu = mu_coefs[0] * x_train[:, 0] + mu_coefs[1] * x_train[:, 1]
        tau = tau_coefs[0] + tau_coefs[1] * x_train[:, 2]

        noise = rng.normal(size=n) * noise_scale
        y_train = (mu + tau * z_train + noise).astype(np.float32)
        return x_train, pi.astype(np.float32), z_train, y_train, mu, tau, rng

    def test_bcf_save_load_npz(self) -> None:
        """Tests saving and loading a BCF model via NPZ preserves prediction equality."""
        x_train, pihat, z_train, y_train, _, _, _ = self._generate_bcf_data(
            n=200, p=5, seed=0
        )

        model = bcf(
            x_train=x_train,
            y_train=y_train,
            z_train=z_train,
            pihat_train=pihat,
            num_trees_mu=2,
            num_trees_tau=2,
            ndpost=3,
            nskip=2,
            standardize=False,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / 'test_bcf.npz'
            model.save_npz(npz_path)

            # Verify schema_version is present in archive
            with np.load(npz_path) as archive:
                assert 'schema_version' in archive
                assert int(archive['schema_version']) == 1

            loaded_model = bcf.load_npz(npz_path)

            preds_orig = model.predict(x_train, pihat_test=pihat)
            preds_loaded = loaded_model.predict(x_train, pihat_test=pihat)

            assert_allclose(preds_loaded['mu'], preds_orig['mu'], allow_non_scalar=True)
            assert_allclose(
                preds_loaded['tau'], preds_orig['tau'], allow_non_scalar=True
            )

    def test_bcf_save_load_npz_standardized(self) -> None:
        """Tests that saving and loading an auto-standardized model preserves scale metadata and unscaling."""
        x_train, pihat, z_train, y_train, _, _, _ = self._generate_bcf_data(
            n=200, p=5, mu_coefs=(10.0, 5.0), tau_coefs=(4.0, 2.0), seed=1
        )

        model = bcf(
            x_train=x_train,
            y_train=y_train,
            z_train=z_train,
            pihat_train=pihat,
            num_trees_mu=2,
            num_trees_tau=2,
            ndpost=3,
            nskip=2,
            standardize=True,
            seed=42,
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

            preds_orig = model.predict(x_train, pihat_test=pihat)
            preds_loaded = loaded_model.predict(x_train, pihat_test=pihat)

            assert_allclose(preds_loaded['mu'], preds_orig['mu'], allow_non_scalar=True)
            assert_allclose(
                preds_loaded['tau'], preds_orig['tau'], allow_non_scalar=True
            )

    def test_bcf_standardization_equivalence(self) -> None:
        """Tests that automatic standardization is numerically equivalent to manual pre-scaling."""
        x_train, pihat, z_train, y_train, _, _, _ = self._generate_bcf_data(
            n=150, p=4, mu_coefs=(5.0, 3.0), tau_coefs=(2.0, 1.0), seed=42
        )

        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        y_scaled = (y_train - y_mean) / y_std

        # Model 1: Auto standardization on raw y_train
        model_auto = bcf(
            x_train=x_train,
            y_train=y_train,
            z_train=z_train,
            pihat_train=pihat,
            num_trees_mu=5,
            num_trees_tau=5,
            ndpost=10,
            nskip=5,
            standardize=True,
            seed=random.key(100),
        )

        # Model 2: Manual pre-scaling with standardize=False
        model_manual = bcf(
            x_train=x_train,
            y_train=y_scaled,
            z_train=z_train,
            pihat_train=pihat,
            num_trees_mu=5,
            num_trees_tau=5,
            ndpost=10,
            nskip=5,
            standardize=False,
            seed=random.key(100),
        )

        preds_auto = model_auto.predict(x_train, pihat_test=pihat)
        preds_manual = model_manual.predict(x_train, pihat_test=pihat)

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

    def test_bcf_statistical_convergence(self) -> None:
        """Internal reproducibility (jax vs jax) and out-of-sample DGP recovery.

        Two independent bartz chains must agree (Rhat near 1), and a
        prior-matched model must recover the known treatment and prognostic
        effects on held-out data.
        """
        n = 100
        p = 5
        x_train, pi, z_train, y_train, _, _, _ = self._generate_bcf_data(
            n=n,
            p=p,
            mu_coefs=(1.0, 0.5),
            tau_coefs=(1.0, 0.5),
            noise_scale=0.2,
            seed=42,
        )

        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        y_scaled = (y_train - y_mean) / y_std

        # held-out test set from the same DGP for out-of-sample recovery
        x_test, pi_test, _, _, mu_test, tau_test, _ = self._generate_bcf_data(
            n=300,
            p=p,
            mu_coefs=(1.0, 0.5),
            tau_coefs=(1.0, 0.5),
            noise_scale=0.2,
            seed=303,
        )

        ndpost = 2500
        nskip = 1500

        # 1. Internal Stability Models (JAX defaults)
        model_jax_a = bcf(
            x_train=x_train,
            y_train=y_scaled,
            z_train=z_train,
            pihat_train=pi.astype(np.float32),
            num_trees_mu=50,
            num_trees_tau=20,
            ndpost=ndpost,
            nskip=nskip,
            sample_sigma2_leaf_mu=False,
            sample_sigma2_leaf_tau=False,
            seed=random.key(42),
        )

        model_jax_b = bcf(
            x_train=x_train,
            y_train=y_scaled,
            z_train=z_train,
            pihat_train=pi.astype(np.float32),
            num_trees_mu=50,
            num_trees_tau=20,
            ndpost=ndpost,
            nskip=nskip,
            sample_sigma2_leaf_mu=False,
            sample_sigma2_leaf_tau=False,
            seed=random.key(123),
        )

        # 2. Structural Alignment Models (Forced Prior Matching)
        leaf_prior_cov_inv_mu = 50.0
        leaf_prior_cov_inv_tau = 40.0

        model_jax_matched = bcf(
            x_train=x_train,
            y_train=y_scaled,
            z_train=z_train,
            pihat_train=pi.astype(np.float32),
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
            seed=random.key(999),
        )

        preds_a = model_jax_a.predict(x_test=x_train, pihat_test=pi.astype(np.float32))
        preds_b = model_jax_b.predict(x_test=x_train, pihat_test=pi.astype(np.float32))

        # 1. Internal reproducibility: two independent bartz chains agree.
        yhat_a = preds_a['mu'] + preds_a['tau'] * z_train
        yhat_b = preds_b['mu'] + preds_b['tau'] * z_train
        rhat_yhat = _rhat_two_chains(yhat_a, yhat_b)
        stacked_tau = np.stack([preds_a['tau'], preds_b['tau']], axis=0)
        rhat_tau_jax = rhat_rank(stacked_tau, split=True)
        assert np.max(rhat_yhat) < 1.06
        assert np.percentile(rhat_tau_jax, 95) < 1.10

        # 2. Out-of-sample recovery of the known DGP, on held-out data. RMSE
        # (not correlation) catches magnitude/offset errors; a constant tau
        # predictor scores ~0.47, so this requires capturing the heterogeneity.
        preds = model_jax_matched.predict(
            x_test=x_test, pihat_test=pi_test.astype(np.float32)
        )
        tau_hat = np.mean(np.array(preds['tau']) * y_std, axis=0)
        mu_hat = np.mean(np.array(preds['mu']) * y_std + y_mean, axis=0)
        assert np.sqrt(np.mean((tau_hat - tau_test) ** 2)) < 0.35
        assert np.sqrt(np.mean((mu_hat - mu_test) ** 2)) < 0.45

    def test_bcf_null_treatment_effect(self) -> None:
        """Verifies that BCF does not find a treatment effect when tau=0."""
        rng = np.random.default_rng(999)
        n = 200
        p = 5
        x_train = rng.normal(size=(n, p)).astype(np.float32)
        pi = 1.0 / (1.0 + np.exp(-x_train[:, 0]))
        z_train = rng.binomial(1, pi).astype(np.float32)

        mu = x_train[:, 0] + x_train[:, 1]
        y_train = (mu + rng.normal(size=n) * 0.5).astype(np.float32)

        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        y_scaled = (y_train - y_mean) / y_std

        model = bcf(
            x_train=x_train,
            y_train=y_scaled,
            z_train=z_train,
            pihat_train=pi.astype(np.float32),
            num_trees_mu=50,
            num_trees_tau=20,
            ndpost=400,
            nskip=400,
            sample_sigma2_leaf_mu=False,
            sample_sigma2_leaf_tau=False,
            seed=random.key(42),
        )

        x_test = rng.normal(size=(20, p)).astype(np.float32)
        pi_test = 1.0 / (1.0 + np.exp(-x_test[:, 0]))
        preds = model.predict(x_test=x_test, pihat_test=pi_test.astype(np.float32))

        tau_samples = preds['tau'] * y_std
        cate_mean = np.mean(tau_samples, axis=0)

        # Point estimates should be close to 0
        assert np.mean(np.abs(cate_mean)) < 0.2

        # 95% Credible interval should cover 0 for >= 90% of units
        lower_bounds = np.percentile(tau_samples, 2.5, axis=0)
        upper_bounds = np.percentile(tau_samples, 97.5, axis=0)
        contains_zero = (lower_bounds <= 0.0) & (upper_bounds >= 0.0)
        assert np.mean(contains_zero) >= 0.90

    def test_bcf_noise_variance_recovery(self) -> None:
        """Verifies that the BCF model recovers the true residual noise variance."""
        rng = np.random.default_rng(777)
        n = 300
        p = 5
        x_train = rng.normal(size=(n, p)).astype(np.float32)
        pi = 1.0 / (1.0 + np.exp(-x_train[:, 0]))
        z_train = rng.binomial(1, pi).astype(np.float32)

        mu = x_train[:, 0] + x_train[:, 1]
        tau = 1.5 + 0.5 * x_train[:, 2]

        true_sigma = 0.5
        y_train = (mu + tau * z_train + rng.normal(scale=true_sigma, size=n)).astype(
            np.float32
        )

        model = bcf(
            x_train=x_train,
            y_train=y_train,
            z_train=z_train,
            pihat_train=pi.astype(np.float32),
            num_trees_mu=100,
            num_trees_tau=30,
            ndpost=600,
            nskip=400,
            seed=random.key(1234),
        )

        # Recovered noise variance is (1.0 / precision) * y_std^2
        recovered_sigma2 = (1.0 / model._main_trace['mu'].error_cov_inv) * (
            model._y_std**2
        )
        posterior_mean_sigma2 = np.mean(recovered_sigma2)

        # True variance is 0.25 (0.5^2). Check it's close.
        assert np.isclose(posterior_mean_sigma2, 0.25, atol=0.10)

    def test_bcf_one_step_residual_invariant(self) -> None:
        """Verifies that R == y - offset - mu_fit - (tau_0 + tau_fit) * Z."""
        x_train, _, z_train, y_train, _, _, _ = self._generate_bcf_data(n=100, seed=42)

        x_train_t = jnp.asarray(x_train.T)
        binner = UniqueQuantileBinner(x_train_t, key=random.key(1))
        x_binned = binner.bin(x_train_t)
        max_split = binner.max_split

        init_state = init_bcf(
            X_unified=x_binned,
            trt=z_train.astype(bool),
            y=y_train,
            offset=0.0,
            max_split_mu=jnp.array(max_split),
            max_split_tau=jnp.array(max_split),
            num_trees_mu=5,
            num_trees_tau=5,
            p_nonterminal_mu=np.ones(5, dtype=np.float32) * 0.95,
            p_nonterminal_tau=np.ones(5, dtype=np.float32) * 0.95,
            leaf_prior_cov_inv_mu=1.0,
            leaf_prior_cov_inv_tau=1.0,
            error_cov_inv=Wishart(
                nu=jnp.float32(1.0),
                rate=jnp.array(1.0, dtype=jnp.float32),
                value=jnp.array(1.0, dtype=jnp.float32),
            ),
        )

        # `resid` is stored scaled; copy `resid_unit` out before the step, which
        # donates the buffer that the returned state shares.
        resid_unit = jnp.copy(init_state.resid_unit)

        new_state = bcf_step(random.key(2), init_state)

        mu_fit_raw = evaluate_forest(new_state.X, new_state.forest).sum(axis=0)
        mu_fit = (
            mu_fit_raw
            if new_state.inv_sdev_scale is None
            else mu_fit_raw / new_state.inv_sdev_scale
        )

        tau_fit_raw = evaluate_forest(new_state.X, new_state.forest_tau).sum(axis=0)

        b_z = jnp.where(z_train == 1, new_state.b1, new_state.b0)
        expected_resid = (
            y_train
            - new_state.forest.offset
            - mu_fit
            - b_z * (new_state.tau_0 + tau_fit_raw)
        )

        # `resid` is stored scaled (``resid_unit * resid = data residual``)
        assert_allclose(
            new_state.resid * resid_unit,
            expected_resid,
            atol=1e-5,
            allow_non_scalar=True,
        )

    def test_bcf_unsplittable_x_reduction(self) -> None:
        """Verifies BCF degenerates to Bayesian linear regression when max_split is 0."""
        rng = np.random.default_rng(2)
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
            seed=random.key(3),
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

    def test_bcf_adaptive_coding(self) -> None:
        """Adaptive coding recovers the known treatment effect out of sample."""
        n = 100
        p = 5
        x_train, pi, z_train, y_train, _, _, _ = self._generate_bcf_data(
            n=n,
            p=p,
            mu_coefs=(1.0, 0.5),
            tau_coefs=(1.0, 0.5),
            noise_scale=0.2,
            seed=100,
        )

        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        y_scaled = (y_train - y_mean) / y_std

        # held-out test set from the same DGP for out-of-sample recovery
        x_test, pi_test, _, _, mu_test, tau_test, _ = self._generate_bcf_data(
            n=300,
            p=p,
            mu_coefs=(1.0, 0.5),
            tau_coefs=(1.0, 0.5),
            noise_scale=0.2,
            seed=404,
        )

        ndpost = 1000
        nskip = 500

        leaf_prior_cov_inv_mu = 50.0
        leaf_prior_cov_inv_tau = 40.0

        model_jax = bcf(
            x_train=x_train,
            y_train=y_scaled,
            z_train=z_train,
            pihat_train=pi.astype(np.float32),
            num_trees_mu=50,
            num_trees_tau=20,
            ndpost=ndpost,
            nskip=nskip,
            leaf_prior_cov_inv_mu=leaf_prior_cov_inv_mu,
            leaf_prior_cov_inv_tau=leaf_prior_cov_inv_tau,
            sigma_df=0.0,
            sigma_scale=0.0,
            adaptive_coding=True,
            seed=random.key(123),
        )

        preds = model_jax.predict(x_test=x_test, pihat_test=pi_test.astype(np.float32))
        cate_hat = np.mean(np.array(preds['tau']) * y_std, axis=0)
        mu_hat = np.mean(np.array(preds['mu']) * y_std + y_mean, axis=0)
        assert np.sqrt(np.mean((cate_hat - tau_test) ** 2)) < 0.4
        assert np.sqrt(np.mean((mu_hat - mu_test) ** 2)) < 0.4

    def test_bcf_leaf_variance_prior_inactive(self) -> None:
        """Verifies that sample_sigma2_leaf=False keeps the prior variance fixed."""
        n = 50
        p = 3
        x_train, pi, z_train, y_train, _, _, _ = self._generate_bcf_data(
            n=n, p=p, seed=42
        )

        y_scaled = (y_train - np.mean(y_train)) / np.std(y_train)

        model = bcf(
            x_train=x_train,
            y_train=y_scaled,
            z_train=z_train,
            pihat_train=pi.astype(np.float32),
            num_trees_mu=10,
            num_trees_tau=5,
            ndpost=10,
            nskip=0,
            sample_sigma2_leaf_mu=False,
            sample_sigma2_leaf_tau=False,
            seed=random.key(42),
        )

        # Check that trace variances are constant
        mu_prior_vars = np.array(model._leaf_prior_cov_inv_mu_trace)
        tau_prior_vars = np.array(model._leaf_prior_cov_inv_tau_trace)

        # Assert variance across the chain (axis 0) is 0
        assert_allclose(np.var(mu_prior_vars, axis=0), 0.0, atol=1e-7)
        assert_allclose(np.var(tau_prior_vars, axis=0), 0.0, atol=1e-7)

        # Compare to active
        model_active = bcf(
            x_train=x_train,
            y_train=y_scaled,
            z_train=z_train,
            pihat_train=pi.astype(np.float32),
            num_trees_mu=10,
            num_trees_tau=5,
            ndpost=10,
            nskip=0,
            sample_sigma2_leaf_mu=True,
            sample_sigma2_leaf_tau=True,
            seed=random.key(42),
        )

        mu_prior_vars_active = np.array(model_active._leaf_prior_cov_inv_mu_trace)
        tau_prior_vars_active = np.array(model_active._leaf_prior_cov_inv_tau_trace)

        assert np.var(mu_prior_vars_active, axis=0).mean() > 1e-4
        assert np.var(tau_prior_vars_active, axis=0).mean() > 1e-4

    def test_bcf_leaf_variance_prior_active_equivalence(self) -> None:
        """Adaptive leaf variance matches StochTree's scale and recovers the DGP."""
        n = 500
        p = 5
        x_train, pi, z_train, y_train, _, _, _ = self._generate_bcf_data(
            n=n,
            p=p,
            mu_coefs=(1.0, 0.5),
            tau_coefs=(1.0, 0.5),
            noise_scale=0.2,
            seed=100,
        )

        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        y_scaled = (y_train - y_mean) / y_std

        # held-out test set from the same DGP for out-of-sample recovery
        x_test, pi_test, _, _, mu_test, tau_test, _ = self._generate_bcf_data(
            n=300,
            p=p,
            mu_coefs=(1.0, 0.5),
            tau_coefs=(1.0, 0.5),
            noise_scale=0.2,
            seed=202,
        )

        ndpost = 1500
        nskip = 1000

        model_jax = bcf(
            x_train=x_train,
            y_train=y_scaled,
            z_train=z_train,
            pihat_train=pi.astype(np.float32),
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
            seed=random.key(123),
        )

        model_st = stochtree.BCFModel()
        model_st.sample(
            X_train=x_train,
            Z_train=z_train,
            y_train=y_train.astype(np.float64),
            propensity_train=pi.astype(np.float32),
            num_mcmc=ndpost,
            num_gfr=0,
            num_burnin=nskip,
            prognostic_forest_params={
                'num_trees': 200,
                'sample_sigma2_leaf': True,
                'sigma2_leaf_shape': 3.0,
                'sigma2_leaf_scale': 4.0 / 200.0,
                'min_samples_leaf': 3,
            },
            treatment_effect_forest_params={
                'num_trees': 50,
                'sample_sigma2_leaf': True,
                'sigma2_leaf_shape': 3.0,
                'sigma2_leaf_scale': 1.0 / 50.0,
                'sample_intercept': True,
                'min_samples_leaf': 3,
            },
            general_params={
                'adaptive_coding': False,
                'random_seed': 42,
                'control_coding_init': 0.0,
                'treated_coding_init': 1.0,
            },
        )

        y_var = np.var(y_train)

        leaf_var_mu_jax = (
            np.mean(1.0 / np.array(model_jax._leaf_prior_cov_inv_mu_trace)) * y_var
        )
        leaf_var_mu_st = np.mean(model_st.leaf_scale_mu_samples) * y_var
        leaf_var_tau_jax = (
            np.mean(1.0 / np.array(model_jax._leaf_prior_cov_inv_tau_trace)) * y_var
        )
        leaf_var_tau_st = np.mean(model_st.leaf_scale_tau_samples) * y_var
        assert_allclose(leaf_var_mu_jax, leaf_var_mu_st, rtol=0.3)
        assert_allclose(leaf_var_tau_jax, leaf_var_tau_st, rtol=0.3)

        preds = model_jax.predict(x_test=x_test, pihat_test=pi_test.astype(np.float32))
        tau_hat = np.mean(np.array(preds['tau']) * y_std, axis=0)
        mu_hat = np.mean(np.array(preds['mu']) * y_std + y_mean, axis=0)
        assert np.sqrt(np.mean((tau_hat - tau_test) ** 2)) < 0.15
        assert np.sqrt(np.mean((mu_hat - mu_test) ** 2)) < 0.15

    def test_predict_potential_outcomes(self) -> None:
        """Tests posterior predictive potential outcome sampling in BCF."""
        x_train, pihat, z_train, y_train, _, _, _ = self._generate_bcf_data(
            n=150, p=5, seed=42
        )

        model = bcf(
            x_train=x_train,
            y_train=y_train,
            z_train=z_train,
            pihat_train=pihat,
            num_trees_mu=5,
            num_trees_tau=5,
            ndpost=20,
            nskip=10,
            standardize=True,
            seed=123,
        )

        # 1. Verify sigma_trace
        # with self.subTest(name='sigma_trace'):
        sigma = model.sigma_trace
        assert sigma.shape == (20,)
        assert bool(jnp.all(sigma > 0.0))

        x_test = x_train[:30]
        pihat_test = pihat[:30]

        # 2. Test shapes, keys, and realized lift consistency
        # with self.subTest(name='shapes_and_lift_consistency'):
        res = model.predict_potential_outcomes(
            x_test=x_test, pihat_test=pihat_test, rho=0.5, key=random.key(42)
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
            x_test=x_test, pihat_test=pihat_test, rho=1.0, key=random.key(42)
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
            x_test=x_test, pihat_test=pihat_test, rho=0.0, key=random.key(42)
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
            x_test=x_test, pihat_test=pihat_test, key=7
        )
        assert res_key_none['y0'].shape == res_key_int['y0'].shape

    def test_bcf_binary_model(self) -> None:
        """Tests binary BCF end-to-end: initialization, offset, and predictions."""
        # Generate data with non-trivial positive rate (~70% positive)
        x_train, pihat, z_train, y_continuous, _, _, _ = self._generate_bcf_data(
            n=200, p=5, seed=0
        )
        y_train = (y_continuous > np.percentile(y_continuous, 30)).astype(np.float32)

        model = bcf(
            x_train=x_train,
            y_train=y_train,
            z_train=z_train,
            pihat_train=pihat,
            num_trees_mu=5,
            num_trees_tau=5,
            ndpost=10,
            nskip=5,
            outcome_type='binary',
            delta_max=0.9,
            seed=random.key(42),
        )

        # Sub-test 1: Verify initialization and offset correctness
        # with self.subTest(name='offset_and_scaling_invariants'):
        assert model._y_std == 1.0
        assert model._y_mean == 0.0
        expected_offset = stats.norm.ppf(np.mean(y_train))
        assert np.isclose(model._offset, expected_offset, atol=0.0001)

        # Sub-test 2: Verify prediction keys and valid probability bounds
        # with self.subTest(name='prediction_probabilities'):
        preds = model.predict(x_train, pihat_test=pihat)
        assert 'mu' in preds
        assert 'tau' in preds
        assert 'tau_prob' in preds
        assert 'p1' in preds
        assert 'p0' in preds

        # Check that raw probabilities are within [0, 1]
        assert np.all((preds['p0'] >= 0.0) & (preds['p0'] <= 1.0))
        assert np.all((preds['p1'] >= 0.0) & (preds['p1'] <= 1.0))

        # Sub-test 3: potential outcomes on a binary model return 0/1 labels
        po = model.predict_potential_outcomes(x_train, pihat_test=pihat, key=0)
        assert np.isin(np.array(po['y0']), (0.0, 1.0)).all()
        assert np.isin(np.array(po['y1']), (0.0, 1.0)).all()

    def test_bcf_binary_requires_0_1(self) -> None:
        """Binary BCF rejects outcomes that are not 0/1."""
        x_train, pihat, z_train, _, _, _, _ = self._generate_bcf_data(n=20, seed=0)
        with pytest.raises(ValueError, match='strictly 0 or 1'):
            bcf(
                x_train=x_train,
                y_train=np.full(20, 2.0, np.float32),
                z_train=z_train,
                pihat_train=pihat,
                outcome_type='binary',
                num_trees_mu=2,
                num_trees_tau=2,
                ndpost=1,
                nskip=0,
                seed=42,
            )

    def test_bcf_treatment_requires_0_1(self) -> None:
        """BCF rejects treatments that are not 0/1."""
        x_train, pihat, _, y_train, _, _, _ = self._generate_bcf_data(n=20, seed=0)
        with pytest.raises(EquinoxRuntimeError, match='must be 0 or 1'):
            bcf(
                x_train=x_train,
                y_train=y_train,
                z_train=np.full(20, 0.5, np.float32),
                pihat_train=pihat,
                num_trees_mu=2,
                num_trees_tau=2,
                ndpost=1,
                nskip=0,
                seed=42,
            )

    def test_bcf_constructor_options(self) -> None:
        """Constructor x_test/z_test, pihat toggle, explicit tau_0 prior, sigma_trace."""
        x_train, pihat, z_train, y_train, _, _, _ = self._generate_bcf_data(
            n=30, seed=0
        )
        x_test, pihat_test, z_test, _, _, _, _ = self._generate_bcf_data(n=15, seed=1)
        ndpost = 3
        model = bcf(
            x_train=x_train,
            y_train=y_train,
            z_train=z_train,
            pihat_train=pihat,
            x_test=x_test,
            z_test=z_test,
            pihat_test=pihat_test,
            include_pihat_in_mu=False,
            tau_0_prior_var=0.5,
            standardize=False,
            num_trees_mu=2,
            num_trees_tau=2,
            ndpost=ndpost,
            nskip=1,
            seed=42,
        )
        assert model._mcmc_state.num_chains() is None
        assert model.sigma_trace.shape == (ndpost,)

    def test_bcf_x_test_format_mismatch(self) -> None:
        """x_test format must match x_train, at construction and at predict."""
        x_train, pihat, z_train, y_train, _, _, _ = self._generate_bcf_data(
            n=30, seed=0
        )
        x_test_df = pd.DataFrame(np.asarray(x_train)[:15])
        with pytest.raises(ValueError, match='does not match x_train'):
            bcf(
                x_train=x_train,
                y_train=y_train,
                z_train=z_train,
                pihat_train=pihat,
                x_test=x_test_df,
                num_trees_mu=2,
                num_trees_tau=2,
                ndpost=2,
                nskip=1,
                seed=42,
            )
        model = bcf(
            x_train=x_train,
            y_train=y_train,
            z_train=z_train,
            pihat_train=pihat,
            num_trees_mu=2,
            num_trees_tau=2,
            ndpost=2,
            nskip=1,
            seed=42,
        )
        with pytest.raises(ValueError, match='does not match x_train'):
            model.predict(x_test_df)
