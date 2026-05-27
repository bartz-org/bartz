# bartz/src/bartz/stochtree/_stochtree.py
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

"""Implement class `BARTModel` that mimics the Python package stochtree."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Literal, TypeVar

import numpy as np
from jax import numpy as jnp
from jax.scipy.special import ndtr
from jaxtyping import Array, Float, Float32
from scipy.stats import norm

from bartz._interface import Bart, DataFrame, PredictKind
from bartz.prepcovars import UniqueQuantileBinner

_T = TypeVar('_T')


@dataclass
class OutcomeModel:
    """Outcome model specification, matching `stochtree.OutcomeModel`.

    Parameters
    ----------
    outcome
        Outcome family. ``'continuous'`` or ``'binary'``.
    link
        Link function. ``'identity'`` for continuous, ``'probit'`` for binary.
    """

    outcome: Literal['continuous', 'binary'] = 'continuous'
    link: Literal['identity', 'probit'] = 'identity'


class NotSampledError(RuntimeError):
    """Raised when calling a method that requires `sample` to have been called."""


@dataclass(kw_only=True)
class _GeneralParams:
    """Mirror of stochtree's ``general_params`` dict, with the keys bartz handles.

    Parameters
    ----------
    cutpoint_grid_size
        Maximum number of cutpoints to consider for each feature.
    standardize
        Whether to standardize the outcome before fitting (continuous case
        only; ignored for probit binary).
    sigma2_init
        Starting value of global variance parameter. If `None`, bartz auto-
        calibrates via its ``sigest='auto'`` heuristic.
    sigma2_global_shape
        Shape parameter of the inverse-gamma prior on the global error
        variance. Stochtree's default ``0`` produces an improper prior; in
        that case bartz falls back to ``sigdf=3`` / ``sigquant=0.9``.
    sigma2_global_scale
        Scale parameter of the inverse-gamma prior on the global error
        variance.
    variable_weights
        Per-predictor sampling weights. Must be strictly positive (bartz does
        not allow zero weights); pass a small positive value to suppress a
        variable.
    random_seed
        Seed for the random number generator.
    keep_every
        Thinning factor for retained MCMC samples.
    num_chains
        Number of independent MCMC chains.
    outcome_model
        Outcome family / link specification. Only ``('continuous','identity')``
        and ``('binary','probit')`` are supported.
    probit_outcome_model
        Deprecated alias of ``outcome_model=OutcomeModel('binary','probit')``.
    """

    cutpoint_grid_size: int = 100
    standardize: bool = True
    sigma2_init: float | None = None
    sigma2_global_shape: float = 0
    sigma2_global_scale: float = 0
    variable_weights: np.ndarray | None = None
    random_seed: int | None = None
    keep_every: int = 1
    num_chains: int = 1
    outcome_model: OutcomeModel | None = None
    probit_outcome_model: bool = False


@dataclass(kw_only=True)
class _MeanForestParams:
    """Mirror of stochtree's ``mean_forest_params`` dict, restricted to the keys bartz handles.

    Parameters
    ----------
    num_trees
        Number of trees in the conditional mean ensemble.
    alpha
        Tree split prior base (``base`` in bartz; stochtree's name is
        ``alpha``).
    beta
        Tree split prior decay (``power`` in bartz; stochtree's name is
        ``beta``).
    min_samples_leaf
        Minimum number of training samples at a leaf.
    max_depth
        Maximum tree depth. Stochtree convention: ``-1`` means unbounded.
    sample_sigma2_leaf
        Stochtree default is ``True`` (sample the leaf-variance prior). bartz
        uses a fixed leaf-variance prior; ``True`` is rejected in
        `__post_init__`. To use this shim, pass
        ``mean_forest_params={'sample_sigma2_leaf': False}`` (matching the
        bartz behavior).
    sigma2_leaf_init
        Initial leaf-variance prior. If `None`, defaults to
        ``1 / num_trees`` (matching stochtree).
    """

    num_trees: int = 200
    alpha: float = 0.95
    beta: float = 2.0
    min_samples_leaf: int = 5
    max_depth: int = 10
    sample_sigma2_leaf: bool = True
    sigma2_leaf_init: float | None = None

    def __post_init__(self) -> None:
        if self.sample_sigma2_leaf:
            msg = (
                'sample_sigma2_leaf=True is not supported (bartz uses a fixed'
                " leaf-variance prior); pass mean_forest_params={'sample_sigma2_leaf':"
                ' False} to acknowledge this.'
            )
            raise NotImplementedError(msg)


def _build_dataclass(cls: type[_T], params: dict | None, name: str) -> _T:
    """Convert a user-supplied dict to a dataclass, with friendly errors."""
    if params is None:
        params = {}
    allowed = {f.name for f in fields(cls)}
    extra = set(params) - allowed
    if extra:
        msg = (
            f'{name} contains unsupported key(s) {sorted(extra)}; valid keys'
            f' are {sorted(allowed)}'
        )
        raise ValueError(msg)
    return cls(**params)


class BARTModel:
    R"""
    BART model with a `stochtree`-compatible interface, powered by bartz.

    This class mimics `stochtree.BARTModel` so that bartz can be used as a
    drop-in reference implementation for testing. The intersection of features
    is targeted: continuous regression (Gaussian outcome, identity link) and
    binary classification (probit link) on tabular covariates.

    Use the same idiomatic pattern as `stochtree.BARTModel`::

        m = BARTModel()
        m.sample(
            X_train=X, y_train=y, X_test=X_test,
            num_gfr=0, num_mcmc=200,
            mean_forest_params={'sample_sigma2_leaf': False},
        )
        yhat = m.predict(X_new, terms='y_hat', type='mean')

    Notes
    -----
    Differences from `stochtree`, by design:

    - ``num_gfr`` has no default and must be set explicitly to ``0`` (bartz
      has no grow-from-root sampler).
    - ``mean_forest_params['sample_sigma2_leaf']`` is rejected when ``True``
      (bartz uses a fixed leaf-variance prior); pass ``False`` explicitly.
    - The stochtree arguments that select unsupported behavior — leaf-basis
      regression, random effects, heteroskedastic variance forests,
      warm-starting from a previous model — are removed from the signatures
      rather than raising at runtime.
    - With ``sigma2_global_shape == 0`` and ``sigma2_global_scale == 0``
      (stochtree's improper-prior default), bartz falls back to its
      ``sigest='auto'`` / ``sigdf=3`` / ``sigquant=0.9`` automatic
      calibration; pass explicit positive values to align both packages.
    - bartz uses single-precision floats, so outputs differ from stochtree at
      the float32 precision level.

    References
    ----------
    Krantsevich, N., He, J., Hahn, P. R. (2023). "stochtree: Stochastic Tree
    Ensembles in Python".
    """

    # public, set by sample()
    sampled: bool
    standardize: bool
    sample_sigma2_global: bool
    probit_outcome_model: bool
    outcome_model: OutcomeModel
    num_gfr: int
    num_burnin: int
    num_mcmc: int
    num_chains: int
    num_samples: int
    sigma2_init: float | None
    y_bar: float
    y_std: float
    has_rfx: bool
    include_mean_forest: bool
    include_variance_forest: bool

    y_hat_train: Float32[Array, 'n num_samples']
    global_var_samples: Float32[Array, ' num_samples']

    y_hat_test: Float32[Array, 'm num_samples'] | None

    _bart: Bart

    def __init__(self) -> None:
        self.sampled = False

    def is_sampled(self) -> bool:
        """Return whether `sample` has been called."""
        return self.sampled

    def sample(
        self,
        X_train: np.ndarray | DataFrame,
        y_train: np.ndarray,
        X_test: np.ndarray | DataFrame | None = None,
        observation_weights: np.ndarray | None = None,
        *,
        num_gfr: int,
        num_burnin: int = 0,
        num_mcmc: int = 100,
        general_params: dict[str, Any] | None = None,
        mean_forest_params: dict[str, Any] | None = None,
    ) -> None:
        """Fit the model.

        The signature mirrors `stochtree.BARTModel.sample`, with the
        unsupported keyword arguments (``leaf_basis_train``,
        ``rfx_group_ids_train``, ``previous_model_json``,
        ``variance_forest_params``, ``random_effects_params`` and their
        siblings) removed instead of raising at runtime.

        Parameters
        ----------
        X_train
            Training covariates with shape ``(n, p)``.
        y_train
            Training outcomes of length ``n``.
        X_test
            Optional test covariates; if given, predictions are cached on
            them in `y_hat_test`.
        observation_weights
            Optional positive per-observation weights scaling the residual
            variance (``y_i | - ~ N(mu(X_i), sigma^2 / w_i)``).
        num_gfr
            Number of grow-from-root iterations. Must be ``0``.
        num_burnin
            Number of MCMC burn-in iterations.
        num_mcmc
            Number of retained MCMC iterations per chain.
        general_params
            Optional override for the keys of `_GeneralParams`.
        mean_forest_params
            Override for the keys of `_MeanForestParams`. Must explicitly
            disable ``sample_sigma2_leaf``.

        Raises
        ------
        NotImplementedError
            If ``num_gfr`` is non-zero, since bartz has no grow-from-root
            sampler.
        """
        if num_gfr != 0:
            msg = (
                'num_gfr must be 0; the grow-from-root sampler is not available'
                ' in bartz.'
            )
            raise NotImplementedError(msg)

        gp = _build_dataclass(_GeneralParams, general_params, 'general_params')
        mfp = _build_dataclass(
            _MeanForestParams, mean_forest_params, 'mean_forest_params'
        )

        outcome_model = _resolve_outcome_model(
            gp.outcome_model, gp.probit_outcome_model
        )
        is_probit = outcome_model.outcome == 'binary'

        X_train_np, y_train_np = _process_train_inputs(X_train, y_train)
        _, p = X_train_np.shape

        # standardization matches stochtree's y_bar / y_std logic
        standardize = gp.standardize
        y_bar, y_std, y_for_bartz = _standardize_y(y_train_np, is_probit, standardize)

        num_trees = int(mfp.num_trees)
        num_chains = int(gp.num_chains)
        bart_num_chains = None if num_chains == 1 else num_chains

        # leaf-prior: bartz uses sigma_mu = tau_num / (k * sqrt(num_trees));
        # stochtree's sigma2_leaf is the leaf-variance prior. Hold k=2 and solve
        # for tau_num so that the two parameterizations agree.
        sigma2_leaf_init = mfp.sigma2_leaf_init
        if sigma2_leaf_init is None:
            sigma2_leaf_init = 1.0 / num_trees
        bartz_k = 2.0
        tau_num_arg = bartz_k * math.sqrt(num_trees * float(sigma2_leaf_init))

        sigma2_init = gp.sigma2_init
        sigdf, lambda_, sigest_arg = _resolve_variance_prior(
            float(gp.sigma2_global_shape), float(gp.sigma2_global_scale), sigma2_init
        )

        binner = partial(
            UniqueQuantileBinner,
            max_bins=int(gp.cutpoint_grid_size) + 1,
            max_subsample=None,
        )

        variable_weights = _check_variable_weights(gp.variable_weights, p)

        # stochtree's max_depth == -1 means unbounded; bartz uses a 1-based limit
        max_depth = int(mfp.max_depth)
        max_depth_arg = 30 if max_depth < 0 else max_depth + 1

        seed = 0 if gp.random_seed is None else int(gp.random_seed)
        w_arg = (
            None
            if observation_weights is None
            else jnp.asarray(observation_weights, jnp.float32)
        )

        self._bart = Bart(
            x_train=jnp.asarray(X_train_np.T),
            y_train=jnp.asarray(y_for_bartz),
            outcome_type='binary' if is_probit else 'continuous',
            binner=binner,
            varprob=variable_weights,
            sigest=sigest_arg,
            sigdf=sigdf,
            sigquant=0.9,
            k=bartz_k,
            power=float(mfp.beta),
            base=float(mfp.alpha),
            lambda_=lambda_,
            tau_num=tau_num_arg,
            w=w_arg,
            num_trees=num_trees,
            n_save=int(num_mcmc),
            n_burn=int(num_burnin),
            n_skip=int(gp.keep_every),
            printevery=None,
            num_chains=bart_num_chains,
            seed=seed,
            maxdepth=max_depth_arg,
            init_kw={'min_points_per_leaf': int(mfp.min_samples_leaf)},
        )
        self._finalize_sample(
            outcome_model=outcome_model,
            num_burnin=int(num_burnin),
            num_mcmc=int(num_mcmc),
            num_chains=num_chains,
            sigma2_init=sigma2_init,
            y_bar=y_bar,
            y_std=y_std,
            standardize=standardize,
            X_test=X_test,
        )

    def _finalize_sample(
        self,
        *,
        outcome_model: OutcomeModel,
        num_burnin: int,
        num_mcmc: int,
        num_chains: int,
        sigma2_init: float | None,
        y_bar: float,
        y_std: float,
        standardize: bool,
        X_test: np.ndarray | DataFrame | None,
    ) -> None:
        """Populate the public attributes after `_bart` has been constructed."""
        is_probit = outcome_model.outcome == 'binary'
        self.sampled = True
        self.standardize = standardize
        self.sample_sigma2_global = True
        self.probit_outcome_model = is_probit
        self.outcome_model = outcome_model
        self.num_gfr = 0
        self.num_burnin = num_burnin
        self.num_mcmc = num_mcmc
        self.num_chains = num_chains
        self.num_samples = num_mcmc * num_chains
        self.sigma2_init = sigma2_init
        self.y_bar = y_bar
        self.y_std = y_std
        self.has_rfx = False
        self.include_mean_forest = True
        self.include_variance_forest = False

        # cached outputs in stochtree's (n, num_samples) layout, original scale
        self.y_hat_train = self._predict_y_hat_internal('train')
        if X_test is not None:
            X_test_np = _check_X(X_test)
            self.y_hat_test = self._predict_y_hat_internal(jnp.asarray(X_test_np.T))
        else:
            self.y_hat_test = None

        if is_probit:
            self.global_var_samples = jnp.ones((self.num_samples,), dtype=jnp.float32)
        else:
            sigma = self._bart.get_error_sdev()
            self.global_var_samples = (sigma * y_std) ** 2

    def predict(
        self,
        X: np.ndarray | DataFrame,
        *,
        type: str = 'posterior',  # noqa: A002
        terms: str | Sequence[str] = 'all',
        scale: str = 'linear',
    ) -> np.ndarray | dict[str, np.ndarray | None]:
        """Predict at new covariates.

        Parameters
        ----------
        X
            New covariates with shape ``(m, p)``.
        type
            ``'posterior'`` returns one prediction per posterior sample, with
            shape ``(m, num_samples)``. ``'mean'`` averages the posterior
            samples, returning a vector of shape ``(m,)``.
        terms
            One of ``'y_hat'``, ``'mean_forest'``, ``'all'``, or a list. Since
            random effects and a variance forest are not supported, ``'y_hat'``
            and ``'mean_forest'`` produce the same result; ``'rfx'`` and
            ``'variance_forest'`` are accepted (matching stochtree) but the
            corresponding entries are `None`.
        scale
            For probit binary regression: ``'linear'`` returns the eta values,
            ``'probability'`` returns ``Phi(eta)``, ``'class'`` returns 0 / 1.
            Only ``'linear'`` is valid for continuous outcomes.

        Returns
        -------
        Either a single numpy array (for a single requested term) or a dict
        keyed by term name (matching stochtree's behavior).

        Raises
        ------
        NotSampledError
            If `sample` has not been called yet.
        """
        if not self.sampled:
            msg = (
                "This BARTModel instance is not fitted yet. Call 'sample' before"
                ' using this model.'
            )
            raise NotSampledError(msg)
        terms_list = _check_predict_args(type, scale, terms, self.probit_outcome_model)

        X_np = _check_X(X)
        x_bartz = jnp.asarray(X_np.T)
        pred = self._predict_y_hat_internal(x_bartz)

        if self.probit_outcome_model and scale in ('probability', 'class'):
            prob = ndtr(pred)
            pred_out = jnp.where(prob < 0.5, 0, 1) if scale == 'class' else prob
        else:
            pred_out = pred

        if type == 'mean':
            pred_out = jnp.mean(pred_out, axis=1)

        wants_y_hat = ('y_hat' in terms_list) or ('all' in terms_list)
        wants_mean_forest = ('mean_forest' in terms_list) or ('all' in terms_list)
        single = sum([wants_y_hat, wants_mean_forest]) == 1
        if single:
            return pred_out
        return {
            'y_hat': pred_out if wants_y_hat else None,
            'mean_forest_predictions': pred_out if wants_mean_forest else None,
            'rfx_predictions': None,
            'variance_forest_predictions': None,
        }

    def _predict_y_hat_internal(
        self, x: Float[Array, 'p m'] | Literal['train']
    ) -> Float32[Array, 'm num_samples']:
        """Return predictions on the original outcome scale, layout ``(m, num_samples)``."""
        latent = self._bart.predict(x, kind=PredictKind.latent_samples)
        if self.probit_outcome_model:
            # bartz integrates the binary offset into latent; result already on probit scale.
            return latent.T
        if self.standardize:
            return (latent * self.y_std + self.y_bar).T
        return latent.T


def _resolve_outcome_model(
    outcome_model: object, probit_outcome_model: bool
) -> OutcomeModel:
    if outcome_model is not None:
        outcome = getattr(outcome_model, 'outcome', None)
        link = getattr(outcome_model, 'link', None)
        if outcome is None or link is None:
            msg = 'outcome_model must have .outcome and .link attributes'
            raise TypeError(msg)
        if (outcome, link) not in (('continuous', 'identity'), ('binary', 'probit')):
            msg = (
                f'unsupported outcome_model (outcome={outcome!r}, link={link!r});'
                " only ('continuous', 'identity') and ('binary', 'probit') are"
                ' supported.'
            )
            raise NotImplementedError(msg)
        return OutcomeModel(outcome=outcome, link=link)
    if probit_outcome_model:
        return OutcomeModel(outcome='binary', link='probit')
    return OutcomeModel(outcome='continuous', link='identity')


def _standardize_y(
    y_train_np: np.ndarray, is_probit: bool, standardize: bool
) -> tuple[float, float, np.ndarray]:
    """Return ``(y_bar, y_std, y_for_bartz)`` matching stochtree's standardization."""
    if is_probit:
        (n,) = y_train_np.shape
        mean01 = float(np.mean(y_train_np != 0))
        mean01 = min(max(mean01, 1.0 / (n + 1)), n / (n + 1))
        y_bar = float(norm.ppf(mean01))
        return y_bar, 1.0, (y_train_np != 0).astype(np.float32)
    if standardize:
        y_bar = float(np.mean(y_train_np))
        y_std_val = float(np.std(y_train_np))
        y_std = y_std_val if y_std_val > 0 else 1.0
        return y_bar, y_std, ((y_train_np - y_bar) / y_std).astype(np.float32)
    return 0.0, 1.0, y_train_np.astype(np.float32)


def _resolve_variance_prior(
    shape: float, scale: float, sigma2_init: float | None
) -> tuple[float, float | None, float | Literal['auto']]:
    """Translate stochtree's IG(shape, scale) prior to bartz's scaled-inv-chi2.

    Parameters
    ----------
    shape
        Stochtree's ``sigma2_global_shape``.
    scale
        Stochtree's ``sigma2_global_scale``.
    sigma2_init
        Stochtree's ``sigma2_init`` (used only when the prior is improper).

    Returns
    -------
    ``(sigdf, lambda_, sigest_arg)`` for bartz.
    """
    # IG(shape, scale) <=> scaled-inv-chi2(df=2*shape, lambda=scale/shape)
    if shape > 0 and scale > 0:
        # bartz rejects sigest when lambda_ is set; stochtree's sigma2_init is
        # the chain starting value, not the prior scale, so we drop it here.
        return 2.0 * shape, scale / shape, 'auto'
    sigest_arg: float | Literal['auto'] = (
        math.sqrt(float(sigma2_init)) if sigma2_init is not None else 'auto'
    )
    return 3.0, None, sigest_arg


def _check_variable_weights(
    variable_weights: np.ndarray | None, p: int
) -> Float32[Array, ' p'] | None:
    """Validate `variable_weights`, returning the jax array (or None)."""
    if variable_weights is None:
        return None
    arr = jnp.asarray(variable_weights, jnp.float32)
    if arr.shape != (p,):
        msg = f'variable_weights must have shape (p,)=({p},), got {arr.shape}'
        raise ValueError(msg)
    if jnp.any(arr <= 0).item():
        msg = (
            'variable_weights must be strictly positive (bartz does not'
            ' allow zero weights); use a small value to suppress a'
            ' variable.'
        )
        raise ValueError(msg)
    return arr


def _check_predict_args(
    type_: str, scale: str, terms: str | Sequence[str], probit_outcome_model: bool
) -> list[str]:
    """Validate `BARTModel.predict` arguments, returning the normalized terms list."""
    if scale not in ('linear', 'probability', 'class'):
        msg = f"scale must be 'linear', 'probability', or 'class'; got {scale!r}"
        raise ValueError(msg)
    if type_ not in ('posterior', 'mean'):
        msg = f"type must be 'posterior' or 'mean'; got {type_!r}"
        raise ValueError(msg)
    if not probit_outcome_model and scale != 'linear':
        msg = (
            "scale must be 'linear' for non-probit (continuous) regression;"
            f' got {scale!r}'
        )
        raise ValueError(msg)
    if type_ == 'mean' and scale == 'class':
        msg = "scale='class' is incompatible with type='mean'"
        raise ValueError(msg)
    terms_list = [terms] if isinstance(terms, str) else list(terms)
    for t in terms_list:
        if t not in ('y_hat', 'mean_forest', 'rfx', 'variance_forest', 'all'):
            msg = (
                f'unknown term {t!r}; valid terms are y_hat, mean_forest,'
                ' rfx, variance_forest, all'
            )
            raise ValueError(msg)
    return terms_list


def _process_train_inputs(
    X_train: np.ndarray | DataFrame, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(X_train, 'to_numpy') and hasattr(X_train, 'columns'):
        X_np = np.asarray(X_train.to_numpy())
    else:
        X_np = np.asarray(X_train)
    if X_np.ndim == 1:
        X_np = X_np[:, None]
    if X_np.ndim != 2:
        msg = f'X_train must be 2D (n, p); got shape {X_np.shape}'
        raise ValueError(msg)
    y_np = np.asarray(y_train)
    if y_np.ndim != 1:
        msg = f'y_train must be 1D (n,); got shape {y_np.shape}'
        raise ValueError(msg)
    if y_np.shape[0] != X_np.shape[0]:
        msg = (
            f'X_train and y_train length mismatch: X_train has '
            f'{X_np.shape[0]} rows, y_train has {y_np.shape[0]} entries'
        )
        raise ValueError(msg)
    return X_np, y_np


def _check_X(X: np.ndarray | DataFrame) -> np.ndarray:
    if hasattr(X, 'to_numpy') and hasattr(X, 'columns'):
        X_np = np.asarray(X.to_numpy())
    else:
        X_np = np.asarray(X)
    if X_np.ndim == 1:
        X_np = X_np[:, None]
    if X_np.ndim != 2:
        msg = f'X must be 2D (m, p); got shape {X_np.shape}'
        raise ValueError(msg)
    return X_np


__all__ = ['BARTModel', 'NotSampledError', 'OutcomeModel']
