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
from dataclasses import dataclass, field, fields
from functools import partial
from typing import Any, Literal, TypeVar

import numpy as np
from jax import numpy as jnp
from jax.scipy.special import ndtr
from jaxtyping import Array, Float, Float32
from scipy.stats import norm

from bartz._interface import Bart, DataFrame, PredictKind
from bartz.prepcovars import UniqueQuantileBinner

T = TypeVar('T')

_MAX_DEPTH_LIMIT = 16


def _check_int(value: object, name: str) -> None:
    # bool is a subclass of int; reject explicitly to avoid silent coercion.
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f'{name} must be int, got {type(value).__name__}'
        raise TypeError(msg)


def _check_float(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f'{name} must be float, got {type(value).__name__}'
        raise TypeError(msg)


def _check_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        msg = f'{name} must be bool, got {type(value).__name__}'
        raise TypeError(msg)


@dataclass(frozen=True)
class OutcomeModel:
    """Outcome model specification, matching `stochtree.OutcomeModel`.

    Only ``('continuous', 'identity')`` and ``('binary', 'probit')`` are
    supported.
    """

    outcome: Literal['continuous', 'binary'] = 'continuous'
    """Outcome family."""

    link: Literal['identity', 'probit'] = 'identity'
    """Link function. ``'identity'`` for continuous, ``'probit'`` for binary."""

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, str):
            msg = f'outcome must be str, got {type(self.outcome).__name__}'
            raise TypeError(msg)
        if not isinstance(self.link, str):
            msg = f'link must be str, got {type(self.link).__name__}'
            raise TypeError(msg)
        if (self.outcome, self.link) not in (
            ('continuous', 'identity'),
            ('binary', 'probit'),
        ):
            msg = (
                f'unsupported outcome_model (outcome={self.outcome!r}, '
                f"link={self.link!r}); only ('continuous', 'identity') "
                "and ('binary', 'probit') are supported."
            )
            raise NotImplementedError(msg)


class NotSampledError(RuntimeError):
    """Raised when calling a method that requires `sample` to have been called."""


@dataclass(frozen=True, kw_only=True)
class GeneralParams:
    """Mirror of stochtree's ``general_params`` dict, with the keys bartz handles."""

    cutpoint_grid_size: int = 100
    """Maximum number of cutpoints to consider for each feature."""

    standardize: bool = True
    """Whether to standardize the outcome before fitting. Ignored for probit binary."""

    sigma2_init: float | None = None
    """Starting value of the global error variance. If `None`, bartz picks one automatically."""

    sigma2_global_shape: float = 0
    """Shape parameter of the inverse-gamma prior on the global error variance."""

    sigma2_global_scale: float = 0
    """Scale parameter of the inverse-gamma prior on the global error variance."""

    variable_weights: np.ndarray | None = None
    """Per-predictor sampling weights. Must be strictly positive; pass a small positive value to suppress a variable."""

    random_seed: int | None = None
    """Seed for the random number generator."""

    keep_every: int = 1
    """Thinning factor for retained MCMC samples."""

    num_chains: int = 1
    """Number of independent MCMC chains."""

    outcome_model: OutcomeModel = field(default_factory=OutcomeModel)
    """Outcome family and link specification."""

    def __post_init__(self) -> None:
        _check_int(self.cutpoint_grid_size, 'cutpoint_grid_size')
        _check_bool(self.standardize, 'standardize')
        if self.sigma2_init is not None:
            _check_float(self.sigma2_init, 'sigma2_init')
        _check_float(self.sigma2_global_shape, 'sigma2_global_shape')
        _check_float(self.sigma2_global_scale, 'sigma2_global_scale')
        if self.variable_weights is not None and not isinstance(
            self.variable_weights, np.ndarray
        ):
            msg = (
                f'variable_weights must be np.ndarray, got '
                f'{type(self.variable_weights).__name__}'
            )
            raise TypeError(msg)
        if self.random_seed is not None:
            _check_int(self.random_seed, 'random_seed')
        _check_int(self.keep_every, 'keep_every')
        _check_int(self.num_chains, 'num_chains')
        if not isinstance(self.outcome_model, OutcomeModel):
            msg = (
                f'outcome_model must be OutcomeModel, got '
                f'{type(self.outcome_model).__name__}'
            )
            raise TypeError(msg)


@dataclass(frozen=True, kw_only=True)
class MeanForestParams:
    """Mirror of stochtree's ``mean_forest_params`` dict, restricted to the keys bartz handles."""

    num_trees: int = 200
    """Number of trees in the conditional mean ensemble."""

    alpha: float = 0.95
    """Tree split prior base."""

    beta: float = 2.0
    """Tree split prior decay."""

    min_samples_leaf: int = 5
    """Minimum number of training samples at a leaf."""

    max_depth: int = 10
    """Maximum tree depth. Must be a non-negative integer at most ``16``."""

    sample_sigma2_leaf: bool = True
    """Whether to sample the leaf-variance prior. Must be set to ``False``."""

    sigma2_leaf_init: float | None = None
    """Initial leaf-variance prior. If `None`, defaults to ``1 / num_trees``."""

    def __post_init__(self) -> None:
        _check_int(self.num_trees, 'num_trees')
        _check_float(self.alpha, 'alpha')
        _check_float(self.beta, 'beta')
        _check_int(self.min_samples_leaf, 'min_samples_leaf')
        _check_int(self.max_depth, 'max_depth')
        _check_bool(self.sample_sigma2_leaf, 'sample_sigma2_leaf')
        if self.sigma2_leaf_init is not None:
            _check_float(self.sigma2_leaf_init, 'sigma2_leaf_init')

        if self.sample_sigma2_leaf:
            msg = (
                'sample_sigma2_leaf=True is not supported (bartz uses a fixed'
                " leaf-variance prior); pass mean_forest_params={'sample_sigma2_leaf':"
                ' False} to acknowledge this.'
            )
            raise NotImplementedError(msg)
        if self.max_depth < 0:
            msg = (
                f'max_depth={self.max_depth} is not supported; bartz stores trees'
                ' as heap arrays of size 2**max_depth, so the stochtree'
                ' convention max_depth=-1 (unbounded) is rejected. Pass a'
                f' non-negative integer at most {_MAX_DEPTH_LIMIT}.'
            )
            raise NotImplementedError(msg)
        if self.max_depth > _MAX_DEPTH_LIMIT:
            msg = (
                f'max_depth={self.max_depth} exceeds {_MAX_DEPTH_LIMIT}; bartz'
                ' stores trees as heap arrays of size 2**max_depth, so memory'
                ' grows exponentially with depth.'
            )
            raise ValueError(msg)
        if self.sigma2_leaf_init is None:
            object.__setattr__(self, 'sigma2_leaf_init', 1.0 / self.num_trees)


def build_dataclass(cls: type[T], params: dict | None, name: str) -> T:
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

    See `GeneralParams` and `MeanForestParams` for the supported keys in the
    ``general_params`` and ``mean_forest_params`` dicts.

    Notes
    -----
    Differences from `stochtree`, by design:

    - ``num_gfr`` has no default and must be set explicitly to ``0``.
    - ``mean_forest_params['sample_sigma2_leaf']`` must be ``False``.
    - ``mean_forest_params['max_depth']`` must be a non-negative integer at
      most ``16``; stochtree's ``-1`` (unbounded depth) sentinel is not
      accepted.
    - The deprecated ``general_params['probit_outcome_model']`` flag is not
      accepted; pass ``outcome_model=OutcomeModel('binary', 'probit')``
      instead.
    - Leaf-basis regression, random effects, heteroskedastic variance
      forests, and warm-starting from a previous model are not supported.
    - bartz uses single-precision floats, so outputs differ from stochtree
      at the float32 precision level.

    References
    ----------
    Herren, A., Hahn, P. R., Murray, J., Carvalho, C. (2026). "StochTree:
    BART-based modeling in R and Python". arXiv:2512.12051.
    """

    # public, set by sample()
    sampled: bool
    """Whether `sample` has been called."""

    standardize: bool
    """Whether the outcome was standardized before fitting."""

    sample_sigma2_global: bool
    """Whether the global error variance is sampled (always ``True``)."""

    probit_outcome_model: bool
    """Whether the model uses a binary outcome with probit link."""

    outcome_model: OutcomeModel
    """Outcome family and link specification used during fitting."""

    num_gfr: int
    """Number of grow-from-root iterations (always ``0``)."""

    num_burnin: int
    """Number of MCMC burn-in iterations."""

    num_mcmc: int
    """Number of retained MCMC iterations per chain."""

    num_chains: int
    """Number of independent MCMC chains."""

    num_samples: int
    """Total number of retained posterior samples (``num_mcmc * num_chains``)."""

    sigma2_init: float | None
    """Starting value of the global error variance."""

    y_bar: float
    """Mean used to standardize the outcome (``0`` if not standardized)."""

    y_std: float
    """Standard deviation used to standardize the outcome (``1`` if not standardized)."""

    has_rfx: bool
    """Whether the model includes random effects (always ``False``)."""

    include_mean_forest: bool
    """Whether the model includes a conditional mean forest (always ``True``)."""

    include_variance_forest: bool
    """Whether the model includes a variance forest (always ``False``)."""

    y_hat_train: Float32[Array, 'n num_samples']
    """Posterior predictions at the training covariates, in the original outcome scale."""

    global_var_samples: Float32[Array, ' num_samples']
    """Posterior samples of the global error variance. For probit binary regression, an array of ones."""

    y_hat_test: Float32[Array, 'm num_samples'] | None
    """Posterior predictions at `X_test` if it was supplied to `sample`, else `None`."""

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

        The signature mirrors `stochtree.BARTModel.sample`, restricted to the
        keyword arguments bartz supports.

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
            Optional override for the keys of `GeneralParams`.
        mean_forest_params
            Override for the keys of `MeanForestParams`. Must explicitly
            disable ``sample_sigma2_leaf``.

        Raises
        ------
        NotImplementedError
            If ``num_gfr`` is non-zero.
        """
        if num_gfr != 0:
            msg = (
                'num_gfr must be 0; the grow-from-root sampler is not available'
                ' in bartz.'
            )
            raise NotImplementedError(msg)

        gp = build_dataclass(GeneralParams, general_params, 'general_params')
        mfp = build_dataclass(
            MeanForestParams, mean_forest_params, 'mean_forest_params'
        )

        is_probit = gp.outcome_model.outcome == 'binary'

        X_train_np, y_train_np = process_train_inputs(X_train, y_train)
        _, p = X_train_np.shape

        # standardization matches stochtree's y_bar / y_std logic
        y_bar, y_std, y_for_bartz = standardize_y(y_train_np, is_probit, gp.standardize)

        bart_num_chains = None if gp.num_chains == 1 else gp.num_chains

        # leaf-prior: bartz uses sigma_mu = tau_num / (k * sqrt(num_trees));
        # stochtree's sigma2_leaf is the leaf-variance prior. Hold k=2 and solve
        # for tau_num so that the two parameterizations agree.
        bartz_k = 2.0
        tau_num_arg = bartz_k * math.sqrt(mfp.num_trees * mfp.sigma2_leaf_init)

        sigdf, lambda_, sigest_arg = resolve_variance_prior(
            gp.sigma2_global_shape, gp.sigma2_global_scale, gp.sigma2_init
        )

        binner = partial(
            UniqueQuantileBinner, max_bins=gp.cutpoint_grid_size + 1, max_subsample=None
        )

        variable_weights = check_variable_weights(gp.variable_weights, p)

        seed = 0 if gp.random_seed is None else gp.random_seed
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
            power=mfp.beta,
            base=mfp.alpha,
            lambda_=lambda_,
            tau_num=tau_num_arg,
            w=w_arg,
            num_trees=mfp.num_trees,
            n_save=num_mcmc,
            n_burn=num_burnin,
            n_skip=gp.keep_every,
            printevery=None,
            num_chains=bart_num_chains,
            seed=seed,
            maxdepth=mfp.max_depth + 1,
            init_kw={'min_points_per_leaf': mfp.min_samples_leaf},
        )
        self._finalize_sample(
            outcome_model=gp.outcome_model,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            num_chains=gp.num_chains,
            sigma2_init=gp.sigma2_init,
            y_bar=y_bar,
            y_std=y_std,
            standardize=gp.standardize,
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
            X_test_np = check_X(X_test)
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
    ) -> np.ndarray | dict[str, np.ndarray]:
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
            and ``'mean_forest'`` produce the same result.
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
        terms_list = check_predict_args(type, scale, terms, self.probit_outcome_model)

        X_np = check_X(X)
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
        result: dict[str, np.ndarray] = {}
        if wants_y_hat:
            result['y_hat'] = pred_out
        if wants_mean_forest:
            result['mean_forest_predictions'] = pred_out
        return result

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


def standardize_y(
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


def resolve_variance_prior(
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


def check_variable_weights(
    variable_weights: np.ndarray | None, p: int
) -> Float32[Array, ' p'] | None:
    """Validate `variable_weights`, returning the jax array (or None)."""
    if variable_weights is None:
        return None
    arr = jnp.asarray(variable_weights, jnp.float32)
    if arr.shape != (p,):
        msg = f'variable_weights must have shape (p,)=({p},), got {arr.shape}'
        raise ValueError(msg)
    return arr


def check_predict_args(
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
        if t not in ('y_hat', 'mean_forest', 'all'):
            msg = f'unknown term {t!r}; valid terms are y_hat, mean_forest, all'
            raise ValueError(msg)
    return terms_list


def process_train_inputs(
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


def check_X(X: np.ndarray | DataFrame) -> np.ndarray:
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
