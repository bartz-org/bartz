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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from functools import partial

# WORKAROUND(python<3.15): use frozendict instead of MappingProxyType
from types import MappingProxyType
from typing import Any, Literal, TypeVar

from jax import numpy as jnp
from jax.scipy.special import ndtr
from jaxtyping import Array, Float, Float32, Key, Real, Shaped

from bartz._interface import Bart, DataFrame, PredictKind, Series
from bartz._jaxext.scipy.special import ndtri
from bartz.mcmcstep._state import ArrayLike, FloatLike
from bartz.prepcovars import RangeEvenBinner

T = TypeVar('T')

_MAX_DEPTH_LIMIT = 16


@dataclass(frozen=True)
class OutcomeModel:
    """Outcome model specification, matching `stochtree.OutcomeModel`.

    Only ``('continuous', 'identity')`` and ``('binary', 'probit')`` are
    supported.
    """

    outcome: Literal['continuous', 'binary'] = 'continuous'
    """Outcome family."""

    link: Literal['identity', 'probit'] | None = None
    """Link function. If `None`, defaults to ``'identity'`` for ``'continuous'`` and ``'probit'`` for ``'binary'``."""

    def __post_init__(self) -> None:
        if self.link is None:
            default_link = {'continuous': 'identity', 'binary': 'probit'}.get(
                self.outcome
            )
            object.__setattr__(self, 'link', default_link)
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


class NotSampledError(ValueError, AttributeError):
    """Raised when calling a method that requires `sample` to have been called."""


@dataclass(frozen=True, kw_only=True)
class GeneralParams:
    """Mirror of stochtree's ``general_params`` dict, with the keys bartz handles."""

    standardize: bool = True
    """Whether to standardize the outcome before fitting. Ignored for probit binary."""

    sigma2_init: FloatLike | None = None
    """Starting value of the global error variance. Only honored when the variance prior is improper (``sigma2_global_shape=0`` or ``sigma2_global_scale=0``); otherwise raises, since bartz cannot decouple the chain start from the prior scale. If `None` (default), uses ``var(resid_train)`` for continuous and ``1.0`` for probit, matching stochtree."""

    sigma2_global_shape: FloatLike = 0
    """Shape parameter of the inverse-gamma prior on the global error variance. The default ``0`` is mapped to a near-improper prior, since bartz's scaled-inv-chi² cannot represent ``IG(0, 0)`` exactly."""

    sigma2_global_scale: FloatLike = 0
    """Scale parameter of the inverse-gamma prior on the global error variance. The default ``0`` is mapped to a near-improper prior, since bartz's scaled-inv-chi² cannot represent ``IG(0, 0)`` exactly."""

    variable_weights: Float[ArrayLike, ' p'] | None = None
    """Per-predictor sampling weights. Must be strictly positive; pass a small positive value to suppress a variable."""

    random_seed: int | Key[Array, ''] | None = None
    """Seed for the random number generator."""

    keep_every: int = 1
    """Thinning factor for retained MCMC samples."""

    num_chains: int = 1
    """Number of independent MCMC chains."""

    outcome_model: OutcomeModel = field(default_factory=OutcomeModel)
    """Outcome family and link specification."""


@dataclass(frozen=True, kw_only=True)
class MeanForestParams:
    """Mirror of stochtree's ``mean_forest_params`` dict, restricted to the keys bartz handles."""

    num_trees: int = 200
    """Number of trees in the conditional mean ensemble."""

    alpha: FloatLike = 0.95
    """Tree split prior base."""

    beta: FloatLike = 2.0
    """Tree split prior decay."""

    min_samples_leaf: int = 5
    """Minimum number of training samples at a leaf."""

    max_depth: int = 10
    """Maximum tree depth. Must be a non-negative integer at most ``16``."""

    sample_sigma2_leaf: bool = True
    """Whether to sample the leaf-variance prior. Must be set to ``False``."""

    sigma2_leaf_init: FloatLike | None = None
    """Initial leaf-variance prior (held fixed since ``sample_sigma2_leaf=False``). If `None`, matches stochtree's defaults: ``var(resid_train) / num_trees`` for continuous and ``2 / num_trees`` for probit."""

    def __post_init__(self) -> None:
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
    - ``general_params['cutpoint_grid_size']`` is not accepted; bartz uses a
      fixed grid of 256 evenly-spaced bins per predictor. stochtree only
      uses this parameter for the GFR sampler, which bartz does not support.
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

    sigma2_init: FloatLike
    """Starting value of the global error variance actually used to seed the chain."""

    y_bar: Float32[Array, '']
    """Mean used to standardize the outcome (``0`` if not standardized)."""

    y_std: Float32[Array, '']
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
        X_train: Real[ArrayLike, 'n p'] | DataFrame,
        y_train: Real[ArrayLike, ' n'] | Series,
        X_test: Real[ArrayLike, 'm p'] | DataFrame | None = None,
        observation_weights: Float[ArrayLike, ' n'] | Series | None = None,
        *,
        num_gfr: int,
        num_burnin: int = 0,
        num_mcmc: int = 100,
        general_params: dict[str, Any] | None = None,
        mean_forest_params: dict[str, Any] | None = None,
        bart_kwargs: Mapping[str, Any] = MappingProxyType({}),
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
        bart_kwargs
            Additional arguments forwarded to `bartz.Bart`. Use this to set
            ``devices`` and ``rm_const=False`` when wrapping `sample` in
            `jax.jit`.

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

        X_train_arr, y_train_arr = process_train_inputs(X_train, y_train)
        _, p = X_train_arr.shape

        y_bar, y_std, y_for_bartz = standardize_y(
            y_train_arr, is_probit, gp.standardize
        )

        bart_num_chains = None if gp.num_chains == 1 else gp.num_chains

        # variance of the standardized residual, matching stochtree
        # (np.var(resid_train) with ddof=0). For standardize=True it is exactly
        # 1.0; we hardcode that so the value stays trace-time concrete.
        if is_probit:
            var_resid_train: FloatLike = 1.0  # bartz ignores σ² for binary
        elif gp.standardize:
            var_resid_train = 1.0
        else:
            var_resid_train = jnp.var(y_for_bartz)

        # leaf-prior: bartz uses sigma_mu = tau_num / (k * sqrt(num_trees));
        # stochtree's sigma2_leaf is the leaf-variance prior. Hold k=2 and solve
        # for tau_num so that the two parameterizations agree.
        bartz_k = 2.0
        sigma2_leaf_init = resolve_sigma2_leaf_init(
            mfp.sigma2_leaf_init, mfp.num_trees, is_probit, var_resid_train
        )
        tau_num_arg = bartz_k * jnp.sqrt(mfp.num_trees * sigma2_leaf_init)

        if is_probit:
            # stochtree pins σ²=1 for probit; bartz binary branch ignores the
            # variance prior, so we pass any valid placeholders.
            sigdf_arg: FloatLike = 3.0
            lambda_arg: FloatLike | None = None
            sigma2_init_stored: FloatLike = 1.0
        else:
            sigdf_arg, lambda_arg, sigma2_init_stored = resolve_variance_prior(
                gp.sigma2_global_shape,
                gp.sigma2_global_scale,
                gp.sigma2_init,
                var_resid_train,
            )

        binner = partial(RangeEvenBinner, max_bins=256)

        variable_weights = check_variable_weights(gp.variable_weights, p)

        seed = 0 if gp.random_seed is None else gp.random_seed

        kwargs: dict = dict(
            x_train=X_train_arr.T,
            y_train=y_for_bartz,
            outcome_type='binary' if is_probit else 'continuous',
            binner=binner,
            varprob=variable_weights,
            sigest='auto',
            sigdf=sigdf_arg,
            sigquant=0.9,
            k=bartz_k,
            power=mfp.beta,
            base=mfp.alpha,
            lambda_=lambda_arg,
            tau_num=tau_num_arg,
            w=observation_weights,
            num_trees=mfp.num_trees,
            n_save=num_mcmc,
            n_burn=num_burnin,
            n_skip=gp.keep_every,
            printevery=None,
            num_chains=bart_num_chains,
            seed=seed,
            maxdepth=mfp.max_depth + 1,
        )
        kwargs.update(bart_kwargs)
        # match stochtree's gating: only acceptance-time veto on
        # min_samples_leaf, no per-leaf affluence filter (stochtree picks
        # leaves uniformly over all of them). User-supplied init_kw values
        # win on conflicts.
        kwargs = dict(
            kwargs,
            init_kw=dict(
                {
                    'min_points_per_leaf': mfp.min_samples_leaf,
                    'min_points_per_decision_node': None,
                },
                **kwargs.get('init_kw', {}),
            ),
        )
        self._bart = Bart(**kwargs)
        self._finalize_sample(
            outcome_model=gp.outcome_model,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            num_chains=gp.num_chains,
            sigma2_init=sigma2_init_stored,
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
        sigma2_init: FloatLike,
        y_bar: Float32[Array, ''],
        y_std: Float32[Array, ''],
        standardize: bool,
        X_test: Real[ArrayLike, 'm p'] | DataFrame | None,
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
            self.y_hat_test = self._predict_y_hat_internal(check_X(X_test).T)
        else:
            self.y_hat_test = None

        if is_probit:
            self.global_var_samples = jnp.ones((self.num_samples,))
        else:
            sigma = self._bart.get_error_sdev()
            self.global_var_samples = (sigma * y_std) ** 2

    def predict(
        self,
        X: Real[ArrayLike, 'm p'] | DataFrame,
        *,
        type: Literal['posterior', 'mean'] = 'posterior',  # noqa: A002
        terms: Literal['y_hat', 'mean_forest', 'all']
        | Sequence[Literal['y_hat', 'mean_forest', 'all']] = 'all',
        scale: Literal['linear', 'probability', 'class'] = 'linear',
    ) -> Shaped[Array, '...'] | dict[str, Shaped[Array, '...']]:
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
        Either a single jax array (for a single requested term) or a dict keyed by term name (matching stochtree's behavior).

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
        terms_tuple = check_predict_args(type, scale, terms, self.probit_outcome_model)

        pred = self._predict_y_hat_internal(check_X(X).T)

        if self.probit_outcome_model and scale in ('probability', 'class'):
            prob = ndtr(pred)
            pred_out = jnp.where(prob < 0.5, 0, 1) if scale == 'class' else prob
        else:
            pred_out = pred

        if type == 'mean':
            pred_out = jnp.mean(pred_out, axis=1)

        wants_y_hat = ('y_hat' in terms_tuple) or ('all' in terms_tuple)
        wants_mean_forest = ('mean_forest' in terms_tuple) or ('all' in terms_tuple)
        single = sum([wants_y_hat, wants_mean_forest]) == 1
        if single:
            return pred_out
        result: dict[str, Shaped[Array, '...']] = {}
        if wants_y_hat:
            result['y_hat'] = pred_out
        if wants_mean_forest:
            result['mean_forest_predictions'] = pred_out
        return result

    def _predict_y_hat_internal(
        self, x: Real[ArrayLike, 'p m'] | Literal['train']
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
    y_train: Real[ArrayLike, ' n'], is_probit: bool, standardize: bool
) -> tuple[Float32[Array, ''], Float32[Array, ''], Float32[Array, ' n']]:
    """Return ``(y_bar, y_std, y_for_bartz)`` matching stochtree's standardization."""
    y = jnp.asarray(y_train, jnp.float32)
    if is_probit:
        return ndtri(y.mean()), jnp.float32(1.0), (y != 0).astype(jnp.float32)
    if standardize:
        y_bar = y.mean()
        y_std_val = y.std()
        y_std = jnp.where(y_std_val > 0, y_std_val, 1.0)
        return y_bar, y_std, (y - y_bar) / y_std
    return jnp.float32(0.0), jnp.float32(1.0), y


def resolve_sigma2_leaf_init(
    sigma2_leaf_init: FloatLike | None,
    num_trees: int,
    is_probit: bool,
    var_resid_train: FloatLike,
) -> FloatLike:
    """Default `sigma2_leaf_init` per stochtree: probit→2/num_trees, continuous→var(resid)/num_trees."""
    if sigma2_leaf_init is not None:
        return sigma2_leaf_init
    if is_probit:
        return 2.0 / num_trees
    return var_resid_train / num_trees


# Bartz scaled-inv-chi² df used to approximate stochtree's improper IG(0,0).
# Small enough that the prior contribution to the σ² posterior is dominated by
# the data (the posterior is IG((sigdf+n)/2, (sigdf*lambda+sum_sq)/2) for any
# n >= 1), large enough in case the starting MCMC value is computed with a
# regularized inverse
_IMPROPER_PRIOR_SIGDF = 0.01


def resolve_variance_prior(
    shape: FloatLike,
    scale: FloatLike,
    sigma2_init: FloatLike | None,
    var_resid_train: FloatLike,
) -> tuple[FloatLike, FloatLike, FloatLike]:
    """Translate stochtree's IG(shape, scale) prior to bartz's scaled-inv-chi2.

    Bartz initializes σ² at ``lambda_`` (it cannot decouple the prior scale
    from the chain start). For the proper-prior path we match the prior
    exactly and refuse a user-supplied ``sigma2_init``; for the improper-prior
    path we pin ``lambda_`` to the desired initial value and use a tiny
    ``sigdf`` so the prior contribution is negligible.

    Parameters
    ----------
    shape
        Stochtree's ``sigma2_global_shape``.
    scale
        Stochtree's ``sigma2_global_scale``.
    sigma2_init
        Stochtree's ``sigma2_init``. Allowed only when the prior is improper.
    var_resid_train
        Variance of the residual used to standardize-style initialize σ² when
        ``sigma2_init`` is unset and the prior is improper.

    Returns
    -------
    ``(sigdf, lambda_, sigma2_init_stored)`` for bartz; ``sigma2_init_stored``
    is the actual chain starting value, suitable for ``BARTModel.sigma2_init``.

    Raises
    ------
    NotImplementedError
        If `sigma2_init` is set together with a proper variance prior.
    """
    # IG(shape, scale) <=> scaled-inv-chi2(df=2*shape, lambda=scale/shape)
    if shape > 0 and scale > 0:
        if sigma2_init is not None:
            msg = (
                'sigma2_init cannot be set together with a proper variance'
                ' prior (sigma2_global_shape > 0 and sigma2_global_scale > 0):'
                ' bartz initializes sigma² at lambda_=scale/shape and cannot'
                ' separately honor sigma2_init. Drop sigma2_init or use the'
                ' default improper prior (sigma2_global_shape=0,'
                ' sigma2_global_scale=0).'
            )
            raise NotImplementedError(msg)
        lambda_ = scale / shape
        return 2.0 * shape, lambda_, lambda_
    sigma2_start = sigma2_init if sigma2_init is not None else var_resid_train
    return _IMPROPER_PRIOR_SIGDF, sigma2_start, sigma2_start


def check_variable_weights(
    variable_weights: Float[ArrayLike, ' p'] | None, p: int
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
    type_: Literal['posterior', 'mean'],
    scale: Literal['linear', 'probability', 'class'],
    terms: Literal['y_hat', 'mean_forest', 'all']
    | Sequence[Literal['y_hat', 'mean_forest', 'all']],
    probit_outcome_model: bool,
) -> tuple[str, ...]:
    """Validate `BARTModel.predict` arguments, returning the normalized terms tuple."""
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
    terms_tuple = (terms,) if isinstance(terms, str) else tuple(terms)
    for t in terms_tuple:
        if t not in ('y_hat', 'mean_forest', 'all'):
            msg = f'unknown term {t!r}; valid terms are y_hat, mean_forest, all'
            raise ValueError(msg)
    return terms_tuple


def process_train_inputs(
    X_train: Real[ArrayLike, 'n p'] | DataFrame, y_train: Real[ArrayLike, ' n'] | Series
) -> tuple[Real[Array, 'n p'], Real[Array, ' n']]:
    """Convert training inputs to 2-D / 1-D jax arrays and verify their shapes are compatible."""
    X = check_X(X_train, name='X_train')
    y = _coerce_response(y_train, name='y_train')
    if y.shape[0] != X.shape[0]:
        msg = (
            f'X_train and y_train length mismatch: X_train has '
            f'{X.shape[0]} rows, y_train has {y.shape[0]} entries'
        )
        raise ValueError(msg)
    return X, y


def check_X(
    X: Real[ArrayLike, 'n p'] | DataFrame, *, name: str = 'X'
) -> Real[Array, 'n p']:
    """Convert a DataFrame/array-like to a 2-D jax array in ``(n, p)`` layout."""
    if hasattr(X, 'columns') and hasattr(X, 'to_numpy'):
        X = X.to_numpy()
    arr = jnp.asarray(X)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        msg = f'{name} must be 2D (n, p); got shape {arr.shape}'
        raise ValueError(msg)
    return arr


def _coerce_response(
    y: Real[ArrayLike, ' n'] | Series, *, name: str
) -> Real[Array, ' n']:
    """Convert a Series/array-like response to a 1-D jax array."""
    if hasattr(y, 'to_numpy'):
        y = y.to_numpy()
    arr = jnp.asarray(y)
    if arr.ndim != 1:
        msg = f'{name} must be 1D (n,); got shape {arr.shape}'
        raise ValueError(msg)
    return arr
