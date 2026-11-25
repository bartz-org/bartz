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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from functools import partial

# WORKAROUND(python<3.15): use frozendict instead of MappingProxyType
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

from equinox import error_if
from jax import lax, random
from jax import numpy as jnp
from jax.scipy.special import ndtr, ndtri
from jaxtyping import Array, Float, Float32, Key, Real, Shaped

from bartz._interface import Bart, PredictKind, Series
from bartz._jaxext import forbid_x64, is_key, jit, split
from bartz.mcmcloop import MainTrace
from bartz.mcmcstep import Wishart
from bartz.mcmcstep._state import ArrayLike, FloatLike
from bartz.prepcovars import RangeEvenBinner
from bartz.stochtree._preprocess import DataFrame, Preprocessor, to_narwhals

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

_MAX_DEPTH_LIMIT = 16
_NOT_SAMPLED_MSG = (
    "This BARTModel instance is not fitted yet. Call 'sample' before using this model."
)


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
    """Raised when calling a method that requires `BARTModel.sample` to have been called."""


@dataclass(frozen=True, kw_only=True)
class GeneralParams:
    """Mirror of stochtree's ``general_params`` dict, with the keys bartz handles."""

    standardize: bool = True
    """Whether to standardize the outcome before fitting. Ignored for probit binary."""

    sigma2_init: FloatLike | None = None
    """Starting value of the global error variance. If `None` (default), uses ``var(resid_train)`` for continuous and ``1.0`` for probit."""

    sigma2_global_shape: FloatLike = 0.0
    """Shape parameter of the inverse-gamma prior on the global error variance. The default ``0`` is mapped to a near-improper prior, since bartz's scaled-inv-chi² cannot represent ``IG(0, 0)`` exactly."""

    sigma2_global_scale: FloatLike = 0.0
    """Scale parameter of the inverse-gamma prior on the global error variance. The default ``0`` is mapped to a near-improper prior, since bartz's scaled-inv-chi² cannot represent ``IG(0, 0)`` exactly."""

    variable_weights: Float[ArrayLike, ' p'] | None = None
    """Per-predictor sampling weights. Must be strictly positive; pass a small positive value to suppress a variable."""

    random_seed: int | Key[Array, ''] | None = None
    """Seed for the random number generator. Unlike stochtree, the default
    `None` is deterministic (equivalent to seed ``0``) rather than drawing a
    random seed, so repeated fits reproduce by default."""

    keep_every: int = 1
    """Thinning factor for retained MCMC samples."""

    num_chains: int = 1
    """Number of independent MCMC chains."""

    outcome_model: OutcomeModel = field(default_factory=OutcomeModel)
    """Outcome family and link specification. Defaults to continuous with
    identity link."""


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
    """Whether to sample the leaf variance with an inverse-gamma prior, or hold it fixed at `sigma2_leaf_init`."""

    sigma2_leaf_init: FloatLike | None = None
    """Initial leaf variance. If `None`, matches stochtree's defaults of ``var(resid_train) / num_trees`` for continuous and ``2 / num_trees`` for probit."""

    sigma2_leaf_shape: FloatLike = 3.0
    """Twice the shape parameter of the inverse-gamma prior on the leaf variance, i.e., the prior is ``IG(sigma2_leaf_shape / 2, sigma2_leaf_scale / 2)``."""

    sigma2_leaf_scale: FloatLike | None = None
    """Twice the scale parameter of the inverse-gamma prior on the leaf variance. If `None`, defaults to ``var(resid_train) / num_trees`` for continuous and ``1 / num_trees`` for probit."""

    def __post_init__(self) -> None:
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


if TYPE_CHECKING:
    # static only, beartype does not support type[DataclassInstance]
    T = TypeVar('T', bound='DataclassInstance')
else:
    T = TypeVar('T')


def build_dataclass(cls: type[T], params: Mapping[str, Any] | None, name: str) -> T:
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

    This class mimics `stochtree.BARTModel`. The main difference is that
    `bartz` does not implement "grow-from-root" (XBART), so the `stochtree`
    defaults of 0 burnin and 100 saved MCMC iterations are best changed::

        m = BARTModel()
        m.sample(X_train=X, y_train=y, X_test=X_test, num_gfr=0, num_burnin=1000, num_mcmc=1000)
        yhat = m.predict(X_new, terms='y_hat', type='mean')

    See `GeneralParams` and `MeanForestParams` for the supported keys in the
    ``general_params`` and ``mean_forest_params`` dicts.

    Notes
    -----
    Differences from `stochtree`:

    - ``num_gfr`` has no default and must be set explicitly to ``0``.
    - The arguments of `sample` after ``y_train``, and of the other methods
      after the covariates, are keyword-only, because some of stochtree's
      positional arguments are not supported.
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
    - ``observation_weights`` are not supported, because as of stochtree
      0.4.5 they do not work as documented, so there is no stochtree behavior
      to mimic.
    - bartz uses single-precision floats, so outputs differ from stochtree
      at the float32 precision level.
    - ``general_params['random_seed']`` defaults to deterministic behavior
      (seed ``0``) when unset, whereas stochtree draws a random seed. This is
      intentional, to make repeated fits reproducible by default.

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
    """Whether the global error variance is sampled (``False`` for probit binary regression)."""

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

    keep_every: int
    """Thinning factor of the retained MCMC samples."""

    num_samples: int
    """Total number of retained posterior samples (``num_mcmc * num_chains``)."""

    n_train: int
    """Number of training observations."""

    n_test: int
    """Number of test observations (``0`` if `X_test` was not supplied)."""

    has_test: bool
    """Whether `X_test` was supplied to `sample`."""

    num_covariates: int
    """Number of covariates after preprocessing, e.g., after one-hot encoding."""

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

    global_var_samples: Float32[Array, ' num_samples'] | None
    """Posterior samples of the global error variance, or `None` if not sampled."""

    sample_sigma2_leaf: bool
    """Whether the leaf variance is sampled."""

    leaf_scale_samples: Float32[Array, ' num_samples'] | None
    """Posterior samples of the leaf variance, in the standardized outcome scale, or `None` if not sampled."""

    y_hat_test: Float32[Array, 'm num_samples'] | None
    """Posterior predictions at `X_test` if it was supplied to `sample`, else `None`."""

    _bart: Bart
    _preprocessor: Preprocessor | None
    _ppd_key: Key[Array, '']
    """Base key of `sample_posterior_predictive`."""

    _ppd_count: int
    """Number of calls to `sample_posterior_predictive`, folded into `_ppd_key`."""

    def __init__(self) -> None:
        self.sampled = False
        self._preprocessor = None

    def is_sampled(self) -> bool:
        """Return whether `sample` has been called."""
        return self.sampled

    def _prepare_training_inputs(
        self,
        X_train: Real[ArrayLike, 'n p'] | DataFrame,
        y_train: Real[ArrayLike, ' n'] | Series,
        gp: GeneralParams,
    ) -> tuple[Real[Array, 'n p'], Real[Array, ' n'], Float32[Array, ' p'] | None]:
        """Coerce inputs and build variable weights, fitting the DataFrame preprocessor if any."""
        y_train_arr = _coerce_response(y_train, name='y_train')

        df_train = to_narwhals(X_train)
        if df_train is None:
            self._preprocessor = None
            X_train_arr = check_X(X_train, name='X_train')
            _, p = X_train_arr.shape
            varprob = check_variable_weights(gp.variable_weights, p)
        else:
            # The preprocessor decides the default weights: uniform over the
            # *original* columns split across each one-hot expansion (so every
            # original variable keeps an equal splitting budget), or `None` when
            # nothing expands (deferring to bartz's native uniform fast-path).
            self._preprocessor = Preprocessor()
            weights_np = self._preprocessor.fit(
                df_train, variable_weights=gp.variable_weights
            )
            X_train_np = self._preprocessor.transform(df_train)
            if X_train_np.shape[1] == 0:
                msg = 'X_train has no usable columns after preprocessing'
                raise ValueError(msg)
            X_train_arr = jnp.asarray(X_train_np)
            varprob = None if weights_np is None else jnp.asarray(weights_np)

        n, _ = X_train_arr.shape
        if y_train_arr.shape[0] != n:
            msg = (
                f'X_train and y_train length mismatch: X_train has {n} rows,'
                f' y_train has {y_train_arr.shape[0]} entries'
            )
            raise ValueError(msg)
        return X_train_arr, y_train_arr, varprob

    @forbid_x64
    def sample(
        self,
        X_train: Real[ArrayLike, 'n p'] | DataFrame,
        y_train: Real[ArrayLike, ' n'] | Series,
        *,
        X_test: Real[ArrayLike, 'm p'] | DataFrame | None = None,
        observation_weights: Float[ArrayLike, ' n'] | Series | None = None,
        num_gfr: int,
        num_burnin: int = 0,
        num_mcmc: int = 100,
        general_params: Mapping[str, Any] | None = None,
        mean_forest_params: Mapping[str, Any] | None = None,
        bart_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ) -> None:
        """Fit the model.

        Parameters
        ----------
        X_train
            Training covariates with shape ``(n, p)``. Any eager dataframe
            supported by narwhals (e.g., pandas, polars, pyarrow) is accepted,
            with categorical columns encoded like in stochtree.
        y_train
            Training outcomes of length ``n``.
        X_test
            Optional test covariates; if given, predictions are cached on
            them in `y_hat_test`.
        observation_weights
            Not supported, must be `None`.
        num_gfr
            Number of grow-from-root iterations. Must be set explicitly to ``0``.
        num_burnin
            Number of MCMC burn-in iterations. Recommended to set it to 1000.
        num_mcmc
            Number of retained MCMC iterations per chain. Recommended to set it
            to 1000.
        general_params
            Optional override for the keys of `GeneralParams`.
        mean_forest_params
            Optional override for the keys of `MeanForestParams`.
        bart_kwargs
            Additional arguments forwarded to `bartz.Bart`. Use this to set
            ``devices`` and ``rm_const=False`` when wrapping `sample` in
            `jax.jit`.

        Raises
        ------
        NotImplementedError
            If ``num_gfr`` is non-zero or `observation_weights` is set.

        Notes
        -----
        The signature mirrors `stochtree.BARTModel.sample`, restricted to the
        arguments bartz supports. This method invokes `bartz.Bart` under the
        hood.
        """
        if num_gfr != 0:
            msg = (
                'num_gfr must be 0; the grow-from-root sampler is not available'
                ' in bartz.'
            )
            raise NotImplementedError(msg)
        if observation_weights is not None:
            msg = (
                'observation_weights are not supported, because as of stochtree'
                ' 0.4.5 they do not work as documented, so there is no'
                ' stochtree behavior to mimic.'
            )
            raise NotImplementedError(msg)

        gp = build_dataclass(GeneralParams, general_params, 'general_params')
        mfp = build_dataclass(
            MeanForestParams, mean_forest_params, 'mean_forest_params'
        )

        is_probit = gp.outcome_model.outcome == 'binary'

        X_train_arr, y_train_arr, variable_weights = self._prepare_training_inputs(
            X_train, y_train, gp
        )

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

        if mfp.sample_sigma2_leaf:
            leaf_init_kw: dict = {
                'sum_trees_prior_cov_inv': resolve_leaf_variance_prior(
                    mfp.sigma2_leaf_shape,
                    mfp.sigma2_leaf_scale,
                    sigma2_leaf_init,
                    num_trees=mfp.num_trees,
                    is_probit=is_probit,
                    var_resid_train=var_resid_train,
                )
            }
        else:
            leaf_init_kw = {}

        if is_probit:
            # stochtree pins σ²=1 for probit; bartz binary branch ignores the
            # variance prior, so we leave the scale/init at their 'auto'
            # defaults (bartz rejects explicit values for binary outcomes).
            sigma_df_arg: FloatLike = 3.0
            sigma_scale_arg: FloatLike | Literal['auto'] = 'auto'
            sigma_init_arg: FloatLike | Literal['auto'] = 'auto'
            sigma2_init_stored: FloatLike = 1.0
        else:
            sigma_df_arg, sigma_scale_arg, sigma_init_arg, sigma2_init_stored = (
                resolve_variance_prior(
                    gp.sigma2_global_shape,
                    gp.sigma2_global_scale,
                    gp.sigma2_init,
                    var_resid_train,
                )
            )

        binner = partial(RangeEvenBinner, max_bins=256)

        seed = 0 if gp.random_seed is None else gp.random_seed
        if not is_key(seed):
            seed = random.key(seed)
        keys = split(seed)

        kwargs: dict = dict(
            x_train=X_train_arr.T,
            y_train=y_for_bartz,
            outcome_type='binary' if is_probit else 'continuous',
            binner=binner,
            varprob=variable_weights,
            sigma_df=sigma_df_arg,
            sigma_scale=sigma_scale_arg,
            sigma_init=sigma_init_arg,
            k=bartz_k,
            power=mfp.beta,
            base=mfp.alpha,
            tau_num=tau_num_arg,
            num_trees=mfp.num_trees,
            n_save=num_mcmc,
            n_burn=num_burnin,
            n_skip=gp.keep_every,
            printevery=None,
            num_chains=bart_num_chains,
            seed=keys.pop(),
            maxdepth=mfp.max_depth + 1,
            precompute_predict_train=True,
        )
        kwargs.update(bart_kwargs)
        # min_samples_leaf maps to bartz's acceptance-time veto.
        # min_points_per_decision_node = 2 * min_samples_leaf makes the
        # proposal skip leaves too small to split, without changing the target
        # posterior, which thus matches stochtree. The leaf variance prior goes
        # through init_kw because Bart's sigma_mu_df ties the prior scale to
        # the initial value, while stochtree sets them independently.
        # User-supplied init_kw values win on conflicts.
        kwargs = dict(
            kwargs,
            init_kw=dict(
                {
                    'min_points_per_leaf': mfp.min_samples_leaf,
                    'min_points_per_decision_node': 2 * mfp.min_samples_leaf,
                    **leaf_init_kw,
                },
                **kwargs.get('init_kw', {}),
            ),
        )
        self._bart = Bart(**kwargs)
        self._ppd_key = keys.pop()
        self._ppd_count = 0
        n_train, num_covariates = X_train_arr.shape
        self._finalize_sample(
            outcome_model=gp.outcome_model,
            num_burnin=num_burnin,
            num_mcmc=num_mcmc,
            num_chains=gp.num_chains,
            keep_every=gp.keep_every,
            n_train=n_train,
            num_covariates=num_covariates,
            sample_sigma2_leaf=mfp.sample_sigma2_leaf,
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
        keep_every: int,
        n_train: int,
        num_covariates: int,
        sample_sigma2_leaf: bool,
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
        self.sample_sigma2_global = not is_probit
        self.sample_sigma2_leaf = sample_sigma2_leaf
        self.probit_outcome_model = is_probit
        self.outcome_model = outcome_model
        self.num_gfr = 0
        self.num_burnin = num_burnin
        self.num_mcmc = num_mcmc
        self.num_chains = num_chains
        self.keep_every = keep_every
        self.num_samples = num_mcmc * num_chains
        self.n_train = n_train
        self.has_test = X_test is not None
        self.num_covariates = num_covariates
        self.sigma2_init = sigma2_init
        self.y_bar = y_bar
        self.y_std = y_std
        self.has_rfx = False
        self.include_mean_forest = True
        self.include_variance_forest = False

        # cached outputs in stochtree's (n, num_samples) layout, original scale
        self.y_hat_train = self._predict_y_hat_internal('train')
        if X_test is not None:
            self.y_hat_test = self._predict_y_hat_internal(self._prepare_x(X_test).T)
            self.n_test, _ = self.y_hat_test.shape
        else:
            self.y_hat_test = None
            self.n_test = 0

        if is_probit:
            self.global_var_samples = None
        else:
            sigma = self._bart.get_error_sdev()
            self.global_var_samples = (sigma * y_std) ** 2

        if sample_sigma2_leaf:
            self.leaf_scale_samples = get_leaf_scale_samples(
                self._bart._main_trace  # noqa: SLF001
            )
        else:
            self.leaf_scale_samples = None

    @overload
    def predict(
        self,
        X: Real[ArrayLike, 'm p'] | DataFrame,
        *,
        type: Literal['posterior', 'mean'] = 'posterior',
        terms: Literal['y_hat', 'mean_forest'],
        scale: Literal['linear', 'probability', 'class'] = 'linear',
    ) -> Shaped[Array, 'm num_samples'] | Shaped[Array, ' m']: ...

    @overload
    def predict(
        self,
        X: Real[ArrayLike, 'm p'] | DataFrame,
        *,
        type: Literal['posterior', 'mean'] = 'posterior',
        terms: Literal['all'] = 'all',
        scale: Literal['linear', 'probability', 'class'] = 'linear',
    ) -> dict[str, Shaped[Array, 'm num_samples']] | dict[str, Shaped[Array, ' m']]: ...

    @overload
    def predict(
        self,
        X: Real[ArrayLike, 'm p'] | DataFrame,
        *,
        type: Literal['posterior', 'mean'] = 'posterior',
        terms: Sequence[Literal['y_hat', 'mean_forest', 'all']],
        scale: Literal['linear', 'probability', 'class'] = 'linear',
    ) -> (
        Shaped[Array, 'm num_samples']
        | Shaped[Array, ' m']
        | dict[str, Shaped[Array, 'm num_samples']]
        | dict[str, Shaped[Array, ' m']]
    ): ...

    @forbid_x64
    def predict(
        self,
        X: Real[ArrayLike, 'm p'] | DataFrame,
        *,
        type: Literal['posterior', 'mean'] = 'posterior',  # noqa: A002
        terms: Literal['y_hat', 'mean_forest', 'all']
        | Sequence[Literal['y_hat', 'mean_forest', 'all']] = 'all',
        scale: Literal['linear', 'probability', 'class'] = 'linear',
    ) -> (
        Shaped[Array, 'm num_samples']
        | Shaped[Array, ' m']
        | dict[str, Shaped[Array, 'm num_samples']]
        | dict[str, Shaped[Array, ' m']]
    ):
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
        Either a single jax array (for a single requested term) or a dict keyed by term name.

        Raises
        ------
        NotSampledError
            If `sample` has not been called yet.
        """
        if not self.sampled:
            raise NotSampledError(_NOT_SAMPLED_MSG)
        terms_tuple = check_predict_args(type, scale, terms, self.probit_outcome_model)

        pred = self._predict_y_hat_internal(self._prepare_x(X).T)

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

    def compute_contrast(
        self,
        X_0: Real[ArrayLike, 'm p'] | DataFrame,
        X_1: Real[ArrayLike, 'm p'] | DataFrame,
        *,
        type: Literal['posterior', 'mean'] = 'posterior',  # noqa: A002
        scale: Literal['linear', 'probability'] = 'linear',
    ) -> Float32[Array, 'm num_samples'] | Float32[Array, ' m']:
        """Compute the difference between the predictions at `X_1` and at `X_0`.

        Parameters
        ----------
        X_0
            Covariates of the subtracted predictions, with shape ``(m, p)``.
        X_1
            Covariates of the other predictions, with shape ``(m, p)``.
        type
            ``'posterior'`` returns one contrast per posterior sample, with
            shape ``(m, num_samples)``. ``'mean'`` averages them, returning a
            vector of shape ``(m,)``.
        scale
            ``'linear'``, or, for probit binary regression, ``'probability'``
            to contrast ``Phi(eta)``.

        Returns
        -------
        The contrast ``predict(X_1) - predict(X_0)``.

        Raises
        ------
        NotSampledError
            If `sample` has not been called yet.
        """
        if not self.sampled:
            raise NotSampledError(_NOT_SAMPLED_MSG)
        check_interval_scale(scale)
        check_predict_args(type, scale, 'y_hat', self.probit_outcome_model)
        pred_1 = self.predict(X_1, terms='y_hat', scale=scale)
        pred_0 = self.predict(X_0, terms='y_hat', scale=scale)
        contrast = pred_1 - pred_0
        if type == 'mean':
            return contrast.mean(axis=1)
        else:
            return contrast

    @overload
    def compute_posterior_interval(
        self,
        X: Real[ArrayLike, 'm p'] | DataFrame,
        *,
        terms: Literal['y_hat', 'mean_forest'],
        level: FloatLike = 0.95,
        scale: Literal['linear', 'probability'] = 'linear',
    ) -> dict[str, Float32[Array, ' m']]: ...

    @overload
    def compute_posterior_interval(
        self,
        X: Real[ArrayLike, 'm p'] | DataFrame,
        *,
        terms: Literal['all'] = 'all',
        level: FloatLike = 0.95,
        scale: Literal['linear', 'probability'] = 'linear',
    ) -> dict[str, dict[str, Float32[Array, ' m']]]: ...

    @overload
    def compute_posterior_interval(
        self,
        X: Real[ArrayLike, 'm p'] | DataFrame,
        *,
        terms: Sequence[Literal['y_hat', 'mean_forest', 'all']],
        level: FloatLike = 0.95,
        scale: Literal['linear', 'probability'] = 'linear',
    ) -> (
        dict[str, Float32[Array, ' m']] | dict[str, dict[str, Float32[Array, ' m']]]
    ): ...

    def compute_posterior_interval(
        self,
        X: Real[ArrayLike, 'm p'] | DataFrame,
        *,
        terms: Literal['y_hat', 'mean_forest', 'all']
        | Sequence[Literal['y_hat', 'mean_forest', 'all']] = 'all',
        level: FloatLike = 0.95,
        scale: Literal['linear', 'probability'] = 'linear',
    ) -> dict[str, Float32[Array, ' m']] | dict[str, dict[str, Float32[Array, ' m']]]:
        """Compute equal-tailed posterior credible intervals of the predictions.

        Parameters
        ----------
        X
            Covariates with shape ``(m, p)``.
        terms
            Like in `predict`.
        level
            Posterior probability of the interval, between 0 and 1.
        scale
            ``'linear'``, or, for probit binary regression, ``'probability'``
            for intervals on ``Phi(eta)``.

        Returns
        -------
        A dict with keys ``'lower'`` and ``'upper'``, or, for multiple terms, a dict of such dicts keyed like the output of `predict`.

        Raises
        ------
        NotSampledError
            If `sample` has not been called yet.
        """
        if not self.sampled:
            raise NotSampledError(_NOT_SAMPLED_MSG)
        check_interval_scale(scale)
        level = jnp.asarray(level)
        level = error_if(level, ~((level > 0) & (level < 1)), 'level must be in (0, 1)')
        pred = self.predict(X, terms=terms, scale=scale)
        if isinstance(pred, dict):
            return {k: posterior_interval(v, level) for k, v in pred.items()}
        else:
            return posterior_interval(pred, level)

    @forbid_x64
    def sample_posterior_predictive(
        self,
        X: Real[ArrayLike, 'm p'] | DataFrame,
        *,
        num_draws_per_sample: int | None = None,
    ) -> (
        Real[Array, 'm num_samples'] | Real[Array, 'm num_samples num_draws_per_sample']
    ):
        """Sample outcomes from the posterior predictive distribution.

        Parameters
        ----------
        X
            Covariates with shape ``(m, p)``.
        num_draws_per_sample
            Number of draws per posterior sample. If `None`, it is 1 if
            there are at least 1000 posterior samples, else enough to get at
            least 1000 draws per observation.

        Returns
        -------
        The draws, with shape ``(m, num_samples)`` if there is one draw per sample, else ``(m, num_samples, num_draws_per_sample)``.

        Raises
        ------
        NotSampledError
            If `sample` has not been called yet.

        Notes
        -----
        The draws are random, and change at each call. Their sequence is
        determined by ``general_params['random_seed']``.
        """
        if not self.sampled:
            raise NotSampledError(_NOT_SAMPLED_MSG)
        if num_draws_per_sample is None:
            num_draws_per_sample = (
                1 if self.num_samples >= 1000 else math.ceil(1000 / self.num_samples)
            )
        eta = self._predict_y_hat_internal(self._prepare_x(X).T)
        key = random.fold_in(self._ppd_key, self._ppd_count)
        self._ppd_count += 1
        return sample_outcome(key, eta, self.global_var_samples, num_draws_per_sample)

    def has_term(self, term: Literal['mean_forest', 'y_hat', 'all']) -> bool:
        """Return whether the model includes a term.

        Parameters
        ----------
        term
            The term to check.

        Returns
        -------
        `True` for the terms in the signature, else `False`.
        """
        return term in ('mean_forest', 'y_hat', 'all')

    def extract_parameter(
        self,
        term: Literal[
            'sigma2',
            'global_error_scale',
            'sigma2_global',
            'sigma2_leaf',
            'leaf_scale',
            'y_hat_train',
            'y_hat_test',
        ],
    ) -> Float32[Array, ' num_samples'] | Float32[Array, 'n num_samples']:
        """Return the posterior samples of a parameter by name.

        Parameters
        ----------
        term
            ``'sigma2'`` (alias ``'global_error_scale'`` or
            ``'sigma2_global'``) for `global_var_samples`, ``'sigma2_leaf'``
            (alias ``'leaf_scale'``) for `leaf_scale_samples`,
            ``'y_hat_train'``, or ``'y_hat_test'``.

        Returns
        -------
        The samples, indexed by the last axis.

        Raises
        ------
        NotSampledError
            If `sample` has not been called yet.
        ValueError
            If `term` is not valid or the model does not have it.
        """
        if not self.sampled:
            raise NotSampledError(_NOT_SAMPLED_MSG)
        if term in ('sigma2', 'global_error_scale', 'sigma2_global'):
            value = self.global_var_samples
            what = 'global variance parameter samples'
        elif term in ('sigma2_leaf', 'leaf_scale'):
            value = self.leaf_scale_samples
            what = 'leaf variance parameter samples'
        elif term == 'y_hat_train':
            value = self.y_hat_train
            what = 'in-sample mean function prediction samples'
        elif term == 'y_hat_test':
            value = self.y_hat_test
            what = 'test set mean function prediction samples'
        else:
            msg = f'term {term} is not a valid BART model term'
            raise ValueError(msg)
        if value is None:
            msg = f'This model does not have {what}'
            raise ValueError(msg)
        return value

    def summary(self) -> None:
        """Print a description of the model and summaries of its posterior samples.

        Raises
        ------
        NotSampledError
            If `sample` has not been called yet.
        """
        if not self.sampled:
            raise NotSampledError(_NOT_SAMPLED_MSG)
        out = f'BART Model Summary:\n-------------------\n{self}\n'
        if self.global_var_samples is not None:
            out += summarize_samples(
                'Summary of sigma^2 posterior: ', 'samples', self.global_var_samples
            )
        if self.leaf_scale_samples is not None:
            out += summarize_samples(
                'Summary of leaf scale posterior: ', 'samples', self.leaf_scale_samples
            )
        out += summarize_samples(
            'Summary of in-sample posterior mean predictions: \n',
            'observations',
            self.y_hat_train.mean(axis=1),
        )
        if self.y_hat_test is not None:
            out += summarize_samples(
                'Summary of test-set posterior mean predictions: \n',
                'observations',
                self.y_hat_test.mean(axis=1),
            )
        print(out)  # noqa: T201

    def __str__(self) -> str:
        if self.sampled:
            return self._describe()
        else:
            return 'Empty BARTModel() object: not yet sampled'

    __repr__ = __str__

    def _describe(self) -> str:
        """Describe the fitted model with the same text as stochtree."""
        terms = ['mean forest']
        if self.sample_sigma2_global:
            terms.append('global error variance model')
        if self.sample_sigma2_leaf:
            terms.append('mean forest leaf scale model')
        if len(terms) > 2:
            terms_str = f'{", ".join(terms[:-1])}, and {terms[-1]}'
        else:
            terms_str = ' and '.join(terms)
        link = 'with a probit link' if self.probit_outcome_model else 'as gaussian'
        out = (
            f'BARTModel run with {terms_str}\nOutcome was modeled {link} with a'
            ' constant leaf prior for the mean forest'
        )
        if self.standardize:
            out += '\nOutcome was standardized'
        chains = 'chain' if self.num_chains == 1 else 'chains'
        if self.keep_every == 1:
            thinning = 'retaining every iteration (i.e. no thinning)'
        else:
            thinning = f'retaining every {self.keep_every}th iteration (i.e. thinning)'
        out += (
            f'\nThe sampler was run for {self.num_gfr} GFR iterations, with'
            f' {self.num_chains} {chains} of {self.num_burnin} burn-in iterations'
            f' and {self.num_mcmc} MCMC iterations, {thinning}\n'
        )
        return out

    def _prepare_x(self, X: Real[ArrayLike, 'm p'] | DataFrame) -> Real[Array, 'm p']:
        """Convert covariates to a 2-D jax array, replaying the fitted preprocessor if any."""
        if self._preprocessor is None:
            return check_X(X)
        df = to_narwhals(X)
        if df is None:
            msg = (
                'this model was fit on a DataFrame, so prediction covariates must'
                ' also be a DataFrame with the same columns; got a'
                ' non-DataFrame. Passing a raw array would bypass the fitted'
                ' preprocessing (e.g. one-hot encoding) and silently misalign the'
                ' features.'
            )
            raise TypeError(msg)
        return jnp.asarray(self._preprocessor.transform(df))

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


def resolve_leaf_variance_prior(
    shape: FloatLike,
    scale: FloatLike | None,
    sigma2_leaf_init: FloatLike,
    *,
    num_trees: int,
    is_probit: bool,
    var_resid_train: FloatLike,
) -> Wishart:
    """Translate stochtree's leaf variance prior to a `Wishart` on the sum of trees precision.

    Despite the documented ``IG(shape, scale)``, stochtree's sampler uses the
    posterior ``IG(shape / 2 + L / 2, scale / 2 + S / 2)``, i.e., the prior is
    ``IG(shape / 2, scale / 2)``, which is `Wishart` with ``nu = shape`` and
    ``rate = scale`` on the leaf precision. The variance of the sum of trees is
    `num_trees` times the one of a leaf, so its scale is too.
    """
    if scale is None:
        sum_scale = 1.0 if is_probit else var_resid_train
    else:
        sum_scale = num_trees * scale
    return Wishart(
        nu=shape, rate=sum_scale, value=jnp.reciprocal(num_trees * sigma2_leaf_init)
    )


@jit
def get_leaf_scale_samples(trace: MainTrace) -> Float32[Array, ' num_samples']:
    """Return the leaf variance samples, chains concatenated."""
    # the trace has the precision of the sum of trees
    *_, num_trees, _ = trace.split_tree.shape
    prec = trace.sum_trees_prior_cov_inv * num_trees
    if trace.has_chains:
        prec = lax.collapse(prec, 0, 2)
    return jnp.reciprocal(prec)


def resolve_variance_prior(
    shape: FloatLike,
    scale: FloatLike,
    sigma2_init: FloatLike | None,
    var_resid_train: FloatLike,
) -> tuple[Float32[Array, ''], Float32[Array, ''], Float32[Array, ''], FloatLike]:
    """Translate stochtree's IG(shape, scale) prior to bartz's error variance prior.

    The IG(shape, scale) prior on σ² is the scaled-inverse-χ² with
    ``sigma_df = 2*shape`` and prior harmonic mean ``square(sigma_scale) =
    scale/shape``; the chain starts at `sigma2_init` (default
    ``var(resid_train)``), decoupled from the prior. The mapping is branchless so
    `shape` / `scale` may be traced; the unrepresentable IG(0, scale>0) (positive
    rate, zero df) yields a NaN that surfaces downstream rather than an error.

    Parameters
    ----------
    shape
        Stochtree's ``sigma2_global_shape``.
    scale
        Stochtree's ``sigma2_global_scale``.
    sigma2_init
        Stochtree's ``sigma2_init``. If `None`, defaults to `var_resid_train`.
    var_resid_train
        Variance of the residual, the default chain start for σ².

    Returns
    -------
    sigma_df : Float32[Array, '']
        Degrees of freedom of bartz's error variance prior.
    sigma_scale : Float32[Array, '']
        Scale of bartz's prior (sqrt of the prior harmonic mean of the variance).
    sigma_init : Float32[Array, '']
        Initial error standard deviation seeding the chain.
    sigma2_init_stored : FloatLike
        The chain starting value of σ², suitable for ``BARTModel.sigma2_init``.
    """
    shape = jnp.asarray(shape, jnp.float32)
    scale = jnp.asarray(scale, jnp.float32)
    sigma2_start = sigma2_init if sigma2_init is not None else var_resid_train
    sigma_init = jnp.sqrt(jnp.asarray(sigma2_start, jnp.float32))
    # IG(shape, scale) <=> scaled-inv-chi2(df=2*shape, harmonic mean=scale/shape).
    # The `scale > 0` guard keeps IG(0, 0) at harmonic mean 0 (avoiding 0/0) while
    # letting IG(0, scale>0) overflow to inf -> NaN rate, flagging it as invalid.
    harmonic_mean = jnp.where(scale > 0, scale / shape, 0.0)
    return 2.0 * shape, jnp.sqrt(harmonic_mean), sigma_init, sigma2_start


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
    if scale == 'class' and set(terms_tuple) != {'y_hat'}:
        # match stochtree: 'class' converts only the single 'y_hat' term, so it
        # rejects 'mean_forest' and 'all' (the latter also pulls in mean_forest)
        msg = "scale='class' is only supported when requesting a single 'y_hat' term"
        raise ValueError(msg)
    return terms_tuple


def check_interval_scale(scale: str) -> None:
    """Validate `scale` for the methods that do not support ``'class'``."""
    if scale not in ('linear', 'probability'):
        msg = f"scale must be 'linear' or 'probability'; got {scale!r}"
        raise ValueError(msg)


@jit
def posterior_interval(
    samples: Float32[Array, 'm num_samples'], level: FloatLike
) -> dict[str, Float32[Array, ' m']]:
    """Compute the equal-tailed interval along the samples axis."""
    tail = (1 - level) / 2
    lower, upper = jnp.quantile(samples, jnp.array([tail, 1 - tail]), axis=1)
    return {'lower': lower, 'upper': upper}


@jit(static_argnames='num_draws')
def sample_outcome(
    key: Key[Array, ''],
    eta: Float32[Array, 'm num_samples'],
    sigma2: Float32[Array, ' num_samples'] | None,
    num_draws: int,
) -> Real[Array, 'm num_samples'] | Real[Array, 'm num_samples num_draws']:
    """Sample outcomes with mean `eta` and variance `sigma2`, or probit if `sigma2` is `None`."""
    noise = random.normal(key, (*eta.shape, num_draws))
    if sigma2 is None:
        draws = jnp.where(eta[..., None] + noise > 0, 1, 0)
    else:
        draws = eta[..., None] + jnp.sqrt(sigma2)[:, None] * noise
    if num_draws == 1:
        return draws.squeeze(-1)
    else:
        return draws


def summarize_samples(header: str, unit: str, x: Float32[Array, ' n']) -> str:
    """Format the mean, standard deviation and quantiles of `x` like stochtree."""
    probs = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    quantiles = jnp.quantile(x, jnp.array(probs)).tolist()
    (n,) = x.shape
    out = (
        f'{header}{n} {unit}, mean = {x.mean().item():.3f}, standard deviation'
        f' = {x.std().item():.3f}, quantiles:\n'
    )
    for p, q in zip(probs, quantiles, strict=True):
        out += f'  {p * 100:5.1f}%: {q:.3f}\n'
    return out


def check_X(
    X: Real[ArrayLike, 'n p'] | DataFrame, *, name: str = 'X'
) -> Real[Array, 'n p']:
    """Convert a DataFrame/array-like to a 2-D jax array in ``(n, p)`` layout."""
    df = to_narwhals(X)
    if df is not None:
        X = df.to_numpy()
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
    if isinstance(y, Series):
        y = y.to_numpy()
    arr = jnp.asarray(y)
    if arr.ndim != 1:
        msg = f'{name} must be 1D (n,); got shape {arr.shape}'
        raise ValueError(msg)
    return arr
