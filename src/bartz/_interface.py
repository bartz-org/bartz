# bartz/src/bartz/_interface.py
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

"""Main high-level interface of the package."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from enum import Enum
from functools import cached_property, partial
from os import cpu_count

# WORKAROUND(python<3.15): use frozendict instead of MappingProxyType
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypedDict, overload
from warnings import warn

import jax
import jax.numpy as jnp
from equinox import Module, error_if, field
from jax import (
    Device,
    debug_nans,
    device_count,
    device_put,
    jit,
    lax,
    make_mesh,
    random,
    tree,
)
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import ndtr
from jax.sharding import AxisType, Mesh, PartitionSpec
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Float32, Int32, Key, Real, Shaped, UInt
from numpy import ndarray

from bartz import mcmcstep
from bartz._jaxext import equal_shards, is_key, split
from bartz._jaxext.scipy.special import ndtri
from bartz._jaxext.scipy.stats import invgamma
from bartz.grove import (
    TreesTrace,
    check_trace,
    evaluate_forest,
    forest_depth_distr,
    format_tree,
    points_per_node_distr,
)
from bartz.mcmcloop import (
    BurninTrace,
    MainTrace,
    RunMCMCResult,
    compute_varcount,
    evaluate_trace,
    make_default_callback,
    run_mcmc,
)
from bartz.mcmcstep import OutcomeType, make_p_nonterminal
from bartz.mcmcstep._state import (
    ArrayLike,
    FloatLike,
    _inv_via_chol_with_gersh,
    chain_to_axis,
    chain_vmap_axes,
    chainful_axis,
    chol_with_gersh,
    trace_sample_axes,
)
from bartz.prepcovars import Binner, BinnerFactory, UniqueQuantileBinner
from bartz.prepcovars._prepcovars import _sigma2_from_cg, _sigma2_from_ols

CG_MAXITER = 20


class PredictKind(Enum):
    """Kind of output of `Bart.predict`."""

    mean = 'mean'
    """The posterior mean of the conditional mean, shape ``(m,)`` (or
    ``(k, m)`` for multivariate regression)."""

    mean_samples = 'mean_samples'
    """Per-sample conditional mean, shape ``(num_chains * n_save, m)``
    (or ``(num_chains * n_save, k, m)``). For binary regression, this is
    the probit-transformed sum-of-trees."""

    outcome_samples = 'outcome_samples'
    """Samples of the outcome variable, shape ``(num_chains * n_save,
    m)`` (or ``(num_chains * n_save, k, m)``). For binary regression,
    these are Bernoulli draws. For continuous regression, these are
    Gaussian draws with the posterior noise variance."""

    latent_samples = 'latent_samples'
    """Raw sum-of-trees values, shape ``(num_chains * n_save, m)`` (or
    ``(num_chains * n_save, k, m)``)."""


class DataFrame(Protocol):
    """DataFrame duck-type for `Bart`."""

    columns: Sequence[str]
    """The names of the columns."""

    def to_numpy(self) -> ndarray:
        """Convert the dataframe to a 2d numpy array with columns on the second axis."""
        ...


class Series(Protocol):
    """Series duck-type for `Bart`."""

    name: str | None
    """The name of the series."""

    def to_numpy(self) -> ndarray:
        """Convert the series to a 1d numpy array."""
        ...


class Bart(Module):
    R"""
    Nonparametric regression with Bayesian Additive Regression Trees (BART) [2]_.

    Regress `y_train` on `x_train` with a latent mean function represented as
    a sum of decision trees. The inference is carried out by sampling the
    posterior distribution of the tree ensemble with an MCMC.

    Parameters
    ----------
    x_train
        The training predictors.
    y_train
        The training responses. For univariate regression, a 1D array of shape
        `(n,)`. For multivariate regression, a 2D array of shape `(k, n)` where
        `k` is the number of response components, as introduced in [3]_. For
        binary regression, the convention is that non-zero values mean 1, zero
        mean 0, like booleans.
    outcome_type
        The type of regression. ``'continuous'`` for continuous regression,
        ``'binary'`` for binary regression with probit link. For multivariate
        regression, a scalar value applies to all components; alternatively, a
        sequence of per-component types (e.g., ``['binary', 'continuous']``)
        specifies mixed outcome types.
    sparse
        Whether to activate variable selection on the predictors as done in
        [1]_.
    theta
    a
    b
    rho
        Hyperparameters of the sparsity prior used for variable selection.

        The prior distribution on the choice of predictor for each decision rule
        is

        .. math::
            (s_1, \ldots, s_p) \sim
            \operatorname{Dirichlet}(\mathtt{theta}/p, \ldots, \mathtt{theta}/p).

        If `theta` is not specified, it's a priori distributed according to

        .. math::
            \frac{\mathtt{theta}}{\mathtt{theta} + \mathtt{rho}} \sim
            \operatorname{Beta}(\mathtt{a}, \mathtt{b}).

        If not specified, `rho` is set to the number of predictors p. To tune
        the prior, consider setting a lower `rho` to prefer more sparsity.
        If setting `theta` directly, it should be in the ballpark of p or lower
        as well.
    varprob
        The probability distribution over the `p` predictors for choosing a
        predictor to split on in a decision node a priori. Must be > 0. It does
        not need to be normalized to sum to 1. If not specified, use a uniform
        distribution. If ``sparse=True``, this is used as initial value for the
        MCMC.
    binner
        A callable that, given the training predictors and a random key,
        returns a `~bartz.prepcovars.Binner` instance. The default is
        `~bartz.prepcovars.UniqueQuantileBinner`, which places cutpoints at
        the quantiles of each predictor. Other built-in options are
        `~bartz.prepcovars.RangeEvenBinner` (evenly-spaced cutpoints over the
        observed range) and `~bartz.prepcovars.GivenSplitsBinner` (R BART
        ``xinfo`` format). To pass options, use `functools.partial`, e.g.
        ``binner=partial(UniqueQuantileBinner, max_bins=128)``.
    rm_const
        How to treat predictors with no associated decision rules (i.e., there
        are no available cutpoints for that predictor). If `True` (default),
        they are ignored. If `False`, an error is raised if there are any.
    sigest
        An estimate of the residual standard deviation on `y_train`, used to set
        `lambda_`. Ignored if `lambda_` is specified. For multivariate regression,
        can be a scalar (broadcast to all components) or a `(k,)` vector of
        per-component estimates. For mixed outcome types, binary component
        values are ignored. Can be one of the following special values to set
        automatically based on the data:

        'ols-or-variance'
            If less than two datapoints, set ``sigest=1``. If ``n > p``, use the
            OLS error standard deviation estimate (w/ intercept, w/o taking into
            account `w`), else use the standard deviation of `y_train`.
        'gc'
            Use an approximate and regularized version of the OLS residual
            standard deviation estimate.
        'auto' (default)
            Use 'ols-or-variance' if the dataset is smaller than a threshold,
            else 'gc' for larger datasets.
    sigdf
        The degrees of freedom of the scaled inverse-chisquared prior on the
        noise variance. For multivariate regression, the Inverse-Wishart
        degrees of freedom are set to `sigdf + k - 1`.
    sigquant
        The quantile of the prior on the noise variance that shall match
        `sigest` to set the scale of the prior. Ignored if `lambda_` is specified.
    k
        The inverse scale of the prior standard deviation on the latent mean
        function, relative to half the observed range of `y_train`. If `y_train`
        has less than two elements, `k` is ignored and the scale is set to 1.
    power
    base
        Parameters of the prior on tree node generation. The probability that a
        node at depth `d` (0-based) is non-terminal is ``base / (1 + d) **
        power``.
    lambda_
        The prior harmonic mean of the error variance. (The harmonic mean of x
        is 1/mean(1/x).) If not specified, it is set based on `sigest` and
        `sigquant`. For multivariate regression, can be a scalar (broadcast
        to all components) or a `(k,)` vector. For mixed outcome types, binary
        component values are ignored.
    tau_num
        The numerator in the expression that determines the prior standard
        deviation of leaves. If not specified, default to ``(max(y_train) -
        min(y_train)) / 2`` (or 1 if `y_train` has less than two elements) for
        continuous regression, and 3 for binary regression. For multivariate
        regression, the range is computed per component. For mixed outcome
        types, each component uses the default for its type.
    offset
        The prior mean of the latent mean function. If not specified, it is set
        to the mean of `y_train` for continuous regression, and to
        ``Phi^-1(mean(y_train != 0))`` for binary regression. If `y_train` is
        empty, `offset` is set to 0. With binary regression, if `y_train` is
        all zero or all non-zero, it is set to ``Phi^-1(1/(n+1))`` or
        ``Phi^-1(n/(n+1))``, respectively. For multivariate regression, can be
        a scalar (broadcast to all components) or a `(k,)` vector. If not
        specified, it is set to the per-component mean of `y_train`. For mixed
        outcome types, each component uses the default for its type.
    w
        Coefficients that rescale the error standard deviation on each
        datapoint. Not specifying `w` is equivalent to setting it to 1 for all
        datapoints. Note: `w` is ignored in the automatic determination of
        `sigest`, so either the weights should be O(1), or `sigest` should be
        specified by the user. Shape ``(n,)`` applies the same scalar weight
        to every outcome component; for multivariate continuous regression,
        ``(k, n)`` instead supplies a per-component weight per datapoint.
    missing
        Boolean mask with the same shape as `y_train`; `True` marks entries
        to be ignored by the MCMC. Values of `y_train` must be finite
        everywhere, including at masked positions. If 2-D,
        ``error_cov_scale`` must be diagonal.
    num_trees
        The number of trees used to represent the latent mean function.
    n_save
        The number of MCMC samples to save, after burn-in, per chain. The
        total trace length across all chains is ``num_chains * n_save``.
    n_burn
        The number of initial MCMC samples to discard as burn-in. This number
        of samples is discarded from each chain.
    n_skip
        The thinning factor for the MCMC samples, after burn-in.
    printevery
        The number of iterations (including thinned-away ones) between each log
        line. Set to `None` to disable logging. ^C interrupts the MCMC only
        every `printevery` iterations, so with logging disabled it's impossible
        to kill the MCMC conveniently.
    num_chains
        The number of independent Markov chains to run.

        The difference between ``num_chains=None`` and ``num_chains=1`` is that
        in the latter case in the object attributes and some methods there will
        be an explicit chain axis of size 1.
    num_chain_devices
        The number of devices to spread the chains across. Must be a divisor of
        `num_chains`. Each device will run a fraction of the chains. If 'auto'
        (default) and running on cpu, the number of devices is picked
        automatically based on the number of cores and the number of virtual jax
        cpu devices.
    num_data_devices
        The number of devices to split datapoints across. Must be a divisor of
        `n`. This is useful only with very high `n`, about > 1000_000.

        If both num_chain_devices and num_data_devices are specified, the total
        number of devices used is the product of the two.
    devices
        One or more devices used to run the MCMC on. If not specified, the
        computation will follow the placement of the input arrays. If a list of
        devices, this argument can be longer than the number of devices needed.
    seed
        The seed for the random number generator.
    maxdepth
        The maximum depth of the trees. This is 1-based, so with the default
        ``maxdepth=6``, the depths of the levels range from 0 to 5.
    init_kw
        Additional arguments passed to `bartz.mcmcstep.init`.
    run_mcmc_kw
        Additional arguments passed to `bartz.mcmcloop.run_mcmc`.

    References
    ----------
    .. [1] Linero, Antonio R. (2018). “Bayesian Regression Trees for
       High-Dimensional Prediction and Variable Selection”. In: Journal of the
       American Statistical Association 113.522, pp. 626-636.
    .. [2] Hugh A. Chipman, Edward I. George, Robert E. McCulloch "BART:
       Bayesian additive regression trees," The Annals of Applied Statistics,
       Ann. Appl. Stat. 4(1), 266-298, (March 2010).
    .. [3] Um, Seungha, Antonio R. Linero, Debajyoti Sinha, and Dipankar
       Bandyopadhyay (2023). "Bayesian additive regression trees for
       multivariate skewed responses". In: Statistics in Medicine 42.3,
       pp. 246-263.

    """

    _main_trace: MainTrace
    _burnin_trace: BurninTrace
    _mcmc_state: mcmcstep.State
    _binner: Binner
    _binary_mask: Bool[Array, ''] | Bool[Array, ' k']
    # WORKAROUND(jax<0.9.1): use `jax.tree.static` instead of `field(static=True)`
    _x_train_fmt: Any = field(static=True)

    sigest: Float32[Array, ''] | Float32[Array, ' k'] | None = None
    """The estimated standard deviation of the error used to set `lambda_`."""

    _w: Float32[Array, ' n'] | Float32[Array, 'k n'] | None = None

    def __init__(
        self,
        x_train: Real[ArrayLike, 'p n'] | DataFrame,
        y_train: Float32[ArrayLike, ' n']
        | Float32[ArrayLike, 'k n']
        | Series
        | DataFrame,
        *,
        outcome_type: OutcomeType | str | Sequence[OutcomeType | str] = 'continuous',
        sparse: bool = False,
        theta: FloatLike | None = None,
        a: FloatLike = 0.5,
        b: FloatLike = 1.0,
        rho: FloatLike | None = None,
        varprob: Float[ArrayLike, ' p'] | None = None,
        binner: BinnerFactory = UniqueQuantileBinner,
        rm_const: bool = True,
        sigest: FloatLike
        | Float[ArrayLike, ' k']
        | Literal['auto', 'ols-or-variance', 'cg'] = 'auto',
        sigdf: FloatLike = 3.0,
        sigquant: FloatLike = 0.9,
        k: FloatLike = 2.0,
        power: FloatLike = 2.0,
        base: FloatLike = 0.95,
        lambda_: FloatLike | Float[ArrayLike, ' k'] | None = None,
        tau_num: FloatLike | None = None,
        offset: FloatLike | Float[ArrayLike, ' k'] | None = None,
        w: Float[ArrayLike, ' n']
        | Float[ArrayLike, 'k n']
        | Series
        | DataFrame
        | None = None,
        missing: Bool[ArrayLike, ' n']
        | Bool[ArrayLike, 'k n']
        | Series
        | DataFrame
        | None = None,
        num_trees: int = 200,
        n_save: int = 1000,
        n_burn: int = 1000,
        n_skip: int = 1,
        printevery: int | None = 100,
        num_chains: int | None = 4,
        num_chain_devices: int | None | Literal['auto'] = 'auto',
        num_data_devices: int | None = None,
        devices: Literal['cpu', 'gpu'] | Device | Sequence[Device] | None = None,
        seed: int | Key[Array, ''] = 0,
        maxdepth: int = 6,
        init_kw: Mapping = MappingProxyType({}),
        run_mcmc_kw: Mapping = MappingProxyType({}),
    ) -> None:
        # check data and put it in the right format
        x_train, x_train_fmt = _process_predictor_input(x_train)
        y_train = _process_response_input(y_train)
        _check_same_length(x_train, y_train)

        if w is not None:
            # keep=True because `w` is donated downstream but also retained
            # as `self._w` for prediction
            w, self._w = _process_response_input(w, keep=True)
            _check_same_length(x_train, w)

        if missing is not None:
            missing = _process_response_input(missing, dtype=jnp.bool_)
            _check_same_length(x_train, missing)

        # check data types are correct for continuous/binary/multivariate regression
        outcome_type, binary_mask = _check_type_settings(y_train, outcome_type, w)

        # process sparsity settings
        theta, a, b, rho = _process_sparsity_settings(x_train, sparse, theta, a, b, rho)

        # process "standardization" settings
        offset = _process_offset_settings(y_train, binary_mask, offset)
        leaf_prior_cov_inv = _process_leaf_variance_settings(
            y_train, binary_mask, k, num_trees, tau_num
        )
        error_cov_df, error_cov_scale, sigest = _process_error_variance_settings(
            x_train,
            y_train,
            outcome_type,
            binary_mask,
            sigest,
            sigdf,
            sigquant,
            lambda_,
        )

        # split the user-provided seed into an mcmc key and a binner key
        if not is_key(seed):
            seed = random.key(seed)
        keys = split(seed)

        # construct the binner and bin x_train
        binner_obj = binner(x_train, key=keys.pop())
        x_train = binner_obj.bin(x_train)
        # copy max_split because `mcmcstep.init` donates it
        max_split = jnp.array(binner_obj.max_split)

        # setup and run mcmc
        initial_state = _setup_mcmc(
            x_train,
            y_train,
            outcome_type,
            offset,
            w,
            missing,
            max_split,
            leaf_prior_cov_inv,
            error_cov_df,
            error_cov_scale,
            power,
            base,
            maxdepth,
            num_trees,
            init_kw,
            rm_const,
            theta,
            a,
            b,
            rho,
            varprob,
            num_chains,
            num_chain_devices,
            num_data_devices,
            devices,
            sparse,
            n_burn,
        )
        result = _run_mcmc(
            initial_state, n_save, n_burn, n_skip, printevery, keys.pop(), run_mcmc_kw
        )

        # set public attributes
        self.sigest = sigest

        # set private attributes
        self._main_trace = result.main_trace
        self._burnin_trace = result.burnin_trace
        self._mcmc_state = result.final_state
        self._binner = binner_obj
        self._x_train_fmt = x_train_fmt
        self._binary_mask = binary_mask

    def predict(
        self,
        x_test: Real[ArrayLike, 'p m'] | DataFrame | str,
        *,
        kind: PredictKind | str = 'mean',
        key: Key[Array, ''] | None = None,
        w: Float[ArrayLike, ' m']
        | Float[ArrayLike, 'k m']
        | Series
        | DataFrame
        | None = None,
    ) -> (
        Float32[Array, ' m']
        | Float32[Array, 'k m']
        | Float32[Array, 'ndpost m']
        | Float32[Array, 'ndpost k m']
    ):
        """
        Compute predictions at `x_test`.

        Parameters
        ----------
        x_test
            The test predictors, or the string ``'train'`` to compute
            predictions on the training data.
        kind
            The kind of output. See `PredictKind` for details.
        key
            Jax random key, required when ``kind='outcome_samples'``.
        w
            Per-observation error scale for ``kind='outcome_samples'``.
            Required when the model was fit with weights and ``x_test`` is
            new data. Shape matches the shape used at fitting: ``(m,)`` for
            scalar weights, ``(k, m)`` for multivariate vector weights.

        Returns
        -------
        Predictions at `x_test` in the requested format.

        Raises
        ------
        ValueError
            If `x_test` has a different format than `x_train`, or if `w`
            is specified when it should be `None`, or if `w` is not
            specified when it is required.

        """
        # parse arguments
        kind = PredictKind(kind)
        if kind is PredictKind.outcome_samples and key is None:
            msg = '`key` not specified'
            raise ValueError(msg)
        w = self._process_w_test(x_test, kind, w)
        x_test = self._process_x_test(x_test, w)

        # invoke jitted implementation
        return predict(
            key,
            self._main_trace,
            x_test,
            w,
            self._mcmc_state.binary_indices,
            self._mcmc_state.binary_y is not None,
            kind,
        )

    @property
    def offset(self) -> Float32[Array, ''] | Float32[Array, ' k']:
        """The prior mean of the latent mean function."""
        return self._mcmc_state.offset

    @property
    def n_save(self) -> int:
        """The number of posterior samples after burn-in saved per chain."""
        sample_axis = trace_sample_axes(self._main_trace).grow_prop_count
        return self._main_trace.grow_prop_count.shape[sample_axis]

    @property
    def num_chains(self) -> int | None:
        """The number of chains, `None` if scalar."""
        return self._mcmc_state.num_chains()

    @property
    def ndpost(self) -> int:
        """The total number of posterior samples after burn-in across all chains."""
        return self._main_trace.grow_prop_count.size

    @property
    def num_trees(self) -> int:
        """Return the number of trees used in the model."""
        forest = self._mcmc_state.forest
        chain_axis = chain_vmap_axes(forest).split_tree
        # chainless split_tree is (num_trees, half_tree_size); num_trees is core axis 0
        axis = chainful_axis(0, chain_axis)
        return forest.split_tree.shape[axis]

    def get_latent_prec(
        self, only_continuous: bool = False
    ) -> (
        Float32[Array, ' n_burn+n_save']
        | Float32[Array, 'n_burn+n_save k k']
        | Float32[Array, 'num_chains n_burn+n_save']
        | Float32[Array, 'num_chains n_burn+n_save k k']
    ):
        """Return the posterior samples of the latent error precision matrix.

        Parameters
        ----------
        only_continuous
            If `True` and the model has mixed binary-continuous outcomes,
            return only the submatrix for the continuous components.

        Returns
        -------
        MCMC samples of the error precision matrix.

        Notes
        -----
        This method is meant to check for convergence, so it returns the full
        MCMC trace and does not concatenate chains together. For probit
        regression, this returns the precision of the latent error term, not
        the Bernoulli precision for the binary outcome. For heteroskedastic
        regression, the returned precision is the global precision parameter,
        that would have to be divided by a squared weight to get the precision
        on a given datapoint.

        Raises
        ------
        ValueError
             If `only_continuous` is `True` but the model has only binary
             outcomes, so there is no continuous submatrix to return.
        """
        binary_indices = self._mcmc_state.binary_indices
        if (
            only_continuous
            and binary_indices is None
            and self._mcmc_state.binary_y is not None
        ):
            msg = 'Model has only binary outcomes, so there is no continuous submatrix to return.'
            raise ValueError(msg)

        return get_latent_prec(
            self._burnin_trace,
            self._main_trace,
            binary_indices,
            only_continuous=only_continuous,
        )

    def get_error_sdev(
        self, mean: bool = False
    ) -> (
        Float32[Array, ' ndpost']
        | Float32[Array, 'ndpost k']
        | Float32[Array, '']
        | Float32[Array, ' k']
    ):
        """Return the error standard deviation, post-burnin, chains concatenated.

        Parameters
        ----------
        mean
            If `True`, average the error covariance matrix across samples before
            taking the square root, returning a single scalar or vector instead
            of posterior samples.

        Returns
        -------
        Posterior samples (or single estimate) of the error standard deviation; NaN for binary outcomes.

        Notes
        -----
        Binary outcomes do have a standard deviation of course, but it's not
        returned by this method because that would require to evaluate
        predictions on a given X, since the Bernoulli variance is p(1-p).
        """
        # binary outcomes are filled with NaN, so disable the NaN check
        with debug_nans(False):
            return get_error_sdev(self._main_trace, self._binary_mask, mean=mean)

    @cached_property
    def varcount(self) -> Int32[Array, 'ndpost p']:
        """Histogram of predictor usage for decision rules in the trees."""
        p = self._mcmc_state.forest.max_split.size
        return varcount(p, self._main_trace)

    @cached_property
    def varcount_mean(self) -> Float32[Array, ' p']:
        """Average of `varcount` across MCMC iterations."""
        return self.varcount.mean(axis=0)

    @cached_property
    def varprob(self) -> Float32[Array, 'ndpost p']:
        """Posterior samples of the probability of choosing each predictor for a decision rule."""
        return varprob(self._mcmc_state.forest.max_split, self._main_trace)

    @cached_property
    def varprob_mean(self) -> Float32[Array, ' p']:
        """The marginal posterior probability of each predictor being chosen for a decision rule."""
        return self.varprob.mean(axis=0)

    def _process_w_test(
        self,
        x_test: Real[ArrayLike, 'p m'] | DataFrame | str,
        kind: PredictKind,
        w: Float[ArrayLike, ' m'] | Float[ArrayLike, 'k m'] | Series | DataFrame | None,
    ) -> Float32[Array, ' m'] | Float32[Array, 'k m'] | None:
        """Validate and resolve the error weights for prediction.

        Parameters
        ----------
        x_test
            The raw (not yet processed) test predictors, or ``'train'``.
        kind
            The prediction kind.
        w
            User-provided per-observation error scale, or `None`.

        Returns
        -------
        The resolved error scale as a float32 array, or `None` if weights
        are not applicable.

        Raises
        ------
        ValueError
            If `w` is specified when it should be `None`, or missing when
            required.

        """
        x_test_is_train = isinstance(x_test, str) and x_test == 'train'
        has_train_weights = self._w is not None
        is_binary = self._mcmc_state.binary_y is not None
        needs_weights = (
            kind is PredictKind.outcome_samples and not is_binary and has_train_weights
        )

        if not needs_weights:
            if w is not None:
                msg = (
                    '`w` must be `None` in this configuration'
                    " (it is used only with kind='outcome_samples',"
                    ' continuous regression fitted with weights)'
                )
                raise ValueError(msg)
            return None

        if x_test_is_train:
            if w is not None:
                msg = (
                    "`w` must be `None` when x_test='train'"
                    ' (training weights are used automatically)'
                )
                raise ValueError(msg)
            return self._w

        # new test data, model was fit with weights
        if w is None:
            msg = (
                '`w` is required because the model was fit with'
                ' weights and x_test is new data'
            )
            raise ValueError(msg)
        w_test = _process_response_input(w)
        if w_test.ndim != self._w.ndim:
            msg = (
                f'`w` shape mismatch with training weights: got '
                f'{w_test.shape=}, expected {self._w.ndim}D '
                f'(matching the training-weight shape).'
            )
            raise ValueError(msg)
        return w_test

    def _process_x_test(
        self,
        x_test: Real[ArrayLike, 'p m'] | DataFrame | str,
        w: Float32[Array, ' m'] | None,
    ) -> UInt[Array, 'p m']:
        """Convert x_test to binned format suitable for prediction."""
        if isinstance(x_test, str):
            if x_test != 'train':
                msg = (
                    f"x_test must be an array, a DataFrame, or 'train', got {x_test!r}"
                )
                raise ValueError(msg)
            return self._mcmc_state.X
        x_test, x_test_fmt = _process_predictor_input(x_test)
        if x_test_fmt != self._x_train_fmt:
            msg = f'Input format mismatch: {x_test_fmt=} != x_train_fmt={self._x_train_fmt!r}'
            raise ValueError(msg)
        if w is not None:
            _check_same_length(w, x_test)
        return self._binner.bin(x_test)

    def _check_trees(
        self, error: bool = False
    ) -> UInt[Array, 'num_chains n_save num_trees']:
        """Apply `bartz.grove.check_trace` to all the tree draws.

        Parameters
        ----------
        error
            If `True`, throw an error if any invalid trees are found.

        Returns
        -------
        An array where non-zero entries indicate invalid trees.

        Raises
        ------
        RuntimeError
            If `error` is `True` and any invalid trees are found.
        """
        out = check_trees(self._main_trace, self._mcmc_state.forest.max_split)
        if error:
            bad_count = jnp.count_nonzero(out).item()
            if bad_count > 0:
                msg = f'Found {bad_count} invalid trees in the MCMC trace.'
                raise RuntimeError(msg)
        return out

    def _tree_goes_bad(self) -> Bool[Array, 'num_chains n_save num_trees']:
        """Find iterations where a tree becomes invalid.

        Returns
        -------
        An array where ``(i, j, k)`` is `True` if tree `k` is invalid at
        iteration `j` in chain `i` but not at iteration ``j - 1``.
        """
        return tree_goes_bad(self._main_trace, self._mcmc_state.forest.max_split)

    def _check_replicated_trees(self) -> None:
        """Check that the trees are equal across data-sharded devices.

        If the data is sharded across devices, verify that the trees (which
        should be replicated) are identical on all shards.

        Raises
        ------
        RuntimeError
            If the trees differ across devices.
        """
        state = self._mcmc_state
        mesh = state.config.mesh
        if mesh is not None and 'data' in mesh.axis_names:
            replicated_forest = replace(state.forest, leaf_indices=None)
            equal = equal_shards(
                replicated_forest, 'data', in_specs=PartitionSpec(), mesh=mesh
            )
            equal_array = jnp.stack(tree.leaves(equal))
            all_equal = jnp.all(equal_array)
            if not all_equal.item():
                msg = 'The trees differ across data-sharded devices.'
                raise RuntimeError(msg)

    def _compare_resid(
        self, y: Float32[Array, ' n'] | Float32[Array, 'k n'] | None = None
    ) -> tuple[
        Float32[Array, '*num_chains n'] | Float32[Array, '*num_chains k n'],
        Float32[Array, '*num_chains n'] | Float32[Array, '*num_chains k n'],
    ]:
        """Re-compute residuals to compare them with the updated ones.

        Parameters
        ----------
        y
            The response variable. Required for continuous regression (since
            ``State`` does not store ``y`` in continuous mode). Ignored for
            binary regression (where ``State.z`` is used instead).

        Returns
        -------
        resid1
            The final state of the residuals updated during the MCMC.
        resid2
            The residuals computed from the final state of the trees.
        """
        state = self._mcmc_state
        if state.binary_indices is not None:
            assert y is not None, 'y is required for mixed regression'
        elif state.z is None:
            assert y is not None, 'y is required for continuous regression'
        y_arr = jnp.asarray(y) if y is not None else None
        return compare_resid(state, y_arr)

    def _depth_distr(self) -> Int32[Array, '*num_chains n_save d']:
        """Histogram of tree depths for each state of the trees.

        Returns
        -------
        A matrix where each row contains a histogram of tree depths.
        """
        return depth_distr(self._main_trace)

    def _points_per_node_distr(
        self, node_type: str
    ) -> Int32[Array, '*num_chains n_save n+1']:
        return points_per_node_distr_trace(
            self._mcmc_state.X, self._main_trace, node_type
        )

    def _points_per_decision_node_distr(self) -> Int32[Array, '*num_chains n_save n+1']:
        """Histogram of number of points belonging to parent-of-leaf nodes.

        Returns
        -------
        For each chain, a matrix where each row contains a histogram of number of points.
        """
        return self._points_per_node_distr('leaf-parent')

    def _points_per_leaf_distr(self) -> Int32[Array, '*num_chains n_save n+1']:
        """Histogram of number of points belonging to leaves.

        Returns
        -------
        A matrix where each row contains a histogram of number of points.
        """
        return self._points_per_node_distr('leaf')

    def _print_tree(
        self, i_chain: int, i_sample: int, i_tree: int, print_all: bool = False
    ) -> None:
        """Print a single tree in human-readable format.

        Parameters
        ----------
        i_chain
            The index of the MCMC chain.
        i_sample
            The index of the (post-burnin) sample in the chain.
        i_tree
            The index of the tree in the sample.
        print_all
            If `True`, also print the content of unused node slots.
        """
        trace = self._main_trace
        trees = TreesTrace.from_dataclass(trace)
        if trace.has_chains:
            trees_chain_axes = TreesTrace.from_dataclass(chain_vmap_axes(trace))
            # WORKAROUND(python<3.14): use operator.is_none
            trees = tree.map(
                chain_to_axis, trees, trees_chain_axes, is_leaf=lambda x: x is None
            )
        else:
            i_chain = ...
        trees = tree.map(lambda x: x[i_chain, i_sample, i_tree, :], trees)
        s = format_tree(trees, print_all=print_all)
        print(s)  # noqa: T201, this method is intended for debug


def _process_predictor_input(
    x: Real[ArrayLike, 'p n'] | DataFrame,
) -> tuple[Shaped[Array, 'p n'], Any]:
    if hasattr(x, 'columns'):
        fmt = dict(kind='dataframe', columns=x.columns)
        x = x.to_numpy().T
    else:
        fmt = dict(kind='array', num_covar=x.shape[0])
    x = jnp.asarray(x)
    assert x.ndim == 2
    return x, fmt


@overload
def _process_response_input(
    arr: Shaped[ArrayLike, ' n'] | Shaped[ArrayLike, 'k n'] | Series | DataFrame,
    /,
    *,
    keep: Literal[False] = False,
    dtype: DTypeLike = jnp.float32,
) -> Shaped[Array, ' n'] | Shaped[Array, 'k n']: ...


@overload
def _process_response_input(
    arr: Shaped[ArrayLike, ' n'] | Shaped[ArrayLike, 'k n'] | Series | DataFrame,
    /,
    *,
    keep: Literal[True],
    dtype: DTypeLike = jnp.float32,
) -> tuple[
    Shaped[Array, ' n'] | Shaped[Array, 'k n'],
    Shaped[Array, ' n'] | Shaped[Array, 'k n'],
]: ...


def _process_response_input(
    arr: Shaped[ArrayLike, ' n'] | Shaped[ArrayLike, 'k n'] | Series | DataFrame,
    /,
    *,
    keep: bool = False,
    dtype: DTypeLike = jnp.float32,
) -> (
    Shaped[Array, ' n']
    | Shaped[Array, 'k n']
    | tuple[
        Shaped[Array, ' n'] | Shaped[Array, 'k n'],
        Shaped[Array, ' n'] | Shaped[Array, 'k n'],
    ]
):
    if hasattr(arr, 'columns'):
        arr = arr.to_numpy().T
    elif hasattr(arr, 'to_numpy'):
        arr = arr.to_numpy()
    # in normal mode: one unconditional copy, safe to donate downstream.
    # in `keep` mode: convert without copying when possible to get the
    # keep array, then `jnp.copy` to make a separate disposable copy.
    arr = jnp.array(arr, dtype, copy=not keep)
    if arr.ndim < 1 or arr.ndim > 2:
        msg = f'response-like input must be 1D (n,) or 2D (k, n). Got {arr.ndim=}.'
        raise ValueError(msg)
    if keep:
        return jnp.copy(arr), arr
    return arr


def _check_same_length(x1: Array, x2: Array) -> None:
    get_length = lambda x: x.shape[-1]
    assert get_length(x1) == get_length(x2)


def _check_type_settings(
    y_train: Float32[Array, ' n'] | Float32[Array, 'k n'],
    outcome_type: OutcomeType | str | Sequence[OutcomeType | str],
    w: Float[Array, ' n'] | Float[Array, 'k n'] | None,
) -> tuple[OutcomeType | tuple[OutcomeType, ...], Bool[Array, ''] | Bool[Array, ' k']]:
    # standardize outcome_type to OutcomeType or tuple[OutcomeType, ...]
    if isinstance(outcome_type, Sequence) and not isinstance(outcome_type, str):
        outcome_type = tuple(OutcomeType(t) for t in outcome_type)
        num_types = len(outcome_type)
        if len(set(outcome_type)) == 1:
            outcome_type = outcome_type[0]
    else:
        num_types = None
        outcome_type = OutcomeType(outcome_type)

    # validation
    if num_types is not None and (y_train.ndim != 2 or num_types != y_train.shape[0]):
        msg = (
            f'Sequence outcome_type of length {num_types}'
            f' requires y_train.shape=({num_types}, n),'
            f' found {y_train.shape=}.'
        )
        raise ValueError(msg)
    if w is not None and outcome_type is not OutcomeType.continuous:
        msg = 'Weights are not supported when any outcome is binary.'
        raise ValueError(msg)
    if (
        w is not None
        and w.ndim == 2
        and (y_train.ndim != 2 or w.shape[0] != y_train.shape[0])
    ):
        msg = (
            f'2D w (vector per-component weights) requires y_train of '
            f'shape (k, n) with matching k; got {w.shape=}, '
            f'{y_train.shape=}.'
        )
        raise ValueError(msg)

    if isinstance(outcome_type, tuple):
        binary_mask = jnp.array([t is OutcomeType.binary for t in outcome_type])
    else:
        binary_mask = jnp.bool_(outcome_type is OutcomeType.binary)
    binary_mask = jnp.broadcast_to(binary_mask, y_train.shape[:-1])

    return outcome_type, binary_mask


def _process_sparsity_settings(
    x_train: Real[Array, 'p n'],
    sparse: bool,
    theta: FloatLike | None,
    a: FloatLike,
    b: FloatLike,
    rho: FloatLike | None,
) -> (
    tuple[None, None, None, None]
    | tuple[FloatLike, None, None, None]
    | tuple[None, FloatLike, FloatLike, FloatLike]
):
    """Return (theta, a, b, rho)."""
    if not sparse:
        return None, None, None, None
    elif theta is not None:
        return theta, None, None, None
    else:
        if rho is None:
            p, _ = x_train.shape
            rho = float(p)
        return None, a, b, rho


def _process_offset_settings(
    y_train: Float32[Array, ' n'] | Float32[Array, 'k n'],
    binary_mask: Bool[Array, ''] | Bool[Array, ' k'],
    offset: FloatLike | Float[ArrayLike, ' k'] | None,
) -> Float32[Array, ''] | Float32[Array, ' k']:
    """Return offset."""
    if offset is not None:
        off = jnp.asarray(offset, jnp.float32)
        return jnp.broadcast_to(off, y_train.shape[:-1])
    if y_train.shape[-1] < 1:
        return jnp.zeros(y_train.shape[:-1])

    bound = 1 / (1 + y_train.shape[-1])
    binary_offset = ndtri(jnp.clip((y_train != 0).mean(-1), bound, 1 - bound))
    continuous_offset = y_train.mean(-1)
    return jnp.where(binary_mask, binary_offset, continuous_offset)


def _process_leaf_variance_settings(
    y_train: Float32[Array, ' n'] | Float32[Array, 'k n'],
    binary_mask: Bool[Array, ''] | Bool[Array, ' k'],
    k: FloatLike,
    num_trees: int,
    tau_num: FloatLike | None,
) -> Float32[Array, ''] | Float32[Array, 'k k']:
    """Return `leaf_prior_cov_inv`."""
    # determine `tau_num` if not specified
    if tau_num is None:
        if y_train.shape[-1] < 2:
            continuous_tau = jnp.ones(y_train.shape[:-1])
        else:
            continuous_tau = (y_train.max(-1) - y_train.min(-1)) / 2
        tau_num = jnp.where(binary_mask, 3.0, continuous_tau)

    # leaf prior standard deviation
    sigma_mu = tau_num / (k * math.sqrt(num_trees))

    # leaf prior precision matrix
    leaf_prior_cov_inv = jnp.reciprocal(jnp.square(sigma_mu))
    if y_train.ndim == 2:
        leaf_prior_cov_inv = jnp.diag(
            jnp.broadcast_to(leaf_prior_cov_inv, y_train.shape[:-1])
        )
    return leaf_prior_cov_inv


def _process_error_variance_settings(
    x_train: Shaped[Array, 'p n'],
    y_train: Float32[Array, ' n'] | Float32[Array, 'k n'],
    outcome_type: OutcomeType | tuple[OutcomeType, ...],
    binary_mask: Bool[Array, ''] | Bool[Array, ' k'],
    sigest: FloatLike | Float[Array, ' k'] | Literal['auto', 'ols-or-variance', 'cg'],
    sigdf: FloatLike,
    sigquant: FloatLike,
    lambda_: FloatLike | Float[Array, ' k'] | None,
) -> tuple[
    Float32[Array, ''] | None,
    Float32[Array, ''] | Float32[Array, 'k k'] | None,
    Float32[Array, ''] | Float32[Array, ' k'] | None,
]:
    """Return (error_cov_df, error_cov_scale, sigest)."""
    if outcome_type is OutcomeType.binary:
        if not isinstance(sigest, str) or lambda_ is not None:
            msg = 'Do not set `sigest` or `lambda_` for binary regression, they are ignored'
            raise ValueError(msg)
        return None, None, None

    if lambda_ is None:
        # estimate sigest²
        sigest2 = _estimate_sigest2(x_train, y_train, sigest, binary_mask)
        sigest = jnp.sqrt(sigest2)

        # lambda_ from sigest²
        alpha = sigdf / 2
        invchi2 = invgamma.ppf(sigquant, alpha) / 2
        invchi2rid = invchi2 * sigdf
        lambda_ = sigest2 / invchi2rid

    elif not isinstance(sigest, str):
        msg = "Do not set `sigest` if `lambda_` is specified, it's ignored"
        raise ValueError(msg)

    else:
        lambda_ = jnp.where(binary_mask, 0.0, lambda_)
        sigest = None

    # params written in multivariate form
    if y_train.ndim == 2:
        k = y_train.shape[0]
        lambda_ = jnp.broadcast_to(lambda_, (k,))
        error_cov_df = jnp.asarray(sigdf) + k - 1
        error_cov_scale = jnp.diag(sigdf * lambda_)
    else:
        error_cov_df = jnp.asarray(sigdf)
        error_cov_scale = jnp.asarray(sigdf * lambda_)

    return error_cov_df, error_cov_scale, sigest


def _estimate_sigest2(
    x_train: Shaped[Array, 'p n'],
    y_train: Float32[Array, '*k n'],
    sigest: FloatLike | Float[Array, ' k'] | Literal['auto', 'ols-or-variance', 'cg'],
    binary_mask: Bool[Array, '*k'],
) -> Float32[Array, '*k']:
    if not isinstance(sigest, str):
        sigest2 = jnp.square(jnp.asarray(sigest, dtype=jnp.float32))
        sigest2 = jnp.broadcast_to(sigest2, y_train.shape[:-1])
    elif sigest == 'ols-or-variance':
        sigest2 = _sigest2_ols_or_variance(x_train, y_train)
    elif sigest == 'cg':
        sigest2 = _sigest2_cg(x_train, y_train)
    elif sigest == 'auto':
        sigest2 = _sigest2_auto(x_train, y_train)
    else:
        msg = f'unrecognized value {sigest=}'
        raise ValueError(msg)
    return jnp.where(binary_mask, 0.0, sigest2)


def _sigest2_ols_or_variance(
    x_train: Shaped[Array, 'p n'], y_train: Float32[Array, '*k n']
) -> Float32[Array, '*k']:
    """Implement the case `sigest='ols-or-variance'`."""
    p, n = x_train.shape
    if n < 2:
        *k, _ = y_train.shape
        return jnp.ones(k)
    elif n <= p:
        return jnp.var(y_train, axis=-1)
    else:
        return _sigma2_from_ols(x_train, y_train)


def _sigest2_cg(
    x_train: Shaped[Array, 'p n'], y_train: Float32[Array, '*k n']
) -> Float32[Array, '*k']:
    """Implement the case `sigest='cg'`."""
    p, n = x_train.shape
    maxiter = max(1, min(n, p, CG_MAXITER))
    return _sigma2_from_cg(x_train, y_train, maxiter)


def _sigest2_auto(
    x_train: Shaped[Array, 'p n'], y_train: Float32[Array, '*k n']
) -> Float32[Array, ' *k']:
    """Implement the case `sigest='auto'`."""
    p, n = x_train.shape
    threshold = 10_000 * 100**2
    if n * p * p > threshold and min(n, p) > CG_MAXITER:
        return _sigest2_cg(x_train, y_train)
    else:
        return _sigest2_ols_or_variance(x_train, y_train)


def _setup_mcmc(
    x_train: Real[Array, 'p n'],
    y_train: Float32[Array, ' n'] | Float32[Array, 'k n'],
    outcome_type: OutcomeType | tuple[OutcomeType, ...],
    offset: Float32[Array, ''] | Float32[Array, ' k'],
    w: Float[Array, ' n'] | Float[Array, 'k n'] | None,
    missing: Bool[Array, ' n'] | Bool[Array, 'k n'] | None,
    max_split: UInt[Array, ' p'],
    leaf_prior_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'],
    error_cov_df: FloatLike | None,
    error_cov_scale: FloatLike | Float32[Array, 'k k'] | None,
    power: FloatLike,
    base: FloatLike,
    maxdepth: int,
    num_trees: int,
    init_kw: Mapping[str, Any],
    rm_const: bool,
    theta: FloatLike | None,
    a: FloatLike | None,
    b: FloatLike | None,
    rho: FloatLike | None,
    varprob: Float[ArrayLike, ' p'] | None,
    num_chains: int | None,
    num_chain_devices: int | None | Literal['auto'],
    num_data_devices: int | None,
    devices: Literal['cpu', 'gpu'] | Device | Sequence[Device] | None,
    sparse: bool,
    n_burn: int,
) -> mcmcstep.State:
    p_nonterminal = make_p_nonterminal(maxdepth, base, power)

    # process device settings
    device_kw, device = process_device_settings(
        y_train, num_chains, num_chain_devices, num_data_devices, devices
    )

    kw: dict = dict(
        X=x_train,
        y=y_train,
        outcome_type=outcome_type,
        offset=offset,
        error_scale=w,
        missing=missing,
        max_split=max_split,
        num_trees=num_trees,
        p_nonterminal=p_nonterminal,
        leaf_prior_cov_inv=leaf_prior_cov_inv,
        error_cov_df=error_cov_df,
        error_cov_scale=error_cov_scale,
        min_points_per_decision_node=10,
        log_s=process_varprob(varprob, max_split),
        theta=theta,
        a=a,
        b=b,
        rho=rho,
        sparse_on_at=n_burn // 2 if sparse else None,
        **device_kw,
    )

    if rm_const:
        n_empty = jnp.sum(max_split == 0).item()
        kw.update(filter_splitless_vars=n_empty)

    kw.update(init_kw)

    state = mcmcstep.init(**kw)

    # put state on device if requested explicitly by the user
    if device is not None:
        state = device_put(state, device, donate=True)

    return state


def _run_mcmc(
    mcmc_state: mcmcstep.State,
    n_save: int,
    n_burn: int,
    n_skip: int,
    printevery: int | None,
    key: Key[Array, ''],
    run_mcmc_kw: Mapping,
) -> RunMCMCResult:
    # prepare arguments
    kw: dict = dict(n_burn=n_burn, n_skip=n_skip, inner_loop_length=printevery)
    kw.update(
        make_default_callback(
            mcmc_state,
            dot_every=None if printevery is None or printevery == 1 else 1,
            report_every=printevery,
        )
    )
    kw.update(run_mcmc_kw)

    return run_mcmc(key, mcmc_state, n_save, **kw)


@partial(jit, static_argnames='p')
# this is jitted such that lax.collapse below does not create a copy
def varcount(p: int, trace: MainTrace) -> Int32[Array, 'ndpost p']:
    """Histogram of predictor usage for decision rules in the trees, squashing chains."""
    varcount: Int32[Array, '*chains samples p']
    varcount = compute_varcount(p, trace, out_chain_axis=0)
    return lax.collapse(varcount, 0, -1)


@partial(jit, static_argnames='mean')
def get_error_sdev(
    trace: MainTrace,
    binary_mask: Bool[Array, ''] | Bool[Array, ' k'],
    *,
    mean: bool = False,
) -> (
    Float32[Array, ' ndpost']
    | Float32[Array, 'ndpost k']
    | Float32[Array, '']
    | Float32[Array, ' k']
):
    """Error standard deviation, post-burnin, chains concatenated."""
    prec = trace.error_cov_inv
    if trace.has_chains:
        # shape (chains, samples) or (chains, samples, k, k), concatenate chains
        prec = chain_to_axis(prec, chain_vmap_axes(trace).error_cov_inv)
        prec = lax.collapse(prec, 0, 2)
    is_uv = prec.ndim == 1
    if is_uv:
        # univariate case, reshape to 1x1 matrix
        prec = prec[..., None, None]

    # invert precision to covariance, then take diagonal variance
    cov = _inv_via_chol_with_gersh(prec)
    var = jnp.diagonal(cov, axis1=-2, axis2=-1)
    if mean:
        var = var.mean(0)
    sdev = jnp.sqrt(var)
    if is_uv:
        sdev = sdev.squeeze(-1)
    return jnp.where(binary_mask, jnp.nan, sdev)


@partial(jit, static_argnames='only_continuous')
def get_latent_prec(
    burnin_trace: BurninTrace,
    main_trace: MainTrace,
    binary_indices: Int32[Array, ' kb'] | None,
    *,
    only_continuous: bool = False,
) -> (
    Float32[Array, ' n_burn+n_save']
    | Float32[Array, 'n_burn+n_save k k']
    | Float32[Array, 'num_chains n_burn+n_save']
    | Float32[Array, 'num_chains n_burn+n_save k k']
):
    """Latent error precision trace, burn-in + main concatenated."""
    burnin = burnin_trace.error_cov_inv
    main = main_trace.error_cov_inv
    sample_axis = trace_sample_axes(main_trace).error_cov_inv
    prec = jnp.concatenate([burnin, main], axis=sample_axis)
    prec = chain_to_axis(prec, chain_vmap_axes(main_trace).error_cov_inv)
    if only_continuous and binary_indices is not None:
        *_, k, _ = prec.shape
        kc = k - binary_indices.size
        mask = jnp.ones(k, dtype=bool).at[binary_indices].set(False)
        (cont_indices,) = jnp.nonzero(mask, size=kc)
        prec = prec[..., cont_indices[:, None], cont_indices[None, :]]
    return prec


@jit
def varprob(
    max_split: UInt[Array, ' p'], trace: MainTrace
) -> Float32[Array, 'ndpost p']:
    """Posterior samples of predictor selection probability, chains concatenated."""
    p = max_split.size
    varprob = trace.varprob
    if varprob is None:
        ndpost = trace.grow_prop_count.size
        peff = jnp.count_nonzero(max_split)
        out = jnp.where(max_split, 1 / peff, 0)
        return jnp.broadcast_to(out, (ndpost, p))
    varprob = chain_to_axis(varprob, chain_vmap_axes(trace).varprob)
    return varprob.reshape(-1, p)


@jit
def check_trees(
    trace: MainTrace, max_split: UInt[Array, ' p']
) -> UInt[Array, 'num_chains n_save num_trees']:
    """Apply `bartz.grove.check_trace` to all the tree draws."""
    trees = TreesTrace.from_dataclass(trace)
    if trace.has_chains:
        trees_chain_axes = TreesTrace.from_dataclass(chain_vmap_axes(trace))
        # WORKAROUND(python<3.14): use operator.is_none
        trees = tree.map(
            chain_to_axis, trees, trees_chain_axes, is_leaf=lambda x: x is None
        )
    out: UInt[Array, '*chains samples num_trees']
    out = check_trace(trees, max_split)
    if out.ndim < 3:
        out = out[None, :, :]
    return out


@jit
def tree_goes_bad(
    trace: MainTrace, max_split: UInt[Array, ' p']
) -> Bool[Array, 'num_chains n_save num_trees']:
    """Find iterations where a tree becomes invalid."""
    bad = check_trees(trace, max_split).astype(bool)
    bad_before = jnp.pad(bad[:, :-1, :], [(0, 0), (1, 0), (0, 0)])
    return bad & ~bad_before


@jit
def compare_resid(
    state: mcmcstep.State, y: Float32[Array, ' n'] | Float32[Array, 'k n'] | None
) -> tuple[
    Float32[Array, '*num_chains n'] | Float32[Array, '*num_chains k n'],
    Float32[Array, '*num_chains n'] | Float32[Array, '*num_chains k n'],
]:
    """Re-compute residuals to compare them with the updated ones."""
    chain_axes = chain_vmap_axes(state)
    resid1 = chain_to_axis(state.resid, chain_axes.resid)
    z = chain_to_axis(state.z, chain_axes.z) if state.z is not None else None

    forests = TreesTrace.from_dataclass(state.forest)
    if state.has_chains:
        forest_chain_axes = TreesTrace.from_dataclass(chain_axes.forest)
        # WORKAROUND(python<3.14): use operator.is_none
        forests = tree.map(
            chain_to_axis, forests, forest_chain_axes, is_leaf=lambda x: x is None
        )
    trees = evaluate_forest(state.X, forests, sum_batch_axis=-1)

    if state.binary_indices is not None:
        # mixed binary-continuous: z has only binary rows, y has all rows
        ref = jnp.broadcast_to(y, resid1.shape)
        ref = ref.at[..., state.binary_indices, :].set(z)
    elif z is not None:
        ref = z
    else:
        ref = y
    resid2 = ref - (trees + state.offset[..., None])

    return resid1, resid2


@jit
def depth_distr(trace: MainTrace) -> Int32[Array, '*num_chains n_save d']:
    """Histogram of tree depths for each state of the trees."""
    split_tree = chain_to_axis(trace.split_tree, chain_vmap_axes(trace).split_tree)
    out: Int32[Array, '*chains samples d']
    out = forest_depth_distr(split_tree)
    if out.ndim < 3:
        out = out[None, :, :]
    return out


@partial(jit, static_argnames='node_type')
def points_per_node_distr_trace(
    X: UInt[Array, 'p n'], trace: MainTrace, node_type: str
) -> Int32[Array, '*num_chains n_save n+1']:
    """Histogram of number of points per node, for every tree draw in the trace."""
    chain_axes = chain_vmap_axes(trace)
    var_tree = chain_to_axis(trace.var_tree, chain_axes.var_tree)
    split_tree = chain_to_axis(trace.split_tree, chain_axes.split_tree)
    out: Int32[Array, '*chains samples n+1']
    out = points_per_node_distr(X, var_tree, split_tree, node_type, sum_batch_axis=-1)
    if out.ndim < 3:
        out = out[None, :, :]
    return out


class DeviceKwArgs(TypedDict):
    num_chains: int | None
    mesh: Mesh | None
    target_platform: Literal['cpu', 'gpu'] | None


def process_device_settings(
    y_train: Array,
    num_chains: int | None,
    num_chain_devices: int | None | Literal['auto'],
    num_data_devices: int | None,
    devices: Literal['cpu', 'gpu'] | Device | Sequence[Device] | None,
) -> tuple[DeviceKwArgs, Device | None]:
    """Return the arguments for `mcmcstep.init` related to devices, and an optional device where to put the state."""
    platform, device, devices = _determine_devices(y_train, devices)
    num_chain_devices = _determine_num_chain_devices(
        platform, num_chains, num_chain_devices
    )
    mesh, device = _determine_mesh(num_chain_devices, num_data_devices, device, devices)

    # prepare arguments to `init`
    settings = DeviceKwArgs(
        num_chains=num_chains,
        mesh=mesh,
        target_platform=None
        if mesh is not None or hasattr(y_train, 'platform')
        else platform,
        # here we don't take into account the case where the user has set both
        # batch sizes; since the user has to be playing with `init_kw` to do
        # that, we'll let `init` throw the error and the user set
        # `target_platform` themselves so they have a clearer idea how the
        # thing works (i.e.: init won't be happy to receive target_platform if
        # it's not needed because all device-dependent defaults are overridden)
    )

    return settings, device


def _determine_devices(
    y_train: Array, devices: Literal['cpu', 'gpu'] | Device | Sequence[Device] | None
) -> tuple[str, Device | None, tuple[Device, ...]]:
    """Determine the target platform and set of devices for the MCMC, and possibly a single target device."""
    if isinstance(devices, str):
        platform = devices
        devices = jax.devices(platform)
        return platform, devices[0], devices
    elif devices is not None:
        if not hasattr(devices, '__len__'):
            devices = (devices,)
        device = devices[0]
        return device.platform, device, devices
    elif hasattr(y_train, 'platform'):
        # set device=None because if the devices were not specified explicitly
        # we may be in the case where computation will follow data placement,
        # do not disturb jax as the user may be playing with vmap, jit, reshard...
        platform = y_train.platform()
        return platform, None, jax.devices(platform)
    else:
        msg = 'not possible to infer device from `y_train`, please set `devices`'
        raise ValueError(msg)


def _largest_divisor_at_most(n: int, cap: int) -> int:
    """Return the largest divisor of `n` in [1, cap]."""
    for d in range(cap, 0, -1):
        if n % d == 0:
            return d
    return 1  # unreachable: 1 always divides n


def _determine_num_chain_devices(
    platform: str,
    num_chains: int | None,
    num_chain_devices: int | None | Literal['auto'],
) -> int | None:
    """Decide the value of `num_chain_devices` when it's set to 'auto'."""
    if num_chain_devices != 'auto':
        return num_chain_devices
    elif num_chains is None or num_chains == 1 or platform != 'cpu':
        return None
    else:
        num_cores = cpu_count()
        assert num_cores is not None, 'could not determine number of cpu cores'
        num_shards = _largest_divisor_at_most(num_chains, num_cores)

        if num_shards > 1:
            num_jax_cpus = device_count('cpu')
            if num_jax_cpus < num_shards:
                new_num_shards = _largest_divisor_at_most(num_chains, num_jax_cpus)
                msg = (
                    f'`Bart` would like to shard {num_chains} chains across '
                    f'{num_shards} virtual jax cpu devices, but jax is set up '
                    f'with only {num_jax_cpus} cpu devices, so it will use '
                    f'{new_num_shards} devices instead. To enable '
                    'parallelization, please increase the limit with '
                    '`jax.config.update("jax_num_cpu_devices", <num_devices>)`.'
                )
                warn(msg)
                num_shards = new_num_shards

        return num_shards if num_shards > 1 else None


def _determine_mesh(
    num_chain_devices: int | None,
    num_data_devices: int | None,
    device: Device | None,
    devices: Sequence[Device],
) -> tuple[Mesh | None, Device | None]:
    """Create a jax device mesh for `mcmcstep.init()`."""
    if num_chain_devices is None and num_data_devices is None:
        return None, device
    else:
        mesh = dict()
        if num_chain_devices is not None:
            mesh.update(chains=num_chain_devices)
        if num_data_devices is not None:
            mesh.update(data=num_data_devices)
        mesh = make_mesh(
            axis_shapes=tuple(mesh.values()),
            axis_names=tuple(mesh),
            axis_types=(AxisType.Auto,) * len(mesh),
            devices=devices,
        )
        return mesh, None
        # set device=None because `mcmcstep.init` will `device_put` with the
        # mesh already, we don't want to undo its work


def process_varprob(
    varprob: Float[ArrayLike, ' p'] | None, max_split: UInt[Array, ' p']
) -> Float32[Array, ' p'] | None:
    """Convert varprob to log_s."""
    if varprob is None:
        return None
    varprob = jnp.asarray(varprob)
    assert varprob.shape == max_split.shape, 'varprob must have shape (p,)'
    varprob = error_if(varprob, varprob <= 0, 'varprob must be > 0')
    return jnp.log(varprob)


def predict_latent(
    x: UInt[Array, 'p m'], trace: MainTrace
) -> Float32[Array, 'ndpost m'] | Float32[Array, 'ndpost k m']:
    """Evaluate trees on already quantized `x`, and squash chains."""
    return evaluate_trace(x, trace, flatten_chains=True)


@partial(jit, static_argnums=(5, 6))
def predict(
    key: Key[Array, ''] | None,
    trace: MainTrace,
    x_test: UInt[Array, 'p m'],
    w: Float[Array, ' m'] | Float[Array, 'k m'] | None,
    binary_indices: Int32[Array, ' kb'] | None,
    has_binary: bool,
    kind: PredictKind | str,
    /,
) -> (
    Float32[Array, ' m']
    | Float32[Array, 'k m']
    | Float32[Array, 'ndpost m']
    | Float32[Array, 'ndpost k m']
):
    """Implement `Bart.predict`."""
    # get latent i.e. bare sum-of-trees predictions
    latent = predict_latent(x_test, trace)
    if kind is PredictKind.latent_samples:
        return latent

    # sample posterior (uses latent directly, no probit squash needed)
    if kind is PredictKind.outcome_samples:
        assert key is not None
        return sample_outcome(key, trace, latent, w, binary_indices, has_binary)

    # squash predictions to (0, 1) if probit
    if binary_indices is not None:
        indexing = jnp.s_[..., binary_indices, :]
        mean_samples = latent.at[indexing].set(ndtr(latent[indexing]))
    elif has_binary:  # self._mcmc_state.binary_y is not None:
        mean_samples = ndtr(latent)
    else:
        mean_samples = latent

    # take mean or return samples
    if kind is PredictKind.mean:
        return mean_samples.mean(axis=0)
    return mean_samples


@partial(jit, static_argnums=(5,))
def sample_outcome(
    key: Key[Array, ''],
    trace: MainTrace,
    latent: Float32[Array, 'ndpost m'] | Float32[Array, 'ndpost k m'],
    w: Float32[Array, ' m'] | Float32[Array, 'k m'] | None,
    binary_indices: Int32[Array, ' kb'] | None,
    has_binary: bool,
    /,
) -> Float32[Array, 'ndpost m'] | Float32[Array, 'ndpost k m']:
    """Sample from the posterior predictive distribution."""
    # move error_cov_inv chain axis to 0
    prec = chain_to_axis(trace.error_cov_inv, chain_vmap_axes(trace).error_cov_inv)

    if latent.ndim > 2:  # multivariate case
        error_cov_inv = lax.collapse(prec, 0, -2)

        # Cholesky of precision: error_cov_inv = L @ L^T
        L = chol_with_gersh(error_cov_inv)  # (ndpost, k, k)

        # Sample z ~ N(0, I) and solve L^T @ error = z
        # so error = L^{-T} z ~ N(0, L^{-T} L^{-1}) = N(0, Sigma)
        z = random.normal(key, latent.shape)  # (ndpost, k, m)
        error = solve_triangular(L, z, trans='T', lower=True)  # (ndpost, k, m)
        if w is not None:
            # w is (m,) or (k, m) so it always broadcasts right
            error *= w
    elif has_binary:
        # pure binary UV: probit has sigma = 1
        error = random.normal(key, latent.shape)
    else:  # univariate continuous
        sigma = jnp.sqrt(jnp.reciprocal(prec)).reshape(-1)
        error = sigma[..., None] * random.normal(key, latent.shape)
        if w is not None:
            error *= w[None, :]

    outcome = latent + error

    # convert binary outcomes via latent probit thresholding
    if binary_indices is not None:
        idx = jnp.s_[..., binary_indices, :]
        outcome = outcome.at[idx].set(jnp.where(outcome[idx] > 0, 1.0, 0.0))
    elif has_binary:
        outcome = jnp.where(outcome > 0, 1.0, 0.0)

    return outcome
