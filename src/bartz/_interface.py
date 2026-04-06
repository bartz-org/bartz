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
from enum import Enum
from functools import cached_property, partial
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypedDict

import jax
import jax.numpy as jnp
from equinox import Module, error_if, field
from jax import Device, device_put, jit, lax, make_mesh, random
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import ndtr
from jax.sharding import AxisType, Mesh
from jaxtyping import (
    Array,
    Bool,
    Float,
    Float32,
    Int32,
    Integer,
    Key,
    Real,
    Shaped,
    UInt,
)
from numpy import ndarray

from bartz import mcmcloop, mcmcstep, prepcovars
from bartz.jaxext import is_key
from bartz.jaxext.scipy.special import ndtri
from bartz.jaxext.scipy.stats import invgamma
from bartz.mcmcloop import RunMCMCResult, compute_varcount, evaluate_trace, run_mcmc
from bartz.mcmcstep import OutcomeType, make_p_nonterminal
from bartz.mcmcstep._state import chol_with_gersh, get_num_chains

FloatLike = float | Float[Any, '']


class PredictKind(Enum):
    """Kind of output of `Bart.predict`."""

    mean = 'mean'
    """The posterior mean of the conditional mean, shape ``(m,)`` (or
    ``(k, m)`` for multivariate regression)."""

    mean_samples = 'mean_samples'
    """Per-sample conditional mean, shape ``(ndpost, m)`` (or ``(ndpost,
    k, m)``). For binary regression, this is the probit-transformed
    sum-of-trees."""

    outcome_samples = 'outcome_samples'
    """Samples of the outcome variable, shape ``(ndpost, m)`` (or
    ``(ndpost, k, m)``). For binary regression, these are Bernoulli
    draws. For continuous regression, these are Gaussian draws with the
    posterior noise variance."""

    latent_samples = 'latent_samples'
    """Raw sum-of-trees values, shape ``(ndpost, m)`` (or ``(ndpost, k,
    m)``)."""


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
    xinfo
        A matrix with the cutpoins to use to bin each predictor. If not
        specified, it is generated automatically according to `usequants` and
        `numcut`.

        Each row shall contain a sorted list of cutpoints for a predictor. If
        there are less cutpoints than the number of columns in the matrix,
        fill the remaining cells with NaN.

        `xinfo` shall be a matrix even if `x_train` is a dataframe.
    usequants
        Whether to use predictors quantiles instead of a uniform grid to bin
        predictors. Ignored if `xinfo` is specified.
    rm_const
        How to treat predictors with no associated decision rules (i.e., there
        are no available cutpoints for that predictor). If `True` (default),
        they are ignored. If `False`, an error is raised if there are any.
    sigest
        An estimate of the residual standard deviation on `y_train`, used to set
        `lamda`. If not specified, it is estimated by linear regression (with
        intercept, and without taking into account `w`). If `y_train` has less
        than two elements, it is set to 1. If n <= p, it is set to the standard
        deviation of `y_train`. Ignored if `lamda` is specified. For
        multivariate regression, can be a scalar (broadcast to all components)
        or a `(k,)` vector of per-component estimates. For mixed outcome types,
        binary component values are ignored.
    sigdf
        The degrees of freedom of the scaled inverse-chisquared prior on the
        noise variance. For multivariate regression, the Inverse-Wishart
        degrees of freedom are set to `sigdf + k - 1`.
    sigquant
        The quantile of the prior on the noise variance that shall match
        `sigest` to set the scale of the prior. Ignored if `lamda` is specified.
    k
        The inverse scale of the prior standard deviation on the latent mean
        function, relative to half the observed range of `y_train`. If `y_train`
        has less than two elements, `k` is ignored and the scale is set to 1.
    power
    base
        Parameters of the prior on tree node generation. The probability that a
        node at depth `d` (0-based) is non-terminal is ``base / (1 + d) **
        power``.
    lamda
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
        specified by the user. Not supported for multivariate regression.
    num_trees
        The number of trees used to represent the latent mean function.
    numcut
        If `usequants` is `False`: the exact number of cutpoints used to bin the
        predictors, ranging between the minimum and maximum observed values
        (excluded).

        If `usequants` is `True`: the maximum number of cutpoints to use for
        binning the predictors. Each predictor is binned such that its
        distribution in `x_train` is approximately uniform across bins. The
        number of bins is at most the number of unique values appearing in
        `x_train`, or ``numcut + 1``.

        Before running the algorithm, the predictors are compressed to the
        smallest integer type that fits the bin indices, so `numcut` is best set
        to the maximum value of an unsigned integer type, like 255.

        Ignored if `xinfo` is specified.
    ndpost
        The number of MCMC samples to save, after burn-in. `ndpost` is the
        total number of samples across all chains. `ndpost` is rounded up to the
        first multiple of `mc_cores`.
    nskip
        The number of initial MCMC samples to discard as burn-in. This number
        of samples is discarded from each chain.
    keepevery
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
        `num_chains`. Each device will run a fraction of the chains.
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

    _main_trace: mcmcloop.MainTrace
    _burnin_trace: mcmcloop.BurninTrace
    _mcmc_state: mcmcstep.State
    _splits: Real[Array, 'p max_num_splits']
    _x_train_fmt: Any = field(static=True)

    offset: Float32[Array, ''] | Float32[Array, ' k']
    """The prior mean of the latent mean function."""

    sigest: Float32[Array, ''] | Float32[Array, ' k'] | None = None
    """The estimated standard deviation of the error used to set `lamda`."""

    def __init__(
        self,
        x_train: Real[Array, 'p n'] | DataFrame,
        y_train: Float32[Array, ' n'] | Float32[Array, 'k n'] | Series,
        *,
        outcome_type: OutcomeType | str | Sequence[OutcomeType | str] = 'continuous',
        sparse: bool = False,
        theta: FloatLike | None = None,
        a: FloatLike = 0.5,
        b: FloatLike = 1.0,
        rho: FloatLike | None = None,
        varprob: Float[Array, ' p'] | None = None,
        xinfo: Float[Array, 'p n'] | None = None,
        usequants: bool = False,
        rm_const: bool = True,
        sigest: FloatLike | Float[Array, ' k'] | None = None,
        sigdf: FloatLike = 3.0,
        sigquant: FloatLike = 0.9,
        k: FloatLike = 2.0,
        power: FloatLike = 2.0,
        base: FloatLike = 0.95,
        lamda: FloatLike | Float[Array, ' k'] | None = None,
        tau_num: FloatLike | None = None,
        offset: FloatLike | Float[Array, ' k'] | None = None,
        w: Float[Array, ' n'] | Series | None = None,
        num_trees: int = 200,
        numcut: int = 255,
        ndpost: int = 1000,
        nskip: int = 1000,
        keepevery: int = 1,
        printevery: int | None = 100,
        num_chains: int | None = 4,
        num_chain_devices: int | None = None,
        num_data_devices: int | None = None,
        devices: Device | Sequence[Device] | None = None,
        seed: int | Key[Array, ''] = 0,
        maxdepth: int = 6,
        init_kw: Mapping = MappingProxyType({}),
        run_mcmc_kw: Mapping = MappingProxyType({}),
    ) -> None:
        # check data and put it in the right format
        x_train, x_train_fmt = self._process_predictor_input(x_train)
        y_train = self._process_response_input(y_train)
        self._check_same_length(x_train, y_train)

        if w is not None:
            w = self._process_response_input(w)
            self._check_same_length(x_train, w)

        # check data types are correct for continuous/binary/multivariate regression
        outcome_type, binary_mask = self._check_type_settings(y_train, outcome_type, w)

        # process sparsity settings
        theta, a, b, rho = self._process_sparsity_settings(
            x_train, sparse, theta, a, b, rho
        )

        # process "standardization" settings
        offset = self._process_offset_settings(y_train, binary_mask, offset)
        leaf_prior_cov_inv = self._process_leaf_variance_settings(
            y_train, binary_mask, k, num_trees, tau_num
        )
        error_cov_df, error_cov_scale, sigest = self._process_error_variance_settings(
            x_train, y_train, outcome_type, binary_mask, sigest, sigdf, sigquant, lamda
        )

        # determine splits
        splits, max_split = self._determine_splits(x_train, usequants, numcut, xinfo)
        x_train = self._bin_predictors(x_train, splits)

        # setup and run mcmc
        initial_state = self._setup_mcmc(
            x_train,
            y_train,
            outcome_type,
            offset,
            w,
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
            nskip,
        )
        result = self._run_mcmc(
            initial_state, ndpost, nskip, keepevery, printevery, seed, run_mcmc_kw
        )

        # set public attributes
        # set offset from the state because of buffer donation
        self.offset = result.final_state.offset
        self.sigest = sigest

        # set private attributes
        self._main_trace = result.main_trace
        self._burnin_trace = result.burnin_trace
        self._mcmc_state = result.final_state
        self._splits = splits
        self._x_train_fmt = x_train_fmt

    @property
    def ndpost(self) -> int:
        """The total number of posterior samples after burn-in across all chains.

        May be larger than the initialization argument `ndpost` if it was not
        divisible by the number of chains.
        """
        return self._main_trace.grow_prop_count.size

    @property
    def num_trees(self) -> int:
        """Return the number of trees used in the model."""
        return self._mcmc_state.forest.split_tree.shape[-2]

    @cached_property
    def sigma(
        self,
    ) -> (
        Float32[Array, ' nskip+ndpost']
        | Float32[Array, 'nskip+ndpost/mc_cores mc_cores']
        | None
    ):
        """The standard deviation of the error, including burn-in samples.

        Returns `None` for binary regression or multivariate regression.
        """
        if self._mcmc_state.binary_y is not None:
            return None
        # not meaningful for MV (error_cov_inv is a matrix)
        if self._mcmc_state.offset.ndim == 1:
            return None
        return jnp.sqrt(
            jnp.reciprocal(
                jnp.concatenate(
                    [
                        self._burnin_trace.error_cov_inv.T,
                        self._main_trace.error_cov_inv.T,
                    ],
                    axis=0,
                    # error_cov_inv has shape (chains? samples) in the trace
                )
            )
        )

    @cached_property
    def sigma_(self) -> Float32[Array, 'ndpost'] | None:
        """The standard deviation of the error, only over the post-burnin samples and flattened.

        Returns `None` for binary regression or multivariate regression.
        """
        if self._mcmc_state.binary_y is not None:
            return None
        # not meaningful for MV (error_cov_inv is a matrix)
        if self._mcmc_state.offset.ndim == 1:
            return None
        return jnp.sqrt(jnp.reciprocal(self._main_trace.error_cov_inv)).reshape(-1)

    @cached_property
    def sigma_mean(self) -> Float32[Array, ''] | None:
        """The mean of `sigma`, only over the post-burnin samples."""
        if self.sigma_ is None:
            return None
        return self.sigma_.mean()

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
        max_split = self._mcmc_state.forest.max_split
        p = max_split.size
        varprob = self._main_trace.varprob
        if varprob is None:
            peff = jnp.count_nonzero(max_split)
            varprob = jnp.where(max_split, 1 / peff, 0)
            varprob = jnp.broadcast_to(varprob, (self.ndpost, p))
        else:
            varprob = varprob.reshape(-1, p)
        return varprob

    @cached_property
    def varprob_mean(self) -> Float32[Array, ' p']:
        """The marginal posterior probability of each predictor being chosen for a decision rule."""
        return self.varprob.mean(axis=0)

    def predict(
        self,
        x_test: Real[Array, 'p m'] | DataFrame | str,
        *,
        kind: PredictKind | str = 'mean',
        key: Key[Array, ''] | None = None,
        w: Float[Array, ' m'] | Series | None = None,
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
            new data.

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

        # get latent i.e. bare sum-of-trees predictions
        latent = self._predict(x_test)
        if kind is PredictKind.latent_samples:
            return latent

        # sample posterior (uses latent directly, no probit squash needed)
        binary_indices = self._mcmc_state.binary_indices
        if kind is PredictKind.outcome_samples:
            return self._sample_outcome(key, latent, binary_indices, w)

        # squash predictions to (0, 1) if probit
        if binary_indices is not None:
            indexing = jnp.s_[..., binary_indices, :]
            mean_samples = latent.at[indexing].set(ndtr(latent[indexing]))
        elif self._mcmc_state.binary_y is not None:
            mean_samples = ndtr(latent)
        else:
            mean_samples = latent

        # take mean or return samples
        if kind is PredictKind.mean:
            return mean_samples.mean(axis=0)
        return mean_samples

    def _sample_outcome(
        self,
        key: Key[Array, ''],
        latent: Float32[Array, 'ndpost m'] | Float32[Array, 'ndpost k m'],
        binary_indices: Int32[Array, ' kb'] | None,
        w: Float32[Array, ' m'] | None,
    ) -> Float32[Array, 'ndpost m'] | Float32[Array, 'ndpost k m']:
        """Sample from the posterior predictive distribution."""
        if latent.ndim > 2:  # multivariate case
            error_cov_inv = self._main_trace.error_cov_inv
            error_cov_inv = lax.collapse(error_cov_inv, 0, -2)

            # Cholesky of precision: error_cov_inv = L @ L^T
            L = chol_with_gersh(error_cov_inv)  # (ndpost, k, k)

            # Sample z ~ N(0, I) and solve L^T @ error = z
            # so error = L^{-T} z ~ N(0, L^{-T} L^{-1}) = N(0, Sigma)
            z = random.normal(key, latent.shape)  # (ndpost, k, m)
            error = solve_triangular(L, z, trans='T', lower=True)
        elif self._mcmc_state.binary_y is not None:
            # pure binary UV: probit has sigma = 1
            error = random.normal(key, latent.shape)
        else:  # univariate continuous
            error = self.sigma_[..., None] * random.normal(key, latent.shape)
            if w is not None:
                error *= w[None, :]

        outcome = latent + error

        # convert binary outcomes via latent probit thresholding
        if binary_indices is not None:
            idx = jnp.s_[..., binary_indices, :]
            outcome = outcome.at[idx].set(jnp.where(outcome[idx] > 0, 1.0, 0.0))
        elif self._mcmc_state.binary_y is not None:
            outcome = jnp.where(outcome > 0, 1.0, 0.0)

        return outcome

    def _process_w_test(
        self,
        x_test: Real[Array, 'p m'] | DataFrame | str,
        kind: PredictKind,
        w: Float[Array, ' m'] | Series | None,
    ) -> Float32[Array, ' m'] | None:
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
        has_train_weights = self._mcmc_state.prec_scale is not None
        is_binary = self._mcmc_state.binary_y is not None
        is_multivariate = self._mcmc_state.offset.ndim == 1
        needs_weights = (
            kind is PredictKind.outcome_samples
            and not is_binary
            and not is_multivariate
            and has_train_weights
        )

        if not needs_weights:
            if w is not None:
                msg = (
                    '`w` must be `None` in this configuration'
                    " (it is used only with kind='outcome_samples',"
                    ' univariate continuous regression fitted with'
                    ' weights)'
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
            return jnp.reciprocal(jnp.sqrt(self._mcmc_state.prec_scale))

        # new test data, model was fit with weights
        if w is None:
            msg = (
                '`w` is required because the model was fit with'
                ' weights and x_test is new data'
            )
            raise ValueError(msg)
        return self._process_response_input(w)

    def _process_x_test(
        self,
        x_test: Real[Array, 'p m'] | DataFrame | str,
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
        x_test, x_test_fmt = self._process_predictor_input(x_test)
        if x_test_fmt != self._x_train_fmt:
            msg = f'Input format mismatch: {x_test_fmt=} != x_train_fmt={self._x_train_fmt!r}'
            raise ValueError(msg)
        if w is not None:
            self._check_same_length(w, x_test)
        return self._bin_predictors(x_test, self._splits)

    @staticmethod
    def _process_predictor_input(
        x: Real[Any, 'p n'] | DataFrame,
    ) -> tuple[Shaped[Array, 'p n'], Any]:
        if hasattr(x, 'columns'):
            fmt = dict(kind='dataframe', columns=x.columns)
            x = x.to_numpy().T
        else:
            fmt = dict(kind='array', num_covar=x.shape[0])
        x = jnp.asarray(x)
        assert x.ndim == 2
        return x, fmt

    @staticmethod
    def _process_response_input(
        y: Shaped[Array, ' n'] | Shaped[Array, 'k n'] | Series,
    ) -> Float32[Array, ' n'] | Float32[Array, 'k n']:
        if hasattr(y, 'to_numpy'):
            y = y.to_numpy()
        y = jnp.asarray(y, jnp.float32)
        if y.ndim < 1 or y.ndim > 2:
            msg = f'y_train must be 1D (n,) or 2D (k, n). Got {y.ndim=}.'
            raise ValueError(msg)
        return y

    @staticmethod
    def _check_same_length(x1: Array, x2: Array) -> None:
        get_length = lambda x: x.shape[-1]
        assert get_length(x1) == get_length(x2)

    @classmethod
    def _process_error_variance_settings(
        cls,
        x_train: Shaped[Array, 'p n'],
        y_train: Float32[Array, ' n'] | Float32[Array, 'k n'],
        outcome_type: OutcomeType | tuple[OutcomeType, ...],
        binary_mask: Bool[Array, ''] | Bool[Array, ' k'],
        sigest: FloatLike | Float[Array, ' k'] | None,
        sigdf: FloatLike,
        sigquant: FloatLike,
        lamda: FloatLike | Float[Array, ' k'] | None,
    ) -> tuple[
        Float32[Array, ''] | None,
        Float32[Array, ''] | Float32[Array, 'k k'] | None,
        Float32[Array, ''] | Float32[Array, ' k'] | None,
    ]:
        """Return (error_cov_df, error_cov_scale, sigest)."""
        if outcome_type is OutcomeType.binary:
            if sigest is not None or lamda is not None:
                msg = 'Let `sigest=None` and `lamda=None` for binary regression'
                raise ValueError(msg)
            return None, None, None

        if lamda is None:
            # estimate sigest²
            sigest2 = cls._estimate_sigest2(x_train, y_train, sigest, binary_mask)
            sigest = jnp.sqrt(sigest2)

            # lamda from sigest²
            alpha = sigdf / 2
            invchi2 = invgamma.ppf(sigquant, alpha) / 2
            invchi2rid = invchi2 * sigdf
            lamda = sigest2 / invchi2rid

        elif sigest is not None:
            msg = 'Let `sigest=None` if `lamda` is specified'
            raise ValueError(msg)

        else:
            lamda = jnp.where(binary_mask, 0.0, lamda)

        # params written in multivariate form
        if y_train.ndim == 2:
            k = y_train.shape[0]
            lamda = jnp.broadcast_to(lamda, (k,))
            error_cov_df = jnp.asarray(sigdf) + k - 1
            error_cov_scale = jnp.diag(sigdf * lamda)
        else:
            error_cov_df = jnp.asarray(sigdf)
            error_cov_scale = jnp.asarray(sigdf * lamda)

        return error_cov_df, error_cov_scale, sigest

    @classmethod
    def _estimate_sigest2(
        cls,
        x_train: Shaped[Array, 'p n'],
        y_train: Float32[Array, '*k n'],
        sigest: float | Shaped[Array, '*k'] | None,
        binary_mask: Bool[Array, ''] | Bool[Array, ' k'],
    ) -> Float32[Array, '*k']:
        n = y_train.shape[-1]
        if sigest is not None:
            sigest2 = jnp.square(jnp.asarray(sigest, dtype=jnp.float32))
            sigest2 = jnp.broadcast_to(sigest2, y_train.shape[:-1])
        elif n < 2:
            sigest2 = jnp.ones(y_train.shape[:-1])
        elif n <= x_train.shape[0]:
            sigest2 = jnp.var(y_train, axis=-1)
        else:
            sigest2 = cls._linear_regression(x_train, y_train)
        return jnp.where(binary_mask, 0.0, sigest2)

    @staticmethod
    @jit
    def _linear_regression(
        x_train: Shaped[Array, 'p n'],
        y_train: Float32[Array, ' n'] | Float32[Array, 'k n'],
    ) -> Float32[Array, ''] | Float32[Array, ' k']:
        """Return the error variance estimated with OLS with intercept."""
        x_centered = x_train.T - x_train.mean(axis=1)
        y_centered = y_train.T - y_train.mean(axis=-1)
        # centering is equivalent to adding an intercept column
        _, chisq, rank, _ = jnp.linalg.lstsq(x_centered, y_centered)
        chisq = chisq.reshape(y_train.shape[:-1])
        dof = y_train.shape[-1] - rank
        return chisq / dof

    @staticmethod
    def _check_type_settings(
        y_train: Float32[Array, ' n'] | Float32[Array, 'k n'],
        outcome_type: OutcomeType | str | Sequence[OutcomeType | str],
        w: Float[Array, ' n'] | None,
    ) -> tuple[
        OutcomeType | tuple[OutcomeType, ...], Bool[Array, ''] | Bool[Array, ' k']
    ]:
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
        if num_types is not None and (
            y_train.ndim != 2 or num_types != y_train.shape[0]
        ):
            msg = (
                f'Sequence outcome_type of length {num_types}'
                f' requires y_train.shape=({num_types}, n),'
                f' found {y_train.shape=}.'
            )
            raise ValueError(msg)
        if w is not None and not (
            outcome_type is OutcomeType.continuous and y_train.ndim == 1
        ):
            msg = 'Weights are only supported for univariate continuous regression.'
            raise ValueError(msg)

        if isinstance(outcome_type, tuple):
            binary_mask = jnp.array([t is OutcomeType.binary for t in outcome_type])
        else:
            binary_mask = jnp.bool_(outcome_type is OutcomeType.binary)

        return outcome_type, binary_mask

    @staticmethod
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

    @staticmethod
    def _process_offset_settings(
        y_train: Float32[Array, ' n'] | Float32[Array, 'k n'],
        binary_mask: Bool[Array, ''] | Bool[Array, ' k'],
        offset: float | Float32[Any, ''] | Float32[Any, ' k'] | None,
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

    @staticmethod
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

    @staticmethod
    def _determine_splits(
        x_train: Real[Array, 'p n'],
        usequants: bool,
        numcut: int,
        xinfo: Float[Array, 'p n'] | None,
    ) -> tuple[Real[Array, 'p m'], UInt[Array, ' p']]:
        if xinfo is not None:
            if xinfo.ndim != 2 or xinfo.shape[0] != x_train.shape[0]:
                msg = f'{xinfo.shape=} different from expected ({x_train.shape[0]}, *)'
                raise ValueError(msg)
            return prepcovars.parse_xinfo(xinfo)
        elif usequants:
            return prepcovars.quantilized_splits_from_matrix(x_train, numcut + 1)
        else:
            return prepcovars.uniform_splits_from_matrix(x_train, numcut + 1)

    @staticmethod
    def _bin_predictors(
        x: Real[Array, 'p n'], splits: Real[Array, 'p max_num_splits']
    ) -> UInt[Array, 'p n']:
        return prepcovars.bin_predictors(x, splits)

    @staticmethod
    def _setup_mcmc(
        x_train: Real[Array, 'p n'],
        y_train: Float32[Array, ' n'] | Float32[Array, 'k n'],
        outcome_type: OutcomeType | tuple[OutcomeType, ...],
        offset: Float32[Array, ''] | Float32[Array, ' k'],
        w: Float[Array, ' n'] | None,
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
        varprob: Float[Any, ' p'] | None,
        num_chains: int | None,
        num_chain_devices: int | None,
        num_data_devices: int | None,
        devices: Device | Sequence[Device] | None,
        sparse: bool,
        nskip: int,
    ) -> mcmcstep.State:
        p_nonterminal = make_p_nonterminal(maxdepth, base, power)

        # process device settings
        device_kw, device = process_device_settings(
            y_train, num_chains, num_chain_devices, num_data_devices, devices
        )

        kw: dict = dict(
            X=x_train,
            # copy y_train because it's going to be donated in the mcmc loop
            y=jnp.array(y_train),
            outcome_type=outcome_type,
            offset=offset,
            error_scale=w,
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
            sparse_on_at=nskip // 2 if sparse else None,
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

    @classmethod
    def _run_mcmc(
        cls,
        mcmc_state: mcmcstep.State,
        ndpost: int,
        nskip: int,
        keepevery: int,
        printevery: int | None,
        seed: int | Integer[Array, ''] | Key[Array, ''],
        run_mcmc_kw: Mapping,
    ) -> RunMCMCResult:
        # prepare random generator seed
        if is_key(seed):
            key = jnp.copy(seed)
        else:
            key = jax.random.key(seed)

        # round up ndpost
        num_chains = get_num_chains(mcmc_state)
        if num_chains is None:
            num_chains = 1
        n_save = ndpost // num_chains + bool(ndpost % num_chains)

        # prepare arguments
        kw: dict = dict(n_burn=nskip, n_skip=keepevery, inner_loop_length=printevery)
        kw.update(
            mcmcloop.make_default_callback(
                mcmc_state,
                dot_every=None if printevery is None or printevery == 1 else 1,
                report_every=printevery,
            )
        )
        kw.update(run_mcmc_kw)

        return run_mcmc(key, mcmc_state, n_save, **kw)

    def _predict(
        self, x: UInt[Array, 'p m']
    ) -> Float32[Array, 'ndpost m'] | Float32[Array, 'ndpost k m']:
        """Evaluate trees on already quantized `x`."""
        return predict(x, self._main_trace)


@partial(jit, static_argnames='p')
# this is jitted such that lax.collapse below does not create a copy
def varcount(p: int, trace: mcmcloop.MainTrace) -> Int32[Array, 'ndpost p']:
    """Histogram of predictor usage for decision rules in the trees, squashing chains."""
    varcount: Int32[Array, '*chains samples p']
    varcount = compute_varcount(p, trace)
    return lax.collapse(varcount, 0, -1)


@jit
# this is jitted such that lax.collapse below does not create a copy
def predict(
    x: UInt[Array, 'p m'], trace: mcmcloop.MainTrace
) -> Float32[Array, 'ndpost m'] | Float32[Array, 'ndpost k m']:
    """Evaluate trees on already quantized `x`, and squash chains."""
    out = evaluate_trace(x, trace)
    # For MV, out has shape (*trace_shape, k, n); for UV, (*trace_shape, n).
    # We must collapse only the chain/sample dims, not k.
    # Detect MV: leaf_tree has an extra axis compared to split_tree.
    is_mv = trace.leaf_tree.ndim > trace.split_tree.ndim
    end = -2 if is_mv else -1
    return lax.collapse(out, 0, end)


class DeviceKwArgs(TypedDict):
    num_chains: int | None
    mesh: Mesh | None
    target_platform: Literal['cpu', 'gpu'] | None


def process_device_settings(
    y_train: Array,
    num_chains: int | None,
    num_chain_devices: int | None,
    num_data_devices: int | None,
    devices: Device | Sequence[Device] | None,
) -> tuple[DeviceKwArgs, Device | None]:
    """Return the arguments for `mcmcstep.init` related to devices, and an optional device where to put the state."""
    # determine devices
    if devices is not None:
        if not hasattr(devices, '__len__'):
            devices = (devices,)
        device = devices[0]
        platform = device.platform
    elif hasattr(y_train, 'platform'):
        platform = y_train.platform()
        device = None
        # set device=None because if the devices were not specified explicitly
        # we may be in the case where computation will follow data placement,
        # do not disturb jax as the user may be playing with vmap, jit, reshard...
        devices = jax.devices(platform)
    else:
        msg = 'not possible to infer device from `y_train`, please set `devices`'
        raise ValueError(msg)

    # create mesh
    if num_chain_devices is None and num_data_devices is None:
        mesh = None
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
        device = None
        # set device=None because `mcmcstep.init` will `device_put` with the
        # mesh already, we don't want to undo its work

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
        # thing works.
    )

    return settings, device


def process_varprob(
    varprob: Float[Any, ' p'] | None, max_split: UInt[Array, ' p']
) -> Float32[Array, ' p'] | None:
    """Convert varprob to log_s."""
    if varprob is None:
        return None
    varprob = jnp.asarray(varprob)
    assert varprob.shape == max_split.shape, 'varprob must have shape (p,)'
    varprob = error_if(varprob, varprob <= 0, 'varprob must be > 0')
    return jnp.log(varprob)
