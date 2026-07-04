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
import pickle
from collections.abc import Collection, Hashable, Mapping, Sequence
from dataclasses import replace
from enum import Enum
from functools import cached_property
from os import PathLike, cpu_count
from pathlib import Path

# WORKAROUND(python<3.15): use frozendict instead of MappingProxyType
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable
from warnings import warn

import jax
import jax.numpy as jnp
from equinox import Module, error_if, field, tree_at
from jax import Device, debug_nans, device_put, lax, make_mesh, random, tree
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import ndtr, ndtri
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Float32, Int32, Key, Real, Shaped, UInt
from numpy import ndarray

from bartz._jaxext import equal_shards, is_key, jit, project, split
from bartz.grove import (
    TreeHeaps,
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
    MainTraceWithTrainPred,
    RunMCMCResult,
    compute_varcount,
    evaluate_trace,
    make_print_callback,
    make_tqdm_callback,
    run_mcmc,
)
from bartz.mcmcstep import DiagWishart, OutcomeType, Wishart, make_p_nonterminal
from bartz.mcmcstep._axes import (
    chain_to_axis,
    chain_vmap_axes,
    chainful_axis,
    get_has_chains,
    trace_sample_axes,
)
from bartz.mcmcstep._state import (
    ArrayLike,
    FloatLike,
    State,
    _inv_via_chol_with_gersh,
    _leaf_partition_spec,
    chol_with_gersh,
    init,
)
from bartz.prepcovars import Binner, BinnerFactory, UniqueQuantileBinner


class PredictKind(Enum):
    """Kind of output of `Bart.predict`."""

    mean = 'mean'
    """The posterior mean of the conditional mean, shape ``(m,)`` (or
    ``(k, m)`` for multivariate regression)."""

    mean_samples = 'mean_samples'
    """Per-sample conditional mean, shape ``(num_chains * n_save, m)``
    (or ``(num_chains * n_save, k, m)``). For binary regression, this is
    the probit-transformed sum-of-trees, divided by the error scale `w`
    first if the model is heteroskedastic."""

    outcome_samples = 'outcome_samples'
    """Samples of the outcome variable, shape ``(num_chains * n_save,
    m)`` (or ``(num_chains * n_save, k, m)``). For binary regression,
    these are Bernoulli draws. For continuous regression, these are
    Gaussian draws with the posterior noise variance."""

    latent_samples = 'latent_samples'
    """Raw sum-of-trees values, shape ``(num_chains * n_save, m)`` (or
    ``(num_chains * n_save, k, m)``)."""


@runtime_checkable
class DataFrame(Protocol):
    """DataFrame duck-type for `Bart`."""

    @property
    def columns(self) -> Collection[str]:
        """The names of the columns."""
        ...

    def to_numpy(self) -> Shaped[ndarray, '*shape']:
        """Convert the dataframe to a 2d numpy array with columns on the second axis."""
        ...


@runtime_checkable
class Series(Protocol):
    """Series duck-type for `Bart`."""

    @property
    def name(self) -> Hashable:
        """The name of the series."""
        ...

    def to_numpy(self) -> Shaped[ndarray, '*shape']:
        """Convert the series to a 1d numpy array."""
        ...


class SparseConfig(Module):
    R"""
    Configuration of a sparsity-inducing variable selection prior.

    This is the prior of [1]_. Pass an instance to the `sparse` argument of
    `Bart` to activate variable selection on the predictors. The prior on the
    choice of predictor for each decision rule is

    .. math::
        (s_1, \ldots, s_p) \sim
        \operatorname{Dirichlet}(\mathtt{theta}/p, \ldots, \mathtt{theta}/p).

    If `theta` is not specified, it's a priori distributed according to

    .. math::
        \frac{\mathtt{theta}}{\mathtt{theta} + \mathtt{rho}} \sim
        \operatorname{Beta}(\mathtt{a}, \mathtt{b}).

    References
    ----------
    .. [1] Linero, Antonio R. (2018). “Bayesian Regression Trees for
       High-Dimensional Prediction and Variable Selection”. In: Journal of the
       American Statistical Association 113.522, pp. 626-636.
    """

    theta: FloatLike | None = None
    """Concentration of the Dirichlet prior. If not specified, it is sampled
    from a Beta prior parametrized by `a`, `b` and `rho`. If set directly, it
    should be in the ballpark of the predictor count p or lower."""

    a: FloatLike = 0.5
    """Shape parameter of the Beta prior on ``theta / (theta + rho)``."""

    b: FloatLike = 1.0
    """Shape parameter of the Beta prior on ``theta / (theta + rho)``."""

    rho: FloatLike | None = None
    """Scale of the Beta prior on `theta`. If not specified, set to the number
    of predictors p. Lower values prefer more sparsity."""

    augment: bool = field(static=True, default=True)
    """Whether to account exactly for the decision rules forbidden by the
    ancestors of each node when updating the variable selection probabilities,
    using data augmentation. On by default. Setting it to `False` ignores the
    forbidden rules, which is faster but only approximate. This matters most
    with few predictors with few cutpoints each, where the same predictor
    cannot be re-used down a branch."""

    enabled: bool = field(static=True, default=True)
    """Whether variable selection is active."""


class Bart(Module):
    R"""
    Nonparametric regression with Bayesian Additive Regression Trees (BART).

    Regress `y_train` on `x_train` with a latent mean function represented as
    a sum of decision trees [2]_. The inference is carried out by sampling the
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
        specifies mixed outcome types. Binary components in multivariate
        outcomes follow the multivariate probit BART formulation of [4]_.
    sparse
        A `SparseConfig` for the sparsity-inducing variable selection prior of
        [1]_. Disabled by default; pass a `SparseConfig` to enable it.
    varprob
        The probability distribution over the `p` predictors for choosing a
        predictor to split on in a decision node a priori. Must be > 0. It does
        not need to be normalized to sum to 1. If not specified, use a uniform
        distribution. If `sparse` is enabled, this is used as initial value for
        the MCMC.
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
    sigma_df
        The degrees of freedom of the prior on the error precision. For
        multivariate regression with `k` components, the Wishart degrees of
        freedom are set to ``sigma_df + k - 1``.
    sigma_scale
        Sets the scale of the prior on the error precision. If 'auto' (default),
        the prior is scaled so that the error precision equals
        ``diag(1 / var(y_train))`` in expectation, where with weights `error_scale`
        the variance is a precision-weighted one that estimates the unit-weight error
        variance. Otherwise, ``square(sigma_scale)`` is the prior harmonic mean of
        the error variance; for multivariate regression a scalar is broadcast to
        all components. For mixed outcome types, binary components are ignored.
    sigma_init
        The initial value of the error standard deviation in the MCMC. If 'auto'
        (default), the initial error precision is set to ``diag(1 / var(y_train))``,
        with the same precision-weighted variance as `sigma_scale` when weights are
        given. Otherwise, the initial precision is ``diag(1 / square(sigma_init))``;
        for multivariate regression a scalar is broadcast to all components. For
        mixed outcome types, binary components are ignored.
    k
        The inverse scale of the prior standard deviation on the latent mean
        function, relative to half the observed range of `y_train`. If `y_train`
        has less than two elements, `k` is ignored and the scale is set to 1.
    power
    base
        Parameters of the prior on tree node generation. The probability that a
        node at depth `d` (0-based) is non-terminal is ``base / (1 + d) **
        power``.
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
    error_scale
        Coefficients that rescale the error standard deviation on each
        datapoint. Not specifying `error_scale` is equivalent to setting it to 1
        for all datapoints. Shape ``(n,)`` applies the same scalar weight to every
        outcome component; for multivariate regression, ``(k, n)`` instead
        supplies a per-component weight per datapoint. Supported with binary
        (probit) outcomes, where the weight scales the latent error so the
        success probability is ``Phi(latent / error_scale)``, including the
        binary components of a mixed regression.
    missing
        Boolean mask with the same shape as `y_train`; `True` marks entries
        to be ignored by the MCMC. Values of `y_train` must be finite
        everywhere, including at masked positions. If 2-D, the error
        covariance must be diagonal.
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
        line. Set to `None` to disable progress reporting entirely (this ignores
        `pbar`). ^C interrupts the MCMC only every `printevery` iterations, so
        with reporting disabled it's impossible to kill the MCMC conveniently.
    pbar
        If `True`, show a `tqdm` progress bar instead of printing log lines. The
        bar advances every iteration and refreshes the acceptance statistics
        every `printevery` iterations. Ignored if `printevery` is `None`.
    num_chains
        The number of independent Markov chains to run.

        The difference between ``num_chains=None`` and ``num_chains=1`` is that
        in the latter case in the object attributes and some methods there will
        be an explicit chain axis of size 1.
    num_chain_devices
        The number of devices to spread the chains across. Must be a divisor of
        `num_chains`. Each device will run a fraction of the chains. If 'auto'
        (default) and running on cpu, the number of devices is picked
        automatically based on the number of cores and the number of available
        devices (all the virtual jax cpu devices, or the `devices` list if set).
    num_data_devices
        The number of devices to split datapoints across. Must be a divisor of
        `n`. This is useful only with very high `n`, about > 1000_000. `predict`
        parallelizes across the same devices, splitting the test points; the
        number of test points must be a multiple of `num_data_devices` as well.

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
    precompute_predict_train
        If `True`, compute the predictions at the training points during the
        MCMC. Off by default; makes ``predict('train', ...)`` faster at the cost
        of more memory.
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
    .. [4] Goh, Yong Chen, Wuu Kuang Soh, Andrew C. Parnell, and Keefe
       Murphy (2024). "Joint Models for Handling Non-Ignorable Missing
       Data using Bayesian Additive Regression Trees: Application to
       Leaf Photosynthetic Traits Data". arXiv:2412.14946 [stat.ME].

    """

    _main_trace: MainTrace
    _burnin_trace: BurninTrace
    _mcmc_state: State
    _binner: Binner
    _binary_mask: Bool[Array, ''] | Bool[Array, ' k']
    # WORKAROUND(jax<0.9.1): use `jax.tree.static` instead of `field(static=True)`
    _x_train_fmt: Any = field(static=True)
    _device: Device | None = field(static=True)

    def __init__(
        self,
        x_train: Real[ArrayLike, 'p n'] | DataFrame,
        y_train: Float32[ArrayLike, ' n']
        | Float32[ArrayLike, 'k n']
        | Series
        | DataFrame,
        *,
        outcome_type: OutcomeType | str | Sequence[OutcomeType | str] = 'continuous',
        sparse: SparseConfig = SparseConfig(enabled=False),
        varprob: Float[ArrayLike, ' p'] | None = None,
        binner: BinnerFactory = UniqueQuantileBinner,
        rm_const: bool = True,
        sigma_df: FloatLike = 3.0,
        sigma_scale: FloatLike | Float[ArrayLike, ' k'] | Literal['auto'] = 'auto',
        sigma_init: FloatLike | Float[ArrayLike, ' k'] | Literal['auto'] = 'auto',
        k: FloatLike = 2.0,
        power: FloatLike = 2.0,
        base: FloatLike = 0.95,
        tau_num: FloatLike | None = None,
        offset: FloatLike | Float[ArrayLike, ' k'] | None = None,
        error_scale: Float[ArrayLike, ' n']
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
        pbar: bool = True,
        num_chains: int | None = 4,
        num_chain_devices: int | None | Literal['auto'] = 'auto',
        num_data_devices: int | None = None,
        devices: Literal['cpu', 'gpu'] | Device | Sequence[Device] | None = None,
        seed: int | Key[Array, ''] = 0,
        maxdepth: int = 6,
        precompute_predict_train: bool = False,
        init_kw: Mapping = MappingProxyType({}),
        run_mcmc_kw: Mapping = MappingProxyType({}),
    ) -> None:
        # check data and put it in the right format
        x_train, x_train_fmt = _process_predictor_input(x_train)
        y_train = _process_response_input(y_train)
        _check_same_length(x_train, y_train)

        if error_scale is not None:
            # `error_scale` is donated downstream as `init`'s `error_scale`, which
            # keeps it (sharded) as `State.error_scale` for prediction
            error_scale = _process_response_input(error_scale)
            _check_same_length(x_train, error_scale)

        if missing is not None:
            missing = _process_response_input(missing, dtype=jnp.bool_)
            _check_same_length(x_train, missing)

        # check data types are correct for continuous/binary/multivariate regression
        outcome_type, binary_mask = _check_type_settings(
            y_train, outcome_type, error_scale
        )

        # process "standardization" settings
        offset = _process_offset_settings(
            y_train,
            binary_mask,
            missing,
            None if offset is None else jnp.asarray(offset, jnp.float32),
        )
        leaf_prior_cov_inv = _process_leaf_variance_settings(
            y_train,
            binary_mask,
            missing,
            jnp.asarray(k, jnp.float32),
            num_trees,
            None if tau_num is None else jnp.asarray(tau_num, jnp.float32),
        )
        error_cov_inv = _process_error_variance_settings(
            y_train,
            outcome_type,
            binary_mask,
            missing,
            sigma_df,
            sigma_scale,
            sigma_init,
            error_scale,
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
        initial_state, mcmc_key, device = _setup_mcmc(
            x_train,
            y_train,
            outcome_type,
            offset,
            error_scale,
            missing,
            max_split,
            leaf_prior_cov_inv,
            error_cov_inv,
            power,
            base,
            maxdepth,
            num_trees,
            init_kw,
            rm_const,
            sparse,
            varprob,
            num_chains,
            num_chain_devices,
            num_data_devices,
            devices,
            n_burn,
            keys.pop(),
        )
        result = _run_mcmc(
            initial_state,
            n_save,
            n_burn,
            n_skip,
            printevery,
            pbar,
            mcmc_key,
            precompute_predict_train,
            run_mcmc_kw,
        )

        # set private attributes
        self._main_trace = result.main_trace
        self._burnin_trace = result.burnin_trace
        self._mcmc_state = result.final_state
        self._binner = binner_obj
        self._x_train_fmt = x_train_fmt
        self._binary_mask = binary_mask
        self._device = device

    def predict(
        self,
        x_test: Real[ArrayLike, 'p m'] | DataFrame | str,
        *,
        kind: PredictKind | str = 'mean',
        key: Key[Array, ''] | None = None,
        error_scale: Float[ArrayLike, ' m']
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
        error_scale
            Per-observation error scale. Used with ``kind='outcome_samples'``,
            and also with ``kind='mean'`` or ``'mean_samples'`` for binary
            outcomes (since the success probability is
            ``Phi(latent / error_scale)``).
            Required when the model was fit with weights and ``x_test`` is
            new data. Shape matches the shape used at fitting: ``(m,)`` for
            scalar weights, ``(k, m)`` for multivariate vector weights.

        Returns
        -------
        Predictions at `x_test` in the requested format.

        Raises
        ------
        ValueError
            If `x_test` has a different format than `x_train`, or if `error_scale`
            is specified when it should be `None`, or if `error_scale` is not
            specified when it is required, or if the model splits datapoints
            across devices (`num_data_devices`) and the number of test points
            is not a multiple of the number of data devices.

        Notes
        -----
        If the model splits datapoints across devices (`num_data_devices`),
        the test points and the returned predictions are split the same way.
        """
        # parse arguments
        kind = PredictKind(kind)
        if kind is PredictKind.outcome_samples and key is None:
            msg = '`key` not specified'
            raise ValueError(msg)
        error_scale = self._process_error_scale_test(x_test, kind, error_scale)
        x_test_is_train = isinstance(x_test, str) and x_test == 'train'

        # use the predictions precomputed during the MCMC, if available
        if x_test_is_train and isinstance(self._main_trace, MainTraceWithTrainPred):
            return predict_train(
                key,
                self._main_trace,
                error_scale,
                self._mcmc_state.binary_indices,
                self._mcmc_state.z is not None,
                kind,
            )

        x_test = self._process_x_test(x_test, error_scale)

        # place new test data on the devices of the model; the training data
        # is already in place
        if not x_test_is_train:
            x_test, error_scale = self._device_put_test(x_test, error_scale)

        # invoke jitted implementation
        return predict(
            key,
            self._main_trace,
            x_test,
            error_scale,
            self._mcmc_state.binary_indices,
            self._mcmc_state.z is not None,
            kind,
            # the test points are sharded over the mesh 'data' axis (when
            # there is one): the training data at `init`, new test data by
            # `_device_put_test`. `evaluate_trace` can't detect this on its
            # own at trace time, so declare it.
            'shard_and_autobatch',
        )

    def _drop_device_info(self) -> 'Bart':
        """Return a copy of the model without device placement metadata.

        Clear the meshes in the MCMC state config and in the traces, and the
        explicitly requested device. Only this static metadata is dropped: the
        arrays keep their actual placement.
        """
        config = replace(self._mcmc_state.config, mesh=None)
        main_trace = replace(self._main_trace, mesh=None)
        burnin_trace = replace(self._burnin_trace, mesh=None)
        obj = tree_at(
            lambda b: (b._mcmc_state.config, b._main_trace, b._burnin_trace),  # noqa: SLF001
            self,
            (config, main_trace, burnin_trace),
        )
        # `_device` is a static field, out of `tree_at`'s reach, so modify the
        # fresh copy in place
        object.__setattr__(obj, '_device', None)
        return obj

    def dump(self, path: str | PathLike) -> None:
        """Serialize the fitted model to a file with `pickle`.

        Parameters
        ----------
        path
            The file to write to.

        Notes
        -----
        Intended for short-term storage (e.g. caching across processes), not
        long-term archival: the format depends on the versions of bartz, jax and
        equinox. The arrays are copied to host memory and all device/sharding
        placement is dropped; `load` reconstructs a single-device model.
        """
        # drop all device info (`Device` objects are not picklable), then
        # gather any sharded arrays to host (dropping their sharding); the
        # reload is single-device
        obj = self._drop_device_info()
        obj = jax.device_get(obj)
        with Path(path).open('wb') as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | PathLike) -> 'Bart':
        """Load a model saved with `dump`.

        Parameters
        ----------
        path
            The file to read from.

        Returns
        -------
        The deserialized model, on host memory with no device placement.

        Raises
        ------
        TypeError
            If the file does not contain a `Bart` instance.
        """
        with Path(path).open('rb') as file:
            obj = pickle.load(file)  # noqa: S301, the user owns the file
        if not isinstance(obj, cls):
            msg = f'unpickled a {type(obj).__name__}, not a {cls.__name__}'
            raise TypeError(msg)
        return obj

    @property
    def offset(self) -> Float32[Array, ''] | Float32[Array, ' k']:
        """The prior mean of the latent mean function."""
        return self._mcmc_state.forest.offset

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
        Float32[Array, ' n_burn_plus_n_save']
        | Float32[Array, 'n_burn_plus_n_save k k']
        | Float32[Array, 'num_chains n_burn_plus_n_save']
        | Float32[Array, 'num_chains n_burn_plus_n_save k k']
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

        Raises
        ------
        ValueError
             If `only_continuous` is `True` but the model has only binary
             outcomes, so there is no continuous submatrix to return.

        Notes
        -----
        This method is meant to check for convergence, so it returns the full
        MCMC trace and does not concatenate chains together. For probit
        regression, this returns the precision of the latent error term, not
        the Bernoulli precision for the binary outcome. For heteroskedastic
        regression, the returned precision is the global precision parameter,
        that would have to be divided by a squared weight to get the precision
        on a given datapoint.
        """
        binary_indices = self._mcmc_state.binary_indices
        if (
            only_continuous
            and binary_indices is None
            and self._mcmc_state.z is not None
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

    def _process_error_scale_test(
        self,
        x_test: Real[ArrayLike, 'p m'] | DataFrame | str,
        kind: PredictKind,
        error_scale: Float[ArrayLike, ' m']
        | Float[ArrayLike, 'k m']
        | Series
        | DataFrame
        | None,
    ) -> Float32[Array, ' m'] | Float32[Array, 'k m'] | None:
        """Validate and resolve the error weights for prediction.

        Parameters
        ----------
        x_test
            The raw (not yet processed) test predictors, or ``'train'``.
        kind
            The prediction kind.
        error_scale
            User-provided per-observation error scale, or `None`.

        Returns
        -------
        The resolved error scale as a float32 array, or `None` if weights are not applicable.

        Raises
        ------
        ValueError
            If `error_scale` is specified when it should be `None`, or missing
            when required.
        """
        x_test_is_train = isinstance(x_test, str) and x_test == 'train'
        train_error_scale = self._mcmc_state.error_scale
        has_train_weights = train_error_scale is not None
        is_binary = self._mcmc_state.z is not None
        # weights enter the outcome samples of any outcome type, and also the
        # mean of binary outcomes, since P(y=1) = Phi(latent / weight)
        needs_weights = has_train_weights and (
            kind is PredictKind.outcome_samples
            or (is_binary and kind in (PredictKind.mean, PredictKind.mean_samples))
        )

        if not needs_weights:
            if error_scale is not None:
                msg = (
                    '`error_scale` must be `None` in this configuration (weights'
                    " are used with kind='outcome_samples', and with kind='mean'"
                    " or 'mean_samples' for binary outcomes, and only when the"
                    ' model was fit with weights)'
                )
                raise ValueError(msg)
            return None

        if x_test_is_train:
            if error_scale is not None:
                msg = (
                    "`error_scale` must be `None` when x_test='train'"
                    ' (training weights are used automatically)'
                )
                raise ValueError(msg)
            return train_error_scale

        # new test data, model was fit with weights
        if error_scale is None:
            msg = (
                '`error_scale` is required because the model was fit with'
                ' weights and x_test is new data'
            )
            raise ValueError(msg)
        error_scale_test = _process_response_input(error_scale)
        assert train_error_scale is not None  # implied by needs_weights
        # the per-observation axis is checked separately, against x_test
        if error_scale_test.shape[:-1] != train_error_scale.shape[:-1]:
            msg = (
                f'`error_scale` shape mismatch with training weights: got '
                f'{error_scale_test.shape=}, but the leading dimensions must match '
                f'the training weights {train_error_scale.shape=} (only the '
                f'per-observation axis may differ).'
            )
            raise ValueError(msg)
        return error_scale_test

    def _process_x_test(
        self,
        x_test: Real[ArrayLike, 'p m'] | DataFrame | str,
        error_scale: Float32[Array, ' m'] | Float32[Array, 'k m'] | None,
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
        if error_scale is not None:
            _check_same_length(error_scale, x_test)
        return self._binner.bin(x_test)

    def _device_put_test(
        self,
        x_test: UInt[Array, 'p m'],
        error_scale: Float32[Array, ' m'] | Float32[Array, 'k m'] | None,
    ) -> tuple[UInt[Array, 'p m'], Float32[Array, ' m'] | Float32[Array, 'k m'] | None]:
        """Place new test data on the devices of the model.

        Mirror the placement of the training data done at fit time: shard over
        the mesh if there is one (the observation axis over 'data'), else move
        to the device requested explicitly at construction, if any. The inputs
        are donated, so they must not be used elsewhere.
        """
        mesh = self._mcmc_state.config.mesh
        if mesh is not None:
            put = lambda a: device_put(
                a,
                NamedSharding(mesh, _leaf_partition_spec(a.ndim, None, -1, mesh)),
                donate=True,
            )
        elif self._device is not None:
            put = lambda a: device_put(a, self._device, donate=True)
        else:
            return x_test, error_scale
        if error_scale is None:
            return put(x_test), None
        else:
            return put(x_test), put(error_scale)

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
        An array where ``(i, j, k)`` is `True` if tree `k` is invalid at iteration `j` in chain `i` but not at iteration ``j - 1``.
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
            # drop the data-sharded `leaf_indices` (not replicated) before the
            # cross-shard equality check; `None` is a deliberately off-type
            # placeholder, so use `tree_at`, which (unlike `dataclasses.replace`)
            # bypasses the `__init__` type checks
            replicated_forest = tree_at(lambda f: f.leaf_indices, state.forest, None)
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
        self, node_type: Literal['leaf', 'leaf-parent']
    ) -> Int32[Array, '*num_chains n_save n_plus_1']:
        return points_per_node_distr_trace(
            self._mcmc_state.X, self._main_trace, node_type
        )

    def _points_per_decision_node_distr(
        self,
    ) -> Int32[Array, '*num_chains n_save n_plus_1']:
        """Histogram of number of points belonging to parent-of-leaf nodes.

        Returns
        -------
        For each chain, a matrix where each row contains a histogram of number of points.
        """
        return self._points_per_node_distr('leaf-parent')

    def _points_per_leaf_distr(self) -> Int32[Array, '*num_chains n_save n_plus_1']:
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
        trees = _trees_chain_first(trace)
        index = (i_chain, i_sample, i_tree) if trace.has_chains else (i_sample, i_tree)
        # index the heap arrays, leaving the (unbatched) leaf scale alone; the
        # trailing ellipsis covers the extra `k` axis of multivariate leaves
        trees = tree_at(
            lambda t: (t.var_tree, t.split_tree, t.leaf_tree),
            trees,
            replace_fn=lambda x: x[(*index, ...)],
        )
        s = format_tree(trees, print_all=print_all)
        print(s)  # noqa: T201, this method is intended for debug


def _process_predictor_input(
    x: Real[ArrayLike, 'p n'] | DataFrame,
) -> tuple[Shaped[Array, 'p n'], Any]:
    if isinstance(x, DataFrame):
        fmt = dict(kind='dataframe', columns=x.columns)
        x = x.to_numpy().T
    else:
        fmt = dict(kind='array', num_covar=x.shape[0])
    x = jnp.asarray(x)
    assert x.ndim == 2
    return x, fmt


def _process_response_input(
    arr: Shaped[ArrayLike, ' n'] | Shaped[ArrayLike, 'k n'] | Series | DataFrame,
    /,
    *,
    dtype: DTypeLike = jnp.float32,
) -> Shaped[Array, ' n'] | Shaped[Array, 'k n']:
    if isinstance(arr, DataFrame):
        arr = arr.to_numpy().T
    elif isinstance(arr, Series):
        arr = arr.to_numpy()
    # one unconditional copy, safe to donate downstream
    arr = jnp.array(arr, dtype, copy=True)
    if arr.ndim < 1 or arr.ndim > 2:
        msg = f'response-like input must be 1D (n,) or 2D (k, n). Got {arr.ndim=}.'
        raise ValueError(msg)
    return arr


def _check_same_length(x1: Shaped[Array, '... n'], x2: Shaped[Array, '... n']) -> None:
    get_length = lambda x: x.shape[-1]
    assert get_length(x1) == get_length(x2)


def _check_type_settings(
    y_train: Float32[Array, ' n'] | Float32[Array, 'k n'],
    outcome_type: OutcomeType | str | Sequence[OutcomeType | str],
    error_scale: Float[Array, ' n'] | Float[Array, 'k n'] | None,
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
    if (
        error_scale is not None
        and error_scale.ndim == 2
        and (y_train.ndim != 2 or error_scale.shape[0] != y_train.shape[0])
    ):
        msg = (
            f'2D error_scale (vector per-component weights) requires y_train of '
            f'shape (k, n) with matching k; got {error_scale.shape=}, '
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
    x_train: Real[Array, 'p n'], sparse: SparseConfig
) -> (
    tuple[None, None, None, None]
    | tuple[FloatLike, None, None, None]
    | tuple[None, FloatLike, FloatLike, FloatLike]
):
    """Return (theta, a, b, rho)."""
    if not sparse.enabled:
        return None, None, None, None
    elif sparse.theta is not None:
        return sparse.theta, None, None, None
    else:
        rho = sparse.rho
        if rho is None:
            p, _ = x_train.shape
            rho = float(p)
        return None, sparse.a, sparse.b, rho


@jit
def _process_offset_settings(
    y_train: Float32[Array, ' n'] | Float32[Array, 'k n'],
    binary_mask: Bool[Array, ''] | Bool[Array, ' k'],
    missing: Bool[Array, ' n'] | Bool[Array, 'k n'] | None,
    offset: Float32[Array, ''] | Float32[Array, ' k'] | None,
) -> Float32[Array, ''] | Float32[Array, ' k']:
    """Determine and return offset."""
    if offset is not None:
        return jnp.broadcast_to(offset, y_train.shape[:-1])
    if y_train.shape[-1] < 1:
        return jnp.zeros(y_train.shape[:-1])

    if missing is None:
        *_, n_valid = y_train.shape
        prop = (y_train != 0).mean(-1)
        mean = y_train.mean(-1)
    else:
        *_, n = y_train.shape
        n_valid = n - jnp.count_nonzero(missing, axis=-1)
        safe_n = jnp.maximum(n_valid, 1)
        prop = jnp.where(missing, 0, y_train != 0).sum(-1) / safe_n
        mean = jnp.where(missing, 0.0, y_train).sum(-1) / safe_n

    bound = jnp.reciprocal(1.0 + n_valid)
    binary_offset = ndtri(jnp.clip(prop, bound, 1 - bound))
    offset = jnp.where(binary_mask, binary_offset, mean)
    return jnp.where(n_valid > 0, offset, 0.0)


@jit(static_argnums=(4,))
def _process_leaf_variance_settings(
    y_train: Float32[Array, ' n'] | Float32[Array, 'k n'],
    binary_mask: Bool[Array, ''] | Bool[Array, ' k'],
    missing: Bool[Array, ' n'] | Bool[Array, 'k n'] | None,
    k: Float[Array, ''],
    num_trees: int,
    tau_num: Float[Array, ''] | None,
) -> Float32[Array, ''] | Float32[Array, 'k k']:
    """Return `leaf_prior_cov_inv`."""
    # determine `tau_num` if not specified
    *kshape, n = y_train.shape
    if tau_num is None:
        if n < 2:
            continuous_tau = jnp.ones(kshape)
        elif missing is None:
            continuous_tau = (y_train.max(-1) - y_train.min(-1)) / 2
        else:
            n_valid = n - jnp.count_nonzero(missing, axis=-1)
            ymax = jnp.where(missing, -jnp.inf, y_train).max(-1)
            ymin = jnp.where(missing, jnp.inf, y_train).min(-1)
            continuous_tau = jnp.where(n_valid >= 2, (ymax - ymin) / 2, 1.0)
        tau_num = jnp.where(binary_mask, 3.0, continuous_tau)

    # leaf prior standard deviation
    sigma_mu = tau_num / (k * math.sqrt(num_trees))

    # leaf prior precision matrix
    leaf_prior_cov_inv = jnp.reciprocal(jnp.square(sigma_mu))
    if y_train.ndim == 2:
        leaf_prior_cov_inv = jnp.diag(jnp.broadcast_to(leaf_prior_cov_inv, kshape))
    return leaf_prior_cov_inv


def _process_error_variance_settings(
    y_train: Float32[Array, ' n'] | Float32[Array, 'k n'],
    outcome_type: OutcomeType | tuple[OutcomeType, ...],
    binary_mask: Bool[Array, ''] | Bool[Array, ' k'],
    missing: Bool[Array, ' n'] | Bool[Array, 'k n'] | None,
    sigma_df: FloatLike,
    sigma_scale: FloatLike | Float[ArrayLike, ' k'] | Literal['auto'],
    sigma_init: FloatLike | Float[ArrayLike, ' k'] | Literal['auto'],
    error_scale: Float32[Array, ' n'] | Float32[Array, 'k n'] | None,
) -> Wishart | None:
    """Build the error precision prior from the user settings."""
    if outcome_type is OutcomeType.binary:
        if not isinstance(sigma_scale, str) or not isinstance(sigma_init, str):
            msg = (
                'Do not set `sigma_scale` or `sigma_init` for binary regression, '
                'they are ignored'
            )
            raise ValueError(msg)
        return None

    *kdims, _ = y_train.shape  # () or (k,)
    k = kdims[0] if kdims else 1
    nu = jnp.asarray(sigma_df, jnp.float32) + (k - 1)

    # guarded per-component variance of y_train, computed only when an 'auto'
    # spec needs it (this function is not jitted, so it would not be elided)
    if isinstance(sigma_scale, str) or isinstance(sigma_init, str):
        vary = _guarded_response_variance(y_train, error_scale, missing)
    else:
        vary = None

    # prior rate: E[precision] = nu / rate, so rate = nu * var per component
    rate_diag = jnp.where(
        binary_mask, 0.0, nu * _resolve_error_variance(sigma_scale, vary, kdims)
    )

    # initial precision = 1 / var per component (1 for binary components)
    init_var = _resolve_error_variance(sigma_init, vary, kdims)
    init_diag = jnp.where(binary_mask, 1.0, jnp.reciprocal(init_var))

    if y_train.ndim == 2:
        rate, init = jnp.diag(rate_diag), jnp.diag(init_diag)
    else:
        rate, init = rate_diag, init_diag
    return make_error_cov_prior(nu, rate, init, outcome_type, missing)


@jit
def _guarded_response_variance(
    y_train: Float32[Array, ' n'] | Float32[Array, 'k n'],
    error_scale: Float32[Array, ' n'] | Float32[Array, 'k n'] | None,
    missing: Bool[Array, ' n'] | Bool[Array, 'k n'] | None,
) -> Float32[Array, '*k']:
    """Per-component variance of `y_train`, used by the 'auto' error scale.

    A precision-weighted variance (precision ``1 / error_scale ** 2``) estimates
    the unit-weight ``sigma ** 2``; `missing` entries are dropped. The variance
    is guarded to 1 when undefined (fewer than 2 valid points) or non-positive.
    """
    if error_scale is None and missing is None:
        vary = jnp.var(y_train, axis=-1)
        return jnp.where(vary > 0, vary, 1.0)
    else:
        prec = (
            jnp.ones(())
            if error_scale is None
            else jnp.reciprocal(jnp.square(error_scale))
        )
        if missing is not None:
            prec = jnp.where(missing, 0.0, prec)
            y_train = jnp.where(missing, 0.0, y_train)
        n_valid = jnp.count_nonzero(prec, axis=-1)
        wmean = jnp.sum(prec * y_train, axis=-1) / jnp.sum(prec, axis=-1)
        sqdev = prec * jnp.square(y_train - wmean[..., None])
        vary = jnp.sum(sqdev, axis=-1) / n_valid
        # guard on n_valid too: with a single valid point the variance is 0 in
        # exact arithmetic, but float rounding in wmean can leave a tiny
        # positive vary that would slip past the `vary > 0` guard
        return jnp.where((n_valid > 1) & (vary > 0), vary, 1.0)


def _resolve_error_variance(
    spec: FloatLike | Float[ArrayLike, ' k'] | Literal['auto'],
    vary: Float32[Array, '*k'] | None,
    shape: Sequence[int],
) -> Float32[Array, '*k']:
    """Per-component error variance from a scale spec ('auto' uses var(y))."""
    if isinstance(spec, str):
        if spec != 'auto':
            msg = f"unrecognized value {spec!r}, expected 'auto' or a number"
            raise ValueError(msg)
        assert vary is not None  # computed iff some spec is 'auto'
        return vary
    else:
        return jnp.broadcast_to(jnp.square(jnp.asarray(spec, jnp.float32)), shape)


def make_error_cov_prior(
    nu: Float32[Array, ''],
    rate: Float32[Array, ''] | Float32[Array, 'k k'],
    value: Float32[Array, ''] | Float32[Array, 'k k'],
    outcome_type: OutcomeType | tuple[OutcomeType, ...],
    missing: Bool[Array, ' n'] | Bool[Array, 'k n'] | None,
) -> Wishart:
    """Build the error precision prior, diagonal-constrained where required.

    Mixed binary-continuous and partial-missing (2-D mask) regression restrict
    the error covariance to diagonal, so they take a `DiagWishart`; the dense
    cases take a `Wishart`. `init` re-checks this choice. `value` is the initial
    value of the precision.
    """
    if isinstance(outcome_type, tuple):
        binary = [t is OutcomeType.binary for t in outcome_type]
        is_mixed = any(binary) and not all(binary)
    else:
        is_mixed = False
    # a 2-D missingness mask only occurs with multivariate y (checked in `init`)
    partial_missing = missing is not None and missing.ndim == 2
    if is_mixed or partial_missing:
        return DiagWishart(nu=nu, rate=rate, value=value)
    else:
        return Wishart(nu=nu, rate=rate, value=value)


def _setup_mcmc(
    x_train: Real[Array, 'p n'],
    y_train: Float32[Array, ' n'] | Float32[Array, 'k n'],
    outcome_type: OutcomeType | tuple[OutcomeType, ...],
    offset: Float32[Array, ''] | Float32[Array, ' k'],
    error_scale: Float[Array, ' n'] | Float[Array, 'k n'] | None,
    missing: Bool[Array, ' n'] | Bool[Array, 'k n'] | None,
    max_split: UInt[Array, ' p'],
    leaf_prior_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'],
    error_cov_inv: Wishart | None,
    power: FloatLike,
    base: FloatLike,
    maxdepth: int,
    num_trees: int,
    init_kw: Mapping[str, Any],
    rm_const: bool,
    sparse: SparseConfig,
    varprob: Float[ArrayLike, ' p'] | None,
    num_chains: int | None,
    num_chain_devices: int | None | Literal['auto'],
    num_data_devices: int | None,
    devices: Literal['cpu', 'gpu'] | Device | Sequence[Device] | None,
    n_burn: int,
    mcmc_key: Key[Array, ''],
) -> tuple[State, Key[Array, ''], Device | None]:
    p_nonterminal = make_p_nonterminal(maxdepth, base, power)

    # resolve the sparsity prior hyperparameters
    theta, a, b, rho = _process_sparsity_settings(x_train, sparse)

    # process device settings
    device_kw, device = process_device_settings(
        y_train, num_chains, num_chain_devices, num_data_devices, devices
    )

    kw: dict = dict(
        X=x_train,
        y=y_train,
        outcome_type=outcome_type,
        offset=offset,
        error_scale=error_scale,
        missing=missing,
        max_split=max_split,
        num_trees=num_trees,
        p_nonterminal=p_nonterminal,
        leaf_prior_cov_inv=leaf_prior_cov_inv,
        error_cov_inv=error_cov_inv,
        min_points_per_decision_node=10,
        log_s=process_varprob(varprob, max_split),
        theta=theta,
        a=a,
        b=b,
        rho=rho,
        sparse_on_at=n_burn // 2 if sparse.enabled else None,
        augment=sparse.augment,
        **device_kw,
    )

    if rm_const:
        n_empty = jnp.sum(max_split == 0).item()
        kw.update(filter_splitless_vars=n_empty)

    kw.update(init_kw)

    state = init(**kw)

    # put state and mcmc key on device if requested explicitly by the user
    if device is not None:
        mcmc_key, state = device_put((mcmc_key, state), device, donate=True)

    return state, mcmc_key, device


def _run_mcmc(
    mcmc_state: State,
    n_save: int,
    n_burn: int,
    n_skip: int,
    printevery: int | None,
    pbar: bool,
    key: Key[Array, ''],
    precompute_predict_train: bool,
    run_mcmc_kw: Mapping,
) -> RunMCMCResult:
    # prepare arguments
    kw: dict = dict(n_burn=n_burn, n_skip=n_skip, inner_loop_length=printevery)
    if precompute_predict_train:
        kw.update(main_trace_type=MainTraceWithTrainPred)
    # `printevery=None` disables progress reporting entirely: no callback is
    # installed, so the loop traces without any `debug.callback` effect (a tqdm
    # bar would otherwise advance every iteration regardless of `printevery`).
    if printevery is not None:
        if pbar:
            kw.update(make_tqdm_callback(mcmc_state, report_every=printevery))
        else:
            kw.update(
                make_print_callback(
                    mcmc_state,
                    dot_every=None if printevery == 1 else 1,
                    report_every=printevery,
                )
            )
    kw.update(run_mcmc_kw)

    return run_mcmc(key, mcmc_state, n_save, **kw)


@jit(static_argnames='p')
# this is jitted such that lax.collapse below does not create a copy
def varcount(p: int, trace: MainTrace) -> Int32[Array, 'ndpost p']:
    """Histogram of predictor usage for decision rules in the trees, squashing chains."""
    varcount: Int32[Array, '*chains samples p']
    varcount = compute_varcount(p, trace, out_chain_axis=0)
    return lax.collapse(varcount, 0, -1)


@jit(static_argnames='mean')
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


@jit(static_argnames='only_continuous')
def get_latent_prec(
    burnin_trace: BurninTrace,
    main_trace: MainTrace,
    binary_indices: Int32[Array, ' kb'] | None,
    *,
    only_continuous: bool = False,
) -> (
    Float32[Array, ' n_burn_plus_n_save']
    | Float32[Array, 'n_burn_plus_n_save k k']
    | Float32[Array, 'num_chains n_burn_plus_n_save']
    | Float32[Array, 'num_chains n_burn_plus_n_save k k']
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


def _trees_chain_first(obj: TreeHeaps) -> TreesTrace:
    """Extract `obj`'s heap arrays, moving any chain axis to the front.

    Returns a `TreesTrace` whose leading axis is the chain axis when `obj`
    carries one, and the bare per-object heap arrays otherwise.
    """
    trees = project(TreesTrace, obj)
    if get_has_chains(obj):
        axes = trees.axes_from_dataclass(chain_vmap_axes(obj))
        # WORKAROUND(python<3.14): use operator.is_none
        trees = tree.map(chain_to_axis, trees, axes, is_leaf=lambda x: x is None)
    return trees


@jit
def check_trees(
    trace: MainTrace, max_split: UInt[Array, ' p']
) -> UInt[Array, 'num_chains n_save num_trees']:
    """Apply `bartz.grove.check_trace` to all the tree draws."""
    trees = _trees_chain_first(trace)
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
    state: State, y: Float32[Array, ' n'] | Float32[Array, 'k n'] | None
) -> tuple[
    Float32[Array, '*num_chains n'] | Float32[Array, '*num_chains k n'],
    Float32[Array, '*num_chains n'] | Float32[Array, '*num_chains k n'],
]:
    """Re-compute residuals to compare them with the updated ones."""
    chain_axes = chain_vmap_axes(state)
    resid1 = chain_to_axis(state.resid * state.resid_scale[..., None], chain_axes.resid)
    z = chain_to_axis(state.z, chain_axes.z) if state.z is not None else None

    forests = _trees_chain_first(state.forest)
    trees = evaluate_forest(state.X, forests, sum_batch_axis=-1)

    if state.binary_indices is not None:
        # mixed binary-continuous: z has only binary rows, y has all rows
        assert y is not None
        ref = jnp.broadcast_to(y, resid1.shape)
        ref = ref.at[..., state.binary_indices, :].set(z)
    elif z is not None:
        ref = z
    else:
        assert y is not None
        ref = y
    resid2 = ref - (trees + state.forest.offset[..., None])

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


@jit(static_argnames='node_type')
def points_per_node_distr_trace(
    X: UInt[Array, 'p n'], trace: MainTrace, node_type: Literal['leaf', 'leaf-parent']
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


def process_device_settings(
    y_train: Shaped[Array, '...'],
    num_chains: int | None,
    num_chain_devices: int | None | Literal['auto'],
    num_data_devices: int | None,
    devices: Literal['cpu', 'gpu'] | Device | Sequence[Device] | None,
) -> tuple[DeviceKwArgs, Device | None]:
    """Return the arguments for `mcmcstep.init` related to devices, and an optional device where to put the state."""
    # whether the user pinned a concrete pool of devices (vs. inheriting all of
    # the platform's devices); the auto chain sharding may not exceed that pool
    explicit_devices = devices is not None and not isinstance(devices, str)
    platform, device, devices = _determine_devices(y_train, devices)
    num_chain_devices = _determine_num_chain_devices(
        platform,
        num_chains,
        num_chain_devices,
        num_data_devices,
        len(devices),
        explicit_devices,
    )
    mesh, device = _determine_mesh(num_chain_devices, num_data_devices, device, devices)

    # prepare arguments to `init`
    settings = DeviceKwArgs(num_chains=num_chains, mesh=mesh)

    return settings, device


def _determine_devices(
    y_train: Shaped[Array, '...'],
    devices: Literal['cpu', 'gpu'] | Device | Sequence[Device] | None,
) -> tuple[str, Device | None, Sequence[Device]]:
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
        platform = y_train.platform()  # ty: ignore[call-non-callable]
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
    num_data_devices: int | None,
    num_devices: int,
    explicit_devices: bool,
) -> int | None:
    """Resolve and validate `num_chain_devices`, returning the chain mesh axis size or `None`."""
    if num_chain_devices == 'auto':
        num_chain_devices = _auto_num_chain_devices(
            platform, num_chains, num_data_devices, num_devices, explicit_devices
        )

    # an explicit value must be a positive divisor of the number of chains
    if num_chain_devices is not None:
        effective_chains = 1 if num_chains is None else num_chains
        if num_chain_devices < 1 or effective_chains % num_chain_devices:
            chains_desc = (
                'a single chain (num_chains=None)'
                if num_chains is None
                else f'num_chains={num_chains}'
            )
            msg = (
                f'num_chain_devices={num_chain_devices} must be a positive '
                f'divisor of the number of chains ({chains_desc})'
            )
            raise ValueError(msg)

    # there is no chain axis to shard when the chains are scalar
    if num_chains is None:
        return None
    return num_chain_devices


def _auto_num_chain_devices(
    platform: str,
    num_chains: int | None,
    num_data_devices: int | None,
    num_devices: int,
    explicit_devices: bool,
) -> int | None:
    """Pick `num_chain_devices` automatically for multi-chain cpu runs.

    `num_data_devices` reserves devices for the data axis, so the chain axis can
    only use a fraction of them; this keeps the ``chains x data`` mesh within the
    `num_devices` available devices.
    """
    if num_chains is None or num_chains == 1 or platform != 'cpu':
        return None
    data_devices = num_data_devices or 1
    num_cores = cpu_count()
    assert num_cores is not None, 'could not determine number of cpu cores'

    # devices available for the chain axis after reserving for the data axis
    core_budget = max(1, num_cores // data_devices)
    num_shards = _largest_divisor_at_most(num_chains, core_budget)

    if num_shards > 1:
        # the mesh draws from `num_devices` devices, whether those are all the
        # platform's devices or an explicit subset passed by the user
        device_budget = max(1, num_devices // data_devices)
        if device_budget < num_shards:
            new_num_shards = _largest_divisor_at_most(num_chains, device_budget)
            warn(
                _auto_chain_devices_warning(
                    num_chains,
                    num_shards,
                    new_num_shards,
                    device_budget,
                    num_devices,
                    num_data_devices,
                    explicit_devices,
                )
            )
            num_shards = new_num_shards

    return num_shards if num_shards > 1 else None


def _auto_chain_devices_warning(
    num_chains: int,
    desired: int,
    actual: int,
    device_budget: int,
    num_devices: int,
    num_data_devices: int | None,
    explicit_devices: bool,
) -> str:
    """Compose the warning shown when auto chain sharding is capped by the device count."""
    if explicit_devices:
        pool = f'the {num_devices} devices passed in `devices`'
        few = f'only {num_devices} devices were passed in `devices`'
        advice = ''
    else:
        pool = f'the {num_devices} jax cpu devices'
        few = f'jax is set up with only {num_devices} cpu devices'
        advice = (
            ' To enable more parallelization, increase the limit with '
            '`jax.config.update("jax_num_cpu_devices", <num_devices>)`.'
        )
    if num_data_devices:
        limit = (
            f'only {device_budget} of {pool} are free for chains '
            f'(num_data_devices={num_data_devices} reserves the rest)'
        )
    else:
        limit = few
    return (
        f'`Bart` would like to shard {num_chains} chains across {desired} '
        f'devices, but {limit}, so it will use {actual} devices for chains '
        f'instead.{advice}'
    )


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
    x: UInt[Array, 'p m'],
    trace: MainTrace,
    test_points: Literal['none', 'autobatch', 'shard_and_autobatch'] = 'none',
) -> Float32[Array, 'ndpost m'] | Float32[Array, 'ndpost k m']:
    """Evaluate trees on already quantized `x`, and squash chains."""
    return evaluate_trace(x, trace, flatten_chains=True, test_points=test_points)


@jit(static_argnums=(5, 6, 7))
def predict(
    key: Key[Array, ''] | None,
    trace: MainTrace,
    x_test: UInt[Array, 'p m'],
    error_scale: Float[Array, ' m'] | Float[Array, 'k m'] | None,
    binary_indices: Int32[Array, ' kb'] | None,
    has_binary: bool,
    kind: PredictKind | str,
    test_points: Literal['none', 'autobatch', 'shard_and_autobatch'],
    /,
) -> (
    Float32[Array, ' m']
    | Float32[Array, 'k m']
    | Float32[Array, 'ndpost m']
    | Float32[Array, 'ndpost k m']
):
    """Implement `Bart.predict` by evaluating the trees on `x_test`."""
    # get latent i.e. bare sum-of-trees predictions
    latent = predict_latent(x_test, trace, test_points)
    return _predict_from_latent(
        key, trace, latent, error_scale, binary_indices, has_binary, kind
    )


@jit(static_argnums=(4, 5))
def predict_train(
    key: Key[Array, ''] | None,
    trace: MainTraceWithTrainPred,
    error_scale: Float[Array, ' n'] | Float[Array, 'k n'] | None,
    binary_indices: Int32[Array, ' kb'] | None,
    has_binary: bool,
    kind: PredictKind | str,
    /,
) -> (
    Float32[Array, ' n']
    | Float32[Array, 'k n']
    | Float32[Array, 'ndpost n']
    | Float32[Array, 'ndpost k n']
):
    """Implement `Bart.predict('train')` from the precomputed predictions."""
    latent = _flatten_chain_sample(
        trace.train_pred,
        chain_vmap_axes(trace).train_pred,
        trace_sample_axes(trace).train_pred,
    )
    return _predict_from_latent(
        key, trace, latent, error_scale, binary_indices, has_binary, kind
    )


def _predict_from_latent(
    key: Key[Array, ''] | None,
    trace: MainTrace,
    latent: Float32[Array, 'ndpost m'] | Float32[Array, 'ndpost k m'],
    error_scale: Float[Array, ' m'] | Float[Array, 'k m'] | None,
    binary_indices: Int32[Array, ' kb'] | None,
    has_binary: bool,
    kind: PredictKind | str,
) -> (
    Float32[Array, ' m']
    | Float32[Array, 'k m']
    | Float32[Array, 'ndpost m']
    | Float32[Array, 'ndpost k m']
):
    """Turn the sum-of-trees `latent` samples into the requested prediction kind."""
    if kind is PredictKind.latent_samples:
        return latent

    # sample posterior (uses latent directly, no probit squash needed)
    if kind is PredictKind.outcome_samples:
        assert key is not None
        return sample_outcome(
            key, trace, latent, error_scale, binary_indices, has_binary
        )

    # squash predictions to (0, 1) if probit; with heteroskedastic weights
    # P(y=1) = Phi(latent / error_scale)
    if has_binary:  # self._mcmc_state.z is not None
        # error_scale is (m,) or (k, m), so it broadcasts against latent
        arg = latent if error_scale is None else latent / error_scale
        if binary_indices is not None:
            # mixed: squash only the binary rows, leaving continuous rows as-is
            indexing = jnp.s_[..., binary_indices, :]
            mean_samples = latent.at[indexing].set(ndtr(arg[indexing]))
        else:
            mean_samples = ndtr(arg)
    else:
        mean_samples = latent

    # take mean or return samples
    if kind is PredictKind.mean:
        return mean_samples.mean(axis=0)
    return mean_samples


def _flatten_chain_sample(
    arr: Float[Array, '*shape'], chain_axis: int | None, sample_axis: int
) -> Float[Array, '*flat_shape']:
    """Fold the chain axis into the sample axis, matching `predict_latent`'s layout."""
    if chain_axis is None:
        return arr
    arr = jnp.moveaxis(arr, (chain_axis, sample_axis), (0, 1))
    return lax.collapse(arr, 0, 2)


@jit(static_argnums=(5,))
def sample_outcome(
    key: Key[Array, ''],
    trace: MainTrace,
    latent: Float32[Array, 'ndpost m'] | Float32[Array, 'ndpost k m'],
    error_scale: Float32[Array, ' m'] | Float32[Array, 'k m'] | None,
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
        if error_scale is not None:
            # error_scale is (m,) or (k, m) so it always broadcasts right
            error *= error_scale
    else:  # univariate
        # pure binary probit has unit-scale latent error; continuous scales it
        # by `sigma`. Either way, optionally rescaled per datapoint by error_scale.
        error = random.normal(key, latent.shape)
        if not has_binary:
            sigma = jnp.sqrt(jnp.reciprocal(prec)).reshape(-1)
            error *= sigma[..., None]
        if error_scale is not None:
            error *= error_scale[None, :]

    outcome = latent + error

    # convert binary outcomes via latent probit thresholding
    if binary_indices is not None:
        idx = jnp.s_[..., binary_indices, :]
        outcome = outcome.at[idx].set(jnp.where(outcome[idx] > 0, 1.0, 0.0))
    elif has_binary:
        outcome = jnp.where(outcome > 0, 1.0, 0.0)

    return outcome
