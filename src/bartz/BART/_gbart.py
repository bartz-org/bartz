# bartz/src/bartz/BART/_gbart.py
#
# Copyright (c) 2024-2026, The Bartz Contributors
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

"""Implement classes `mc_gbart` and `gbart` that mimic the R BART3 package."""

from collections.abc import Mapping
from functools import cached_property
from os import cpu_count
from types import MappingProxyType
from typing import Any, Literal
from warnings import warn

from equinox import Module
from jax import device_count
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Float32, Int32, Key, Real

from bartz import mcmcloop, mcmcstep
from bartz._interface import Bart, DataFrame, FloatLike, Series
from bartz.jaxext import get_default_device


class mc_gbart(Module):
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
        The training responses.
    x_test
        The test predictors.
    type
        The type of regression. 'wbart' for continuous regression, 'pbart' for
        binary regression with probit link.
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
        deviation of `y_train`. Ignored if `lamda` is specified.
    sigdf
        The degrees of freedom of the scaled inverse-chisquared prior on the
        noise variance.
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
        `sigquant`.
    tau_num
        The numerator in the expression that determines the prior standard
        deviation of leaves. If not specified, default to ``(max(y_train) -
        min(y_train)) / 2`` (or 1 if `y_train` has less than two elements) for
        continuous regression, and 3 for binary regression.
    offset
        The prior mean of the latent mean function. If not specified, it is set
        to the mean of `y_train` for continuous regression, and to
        ``Phi^-1(mean(y_train))`` for binary regression. If `y_train` is empty,
        `offset` is set to 0. With binary regression, if `y_train` is all
        `False` or `True`, it is set to ``Phi^-1(1/(n+1))`` or
        ``Phi^-1(n/(n+1))``, respectively.
    w
        Coefficients that rescale the error standard deviation on each
        datapoint. Not specifying `w` is equivalent to setting it to 1 for all
        datapoints. Note: `w` is ignored in the automatic determination of
        `sigest`, so either the weights should be O(1), or `sigest` should be
        specified by the user.
    ntree
        The number of trees used to represent the latent mean function. By
        default 200 for continuous regression and 50 for binary regression.
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
        The thinning factor for the MCMC samples, after burn-in. By default, 1
        for continuous regression and 10 for binary regression.
    printevery
        The number of iterations (including thinned-away ones) between each log
        line. Set to `None` to disable logging. ^C interrupts the MCMC only
        every `printevery` iterations, so with logging disabled it's impossible
        to kill the MCMC conveniently.
    mc_cores
        The number of independent MCMC chains.
    seed
        The seed for the random number generator.
    bart_kwargs
        Additional arguments passed to `bartz.Bart`.

    Notes
    -----
    This interface imitates the function ``mc_gbart`` from the R package `BART3
    <https://github.com/rsparapa/bnptools>`_, but with these differences:

    - If `x_train` and `x_test` are matrices, they have one predictor per row
      instead of per column.
    - If ``usequants=False``, R BART3 switches to quantiles anyway if there are
      less predictor values than the required number of bins, while bartz
      always follows the specification.
    - Some functionality is missing.
    - The error variance parameter is called `lamda` instead of `lambda`.
    - There are some additional attributes, and some missing.
    - The trees have a maximum depth of 6.
    - `rm_const` refers to predictors without decision rules instead of
      predictors that are constant in `x_train`.
    - If `rm_const=True` and some variables are dropped, the predictors
      matrix/dataframe passed to `predict` should still include them.

    References
    ----------
    .. [1] Linero, Antonio R. (2018). "Bayesian Regression Trees for
       High-Dimensional Prediction and Variable Selection". In: Journal of the
       American Statistical Association 113.522, pp. 626-636.
    .. [2] Hugh A. Chipman, Edward I. George, Robert E. McCulloch "BART:
       Bayesian additive regression trees," The Annals of Applied Statistics,
       Ann. Appl. Stat. 4(1), 266-298, (March 2010).
    """

    _bart: Bart

    def __init__(
        self,
        x_train: Real[Array, 'p n'] | DataFrame,
        y_train: Bool[Array, ' n'] | Float32[Array, ' n'] | Series,
        *,
        x_test: Real[Array, 'p m'] | DataFrame | None = None,
        type: Literal['wbart', 'pbart'] = 'wbart',  # noqa: A002
        sparse: bool = False,
        theta: FloatLike | None = None,
        a: FloatLike = 0.5,
        b: FloatLike = 1.0,
        rho: FloatLike | None = None,
        varprob: Float[Array, ' p'] | None = None,
        xinfo: Float[Array, 'p n'] | None = None,
        usequants: bool = False,
        rm_const: bool = True,
        sigest: FloatLike | None = None,
        sigdf: FloatLike = 3.0,
        sigquant: FloatLike = 0.9,
        k: FloatLike = 2.0,
        power: FloatLike = 2.0,
        base: FloatLike = 0.95,
        lamda: FloatLike | None = None,
        tau_num: FloatLike | None = None,
        offset: FloatLike | None = None,
        w: Float[Array, ' n'] | None = None,
        ntree: int | None = None,
        numcut: int = 100,
        ndpost: int = 1000,
        nskip: int = 100,
        keepevery: int | None = None,
        printevery: int | None = 100,
        mc_cores: int = 2,
        seed: int | Key[Array, ''] = 0,
        bart_kwargs: Mapping = MappingProxyType({}),
    ) -> None:
        kwargs: dict = dict(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            type=type,
            sparse=sparse,
            theta=theta,
            a=a,
            b=b,
            rho=rho,
            varprob=varprob,
            xinfo=xinfo,
            usequants=usequants,
            rm_const=rm_const,
            sigest=sigest,
            sigdf=sigdf,
            sigquant=sigquant,
            k=k,
            power=power,
            base=base,
            lamda=lamda,
            tau_num=tau_num,
            offset=offset,
            w=w,
            ntree=ntree,
            numcut=numcut,
            ndpost=ndpost,
            nskip=nskip,
            keepevery=keepevery,
            printevery=printevery,
            seed=seed,
            maxdepth=6,
            **process_mc_cores(y_train, mc_cores),
        )
        kwargs.update(bart_kwargs)
        self._bart = Bart(**kwargs)

    # Public attributes from Bart

    @property
    def ndpost(self) -> int:
        """The number of MCMC samples saved, after burn-in."""
        return self._bart.ndpost

    @property
    def offset(self) -> Float32[Array, '']:
        """The prior mean of the latent mean function."""
        return self._bart.offset

    @property
    def sigest(self) -> Float32[Array, ''] | None:
        """The estimated standard deviation of the error used to set `lamda`."""
        return self._bart.sigest

    @property
    def yhat_test(self) -> Float32[Array, 'ndpost m'] | None:
        """The conditional posterior mean at `x_test` for each MCMC iteration."""
        return self._bart.yhat_test

    # Private attributes from Bart

    @property
    def _main_trace(self) -> mcmcloop.MainTrace:
        return self._bart._main_trace  # noqa: SLF001

    @property
    def _burnin_trace(self) -> mcmcloop.BurninTrace:
        return self._bart._burnin_trace  # noqa: SLF001

    @property
    def _mcmc_state(self) -> mcmcstep.State:
        return self._bart._mcmc_state  # noqa: SLF001

    @property
    def _splits(self) -> Real[Array, 'p max_num_splits']:
        return self._bart._splits  # noqa: SLF001

    @property
    def _x_train_fmt(self) -> Any:
        return self._bart._x_train_fmt  # noqa: SLF001

    # Cached properties from Bart

    @cached_property
    def prob_test(self) -> Float32[Array, 'ndpost m'] | None:
        """The posterior probability of y being True at `x_test` for each MCMC iteration."""
        return self._bart.prob_test

    @cached_property
    def prob_test_mean(self) -> Float32[Array, ' m'] | None:
        """The marginal posterior probability of y being True at `x_test`."""
        return self._bart.prob_test_mean

    @cached_property
    def prob_train(self) -> Float32[Array, 'ndpost n'] | None:
        """The posterior probability of y being True at `x_train` for each MCMC iteration."""
        return self._bart.prob_train

    @cached_property
    def prob_train_mean(self) -> Float32[Array, ' n'] | None:
        """The marginal posterior probability of y being True at `x_train`."""
        return self._bart.prob_train_mean

    @cached_property
    def sigma(
        self,
    ) -> (
        Float32[Array, ' nskip+ndpost']
        | Float32[Array, 'nskip+ndpost/mc_cores mc_cores']
        | None
    ):
        """The standard deviation of the error, including burn-in samples."""
        return self._bart.sigma

    @cached_property
    def sigma_(self) -> Float32[Array, 'ndpost'] | None:
        """The standard deviation of the error, only over the post-burnin samples and flattened."""
        return self._bart.sigma_

    @cached_property
    def sigma_mean(self) -> Float32[Array, ''] | None:
        """The mean of `sigma`, only over the post-burnin samples."""
        return self._bart.sigma_mean

    @cached_property
    def varcount(self) -> Int32[Array, 'ndpost p']:
        """Histogram of predictor usage for decision rules in the trees."""
        return self._bart.varcount

    @cached_property
    def varcount_mean(self) -> Float32[Array, ' p']:
        """Average of `varcount` across MCMC iterations."""
        return self._bart.varcount_mean

    @cached_property
    def varprob(self) -> Float32[Array, 'ndpost p']:
        """Posterior samples of the probability of choosing each predictor for a decision rule."""
        return self._bart.varprob

    @cached_property
    def varprob_mean(self) -> Float32[Array, ' p']:
        """The marginal posterior probability of each predictor being chosen for a decision rule."""
        return self._bart.varprob_mean

    @cached_property
    def yhat_test_mean(self) -> Float32[Array, ' m'] | None:
        """The marginal posterior mean at `x_test`.

        Not defined with binary regression because it's error-prone, typically
        the right thing to consider would be `prob_test_mean`.
        """
        return self._bart.yhat_test_mean

    @cached_property
    def yhat_train(self) -> Float32[Array, 'ndpost n']:
        """The conditional posterior mean at `x_train` for each MCMC iteration."""
        return self._bart.yhat_train

    @cached_property
    def yhat_train_mean(self) -> Float32[Array, ' n'] | None:
        """The marginal posterior mean at `x_train`.

        Not defined with binary regression because it's error-prone, typically
        the right thing to consider would be `prob_train_mean`.
        """
        return self._bart.yhat_train_mean

    # Public methods from Bart

    def predict(
        self, x_test: Real[Array, 'p m'] | DataFrame
    ) -> Float32[Array, 'ndpost m']:
        """
        Compute the posterior mean at `x_test` for each MCMC iteration.

        Parameters
        ----------
        x_test
            The test predictors.

        Returns
        -------
        The conditional posterior mean at `x_test` for each MCMC iteration.
        """
        return self._bart.predict(x_test)


class gbart(mc_gbart):
    """Subclass of `mc_gbart` that forces `mc_cores=1`."""

    def __init__(self, *args, **kwargs) -> None:
        if 'mc_cores' in kwargs:
            msg = "gbart.__init__() got an unexpected keyword argument 'mc_cores'"
            raise TypeError(msg)
        kwargs.update(mc_cores=1)
        super().__init__(*args, **kwargs)


def process_mc_cores(y_train: Array | Any, mc_cores: int) -> dict[str, Any]:
    """Determine the arguments to pass to `Bart` to configure multiple chains."""
    # one chain, leave default configuration which is num_chains=None
    if abs(mc_cores) == 1:
        return {}

    # determine if we are on cpu; this point may raise an exception
    platform = get_platform(y_train, mc_cores)

    # set the num_chains argument
    mc_cores = abs(mc_cores)
    kwargs = dict(num_chains=mc_cores)

    # if on cpu, try to shard the chains across multiple virtual cpus
    if platform == 'cpu':
        # determine number of logical cpu cores
        num_cores = cpu_count()
        assert num_cores is not None, 'could not determine number of cpu cores'

        # determine number of shards that evenly divides chains
        for num_shards in range(num_cores, 0, -1):
            if mc_cores % num_shards == 0:
                break

        # handle the case where there are less jax cpu devices that that
        if num_shards > 1:
            num_jax_cpus = device_count('cpu')
            if num_jax_cpus < num_shards:
                for new_num_shards in range(num_jax_cpus, 0, -1):
                    if mc_cores % new_num_shards == 0:
                        break
                msg = (
                    f'`mc_gbart` would like to shard {mc_cores} chains across '
                    f'{num_shards} virtual jax cpu devices, but jax is set up '
                    f'with only {num_jax_cpus} cpu devices, so it will use '
                    f'{new_num_shards} devices instead. To enable '
                    'parallelization, please increase the limit with '
                    '`jax.config.update("jax_num_cpu_devices", <num_devices>)`.'
                )
                warn(msg)
                num_shards = new_num_shards

        # set the number of shards
        if num_shards > 1:
            kwargs.update(num_chain_devices=num_shards)

    return kwargs


def get_platform(y_train: Array | Any, mc_cores: int) -> str:
    """Get the platform for `process_mc_cores` from `y_train` or the default device."""
    if isinstance(y_train, Array) and hasattr(y_train, 'platform'):
        return y_train.platform()
    elif (
        not isinstance(y_train, Array) and hasattr(jnp.zeros(()), 'platform')
        # this condition means: y_train is not an array, but we are not under
        # jit, so y_train is going to be converted to an array on the default
        # device
    ) or mc_cores < 0:
        return get_default_device().platform
    else:
        msg = (
            'Could not determine the platform from `y_train`, maybe `mc_gbart` '
            'was used with a `jax.jit`ted function? The platform is needed to '
            'determine whether the computation is going to run on CPU to '
            'automatically shard the chains across multiple virtual CPU '
            'devices. To acknowledge this problem and circumvent it '
            'by using the current default jax device, negate `mc_cores`.'
        )
        raise RuntimeError(msg)
