# bartz/tests/test_interface.py
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

"""Test `bartz.Bart`.

This is the main suite of tests.
"""

import math
import pickle
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, redirect_stderr
from dataclasses import dataclass, replace
from functools import partial
from gc import collect
from inspect import signature
from io import StringIO
from os import cpu_count
from pathlib import Path
from typing import Any, Literal, NamedTuple
from weakref import ReferenceType, ref

import jax
import numpy
import pandas as pd
import polars as pl
import pytest
from equinox import EquinoxRuntimeError
from jax import (
    Device,
    block_until_ready,
    config,
    debug_infs,
    debug_key_reuse,
    debug_nans,
    jit,
    lax,
    no_tracing,
    random,
    tree,
    vmap,
)
from jax import numpy as jnp
from jax.nn import softmax
from jax.scipy.special import ndtr
from jax.sharding import Mesh, SingleDeviceSharding
from jax.tree_util import KeyPath, keystr
from jaxtyping import (
    Array,
    Bool,
    Float,
    Float32,
    Int32,
    Key,
    PyTree,
    Real,
    Shaped,
    UInt,
)
from numpy.testing import assert_array_less
from pytest import CaptureFixture, FixtureRequest  # noqa: PT013
from pytest_subtests import SubTests

from bartz import Bart as OriginalBart
from bartz import PredictKind, SparseConfig
from bartz._interface import (
    DataFrame,
    Series,
    _guarded_response_variance,
    predict_latent,
)
from bartz._jaxext import (
    get_default_device,
    get_default_devices,
    get_device_count,
    is_key,
    jaxtyping_disabled,
    minimal_unsigned_dtype,
    project,
    split,
)
from bartz._typing import kwdict
from bartz.debug import MinimalTrace, PriorSample, sample_prior
from bartz.grove import (
    check_trace,
    describe_error,
    forest_depth_distr,
    is_actual_leaf,
    tree_actual_depth,
    tree_depth,
    tree_depths,
)
from bartz.mcmcloop import (
    Callback,
    MainTraceWithTrainPred,
    compute_varcount,
    evaluate_trace,
)
from bartz.mcmcloop._callback import _TQDM_REGISTRY
from bartz.mcmcloop._loop import _inner_loop_counter
from bartz.mcmcstep import BatchedReduction, State, step
from bartz.mcmcstep._axes import chain_to_axis, chain_vmap_axes
from bartz.mcmcstep._state import init
from bartz.prepcovars import GivenSplitsBinner, RangeEvenBinner, UniqueQuantileBinner
from bartz.testing import DiscreteUniform, Gamma, gen_data
from tests.test_mcmcstep import (
    check_sharding,
    get_normal_spec,
    normalize_spec,
    reduce_reference,
)
from tests.util import (
    assert_allclose,
    assert_array_equal,
    assert_close_matrices,
    assert_different_matrices,
    clipped_logit,
    condf,
    nnone,
    periodic_sigint,
    rhat_rank,
)


class Bart(OriginalBart):
    """Wrapper that enables debug checks by default."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._check_trees(error=True)
        self._check_replicated_trees()


# All test data comes from `bartz.testing.gen_data`; the wrappers below adapt
# it for missingness masks and exact sparsity.

# `gen_data` settings shared by all its call sites: enough signal that BART can
# identify it, and enough noise that the convergence-test posteriors mix well.
# q = 0 because the small variants have p = 2, and q must be < p; the binary-X
# variants override it because m = 2 requires q >= 2.
GEN_KW: kwdict = dict(q=0, sigma2_lin=0.5, sigma2_quad=0.5, sigma2_eps=0.04)


def gen_sparse_data(
    key: Key[Array, ''], *, n: int, p: int, peff: int
) -> tuple[Float[Array, 'p n'], Float[Array, ' n'], Bool[Array, ' p']]:
    """Generate continuous data that depends on exactly `peff` of `p` predictors.

    The signal is a `gen_data` DGP on `peff` predictors (binary coefficient
    draws, so they are equally important), padded with irrelevant predictors
    from the same family up to `p` and shuffled. Returns ``(X, y, mask)``
    where ``mask`` flags the relevant predictors.
    """
    keys = split(key, 3)
    # lower noise than GEN_KW: the relevant/irrelevant varprob contrast needs a
    # higher signal-to-noise ratio than the convergence tests
    gen_kw: kwdict = dict(GEN_KW, sigma2_eps=0.01)
    dgp = gen_data(keys.pop(), n=n, p=peff, **gen_kw)
    padding = dgp.params.x_distr.sample(keys.pop(), (p - peff, n))
    X = jnp.concatenate([dgp.x, padding], axis=0)
    mask = jnp.arange(p) < peff
    perm = random.permutation(keys.pop(), p)
    return X[perm, :], dgp.y, mask[perm]


def gen_missing(key: Key[Array, ''], shape: tuple[int, int]) -> Bool[Array, 'k n']:
    """Generate a boolean missingness mask that is not completely at random.

    The mask is the binary outcome of its own `gen_data` DGP, so the entries
    are driven by latent predictors and correlated within a datapoint, instead
    of i.i.d. coin flips.
    """
    k, n = shape
    dgp = gen_data(key, n=n, p=k, k=k, lambda_=0.5, outcome_type='binary', **GEN_KW)
    return dgp.y.astype(bool)


def _bart_default(kw: dict[str, Any], param_name: str) -> Any:  # noqa: ANN401
    """Do `kw.get(param_name, <default in Bart>)`."""
    param = signature(OriginalBart).parameters[param_name]
    if param.default is param.empty:
        return kw[param_name]
    return kw.get(param_name, param.default)


def _init_default(kw: dict[str, Any], param_name: str) -> Any:  # noqa: ANN401
    """Do `kw['init_kw'].get(param_name, <default in init>)`."""
    default = signature(init).parameters[param_name].default
    return kw.get('init_kw', {}).get(param_name, default)


class BartKW(NamedTuple):
    """Keyword arguments for `Bart` plus associated test data."""

    kw: dict[str, Any]
    x_test: Real[Array, 'p m']
    w_test: Float32[Array, ' m'] | Float32[Array, 'k m'] | None = None

    @property
    def uses_quantile_binner(self) -> bool:
        """Whether `kw['binner']` (a class or `partial`) is `UniqueQuantileBinner`."""
        binner = self.kw.get('binner')
        return getattr(binner, 'func', binner) is UniqueQuantileBinner

    @property
    def any_binary(self) -> bool:
        """Whether `kw['outcome_type']` includes any binary component."""
        outcome_type = _bart_default(self.kw, 'outcome_type')
        if isinstance(outcome_type, str):
            return outcome_type == 'binary'
        return 'binary' in outcome_type

    @property
    def all_binary(self) -> bool:
        """Whether all components of `kw['outcome_type']` are binary."""
        outcome_type = _bart_default(self.kw, 'outcome_type')
        if isinstance(outcome_type, str):
            return outcome_type == 'binary'
        return all(t == 'binary' for t in outcome_type)

    @property
    def is_mixed(self) -> bool:
        """Whether `kw['outcome_type']` has both binary and continuous components."""
        outcome_type = _bart_default(self.kw, 'outcome_type')
        if isinstance(outcome_type, str):
            return False
        return any(t == 'binary' for t in outcome_type) and any(
            t != 'binary' for t in outcome_type
        )

    @property
    def binary_mask(self) -> Bool[Array, '*k']:
        """Bool mask of binary components; shape matches ``y_train.shape[:-1]``."""
        outcome_type = _bart_default(self.kw, 'outcome_type')
        if isinstance(outcome_type, str):
            mask = jnp.bool_(outcome_type == 'binary')
        else:
            mask = jnp.array([t == 'binary' for t in outcome_type])
        return jnp.broadcast_to(mask, self.kshape)

    def w_for_predict(
        self, kind: str, *, on_train: bool
    ) -> Float32[Array, ' m'] | Float32[Array, 'k m'] | None:
        """Return the `w` to pass to `Bart.predict`, or `None`, given kind and target.

        On new test data error scales are needed for outcome samples (any
        outcome type) and for the mean of binary outcomes; on the training data
        they are supplied automatically, so `None` is returned.
        """
        if on_train:
            return None
        elif kind == 'outcome_samples' or (
            self.any_binary and kind in ('mean', 'mean_samples')
        ):
            return self.w_test
        else:
            return None

    @property
    def p(self) -> int:
        """Number of predictors (``kw['x_train'].shape[0]``)."""
        p, _ = self.kw['x_train'].shape
        return p

    @property
    def n(self) -> int:
        """Number of training observations (``kw['x_train'].shape[1]``)."""
        _, n = self.kw['x_train'].shape
        return n

    @property
    def kshape(self) -> tuple[int, ...]:
        """Outer dimensions of `kw['y_train']`: ``()`` UV, ``(k,)`` MV."""
        return self.kw['y_train'].shape[:-1]

    @property
    def k(self) -> int | None:
        """Number of outcomes; ``None`` if univariate."""
        shape = self.kshape
        if not shape:
            return None
        (k,) = shape
        return k

    @property
    def num_chains(self) -> int | None:
        """Value of ``kw['num_chains']`` (or its `Bart` default)."""
        return _bart_default(self.kw, 'num_chains')

    @property
    def n_save(self) -> int:
        """Value of ``kw['n_save']`` (or its `Bart` default)."""
        return _bart_default(self.kw, 'n_save')

    @property
    def ndpost(self) -> int:
        """Total number of posterior samples across chains."""
        return self.n_save * (self.num_chains or 1)

    @property
    def num_trees(self) -> int:
        """Value of ``kw['num_trees']`` (or its `Bart` default)."""
        return _bart_default(self.kw, 'num_trees')

    @property
    def maxdepth(self) -> int:
        """Value of ``kw['maxdepth']`` (or its `Bart` default)."""
        return _bart_default(self.kw, 'maxdepth')

    @property
    def sparse(self) -> bool:
        """Whether ``kw['sparse']`` (or its `Bart` default) enables sparsity."""
        return _bart_default(self.kw, 'sparse').enabled

    @property
    def max_bins(self) -> int:
        """Upper bound on the number of bins implied by `kw['binner']`."""
        binner = _bart_default(self.kw, 'binner')
        if isinstance(binner, partial):
            subcls = binner.func
            partial_kwargs = binner.keywords
        else:
            subcls = binner
            partial_kwargs = {}
        bound = signature(subcls).bind_partial(**partial_kwargs)
        bound.apply_defaults()
        defaults = dict(bound.arguments)
        if subcls is GivenSplitsBinner:
            return defaults['xinfo'].shape[1] + 1
        elif subcls in (UniqueQuantileBinner, RangeEvenBinner):
            return defaults['max_bins']
        else:
            msg = f'Cannot deduce max_bins for binner of type {subcls.__name__}'
            raise NotImplementedError(msg)


def make_kw(key: Key[Array, ''], variant: int) -> BartKW:
    """Return keyword arguments for `Bart` and test predictors."""
    keys = split(key, 10)  # 10 is just some high number
    n = 20
    # `predict` splits new test data like the training data, so `nt` must be a
    # multiple of the `num_data_devices=2` used by some variants (and by the
    # sharding-equivalence tests, which apply it to any variant)
    nt = 22
    p = 2
    high_p = 257  # > 256 to use uint16 for var_trees.
    # num_trees differs from n and nt so that the datapoint, test-datapoint, and
    # tree axes have distinct lengths: an accidental misalignment of these axes
    # then cannot pass by coincidence (e.g. it lets tests locate the datapoint
    # axis by its length, see `test_permutation_invariance`)
    common = dict(num_trees=19, n_save=50, n_burn=50, seed=keys.pop())

    match variant:
        # continuous regression with some settings that induce large types,
        # sparsity with free theta
        case 1:
            train, test = gen_data(
                keys.pop(), n=n + nt, p=p, s_distr=Gamma(2.0), **GEN_KW
            ).split(n)
            bkw = BartKW(
                kw=dict(
                    x_train=train.x,
                    y_train=train.y,
                    outcome_type='continuous',
                    # non-default augment; variant 3 covers the default True
                    sparse=SparseConfig(augment=False),
                    **common,
                    printevery=50,
                    binner=partial(RangeEvenBinner, max_bins=257),
                    # > 256 to use uint16 for X and split_trees
                    num_chains=None,
                    maxdepth=9,  # > 8 to use uint16 for leaf_indices
                    init_kw=dict(
                        resid_reduction_config=BatchedReduction(num_batches=None),
                        count_reduction_config=BatchedReduction(num_batches=None),
                        prec_reduction_config=BatchedReduction(num_batches=None),
                        prec_count_num_trees=5,
                        save_ratios=True,
                        min_points_per_leaf=5,
                    ),
                ),
                x_test=test.x,
            )

        # binary regression with binary X and high p
        case 2:
            gen_kw: kwdict = dict(GEN_KW, q=2)
            train, test = gen_data(
                keys.pop(),
                n=n + nt,
                p=high_p,
                x_distr=DiscreteUniform(2),
                outcome_type='binary',
                **gen_kw,
            ).split(n)
            bkw = BartKW(
                kw=dict(
                    x_train=train.x,
                    y_train=train.y,
                    outcome_type='binary',
                    **common,
                    n_skip=1,  # the mc_gbart default with binary would be 10
                    printevery=None,
                    # quantile-binned, with binary X, to check the case in
                    # which the splits are less than the statically known
                    # maximum; full-sample quantilization for determinism
                    binner=partial(
                        UniqueQuantileBinner, max_bins=256, max_subsample=None
                    ),
                    num_chains=2,
                    maxdepth=6,
                    num_data_devices=min(2, get_device_count()),
                    num_chain_devices=None,  # for mc_gbart, turn autoshard off
                    # on for some variants to cover both trace types
                    precompute_predict_train=True,
                    init_kw=dict(
                        save_ratios=False,
                        min_points_per_decision_node=None,
                        min_points_per_leaf=None,
                    ),
                ),
                x_test=test.x,
            )

        # continuous regression with error scales and sparsity with fixed theta
        case 3:
            train, test = gen_data(
                keys.pop(),
                n=n + nt,
                p=p,
                s_distr=Gamma(2.0),
                het_strength=0.5,
                het_shape='scalar',
                **GEN_KW,
            ).split(n)
            bkw = BartKW(
                kw=dict(
                    x_train=train.x,
                    y_train=train.y,
                    outcome_type='continuous',
                    error_scale=train.error_scale,
                    sparse=SparseConfig(theta=2.0),
                    varprob=jnp.array([0.2, 0.8]),
                    **common,
                    printevery=50,
                    binner=partial(
                        UniqueQuantileBinner, max_bins=11, max_subsample=None
                    ),
                    num_chains=2,
                    num_chain_devices=min(2, get_device_count()),
                    maxdepth=8,  # 8 to check if leaf_indices changes type too soon
                    init_kw=dict(
                        save_ratios=True,
                        resid_reduction_config=BatchedReduction(num_batches=16),
                        count_reduction_config=BatchedReduction(num_batches=16),
                        prec_reduction_config=BatchedReduction(num_batches=16),
                        prec_count_num_trees=7,
                        min_points_per_leaf=5,
                    ),
                ),
                x_test=test.x,
                w_test=test.error_scale,
            )

        # multivariate continuous regression with error scales and some
        # settings that induce large types, sparsity with free theta
        case 4:
            train, test = gen_data(
                keys.pop(),
                n=n + nt,
                p=p,
                k=2,
                lambda_=0.5,
                s_distr=Gamma(2.0),
                het_strength=0.5,
                het_shape='scalar',
                **GEN_KW,
            ).split(n)
            bkw = BartKW(
                kw=dict(
                    x_train=train.x,
                    y_train=train.y,
                    outcome_type='continuous',
                    error_scale=train.error_scale,
                    sparse=SparseConfig(),
                    **common,
                    printevery=50,
                    binner=partial(RangeEvenBinner, max_bins=257),
                    # > 256 to use uint16 for X and split_trees
                    num_chains=None,
                    maxdepth=9,  # > 8 to use uint16 for leaf_indices
                    # request the (already) default device explicitly, to
                    # exercise the explicit-device paths (`Bart._device`)
                    # without changing placement or compilation
                    devices=get_default_device(),
                    precompute_predict_train=True,
                    init_kw=dict(
                        resid_reduction_config=BatchedReduction(num_batches=None),
                        count_reduction_config=BatchedReduction(num_batches=None),
                        prec_reduction_config=BatchedReduction(num_batches=None),
                        prec_count_num_trees=5,
                        save_ratios=True,
                        min_points_per_leaf=5,
                        leaf_dtype=jnp.float32,
                        prec_scale_dtype=jnp.float32,
                    ),
                ),
                x_test=test.x,
                w_test=test.error_scale,
            )

        # multivariate heteroskedastic binary regression with binary X and high
        # p; per-component error scales, to exercise the all-binary 2-D latent-scale paths
        # in `step_z` and the mean squashing of `predict` (the scalar all-binary
        # paths are covered by `test_heteroskedastic_probit`)
        case 5:
            gen_kw: kwdict = dict(GEN_KW, q=2)
            train, test = gen_data(
                keys.pop(),
                n=n + nt,
                p=high_p,
                x_distr=DiscreteUniform(2),
                k=2,
                lambda_=0.5,
                outcome_type='binary',
                het_strength=0.5,
                het_shape='vector',
                **gen_kw,
            ).split(n)
            bkw = BartKW(
                kw=dict(
                    x_train=train.x,
                    y_train=train.y,
                    outcome_type='binary',
                    error_scale=train.error_scale,
                    **common,
                    printevery=None,
                    # quantile-binned with binary X, deterministic via
                    # full-sample quantilization
                    binner=partial(
                        UniqueQuantileBinner, max_bins=256, max_subsample=None
                    ),
                    num_chains=2,
                    maxdepth=6,
                    num_data_devices=min(2, get_device_count()),
                    init_kw=dict(
                        save_ratios=False,
                        min_points_per_decision_node=None,
                        min_points_per_leaf=None,
                    ),
                ),
                x_test=test.x,
                w_test=test.error_scale,
            )

        # multivariate heteroskedastic mixed binary-continuous regression with
        # sparsity with fixed theta and missing subunits
        case 6:
            outcome_type = ('continuous', 'binary', 'binary')
            # p = 3 instead of 2 because gen_data requires p >= k
            train, test = gen_data(
                keys.pop(),
                n=n + nt,
                p=3,
                k=3,
                lambda_=0.5,
                s_distr=Gamma(2.0),
                outcome_type=outcome_type,
                het_strength=0.5,
                het_shape='scalar',
                **GEN_KW,
            ).split(n)
            bkw = BartKW(
                kw=dict(
                    x_train=train.x,
                    y_train=train.y,
                    outcome_type=outcome_type,
                    error_scale=train.error_scale,
                    missing=gen_missing(keys.pop(), (len(outcome_type), n)),
                    # non-default augment; variant 4 covers the default True
                    sparse=SparseConfig(theta=2.0, augment=False),
                    varprob=jnp.array([0.2, 0.3, 0.5]),
                    **common,
                    printevery=50,
                    binner=partial(
                        UniqueQuantileBinner, max_bins=11, max_subsample=None
                    ),
                    num_chains=2,
                    num_chain_devices=min(2, get_device_count()),
                    maxdepth=8,  # 8 to check if leaf_indices changes type too soon
                    precompute_predict_train=True,
                    init_kw=dict(
                        save_ratios=True,
                        resid_reduction_config=BatchedReduction(num_batches=16),
                        count_reduction_config=BatchedReduction(num_batches=16),
                        prec_reduction_config=BatchedReduction(num_batches=16),
                        prec_count_num_trees=7,
                        min_points_per_leaf=5,
                        # narrow leaf storage is fragile across the disparate
                        # scales of mixed outcome components
                        leaf_dtype=jnp.float32,
                        prec_scale_dtype=jnp.float32,
                    ),
                ),
                x_test=test.x,
                w_test=test.error_scale,
            )

        # multivariate continuous regression with per-component error scales
        case 7:  # pragma: no branch
            train, test = gen_data(
                keys.pop(),
                n=n + nt,
                p=p,
                k=2,
                lambda_=0.5,
                s_distr=Gamma(2.0),
                het_strength=0.5,
                het_shape='vector',
                **GEN_KW,
            ).split(n)
            bkw = BartKW(
                kw=dict(
                    x_train=train.x,
                    y_train=train.y,
                    outcome_type='continuous',
                    error_scale=train.error_scale,
                    **common,
                    num_chains=None,
                ),
                x_test=test.x,
                w_test=test.error_scale,
            )

        case _:  # pragma: no cover
            msg = f'Unknown variant {variant}'
            raise ValueError(msg)

    return bkw


# test only the multivariate variants, because the other ones are tested in
# test_BART.py
@pytest.fixture(
    params=(
        pytest.param(4, id='v4'),
        pytest.param(5, id='v5'),
        pytest.param(6, id='v6'),
        pytest.param(7, id='v7'),
    ),
    scope='module',
)
def variant(request: FixtureRequest) -> int:
    """Return a parametrized indicator to select different BART configurations."""
    return request.param


@pytest.fixture
def bkw(keys: split, variant: int) -> BartKW:
    """Return keyword arguments for Bart and test predictors."""
    return make_kw(keys.pop(), variant)


def set_num_datapoints(kw: dict, n: int) -> dict:
    """Set the number of datapoints in the kw dictionary."""
    assert n <= kw['y_train'].shape[-1]
    kw = kw.copy()
    kw['x_train'] = kw['x_train'][:, :n]
    kw['y_train'] = kw['y_train'][..., :n]
    if kw.get('error_scale') is not None:
        kw['error_scale'] = kw['error_scale'][..., :n]
    if kw.get('missing') is not None:
        kw['missing'] = kw['missing'][..., :n]
    return kw


def test_meta_bkw_is_not_shared(bkw: BartKW) -> None:
    """Check that the kw dictionary is not shared across tests."""
    bkw.kw.clear()
    # this will make other tests fail if it's a shared dictionary


@dataclass(frozen=True)
class CachedBart:
    """Pre-computed BART run shared between multiple tests that do not change the arguments."""

    bkw: BartKW
    bart: Bart


class TestWithCachedBart:
    """Group of slow tests that check the same BART run, for efficiency."""

    @pytest.fixture(scope='class')
    def cachedbart(self, variant: int) -> CachedBart:
        """Return a pre-computed Bart."""
        key = random.key(0x139CD0C0)
        keys = random.split(key, 10)  # 10 is just some high number
        key = keys[variant]
        bkw = make_kw(key, variant)
        kw = bkw.kw

        nchains = 4
        kw.update(
            num_trees=max(2 * bkw.n, bkw.p),
            n_burn=2000,
            n_save=2000,
            n_skip=1,
            num_chains=nchains,
            # devices=None because the explicit single device passed by some
            # variants leaves no room for the chain sharding set up
            # automatically for the multiple chains
            devices=None,
        )
        init_kw = dict(kw.get('init_kw', {}))
        init_kw.update(min_points_per_decision_node=10, min_points_per_leaf=5)
        kw['init_kw'] = init_kw

        return CachedBart(bkw=bkw, bart=Bart(**kw))

    def test_residuals_accuracy(self, cachedbart: CachedBart) -> None:
        """Check that running residuals are close to the recomputed final residuals.

        On the float16 variants this also exercises that the running residuals
        are accumulated from the rounded stored leaves, not the exact draws.
        """
        accum_resid, actual_resid = cachedbart.bart._compare_resid()
        assert_close_matrices(accum_resid, actual_resid, rtol=1e-4, reduce_rank=True)

    def test_convergence(self, cachedbart: CachedBart, subtests: SubTests) -> None:
        """Run multiple chains and check convergence with rhat."""
        bart = cachedbart.bart
        bkw = cachedbart.bkw
        num_chains = bkw.num_chains
        nsamples = bkw.n_save

        with subtests.test('yhat_train'):
            yhat_train = bart.predict('train', kind='latent_samples')
            yhat_train_chains = yhat_train.reshape(num_chains, nsamples, -1)
            rhat_yhat_train = rhat_rank(yhat_train_chains, split=True)
            assert_array_less(rhat_yhat_train, 1.15)

        if bkw.all_binary:
            with subtests.test('prob_train'):
                prob_train = bart.predict('train', kind='mean_samples')
                prob_train_chains = prob_train.reshape(num_chains, nsamples, -1)
                rhat_prob_train = rhat_rank(
                    clipped_logit(prob_train_chains, 1e-5), split=True
                )
                assert_array_less(rhat_prob_train, 1.005)

        elif bkw.is_mixed:
            with subtests.test('sigma'):
                # mixed regression: check get_error_sdev, dropping binary
                # components (NaN sdev)
                with debug_nans(False):
                    sigma = bart.get_error_sdev().reshape(num_chains, nsamples, -1)
                    binary_mask = bkw.binary_mask
                    if (
                        binary_mask.ndim > 0
                    ):  # pragma: no branch, always on in mv variants
                        sigma = sigma[:, :, ~binary_mask]
                rhat_sigma = rhat_rank(sigma, split=True)
                assert_array_less(rhat_sigma, 1.05)

        else:
            with subtests.test('error_cov_inv'):
                # all continuous: check full precision matrix convergence
                # using upper triangular elements (matrix is symmetric).
                # When the error covariance is constrained diagonal (2-D
                # inv_sdev_scale), off-diagonal entries are deterministically
                # zero, so rhat is undefined there; check only the diagonal.
                error_cov_inv = chain_to_axis(
                    bart._main_trace.error_cov_inv,
                    chain_vmap_axes(bart._main_trace).error_cov_inv,
                )
                if error_cov_inv.ndim == 2:  # pragma: no cover, only mv by default
                    error_cov_inv = error_cov_inv[:, :, None, None]
                _, _, k, _ = error_cov_inv.shape
                inv_sdev = bart._mcmc_state.inv_sdev_scale
                if inv_sdev is not None and inv_sdev.ndim == 2:
                    ti = tj = jnp.arange(k)
                else:
                    ti, tj = jnp.triu_indices(k)
                error_cov_inv = error_cov_inv[:, :, ti, tj]
                rhat_prec = rhat_rank(error_cov_inv, split=True)
                assert_array_less(rhat_prec, 1.07)

        if bkw.p < bkw.n:
            with subtests.test('varcount'):
                varcount_vals = bart.varcount.reshape(num_chains, nsamples, bkw.p)
                rhat_varcount = rhat_rank(varcount_vals, split=True)
                assert_array_less(rhat_varcount, 1.7)

            if bkw.sparse:  # pragma: no branch
                with subtests.test('varprob'):
                    varprob_vals = bart.varprob.reshape(num_chains, nsamples, bkw.p)
                    rhat_varprob = rhat_rank(varprob_vals[:, :, 1:], split=True)
                    assert_array_less(rhat_varprob, 1.6)

    def test_different_chains(self, cachedbart: CachedBart) -> None:
        """Check that different chains give different results."""
        bart = cachedbart.bart

        step_theta = bart._mcmc_state.forest.rho is not None

        def assert_different(x: PyTree[Array], **kwargs: Any) -> None:
            def assert_different(
                path: KeyPath, x: Shaped[Array, '...'] | None, chain_axis: int | None
            ) -> None:
                str_path = keystr(path)
                if (
                    (str_path.endswith('.theta') and not step_theta)
                    or (
                        # '.error_cov_inv' on the trace, '.…value' on the state;
                        # fixed covariance matrix means all chains are equal
                        str_path.endswith(('.error_cov_inv', '.error_cov_inv.value'))
                        and bart._mcmc_state.error_cov_inv.nu is None
                    )
                    # means over datapoints (and steps): they concentrate (and
                    # round to a power of two), so the chains legitimately
                    # nearly coincide
                    or str_path.endswith(
                        ('.resid_inexact_integral', '.resid_eff_scale')
                    )
                    or (
                        x is not None
                        and jnp.issubdtype(x.dtype, jnp.integer)
                        and x.ndim == 1
                        # too few integers, may collide
                    )
                ):
                    return
                if x is not None and chain_axis is not None:
                    # widen first so a reduced-precision leaf (e.g. float16
                    # leaf_tree) and its mean reference share a dtype
                    x = x.astype(jnp.float32)
                    ref = jnp.broadcast_to(x.mean(chain_axis, keepdims=True), x.shape)
                    assert_different_matrices(
                        x,
                        ref,
                        reduce_rank=True,
                        ord='fro' if x.ndim >= 2 else 2,
                        atol=0,
                        err_msg=f'chain samples are not different for {str_path}\n',
                        **kwargs,
                    )

            axes = chain_vmap_axes(x)
            tree.map_with_path(assert_different, x, axes, is_leaf=lambda x: x is None)

        assert_different(bart._mcmc_state, rtol=0.01)
        assert_different(bart._main_trace, rtol=0.01)
        assert_different(bart._burnin_trace, rtol=0.01)


def test_sequential_guarantee(bkw: BartKW, subtests: SubTests) -> None:
    """Check that the way iterations are saved does not influence the result."""
    kw = bkw.kw
    kw['n_skip'] = 1
    bart1 = Bart(**kw)

    num_chains = bkw.num_chains or 1
    y_shape = kw['y_train'].shape

    # run moving some samples from burn-in to main
    kw2 = dict(kw)
    kw2['seed'] = random.clone(kw2['seed'])
    if bkw.sparse:
        init_kw = dict(kw2.get('init_kw', {}))
        init_kw.setdefault('sparse_on_at', kw2['n_burn'] // 2)
        kw2['init_kw'] = init_kw
    delta = 1
    kw2['n_burn'] -= delta
    kw2['n_save'] += delta
    bart2 = Bart(**kw2)
    bart2_yhat_train = (
        bart2.predict('train', kind='latent_samples')
        .reshape(num_chains, kw2['n_save'], *y_shape)[:, delta:]
        .reshape(num_chains * bkw.n_save, *y_shape)
    )

    with subtests.test('shift burn-in'):
        rtol = (
            0
            if bart1.predict('train', kind='latent_samples').platform() == 'cpu'  # ty: ignore[unresolved-attribute]
            else 2e-6
        )
        assert_close_matrices(
            bart1.predict('train', kind='latent_samples'),
            bart2_yhat_train,
            rtol=rtol,
            reduce_rank=True,
        )

    # run keeping 1 every 2 samples
    kw3 = dict(kw)
    kw3['seed'] = random.clone(kw3['seed'])
    kw3['n_skip'] = 2
    bart3 = Bart(**kw3)
    bart1_yhat_train = bart1.predict('train', kind='latent_samples').reshape(
        num_chains, kw3['n_save'], *y_shape
    )[:, 1::2, :, ...]
    bart3_yhat_train = bart3.predict('train', kind='latent_samples').reshape(
        num_chains, kw3['n_save'], *y_shape
    )[:, : bart1_yhat_train.shape[1], :, ...]

    with subtests.test('change thinning'):
        rtol = (
            0
            if bart1.predict('train', kind='latent_samples').platform() == 'cpu'  # ty: ignore[unresolved-attribute]
            else 2e-6
        )
        assert_close_matrices(
            bart1_yhat_train, bart3_yhat_train, rtol=rtol, reduce_rank=True
        )


def test_tree_structure_changes(bkw: BartKW, subtests: SubTests) -> None:
    """Check how the tree structures change between consecutive MCMC iterations."""
    kw = bkw.kw
    kw['n_skip'] = 1  # so each saved sample is one MCMC step past the previous
    bart = Bart(**kw)

    trace = bart._main_trace
    axes = chain_vmap_axes(trace)

    # move the chain axis (if any) to the front, so that the leading `...`
    # absorbs it uniformly whether or not it is there (CHAIN_AXIS is a global
    # config, so we can't assume where the chain axis sits in the raw layout)
    split_tree = chain_to_axis(trace.split_tree, axes.split_tree)
    grow_acc = chain_to_axis(trace.grow_acc_count, axes.grow_acc_count)
    prune_acc = chain_to_axis(trace.prune_acc_count, axes.prune_acc_count)

    # a tree structure is fully determined by its split tree (split_tree == 0
    # marks inactive nodes); accepted grow/prune moves are exactly the ones that
    # change it, while the optimistically-updated var_tree may differ on rejected
    # grows too, so it must not be compared
    differs = split_tree[..., 1:, :, :] != split_tree[..., :-1, :, :]

    with subtests.test('acceptance count matches changed trees'):
        # number of trees whose structure changed between consecutive iterations
        num_changed = jnp.sum(jnp.any(differs, axis=-1), axis=-1)
        acc_count = grow_acc[..., 1:] + prune_acc[..., 1:]
        assert_array_equal(num_changed, acc_count)

    with subtests.test('at most one changed split node per tree'):
        # a single grow/prune move adds/removes exactly one decision node, so at
        # most one split node changes per tree between consecutive iterations
        num_changed_nodes = jnp.sum(differs, axis=-1)
        assert_array_less(num_changed_nodes, 2)

    with subtests.test('acceptance count does not exceed proposal count'):
        # each tree proposes at most one move (grow or prune) per iteration, so
        # proposals total at most one per tree, and acceptances are a subset
        grow_prop = chain_to_axis(trace.grow_prop_count, axes.grow_prop_count)
        prune_prop = chain_to_axis(trace.prune_prop_count, axes.prune_prop_count)
        *_, num_trees, _ = split_tree.shape
        assert jnp.all(grow_acc <= grow_prop)
        assert jnp.all(prune_acc <= prune_prop)
        assert jnp.all(grow_prop + prune_prop <= num_trees)


def test_multivariate_leaf_prior_covariance(bkw: BartKW) -> None:
    """Multivariate leaf draws must carry the full leaf-prior covariance.

    A strongly informative, correlated leaf prior makes the likelihood
    negligible, so every heap node is resampled essentially from its prior
    ``N(0, leaf_prior_cov)`` each sweep. Nodes that are not actual leaves carry
    zero likelihood precision, hence are drawn exactly from the prior; pooling
    every heap node (over chains, samples, trees, and positions) thus estimates
    ``leaf_prior_cov`` with ~``tree_size`` times more draws than the actual
    leaves alone, at no extra sampling cost.

    This catches sampling the leaf noise with a wrong covariance, e.g. as
    ``z / diag(L)`` instead of ``L^-T z`` (a `solve_triangular` missing
    ``lower=True``), which would zero the off-diagonal correlations.
    """
    if bkw.any_binary or bkw.k is None:
        pytest.skip('only meaningful for all-continuous multivariate outcomes')
    k = bkw.k

    # correlated leaf prior, scaled to a large precision so it dominates the
    # likelihood and the leaves sample (essentially) from the prior
    rho = 0.7
    scale = 1e6
    corr = (1 - rho) * jnp.eye(k) + rho
    leaf_prior_cov_inv = scale * jnp.linalg.inv(corr)
    leaf_prior_cov = corr / scale  # = inv(leaf_prior_cov_inv), before init donates it

    kw = dict(
        bkw.kw,
        init_kw=dict(bkw.kw.get('init_kw', {}), leaf_prior_cov_inv=leaf_prior_cov_inv),
    )
    bart = Bart(**kw)

    # pool every heap node (each an independent prior draw) over chains, samples,
    # trees, and positions; leaf_tree is (..., k, tree_size), stored in
    # prior-sd units, so convert it to data units first
    trace = bart._main_trace
    leaf_tree = trace.leaf_unit[..., None] * trace.leaf_tree
    leaves = jnp.moveaxis(leaf_tree, -2, -1).reshape(-1, k)
    empirical_cov = jnp.cov(leaves.T)

    # the large pool drives the 2-norm sampling error well below 0.01 (measured
    # <0.007); the off-diagonal-zeroing bug instead deviates by ~0.4
    assert_close_matrices(empirical_cov, leaf_prior_cov, rtol=0.02)


def test_check_trees_detects_corruption(bkw: BartKW) -> None:
    """`_check_trees` flags trees corrupted mid-MCMC (guard against false negatives).

    A callback injected through ``run_mcmc_kw`` rewrites every decision node to
    split on an out-of-range variable index after each MCMC step. The corruption
    is deliberately benign for the dynamics: JAX clips the out-of-bounds gather,
    so each tree merely splits on the last predictor instead of the intended
    one, the run stays finite (no NaNs/infs) and completes normally. But the
    saved trees violate the ``check_var_in_bounds`` invariant, which
    `_check_trees` (used by the `Bart` test wrapper's ``__init__``) must catch.
    """

    class Corrupt(Callback):
        def __call__(self, *, state: State, **_: Any) -> tuple[State, 'Corrupt']:
            forest = state.forest
            # max_split.size is the first out-of-range variable index; the var
            # dtype is sized for indices up to max_split.size - 1, so guard that
            # this one extra value still fits (it does whenever the count is not
            # exactly at a dtype boundary, which holds for every variant)
            oob_value = forest.max_split.size
            assert minimal_unsigned_dtype(oob_value) == forest.var_tree.dtype
            oob_var = jnp.array(oob_value, forest.var_tree.dtype)
            var_tree = jnp.where(forest.split_tree != 0, oob_var, forest.var_tree)
            forest = replace(forest, var_tree=var_tree)
            return replace(state, forest=forest), self

    kw: kwdict = dict(bkw.kw, run_mcmc_kw=dict(callback=Corrupt()))

    # build via the unwrapped class so the (finite) run completes without the
    # wrapper's automatic check aborting it, then inspect the corrupted trace
    bart = OriginalBart(**kw)

    bad = bart._check_trees()
    # every decision node now points at an out-of-range variable; trees that
    # happen to be a bare root leaf carry no decision node and stay valid
    assert jnp.any(bad), 'corruption went undetected (false negative)'
    for code in jnp.unique(bad[bad != 0]).tolist():
        assert describe_error(code) == ['check_var_in_bounds']

    # end-to-end: `Bart` (the wrapper used throughout the tests) runs
    # `_check_trees(error=True)` in `__init__`, so the corrupted run must abort
    # construction. Testing the real wrapper, not just `_check_trees`, also
    # catches a wrapper that silently fails to invoke or propagate the check.
    with pytest.raises(RuntimeError, match='invalid trees'):
        Bart(**dict(kw, seed=random.clone(kw['seed'])))


def test_missing_ignored(bkw: BartKW, keys: split) -> None:
    """Check masked datapoints do not affect the mcmc."""
    # unconditionally set missingness
    kw = bkw.kw
    y_train = kw['y_train']
    missing = gen_missing(keys.pop(), y_train.shape)
    kw['missing'] = missing

    # run bart with clean data in the slots marked as missing
    bart1 = Bart(**kw)

    class InjectGarbage(Callback):
        def __call__(
            self, *, key: Key[Array, ''], state: State, **_: Any
        ) -> tuple[State, 'InjectGarbage']:
            """Refill the masked positions of the state with noise after each step."""
            ks = split(key)

            def noisy(
                x: Float[Array, ' *shape'], mask: Bool[Array, ' *mask_shape']
            ) -> Float[Array, ' *shape']:
                # note: don't use nan, and steer clear of overflow, because some
                # masking is done via multiplication by 0
                scale = jnp.finfo(x.dtype).max / 128.0
                noise = scale * random.normal(ks.pop(), x.shape, x.dtype)
                return jnp.where(mask, noise, x)

            return replace(
                state, y=noisy(state.y, missing), resid=noisy(state.resid, missing)
            ), self

    # prepare huge/NaN garbage to inject at the missing data locations
    garbage = random.normal(keys.pop(), y_train.shape) * 1e6
    garbage = jnp.where(
        random.bernoulli(keys.pop(), 0.5, garbage.shape), garbage, jnp.nan
    )

    # run bart with garbage at missing locations
    kw2 = dict(
        kw,
        seed=random.clone(kw['seed']),
        y_train=jnp.where(missing, garbage, y_train),
        run_mcmc_kw=dict(callback=InjectGarbage()),
    )
    bart2 = Bart(**kw2)

    # note: don't use predict('train') because that may rely on `resid` being clean
    yhat1 = bart1.predict(kw['x_train'], kind='latent_samples')
    yhat2 = bart2.predict(kw['x_train'], kind='latent_samples')
    rtol = 0 if yhat1.platform() == 'cpu' else 1e-5  # ty: ignore[unresolved-attribute]
    assert_close_matrices(yhat1, yhat2, rtol=rtol, reduce_rank=True)

    # check resid_inexact_integral separately because it's not upstream of anything
    assert_close_matrices(
        bart1._mcmc_state.resid_inexact_integral,
        bart2._mcmc_state.resid_inexact_integral,
        rtol=rtol,
        reduce_rank=True,
    )


def test_binary_rejects_sigma_settings(bkw: BartKW, subtests: SubTests) -> None:
    """Binary regression rejects `sigma_scale`/`sigma_init`.

    The rejection fires only when every outcome is binary (the outcome type then
    collapses to the scalar ``binary``); skip the variants that would not raise.
    """
    if not bkw.all_binary:
        pytest.skip('rejection only fires for all-binary outcomes')
    for param in ('sigma_scale', 'sigma_init'):
        with (
            subtests.test(param=param),
            pytest.raises(ValueError, match='Do not set `sigma_scale` or `sigma_init`'),
        ):
            Bart(**dict(bkw.kw, **{param: 1.0}))


def test_outcome_samples_error_scale(keys: split) -> None:
    """`outcome_samples` applies `error_scale` in univariate continuous regression.

    The univariate error-scaled branch of `sample_outcome` is reachable only
    here: the BART3 wrappers expose no `outcome_samples`, and the `bkw` fixture
    is multivariate-only, so this uses the univariate variant directly.
    """
    bkw = make_kw(keys.pop(), 3)  # univariate continuous with error scales
    bart = Bart(**bkw.kw)
    out = bart.predict(
        bkw.x_test, kind='outcome_samples', error_scale=bkw.w_test, key=keys.pop()
    )
    assert jnp.all(jnp.isfinite(out))


def _assert_predictions_finite(bart: Bart, bkw: BartKW, keys: split) -> None:
    """Assert every prediction kind is finite, on train and test predictors."""
    for x_test, on_train in (('train', True), (bkw.x_test, False)):
        for kind in ('mean', 'mean_samples', 'latent_samples'):
            w = bkw.w_for_predict(kind, on_train=on_train)
            pred = bart.predict(x_test, kind=kind, error_scale=w)
            assert jnp.all(jnp.isfinite(pred)), (x_test, kind)
        w = bkw.w_for_predict('outcome_samples', on_train=on_train)
        out = bart.predict(
            x_test, kind='outcome_samples', error_scale=w, key=keys.pop()
        )
        assert jnp.all(jnp.isfinite(out)), (x_test, 'outcome_samples')
    # error sdev is NaN by design for binary components, so only check it
    # when there are none
    if not bkw.any_binary:
        assert jnp.all(jnp.isfinite(bart.get_error_sdev()))


def test_constant_y_train(bkw: BartKW, keys: split) -> None:
    """Behavior when `y_train` is completely constant.

    A constant response has zero sample variance, but the error-precision prior
    guards against it (``var(y_train)`` is replaced by 1 where zero), so the
    defaults yield a well-behaved MCMC for any outcome type. The leaf scale
    default ``tau_num = (max - min) / 2 = 0`` is left in place to confirm it is
    *not* fatal: it merely pins the leaves to zero.
    """
    kw = dict(bkw.kw)
    # a single constant value: continuous components are flat, binary ones are
    # all-ones (so the all-zero/all-one offset clipping kicks in)
    kw['y_train'] = jnp.full_like(kw['y_train'], 1.0)

    bart = Bart(**kw)
    _assert_predictions_finite(bart, bkw, keys)
    if not bkw.all_binary:
        # tau_num kept at its degenerate default (0): the infinite leaf-prior
        # precision pins every leaf to exactly zero
        leaf_tree = bart._mcmc_state.forest.leaf_tree
        assert_array_equal(leaf_tree, jnp.zeros_like(leaf_tree))


def test_constant_predictor(bkw: BartKW, subtests: SubTests) -> None:
    """A constant predictor has no available cutpoints.

    With ``rm_const=False`` the splitless predictor triggers an error; with
    ``rm_const=True`` (default) predictor 0 is ignored, which must show up in
    every downstream signal: it is splitless, listed in ``blocked_vars``, never
    counted in `varcount`, assigned zero `varprob`, and never selected for a
    decision rule in the `var_tree` of the main trace.
    """
    kw = dict(bkw.kw)

    # make predictor 0 constant so it has no available cutpoints; force a
    # quantile binner because e.g. RangeEvenBinner would still place cutpoints
    # over the degenerate range, leaving the predictor formally splittable
    x = kw['x_train']
    kw['x_train'] = x.at[0, :].set(x[0, 0])
    kw['binner'] = partial(UniqueQuantileBinner, max_subsample=None)

    bart = Bart(**kw)

    forest = bart._mcmc_state.forest
    with subtests.test('predictor 0 is splitless'):
        assert forest.max_split[0] == 0

    with subtests.test('blocked_vars lists predictor 0'):
        (expected,) = jnp.nonzero(forest.max_split == 0)
        assert forest.blocked_vars is not None
        assert_array_equal(jnp.sort(forest.blocked_vars), expected, strict=False)
        assert jnp.any(forest.blocked_vars == 0)

    with subtests.test('varcount ignores predictor 0'):
        assert jnp.all(bart.varcount[:, 0] == 0)
        assert jnp.any(bart.varcount[:, 1:] > 0)  # the test is not vacuous

    with subtests.test('varprob ignores predictor 0'):
        assert jnp.all(bart.varprob[:, 0] == 0)

    with subtests.test('predictor 0 absent from var_tree'):
        trace = bart._main_trace
        axes = chain_vmap_axes(trace)
        var_tree = chain_to_axis(trace.var_tree, axes.var_tree)
        split_tree = chain_to_axis(trace.split_tree, axes.split_tree)
        # only active decision nodes carry a meaningful variable; predictor 0
        # (var index 0) must never appear among them
        active = split_tree > 0
        assert jnp.any(active)  # the test is not vacuous
        assert jnp.all(var_tree[active] != 0)

    with (
        subtests.test('rm_const=False raises'),
        pytest.raises(EquinoxRuntimeError, match='predictors with no splits'),
    ):
        Bart(**dict(kw, rm_const=False, seed=random.clone(kw['seed'])))


def test_output_shapes(bkw: BartKW, keys: split) -> None:
    """Check the output shapes of the Bart predictions and attributes."""
    kw = bkw.kw
    bart = Bart(**kw)

    ndpost = bkw.ndpost
    _, m = bkw.x_test.shape
    k = bkw.kshape

    assert bart.offset.shape == k

    assert bart.predict('train', kind='mean_samples').shape == (ndpost, *k, bkw.n)
    assert bart.predict('train', kind='mean').shape == (*k, bkw.n)
    assert bart.predict('train', kind='latent_samples').shape == (ndpost, *k, bkw.n)
    assert bart.predict('train', kind='outcome_samples', key=keys.pop()).shape == (
        ndpost,
        *k,
        bkw.n,
    )

    mean_w = bkw.w_for_predict('mean', on_train=False)
    assert bart.predict(bkw.x_test, kind='mean_samples', error_scale=mean_w).shape == (
        ndpost,
        *k,
        m,
    )
    assert bart.predict(bkw.x_test, kind='mean', error_scale=mean_w).shape == (*k, m)
    assert bart.predict(bkw.x_test, kind='latent_samples').shape == (ndpost, *k, m)
    assert bart.predict(
        bkw.x_test, kind='outcome_samples', error_scale=bkw.w_test, key=keys.pop()
    ).shape == (ndpost, *k, m)

    with debug_nans(False):
        assert bart.get_error_sdev().shape == (ndpost, *k)
        assert bart.get_error_sdev(mean=True).shape == k

    assert bart.varcount.shape == (ndpost, bkw.p)
    assert bart.varcount_mean.shape == (bkw.p,)
    assert bart.varprob.shape == (ndpost, bkw.p)
    assert bart.varprob_mean.shape == (bkw.p,)

    # get_latent_prec shape
    n_burn = kw['n_burn']
    prec = bart.get_latent_prec()
    if bkw.num_chains is not None:
        assert prec.shape == (bkw.num_chains, n_burn + bkw.n_save, *k, *k)
    else:
        assert prec.shape == (n_burn + bkw.n_save, *k, *k)


def test_output_types(bkw: BartKW, keys: split) -> None:
    """Check the output types of all the attributes of Bart."""
    kw = bkw.kw
    bart = Bart(**kw)

    assert bart.offset.dtype == jnp.float32
    assert isinstance(bart.n_save, int)
    assert bart.predict('train', kind='mean').dtype == jnp.float32
    assert bart.predict('train', kind='mean_samples').dtype == jnp.float32
    assert bart.predict('train', kind='latent_samples').dtype == jnp.float32
    assert (
        bart.predict('train', kind='outcome_samples', key=keys.pop()).dtype
        == jnp.float32
    )
    assert bart.get_latent_prec().dtype == jnp.float32
    with debug_nans(False):
        assert bart.get_error_sdev().dtype == jnp.float32
        assert bart.get_error_sdev(mean=True).dtype == jnp.float32
    assert bart.varcount.dtype == jnp.int32
    assert bart.varcount_mean.dtype == jnp.float32
    assert bart.varprob.dtype == jnp.float32
    assert bart.varprob_mean.dtype == jnp.float32

    # also test the internal leaf dtype directly, since it's not reflected in the
    # public API anywhere
    assert bart._main_trace.leaf_tree.dtype == _init_default(bkw.kw, 'leaf_dtype')

    # likewise the internal prec_scale and inv_sdev_scale dtypes, which share
    # `prec_scale_dtype` (both only set for error-scaled/missing data)
    prec_scale = bart._mcmc_state.prec_scale
    if prec_scale is not None:
        assert prec_scale.dtype == _init_default(bkw.kw, 'prec_scale_dtype')
    inv_sdev_scale = bart._mcmc_state.inv_sdev_scale
    if inv_sdev_scale is not None:
        assert inv_sdev_scale.dtype == _init_default(bkw.kw, 'prec_scale_dtype')


def test_leaf_unit(bkw: BartKW) -> None:
    """`leaf_unit` is the marginal prior leaf standard deviation, rounded to a power of two."""
    forest = Bart(**bkw.kw)._mcmc_state.forest
    cov_inv = nnone(forest.leaf_prior_cov_inv)
    if cov_inv.ndim:
        marginal_std = jnp.sqrt(jnp.diagonal(jnp.linalg.inv(cov_inv)))
    else:  # pragma: no cover, always mv with defaults
        marginal_std = jnp.sqrt(jnp.reciprocal(cov_inv))
    expected = 2 ** jnp.round(jnp.log2(marginal_std))
    assert forest.leaf_unit.dtype == jnp.float32
    assert_close_matrices(forest.leaf_unit, expected, rtol=1e-5)
    # exactly a power of two: rounding the log2 back is a no-op
    pow2 = 2 ** jnp.round(jnp.log2(forest.leaf_unit))
    assert_array_equal(forest.leaf_unit, pow2)


def test_inv_sdev_unit(bkw: BartKW) -> None:
    """`inv_sdev_unit` is the rms of the reciprocal error scales, rounded to a power of two."""
    state = Bart(**bkw.kw)._mcmc_state

    assert state.inv_sdev_unit.dtype == jnp.float32

    # exactly a power of two: rounding the log2 back is a no-op
    pow2 = 2 ** jnp.round(jnp.log2(state.inv_sdev_unit))
    assert_array_equal(state.inv_sdev_unit, pow2)

    if state.inv_sdev_scale is None:
        assert_array_equal(state.inv_sdev_unit, jnp.ones(()))
    else:
        stored = state.inv_sdev_scale.astype(jnp.float32)
        rtol = condf(state.inv_sdev_scale, 1e-6, 1e-3)

        # the unit times the stored values reconstructs the reciprocal error scales,
        # zeroed at masked datapoints
        inv_sdev = state.inv_sdev_unit[..., None] * stored
        scale = jnp.ones(()) if state.error_scale is None else state.error_scale
        expected = jnp.where(stored != 0, jnp.reciprocal(scale), 0.0)
        expected = jnp.broadcast_to(expected, inv_sdev.shape)
        assert_close_matrices(inv_sdev, expected, rtol=rtol, reduce_rank=True)

        # the unit is the rms of the reciprocal error scales over non-missing
        # datapoints, so the stored mean square lands within the pow2 bracket
        ms = jnp.sum(jnp.square(stored), axis=-1)
        ms = ms / jnp.maximum(jnp.sum(stored != 0, axis=-1), 1)
        assert jnp.all((ms > 0.5 * (1 - 1e-3)) & (ms < 2.0 * (1 + 1e-3)))

        # prec_scale is the square (per-component outer product) of the stored
        # reciprocal error scales, in the same units
        prec_scale = nnone(state.prec_scale).astype(jnp.float32)
        if prec_scale.ndim == stored.ndim:
            expected_prec = jnp.square(stored)
        else:
            expected_prec = jnp.einsum('an,bn->abn', stored, stored)
        assert_close_matrices(
            prec_scale,
            expected_prec,
            rtol=condf(prec_scale, 1e-6, 1e-3),
            reduce_rank=True,
        )


def test_float16_close_to_float32(bkw: BartKW) -> None:
    """float16 leaf storage tracks float32 within float16 precision.

    A single MCMC step is taken with the same seed for both dtypes, so the tree
    draws are identical and the only difference is the rounding of the leaves.
    """
    prev_init_kw: kwdict = bkw.kw.get('init_kw', {})
    base_kw = dict(bkw.kw, n_burn=0, n_save=1)
    latent = {}
    for dtype in (jnp.float16, jnp.float32):
        init_kw = dict(prev_init_kw, leaf_dtype=dtype)
        bart = Bart(**dict(base_kw, init_kw=init_kw))
        latent[dtype] = bart.predict('train', kind='latent_samples')
    assert_close_matrices(
        latent[jnp.float16], latent[jnp.float32], rtol=1e-2, reduce_rank=True
    )


def test_leaf_quantization(bkw: BartKW, keys: split) -> None:
    """With `leaf_quantization`, leaves are stored as multiples of the quantum."""
    leaf_quantization = 3
    bart = Bart(
        **dict(
            bkw.kw,
            init_kw=dict(
                bkw.kw.get('init_kw', {}), leaf_quantization=leaf_quantization
            ),
        )
    )
    state = bart._mcmc_state
    nmant = jnp.finfo(state.resid.dtype).nmant

    # the grid follows the residual scale measured before the leaves are
    # sampled, so compute the quantum from the pre-step state (also because
    # `step` donates it), step once, and check the freshly sampled leaves
    quantum = (state.resid_eff_scale / state.forest.leaf_unit) * 2.0 ** (
        leaf_quantization - nmant
    )
    state = step(keys.pop(), state)

    # align the (*chains, [k]) quantum with the (*chains, num_trees, [k,]
    # tree_size) leaves
    if state.resid_unit.ndim:
        quantum = quantum[..., None, :, None]
    else:  # pragma: no cover, always mv with defaults
        quantum = quantum[..., None, None]

    leaves = state.forest.leaf_tree / quantum
    assert_array_equal(leaves, jnp.round(leaves))

    # minimal check of sum_trees_eps()
    eps = state.sum_trees_eps()
    assert eps.shape == bkw.kshape
    assert jnp.all(jnp.isfinite(eps))
    assert_array_less(0, eps)


def test_resid_eff_scale(bkw: BartKW) -> None:
    """`resid_eff_scale` is the residual RMS in data units, rounded to a power of two."""
    state = Bart(**bkw.kw)._mcmc_state

    # exactly a power of two: rounding the log2 back is a no-op
    pow2 = 2 ** jnp.round(jnp.log2(state.resid_eff_scale))
    assert_array_equal(state.resid_eff_scale, pow2)

    if state.error_cov_inv.nu is None:
        # never stepped without an error covariance update: stays at its
        # initial value `resid_unit`
        expected = jnp.broadcast_to(state.resid_unit, state.resid_eff_scale.shape)
        assert_array_equal(state.resid_eff_scale, expected)
    else:
        # the last update measured the residuals as they are in the final
        # state, so recompute their (precision-scaled, masked) rms
        resid = state.resid * state.resid_unit[..., None]
        if state.inv_sdev_scale is None:  # pragma: no cover, error-scaled by default
            n_eff = resid.shape[-1]
        else:
            # `inv_sdev_scale` is stored in units of `inv_sdev_unit`
            resid = resid * state.inv_sdev_scale * state.inv_sdev_unit[..., None]
            n_eff = jnp.sum(state.inv_sdev_scale != 0, axis=-1)
        rms = jnp.sqrt(jnp.sum(jnp.square(resid), axis=-1) / n_eff)
        # `resid_eff_scale` is this rms rounded to a power of two, but
        # reproducing the step's exact float32 reduction (narrow-dtype
        # accumulation, the per-path scaling and masking) bit-for-bit is
        # brittle, and pow2 rounding turns a 1-ulp difference near a boundary
        # into a full factor of two. so check agreement loosely: pow2 rounding
        # moves the value by at most sqrt(2), which rtol=0.5 covers
        assert_close_matrices(state.resid_eff_scale, rms, rtol=0.5)


# without quantization the drift charge counts every tree update, but in a
# converged chain many updates are exact (the resampled leaves store
# identically), so that case is graded on a wider bracket
@pytest.mark.parametrize(
    ('num_trees', 'leaf_quantization', 'bound'),
    [(16, 1, 8), (256, 1, 8), (16, None, 16)],
)
def test_sum_trees_eps(
    num_trees: int, leaf_quantization: int | None, bound: int, keys: split
) -> None:
    """`State.sum_trees_eps` brackets the observed accuracy of the sum of trees.

    Fit an exactly representable function (linear in binary covariates, split
    at 0.5) with noise low enough that the error floor set by the float16
    storage dominates, and check the accuracy estimate against the larger of
    that floor and the drift of the running residuals (the difference between
    the precomputed train predictions and the exact ones recomputed from the
    saved trees).
    """
    n, p, sigma = 300, 5, 1e-4
    x = random.bernoulli(keys.pop(), 0.5, (p, n)).astype(jnp.float32)
    coef = random.normal(keys.pop(), (p,))
    f = coef @ x
    y = f + sigma * random.normal(keys.pop(), (n,))
    bart = Bart(
        x,
        y,
        num_trees=num_trees,
        n_burn=2000,
        n_save=100,
        num_chains=None,
        seed=keys.pop(),
        sigma_scale=sigma,
        precompute_predict_train=True,
        init_kw=dict(
            leaf_dtype=jnp.float16,
            resid_dtype=jnp.float16,
            leaf_quantization=leaf_quantization,
        ),
    )
    pred = bart.predict(x, kind='latent_samples')
    pred_running = bart.predict('train', kind='latent_samples')
    err = jnp.sqrt(jnp.mean(jnp.square(pred - f)))
    err_drift = jnp.sqrt(jnp.mean(jnp.square(pred_running[-1, :] - pred[-1, :])))
    eps = bart._mcmc_state.sum_trees_eps()
    # the estimate sits at ~3x the observed rms error across configs (the eps
    # is a grid spacing, the error floor lands at ~spacing/sqrt(12))
    assert err <= eps <= bound * jnp.maximum(err, err_drift)


@pytest.mark.parametrize('leaf_quantization', [None, 0])
def test_sum_trees_eps_drift(leaf_quantization: int | None, keys: split) -> None:
    """The drift term of `State.sum_trees_eps` tracks the running residuals error.

    Run on noisy data with float16 residuals and train-prediction
    precomputation, then compare the drift term against the accumulated
    numerical error of the running residuals, observed as the difference
    between the precomputed train predictions and the exact ones recomputed
    from the saved trees. Without quantization every residual update rounds,
    so the drift term dominates and bounds the observed error; with
    quantization most updates are exact and the drift term must not blow up
    above the observed error.
    """
    dgp = gen_data(
        keys.pop(), n=500, p=10, q=0, sigma2_lin=1.0, sigma2_quad=1.0, sigma2_eps=1.0
    )
    bart = Bart(
        dgp.x,
        dgp.y,
        num_trees=16,
        n_burn=0,
        n_save=2000,
        num_chains=None,
        seed=keys.pop(),
        precompute_predict_train=True,
        init_kw=dict(
            leaf_dtype=jnp.float16,
            resid_dtype=jnp.float16,
            leaf_quantization=leaf_quantization,
        ),
    )
    pred_running = bart.predict('train', kind='latent_samples')
    pred_actual = bart.predict(dgp.x, kind='latent_samples')
    err = jnp.sqrt(jnp.mean(jnp.square(pred_running[-1, :] - pred_actual[-1, :])))
    resolution, drift, _ = bart._mcmc_state._sum_trees_eps()
    # the drift estimate is conservative, but must stay within a bounded
    # factor of the observed accumulated error
    assert drift <= 16 * err
    # the overall estimate must cover the observed accumulated error (the
    # slack absorbs the slow sub-resolution error growth the snapshot-based
    # estimate cannot track)
    assert err <= 4 * jnp.maximum(resolution, drift)
    if leaf_quantization is None:
        # every residual update rounds: the drift term dominates and is an
        # upper bound on the observed error
        assert resolution < drift
        assert err <= drift


def test_sum_trees_eps_snap(keys: split) -> None:
    """The snap term of `State.sum_trees_eps` covers coarse-quantization distortion.

    With many trees the leaf full conditional shrinks below the (fixed) leaf
    quantum: leaf sampling degenerates into snapping the full conditional mean
    to the grid, most leaves pin at zero, and the fit degrades statistically
    while the stored sums stay numerically exact. The resolution and drift
    terms are blind to this; check that the snap term flags it by covering
    the observed error of the posterior mean.
    """
    n = 250
    dgp = gen_data(
        keys.pop(), n=n, p=5, q=0, sigma2_lin=1.0, sigma2_quad=1.0, sigma2_eps=1.0
    )
    bart = Bart(
        dgp.x,
        dgp.y,
        num_trees=12800,
        n_burn=400,
        n_save=200,
        num_chains=None,
        seed=keys.pop(),
        init_kw=dict(
            leaf_dtype=jnp.float16, resid_dtype=jnp.float16, leaf_quantization=6
        ),
    )
    post_mean = bart.predict(dgp.x, kind='latent_samples').mean(axis=0)
    err = jnp.sqrt(jnp.mean(jnp.square(post_mean - dgp.mu)))
    resolution, drift, snap = bart._mcmc_state._sum_trees_eps()
    # the numerical terms alone under-report the error
    assert jnp.maximum(resolution, drift) < err
    # the snap term covers it, within a bounded factor. the quantum scales
    # with the measured residual scale, and coarse quantization is what
    # inflates the residuals in the first place (leaves snap to the grid, the
    # fit degrades), so distortion -> larger residuals -> larger quantum ->
    # more distortion, until the loop settles. at that equilibrium snap and
    # the observed error track each other up to a constant
    assert err <= 1.5 * snap <= 12 * err
    assert_array_equal(bart._mcmc_state.sum_trees_eps(), snap)


def test_output_ranges(bkw: BartKW, keys: split) -> None:
    """Check value constraints on Bart outputs."""
    kw = bkw.kw
    bart = Bart(**kw)

    binary_mask = bkw.binary_mask

    mean = bart.predict('train', kind='mean')
    bmean = mean[binary_mask, ...]
    assert jnp.all((bmean >= 0) & (bmean <= 1))

    outcomes = bart.predict('train', kind='outcome_samples', key=keys.pop())
    boutcomes = outcomes[:, binary_mask, ...]
    assert jnp.all((boutcomes == 0) | (boutcomes == 1))

    with debug_nans(False):
        sdev = bart.get_error_sdev()
        assert jnp.all(jnp.isnan(sdev[..., binary_mask]))
        assert jnp.all(jnp.isfinite(sdev[..., ~binary_mask]))
        assert jnp.all(sdev[..., ~binary_mask] > 0)

        sdev_mean = bart.get_error_sdev(mean=True)
        assert jnp.all(jnp.isnan(sdev_mean[binary_mask]))
        assert jnp.all(jnp.isfinite(sdev_mean[~binary_mask]))
        assert jnp.all(sdev_mean[~binary_mask] > 0)

    # get_latent_prec: symmetry and positive definiteness
    if bkw.k is not None:  # pragma: no branch, always mv with defaults
        prec = bart.get_latent_prec()
        assert_close_matrices(prec, prec.mT, rtol=1e-6, reduce_rank=True)
        eigvals = jnp.linalg.eigvalsh(prec)
        assert jnp.all(eigvals > 0)


def test_predict_means(bkw: BartKW, keys: split, subtests: SubTests) -> None:
    """Check that the various outputs of `Bart.predict` are consistent."""
    bart = Bart(**bkw.kw)

    mean = bart.predict('train', kind='mean')  # (*k, n)
    mean_samples = bart.predict('train', kind='mean_samples')  # (ndpost, *k, n)
    latent_samples = bart.predict('train', kind='latent_samples')  # (ndpost, *k, n)
    outcome_samples = bart.predict('train', kind='outcome_samples', key=keys.pop())
    # outcome_samples has shape (num_chains*n_save, *k, n)

    with subtests.test('mean_samples vs mean'):
        mean_from_mean_samples = mean_samples.mean(0)
        assert_close_matrices(mean, mean_from_mean_samples, rtol=1e-7)

    with subtests.test('latent_samples vs mean_samples'):
        binary_mask = bkw.binary_mask
        cont_mean_samples = mean_samples[:, ~binary_mask, ...]
        cont_latent_samples = latent_samples[:, ~binary_mask, ...]
        assert_close_matrices(
            cont_mean_samples.T, cont_latent_samples.T, reduce_rank=True
        )

    with subtests.test('outcome_samples vs mean'):
        mean_from_os = outcome_samples.mean(0)
        sdev = outcome_samples.std(0) / jnp.sqrt(outcome_samples.shape[0])
        t = jnp.abs(mean - mean_from_os) / sdev
        assert jnp.all(t < 5)

    with subtests.test('binary mean uses the latent scale'):
        # for binary components P(y=1) = Phi(latent / w); continuous ones are
        # the plain latent mean (error scales do not enter their mean)
        w = bkw.kw.get('error_scale')
        arg = latent_samples if w is None else latent_samples / w
        expected = jnp.where(
            bkw.binary_mask[..., None], ndtr(arg), latent_samples
        ).mean(0)
        assert_close_matrices(mean, expected, rtol=1e-5)


@pytest.mark.parametrize(
    ('outcome_type', 'k', 'het_shape'),
    [
        # univariate pure-binary with a scalar error scale, and mixed
        # binary-continuous with a per-component error scale. Together with
        # `bkw` variants v5 (all-binary, vector) and v6 (mixed, scalar) these
        # complete the {pure, mixed} x {scalar, vector} shape matrix; the
        # mixed-vector case is the only one that exercises the 2-D binary-row
        # restriction in `step_z`.
        pytest.param('binary', None, 'scalar', id='binary-scalar'),
        pytest.param(('continuous', 'binary'), 2, 'vector', id='mixed-vector'),
    ],
)
def test_heteroskedastic_probit(
    keys: split,
    subtests: SubTests,
    outcome_type: str | tuple[str, ...],
    k: int | None,
    het_shape: Literal['scalar', 'vector'],
) -> None:
    """`error_scale` scales the probit latent error, so P(y=1) = Phi(latent / w).

    `test_predict_means` and `test_output_ranges` already cover the mean-squash
    relationship (for the other two error-scale-shape combinations), the outcome
    thresholding and the outcome-vs-mean consistency via the `bkw` fixture; this
    adds the two combinations `bkw` lacks and the new-test-data error-scale
    checks.
    """
    n = 20
    train, test = gen_data(
        keys.pop(),
        n=2 * n,
        p=3,
        k=k,
        lambda_=None if k is None else 0.5,
        outcome_type=outcome_type,
        het_strength=0.5,
        het_shape=het_shape,
        **GEN_KW,
    ).split(n)
    w_train, w_test = train.error_scale, test.error_scale
    assert w_train is not None
    assert w_test is not None
    bart = Bart(
        x_train=train.x,
        y_train=train.y,
        outcome_type=outcome_type,
        error_scale=w_train,
        num_trees=19,
        n_save=50,
        n_burn=50,
        seed=keys.pop(),
        printevery=None,
    )

    latent = bart.predict('train', kind='latent_samples')
    mean = bart.predict('train', kind='mean')

    with subtests.test('binary mean uses the latent scale'):
        # binary components: Phi(latent / w); continuous ones: untouched latent
        if isinstance(outcome_type, tuple):
            binary_mask = jnp.array([t == 'binary' for t in outcome_type])[:, None]
        else:
            binary_mask = jnp.array(True)
        expected = jnp.where(binary_mask, ndtr(latent / w_train), latent).mean(0)
        assert_close_matrices(mean, expected, rtol=1e-5)

    with subtests.test('error scales required on new test data'):
        msg = 'required because the model was fit with'
        for kind in ('mean', 'mean_samples', 'outcome_samples'):
            extra: dict = {'key': keys.pop()} if kind == 'outcome_samples' else {}
            with pytest.raises(ValueError, match=msg):
                bart.predict(test.x, kind=kind, **extra)
            pred = bart.predict(test.x, kind=kind, error_scale=w_test, **extra)
            assert jnp.all(jnp.isfinite(pred))

    if w_test.ndim == 2:
        with subtests.test('test error scales must match the component dimension'):
            # a 2-D test error scale with the wrong number of components is rejected
            bad = jnp.ones((w_test.shape[0] + 1, w_test.shape[1]))
            with pytest.raises(
                ValueError, match='shape mismatch with the training error scales'
            ):
                bart.predict(test.x, kind='mean', error_scale=bad)


def test_predict(bkw: BartKW) -> None:
    """Check that predict with x_train matches predict with 'train'."""
    kw = bkw.kw
    bart = Bart(**kw)
    yhat_train = bart.predict(kw['x_train'], kind='latent_samples')
    assert_close_matrices(
        bart.predict('train', kind='latent_samples'),
        yhat_train,
        rtol=1e-5,
        reduce_rank=True,
    )


def test_precompute_predict_train(bkw: BartKW, subtests: SubTests) -> None:
    """`precompute_predict_train` gives the same train predictions, up to rounding."""
    bart_off = Bart(**dict(bkw.kw, precompute_predict_train=False))
    bart_on = Bart(**dict(bkw.kw, precompute_predict_train=True))

    with subtests.test('trace type'):
        assert not isinstance(bart_off._main_trace, MainTraceWithTrainPred)
        assert isinstance(bart_on._main_trace, MainTraceWithTrainPred)

    # `outcome_samples` is left out: for binary outcomes a tiny latent difference
    # can flip a thresholded 0/1 draw, which is not an approximate mismatch
    for kind in ('latent_samples', 'mean', 'mean_samples'):
        with subtests.test('predict', kind=kind):
            assert_close_matrices(
                bart_on.predict('train', kind=kind),
                bart_off.predict('train', kind=kind),
                rtol=1e-5,
                reduce_rank=True,
            )


def test_count_prec_tree_caches_valid(bkw: BartKW, subtests: SubTests) -> None:
    """The cached `count_tree`/`prec_tree` stay valid at the actual leaves.

    `step` updates the cached per-leaf counts and precision sums only at the
    nodes involved in the moves; after the MCMC run, at every actual leaf,
    they must match a from-scratch reduction over the datapoints (entries
    elsewhere may be dirty).
    """
    bart = Bart(**bkw.kw)
    state = bart._mcmc_state
    forest = state.forest

    with subtests.test('presence'):
        # which caches exist follows the configuration
        assert (forest.count_tree is not None) == (
            state.prec_scale is None
            or forest.min_points_per_decision_node is not None
            or forest.min_points_per_leaf is not None
        )
        assert (forest.prec_tree is not None) == (state.prec_scale is not None)

    # flatten the chain axis (if any) into the tree axis. `leaf_indices` carries
    # its chain axis after the tree axis (unlike `split_tree`), so move it to the
    # front first, to match `split_tree`'s chain-leading flattening order
    _, n = state.X.shape
    half = forest.split_tree.shape[-1]
    split_tree = forest.split_tree.reshape(-1, half)
    leaf_indices = chain_to_axis(
        forest.leaf_indices, chain_vmap_axes(forest).leaf_indices
    ).reshape(-1, n)

    # mask of the actual leaves over the full heap, per tree
    leaf_mask = vmap(partial(is_actual_leaf, add_bottom_level=True))(split_tree)
    # sanity: the trees do grow beyond the root (else the check is vacuous)
    assert jnp.any(leaf_mask[..., 2:])

    def check_cache(
        cache: Shaped[Array, '#chains_x_trees *k_k 2*half'],
        values: int | Float32[Array, ' n'] | Float32[Array, 'k k n'],
        comparison: Callable[..., None],
    ) -> None:
        # the fresh value is a from-scratch reduction over the datapoints, the
        # same baseline `TestReduction` checks the optimized reductions against
        *_, tree_size = cache.shape
        core = jnp.asarray(values).shape[:-1]
        cache = cache.reshape(-1, *core, tree_size)
        fresh = vmap(
            partial(
                reduce_reference,
                values,
                size=tree_size,
                dtype=cache.dtype,
                data_sharded=False,
            )
        )(leaf_indices)
        mask = leaf_mask.reshape(-1, *(1,) * len(core), tree_size)
        comparison(jnp.where(mask, cache, 0), jnp.where(mask, fresh, 0))

    with subtests.test('values'):
        if forest.count_tree is not None:
            check_cache(forest.count_tree, 1, assert_array_equal)
        if forest.prec_tree is not None:
            assert state.prec_scale is not None
            check_cache(
                forest.prec_tree,
                state.prec_scale,
                partial(assert_close_matrices, rtol=1e-5, reduce_rank=True),
            )


class TestVarprobAttr:
    """Test the `Bart.varprob` attribute."""

    def test_basic_properties(self, bkw: BartKW) -> None:
        """Basic checks of the `varprob` attribute."""
        kw = bkw.kw
        bart = Bart(**kw)

        assert jnp.all(bart.varprob >= 0)
        assert jnp.all(bart.varprob <= 1)
        varprob_sum = bart.varprob.sum(axis=1)
        assert_close_matrices(varprob_sum, jnp.ones_like(varprob_sum), rtol=1e-6)

        if not bkw.sparse:
            unique = jnp.unique(bart.varprob)
            assert unique.size in (1, 2)
            if unique.size == 2:  # pragma: no cover
                assert unique[0] == 0

        assert_array_equal(bart.varprob_mean, bart.varprob.mean(axis=0))

    def test_blocked_vars(self, keys: split) -> None:
        """Check that varprob = 0 on predictors blocked a priori."""
        dgp = gen_data(keys.pop(), n=30, p=2, **GEN_KW)
        with debug_nans(False):
            xinfo = jnp.array([[jnp.nan], [0]])
        bart = Bart(
            x_train=dgp.x,
            y_train=dgp.y,
            binner=partial(GivenSplitsBinner, xinfo=xinfo),
            seed=keys.pop(),
        )
        assert_array_equal(bart._mcmc_state.forest.max_split, [0, 1], strict=False)
        assert_array_equal(bart.varprob_mean, [0, 1], strict=False)
        assert jnp.all(bart.varprob_mean == bart.varprob)


@pytest.mark.parametrize(
    ('theta', 'augment'), [('fixed', False), ('free', False), ('free', True)]
)
def test_variable_selection(
    keys: split, theta: Literal['fixed', 'free'], augment: bool
) -> None:
    """Check that variable selection works, with and without augmentation."""
    p = 100
    peff = 5
    n = 1000

    X, y, mask = gen_sparse_data(keys.pop(), n=n, p=p, peff=peff)

    bart = Bart(
        x_train=X,
        y_train=y,
        n_burn=1000,
        sparse=SparseConfig(
            augment=augment, theta=float(peff) if theta == 'fixed' else None
        ),
        seed=keys.pop(),
    )

    assert bart.varprob_mean[mask].sum() >= 0.9
    assert bart.varprob_mean[mask].min().item() > 0.5 / peff
    assert bart.varprob_mean[~mask].max().item() < 1 / (p - peff)


def test_scale_shift(bkw: BartKW) -> None:
    """Check self-consistency of rescaling the inputs.

    For mixed binary-continuous outcomes, only the continuous components
    of `y_train` are rescaled, and binary components are matched between
    `bart1` and `bart2` via `bkw.binary_mask`.
    """
    kw = bkw.kw

    if bkw.all_binary:
        pytest.skip('Cannot rescale binary responses.')

    mask = bkw.binary_mask

    bart1 = Bart(**kw)

    # mixed outcomes use float32 leaves (set in `make_kw`): heterogeneous
    # component scales make the two float16 chains bifurcate for some seeds,
    # even though the scaling itself is correct (see
    # test_chol_with_gersh_disparate_scales)
    if bkw.is_mixed:
        assert bart1._mcmc_state.forest.leaf_tree.dtype == jnp.float32

    # blow the response up far enough that the leaves would overflow float16 if
    # they were stored in data units instead of units of leaf_unit; leaves are
    # summed across trees, so the per-leaf magnitude is the total scale over
    # roughly the square root of the number of trees
    overflow_float16_scale = (
        math.sqrt(kw['num_trees']) * 2 * float(jnp.finfo(jnp.float16).max)
    )
    offset = 0.4703189 * overflow_float16_scale
    scale = 0.5294714 * overflow_float16_scale

    y = jnp.where(mask[..., None], kw['y_train'], offset + kw['y_train'] * scale)

    x_offset = -0.6184722
    x_scale = 1.8521347
    x = x_offset + x_scale * kw['x_train']

    kw2 = dict(kw)
    kw2.update(x_train=x, y_train=y, seed=random.clone(kw['seed']))
    bart2 = Bart(**kw2)

    assert_close_matrices(
        bart1.offset,
        jnp.where(mask, bart2.offset, (bart2.offset - offset) / scale),
        rtol=1e-5,
    )
    assert_array_equal(
        nnone(bart1._mcmc_state.error_cov_inv.nu),
        nnone(bart2._mcmc_state.error_cov_inv.nu),
    )

    masked_scale = jnp.where(mask, 1.0, scale)
    if masked_scale.ndim:
        masked_scale = masked_scale[:, None]
    cov_scale = masked_scale * masked_scale.T

    assert_close_matrices(
        nnone(bart1._mcmc_state.forest.leaf_prior_cov_inv),
        nnone(bart2._mcmc_state.forest.leaf_prior_cov_inv) * cov_scale,
        rtol=1e-6,
    )

    assert_close_matrices(
        nnone(bart1._mcmc_state.error_cov_inv.rate) * cov_scale,
        nnone(bart2._mcmc_state.error_cov_inv.rate),
        rtol=1e-6,
    )

    mask_pred = mask[..., None]

    # use a single loose tolerance: the heteroskedastic binary-latent step
    # accumulates float32 rounding, predictions derived from reduced-precision
    # stored leaves hold only to the rounding floor, and 1e-3 covers all cases
    leaf_tree = bart1._mcmc_state.forest.leaf_tree
    pred_rtol = 1e-3

    yhat1 = bart1.predict('train', kind='latent_samples')
    yhat2 = bart2.predict('train', kind='latent_samples')
    assert_close_matrices(
        yhat1,
        jnp.where(mask_pred, yhat2, (yhat2 - offset) / scale),
        rtol=pred_rtol,
        reduce_rank=True,
    )

    mean1 = bart1.predict('train', kind='mean')
    mean2 = bart2.predict('train', kind='mean')
    assert_close_matrices(
        mean1,
        jnp.where(mask_pred, mean2, (mean2 - offset) / scale),
        rtol=pred_rtol,
        reduce_rank=True,
    )

    yhat_test1 = bart1.predict(kw['x_train'], kind='latent_samples')
    yhat_test2 = bart2.predict(x, kind='latent_samples')
    assert_close_matrices(
        yhat_test1,
        jnp.where(mask_pred, yhat_test2, (yhat_test2 - offset) / scale),
        rtol=pred_rtol,
        reduce_rank=True,
    )

    # binary mean predictions on new test data need the (training) error
    # scales; the two models share the same `w`, and continuous-only outcomes
    # ignore it
    mean_w = bart1._mcmc_state.error_scale if bkw.any_binary else None
    yhat_test_mean1 = bart1.predict(kw['x_train'], kind='mean', error_scale=mean_w)
    yhat_test_mean2 = bart2.predict(x, kind='mean', error_scale=mean_w)
    assert_close_matrices(
        yhat_test_mean1,
        jnp.where(mask_pred, yhat_test_mean2, (yhat_test_mean2 - offset) / scale),
        rtol=pred_rtol,
        reduce_rank=True,
    )

    # mixed outcomes accumulate more float32 rounding through the binary-latent
    # step, so the sdev comparisons get a looser tolerance; with reduced leaf
    # precision the rounding floor dominates instead
    rtol = condf(leaf_tree, 1e-4 if bkw.is_mixed else 1e-5, 1e-3)

    # binary positions of get_error_sdev are NaN; replace with 0 to compare.
    with debug_nans(False):
        sdev_actual = jnp.where(mask, 0.0, bart1.get_error_sdev())
        sdev_expected = jnp.where(mask, 0.0, bart2.get_error_sdev() / scale)
        sdev_mean_actual = jnp.where(mask, 0.0, bart1.get_error_sdev(mean=True))
        sdev_mean_expected = jnp.where(
            mask, 0.0, bart2.get_error_sdev(mean=True) / scale
        )
    assert_close_matrices(sdev_actual, sdev_expected, rtol=rtol, reduce_rank=True)
    assert_close_matrices(sdev_mean_actual, sdev_mean_expected, rtol=rtol)


def test_permutation_invariance(bkw: BartKW, keys: split) -> None:
    """Check that `Bart` is invariant under permutation of the datapoints.

    Permuting the inputs along their datapoint axis and inverse-permuting the
    datapoint axis of the fitted arrays must recover the original fit. Binary
    outcomes are excluded: their latent-variable augmentation draws a
    per-datapoint normal keyed by the datapoint position, so permuting the
    data changes the augmentation, and hence the fit.
    """
    if bkw.any_binary:
        pytest.skip('binary latent augmentation is position-dependent')

    kw = bkw.kw
    bart1 = Bart(**kw)

    # permute every input with a datapoint axis (always the last one)
    n = bkw.n
    perm = random.permutation(keys.pop(), n)
    kw2 = dict(kw, seed=random.clone(kw['seed']))
    kw2['x_train'] = kw['x_train'][:, perm]
    kw2['y_train'] = kw['y_train'][..., perm]
    if kw.get('error_scale') is not None:
        kw2['error_scale'] = kw['error_scale'][..., perm]
    if kw.get('missing') is not None:
        kw2['missing'] = kw['missing'][..., perm]
    bart2 = Bart(**kw2)

    # inverse-permute every fitted array along its datapoint axis; `n` differs
    # from every other axis length (see `make_kw`), so the datapoint axis is the
    # unique axis with that length
    inv = jnp.argsort(perm)

    def unpermute(x: Shaped[Array, '*shape']) -> Shaped[Array, '*shape']:
        if n not in getattr(x, 'shape', ()):
            return x
        (axis,) = (i for i, length in enumerate(x.shape) if length == n)
        return jnp.take(x, inv, axis=axis)

    bart2 = tree.map(unpermute, bart2)

    # reduced-precision leaves quantize the permutation-dependent float32
    # reductions, so invariance holds only to the rounding floor; the leaves and
    # everything derived from them carry this loss
    leaf_tree = bart1._mcmc_state.forest.leaf_tree

    def check_equal(
        path: KeyPath, x1: Shaped[Array, '*shape'], x2: Shaped[Array, '*shape']
    ) -> None:
        rtol = condf(
            leaf_tree,
            1e-4 if x1.platform() != 'cpu' else 1e-5,  # ty: ignore[unresolved-attribute]
            1e-3,
        )
        assert_close_matrices(
            x2, x1, err_msg=f'{keystr(path)}: ', rtol=rtol, reduce_rank=True
        )

    tree.map_with_path(check_equal, bart1, bart2)


def test_min_points_per_decision_node(bkw: BartKW) -> None:
    """Check that the limit of at least 10 datapoints per decision node is respected."""
    kw = bkw.kw
    init_kw = dict(kw.get('init_kw', {}))
    init_kw['min_points_per_leaf'] = None
    kw['init_kw'] = init_kw
    bart = Bart(**kw)
    distr = bart._points_per_decision_node_distr()
    distr_marg = distr.sum(axis=(0, 1))

    min_points = init_kw.get('min_points_per_decision_node', 10)

    if min_points is None:
        assert distr_marg[9] > 0
    else:
        assert jnp.all(distr_marg[:min_points] == 0)
        assert jnp.any(distr_marg[min_points:] > 0)


def test_min_points_per_leaf(bkw: BartKW) -> None:
    """Check that the limit of at least 5 datapoints per leaf is respected."""
    kw = bkw.kw
    init_kw = dict(kw.get('init_kw', {}))
    init_kw['min_points_per_decision_node'] = None
    kw['init_kw'] = init_kw
    bart = Bart(**kw)
    distr = bart._points_per_leaf_distr()
    distr_marg = distr.sum(axis=(0, 1))

    min_points = init_kw.get('min_points_per_leaf')  # default None

    if min_points is None:
        assert distr_marg[4] > 0
    else:
        assert jnp.all(distr_marg[:min_points] == 0)
        assert distr_marg[min_points] > 0


@pytest.mark.parametrize('num_datapoints', [0, 1])
def test_zero_or_one_datapoint(bkw: BartKW, num_datapoints: int) -> None:
    """Check automatic data scaling with 0 or 1 datapoints."""
    kw = set_num_datapoints(bkw.kw, num_datapoints)

    if num_datapoints == 0 or bkw.uses_quantile_binner:
        nsplits = 10
        xinfo = jnp.broadcast_to(
            jnp.arange(nsplits, dtype=jnp.float32), (bkw.p, nsplits)
        )
        kw.update(binner=partial(GivenSplitsBinner, xinfo=xinfo))

    kw.update(num_data_devices=None)

    init_kw = dict(kw.get('init_kw', {}))
    init_kw.update(
        save_ratios=True, min_points_per_decision_node=None, min_points_per_leaf=None
    )
    kw['init_kw'] = init_kw

    bart = Bart(**kw)

    assert bart.predict('train', kind='latent_samples').shape == (
        bkw.ndpost,
        *bkw.kshape,
        num_datapoints,
    )

    # check bart.offset
    mask = bkw.binary_mask
    if num_datapoints == 0 or bkw.all_binary:
        assert_array_equal(bart.offset, jnp.zeros_like(bart.offset))
    else:
        # continuous offset is the single datapoint, or 0 if it is missing
        expected = jnp.where(mask[..., None], 0.0, kw['y_train'])
        if kw.get('missing') is not None:
            expected = jnp.where(kw['missing'], 0.0, expected)
        assert_close_matrices(bart.offset[..., None], expected, rtol=1e-6)

    # set expected tau_num and check the error prior default
    if bkw.all_binary:
        tau_num = 3
    else:
        tau_num = jnp.where(mask, 3.0, 1.0)
        # var(y_train) is 0 (n=1) or undefined (n=0), guarded to 1, so the
        # default prior rate is nu for the continuous components
        nu = nnone(bart._mcmc_state.error_cov_inv.nu)
        rate = jnp.diag(nnone(bart._mcmc_state.error_cov_inv.rate))
        assert_close_matrices(
            rate[~mask], jnp.broadcast_to(nu, rate[~mask].shape), rtol=1e-6
        )

    # check leaf_prior_cov_inv
    expected_cov_inv = (2**2 * bkw.num_trees) / tau_num**2
    leaf_prior_cov_inv = nnone(bart._mcmc_state.forest.leaf_prior_cov_inv)
    if leaf_prior_cov_inv.ndim == 2:  # pragma: no branch, always mv with defaults
        expected_cov_inv = jnp.eye(leaf_prior_cov_inv.shape[0]) * expected_cov_inv
    assert_close_matrices(leaf_prior_cov_inv, expected_cov_inv, rtol=1e-6)

    assert bart._burnin_trace.log_likelihood is not None
    assert_close_matrices(
        bart._burnin_trace.log_likelihood,
        jnp.zeros_like(bart._burnin_trace.log_likelihood),
        atol=1e-4,
        reduce_rank=True,
    )
    assert bart._main_trace.log_likelihood is not None
    assert_close_matrices(
        bart._main_trace.log_likelihood,
        jnp.zeros_like(bart._main_trace.log_likelihood),
        atol=1e-4,
        reduce_rank=True,
    )


def test_two_datapoints(bkw: BartKW) -> None:
    """Check automatic data scaling with 2 datapoints."""
    kw = set_num_datapoints(bkw.kw, 2)
    if kw.get('missing') is not None:
        # un-mask fully missing datapoints: with a single effective datapoint
        # every move likelihood ratio is exactly zero (the lone point falls
        # entirely on one side of any split and the marginal likelihood
        # cancels), which would break the log_likelihood assertions below
        kw['missing'] = kw['missing'] & ~jnp.all(kw['missing'], axis=0)
    init_kw = dict(kw.get('init_kw', {}))
    init_kw.update(
        save_ratios=True, min_points_per_decision_node=None, min_points_per_leaf=None
    )
    kw['init_kw'] = init_kw
    bart = Bart(**kw)
    if not bkw.all_binary:
        # the default prior rate is nu * (precision-weighted) var(y_train) per
        # continuous component, see `_guarded_response_variance`
        mask = bkw.binary_mask
        nu = nnone(bart._mcmc_state.error_cov_inv.nu)
        vary = _guarded_response_variance(
            kw['y_train'], kw.get('error_scale'), kw.get('missing')
        )
        rate = jnp.diag(nnone(bart._mcmc_state.error_cov_inv.rate))
        assert_close_matrices(rate[~mask], nu * vary[~mask], rtol=1e-6)
    if bkw.uses_quantile_binner:
        assert jnp.all(bart._mcmc_state.forest.max_split <= 1)
    assert not jnp.all(bart._burnin_trace.log_likelihood == 0.0)
    assert not jnp.all(bart._main_trace.log_likelihood == 0.0)


def test_few_datapoints(bkw: BartKW) -> None:
    """Check that the trees cannot grow if there are not enough datapoints."""
    kw = bkw.kw
    init_kw = dict(kw.get('init_kw', {}))
    init_kw.update(min_points_per_decision_node=10, min_points_per_leaf=None)
    # precompute off so the constancy check is exact (tree evaluation)
    kw1 = dict(
        set_num_datapoints(kw, 8), init_kw=init_kw, precompute_predict_train=False
    )
    bart = Bart(**kw1)
    yhat = bart.predict('train', kind='latent_samples')
    assert jnp.all(yhat == yhat[..., :, :1])

    init_kw2 = dict(kw.get('init_kw', {}))
    init_kw2.update(min_points_per_decision_node=None, min_points_per_leaf=5)
    kw2 = dict(
        set_num_datapoints(kw, 8),
        init_kw=init_kw2,
        seed=random.clone(kw['seed']),
        precompute_predict_train=False,
    )
    bart = Bart(**kw2)
    yhat = bart.predict('train', kind='latent_samples')
    assert jnp.all(yhat == yhat[..., :, :1])


def test_xinfo() -> None:
    """Simple check that `GivenSplitsBinner` works as the `binner`."""
    with debug_nans(False):
        xinfo = jnp.array(
            [[1.1, 2.3, jnp.nan], [-50, 10, 20], [jnp.nan, jnp.nan, jnp.nan]]
        )
    kw = dict(
        x_train=jnp.empty((3, 0)),
        y_train=jnp.empty(0),
        n_save=0,
        n_burn=0,
        binner=partial(GivenSplitsBinner, xinfo=xinfo),
    )
    bart = Bart(**kw)

    xinfo_wo_nan = jnp.where(jnp.isnan(xinfo), jnp.finfo(jnp.float32).max, xinfo)
    assert_array_equal(bart._binner._splits, xinfo_wo_nan)
    assert_array_equal(bart._mcmc_state.forest.max_split, [2, 3, 0], strict=False)


def test_xinfo_wrong_p() -> None:
    """Check that `GivenSplitsBinner` rejects mismatched shapes."""
    with debug_nans(False):
        xinfo = jnp.array(
            [[1.1, 2.3, jnp.nan], [-50, 10, 20], [jnp.nan, jnp.nan, jnp.nan]]
        )
    kw = dict(
        x_train=jnp.empty((5, 0)),
        y_train=jnp.empty(0),
        n_save=0,
        n_burn=0,
        binner=partial(GivenSplitsBinner, xinfo=xinfo),
    )
    # `xinfo`'s p (3) deliberately mismatches `x_train`'s p (5); disable
    # jaxtyping so the cross-axis check doesn't pre-empt the `ValueError`
    with jaxtyping_disabled(), pytest.raises(ValueError, match=r'xinfo\.shape'):
        Bart(**kw)


# The sparse variants check the variable-selection prior (`log_s`, and `theta`
# when free). Their Beta(a, b) prior on theta/(theta+rho) is kept concentrated:
# a diffuse prior (e.g. the a=0.5, b=1 default) makes the theta<->log_s Gibbs
# loop mix too slowly to reach the prior within the burn-in budget, the more so
# with blocked variables (nsplits=1). They use `augment=True` so the MCMC
# samples the exact `log_s` full conditional, matching `sample_prior`. The
# blocked low-p cases ((1, 1), (3, 2)) stress-test that augmentation: without it
# the variable usage is badly biased there (`s` too uniform).
@pytest.mark.parametrize(
    ('p', 'nsplits', 'sparsity'),
    [
        (1, 1, None),
        (3, 2, None),
        (10, 1, None),
        (10, 255, None),
        (10, 1, 'fixed'),
        (10, 1, 'free'),
        (10, 255, 'free'),
        (1, 1, 'free'),
        (3, 2, 'free'),
    ],
)
def test_prior(
    keys: split, p: int, nsplits: int, sparsity: str | None, subtests: SubTests
) -> None:
    """Check that the posterior without data is equivalent to the prior."""
    # few, heavily-blocked variables make the variable-selection Gibbs loop mix
    # slowly, so the small-p sparse cases need a longer chain to reach the prior.
    n_burn, n_save = (6000, 2000) if sparsity is not None and p <= 3 else (3000, 1000)
    bart = run_bart_like_prior(
        keys.pop(), p, nsplits, subtests, sparsity, n_burn, n_save
    )

    prior, prior_trace = sample_prior_like(keys.pop(), bart, subtests)

    with subtests.test('number of stub trees'):
        nstub_mcmc = count_stub_trees(bart._main_trace.split_tree)
        nstub_prior = count_stub_trees(prior_trace.split_tree)
        rhat_nstub = rhat_rank([nstub_mcmc, nstub_prior], split=False)
        assert_array_less(rhat_nstub, 1.01)

    if (p, nsplits) != (1, 1):
        with subtests.test('number of simple trees'):
            nsimple_mcmc = count_simple_trees(bart._main_trace.split_tree)
            nsimple_prior = count_simple_trees(prior_trace.split_tree)
            rhat_nsimple = rhat_rank([nsimple_mcmc, nsimple_prior], split=False)
            assert_array_less(rhat_nsimple, 1.01)

        varcount_prior = compute_varcount(
            bart._mcmc_state.forest.max_split.size, prior_trace
        )

        if sparsity is None:
            # under variable selection the per-variable counts are exchangeable
            # but their labelling mixes very slowly (a sparse `s` sticks on a
            # variable for many iterations), so this never converges; the
            # label-invariant `effective predictors` subtest checks `s` instead.
            with subtests.test('varcount'):
                rhat_varcount = rhat_rank([bart.varcount, varcount_prior], split=False)
                # `rhat_varcount` is a max over `p` rank-rhat values; its upper
                # tail reaches ~1.05 even when the MCMC matches the prior exactly
                # (checked over 100 seeds for both p=3 and p=10), so a tighter
                # bound would false-fail on ~10% of seeds.
                assert_array_less(rhat_varcount, 1.05)

        with subtests.test('number of nodes'):
            sum_varcount_mcmc = bart.varcount.sum(axis=1)
            sum_varcount_prior = varcount_prior.sum(axis=1)
            rhat_sum_varcount = rhat_rank(
                [sum_varcount_mcmc, sum_varcount_prior], split=False
            )
            assert_array_less(rhat_sum_varcount, 1.02)

        with subtests.test('imbalance index'):
            imb_mcmc = avg_imbalance_index(bart._main_trace.split_tree)
            imb_prior = avg_imbalance_index(prior_trace.split_tree)
            rhat_imb = rhat_rank([imb_mcmc, imb_prior], split=False)
            assert_array_less(rhat_imb, 1.02)

        with subtests.test('average max tree depth'):
            maxd_mcmc = avg_max_tree_depth(bart._main_trace.split_tree)
            maxd_prior = avg_max_tree_depth(prior_trace.split_tree)
            rhat_maxd = rhat_rank([maxd_mcmc, maxd_prior], split=False)
            assert_array_less(rhat_maxd, 1.02)

        with subtests.test('max tree depth distribution'):
            dd_mcmc = bart._depth_distr()
            dd_prior = forest_depth_distr(prior_trace.split_tree)
            rhat_dd = rhat_rank([dd_mcmc.squeeze(0), dd_prior], split=False)
            assert_array_less(rhat_dd, 1.02)

    with subtests.test('y_test'):
        assert nsplits <= 255  # `X` is uint8, so the split count must fit a byte
        X = random.randint(keys.pop(), (p, 30), 0, nsplits + 1, jnp.uint8)
        yhat_mcmc = predict_latent(X, bart._main_trace)
        yhat_prior = evaluate_trace(X, prior_trace)
        rhat_yhat = rhat_rank([yhat_mcmc, yhat_prior], split=False)
        assert_array_less(rhat_yhat, 1.01)

    if sparsity is not None:
        check_sparsity_prior(bart, prior, subtests)


def check_sparsity_prior(bart: Bart, prior: PriorSample, subtests: SubTests) -> None:
    """Check the MCMC matches the prior on the variable-selection parameters."""
    # `effective predictors` is a label-invariant summary of the variable
    # selection probabilities `s`, so unlike the per-variable counts it is immune
    # to the slow mixing of which variables carry the mass. With p=1 it is
    # degenerate (s is always [1], and the augmentation is a no-op), so skip it.
    if bart._mcmc_state.forest.max_split.size > 1:
        with subtests.test('effective predictors'):
            eff_mcmc = effective_predictors(nnone(bart._main_trace.varprob))
            eff_prior = effective_predictors(softmax(nnone(prior.log_s), axis=-1))
            rhat_eff = rhat_rank([eff_mcmc, eff_prior], split=False)
            # the bound is loose because `s` mixes slowly with few blocked
            # variables; it still easily catches a broken augmentation, which
            # biases the usage enough to push this rhat well past 1.3.
            assert_array_less(rhat_eff, 1.08)

    if bart._mcmc_state.forest.rho is not None:  # `theta` is sampled
        with subtests.test('theta'):
            rhat_theta = rhat_rank(
                [nnone(bart._main_trace.theta), nnone(prior.theta)], split=False
            )
            assert_array_less(rhat_theta, 1.05)


def sparsity_prior_params(sparsity: str | None, p: int) -> kwdict:
    """Sparsity-prior hyperparameters shared by the MCMC and the prior draw."""
    params = {'fixed': dict(theta=float(p)), 'free': dict(a=20.0, b=20.0, rho=float(p))}
    return {} if sparsity is None else params[sparsity]


def effective_predictors(
    varprob: Float32[Array, '*batch p'],
) -> Float32[Array, ' *batch']:
    """Effective number of predictors, the exp of the split-probability entropy."""
    entropy = -jnp.sum(jnp.where(varprob > 0, varprob * jnp.log(varprob), 0.0), axis=-1)
    return jnp.exp(entropy)


def run_bart_like_prior(
    key: Key[Array, ''],
    p: int,
    nsplits: int,
    subtests: SubTests,
    sparsity: str | None,
    n_burn: int,
    n_save: int,
) -> Bart:
    """Run `Bart` without datapoints to sample the prior distribution."""
    xinfo = jnp.broadcast_to(jnp.arange(nsplits, dtype=jnp.float32), (p, nsplits))
    sparse_kw = (
        {}
        if sparsity is None
        else dict(
            sparse=SparseConfig(augment=True, **sparsity_prior_params(sparsity, p))
        )
    )

    bart = Bart(
        x_train=jnp.empty((p, 0)),
        y_train=jnp.empty(0),
        num_trees=20,
        n_save=n_save,
        n_burn=n_burn,
        printevery=None,
        binner=partial(GivenSplitsBinner, xinfo=xinfo),
        seed=key,
        num_chains=None,
        init_kw=dict(
            min_points_per_decision_node=None,
            min_points_per_leaf=None,
            save_ratios=True,
        ),
        **sparse_kw,
    )

    with subtests.test('likelihood ratio = 1'):
        assert bart._burnin_trace.log_likelihood is not None
        assert_close_matrices(
            bart._burnin_trace.log_likelihood,
            jnp.zeros_like(bart._burnin_trace.log_likelihood),
            atol=1e-5,
            reduce_rank=True,
        )
        assert bart._main_trace.log_likelihood is not None
        assert_close_matrices(
            bart._main_trace.log_likelihood,
            jnp.zeros_like(bart._main_trace.log_likelihood),
            atol=1e-5,
            reduce_rank=True,
        )

    return bart


def sample_prior_like(
    key: Key[Array, ''], bart: Bart, subtests: SubTests
) -> tuple[PriorSample, MinimalTrace]:
    """Sample from the prior with the same settings used in `bart`."""
    forest = bart._mcmc_state.forest
    p_nonterminal = forest.p_nonterminal
    max_depth = tree_depth(p_nonterminal)
    indices = 2 ** jnp.arange(max_depth - 1)
    p_nonterminal = p_nonterminal[indices]

    # mirror the variable-selection prior the MCMC used, read back from the state:
    # free theta (a, b, rho set), fixed theta, or no variable selection
    if forest.rho is not None:
        sparse_kw = dict(a=forest.a, b=forest.b, rho=forest.rho)
    elif forest.theta is not None:
        sparse_kw = dict(theta=forest.theta)
    else:
        sparse_kw = {}

    prior_trees = sample_prior(
        key,
        bart.n_save,
        len(forest.leaf_tree),
        forest.max_split,
        p_nonterminal,
        jnp.sqrt(jnp.reciprocal(nnone(forest.leaf_prior_cov_inv))),
        **sparse_kw,
    )

    with subtests.test('check prior trees'):
        bad = check_trace(prior_trees, forest.max_split)
        bad_count = jnp.count_nonzero(bad)
        assert bad_count == 0

    return prior_trees, project(MinimalTrace, replace(prior_trees, offset=bart.offset))


def count_stub_trees(
    split_tree: UInt[Array, '*batch_shape ntree 2**(d-1)'],
) -> Int32[Array, '*batch_shape']:
    """Count the number of trees with only a root node."""
    return (~split_tree.any(-1)).sum(-1)


def count_simple_trees(
    split_tree: UInt[Array, '*batch_shape ntree 2**(d-1)'],
) -> Int32[Array, '*batch_shape']:
    """Count the number of trees with 2 layers."""
    return (split_tree.astype(bool).sum(-1) == 1).sum(-1)


@partial(jnp.vectorize, signature='(n,m)->()')
def avg_imbalance_index(
    split_tree: UInt[Array, '*batch_shape ntree 2**(d-1)'],
) -> Float32[Array, '*batch_shape']:
    """Measure average tree imbalance in the forest."""
    is_leaf = vmap(partial(is_actual_leaf, add_bottom_level=True))(split_tree)
    depths = tree_depths(is_leaf.shape[-1])
    depths = jnp.broadcast_to(depths, is_leaf.shape)
    index = jnp.std(depths, where=is_leaf, axis=-1)
    return index.mean(-1)


@partial(jnp.vectorize, signature='(n,m)->()')
def avg_max_tree_depth(
    split_tree: UInt[Array, '*batch_shape ntree 2**(d-1)'],
) -> Float32[Array, '*batch_shape']:
    """Measure average maximum tree depth in the forest."""
    depth = vmap(tree_actual_depth)(split_tree)
    return depth.mean(-1)


def test_jit(bkw: BartKW) -> None:
    """Test that jitting around the whole interface works."""
    kw = bkw.kw
    kw.update(printevery=None, rm_const=False)

    platform = kw['y_train'].platform()
    kw.update(devices=jax.devices(platform))

    X = kw.pop('x_train')
    y = kw.pop('y_train')
    w = kw.pop('error_scale', None)
    key = kw.pop('seed')

    def task(
        X: Shaped[Array, 'p n'],
        y: Shaped[Array, ' n'] | Shaped[Array, 'k n'],
        w: Float32[Array, ' n'] | None,
        key: Key[Array, ''],
    ) -> tuple[State, Shaped[Array, 'ndpost n'] | Shaped[Array, 'ndpost k n']]:
        bart = OriginalBart(X, y, error_scale=w, **kw, seed=key)
        return bart._mcmc_state, bart.predict('train', kind='latent_samples')

    task_compiled = jit(task)

    _state1, pred1 = task(X, y, w, key)
    _state2, pred2 = task_compiled(X, y, w, random.clone(key))

    assert_close_matrices(pred1, pred2, rtol=1e-5, reduce_rank=True)


def test_vmap(bkw: BartKW, keys: split) -> None:
    """Test that jit(vmap(...))ing around the whole interface works.

    Vmaps `Bart` over a leading repetition axis added to every input carrying
    the data dimension ``n`` (``x_train``, ``y_train``, ``w``, ``missing``),
    and checks each leaf of the vmapped `Bart` matches the stack of the
    corresponding leaves of the per-repetition `Bart` objects.
    """
    kw = bkw.kw
    kw.update(printevery=None, rm_const=False)

    platform = kw['y_train'].platform()
    kw.update(devices=jax.devices(platform))

    binary_mask = bkw.binary_mask  # read before popping y_train

    X = kw.pop('x_train')
    y = kw.pop('y_train')
    w = kw.pop('error_scale', None)
    missing = kw.pop('missing', None)
    kw.pop('seed')  # replaced by a distinct per-repetition key below

    nrep = 3
    key = keys.pop(nrep)  # one MCMC seed per repetition

    def tile(a: Shaped[Array, '...']) -> Shaped[Array, 'nrep ...']:
        return jnp.broadcast_to(a, (nrep, *a.shape))

    # tile each input over the repetition axis, then add a per-repetition
    # perturbation that keeps it valid (positive `w`, boolean `missing`,
    # binary outcome components in ``{0, 1}``)
    Xb = tile(X) + 0.1 * random.normal(keys.pop(), (nrep, *X.shape))

    is_binary = jnp.broadcast_to(binary_mask[..., None], y.shape)
    yb_cont = tile(y) + 0.1 * random.normal(keys.pop(), (nrep, *y.shape))
    yb_bin = tile(y).astype(bool) ^ random.bernoulli(keys.pop(), 0.1, (nrep, *y.shape))
    yb = jnp.where(tile(is_binary), yb_bin.astype(y.dtype), yb_cont)

    if w is None:
        wb = None
    else:
        wb = tile(w) * jnp.exp(0.1 * random.normal(keys.pop(), (nrep, *w.shape)))

    if missing is None:
        mb = None
    else:
        mb = tile(missing) ^ random.bernoulli(keys.pop(), 0.1, (nrep, *missing.shape))

    def task(
        X: Shaped[Array, 'p n'],
        y: Shaped[Array, ' n'] | Shaped[Array, 'k n'],
        w: Float32[Array, ' n'] | Float32[Array, 'k n'] | None,
        missing: Bool[Array, ' n'] | Bool[Array, 'k n'] | None,
        key: Key[Array, ''],
    ) -> OriginalBart:
        return OriginalBart(X, y, error_scale=w, missing=missing, **kw, seed=key)

    batched = jit(vmap(task))(Xb, yb, wb, mb, key)

    key_singles = random.clone(key)  # avoid reusing the keys consumed by vmap
    singles = [
        task(
            Xb[i, ...],
            yb[i, ...],
            None if wb is None else wb[i, ...],
            None if mb is None else mb[i, ...],
            key_singles[i],
        )
        for i in range(nrep)
    ]
    stacked = tree.map(lambda *leaves: jnp.stack(leaves), *singles)

    # reduced-precision leaves quantize the slightly different float32 reductions
    # of vmapped vs looped runs, so equivalence holds only to the rounding floor;
    # the leaves and everything derived from them carry this loss
    rtol = condf(
        singles[0]._mcmc_state.forest.leaf_tree,
        1e-4 if platform != 'cpu' else 1e-5,
        1e-3,
    )

    def check(a: Shaped[Array, '*shape'], b: Shaped[Array, '*shape']) -> None:
        assert_close_matrices(a, b, rtol=rtol, reduce_rank=True)

    tree.map(check, batched, stacked)


# WORKAROUND(jax<0.7.2): before 0.7.2, callbacks (pure_callback in
# gammainccinv, equinox debug checks) disable JAX's C++ pjit fastpath cache, so
# no_tracing raises a false positive without any actual jaxpr re-tracing. Remove
# this skip once the oldest supported jax is >= 0.7.2.
@pytest.mark.skipif(
    jax.__version_info__ < (0, 7, 2),
    reason='no_tracing gives false positives with jax < 0.7.2: callbacks '
    '(pure_callback in gammainccinv, equinox debug checks) disable '
    "JAX's C++ pjit fastpath cache, so no_tracing raises without any actual "
    'jaxpr re-tracing (see test_no_recompilation_inner_loop_counter for the '
    'version-robust check)',
)
def test_no_recompilation_no_tracing(bkw: BartKW) -> None:
    """Check that running the same `Bart` invocation twice does not retrace.

    `jax.no_tracing` has various false positives, so this test has a few
    workarounds that reduce coverage, which is compensated by
    `test_no_recompilation_inner_loop_counter`.
    """
    kw: kwdict = dict(
        bkw.kw,
        # no callback because all our callbacks use `jax.debug.callback` which
        # blocks the jit fast path which `no_tracing` counts as tracing
        run_mcmc_kw=dict(bkw.kw.get('run_mcmc_kw', {}), callback=None),
    )

    # use OriginalBart because the test wrapper's `_check_replicated_trees`
    # always retraces due to its `shard_map`
    OriginalBart(**kw)

    kw2: kwdict = dict(kw, seed=random.clone(kw['seed']))
    with no_tracing():
        OriginalBart(**kw2)


def test_no_recompilation_inner_loop_counter(bkw: BartKW) -> None:
    """Check the MCMC inner loop is not retraced on the second `Bart` call.

    Uses the `_CallCounter` already wrapping `_run_mcmc_inner_loop`. The
    counter is reset at the start of each `run_mcmc` and increments only on
    actual Python-side tracing of the inner loop; if the JIT cache hits it
    stays at 0. Unlike `test_no_recompilation_no_tracing`, this works with
    the default ``printevery`` (which uses `debug.callback`), but only
    covers `_run_mcmc_inner_loop`.
    """
    kw = bkw.kw
    Bart(**kw)
    kw2 = dict(kw, seed=random.clone(kw['seed']))
    Bart(**kw2)
    assert _inner_loop_counter.n_calls == 0


def test_print_callback_terminates_dot_line(
    bkw: BartKW, capsys: CaptureFixture[str]
) -> None:
    """MCMC logging ends with a newline when the last iteration only prints a dot."""
    kw = bkw.kw
    kw['pbar'] = False  # this test is about the print-callback line format
    n_iters = kw['n_burn'] + kw['n_save'] * _bart_default(kw, 'n_skip')
    printevery = _bart_default(kw, 'printevery')
    # ensure the last iteration falls outside a report boundary so it prints
    # only a dot, exercising the previously buggy path
    if printevery is None or n_iters % printevery == 0:
        kw['printevery'] = n_iters - 1
    block_until_ready(Bart(**kw))
    captured = capsys.readouterr()
    assert captured.out.endswith('.\n'), repr(captured.out[-200:])


def test_pbar(bkw: BartKW, capsys: CaptureFixture[str]) -> None:
    """The `pbar=True` progress bar runs to completion across configurations.

    Parametrized over `bkw`, so it also covers chains sharded across devices
    (e.g. variant v6). This exercises that `TqdmCallback` uses unordered debug
    callbacks, since ordered ones are unsupported with more than one device.
    """
    # force a concrete `printevery`: some bkw variants set it to None, which now
    # disables the bar entirely (covered by test_pbar_disabled_by_printevery_none)
    block_until_ready(Bart(**dict(bkw.kw, pbar=True, printevery=10)))
    assert '100%' in capsys.readouterr().err  # the bar ran and reached the end


def test_pbar_disabled_by_printevery_none(
    bkw: BartKW, capsys: CaptureFixture[str]
) -> None:
    """`printevery=None` disables the bar entirely, even with `pbar=True`."""
    n_bars_before = len(_TQDM_REGISTRY)
    block_until_ready(Bart(**dict(bkw.kw, pbar=True, printevery=None)))
    assert '%' not in capsys.readouterr().err  # no bar was drawn
    assert len(_TQDM_REGISTRY) == n_bars_before  # no bar was even created


def test_print_tree(bkw: BartKW, capsys: CaptureFixture[str]) -> None:
    """Smoke-test `Bart._print_tree`."""
    bart = Bart(**bkw.kw)
    capsys.readouterr()  # discard MCMC progress output
    bart._print_tree(0, 0, 0)
    assert capsys.readouterr().out


@pytest.mark.flaky
# it's flaky because the interrupt may be caught and converted by jax internals (#33054)
@pytest.mark.timeout(32)
def test_interrupt(bkw: BartKW) -> None:
    """Test that the MCMC can be interrupted with ^C."""
    kw = bkw.kw
    kw.update(printevery=1, n_save=0, n_burn=10000)

    # Send the first ^C after 3 s, if the time was too short, it would interrupt
    # a first interruptible phase of jax compilation. Then send ^C every second,
    # in case the first ^C landed during a second non-interruptible compilation
    # phase that eats ^C and ignores it.
    with (
        pytest.raises(KeyboardInterrupt),
        periodic_sigint(first_after=3.0, interval=1.0),
    ):
        block_until_ready(Bart(**kw))


def test_dataframe_series_duck_types() -> None:
    """Check pandas and polars frames satisfy the `DataFrame`/`Series` ducks."""
    data = numpy.zeros((3, 2))
    for frame in pd.DataFrame(data), pl.DataFrame(data):
        assert isinstance(frame, DataFrame)
    for series in pd.Series(data[:, 0]), pl.Series(data[:, 0]):
        assert isinstance(series, Series)
    # a bare array is neither, lacking `columns`/`name`/`to_numpy`
    assert not isinstance(data, DataFrame)
    assert not isinstance(data, Series)


def test_polars(bkw: BartKW) -> None:
    """Test passing data as DataFrame and Series."""
    kw = bkw.kw

    bart = Bart(**kw)
    pred = bart.predict(bkw.x_test, kind='latent_samples')

    def to_polars(a: Shaped[Array, '...'] | None) -> pl.Series | pl.DataFrame | None:
        if a is None:
            return None
        arr = numpy.array(a)
        if arr.ndim == 1:
            return pl.Series(arr)
        return pl.DataFrame(arr.T)

    kw2 = dict(kw)
    kw2.update(
        seed=random.clone(kw2['seed']),
        x_train=to_polars(kw['x_train']),
        y_train=to_polars(kw['y_train']),
        error_scale=to_polars(kw.get('error_scale')),
        missing=to_polars(kw.get('missing')),
    )
    bart2 = Bart(**kw2)
    x_test_pl = pl.DataFrame(numpy.array(bkw.x_test).T)
    pred2 = bart2.predict(x_test_pl, kind='latent_samples')

    rtol = 0 if pred.platform() == 'cpu' else 2e-6  # ty: ignore[unresolved-attribute]

    assert_close_matrices(
        bart.predict('train', kind='latent_samples'),
        bart2.predict('train', kind='latent_samples'),
        rtol=rtol,
        reduce_rank=True,
    )
    if not bkw.any_binary:
        assert_close_matrices(bart.get_error_sdev(), bart2.get_error_sdev(), rtol=rtol)
    assert_close_matrices(pred, pred2, rtol=rtol, reduce_rank=True)


def test_data_format_mismatch(bkw: BartKW) -> None:
    """Test that passing predictors with mismatched formats raises an error."""
    kw = bkw.kw
    kw.update(
        x_train=pl.DataFrame(numpy.array(kw['x_train']).T),
        error_scale=None
        if kw.get('error_scale') is None
        else pl.Series(numpy.array(kw['error_scale'])),
    )
    bart = Bart(**kw)
    w = bkw.w_for_predict('mean', on_train=False)
    with pytest.raises(ValueError, match='format mismatch'):
        bart.predict(numpy.array(bkw.x_test), error_scale=w)


def test_automatic_integer_types(bkw: BartKW) -> None:
    """Test that integer variables in the MCMC state have the correct type."""
    kw = bkw.kw
    bart = Bart(**kw)

    def select_type(cond: bool) -> type:
        return jnp.uint8 if cond else jnp.uint16

    leaf_indices_type = select_type(bkw.maxdepth <= 8)
    split_trees_type = X_type = select_type(bkw.max_bins <= 256)
    var_trees_type = select_type(bkw.p <= 256)

    assert bart._mcmc_state.forest.var_tree.dtype == var_trees_type
    assert bart._mcmc_state.forest.split_tree.dtype == split_trees_type
    assert bart._mcmc_state.forest.leaf_indices.dtype == leaf_indices_type
    assert bart._mcmc_state.X.dtype == X_type
    assert bart._mcmc_state.forest.max_split.dtype == split_trees_type


def check_chains_data_sharding(
    x: Shaped[Array, '...'], chains: bool, data: bool, mesh: Mesh | None
) -> None:
    """Check `x` is sharded as indicated by the boolean flags."""
    if mesh is None:
        assert isinstance(x.sharding, SingleDeviceSharding)
        return
    expected_spec = [None] * x.ndim
    if data and 'data' in mesh.axis_names:
        expected_spec[-1] = 'data'
    if chains and 'chains' in mesh.axis_names:
        expected_spec[0] = 'chains'
    expected_spec = normalize_spec(expected_spec, mesh, x.shape)
    assert get_normal_spec(x) == expected_spec


def get_expect_sharded(kw: dict) -> bool:
    """Check whether we expect sharding to be set up based on the arguments."""
    num_chain_devices = kw.get('num_chain_devices', 'auto')
    num_chains = kw.get('num_chains', 4)
    return (
        hasattr(num_chain_devices, '__index__')
        or kw.get('num_data_devices') is not None
        or (
            num_chain_devices == 'auto'
            and num_chains is not None
            and num_chains > 1
            and get_device_count() > 1
            and get_default_device().platform == 'cpu'
        )
    )


def test_sharding(bkw: BartKW, variant: int, keys: split) -> None:
    """Check that chains and data shards live on their own devices throughout the interface."""
    # WORKAROUND(jax<0.7): sharding bug, no time to fix
    if jax.__version_info__ < (0, 7, 0) and variant in (2, 5):
        pytest.xfail('Sharding bug in bartz with jax<0.7.')
    kw = bkw.kw
    bart = Bart(**kw)

    expect_sharded = get_expect_sharded(kw)
    mesh = bart._mcmc_state.config.mesh
    assert expect_sharded == (mesh is not None)

    # check internal attributes that have chains/data metadata, so their
    # expected sharding can be automatically inferred by `check_sharding`
    check = partial(check_sharding, mesh=mesh)
    check(bart._mcmc_state)
    check(bart._burnin_trace)
    check(bart._main_trace)

    check = partial(check_chains_data_sharding, mesh=mesh)

    for kind in PredictKind:
        extra: dict = {'key': keys.pop()} if kind is PredictKind.outcome_samples else {}
        # the mean collapses the chain axis; the other kinds keep it
        chains = kind is not PredictKind.mean
        yhat_train = bart.predict('train', kind=kind, **extra)
        check(yhat_train, chains=chains, data=True)

        extra['error_scale'] = bkw.w_for_predict(kind.value, on_train=False)
        yhat_test = bart.predict(bkw.x_test, kind=kind, **extra)
        check(yhat_test, chains=chains, data=True)

    assert bart.offset.is_fully_replicated
    assert bart.varcount_mean.is_fully_replicated
    assert bart.varprob_mean.is_fully_replicated


class TestVarprobParam:
    """Test the `varprob` parameter."""

    def test_biased_predictor_choice(self, keys: split, bkw: BartKW) -> None:
        """Check that if `varprob[i]` is high then predictor `i` is used more than others."""
        kw = bkw.kw
        i = random.randint(keys.pop(), (), 0, bkw.p)
        vp = jnp.full(bkw.p, 0.001).at[i].set(1)
        vp /= vp.sum()
        kw.update(sparse=SparseConfig(enabled=False), varprob=vp)
        bart = Bart(**kw)
        vc = bart.varcount_mean
        vc /= vc.sum()
        assert vc[i] > vp[i] * 0.6

    def test_positive(self, bkw: BartKW, subtests: SubTests) -> None:
        """Check that an error is raised if varprob is not > 0."""
        kw = bkw.kw

        with subtests.test('not negative'):
            assert bkw.p > 1
            varprob = jnp.ones(bkw.p).at[0].set(-1.0)
            kw_neg = dict(kw, varprob=varprob)
            with pytest.raises(EquinoxRuntimeError, match='varprob must be > 0'):
                Bart(**kw_neg)

        with subtests.test('not 0'):
            varprob = jnp.zeros(bkw.p).at[0].set(1.0)
            kw_zero = dict(kw, varprob=varprob)
            with pytest.raises(EquinoxRuntimeError, match='varprob must be > 0'):
                Bart(**kw_zero)


def run_bart_and_block(bkw: BartKW, keys: split) -> None:
    """Run bart and block until all outputs are ready."""
    bart = Bart(**bkw.kw)
    stuff = (
        bart,
        bart.predict('train', kind='latent_samples'),
        bart.predict('train', kind='mean'),
        bart.predict('train', kind='mean_samples'),
        bart.predict('train', kind='outcome_samples', key=keys.pop()),
        bart.predict(bkw.x_test, kind='latent_samples'),
        bart.predict(
            bkw.x_test,
            kind='mean',
            error_scale=bkw.w_for_predict('mean', on_train=False),
        ),
        bart.predict(
            bkw.x_test,
            kind='mean_samples',
            error_scale=bkw.w_for_predict('mean_samples', on_train=False),
        ),
        bart.predict(
            bkw.x_test, kind='outcome_samples', key=keys.pop(), error_scale=bkw.w_test
        ),
        bart.get_error_sdev(),
        bart.get_latent_prec(),
        bart.varcount,
        bart.varprob,
    )
    block_until_ready(stuff)


@contextmanager
def array_garbage_collection_guard(value: str) -> Iterator[None]:
    """Implement `jax.array_garbage_collection_guard`, added in jax v0.9.1."""
    # WORKAROUND(jax<0.9.1): replace with `jax.array_garbage_collection_guard`
    setting = 'jax_array_garbage_collection_guard'
    prev = getattr(config, setting)
    config.update(setting, value)
    try:
        yield
    finally:
        config.update(setting, prev)


@contextmanager
def catch_array_gc_guard() -> Iterator[None]:
    """Catch array GC guard log messages and raise as exceptions."""
    # we need this because array_garbage_collection_guard('fatal')
    # terminates the process, and the 'log' messages are emitted from C++
    # code that swallows Python exceptions, so we can only check after the
    # context exits
    buf = StringIO()
    with redirect_stderr(buf), array_garbage_collection_guard('log'):
        yield
    captured = buf.getvalue()
    errmsg = '`jax.Array` was deleted by the Python garbage collector'
    if errmsg in captured:
        raise RuntimeError(captured)


def create_array_cycle() -> ReferenceType[Array]:
    """Create a reference cycle of two jax.Arrays."""
    n1 = jnp.ones((2, 2))
    n2 = jnp.zeros((2, 2))
    n1.next = n2  # ty:ignore[unresolved-attribute]
    n2.next = n1  # ty:ignore[unresolved-attribute]
    return ref(n1)


def test_catch_array_gc_guard() -> None:
    """Test `catch_array_gc_guard`."""
    collect()
    with (  # noqa: PT012, in case a gc collection happened before the explicit one
        pytest.raises(RuntimeError, match='deleted by the Python garbage collector'),
        catch_array_gc_guard(),
    ):
        weak = create_array_cycle()
        assert weak() is not None
        collect()
    assert weak() is None


def test_debug_checks(keys: split, bkw: BartKW) -> None:
    """Run with invasive jax debug options active."""
    with debug_nans(True), debug_infs(True), debug_key_reuse(True):
        run_bart_and_block(bkw, keys)


def test_no_array_gc(keys: split, bkw: BartKW) -> None:
    """Check that no reference cycles cause `jax.Array`s to be deleted by gc.

    Kept separate from `test_debug_checks` because `debug_nans` and `debug_infs`
    force jax through a slow Python dispatch path that itself creates cycles,
    yielding spurious failures unrelated to bartz code.

    Runs with `jaxtyping_disabled` because the runtime type-checking import hook
    (on during the tests) wraps every function with `beartype`, whose closures
    introduce reference cycles of their own; this check is about cycles in the
    bartz code, not in the test-only instrumentation.
    """
    collect()
    with catch_array_gc_guard(), jaxtyping_disabled():
        run_bart_and_block(bkw, keys)
        collect()


def test_equiv_sharding(bkw: BartKW, subtests: SubTests) -> None:
    """Check that the result is the same with/without sharding."""
    if get_device_count() < 2:  # this branch is covered in single cpu tests config
        pytest.skip('Need at least 2 devices for this test')
    if bkw.any_binary:
        # Binary regression uses `step_z`, which on data sharding folds the
        # shard index into the key to decorrelate per-datapoint draws — this
        # intentionally breaks bit-equivalence with the unsharded execution.
        pytest.skip('step_z breaks sharding equivalence on binary outcomes')

    baseline_kw = tree.map(lambda x: x, bkw.kw)
    # devices=None because the explicit single device passed by some variants
    # would leave no room to build the sharding meshes (and pin `_device`,
    # breaking treedef equality with the sharded fits)
    baseline_kw.update(
        num_chain_devices=None,
        num_data_devices=None,
        n_burn=0,
        n_save=10,
        num_chains=2,
        devices=None,
    )
    bart = Bart(**baseline_kw)

    # float32 arrays pin the MCMC trajectory and otherwise match exactly across
    # sharding; a narrow prec_scale, read in the data-parallel per-leaf
    # reductions, lets those reductions (and the error-variance trajectory they
    # feed) diverge a bit more when data sharding reorders them
    prec_scale = bart._mcmc_state.prec_scale
    strict_rtol = 1e-5 if prec_scale is None else condf(prec_scale, 1e-5, 1e-3)

    def check_equal(
        path: KeyPath, xb: Shaped[Array, '*shape'], xs: Shaped[Array, '*shape']
    ) -> None:
        # Leaves stored in a reduced-precision dtype can differ by a rounding
        # step when data sharding reorders the per-leaf reductions, so compare
        # them only at their storage precision.
        reduced = jnp.issubdtype(xs.dtype, jnp.floating) and xs.dtype.itemsize < 4
        assert_close_matrices(
            xs,
            xb,
            err_msg=f'{keystr(path)}: ',
            rtol=1e-2 if reduced else strict_rtol,
            reduce_rank=True,
        )

    # the sharded fits go through `_drop_device_info` because the mesh is
    # static metadata, so it must be cleared to make treedefs match the
    # unsharded baseline before comparing leaves

    with subtests.test('shard chains'):
        chains_kw = tree.map(lambda x: x, baseline_kw)
        chains_kw.update(num_chain_devices=2)
        bart_chains = Bart(**chains_kw)._drop_device_info()
        tree.map_with_path(check_equal, bart, bart_chains)

    with subtests.test('shard data'):
        data_kw = tree.map(lambda x: x, baseline_kw)
        data_kw.update(num_data_devices=2)
        bart_data = Bart(**data_kw)._drop_device_info()
        tree.map_with_path(check_equal, bart, bart_data)

    if get_device_count() >= 4:  # pragma: no branch
        with subtests.test('shard data and chains'):
            both_kw = tree.map(lambda x: x, baseline_kw)
            both_kw.update(num_chain_devices=2, num_data_devices=2)
            bart_both = Bart(**both_kw)._drop_device_info()
            tree.map_with_path(check_equal, bart, bart_both)


@pytest.mark.parametrize(
    ('num_chains', 'num_chain_devices'),
    [
        (None, 2),  # cannot shard a scalar (single) chain across 2 devices
        (1, 2),  # 2 does not divide 1
        (4, 3),  # 3 does not divide 4
        (4, 0),  # not a positive number of devices
    ],
)
def test_num_chain_devices_invalid(
    num_chains: int | None, num_chain_devices: int, keys: split
) -> None:
    """`Bart` rejects a `num_chain_devices` that does not divide the chains."""
    dgp = gen_data(keys.pop(), n=10, p=2, **GEN_KW)
    with pytest.raises(ValueError, match='must be a positive divisor'):
        Bart(
            dgp.x,
            dgp.y,
            seed=keys.pop(),
            n_save=0,
            n_burn=0,
            num_chains=num_chains,
            num_chain_devices=num_chain_devices,
        )


def test_num_chains_none_with_chain_device(keys: split) -> None:
    """`num_chains=None` ignores a harmless 1-device request (no chain axis)."""
    dgp = gen_data(keys.pop(), n=10, p=2, **GEN_KW)
    bart = Bart(
        dgp.x,
        dgp.y,
        seed=keys.pop(),
        n_save=10,
        n_burn=10,
        num_chains=None,
        num_chain_devices=1,
    )
    assert bart.num_chains is None
    mesh = bart._mcmc_state.config.mesh
    assert mesh is None or 'chains' not in mesh.axis_names


def test_auto_chains_fit_with_data_sharding(keys: split) -> None:
    """Auto chain sharding shrinks to leave room for data sharding."""
    total = get_device_count()
    if total < 2:
        pytest.skip('Need at least 2 devices to reserve some for data sharding.')
    # reserve every device for the data axis, so auto chain sharding must back
    # off to a single chain device; pre-fix this overcommitted the devices and
    # `make_mesh` raised.
    n = 12 * total  # divisible by total so num_data_devices=total is valid
    dgp = gen_data(keys.pop(), n=n, p=2, **GEN_KW)
    bart = Bart(
        dgp.x,
        dgp.y,
        seed=keys.pop(),
        num_trees=5,
        n_save=10,
        n_burn=10,
        num_chains=4,
        num_data_devices=total,
    )
    mesh = bart._mcmc_state.config.mesh
    assert mesh is not None
    assert mesh.size <= total


def test_auto_chains_within_explicit_devices(bkw: BartKW) -> None:
    """Auto chain sharding stays within an explicitly passed device list."""
    if get_default_device().platform != 'cpu':  # pragma: no cover
        pytest.skip('Auto chain sharding only kicks in on cpu.')
    if get_device_count() < 2:
        pytest.skip('Need >1 device to hand `Bart` fewer than it would use.')
    if (cpu_count() or 1) < 2:  # pragma: no cover
        pytest.skip('Auto chain sharding needs >1 core to want >1 device.')
    if bkw.num_chains is None or bkw.num_chains < 2:
        pytest.skip('Need >1 chain for chain sharding to kick in.')
    # Hand `Bart` a single device with auto chain sharding: it must cap the chain
    # axis at that device, not at the full jax cpu device count (pre-fix it took
    # the full count and `make_mesh` raised). Clearing the variants' explicit
    # data/chain device counts lets the single device suffice once it caps.
    kw = dict(
        bkw.kw,
        num_chain_devices='auto',
        num_data_devices=None,
        devices=get_default_device(),
    )
    with pytest.warns(UserWarning, match='passed in `devices`'):
        bart = Bart(**kw)
    mesh = bart._mcmc_state.config.mesh
    assert mesh is None or mesh.size == 1


def test_num_trees(bkw: BartKW, subtests: SubTests) -> None:
    """Test the number of trees."""
    kw = bkw.kw
    kw.update(n_burn=0, n_save=0)

    with subtests.test('given num_trees'):
        bart = Bart(**kw)
        assert bart.num_trees == kw['num_trees']

    with subtests.test('default num_trees'):
        kw2 = {k: v for k, v in kw.items() if k != 'num_trees'}
        bart = Bart(**kw2)
        assert bart.num_trees == 200


def test_dump_load_roundtrip(bkw: BartKW, tmp_path: Path) -> None:
    """`dump`/`load` preserve every array in the model, dropping only the mesh."""
    # keep `bkw.kw` unchanged so the MCMC reuses an already-compiled shape
    # rather than triggering a fresh (slower) compilation
    bart = Bart(**bkw.kw)

    path = tmp_path / 'bart.pkl'
    bart.dump(path)
    loaded = Bart.load(path)

    assert isinstance(loaded, Bart)
    # the device mesh and the explicit device are the only things dropped; the
    # reload is single-device
    assert loaded._mcmc_state.config.mesh is None
    assert loaded._device is None

    # every array must survive identically: values are gathered to host, not
    # recomputed. The mesh lives in the static tree structure, not in the
    # leaves, so it does not interfere with a leaf-by-leaf comparison. We
    # cannot use `tree.map_with_path` (as the sharding tests do) because an
    # equinox `Module` does not round-trip its static metadata to an identical
    # tree structure through pickle, so we compare the flattened (path, leaf)
    # pairs instead.
    original = tree.flatten_with_path(bart)[0]
    restored = tree.flatten_with_path(loaded)[0]
    assert len(original) == len(restored)
    for (path, leaf), (rpath, rleaf) in zip(original, restored, strict=True):
        assert path == rpath, f'structure mismatch: {keystr(path)} != {keystr(rpath)}'
        assert_array_equal(rleaf, leaf, err_msg=f'{keystr(path)}: ')


def test_load_wrong_type(tmp_path: Path) -> None:
    """`load` rejects a file that does not contain a `Bart`."""
    path = tmp_path / 'notbart.pkl'
    with path.open('wb') as file:
        pickle.dump([1, 2, 3], file)
    with pytest.raises(TypeError, match='not a Bart'):
        Bart.load(path)


class ExampleData(NamedTuple):
    """Small nonsense MV dataset for TestMVBartInterface."""

    x: Float32[Array, 'p n']
    y: Float32[Array, 'k n']
    w: Float32[Array, ' n']
    kwargs: dict[str, Any]


class TestMVBartInterface:
    """Some specific multivariate tests with their own data in and parametrization."""

    @pytest.fixture
    def example_data(self, keys: split) -> ExampleData:
        """Return a small nonsense MV dataset (x, y, w)."""
        return ExampleData(
            x=random.normal(keys.pop(), (2, 5)),
            y=random.normal(keys.pop(), (3, 5)),
            w=random.normal(keys.pop(), (5,)),
            kwargs=dict(num_trees=5, n_save=0, n_burn=0, num_chains=None),
        )

    @pytest.mark.parametrize('outcome_mode', ['continuous', 'mixed'])
    def test_scalar_params(
        self, example_data: ExampleData, subtests: SubTests, outcome_mode: str
    ) -> None:
        """Test that scalar configuration params are broadcasted."""
        x, y, _, kw = example_data
        k, _ = y.shape
        if outcome_mode == 'mixed':
            outcome_type = ['binary'] + ['continuous'] * (k - 1)
        else:
            outcome_type = 'continuous'
        kw.update(x_train=x, y_train=y, outcome_type=outcome_type)

        with subtests.test('offset'):
            bart = Bart(offset=0.0, **kw)
            assert bart.offset.shape == (k,)

        with subtests.test('sigma_scale'):
            bart = Bart(sigma_scale=1.0, **kw)
            assert nnone(bart._mcmc_state.error_cov_inv.rate).shape == (k, k)

        with subtests.test('sigma_init'):
            # no MCMC steps (n_burn = n_save = 0), so the state holds the init:
            # a precision of 1 / sigma_init**2 = 1 broadcast to all components
            bart = Bart(sigma_init=1.0, **kw)
            assert_array_equal(nnone(bart._mcmc_state.error_cov_inv.value), jnp.eye(k))

    def test_outcome_type_length_mismatch(self, example_data: ExampleData) -> None:
        """Sequence outcome_type with wrong length should raise."""
        x, y, _, kw = example_data
        with pytest.raises(ValueError, match='length'):
            Bart(x_train=x, y_train=y, **kw, outcome_type=['continuous', 'continuous'])

    def test_sequence_outcome_type_requires_2d(self, example_data: ExampleData) -> None:
        """Sequence outcome_type with 1D y should raise."""
        x, y, _, kw = example_data
        with pytest.raises(ValueError, match=r'y_train\.shape=\(1, n\)'):
            Bart(x_train=x, y_train=y[0, :], **kw, outcome_type=['continuous'])


def test_uv_mv_k1_equivalence(bkw: BartKW) -> None:
    """Test that Bart initializes equivalent states for UV and MV(k=1)."""
    outcome_type = bkw.kw['outcome_type']
    if isinstance(outcome_type, Sequence) and not isinstance(outcome_type, str):
        outcome_type = outcome_type[0]

    def to_uv_mv(arr: Shaped[Array, '...'] | None) -> tuple[Array | None, Array | None]:
        """Split a ``(n,)`` or ``(k, n)`` array into UV and MV(k=1) versions."""
        if arr is None:
            return None, None
        if arr.ndim == 1:
            arr = arr[None, :]
        arr_mv = arr[:1, :]
        return arr_mv.squeeze(0), arr_mv

    y_uv, y_mv = to_uv_mv(bkw.kw['y_train'])
    w_uv, w_mv = to_uv_mv(bkw.kw.get('error_scale'))
    missing_uv, missing_mv = to_uv_mv(bkw.kw.get('missing'))

    bkw.kw.update(outcome_type=outcome_type, n_burn=0, n_save=0)
    for key in ('y_train', 'error_scale', 'missing'):
        bkw.kw.pop(key, None)
    bart_uv = Bart(y_train=y_uv, error_scale=w_uv, missing=missing_uv, **bkw.kw)
    bart_mv = Bart(y_train=y_mv, error_scale=w_mv, missing=missing_mv, **bkw.kw)

    state_uv = bart_uv._mcmc_state
    state_mv = bart_mv._mcmc_state

    # Move chain axis to position 0 so squeezes target the k axis at -2 / -1.
    uv_axes = chain_vmap_axes(state_uv)
    mv_axes = chain_vmap_axes(state_mv)

    # Residuals and error covariance
    assert_close_matrices(
        chain_to_axis(state_uv.resid, uv_axes.resid),
        chain_to_axis(state_mv.resid, mv_axes.resid).squeeze(-2),
        rtol=1e-7,
    )
    assert_close_matrices(
        chain_to_axis(state_uv.error_cov_inv.value, uv_axes.error_cov_inv.value),
        chain_to_axis(
            state_mv.error_cov_inv.value, mv_axes.error_cov_inv.value
        ).squeeze((-2, -1)),
        rtol=1e-6,
    )

    # Prior parameters (scalar floats, only equal up to GPU rounding)
    assert_allclose(bart_uv.offset, bart_mv.offset.squeeze(0), rtol=1e-6)
    assert_allclose(
        nnone(state_uv.forest.leaf_prior_cov_inv),
        nnone(state_mv.forest.leaf_prior_cov_inv).reshape(()),
        rtol=1e-6,
    )
    if outcome_type == 'continuous':
        assert_array_equal(
            nnone(state_uv.error_cov_inv.nu), nnone(state_mv.error_cov_inv.nu)
        )
        assert_allclose(
            nnone(state_uv.error_cov_inv.rate),
            nnone(state_mv.error_cov_inv.rate).reshape(()),
            rtol=1e-6,
        )

    # Forest structure
    uv_forest_axes = chain_vmap_axes(state_uv.forest)
    mv_forest_axes = chain_vmap_axes(state_mv.forest)
    assert_array_equal(
        chain_to_axis(state_uv.forest.var_tree, uv_forest_axes.var_tree),
        chain_to_axis(state_mv.forest.var_tree, mv_forest_axes.var_tree),
    )
    assert_array_equal(
        chain_to_axis(state_uv.forest.split_tree, uv_forest_axes.split_tree),
        chain_to_axis(state_mv.forest.split_tree, mv_forest_axes.split_tree),
    )
    assert_array_equal(
        chain_to_axis(state_uv.forest.leaf_tree, uv_forest_axes.leaf_tree),
        chain_to_axis(state_mv.forest.leaf_tree, mv_forest_axes.leaf_tree).squeeze(-2),
    )
    assert_array_equal(
        chain_to_axis(state_uv.forest.leaf_indices, uv_forest_axes.leaf_indices),
        chain_to_axis(state_mv.forest.leaf_indices, mv_forest_axes.leaf_indices),
    )


def test_get_latent_prec_only_continuous(bkw: BartKW) -> None:
    """get_latent_prec(only_continuous=True) removes binary components."""
    kw = bkw.kw
    if bkw.k is None:  # pragma: no cover, bc defaults are mv only
        pytest.skip('UV variant')

    bart = Bart(**kw)
    if bkw.all_binary:
        with pytest.raises(ValueError, match='only binary'):
            bart.get_latent_prec(only_continuous=True)
        return

    prec = bart.get_latent_prec(only_continuous=True)
    kc = int(jnp.sum(~bkw.binary_mask))

    n_burn = kw['n_burn']
    if bkw.num_chains is not None:
        assert prec.shape == (bkw.num_chains, n_burn + bkw.n_save, kc, kc)
    else:
        assert prec.shape == (n_burn + bkw.n_save, kc, kc)


def test_get_error_sdev_values(bkw: BartKW) -> None:
    """get_error_sdev matches manual computation from precision matrices."""
    kw = bkw.kw
    if bkw.all_binary:
        pytest.skip('binary variant')
    bart = Bart(**kw)
    n_burn = kw['n_burn']

    with debug_nans(False):
        sdev = bart.get_error_sdev()
        if sdev.ndim == 1:  # pragma: no cover, bc defaults are mv only
            sdev = sdev[:, None]  # reshape as vector of length 1

    # manual: invert each precision matrix, take sqrt of diagonal
    prec = bart.get_latent_prec()
    if prec.ndim < 3:  # pragma: no cover, bc defaults are mv only
        prec = prec[..., :, None, None]  # reshape as 1x1 matrix
    prec = prec[..., n_burn:, :, :]  # skip burnin
    prec = lax.collapse(prec, 0, -2)  # flatten chains

    cov = jnp.linalg.inv(prec)
    sdev_ref = jnp.sqrt(jnp.diagonal(cov, axis1=-2, axis2=-1))

    # for mixed, compare only continuous components (binary have NaN sdev)
    mask = jnp.atleast_1d(~bkw.binary_mask)
    sdev = sdev[:, mask]
    sdev_ref = sdev_ref[:, mask]

    assert_close_matrices(sdev, sdev_ref, rtol=1e-5)


def test_devices_platform(bkw: BartKW) -> None:
    """Check that passing `devices='cpu'/'gpu'` ends up on the expected device."""
    bart1 = Bart(**bkw.kw)
    platform = bart1._main_trace.grow_prop_count.platform()  # ty: ignore[unresolved-attribute]
    kw2 = dict(bkw.kw, devices=platform)
    bart2 = Bart(**kw2)
    # `_device` records whether `devices` was passed explicitly, so it may
    # legitimately differ; the actual placement is compared leaf by leaf
    assert_identical_bart(bart1._drop_device_info(), bart2._drop_device_info())


def assert_identical_bart(bart1: OriginalBart, bart2: OriginalBart) -> None:
    """Check that two `Bart` objects are equal."""

    def check_same(
        path: KeyPath, x1: Shaped[Array, '*shape'], x2: Shaped[Array, '*shape']
    ) -> None:
        assert x1.shape == x2.shape
        assert x1.dtype == x2.dtype
        assert x1.sharding.is_equivalent_to(x2.sharding, x1.ndim)
        if jnp.issubdtype(x1.dtype, jnp.floating):
            rtol = 1e-4 if x1.platform() != 'cpu' else 1e-5  # ty: ignore[unresolved-attribute]
            assert_close_matrices(
                x1, x2, rtol=rtol, err_msg=keystr(path), reduce_rank=True
            )
        else:
            assert_array_equal(x1, x2, strict=True, err_msg=keystr(path))

    tree.map_with_path(check_same, bart1, bart2)

    treedef1 = tree.structure(bart1)
    treedef2 = tree.structure(bart2)
    assert treedef1 == treedef2


def test_numpy_input(bkw: BartKW) -> None:
    """Check if all numerical inputs are numpy arrays, everything works as usual."""
    bart1 = Bart(**bkw.kw)

    def to_numpy_array(
        x: Shaped[Array, '*shape'] | object,
    ) -> Shaped[numpy.ndarray, '*shape'] | object:
        if isinstance(x, (Array, float)) and not is_key(x):
            return numpy.asarray(x)
        else:
            return x

    kw2 = tree.map(to_numpy_array, bkw.kw)
    bart2 = Bart(**kw2)

    assert_identical_bart(bart1, bart2)


def assert_bart_on_device(bart: Bart, device: Device) -> None:
    """Check that the MCMC state and traces of ``bart`` live on ``device``."""

    def check(path: KeyPath, x: Shaped[Array, '...'] | object) -> None:
        if isinstance(x, Array):
            assert x.devices() == {device}, (
                f'{keystr(path)}: expected {device}, got {x.devices()}'
            )

    tree.map_with_path(check, (bart._mcmc_state, bart._main_trace, bart._burnin_trace))


class TestDevicePlacement:
    """Test how `Bart` selects the device for the result.

    Skipped if only one device is available.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_single_device(self) -> None:
        if get_device_count() < 2:  # this branch is covered in single cpu tests config
            pytest.skip('Need at least 2 devices for device placement tests')

    @pytest.fixture
    def other_device(self) -> Device:
        """Return a device different from the default one."""
        default = get_default_device()
        return next(d for d in get_default_devices() if d != default)

    @pytest.fixture
    def single_device_kw(self, bkw: BartKW) -> dict:
        """``bkw.kw`` with no device arguments."""
        return dict(bkw.kw, num_chain_devices=None, num_data_devices=None, devices=None)

    @pytest.fixture
    def committed_kw(self, single_device_kw: dict, other_device: Device) -> dict:
        """Like ``single_device_kw`` with all JAX inputs committed to ``other_device``."""

        def commit(x: Any) -> Any:  # noqa: ANN401
            if isinstance(x, Array):
                return jax.device_put(x, other_device)
            return x

        return tree.map(commit, single_device_kw)

    @pytest.fixture(params=['jax', 'numpy'])
    def uncommitted_kw(self, single_device_kw: dict, request: FixtureRequest) -> dict:
        """``single_device_kw`` with arrays as JAX or NumPy (both uncommitted)."""
        if request.param == 'jax':
            return single_device_kw

        def to_numpy(x: Any) -> Any:  # noqa: ANN401
            if isinstance(x, Array) and not is_key(x):
                return numpy.asarray(x)
            return x

        # numpy doesn't support jax keys; this test only checks device
        # placement, not values, so swapping the seed is safe
        return dict(tree.map(to_numpy, single_device_kw), seed=0)

    def test_uncommitted_default(self, uncommitted_kw: dict) -> None:
        """Uncommitted inputs put the result on the default device."""
        bart = Bart(**uncommitted_kw)
        assert_bart_on_device(bart, get_default_device())

    def test_uncommitted_with_devices(
        self, uncommitted_kw: dict, other_device: Device
    ) -> None:
        """With uncommitted inputs, ``devices`` selects the result device."""
        bart = Bart(**dict(uncommitted_kw, devices=other_device))
        assert_bart_on_device(bart, other_device)

    def test_committed_followed(self, committed_kw: dict, other_device: Device) -> None:
        """Committed inputs determine the result device when ``devices`` is unset."""
        bart = Bart(**committed_kw)
        assert_bart_on_device(bart, other_device)

    def test_committed_inconsistent_errors(
        self, single_device_kw: dict, other_device: Device
    ) -> None:
        """Committed inputs on different devices raise an error."""
        kw = dict(
            single_device_kw,
            x_train=jax.device_put(single_device_kw['x_train'], get_default_device()),
            y_train=jax.device_put(single_device_kw['y_train'], other_device),
        )
        with pytest.raises(ValueError, match='incompatible devices'):
            Bart(**kw)

    def test_devices_overrides_committed(self, committed_kw: dict) -> None:
        """When both committed inputs and ``devices`` are set, ``devices`` wins."""
        bart = Bart(**dict(committed_kw, devices=get_default_device()))
        assert_bart_on_device(bart, get_default_device())

    def test_predict_on_devices(
        self, bkw: BartKW, single_device_kw: dict, other_device: Device
    ) -> None:
        """`predict` places new test data on the device requested at fit time."""
        bart = Bart(**dict(single_device_kw, devices=other_device))
        w = bkw.w_for_predict('mean', on_train=False)
        pred = bart.predict(bkw.x_test, kind='mean', error_scale=w)
        assert pred.devices() == {other_device}


@pytest.mark.parametrize('param', ['sigma_scale', 'sigma_init'])
def test_sigma_wrong_special_value(bkw: BartKW, param: str) -> None:
    """Trigger error on unrecognized `sigma_scale`/`sigma_init` value."""
    value = 'ohohoh'
    if bkw.all_binary:
        pytest.skip('Parameter ignored with binary outcomes.')
    kw = dict(bkw.kw, **{param: value})
    # the import-hook type checker would reject the invalid literal before
    # `Bart` raises its own `ValueError`, so disable it here
    with jaxtyping_disabled(), pytest.raises(ValueError, match=value):
        Bart(**kw)
