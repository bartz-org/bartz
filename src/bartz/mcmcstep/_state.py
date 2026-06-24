# bartz/src/bartz/mcmcstep/_state.py
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

"""Module defining the BART MCMC state and initialization."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from enum import Enum
from functools import partial, wraps
from typing import Literal, TypedDict, TypeVar, cast

import jax
import numpy
from equinox import error_if, filter_jit
from jax import NamedSharding, device_put, lax, make_mesh, random, shard_map, tree, vmap
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.sharding import AxisType, Mesh, PartitionSpec
from jaxtyping import (
    Array,
    Bool,
    Float,
    Float32,
    Int32,
    Integer,
    Key,
    PyTree,
    Shaped,
    UInt,
    UInt32,
)
from numpy import ndarray

from bartz._jaxext import Module, field, jaxtyping_disabled, jit, minimal_unsigned_dtype
from bartz.grove import tree_depths
from bartz.mcmcstep._axes import CHAIN_AXIS, chain_vmap_axes, data_vmap_axes
from bartz.mcmcstep._lazy import (
    _is_lazy_or_none,
    _lazy,
    _lazy_from_array,
    _LazyArray,
    _wrap_chain,
    add_dummy_axis,
)
from bartz.mcmcstep._reduction import BatchedReduction, ReductionConfig

ArrayLike = Array | ndarray

FloatLike = float | Float[ArrayLike, '']


class OutcomeType(Enum):
    """Likelihood types for each outcome component in the regression."""

    continuous = 'continuous'
    """Continuous outcome with Normal error."""

    binary = 'binary'
    """Binary outcome in {0, 1} with probit link."""


T = TypeVar('T')


class Wishart(Module):
    """A precision matrix with a Wishart prior, bundled with its current value.

    Represents a random precision (inverse covariance) ``value`` drawn from a
    Wishart prior with degrees of freedom `nu` and rate matrix `rate`. The
    univariate case (``k = 1``) is the Gamma special case; the relationship to
    the inverse-gamma prior on the variance is ``alpha = nu / 2``,
    ``beta = rate / 2``. The prior mean of the precision is ``nu * rate^-1``.

    Set `nu` and `rate` to `None` to represent a precision held fixed at `value`
    with no prior (e.g. the identity in binary regression).
    """

    nu: Float32[Array, ''] | None
    """Degrees of freedom of the Wishart prior, or `None` if there is no prior."""

    rate: Float32[Array, ''] | Float32[Array, 'k k'] | None
    """The rate matrix of the Wishart prior (scalar for univariate), or `None`
    if there is no prior. Equal to the inverse-gamma ``scale`` in the
    univariate case."""

    value: Float32[Array, '*chains k k'] | Float32[Array, '*chains'] = field(
        chains=CHAIN_AXIS
    )
    """The precision matrix (scalar for univariate)."""

    def __init__(
        self,
        nu: FloatLike | None,
        rate: FloatLike | Float[ArrayLike, 'k k'] | None,
        value: FloatLike
        | Float[ArrayLike, '*chains k k']
        | Float[ArrayLike, '*chains'],
    ) -> None:
        # `init` passes a deferred `_LazyArray` (cast to `Array`) for `value` to
        # route it through sharding.
        assert (nu is None) == (rate is None), 'set both or neither of nu and rate'
        self.nu = None if nu is None else jnp.asarray(nu, jnp.float32)
        self.rate = None if rate is None else jnp.asarray(rate, jnp.float32)
        if isinstance(value, _LazyArray):
            self.value = cast(Array, value)
        else:
            self.value = jnp.asarray(value, jnp.float32)


class DiagWishart(Wishart):
    """A diagonal precision matrix with independent chi-square diagonal entries.

    Despite the name this is not a Wishart restricted to diagonal matrices, but
    a convenience type: a diagonal precision whose entries are mutually
    independent, each with its own Gamma (scaled chi-square) prior. Only the
    multivariate (matrix) case is supported.

    A component with `rate` 0 has no prior; its precision is held fixed at its
    `value` (1 for the binary components of a mixed regression).

    Used for mixed binary-continuous regression and for continuous multivariate
    regression with per-datapoint missingness.
    """

    def __init__(
        self,
        nu: FloatLike | None,
        rate: FloatLike | Float[ArrayLike, 'k k'] | None,
        value: FloatLike
        | Float[ArrayLike, '*chains k k']
        | Float[ArrayLike, '*chains'],
    ) -> None:
        # explicit (delegating) init so the static checker uses this signature
        # instead of synthesizing a stricter one from the inherited fields
        assert rate is None or jnp.ndim(rate) == 2, (
            'DiagWishart supports only the multivariate (matrix) case'
        )
        super().__init__(nu, rate, value)


class Forest(Module):
    """Represents the MCMC state of a sum of trees."""

    # Heap-array fields follow the `bartz.grove.TreesTrace` convention: the
    # union-free integer trees are declared before `leaf_tree` and carry the
    # bindable `half_tree_size` axis, while `leaf_tree` (and `p_nonterminal`) are
    # checked against `2*half_tree_size`. Declaring a union-free `*chains` field
    # first binds the variadic chain axis (plus `num_trees` and `half_tree_size`)
    # before `leaf_tree`'s `... | ... k ...` union is evaluated, so the runtime
    # typechecker can't mis-bind `*chains` against the `k` axis of a multivariate
    # forest (the layouts are otherwise rank-ambiguous). No anchor field needed.
    var_tree: UInt[Array, '*chains num_trees half_tree_size'] = field(chains=CHAIN_AXIS)
    """The decision axes."""

    split_tree: UInt[Array, '*chains num_trees half_tree_size'] = field(
        chains=CHAIN_AXIS
    )
    """The decision boundaries."""

    affluence_tree: Bool[Array, '*chains num_trees half_tree_size'] = field(
        chains=CHAIN_AXIS
    )
    """Marks leaves that can be grown."""

    leaf_tree: (
        Float32[Array, '*chains num_trees 2*half_tree_size']
        | Float32[Array, '*chains num_trees k 2*half_tree_size']
    ) = field(chains=CHAIN_AXIS)
    """The leaf values."""

    grow_prop_count: Int32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    """The number of grow proposals made during one full MCMC cycle."""

    prune_prop_count: Int32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    """The number of prune proposals made during one full MCMC cycle."""

    grow_acc_count: Int32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    """The number of grow moves accepted during one full MCMC cycle."""

    prune_acc_count: Int32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    """The number of prune moves accepted during one full MCMC cycle."""

    max_split: UInt[Array, ' p']
    """The maximum split index for each predictor."""

    blocked_vars: UInt[Array, ' q'] | None
    """Indices of variables that are not used. This shall include at least
    the `i` such that ``max_split[i] == 0``, otherwise behavior is
    undefined."""

    p_nonterminal: Float32[Array, ' 2*half_tree_size']
    """The prior probability of each node being nonterminal, conditional on
    its ancestors. Includes the nodes at maximum depth which should be set
    to 0."""

    p_propose_grow: Float32[Array, ' half_tree_size']
    """The unnormalized probability of picking a leaf for a grow proposal."""

    leaf_indices: UInt[Array, '*chains num_trees n'] = field(chains=CHAIN_AXIS, data=-1)
    """The index of the leaf each datapoints falls into, for each tree."""

    count_tree: UInt32[Array, '*chains num_trees 2*half_tree_size'] | None = field(
        chains=CHAIN_AXIS
    )
    """The number of datapoints per leaf. Valid at the leaves and at the nodes
    involved in the latest moves, dirty elsewhere. `None` if the error
    precision is weighted and there are no minimum-points-per-node
    constraints, which makes the counts unused."""

    prec_tree: (
        Float32[Array, '*chains num_trees 2*half_tree_size']
        | Float32[Array, '*chains num_trees k k 2*half_tree_size']
        | None
    ) = field(chains=CHAIN_AXIS)
    """The likelihood precision scale summed over the datapoints in each leaf.
    Valid at the leaves and at the nodes involved in the latest moves, dirty
    elsewhere. `None` if the error precision is not weighted, in which case
    `count_tree` takes its place."""

    min_points_per_decision_node: Int32[Array, ''] | None
    """The minimum number of data points in a decision node."""

    min_points_per_leaf: Int32[Array, ''] | None
    """The minimum number of data points in a leaf node."""

    log_trans_prior: Float32[Array, '*chains num_trees'] | None = field(
        chains=CHAIN_AXIS
    )
    """The log transition and prior Metropolis-Hastings ratio for the
    proposed move on each tree."""

    log_likelihood: Float32[Array, '*chains num_trees'] | None = field(
        chains=CHAIN_AXIS
    )
    """The log likelihood ratio."""

    leaf_prior_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'] | None
    """The prior precision matrix of a leaf, conditional on the tree structure.
    For the univariate case (k=1), this is a scalar (the inverse variance).
    The prior covariance of the sum of trees is
    ``num_trees * leaf_prior_cov_inv^-1``."""

    log_s: Float32[Array, '*chains p'] | None = field(chains=CHAIN_AXIS)
    """The logarithm of the prior probability for choosing a variable to split
    along in a decision rule, conditional on the ancestors. Not normalized.
    If `None`, use a uniform distribution."""

    theta: Float32[Array, '*chains'] | None = field(chains=CHAIN_AXIS)
    """The concentration parameter for the Dirichlet prior on the variable
    distribution `s`. Required only to update `log_s`."""

    a: Float32[Array, ''] | None
    """Parameter of the prior on `theta`. Required only to sample `theta`."""

    b: Float32[Array, ''] | None
    """Parameter of the prior on `theta`. Required only to sample `theta`."""

    rho: Float32[Array, ''] | None
    """Parameter of the prior on `theta`. Required only to sample `theta`."""

    @property
    def has_chains(self) -> bool:
        """Whether this forest carries an explicit chain axis."""
        return self.var_tree.ndim > 2


class StepConfig(Module):
    """Options for the MCMC step."""

    steps_done: Int32[Array, '']
    """The number of MCMC steps completed so far."""

    sparse_on_at: Int32[Array, ''] | None
    """After how many steps to turn on variable selection."""

    resid_reduction_config: ReductionConfig
    """How to sum the residuals in each leaf."""

    count_reduction_config: ReductionConfig
    """How to count the datapoints in each leaf."""

    prec_reduction_config: ReductionConfig
    """How to sum the likelihood precisions in each leaf."""

    prec_count_num_trees: int | None = field(static=True)
    """Batch size for processing trees to compute count and prec trees."""

    sequential_unroll: int | bool = field(static=True)
    """How much to unroll the sequential accept/reject loop over trees in
    `step`. See the ``unroll`` argument of `jax.lax.scan`."""

    mesh: Mesh | None = field(static=True)
    """The mesh used to shard data and computation across multiple devices."""

    @property
    def data_sharded(self) -> bool:
        """Whether the data axis is sharded across devices."""
        return self.mesh is not None and 'data' in self.mesh.axis_names


class State(Module):
    """Represents the MCMC state of BART."""

    _chain_anchor: Float32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    """Unused per-chain scalar, declared first as a runtime-typechecker anchor.
    Its single (union-free) ``*chains`` annotation binds the variadic chain axis
    before the ``... | ... k ...`` unions of `z`/`resid` (z over the
    binary-outcome ``kb`` axis) are checked; otherwise those can mis-bind
    ``*chains`` against the outcome axis for a multivariate-without-chains state
    (the layouts are rank-ambiguous). Unlike `Forest`, `State` has no genuine
    union-free chain field to reorder into this slot, so a dummy one is
    carried."""

    X: UInt[Array, 'p n'] = field(data=-1)
    """The predictors."""

    binary_y: None | Bool[Array, ' n'] | Bool[Array, 'kb n'] = field(data=-1)
    """The response as booleans for binary regression, `None` for continuous.
    In the mixed binary-continuous case, only the binary outcome components
    are stored, with shape ``(kb, n)``."""

    z: None | Float32[Array, '*chains n'] | Float32[Array, '*chains kb n'] = field(
        chains=CHAIN_AXIS, data=-1
    )
    """The latent variable for binary regression. `None` in continuous
    regression. In the mixed binary-continuous case, only the binary outcome
    components are stored, with shape ``(*chains, kb, n)``."""

    binary_indices: None | Int32[Array, ' kb']
    """The indices of binary outcome components in the full list of outcome
    components. `None` when there are no binary components."""

    offset: Float32[Array, ''] | Float32[Array, ' k']
    """Constant shift added to the sum of trees."""

    resid: Float32[Array, '*chains n'] | Float32[Array, '*chains k n'] = field(
        chains=CHAIN_AXIS, data=-1
    )
    """The residuals (`y` or `z` minus sum of trees)."""

    error_cov_inv: Wishart
    """The inverse error covariance with its Wishart prior. The current value is
    ``error_cov_inv.value`` (scalar for univariate, matrix for multivariate);
    identity with no prior in binary regression."""

    prec_scale: Float32[Array, ' n'] | Float32[Array, 'k k n'] | None = field(data=-1)
    """The scale on the error precision. `None` in binary regression. With
    scalar per-datapoint weights, shape ``(n,)`` and value
    ``1 / error_scale ** 2``. With vector per-datapoint weights, shape ``(k, k, n)``
    and value ``1/outer(error_scale, error_scale)`` repeated over datapoints."""

    inv_sdev_scale: Float32[Array, ' n'] | Float32[Array, 'k n'] | None = field(data=-1)
    """The reciprocal of the per-observation error standard-deviation scale.
    `None` in binary regression. Shape ``(n,)`` for scalar weights, or
    ``(k, n)`` for per-component vector weights."""

    forest: Forest
    """The sum of trees model."""

    config: StepConfig
    """Metadata and configurations for the MCMC step."""

    @property
    def has_chains(self) -> bool:
        """Whether this state carries an explicit chain axis."""
        return self.forest.has_chains

    def num_chains(self) -> int | None:
        """Return the number of chains, or `None` if not multichain."""
        if not self.has_chains:
            return None
        c = chain_vmap_axes(self.forest).var_tree
        return self.forest.var_tree.shape[c]


def _check_diagonal(rate: Float32[Array, 'k k']) -> Float32[Array, 'k k']:
    """Raise if the Wishart `rate` is not diagonal."""
    diag = jnp.diag(jnp.diag(rate))
    return error_if(rate, jnp.any(rate != diag), 'error_cov_inv.rate must be diagonal')


def _check_binary_unit_precision(
    value: Float32[Array, 'k k'], binary_mask: Sequence[bool]
) -> Float32[Array, 'k k']:
    """Raise if the binary diagonal entries of `value` are not fixed at 1."""
    binary = jnp.array(binary_mask)
    off_unit = jnp.any(binary & (jnp.diag(value) != 1.0))
    return error_if(
        value,
        off_unit,
        'binary error precision must be 1 (the default for a zero rate)',
    )


def _init_shape_shifting_parameters(
    y: Float32[Array, ' n'] | Float32[Array, 'k n'],
    outcome_type: OutcomeType | list[OutcomeType],
    offset: Float32[Array, ''] | Float32[Array, ' k'],
    error_scale: Float32[ArrayLike, ' n'] | Float32[ArrayLike, 'k n'] | None,
    error_cov_inv: Wishart | None,
    leaf_prior_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'],
    missing: Bool[ArrayLike, ' n'] | Bool[ArrayLike, 'k n'] | None,
) -> tuple[bool, tuple[int, ...], Wishart, None | Int32[Array, ' kb']]:
    """
    Check and initialize parameters that change array type/shape based on outcome kind.

    Parameters
    ----------
    y
        The response variable (used only for shape checks).
    outcome_type
        Whether the regression is continuous or binary. Can be a list of
        `OutcomeType` for per-component specification in the multivariate case.
    offset
        The offset to add to the predictions.
    error_scale
        Per-observation error scale (univariate only).
    error_cov_inv
        The Wishart prior on the error precision and its initial value, or
        `None` for binary regression. The mixed and partial-missing diagonal
        modes require a `DiagWishart`; in the mixed case the binary components
        must have an initial precision of 1.
    leaf_prior_cov_inv
        The inverse of the leaf prior covariance.
    missing
        The per-datapoint missingness mask, used to detect partial missingness
        (2-D mask) so that diagonal-mode initialization is selected.

    Returns
    -------
    is_binary
        Whether all outcomes are binary.
    kshape
        The outcome shape, empty for univariate, (k,) for multivariate.
    error_cov_inv
        The Wishart prior with its initial value resolved for the outcome kind.
    binary_indices
        The indices of binary outcome components, or `None` if there are none.
    """
    kshape = offset.shape

    # determine per-component outcome kinds
    if isinstance(outcome_type, list):
        assert kshape, 'per-component outcome_type requires multivariate y'
        (k,) = kshape
        assert len(outcome_type) == k
        binary_mask = [t is OutcomeType.binary for t in outcome_type]
        is_binary = all(binary_mask)
        is_mixed = any(binary_mask) and not is_binary
    else:
        is_binary = outcome_type is OutcomeType.binary
        is_mixed = False

    if is_mixed:
        binary_indices = jnp.array([i for i, b in enumerate(binary_mask) if b])
    else:
        binary_indices = None

    partial_missing = missing is not None and missing.ndim == 2 and kshape

    # All-binary: no prior, the precision is fixed at the identity.
    if is_binary:
        assert error_scale is None
        assert error_cov_inv is None, 'no error covariance prior in binary regression'
        value = jnp.eye(kshape[0]) if kshape else jnp.array(1.0)
        error_cov_inv = Wishart(nu=None, rate=None, value=value)

    # Mixed binary-continuous, or continuous-mv with 2-D missingness: diagonal
    # error covariance, updated component-wise. The caller must supply a
    # `DiagWishart`; in the mixed case the binary components must have unit
    # initial precision (see `DiagWishart`).
    elif is_mixed or partial_missing:
        if is_mixed:
            assert error_scale is None
        assert isinstance(error_cov_inv, DiagWishart), (
            'mixed binary-continuous or partial-missing regression requires a '
            'DiagWishart error_cov_inv prior'
        )
        assert error_cov_inv.rate is not None
        assert error_cov_inv.rate.shape == 2 * kshape
        assert error_cov_inv.value.shape == 2 * kshape
        rate = _check_diagonal(error_cov_inv.rate)
        value = _check_diagonal(error_cov_inv.value)
        if is_mixed:
            value = _check_binary_unit_precision(value, binary_mask)
        error_cov_inv = replace(error_cov_inv, rate=rate, value=value)

    # All-continuous: a dense `Wishart`.
    else:
        assert (
            error_scale is None
            or error_scale.shape == y.shape  # (k, n)
            or error_scale.shape == y.shape[-1:]  # (n,)
        )
        assert error_cov_inv is not None
        assert type(error_cov_inv) is Wishart, (
            'continuous regression requires a dense Wishart error_cov_inv prior'
        )
        rate = error_cov_inv.rate
        assert rate is not None
        assert rate.shape == 2 * kshape
        assert error_cov_inv.value.shape == 2 * kshape

    assert y.shape[:-1] == kshape
    assert leaf_prior_cov_inv.shape == 2 * kshape

    return is_binary, kshape, error_cov_inv, binary_indices


def _check_splitless_vars(
    filter_splitless_vars: int,
    max_split: UInt[Array, ' p'],
    offset: Float32[Array, ''] | Float32[Array, ' k'],
) -> Float32[Array, ''] | Float32[Array, ' k']:
    """Check there aren't too many deactivated predictors."""
    msg = (
        f'there are more than {filter_splitless_vars=} predictors with no splits, '
        'please increase `filter_splitless_vars` or investigate the missing splits'
    )
    return error_if(offset, jnp.sum(max_split == 0) > filter_splitless_vars, msg)


def _parse_outcome_type(
    outcome_type: 'OutcomeType | str | Sequence[OutcomeType | str]',
) -> 'OutcomeType | list[OutcomeType]':
    """Normalize outcome_type to enum (or list of enums)."""
    if isinstance(outcome_type, Sequence) and not isinstance(outcome_type, str):
        return [OutcomeType(t) for t in outcome_type]
    else:
        return OutcomeType(outcome_type)


def _parse_p_nonterminal(
    p_nonterminal: Float32[ArrayLike, ' d_minus_1'],
) -> Float32[Array, ' d_minus_1+1']:
    """Check it's in (0, 1) and pad with a 0 at the end."""
    p_nonterminal = jnp.asarray(p_nonterminal)
    ok = (p_nonterminal > 0) & (p_nonterminal < 1)
    p_nonterminal = error_if(p_nonterminal, ~ok, 'p_nonterminal must be in (0, 1)')
    return jnp.pad(p_nonterminal, (0, 1))


def make_p_nonterminal(
    d: int, alpha: FloatLike = 0.95, beta: FloatLike = 2.0
) -> Float32[Array, ' {d}-1']:
    """Prepare the `p_nonterminal` argument to `init`.

    It is calculated according to the formula:

        P_nt(depth) = alpha / (1 + depth)^beta,     with depth 0-based

    Parameters
    ----------
    d
        The maximum depth of the trees (d=1 means tree with only root node)
    alpha
        The a priori probability of the root node having children, conditional
        on it being possible
    beta
        The exponent of the power decay of the probability of having children
        with depth.

    Returns
    -------
    An array of probabilities, one per tree level but the last.
    """
    assert d >= 1
    depth = jnp.arange(d - 1)
    return alpha / (1 + depth).astype(float) ** beta


def init(
    *,
    X: UInt[ArrayLike, 'p n'],
    y: Float32[ArrayLike, ' n'] | Float32[ArrayLike, ' k n'],
    outcome_type: OutcomeType | str | Sequence[OutcomeType | str] = 'continuous',
    offset: FloatLike | Float[ArrayLike, ' k'],
    max_split: UInt[ArrayLike, ' p'],
    num_trees: int,
    p_nonterminal: Float32[ArrayLike, ' d_minus_1'],
    leaf_prior_cov_inv: FloatLike | Float[ArrayLike, 'k k'],
    error_cov_inv: Wishart | None = None,
    error_scale: Float32[ArrayLike, ' n'] | Float32[ArrayLike, 'k n'] | None = None,
    missing: Bool[ArrayLike, ' n'] | Bool[ArrayLike, 'k n'] | None = None,
    min_points_per_decision_node: int | Integer[ArrayLike, ''] | None = None,
    # the gpu batch targets of the default reductions were tuned on an A4000
    resid_reduction_config: ReductionConfig = BatchedReduction(auto_gpu_target=1024),
    count_reduction_config: ReductionConfig = BatchedReduction(auto_gpu_target=2048),
    prec_reduction_config: ReductionConfig = BatchedReduction(auto_gpu_target=1024),
    prec_count_num_trees: int | None | Literal['auto'] = 'auto',
    sequential_unroll: int | bool = 2,
    save_ratios: bool = False,
    filter_splitless_vars: int = 0,
    min_points_per_leaf: int | Integer[ArrayLike, ''] | None = None,
    log_s: Float32[ArrayLike, ' p'] | None = None,
    theta: FloatLike | None = None,
    a: FloatLike | None = None,
    b: FloatLike | None = None,
    rho: FloatLike | None = None,
    sparse_on_at: int | Integer[ArrayLike, ''] | None = None,
    num_chains: int | None = None,
    mesh: Mesh | dict[str, int] | None = None,
) -> State:
    """
    Make a BART posterior sampling MCMC initial state.

    Parameters
    ----------
    X
        The predictors. Note this is trasposed compared to the usual convention.
    y
        The response. If two-dimensional, the outcome is multivariate with the
        first axis indicating the component. For binary data, non-zero means 1,
        zero means 0.
    outcome_type
        Whether the regression is continuous or binary (probit). Can also be a
        sequence of `OutcomeType` values, one per outcome component, for mixed
        binary-continuous multivariate regression.
    offset
        Constant shift added to the sum of trees. 0 if not specified.
    max_split
        The maximum split index for each variable. All split ranges start at 1.
    num_trees
        The number of trees in the forest.
    p_nonterminal
        The probability of a nonterminal node at each depth. The maximum depth
        of trees is fixed by the length of this array. Use `make_p_nonterminal`
        to set it with the conventional formula.
    leaf_prior_cov_inv
        The prior precision matrix of a leaf, conditional on the tree structure.
        For the univariate case (k=1), this is a scalar (the inverse variance).
        The prior covariance of the sum of trees is
        ``num_trees * leaf_prior_cov_inv^-1``. The prior mean of leaves is
        always zero.
    error_cov_inv
        The Wishart prior on the inverse error covariance, together with its
        initial value (see `Wishart`). Leave it unspecified for binary
        regression. The mixed binary-continuous and partial-missing diagonal
        modes require a `DiagWishart`; in the mixed case the binary components
        must have an initial precision of 1 (see `DiagWishart`).
    error_scale
        Each error is scaled by the corresponding factor in `error_scale`. If
        ``error_scale[..., i]`` is a scalar, each error variance or covariance
        matrix is multiplied by ``error_scale[..., i] ** 2``. If
        ``error_scale[:, i]`` is a vector, then the covariance matrix is
        rescaled by its outer product. Not supported for binary or mixed
        binary-continuous regression. If not specified, defaults to 1 for all
        points, but potentially skipping calculations.
    missing
        Boolean mask, same shape as `y`; `True` marks entries to be ignored
        by the MCMC. Values of `y` must be finite everywhere, including at
        masked positions. If 2-D, `error_cov_inv.rate` must be diagonal.
    min_points_per_decision_node
        The minimum number of data points in a decision node. 0 if not
        specified.
    resid_reduction_config
    count_reduction_config
    prec_reduction_config
        How to sum the residuals, count the datapoints, and sum the likelihood
        precisions in each leaf, respectively. See `ReductionConfig` and its
        subclasses. The defaults pick the batching automatically based on the
        platform the MCMC is compiled for, resolved at run time, so the state
        need not know the platform in advance.
    prec_count_num_trees
        The number of trees to process at a time when counting datapoints or
        computing the likelihood precision. If `None`, do all trees at once,
        which may use too much memory. If 'auto' (default), it's chosen
        automatically.
    sequential_unroll
        How much to unroll the sequential accept/reject loop over trees in
        `step`. See the ``unroll`` argument of `jax.lax.scan`. Unrolling may
        speed up the MCMC at the cost of longer compilation. 1 means no
        unrolling; the default is 2.
    save_ratios
        Whether to save the Metropolis-Hastings ratios.
    filter_splitless_vars
        The maximum number of variables without splits that can be ignored. If
        there are more, `init` raises an exception.
    min_points_per_leaf
        The minimum number of datapoints in a leaf node. 0 if not specified.
        Unlike `min_points_per_decision_node`, this constraint is not taken into
        account in the Metropolis-Hastings ratio because it would be expensive
        to compute. Grow moves that would violate this constraint are vetoed.
        This parameter is independent of `min_points_per_decision_node` and
        there is no check that they are coherent. It makes sense to set
        ``min_points_per_decision_node >= 2 * min_points_per_leaf``.
    log_s
        The logarithm of the prior probability for choosing a variable to split
        along in a decision rule, conditional on the ancestors. Not normalized.
        If not specified, use a uniform distribution. If not specified and
        `theta` or `rho`, `a`, `b` are, it's initialized automatically.
    theta
        The concentration parameter for the Dirichlet prior on `s`. Required
        only to update `log_s`. If not specified, and `rho`, `a`, `b` are
        specified, it's initialized automatically.
    a
    b
    rho
        Parameters of the prior on `theta`. Required only to sample `theta`.
    sparse_on_at
        After how many MCMC steps to turn on variable selection.
    num_chains
        The number of independent MCMC chains to represent in the state. Single
        chain with scalar values if not specified.
    mesh
        A jax mesh used to shard data and computation across multiple devices.
        If it has a 'chains' axis, that axis is used to shard the chains. If it
        has a 'data' axis, that axis is used to shard the datapoints.

        As a shorthand, if a dictionary mapping axis names to axis size is
        passed, the corresponding mesh is created, e.g., ``dict(chains=4,
        data=2)`` will let jax pick 8 devices to split chains (which must be a
        multiple of 4) across 4 pairs of devices, where in each pair the data is
        split in two.

        Note: if a mesh is passed, the arrays are always sharded according to
        it. In particular even if the mesh has no 'chains' or 'data' axis, the
        arrays will be replicated on all devices in the mesh.

    Returns
    -------
    An initialized BART MCMC state.

    Raises
    ------
    ValueError
        If arguments unused in binary regression are set.

    Notes
    -----
    In decision nodes, the values in ``X[i, :]`` are compared to a cutpoint out
    of the range ``[1, 2, ..., max_split[i]]``. A point belongs to the left
    child iff ``X[i, j] < cutpoint``. Thus it makes sense for ``X[i, :]`` to be
    integers in the range ``[0, 1, ..., max_split[i]]``.

    In general the arrays passed to this function as arguments may be donated,
    invalidating them. Create copies before passing them to `init` if this
    happens and you need them again.
    """
    # convert to array all array-like arguments that are used in other
    # configurations but don't need further processing themselves
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    assert y.dtype == jnp.float32
    offset = jnp.asarray(offset)
    leaf_prior_cov_inv = jnp.asarray(leaf_prior_cov_inv)
    max_split = jnp.asarray(max_split)
    if error_scale is not None:
        error_scale = jnp.asarray(error_scale)
    if missing is not None:
        missing = jnp.asarray(missing)
    assert missing is None or missing.ndim <= y.ndim

    # normalize outcome_type to enum (or list of enums)
    outcome_type = _parse_outcome_type(outcome_type)

    # check p_nonterminal and pad it with a 0 at the end (still not final shape)
    p_nonterminal = _parse_p_nonterminal(p_nonterminal)

    # process arguments that change depending on outcome type
    is_binary, kshape, error_cov_inv, binary_indices = _init_shape_shifting_parameters(
        y, outcome_type, offset, error_scale, error_cov_inv, leaf_prior_cov_inv, missing
    )

    # extract array sizes from arguments
    (max_depth,) = p_nonterminal.shape
    p, n = X.shape

    # check and initialize sparsity parameters
    if not _all_none_or_not_none(rho, a, b):
        msg = 'rho, a, b are not either all `None` or all set'
        raise ValueError(msg)
    if theta is None and rho is not None:
        theta = rho
    if log_s is None and theta is not None:
        log_s = jnp.zeros(max_split.size)
    if not _all_none_or_not_none(theta, sparse_on_at):
        msg = 'sparsity params (either theta or rho,a,b) and sparse_on_at must be either all None or all set'
        raise ValueError(msg)

    # determine settings for reductions
    mesh = _parse_mesh(num_chains, mesh)
    red_cfg = _parse_reduction_configs(
        resid_reduction_config,
        count_reduction_config,
        prec_reduction_config,
        prec_count_num_trees,
        y,
        num_trees,
        num_chains,
        mesh,
    )

    # check there aren't too many deactivated predictors
    offset = _check_splitless_vars(filter_splitless_vars, max_split, offset)

    tree_size = 2**max_depth

    # Assemble the state, shard it, then fill in the post-shard fields. This
    # whole region runs with type-checking disabled because the state carries
    # deliberately wrong-typed intermediates parked in its fields for sharding:
    # `_LazyArray` leaves (each chain-bearing leaf is built at its core no-chain
    # shape, then `_add_chains` wraps it to broadcast in the chain axis), the raw
    # float `y` in the bool `binary_y` slot, and the user `missing` mask and
    # `error_scale` in the scale slots. The context ends once every field has
    # been replaced by its final, correctly-typed array.
    with jaxtyping_disabled():
        state = State(
            _chain_anchor=_lazy(jnp.zeros, ()),  # typechecker chain anchor
            X=X,
            binary_y=y,  # temporary to be sharded together with everything else
            z=(
                _lazy(jnp.full, y.shape, offset[..., None])
                if is_binary
                else _lazy(
                    jnp.full, (binary_indices.size, n), offset[binary_indices, None]
                )
                if binary_indices is not None
                else None
            ),
            binary_indices=binary_indices,
            offset=offset,
            resid=(
                _lazy(jnp.zeros, y.shape)
                if is_binary
                # resid is created later after y and offset are sharded
                else cast(Array, None)
            ),
            # only `value` carries the chain axis, so it becomes the lazy leaf;
            # the prior params `nu`/`rate` are shared across chains
            error_cov_inv=replace(
                error_cov_inv, value=_lazy_from_array(error_cov_inv.value)
            ),
            # temporarily store user inputs in these slots so they get sharded
            # with everything else; `_compute_scales` replaces them post-shard.
            prec_scale=error_scale,
            inv_sdev_scale=missing,
            forest=Forest(
                leaf_tree=_lazy(
                    jnp.zeros, (num_trees, *kshape, tree_size), jnp.float32
                ),
                var_tree=_lazy(
                    jnp.zeros,
                    (num_trees, tree_size // 2),
                    minimal_unsigned_dtype(p - 1),
                ),
                split_tree=_lazy(
                    jnp.zeros, (num_trees, tree_size // 2), max_split.dtype
                ),
                affluence_tree=_lazy(
                    _initial_affluence_tree,
                    (num_trees, tree_size // 2),
                    n,
                    min_points_per_decision_node,
                ),
                blocked_vars=_get_blocked_vars(filter_splitless_vars, max_split),
                max_split=max_split,
                grow_prop_count=_lazy(jnp.zeros, (), int),
                grow_acc_count=_lazy(jnp.zeros, (), int),
                prune_prop_count=_lazy(jnp.zeros, (), int),
                prune_acc_count=_lazy(jnp.zeros, (), int),
                p_nonterminal=p_nonterminal[tree_depths(tree_size)],
                p_propose_grow=p_nonterminal[tree_depths(tree_size // 2)],
                leaf_indices=_lazy(
                    jnp.ones, (num_trees, n), minimal_unsigned_dtype(tree_size - 1)
                ),
                # the counts serve the minimum-points constraints and stand in
                # for the precisions when the error precision is unweighted
                # (`prec_scale` is set iff `error_scale` or `missing` is given)
                count_tree=(
                    _lazy(_initial_count_tree, (num_trees, tree_size), n)
                    if min_points_per_decision_node is not None
                    or min_points_per_leaf is not None
                    or (error_scale is None and missing is None)
                    else None
                ),
                # prec_tree is created later, it needs the sharded prec_scale
                prec_tree=None,
                min_points_per_decision_node=_asarray_or_none(
                    min_points_per_decision_node
                ),
                min_points_per_leaf=_asarray_or_none(min_points_per_leaf),
                log_trans_prior=_lazy(jnp.zeros, (num_trees,)) if save_ratios else None,
                log_likelihood=_lazy(jnp.zeros, (num_trees,)) if save_ratios else None,
                leaf_prior_cov_inv=leaf_prior_cov_inv,
                log_s=_lazy_from_array(_asarray_or_none(log_s)),
                theta=_lazy_from_array(_asarray_or_none(theta)),
                rho=_asarray_or_none(rho),
                a=_asarray_or_none(a),
                b=_asarray_or_none(b),
            ),
            config=StepConfig(
                steps_done=jnp.int32(0),
                sparse_on_at=_asarray_or_none(sparse_on_at),
                sequential_unroll=sequential_unroll,
                mesh=mesh,
                **red_cfg,
            ),
        )

        # add the chain axis to every chain-marked leaf at the position
        # declared by its field metadata
        state = _add_chains(state, num_chains)

        # delete big input arrays such that they can be deleted as soon as they
        # are sharded, only those arrays that contain an (n,) sized axis
        del X, error_scale, missing, y

        # move all arrays to the appropriate device
        state = _shard_state(state)

        # calculate initial resid in the continuous outcome case, such that y
        # and offset are already sharded if needed
        if state.resid is None:
            state = _set_initial_resid(state, binary_indices, num_chains)

        # calculate initial binary_y
        if is_binary or binary_indices is not None:
            assert state.binary_y is not None  # holds y at this point
            binary_y = _LazyArray(
                _initial_binary_y,
                state.binary_y.shape
                if binary_indices is None
                else (binary_indices.size, n),
                state.binary_y,  # this is actually y
                binary_indices,
            )
            binary_y = _shard_leaf(binary_y, None, -1, state.config.mesh)
        else:
            binary_y = None
        state = replace(state, binary_y=binary_y)

        # calculate prec_scale and inv_sdev_scale after sharding to do the
        # calculation on the right devices. Pre-shard, `state.prec_scale` holds
        # the user-supplied `error_scale` and `state.inv_sdev_scale` holds the
        # user-supplied `missing` mask.
        if state.prec_scale is not None or state.inv_sdev_scale is not None:
            inv_sdev_scale, prec_scale = _compute_scales(
                state.prec_scale, state.inv_sdev_scale
            )
            state = replace(state, inv_sdev_scale=inv_sdev_scale, prec_scale=prec_scale)

            # calculate the initial prec_tree from the sharded prec_scale
            state = _set_initial_prec_tree(state, num_chains, num_trees, tree_size)

    # all the wrong-typed intermediates have now been replaced by their final
    # values, so type-checking can resume; make all types strong to avoid
    # unwanted recompilations
    return _remove_weak_types(state)


def _set_initial_resid(
    state: 'State', binary_indices: Int32[Array, ' kb'] | None, num_chains: int | None
) -> 'State':
    """Build the continuous-outcome `resid` and shard it.

    Called post-shard so the captured ``state.binary_y`` and ``state.offset``
    are already on the target devices. Sharding axes are read via
    `chain_vmap_axes` / `data_vmap_axes` on a shape preview where the new
    `resid` leaf has the chain-extended ``ndim`` (inflated by a placeholder
    when `num_chains` is not `None`).
    """
    assert state.binary_y is not None  # holds y at this point
    inner = _LazyArray(
        _initial_resid,
        state.binary_y.shape,
        state.binary_y,
        state.offset,
        binary_indices,
    )
    preview_resid = add_dummy_axis(inner) if num_chains is not None else inner
    preview = replace(state, resid=preview_resid)
    chain_axis = chain_vmap_axes(preview).resid
    data_axis = data_vmap_axes(preview).resid
    resid = _wrap_chain(inner, chain_axis, num_chains)
    resid = _shard_leaf(resid, chain_axis, data_axis, state.config.mesh)
    return replace(state, resid=resid)


def _initial_resid(
    shape: tuple[int, ...],
    y: Float32[Array, ' n'] | Float32[Array, 'k n'],
    offset: Float32[Array, ''] | Float32[Array, ' k'],
    binary_indices: Int32[Array, ' kb'] | None,
) -> Float32[Array, ' n'] | Float32[Array, 'k n']:
    """Calculate the initial value for `State.resid` in the continuous outcome case.

    In the mixed binary-continuous case, binary rows are zeroed out (their
    residual starts at ``z - trees - offset = 0``).
    """
    resid = jnp.broadcast_to(y - offset[..., None], shape)
    if binary_indices is not None:
        resid = resid.at[..., binary_indices, :].set(0.0)
    return resid


def _initial_binary_y(
    shape: tuple[int, ...],
    y: Float32[Array, 'k n'] | Float32[Array, ' n'],
    binary_indices: Int32[Array, ' kb'] | None,
) -> Bool[Array, 'kb n'] | Bool[Array, ' n']:
    """Extract and convert the binary outcome components from ``y``."""
    if binary_indices is None:
        out = y.astype(bool)
    else:
        out = y[binary_indices, :].astype(bool)
    assert out.shape == shape
    return out


def _initial_affluence_tree(
    shape: tuple[int, ...], n: int, min_points_per_decision_node: int | None
) -> Shaped[Array, '...']:
    """Create the initial value of `Forest.affluence_tree`."""
    return (
        jnp.zeros(shape, bool)
        .at[..., 1]
        .set(
            True
            if min_points_per_decision_node is None
            else n >= min_points_per_decision_node
        )
    )


def _initial_count_tree(shape: tuple[int, ...], n: int) -> Shaped[Array, '...']:
    """Create the initial value of `Forest.count_tree`: all datapoints in the root."""
    return jnp.zeros(shape, jnp.uint32).at[..., 1].set(n)


def _set_initial_prec_tree(
    state: State, num_chains: int | None, num_trees: int, tree_size: int
) -> State:
    """Build the cached per-leaf precision for root-only trees and shard it.

    Called post-shard so the captured ``state.prec_scale`` is already on the
    target devices; mirrors `_set_initial_resid`.
    """
    assert state.prec_scale is not None
    shape = (num_trees, *state.prec_scale.shape[:-1], tree_size)
    inner = _LazyArray(_initial_prec_tree, shape, state.prec_scale)
    preview_tree = add_dummy_axis(inner) if num_chains is not None else inner
    preview = replace(state, forest=replace(state.forest, prec_tree=preview_tree))
    chain_axis = chain_vmap_axes(preview).forest.prec_tree
    prec_tree = _wrap_chain(inner, chain_axis, num_chains)
    prec_tree = _shard_leaf(prec_tree, chain_axis, None, state.config.mesh)
    return replace(state, forest=replace(state.forest, prec_tree=prec_tree))


def _initial_prec_tree(
    shape: tuple[int, ...], prec_scale: Float32[Array, ' n'] | Float32[Array, 'k k n']
) -> Float32[Array, 'num_trees tree_size'] | Float32[Array, 'num_trees k k tree_size']:
    """Create the initial value of `Forest.prec_tree`: all datapoints in the root."""
    return jnp.zeros(shape, jnp.float32).at[..., 1].set(prec_scale.sum(axis=-1))


@jit(donate_argnums=(0, 1))
def _compute_scales(
    error_scale: Float32[Array, ' n'] | Float32[Array, 'k n'] | None,
    missing: Bool[Array, ' n'] | Bool[Array, 'k n'] | None,
) -> tuple[
    Float32[Array, ' n'] | Float32[Array, 'k n'],
    Float32[Array, ' n'] | Float32[Array, 'k k n'],
]:
    """Compute ``inv_sdev_scale`` and ``prec_scale``.

    This is a separate function to use donate_argnums to avoid intermediate
    copies. At least one of `error_scale` and `missing` must be non-None.
    """
    if error_scale is None:
        inv_sdev_scale = jnp.array(1.0)
    else:
        inv_sdev_scale = jnp.reciprocal(error_scale)
    if missing is not None:
        inv_sdev_scale = jnp.where(missing, 0.0, inv_sdev_scale)
    if inv_sdev_scale.ndim == 1:
        prec_scale = jnp.square(inv_sdev_scale)
    else:
        prec_scale = jnp.einsum('an,bn->abn', inv_sdev_scale, inv_sdev_scale)
    return inv_sdev_scale, prec_scale


def _get_blocked_vars(
    filter_splitless_vars: int, max_split: UInt[Array, ' p']
) -> None | UInt[Array, ' q']:
    """Initialize the `blocked_vars` field."""
    if filter_splitless_vars:
        (p,) = max_split.shape
        (blocked_vars,) = jnp.nonzero(
            max_split == 0, size=filter_splitless_vars, fill_value=p
        )
        return blocked_vars.astype(minimal_unsigned_dtype(p))
        # see `fully_used_variables` for the type cast
    else:
        return None


def _add_chains(state: 'State', num_chains: int | None) -> 'State':
    """Extend chain-marked `_LazyArray` leaves to include a chain axis of size `num_chains`.

    Walks `state`, asks `chain_vmap_axes` where each leaf's chain axis lives,
    and wraps the carried `_LazyArray` so its factory creates the core array
    and then broadcasts a chain axis in at that position. To make
    `chain_vmap_axes` normalize against the chain-extended ``ndim``, the
    lookup is done on a shape preview built via `add_dummy_axis`. No-op when
    `num_chains` is `None`.

    Chain-marked leaves are required to be `_LazyArray` (or `None`); eager
    arrays at chain-marked positions are rejected so that all chain insertion
    happens at concretization time inside `_shard_state`.
    """
    if num_chains is None:
        return state
    preview = add_dummy_axis(state)
    chain_axes = chain_vmap_axes(preview)

    def wrap(leaf: object, chain_axis: int | None) -> object:
        if chain_axis is None or leaf is None:
            return leaf
        assert isinstance(leaf, _LazyArray), (
            f'expected _LazyArray for chain-marked leaf, got {type(leaf).__name__}'
        )
        return _wrap_chain(leaf, chain_axis, num_chains)

    return tree.map(wrap, state, chain_axes, is_leaf=_is_lazy_or_none)


def _parse_mesh(
    num_chains: int | None, mesh: Mesh | dict[str, int] | None
) -> Mesh | None:
    """Parse the `mesh` argument."""
    if mesh is None:
        return None

    # convert dict format to actual mesh
    if not isinstance(mesh, Mesh):
        assert set(mesh).issubset({'chains', 'data'})
        mesh = make_mesh(
            tuple(mesh.values()), tuple(mesh), axis_types=(AxisType.Auto,) * len(mesh)
        )

    # the chains mesh axis must be consistent with the number of chains
    if 'chains' in mesh.axis_names:
        if num_chains is None:
            msg = "mesh has a 'chains' axis but num_chains is None (scalar, no chain axis)"
            raise ValueError(msg)
        chains_axis = get_axis_size(mesh, 'chains')
        if num_chains % chains_axis:
            msg = (
                f"mesh 'chains' axis of size {chains_axis} does not divide "
                f'num_chains={num_chains}'
            )
            raise ValueError(msg)

    # check the axes we use are in auto mode
    assert 'chains' not in mesh.axis_names or 'chains' in mesh.auto_axes
    assert 'data' not in mesh.axis_names or 'data' in mesh.auto_axes

    return mesh


@partial(filter_jit, donate='all')
# jit and donate because otherwise type conversion would create copies
def _remove_weak_types(x: PyTree[Array, 'T']) -> PyTree[Array, 'T']:
    """Make all types strong.

    This is to avoid recompilation in `run_mcmc` or `step`.
    """

    def remove_weak(x: T) -> T:
        if isinstance(x, Array) and x.weak_type:
            return cast(T, x.astype(x.dtype))
        else:
            return x

    return tree.map(remove_weak, x)


def _shard_state(state: State) -> State:
    """Place all arrays on the appropriate devices, and instantiate lazily defined arrays."""
    mesh = state.config.mesh
    shard_leaf = partial(_shard_leaf, mesh=mesh)
    return tree.map(
        shard_leaf,
        state,
        chain_vmap_axes(state),
        data_vmap_axes(state),
        is_leaf=lambda x: x is None or isinstance(x, _LazyArray),
    )


def _leaf_partition_spec(
    ndim: int, chain_axis: int | None, data_axis: int | None, mesh: Mesh
) -> PartitionSpec:
    """Build a `PartitionSpec` for a leaf with the given chain/data axes."""
    spec = [None] * ndim
    if chain_axis is not None and 'chains' in mesh.axis_names:
        spec[chain_axis] = 'chains'
    if data_axis is not None and 'data' in mesh.axis_names:
        spec[data_axis] = 'data'

    # remove trailing Nones to be consistent with jax's output, it's useful
    # for comparing shardings during debugging
    while spec and spec[-1] is None:
        spec.pop()

    return PartitionSpec(*spec)


def _shard_leaf(
    x: Shaped[Array, '*shape'] | None | Shaped[_LazyArray, '*shape'],
    chain_axis: int | None,
    data_axis: int | None,
    mesh: Mesh | None,
) -> Shaped[Array, '*shape'] | None:
    """Create `x` if it's lazy and shard it."""
    if x is None:
        return None

    if mesh is None:
        sharding = None
    else:
        spec = _leaf_partition_spec(x.ndim, chain_axis, data_axis, mesh)
        sharding = NamedSharding(mesh, spec)

    if isinstance(x, _LazyArray):
        x = _concretize_lazy_array(x, sharding)
    elif sharding is not None:
        x = device_put(x, sharding, donate=True)

    return x


@filter_jit
# jit such that in recent jax versions the shards are created on the right
# devices immediately instead of being created on the wrong device and then
# copied
def _concretize_lazy_array(
    x: Shaped[_LazyArray, '*shape'], sharding: NamedSharding | None
) -> Shaped[Array, '*shape']:
    """Create an array from an abstract spec on the appropriate devices."""
    x = x()
    if sharding is not None:
        x = lax.with_sharding_constraint(x, sharding)
    return x


def _all_none_or_not_none(*args: object) -> bool:
    is_none = [x is None for x in args]
    return all(is_none) or not any(is_none)


def _asarray_or_none(x: object) -> Shaped[Array, '...'] | None:
    if x is None:
        return None
    return jnp.asarray(x)


class _ReductionConfig(TypedDict):
    """Fields of `StepConfig` related to reductions."""

    resid_reduction_config: ReductionConfig
    count_reduction_config: ReductionConfig
    prec_reduction_config: ReductionConfig
    prec_count_num_trees: int | None


def _parse_reduction_configs(
    resid_reduction_config: ReductionConfig,
    count_reduction_config: ReductionConfig,
    prec_reduction_config: ReductionConfig,
    prec_count_num_trees: int | None | Literal['auto'],
    y: Float32[Array, ' n'] | Float32[Array, ' k n'] | Bool[Array, ' n'],
    num_trees: int,
    num_chains: int | None,
    mesh: Mesh | None,
) -> _ReductionConfig:
    """Determine settings for indexed reduces."""
    n = y.shape[-1]
    n //= get_axis_size(mesh, 'data')  # per-device datapoints
    # chains are vmapped together on each device, so they share the per-step
    # memory of the per-tree reduction
    chains_per_device = (num_chains or 1) // get_axis_size(mesh, 'chains')
    # the reduction configs carry their own datapoint-batch settings (resolved
    # per-platform at run time when 'auto', see `ReductionConfig`), so they are
    # stored verbatim; only `prec_count_num_trees`, which does not depend on the
    # platform, is resolved here
    return dict(
        resid_reduction_config=resid_reduction_config,
        count_reduction_config=count_reduction_config,
        prec_reduction_config=prec_reduction_config,
        prec_count_num_trees=_parse_prec_count_num_trees(
            prec_count_num_trees, num_trees, n * chains_per_device
        ),
    )


def _parse_prec_count_num_trees(
    prec_count_num_trees: int | None | Literal['auto'], num_trees: int, n: int
) -> int | None:
    """Return the number of trees to process at a time or determine it automatically."""
    if prec_count_num_trees != 'auto':
        return prec_count_num_trees
    max_n_by_ntree = 2**27  # about 100M
    pcnt = max_n_by_ntree // max(1, n)
    pcnt = min(num_trees, pcnt)
    pcnt = max(1, pcnt)
    pcnt = _search_divisor(
        pcnt, num_trees, max(1, pcnt // 2), max(1, min(num_trees, pcnt * 2))
    )
    if pcnt >= num_trees:
        pcnt = None
    return pcnt


def _search_divisor(target_divisor: int, dividend: int, low: int, up: int) -> int:
    """Find the divisor closest to `target_divisor` in [low, up] if `target_divisor` is not already.

    If there is none, give up and return `target_divisor`.
    """
    assert target_divisor >= 1
    assert 1 <= low <= up <= dividend
    if dividend % target_divisor == 0:
        return target_divisor
    candidates = numpy.arange(low, up + 1)
    divisors = candidates[dividend % candidates == 0]
    if divisors.size == 0:
        return target_divisor
    penalty = numpy.abs(divisors - target_divisor)
    closest = numpy.argmin(penalty)
    return divisors[closest].item()


def get_axis_size(mesh: Mesh | None, axis_name: str) -> int:
    if mesh is None or axis_name not in mesh.shape:
        return 1
    else:
        return mesh.shape[axis_name]


def chol_with_gersh(
    mat: Float32[Array, '*batch_shape k k'], absolute_eps: bool = False
) -> Float32[Array, '*batch_shape k k']:
    """Cholesky with Gershgorin stabilization, supports batching."""
    return _chol_with_gersh_impl(mat, absolute_eps)


@partial(jnp.vectorize, signature='(k,k)->(k,k)', excluded=(1,))
def _chol_with_gersh_impl(
    mat: Float32[Array, '*batch_shape k k'], absolute_eps: bool
) -> Float32[Array, '*batch_shape k k']:
    rho = jnp.max(jnp.sum(jnp.abs(mat), axis=1), initial=0.0)
    eps = jnp.finfo(mat.dtype).eps
    u = mat.shape[0] * rho * eps
    if absolute_eps:
        u += eps
    mat = mat.at[jnp.diag_indices_from(mat)].add(u)
    return jnp.linalg.cholesky(mat)


def _inv_via_chol_with_gersh(
    mat: Float32[Array, '*batch_shape k k'],
) -> Float32[Array, '*batch_shape k k']:
    """Compute matrix inverse via Cholesky with Gershgorin stabilization.

    DO NOT USE THIS FUNCTION UNLESS YOU REALLY NEED TO.
    """
    # mat = L L^T
    # mat^-1 = L^-T L^-1 = L^-T I L^-1 = L^-T (L^-T I)^T
    # I suspect this to be more accurate than (L^-1 I)^T (L^-1 I)
    L = chol_with_gersh(mat)
    eye = jnp.broadcast_to(jnp.eye(mat.shape[-1]), mat.shape)
    Ltinv = solve_triangular(L, eye, trans='T', lower=True)
    return solve_triangular(L, Ltinv.mT, trans='T', lower=True)


def split_key_for_chains(
    fun: Callable[[Key[Array, ''] | Key[Array, ' num_chains'], State], State],
) -> Callable[[Key[Array, ''], State], State]:
    """Split a single PRNG key into per-chain keys before calling `fun`.

    When the state is multichain, the input key is split into
    ``state.num_chains()`` keys. For single-chain states, the key is passed
    through unchanged.
    """

    @wraps(fun)
    def wrapped(key: Key[Array, ''], state: State) -> State:
        num_chains = state.num_chains()
        if num_chains is None:
            return fun(key, state)
        keys = random.split(key, num_chains)
        return fun(keys, state)

    return wrapped


def partition_specs(x: PyTree, mesh: Mesh) -> PyTree[PartitionSpec]:
    """Per-leaf `PartitionSpec`s derived from chain/data `field` markers.

    Each array leaf is sharded over ``'chains'`` along its chain axis and over
    ``'data'`` along its data axis, when those axes are marked (see `field`)
    and present in `mesh`; all other axes are replicated.

    Parameters
    ----------
    x
        A pytree of arrays carrying chain/data `field` markers.
    mesh
        The device mesh to shard over.

    Returns
    -------
    A pytree matching `x` with a `PartitionSpec` in place of each array leaf.
    """
    return tree.map(
        lambda leaf, ca, da: _leaf_partition_spec(leaf.ndim, ca, da, mesh),
        x,
        chain_vmap_axes(x),
        data_vmap_axes(x),
    )


def shard_map_state(
    fun: Callable[[Key[Array, ''] | Key[Array, ' num_chains'], State], State],
) -> Callable[[Key[Array, ''] | Key[Array, ' num_chains'], State], State]:
    """Wrap a ``(keys, state) -> state`` function in a manual `jax.shard_map`.

    Uses `state.config.mesh` (static). No-op when the mesh is `None`. The keys
    input is sharded across ``'chains'`` when the state is multichain and
    ``'chains'`` is in the mesh; otherwise the keys are replicated. State
    leaves are sharded according to their `chains`/`data` field metadata. The
    output sharding matches the input sharding.
    """

    @wraps(fun)
    def wrapped(key: Key[Array, ''] | Key[Array, ' num_chains'], state: State) -> State:
        mesh = state.config.mesh
        if mesh is None:
            return fun(key, state)

        if state.has_chains and 'chains' in mesh.axis_names:
            key_spec = PartitionSpec('chains')
        else:
            key_spec = PartitionSpec()

        state_specs = partition_specs(state, mesh)

        mapped = shard_map(
            fun,
            mesh=mesh,
            in_specs=(key_spec, state_specs),
            out_specs=state_specs,
            **_get_shard_map_patch_kwargs(),
        )
        return mapped(key, state)

    return wrapped


def vmap_chains(
    fun: Callable[[Key[Array, ''], State], State],
) -> Callable[[Key[Array, ' num_chains'] | Key[Array, ''], State], State]:
    """Vmap a ``(key, state) -> state`` function over chain axes.

    When the state is multichain, `keys` must have a leading chain axis and
    `fun` is vmapped over it together with the chain axes of `state`. For
    single-chain states, the function is called unchanged.
    """

    @wraps(fun)
    def wrapped(
        keys: Key[Array, ' num_chains'] | Key[Array, ''], state: State
    ) -> State:
        if not state.has_chains:
            return fun(keys, state)
        state_axes = chain_vmap_axes(state)
        vmapped_fun = vmap(fun, in_axes=(0, state_axes), out_axes=state_axes)
        return vmapped_fun(keys, state)

    return wrapped


class _ShardMapPatchKwargs(TypedDict, total=False):
    check_vma: bool


def _get_shard_map_patch_kwargs() -> _ShardMapPatchKwargs:
    # bug: jax 0.8.1-0.8.2: vmap(shard_map(psum)), jax#34249; the
    # jax_disable_vmap_shmap_error config did not work.

    # WORKAROUND(jax<=0.8.2): remove this whole function when jax > 0.8.2
    buggy = ('0.8.1', '0.8.2')
    if jax.__version__ in buggy:
        return {'check_vma': False}
    return {}
