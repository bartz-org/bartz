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

import inspect
import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import fields, replace
from enum import Enum
from functools import partial, wraps
from typing import Any, Literal, TypedDict, TypeVar

import jax
import numpy
from equinox import Module, error_if, filter_jit
from equinox import field as eqx_field
from jax import (
    NamedSharding,
    ShapeDtypeStruct,
    device_put,
    jit,
    lax,
    make_mesh,
    random,
    tree,
    vmap,
)
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.sharding import AxisType, Mesh, PartitionSpec
from numpy.lib.array_utils import normalize_axis_index

# WORKAROUND(jax<0.6.1): shard_map was promoted from jax.experimental to top-level in 0.6.1
try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map
from jaxtyping import Array, Bool, Float, Float32, Int32, Integer, Key, PyTree, UInt
from numpy import ndarray

from bartz._jaxext import get_default_device, minimal_unsigned_dtype
from bartz.grove import tree_depths

ArrayLike = Array | ndarray

FloatLike = float | Float[ArrayLike, '']

# Default position of the chain axis in chain-bearing leaves; see `field`.
CHAIN_AXIS = 0


class OutcomeType(Enum):
    """Likelihood types for each outcome component in the regression."""

    continuous = 'continuous'
    """Continuous outcome with Normal error."""

    binary = 'binary'
    """Binary outcome in {0, 1} with probit link."""


def field(  # noqa: ANN202
    *,
    chains: int | None = None,
    data: int | None = None,
    samples: int | None = None,
    **kwargs: Any,
):
    """Extend `equinox.field` with chain/data/sample axis markers.

    Parameters
    ----------
    chains
        Index of the chain axis for the field's arrays, or `None` if the field
        has no chain axis. Any int is accepted, including negative indices with
        the usual numpy semantics (e.g. ``-1`` for the last axis); the index is
        normalized per-leaf against the leaf's ``ndim`` by `chain_vmap_axes`.
    data
    samples
        Indices of the data/sample axes for the field's arrays, declared in the
        chain-less "core" layout. `None` if the field has no data/sample axis.
        The index is normalized per-leaf against the core ``ndim`` (the leaf's
        ``ndim`` minus 1 when a chain axis is present, else the leaf's
        ``ndim``); the chain axis, if any, is treated as inserted after the
        data/sample axis exists, so `data_vmap_axes`/`trace_sample_axes` shift
        the returned sample index up by 1 when the chain position is at or
        before the core data/sample index.
    **kwargs
        Other parameters passed to `equinox.field`.

    Returns
    -------
    A dataclass field descriptor with the axis indices in the metadata, unset
    if `None`.
    """
    metadata = dict(kwargs.pop('metadata', {}))
    assert 'chains' not in metadata
    assert 'data' not in metadata
    assert 'samples' not in metadata
    for name, value in (('chains', chains), ('data', data), ('samples', samples)):
        # bool is a subclass of int; reject it so a boolean value does not
        # silently mean axis 0 or 1.
        assert not isinstance(value, bool), (
            f'{name!r} marker must be an int axis index or None, not bool'
        )
        if value is not None:
            metadata[name] = value
    return eqx_field(metadata=metadata, **kwargs)


def chain_vmap_axes(x: PyTree[Module | Any, 'T']) -> PyTree[int | None, 'T']:
    """Determine vmapping axes for chains.

    This function determines the argument to the `in_axes` or `out_axes`
    parameter of `jax.vmap` to vmap over all and only the chain axes found in the
    pytree `x`.

    Parameters
    ----------
    x
        A pytree. Subpytrees that are Module attributes marked with
        ``field(chains=<int>)`` are considered to have a chain axis at that
        index. `x` (or one of its subtrees) must define a `has_chains` property
        (see `get_has_chains`).

    Returns
    -------
    A pytree with the same structure as `x`, with each leaf set to the chain
    axis index declared by its owning ``field(chains=...)`` marker, normalized
    against the leaf's ``ndim`` (so the returned indices are non-negative), or
    `None` for unmarked leaves. If `has_chains` is `False`, every leaf is
    `None`.
    """
    if not get_has_chains(x):
        return _find_metadata(x, 'chains', marker_value=_none_marker)

    return _find_metadata(x, 'chains')


def _none_marker(leaf: object, raw: int) -> None:  # noqa: ARG001
    """Marker mapper that always returns `None`."""
    return None  # noqa: RET501


def data_vmap_axes(x: PyTree[Module | Any, 'T']) -> PyTree[int | None, 'T']:
    """Determine vmapping axes for data.

    Parameters
    ----------
    x
        A pytree. Subpytrees that are Module attributes marked with
        ``field(data=<int>)`` are considered to have a data axis at that
        position in the chain-less layout. `x` (or one of its subtrees) must
        define a `has_chains` property (see `get_has_chains`).

    Returns
    -------
    A pytree with the same structure as `x`, with each leaf set to the data axis index (normalized and chain-shifted), or `None` for unmarked leaves.
    """
    chain_axes = chain_vmap_axes(x)
    data_raw = _find_metadata(x, 'data', marker_value=_raw_marker)
    return tree.map(
        _compute_core_axis, x, data_raw, chain_axes, is_leaf=_is_core_axis_leaf
    )


def trace_sample_axes(trace: PyTree[Module | Any, 'T']) -> PyTree[int | None, 'T']:
    """Determine the position of the sample axis for each leaf of a trace.

    Parameters
    ----------
    trace
        A trace pytree (typically a `~bartz.mcmcloop.BurninTrace` or
        `~bartz.mcmcloop.MainTrace`). `trace` (or one of its subtrees) must
        define a `has_chains` property.

    Returns
    -------
    A pytree with the same structure as `trace` but with sample axes in the leaves, see `field`.
    """
    chain_axes = chain_vmap_axes(trace)
    sample_raw = _find_metadata(trace, 'samples', marker_value=_raw_marker)
    return tree.map(
        _compute_core_axis, trace, sample_raw, chain_axes, is_leaf=_is_core_axis_leaf
    )


def _raw_marker(leaf: object, raw: int) -> int:  # noqa: ARG001
    """Marker mapper that returns the raw marker value."""
    return raw


def _is_core_axis_leaf(x: object) -> bool:
    """Treat `None` and `_LazyArray` as leaves when resolving core-axis markers."""
    return x is None or _is_lazy_array(x)


def _compute_core_axis(
    leaf: object, raw: int | None, chain_axis: int | None
) -> int | None:
    """Combine a raw core-layout marker and a (normalized) chain position."""
    if raw is None:
        return None
    has_chain = chain_axis is not None
    core_ndim = leaf.ndim - (1 if has_chain else 0)
    axis = normalize_axis_index(raw, core_ndim)
    if has_chain and chain_axis <= axis:
        axis += 1
    return axis


T = TypeVar('T')


class _HasChainsFound(Exception):
    """Internal control-flow signal carrying a found `has_chains` value."""

    def __init__(self, value: bool) -> None:
        self.value = value


def get_has_chains(x: PyTree) -> bool:
    """Return the `has_chains` flag from the first node in `x` that defines it.

    Walks `x` and stops at the first node exposing a `has_chains` attribute,
    returning its value. The walk uses `jax.tree.map` with an `is_leaf` callback
    that raises a custom exception to short-circuit traversal.

    Parameters
    ----------
    x
        A pytree, possibly containing nodes that define a `has_chains`
        attribute.

    Returns
    -------
    The value of `has_chains` on the first matching node.

    Raises
    ------
    ValueError
        If no node in `x` defines a `has_chains` property.
    """

    def is_leaf(node: object) -> bool:
        try:
            value = node.has_chains
        except AttributeError:
            return False
        raise _HasChainsFound(value)

    try:
        tree.map(lambda _: None, x, is_leaf=is_leaf)
    except _HasChainsFound as exc:
        return exc.value
    msg = 'no `has_chains` property found in the pytree'
    raise ValueError(msg)


def _normalize_axis_for_leaf(leaf: object, raw: int) -> int:
    """Normalize a marker axis index against `leaf.ndim`.

    Raises `numpy.exceptions.AxisError` if `raw` is out of bounds for
    `leaf.ndim`.
    """
    return normalize_axis_index(raw, leaf.ndim)


def _is_lazy_array(x: object) -> bool:
    return isinstance(x, _LazyArray)


def _is_module(x: object) -> bool:
    return isinstance(x, Module) and not _is_lazy_array(x)


def _find_metadata(
    x: PyTree[Any, ' S'],
    key: Hashable,
    *,
    marker_value: Callable[[object, int], object] = _normalize_axis_for_leaf,
    default_value: object = None,
) -> PyTree[Any, ' S']:
    """Walk `x` replacing marked subtrees with derived values.

    For each Module field whose metadata contains `key`, the field's subtree
    is replaced by mapping ``marker_value(leaf, raw)`` over its leaves, where
    `raw` is the unnormalized metadata value; leaves outside any marked field
    become `default_value`.
    """
    if _is_module(x):
        args = []
        for f in fields(x):
            v = getattr(x, f.name)
            if f.metadata.get('static', False):
                args.append(v)
            elif key in f.metadata:
                raw = f.metadata[key]
                args.append(
                    tree.map(
                        lambda leaf, raw=raw: marker_value(leaf, raw),
                        v,
                        is_leaf=_is_lazy_array,
                    )
                )
            else:
                args.append(
                    _find_metadata(
                        v, key, marker_value=marker_value, default_value=default_value
                    )
                )
        return x.__class__(*args)

    def get_axes(x: object) -> PyTree:
        if _is_module(x):
            return _find_metadata(
                x, key, marker_value=marker_value, default_value=default_value
            )
        return tree.map(lambda _: default_value, x, is_leaf=_is_lazy_array)

    return tree.map(get_axes, x, is_leaf=lambda x: isinstance(x, Module))


class Forest(Module):
    """Represents the MCMC state of a sum of trees."""

    leaf_tree: (
        Float32[Array, '*chains num_trees 2**d']
        | Float32[Array, '*chains num_trees k 2**d']
    ) = field(chains=CHAIN_AXIS)
    """The leaf values."""

    var_tree: UInt[Array, '*chains num_trees 2**(d-1)'] = field(chains=CHAIN_AXIS)
    """The decision axes."""

    split_tree: UInt[Array, '*chains num_trees 2**(d-1)'] = field(chains=CHAIN_AXIS)
    """The decision boundaries."""

    affluence_tree: Bool[Array, '*chains num_trees 2**(d-1)'] = field(chains=CHAIN_AXIS)
    """Marks leaves that can be grown."""

    max_split: UInt[Array, ' p']
    """The maximum split index for each predictor."""

    blocked_vars: UInt[Array, ' q'] | None
    """Indices of variables that are not used. This shall include at least
    the `i` such that ``max_split[i] == 0``, otherwise behavior is
    undefined."""

    p_nonterminal: Float32[Array, ' 2**d']
    """The prior probability of each node being nonterminal, conditional on
    its ancestors. Includes the nodes at maximum depth which should be set
    to 0."""

    p_propose_grow: Float32[Array, ' 2**(d-1)']
    """The unnormalized probability of picking a leaf for a grow proposal."""

    leaf_indices: UInt[Array, '*chains num_trees n'] = field(chains=CHAIN_AXIS, data=-1)
    """The index of the leaf each datapoints falls into, for each tree."""

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

    grow_prop_count: Int32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    """The number of grow proposals made during one full MCMC cycle."""

    prune_prop_count: Int32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    """The number of prune proposals made during one full MCMC cycle."""

    grow_acc_count: Int32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    """The number of grow moves accepted during one full MCMC cycle."""

    prune_acc_count: Int32[Array, '*chains'] = field(chains=CHAIN_AXIS)
    """The number of prune moves accepted during one full MCMC cycle."""

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
    """Parameter of the prior on `theta`. Required only to sample `theta`.
    See `step_theta`."""

    b: Float32[Array, ''] | None
    """Parameter of the prior on `theta`. Required only to sample `theta`.
    See `step_theta`."""

    rho: Float32[Array, ''] | None
    """Parameter of the prior on `theta`. Required only to sample `theta`.
    See `step_theta`."""

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

    resid_num_batches: int | None = field(static=True)
    """The number of batches for computing the sum of residuals. If
    `None`, they are computed with no batching."""

    count_num_batches: int | None = field(static=True)
    """The number of batches for computing counts. If
    `None`, they are computed with no batching."""

    prec_num_batches: int | None = field(static=True)
    """The number of batches for computing precision scales. If
    `None`, they are computed with no batching."""

    prec_count_num_trees: int | None = field(static=True)
    """Batch size for processing trees to compute count and prec trees."""

    mesh: Mesh | None = field(static=True)
    """The mesh used to shard data and computation across multiple devices."""

    @property
    def data_sharded(self) -> bool:
        """Whether the data axis is sharded across devices."""
        return self.mesh is not None and 'data' in self.mesh.axis_names


class State(Module):
    """Represents the MCMC state of BART."""

    X: UInt[Array, 'p n'] = field(data=-1)
    """The predictors."""

    binary_y: None | Bool[Array, ' n'] | Bool[Array, 'k n'] = field(data=-1)
    """The response as booleans for binary regression, `None` for continuous.
    In the mixed binary-continuous case, only the binary outcome components
    are stored, with shape ``(kb, n)``."""

    z: None | Float32[Array, '*chains n'] | Float32[Array, '*chains k n'] = field(
        chains=CHAIN_AXIS, data=-1
    )
    """The latent variable for binary regression. `None` in continuous
    regression. In the mixed binary-continuous case, only the binary outcome
    components are stored, with shape ``(*chains, kb, n)``."""

    binary_indices: None | Int32[Array, ' kb']
    """The indices of binary outcome components in the full list of outcome
    components. `None` when there are no binary components. Filled in by
    `init` and used by `step_z` to update only the binary rows of `resid`."""

    offset: Float32[Array, ''] | Float32[Array, ' k']
    """Constant shift added to the sum of trees."""

    resid: Float32[Array, '*chains n'] | Float32[Array, '*chains k n'] = field(
        chains=CHAIN_AXIS, data=-1
    )
    """The residuals (`y` or `z` minus sum of trees)."""

    error_cov_inv: Float32[Array, '*chains'] | Float32[Array, '*chains k k'] = field(
        chains=CHAIN_AXIS
    )
    """The inverse error covariance (scalar for univariate, matrix for multivariate).
    Identity in binary regression."""

    prec_scale: Float32[Array, ' n'] | Float32[Array, 'k k n'] | None = field(data=-1)
    """The scale on the error precision. `None` in binary regression. With
    scalar per-datapoint weights, shape ``(n,)`` and value
    ``1 / error_scale ** 2``. With vector per-datapoint weights, shape ``(k, k, n)``
    and value ``1/outer(error_scale, error_scale)`` repeated over datapoints."""

    inv_sdev_scale: Float32[Array, ' n'] | Float32[Array, 'k n'] | None = field(data=-1)
    """The reciprocal of the per-observation error standard-deviation scale.
    `None` in binary regression. Shape ``(n,)`` for scalar weights, or
    ``(k, n)`` for per-component vector weights."""

    error_cov_df: Float32[Array, ''] | None
    """The df parameter of the inverse Wishart prior on the noise
    covariance. For the univariate case, the relationship to the inverse
    gamma prior parameters is ``alpha = df / 2``.
    `None` in binary regression."""

    error_cov_scale: Float32[Array, ''] | Float32[Array, 'k k'] | None
    """The scale parameter of the inverse Wishart prior on the noise
    covariance. For the univariate case, the relationship to the inverse
    gamma prior parameters is ``beta = scale / 2``.
    `None` in binary regression."""

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


def _check_diagonal(error_cov_scale: Float32[Array, 'k k']) -> Float32[Array, 'k k']:
    """Raise if `error_cov_scale` is not diagonal."""
    diag = jnp.diag(jnp.diag(error_cov_scale))
    return error_if(
        error_cov_scale,
        jnp.any(error_cov_scale != diag),
        'error_cov_scale must be diagonal',
    )


def _init_diag_error_cov_inv(
    error_cov_df: Float32[Array, ''],
    error_cov_scale: Float32[Array, 'k k'],
    binary_mask: Sequence[bool] | None = None,
) -> Float32[Array, 'k k']:
    """Initialize diagonal `error_cov_inv` from inverse-gamma mode per component."""
    scale_diag = jnp.diag(error_cov_scale)
    inv_diag = error_cov_df / jnp.where(scale_diag, scale_diag, 1.0)
    if binary_mask is not None:
        inv_diag = jnp.where(jnp.array(binary_mask), 1.0, inv_diag)
    return jnp.diag(inv_diag)


def _init_shape_shifting_parameters(
    y: Float32[Array, ' n'] | Float32[Array, 'k n'],
    outcome_type: OutcomeType | list[OutcomeType],
    offset: Float32[Array, ''] | Float32[Array, ' k'],
    error_scale: Float32[ArrayLike, ' n'] | Float32[ArrayLike, 'k n'] | None,
    error_cov_df: float | Float32[ArrayLike, ''] | None,
    error_cov_scale: float | Float32[ArrayLike, ''] | Float32[ArrayLike, 'k k'] | None,
    leaf_prior_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'],
    missing: Bool[ArrayLike, ' n'] | Bool[ArrayLike, 'k n'] | None,
) -> tuple[
    bool,
    tuple[()] | tuple[int],
    None | Float32[Array, ''],
    None | Float32[Array, ''],
    None | Float32[Array, ''],
    None | Int32[Array, ' kb'],
]:
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
    error_cov_df
        The error covariance degrees of freedom.
    error_cov_scale
        The error covariance scale.
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
        The initialized error covariance inverse.
    error_cov_df
        The error covariance degrees of freedom (as array).
    error_cov_scale
        The error covariance scale (as array).
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

    # All-binary
    if is_binary:
        assert error_scale is None
        assert error_cov_df is None
        assert error_cov_scale is None
        if kshape:
            error_cov_inv = jnp.eye(kshape[0])
        else:
            error_cov_inv = jnp.array(1.0)

    # Mixed binary-continuous, or continuous-mv with 2-D missingness:
    # diagonal error covariance, updated component-wise.
    elif is_mixed or partial_missing:
        if is_mixed:
            assert error_scale is None
        error_cov_df = jnp.asarray(error_cov_df)
        error_cov_scale = _check_diagonal(jnp.asarray(error_cov_scale))
        assert error_cov_scale.shape == 2 * kshape
        error_cov_inv = _init_diag_error_cov_inv(
            error_cov_df, error_cov_scale, binary_mask if is_mixed else None
        )

    # All-continuous
    else:
        assert (
            error_scale is None
            or error_scale.shape == y.shape  # (k, n)
            or error_scale.shape == y.shape[-1:]  # (n,)
        )
        error_cov_df = jnp.asarray(error_cov_df)
        error_cov_scale = jnp.asarray(error_cov_scale)
        assert error_cov_scale.shape == 2 * kshape

        # Multivariate vs univariate
        if kshape:
            error_cov_inv = error_cov_df * _inv_via_chol_with_gersh(error_cov_scale)
        else:
            # inverse gamma prior: alpha = df / 2, beta = scale / 2
            error_cov_inv = error_cov_df / error_cov_scale

    assert y.shape[:-1] == kshape
    assert leaf_prior_cov_inv.shape == 2 * kshape

    return (
        is_binary,
        kshape,
        error_cov_inv,
        error_cov_df,
        error_cov_scale,
        binary_indices,
    )


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


class _LazyArray(Module):
    """Like `functools.partial` but specialized to array-creating functions like `jax.numpy.zeros`."""

    array_creator: Callable
    shape: tuple[int, ...]
    args: tuple

    def __init__(
        self, array_creator: Callable, shape: tuple[int, ...], *args: Any
    ) -> None:
        self.array_creator = array_creator
        self.shape = shape
        self.args = args

    def __call__(self, **kwargs: Any) -> T:
        return self.array_creator(self.shape, *self.args, **kwargs)

    @property
    def ndim(self) -> int:
        return len(self.shape)


DummyArray = Array | ShapeDtypeStruct | _LazyArray


def add_dummy_axis(x: PyTree[DummyArray, 'T']) -> PyTree[ShapeDtypeStruct, 'T']:
    """Replace array-like leaves with a rank-inflated placeholder."""

    def replace_leaf(leaf: DummyArray) -> ShapeDtypeStruct:
        return ShapeDtypeStruct((0,) * (leaf.ndim + 1), jnp.float32)

    return tree.map(replace_leaf, x, is_leaf=lambda x: isinstance(x, _LazyArray))


def _return_array(shape: tuple[int, ...], arr: Array, **kwargs: Any) -> Array:  # noqa: ARG001
    """`_LazyArray` factory that returns an already-built array."""
    return arr


def _lazy_from_array(arr: Array | None) -> _LazyArray | None:
    """Wrap an existing array as a `_LazyArray` reporting `arr.shape`, or pass `None`."""
    if arr is None:
        return None
    return _LazyArray(_return_array, arr.shape, arr)


def _broadcast_chain(
    shape: tuple[int, ...], inner: _LazyArray, chain_axis: int, **kwargs: Any
) -> Array:
    """Concretize `inner` then insert and broadcast a chain axis at `chain_axis`."""
    arr = inner(**kwargs)
    arr = jnp.expand_dims(arr, chain_axis)
    return jnp.broadcast_to(arr, shape)


def _wrap_chain(
    inner: _LazyArray, chain_axis: int | None, num_chains: int | None
) -> _LazyArray:
    """Wrap `inner` so its factory inserts and broadcasts `num_chains` at `chain_axis`. No-op when `chain_axis` is `None`."""
    if chain_axis is None:
        return inner
    new_shape = (*inner.shape[:chain_axis], num_chains, *inner.shape[chain_axis:])
    return _LazyArray(_broadcast_chain, new_shape, inner, chain_axis)


def _is_lazy_or_none(x: object) -> bool:
    """`tree.map(is_leaf=...)` predicate that stops at `_LazyArray` or `None`."""
    return x is None or isinstance(x, _LazyArray)


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
    error_cov_df: FloatLike | None = None,
    error_cov_scale: FloatLike | Float[ArrayLike, 'k k'] | None = None,
    error_scale: Float32[ArrayLike, ' n'] | Float32[ArrayLike, 'k n'] | None = None,
    missing: Bool[ArrayLike, ' n'] | Bool[ArrayLike, 'k n'] | None = None,
    min_points_per_decision_node: int | Integer[ArrayLike, ''] | None = None,
    resid_num_batches: int | None | Literal['auto'] = 'auto',
    count_num_batches: int | None | Literal['auto'] = 'auto',
    prec_num_batches: int | None | Literal['auto'] = 'auto',
    prec_count_num_trees: int | None | Literal['auto'] = 'auto',
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
    target_platform: Literal['cpu', 'gpu'] | None = None,
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
    error_cov_df
    error_cov_scale
        The df and scale parameters of the inverse Wishart prior on the error
        covariance. For the univariate case, the relationship to the inverse
        gamma prior parameters is ``alpha = df / 2``, ``beta = scale / 2``.
        Leave unspecified for binary regression.
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
        masked positions. If 2-D, `error_cov_scale` must be diagonal.
    min_points_per_decision_node
        The minimum number of data points in a decision node. 0 if not
        specified.
    resid_num_batches
    count_num_batches
    prec_num_batches
        The number of batches, along datapoints, for summing the residuals,
        counting the number of datapoints in each leaf, and computing the
        likelihood precision in each leaf, respectively. `None` for no batching.
        If 'auto', it's chosen automatically based on the target platform; see
        the description of `target_platform` below for how it is determined.
    prec_count_num_trees
        The number of trees to process at a time when counting datapoints or
        computing the likelihood precision. If `None`, do all trees at once,
        which may use too much memory. If 'auto' (default), it's chosen
        automatically.
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
    target_platform
        Platform ('cpu' or 'gpu') used to determine the number of batches
        automatically. If `mesh` is specified, the platform is inferred from the
        devices in the mesh. Otherwise, if `y` is a concrete array (i.e., `init`
        is not invoked in a `jax.jit` context), the platform is set to the
        platform of `y`. Otherwise, use `target_platform`.

        To avoid confusion, in all cases where the `target_platform` argument
        would be ignored, `init` raises an exception if `target_platform` is
        set.

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
    assert missing is None or missing.ndim <= y.ndim

    # normalize outcome_type to enum (or list of enums)
    outcome_type = _parse_outcome_type(outcome_type)

    # check p_nonterminal and pad it with a 0 at the end (still not final shape)
    p_nonterminal = _parse_p_nonterminal(p_nonterminal)

    # process arguments that change depending on outcome type
    is_binary, kshape, error_cov_inv, error_cov_df, error_cov_scale, binary_indices = (
        _init_shape_shifting_parameters(
            y,
            outcome_type,
            offset,
            error_scale,
            error_cov_df,
            error_cov_scale,
            leaf_prior_cov_inv,
            missing,
        )
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

    # determine batch sizes for reductions
    mesh = _parse_mesh(num_chains, mesh)
    target_platform = _parse_target_platform(
        y, mesh, target_platform, resid_num_batches, count_num_batches, prec_num_batches
    )
    red_cfg = _parse_reduction_configs(
        resid_num_batches,
        count_num_batches,
        prec_num_batches,
        prec_count_num_trees,
        y,
        num_trees,
        num_chains,
        mesh,
        target_platform,
    )

    # check there aren't too many deactivated predictors
    offset = _check_splitless_vars(filter_splitless_vars, max_split, offset)

    tree_size = 2**max_depth

    # initialize all remaining stuff and put it in an unsharded state. Every
    # chain-bearing leaf is built as a `_LazyArray` at its core (no-chain)
    # shape; `_add_chains` then wraps each one to broadcast in the chain axis.
    state = State(
        X=X,
        binary_y=y,  # temporary to be sharded together with everything else
        z=(
            _LazyArray(jnp.full, y.shape, offset[..., None])
            if is_binary
            else _LazyArray(
                jnp.full, (binary_indices.size, n), offset[binary_indices, None]
            )
            if binary_indices is not None
            else None
        ),
        binary_indices=binary_indices,
        offset=offset,
        resid=(
            _LazyArray(jnp.zeros, y.shape)
            if is_binary
            else None  # resid is created later after y and offset are sharded
        ),
        error_cov_inv=_lazy_from_array(error_cov_inv),
        # temporarily store user inputs in these slots so they get sharded
        # with everything else; `_compute_scales` replaces them post-shard.
        prec_scale=error_scale,
        inv_sdev_scale=missing,
        error_cov_df=error_cov_df,
        error_cov_scale=error_cov_scale,
        forest=Forest(
            leaf_tree=_LazyArray(
                jnp.zeros, (num_trees, *kshape, tree_size), jnp.float32
            ),
            var_tree=_LazyArray(
                jnp.zeros, (num_trees, tree_size // 2), minimal_unsigned_dtype(p - 1)
            ),
            split_tree=_LazyArray(
                jnp.zeros, (num_trees, tree_size // 2), max_split.dtype
            ),
            affluence_tree=_LazyArray(
                _initial_affluence_tree,
                (num_trees, tree_size // 2),
                n,
                min_points_per_decision_node,
            ),
            blocked_vars=_get_blocked_vars(filter_splitless_vars, max_split),
            max_split=max_split,
            grow_prop_count=_LazyArray(jnp.zeros, (), int),
            grow_acc_count=_LazyArray(jnp.zeros, (), int),
            prune_prop_count=_LazyArray(jnp.zeros, (), int),
            prune_acc_count=_LazyArray(jnp.zeros, (), int),
            p_nonterminal=p_nonterminal[tree_depths(tree_size)],
            p_propose_grow=p_nonterminal[tree_depths(tree_size // 2)],
            leaf_indices=_LazyArray(
                jnp.ones, (num_trees, n), minimal_unsigned_dtype(tree_size - 1)
            ),
            min_points_per_decision_node=_asarray_or_none(min_points_per_decision_node),
            min_points_per_leaf=_asarray_or_none(min_points_per_leaf),
            log_trans_prior=_LazyArray(jnp.zeros, (num_trees,))
            if save_ratios
            else None,
            log_likelihood=_LazyArray(jnp.zeros, (num_trees,)) if save_ratios else None,
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
            mesh=mesh,
            **red_cfg,
        ),
    )

    # add the chain axis to every chain-marked leaf at the position declared
    # by its field metadata
    state = _add_chains(state, num_chains)

    # delete big input arrays such that they can be deleted as soon as they
    # are sharded, only those arrays that contain an (n,) sized axis
    del X, error_scale, missing, y

    # move all arrays to the appropriate device
    state = _shard_state(state)

    # calculate initial resid in the continuous outcome case, such that y and
    # offset are already sharded if needed
    if state.resid is None:
        state = _set_initial_resid(state, binary_indices, num_chains)

    # calculate initial binary_y
    if is_binary or binary_indices is not None:
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

    # make all types strong to avoid unwanted recompilations
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
) -> Array:
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


@partial(jit, donate_argnums=(0, 1))
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
        inv_sdev_scale = 1.0
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
    if isinstance(mesh, dict):
        assert set(mesh).issubset({'chains', 'data'})
        mesh = make_mesh(
            tuple(mesh.values()), tuple(mesh), axis_types=(AxisType.Auto,) * len(mesh)
        )

    # check there's no chain mesh axis if there are no chains
    if num_chains is None:
        assert 'chains' not in mesh.axis_names

    # check the axes we use are in auto mode
    assert 'chains' not in mesh.axis_names or 'chains' in mesh.auto_axes
    assert 'data' not in mesh.axis_names or 'data' in mesh.auto_axes

    return mesh


def _parse_target_platform(
    y: Array,
    mesh: Mesh | None,
    target_platform: Literal['cpu', 'gpu'] | None,
    resid_num_batches: int | None | Literal['auto'],
    count_num_batches: int | None | Literal['auto'],
    prec_num_batches: int | None | Literal['auto'],
) -> Literal['cpu', 'gpu'] | None:
    if mesh is not None:
        assert target_platform is None, 'mesh provided, do not set target_platform'
        return mesh.devices.flat[0].platform
    elif hasattr(y, 'platform'):
        assert target_platform is None, 'device inferred from y, unset target_platform'
        return y.platform()
    elif (
        resid_num_batches == 'auto'
        or count_num_batches == 'auto'
        or prec_num_batches == 'auto'
    ):
        assert target_platform in ('cpu', 'gpu')
        return target_platform
    else:
        assert target_platform is None, 'target_platform not used, unset it'
        return target_platform


@partial(filter_jit, donate='all')
# jit and donate because otherwise type conversion would create copies
def _remove_weak_types(x: PyTree[Array, 'T']) -> PyTree[Array, 'T']:
    """Make all types strong.

    This is to avoid recompilation in `run_mcmc` or `step`.
    """

    def remove_weak(x: T) -> T:
        if isinstance(x, Array) and x.weak_type:
            return x.astype(x.dtype)
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
    x: Array | None | _LazyArray,
    chain_axis: int | None,
    data_axis: int | None,
    mesh: Mesh | None,
) -> Array | None:
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
def _concretize_lazy_array(x: _LazyArray, sharding: NamedSharding | None) -> Array:
    """Create an array from an abstract spec on the appropriate devices."""
    x = x()
    if sharding is not None:
        x = lax.with_sharding_constraint(x, sharding)
    return x


def _all_none_or_not_none(*args: object) -> bool:
    is_none = [x is None for x in args]
    return all(is_none) or not any(is_none)


def _asarray_or_none(x: object) -> Array | None:
    if x is None:
        return None
    return jnp.asarray(x)


def _get_platform(mesh: Mesh | None) -> str:
    if mesh is None:
        return get_default_device().platform
    else:
        return mesh.devices.flat[0].platform


class _ReductionConfig(TypedDict):
    """Fields of `StepConfig` related to reductions."""

    resid_num_batches: int | None
    count_num_batches: int | None
    prec_num_batches: int | None
    prec_count_num_trees: int | None


def _parse_reduction_configs(
    resid_num_batches: int | None | Literal['auto'],
    count_num_batches: int | None | Literal['auto'],
    prec_num_batches: int | None | Literal['auto'],
    prec_count_num_trees: int | None | Literal['auto'],
    y: Float32[Array, ' n'] | Float32[Array, ' k n'] | Bool[Array, ' n'],
    num_trees: int,
    num_chains: int | None,
    mesh: Mesh | None,
    target_platform: Literal['cpu', 'gpu'] | None,
) -> _ReductionConfig:
    """Determine settings for indexed reduces."""
    n = y.shape[-1]
    n //= get_axis_size(mesh, 'data')  # per-device datapoints
    # chains are vmapped together on each device, so they share the per-step
    # memory of the per-tree reduction
    chains_per_device = (num_chains or 1) // get_axis_size(mesh, 'chains')
    parse_num_batches = partial(_parse_num_batches, target_platform, n)
    return dict(
        resid_num_batches=parse_num_batches(resid_num_batches, 'resid'),
        count_num_batches=parse_num_batches(count_num_batches, 'count'),
        prec_num_batches=parse_num_batches(prec_num_batches, 'prec'),
        prec_count_num_trees=_parse_prec_count_num_trees(
            prec_count_num_trees, num_trees, n * chains_per_device
        ),
    )


def _parse_num_batches(
    target_platform: Literal['cpu', 'gpu'] | None,
    n: int,
    num_batches: int | None | Literal['auto'],
    which: Literal['resid', 'count', 'prec'],
) -> int | None:
    """Return the number of batches or determine it automatically."""
    final_round = partial(_final_round, n)
    if num_batches != 'auto':
        nb = num_batches
    elif target_platform == 'cpu':
        nb = final_round(16)
    elif target_platform == 'gpu':
        nb = dict(resid=1024, count=2048, prec=1024)[which]  # on an A4000
        nb = final_round(nb)
    return nb


def _final_round(n: int, num: float) -> int | None:
    """Bound batch size, round number of batches to a power of 2, and disable batching if there's only 1 batch."""
    # at least some elements per batch
    num = min(n // 32, num)

    # round to the nearest power of 2 because I guess XLA and the hardware
    # will like that (not sure about this, maybe just multiple of 32?)
    num = 2 ** round(math.log2(num)) if num else 0

    # disable batching if the batch is as large as the whole dataset
    return num if num > 1 else None


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

        chain_axes = chain_vmap_axes(state)
        data_axes = data_vmap_axes(state)

        state_specs = tree.map(
            lambda x, ca, da: _leaf_partition_spec(x.ndim, ca, da, mesh),
            state,
            chain_axes,
            data_axes,
        )

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


def _get_shard_map_patch_kwargs() -> dict[str, bool]:
    # bug 1: jax 0.6.0-0.7.0: the VMA/check_rep checker rejects
    # random.gamma's internal while_loop when the key axis is sharded but the
    # alpha is replicated; fixed in jax 0.7.1.

    # bug 2: jax 0.8.1-0.8.2: vmap(shard_map(psum)), jax#34249; the
    # jax_disable_vmap_shmap_error config did not work.

    # WORKAROUND(jax<=0.8.2): remove this whole function when jax > 0.8.2
    buggy = ('0.6.0', '0.6.1', '0.6.2', '0.7.0', '0.8.1', '0.8.2')
    if jax.__version__ in buggy:
        # WORKAROUND(jax<0.6.1): check_rep instead of check_vma before then
        params = inspect.signature(shard_map).parameters
        flag = 'check_rep' if 'check_rep' in params else 'check_vma'
        return {flag: False}
    return {}
