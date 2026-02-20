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

from collections.abc import Callable, Hashable
from dataclasses import fields, replace
from functools import partial, wraps
from math import log2
from typing import Any, Literal, TypedDict, TypeVar

import numpy
from equinox import Module, error_if, filter_jit
from equinox import field as eqx_field
from jax import (
    NamedSharding,
    device_put,
    eval_shape,
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
from jaxtyping import Array, Bool, Float32, Int32, Integer, PyTree, Shaped, UInt

from bartz.grove import tree_depths
from bartz.jaxext import get_default_device, is_key, minimal_unsigned_dtype


def field(*, chains: bool = False, data: bool = False, **kwargs: Any):  # noqa: ANN202
    """Extend `equinox.field` with two new parameters.

    Parameters
    ----------
    chains
        Whether the arrays in the field have an optional first axis that
        represents independent Markov chains.
    data
        Whether the last axis of the arrays in the field represent units of
        the data.
    **kwargs
        Other parameters passed to `equinox.field`.

    Returns
    -------
    A dataclass field descriptor with the special attributes in the metadata, unset if False.
    """
    metadata = dict(kwargs.pop('metadata', {}))
    assert 'chains' not in metadata
    assert 'data' not in metadata
    if chains:
        metadata['chains'] = True
    if data:
        metadata['data'] = True
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
        ``field(..., chains=True)`` are considered to have a leading chain axis.

    Returns
    -------
    A pytree with the same structure as `x` with 0 or None in the leaves.
    """
    return _find_metadata(x, 'chains', 0, None)


def data_vmap_axes(x: PyTree[Module | Any, 'T']) -> PyTree[int | None, 'T']:
    """Determine vmapping axes for data.

    This is analogous to `chain_vmap_axes` but returns -1 for all fields
    marked with ``field(..., data=True)``.
    """
    return _find_metadata(x, 'data', -1, None)


T = TypeVar('T')


def _find_metadata(
    x: PyTree[Any, ' S'], key: Hashable, if_true: T, if_false: T
) -> PyTree[T, ' S']:
    """Replace all subtrees of x marked with a metadata key."""

    def is_lazy_array(x: object) -> bool:
        return isinstance(x, _LazyArray)

    def is_module(x: object) -> bool:
        return isinstance(x, Module) and not is_lazy_array(x)

    if is_module(x):
        args = []
        for f in fields(x):
            v = getattr(x, f.name)
            if f.metadata.get('static', False):
                args.append(v)
            elif f.metadata.get(key, False):
                subtree = tree.map(lambda _: if_true, v, is_leaf=is_lazy_array)
                args.append(subtree)
            else:
                args.append(_find_metadata(v, key, if_true, if_false))
        return x.__class__(*args)

    def get_axes(x: object) -> PyTree[T]:
        if is_module(x):
            return _find_metadata(x, key, if_true, if_false)
        else:
            return tree.map(lambda _: if_false, x, is_leaf=is_lazy_array)

    def is_leaf(x: object) -> bool:
        return isinstance(x, Module)  # this catches _LazyArray as well

    return tree.map(get_axes, x, is_leaf=is_leaf)


class Forest(Module):
    """Represents the MCMC state of a sum of trees."""

    leaf_tree: (
        Float32[Array, '*chains num_trees 2**d']
        | Float32[Array, '*chains num_trees k 2**d']
    ) = field(chains=True)
    """The leaf values."""

    var_tree: UInt[Array, '*chains num_trees 2**(d-1)'] = field(chains=True)
    """The decision axes."""

    split_tree: UInt[Array, '*chains num_trees 2**(d-1)'] = field(chains=True)
    """The decision boundaries."""

    affluence_tree: Bool[Array, '*chains num_trees 2**(d-1)'] = field(chains=True)
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

    leaf_indices: UInt[Array, '*chains num_trees n'] = field(chains=True, data=True)
    """The index of the leaf each datapoints falls into, for each tree."""

    min_points_per_decision_node: Int32[Array, ''] | None
    """The minimum number of data points in a decision node."""

    min_points_per_leaf: Int32[Array, ''] | None
    """The minimum number of data points in a leaf node."""

    log_trans_prior: Float32[Array, '*chains num_trees'] | None = field(chains=True)
    """The log transition and prior Metropolis-Hastings ratio for the
    proposed move on each tree."""

    log_likelihood: Float32[Array, '*chains num_trees'] | None = field(chains=True)
    """The log likelihood ratio."""

    grow_prop_count: Int32[Array, '*chains'] = field(chains=True)
    """The number of grow proposals made during one full MCMC cycle."""

    prune_prop_count: Int32[Array, '*chains'] = field(chains=True)
    """The number of prune proposals made during one full MCMC cycle."""

    grow_acc_count: Int32[Array, '*chains'] = field(chains=True)
    """The number of grow moves accepted during one full MCMC cycle."""

    prune_acc_count: Int32[Array, '*chains'] = field(chains=True)
    """The number of prune moves accepted during one full MCMC cycle."""

    leaf_prior_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'] | None
    """The prior precision matrix of a leaf, conditional on the tree structure.
    For the univariate case (k=1), this is a scalar (the inverse variance).
    The prior covariance of the sum of trees is
    ``num_trees * leaf_prior_cov_inv^-1``."""

    log_s: Float32[Array, '*chains p'] | None = field(chains=True)
    """The logarithm of the prior probability for choosing a variable to split
    along in a decision rule, conditional on the ancestors. Not normalized.
    If `None`, use a uniform distribution."""

    theta: Float32[Array, '*chains'] | None = field(chains=True)
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

    def num_chains(self) -> int | None:
        """Return the number of chains, or `None` if not multichain."""
        # maybe this should be replaced by chain_shape() -> () | (int,)
        if self.var_tree.ndim == 2:
            return None
        else:
            return self.var_tree.shape[0]


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


class State(Module):
    """Represents the MCMC state of BART."""

    X: UInt[Array, 'p n'] = field(data=True)
    """The predictors."""

    y: Float32[Array, ' n'] | Float32[Array, ' k n'] | Bool[Array, ' n'] = field(
        data=True
    )
    """The response. If the data type is `bool`, the model is binary regression."""

    z: None | Float32[Array, '*chains n'] = field(chains=True, data=True)
    """The latent variable for binary regression. `None` in continuous
    regression."""

    offset: Float32[Array, ''] | Float32[Array, ' k']
    """Constant shift added to the sum of trees."""

    resid: Float32[Array, '*chains n'] | Float32[Array, '*chains k n'] = field(
        chains=True, data=True
    )
    """The residuals (`y` or `z` minus sum of trees)."""

    error_cov_inv: Float32[Array, '*chains'] | Float32[Array, '*chains k k'] | None = (
        field(chains=True)
    )
    """The inverse error covariance (scalar for univariate, matrix for multivariate).
    `None` in binary regression."""

    prec_scale: Float32[Array, ' n'] | None = field(data=True)
    """The scale on the error precision, i.e., ``1 / error_scale ** 2``.
    `None` in binary regression."""

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


def _init_shape_shifting_parameters(
    y: Float32[Array, ' n'] | Float32[Array, 'k n'] | Bool[Array, ' n'],
    offset: Float32[Array, ''] | Float32[Array, ' k'],
    error_scale: Float32[Any, ' n'] | None,
    error_cov_df: float | Float32[Any, ''] | None,
    error_cov_scale: float | Float32[Any, ''] | Float32[Any, 'k k'] | None,
    leaf_prior_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'],
) -> tuple[
    bool,
    tuple[()] | tuple[int],
    None | Float32[Array, ''],
    None | Float32[Array, ''],
    None | Float32[Array, ''],
]:
    """
    Check and initialize parameters that change array type/shape based on outcome kind.

    Parameters
    ----------
    y
        The response variable; the outcome type is deduced from `y` and then
        all other parameters are checked against it.
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

    Returns
    -------
    is_binary
        Whether the outcome is binary.
    kshape
        The outcome shape, empty for univariate, (k,) for multivariate.
    error_cov_inv
        The initialized error covariance inverse.
    error_cov_df
        The error covariance degrees of freedom (as array).
    error_cov_scale
        The error covariance scale (as array).

    Raises
    ------
    ValueError
        If `y` is binary and multivariate.
    """
    # determine outcome kind, binary/continuous x univariate/multivariate
    is_binary = y.dtype == bool
    kshape = y.shape[:-1]

    # Binary vs continuous
    if is_binary:
        if kshape:
            msg = 'Binary multivariate regression not supported, open an issue at https://github.com/bartz-org/bartz/issues if you need it.'
            raise ValueError(msg)
        assert error_scale is None
        assert error_cov_df is None
        assert error_cov_scale is None
        error_cov_inv = None
    else:
        error_cov_df = jnp.asarray(error_cov_df)
        error_cov_scale = jnp.asarray(error_cov_scale)
        assert error_cov_scale.shape == 2 * kshape

        # Multivariate vs univariate
        if kshape:
            error_cov_inv = error_cov_df * _inv_via_chol_with_gersh(error_cov_scale)
        else:
            # inverse gamma prior: alpha = df / 2, beta = scale / 2
            error_cov_inv = error_cov_df / error_cov_scale

    assert leaf_prior_cov_inv.shape == 2 * kshape
    assert offset.shape == kshape

    return is_binary, kshape, error_cov_inv, error_cov_df, error_cov_scale


def _parse_p_nonterminal(
    p_nonterminal: Float32[Any, ' d_minus_1'],
) -> Float32[Array, ' d_minus_1+1']:
    """Check it's in (0, 1) and pad with a 0 at the end."""
    p_nonterminal = jnp.asarray(p_nonterminal)
    ok = (p_nonterminal > 0) & (p_nonterminal < 1)
    p_nonterminal = error_if(p_nonterminal, ~ok, 'p_nonterminal must be in (0, 1)')
    return jnp.pad(p_nonterminal, (0, 1))


def make_p_nonterminal(
    d: int,
    alpha: float | Float32[Array, ''] = 0.95,
    beta: float | Float32[Array, ''] = 2.0,
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


def init(
    *,
    X: UInt[Any, 'p n'],
    y: Float32[Any, ' n'] | Float32[Any, ' k n'] | Bool[Any, ' n'],
    offset: float | Float32[Any, ''] | Float32[Any, ' k'],
    max_split: UInt[Any, ' p'],
    num_trees: int,
    p_nonterminal: Float32[Any, ' d_minus_1'],
    leaf_prior_cov_inv: float | Float32[Any, ''] | Float32[Array, 'k k'],
    error_cov_df: float | Float32[Any, ''] | None = None,
    error_cov_scale: float | Float32[Any, ''] | Float32[Array, 'k k'] | None = None,
    error_scale: Float32[Any, ' n'] | None = None,
    min_points_per_decision_node: int | Integer[Any, ''] | None = None,
    resid_num_batches: int | None | Literal['auto'] = 'auto',
    count_num_batches: int | None | Literal['auto'] = 'auto',
    prec_num_batches: int | None | Literal['auto'] = 'auto',
    prec_count_num_trees: int | None | Literal['auto'] = 'auto',
    save_ratios: bool = False,
    filter_splitless_vars: int = 0,
    min_points_per_leaf: int | Integer[Any, ''] | None = None,
    log_s: Float32[Any, ' p'] | None = None,
    theta: float | Float32[Any, ''] | None = None,
    a: float | Float32[Any, ''] | None = None,
    b: float | Float32[Any, ''] | None = None,
    rho: float | Float32[Any, ''] | None = None,
    sparse_on_at: int | Integer[Any, ''] | None = None,
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
        The response. If the data type is `bool`, the regression model is binary
        regression with probit. If two-dimensional, the outcome is multivariate
        with the first axis indicating the component.
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
        Each error is scaled by the corresponding factor in `error_scale`, so
        the error variance for ``y[i]`` is ``sigma2 * error_scale[i] ** 2``.
        Not supported for binary regression. If not specified, defaults to 1 for
        all points, but potentially skipping calculations.
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
        If `y` is boolean and arguments unused in binary regression are set.

    Notes
    -----
    In decision nodes, the values in ``X[i, :]`` are compared to a cutpoint out
    of the range ``[1, 2, ..., max_split[i]]``. A point belongs to the left
    child iff ``X[i, j] < cutpoint``. Thus it makes sense for ``X[i, :]`` to be
    integers in the range ``[0, 1, ..., max_split[i]]``.
    """
    # convert to array all array-like arguments that are used in other
    # configurations but don't need further processing themselves
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    leaf_prior_cov_inv = jnp.asarray(leaf_prior_cov_inv)
    max_split = jnp.asarray(max_split)

    # check p_nonterminal and pad it with a 0 at the end (still not final shape)
    p_nonterminal = _parse_p_nonterminal(p_nonterminal)

    # process arguments that change depending on outcome type
    is_binary, kshape, error_cov_inv, error_cov_df, error_cov_scale = (
        _init_shape_shifting_parameters(
            y, offset, error_scale, error_cov_df, error_cov_scale, leaf_prior_cov_inv
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

    # process multichain settings
    chain_shape = () if num_chains is None else (num_chains,)
    resid_shape = chain_shape + y.shape
    add_chains = partial(_add_chains, chain_shape=chain_shape)

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
        mesh,
        target_platform,
    )

    # check there aren't too many deactivated predictors
    msg = (
        f'there are more than {filter_splitless_vars=} predictors with no splits, '
        'please increase `filter_splitless_vars` or investigate the missing splits'
    )
    offset = error_if(offset, jnp.sum(max_split == 0) > filter_splitless_vars, msg)

    # determine shapes for trees
    tree_shape = (*chain_shape, num_trees)
    tree_size = 2**max_depth

    # initialize all remaining stuff and put it in an unsharded state
    state = State(
        X=X,
        y=y,
        z=_LazyArray(jnp.full, resid_shape, offset) if is_binary else None,
        offset=offset,
        resid=_LazyArray(jnp.zeros, resid_shape)
        if is_binary
        else None,  # in this case, resid is created later after y and offset are sharded
        error_cov_inv=add_chains(error_cov_inv),
        prec_scale=error_scale,  # temporarily set to error_scale, fix after sharding
        error_cov_df=error_cov_df,
        error_cov_scale=error_cov_scale,
        forest=Forest(
            leaf_tree=_LazyArray(
                jnp.zeros, (*tree_shape, *kshape, tree_size), jnp.float32
            ),
            var_tree=_LazyArray(
                jnp.zeros, (*tree_shape, tree_size // 2), minimal_unsigned_dtype(p - 1)
            ),
            split_tree=_LazyArray(
                jnp.zeros, (*tree_shape, tree_size // 2), max_split.dtype
            ),
            affluence_tree=_LazyArray(
                _initial_affluence_tree,
                (*tree_shape, tree_size // 2),
                n,
                min_points_per_decision_node,
            ),
            blocked_vars=_get_blocked_vars(filter_splitless_vars, max_split),
            max_split=max_split,
            grow_prop_count=_LazyArray(jnp.zeros, chain_shape, int),
            grow_acc_count=_LazyArray(jnp.zeros, chain_shape, int),
            prune_prop_count=_LazyArray(jnp.zeros, chain_shape, int),
            prune_acc_count=_LazyArray(jnp.zeros, chain_shape, int),
            p_nonterminal=p_nonterminal[tree_depths(tree_size)],
            p_propose_grow=p_nonterminal[tree_depths(tree_size // 2)],
            leaf_indices=_LazyArray(
                jnp.ones, (*tree_shape, n), minimal_unsigned_dtype(tree_size - 1)
            ),
            min_points_per_decision_node=_asarray_or_none(min_points_per_decision_node),
            min_points_per_leaf=_asarray_or_none(min_points_per_leaf),
            log_trans_prior=_LazyArray(jnp.zeros, (*chain_shape, num_trees))
            if save_ratios
            else None,
            log_likelihood=_LazyArray(jnp.zeros, (*chain_shape, num_trees))
            if save_ratios
            else None,
            leaf_prior_cov_inv=leaf_prior_cov_inv,
            log_s=add_chains(_asarray_or_none(log_s)),
            theta=add_chains(_asarray_or_none(theta)),
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

    # delete big input arrays such that they can be deleted as soon as they
    # are sharded, only those arrays that contain an (n,) sized axis
    del X, y, error_scale

    # move all arrays to the appropriate device
    state = _shard_state(state)

    # calculate initial resid in the continuous outcome case, such that y and
    # offset are already sharded if needed
    if state.resid is None:
        resid = _LazyArray(_initial_resid, resid_shape, state.y, state.offset)
        resid = _shard_leaf(resid, 0, -1, state.config.mesh)
        state = replace(state, resid=resid)

    # calculate prec_scale after sharding to do the calculation on the right
    # devices
    if state.prec_scale is not None:
        prec_scale = _compute_prec_scale(state.prec_scale)
        state = replace(state, prec_scale=prec_scale)

    # make all types strong to avoid unwanted recompilations
    return _remove_weak_types(state)


def _initial_resid(
    shape: tuple[int, ...],
    y: Float32[Array, ' n'] | Float32[Array, 'k n'],
    offset: Float32[Array, ''] | Float32[Array, ' k'],
) -> Float32[Array, ' n'] | Float32[Array, 'k n']:
    """Calculate the initial value for `State.resid` in the continuous outcome case."""
    return jnp.broadcast_to(y - offset[..., None], shape)


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


@partial(jit, donate_argnums=(0,))
def _compute_prec_scale(error_scale: Float32[Array, ' n']) -> Float32[Array, ' n']:
    """Compute 1 / error_scale**2.

    This is a separate function to use donate_argnums to avoid intermediate
    copies.
    """
    return jnp.reciprocal(jnp.square(error_scale))


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


def _add_chains(
    x: Shaped[Array, '*shape'] | None, chain_shape: tuple[int, ...]
) -> Shaped[Array, '*shape'] | Shaped[Array, ' num_chains *shape'] | None:
    """Broadcast `x` to all chains."""
    if x is None:
        return None
    else:
        return jnp.broadcast_to(x, chain_shape + x.shape)


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
    assert 'chains' not in mesh.axis_names or 'chains' in _auto_axes(mesh)
    assert 'data' not in mesh.axis_names or 'data' in _auto_axes(mesh)

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


def _auto_axes(mesh: Mesh) -> list[str]:
    """Re-implement `Mesh.auto_axes` because that's missing in jax v0.5."""
    # Mesh.auto_axes added in jax v0.6.0
    return [
        n
        for n, t in zip(mesh.axis_names, mesh.axis_types, strict=True)
        if t == AxisType.Auto
    ]


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
        spec = [None] * x.ndim
        if chain_axis is not None and 'chains' in mesh.axis_names:
            spec[chain_axis] = 'chains'
        if data_axis is not None and 'data' in mesh.axis_names:
            spec[data_axis] = 'data'

        # remove trailing Nones to be consistent with jax's output, it's useful
        # for comparing shardings during debugging
        while spec and spec[-1] is None:
            spec.pop()

        spec = PartitionSpec(*spec)
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
    mesh: Mesh | None,
    target_platform: Literal['cpu', 'gpu'] | None,
) -> _ReductionConfig:
    """Determine settings for indexed reduces."""
    n = y.shape[-1]
    n //= get_axis_size(mesh, 'data')  # per-device datapoints
    parse_num_batches = partial(_parse_num_batches, target_platform, n)
    return dict(
        resid_num_batches=parse_num_batches(resid_num_batches, 'resid'),
        count_num_batches=parse_num_batches(count_num_batches, 'count'),
        prec_num_batches=parse_num_batches(prec_num_batches, 'prec'),
        prec_count_num_trees=_parse_prec_count_num_trees(
            prec_count_num_trees, num_trees, n
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
    num = 2 ** round(log2(num)) if num else 0

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
    if mesh is None or axis_name not in mesh.axis_names:
        return 1
    else:
        i = mesh.axis_names.index(axis_name)
        return mesh.axis_sizes[i]


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


def _inv_via_chol_with_gersh(mat: Float32[Array, 'k k']) -> Float32[Array, 'k k']:
    """Compute matrix inverse via Cholesky with Gershgorin stabilization.

    DO NOT USE THIS FUNCTION UNLESS YOU REALLY NEED TO.
    """
    L = chol_with_gersh(mat)
    I = jnp.eye(mat.shape[0], dtype=mat.dtype)
    L_inv = solve_triangular(L, I, lower=True)
    return L_inv.T @ L_inv


def get_num_chains(x: PyTree) -> int | None:
    """Get the number of chains of a pytree.

    Find all nodes in the structure that define 'num_chains()', stopping
    traversal at nodes that define it. Check all values obtained invoking
    `num_chains` are equal, then return it.
    """
    leaves, _ = tree.flatten(x, is_leaf=lambda x: hasattr(x, 'num_chains'))
    num_chains = [x.num_chains() for x in leaves if hasattr(x, 'num_chains')]
    ref = num_chains[0]
    assert all(c == ref for c in num_chains)
    return ref


def _chain_axes_with_keys(x: PyTree) -> PyTree[int | None]:
    """Return `chain_vmap_axes(x)` but also set to 0 for random keys."""
    axes = chain_vmap_axes(x)

    def axis_if_key(x: object, axis: int | None) -> int | None:
        if is_key(x):
            return 0
        else:
            return axis

    return tree.map(axis_if_key, x, axes)


def _get_mc_out_axes(
    fun: Callable[[tuple, dict], PyTree], args: PyTree, in_axes: PyTree[int | None]
) -> PyTree[int | None]:
    """Decide chain vmap axes for outputs."""
    vmapped_fun = vmap(fun, in_axes=in_axes)
    out = eval_shape(vmapped_fun, *args)
    return chain_vmap_axes(out)


def _find_mesh(x: PyTree) -> Mesh | None:
    """Find the mesh used for chains."""

    class MeshFound(Exception):
        pass

    def find_mesh(x: object) -> None:
        if isinstance(x, State):
            raise MeshFound(x.config.mesh)

    try:
        tree.map(find_mesh, x, is_leaf=lambda x: isinstance(x, State))
    except MeshFound as e:
        return e.args[0]
    else:
        raise ValueError


def _split_all_keys(x: PyTree, num_chains: int) -> PyTree:
    """Split all random keys in `num_chains` keys."""
    mesh = _find_mesh(x)

    def split_key(x: object) -> object:
        if is_key(x):
            x = random.split(x, num_chains)
            if mesh is not None and 'chains' in mesh.axis_names:
                x = device_put(x, NamedSharding(mesh, PartitionSpec('chains')))
        return x

    return tree.map(split_key, x)


def vmap_chains(
    fun: Callable[..., T], *, auto_split_keys: bool = False
) -> Callable[..., T]:
    """Apply vmap on chain axes automatically if the inputs are multichain."""

    @wraps(fun)
    def auto_vmapped_fun(*args: Any, **kwargs: Any) -> T:
        all_args = args, kwargs
        num_chains = get_num_chains(all_args)
        if num_chains is not None:
            if auto_split_keys:
                all_args = _split_all_keys(all_args, num_chains)

            def wrapped_fun(args: tuple[Any, ...], kwargs: dict[str, Any]) -> T:
                return fun(*args, **kwargs)

            mc_in_axes = _chain_axes_with_keys(all_args)
            mc_out_axes = _get_mc_out_axes(wrapped_fun, all_args, mc_in_axes)
            vmapped_fun = vmap(wrapped_fun, in_axes=mc_in_axes, out_axes=mc_out_axes)
            return vmapped_fun(*all_args)

        else:
            return fun(*args, **kwargs)

    return auto_vmapped_fun
