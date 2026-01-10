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
from dataclasses import Field, fields
from functools import partial, wraps
from math import ceil, log2
from typing import Any, Literal, TypeVar

from equinox import Module, error_if
from equinox import field as eqx_field
from jax import NamedSharding, device_put, eval_shape, make_mesh, random, tree, vmap
from jax import numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.sharding import AxisType, Mesh, PartitionSpec
from jax.tree import flatten
from jaxtyping import Array, Bool, Float32, Int32, Integer, PyTree, Shaped, UInt

from bartz.grove import make_tree, tree_depths
from bartz.jaxext import get_default_device, is_key, minimal_unsigned_dtype


def field(*, chains: bool = False, data: bool = False, **kwargs) -> Field:
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
    if isinstance(x, Module):
        args = []
        for f in fields(x):
            v = getattr(x, f.name)
            if f.metadata.get('static', False):
                args.append(v)
            elif f.metadata.get(key, False):
                subtree = tree.map(lambda _: if_true, v)
                args.append(subtree)
            else:
                args.append(_find_metadata(v, key, if_true, if_false))
        return x.__class__(*args)

    def is_leaf(x) -> bool:
        return isinstance(x, Module)

    def get_axes(x: Module | Any) -> PyTree[T]:
        if isinstance(x, Module):
            return _find_metadata(x, key, if_true, if_false)
        else:
            return tree.map(lambda _: if_false, x)

    return tree.map(get_axes, x, is_leaf=is_leaf)


class Forest(Module):
    """
    Represents the MCMC state of a sum of trees.

    Parameters
    ----------
    leaf_tree
        The leaf values.
    var_tree
        The decision axes.
    split_tree
        The decision boundaries.
    affluence_tree
        Marks leaves that can be grown.
    max_split
        The maximum split index for each predictor.
    blocked_vars
        Indices of variables that are not used. This shall include at least
        the `i` such that ``max_split[i] == 0``, otherwise behavior is
        undefined.
    p_nonterminal
        The prior probability of each node being nonterminal, conditional on
        its ancestors. Includes the nodes at maximum depth which should be set
        to 0.
    p_propose_grow
        The unnormalized probability of picking a leaf for a grow proposal.
    leaf_indices
        The index of the leaf each datapoints falls into, for each tree.
    min_points_per_decision_node
        The minimum number of data points in a decision node.
    min_points_per_leaf
        The minimum number of data points in a leaf node.
    log_trans_prior
        The log transition and prior Metropolis-Hastings ratio for the
        proposed move on each tree.
    log_likelihood
        The log likelihood ratio.
    grow_prop_count
    prune_prop_count
        The number of grow/prune proposals made during one full MCMC cycle.
    grow_acc_count
    prune_acc_count
        The number of grow/prune moves accepted during one full MCMC cycle.
    leaf_prior_cov_inv
        The prior precision matrix of a leaf, conditional on the tree structure.
        For the univariate case (k=1), this is a scalar (the inverse variance).
        The prior covariance of the sum of trees is
        ``num_trees * leaf_prior_cov_inv^-1``.
    log_s
        The logarithm of the prior probability for choosing a variable to split
        along in a decision rule, conditional on the ancestors. Not normalized.
        If `None`, use a uniform distribution.
    theta
        The concentration parameter for the Dirichlet prior on the variable
        distribution `s`. Required only to update `s`.
    a
    b
    rho
        Parameters of the prior on `theta`. Required only to sample `theta`.
        See `step_theta`.
    """

    leaf_tree: (
        Float32[Array, '*chains num_trees 2**d']
        | Float32[Array, '*chains num_trees k 2**d']
    ) = field(chains=True)
    var_tree: UInt[Array, '*chains num_trees 2**(d-1)'] = field(chains=True)
    split_tree: UInt[Array, '*chains num_trees 2**(d-1)'] = field(chains=True)
    affluence_tree: Bool[Array, '*chains num_trees 2**(d-1)'] = field(chains=True)
    max_split: UInt[Array, ' p']
    blocked_vars: UInt[Array, ' q'] | None
    p_nonterminal: Float32[Array, ' 2**d']
    p_propose_grow: Float32[Array, ' 2**(d-1)']
    leaf_indices: UInt[Array, '*chains num_trees n'] = field(chains=True, data=True)
    min_points_per_decision_node: Int32[Array, ''] | None
    min_points_per_leaf: Int32[Array, ''] | None
    log_trans_prior: Float32[Array, '*chains num_trees'] | None = field(chains=True)
    log_likelihood: Float32[Array, '*chains num_trees'] | None = field(chains=True)
    grow_prop_count: Int32[Array, '*chains'] = field(chains=True)
    prune_prop_count: Int32[Array, '*chains'] = field(chains=True)
    grow_acc_count: Int32[Array, '*chains'] = field(chains=True)
    prune_acc_count: Int32[Array, '*chains'] = field(chains=True)
    leaf_prior_cov_inv: Float32[Array, ''] | Float32[Array, 'k k'] | None
    log_s: Float32[Array, '*chains p'] | None = field(chains=True)
    theta: Float32[Array, '*chains'] | None = field(chains=True)
    a: Float32[Array, ''] | None
    b: Float32[Array, ''] | None
    rho: Float32[Array, ''] | None

    def num_chains(self) -> int | None:
        """Return the number of chains, or `None` if not multichain."""
        # maybe this should be replaced by chain_shape() -> () | (int,)
        if self.var_tree.ndim == 2:
            return None
        else:
            return self.var_tree.shape[0]


class StepConfig(Module):
    """Options for the MCMC step.

    Parameters
    ----------
    steps_done
        The number of MCMC steps completed so far.
    sparse_on_at
        After how many steps to turn on variable selection.
    resid_batch_size
    count_batch_size
        The data batch sizes for computing the sufficient statistics. If `None`,
        they are computed with no batching.
    mesh
        The mesh used to shard data and computation across multiple devices.
    """

    steps_done: Int32[Array, '']
    sparse_on_at: Int32[Array, ''] | None
    resid_batch_size: int | None = field(static=True)
    count_batch_size: int | None = field(static=True)
    mesh: Mesh | None = field(static=True)


class State(Module):
    """
    Represents the MCMC state of BART.

    Parameters
    ----------
    X
        The predictors.
    y
        The response. If the data type is `bool`, the model is binary regression.
    resid
        The residuals (`y` or `z` minus sum of trees).
    z
        The latent variable for binary regression. `None` in continuous
        regression.
    offset
        Constant shift added to the sum of trees.
    error_cov_inv
        The inverse error covariance (scalar for univariate, matrix for multivariate).
        `None` in binary regression.
    prec_scale
        The scale on the error precision, i.e., ``1 / error_scale ** 2``.
        `None` in binary regression.
    error_cov_df
    error_cov_scale
        The df and scale parameters of the inverse Wishart prior on the noise
        covariance. For the univariate case, the relationship to the inverse
        gamma prior parameters is ``alpha = df / 2``, ``beta = scale / 2``.
        `None` in binary regression.
    forest
        The sum of trees model.
    config
        Metadata and configurations for the MCMC step.
    """

    X: UInt[Array, 'p n'] = field(data=True)
    y: Float32[Array, ' n'] | Float32[Array, ' k n'] | Bool[Array, ' n'] = field(
        data=True
    )
    z: None | Float32[Array, '*chains n'] = field(chains=True, data=True)
    offset: Float32[Array, ''] | Float32[Array, ' k']
    resid: Float32[Array, '*chains n'] | Float32[Array, '*chains k n'] = field(
        chains=True, data=True
    )
    error_cov_inv: Float32[Array, '*chains'] | Float32[Array, '*chains k k'] | None = (
        field(chains=True)
    )
    prec_scale: Float32[Array, ' n'] | None = field(data=True)
    error_cov_df: Float32[Array, ''] | None
    error_cov_scale: Float32[Array, ''] | Float32[Array, 'k k'] | None
    forest: Forest
    config: StepConfig


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
    resid_batch_size: int | None | Literal['auto'] = 'auto',
    count_batch_size: int | None | Literal['auto'] = 'auto',
    save_ratios: bool = False,
    filter_splitless_vars: bool = True,
    min_points_per_leaf: int | Integer[Any, ''] | None = None,
    log_s: Float32[Any, ' p'] | None = None,
    theta: float | Float32[Any, ''] | None = None,
    a: float | Float32[Any, ''] | None = None,
    b: float | Float32[Any, ''] | None = None,
    rho: float | Float32[Any, ''] | None = None,
    sparse_on_at: int | Integer[Any, ''] | None = None,
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
        of trees is fixed by the length of this array.
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
    resid_batch_size
    count_batch_size
        The batch sizes, along datapoints, for summing the residuals and
        counting the number of datapoints in each leaf. `None` for no batching.
        If 'auto', it's chosen automatically based on the devices used in `mesh`
        if specified, else the default device. Note: in the latter case, this
        won't respect the device of `y` if it's different from the default
        device.
    save_ratios
        Whether to save the Metropolis-Hastings ratios.
    filter_splitless_vars
        Whether to check `max_split` for variables without available cutpoints.
        If any are found, they are put into a list of variables to exclude from
        the MCMC. If `False`, no check is performed, but the results may be
        wrong if any variable is blocked. The function is jax-traceable only
        if this is set to `False`.
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
    tree_shape = (*chain_shape, num_trees)
    add_chains = partial(_add_chains, chain_shape=chain_shape)

    # determine batch sizes for reductions
    mesh = _parse_mesh(num_chains, mesh)
    resid_batch_size, count_batch_size = _choose_suffstat_batch_size(
        resid_batch_size, count_batch_size, y, max_depth, num_trees, num_chains, mesh
    )

    # initialize all remaining stuff and put it in an unsharded state
    state = State(
        X=X,
        y=y,
        z=jnp.full(resid_shape, offset) if is_binary else None,
        offset=offset,
        resid=jnp.zeros(resid_shape)
        if is_binary
        else jnp.broadcast_to(y - offset[..., None], resid_shape),
        error_cov_inv=add_chains(error_cov_inv),
        prec_scale=(
            None if error_scale is None else jnp.reciprocal(jnp.square(error_scale))
        ),
        error_cov_df=error_cov_df,
        error_cov_scale=error_cov_scale,
        forest=Forest(
            leaf_tree=make_tree(max_depth, jnp.float32, tree_shape + kshape),
            var_tree=make_tree(
                max_depth - 1, minimal_unsigned_dtype(p - 1), tree_shape
            ),
            split_tree=make_tree(max_depth - 1, max_split.dtype, tree_shape),
            affluence_tree=(
                make_tree(max_depth - 1, bool, tree_shape)
                .at[..., 1]
                .set(
                    True
                    if min_points_per_decision_node is None
                    else n >= min_points_per_decision_node
                )
            ),
            blocked_vars=_get_blocked_vars(filter_splitless_vars, max_split),
            max_split=max_split,
            grow_prop_count=jnp.zeros(chain_shape, int),
            grow_acc_count=jnp.zeros(chain_shape, int),
            prune_prop_count=jnp.zeros(chain_shape, int),
            prune_acc_count=jnp.zeros(chain_shape, int),
            p_nonterminal=p_nonterminal[tree_depths(2**max_depth)],
            p_propose_grow=p_nonterminal[tree_depths(2 ** (max_depth - 1))],
            leaf_indices=jnp.ones(
                (*tree_shape, n), minimal_unsigned_dtype(2**max_depth - 1)
            ),
            min_points_per_decision_node=_asarray_or_none(min_points_per_decision_node),
            min_points_per_leaf=_asarray_or_none(min_points_per_leaf),
            log_trans_prior=jnp.zeros((*chain_shape, num_trees))
            if save_ratios
            else None,
            log_likelihood=jnp.zeros((*chain_shape, num_trees))
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
            resid_batch_size=resid_batch_size,
            count_batch_size=count_batch_size,
            mesh=mesh,
        ),
    )

    # move all arrays to the appropriate device
    return _shard_state(state)


def _get_blocked_vars(
    filter_splitless_vars: bool, max_split: UInt[Array, ' p']
) -> None | UInt[Array, ' q']:
    """Initialize the `blocked_vars` field."""
    if filter_splitless_vars:
        (p,) = max_split.shape
        (blocked_vars,) = jnp.nonzero(max_split == 0)
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


def _auto_axes(mesh: Mesh) -> list[str]:
    """Re-implement `Mesh.auto_axes` because that's missing in jax 5."""
    return [
        n
        for n, t in zip(mesh.axis_names, mesh.axis_types, strict=True)
        if t == AxisType.Auto
    ]


def _shard_state(state: State) -> State:
    """Place all fields in the state on the appropriate devices."""
    mesh = state.config.mesh
    if mesh is None:
        return state

    def shard_leaf(
        x: Array | None, chain_axis: int | None, data_axis: int | None
    ) -> Array | None:
        if x is None:
            return None

        spec = [None] * x.ndim
        if chain_axis is not None and 'chains' in mesh.axis_names:
            spec[chain_axis] = 'chains'
        if data_axis is not None and 'data' in mesh.axis_names:
            spec[data_axis] = 'data'

        spec = PartitionSpec(*spec)
        return device_put(x, NamedSharding(mesh, spec))

    return tree.map(
        shard_leaf,
        state,
        chain_vmap_axes(state),
        data_vmap_axes(state),
        is_leaf=lambda x: x is None,
    )


def _all_none_or_not_none(*args):
    is_none = [x is None for x in args]
    return all(is_none) or not any(is_none)


def _asarray_or_none(x):
    if x is None:
        return None
    return jnp.asarray(x)


def _get_platform(mesh: Mesh | None) -> str:
    if mesh is None:
        return get_default_device().platform
    else:
        return mesh.devices.flat[0].platform


def _choose_suffstat_batch_size(
    resid_batch_size: int | None | Literal['auto'],
    count_batch_size: int | None | Literal['auto'],
    y: Float32[Array, ' n'] | Float32[Array, ' k n'] | Bool[Array, ' n'],
    max_depth: int,
    num_trees: int,
    num_chains: int | None,
    mesh: Mesh | None,
) -> tuple[int | None, int | None]:
    """Determine batch sizes for reductions."""
    # get number of outcomes and of datapoints
    k, n = _get_k_n(y)

    # get per-device values
    if num_chains is None:
        num_chains = 1
    num_chains //= _get_axis_size(mesh, 'chains')
    n //= _get_axis_size(mesh, 'data')

    # compute auxiliary sizes
    batch_size = k * num_chains
    unbatched_accum_bytes_times_batch_size = num_trees * 2**max_depth * 4 * n

    platform = _get_platform(mesh)

    def final_round(s: float) -> int | None:
        # multiply by batch_size because if the calculation is already
        # parallelizable over batching dims there is correspondingly less need
        # to parallelize across datapoints
        s *= batch_size

        # at least 1, i.e., each datapoint is its own batch
        s = max(1, s)

        # round to the nearest power of 2 because I guess XLA and the hardware
        # will like that
        s = 2 ** round(log2(s))

        # disable batching if the batch is as large as the whole dataset
        return s if s < n else None

    if resid_batch_size != 'auto':
        rbs = resid_batch_size
    elif platform == 'cpu':
        rbs = final_round(n / 6)
        # instead of 6 I guess I should have in general the number of "good"
        # physical cores
    elif platform == 'gpu':
        rbs = final_round((2 * n) ** (1 / 3))

    if count_batch_size != 'auto':
        cbs = count_batch_size
    elif platform == 'cpu':
        cbs = None
    elif platform == 'gpu':
        cbs = (n / 16) ** 0.5

        # ensure we don't exceed ~512MiB of memory usage per device
        max_memory = 2**29
        min_batch_size = ceil(unbatched_accum_bytes_times_batch_size / max_memory)
        cbs = max(cbs, min_batch_size)

        cbs = final_round(cbs)

    return rbs, cbs


def _get_axis_size(mesh: Mesh | None, axis_name: str) -> int:
    if mesh is None or axis_name not in mesh.axis_names:
        return 1
    else:
        i = mesh.axis_names.index(axis_name)
        return mesh.axis_sizes[i]


def _get_k_n(y: Array) -> tuple[int, int]:
    if y.ndim == 2:
        return y.shape
    else:
        (n,) = y.shape
        return 1, n


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
    leaves, _ = flatten(x, is_leaf=lambda x: hasattr(x, 'num_chains'))
    num_chains = [x.num_chains() for x in leaves if hasattr(x, 'num_chains')]
    ref = num_chains[0]
    assert all(c == ref for c in num_chains)
    return ref


def _chain_axes_with_keys(x: PyTree) -> PyTree[int | None]:
    """Return `chain_vmap_axes(x)` but also set to 0 for random keys."""
    axes = chain_vmap_axes(x)

    def axis_if_key(x, axis):
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

    def find_mesh(x: State | Any):
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

    def split_key(x):
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
    def auto_vmapped_fun(*args, **kwargs) -> T:
        all_args = args, kwargs
        num_chains = get_num_chains(all_args)
        if num_chains is not None:
            if auto_split_keys:
                all_args = _split_all_keys(all_args, num_chains)

            def wrapped_fun(args, kwargs):
                return fun(*args, **kwargs)

            mc_in_axes = _chain_axes_with_keys(all_args)
            mc_out_axes = _get_mc_out_axes(wrapped_fun, all_args, mc_in_axes)
            vmapped_fun = vmap(wrapped_fun, in_axes=mc_in_axes, out_axes=mc_out_axes)
            return vmapped_fun(*all_args)

        else:
            return fun(*args, **kwargs)

    return auto_vmapped_fun
