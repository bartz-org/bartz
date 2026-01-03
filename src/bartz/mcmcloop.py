# bartz/src/bartz/mcmcloop.py
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

"""Functions that implement the full BART posterior MCMC loop.

The entry points are `run_mcmc` and `make_default_callback`.
"""

from collections.abc import Callable
from dataclasses import fields
from functools import partial, wraps
from typing import Any, Protocol

import jax
import numpy
from equinox import Module
from jax import debug, jit, lax, tree
from jax import numpy as jnp
from jax.nn import softmax
from jaxtyping import Array, Bool, Float32, Int32, Integer, Key, PyTree, Shaped, UInt

from bartz import grove, jaxext, mcmcstep
from bartz._profiler import (
    callback_if_not_profiling,
    cond_if_not_profiling,
    jit_if_not_profiling,
    scan_if_not_profiling,
)
from bartz.mcmcstep import State


class BurninTrace(Module):
    """MCMC trace with only diagnostic values."""

    error_cov_inv: (
        Float32[Array, '*trace_length'] | Float32[Array, '*trace_length k k'] | None
    )
    theta: Float32[Array, '*trace_length'] | None
    grow_prop_count: Int32[Array, '*trace_length']
    grow_acc_count: Int32[Array, '*trace_length']
    prune_prop_count: Int32[Array, '*trace_length']
    prune_acc_count: Int32[Array, '*trace_length']
    log_likelihood: Float32[Array, '*trace_length'] | None
    log_trans_prior: Float32[Array, '*trace_length'] | None

    @classmethod
    def from_state(cls, state: State) -> 'BurninTrace':
        """Create a single-item burn-in trace from a MCMC state."""
        return cls(
            error_cov_inv=state.error_cov_inv,
            theta=state.forest.theta,
            grow_prop_count=state.forest.grow_prop_count,
            grow_acc_count=state.forest.grow_acc_count,
            prune_prop_count=state.forest.prune_prop_count,
            prune_acc_count=state.forest.prune_acc_count,
            log_likelihood=state.forest.log_likelihood,
            log_trans_prior=state.forest.log_trans_prior,
        )


class MainTrace(BurninTrace):
    """MCMC trace with trees and diagnostic values."""

    leaf_tree: Float32[Array, '*trace_length 2**d']
    var_tree: UInt[Array, '*trace_length 2**(d-1)']
    split_tree: UInt[Array, '*trace_length 2**(d-1)']
    offset: Float32[Array, '*trace_length']
    varprob: Float32[Array, '*trace_length p'] | None

    @classmethod
    def from_state(cls, state: State) -> 'MainTrace':
        """Create a single-item main trace from a MCMC state."""
        # compute varprob
        log_s = state.forest.log_s
        if log_s is None:
            varprob = None
        else:
            varprob = softmax(log_s, where=state.forest.max_split.astype(bool))

        return cls(
            leaf_tree=state.forest.leaf_tree,
            var_tree=state.forest.var_tree,
            split_tree=state.forest.split_tree,
            offset=state.offset,
            varprob=varprob,
            **vars(BurninTrace.from_state(state)),
        )


CallbackState = PyTree[Any, 'T']


class Callback(Protocol):
    """Callback type for `run_mcmc`."""

    def __call__(
        self,
        *,
        key: Key[Array, ''],
        bart: State,
        burnin: Bool[Array, ''],
        i_total: Int32[Array, ''],
        i_skip: Int32[Array, ''],
        callback_state: CallbackState,
        n_burn: Int32[Array, ''],
        n_save: Int32[Array, ''],
        n_skip: Int32[Array, ''],
        i_outer: Int32[Array, ''],
        inner_loop_length: int,
    ) -> tuple[State, CallbackState] | None:
        """Do an arbitrary action after an iteration of the MCMC.

        Parameters
        ----------
        key
            A key for random number generation.
        bart
            The MCMC state just after updating it.
        burnin
            Whether the last iteration was in the burn-in phase.
        i_total
            The index of the last MCMC iteration (0-based).
        i_skip
            The number of MCMC updates from the last saved state. The initial
            state counts as saved, even if it's not copied into the trace.
        callback_state
            The callback state, initially set to the argument passed to
            `run_mcmc`, afterwards to the value returned by the last invocation
            of the callback.
        n_burn
        n_save
        n_skip
            The corresponding `run_mcmc` arguments as-is.
        i_outer
            The index of the last outer loop iteration (0-based).
        inner_loop_length
            The number of MCMC iterations in the inner loop.

        Returns
        -------
        bart : State
            A possibly modified MCMC state. To avoid modifying the state,
            return the `bart` argument passed to the callback as-is.
        callback_state : CallbackState
            The new state to be passed on the next callback invocation.

        Notes
        -----
        For convenience, the callback may return `None`, and the states won't
        be updated.
        """
        ...


class _Carry(Module):
    """Carry used in the loop in `run_mcmc`."""

    bart: State
    i_total: Int32[Array, '']
    key: Key[Array, '']
    burnin_trace: PyTree[Shaped[Array, 'n_burn *']]
    main_trace: PyTree[Shaped[Array, 'n_save *']]
    callback_state: CallbackState


def run_mcmc(
    key: Key[Array, ''],
    bart: State,
    n_save: int,
    *,
    n_burn: int = 0,
    n_skip: int = 1,
    inner_loop_length: int | None = None,
    callback: Callback | None = None,
    callback_state: CallbackState = None,
    burnin_extractor: Callable[[State], PyTree] = BurninTrace.from_state,
    main_extractor: Callable[[State], PyTree] = MainTrace.from_state,
) -> tuple[State, PyTree[Shaped[Array, 'n_burn *']], PyTree[Shaped[Array, 'n_save *']]]:
    """
    Run the MCMC for the BART posterior.

    Parameters
    ----------
    key
        A key for random number generation.
    bart
        The initial MCMC state, as created and updated by the functions in
        `bartz.mcmcstep`. The MCMC loop uses buffer donation to avoid copies,
        so this variable is invalidated after running `run_mcmc`. Make a copy
        beforehand to use it again.
    n_save
        The number of iterations to save.
    n_burn
        The number of initial iterations which are not saved.
    n_skip
        The number of iterations to skip between each saved iteration, plus 1.
        The effective burn-in is ``n_burn + n_skip - 1``.
    inner_loop_length
        The MCMC loop is split into an outer and an inner loop. The outer loop
        is in Python, while the inner loop is in JAX. `inner_loop_length` is the
        number of iterations of the inner loop to run for each iteration of the
        outer loop. If not specified, the outer loop will iterate just once,
        with all iterations done in a single inner loop run. The inner stride is
        unrelated to the stride used for saving the trace.
    callback
        An arbitrary function run during the loop after updating the state. For
        the signature, see `Callback`. The callback is called under the jax jit,
        so the argument values are not available at the time the Python code is
        executed. Use the utilities in `jax.debug` to access the values at
        actual runtime. The callback may return new values for the MCMC state
        and the callback state.
    callback_state
        The initial custom state for the callback.
    burnin_extractor
    main_extractor
        Functions that extract the variables to be saved respectively only in
        the main trace and in both traces, given the MCMC state as argument.
        Must return a pytree, and must be vmappable.

    Returns
    -------
    bart : State
        The final MCMC state.
    burnin_trace : PyTree[Shaped[Array, 'n_burn *']]
        The trace of the burn-in phase. For the default layout, see `BurninTrace`.
    main_trace : PyTree[Shaped[Array, 'n_save *']]
        The trace of the main phase. For the default layout, see `MainTrace`.

    Notes
    -----
    The number of MCMC updates is ``n_burn + n_skip * n_save``. The traces do
    not include the initial state, and include the final state.
    """
    burnin_trace = _empty_trace(n_burn, bart, burnin_extractor)
    main_trace = _empty_trace(n_save, bart, main_extractor)

    # determine number of iterations for inner and outer loops
    n_iters = n_burn + n_skip * n_save
    if inner_loop_length is None:
        inner_loop_length = n_iters
    if inner_loop_length:
        n_outer = n_iters // inner_loop_length + bool(n_iters % inner_loop_length)
    else:
        n_outer = 1
        # setting to 0 would make for a clean noop, but it's useful to keep the
        # same code path for benchmarking and testing

    carry = _Carry(bart, jnp.int32(0), key, burnin_trace, main_trace, callback_state)
    for i_outer in range(n_outer):
        carry = _run_mcmc_inner_loop(
            carry,
            inner_loop_length,
            callback,
            burnin_extractor,
            main_extractor,
            n_burn,
            n_save,
            n_skip,
            i_outer,
            n_iters,
        )

    return carry.bart, carry.burnin_trace, carry.main_trace


@partial(jit, static_argnums=(0, 2))
def _empty_trace(
    length: int, bart: State, extractor: Callable[[State], PyTree]
) -> PyTree:
    return jax.vmap(extractor, in_axes=None, out_axes=0, axis_size=length)(bart)


@jit
def _compute_i_skip(
    i_total: Int32[Array, ''], n_burn: Int32[Array, ''], n_skip: Int32[Array, '']
) -> Int32[Array, '']:
    """Compute the `i_skip` argument passed to `callback`."""
    burnin = i_total < n_burn
    return jnp.where(
        burnin,
        i_total + 1,
        (i_total - n_burn + 1) % n_skip
        + jnp.where(i_total - n_burn + 1 < n_skip, n_burn, 0),
    )


@partial(jit_if_not_profiling, donate_argnums=(0,), static_argnums=(1, 2, 3, 4))
def _run_mcmc_inner_loop(
    carry: _Carry,
    inner_loop_length: int,
    callback: Callback | None,
    burnin_extractor: Callable[[State], PyTree],
    main_extractor: Callable[[State], PyTree],
    n_burn: Int32[Array, ''],
    n_save: Int32[Array, ''],
    n_skip: Int32[Array, ''],
    i_outer: Int32[Array, ''],
    n_iters: Int32[Array, ''],
) -> _Carry:
    def loop_impl(carry: _Carry) -> _Carry:
        """Loop body to run if i_total < n_iters."""
        # split random key
        keys = jaxext.split(carry.key, 3)
        key = keys.pop()

        # update state
        bart = mcmcstep.step(keys.pop(), carry.bart)

        # invoke callback
        callback_state = carry.callback_state
        if callback is not None:
            i_skip = _compute_i_skip(carry.i_total, n_burn, n_skip)
            rt = callback(
                key=keys.pop(),
                bart=bart,
                burnin=carry.i_total < n_burn,
                i_total=carry.i_total,
                i_skip=i_skip,
                callback_state=callback_state,
                n_burn=n_burn,
                n_save=n_save,
                n_skip=n_skip,
                i_outer=i_outer,
                inner_loop_length=inner_loop_length,
            )
            if rt is not None:
                bart, callback_state = rt

        # save to trace
        burnin_trace, main_trace = _save_state_to_trace(
            carry.burnin_trace,
            carry.main_trace,
            burnin_extractor,
            main_extractor,
            bart,
            carry.i_total,
            n_burn,
            n_skip,
        )

        return _Carry(
            bart=bart,
            i_total=carry.i_total + 1,
            key=key,
            burnin_trace=burnin_trace,
            main_trace=main_trace,
            callback_state=callback_state,
        )

    def loop_noop(carry: _Carry) -> _Carry:
        """Loop body to run if i_total >= n_iters; it does nothing."""
        return carry

    def loop(carry: _Carry, _) -> tuple[_Carry, None]:
        carry = cond_if_not_profiling(
            carry.i_total < n_iters, loop_impl, loop_noop, carry
        )
        return carry, None

    carry, _ = scan_if_not_profiling(loop, carry, None, inner_loop_length)
    return carry


@partial(jit, donate_argnums=(0, 1), static_argnums=(2, 3))
# this is jitted because under profiling _run_mcmc_inner_loop and the loop
# within it are not, so I need the donate_argnums feature of jit to avoid
# creating copies of the traces
def _save_state_to_trace(
    burnin_trace: PyTree,
    main_trace: PyTree,
    burnin_extractor: Callable[[State], PyTree],
    main_extractor: Callable[[State], PyTree],
    bart: State,
    i_total: Int32[Array, ''],
    n_burn: Int32[Array, ''],
    n_skip: Int32[Array, ''],
) -> tuple[PyTree, PyTree]:
    burnin_idx = i_total
    main_idx = (i_total - n_burn) // n_skip
    noop_idx = jnp.iinfo(jnp.int32).max
    noop_cond = i_total < n_burn
    main_idx = jnp.where(noop_cond, noop_idx, main_idx)

    burnin_trace = _pytree_at_set(burnin_trace, burnin_idx, burnin_extractor(bart))
    main_trace = _pytree_at_set(main_trace, main_idx, main_extractor(bart))

    return burnin_trace, main_trace


def _pytree_at_set(
    dest: PyTree[Array, ' T'], index: Int32[Array, ''], val: PyTree[Array]
) -> PyTree[Array, ' T']:
    """Map ``dest.at[index].set(val)`` over pytrees."""

    def at_set(dest, val):
        if dest.size:
            return dest.at[index, ...].set(val, mode='drop')
        else:
            # this handles the case where an array is empty because jax refuses
            # to index into an array of length 0, even if just in the abstract
            return dest

    return tree.map(at_set, dest, val)


def make_default_callback(
    *,
    dot_every: int | Integer[Array, ''] | None = 1,
    report_every: int | Integer[Array, ''] | None = 100,
    sparse_on_at: int | Integer[Array, ''] | None = None,
) -> dict[str, Any]:
    """
    Prepare a default callback for `run_mcmc`.

    The callback prints a dot on every iteration, and a longer
    report outer loop iteration, and can do variable selection.

    Parameters
    ----------
    dot_every
        A dot is printed every `dot_every` MCMC iterations, `None` to disable.
    report_every
        A one line report is printed every `report_every` MCMC iterations,
        `None` to disable.
    sparse_on_at
        If specified, variable selection is activated starting from this
        iteration. If `None`, variable selection is not used.

    Returns
    -------
    A dictionary with the arguments to pass to `run_mcmc` as keyword arguments to set up the callback.

    Examples
    --------
    >>> run_mcmc(..., **make_default_callback())
    """

    def asarray_or_none(val: None | Any) -> None | Array:
        return None if val is None else jnp.asarray(val)

    return dict(
        callback=_default_callback,
        callback_state=(
            PrintCallbackState(
                asarray_or_none(dot_every), asarray_or_none(report_every)
            ),
            SparseCallbackState(asarray_or_none(sparse_on_at)),
        ),
    )


def _default_callback(*, bart, callback_state, **kwargs):
    print_state, sparse_state = callback_state
    bart, _ = sparse_callback(callback_state=sparse_state, bart=bart, **kwargs)
    print_callback(callback_state=print_state, bart=bart, **kwargs)
    return bart, callback_state
    # here I assume that the callbacks don't update their states


class PrintCallbackState(Module):
    """State for `print_callback`.

    Parameters
    ----------
    dot_every
        A dot is printed every `dot_every` MCMC iterations, `None` to disable.
    report_every
        A one line report is printed every `report_every` MCMC iterations,
        `None` to disable.
    """

    dot_every: Int32[Array, ''] | None
    report_every: Int32[Array, ''] | None


def print_callback(
    *,
    bart: State,
    burnin: Bool[Array, ''],
    i_total: Int32[Array, ''],
    n_burn: Int32[Array, ''],
    n_save: Int32[Array, ''],
    n_skip: Int32[Array, ''],
    callback_state: PrintCallbackState,
    **_,
):
    """Print a dot and/or a report periodically during the MCMC."""
    if callback_state.dot_every is not None:
        dot_cond = (i_total + 1) % callback_state.dot_every == 0
        cond_if_not_profiling(
            dot_cond,
            lambda: callback_if_not_profiling(
                lambda: print('.', end='', flush=True),  # noqa: T201
                ordered=True,
            ),
            # logging can't do in-line printing so I'll stick to print
            lambda: None,
        )

    if callback_state.report_every is not None:

        def print_report():
            debug.callback(
                _print_report,
                burnin=burnin,
                i_total=i_total,
                n_iters=n_burn + n_save * n_skip,
                grow_prop_count=bart.forest.grow_prop_count,
                grow_acc_count=bart.forest.grow_acc_count,
                prune_acc_count=bart.forest.prune_acc_count,
                prop_total=len(bart.forest.leaf_tree),
                fill=grove.forest_fill(bart.forest.split_tree),
                ordered=True,
            )

        report_cond = (i_total + 1) % callback_state.report_every == 0

        # print a newline after dots
        if callback_state.dot_every is not None:
            cond_if_not_profiling(
                report_cond & dot_cond,
                lambda: callback_if_not_profiling(print, ordered=True),
                lambda: None,
            )

        cond_if_not_profiling(report_cond, print_report, lambda: None)


def _convert_jax_arrays_in_args(func: Callable) -> Callable:
    """Remove jax arrays from a function arguments.

    Converts all `jax.Array` instances in the arguments to either Python scalars
    or numpy arrays.
    """

    def convert_jax_arrays(pytree: PyTree) -> PyTree:
        def convert_jax_array(val: Any) -> Any:
            if not isinstance(val, jax.Array):
                return val
            elif val.shape:
                return numpy.array(val)
            else:
                return val.item()

        return tree.map(convert_jax_array, pytree)

    @wraps(func)
    def new_func(*args, **kw):
        args = convert_jax_arrays(args)
        kw = convert_jax_arrays(kw)
        return func(*args, **kw)

    return new_func


@_convert_jax_arrays_in_args
# convert all jax arrays in arguments because operations on them could lead to
# deadlock with the main thread
def _print_report(
    *,
    burnin: bool,
    i_total: int,
    n_iters: int,
    grow_prop_count: int,
    grow_acc_count: int,
    prune_acc_count: int,
    prop_total: int,
    fill: float,
):
    """Print the report for `print_callback`."""
    grow_prop = grow_prop_count / prop_total
    move_acc = (grow_acc_count + prune_acc_count) / prop_total

    suffix = ' (burnin)' if burnin else ''

    print(  # noqa: T201, see print_callback for why not logging
        f'Iteration {i_total + 1}/{n_iters}, '
        f'grow prob: {grow_prop:.0%}, '
        f'move acc: {move_acc:.0%}, '
        f'fill: {fill:.0%}{suffix}'
    )


class SparseCallbackState(Module):
    """State for `sparse_callback`.

    Parameters
    ----------
    sparse_on_at
        If specified, variable selection is activated starting from this
        iteration. If `None`, variable selection is not used.
    """

    sparse_on_at: Int32[Array, ''] | None


def sparse_callback(
    *,
    key: Key[Array, ''],
    bart: State,
    i_total: Int32[Array, ''],
    callback_state: SparseCallbackState,
    **_,
):
    """Perform variable selection, see `mcmcstep.step_sparse`."""
    if callback_state.sparse_on_at is not None:
        bart = cond_if_not_profiling(
            i_total < callback_state.sparse_on_at,
            lambda: bart,
            lambda: mcmcstep.step_sparse(key, bart),
        )
    return bart, callback_state


class Trace(grove.TreeHeaps, Protocol):
    """Protocol for a MCMC trace."""

    offset: Float32[Array, ' trace_length']


class TreesTrace(Module):
    """Implementation of `bartz.grove.TreeHeaps` for an MCMC trace."""

    leaf_tree: Float32[Array, 'trace_length num_trees 2**d']
    var_tree: UInt[Array, 'trace_length num_trees 2**(d-1)']
    split_tree: UInt[Array, 'trace_length num_trees 2**(d-1)']

    @classmethod
    def from_dataclass(cls, obj: grove.TreeHeaps):
        """Create a `TreesTrace` from any `bartz.grove.TreeHeaps`."""
        return cls(**{f.name: getattr(obj, f.name) for f in fields(cls)})


@jax.jit
def evaluate_trace(
    trace: Trace, X: UInt[Array, 'p n']
) -> Float32[Array, 'trace_length n']:
    """
    Compute predictions for all iterations of the BART MCMC.

    Parameters
    ----------
    trace
        A trace of the BART MCMC, as returned by `run_mcmc`.
    X
        The predictors matrix, with `p` predictors and `n` observations.

    Returns
    -------
    The predictions for each iteration of the MCMC.
    """
    evaluate_trees = partial(grove.evaluate_forest, sum_trees=False)
    evaluate_trees = jaxext.autobatch(evaluate_trees, 2**29, (None, 0))
    trees = TreesTrace.from_dataclass(trace)

    def loop(_, item):
        offset, trees = item
        values = evaluate_trees(X, trees)
        return None, offset[..., None] + jnp.sum(values, axis=0, dtype=jnp.float32)

    _, y = lax.scan(loop, None, (trace.offset, trees))
    return y


@partial(jax.jit, static_argnums=(0,))
def compute_varcount(
    p: int, trace: grove.TreeHeaps
) -> Int32[Array, 'trace_length {p}']:
    """
    Count how many times each predictor is used in each MCMC state.

    Parameters
    ----------
    p
        The number of predictors.
    trace
        A trace of the BART MCMC, as returned by `run_mcmc`.

    Returns
    -------
    Histogram of predictor usage in each MCMC state.
    """
    vmapped_var_histogram = jax.vmap(grove.var_histogram, in_axes=(None, 0, 0))
    return vmapped_var_histogram(p, trace.var_tree, trace.split_tree)
