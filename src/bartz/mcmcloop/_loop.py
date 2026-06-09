# bartz/src/bartz/mcmcloop/_loop.py
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

"""Implement `run_mcmc`, the MCMC loop driver."""

from collections.abc import Callable
from functools import partial, update_wrapper
from typing import (
    Any,
    Generic,
    NamedTuple,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from equinox import Module
from jax import (
    NamedSharding,
    device_put,
    eval_shape,
    lax,
    named_call,
    random,
    tree,
    vmap,
)
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Bool, Int32, Key, PyTree, Shaped

from bartz._jaxext import jit, jit_active, split
from bartz.mcmcloop._trace import BurninTrace, MainTrace
from bartz.mcmcstep import State, step
from bartz.mcmcstep._axes import trace_sample_axes
from bartz.mcmcstep._lazy import add_dummy_axis

# WORKAROUND(python<3.12): use `type CallbackState = PyTree[Any, 'T']`
CallbackState: TypeAlias = PyTree[Any, 'T']


class RunMCMCResult(NamedTuple):
    """Return value of `run_mcmc`."""

    final_state: State
    """The final MCMC state."""

    burnin_trace: BurninTrace
    """The trace of the burn-in phase."""

    main_trace: MainTrace
    """The trace of the main phase."""


@runtime_checkable
class Callback(Protocol):
    """Callback type for `run_mcmc`."""

    def __call__(
        self,
        *,
        key: Key[Array, ''],
        state: State,
        burnin: Bool[Array, ''],
        i_total: Int32[Array, ''],
        callback_state: CallbackState,
        n_burn: Int32[Array, ''],
        n_save: Int32[Array, ''],
        n_skip: Int32[Array, ''],
        i_outer: Int32[Array, ''],
        inner_loop_length: Int32[Array, ''],
    ) -> tuple[State, CallbackState] | None:
        """Do an arbitrary action after an iteration of the MCMC.

        Parameters
        ----------
        key
            A key for random number generation.
        state
            The MCMC state just after updating it.
        burnin
            Whether the last iteration was in the burn-in phase.
        i_total
            The index of the last MCMC iteration (0-based).
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
        state : State
            A possibly modified MCMC state. To avoid modifying the state,
            return the `state` argument passed to the callback as-is.
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

    state: State
    i_total: Int32[Array, '']
    key: Key[Array, '']
    burnin_trace: BurninTrace
    main_trace: MainTrace
    callback_state: CallbackState


def run_mcmc(
    key: Key[Array, ''],
    state: State,
    n_save: int,
    *,
    n_burn: int = 0,
    n_skip: int = 1,
    inner_loop_length: int | None = None,
    callback: Callback | None = None,
    callback_state: CallbackState = None,
) -> RunMCMCResult:
    """
    Run the MCMC for the BART posterior.

    Parameters
    ----------
    key
        A key for random number generation.
    state
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

    Returns
    -------
    A namedtuple with the final state, the burn-in trace, and the main trace.

    Raises
    ------
    RuntimeError
        If `run_mcmc` detects it's being invoked in a `jax.jit`-wrapped context and
        with settings that would create unrolled loops in the trace.

    Notes
    -----
    The number of MCMC updates is ``n_burn + n_skip * n_save``. The traces do
    not include the initial state, and include the final state.

    Resuming is exact: passing the returned `~RunMCMCResult.final_state` and the same `key` to
    a new call continues the run as if it had not stopped, so splitting a run
    into several consecutive calls gives the same result as a single call.
    """
    # copy the key so buffer donation does not invalidate the caller's copy
    key = jnp.copy(key)

    # create empty traces
    burnin_trace = _empty_trace(n_burn, state, BurninTrace)
    main_trace = _empty_trace(n_save, state, MainTrace)

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

    # error if under jit and there are unrolled loops
    if jit_active() and n_outer > 1:
        msg = (
            '`run_mcmc` was called within a jit-compiled function and '
            'there are more than 1 outer loops, '
            'please either do not jit or set `inner_loop_length=None`'
        )
        raise RuntimeError(msg)

    replicate = partial(_replicate, mesh=state.config.mesh)
    carry = _Carry(
        state,
        replicate(jnp.int32(0)),
        replicate(key),
        burnin_trace,
        main_trace,
        callback_state,
    )
    _inner_loop_counter.reset_call_counter()
    for i_outer in range(n_outer):
        carry = _run_mcmc_inner_loop(
            carry, inner_loop_length, callback, n_burn, n_save, n_skip, i_outer, n_iters
        )

    return RunMCMCResult(carry.state, carry.burnin_trace, carry.main_trace)


def _replicate(
    x: Shaped[Array, '*shape'], mesh: Mesh | None
) -> Shaped[Array, '*shape']:
    if mesh is None:
        return x
    else:
        return device_put(x, NamedSharding(mesh, PartitionSpec()))


TraceT = TypeVar('TraceT', bound=BurninTrace)


@jit(static_argnums=(0, 2))
def _empty_trace(length: int, state: State, trace_cls: type[TraceT]) -> TraceT:
    example_output = eval_shape(trace_cls.from_state, state)
    out_axes = trace_sample_axes(add_dummy_axis(example_output))

    return vmap(
        trace_cls.from_state, in_axes=None, out_axes=out_axes, axis_size=length
    )(state)


T = TypeVar('T')


class _CallCounter(Generic[T]):
    """Wrap a callable to check it's not called more than once."""

    def __init__(self, func: Callable[..., T]) -> None:
        self.func = func
        self.n_calls = 0
        update_wrapper(self, func)

    def reset_call_counter(self) -> None:
        """Reset the call counter."""
        self.n_calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        if self.n_calls:
            msg = (
                'The inner loop of `run_mcmc` was traced more than once, '
                'which indicates a double compilation of the MCMC code. This '
                'probably depends on the input state having different type from the '
                'output state. Check the input is in a format that is the '
                'same jax would output, e.g., all arrays and scalars are jax '
                'arrays, with the right shardings.'
            )
            raise RuntimeError(msg)
        self.n_calls += 1
        return self.func(*args, **kwargs)


def _run_mcmc_inner_loop_impl(
    carry: _Carry,
    inner_loop_length: Int32[Array, ''],
    callback: Callback | None,
    n_burn: Int32[Array, ''],
    n_save: Int32[Array, ''],
    n_skip: Int32[Array, ''],
    i_outer: Int32[Array, ''],
    n_iters: Int32[Array, ''],
) -> _Carry:
    # determine number of iterations for this loop batch
    i_upper = jnp.minimum(carry.i_total + inner_loop_length, n_iters)

    def cond(carry: _Carry) -> Bool[Array, '']:
        """Whether to continue the MCMC loop."""
        return carry.i_total < i_upper

    def body(carry: _Carry) -> _Carry:
        """Update the MCMC state."""
        iter_key = random.fold_in(carry.key, carry.state.config.steps_done)
        keys = split(iter_key, 2)

        # update state
        state = step(keys.pop(), carry.state)

        # invoke callback
        callback_state = carry.callback_state
        if callback is not None:
            rt = callback(
                key=keys.pop(),
                state=state,
                burnin=carry.i_total < n_burn,
                i_total=carry.i_total,
                callback_state=callback_state,
                n_burn=n_burn,
                n_save=n_save,
                n_skip=n_skip,
                i_outer=i_outer,
                inner_loop_length=inner_loop_length,
            )
            if rt is not None:
                state, callback_state = rt

        # save to trace
        burnin_trace, main_trace = _save_state_to_trace(
            carry.burnin_trace, carry.main_trace, state, carry.i_total, n_burn, n_skip
        )

        return _Carry(
            state=state,
            i_total=carry.i_total + 1,
            key=carry.key,
            burnin_trace=burnin_trace,
            main_trace=main_trace,
            callback_state=callback_state,
        )

    return lax.while_loop(cond, body, carry)


# Wrap the inner loop in an explicit `_CallCounter`, kept in `_inner_loop_counter`
# so `run_mcmc` can reset it directly instead of reaching into jit internals,
# then jit the wrapped callable.
_inner_loop_counter: _CallCounter[_Carry] = _CallCounter(_run_mcmc_inner_loop_impl)
_run_mcmc_inner_loop = jit(donate_argnums=(0,), static_argnums=(2,))(
    _inner_loop_counter
)


@named_call
def _save_state_to_trace(
    burnin_trace: BurninTrace,
    main_trace: MainTrace,
    state: State,
    i_total: Int32[Array, ''],
    n_burn: Int32[Array, ''],
    n_skip: Int32[Array, ''],
) -> tuple[BurninTrace, MainTrace]:
    # trace index where to save during burnin; out-of-bounds => noop after
    # burnin
    burnin_idx = i_total

    # trace index where to save during main phase; force it out-of-bounds
    # during burnin
    main_idx = (i_total - n_burn) // n_skip
    noop_idx = jnp.iinfo(jnp.int32).max
    noop_cond = i_total < n_burn
    main_idx = jnp.where(noop_cond, noop_idx, main_idx)

    # prepare array index
    burnin_trace = _set(burnin_trace, burnin_idx, BurninTrace.from_state(state))
    main_trace = _set(main_trace, main_idx, MainTrace.from_state(state))

    return burnin_trace, main_trace


def _set(
    trace: PyTree[Array, ' T'], index: Int32[Array, ''], val: PyTree[Array, ' T']
) -> PyTree[Array, ' T']:
    """Do ``trace[index] = val`` but fancier."""
    # WORKAROUND(jax<0.7.1): once we bump jax to v0.7.1 we can use mutable
    # arrays to save the trace instead of this functional update.
    sample_axes = trace_sample_axes(trace)

    # `trace` is `(*chains, samples, *shape)` and `val` the same without the
    # `samples` axis. The optional `chains` axis cannot share an annotation with
    # the variadic `*shape` (two variadics are ambiguous), and a union of the
    # with/without-`chains` layouts is rank-ambiguous under the runtime checker,
    # so the trace/val shapes are kept independent; their relationship is
    # enforced dynamically. The return has the trace shape.
    def at_set(
        trace: Shaped[Array, '*chains_samples_core'],
        val: Shaped[Array, '*chains_core'],
        sample_axis: int | None,
    ) -> Shaped[Array, '*chains_samples_core']:
        if sample_axis is None or trace.size == 0:
            # `sample_axis is None`: fields without a `samples` marker have
            # no per-iteration slot to update.
            # `trace.size == 0`: jax refuses to index into an axis of length
            # 0, even in the abstract.
            return trace

        ndindex = (slice(None),) * sample_axis + (index, ...)
        return trace.at[ndindex].set(val, mode='drop')

    return tree.map(at_set, trace, val, sample_axes)
