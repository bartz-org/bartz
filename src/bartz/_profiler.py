# bartz/src/bartz/_profiler.py
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

"""Module with utilities related to profiling bartz."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar

from jax import block_until_ready, debug, jit
from jax.lax import cond, scan
from jax.profiler import TraceAnnotation
from jaxtyping import Array, Bool

from bartz.mcmcstep._state import vmap_chains

PROFILE_MODE: bool = False

T = TypeVar('T')
Carry = TypeVar('Carry')


def get_profile_mode() -> bool:
    """Return the current profile mode status.

    Returns
    -------
    True if profile mode is enabled, False otherwise.
    """
    return PROFILE_MODE


def set_profile_mode(value: bool, /) -> None:
    """Set the profile mode status.

    Parameters
    ----------
    value
        If True, enable profile mode. If False, disable it.
    """
    global PROFILE_MODE  # noqa: PLW0603
    PROFILE_MODE = value


@contextmanager
def profile_mode(value: bool, /) -> Iterator[None]:
    """Context manager to temporarily set profile mode.

    Parameters
    ----------
    value
        Profile mode value to set within the context.

    Examples
    --------
    >>> with profile_mode(True):
    ...     # Code runs with profile mode enabled
    ...     pass

    Notes
    -----
    In profiling mode, the MCMC loop is not compiled into a single function, but
    instead compiled in smaller pieces that are instrumented to show up in the
    jax tracer and Python profiling statistics. Search for function names
    starting with 'jab' (see `jit_and_block_if_profiling`).

    Jax tracing is not enabled by this context manager and if used must be
    handled separately by the user; this context manager only makes sure that
    the execution flow will be more interpretable in the traces if the tracer is
    used.
    """
    old_value = get_profile_mode()
    set_profile_mode(value)
    try:
        yield
    finally:
        set_profile_mode(old_value)


def jit_and_block_if_profiling(
    func: Callable[..., T], block_before: bool = False, **kwargs
) -> Callable[..., T]:
    """Apply JIT compilation and block if profiling is enabled.

    When profile mode is off, the function runs without JIT. When profile mode
    is on, the function is JIT compiled and blocks outputs to ensure proper
    timing.

    Parameters
    ----------
    func
        Function to wrap.
    block_before
        If True block inputs before passing them to the JIT-compiled function.
        This ensures that any pending computations are completed before entering
        the JIT-compiled function. This phase is not included in the trace
        event.
    **kwargs
        Additional arguments to pass to `jax.jit`.

    Returns
    -------
    Wrapped function.

    Notes
    -----
    Under profiling mode, the function invocation is handled such that a custom
    jax trace event with name `jab[<func_name>]` is created. The statistics on
    the actual Python function will be off, while the function
    `jab_inner_wrapper` represents the actual execution time.
    """
    jitted_func = jit(func, **kwargs)

    event_name = f'jab[{func.__name__}]'

    # this wrapper is meant to measure the time spent executing the function
    def jab_inner_wrapper(*args, **kwargs) -> T:
        with TraceAnnotation(event_name):
            result = jitted_func(*args, **kwargs)
            return block_until_ready(result)

    @wraps(func)
    def jab_outer_wrapper(*args: Any, **kwargs: Any) -> T:
        if get_profile_mode():
            if block_before:
                args, kwargs = block_until_ready((args, kwargs))
            return jab_inner_wrapper(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return jab_outer_wrapper


def jit_if_profiling(func: Callable[..., T], *args, **kwargs) -> Callable[..., T]:
    """Apply JIT compilation only when profiling.

    Parameters
    ----------
    func
        Function to wrap.
    *args
    **kwargs
        Additional arguments to pass to `jax.jit`.

    Returns
    -------
    Wrapped function.
    """
    jitted_func = jit(func, *args, **kwargs)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        if get_profile_mode():
            return jitted_func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


def jit_if_not_profiling(func: Callable[..., T], *args, **kwargs) -> Callable[..., T]:
    """Apply JIT compilation only when not profiling.

    When profile mode is off, the function is JIT compiled. When profile mode is
    on, the function runs as-is.

    Parameters
    ----------
    func
        Function to wrap.
    *args
    **kwargs
        Additional arguments to pass to `jax.jit`.

    Returns
    -------
    Wrapped function.
    """
    jitted_func = jit(func, *args, **kwargs)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        if get_profile_mode():
            return func(*args, **kwargs)
        else:
            return jitted_func(*args, **kwargs)

    wrapper._fun = func  # used by run_mcmc  # noqa: SLF001

    return wrapper


def scan_if_not_profiling(
    f: Callable[[Carry, None], tuple[Carry, None]],
    init: Carry,
    xs: None,
    length: int,
    /,
) -> tuple[Carry, None]:
    """Restricted replacement for `jax.lax.scan` that uses a Python loop when profiling.

    Parameters
    ----------
    f
        Scan body function with signature (carry, None) -> (carry, None).
    init
        Initial carry value.
    xs
        Input values to scan over (not supported).
    length
        Integer specifying the number of loop iterations.

    Returns
    -------
    Tuple of (final_carry, None) (stacked outputs not supported).
    """
    assert xs is None
    if get_profile_mode():
        carry = init
        for _i in range(length):
            carry, _ = f(carry, None)
        return carry, None

    else:
        return scan(f, init, None, length)


def cond_if_not_profiling(
    pred: bool | Bool[Array, ''],
    true_fun: Callable[..., T],
    false_fun: Callable[..., T],
    /,
    *operands,
) -> T:
    """Restricted replacement for `jax.lax.cond` that uses a Python if when profiling.

    Parameters
    ----------
    pred
        Boolean predicate to choose which function to execute.
    true_fun
        Function to execute if `pred` is True.
    false_fun
        Function to execute if `pred` is False.
    *operands
        Arguments passed to `true_fun` and `false_fun`.

    Returns
    -------
    Result of either `true_fun()` or `false_fun()`.
    """
    if get_profile_mode():
        if pred:
            return true_fun(*operands)
        else:
            return false_fun(*operands)
    else:
        return cond(pred, true_fun, false_fun, *operands)


def callback_if_not_profiling(
    callback: Callable[..., None], *args: Any, ordered: bool = False, **kwargs: Any
) -> None:
    """Restricted replacement for `jax.debug.callback` that calls the callback directly in profiling mode."""
    if get_profile_mode():
        callback(*args, **kwargs)
    else:
        debug.callback(callback, *args, ordered=ordered, **kwargs)


def vmap_chains_if_profiling(fun: Callable[..., T], **kwargs) -> Callable[..., T]:
    """Apply `vmap_chains` only when profile mode is enabled."""
    new_fun = vmap_chains(fun, **kwargs)

    @wraps(fun)
    def wrapper(*args, **kwargs):
        if get_profile_mode():
            return new_fun(*args, **kwargs)
        else:
            return fun(*args, **kwargs)

    return wrapper


def vmap_chains_if_not_profiling(fun: Callable[..., T], **kwargs) -> Callable[..., T]:
    """Apply `vmap_chains` only when profile mode is disabled."""
    new_fun = vmap_chains(fun, **kwargs)

    @wraps(fun)
    def wrapper(*args, **kwargs):
        if get_profile_mode():
            return fun(*args, **kwargs)
        else:
            return new_fun(*args, **kwargs)

    return wrapper
