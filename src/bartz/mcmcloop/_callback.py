# bartz/src/bartz/mcmcloop/_callback.py
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

"""Default progress-printing callback for `run_mcmc`."""

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import numpy
from equinox import Module
from jax import debug, lax, tree
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Int32, Integer, PyTree

from bartz.grove import forest_fill
from bartz.mcmcloop._loop import _replicate
from bartz.mcmcstep import State
from bartz.mcmcstep._state import chain_to_axis, chain_vmap_axes, chainful_axis


def make_print_callback(
    state: State,
    *,
    dot_every: int | Integer[Array, ''] | None = 1,
    report_every: int | Integer[Array, ''] | None = 100,
) -> dict[str, Any]:
    """
    Prepare a progress-printing callback for `run_mcmc`.

    The callback prints a dot on every iteration, and a longer report
    periodically.

    Parameters
    ----------
    state
        The bart state to use the callback with, used to determine device
        sharding.
    dot_every
        A dot is printed every `dot_every` MCMC iterations, `None` to disable.
    report_every
        A one line report is printed every `report_every` MCMC iterations,
        `None` to disable.

    Returns
    -------
    A dictionary with the arguments to pass to `run_mcmc` as keyword arguments to set up the callback.

    Examples
    --------
    >>> run_mcmc(key, state, ..., **make_print_callback(state, ...))
    """

    def as_replicated_array_or_none(val: ArrayLike | None) -> None | Array:
        return None if val is None else _replicate(jnp.asarray(val), state.config.mesh)

    return dict(
        callback=print_callback,
        callback_state=PrintCallbackState(
            as_replicated_array_or_none(dot_every),
            as_replicated_array_or_none(report_every),
        ),
    )


class PrintCallbackState(Module):
    """State for `print_callback`."""

    dot_every: Int32[Array, ''] | None
    """A dot is printed every `dot_every` MCMC iterations, `None` to disable."""

    report_every: Int32[Array, ''] | None
    """A one line report is printed every `report_every` MCMC iterations,
    `None` to disable."""


def print_callback(
    *,
    bart: State,
    burnin: Bool[Array, ''],
    i_total: Int32[Array, ''],
    n_burn: Int32[Array, ''],
    n_save: Int32[Array, ''],
    n_skip: Int32[Array, ''],
    callback_state: PrintCallbackState,
    **_: Any,
) -> None:
    """Print a dot and/or a report periodically during the MCMC."""
    report_every = callback_state.report_every
    dot_every = callback_state.dot_every
    it = i_total + 1

    def get_cond(every: Int32[Array, ''] | None) -> bool | Bool[Array, '']:
        return False if every is None else it % every == 0

    report_cond = get_cond(report_every)
    dot_cond = get_cond(dot_every)

    def line_report_branch() -> None:
        if report_every is None:
            return
        if dot_every is None:
            print_newline = False
        else:
            print_newline = it % report_every > it % dot_every
        chain_axis = chain_vmap_axes(bart.forest).split_tree
        num_trees_axis = chainful_axis(0, chain_axis)  # (num_trees, hts)
        split_tree = chain_to_axis(bart.forest.split_tree, chain_axis)
        debug.callback(
            _print_report,
            print_dot=dot_cond,
            print_newline=print_newline,
            burnin=burnin,
            it=it,
            n_iters=n_burn + n_save * n_skip,
            num_chains=bart.num_chains(),
            grow_prop_count=bart.forest.grow_prop_count.mean(),
            grow_acc_count=bart.forest.grow_acc_count.mean(),
            prune_acc_count=bart.forest.prune_acc_count.mean(),
            prop_total=bart.forest.split_tree.shape[num_trees_axis],
            fill=forest_fill(split_tree),
        )

    def just_dot_branch() -> None:
        if dot_every is None:
            return
        # terminate the dot line on the final iteration so subsequent output
        # doesn't continue on the same line as the dots
        last_iter = it == n_burn + n_save * n_skip
        lax.cond(
            last_iter,
            lambda: debug.callback(lambda: print('.', flush=True)),  # noqa: T201
            lambda: debug.callback(lambda: print('.', end='', flush=True)),  # noqa: T201
        )
        # logging can't do in-line printing so we use print

    lax.cond(
        report_cond,
        line_report_branch,
        lambda: lax.cond(dot_cond, just_dot_branch, lambda: None),
    )


T = TypeVar('T')


def _convert_jax_arrays_in_args(func: Callable[..., T]) -> Callable[..., T]:
    """Remove jax arrays from a function arguments.

    Converts all `jax.Array` instances in the arguments to either Python scalars
    or numpy arrays.
    """

    def convert_jax_arrays(pytree: PyTree) -> PyTree:
        def convert_jax_array(val: object) -> object:
            if not isinstance(val, Array):
                return val
            elif val.shape:
                return numpy.array(val)
            else:
                return val.item()

        return tree.map(convert_jax_array, pytree)

    @wraps(func)
    def new_func(*args: Any, **kw: Any) -> T:
        args = convert_jax_arrays(args)
        kw = convert_jax_arrays(kw)
        return func(*args, **kw)

    return new_func


@_convert_jax_arrays_in_args
# convert all jax arrays in arguments because operations on them could lead to
# deadlock with the main thread
def _print_report(
    *,
    print_dot: bool,
    print_newline: bool,
    burnin: bool,
    it: int,
    n_iters: int,
    num_chains: int | None,
    grow_prop_count: float,
    grow_acc_count: float,
    prune_acc_count: float,
    prop_total: int,
    fill: float,
) -> None:
    """Print the report for `print_callback`."""
    # compute fractions
    grow_prop = grow_prop_count / prop_total
    move_acc = (grow_acc_count + prune_acc_count) / prop_total

    # determine prefix
    if print_dot:
        prefix = '.\n'
    elif print_newline:
        prefix = '\n'
    else:
        prefix = ''

    # determine suffix in parentheses
    msgs = []
    if num_chains is not None:
        msgs.append(f'avg. {num_chains} chains')
    if burnin:
        msgs.append('burnin')
    suffix = f' ({", ".join(msgs)})' if msgs else ''

    print(  # noqa: T201, see print_callback for why not logging
        f'{prefix}Iteration {it}/{n_iters}, '
        f'grow prob: {grow_prop:.0%}, '
        f'move acc: {move_acc:.0%}, '
        f'fill: {fill:.0%}{suffix}'
    )
