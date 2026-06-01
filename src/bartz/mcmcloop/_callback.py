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

"""Progress-reporting callbacks for `run_mcmc`."""

import itertools
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import wraps
from typing import Any, TypeVar

import numpy
from equinox import Module
from jax import debug, lax, tree
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float32, Int32, Integer, PyTree
from tqdm.auto import tqdm

from bartz.grove import forest_mean_leaves
from bartz.mcmcloop._loop import _replicate
from bartz.mcmcstep import State
from bartz.mcmcstep._axes import chain_to_axis, chain_vmap_axes, chainful_axis


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
        debug.callback(
            _print_report,
            print_dot=dot_cond,
            print_newline=print_newline,
            burnin=burnin,
            it=it,
            n_iters=n_burn + n_save * n_skip,
            **_forest_stats(bart),
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


def make_tqdm_callback(
    state: State,
    *,
    update_every: int = 1,
    report_every: int | None = 100,
    **tqdm_kwargs: Any,
) -> dict[str, Any]:
    """
    Prepare a `tqdm` progress-bar callback for `run_mcmc`.

    The callback shows a progress bar that advances with the MCMC iterations,
    optionally annotated with the proposal acceptance statistics.

    Parameters
    ----------
    state
        The bart state to use the callback with, used to determine device
        sharding.
    update_every
        The bar position is refreshed every `update_every` MCMC iterations
        (`tqdm` further throttles the actual redraw rate on its own).
    report_every
        The acceptance statistics shown next to the bar are refreshed every
        `report_every` MCMC iterations, `None` to omit them.
    **tqdm_kwargs
        Additional keyword arguments forwarded to the `tqdm.tqdm` constructor,
        e.g., ``desc``, ``file``, or ``disable``.

    Returns
    -------
    A dictionary with the arguments to pass to `run_mcmc` as keyword arguments to set up the callback.

    Examples
    --------
    >>> run_mcmc(key, state, ..., **make_tqdm_callback(state, ...))

    Notes
    -----
    Works with chains sharded across multiple devices. If the run is interrupted
    (e.g. with ^C), the bar is left as-is; the next `make_tqdm_callback` call
    closes it, so a subsequent run starts from a clean line.
    """
    _close_stale_bars()  # clean up after any previous run that was interrupted
    bar_id = next(_tqdm_bar_counter)
    _tqdm_registry[bar_id] = _TqdmEntry(tqdm_kwargs)

    def as_replicated_array(val: ArrayLike) -> Array:
        return _replicate(jnp.asarray(val), state.config.mesh)

    return dict(
        callback=tqdm_callback,
        callback_state=TqdmCallbackState(
            bar_id=as_replicated_array(jnp.int32(bar_id)),
            update_every=as_replicated_array(jnp.int32(update_every)),
            report_every=None
            if report_every is None
            else as_replicated_array(jnp.int32(report_every)),
        ),
    )


class TqdmCallbackState(Module):
    """State for `tqdm_callback`."""

    bar_id: Int32[Array, '']
    """Handle identifying the bar in the module-level `tqdm` bar registry."""

    update_every: Int32[Array, '']
    """The bar position is refreshed every `update_every` MCMC iterations."""

    report_every: Int32[Array, ''] | None
    """The acceptance statistics are refreshed every `report_every` MCMC
    iterations, `None` to omit them."""


def tqdm_callback(
    *,
    bart: State,
    i_total: Int32[Array, ''],
    n_burn: Int32[Array, ''],
    n_save: Int32[Array, ''],
    n_skip: Int32[Array, ''],
    callback_state: TqdmCallbackState,
    **_: Any,
) -> None:
    """Advance a `tqdm` progress bar during the MCMC."""
    it = i_total + 1
    n_iters = n_burn + n_save * n_skip
    bar_id = callback_state.bar_id
    last = it == n_iters

    # The callbacks are unordered: `ordered=True` is unsupported with more than
    # one device, and we need this to work with chains sharded across devices.
    # `_tqdm_advance` is therefore robust to out-of-order invocations.

    # refresh the statistics first so they tend to be visible by the time the
    # bar is advanced
    report_every = callback_state.report_every
    if report_every is not None:

        def report_branch() -> None:
            debug.callback(_tqdm_report, bar_id, n_iters, **_forest_stats(bart))

        lax.cond((it % report_every == 0) | last, report_branch, lambda: None)

    lax.cond(
        (it % callback_state.update_every == 0) | last,
        lambda: debug.callback(_tqdm_advance, bar_id, it, n_iters),
        lambda: None,
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


def _forest_stats(bart: State) -> dict[str, Float32[Array, ''] | int | None]:
    """Cross-chain proposal/acceptance/leaves statistics shown during the MCMC."""
    chain_axis = chain_vmap_axes(bart.forest).split_tree
    num_trees_axis = chainful_axis(0, chain_axis)  # (num_trees, hts)
    split_tree = chain_to_axis(bart.forest.split_tree, chain_axis)
    prop_total = bart.forest.split_tree.shape[num_trees_axis]
    return dict(
        num_chains=bart.num_chains(),
        grow_prop=bart.forest.grow_prop_count.mean() / prop_total,
        move_acc=(
            bart.forest.grow_acc_count.mean() + bart.forest.prune_acc_count.mean()
        )
        / prop_total,
        mean_leaves=forest_mean_leaves(split_tree),
        max_leaves=split_tree.shape[-1],
    )


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
    grow_prop: float,
    move_acc: float,
    mean_leaves: float,
    max_leaves: int,
) -> None:
    """Print the report for `print_callback`."""
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
        f'leaves: {mean_leaves:.1f}/{max_leaves}{suffix}'
    )


@dataclass(frozen=True)
class _TqdmEntry:
    """An entry in the `tqdm` bar registry."""

    kwargs: dict[str, Any]
    """Keyword arguments to construct the bar with, from `make_tqdm_callback`."""

    bar: tqdm | None = None
    """The bar, created lazily on the first callback invocation, `None` until then."""


# tqdm carries Python state that cannot live in a jax pytree, so the bars are
# kept here and referenced from the jax loop through the integer handle stored
# in `TqdmCallbackState.bar_id` (a traceable scalar, so the loop pytree stays
# stable across runs and is not recompiled).
_tqdm_registry: dict[int, _TqdmEntry] = {}
_tqdm_bar_counter = itertools.count()

# tqdm's default layout, but without the ': ' that `format_meter` forces after a
# non-empty description; the label is set as a `{desc}` ending in a space instead
_TQDM_BAR_FORMAT = (
    '{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
    '[{elapsed}<{remaining}, {rate_fmt}{postfix}]'
)


def _close_stale_bars() -> None:
    """Close and drop any bars left over from a previous (e.g. interrupted) run."""
    for entry in _tqdm_registry.values():
        if entry.bar is not None:
            entry.bar.close()
    _tqdm_registry.clear()


def _get_or_create_bar(bar_id: int, n_iters: int) -> tqdm | None:
    """Return the bar for `bar_id`, creating it on first use, `None` if finished."""
    entry = _tqdm_registry.get(bar_id)
    if entry is None:
        # the bar was already closed (the loop finished, possibly out of order)
        return None
    if entry.bar is None:
        bar = tqdm(**{'total': n_iters, 'bar_format': _TQDM_BAR_FORMAT, **entry.kwargs})
        _tqdm_registry[bar_id] = replace(entry, bar=bar)
        return bar
    return entry.bar


@_convert_jax_arrays_in_args
# convert all jax arrays in arguments, see _print_report for why
def _tqdm_advance(bar_id: int, it: int, n_iters: int) -> None:
    """Advance the bar towards absolute position `it`, closing it at the end."""
    bar = _get_or_create_bar(bar_id, n_iters)
    if bar is None:
        return
    bar.update(max(0, it - bar.n))  # forward-only: callbacks may arrive out of order
    if it >= n_iters:
        bar.close()
        del _tqdm_registry[bar_id]


@_convert_jax_arrays_in_args
# convert all jax arrays in arguments, see _print_report for why
def _tqdm_report(
    bar_id: int,
    n_iters: int,
    *,
    num_chains: int | None,
    move_acc: float,
    mean_leaves: float,
    max_leaves: int,
    **_: Any,
) -> None:
    """Set the bar description and acceptance-statistics postfix."""
    bar = _get_or_create_bar(bar_id, n_iters)
    if bar is None:
        return
    # set_description_str (not set_description) to avoid tqdm's ': ' suffix; the
    # trailing space separates the label from the bar
    bar.set_description_str('train ', refresh=False)
    # keep this terse so the bar stays narrow, e.g. '4ch acc 25% leaves 3.4/32'
    msgs = []
    if num_chains is not None:
        msgs.append(f'{num_chains}ch')
    msgs.append(f'acc {move_acc:.0%}')
    msgs.append(f'leaves {mean_leaves:.1f}/{max_leaves}')
    bar.set_postfix_str(' '.join(msgs))
