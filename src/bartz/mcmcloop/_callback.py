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
from functools import partial, wraps
from typing import Any, TypeVar

import numpy
from equinox import Module, field
from jax import debug, eval_shape, lax, tree
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike, Bool, Float32, Int32, Integer, PyTree
from tqdm.auto import tqdm

from bartz.grove import forest_mean_leaves
from bartz.mcmcloop._loop import _replicate
from bartz.mcmcstep import State
from bartz.mcmcstep._axes import chain_to_axis, chain_vmap_axes, chainful_axis


class StatsReport(Module):
    """Forest diagnostics produced by `StatsAccumulator.report` for one report."""

    grow_prop: Float32[Array, '']
    """Fraction of trees proposed for a grow move."""

    move_acc: Float32[Array, '']
    """Fraction of trees on which a grow or prune move was accepted."""

    mean_leaves: Float32[Array, '']
    """Mean number of leaves per tree."""

    peff: Float32[Array, ''] | None
    """Effective number of predictors, or `None` when variable selection is off."""

    n_samples: Int32[Array, ''] | None
    """Number of iterations averaged over, or `None` when not averaging."""

    num_chains: int | None = field(static=True)
    """Number of chains averaged over, or `None` when single-chain."""

    max_leaves: int = field(static=True)
    """Maximum possible number of leaves per tree."""

    p: int | None = field(static=True)
    """Number of predictors, or `None` when variable selection is off."""


class StatsAccumulator(Module):
    """Running average of the forest diagnostics shown during the MCMC.

    When enabled, it sums the per-iteration statistics so a report shows their
    average over the iterations since the previous report. When disabled it
    carries no running state and a report shows the latest iteration only.
    """

    sums: dict[str, Float32[Array, '']] | None
    """Running sums of the averaged statistics, or `None` when disabled."""

    count: Int32[Array, '']
    """Number of iterations accumulated since the last reset."""

    @classmethod
    def initial(cls, state: State, *, enabled: bool) -> 'StatsAccumulator':
        """Create a zeroed accumulator, inert unless `enabled`."""
        if enabled:
            # only the structure is needed, so avoid computing the statistics
            shapes = eval_shape(cls._avg_stats, state)
            sums = tree.map(lambda s: jnp.zeros(s.shape, s.dtype), shapes)
        else:
            sums = None
        return cls(sums=sums, count=jnp.int32(0))

    def update(self, state: State) -> 'StatsAccumulator':
        """Add the latest iteration's statistics; no-op when disabled."""
        if self.sums is None:
            return self
        sums = tree.map(jnp.add, self.sums, self._avg_stats(state))
        return replace(self, sums=sums, count=self.count + 1)

    def reset_if(self, cond: bool | Bool[Array, '']) -> 'StatsAccumulator':
        """Zero the running sums where `cond` holds; no-op when disabled."""
        if self.sums is None:
            return self
        sums = tree.map(lambda s: jnp.where(cond, 0, s), self.sums)
        return replace(self, sums=sums, count=jnp.where(cond, 0, self.count))

    def report(self, state: State) -> StatsReport:
        """Statistics to display: the windowed average if enabled, else the latest."""
        if self.sums is None:
            averaged = self._avg_stats(state)
            n_samples = None
        else:
            averaged = tree.map(lambda s: s / self.count, self.sums)
            n_samples = self.count
        return StatsReport(**averaged, **self._static_stats(state), n_samples=n_samples)

    @staticmethod
    def _avg_stats(state: State) -> dict[str, Float32[Array, ''] | None]:
        """Per-iteration diagnostics that are averaged over the report window."""
        forest = state.forest
        chain_axis = chain_vmap_axes(forest).split_tree
        num_trees_axis = chainful_axis(0, chain_axis)  # (num_trees, hts)
        split_tree = chain_to_axis(forest.split_tree, chain_axis)
        prop_total = forest.split_tree.shape[num_trees_axis]

        log_s = forest.log_s
        if log_s is None:
            peff = None
        else:
            log_s = chain_to_axis(log_s, chain_vmap_axes(forest).log_s)
            peff = StatsAccumulator._effective_predictors(log_s)

        return dict(
            grow_prop=forest.grow_prop_count.mean() / prop_total,
            move_acc=(forest.grow_acc_count.mean() + forest.prune_acc_count.mean())
            / prop_total,
            mean_leaves=forest_mean_leaves(split_tree),
            peff=peff,
        )

    @staticmethod
    def _static_stats(state: State) -> dict[str, int | None]:
        """Per-iteration diagnostics shown as-is, constant over the run."""
        forest = state.forest
        split_tree = chain_to_axis(
            forest.split_tree, chain_vmap_axes(forest).split_tree
        )
        log_s = forest.log_s
        if log_s is None:
            p = None
        else:
            *_, p = chain_to_axis(log_s, chain_vmap_axes(forest).log_s).shape
        return dict(num_chains=state.num_chains(), max_leaves=split_tree.shape[-1], p=p)

    @staticmethod
    def _effective_predictors(log_s: Float32[Array, '*chains p']) -> Float32[Array, '']:
        """Effective number of predictors used for splitting across all chains.

        Perplexity (exponential of the Shannon entropy) of the split-variable
        distribution ``s = softmax(log_s)`` pooled (averaged) over chains. It is
        1 when all chains concentrate on a single shared predictor and ``p`` when
        the pooled distribution is uniform; in general a pooled distribution
        spread evenly over ``k`` predictors gives ``k``. Chains are pooled before
        taking the entropy because predictions average over all chains, so a
        predictor used by any chain counts as used.
        """
        *_, p = log_s.shape
        # normalize each chain
        log_prob = log_s - logsumexp(log_s, axis=-1, keepdims=True)
        per_chain = log_prob.reshape(-1, p)
        num_chains, _ = per_chain.shape
        # mix over chains, i.e., logmeanexp over the chain axis
        log_pool = logsumexp(per_chain, axis=0) - jnp.log(num_chains)
        prob = jnp.exp(log_pool)
        # the where avoids the 0 * -inf = nan term where a probability is 0, the
        # same guard `jax.scipy.special.entr` uses, but reusing the log we have
        entropy = -jnp.sum(prob * jnp.where(prob, log_pool, 1.0))
        return jnp.exp(entropy)


def make_print_callback(
    state: State,
    *,
    dot_every: int | Integer[Array, ''] | None = 1,
    report_every: int | Integer[Array, ''] | None = 100,
    average: bool = True,
) -> dict[str, Any]:
    """
    Prepare a progress-printing callback for `run_mcmc`.

    The callback prints a dot on every iteration, and a longer report
    periodically.

    Parameters
    ----------
    state
        The MCMC state to use the callback with, used to determine device
        sharding.
    dot_every
        A dot is printed every `dot_every` MCMC iterations, `None` to disable.
    report_every
        A one line report is printed every `report_every` MCMC iterations,
        `None` to disable.
    average
        If `True`, the reported statistics are averaged over the iterations
        since the previous report; if `False`, they reflect the current
        iteration only. Ignored when `report_every` is `None`.

    Returns
    -------
    A dictionary with the arguments to pass to `run_mcmc` as keyword arguments to set up the callback.

    Examples
    --------
    >>> run_mcmc(key, state, ..., **make_print_callback(state, ...))
    """

    def as_replicated_array_or_none(val: ArrayLike | None) -> None | Array:
        return None if val is None else _replicate(jnp.asarray(val), state.config.mesh)

    accumulator = tree.map(
        partial(_replicate, mesh=state.config.mesh),
        StatsAccumulator.initial(state, enabled=average and report_every is not None),
    )

    return dict(
        callback=print_callback,
        callback_state=PrintCallbackState(
            as_replicated_array_or_none(dot_every),
            as_replicated_array_or_none(report_every),
            accumulator,
        ),
    )


class PrintCallbackState(Module):
    """State for `print_callback`."""

    dot_every: Int32[Array, ''] | None
    """A dot is printed every `dot_every` MCMC iterations, `None` to disable."""

    report_every: Int32[Array, ''] | None
    """A one line report is printed every `report_every` MCMC iterations,
    `None` to disable."""

    accumulator: StatsAccumulator
    """Running average of the reported statistics, inert unless averaging."""


def print_callback(
    *,
    state: State,
    burnin: Bool[Array, ''],
    i_total: Int32[Array, ''],
    n_burn: Int32[Array, ''],
    n_save: Int32[Array, ''],
    n_skip: Int32[Array, ''],
    callback_state: PrintCallbackState,
    **_: Any,
) -> tuple[State, PrintCallbackState]:
    """Print a dot and/or a report periodically during the MCMC."""
    report_every = callback_state.report_every
    dot_every = callback_state.dot_every
    it = i_total + 1

    accumulator = callback_state.accumulator.update(state)

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
            accumulator.report(state),
            print_dot=dot_cond,
            print_newline=print_newline,
            burnin=burnin,
            it=it,
            n_iters=n_burn + n_save * n_skip,
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

    accumulator = accumulator.reset_if(report_cond)
    return state, replace(callback_state, accumulator=accumulator)


def make_tqdm_callback(
    state: State,
    *,
    update_every: int = 1,
    report_every: int | None = 100,
    average: bool = True,
    **tqdm_kwargs: Any,
) -> dict[str, Any]:
    """
    Prepare a `tqdm` progress-bar callback for `run_mcmc`.

    The callback shows a progress bar that advances with the MCMC iterations,
    optionally annotated with the proposal acceptance statistics.

    Parameters
    ----------
    state
        The MCMC state to use the callback with, used to determine device
        sharding.
    update_every
        The bar position is refreshed every `update_every` MCMC iterations
        (`tqdm` further throttles the actual redraw rate on its own).
    report_every
        The acceptance statistics shown next to the bar are refreshed every
        `report_every` MCMC iterations, `None` to omit them.
    average
        If `True`, the statistics shown are averaged over the iterations since
        the previous refresh; if `False`, they reflect the current iteration
        only. Ignored when `report_every` is `None`.
    **tqdm_kwargs
        Additional keyword arguments forwarded to the `tqdm.tqdm` constructor,
        e.g., ``desc``, ``file``, or ``disable``.

    Returns
    -------
    A dictionary with the arguments to pass to `run_mcmc` as keyword arguments to set up the callback.

    Notes
    -----
    Works with chains sharded across multiple devices. If the run is interrupted
    (e.g. with ^C), the bar is left as-is; the next `make_tqdm_callback` call
    closes it, so a subsequent run starts from a clean line.

    Examples
    --------
    >>> run_mcmc(key, state, ..., **make_tqdm_callback(state, ...))
    """
    _close_stale_bars()  # clean up after any previous run that was interrupted
    bar_id = next(_TQDM_BAR_COUNTER)
    _TQDM_REGISTRY[bar_id] = _TqdmEntry(tqdm_kwargs)

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
            accumulator=tree.map(
                partial(_replicate, mesh=state.config.mesh),
                StatsAccumulator.initial(
                    state, enabled=average and report_every is not None
                ),
            ),
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

    accumulator: StatsAccumulator
    """Running average of the reported statistics, inert unless averaging."""


def tqdm_callback(
    *,
    state: State,
    i_total: Int32[Array, ''],
    n_burn: Int32[Array, ''],
    n_save: Int32[Array, ''],
    n_skip: Int32[Array, ''],
    callback_state: TqdmCallbackState,
    **_: Any,
) -> tuple[State, TqdmCallbackState]:
    """Advance a `tqdm` progress bar during the MCMC."""
    it = i_total + 1
    n_iters = n_burn + n_save * n_skip
    bar_id = callback_state.bar_id
    last = it == n_iters

    accumulator = callback_state.accumulator.update(state)

    # The callbacks are unordered: `ordered=True` is unsupported with more than
    # one device, and we need this to work with chains sharded across devices.
    # `_tqdm_advance` is therefore robust to out-of-order invocations.

    # refresh the statistics first so they tend to be visible by the time the
    # bar is advanced
    report_every = callback_state.report_every
    if report_every is not None:
        report_cond = (it % report_every == 0) | last

        def report_branch() -> None:
            debug.callback(_tqdm_report, accumulator.report(state), bar_id, n_iters)

        lax.cond(report_cond, report_branch, lambda: None)
        accumulator = accumulator.reset_if(report_cond)

    lax.cond(
        (it % callback_state.update_every == 0) | last,
        lambda: debug.callback(_tqdm_advance, bar_id, it, n_iters),
        lambda: None,
    )

    return state, replace(callback_state, accumulator=accumulator)


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
    report: StatsReport,
    *,
    print_dot: bool,
    print_newline: bool,
    burnin: bool,
    it: int,
    n_iters: int,
) -> None:
    """Print the report for `print_callback`."""
    # determine prefix
    if print_dot:
        prefix = '.\n'
    elif print_newline:
        prefix = '\n'
    else:
        prefix = ''

    # determine suffix in parentheses: what the statistics are averaged over
    avg_over = []
    if report.num_chains is not None:
        avg_over.append(f'{report.num_chains} chains')
    if report.n_samples is not None:
        avg_over.append(f'{report.n_samples} samples')
    msgs = []
    if avg_over:
        msgs.append('avg. ' + ' x '.join(avg_over))
    if burnin:
        msgs.append('burnin')
    suffix = f' ({", ".join(msgs)})' if msgs else ''

    # variable-selection concentration, only shown when it is enabled
    if report.peff is None:
        var_msg = ''
    else:
        var_msg = f'var: {report.peff:.1f}/{report.p}, '

    print(  # noqa: T201, see print_callback for why not logging
        f'{prefix}Iteration {it}/{n_iters}, '
        f'grow prob: {report.grow_prop:.0%}, '
        f'move acc: {report.move_acc:.0%}, '
        f'{var_msg}'
        f'leaves: {report.mean_leaves:.1f}/{report.max_leaves}{suffix}'
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
_TQDM_REGISTRY: dict[int, _TqdmEntry] = {}
_TQDM_BAR_COUNTER = itertools.count()

# tqdm's default layout, but without the ': ' that `format_meter` forces after a
# non-empty description; the label is set as a `{desc}` ending in a space instead
_TQDM_BAR_FORMAT = (
    '{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
    '[{elapsed}<{remaining}, {rate_fmt}{postfix}]'
)


def _close_stale_bars() -> None:
    """Close and drop any bars left over from a previous (e.g. interrupted) run."""
    for entry in _TQDM_REGISTRY.values():
        if entry.bar is not None:
            entry.bar.close()
    _TQDM_REGISTRY.clear()


def _get_or_create_bar(bar_id: int, n_iters: int) -> tqdm | None:
    """Return the bar for `bar_id`, creating it on first use, `None` if finished."""
    entry = _TQDM_REGISTRY.get(bar_id)
    if entry is None:
        # the bar was already closed (the loop finished, possibly out of order)
        return None
    if entry.bar is None:
        bar = tqdm(**{'total': n_iters, 'bar_format': _TQDM_BAR_FORMAT, **entry.kwargs})
        _TQDM_REGISTRY[bar_id] = replace(entry, bar=bar)
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
        del _TQDM_REGISTRY[bar_id]


@_convert_jax_arrays_in_args
# convert all jax arrays in arguments, see _print_report for why
def _tqdm_report(report: StatsReport, bar_id: int, n_iters: int) -> None:
    """Set the bar description and acceptance-statistics postfix."""
    bar = _get_or_create_bar(bar_id, n_iters)
    if bar is None:
        return
    # set_description_str (not set_description) to avoid tqdm's ': ' suffix; the
    # trailing space separates the label from the bar
    bar.set_description_str('train ', refresh=False)
    # keep this terse so the bar stays narrow, e.g. '4ch 100sa acc 25% leaves 3.4/32'
    msgs = []
    if report.num_chains is not None:
        msgs.append(f'{report.num_chains}ch')
    if report.n_samples is not None:
        msgs.append(f'{report.n_samples}sa')
    msgs.append(f'acc {report.move_acc:.0%}')
    if report.peff is not None:
        msgs.append(f'var {report.peff:.1f}/{report.p}')
    msgs.append(f'leaves {report.mean_leaves:.1f}/{report.max_leaves}')
    bar.set_postfix_str(' '.join(msgs))
