# bartz/tests/util.py
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

"""Functions intended to be shared across the test suite."""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from operator import ge, le
from os import getpid, kill
from signal import SIGINT
from threading import Event, Thread
from time import monotonic
from typing import Any

import numpy as np
from jax import numpy as jnp
from jaxtyping import ArrayLike, Float
from numpy.testing import assert_allclose as _np_assert_allclose  # noqa: TID251
from numpy.testing import assert_array_equal as _np_assert_array_equal  # noqa: TID251
from scipy import linalg, stats

from bartz.grove import TreesTrace, check_trace, describe_error
from bartz.jaxext import minimal_unsigned_dtype


def manual_tree(
    leaf: list[list[float]],
    var: list[list[int]],
    split: list[list[int]],
    /,
    *,
    ignore_errors: Sequence[str] = (),
) -> TreesTrace:
    """Facilitate the hardcoded definition of tree heaps."""
    assert len(leaf) == len(var) + 1 == len(split) + 1

    def check_powers_of_2(seq: list[list]) -> bool:
        """Check if the lengths of the lists in `seq` are powers of 2."""
        return all(len(x) == 2**i for i, x in enumerate(seq))

    check_powers_of_2(leaf)
    check_powers_of_2(var)
    check_powers_of_2(split)

    tree = TreesTrace(
        jnp.concatenate([jnp.zeros(1), *map(jnp.array, leaf)]),
        jnp.concatenate([jnp.zeros(1, int), *map(jnp.array, var)]),
        jnp.concatenate([jnp.zeros(1, int), *map(jnp.array, split)]),
    )

    p = jnp.max(tree.var_tree) + 1
    var_type = minimal_unsigned_dtype(p - 1)
    split_type = minimal_unsigned_dtype(jnp.max(tree.split_tree))
    max_split = jnp.full(p, jnp.max(tree.split_tree), split_type)
    tree = replace(
        tree,
        var_tree=tree.var_tree.astype(var_type),
        split_tree=tree.split_tree.astype(split_type),
    )

    error = check_trace(tree, max_split)
    descr = describe_error(error)
    bad = any(d not in ignore_errors for d in descr)
    assert not bad, descr

    return tree


def assert_close_matrices(
    actual: ArrayLike,
    desired: ArrayLike,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    tozero: bool = False,
    negate: bool = False,
    ord: int | float | str | None = 2,  # noqa: A002
    err_msg: str = '',
    reduce_rank: bool = False,
) -> None:
    """
    Check if two matrices are similar.

    Parameters
    ----------
    actual
    desired
        The two matrices to be compared. Must be scalars, vectors, or 2d arrays.
        Scalars and vectors are intepreted as 1x1 and Nx1 matrices, but the two
        arrays must have the same shape and dtype beforehand.
    rtol
    atol
        Relative and absolute tolerances for the comparison. The closeness
        condition is:

            ||actual - desired|| <= atol + rtol * ||desired||,

        where the norm is the matrix 2-norm, i.e., the maximum (in absolute
        value) singular value.
    tozero
        If True, use the following codition instead:

            ||actual|| <= atol + rtol * ||desired||

        So `actual` is compared to zero, and `desired` is only used as a
        reference to set the threshold.
    negate
        If True, invert the inequality, replacing <= with >=. This makes the
        function check the two matrices are different instead of similar.
    ord
        Passed to `numpy.linalg.norm` to specify the matrix norm to use, the
        default is 2 which differs from numpy.
    err_msg
        Prefix prepended to the error message (without adding newlines).
    reduce_rank
        If True, reduce the input arrays to 2d by collapsing leading dimensions.

    Notes
    -----
    Boolean values are converted to uint8.
    """
    actual = np.asarray(actual)
    desired = np.asarray(desired)

    assert actual.shape == desired.shape
    assert actual.dtype == desired.dtype

    if actual.dtype == bool:
        actual = actual.astype(np.uint8)
        desired = desired.astype(np.uint8)

    if actual.size > 0:
        actual = np.atleast_1d(actual)
        desired = np.atleast_1d(desired)

        if actual.ndim > 2 and reduce_rank:
            n = actual.shape[-1]
            actual = actual.reshape(-1, n)
            desired = desired.reshape(-1, n)

        if tozero:
            expr = 'actual'
            ref = 'zero'
        else:
            expr = 'actual - desired'
            ref = 'desired'

        if negate:
            cond = 'different'
            op = ge
        else:
            cond = 'close'
            op = le

        dnorm = linalg.norm(desired, ord)
        adnorm = linalg.norm(eval(expr), ord)  # noqa: S307, expr is a literal
        ratio = adnorm / dnorm if dnorm else np.nan

        msg = f"""{err_msg}\
matrices actual and {ref} are not {cond} enough in {ord}-norm
matrix shape: {desired.shape}
norm(desired) = {dnorm:.2g}
norm({expr}) = {adnorm:.2g}  (atol = {atol:.2g})
ratio = {ratio:.2g}  (rtol = {rtol:.2g})"""

        assert op(adnorm, atol + rtol * dnorm), msg


def assert_different_matrices(*args: ArrayLike, **kwargs: Any) -> None:
    """Invoke `assert_close_matrices` with negate=True and default inf tolerance."""
    default_kwargs: dict = dict(rtol=np.inf, atol=np.inf)
    default_kwargs.update(kwargs)
    assert_close_matrices(*args, negate=True, **default_kwargs)


def assert_allclose(
    actual: ArrayLike,
    desired: ArrayLike,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    **kwargs: Any,
) -> None:
    """Wrap `numpy.testing.assert_allclose` with zero default tolerances."""
    _np_assert_allclose(actual, desired, rtol=rtol, atol=atol, **kwargs)


def assert_array_equal(
    actual: ArrayLike, desired: ArrayLike, *, strict: bool = True, **kwargs: Any
) -> None:
    """Wrap `numpy.testing.assert_array_equal` with `strict=True` default."""
    _np_assert_array_equal(actual, desired, strict=strict, **kwargs)


class PeriodicSigintTimer:
    """Periodically send SIGINT (^C) to the main thread.

    Parameters
    ----------
    first_after
        Time in seconds to wait before sending the first SIGINT.
    interval
        Time in seconds between subsequent SIGINTs.
    """

    def __init__(self, *, first_after: float, interval: float) -> None:
        self.first_after = max(0.0, float(first_after))
        self.interval = max(0.001, float(interval))
        self.pid = getpid()
        self._stop = Event()
        self._thread: Thread | None = None
        self.sent = 0

    def _run(self) -> None:
        """Run the main loop of the timer."""
        t0 = monotonic()

        # Wait initial delay (cancellable)
        self._stop.wait(self.first_after)

        # Periodically send SIGINT until stopped
        while not self._stop.is_set():  # pragma: no branch
            kill(self.pid, SIGINT)
            self.sent += 1
            elapsed = monotonic() - t0
            print(f'[PeriodicSigintTimer] sent SIGINT #{self.sent} at t={elapsed:.2f}s')
            self._stop.wait(self.interval)

    def start(self) -> None:
        """Start the timer."""
        assert self._thread is None, 'Timer already started'
        self._thread = Thread(target=self._run, name='PeriodicSigintTimer', daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Stop the timer."""
        assert self._thread is not None, 'Timer not started'
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():  # pragma: no cover
            msg = '[PeriodicSigintTimer] failed to stop timer'
            raise RuntimeError(msg)
        print(f'[PeriodicSigintTimer] stopped after {self.sent} SIGINT(s)')


@contextmanager
def periodic_sigint(
    *, first_after: float, interval: float
) -> Generator[PeriodicSigintTimer, None, None]:
    """Context manager to periodically send SIGINT to the main thread."""
    timer = PeriodicSigintTimer(first_after=first_after, interval=interval)
    try:
        timer.start()
        yield timer
    finally:
        if timer._thread is not None:
            timer.cancel()


def rhat_rank(
    data: ArrayLike, *, split: bool, chain_axis: int = 0, draw_axis: int = 1
) -> Float[np.ndarray, ' *leading']:
    """Elementwise rank-normalized (split-)Rhat.

    Port of ``arviz_stats.rhat`` with ``method='rank'``, the rank-normalized
    folded split-Rhat of Vehtari et al. (2021),
    https://arxiv.org/abs/1903.08008. Vectorized over the leading shape.

    Parameters
    ----------
    data
        Array of MCMC draws with chains along ``chain_axis`` and draws along
        ``draw_axis``.
    split
        If True, split each chain in half along the draw axis before computing
        Rhat (the standard rank-normalized split-Rhat). If False, skip the
        split step and compute the rank-normalized Rhat on the chains as
        given.
    chain_axis
    draw_axis
        Axes of ``data`` indexing chains and draws. Default to the arviz
        ``(chain, draw, ...)`` layout.

    Returns
    -------
    Array with ``chain_axis`` and ``draw_axis`` of ``data`` removed; each entry
    is the rank-normalized (split-)Rhat over the corresponding (chain, draw)
    slice. NaNs in a slice propagate to ``nan`` in its Rhat.

    Raises
    ------
    ValueError
        If the chain dimension has fewer than 2 entries, or the draw dimension
        has fewer than 4 (with ``split=True``) or 2 (with ``split=False``).
    """
    ary = np.asarray(data, float)
    ary = np.moveaxis(ary, (chain_axis, draw_axis), (-2, -1))
    *_, n_chain, n_draw = ary.shape
    min_draw = 4 if split else 2
    if n_chain < 2 or n_draw < min_draw:
        msg = (
            f'need at least 2 chains and {min_draw} draws, got {n_chain} '
            f'chains and {n_draw} draws'
        )
        raise ValueError(msg)

    prepared = _split_chains_v(ary) if split else ary
    rhat_bulk = _rhat_v(_z_scale_v(prepared))
    folded = np.abs(prepared - np.median(prepared, axis=(-2, -1), keepdims=True))
    rhat_tail = _rhat_v(_z_scale_v(folded))
    return np.maximum(rhat_bulk, rhat_tail)


def _split_chains_v(
    ary: Float[np.ndarray, '*leading chain draw'],
) -> Float[np.ndarray, '*leading two_chain half_draw']:
    """Split each chain in half along the draw axis; stack along chain axis."""
    half = ary.shape[-1] // 2
    return np.concatenate([ary[..., :half], ary[..., -half:]], axis=-2)


def _z_scale_v(
    ary: Float[np.ndarray, '*leading chain draw'],
) -> Float[np.ndarray, '*leading chain draw']:
    """Rank-normalize jointly over the last two axes (chain, draw)."""
    flat = ary.reshape(*ary.shape[:-2], -1)
    rank = stats.rankdata(flat, method='average', axis=-1)
    z = stats.norm.ppf((rank - 3 / 8) / (rank.shape[-1] + 1 / 4))
    return z.reshape(ary.shape)


def _rhat_v(
    ary: Float[np.ndarray, '*leading chain draw'],
) -> Float[np.ndarray, ' *leading']:
    """Classic Gelman-Rubin Rhat reducing the last two ``(chain, draw)`` axes."""
    n = ary.shape[-1]
    chain_mean = ary.mean(axis=-1)
    chain_var = ary.var(axis=-1, ddof=1)
    between = n * chain_mean.var(axis=-1, ddof=1)
    within = chain_var.mean(axis=-1)
    return np.sqrt((between / within + n - 1) / n)
