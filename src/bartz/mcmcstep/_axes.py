# bartz/src/bartz/mcmcstep/_axes.py
#
# Copyright (c) 2026, The Bartz Contributors
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

"""Batch-axis bookkeeping: `field` markers and the chain/data/sample resolvers.

Module dataclasses tag their array fields with chain/data/sample axis positions
via `field`; the resolvers here read those markers off a pytree to build the
`in_axes`/`out_axes` for `jax.vmap` and the axis positions used when reshaping.
"""

import os
from collections.abc import Callable, Hashable
from dataclasses import fields
from typing import Any, TypeVar

from equinox import Module as EquinoxModule
from jax import numpy as jnp
from jax import tree
from jaxtyping import Array, PyTree
from numpy.lib.array_utils import normalize_axis_index

from bartz.mcmcstep._lazy import DummyArray, _LazyArray

# Structure variable for the `PyTree[..., 'T']` annotations below.
T = TypeVar('T')

# Default position of the chain axis in chain-bearing leaves; see `field`.
CHAIN_AXIS = int(os.environ.get('CHAIN_AXIS', '0'))


def chain_vmap_axes(x: PyTree[EquinoxModule | Any, 'T']) -> PyTree[int | None, 'T ...']:
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
    A pytree with the same structure as `x`, with each leaf set to the chain axis index declared by its owning ``field(chains=...)`` marker, normalized against the leaf's ``ndim`` (so the returned indices are non-negative), or `None` for unmarked leaves. If `has_chains` is `False`, every leaf is `None`.
    """
    if not get_has_chains(x):
        return _find_metadata(x, 'chains', marker_value=_none_marker)

    return _find_metadata(x, 'chains')


def _none_marker(leaf: object, raw: int) -> None:  # noqa: ARG001
    """Marker mapper that always returns `None`."""
    return None  # noqa: RET501


def data_vmap_axes(x: PyTree[EquinoxModule | Any, 'T']) -> PyTree[int | None, 'T ...']:
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


def trace_sample_axes(
    trace: PyTree[EquinoxModule | Any, 'T'],
) -> PyTree[int | None, 'T ...']:
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


def chainful_axis(core_axis: int, chain_axis: int | None) -> int:
    """Position of a chainless-layout axis in the corresponding chainful array.

    Parameters
    ----------
    core_axis
        Non-negative axis position in the chainless ("core") layout.
    chain_axis
        Non-negative position of the chain axis in the chainful layout, or
        `None` if there is no chain axis.

    Returns
    -------
    The non-negative position of `core_axis` after inserting the chain axis at `chain_axis`.
    """
    if chain_axis is None or core_axis < chain_axis:
        return core_axis
    return core_axis + 1


def chain_to_axis(arr: Array, chain_axis: int | None, target: int = 0) -> Array:
    """Move `chain_axis` of `arr` to position `target`.

    Helper for the common pattern of normalizing the chain axis position in
    arrays derived from chain-marked Module fields. Pair it with
    `chain_vmap_axes` to fetch the source axis from a dataclass.

    Parameters
    ----------
    arr
        Array to be reordered.
    chain_axis
        Source position of the chain axis, or `None` for arrays with no chain
        axis (in which case `arr` is returned unchanged).
    target
        Destination position of the chain axis.

    Returns
    -------
    The reordered array.
    """
    if chain_axis is None:
        return arr
    return jnp.moveaxis(arr, chain_axis, target)


def _compute_core_axis(
    leaf: DummyArray | None, raw_axis: int | None, chain_axis: int | None
) -> int | None:
    """Combine a raw core-layout marker and a (normalized) chain position."""
    if raw_axis is None:
        return None
    assert leaf is not None
    has_chain = chain_axis is not None
    core_ndim = leaf.ndim - (1 if has_chain else 0)
    axis = normalize_axis_index(raw_axis, core_ndim)
    return chainful_axis(axis, chain_axis)


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
        value = getattr(node, 'has_chains', None)
        if value is None:
            return False
        raise _HasChainsFound(value)

    try:
        tree.map(lambda _: None, x, is_leaf=is_leaf)
    except _HasChainsFound as exc:
        return exc.value
    msg = 'no `has_chains` property found in the pytree'
    raise ValueError(msg)


def _normalize_axis_for_leaf(leaf: DummyArray, raw: int) -> int:
    """Normalize a marker axis index against `leaf.ndim`.

    Raises `numpy.exceptions.AxisError` if `raw` is out of bounds for
    `leaf.ndim`.
    """
    return normalize_axis_index(raw, leaf.ndim)


def _is_lazy_array(x: object) -> bool:
    return isinstance(x, _LazyArray)


def _is_module(x: object) -> bool:
    return isinstance(x, EquinoxModule) and not _is_lazy_array(x)


def _find_metadata(
    x: PyTree[Any, ' S'],
    key: Hashable,
    *,
    marker_value: Callable[[DummyArray, int], object] = _normalize_axis_for_leaf,
    default_value: object = None,
) -> PyTree[Any, ' S ...']:
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
        # rebuild bypassing the (type-checked) __init__: the result is a
        # same-structure pytree whose leaves are axis markers (int/None), not
        # the arrays the field annotations require.
        out = object.__new__(type(x))
        for f, value in zip(fields(x), args, strict=True):
            object.__setattr__(out, f.name, value)
        return out

    def get_axes(x: object) -> PyTree:
        if _is_module(x):
            return _find_metadata(
                x, key, marker_value=marker_value, default_value=default_value
            )
        return tree.map(lambda _: default_value, x, is_leaf=_is_lazy_array)

    return tree.map(get_axes, x, is_leaf=lambda x: isinstance(x, EquinoxModule))
