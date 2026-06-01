# bartz/src/bartz/mcmcstep/_lazy.py
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

"""Deferred array construction used to lay out the MCMC state before sharding."""

from collections.abc import Callable
from typing import Any, TypeVar

from equinox import Module
from jax import ShapeDtypeStruct, tree
from jax import numpy as jnp
from jaxtyping import Array, PyTree

T = TypeVar('T')


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


DummyArray = Array | ShapeDtypeStruct | _LazyArray


# WORKAROUND(jaxtyping<0.3.9): a shared structure variable
# (PyTree[DummyArray, 'T'] -> PyTree[ShapeDtypeStruct, 'T']) is mis-bound to a
# single leaf when the leaf type is a union containing a Module (here
# `_LazyArray`), so the return-value check spuriously fails. Drop the structure
# variable; `tree.map` preserves the structure anyway. Restore 'T' on both
# annotations once the jaxtyping floor reaches 0.3.9.
def add_dummy_axis(x: PyTree[DummyArray]) -> PyTree[ShapeDtypeStruct]:
    """Replace array-like leaves with a rank-inflated placeholder."""

    def replace_leaf(leaf: DummyArray) -> ShapeDtypeStruct:
        return ShapeDtypeStruct((0,) * (leaf.ndim + 1), jnp.float32)

    return tree.map(replace_leaf, x, is_leaf=lambda x: isinstance(x, _LazyArray))


def _return_array(shape: tuple[int, ...], arr: Array, **kwargs: Any) -> Array:  # noqa: ARG001
    """`_LazyArray` factory that returns an already-built array."""
    return arr


def _lazy_from_array(arr: Array | None) -> _LazyArray | None:
    """Wrap an existing array as a `_LazyArray` reporting `arr.shape`, or pass `None`."""
    if arr is None:
        return None
    return _LazyArray(_return_array, arr.shape, arr)


def _broadcast_chain(
    shape: tuple[int, ...], inner: _LazyArray, chain_axis: int, **kwargs: Any
) -> Array:
    """Concretize `inner` then insert and broadcast a chain axis at `chain_axis`."""
    arr = inner(**kwargs)
    arr = jnp.expand_dims(arr, chain_axis)
    return jnp.broadcast_to(arr, shape)


def _wrap_chain(
    inner: _LazyArray, chain_axis: int | None, num_chains: int | None
) -> _LazyArray:
    """Wrap `inner` so its factory inserts and broadcasts `num_chains` at `chain_axis`. No-op when `chain_axis` is `None`."""
    if chain_axis is None:
        return inner
    new_shape = (*inner.shape[:chain_axis], num_chains, *inner.shape[chain_axis:])
    return _LazyArray(_broadcast_chain, new_shape, inner, chain_axis)


def _is_lazy_or_none(x: object) -> bool:
    """`tree.map(is_leaf=...)` predicate that stops at `_LazyArray` or `None`."""
    return x is None or isinstance(x, _LazyArray)
