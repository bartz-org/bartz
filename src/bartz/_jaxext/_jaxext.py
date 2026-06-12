# bartz/src/bartz/_jaxext/_jaxext.py
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

"""Implementation of miscellaneous jax extension utilities."""

import math
import sys
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from functools import partial
from typing import Any

import jax
from jax import Device, ensure_compile_time_eval, lax, random, shard_map, tree, vmap
from jax import numpy as jnp
from jax.dtypes import prng_key
from jax.scipy.special import ndtr
from jax.sharding import PartitionSpec
from jax.typing import DTypeLike
from jaxtyping import (
    Array,
    Bool,
    Float32,
    Integer,
    Key,
    PyTree,
    Scalar,
    ScalarLike,
    Shaped,
)
from jaxtyping import config as jaxtyping_config

from bartz._jaxext._jit import jit
from bartz._jaxext.scipy.special import ndtri

if sys.version_info >= (3, 13):
    from typing import TypeIs
else:  # WORKAROUND(python<3.13): typing.TypeIs was added in 3.13
    from typing_extensions import TypeIs


@contextmanager
def jaxtyping_disabled() -> Generator[None, None, None]:
    """Temporarily disable jaxtyping runtime type-checking.

    This also disables `beartype`, because the jaxtyping import hook applies it
    as ``jaxtyped(typechecker=beartype)`` and `jaxtyped` short-circuits to the
    undecorated function when type-checking is disabled. Used to park
    deliberately wrong-typed intermediates (e.g. `_LazyArray` leaves) in an
    `equinox.Module` during construction.
    """
    old = jaxtyping_config.jaxtyping_disable
    jaxtyping_config.update('jaxtyping_disable', True)
    try:
        yield
    finally:
        jaxtyping_config.update('jaxtyping_disable', old)


def vmap_nodoc(fun: Callable, *args: Any, **kw: Any) -> Callable:
    """
    Acts like `jax.vmap` but preserves the docstring of the function unchanged.

    This is useful if the docstring already takes into account that the
    arguments have additional axes due to vmap.
    """
    doc = fun.__doc__
    fun = vmap(fun, *args, **kw)
    fun.__doc__ = doc
    return fun


def minimal_unsigned_dtype(value: int) -> DTypeLike:
    """Return the smallest unsigned integer dtype that can represent `value`."""
    if value < 2**8:
        return jnp.uint8
    if value < 2**16:
        return jnp.uint16
    if value < 2**32:
        return jnp.uint32
    return jnp.uint64


@jit(static_argnums=(1,))
def unique(
    x: Shaped[Array, ' _'], size: int, fill_value: ScalarLike
) -> tuple[Shaped[Array, ' {size}'], int | Integer[Array, '']]:
    """
    Restricted version of `jax.numpy.unique` that uses less memory.

    Parameters
    ----------
    x
        The input array.
    size
        The length of the output.
    fill_value
        The value to fill the output with if `size` is greater than the number
        of unique values in `x`.

    Returns
    -------
    out : Shaped[Array, '{size}']
        The unique values in `x`, sorted, and right-padded with `fill_value`.
    actual_length : int
        The number of used values in `out`.
    """
    if x.size == 0:
        return jnp.full(size, fill_value, x.dtype), 0
    if size == 0:
        return jnp.empty(0, x.dtype), 0
    x = jnp.sort(x)

    def loop(
        carry: tuple[Scalar, Scalar, Shaped[Array, ' size']], x: Scalar
    ) -> tuple[tuple[Scalar, Scalar, Shaped[Array, ' size']], None]:
        i_out, last, out = carry
        i_out = jnp.where(x == last, i_out, i_out + 1)
        out = out.at[i_out].set(x)
        return (i_out, x, out), None

    carry = 0, x[0], jnp.full(size, fill_value, x.dtype)

    def run(unroll: int) -> tuple[Shaped[Array, ' size'], Scalar]:
        (actual_length, _, out), _ = lax.scan(loop, carry, x[:size], unroll=unroll)
        return out, actual_length + 1

    # The optimal scan unroll is opposite on cpu and gpu (benchmarked):
    # - gpu: the loop is dominated by per-step overhead, so a large unroll is up
    #   to ~6x faster; the run time plateaus by ~32 while compile time then grows
    #   steeply, so 32 is the sweet spot.
    # - cpu: past ~6 the backend stops aliasing `out` in place and copies the
    #   size-`size` buffer each step (O(size**2), ~100x slower), so 2 is safest.
    # `default` (cpu, tpu, untested backends) takes the conservative value.
    return lax.platform_dependent(
        cuda=partial(run, 32), rocm=partial(run, 32), default=partial(run, 2)
    )


class split:
    """
    Split a key into `num` keys.

    Parameters
    ----------
    key
        The key to split.
    num
        The number of keys to split into.
    """

    _keys: tuple[Key[Array, ''], ...]
    _num_used: int

    def __init__(self, key: Key[Array, ''], num: int = 2) -> None:
        self._keys = _split_unpack(key, num)
        self._num_used = 0

    def __len__(self) -> int:
        return len(self._keys) - self._num_used

    def pop(self, shape: int | tuple[int, ...] = ()) -> Key[Array, ' *shape']:
        """
        Pop one or more keys from the list.

        Parameters
        ----------
        shape
            The shape of the keys to pop. If empty (default), a single key is
            popped and returned. If not empty, the popped key is split and
            reshaped to the target shape.

        Returns
        -------
        The popped keys as a jax array with the requested shape.

        Raises
        ------
        IndexError
            If the list is empty.
        """
        if len(self) == 0:
            msg = 'No keys left to pop'
            raise IndexError(msg)
        if not isinstance(shape, tuple):
            shape = (shape,)
        key = self._keys[self._num_used]
        self._num_used += 1
        if shape:
            key = _split_shaped(key, shape)
        return key


@jit(static_argnums=(1,))
def _split_unpack(key: Key[Array, ''], num: int) -> tuple[Key[Array, ''], ...]:
    keys = random.split(key, num)
    return tuple(keys)


@jit(static_argnums=(1,))
def _split_shaped(key: Key[Array, ''], shape: tuple[int, ...]) -> Key[Array, ' *shape']:
    num = math.prod(shape)
    keys = random.split(key, num)
    return keys.reshape(shape)


def truncated_normal_onesided(
    key: Key[Array, ''],
    shape: Sequence[int],
    upper: Bool[Array, '...'],
    bound: Float32[Array, '...'],
    *,
    clip: bool = True,
) -> Float32[Array, '...']:
    """
    Sample from a one-sided truncated standard normal distribution.

    Parameters
    ----------
    key
        JAX random key.
    shape
        Shape of output array, broadcasted with other inputs.
    upper
        True for (-∞, bound], False for [bound, ∞).
    bound
        The truncation boundary.
    clip
        Whether to clip the truncated uniform samples to (0, 1) before
        transforming them to truncated normal. Intended for debugging purposes.

    Returns
    -------
    Array of samples from the truncated normal distribution.
    """
    # Pseudocode:
    # | if upper:
    # |     if bound < 0:
    # |         ndtri(uniform(0, ndtr(bound))) =
    # |         ndtri(ndtr(bound) * u)
    # |     if bound > 0:
    # |         -ndtri(uniform(ndtr(-bound), 1)) =
    # |         -ndtri(ndtr(-bound) + ndtr(bound) * (1 - u))
    # | if not upper:
    # |     if bound < 0:
    # |         ndtri(uniform(ndtr(bound), 1)) =
    # |         ndtri(ndtr(bound) + ndtr(-bound) * (1 - u))
    # |     if bound > 0:
    # |         -ndtri(uniform(0, ndtr(-bound))) =
    # |         -ndtri(ndtr(-bound) * u)
    shape = jnp.broadcast_shapes(shape, upper.shape, bound.shape)
    bound_pos = bound > 0
    ndtr_bound = ndtr(bound)
    ndtr_neg_bound = ndtr(-bound)
    scale = jnp.where(upper, ndtr_bound, ndtr_neg_bound)
    shift = jnp.where(upper, ndtr_neg_bound, ndtr_bound)
    u = random.uniform(key, shape)
    left_u = scale * (1 - u)  # ~ uniform in (0, ndtr(±bound)]
    right_u = shift + scale * u  # ~ uniform in [ndtr(∓bound), 1)
    truncated_u = jnp.where(upper ^ bound_pos, left_u, right_u)
    if clip:
        # on gpu the accuracy is lower and sometimes u can reach the boundaries
        zero = jnp.zeros((), truncated_u.dtype)
        one = jnp.ones((), truncated_u.dtype)
        truncated_u = jnp.clip(
            truncated_u, jnp.nextafter(zero, one), jnp.nextafter(one, zero)
        )
    truncated_norm = ndtri(truncated_u)
    return jnp.where(bound_pos, -truncated_norm, truncated_norm)


def get_default_device() -> Device:
    """Get the current default JAX device."""
    with ensure_compile_time_eval():
        return jnp.empty(0).device


def get_default_devices() -> list[Device]:
    """Get all JAX devices on the default platform."""
    return jax.devices(get_default_device().platform)


def get_device_count() -> int:
    """Get the number of available devices on the default platform."""
    return len(get_default_devices())


def is_key(x: object) -> TypeIs[Key[Array, ' *shape']]:
    """Determine if `x` is a jax random key."""
    return isinstance(x, Array) and jnp.issubdtype(x.dtype, prng_key)


def jit_active() -> bool:
    """Check if we are under jit."""
    return not hasattr(jnp.empty(0), 'platform')


def _equal_shards(x: Shaped[Array, '...'], axis_name: str) -> Bool[Array, '']:
    """Check if all shards of `x` are equal, to be used in a `shard_map` context."""
    size = lax.axis_size(axis_name)
    perm = [(i, (i + 1) % size) for i in range(size)]
    perm_x = lax.ppermute(x, axis_name, perm)
    diff = jnp.any(x != perm_x)
    return jnp.logical_not(lax.psum(diff, axis_name))


def equal_shards(
    x: PyTree[Array, ' S'], axis_name: str, **shard_map_kwargs: Any
) -> PyTree[Bool[Array, ''], ' S']:
    """Check that all shards of `x` are equal across axis `axis_name`.

    Parameters
    ----------
    x
        A pytree of arrays to check. Each array is checked separately.
    axis_name
        The mesh axis name across which equality is checked. It's not checked
        across other axes.
    **shard_map_kwargs
        Additional arguments passed to `jax.shard_map` to set up the function
        that checks equality. You may need to specify `in_specs` passing
        the (pytree of) `jax.sharding.PartitionSpec` that specifies how `x`
        is sharded, if the axes are not explicit, and `mesh` if there is not
        a default mesh set by `jax.set_mesh`.

    Returns
    -------
    A pytree of booleans indicating whether each leaf is equal across devices along the mesh axis.
    """
    equal_shards_leaf = partial(_equal_shards, axis_name=axis_name)

    def check_equal(x: PyTree[Array, ' S']) -> PyTree[Bool[Array, ''], ' S']:
        return tree.map(equal_shards_leaf, x)

    sharded_check_equal = shard_map(
        check_equal, out_specs=PartitionSpec(), **shard_map_kwargs
    )

    return sharded_check_equal(x)
