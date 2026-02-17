# bartz/src/bartz/jaxext/__init__.py
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

"""Additions to jax."""

import math
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from functools import partial
from typing import Any

try:
    from jax import shard_map  # available since jax v0.6.1
except ImportError:
    from jax.experimental.shard_map import shard_map

import jax
from jax import (
    Device,
    debug_key_reuse,
    device_count,
    ensure_compile_time_eval,
    jit,
    lax,
    random,
    tree,
    vmap,
)
from jax import numpy as jnp
from jax.dtypes import prng_key
from jax.scipy.special import ndtr
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float32, Key, PyTree, Scalar, Shaped

from bartz.jaxext._autobatch import autobatch  # noqa: F401
from bartz.jaxext.scipy.special import ndtri


def vmap_nodoc(fun: Callable, *args: Any, **kw: Any) -> Callable:
    """
    Acts like `jax.vmap` but preserves the docstring of the function unchanged.

    This is useful if the docstring already takes into account that the
    arguments have additional axes due to vmap.
    """
    doc = fun.__doc__
    fun = jax.vmap(fun, *args, **kw)
    fun.__doc__ = doc
    return fun


def minimal_unsigned_dtype(value: int) -> jnp.dtype:
    """Return the smallest unsigned integer dtype that can represent `value`."""
    if value < 2**8:
        return jnp.uint8
    if value < 2**16:
        return jnp.uint16
    if value < 2**32:
        return jnp.uint32
    return jnp.uint64


@partial(jax.jit, static_argnums=(1,))
def unique(
    x: Shaped[Array, ' _'], size: int, fill_value: Scalar
) -> tuple[Shaped[Array, ' {size}'], int]:
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
        carry: tuple[Scalar, Scalar, Shaped[Array, ' {size}']], x: Scalar
    ) -> tuple[tuple[Scalar, Scalar, Shaped[Array, ' {size}']], None]:
        i_out, last, out = carry
        i_out = jnp.where(x == last, i_out, i_out + 1)
        out = out.at[i_out].set(x)
        return (i_out, x, out), None

    carry = 0, x[0], jnp.full(size, fill_value, x.dtype)
    (actual_length, _, out), _ = lax.scan(loop, carry, x[:size])
    return out, actual_length + 1


class split:
    """
    Split a key into `num` keys.

    Parameters
    ----------
    key
        The key to split.
    num
        The number of keys to split into.

    Notes
    -----
    Unlike `jax.random.split`, this class supports a vector of keys as input. In
    this case, it behaves as if everything had been vmapped over, so `keys.pop`
    has an additional initial output dimension equal to the number of input
    keys, and the deterministic dependency respects this axis.
    """

    _keys: tuple[Key[Array, '*batch'], ...]
    _num_used: int

    def __init__(self, key: Key[Array, '*batch'], num: int = 2) -> None:
        if key.ndim:
            context = debug_key_reuse(False)
        else:
            context = nullcontext()
        with context:
            # jitted-vmapped key split seems to be triggering a false positive
            # with key reuse checks
            self._keys = _split_unpack(key, num)
        self._num_used = 0

    def __len__(self) -> int:
        return len(self._keys) - self._num_used

    def pop(self, shape: int | tuple[int, ...] = ()) -> Key[Array, '*batch {shape}']:
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


@partial(jit, static_argnums=(1,))
def _split_unpack(
    key: Key[Array, '*batch'], num: int
) -> tuple[Key[Array, '*batch'], ...]:
    if key.ndim == 0:
        keys = random.split(key, num)
    elif key.ndim == 1:
        keys = vmap(random.split, in_axes=(0, None), out_axes=1)(key, num)
    return tuple(keys)


@partial(jit, static_argnums=(1,))
def _split_shaped(
    key: Key[Array, '*batch'], shape: tuple[int, ...]
) -> Key[Array, '*batch {shape}']:
    num = math.prod(shape)
    if key.ndim == 0:
        keys = random.split(key, num)
    elif key.ndim == 1:
        keys = vmap(random.split, in_axes=(0, None))(key, num)
    return keys.reshape(*key.shape, *shape)


def truncated_normal_onesided(
    key: Key[Array, ''],
    shape: Sequence[int],
    upper: Bool[Array, '*'],
    bound: Float32[Array, '*'],
    *,
    clip: bool = True,
) -> Float32[Array, '*']:
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


def get_device_count() -> int:
    """Get the number of available devices on the default platform."""
    device = get_default_device()
    return device_count(device.platform)


def is_key(x: object) -> bool:
    """Determine if `x` is a jax random key."""
    return isinstance(x, Array) and jnp.issubdtype(x.dtype, prng_key)


def jit_active() -> bool:
    """Check if we are under jit."""
    return not hasattr(jnp.empty(0), 'platform')


def _equal_shards(x: Array, axis_name: str) -> Bool[Array, '']:
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
