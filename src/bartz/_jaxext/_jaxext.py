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

import inspect
import math
import os
import re
import sys
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial, wraps
from typing import Any, NamedTuple, ParamSpec, TypeVar

import jax
from equinox import tree_at
from jax import (
    Device,
    ShapeDtypeStruct,
    ensure_compile_time_eval,
    eval_shape,
    lax,
    random,
    shard_map,
    tree,
    vmap,
)
from jax import numpy as jnp
from jax.custom_batching import custom_vmap
from jax.dtypes import prng_key
from jax.extend.backend import backends
from jax.scipy.special import ndtr, ndtri
from jax.sharding import PartitionSpec
from jax.tree_util import keystr
from jax.typing import DTypeLike
from jaxtyping import (
    Array,
    Bool,
    Float32,
    Int32,
    Integer,
    Key,
    PyTree,
    Scalar,
    ScalarLike,
    Shaped,
    UInt,
)
from jaxtyping import config as jaxtyping_config

from bartz._jaxext._autobatch import vary, varying_axes
from bartz._jaxext._equinox import Module
from bartz._jaxext._jit import jit

if sys.version_info >= (3, 13):
    from typing import TypeIs
else:  # WORKAROUND(python<3.13): typing.TypeIs was added in 3.13
    from typing_extensions import TypeIs

# the private context manager sets the flag per thread, while the public
# `jax.config.update` sets it globally, so a context manager built on the
# latter would not be thread-safe
try:
    from jax._src.config import explicit_x64_dtypes
except ImportError:  # private jax API, used only as a safety net in `forbid_x64`

    def explicit_x64_dtypes(_mode: str) -> AbstractContextManager[None]:
        """Do nothing in place of the missing jax context manager."""
        return nullcontext()


_P = ParamSpec('_P')
_R = TypeVar('_R')
_T = TypeVar('_T')
C = TypeVar('C')
S = TypeVar('S')
X = TypeVar('X')
Y = TypeVar('Y')


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


def float32_matmuls(fun: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run/trace `fun` under full-float32 matmul precision."""

    @wraps(fun)
    def wrapper(*args: _P.args, **kw: _P.kwargs) -> _R:
        with jax.default_matmul_precision('float32'):
            return fun(*args, **kw)

    return wrapper


_X64_DTYPES = frozenset(
    map(jnp.dtype, (jnp.float64, jnp.int64, jnp.uint64, jnp.complex128))
)


def forbid_x64(fun: Callable[_P, _R]) -> Callable[_P, _R]:
    """Make `fun` raise an error on 64-bit types.

    Raises if ``jax_enable_x64`` is set or an argument contains a 64-bit jax
    array, and runs/traces `fun` with ``jax_explicit_x64_dtypes='error'``, so
    that jax raises on most operations involving 64-bit arrays. Combined with
    `jit`, put it inside, to keep the methods of the jitted function.

    User code called by `fun` can still use 64-bit types within
    ``jax.enable_x64(True)``; the error raised by jax gets a note saying so.
    """
    signature = inspect.signature(fun)

    @wraps(fun)
    def wrapper(*args: _P.args, **kw: _P.kwargs) -> _R:
        if jax.config.jax_enable_x64:
            msg = (
                'bartz does not support 64-bit types, but `jax_enable_x64` is '
                'set. Disable it, or call bartz within `with jax.enable_x64(False):`.'
            )
            raise RuntimeError(msg)
        for name, value in signature.bind(*args, **kw).arguments.items():
            for path, x in tree.leaves_with_path(value):
                if isinstance(x, jax.Array) and x.dtype in _X64_DTYPES:
                    msg = (
                        'bartz does not support 64-bit types, but '
                        f'`{name}{keystr(path)}` has dtype {x.dtype}. '
                        'Convert it to 32 bit.'
                    )
                    raise TypeError(msg)
        with explicit_x64_dtypes('error'):
            try:
                return fun(*args, **kw)
            except ValueError as exc:
                note = (
                    'bartz does not support 64-bit types. To use them in your '
                    'own code called by bartz, e.g., a callback, wrap it in '
                    '`with jax.enable_x64(True):` and convert the results to 32 '
                    'bit within the block.'
                )
                notes = getattr(exc, '__notes__', ())
                if 'Explicitly requested dtype' in str(exc) and note not in notes:
                    exc.add_note(note)
                raise

    return wrapper


def sliced_map(
    f: Callable[[PyTree[Array, ' I']], PyTree[Array, ' O']],
    xs: PyTree[Array, ' I'],
    *,
    batch_size: int,
) -> PyTree[Array, ' O']:
    """
    Like `jax.lax.map` with `batch_size`, but read `xs` by slicing.

    The scan body slices each batch out of the closed-over `xs` with
    `lax.dynamic_slice_in_dim` instead of consuming `xs` reshaped into scan xs.
    Under `vmap`, the batching rule for values closed over by a scan moves the
    batch axes to the front, so leading batch axes in the leaves of `xs` are
    consumed in place, while `lax.map` would transpose them to sit after the
    scanned and `batch_size` axes.

    Parameters
    ----------
    f
        The function to apply elementwise, taking and returning pytrees of
        arrays.
    xs
        A pytree of arrays to map over along their first axes.
    batch_size
        The number of elements to process at a time, vectorized. The remainder
        of the division of the number of elements by `batch_size` is processed
        as one additional smaller batch.

    Returns
    -------
    A pytree of stacked outputs of `f`, like `jax.lax.map`.
    """
    (num_el,) = {x.shape[0] for x in tree.leaves(xs)}

    # shortcut if no batching needed
    if batch_size >= num_el:
        return vmap(f)(xs)

    num_batches, remainder = divmod(num_el, batch_size)

    def apply_batch(start: Integer[Array, ''] | int, size: int) -> PyTree[Array, ' O']:
        def slice_batch(
            x: Shaped[Array, ' num_el *shape'],
        ) -> Shaped[Array, ' size *shape']:
            return lax.dynamic_slice_in_dim(x, start, size, axis=0)

        return vmap(f)(tree.map(slice_batch, xs))

    def loop(_: None, start: Integer[Array, '']) -> tuple[None, PyTree[Array, ' O']]:
        return None, apply_batch(start, batch_size)

    _, out = lax.scan(loop, None, batch_size * jnp.arange(num_batches))
    out = tree.map(lambda x: x.reshape(num_batches * batch_size, *x.shape[2:]), out)
    if remainder:
        start = num_batches * batch_size
        # WORKAROUND(jax<0.12.1): with more cpu devices, a program loaded from
        # the compilation cache may run the collectives in the loop and in the
        # remainder at the same time, and crash. The barrier does not order
        # them, it only changes what xla compiles, to a program that did not
        # crash in the cases tried.
        # WORKAROUND(jax<0.11.1): the outputs of a barrier vary over the manual
        # axes of all its operands, which changes the type of `out`, so do
        # without; once unsupported, drop the condition
        if jax.__version_info__ >= (0, 11, 1):
            out, start = lax.optimization_barrier((out, start))
        rest = apply_batch(start, remainder)
        out = tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), out, rest)
    return out


def minimal_unsigned_dtype(value: int) -> DTypeLike:
    """Return the smallest unsigned integer dtype that can represent `value`."""
    if value < 2**8:
        return jnp.uint8
    if value < 2**16:
        return jnp.uint16
    if value < 2**32:
        return jnp.uint32
    return jnp.uint64


def minimal_unsigned_arange(size: int) -> UInt[Array, ' {size}']:
    """Return the integers below `size` in the smallest unsigned integer dtype."""
    return jnp.arange(size, dtype=minimal_unsigned_dtype(size - 1))


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

    carry = jnp.array(0), x[0], jnp.full(size, fill_value, x.dtype)

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
        Whether to clip the samples to keep them finite and within the
        truncation region. Leave on, off only for debugging.

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
        # `truncated_u` can reach 0 or 1, where ndtri is infinite: ndtr(±bound)
        # underflows for extreme bounds, u can come out exactly 0, and on gpu the
        # accuracy is lower. The lower target is the smallest normal number rather
        # than the smallest subnormal one because xla flushes subnormals to zero on
        # cpu, and ndtri is -inf on subnormals anyway.
        zero = jnp.zeros((), truncated_u.dtype)
        one = jnp.ones((), truncated_u.dtype)
        smallest_normal = jnp.finfo(truncated_u.dtype).smallest_normal
        truncated_u = jnp.clip(truncated_u, smallest_normal, jnp.nextafter(one, zero))
    truncated_norm = ndtri(truncated_u)
    sample = jnp.where(bound_pos, -truncated_norm, truncated_norm)
    if clip:
        # the clip above caps |sample| at ndtri of the target, ~12.9 in float32,
        # which for extreme bounds falls short of the truncation region, and
        # rounding can put the sample a ulp on the wrong side anyway. Move it onto
        # the boundary, where the distribution concentrates for such bounds (its
        # mean is bound + 1/bound).
        sample = jnp.where(
            upper, jnp.minimum(sample, bound), jnp.maximum(sample, bound)
        )
    return sample


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


def get_cpu_thread_count() -> int | None:
    """Get the number of threads xla uses on cpu, `None` without a cpu backend.

    Xla does not expose the count, so this replicates how xla sets it.
    """
    # the first of these environment variables that is an integer replaces the
    # number of cpus available to the process
    env = (os.environ.get(name, '') for name in ('PJRT_NPROC', 'NPROC'))
    env = [int(value) for value in env if re.fullmatch(r'\s*[+-]?\d+\s*', value)]
    if env:
        count = env[0]
    elif hasattr(os, 'sched_getaffinity'):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count() or 1
    if 'cpu' in backends():
        # there is at least one thread per device
        return max(count, jax.local_device_count('cpu'))
    else:
        return None


def is_key(x: object) -> TypeIs[Key[Array, ' *shape']]:
    """Determine if `x` is a jax random key."""
    return isinstance(x, Array) and jnp.issubdtype(x.dtype, prng_key)


def jit_active() -> bool:
    """Check if we are under jit."""
    return not hasattr(jnp.empty(0), 'platform')


def cpu_barrier(x: _T) -> _T:
    """Apply `lax.optimization_barrier` on cpu only, to steer xla:cpu fusion."""
    return lax.platform_dependent(x, cpu=lax.optimization_barrier, default=lambda x: x)


@custom_vmap
def vmap_max(x: Integer[Array, '']) -> Integer[Array, '']:
    """Take the maximum over the axes of the enclosing vmaps, if any.

    Parameters
    ----------
    x
        A non-negative integer.

    Returns
    -------
    The maximum of `x` over the batch, `x` itself if not vmapped, 0 if the batch is empty.

    Notes
    -----
    This is a workaround for collectives over vmap named axes not working under
    `jax.shard_map`, see jax issue #42649. Used by `variable_length_scan` and
    `variable_length_update`.
    """
    return x


@vmap_max.def_vmap
def vmap_max_rule(
    axis_size: int,  # noqa: ARG001
    in_batched: Sequence[bool],
    x: Integer[Array, ' batch'],
) -> tuple[Integer[Array, ''], bool]:
    # jax calls the rule only if some argument is batched, here the only one
    (batched,) = in_batched
    assert batched
    # the max is batched as usual by any outer vmap, so apply again to reduce it
    return vmap_max(jnp.max(x, axis=0, initial=0)), False


class InPlace(Module):
    """An element of a batched array, that reads the whole array when indexed.

    It supports only indexing, see `sequential_vmap`.
    """

    array: Shaped[Array, ' batch *shape']
    """The whole batched array."""

    index: Int32[Array, '']
    """The index of the element along the leading axis of `array`."""

    def __getitem__(self, key: object) -> Shaped[Array, '...']:
        key = key if isinstance(key, tuple) else (key,)
        return self.array[(self.index, *key)]


# the arguments are not typed with a `ParamSpec`, because jax passes them to the
# batching rule as an untyped tuple
def sequential_vmap(
    f: Callable[..., _R],
    *,
    in_place: Callable[..., tuple[Array, ...] | Array] | None = None,
) -> Callable[..., _R]:
    """
    Like `jax.custom_batching.sequential_vmap`, but with some arguments read in place.

    Parameters
    ----------
    f
        The function to apply, with positional arguments only.
    in_place
        Takes the arguments of `f` and returns the arrays, or a tuple of
        arrays, to read in place. "In place" means that instead of slicing the
        array to pass it to `f`, the full unsliced array is wrapped as an
        `InPlace` object that's passed to `f`.

    Returns
    -------
    `f`, with a sequential batching rule.

    Notes
    -----
    The purpose of this function is to avoid redundant copies of arrays in the
    case where an argument to `f` is passed into a loop (e.g.,
    `lax.while_loop`) inside `f`, which forces the materialization of the slice
    of the array.
    """
    g = custom_vmap(f)

    @g.def_vmap
    def rule(
        axis_size: int, in_batched: Sequence[PyTree[bool]], *args: Any
    ) -> tuple[_R, PyTree[bool]]:
        in_batched = tuple(in_batched)
        if axis_size == 0:
            # shortcut because `InPlace` does not support empty batch axis, so
            # `lax.map`'s own shortcut would not work on our wrapped function
            in_axes = tree.map(lambda batched: 0 if batched else None, in_batched)
            out = vmap(f, in_axes=in_axes)(*args)
            return out, tree.map(lambda _: True, out)

        def wrapped_in_place(args: tuple) -> PyTree:
            # `tree_at` passes the arguments as one tuple, `in_place` takes
            # them apart like `f`
            assert in_place is not None
            return in_place(*args)

        if in_place is not None:
            # the arrays read in place are not sliced
            in_place_batched = wrapped_in_place(in_batched)
            in_batched = tree_at(
                wrapped_in_place, in_batched, replace_fn=lambda _: False
            )

        # flatten everything and extract vmapped arguments
        args_leaves, args_treedef = tree.flatten(args)
        flat_in_batched = tree.leaves(in_batched)
        flat_vmapped_args = [
            x
            for x, batched in zip(args_leaves, flat_in_batched, strict=True)
            if batched
        ]

        def apply(
            index_and_vmapped_args: tuple[Int32[Array, ''], Sequence[Array]],
        ) -> _R:
            """Apply `InPlace` where needed, then call `f`."""
            index, flat_vmapped_args_at = index_and_vmapped_args

            # re-insert unvmapped arguments and unflatten
            it = iter(flat_vmapped_args_at)
            args_at = tree.unflatten(
                args_treedef,
                [
                    next(it) if batched else x
                    for x, batched in zip(args_leaves, flat_in_batched, strict=True)
                ],
            )

            # apply `InPlace` where `in_place` indicates to do so
            if in_place is not None:
                views = tree.map(
                    lambda x, batched: InPlace(x, index) if batched else x,
                    wrapped_in_place(args),
                    in_place_batched,
                )
                args_at = tree_at(wrapped_in_place, args_at, views)

            # call `f`
            return f(*args_at)

        out = lax.map(apply, (jnp.arange(axis_size), flat_vmapped_args))
        return out, tree.map(lambda _: True, out)

    return g


def variable_length_scan(
    f: Callable[[C, X, Bool[Array, ''] | None], tuple[C, Y]],
    init: C,
    xs: X,
    length: Integer[Array, ''] | None = None,
    *,
    block_size: int = 1,
    unroll: int | bool = 1,
) -> tuple[C, Y]:
    """
    Like `jax.lax.scan`, but only over the first `length` elements of `xs`.

    Parameters
    ----------
    f
        Like the function scanned by `jax.lax.scan`, but with a third argument
        that says whether the element is within the first `length`. If not,
        `f` shall return the carry unchanged, and its other output is replaced
        by zeros. `variable_length_scan` does not undo the carry update itself
        with a select over the whole carry, because `f` may have a cheaper way,
        like zeroing a delta before adding it. The argument is `None` if
        `length` is.
    init
        The initial carry.
    xs
        The elements to scan over, along the leading axis.
    length
        How many elements to scan over (can be larger than the actual length of
        `xs`, the excess will be skipped). If `None`, all of them, with a loop
        of fixed length. Under `jax.vmap`, if `length` is batched, the loop
        runs up to the maximum `length` in the batch, to keep the loop
        condition unbatched.
    block_size
        If `length` is set, the loop condition is evaluated every `block_size`
        iterations instead of at every iteration. The purpose of this setting
        is avoiding overhead on gpu due to the condition requiring a host trip.
        If `block_size` is not a divisor of `length`, the remainder iterations
        in the last block are run anyway, to avoid double compilation of the
        loop body.
    unroll
        How much to unroll the loop (within each block), passed to `jax.lax.scan`.

    Returns
    -------
    carry : C
        The final carry.
    ys : Y
        The stacked outputs of `f`, zero past `length`.
    """
    # extract length of `xs`
    (len_x,) = {x.shape[0] for x in tree.leaves(xs)}

    # shortcut through `lax.scan`
    if length is None or len_x == 0:
        return lax.scan(lambda carry, x: f(carry, x, None), init, xs, unroll=unroll)

    # clamp the lengths to the number of elements
    length = jnp.minimum(length, len_x)
    block_size = max(1, min(block_size, len_x))

    # compute number of blocks (in the output, not effectively looped over)
    num_blocks_out = -(-len_x // block_size)

    def scan_block(i: Integer[Array, ''] | int, carry: C) -> tuple[C, Y]:
        """Do `block_size` iterations."""

        def step(carry: C, j: Int32[Array, '']) -> tuple[C, Y]:
            """Do a single iteration."""
            # the elements are indexed out of the closed-over `xs`, instead of
            # passing the block as scan xs, such that the block is not copied;
            # the index is clamped in the last block, past `length` anyway
            position = i * block_size + j
            x = tree.map(lambda x: dynamic_slice_in_dim(x, position, 1).squeeze(0), xs)
            valid = position < length
            carry, y = f(carry, x, valid)
            # zero the outputs of the elements past `length`, like those of the
            # blocks never visited, such that they do not depend on
            # `block_size`, or on the other lengths in a vmap batch
            return carry, tree.map(lambda y: jnp.where(valid, y, jnp.zeros_like(y)), y)

        return lax.scan(step, carry, jnp.arange(block_size), unroll=unroll)

    # Allocate `ys`, the output is written in it block by block. Initialize to
    # zero because we promise the unused part of the tail to be zero. Match the
    # sharding of `y` to make the initial loop carry type match the running
    # loop carry type.
    _, dummy_y_block = eval_shape(scan_block, 0, init)
    ys = tree.map(
        lambda y: zeros_varying_like(y, (num_blocks_out, *y.shape)), dummy_y_block
    )

    # compute number of blocks to loop over
    num_blocks_loop = (vmap_max(length) + block_size - 1) // block_size

    def cond(state: BlocksLoopState) -> Bool[Array, '']:
        """Return whether the loop over blocks shall continue."""
        return state.i_block < num_blocks_loop

    def body(state: BlocksLoopState) -> BlocksLoopState:
        """Process one block."""
        carry, y_block = scan_block(state.i_block, state.carry)
        ys = tree.map(
            lambda ys, y: dynamic_update_slice_in_dim(ys, y[None, ...], state.i_block),
            state.ys,
            y_block,
        )
        return BlocksLoopState(state.i_block + 1, carry, ys)

    # loop over blocks, squash block axis from the output and return
    state = lax.while_loop(cond, body, BlocksLoopState(jnp.int32(0), init, ys))
    ys = tree.map(
        lambda y: y.reshape(num_blocks_out * block_size, *y.shape[2:])[:len_x, ...],
        state.ys,
    )
    return state.carry, ys


def dynamic_slice_in_dim(
    x: Shaped[Array, '...'], start: Integer[Array, ''] | int, size: int, axis: int = 0
) -> Shaped[Array, '...']:
    """
    Like `jax.lax.dynamic_slice_in_dim`, but still a slice under `jax.vmap`.

    If `start` is not batched, the batching rule slices the batched `x` along
    the shifted `axis`, while the one of `jax.lax.dynamic_slice_in_dim` turns
    the slice into a gather.

    Parameters
    ----------
    x
        The array to slice.
    start
        The index of the first element of the slice.
    size
        The length of the slice.
    axis
        The (non-negative) axis to slice along.

    Returns
    -------
    The slice ``x[..., start : start + size, ...]``.
    """

    def lax_take(
        x: Shaped[Array, '...'], start: Integer[Array, ''] | int
    ) -> Shaped[Array, '...']:
        return lax.dynamic_slice_in_dim(x, start, size, axis)

    take = custom_vmap(lax_take)

    @take.def_vmap
    def rule(
        axis_size: int,
        in_batched: Sequence[bool],
        x: Shaped[Array, '...'],
        start: Integer[Array, '...'] | int,  # jax passes a constant as a python scalar
    ) -> tuple[Shaped[Array, '...'], bool]:
        del axis_size

        # jax moves the batch axis of the batched arguments to the front, and
        # expects it there in the output
        x_batched, start_batched = in_batched
        if start_batched:
            in_axes = (0 if x_batched else None, 0)
            return vmap(lax_take, in_axes)(x, start), True
        else:
            return dynamic_slice_in_dim(x, start, size, axis + 1), True

    return take(x, start)


def dynamic_update_slice_in_dim(
    x: Shaped[Array, '...'],
    update: Shaped[Array, '...'],
    start: Integer[Array, ''] | int,
    axis: int = 0,
) -> Shaped[Array, '...']:
    """
    Like `jax.lax.dynamic_update_slice_in_dim`, but still a slice update under `jax.vmap`.

    If `start` is not batched, the batching rule updates the batched `x` along
    the shifted `axis`, while the one of `jax.lax.dynamic_update_slice_in_dim`
    turns the update into a scatter, that xla does not do in place.

    Parameters
    ----------
    x
        The array to update.
    update
        The values to write, with the shape of `x` but for the length along
        `axis`.
    start
        The index of the first element to overwrite.
    axis
        The (non-negative) axis to update along.

    Returns
    -------
    `x` with ``x[..., start : start + update.shape[axis], ...]`` set to `update`.
    """

    def lax_put(
        x: Shaped[Array, '...'],
        update: Shaped[Array, '...'],
        start: Integer[Array, ''] | int,
    ) -> Shaped[Array, '...']:
        return lax.dynamic_update_slice_in_dim(x, update, start, axis)

    put = custom_vmap(lax_put)

    @put.def_vmap
    def rule(
        axis_size: int,
        in_batched: Sequence[bool],
        x: Shaped[Array, '...'],
        update: Shaped[Array, '...'],
        start: Integer[Array, '...'] | int,
    ) -> tuple[Shaped[Array, '...'], bool]:
        x_batched, update_batched, start_batched = in_batched
        if start_batched:
            in_axes = (0 if x_batched else None, 0 if update_batched else None, 0)
            return vmap(lax_put, in_axes)(x, update, start), True
        else:
            if not x_batched:
                x = jnp.broadcast_to(x, (axis_size, *x.shape))
            if not update_batched:
                update = jnp.broadcast_to(update, (axis_size, *update.shape))
            return dynamic_update_slice_in_dim(x, update, start, axis + 1), True

    return put(x, update, start)


def variable_length_update(
    update: Callable[[S, X], S],
    post: Callable[[S, X], Y],
    state: S,
    xs: X,
    length: Integer[Array, ''] | None = None,
    *,
    block_size: int | None = None,
) -> tuple[S, Y]:
    """
    Update the first `length` elements of arrays, one block at a time.

    The semantics are given by this Python implementation, with numpy arrays
    in place of the pytrees of jax arrays::

        def variable_length_update(update, post, state, xs, length, block_size):
            ys = np.zeros_like(post(state, xs))
            for start in range(0, length, block_size):
                block = slice(start, start + block_size)
                state[block] = update(state[block], xs[block])
                ys[block] = post(state[block], xs[block])
            ys[length:] = 0
            return state, ys

    The number of iterations is set at run time, so the cost scales with
    `length` while the arrays keep a fixed size, and `state` is overwritten
    block by block without copying it.

    Parameters
    ----------
    update
        Computes the new `state`. It shall be vectorized along the leading
        axis, such that the result does not depend on the blocks.
    post
        Computes other outputs from the new `state`, vectorized like `update`.
    state
        The arrays to update.
    xs
        The other inputs of `update` and `post`.
    length
        How many elements to process. If `None`, all of them, at once. Under
        `jax.vmap`, the loop runs up to the maximum `length` in the batch.
    block_size
        The number of elements in a block, a divisor of the number of
        elements. If `None`, all of them: a single block is processed whole
        without a loop, whatever the `length`.

    Returns
    -------
    state : S
        The updated `state`, left as it was in the blocks not processed.
    ys : Y
        The outputs of `post`, zero past `length`.

    Raises
    ------
    ValueError
        If `block_size` is not a divisor of the number of elements.

    Notes
    -----
    `post` exists, instead of `update` returning the other outputs too, because
    xla writes a block back over the old one in place only if nothing else
    reads the new block before; `post` reads it after, out of the whole array.

    If `block_size` is equal to the number of elements (one block) then
    everything is processed even in the ``length == 0`` case.
    """
    (num_el,) = {x.shape[0] for x in tree.leaves((state, xs))}
    if block_size is None:
        block_size = num_el
    if block_size < 1 or num_el % block_size:
        msg = f'{block_size=} is not a divisor of the {num_el} elements'
        raise ValueError(msg)

    if length is None or block_size == num_el:
        # all the elements at once, a single block needs no loop
        state = update(state, xs)
        ys = post(state, xs)
    else:
        state, ys = update_blocks(
            update, post, state, xs, length, block_size=block_size
        )

    if length is not None:
        # zero the outputs past `length`: the last block processed computes
        # them and the blocks after it do not, so they would depend on
        # `block_size`, and on the other lengths in a vmap batch
        valid = jnp.arange(num_el) < length
        ys = tree.map(
            lambda y: jnp.where(
                valid.reshape(-1, *[1] * (y.ndim - 1)), y, jnp.zeros_like(y)
            ),
            ys,
        )

    return state, ys


def update_blocks(
    update: Callable[[S, X], S],
    post: Callable[[S, X], Y],
    state: S,
    xs: X,
    length: Integer[Array, ''],
    *,
    block_size: int,
) -> tuple[S, Y]:
    """Implement `variable_length_update` over more than one block."""
    (num_el,) = {x.shape[0] for x in tree.leaves((state, xs))}

    # Allocate `ys`, the loop writes the outputs in it block by block. Zeros,
    # to define the outputs of the blocks not processed. They vary over the
    # same mesh axes as the outputs of `post`, for the initial loop carry to
    # have the type of the running one.
    dummy_ys = eval_shape(lambda: post(update(state, xs), xs))
    ys = tree.map(lambda y: zeros_varying_like(y, y.shape), dummy_ys)

    # the blocks to process, up to the maximum `length` in a vmap batch to
    # keep the loop condition unbatched
    num_blocks = -(-jnp.clip(vmap_max(length), 0, num_el) // block_size)

    def cond(loop: BlocksLoopState) -> Bool[Array, '']:
        return loop.i_block < num_blocks

    def body(loop: BlocksLoopState) -> BlocksLoopState:
        start = loop.i_block * block_size

        def take(
            x: Shaped[Array, ' num_el *shape'],
        ) -> Shaped[Array, ' block_size *shape']:
            return dynamic_slice_in_dim(x, start, block_size)

        def put(
            x: Shaped[Array, ' num_el *shape'],
            block: Shaped[Array, ' block_size *shape'],
        ) -> Shaped[Array, ' num_el *shape']:
            return dynamic_update_slice_in_dim(x, block, start)

        xs_block = tree.map(take, xs)
        state = update(tree.map(take, loop.carry), xs_block)
        state = tree.map(put, loop.carry, state)
        # `post` gets the new block sliced back out of `state`, not the output
        # of `update`: read by something else first, xla would not write the
        # block in place
        ys = tree.map(put, loop.ys, post(tree.map(take, state), xs_block))
        return BlocksLoopState(loop.i_block + 1, state, ys)

    loop = lax.while_loop(cond, body, BlocksLoopState(jnp.int32(0), state, ys))
    return loop.carry, loop.ys


# WORKAROUND(beartype<=0.23.1): beartype rejects generic NamedTuple hints
# (beartype#748), so `carry` and `ys` are typed as any pytree. Once fixed, make
# this `NamedTuple, Generic[C, Y]` and annotate `cond` and `body` in
# `variable_length_scan` with `BlocksLoopState[C, Y]`.
class BlocksLoopState(NamedTuple):
    """The carry of the loop over blocks in `variable_length_scan` and `variable_length_update`."""

    i_block: Int32[Array, '']
    """The index of the next block."""

    carry: PyTree
    """The carry of the scan, or the arrays updated in place."""

    ys: PyTree
    """The stacked outputs, with an axis over blocks in `variable_length_scan`."""


def zeros_varying_like(
    x: Shaped[Array, '*any'] | ShapeDtypeStruct, shape: tuple[int, ...]
) -> Shaped[Array, '*shape']:
    """Make zeros with the dtype of `x`, varying over the same mesh axes as `x`."""
    zeros = jnp.zeros(shape, x.dtype)
    axes = tuple(varying_axes(x))
    return vary(zeros, axes) if axes else zeros


def _equal_shards(x: Shaped[Array, '...'], axis_name: str) -> Bool[Array, '']:
    """Check if all shards of `x` are equal, to be used in a `shard_map` context."""
    size = lax.axis_size(axis_name)
    perm = [(i, (i + 1) % size) for i in range(size)]
    perm_x = lax.ppermute(x, axis_name, perm)
    diff = jnp.any(x != perm_x)
    return jnp.logical_not(lax.psum(diff, axis_name))


@jit(static_argnames=('axis_name', 'in_specs', 'mesh', 'axis_names', 'check_vma'))
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
