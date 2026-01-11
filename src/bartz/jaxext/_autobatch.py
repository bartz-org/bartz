# bartz/src/bartz/jaxext/_autobatch.py
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

"""Implementation of `autobatch`."""

import math
from collections.abc import Callable
from functools import partial, wraps
from warnings import warn

from jax.typing import DTypeLike

try:
    from numpy.lib.array_utils import normalize_axis_index  # numpy 2
except ImportError:
    from numpy.core.numeric import normalize_axis_index  # numpy 1

from jax import ShapeDtypeStruct, eval_shape, jit
from jax import numpy as jnp
from jax.lax import scan
from jax.tree import flatten as tree_flatten
from jax.tree import map as tree_map
from jax.tree import reduce as tree_reduce
from jaxtyping import Array, PyTree, Shaped


def expand_axes(axes, tree):
    """Expand `axes` such that they match the pytreedef of `tree`."""

    def expand_axis(axis, subtree):
        return tree_map(lambda _: axis, subtree)

    return tree_map(expand_axis, axes, tree, is_leaf=lambda x: x is None)


def normalize_axes(
    axes: PyTree[int | None, ' T'], tree: PyTree[Array, ' T']
) -> PyTree[int | None, ' T']:
    """Normalize axes to be non-negative and valid for the corresponding arrays in the tree."""

    def normalize_axis(axis: int | None, x: Array) -> int | None:
        if axis is None:
            return None
        else:
            return normalize_axis_index(axis, len(x.shape))

    return tree_map(normalize_axis, axes, tree, is_leaf=lambda x: x is None)


def check_no_nones(axes, tree):
    def check_not_none(_, axis):
        assert axis is not None

    tree_map(check_not_none, tree, axes, is_leaf=lambda x: x is None)


def remove_axis(
    x: PyTree[ShapeDtypeStruct, ' T'], axis: PyTree[int, ' T'], ufunc: jnp.ufunc
) -> PyTree[ShapeDtypeStruct, ' T']:
    """Remove an axis from dummy arrays and change the type to reduction type."""

    def remove_axis(x: ShapeDtypeStruct, axis: int) -> ShapeDtypeStruct:
        new_shape = x.shape[:axis] + x.shape[axis + 1 :]
        new_dtype = reduction_dtype(ufunc, x.dtype)
        return ShapeDtypeStruct(new_shape, new_dtype)

    return tree_map(remove_axis, x, axis)


def extract_size(axes, tree):
    """Get the size of each array in tree at the axis in axes, check they are equal and return it."""

    def get_size(x, axis):
        if axis is None:
            return None
        else:
            return x.shape[axis]

    sizes = tree_map(get_size, tree, axes)
    sizes, _ = tree_flatten(sizes)
    assert all(s == sizes[0] for s in sizes)
    return sizes[0]


def sum_nbytes(tree):
    def nbytes(x):
        return math.prod(x.shape) * x.dtype.itemsize

    return tree_reduce(lambda size, x: size + nbytes(x), tree, 0)


def next_divisor_small(dividend, min_divisor):
    for divisor in range(min_divisor, int(math.sqrt(dividend)) + 1):
        if dividend % divisor == 0:
            return divisor
    return dividend


def next_divisor_large(dividend, min_divisor):
    max_inv_divisor = dividend // min_divisor
    for inv_divisor in range(max_inv_divisor, 0, -1):
        if dividend % inv_divisor == 0:
            return dividend // inv_divisor
    return dividend


def next_divisor(dividend, min_divisor):
    """Return divisor >= min_divisor such that divided % divisor == 0."""
    if dividend == 0:
        return min_divisor
    if min_divisor * min_divisor <= dividend:
        return next_divisor_small(dividend, min_divisor)
    return next_divisor_large(dividend, min_divisor)


def pull_nonbatched(axes, tree):
    def pull_nonbatched(x, axis):
        if axis is None:
            return None
        else:
            return x

    return tree_map(pull_nonbatched, tree, axes), tree


def push_nonbatched(axes, tree, original_tree):
    def push_nonbatched(original_x, x, axis):
        if axis is None:
            return original_x
        else:
            return x

    return tree_map(push_nonbatched, original_tree, tree, axes)


def move_axes_out(axes, tree):
    def move_axis_out(x, axis):
        return jnp.moveaxis(x, axis, 0)

    return tree_map(move_axis_out, tree, axes)


def move_axes_in(axes, tree):
    def move_axis_in(x, axis):
        return jnp.moveaxis(x, 0, axis)

    return tree_map(move_axis_in, tree, axes)


def batch(tree: PyTree[Array, ' T'], nbatches: int) -> PyTree[Array, ' T']:
    """Split the first axis into two axes, the first of size `nbatches`."""

    def batch(x):
        return x.reshape(nbatches, x.shape[0] // nbatches, *x.shape[1:])

    return tree_map(batch, tree)


def unbatch(tree: PyTree[Array, ' T']) -> PyTree[Array, ' T']:
    """Merge the first two axes into a single axis."""

    def unbatch(x):
        return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])

    return tree_map(unbatch, tree)


def reduce(
    ufunc: jnp.ufunc,
    x: PyTree[Array, ' T'],
    axes: PyTree[int, ' T'],
    initial: PyTree[Array, ' T'] | None,
) -> PyTree[Array, ' T']:
    """Reduce each array in `x` along the axes in `axes` starting from `initial` using `ufunc.reduce`."""
    if initial is None:

        def reduce(x: Array, axis: int) -> Array:
            return ufunc.reduce(x, axis=axis)

        return tree_map(reduce, x, axes)

    else:

        def reduce(x: Array, initial: Array, axis: int) -> Array:
            reduced = ufunc.reduce(x, axis=axis)
            return ufunc(initial, reduced)

        return tree_map(reduce, x, initial, axes)


def identity(
    ufunc: jnp.ufunc, x: PyTree[ShapeDtypeStruct, ' T']
) -> PyTree[ShapeDtypeStruct, ' T']:
    """Get the identity element for `ufunc` and each array in `x`."""

    def identity(x: ShapeDtypeStruct) -> Array:
        identity = identity_for(ufunc, x.dtype)
        return jnp.broadcast_to(identity, x.shape)

    return tree_map(identity, x)


def reduction_dtype(ufunc: jnp.ufunc, input_dtype: DTypeLike) -> DTypeLike:
    """Return the output dtype for a reduction with `ufunc` on inputs of type `dtype`."""
    return ufunc.reduce(jnp.empty(1, input_dtype)).dtype


def identity_for(ufunc: jnp.ufunc, input_dtype: DTypeLike) -> Shaped[Array, '']:
    """Return the identity for ufunc, filling in missing cases."""
    # get output type from input type, e.g., int8 is accumulated to int32
    dtype = reduction_dtype(ufunc, input_dtype)

    # return as explicitly typed array
    return jnp.array(ufunc.identity, dtype)


def check_same(tree1, tree2):
    def check_same(x1, x2):
        assert x1.shape == x2.shape
        assert x1.dtype == x2.dtype

    tree_map(check_same, tree1, tree2)


def autobatch(
    func: Callable,
    max_io_nbytes: int,
    in_axes: PyTree[int | None] = 0,
    out_axes: PyTree[int] = 0,
    *,
    return_nbatches: bool = False,
    reduce_ufunc: jnp.ufunc | None = None,
) -> Callable:
    """
    Batch a function such that each batch is smaller than a threshold.

    Parameters
    ----------
    func
        A jittable function with positional arguments only, with inputs and
        outputs pytrees of arrays.
    max_io_nbytes
        The maximum number of input + output bytes in each batch (excluding
        unbatched arguments.)
    in_axes
        A tree matching (a prefix of) the structure of the function input,
        indicating along which axes each array should be batched. A `None` axis
        indicates to not batch an argument.
    out_axes
        The same for outputs (but non-batching is not allowed).
    return_nbatches
        If True, the number of batches is returned as a second output.
    reduce_ufunc
        Function used to reduce the output along the batched axis (e.g.,
        `jax.numpy.add`).

    Returns
    -------
    A function with the same signature as `func`, save for the return value if `return_nbatches`.

    Notes
    -----
    Unless `return_nbatches` or `reduce_ufunc` are set, `autobatch` at given
    arguments is idempotent. Furthermore, `autobatch` can be applied multiple
    times over multiple axes with the same `max_io_nbytes` limit to work on
    multiple axes; in this case it won't unnecessarily loop over additional axes
    if one or more outer `autobatch` are already sufficient.

    To handle memory used in intermediate values: assuming all intermediate
    values have size that scales linearly with the axis batched over, say the
    batched input/output total size is ``batched_size * core_io_size``, and the
    intermediate values have size ``batched_size * core_int_size``, then to take
    them into account divide `max_io_nbytes` by ``(1 + core_int_size /
    core_io_size)``.
    """

    @jit
    @wraps(func)
    def autobatch_wrapper(*args):
        return batched_func(
            func, max_io_nbytes, in_axes, out_axes, return_nbatches, reduce_ufunc, args
        )

    return autobatch_wrapper


def batched_func(
    func: Callable,
    max_io_nbytes: int,
    in_axes: PyTree[int | None],
    out_axes: PyTree[int],
    return_nbatches: bool,
    reduce_ufunc: jnp.ufunc | None,
    args: tuple[PyTree[Array], ...],
) -> PyTree[Array]:
    """Implement the wrapper used in `autobatch`."""
    # determine the output structure of the function
    example_result = eval_shape(func, *args)

    # expand the axes pytrees if they are prefixes
    in_axes = expand_axes(in_axes, args)
    out_axes = expand_axes(out_axes, example_result)
    check_no_nones(out_axes, example_result)

    # check the axes are valid
    in_axes = normalize_axes(in_axes, args)
    out_axes = normalize_axes(out_axes, example_result)

    # get the size of the batched axis
    size = extract_size((in_axes, out_axes), (args, example_result))

    # split arguments in batched and not batched
    original_args = args
    args, nonbatched_args = pull_nonbatched(in_axes, args)

    # determine the number of batches to respect the memory limit
    total_nbytes = sum_nbytes((args, example_result))
    min_nbatches = total_nbytes // max_io_nbytes + bool(total_nbytes % max_io_nbytes)
    min_nbatches = max(1, min_nbatches)
    nbatches = next_divisor(size, min_nbatches)
    assert 1 <= nbatches <= max(1, size)
    assert size % nbatches == 0
    assert total_nbytes % nbatches == 0

    # warn if the memory limit could not be respected
    batch_nbytes = total_nbytes // nbatches
    if batch_nbytes > max_io_nbytes:
        assert size == nbatches
        msg = f'batch_nbytes = {batch_nbytes} > max_io_nbytes = {max_io_nbytes}'
        warn(msg)

    # squeeze out the output dims that will be reduced
    if reduce_ufunc is not None:
        example_result = remove_axis(example_result, out_axes, reduce_ufunc)

    if nbatches > 1:
        # prepare arguments for looping
        args = move_axes_out(in_axes, args)
        args = batch(args, nbatches)

        # prepare carry for reduction
        if reduce_ufunc is None:
            initial = None
        else:
            initial = identity(reduce_ufunc, example_result)

        # loop and invoke the function in batches
        loop = partial(
            batching_loop,
            func=func,
            nonbatched_args=nonbatched_args,
            in_axes=in_axes,
            out_axes=out_axes,
            reduce_ufunc=reduce_ufunc,
        )
        reduced_result, result = scan(loop, initial, args)

        # remove auxiliary batching axis and reverse transposition
        if reduce_ufunc is None:
            assert reduced_result is None
            result = unbatch(result)
            result = move_axes_in(out_axes, result)
        else:
            assert result is None
            result = reduced_result

    # trivial case: no batching needed
    else:
        result = func(*original_args)
        if reduce_ufunc is not None:
            result = reduce(reduce_ufunc, result, out_axes, None)

    check_same(example_result, result)

    if return_nbatches:
        return result, nbatches
    return result


def batching_loop(
    initial, args, *, func, nonbatched_args, in_axes, out_axes, reduce_ufunc
):
    """Implement the batching loop in `autobatch`."""
    # evaluate the function
    args = move_axes_in(in_axes, args)
    args = push_nonbatched(in_axes, args, nonbatched_args)
    result = func(*args)

    # unreduced case: transpose for concatenation and return
    if reduce_ufunc is None:
        result = move_axes_out(out_axes, result)
        return None, result

    # reduced case: reduce starting from initial
    else:
        reduced_result = reduce(reduce_ufunc, result, out_axes, initial)
        return reduced_result, None
