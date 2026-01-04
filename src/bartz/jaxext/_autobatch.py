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
from functools import wraps
from warnings import warn

from jax import eval_shape, jit
from jax import numpy as jnp
from jax.lax import scan
from jax.tree import flatten as tree_flatten
from jax.tree import map as tree_map
from jax.tree import reduce as tree_reduce
from jaxtyping import PyTree


def expand_axes(axes, tree):
    """Expand `axes` such that they match the pytreedef of `tree`."""

    def expand_axis(axis, subtree):
        return tree_map(lambda _: axis, subtree)

    return tree_map(expand_axis, axes, tree, is_leaf=lambda x: x is None)


def check_no_nones(axes, tree):
    def check_not_none(_, axis):
        assert axis is not None

    tree_map(check_not_none, tree, axes)


def extract_size(axes, tree):
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


def batch(tree, nbatches):
    def batch(x):
        return x.reshape(nbatches, x.shape[0] // nbatches, *x.shape[1:])

    return tree_map(batch, tree)


def unbatch(tree):
    def unbatch(x):
        return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])

    return tree_map(unbatch, tree)


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
    return_nbatches: bool = False,
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

    Returns
    -------
    A function with the same signature as `func`, save for the return value if `return_nbatches`.
    """
    initial_in_axes = in_axes
    initial_out_axes = out_axes

    @jit
    @wraps(func)
    def batched_func(*args):
        example_result = eval_shape(func, *args)

        in_axes = expand_axes(initial_in_axes, args)
        out_axes = expand_axes(initial_out_axes, example_result)
        check_no_nones(out_axes, example_result)

        size = extract_size((in_axes, out_axes), (args, example_result))

        original_args = args
        args, nonbatched_args = pull_nonbatched(in_axes, args)

        total_nbytes = sum_nbytes((args, example_result))
        min_nbatches = total_nbytes // max_io_nbytes + bool(
            total_nbytes % max_io_nbytes
        )
        min_nbatches = max(1, min_nbatches)
        nbatches = next_divisor(size, min_nbatches)
        assert 1 <= nbatches <= max(1, size)
        assert size % nbatches == 0
        assert total_nbytes % nbatches == 0

        batch_nbytes = total_nbytes // nbatches
        if batch_nbytes > max_io_nbytes:
            assert size == nbatches
            msg = f'batch_nbytes = {batch_nbytes} > max_io_nbytes = {max_io_nbytes}'
            warn(msg)

        def loop(_, args):
            args = move_axes_in(in_axes, args)
            args = push_nonbatched(in_axes, args, nonbatched_args)
            result = func(*args)
            result = move_axes_out(out_axes, result)
            return None, result

        if nbatches > 1:
            args = move_axes_out(in_axes, args)
            args = batch(args, nbatches)
            _, result = scan(loop, None, args)
            result = unbatch(result)
            result = move_axes_in(out_axes, result)
        else:
            result = func(*original_args)

        check_same(example_result, result)

        if return_nbatches:
            return result, nbatches
        return result

    return batched_func
