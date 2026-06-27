# bartz/src/bartz/_jaxext/scipy/special.py
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

"""Mockup of the :external:py:mod:`scipy.special` module."""

from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
from jax import ShapeDtypeStruct, pure_callback
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, ArrayLike, Float, Shaped
from scipy.special import gammainccinv as scipy_gammainccinv

from bartz._jaxext._jit import jit


def _float_type(*args: DTypeLike | Shaped[ArrayLike, '...']) -> jnp.dtype:
    """Determine the jax floating point result type given operands/types."""
    t = jnp.result_type(*args)
    return jnp.sin(jnp.empty(0, t)).dtype


def _castto(
    func: Callable[..., np.ndarray], dtype: DTypeLike
) -> Callable[..., np.ndarray]:
    # `func` is a host (numpy) routine wrapped for `jax.pure_callback`, so the
    # callback returns a numpy value. `np.asarray` normalizes the scalar case
    # (numpy returns a 0-d scalar, not an `ndarray`, for scalar inputs).
    @wraps(func)
    def newfunc(*args: Any, **kw: Any) -> Shaped[np.ndarray, '...']:
        return np.asarray(func(*args, **kw), dtype)

    return newfunc


@jit
def gammainccinv(
    a: Float[ArrayLike, '...'], y: Float[ArrayLike, '...']
) -> Float[Array, '...']:
    """Survival function inverse of the Gamma(a, 1) distribution."""
    shape = jnp.broadcast_shapes(jnp.shape(a), jnp.shape(y))
    dtype = _float_type(a, y)
    dummy = ShapeDtypeStruct(shape, dtype)
    ufunc = _castto(scipy_gammainccinv, dtype)
    return pure_callback(ufunc, dummy, a, y, vmap_method='expand_dims')
