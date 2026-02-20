# bartz/src/bartz/jaxext/scipy/special.py
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

from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any

from jax import ShapeDtypeStruct, jit, pure_callback
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float
from scipy.special import gammainccinv as scipy_gammainccinv


def _float_type(*args: DTypeLike | Array) -> jnp.dtype:
    """Determine the jax floating point result type given operands/types."""
    t = jnp.result_type(*args)
    return jnp.sin(jnp.empty(0, t)).dtype


def _castto(func: Callable[..., Array], dtype: DTypeLike) -> Callable[..., Array]:
    @wraps(func)
    def newfunc(*args: Any, **kw: Any) -> Array:
        return func(*args, **kw).astype(dtype)

    return newfunc


@jit
def gammainccinv(a: Float[Array, '*'], y: Float[Array, '*']) -> Float[Array, '*']:
    """Survival function inverse of the Gamma(a, 1) distribution."""
    shape = jnp.broadcast_shapes(a.shape, y.shape)
    dtype = _float_type(a.dtype, y.dtype)
    dummy = ShapeDtypeStruct(shape, dtype)
    ufunc = _castto(scipy_gammainccinv, dtype)
    return pure_callback(ufunc, dummy, a, y, vmap_method='expand_dims')


################# COPIED AND ADAPTED FROM JAX ##################
# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from jax import debug_infs, lax


def ndtri(p: Float[Array, '*']) -> Float[Array, '*']:
    """Compute the inverse of the CDF of the Normal distribution function.

    This is a patch of `jax.scipy.special.ndtri`.
    """
    dtype = lax.dtype(p)
    if dtype not in (jnp.float32, jnp.float64):
        msg = f'x.dtype={dtype} is not supported, see docstring for supported types.'
        raise TypeError(msg)
    return _ndtri(p)


def _ndtri(p: Float[Array, '...']) -> Float[Array, '...']:
    # Constants used in piece-wise rational approximations. Taken from the cephes
    # library:
    # https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html
    p0 = list(
        reversed(
            [
                -5.99633501014107895267e1,
                9.80010754185999661536e1,
                -5.66762857469070293439e1,
                1.39312609387279679503e1,
                -1.23916583867381258016e0,
            ]
        )
    )
    q0 = list(
        reversed(
            [
                1.0,
                1.95448858338141759834e0,
                4.67627912898881538453e0,
                8.63602421390890590575e1,
                -2.25462687854119370527e2,
                2.00260212380060660359e2,
                -8.20372256168333339912e1,
                1.59056225126211695515e1,
                -1.18331621121330003142e0,
            ]
        )
    )
    p1 = list(
        reversed(
            [
                4.05544892305962419923e0,
                3.15251094599893866154e1,
                5.71628192246421288162e1,
                4.40805073893200834700e1,
                1.46849561928858024014e1,
                2.18663306850790267539e0,
                -1.40256079171354495875e-1,
                -3.50424626827848203418e-2,
                -8.57456785154685413611e-4,
            ]
        )
    )
    q1 = list(
        reversed(
            [
                1.0,
                1.57799883256466749731e1,
                4.53907635128879210584e1,
                4.13172038254672030440e1,
                1.50425385692907503408e1,
                2.50464946208309415979e0,
                -1.42182922854787788574e-1,
                -3.80806407691578277194e-2,
                -9.33259480895457427372e-4,
            ]
        )
    )
    p2 = list(
        reversed(
            [
                3.23774891776946035970e0,
                6.91522889068984211695e0,
                3.93881025292474443415e0,
                1.33303460815807542389e0,
                2.01485389549179081538e-1,
                1.23716634817820021358e-2,
                3.01581553508235416007e-4,
                2.65806974686737550832e-6,
                6.23974539184983293730e-9,
            ]
        )
    )
    q2 = list(
        reversed(
            [
                1.0,
                6.02427039364742014255e0,
                3.67983563856160859403e0,
                1.37702099489081330271e0,
                2.16236993594496635890e-1,
                1.34204006088543189037e-2,
                3.28014464682127739104e-4,
                2.89247864745380683936e-6,
                6.79019408009981274425e-9,
            ]
        )
    )

    dtype = lax.dtype(p).type
    shape = jnp.shape(p)

    def _create_polynomial(
        var: Float[Array, '...'], coeffs: Sequence[float]
    ) -> Float[Array, '...']:
        """Compute n_th order polynomial via Horner's method."""
        coeffs = np.array(coeffs, dtype)
        if not coeffs.size:
            return jnp.zeros_like(var)
        return coeffs[0] + _create_polynomial(var, coeffs[1:]) * var

    maybe_complement_p = jnp.where(p > dtype(-np.expm1(-2.0)), dtype(1.0) - p, p)
    # Write in an arbitrary value in place of 0 for p since 0 will cause NaNs
    # later on. The result from the computation when p == 0 is not used so any
    # number that doesn't result in NaNs is fine.
    sanitized_mcp = jnp.where(
        maybe_complement_p == dtype(0.0),
        jnp.full(shape, dtype(0.5)),
        maybe_complement_p,
    )

    # Compute x for p > exp(-2): x/sqrt(2pi) = w + w**3 P0(w**2)/Q0(w**2).
    w = sanitized_mcp - dtype(0.5)
    ww = lax.square(w)
    x_for_big_p = w + w * ww * (_create_polynomial(ww, p0) / _create_polynomial(ww, q0))
    x_for_big_p *= -dtype(np.sqrt(2.0 * np.pi))

    # Compute x for p <= exp(-2): x = z - log(z)/z - (1/z) P(1/z) / Q(1/z),
    # where z = sqrt(-2. * log(p)), and P/Q are chosen between two different
    # arrays based on whether p < exp(-32).
    z = lax.sqrt(dtype(-2.0) * lax.log(sanitized_mcp))
    first_term = z - lax.log(z) / z
    second_term_small_p = (
        _create_polynomial(dtype(1.0) / z, p2)
        / _create_polynomial(dtype(1.0) / z, q2)
        / z
    )
    second_term_otherwise = (
        _create_polynomial(dtype(1.0) / z, p1)
        / _create_polynomial(dtype(1.0) / z, q1)
        / z
    )
    x_for_small_p = first_term - second_term_small_p
    x_otherwise = first_term - second_term_otherwise

    x = jnp.where(
        sanitized_mcp > dtype(np.exp(-2.0)),
        x_for_big_p,
        jnp.where(z >= dtype(8.0), x_for_small_p, x_otherwise),
    )

    x = jnp.where(p > dtype(1.0 - np.exp(-2.0)), x, -x)
    with debug_infs(False):
        infinity = jnp.full(shape, dtype(np.inf))
        neg_infinity = -infinity
    return jnp.where(
        p == dtype(0.0), neg_infinity, jnp.where(p == dtype(1.0), infinity, x)
    )


################################################################
