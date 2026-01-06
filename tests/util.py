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

from collections.abc import Sequence
from dataclasses import replace
from operator import ge, le
from pathlib import Path
from typing import Any

import numpy as np
import tomli
from jax import numpy as jnp
from jax import vmap
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, ArrayLike, Float, Real
from scipy import linalg

from bartz.debug import check_tree, describe_error
from bartz.jaxext import minimal_unsigned_dtype
from bartz.mcmcloop import TreesTrace


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

    def check_powers_of_2(seq: list[list]):
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

    error = check_tree(tree, max_split)
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
):
    """
    Check if two matrices are similar.

    Parameters
    ----------
    actual
    desired
        The two matrices to be compared. Must be scalars, vectors, or 2d arrays.
        Scalars and vectors are intepreted as 1x1 and Nx1 matrices, but the two
        arrays must have the same shape beforehand.
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

    Raises
    ------
    ValueError
        If the two matrices have different shapes.
    """
    actual = np.asarray(actual)
    desired = np.asarray(desired)
    if actual.shape != desired.shape:
        msg = f'{actual.shape=} != {desired.shape=}'
        raise ValueError(msg)
    if actual.size > 0:
        actual = np.atleast_1d(actual)
        desired = np.atleast_1d(desired)

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

        msg = f"""\
matrices actual and {ref} are not {cond} enough in {ord}-norm
matrix shape: {desired.shape}
norm(desired) = {dnorm:.2g}
norm({expr}) = {adnorm:.2g}  (atol = {atol:.2g})
ratio = {ratio:.2g}  (rtol = {rtol:.2g})"""

        assert op(adnorm, atol + rtol * dnorm), msg


def assert_different_matrices(*args, **kwargs):
    """Invoke `assert_close_matrices` with negate=True and default inf tolerance."""
    default_kwargs: dict = dict(rtol=np.inf, atol=np.inf)
    default_kwargs.update(kwargs)
    assert_close_matrices(*args, negate=True, **default_kwargs)


def get_old_python_str() -> str:
    """Read the oldest supported Python from pyproject.toml."""
    with Path('pyproject.toml').open('rb') as file:
        return tomli.load(file)['project']['requires-python'].removeprefix('>=')


def get_old_python_tuple() -> tuple[int, int]:
    """Read the oldest supported Python from pyproject.toml as a tuple."""
    ver_str = get_old_python_str()
    major, minor = ver_str.split('.')
    return int(major), int(minor)


def get_version() -> str:
    """Read the bartz version from pyproject.toml."""
    with Path('pyproject.toml').open('rb') as file:
        return tomli.load(file)['project']['version']


def update_version():
    """Update the version file."""
    version = get_version()
    Path('src/bartz/_version.py').write_text(f'__version__ = {version!r}\n')


def multivariate_rhat(chains: Real[Any, 'chain sample dim']) -> Float[Array, '']:
    """
    Compute the multivariate Gelman-Rubin R-hat.

    Parameters
    ----------
    chains
        Independent chains of samples of a vector.

    Returns
    -------
    Multivariate R-hat statistic.

    Raises
    ------
    ValueError
        If there are not enough chains or samples.
    """
    chains = jnp.asarray(chains)
    m, n, p = chains.shape

    if m < 2:  # pragma: no cover
        msg = 'Need at least 2 chains'
        raise ValueError(msg)
    if n < 2:  # pragma: no cover
        msg = 'Need at least 2 samples per chain'
        raise ValueError(msg)

    chain_means = jnp.mean(chains, axis=1)

    def compute_chain_cov(chain_samples, chain_mean):
        centered = chain_samples - chain_mean
        return jnp.dot(centered.T, centered) / (n - 1)

    within_chain_covs = vmap(compute_chain_cov)(chains, chain_means)
    W = jnp.mean(within_chain_covs, axis=0)

    overall_mean = jnp.mean(chain_means, axis=0)
    chain_mean_diffs = chain_means - overall_mean
    B = (n / (m - 1)) * jnp.dot(chain_mean_diffs.T, chain_mean_diffs)

    V_hat = ((n - 1) / n) * W + ((m + 1) / (m * n)) * B

    # Add regularization to W for numerical stability
    gershgorin = jnp.max(jnp.sum(jnp.abs(W), axis=1))
    regularization = jnp.finfo(W.dtype).eps * len(W) * gershgorin
    W_reg = W + regularization * jnp.eye(p)

    # Compute max(eigvals(W^-1 V_hat))
    L = jnp.linalg.cholesky(W_reg)
    # Solve L @ L.T @ x = V_hat @ x = λ @ W @ x
    # This is equivalent to solving (L^-1 V_hat L^-T) @ y = λ @ y
    L_1V = solve_triangular(L, V_hat, lower=True)
    L_1VL_T = solve_triangular(L, L_1V.T, lower=True).T
    eigenvals = jnp.linalg.eigvalsh(L_1VL_T)

    return jnp.max(eigenvals)


def rhat(chains: Real[Any, 'chain sample']) -> Float[Array, '']:
    """
    Compute the univariate Gelman-Rubin R-hat.

    Parameters
    ----------
    chains
        Independent chains of samples of a scalar.

    Returns
    -------
    Univariate R-hat statistic.
    """
    chains = jnp.asarray(chains)
    return multivariate_rhat(chains[:, :, None])
