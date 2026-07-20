# bartz/tests/test_jaxext.py
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

"""Test bartz._jaxext."""

from contextlib import nullcontext
from functools import partial
from itertools import product
from warnings import catch_warnings, simplefilter

import numpy
import pytest
from jax import (
    NamedSharding,
    device_put,
    grad,
    jvp,
    lax,
    make_mesh,
    random,
    shard_map,
    tree,
    vmap,
)

# WORKAROUND(jax<0.7.1): top-level `jax.enable_x64` was added later; on older jax
# it lives in `jax.experimental` (removed in newer jax). Use whichever exists.
try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64  # ty: ignore[unresolved-import]
from jax import numpy as jnp
from jax.sharding import AxisType, Mesh, PartitionSpec
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Float32, Key, Shaped
from pytest_subtests import SubTests
from scipy.special import ndtr as scipy_ndtr
from scipy.stats import anderson_ksamp, ks_1samp, truncnorm
from scipy.stats import invgamma as scipy_invgamma
from scipy.stats import loggamma as scipy_loggamma
from scipy.stats import poisson as scipy_poisson
from scipy.stats import uniform as scipy_uniform

from bartz._jaxext import (
    autobatch,
    equal_shards,
    get_default_devices,
    get_device_count,
    is_key,
    jit,
    sliced_map,
    split,
    truncated_normal_onesided,
    unique,
)
from bartz._jaxext.random import loggamma, poisson
from bartz._jaxext.random._poisson import poisson_from_normal
from bartz._jaxext.scipy.stats import invgamma
from tests.util import (
    assert_allclose,
    assert_array_equal,
    assert_close_matrices,
    int_seed,
)


class TestUnique:
    """Test _jaxext.unique."""

    def test_sort(self) -> None:
        """Check that it's equivalent to sort if no values are repeated."""
        x = jnp.arange(10)[::-1]
        out, length = unique(x, x.size, 666)
        assert_array_equal(jnp.sort(x), out)
        assert out.dtype == x.dtype
        assert length == x.size

    def test_fill(self) -> None:
        """Check that the trailing fill value is used correctly."""
        x = jnp.ones(10)
        out, length = unique(x, x.size, 666)
        assert_array_equal(jnp.asarray([1.0] + 9 * [666.0]), out)
        assert out.dtype == x.dtype
        assert length == 1

    def test_empty_input(self) -> None:
        """Check that the function works on empty input."""
        x = jnp.array([])
        out, length = unique(x, 2, 666)
        assert_array_equal(jnp.asarray([666.0, 666.0]), out)
        assert out.dtype == x.dtype
        assert length == 0

    def test_empty_output(self) -> None:
        """Check that the function works if the output is forced to be empty."""
        x = jnp.array([1, 1, 1])
        out, length = unique(x, 0, 666)
        assert_array_equal(jnp.zeros(0, dtype=x.dtype), out)
        assert out.dtype == x.dtype
        assert length == 0


class TestAutoBatch:
    """Test _jaxext.autobatch."""

    @pytest.mark.parametrize('target_nbatches', [1, 7])
    @pytest.mark.parametrize('with_margin', [False, True])
    @pytest.mark.parametrize('additional_size', [3, 0])
    def test_batch_size(
        self, keys: split, target_nbatches: int, with_margin: bool, additional_size: int
    ) -> None:
        """Check batch sizes are correct in various conditions."""

        def func(
            a: Float[Array, 'n m'], b: Float[Array, ' n'], c: Float[Array, 'p n']
        ) -> tuple[Float[Array, ' n'], Float[Array, 'p n']]:
            return (a * b[:, None]).sum(1), c * b[None, :]

        atomic_batch_size = additional_size + 12
        multiplier = 2
        batch_size = multiplier * atomic_batch_size
        if with_margin:
            batch_size += 1
        size = target_nbatches * multiplier

        a = random.uniform(keys.pop(), (size, additional_size))
        b = random.uniform(keys.pop(), (size,))
        c = random.uniform(keys.pop(), (5, size))

        assert atomic_batch_size == a.shape[1] + 1 + c.shape[0] + 1 + c.shape[0]

        batch_nbytes = batch_size * a.itemsize
        batched_func = autobatch(
            func, batch_nbytes, (0, 0, 1), (0, 1), return_nbatches=True
        )
        batched_func_nobatches = autobatch(func, batch_nbytes, (0, 0, 1), (0, 1))

        out1 = func(a, b, c)
        out2, nbatches = batched_func(a, b, c)
        out3 = batched_func_nobatches(a, b, c)

        assert nbatches == target_nbatches

        for o2, o3 in zip(out2, out3, strict=True):
            numpy.testing.assert_array_max_ulp(o2, o3)
        for o1, o2 in zip(out1, out2, strict=True):
            numpy.testing.assert_array_max_ulp(o1, o2)

    @pytest.mark.parametrize('max_memory', [32, 1024])
    # test with large max memory to trigger noop code path
    def test_unbatched_arg(self, max_memory: int) -> None:
        """Check the function with batching disabled on a scalar argument."""

        def func(a: Shaped[Array, ' n'], b: int) -> Shaped[Array, ' n']:
            return a + b

        batched_func = autobatch(func, max_memory, (0, None))

        a = jnp.arange(100)
        b = 2

        out1 = func(a, b)
        out2 = batched_func(a, b)

        numpy.testing.assert_array_max_ulp(out1, out2)

    def test_batch_axis_pytree(self) -> None:
        """Check the that a batch axis can be specified for a whole sub-pytree."""

        def func(a: int, b: dict[str, Shaped[Array, ' n']]) -> Shaped[Array, ' n']:
            return a + b['foo'] + b['bar']

        batched_func = autobatch(func, 32, (None, 0))

        a = 2
        b = dict(foo=jnp.arange(100), bar=jnp.arange(100))

        out1 = func(a, b)
        out2 = batched_func(a, b)

        numpy.testing.assert_array_max_ulp(out1, out2)

    def test_large_batch_warning(self) -> None:
        """Check the function emits a warning if the size limit can't be honored."""
        x = jnp.arange(10_000).reshape(10, 1000)

        def f(x: Shaped[Array, 'n m']) -> Shaped[Array, 'n m']:
            return x

        g = autobatch(f, 100)
        with pytest.warns(UserWarning, match=' > max_io_nbytes = '):
            g(x)

    def test_empty_values(self) -> None:
        """Check that the function works with batchable empty arrays."""
        x = jnp.empty((10, 0))

        def f(x: Shaped[Array, 'n m']) -> Shaped[Array, 'n m']:
            return x

        g = autobatch(f, 100, return_nbatches=True)
        y, nbatches = g(x)
        assert nbatches == 1
        assert jnp.all(y == x)

    def test_zero_size(self) -> None:
        """Check the function works with a batch axis with length 0."""
        x = jnp.empty((0, 10))

        def f(x: Shaped[Array, 'n m']) -> Shaped[Array, 'n m']:
            return x

        g = autobatch(f, 100, return_nbatches=True)
        y, nbatches = g(x)
        assert nbatches == 1
        assert jnp.all(y == x)

    def test_reduction_basic(self, keys: split, subtests: SubTests) -> None:
        """Check that reduction produces the expected result."""
        # use an internal loop instead of pytest.mark.parametrize because there
        # are too many combinations of parameters
        # explicit names because the ufunc type stubs lack `__name__`
        ops = [
            (None, None, lambda x, **_kw: x),
            ('add', jnp.add, jnp.sum),
            ('logical_and', jnp.logical_and, jnp.all),
        ]
        shape_axes = [
            ((10,), 0),
            ((10, 100), 0),
            ((10, 100), 1),
            ((10, 100), -1),
            ((10, 100), -2),
            ((0,), 0),
            ((10, 0), 0),
        ]
        max_io_nbytes_list = [1, 100, 100_000_000]
        nins = [1, 2]
        dtypes = [jnp.float32, jnp.int8, jnp.bool_]

        key = keys.pop()

        for op, shape_axis, max_io_nbytes, nin, dtype in product(
            ops, shape_axes, max_io_nbytes_list, nins, dtypes
        ):
            ufunc_name, ufunc, reduction = op
            shape, axis = shape_axis

            with subtests.test(
                ufunc=ufunc_name,
                shape=shape,
                axis=axis,
                max_io_nbytes=max_io_nbytes,
                nin=nin,
                dtype=dtype.dtype.name,
            ):

                def func(
                    *args: Shaped[Array, '*shape'], nin: int = nin
                ) -> Shaped[Array, '*shape'] | tuple[Shaped[Array, '*shape'], ...]:
                    out = sum(args)
                    assert isinstance(out, Array)  # rule out the empty-sum 0
                    if nin == 1:
                        return out
                    else:
                        return tuple(i * out for i in range(1, nin + 1))

                keys = split(key)
                key = keys.pop()

                if jnp.issubdtype(dtype, jnp.floating):
                    args = random.uniform(keys.pop(), (nin, *shape), dtype)
                elif jnp.issubdtype(dtype, jnp.integer):
                    args = random.randint(
                        keys.pop(),
                        (nin, *shape),
                        jnp.iinfo(dtype).min // 2,
                        (jnp.iinfo(dtype).max + 1) // 2,
                        dtype,
                    )
                elif jnp.issubdtype(dtype, jnp.bool_):  # pragma: no branch
                    args = random.bernoulli(keys.pop(), 0.5, (nin, *shape))

                expected = tree.map(partial(reduction, axis=axis), func(*args))

                batched_func = autobatch(
                    func,
                    max_io_nbytes,
                    axis,
                    axis,
                    reduce_ufunc=ufunc,
                    return_nbatches=True,
                )
                with catch_warnings(record=True) as caught_warnings:
                    result, nbatches = batched_func(*args)

                # Check at most one warning is raised
                assert len(caught_warnings) <= 1

                if caught_warnings:
                    (w,) = caught_warnings
                    assert issubclass(w.category, UserWarning)
                    assert 'batch_nbytes =' in str(w.message)
                    assert '> max_io_nbytes =' in str(w.message)
                    assert nbatches == max(1, shape[axis])

                tree.map(partial(assert_close_matrices, rtol=1e-6), result, expected)

    def test_reduction_with_unbatched_input(self, keys: split) -> None:
        """Check reduction works with unbatched (None) input arguments."""

        def func(x: Float[Array, 'n m'], scalar: float) -> Float[Array, 'n m']:
            return x * scalar

        x = random.uniform(keys.pop(), (50, 8))
        scalar = 3.0
        expected = func(x, scalar).sum(axis=0)

        batched_func = autobatch(func, 100, (0, None), 0, reduce_ufunc=jnp.add)
        result = batched_func(x, scalar)

        assert result.shape == (8,)
        assert_close_matrices(result, expected, rtol=1e-6)

    def test_reduction_with_return_nbatches(self, keys: split) -> None:
        """Check reduce_ufunc works together with return_nbatches."""

        def func(x: Float[Array, 'n m']) -> Float[Array, 'n m']:
            return x

        x = random.uniform(keys.pop(), (100, 10))
        expected = x.sum(axis=0)

        batched_func = autobatch(
            func, 200, 0, 0, return_nbatches=True, reduce_ufunc=jnp.add
        )
        result, nbatches = batched_func(x)

        assert nbatches.shape == ()
        assert jnp.issubdtype(nbatches.dtype, jnp.integer)

        assert result.shape == (10,)
        assert_close_matrices(result, expected, rtol=1e-6)

    @pytest.mark.parametrize('batched', [True, False])
    def test_none_output_leaf(self, keys: split, batched: bool) -> None:
        """Check that `None` output leaves are passed through unchanged."""

        def func(x: Float[Array, ' n']) -> tuple[Float[Array, ' n'], None]:
            return x * 2, None

        x = random.uniform(keys.pop(), (100,))
        max_io_nbytes = 32 if batched else 1_000_000
        batched_func = autobatch(func, max_io_nbytes, 0, 0, return_nbatches=True)

        out1 = func(x)
        (out2_arr, out2_none), nbatches = batched_func(x)

        if batched:
            assert nbatches > 1
        else:
            assert nbatches == 1
        assert out2_none is None
        numpy.testing.assert_array_max_ulp(out1[0], out2_arr)

    def test_out_axes_none_mismatch(self, keys: split) -> None:
        """Check `None` in `out_axes` at non-`None` output positions errors."""

        def func(x: Float[Array, ' n']) -> Float[Array, ' n']:
            return x * 2

        x = random.uniform(keys.pop(), (10,))
        # the message wording depends on the jax version and on which pytree
        # prefix check trips first
        match = 'Expected None|different types at key path'
        with pytest.raises(ValueError, match=match):
            autobatch(func, 32, 0, None)(x)


class TestSlicedMap:
    """Test _jaxext.sliced_map."""

    @staticmethod
    def f(
        args: tuple[Float[Array, ''], dict[str, Float[Array, ' m']]],
    ) -> tuple[Float[Array, ''], tuple[Float[Array, ''], None]]:
        """Do something; example first argument for `lax.map`."""
        x, d = args
        return x * d['w'].sum(), (x + 1, None)

    def xs(
        self, key: Key[Array, ''], shape: tuple[int, ...] = (10,)
    ) -> tuple[Float[Array, '*batch n'], dict[str, Float[Array, '*batch n m']]]:
        """Return example second argument for `lax.map`."""
        keys = split(key)
        x = random.uniform(keys.pop(), shape)
        w = random.uniform(keys.pop(), (*shape, 3))
        return x, dict(w=w)

    @pytest.mark.parametrize('batch_size', [1, 3, 5, 10, 12])
    # 3: remainder batch, 5: exact division, 10 and 12: single-batch shortcut
    def test_matches_lax_map(self, keys: split, batch_size: int) -> None:
        """Check the output is identical to `lax.map` with `batch_size`."""
        xs = self.xs(keys.pop())
        out = sliced_map(self.f, xs, batch_size=batch_size)
        expected = lax.map(self.f, xs, batch_size=batch_size)
        assert out[1][1] is None
        tree.map(assert_array_equal, out, expected)

    @pytest.mark.parametrize('batch_size', [3, 5])
    def test_under_vmap(self, keys: split, batch_size: int) -> None:
        """Check consistency with `lax.map` when the inputs are closed-over batched values."""
        x, d = self.xs(keys.pop(), (2, 10))

        def with_sliced_map(
            x: Float[Array, ' n'], w: Float[Array, 'n m']
        ) -> tuple[Float[Array, ' n'], tuple[Float[Array, ' n'], None]]:
            return sliced_map(self.f, (x, dict(w=w)), batch_size=batch_size)

        def with_lax_map(
            x: Float[Array, ' n'], w: Float[Array, 'n m']
        ) -> tuple[Float[Array, ' n'], tuple[Float[Array, ' n'], None]]:
            return lax.map(self.f, (x, dict(w=w)), batch_size=batch_size)

        out = vmap(with_sliced_map)(x, d['w'])
        expected = vmap(with_lax_map)(x, d['w'])
        tree.map(assert_array_equal, out, expected)

    def test_mismatched_leading_axes(self, keys: split) -> None:
        """Check that leaves with different leading axis sizes are an error."""
        x, d = self.xs(keys.pop())
        with pytest.raises(ValueError, match='values to unpack'):
            sliced_map(self.f, (x[:5], d), batch_size=3)


def different_keys(keya: Key[Array, ''], keyb: Key[Array, '']) -> bool:
    """Return True iff two jax random keys are different."""
    return jnp.any(random.key_data(keya) != random.key_data(keyb)).item()


def test_split(keys: split) -> None:
    """Test _jaxext.split."""
    key = keys.pop()
    ks = split(key, 3)

    assert len(ks) == 3
    key1 = ks.pop()
    assert len(ks) == 2
    key2 = ks.pop()
    assert len(ks) == 1
    key3 = ks.pop()
    assert len(ks) == 0

    with pytest.raises(IndexError):
        ks.pop()

    assert different_keys(key, key1)
    assert different_keys(key, key2)
    assert different_keys(key, key3)
    assert different_keys(key1, key2)
    assert different_keys(key1, key3)
    assert different_keys(key2, key3)

    ks = split(random.clone(key), 3)
    key1a = ks.pop()
    key2a = ks.pop(2)
    key3a = ks.pop()

    assert not different_keys(key1, key1a)
    assert not different_keys(random.split(key2), key2a)
    assert not different_keys(key3, key3a)

    ks = split(keys.pop(), 1)
    key = ks.pop((2, 3, 5))
    assert key.shape == (2, 3, 5)
    assert len(ks) == 0

    ks = split(keys.pop())
    assert len(ks) == 2


class TestJaxPatches:
    """Check that some jax stuff I patch is correct and still to be patched."""

    def test_invgamma_missing(self) -> None:
        """Check that jax does not implement the inverse gamma distribution."""
        with pytest.raises(ImportError, match=r'gammainccinv'):
            from jax.scipy.special import (  # noqa: PLC0415
                gammainccinv,  # noqa: F401 # ty: ignore[unresolved-import]
            )
        with pytest.raises(ImportError, match=r'invgamma'):
            from jax.scipy.stats import (  # noqa: PLC0415
                invgamma,  # noqa: F401 # ty: ignore[unresolved-import]
            )

    def test_invgamma_correct(self, keys: split) -> None:
        """Compare my implementation of invgamma against scipy's."""
        p = random.uniform(keys.pop(), (100,), float, 0.01, 0.99)
        alpha = 3.5
        x0 = scipy_invgamma.ppf(p, alpha)
        x1 = invgamma.ppf(p, alpha)
        assert_close_matrices(x1, x0.astype(x1.dtype), rtol=1e-6)


class TestTruncatedNormalOneSided:
    """Test `_jaxext.truncated_normal_onesided`."""

    def test_truncated_normal_incorrect(self, keys: split) -> None:
        """Check that `jax.random.truncated_normal` is wrong out of 5 sigma."""
        nsamples = 1000
        lower, upper = jnp.array([(-100.0, -5.0), (5.0, 100.0)]).T
        x = random.truncated_normal(
            keys.pop(), lower[:, None], upper[:, None], (*lower.shape, nsamples)
        )
        for sample, l, u in zip(x, lower, upper, strict=True):
            test = ks_1samp(sample, truncnorm(l, u).cdf)
            assert test.pvalue < 0.01

    def test_correct(self, keys: split) -> None:
        """Check the samples come from the right distribution."""
        nparams = 20
        nsamples = 1000
        upper = random.bernoulli(keys.pop(), 0.5, (nparams,))
        bound = random.uniform(keys.pop(), (nparams,), float, -10, 10)
        x = truncated_normal_onesided(
            keys.pop(), (nparams, nsamples), upper[:, None], bound[:, None]
        )
        for sample, u, b in zip(x, upper, bound, strict=True):
            left = -jnp.inf if u else b
            right = b if u else jnp.inf
            test = ks_1samp(sample, truncnorm(left, right).cdf)
            assert test.pvalue > 0.01

    def test_accurate(self, keys: split) -> None:
        """Check that it does not over/under shoot."""
        x = truncated_normal_onesided(keys.pop(), (), jnp.bool_(True), jnp.float32(-12))
        assert -12.1 <= x < -12
        x = truncated_normal_onesided(keys.pop(), (), jnp.bool_(False), jnp.float32(12))
        assert 12 < x <= 12.1

    def test_finite(self, keys: split) -> None:
        """Check that the outputs are always finite."""
        # shape and n_loops combined shall be enough that all possible
        # float32 values in [0, 1) are drawn by random.uniform
        shape = (1_000_000,)
        n_loops = 100

        loop_keys = keys.pop(n_loops)

        platform = loop_keys.device.platform  # ty: ignore[unresolved-attribute]
        clip = platform == 'gpu'

        @jit
        def loop_body(key: Key[Array, '']) -> Float32[Array, ' n']:
            keys = split(key, 3)
            upper = random.bernoulli(keys.pop(), 0.5, shape)
            bound = random.uniform(keys.pop(), shape, float, -1, 1)
            return truncated_normal_onesided(keys.pop(), shape, upper, bound, clip=clip)

        for key in loop_keys:
            vals = loop_body(key)
            assert jnp.all(jnp.isfinite(vals))


class TestLoggamma:
    """Test `_jaxext.random.loggamma`."""

    @pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
    @pytest.mark.parametrize(
        'alpha', [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    )
    def test_distribution(
        self, keys: split, alpha: float, dtype: DTypeLike, subtests: SubTests
    ) -> None:
        """Check the samples follow log-Gamma(alpha, 1) against a cdf and a sample."""
        nsamples = 100_000  # tested up to 100M
        x64 = enable_x64(True) if dtype == jnp.float64 else nullcontext()
        with x64:
            sample = loggamma(keys.pop(), alpha, (nsamples,), dtype)
            assert sample.dtype == jnp.dtype(dtype)
            # no sample underflows to -inf for small alpha
            assert jnp.all(jnp.isfinite(sample))

        dist = scipy_loggamma(alpha)

        with subtests.test('KS'):
            ks = ks_1samp(sample, dist.cdf)
            assert ks.pvalue > 1e-3

        # the anderson-darling test is more sensitive in the tails
        with subtests.test('AD'):
            reference = dist.rvs(size=nsamples, random_state=int_seed(keys.pop()))
            with catch_warnings():
                # AD caps/floors its reported p-value to [0.001, 0.25], warning on it
                simplefilter('ignore')
                ad = anderson_ksamp([sample, reference])
            # AD floors its p-value at 0.001, so we cut on the statistic instead
            assert ad.statistic <= ad.critical_values[-1]  # 0.001 threshold

    @pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16])
    @pytest.mark.parametrize('alpha', [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
    def test_distribution_low_precision(
        self, keys: split, dtype: DTypeLike, alpha: float, subtests: SubTests
    ) -> None:
        """Like `test_distribution` but for the low-precision float16/bfloat16.

        Tested over the alpha range each dtype can represent, truncating the left
        tail that underflows to -inf (only float16 underflows; bfloat16 has the
        same exponent range as float32). A larger nsamples eventually resolves the
        dtype's discretization and would fail.
        """
        # bfloat16's 8-bit mantissa cannot resolve the distribution at larger alpha
        if dtype == jnp.bfloat16 and alpha > 1e0:
            pytest.skip('bfloat16 too coarse to resolve the distribution here')
        nsamples = 100_000
        sample = loggamma(keys.pop(), alpha, (nsamples,), dtype)
        assert sample.dtype == jnp.dtype(dtype)

        # the deep left tail can underflow below the smallest value of `dtype`; drop
        # those and compare against the cdf conditioned on the representable range
        floor = jnp.finfo(dtype).min.item()
        finite = sample[sample >= floor]
        finite = finite.astype(jnp.float32)  # cast bc KS preserves dtype internally
        assert finite.size > 0.99 * nsamples  # underflow is a rare tail event here

        dist = scipy_loggamma(alpha)
        cdf_floor = dist.cdf(floor)

        def truncated_cdf(x: Float[numpy.ndarray, ' _']) -> Float[numpy.ndarray, ' _']:
            """Cdf conditioned on the value being representable (>= floor)."""
            return (dist.cdf(x) - cdf_floor) / (1 - cdf_floor)

        with subtests.test('KS'):
            ks = ks_1samp(finite, truncated_cdf)
            assert ks.pvalue > 1e-3

        with subtests.test('AD'):
            reference = dist.rvs(size=nsamples, random_state=int_seed(keys.pop()))
            reference = reference[reference >= floor]  # match the sample truncation
            with catch_warnings():
                # AD caps/floors its reported p-value to [0.001, 0.25], warning on it
                simplefilter('ignore')
                ad = anderson_ksamp([finite, reference])
            # AD floors its p-value at 0.001, so we cut on the statistic instead
            assert ad.statistic <= ad.critical_values[-1]  # 0.001 threshold

    @pytest.mark.parametrize('shape', [(), (12,), (3, 4), (2, 3, 2), (1, 12, 1)])
    def test_shape_consistency(self, keys: split, shape: tuple[int, ...]) -> None:
        """A shaped draw equals the flat draw reshaped, given the same key."""
        key = keys.pop()
        alpha = 1.3
        sample = loggamma(key, alpha, shape)
        assert sample.shape == shape
        flat = loggamma(random.clone(key), alpha, (sample.size,))
        assert_array_equal(sample.reshape(sample.size), flat)

    @pytest.mark.parametrize('n_uniforms', [0, 1, 2, 3, 4, 6, 8])
    def test_n_uniforms(self, keys: split, n_uniforms: int) -> None:
        """At high alpha the base draw dominates, so any n_uniforms matches."""
        alpha = 100.0
        nsamples = 100_000
        sample = loggamma(keys.pop(), alpha, (nsamples,), n_uniforms=n_uniforms)
        ks = ks_1samp(sample, scipy_loggamma(alpha).cdf)
        assert ks.pvalue > 1e-3


@partial(jit, static_argnums=(2,))
def _poisson_sample_and_tangent(
    key: Key[Array, ''], lambda_: Float[Array, ''], nsamples: int
) -> tuple[Float[Array, ' nsamples'], Float[Array, ' nsamples']]:
    """Sample Poisson(lambda_) values and their derivative w.r.t. lambda_."""

    def f(lambda_: Float[Array, '']) -> Float[Array, ' nsamples']:
        return poisson(key, jnp.full((nsamples,), lambda_), dtype=float)

    return jvp(f, (lambda_,), (jnp.ones_like(lambda_),))


class TestPoisson:
    """Test `_jaxext.random.poisson`."""

    nsamples = 200_000  # the tests below pass unchanged up to 10M samples

    @pytest.mark.parametrize('x64', [False, True])
    # 2**24 is the float32 integer limit: beyond it the samples land on the
    # coarser float lattice, so it is the largest lambda_ with an exact pmf
    @pytest.mark.parametrize('lambda_', [0.05, 0.5, 5.0, 7.0, 20.0, 1e3, 1e5, 2.0**24])
    def test_distribution(
        self, keys: split, lambda_: float, x64: bool, subtests: SubTests
    ) -> None:
        """Check the samples follow Poisson(lambda_) against the cdf and a sample."""
        nsamples = self.nsamples
        ctx = enable_x64(True) if x64 else nullcontext()
        with ctx:
            sample = poisson(keys.pop(), lambda_, (nsamples,))
            assert jnp.issubdtype(sample.dtype, jnp.integer)
            assert jnp.all(sample >= 0)
        sample = numpy.asarray(sample)

        dist = scipy_poisson(lambda_)

        # the plain KS test breaks on a discrete cdf, so map the sample to
        # exactly uniform under H0 with the randomized probability integral
        # transform: U = F(X - 1) + V pmf(X), V ~ uniform
        with subtests.test('KS'):
            rng = numpy.random.default_rng(int_seed(keys.pop()))
            v = rng.uniform(size=sample.shape)
            u = dist.cdf(sample - 1) + v * dist.pmf(sample)
            ks = ks_1samp(u, scipy_uniform.cdf)
            assert ks.pvalue > 1e-3

        # the anderson-darling test is more sensitive in the tails; its midrank
        # correction handles the heavily tied discrete samples
        with subtests.test('AD'):
            reference = dist.rvs(size=nsamples, random_state=int_seed(keys.pop()))
            with catch_warnings():
                # AD caps/floors its reported p-value to [0.001, 0.25], warning on it
                simplefilter('ignore')
                ad = anderson_ksamp([sample, reference])
            # AD floors its p-value at 0.001, so we cut on the statistic instead
            assert ad.statistic <= ad.critical_values[-1]  # 0.001 threshold

    def test_zero(self, keys: split) -> None:
        """Check lambda_ = 0 yields all zeros."""
        sample = poisson(keys.pop(), 0.0, (self.nsamples,))
        assert_array_equal(sample, jnp.zeros(sample.shape, sample.dtype))

    @pytest.mark.parametrize('lambda_', [4.5, 8.5])
    @pytest.mark.parametrize('shape', [(), (12,), (3, 4), (2, 3, 2), (1, 12, 1)])
    def test_shape_consistency(
        self, keys: split, lambda_: float, shape: tuple[int, ...]
    ) -> None:
        """A shaped draw equals the flat draw reshaped, given the same key."""
        key = keys.pop()
        sample = poisson(key, lambda_, shape)
        assert sample.shape == shape
        flat = poisson(random.clone(key), lambda_, (sample.size,))
        assert_array_equal(sample.reshape(sample.size), flat)

    @pytest.mark.parametrize('lambda_', [4.5, 8.5])
    @pytest.mark.parametrize('lambda_shape', [(), (4,), (1, 4)])
    def test_lambda_broadcast(
        self, keys: split, lambda_: float, lambda_shape: tuple[int, ...]
    ) -> None:
        """A draw with unbroadcasted lambda_ equals the pre-broadcasted draw."""
        shape = (3, 4)
        key = keys.pop()
        sample = poisson(key, jnp.full(lambda_shape, lambda_), shape)
        assert sample.shape == shape
        expected = poisson(random.clone(key), jnp.full(shape, lambda_), shape)
        assert_array_equal(sample, expected)

    def test_lambda_broadcast_grad(self, keys: split) -> None:
        """Reverse mode w.r.t. an unbroadcasted lambda_ sums over the broadcast."""
        lambda_ = jnp.array([2.0, 20.0])
        key = keys.pop()

        def f(lambda_: Float[Array, ' 2']) -> Float[Array, '5 2']:
            return poisson(key, lambda_, (5, 2), dtype=float)

        _, tangent = jvp(f, (lambda_,), (jnp.ones_like(lambda_),))
        gradient = grad(lambda lambda_: f(lambda_).sum())(lambda_)
        assert_close_matrices(gradient, tangent.sum(axis=0), rtol=1e-6)

    @pytest.mark.parametrize('shape', [(3,), ()])
    def test_lambda_shape_mismatch(self, keys: split, shape: tuple[int, ...]) -> None:
        """A lambda_ that does not broadcast to shape is an error."""
        with pytest.raises(ValueError, match='broadcast'):
            poisson(keys.pop(), jnp.ones(5), shape)

    @pytest.mark.parametrize('lambda_', [0.1, 0.5, 2.0, 5.0, 20.0, 1e3, 2.0**24])
    def test_derivative(self, keys: split, lambda_: float, subtests: SubTests) -> None:
        """Averaged sample derivatives estimate derivatives of moments.

        Single-sample derivatives of a discrete variable are meaningless; the
        implicit-reparameterization tangent is defined to estimate derivatives
        of expectations when averaged. Check the first two moments, whose
        derivatives w.r.t. lambda_ are 1 and 2 lambda_ + 1; below the branch
        split the estimates are exactly unbiased. The 1% tolerance is the
        tightest power of ten: it is set by the sampling noise at the smallest
        lambda_, where the tangent variance ~ 1/(3 lambda_) peaks.
        """
        # more samples at small lambda_ to beat the 1/(3 lambda_) variance
        nsamples = self.nsamples * (16 if lambda_ < 1 else 1)
        sample, tangent = _poisson_sample_and_tangent(keys.pop(), lambda_, nsamples)
        assert jnp.all(jnp.isfinite(tangent))

        # average in float64: at lambda_ = 2**24 the float32 reduction error
        # on the second moment would be comparable to the 1% tolerance
        sample = numpy.asarray(sample, numpy.float64)
        tangent = numpy.asarray(tangent, numpy.float64)

        with subtests.test('mean'):
            assert_allclose(numpy.mean(tangent), 1.0, rtol=0.01)

        with subtests.test('second moment'):
            assert_allclose(
                numpy.mean(2 * sample * tangent), 2 * lambda_ + 1, rtol=0.01
            )

    @pytest.mark.parametrize(
        ('lambda_', 'bound'),
        [
            (0.1, 1e-6),
            (1.0, 1e-6),
            (5.0, 1e-5),
            (6.9, 1e-5),
            (7.0, 1e-4),
            (10.0, 1e-4),
            (20.0, 1e-5),
            (50.0, 1e-5),
            (1e6, 1e-4),
            (1.6e7, 1e-4),  # just below 2**24, pmf not available above
        ],
    )
    def test_total_variation(self, lambda_: float, bound: float) -> None:
        """Bound the exact total variation error of the sampler.

        The sampler maps a normal variate monotonically to an integer, so its
        implied pmf can be computed exactly (no sampling) by locating the jumps
        of the map with bisection. The total variation distance to the true
        Poisson pmf is dominated by float roundoff below the branch split and
        by the Peizer-Pratt approximation error just above it; the bounds are
        the observed values rounded up to a power of ten (or two).
        """
        # enumerate a window covering all but ~1e-12 of both pmfs; k0 - 1 is
        # included as the base of the cdf differences (never sampled, so its
        # implied cdf bisects to ~0 when k0 = 0)
        k0 = max(int(scipy_poisson.ppf(1e-12, lambda_)) - 2, 0)
        k1 = int(scipy_poisson.isf(1e-12, lambda_)) + 2
        k = k0 - 1 + jnp.arange(k1 - k0 + 2, dtype=jnp.float32)

        # invariant: sampler(lo) <= k < sampler(hi), z jump locations in between
        lo = jnp.full(k.shape, -8.0, jnp.float32)
        hi = jnp.full(k.shape, 8.0, jnp.float32)
        for _ in range(60):
            mid = (lo + hi) / 2
            below = poisson_from_normal(mid, jnp.float32(lambda_)) <= k
            lo = jnp.where(below, mid, lo)
            hi = jnp.where(below, hi, mid)

        implied_cdf = scipy_ndtr(numpy.asarray(lo, numpy.float64))
        implied_pmf = numpy.diff(implied_cdf)
        exact_pmf = scipy_poisson.pmf(numpy.arange(k0, k1 + 1), lambda_)

        tv = 0.5 * numpy.abs(implied_pmf - exact_pmf).sum()
        # out-of-window mass of both pmfs bounds its TV contribution
        tv += 0.5 * (implied_cdf[0] + scipy_poisson.cdf(k0 - 1, lambda_))
        tv += 0.5 * ((1 - implied_cdf[-1]) + scipy_poisson.sf(k1, lambda_))
        assert tv < bound


def test_is_key(keys: split) -> None:
    """Test _jaxext.is_key."""
    # JAX keys should be recognized
    key = keys.pop()
    assert is_key(key)

    # Array of keys should be recognized
    assert is_key(keys.pop((2, 5)))

    # Non-JAX objects should not be recognized
    assert not is_key(42)
    assert not is_key(3.14)
    assert not is_key('not a key')
    assert not is_key(None)
    assert not is_key([1, 2, 3])
    assert not is_key({'a': 1})

    # JAX arrays that are not keys should not be recognized
    assert not is_key(jnp.array([1, 2, 3]))
    assert not is_key(jnp.zeros((2,), dtype=jnp.uint32))
    assert not is_key(jnp.ones(()))

    # NumPy arrays should not be recognized
    assert not is_key(numpy.array([1, 2, 3]))


def make_broken_replicated_array(
    x: Shaped[Array, '*shape'], axis_name: str, mesh: Mesh
) -> Shaped[Array, '*shape']:
    """Replicate `x` across devices, but make it different on each device across an axis."""

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=PartitionSpec(),
        out_specs=PartitionSpec(),
        # this disables the check that would notice the inconsistency
        check_vma=False,
    )
    def breaker(x: Shaped[Array, '*shape']) -> Shaped[Array, '*shape']:
        return x + lax.axis_index(axis_name)

    return breaker(x)


def test_make_broken_replicated_array() -> None:
    """Test `make_broken_replicated_array`."""
    nd = get_device_count()
    if nd < 2:  # branch covered in single jax cpu test config
        pytest.skip('Requires at least 2 devices')
    mesh = make_mesh(
        (nd,), ('a',), axis_types=(AxisType.Auto,), devices=get_default_devices()
    )
    x = jnp.arange(nd)
    xb = make_broken_replicated_array(x, 'a', mesh)
    for i, shard in enumerate(xb.addressable_shards):
        data: Shaped[Array, '...'] = shard.data
        if i == 0:
            assert_array_equal(data, x, strict=True)
        else:
            assert jnp.all(data != x)


@pytest.mark.parametrize('equal', [True, False])
@pytest.mark.parametrize('replicated', [True, False])
def test_equal_shards(equal: bool, replicated: bool) -> None:
    """Test `_jaxext.equal_shards`."""
    nd = get_device_count()
    if nd < 2:  # branch covered in single jax cpu test config
        pytest.skip('Requires at least 2 devices')

    # define mesh
    mesh = make_mesh(
        (nd,), ('a',), axis_types=(AxisType.Auto,), devices=get_default_devices()
    )

    # create dummy array
    if equal:
        x = jnp.zeros(nd)
    elif replicated:
        x = jnp.zeros(nd)
        x = make_broken_replicated_array(x, 'a', mesh)
    else:
        x = jnp.arange(nd)

    # shard x
    spec = PartitionSpec() if replicated else PartitionSpec('a')
    sharding = NamedSharding(mesh, spec)
    x = device_put(x, sharding)

    # check the shards are equal or different
    result = equal_shards(x, 'a', mesh=mesh, in_specs=spec)
    assert result.item() == equal
