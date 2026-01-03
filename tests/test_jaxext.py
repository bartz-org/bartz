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

"""Test bartz.jaxext."""

import numpy
import pytest
from jax import debug_infs, jit, random
from jax import numpy as jnp
from jax.scipy.special import ndtri
from numpy.testing import assert_allclose
from scipy.stats import invgamma as scipy_invgamma
from scipy.stats import ks_1samp, truncnorm

from bartz import jaxext
from bartz.jaxext.scipy.special import ndtri as patched_ndtri
from bartz.jaxext.scipy.stats import invgamma


class TestUnique:
    """Test jaxext.unique."""

    def test_sort(self):
        """Check that it's equivalent to sort if no values are repeated."""
        x = jnp.arange(10)[::-1]
        out, length = jaxext.unique(x, x.size, 666)
        numpy.testing.assert_array_equal(jnp.sort(x), out)
        assert out.dtype == x.dtype
        assert length == x.size

    def test_fill(self):
        """Check that the trailing fill value is used correctly."""
        x = jnp.ones(10)
        out, length = jaxext.unique(x, x.size, 666)
        numpy.testing.assert_array_equal([1] + 9 * [666], out)
        assert out.dtype == x.dtype
        assert length == 1

    def test_empty_input(self):
        """Check that the function works on empty input."""
        x = jnp.array([])
        out, length = jaxext.unique(x, 2, 666)
        numpy.testing.assert_array_equal([666, 666], out)
        assert out.dtype == x.dtype
        assert length == 0

    def test_empty_output(self):
        """Check that the function works if the output is forced to be empty."""
        x = jnp.array([1, 1, 1])
        out, length = jaxext.unique(x, 0, 666)
        numpy.testing.assert_array_equal([], out)
        assert out.dtype == x.dtype
        assert length == 0


class TestAutoBatch:
    """Test jaxext.autobatch."""

    @pytest.mark.parametrize('target_nbatches', [1, 7])
    @pytest.mark.parametrize('with_margin', [False, True])
    @pytest.mark.parametrize('additional_size', [3, 0])
    def test_batch_size(self, keys, target_nbatches, with_margin, additional_size):
        """Check batch sizes are correct in various conditions."""

        def func(a, b, c):
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
        batched_func = jaxext.autobatch(
            func, batch_nbytes, (0, 0, 1), (0, 1), return_nbatches=True
        )
        batched_func_nobatches = jaxext.autobatch(func, batch_nbytes, (0, 0, 1), (0, 1))

        out1 = func(a, b, c)
        out2, nbatches = batched_func(a, b, c)
        out3 = batched_func_nobatches(a, b, c)

        assert nbatches == target_nbatches

        for o2, o3 in zip(out2, out3, strict=True):
            numpy.testing.assert_array_max_ulp(o2, o3)
        for o1, o2 in zip(out1, out2, strict=True):
            numpy.testing.assert_array_max_ulp(o1, o2)

    def test_unbatched_arg(self):
        """Check the function with batching disabled on a scalar argument."""

        def func(a, b):
            return a + b

        batched_func = jaxext.autobatch(func, 32, (0, None))

        a = jnp.arange(100)
        b = 2

        out1 = func(a, b)
        out2 = batched_func(a, b)

        numpy.testing.assert_array_max_ulp(out1, out2)

    def test_batch_axis_pytree(self):
        """Check the that a batch axis can be specified for a whole sub-pytree."""

        def func(a, b):
            return a + b['foo'] + b['bar']

        batched_func = jaxext.autobatch(func, 32, (None, 0))

        a = 2
        b = dict(foo=jnp.arange(100), bar=jnp.arange(100))

        out1 = func(a, b)
        out2 = batched_func(a, b)

        numpy.testing.assert_array_max_ulp(out1, out2)

    def test_large_batch_warning(self):
        """Check the function emits a warning if the size limit can't be honored."""
        x = jnp.arange(10_000).reshape(10, 1000)

        def f(x):
            return x

        g = jaxext.autobatch(f, 100)
        with pytest.warns(UserWarning, match=' > max_io_nbytes = '):
            g(x)

    def test_empty_values(self):
        """Check that the function works with batchable empty arrays."""
        x = jnp.empty((10, 0))

        def f(x):
            return x

        g = jaxext.autobatch(f, 100, return_nbatches=True)
        y, nbatches = g(x)
        assert nbatches == 1
        assert jnp.all(y == x)

    def test_zero_size(self):
        """Check the function works with a batch axis with length 0."""
        x = jnp.empty((0, 10))

        def f(x):
            return x

        g = jaxext.autobatch(f, 100, return_nbatches=True)
        y, nbatches = g(x)
        assert nbatches == 1
        assert jnp.all(y == x)


def different_keys(keya, keyb):
    """Return True iff two jax random keys are different."""
    return jnp.any(random.key_data(keya) != random.key_data(keyb)).item()


def test_split(keys):
    """Test jaxext.split."""
    key = keys.pop()
    ks = jaxext.split(key, 3)

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

    ks = jaxext.split(random.clone(key), 3)
    key1a = ks.pop()
    key2a = ks.pop(2)
    key3a = ks.pop()

    assert not different_keys(key1, key1a)
    assert not different_keys(random.split(key2), key2a)
    assert not different_keys(key3, key3a)

    ks = jaxext.split(keys.pop(), 1)
    key = ks.pop((2, 3, 5))
    assert key.shape == (2, 3, 5)
    assert len(ks) == 0

    ks = jaxext.split(keys.pop())
    assert len(ks) == 2


class TestJaxPatches:
    """Check that some jax stuff I patch is correct and still to be patched."""

    def test_invgamma_missing(self):
        """Check that jax does not implement the inverse gamma distribution."""
        with pytest.raises(ImportError, match=r'gammainccinv'):
            from jax.scipy.special import gammainccinv  # noqa: F401, PLC0415
        with pytest.raises(ImportError, match=r'invgamma'):
            from jax.scipy.stats import invgamma  # noqa: F401, PLC0415

    def test_invgamma_correct(self, keys):
        """Compare my implementation of invgamma against scipy's."""
        p = random.uniform(keys.pop(), (100,), float, 0.01, 0.99)
        alpha = 3.5
        x0 = scipy_invgamma.ppf(p, alpha)
        x1 = invgamma.ppf(p, alpha)
        assert_allclose(x1, x0, rtol=1e-6)

    @pytest.mark.xfail(reason='Fixed in jax 0.6.2.')
    def test_ndtri_bugged(self, keys):
        """Check that `jax.scipy.special.ndtri` triggers `jax.debug_infs`."""
        x = random.uniform(keys.pop(), (100,), float, 0.01, 0.99)
        with debug_infs(True), pytest.raises(FloatingPointError, match=r'inf'):
            ndtri(x)

    def test_ndtri_correct(self, keys):
        """Check that my copy-pasted ndtri impl is equivalent to the jax one."""
        x = random.uniform(keys.pop(), (100,), float, 0.01, 0.99)
        with debug_infs(False):
            y1 = ndtri(x)
        y2 = patched_ndtri(x)
        assert_allclose(y2, y1, rtol=2e-7, atol=0)  # no atol because in (-∞, ∞)


class TestTruncatedNormalOneSided:
    """Test `jaxext.truncated_normal_onesided`."""

    def test_truncated_normal_incorrect(self, keys):
        """Check that `jax.random.truncated_normal` is wrong out of 5 sigma."""
        nsamples = 1000
        lower, upper = jnp.array([(-100.0, -5.0), (5.0, 100.0)]).T
        x = random.truncated_normal(
            keys.pop(), lower[:, None], upper[:, None], (*lower.shape, nsamples)
        )
        for sample, l, u in zip(x, lower, upper, strict=True):
            test = ks_1samp(sample, truncnorm(l, u).cdf)
            assert test.pvalue < 0.01

    def test_correct(self, keys):
        """Check the samples come from the right distribution."""
        nparams = 20
        nsamples = 1000
        upper = random.bernoulli(keys.pop(), 0.5, (nparams,))
        bound = random.uniform(keys.pop(), (nparams,), float, -10, 10)
        x = jaxext.truncated_normal_onesided(
            keys.pop(), (nparams, nsamples), upper[:, None], bound[:, None]
        )
        for sample, u, b in zip(x, upper, bound, strict=True):
            left = -jnp.inf if u else b
            right = b if u else jnp.inf
            test = ks_1samp(sample, truncnorm(left, right).cdf)
            assert test.pvalue > 0.01

    def test_accurate(self, keys):
        """Check that it does not over/under shoot."""
        x = jaxext.truncated_normal_onesided(
            keys.pop(), (), jnp.bool_(True), jnp.float32(-12)
        )
        assert -12.1 <= x < -12
        x = jaxext.truncated_normal_onesided(
            keys.pop(), (), jnp.bool_(False), jnp.float32(12)
        )
        assert 12 < x <= 12.1

    def test_finite(self, keys):
        """Check that the outputs are always finite."""
        # shape and n_loops combined shall be enough that all possible
        # float32 values in [0, 1) are drawn by random.uniform
        shape = (1_000_000,)
        n_loops = 100

        keys = random.split(keys.pop(), n_loops)

        platform = keys.device.platform
        clip = platform == 'gpu'

        @jit
        def loop_body(key):
            keys = jaxext.split(key, 3)
            upper = random.bernoulli(keys.pop(), 0.5, shape)
            bound = random.uniform(keys.pop(), shape, float, -1, 1)
            return jaxext.truncated_normal_onesided(
                keys.pop(), shape, upper, bound, clip=clip
            )

        for key in keys:
            vals = loop_body(key)
            assert jnp.all(jnp.isfinite(vals))


def test_is_key(keys):
    """Test jaxext.is_key."""
    # JAX keys should be recognized
    key = keys.pop()
    assert jaxext.is_key(key)

    # Array of keys should be recognized
    assert jaxext.is_key(keys.pop((2, 5)))

    # Non-JAX objects should not be recognized
    assert not jaxext.is_key(42)
    assert not jaxext.is_key(3.14)
    assert not jaxext.is_key('not a key')
    assert not jaxext.is_key(None)
    assert not jaxext.is_key([1, 2, 3])
    assert not jaxext.is_key({'a': 1})

    # JAX arrays that are not keys should not be recognized
    assert not jaxext.is_key(jnp.array([1, 2, 3]))
    assert not jaxext.is_key(jnp.zeros((2,), dtype=jnp.uint32))
    assert not jaxext.is_key(jnp.ones(()))

    # NumPy arrays should not be recognized
    assert not jaxext.is_key(numpy.array([1, 2, 3]))
