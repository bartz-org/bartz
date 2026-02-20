# bartz/tests/test_meta.py
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

"""Test properties of pytest itself or other utilities."""

from functools import partial

import pytest
from jax import debug_nans, jit, random
from jax import numpy as jnp
from jax.errors import KeyReuseError
from jaxtyping import Array, Float, Key

from bartz.jaxext import split


@pytest.fixture
def keys1(keys: split) -> split:
    """Pass-through the `keys` fixture."""
    return keys


@pytest.fixture
def keys2(keys: split) -> split:
    """Pass-through the `keys` fixture."""
    return keys


def test_random_keys_do_not_depend_on_fixture(keys1: split, keys2: split) -> None:
    """Check that the `keys` fixture is per-test-case, not per-fixture."""
    assert keys1 is keys2


def test_number_of_random_keys(keys: split) -> None:
    """Check the fixed number of available keys.

    This is here just as reference for the `test_random_keys_are_consumed` test
    below.
    """
    assert len(keys) == 128


@pytest.fixture
def consume_one_key(keys: split) -> Key[Array, '']:  # noqa: D103
    return keys.pop()


@pytest.fixture
def consume_another_key(keys: split) -> Key[Array, '']:  # noqa: D103
    return keys.pop()


def test_random_keys_are_consumed(
    consume_one_key: Key[Array, ''],  # noqa: ARG001
    consume_another_key: Key[Array, ''],  # noqa: ARG001
    keys: split,
) -> None:
    """Check that the random keys in `keys` can't be used more than once."""
    assert len(keys) == 126


def test_debug_key_reuse(keys: split) -> None:
    """Check that the jax debug_key_reuse option works."""
    key = keys.pop()
    random.uniform(key)
    with pytest.raises(KeyReuseError):
        random.uniform(key)


def test_debug_key_reuse_within_jit(keys: split) -> None:
    """Check that the jax debug_key_reuse option works within a jitted function."""

    @jit
    def func(key: Key[Array, '']) -> Float[Array, '']:
        return random.uniform(key) + random.uniform(key)

    with pytest.raises(KeyReuseError):
        func(keys.pop())


class TestJaxNoCopyBehavior:
    """Check whether jax makes actual copies of arrays in various conditions."""

    def test_unconditional_buffer_donation(self) -> None:
        """Test jax donates buffers even if they are small."""
        # donation disabled under debug_nans, see jax/issues/#33949
        with debug_nans(False):
            # check buffer donation works unconditionally
            x = jnp.arange(100)
            xp = x.unsafe_buffer_pointer()

            @partial(jit, donate_argnums=(0,))
            def noop(x: Array) -> Array:
                return x

            y = noop(x)
            yp = y.unsafe_buffer_pointer()

            assert xp == yp
            with pytest.raises(RuntimeError, match=r'delete'):
                x[0]

    def test_jnp_array_copy_no_jit(self) -> None:
        """Test jnp.array makes copies outside jitted functions."""
        y = jnp.arange(100)
        yp = y.unsafe_buffer_pointer()

        z = jnp.array(y)
        zp = z.unsafe_buffer_pointer()

        assert zp != yp

    def test_jnp_array_no_copy_jit(self) -> None:
        """Check jnp.array does not make copies within jit."""
        # donation disabled under debug_nans, see jax/issues/#33949
        with debug_nans(False):
            y = jnp.arange(100)
            yp = y.unsafe_buffer_pointer()

            @partial(jit, donate_argnums=(0,))
            def array(x: Array) -> Array:
                return jnp.array(x)

            q = array(y)
            qp = q.unsafe_buffer_pointer()

            assert qp == yp
