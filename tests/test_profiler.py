# bartz/tests/test_profiler.py
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

"""Test `bartz._profiler`."""

from cProfile import Profile
from functools import partial
from pstats import Stats
from time import perf_counter, sleep

import pytest
from jax import debug_infs, debug_nans, jit, pure_callback, random
from jax import numpy as jnp
from numpy.testing import assert_array_equal

from bartz._profiler import (
    cond_if_not_profiling,
    get_profile_mode,
    jit_and_block_if_profiling,
    jit_if_not_profiling,
    profile_mode,
    scan_if_not_profiling,
    set_profile_mode,
)
from bartz.jaxext import get_default_device


class TestFlag:
    """Test the functionality of the global profile mode flag."""

    def test_initial_state(self):
        """Check profiling mode is off by default."""
        assert not get_profile_mode()

    def test_getter_setter(self):
        """Test setting and getting the profile mode."""
        set_profile_mode(True)
        assert get_profile_mode()
        set_profile_mode(False)
        assert not get_profile_mode()

    def test_context_manager(self):
        """Test the profile mode context manager."""
        with profile_mode(True):
            assert get_profile_mode()
        assert not get_profile_mode()

        set_profile_mode(True)
        with profile_mode(False):
            assert not get_profile_mode()
        assert get_profile_mode()
        set_profile_mode(False)

        with profile_mode(True):
            assert get_profile_mode()
            with profile_mode(False):
                assert not get_profile_mode()
            assert get_profile_mode()
        assert not get_profile_mode()


class TestScanIfNotProfiling:
    """Test `scan_if_not_profiling`."""

    @pytest.mark.parametrize('mode', [True, False])
    def test_result(self, mode: bool):
        """Test that `scan_if_not_profiling` has the right output on a simple example."""

        def body(carry, _):
            return carry + 1, None

        with profile_mode(mode):
            carry, ys = scan_if_not_profiling(body, 0, None, 5)
            assert ys is None
            assert carry == 5

    def test_does_not_jit(self):
        """Check that `scan_if_not_profiling` does not jit the function in profiling mode."""

        def body(carry, _):
            return carry.block_until_ready(), None
            # block_until_ready errors under jit

        with profile_mode(True):
            scan_if_not_profiling(body, jnp.int32(0), None, 5)

        with pytest.raises(
            AttributeError,
            match='DynamicJaxprTracer has no attribute block_until_ready',
        ):
            scan_if_not_profiling(body, 0, None, 5)


class TestCondIfNotProfiling:
    """Test `cond_if_not_profiling`."""

    @pytest.mark.parametrize('mode', [True, False])
    @pytest.mark.parametrize('pred', [True, False])
    def test_result(self, mode: bool, pred: bool):
        """Test that `cond_if_not_profiling` has the right output on a simple example."""
        with profile_mode(mode):
            out = cond_if_not_profiling(
                pred, lambda x: x - 1, lambda x: x + 1, jnp.int32(5)
            )
            assert out == (4 if pred else 6)

    def test_does_not_jit(self):
        """Check that `cond_if_not_profiling` does not jit the function in profiling mode."""
        with profile_mode(True):
            cond_if_not_profiling(
                True,
                lambda x: x.block_until_ready(),
                lambda x: x.block_until_ready(),
                jnp.int32(5),
            )

        with pytest.raises(
            AttributeError,
            match='DynamicJaxprTracer has no attribute block_until_ready',
        ):
            cond_if_not_profiling(
                False,
                lambda x: x.block_until_ready(),
                lambda x: x.block_until_ready(),
                jnp.int32(5),
            )


class TestJitIfNotProfiling:
    """Test `jit_if_not_profiling`."""

    @pytest.mark.parametrize('mode', [True, False])
    def test_result(self, mode: bool):
        """Test that `jit_if_not_profiling` has the right output in both modes."""

        def func(x):
            return x * 2 + 1

        jitted_func = jit_if_not_profiling(func)

        with profile_mode(mode):
            result = jitted_func(5)
            assert result == 11

    def test_does_not_jit(self):
        """Check that `jit_if_not_profiling` does not jit the function in profiling mode."""

        def func(x):
            return x.block_until_ready()
            # block_until_ready errors under jit

        jitted_func = jit_if_not_profiling(func)

        with profile_mode(True):
            result = jitted_func(jnp.int32(42))
            assert result == 42

        with pytest.raises(
            AttributeError,
            match='DynamicJaxprTracer has no attribute block_until_ready',
        ):
            jitted_func(jnp.int32(42))


class TestJitAndBlockIfProfiling:
    """Test `jit_and_block_if_profiling`."""

    @pytest.mark.parametrize('mode', [True, False])
    def test_result(self, mode: bool):
        """Test that `jit_and_block_if_profiling` has the right output in both modes."""

        def func(x):
            return x * 2 + 1

        jitted_func = jit_and_block_if_profiling(func)

        with profile_mode(mode):
            result = jitted_func(5)
            assert result == 11

    def test_jits_when_profiling(self):
        """Check that `jit_and_block_if_profiling` jits when profiling is enabled."""

        def func(x):
            return x.block_until_ready()
            # block_until_ready errors under jit

        jitted_func = jit_and_block_if_profiling(func)

        # When profiling is ON, function IS jitted, so should error
        with (
            pytest.raises(
                AttributeError,
                match='DynamicJaxprTracer has no attribute block_until_ready',
            ),
            profile_mode(True),
        ):
            jitted_func(0)

        # When profiling is OFF, function is NOT jitted, so should work
        with profile_mode(False):
            jitted_func(jnp.int32(0))

    def test_static_args(self):
        """Check that it works with static arguments."""

        def func(n: int):
            return jnp.arange(n)

        jitted_func = jit_and_block_if_profiling(func, static_argnums=(0,))

        with profile_mode(True):
            result = jitted_func(5)
            assert_array_equal(result, jnp.arange(5))

    @pytest.mark.flaky(max_runs=3)
    # flaky because it involves comparing time measurements done on the fly
    def test_blocks_execution(self):
        """Check that `jit_and_block_if_profiling` blocks execution when profiling."""
        with debug_nans(False), debug_infs(False):
            platform = get_default_device().platform
            match platform:
                case 'cpu':
                    n = 2000
                case 'gpu':
                    n = 10_000
                case _:  # pragma: no cover
                    msg = f'Unsupported platform for timing test: {platform}'
                    raise RuntimeError(msg)

            func = lambda: idle(n)  # about 50-100 ms
            jit_func = jit(func)
            jab_func = jit_and_block_if_profiling(func)

            # Time the jitted function
            for _ in range(3):
                jit_func().block_until_ready()  # Warm-up
            start = perf_counter()
            jit_func().block_until_ready()
            expected = perf_counter() - start

            # Check execution is async
            start = perf_counter()
            result = jit_func()
            elapsed = perf_counter() - start
            result.block_until_ready()  # Ensure completion
            assert elapsed < expected / 2

            # Test profiling mode first (should block and wait >= expected)
            with profile_mode(True):
                jab_func()  # Warm-up
                start = perf_counter()
                jab_func()
                elapsed = perf_counter() - start
                assert elapsed >= expected * 0.9, (
                    f'Expected blocking to wait >= {expected:#.2g}s, got {elapsed:#.2g}s'
                )

            # Test non-profiling mode (should be async, < expected)
            jab_func().block_until_ready()  # Warm-up
            start = perf_counter()
            result = jab_func()
            elapsed = perf_counter() - start
            result.block_until_ready()  # Ensure completion
            assert elapsed < expected / 2, (
                f'Expected async execution << {expected:#.2g}s, got {elapsed:#.2g}s'
            )

    def test_profile(self):
        """Test `jit_and_block_if_profiling` under the Python profiler."""
        runtime = 0.1

        @jit_and_block_if_profiling
        def awlkugh():  # weird name to make sure identifiers are legit
            x = jnp.int32(0)

            def sleeper(x):
                sleep(runtime)
                return x

            return pure_callback(sleeper, x, x)

        with profile_mode(True):
            for _ in range(2):
                awlkugh()  # warm-up

        with Profile() as prof, profile_mode(True):
            awlkugh()

        stats = Stats(prof).get_stats_profile()

        assert 'awlkugh' not in stats.func_profiles
        # it's not there because it was traced during warm-up

        p_run = stats.func_profiles['jab_inner_wrapper']

        assert runtime < p_run.cumtime < 10 * runtime


@partial(jit, static_argnums=(0,))
def idle(n: int):
    """Waste time in jax computation."""
    key = random.key(0)
    x = random.normal(key, (n, n))
    return x @ x
