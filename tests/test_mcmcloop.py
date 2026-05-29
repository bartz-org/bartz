# bartz/tests/test_mcmcloop.py
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

"""Test `bartz.mcmcloop`."""

from dataclasses import replace
from functools import partial
from typing import Any

import pytest
from equinox import filter_jit
from jax import NamedSharding, debug_key_reuse, device_put, jit, make_mesh, tree, vmap
from jax import numpy as jnp
from jax.sharding import AxisType, PartitionSpec
from jax.tree_util import KeyPath
from jaxtyping import Array, Float32, UInt8
from pytest import FixtureRequest  # noqa: PT013

from bartz._jaxext import get_default_device, split
from bartz.mcmcloop import BurninTrace, MainTrace, run_mcmc
from bartz.mcmcstep import State, init, make_p_nonterminal
from bartz.mcmcstep._state import trace_sample_axes
from tests.util import assert_array_equal


def gen_data(
    p: int, n: int, k: int | None
) -> tuple[
    UInt8[Array, '{p} {n}'],
    Float32[Array, ' {n}'] | Float32[Array, '{k} {n}'],
    UInt8[Array, ' {p}'],
]:
    """Generate pretty nonsensical data."""
    X = jnp.arange(p * n, dtype=jnp.uint8).reshape(p, n)
    X = vmap(jnp.roll)(X, jnp.arange(p))
    max_split = jnp.full(p, 255, jnp.uint8)
    if k is None:
        shift = 0
    else:
        shift = jnp.linspace(0, 2 * jnp.pi, k, endpoint=False)[:, None]
    y = jnp.cos(jnp.linspace(0, 2 * jnp.pi / 32 * n, n) + shift)
    return X, y, max_split


@filter_jit
def simple_init(
    p: int, n: int, ntree: int, k: int | None = None, **kwargs: Any
) -> State:
    """Simplified version of `bartz.mcmcstep.init` with data pre-filled."""
    X, y, max_split = gen_data(p, n, k)
    eye = 1.0 if k is None else jnp.eye(k)
    return init(
        X=X,
        y=y,
        offset=0.0 if k is None else jnp.zeros(k),
        max_split=max_split,
        num_trees=ntree,
        p_nonterminal=make_p_nonterminal(6),
        leaf_prior_cov_inv=eye,
        error_cov_df=2,
        error_cov_scale=2 * eye,
        min_points_per_decision_node=10,
        target_platform=get_default_device().platform,
        **kwargs,
    )


def cat_traces(
    trace_a: MainTrace | BurninTrace, trace_b: MainTrace | BurninTrace
) -> MainTrace | BurninTrace:
    """Concatenate two traces along their per-leaf sample axis."""
    sample_axes = trace_sample_axes(trace_a)

    def cat(a: Array, b: Array, axis: int | None) -> Array:
        if axis is None:
            assert_array_equal(a, b)
            return a
        return jnp.concatenate([a, b], axis=axis)

    return tree.map(cat, trace_a, trace_b, sample_axes)


class TestRunMcmc:
    """Test `mcmcloop.run_mcmc`."""

    @pytest.fixture(params=[None, 0, 1, 4], scope='class')
    def num_chains(self, request: FixtureRequest) -> int | None:
        """Return number of chains."""
        return request.param

    @pytest.fixture(params=[None, 0, 1, 4], scope='class')
    def k(self, request: FixtureRequest) -> int | None:
        """Return number of outcomes."""
        return request.param

    @pytest.fixture(scope='class')
    def initial_state(self, num_chains: int | None, k: int | None) -> State:
        """Prepare state for tests."""
        return simple_init(10, 100, 20, k, num_chains=num_chains)

    def test_final_state_overflow(self, keys: split, initial_state: State) -> None:
        """Check that the final state is the one in the trace even if there's overflow."""
        initial_state_copy = tree.map(jnp.copy, initial_state)  # donated
        with debug_key_reuse(False):
            final_state, _, main_trace = run_mcmc(
                keys.pop(), initial_state_copy, 10, inner_loop_length=9
            )

        sample_axes = trace_sample_axes(main_trace)

        def last_sample(arr: Array, axis: int) -> Array:
            return jnp.take(arr, -1, axis=axis)

        assert_array_equal(
            final_state.forest.leaf_tree,
            last_sample(main_trace.leaf_tree, sample_axes.leaf_tree),
        )
        assert_array_equal(
            final_state.forest.var_tree,
            last_sample(main_trace.var_tree, sample_axes.var_tree),
        )
        assert_array_equal(
            final_state.forest.split_tree,
            last_sample(main_trace.split_tree, sample_axes.split_tree),
        )
        assert_array_equal(
            final_state.error_cov_inv,
            last_sample(main_trace.error_cov_inv, sample_axes.error_cov_inv),
        )

    def test_zero_iterations(self, keys: split, initial_state: State) -> None:
        """Check 0 iterations produces a noop."""
        initial_state_copy = tree.map(jnp.copy, initial_state)  # donated
        with debug_key_reuse(False):
            final_state, burnin_trace, main_trace = run_mcmc(
                keys.pop(), initial_state_copy, 0, n_burn=0
            )

        tree.map(partial(assert_array_equal, strict=True), initial_state, final_state)

        def assert_empty_trace(
            _path: KeyPath, x: Array | None, sample_axis: int | None
        ) -> None:
            if x is not None and sample_axis is not None:
                assert x.shape[sample_axis] == 0

        def check_trace(trace: MainTrace | BurninTrace) -> None:
            tree.map_with_path(
                assert_empty_trace,
                trace,
                trace_sample_axes(trace),
                is_leaf=lambda x: x is None,
            )

        check_trace(burnin_trace)
        check_trace(main_trace)

    def test_predicted_double_compilation(self, keys: split) -> None:
        """Check that an error is raised under jit if the configuration would lead to double compilation."""
        initial_state = simple_init(10, 100, 20)

        compiled_run_mcmc = jit(
            run_mcmc, static_argnames=('n_save', 'inner_loop_length')
        )

        msg = r'there are more than 1 outer loops'
        with pytest.raises(RuntimeError, match=msg):
            compiled_run_mcmc(keys.pop(), initial_state, 2, inner_loop_length=1)

    def test_detected_double_compilation(self, keys: split) -> None:
        """Check that double compilation is detected."""
        state = simple_init(10, 100, 20)

        mesh = make_mesh((1,), ('a',), axis_types=(AxisType.Auto,))
        sharding = NamedSharding(mesh, PartitionSpec())
        resid = device_put(state.resid, sharding)
        state = replace(state, resid=resid)

        with pytest.raises(
            RuntimeError, match='The inner loop of `run_mcmc` was traced more than once'
        ):
            run_mcmc(keys.pop(), state, 2, inner_loop_length=1)

    def test_steps_done_increments(self, keys: split, initial_state: State) -> None:
        """Check the global step counter advances by the number of MCMC updates."""
        n_burn, n_skip, n_save = 3, 2, 4
        with debug_key_reuse(False):
            final_state, *_ = run_mcmc(
                keys.pop(),
                tree.map(jnp.copy, initial_state),  # donated
                n_save,
                n_burn=n_burn,
                n_skip=n_skip,
            )
        expected = initial_state.config.steps_done + n_burn + n_skip * n_save
        assert_array_equal(final_state.config.steps_done, expected)

    def test_restartable(self, keys: split, initial_state: State) -> None:
        """Check chaining `run_mcmc` with the same key reproduces a single run.

        The key is folded with the state's persistent step counter, so resuming
        from the output state with the same key continues the exact same key
        sequence, no matter how the total number of iterations is split.
        """
        key = keys.pop()
        with debug_key_reuse(False):
            # one run of 5 iterations
            final_single, _, main_single = run_mcmc(
                key, tree.map(jnp.copy, initial_state), 5, n_burn=0, n_skip=1
            )
            # the same 5 iterations split as 2 + 3, resuming from the output state
            mid, _, main_a = run_mcmc(
                key, tree.map(jnp.copy, initial_state), 2, n_burn=0, n_skip=1
            )
            final_split, _, main_b = run_mcmc(key, mid, 3, n_burn=0, n_skip=1)

        tree.map(assert_array_equal, final_single, final_split)
        tree.map(assert_array_equal, main_single, cat_traces(main_a, main_b))

    def test_inner_loop_length_invariance(
        self, keys: split, initial_state: State
    ) -> None:
        """Check the inner loop chunking does not affect the results."""
        key = keys.pop()
        with debug_key_reuse(False):
            final_whole, burnin_whole, main_whole = run_mcmc(
                key, tree.map(jnp.copy, initial_state), 4, n_burn=2, n_skip=1
            )
            final_chunked, burnin_chunked, main_chunked = run_mcmc(
                key,
                tree.map(jnp.copy, initial_state),
                4,
                n_burn=2,
                n_skip=1,
                inner_loop_length=1,
            )

        tree.map(assert_array_equal, final_whole, final_chunked)
        tree.map(assert_array_equal, main_whole, main_chunked)
        tree.map(assert_array_equal, burnin_whole, burnin_chunked)
