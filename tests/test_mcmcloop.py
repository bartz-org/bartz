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

from functools import partial

import pytest
from equinox import filter_jit
from jax import debug_key_reuse, jit, vmap
from jax import numpy as jnp
from jax.tree import map_with_path
from jax.tree_util import tree_map
from jaxtyping import Array, Float32, UInt8
from numpy.testing import assert_array_equal
from pytest import FixtureRequest  # noqa: PT013
from pytest_subtests import SubTests

from bartz import profile_mode
from bartz.jaxext import get_default_device, split
from bartz.mcmcloop import run_mcmc
from bartz.mcmcstep import State, init
from bartz.mcmcstep._state import chain_vmap_axes


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


def make_p_nonterminal(maxdepth: int) -> Float32[Array, ' {maxdepth}-1']:
    """Prepare the p_nonterminal argument to `mcmcstep.init`."""
    depth = jnp.arange(maxdepth - 1)
    base = 0.95
    power = 2
    return base / (1 + depth).astype(float) ** power


@filter_jit
def simple_init(p: int, n: int, ntree: int, k: int | None = None, **kwargs) -> State:
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

    def test_final_state_overflow(self, keys: split, initial_state: State):
        """Check that the final state is the one in the trace even if there's overflow."""
        with debug_key_reuse(initial_state.forest.num_chains() != 0):
            final_state, _, main_trace = run_mcmc(
                keys.pop(), initial_state, 10, inner_loop_length=9
            )

        if initial_state.forest.num_chains() is None:
            last_index = -1
        else:
            last_index = (slice(None), -1)

        assert_array_equal(
            final_state.forest.leaf_tree, main_trace.leaf_tree[last_index]
        )
        assert_array_equal(final_state.forest.var_tree, main_trace.var_tree[last_index])
        assert_array_equal(
            final_state.forest.split_tree, main_trace.split_tree[last_index]
        )
        assert_array_equal(
            final_state.error_cov_inv, main_trace.error_cov_inv[last_index]
        )

    def test_zero_iterations(self, keys: split, initial_state: State):
        """Check 0 iterations produces a noop."""
        with debug_key_reuse(initial_state.forest.num_chains() != 0):
            final_state, burnin_trace, main_trace = run_mcmc(
                keys.pop(), initial_state, 0, n_burn=0
            )

        tree_map(partial(assert_array_equal, strict=True), initial_state, final_state)

        def assert_empty_trace(_path, x, chain_axis):
            if initial_state.forest.num_chains() is None or chain_axis is None:
                sample_axis = 0
            else:
                sample_axis = 1
            if x is not None:
                assert x.shape[sample_axis] == 0

        def check_trace(trace):
            map_with_path(
                assert_empty_trace,
                trace,
                chain_vmap_axes(trace),
                is_leaf=lambda x: x is None,
            )

        check_trace(burnin_trace)
        check_trace(main_trace)

    def test_jit_error(self, keys: split, subtests: SubTests):
        """Check that an error is raise under jit in some conditions."""
        initial_state = simple_init(10, 100, 20)

        compiled_run_mcmc = jit(
            run_mcmc, static_argnames=('n_save', 'inner_loop_length')
        )

        msg = r'there are either more than 1 outer loops'

        with subtests.test('outer loops'), pytest.raises(RuntimeError, match=msg):
            compiled_run_mcmc(keys.pop(), initial_state, 2, inner_loop_length=1)

        with (
            subtests.test('profile mode'),
            profile_mode(True),
            pytest.raises(RuntimeError, match=msg),
        ):
            compiled_run_mcmc(keys.pop(), initial_state, 1)
