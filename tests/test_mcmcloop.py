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

import io
from dataclasses import replace
from functools import partial
from typing import Any, Literal

import pytest
from equinox import filter_jit
from jax import (
    NamedSharding,
    block_until_ready,
    debug_key_reuse,
    device_put,
    jit,
    make_mesh,
    tree,
)
from jax import numpy as jnp
from jax.sharding import AxisType, Mesh, PartitionSpec
from jax.tree_util import KeyPath
from jaxtyping import Array, Shaped, UInt8
from pytest import FixtureRequest  # noqa: PT013

from bartz._jaxext import get_default_devices, get_device_count, split
from bartz.mcmcloop import (
    BurninTrace,
    MainTrace,
    evaluate_trace,
    make_tqdm_callback,
    run_mcmc,
)
from bartz.mcmcloop._callback import _TQDM_REGISTRY, _tqdm_advance
from bartz.mcmcloop._loop import _inner_loop_counter
from bartz.mcmcstep import State, init, make_p_nonterminal
from bartz.mcmcstep._axes import trace_sample_axes
from bartz.testing import gen_nonsense_data
from tests.util import assert_array_equal, assert_close_matrices, nnone


@filter_jit
def simple_init(
    p: int, n: int, ntree: int, k: int | None = None, **kwargs: Any
) -> State:
    """Simplified version of `bartz.mcmcstep.init` with data pre-filled."""
    X, y, max_split = gen_nonsense_data(p, n, k)
    eye = 1.0 if k is None else jnp.eye(k)
    return init(
        X=X,
        y=y,
        offset=0.0 if k is None else jnp.zeros(k),
        max_split=max_split,
        num_trees=ntree,
        p_nonterminal=make_p_nonterminal(6),
        leaf_prior_cov_inv=eye,
        error_cov_df=2.0,
        error_cov_scale=2 * eye,
        min_points_per_decision_node=10,
        **kwargs,
    )


def cat_traces(
    trace_a: MainTrace | BurninTrace, trace_b: MainTrace | BurninTrace
) -> MainTrace | BurninTrace:
    """Concatenate two traces along their per-leaf sample axis."""
    sample_axes = trace_sample_axes(trace_a)

    def cat(
        a: Shaped[Array, '...'], b: Shaped[Array, '...'], axis: int | None
    ) -> Shaped[Array, '...']:
        if axis is None:
            assert_array_equal(a, b)
            return a
        return jnp.concatenate([a, b], axis=axis)

    return tree.map(cat, trace_a, trace_b, sample_axes)


def assert_trace_close(
    actual: Shaped[Array, '*shape'], desired: Shaped[Array, '*shape']
) -> None:
    """Compare state/trace leaves, tolerating GPU floating-point rounding.

    Integer and boolean leaves must match exactly; floating-point leaves are
    compared up to a relative tolerance, because the order of floating-point
    reductions on GPU is not bit-reproducible across different iteration
    chunkings.
    """
    if jnp.issubdtype(actual.dtype, jnp.floating):
        assert_close_matrices(actual, desired, rtol=1e-4, reduce_rank=True)
    else:
        assert_array_equal(actual, desired)


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

        def last_sample(arr: Shaped[Array, '...'], axis: int) -> Shaped[Array, '...']:
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
            _path: KeyPath, x: Shaped[Array, '...'] | None, sample_axis: int | None
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

    @pytest.mark.parametrize('average', [True, False])
    def test_tqdm_callback(
        self, keys: split, initial_state: State, average: bool
    ) -> None:
        """Check the tqdm progress bar runs to completion and is cleaned up.

        Parametrized over ``average`` to also exercise the stats-accumulator
        path where the running sums are absent (``average=False``), so each
        report shows the latest iteration only.
        """
        buf = io.StringIO()
        kw = make_tqdm_callback(
            initial_state, report_every=2, file=buf, mininterval=0, average=average
        )
        bar_id = kw['callback_state'].bar_id.item()
        state = tree.map(jnp.copy, initial_state)  # donated
        with debug_key_reuse(False):
            # block so the (unordered, async) progress callbacks have all fired
            block_until_ready(run_mcmc(keys.pop(), state, 4, n_burn=2, **kw))

        out = buf.getvalue()
        assert '100%' in out  # the bar reached the end
        assert '6/6' in out  # n_burn + n_save * n_skip iterations
        assert bar_id not in _TQDM_REGISTRY  # the bar was closed and removed

    def test_tqdm_callback_cleans_up_interrupted_bar(self) -> None:
        """A new tqdm callback closes a bar left open by an interrupted run."""
        state = simple_init(10, 100, 20)
        buf = io.StringIO()
        kw = make_tqdm_callback(state, file=buf, mininterval=0)
        bar_id = kw['callback_state'].bar_id.item()
        # simulate an interrupted run: advance the bar partway, never finishing
        _tqdm_advance(bar_id, 3, 10)
        assert not nnone(_TQDM_REGISTRY[bar_id].bar).disable  # still open

        make_tqdm_callback(state, file=io.StringIO())  # triggers cleanup
        assert bar_id not in _TQDM_REGISTRY  # the stale bar was closed and dropped
        assert buf.getvalue().endswith('\n')  # closed cleanly, not left mid-line

    def test_tqdm_no_recompilation(self, keys: split) -> None:
        """Check two MCMC runs with a tqdm callback share the compiled inner loop.

        The tqdm bar lives in a module-level registry and is referenced from the
        loop only through an integer handle carried as a traceable scalar, so the
        loop pytree is identical across runs and `_run_mcmc_inner_loop` is not
        retraced (verified via the `_CallCounter`, see
        ``test_no_recompilation_inner_loop_counter`` in ``test_interface``).
        """
        state = simple_init(10, 100, 20)

        def run() -> None:
            kw = make_tqdm_callback(state, disable=True)
            with debug_key_reuse(False):
                run_mcmc(keys.pop(), tree.map(jnp.copy, state), 4, n_burn=2, **kw)

        run()
        run()
        assert _inner_loop_counter.n_calls == 0

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

        mesh = make_mesh(
            (1,), ('a',), axis_types=(AxisType.Auto,), devices=get_default_devices()
        )
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

        tree.map(assert_trace_close, final_single, final_split)
        tree.map(assert_trace_close, main_single, cat_traces(main_a, main_b))

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

        tree.map(assert_trace_close, final_whole, final_chunked)
        tree.map(assert_trace_close, main_whole, main_chunked)
        tree.map(assert_trace_close, burnin_whole, burnin_chunked)


# shared test-point count, reused across cases to hit the jax compilation cache
_N_TEST = 60


def _eval_test_points(n_test: int = _N_TEST) -> UInt8[Array, '6 {n_test}']:
    """Generate quantized test points compatible with `simple_init`'s data."""
    return gen_nonsense_data(6, n_test, None)[0]


def _make_mesh(axes: dict[str, int]) -> Mesh:
    """Build an auto-mode mesh from an ``{axis_name: size}`` mapping."""
    return make_mesh(
        tuple(axes.values()),
        tuple(axes),
        axis_types=(AxisType.Auto,) * len(axes),
        devices=get_default_devices(),
    )


class TestEvaluateTrace:
    """Test test-point batching and sharding in `mcmcloop.evaluate_trace`."""

    @pytest.fixture(params=[None, 4], scope='class')
    def num_chains(self, request: FixtureRequest) -> int | None:
        """Return number of chains."""
        return request.param

    @pytest.fixture(params=[None, 3], scope='class')
    def k(self, request: FixtureRequest) -> int | None:
        """Return number of outcomes."""
        return request.param

    def _trace(
        self, keys: split, num_chains: int | None, k: int | None, mesh: Mesh | None
    ) -> MainTrace:
        state = simple_init(6, 80, 8, k, num_chains=num_chains, mesh=mesh)
        with debug_key_reuse(False):
            return run_mcmc(keys.pop(), state, 5, n_burn=2).main_trace

    def _assert_modes_match(
        self, keys: split, num_chains: int | None, k: int | None, mesh: Mesh | None
    ) -> None:
        """Check the three `test_points` modes agree, and check output sharding."""
        trace = self._trace(keys, num_chains, k, mesh)
        X = _eval_test_points()

        base = evaluate_trace(X, trace, test_points='none')
        auto = evaluate_trace(X, trace, test_points='autobatch')
        assert_close_matrices(auto, base, rtol=1e-5, reduce_rank=True)

        if mesh is not None and 'data' in mesh.axis_names:
            X = device_put(X, NamedSharding(mesh, PartitionSpec(None, 'data')))
        shrd = evaluate_trace(X, trace, test_points='shard_and_autobatch')
        assert_close_matrices(shrd, base, rtol=1e-5, reduce_rank=True)

        # the output `n` axis must be sharded over 'data' when there is one;
        # this assumes the chain axis is leading, so `n` is the last axis
        if mesh is not None and 'data' in mesh.axis_names:
            shard_shape = shrd.sharding.shard_shape(shrd.shape)
            assert shard_shape[-1] == _N_TEST // mesh.shape['data']

    def test_no_mesh(self, keys: split, num_chains: int | None, k: int | None) -> None:
        """Modes agree with no mesh (autobatch with no sharding)."""
        self._assert_modes_match(keys, num_chains, k, None)

    def test_data_mesh(
        self, keys: split, num_chains: int | None, k: int | None
    ) -> None:
        """Modes agree with a data-sharding mesh."""
        if get_device_count() < 2:
            pytest.skip('need at least 2 devices')
        self._assert_modes_match(keys, num_chains, k, _make_mesh({'data': 2}))

    def test_chains_and_data_mesh(
        self, keys: split, num_chains: int | None, k: int | None
    ) -> None:
        """Modes agree with a mesh sharding both chains and data."""
        if num_chains is None:
            pytest.skip('no chains to shard')
        if get_device_count() < 4:
            pytest.skip('need at least 4 devices')
        mesh = _make_mesh({'chains': 2, 'data': 2})
        self._assert_modes_match(keys, num_chains, k, mesh)

    def test_shard_fallback_without_data_axis(self, keys: split) -> None:
        """`shard_and_autobatch` without a 'data' mesh axis falls back to batching."""
        if get_device_count() < 2:
            pytest.skip('need at least 2 devices')
        mesh = _make_mesh({'chains': 2})
        trace = self._trace(keys, 4, None, mesh)
        X = _eval_test_points()
        base = evaluate_trace(X, trace, test_points='none')
        shrd = evaluate_trace(X, trace, test_points='shard_and_autobatch')
        assert_close_matrices(shrd, base, rtol=1e-5, reduce_rank=True)

    @pytest.mark.parametrize('mode', ['autobatch', 'shard_and_autobatch'])
    def test_low_budget_loops_and_warns(
        self, keys: split, mode: Literal['autobatch', 'shard_and_autobatch']
    ) -> None:
        """An absurdly low budget forces every loop to run and stays correct.

        The test points cannot be reduced below one element, so the innermost
        tree autobatch can't honor the limit and warns; catching that warning
        checks the whole batching stack end-to-end.
        """
        if mode == 'shard_and_autobatch' and get_device_count() < 2:
            pytest.skip('need at least 2 devices')
        mesh = _make_mesh({'data': 2}) if mode == 'shard_and_autobatch' else None

        trace = self._trace(keys, None, None, mesh)
        X = _eval_test_points()
        base = evaluate_trace(X, trace, test_points='none')

        if mesh is not None:
            X = device_put(X, NamedSharding(mesh, PartitionSpec(None, 'data')))
        with pytest.warns(UserWarning, match='max_io_nbytes'):
            looped = evaluate_trace(X, trace, test_points=mode, max_io_nbytes=1)
        assert_close_matrices(looped, base, rtol=1e-5)
