# bartz/src/bartz/debug/_debuggbart.py
#
# Copyright (c) 2026, The Bartz Contributors
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

"""Debugging utilities. The main functionality is the class `debug_mc_gbart`."""

from dataclasses import replace
from typing import Any

from equinox import error_if
from jax import numpy as jnp
from jax import tree
from jax.sharding import PartitionSpec
from jax.tree_util import tree_map
from jaxtyping import Array, Bool, Float32, Int32, UInt

from bartz.BART import gbart, mc_gbart
from bartz.debug._check import check_trace
from bartz.grove import (
    evaluate_forest,
    forest_depth_distr,
    format_tree,
    points_per_node_distr,
)
from bartz.jaxext import equal_shards
from bartz.mcmcloop import TreesTrace


class debug_mc_gbart(mc_gbart):
    """A subclass of `mc_gbart` that adds debugging functionality.

    Parameters
    ----------
    *args
        Passed to `mc_gbart`.
    check_trees
        If `True`, check all trees with `check_trace` after running the MCMC,
        and assert that they are all valid.
    check_replicated_trees
        If the data is sharded across devices, check that the trees are equal
        on all devices in the final state. Set to `False` to allow jax tracing.
    **kwargs
        Passed to `mc_gbart`.
    """

    def __init__(
        self,
        *args: Any,
        check_trees: bool = True,
        check_replicated_trees: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if check_trees:
            bad = self.check_trees()
            bad_count = jnp.count_nonzero(bad)
            self._bart.__dict__['offset'] = error_if(
                self._bart.offset, bad_count > 0, 'invalid trees found in trace'
            )

        state = self._mcmc_state
        mesh = state.config.mesh
        if check_replicated_trees and mesh is not None and 'data' in mesh.axis_names:
            replicated_forest = replace(state.forest, leaf_indices=None)
            equal = equal_shards(
                replicated_forest, 'data', in_specs=PartitionSpec(), mesh=mesh
            )
            equal_array = jnp.stack(tree.leaves(equal))
            all_equal = jnp.all(equal_array)
            # we could use error_if here for traceability, but last time we
            # tried it hanged on error, maybe it was due to sharding.
            assert all_equal.item(), 'the trees are different across devices'

    def print_tree(
        self, i_chain: int, i_sample: int, i_tree: int, print_all: bool = False
    ) -> None:
        """Print a single tree in human-readable format.

        Parameters
        ----------
        i_chain
            The index of the MCMC chain.
        i_sample
            The index of the (post-burnin) sample in the chain.
        i_tree
            The index of the tree in the sample.
        print_all
            If `True`, also print the content of unused node slots.
        """
        tree = TreesTrace.from_dataclass(self._main_trace)
        tree = tree_map(lambda x: x[i_chain, i_sample, i_tree, :], tree)
        s = format_tree(tree, print_all=print_all)
        print(s)  # noqa: T201, this method is intended for debug

    def sigma_harmonic_mean(self, prior: bool = False) -> Float32[Array, ' mc_cores']:
        """Return the harmonic mean of the error variance.

        Parameters
        ----------
        prior
            If `True`, use the prior distribution, otherwise use the full
            conditional at the last MCMC iteration.

        Returns
        -------
        The harmonic mean 1/E[1/sigma^2] in the selected distribution.
        """
        bart = self._mcmc_state
        assert bart.error_cov_df is not None
        assert bart.z is None
        # inverse gamma prior: alpha = df / 2, beta = scale / 2
        if prior:
            alpha = bart.error_cov_df / 2
            beta = bart.error_cov_scale / 2
        else:
            alpha = bart.error_cov_df / 2 + bart.resid.size / 2
            norm2 = jnp.einsum('ij,ij->i', bart.resid, bart.resid)
            beta = bart.error_cov_scale / 2 + norm2 / 2
        error_cov_inv = alpha / beta
        return jnp.sqrt(jnp.reciprocal(error_cov_inv))

    def compare_resid(
        self,
    ) -> tuple[Float32[Array, 'mc_cores n'], Float32[Array, 'mc_cores n']]:
        """Re-compute residuals to compare them with the updated ones.

        Returns
        -------
        resid1 : Float32[Array, 'mc_cores n']
            The final state of the residuals updated during the MCMC.
        resid2 : Float32[Array, 'mc_cores n']
            The residuals computed from the final state of the trees.
        """
        bart = self._mcmc_state
        resid1 = bart.resid

        forests = TreesTrace.from_dataclass(bart.forest)
        trees = evaluate_forest(bart.X, forests, sum_batch_axis=-1)

        if bart.z is not None:
            ref = bart.z
        else:
            ref = bart.y
        resid2 = ref - (trees + bart.offset)

        return resid1, resid2

    def avg_acc(
        self,
    ) -> tuple[Float32[Array, ' mc_cores'], Float32[Array, ' mc_cores']]:
        """Compute the average acceptance rates of tree moves.

        Returns
        -------
        acc_grow : Float32[Array, 'mc_cores']
            The average acceptance rate of grow moves.
        acc_prune : Float32[Array, 'mc_cores']
            The average acceptance rate of prune moves.
        """
        trace = self._main_trace

        def acc(prefix: str) -> Float32[Array, ' mc_cores']:
            acc = getattr(trace, f'{prefix}_acc_count')
            prop = getattr(trace, f'{prefix}_prop_count')
            return acc.sum(axis=1) / prop.sum(axis=1)

        return acc('grow'), acc('prune')

    def avg_prop(
        self,
    ) -> tuple[Float32[Array, ' mc_cores'], Float32[Array, ' mc_cores']]:
        """Compute the average proposal rate of grow and prune moves.

        Returns
        -------
        prop_grow : Float32[Array, 'mc_cores']
            The fraction of times grow was proposed instead of prune.
        prop_prune : Float32[Array, 'mc_cores']
            The fraction of times prune was proposed instead of grow.

        Notes
        -----
        This function does not take into account cases where no move was
        proposed.
        """
        trace = self._main_trace

        def prop(prefix: str) -> Array:
            return getattr(trace, f'{prefix}_prop_count').sum(axis=1)

        pgrow = prop('grow')
        pprune = prop('prune')
        total = pgrow + pprune
        return pgrow / total, pprune / total

    def avg_move(
        self,
    ) -> tuple[Float32[Array, ' mc_cores'], Float32[Array, ' mc_cores']]:
        """Compute the move rate.

        Returns
        -------
        rate_grow : Float32[Array, 'mc_cores']
            The fraction of times a grow move was proposed and accepted.
        rate_prune : Float32[Array, 'mc_cores']
            The fraction of times a prune move was proposed and accepted.
        """
        agrow, aprune = self.avg_acc()
        pgrow, pprune = self.avg_prop()
        return agrow * pgrow, aprune * pprune

    def depth_distr(self) -> Int32[Array, 'mc_cores ndpost/mc_cores d']:
        """Histogram of tree depths for each state of the trees.

        Returns
        -------
        A matrix where each row contains a histogram of tree depths.
        """
        out: Int32[Array, '*chains samples d']
        out = forest_depth_distr(self._main_trace.split_tree)
        if out.ndim < 3:
            out = out[None, :, :]
        return out

    def _points_per_node_distr(
        self, node_type: str
    ) -> Int32[Array, 'mc_cores ndpost/mc_cores n+1']:
        out: Int32[Array, '*chains samples n+1']
        out = points_per_node_distr(
            self._mcmc_state.X,
            self._main_trace.var_tree,
            self._main_trace.split_tree,
            node_type,
            sum_batch_axis=-1,
        )
        if out.ndim < 3:
            out = out[None, :, :]
        return out

    def points_per_decision_node_distr(
        self,
    ) -> Int32[Array, 'mc_cores ndpost/mc_cores n+1']:
        """Histogram of number of points belonging to parent-of-leaf nodes.

        Returns
        -------
        For each chain, a matrix where each row contains a histogram of number of points.
        """
        return self._points_per_node_distr('leaf-parent')

    def points_per_leaf_distr(self) -> Int32[Array, 'mc_cores ndpost/mc_cores n+1']:
        """Histogram of number of points belonging to leaves.

        Returns
        -------
        A matrix where each row contains a histogram of number of points.
        """
        return self._points_per_node_distr('leaf')

    def check_trees(self) -> UInt[Array, 'mc_cores ndpost/mc_cores ntree']:
        """Apply `check_trace` to all the tree draws."""
        out: UInt[Array, '*chains samples num_trees']
        out = check_trace(self._main_trace, self._mcmc_state.forest.max_split)
        if out.ndim < 3:
            out = out[None, :, :]
        return out

    def tree_goes_bad(self) -> Bool[Array, 'mc_cores ndpost/mc_cores ntree']:
        """Find iterations where a tree becomes invalid.

        Returns
        -------
        A where (i,j) is `True` if tree j is invalid at iteration i but not i-1.
        """
        bad = self.check_trees().astype(bool)
        bad_before = jnp.pad(bad[:, :-1, :], [(0, 0), (1, 0), (0, 0)])
        return bad & ~bad_before


class debug_gbart(debug_mc_gbart, gbart):
    """A subclass of `gbart` that adds debugging functionality.

    Parameters
    ----------
    *args
        Passed to `gbart`.
    check_trees
        If `True`, check all trees with `check_trace` after running the MCMC,
        and assert that they are all valid.
    check_replicated_trees
        If the data is sharded across devices, check that the trees are equal
        on all devices in the final state. Set to `False` to allow jax tracing.
    **kw
        Passed to `gbart`.
    """
