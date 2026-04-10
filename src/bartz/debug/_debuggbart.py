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

from typing import Any

from jax import numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import Array, Bool, Float32, Int32, UInt

from bartz.BART import gbart, mc_gbart
from bartz.grove import TreesTrace, format_tree


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
            self._bart.check_trees(error=True)
        if check_replicated_trees:
            self._bart.check_replicated_trees()

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
        self, y: Float32[Array, ' n'] | Float32[Array, 'k n'] | None = None
    ) -> tuple[Float32[Array, 'mc_cores n'], Float32[Array, 'mc_cores n']]:
        """Re-compute residuals to compare them with the updated ones."""
        return self._bart.compare_resid(y)

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
        """Histogram of tree depths for each state of the trees."""
        return self._bart.depth_distr()

    def _points_per_node_distr(
        self, node_type: str
    ) -> Int32[Array, 'mc_cores ndpost/mc_cores n+1']:
        return self._bart._points_per_node_distr(node_type)  # noqa: SLF001

    def points_per_decision_node_distr(
        self,
    ) -> Int32[Array, 'mc_cores ndpost/mc_cores n+1']:
        """Histogram of number of points belonging to parent-of-leaf nodes."""
        return self._bart.points_per_decision_node_distr()

    def points_per_leaf_distr(self) -> Int32[Array, 'mc_cores ndpost/mc_cores n+1']:
        """Histogram of number of points belonging to leaves."""
        return self._bart.points_per_leaf_distr()

    def check_trees(self) -> UInt[Array, 'mc_cores ndpost/mc_cores ntree']:
        """Apply `check_trace` to all the tree draws."""
        return self._bart.check_trees()

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
