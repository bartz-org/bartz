# bartz/src/bartz/bcf/_trace.py
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

"""Trace dataclasses of the BCF MCMC, for `run_mcmc`."""

from dataclasses import replace

from jaxtyping import Array, Float32

from bartz._jaxext import field
from bartz.bcf._state import BCFState, swap_mu_tau_forests
from bartz.mcmcloop._trace import BurninTrace, MainTrace, Trace
from bartz.mcmcstep._axes import CHAIN_AXIS
from bartz.mcmcstep._state import State


class BCFBurninTrace(Trace):
    """Burn-in trace of the BCF MCMC, the per-forest diagnostics and the scalar parameters."""

    mu: BurninTrace
    """The trace of the prognostic forest."""

    tau: BurninTrace
    """The trace of the treatment forest."""

    tau_0: Float32[Array, '*chains_and_samples'] = field(chains=CHAIN_AXIS, samples=0)
    """The treatment effect intercept."""

    b: Float32[Array, '*chains_and_samples 2'] = field(chains=CHAIN_AXIS, samples=0)
    """The adaptive coding weights for untreated and treated units."""

    @classmethod
    def from_state(cls, state: State) -> 'BCFBurninTrace':
        """Create a single-item burn-in trace from a BCF state."""
        assert isinstance(state, BCFState)
        return cls(
            mu=BurninTrace.from_state(state),
            tau=BurninTrace.from_state(swap_mu_tau_forests(state)),
            tau_0=state.tau_0,
            b=state.b,
        )

    def finalize(self) -> 'BCFBurninTrace':
        """Finalize the traces of the two forests."""
        return replace(self, mu=self.mu.finalize(), tau=self.tau.finalize())


class BCFMainTrace(BCFBurninTrace):
    """Main trace of the BCF MCMC, with the trees of both forests."""

    mu: MainTrace
    """The trace of the prognostic forest."""

    tau: MainTrace
    """The trace of the treatment forest."""

    @classmethod
    def from_state(cls, state: State) -> 'BCFMainTrace':
        """Create a single-item main trace from a BCF state."""
        assert isinstance(state, BCFState)
        kw: dict = dict(
            vars(BCFBurninTrace.from_state(state)),
            mu=MainTrace.from_state(state),
            tau=MainTrace.from_state(swap_mu_tau_forests(state)),
        )
        return cls(**kw)
