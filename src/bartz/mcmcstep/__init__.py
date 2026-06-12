# bartz/src/bartz/mcmcstep/__init__.py
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

"""
Functions that implement the BART posterior MCMC initialization and update step.

Initialization and stepping
---------------------------
.. autosummary::
    :toctree:

    init
    step
    make_p_nonterminal
    OutcomeType

MCMC state
----------
.. autosummary::
    :toctree:

    State
    Forest
    StepConfig
    Wishart
    DiagWishart

Reduction strategies
--------------------
Configurations for the per-leaf scatter-add reductions, to pass to `init`.

.. autosummary::
    :toctree:

    ReductionConfig
    BatchedReduction
    OneHotReduction
    PallasReduction
"""

# ruff: noqa: F401

from bartz.mcmcstep._reduction import (
    BatchedReduction,
    OneHotReduction,
    PallasReduction,
    ReductionConfig,
)
from bartz.mcmcstep._state import (
    DiagWishart,
    Forest,
    OutcomeType,
    State,
    StepConfig,
    Wishart,
    init,
    make_p_nonterminal,
)
from bartz.mcmcstep._step import step
