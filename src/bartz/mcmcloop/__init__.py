# bartz/src/bartz/mcmcloop/__init__.py
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

"""Functions that implement the full BART posterior MCMC loop.

Running the MCMC
----------------
.. autosummary::
    :toctree:

    run_mcmc
    RunMCMCResult
    BurninTrace
    MainTrace
    MainTraceWithTrainPred

Evaluating the trace
--------------------
.. autosummary::
    :toctree:

    evaluate_trace
    compute_varcount
    EvaluableTrace

Progress callbacks
------------------
The entry points are `make_print_callback` and `make_tqdm_callback`.

.. autosummary::
    :toctree:

    make_print_callback
    make_tqdm_callback
    Callback
    CallbackTuple
    PrintCallback
    TqdmCallback
    StatsAccumulator
    StatsReport
"""

# ruff: noqa: F401

from bartz.mcmcloop._callback import (
    CallbackTuple,
    PrintCallback,
    StatsAccumulator,
    StatsReport,
    TqdmCallback,
    make_print_callback,
    make_tqdm_callback,
)
from bartz.mcmcloop._evaluate import EvaluableTrace, compute_varcount, evaluate_trace
from bartz.mcmcloop._loop import Callback, RunMCMCResult, run_mcmc
from bartz.mcmcloop._trace import BurninTrace, MainTrace, MainTraceWithTrainPred
