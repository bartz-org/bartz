# bartz/src/bartz/debug/__init__.py
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

"""
Debugging utilities.

  - `check_trace`: check the validity of a set of trees.
  - `debug_mc_gbart`: version of `mc_gbart` with debug checks and methods.
  - `trees_BART_to_bartz`: convert an R package BART3 trace to a bartz trace.
  - `sample_prior`: sample the bart prior.
"""

# ruff: noqa: F401

from bartz.debug._check import check_trace, describe_error
from bartz.debug._debuggbart import debug_gbart, debug_mc_gbart
from bartz.debug._prior import SamplePriorTrees, sample_prior
from bartz.debug._traceconv import BARTTraceMeta, TraceWithOffset, trees_BART_to_bartz
