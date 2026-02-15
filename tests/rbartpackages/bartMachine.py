# bartz/tests/rbartpackages/bartMachine.py
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

"""Python wrapper of the R package bartMachine."""

# ruff: noqa: D102, ANN201, ANN002, ANN003

from rpy2 import robjects

from tests.rbartpackages._base import RObjectBase, rmethod


class bartMachine(RObjectBase):  # noqa: D101, because the doc is pulled from R
    _rfuncname = 'bartMachine::bartMachine'

    def __init__(
        self, *args, num_cores: int | None = None, megabytes: int = 5000, **kw
    ) -> None:
        robjects.r(f'options(java.parameters = "-Xmx{megabytes:d}m")')
        robjects.r('loadNamespace("bartMachine")')
        if num_cores is not None:
            robjects.r(f'bartMachine::set_bart_machine_num_cores({int(num_cores)})')
        super().__init__(*args, **kw)

    @rmethod
    def predict(self, *args, **kw): ...

    @rmethod
    def get_posterior(self, *args, **kw): ...

    @rmethod
    def get_sigsqs(self, *args, **kw): ...
