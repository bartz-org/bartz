# bartz/tests/rbartpackages/dbarts.py
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

"""Python wrapper of the R package `dbarts`."""

# ruff: noqa: D101, D102, ANN201, ANN002, ANN003

from rpy2 import robjects

from tests.rbartpackages._base import RObjectBase, rmethod


class bart(RObjectBase):
    """

    Python interface to dbarts::bart.

    The named numeric vector form of the `splitprobs` parameter must be
    specified as a dictionary in Python.

    """

    _rfuncname = 'dbarts::bart'
    _split_probs = 'splitprobs'

    def __init__(self, *args, **kw) -> None:
        split_probs = kw.get(self._split_probs)
        if isinstance(split_probs, dict):
            values = list(split_probs.values())
            names = list(split_probs.keys())
            split_probs = robjects.FloatVector(values)
            split_probs = robjects.r('setNames')(split_probs, names)
            kw[self._split_probs] = split_probs

        super().__init__(*args, **kw)

    @rmethod
    def predict(self, *args, **kw): ...

    @rmethod
    def extract(self, *args, **kw): ...

    @rmethod
    def fitted(self, *args, **kw): ...


class bart2(bart):
    """

    Python interface to dbarts::bart2.

    The named numeric vector form of the `split_probs` parameter must be
    specified as a dictionary in Python.

    """

    _rfuncname = 'dbarts::bart2'
    _split_probs = 'split_probs'

    def __init__(self, formula: str, *args, **kw) -> None:
        rformula = robjects.Formula(formula)
        super().__init__(rformula, *args, **kw)


class rbart_vi(bart2):
    """

    Python interface to dbarts::rbart_vi.

    The named numeric vector form of the `split_probs` parameter must be
    specified as a dictionary in Python.

    """

    _rfuncname = 'dbarts::rbart_vi'


class dbarts(RObjectBase):
    _rfuncname = 'dbarts::dbarts'

    @rmethod
    def run(self, *args, **kw): ...

    @rmethod
    def sampleTreesFromPrior(self, *args, **kw): ...

    @rmethod
    def sampleNodeParametersFromPrior(self, *args, **kw): ...

    @rmethod
    def copy(self, *args, **kw): ...

    @rmethod
    def show(self, *args, **kw): ...

    @rmethod
    def predict(self, *args, **kw): ...

    @rmethod
    def setControl(self, *args, **kw): ...

    @rmethod
    def setModel(self, *args, **kw): ...

    @rmethod
    def setData(self, *args, **kw): ...

    @rmethod
    def setResponse(self, *args, **kw): ...

    @rmethod
    def setOffset(self, *args, **kw): ...

    @rmethod
    def setSigma(self, *args, **kw): ...

    @rmethod
    def setPredictor(self, *args, **kw): ...

    @rmethod
    def setTestPredictor(self, *args, **kw): ...

    @rmethod
    def setTestPredictorAndOffset(self, *args, **kw): ...

    @rmethod
    def setTestOffset(self, *args, **kw): ...

    @rmethod
    def printTrees(self, *args, **kw): ...

    @rmethod
    def plotTree(self, *args, **kw): ...


class dbartsControl(RObjectBase):
    _rfuncname = 'dbarts::dbartsControl'
