# bartz/tests/rbartpackages/BART.py
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

"""Wrapper for the R package BART."""

from typing import NamedTuple, TypedDict, cast

import numpy as np
from jaxtyping import AbstractDtype, Bool, Float64, Int32
from numpy import ndarray
from rpy2.rlike.container import NamedList

from tests.rbartpackages._base import RObjectBase, rmethod


class TreeDraws(TypedDict):
    """Type of the `treedraws` attribute of `mc_gbart`."""

    cutpoints: dict[int | str, Float64[ndarray, ' numcut[i]']]
    trees: str


class String(AbstractDtype):
    """Represent a `numpy.str_` data dtype."""

    dtypes = r'<U\d+'


class ProcTime(NamedTuple):
    """Python representation of the output of R's `proc.time`."""

    user_self: float
    sys_self: float
    elapsed: float
    user_child: float
    sys_child: float


class mc_gbart(RObjectBase):
    """Using the `x_test` argument may create problems, try not passing it."""

    _rfuncname = 'BART::mc.gbart'

    hostname: Bool[ndarray, ' mc_cores'] | String[ndarray, ' mc_cores']
    ndpost: int
    offset: float
    prob_test: None | Float64[ndarray, 'ndpost/mc_cores m'] = None
    prob_test_mean: None | Float64[ndarray, ' m'] = None
    prob_train: None | Float64[ndarray, 'ndpost/mc_cores n'] = None
    prob_train_mean: None | Float64[ndarray, ' n'] = None
    proc_time: ProcTime
    rm_const: Int32[ndarray, '<=p']
    sigma: (
        Float64[ndarray, ' nskip+ndpost']
        | Float64[ndarray, 'nskip+ndpost/mc_cores mc_cores']
        | None
    ) = None
    sigma_mean: float | None = None
    treedraws: TreeDraws
    varcount: Int32[ndarray, 'ndpost p']
    varcount_mean: Float64[ndarray, ' p']
    varprob: Float64[ndarray, 'ndpost p']
    varprob_mean: Float64[ndarray, ' p']
    yhat_test: Float64[ndarray, 'ndpost m'] | None = None
    yhat_test_mean: Float64[ndarray, ' m'] | None = None
    yhat_train: Float64[ndarray, 'ndpost n']
    yhat_train_mean: Float64[ndarray, ' n'] | None = None

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)

        # fix up attributes
        self.ndpost = self.ndpost.astype(int).item()
        self.offset = self.offset.item()
        self.proc_time = ProcTime(*map(float, self.proc_time))

        if np.all(self.rm_const < 0):
            _, p = self.varcount.shape
            rm_const = np.ones(p, bool)
            rm_const[-self.rm_const - 1] = False
            self.rm_const = np.arange(p, dtype=np.int32)[rm_const]
        elif np.all(self.rm_const > 0):
            self.rm_const -= 1
        else:
            msg = 'failed to parse rm.const because indices change sign'
            raise ValueError(msg)

        if self.sigma_mean is not None:
            self.sigma_mean = self.sigma_mean.item()

        r_treedraws = cast(NamedList, self.treedraws)
        cutpoints: NamedList = r_treedraws.getbyname('cutpoints')
        self.treedraws = {
            'cutpoints': {
                i if it.name is None else it.name.item(): it.value
                for i, it in enumerate(cutpoints.items())
            },
            'trees': r_treedraws.getbyname('trees').item(),
        }

    @rmethod
    def predict(
        self, newdata: Float64[ndarray, 'm p'], *args, **kwargs
    ) -> Float64[ndarray, 'ndpost/mc_cores m']:
        """Compute predictions."""
        ...


class bartModelMatrix(RObjectBase):  # noqa: D101 because the R doc is added automatically
    _rfuncname = 'BART::bartModelMatrix'


class gbart(mc_gbart):  # noqa: D101 because the R doc is added automatically
    _rfuncname = 'BART::gbart'

    sigma: Float64[ndarray, ' nskip+ndpost'] | None = None
