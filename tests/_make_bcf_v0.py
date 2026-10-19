# bartz/tests/_make_bcf_v0.py
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

"""Write the archive of format 0, run once with bartz 0.13 outside the repository.

``uv run --isolated --no-project --with bartz==0.13.0 path/to/_make_bcf_v0.py``
"""

from pathlib import Path

import numpy as np
from jax import random

from bartz.bcf import bcf

keys = list(random.split(random.key(2026_10_14_09_46), 5))
n, p = 20, 2
x = random.normal(keys.pop(), (n, p))
z = random.bernoulli(keys.pop(), 0.5, (n,)).astype(float)
pihat = random.uniform(keys.pop(), (n,), minval=0.2, maxval=0.8)
y = x[:, 0] + z + random.normal(keys.pop(), (n,))
model = bcf(
    x,
    y,
    z,
    pihat_train=pihat,
    x_test=x,
    z_test=z,
    pihat_test=pihat,
    num_trees_mu=2,
    num_trees_tau=2,
    ndpost=2,
    nskip=2,
    num_chains=2,
    seed=keys.pop(),
)

directory = Path(__file__).parent / 'npz'
model.save_npz(directory / 'bcf_v0.npz')
pred = model.predict(x, pihat_test=pihat)
np.savez(
    directory / 'bcf_v0.outputs.npz',
    x=x,
    pihat=pihat,
    mu=pred['mu'],
    tau=pred['tau'],
    sigma=model.sigma_trace,
)
