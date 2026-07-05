# bartz/scripts/resid-accuracy.py
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

"""Measure the accuracy degradation of the running residuals along the MCMC.

Runs `Bart` with float16 residuals (to amplify the accumulation error) and
train-prediction precomputation (to propagate the residuals error to the
predictions), then compares the train predictions accumulated during the MCMC
against the exact ones recomputed from the saved trees. The relative error
``||pred_running - pred_actual|| / ||y - pred_actual||`` is plotted against the
MCMC iteration, and against iteration * num_trees, for a series of `num_trees`
settings. The error is normalized by the current residual magnitude because
the float16 rounding error injected at each update is proportional to it.
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from jaxtyping import Array, Float32, Key

from bartz import Bart
from bartz._jaxext import split
from bartz.testing import gen_data

N = 1000
P = 10
N_SAVE = 10_000
NUM_TREES = (4096, 1024, 256, 64, 16, 4)
DTYPE = 'float32'


def rms(x: Float32[Array, '... n']) -> Float32[Array, '...']:
    """Root mean square over the last axis."""
    return jnp.sqrt(jnp.mean(jnp.square(x), axis=-1))


def relative_error(
    key: Key[Array, ''], num_trees: int
) -> tuple[Float32[Array, ' n_save'], jnp.dtype]:
    """Return the per-iteration relative rms error and the residuals dtype."""
    keys = split(key)
    dgp = gen_data(
        keys.pop(), n=N, p=P, q=0, sigma2_lin=1.0, sigma2_quad=1.0, sigma2_eps=1.0
    )
    bart = Bart(
        dgp.x,
        dgp.y,
        num_trees=num_trees,
        n_burn=0,
        n_save=N_SAVE,
        num_chains=None,
        seed=keys.pop(),
        init_kw=dict(leaf_dtype=jnp.dtype(DTYPE), resid_dtype=jnp.dtype(DTYPE)),
        precompute_predict_train=True,
    )
    pred_running = bart.predict('train', kind='latent_samples')
    pred_actual = bart.predict(dgp.x, kind='latent_samples')
    err = rms(pred_running - pred_actual) / rms(dgp.y - pred_actual)
    return err, bart._mcmc_state.resid.dtype  # noqa: SLF001


def main() -> None:
    """Run the MCMCs, plot the errors, save the figure."""
    keys = split(random.key(202607050), len(NUM_TREES))
    results = {nt: relative_error(keys.pop(), nt) for nt in NUM_TREES}
    errors = {nt: err for nt, (err, _) in results.items()}
    (resid_dtype,) = {dtype for _, dtype in results.values()}

    fig, (ax_iter, ax_steps) = plt.subplots(
        1,
        2,
        num='resid-accuracy',
        figsize=(10, 4.5),
        layout='constrained',
        sharey=True,
        clear=True,
    )

    iteration = 1 + jnp.arange(N_SAVE)
    for i, (nt, err) in enumerate(errors.items()):
        ax_iter.plot(iteration, err, color=f'C{i}', label=f'{nt} trees')
        ax_steps.plot(iteration * nt, err, color=f'C{i}', label=f'{nt} trees')

    ax_iter.set(
        xlabel='MCMC iteration',
        ylabel='rms(pred_running - pred_actual) / rms(y - pred_actual)',
    )
    ax_steps.set(xlabel='MCMC iteration * num_trees')
    eps = jnp.finfo(resid_dtype).eps
    for ax in (ax_iter, ax_steps):
        ax.axhline(eps, color='black', linestyle='--', label=f'{resid_dtype} eps')
        ax.set(xscale='log', yscale='log')
        ax.grid(which='major', linestyle='--')
        ax.grid(which='minor', linestyle=':')
        ax.legend()

    out = Path(__file__).with_suffix('.png')
    fig.savefig(out, dpi=150)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
