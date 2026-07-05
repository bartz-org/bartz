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

from argparse import ArgumentParser
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
NUM_TREES = (1024, 256, 64, 16, 4)


def rms(x: Float32[Array, '... n']) -> Float32[Array, '...']:
    """Root mean square over the last axis."""
    return jnp.sqrt(jnp.mean(jnp.square(x), axis=-1))


def relative_error(
    key: Key[Array, ''], num_trees: int, dtype: jnp.dtype, quantization: int | None
) -> tuple[Float32[Array, ' n_save'], Float32[Array, ''], jnp.dtype]:
    """Return the relative rms error, the scaled resolution, and the residuals dtype."""
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
        precompute_predict_train=True,
        init_kw=dict(
            leaf_dtype=dtype, resid_dtype=dtype, leaf_quantization=quantization
        ),
    )
    pred_running = bart.predict('train', kind='latent_samples')
    pred_actual = bart.predict(dgp.x, kind='latent_samples')
    resid_rms = rms(dgp.y - pred_actual)
    err = rms(pred_running - pred_actual) / resid_rms
    state = bart._mcmc_state  # noqa: SLF001
    return err, state.sum_trees_eps() / resid_rms.mean(), state.resid.dtype


def main() -> None:
    """Run the MCMCs, plot the errors, save the figure."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dtype',
        type=jnp.dtype,
        default=jnp.dtype('float16'),
        help='dtype of leaves and residuals (default: float16)',
    )
    parser.add_argument(
        '--quantization',
        type=lambda s: None if s.lower() == 'none' else int(s),
        default=1,
        help='log2 of the leaf quantization step, or "none" (default: 1)',
    )
    args = parser.parse_args()

    keys = split(random.key(202607050), len(NUM_TREES))
    results = {
        nt: relative_error(keys.pop(), nt, args.dtype, args.quantization)
        for nt in NUM_TREES
    }
    errors = {nt: err for nt, (err, _, _) in results.items()}
    sum_trees_eps = {nt: eps for nt, (_, eps, _) in results.items()}
    (resid_dtype,) = {dtype for _, _, dtype in results.values()}

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
        ax.plot([], [], color='black', linestyle=':', label='sum_trees_eps')
        for i, ste in enumerate(sum_trees_eps.values()):
            ax.axhline(ste, color=f'C{i}', linestyle=':')
        ax.set(xscale='log', yscale='log')
        ax.grid(which='major', linestyle='--')
        ax.grid(which='minor', linestyle=':')
        ax.legend(loc='upper left')

    out = Path(__file__)
    out = out.parent / f'{out.name}-{args.dtype}-{args.quantization}.png'
    print(f'write {out}...')
    fig.savefig(out, dpi=150)


if __name__ == '__main__':
    main()
