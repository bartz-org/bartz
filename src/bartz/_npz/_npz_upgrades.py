# bartz/src/bartz/_npz/_npz_upgrades.py
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

"""Convert npz archives of older formats to the current one."""

import json
from collections.abc import Callable
from typing import Any

import numpy as np
from jaxtyping import Shaped

# the skeleton of an archive (see `bartz._npz._npz`), and the arrays it references
Skeleton = Any
Arrays = dict[str, np.ndarray]


def module(cls: str, **fields: Skeleton) -> Skeleton:
    """Make the skeleton of a dataclass."""
    return dict(type='module', cls=cls, fields=fields)


def array(
    arrays: Arrays, key: str, value: Shaped[np.ndarray, '...'] | None
) -> Skeleton:
    """Add `value` to `arrays` and make its skeleton, `None` passes through."""
    if value is None:
        return None
    else:
        arrays[key] = value
        return dict(type='array', key=key)


def upgrade_bartz_0_13_bcf(_: None, old: Arrays) -> tuple[Skeleton, Arrays]:
    """Convert a `bcf` archive written by bartz 0.13 to prediction-only form."""
    arrays: Arrays = {}

    def main_trace(name: str) -> Skeleton:
        def get(field: str) -> Skeleton:
            key = f'main_trace.{name}.{field}'
            return array(arrays, f'_main_trace/{name}/{field}', old.get(key))

        return module(
            'MainTrace',
            has_chains=old[f'main_trace.{name}.has_chains'].item(),
            mesh=None,
            **{
                field: get(field)
                for field in (
                    'grow_prop_count',
                    'grow_acc_count',
                    'prune_prop_count',
                    'prune_acc_count',
                    'error_cov_inv',
                    'leaf_prior_cov_inv',
                    'theta',
                    'log_likelihood',
                    'log_trans_prior',
                    'leaf_tree',
                    'var_tree',
                    'split_tree',
                    'offset',
                    'leaf_unit',
                    'varprob',
                )
            },
        )

    root = module(
        'bcf',
        _mcmc_state=None,
        _binner=module(
            'UniqueQuantileBinner',
            _splits=array(arrays, '_binner/_splits', old['binner._splits']),
            max_split=array(arrays, '_binner/max_split', old['binner.max_split']),
        ),
        _main_trace=dict(
            type='dict', items=dict(mu=main_trace('mu'), tau=main_trace('tau'))
        ),
        _burnin_trace=None,
        _tau_0_trace=array(arrays, '_tau_0_trace', old['tau_0_trace']),
        _b_trace=array(
            arrays, '_b_trace', np.stack([old['b0_trace'], old['b1_trace']], axis=1)
        ),
        _leaf_prior_cov_inv_mu_trace=None,
        _leaf_prior_cov_inv_tau_trace=None,
        _x_train_fmt=dict(type='dict', items=json.loads(old['x_train_fmt'].item())),
        _standardize=old['standardize'].item(),
        _y_mean=array(arrays, '_y_mean', old['_y_mean']),
        _y_std=array(arrays, '_y_std', old['_y_std']),
        _outcome_type=old['_outcome_type'].item(),
        _offset=array(arrays, '_offset', old['_offset']),
        **{
            key: array(arrays, key, old.get(key))
            for key in ('_mu_test', '_tau_test', '_yhat_test')
        },
    )
    return root, arrays


# Step k converts format k to k+1. Steps are frozen once released. Use only
# json/numpy, never bartz classes. Fill new fields with the value that
# reproduces the old behavior, not the current default; if that is impossible,
# convert to the prediction-only form, with `None` in all the fields not needed
# to predict.
UPGRADES: tuple[Callable[[Skeleton, Arrays], tuple[Skeleton, Arrays]], ...] = (
    upgrade_bartz_0_13_bcf,
)
