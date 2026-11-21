# bartz/src/bartz/npz/_npz_upgrades.py
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
from collections.abc import Callable, Iterator
from copy import deepcopy
from typing import Any

import numpy as np
from jaxtyping import Bool, Shaped

# the skeleton of an archive (see `bartz.npz._npz`), and the arrays it references
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


def z_test(old: Arrays) -> Bool[np.ndarray, ' m'] | None:
    """Recover the treatment at the test points from `_yhat_test`.

    `_yhat_test` is ``mu + z * tau``, which is exactly ``mu`` or nan if z = 0.
    """
    if '_yhat_test' in old:
        yhat = old['_yhat_test']
        return np.any((yhat != old['_mu_test']) & ~np.isnan(yhat), axis=0)
    else:
        return None


def upgrade_bartz_0_13_bcf(_: None, old: Arrays) -> tuple[Skeleton, Arrays]:
    """Convert a `bcf` archive written by bartz 0.13 to prediction-only form."""
    arrays: Arrays = {}

    def main_trace(name: str) -> Skeleton:
        def get(field: str) -> Skeleton:
            key = f'main_trace.{name}.{field}'
            return array(arrays, f'_main_trace/{name}/{field}', old.get(key))

        # bartz 0.13 does not track the tree life, fill in a value that stands
        # out if used
        shape = old[f'main_trace.{name}.grow_prop_count'].shape
        mean_tree_life = np.full(shape, np.finfo(np.float32).min, np.float32)

        return module(
            'MainTrace',
            has_chains=old[f'main_trace.{name}.has_chains'].item(),
            mesh=None,
            mean_tree_life=array(
                arrays, f'_main_trace/{name}/mean_tree_life', mean_tree_life
            ),
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

    # unfold the chains out of the samples of the scalar parameters
    shape = old['main_trace.mu.grow_prop_count'].shape
    tau_0 = old['tau_0_trace']
    b = np.stack([old['b0_trace'], old['b1_trace']], axis=-1)

    # bartz 0.13 can not save models fit on a dataframe
    x_train_fmt = json.loads(old['x_train_fmt'].item())
    root = module(
        'bcf',
        _mcmc_state=None,
        _binner=module(
            'UniqueQuantileBinner',
            _splits=array(arrays, '_binner/_splits', old['binner._splits']),
            max_split=array(arrays, '_binner/max_split', old['binner.max_split']),
        ),
        _main_trace=module(
            'BCFMainTrace',
            mu=main_trace('mu'),
            tau=main_trace('tau'),
            tau_0=array(arrays, '_main_trace/tau_0', tau_0.reshape(shape)),
            b=array(arrays, '_main_trace/b', b.reshape(*shape, 2)),
        ),
        _burnin_trace=None,
        _x_train_fmt=dict(type='dict', items=x_train_fmt),
        _has_pihat=old['binner.max_split'].size == x_train_fmt['num_covar'] + 1,
        _y_mean=array(arrays, '_y_mean', old['_y_mean']),
        _y_std=array(arrays, '_y_std', old['_y_std']),
        _outcome_type=old['_outcome_type'].item(),
        _mu_test=array(arrays, '_mu_test', old.get('_mu_test')),
        _tau_test=array(arrays, '_tau_test', old.get('_tau_test')),
        _z_test=array(arrays, '_z_test', z_test(old)),
    )
    return root, arrays


def upgrade_used_trees(root: Skeleton, old: Arrays) -> tuple[Skeleton, Arrays]:
    """Add `Forest.used_trees` with all the trees in use, and `StepConfig` options for it.

    `StepConfig.random_tree_scan` is off, while
    `StepConfig.sequential_block_size` and `StepConfig.parallel_block_size` are
    automatic, which has no effect with all the trees in use.
    """

    def walk(node: Skeleton) -> Skeleton:
        if type(node) is list:
            return [walk(v) for v in node]
        elif type(node) is not dict or node['type'] == 'array':
            return node
        elif node['type'] == 'module':
            fields = {k: walk(v) for k, v in node['fields'].items()}
            if node['cls'] == 'Forest':
                fields = dict(**fields, used_trees=None)
            elif node['cls'] == 'StepConfig':
                fields = dict(
                    **fields,
                    random_tree_scan=False,
                    sequential_block_size=None,
                    parallel_block_size=None,
                )
            return module(node['cls'], **fields)
        elif type(node['items']) is dict:
            return dict(node, items={k: walk(v) for k, v in node['items'].items()})
        else:
            return dict(node, items=[walk(v) for v in node['items']])

    return walk(root), old


def rescale(arrays: Arrays, node: Skeleton, factor: float) -> None:
    """Multiply by `factor` the array `node` refers to, if any."""
    if node is not None:
        x = arrays[node['key']]
        arrays[node['key']] = (x.astype(np.float64) * factor).astype(x.dtype)


def count_trees(arrays: Arrays, fields: dict[str, Skeleton]) -> int:
    """Return the number of trees of a forest or of a main trace."""
    *_, num_trees, _ = arrays[fields['split_tree']['key']].shape
    return num_trees


def trace_pairs(
    main: Skeleton, burnin: Skeleton
) -> Iterator[tuple[Skeleton, Skeleton]]:
    """Pair each main trace with the burn-in trace of the same forest, if any."""
    if main is None:
        pass
    elif 'leaf_prior_cov_inv' in main['fields']:
        yield main, burnin
    else:
        # the traces of the two forests of `bcf`
        for name in 'mu', 'tau':
            yield (
                main['fields'][name],
                None if burnin is None else burnin['fields'][name],
            )


def rename_leaf_prior(node: Skeleton) -> Skeleton:
    """Rename the field `leaf_prior_cov_inv` of a dataclass in place, and return it."""
    fields = node['fields']
    fields['sum_trees_prior_cov_inv'] = fields.pop('leaf_prior_cov_inv')
    return fields['sum_trees_prior_cov_inv']


def upgrade_sum_trees_prior(root: Skeleton, old: Arrays) -> tuple[Skeleton, Arrays]:
    """Convert `leaf_prior_cov_inv` to `sum_trees_prior_cov_inv`.

    In `Forest` and in the traces, the former is the prior precision of a leaf,
    the latter the one of the sum of trees, smaller by a factor of the number
    of trees.
    """
    arrays = dict(old)
    root = deepcopy(root)
    fields = root['fields']

    # the burn-in traces do not have the trees, to count them
    for main, burnin in trace_pairs(fields['_main_trace'], fields['_burnin_trace']):
        n = count_trees(arrays, main['fields'])
        for trace in main, burnin:
            if trace is not None:
                rescale(arrays, rename_leaf_prior(trace), 1 / n)

    state = fields['_mcmc_state']
    state_fields = [] if state is None else state['fields'].values()
    for forest in state_fields:
        if type(forest) is dict and forest.get('cls') == 'Forest':
            n = count_trees(arrays, forest['fields'])
            prior = rename_leaf_prior(forest)['fields']
            rescale(arrays, prior['value'], 1 / n)
            rescale(arrays, prior['rate'], float(n))

    return root, arrays


def upgrade_bartz_0_14(root: Skeleton, old: Arrays) -> tuple[Skeleton, Arrays]:
    """Convert an archive written by bartz 0.14."""
    return upgrade_sum_trees_prior(*upgrade_used_trees(root, old))


# Step k converts format k to k+1. Steps are frozen once released. Use only
# json/numpy, never bartz classes. Fill new fields with the value that
# reproduces the old behavior, not the current default; if that is impossible,
# convert to the prediction-only form, with `None` in all the fields not needed
# to predict; in cases that don't fall in the prediction-only paradigm, decide
# a fill strategy which may not be `None` if the field is not already optional.
UPGRADES: tuple[Callable[[Skeleton, Arrays], tuple[Skeleton, Arrays]], ...] = (
    upgrade_bartz_0_13_bcf,
    upgrade_bartz_0_14,
)
