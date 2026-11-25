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
from collections.abc import Callable
from typing import Any

import numpy as np
from jaxtyping import Bool, Shaped, UInt

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


def is_module(node: Skeleton) -> bool:
    """Whether `node` is the skeleton of a dataclass."""
    return type(node) is dict and node['type'] == 'module'


# the classes with a leaf prior precision in bartz 0.14: the ones that store
# the trees, and the one that does not
WITH_TREES = ('Forest', 'MainTrace', 'MainTraceWithTrainPred')
WITHOUT_TREES = 'BurninTrace'


def count_trees(arrays: Arrays, node: Skeleton, main: Skeleton) -> int:
    """Return the number of trees of `node`, a forest or a trace.

    A burn-in trace lacks the trees to count, so it takes the number from
    `main`, the main trace saved along it.
    """
    source = main if node['cls'] == WITHOUT_TREES else node
    if is_module(source) and source['cls'] in WITH_TREES:
        *_, num_trees, _ = arrays[source['fields']['split_tree']['key']].shape
        return num_trees
    else:
        msg = (
            f'cannot load a {WITHOUT_TREES} saved with bartz 0.14 outside of the '
            'burn-in trace of a Bart or bcf: its conversion needs the number of '
            'trees of the main trace saved along it'
        )
        raise ValueError(msg)


def scaled(old: Arrays, new: Arrays, node: Skeleton, factor: float) -> Skeleton:
    """Add to `new` the array `node` refers to in `old`, times `factor`."""
    if node is None:
        return None
    else:
        x = old[node['key']]
        # `asarray` because numpy turns 0d arrays into scalars
        x = np.asarray(x.astype(np.float64) * factor).astype(x.dtype)
        return array(new, node['key'], x)


def leaf_prior_to_sum(
    old: Arrays, new: Arrays, prior: Skeleton, num_trees: int
) -> Skeleton:
    """Convert a leaf prior precision, a `Wishart` or its samples, to the one of the sum of trees."""
    if is_module(prior):
        assert prior['cls'] == 'Wishart'
        fields = prior['fields']
        fields = dict(
            fields,
            value=scaled(old, new, fields['value'], 1 / num_trees),
            rate=scaled(old, new, fields['rate'], float(num_trees)),
        )
        return module(prior['cls'], **fields)
    else:
        return scaled(old, new, prior, 1 / num_trees)


def main_counterpart(node: Skeleton, main: Skeleton, name: str) -> Skeleton:
    """Return the part of the main trace matching the field `name` of `node`, if any.

    `main` is the part that matches `node`, if `node` is within a burn-in trace.
    """
    if node['cls'] in ('Bart', 'bcf') and name == '_burnin_trace':
        return node['fields']['_main_trace']
    elif is_module(main):
        return main['fields'].get(name)
    else:
        return None


def custom_class_error(node: Skeleton, field: str) -> ValueError:
    """Make the error for a custom class with a field that needs converting."""
    msg = (
        f'cannot load the custom class {node["cls"]!r} saved with bartz 0.14: '
        f'its field `{field}` needs a conversion known only for the classes of bartz'
    )
    return ValueError(msg)


def upgrade_sum_trees_prior(root: Skeleton, old: Arrays) -> tuple[Skeleton, Arrays]:
    """Convert `leaf_prior_cov_inv` to `sum_trees_prior_cov_inv`.

    In `Forest` and in the traces, the former is the prior precision of a leaf,
    the latter the one of the sum of trees, smaller by a factor of the number
    of trees. A burn-in trace lacks the trees to count, so it takes the number
    from the main trace of the same `Bart` or `bcf`.

    Only the classes of bartz, recognized by name, are converted. A burn-in
    trace anywhere else, or a custom class with a `leaf_prior_cov_inv` field,
    e.g., a subclass of a trace, raises `ValueError`.
    """
    new: Arrays = {}

    def walk(node: Skeleton, main: Skeleton = None) -> Skeleton:
        if type(node) is list:
            return [walk(v) for v in node]
        elif type(node) is not dict or node['type'] == 'array':
            return node
        elif node['type'] == 'module':
            fields = {
                k: walk(v, main_counterpart(node, main, k))
                for k, v in node['fields'].items()
            }
            if node['cls'] in (*WITH_TREES, WITHOUT_TREES):
                num_trees = count_trees(old, node, main)
                prior = fields['leaf_prior_cov_inv']
                prior = leaf_prior_to_sum(old, new, prior, num_trees)
                fields = {k: v for k, v in fields.items() if k != 'leaf_prior_cov_inv'}
                fields = dict(fields, sum_trees_prior_cov_inv=prior)
            elif 'leaf_prior_cov_inv' in fields:
                raise custom_class_error(node, 'leaf_prior_cov_inv')
            return module(node['cls'], **fields)
        elif type(node['items']) is dict:
            return dict(node, items={k: walk(v) for k, v in node['items'].items()})
        else:
            return dict(node, items=[walk(v) for v in node['items']])

    return walk(root), {**old, **new}


def has_peff(sums: Skeleton) -> bool:
    """Whether `sums` is the skeleton of a dict with a `peff` item."""
    return type(sums) is dict and sums['type'] == 'dict' and 'peff' in sums['items']


def upgrade_peff_binary_mask(root: Skeleton, old: Arrays) -> tuple[Skeleton, Arrays]:
    """Rename the running sum `peff` to `var_perplexity`, and drop `Bart._binary_mask`.

    The latter is redundant with the MCMC state.

    Only the classes of bartz, recognized by name, are converted. A custom
    class with a `_binary_mask` field, or with a `peff` in a `sums` field,
    e.g., a subclass of `Bart` or `StatsAccumulator`, raises `ValueError`.
    """
    dropped: set[str] = set()

    def walk(node: Skeleton) -> Skeleton:
        if type(node) is list:
            return [walk(v) for v in node]
        elif type(node) is not dict or node['type'] == 'array':
            return node
        elif node['type'] == 'module':
            fields = {k: walk(v) for k, v in node['fields'].items()}
            if node['cls'] == 'StatsAccumulator' and fields['sums'] is not None:
                sums = fields['sums']
                items = {
                    'var_perplexity' if k == 'peff' else k: v
                    for k, v in sums['items'].items()
                }
                fields = dict(fields, sums=dict(sums, items=items))
            elif node['cls'] == 'Bart':
                dropped.add(fields['_binary_mask']['key'])
                fields = {k: v for k, v in fields.items() if k != '_binary_mask'}
            elif '_binary_mask' in fields:
                raise custom_class_error(node, '_binary_mask')
            elif has_peff(fields.get('sums')):
                raise custom_class_error(node, 'sums')
            return module(node['cls'], **fields)
        elif type(node['items']) is dict:
            return dict(node, items={k: walk(v) for k, v in node['items'].items()})
        else:
            return dict(node, items=[walk(v) for v in node['items']])

    root = walk(root)
    return root, {k: v for k, v in old.items() if k not in dropped}


def compress_leaf_tree(
    split_tree: UInt[np.ndarray, '*batch_shape half_tree_size'],
    leaf_tree: Shaped[np.ndarray, '*batch_shape 2*half_tree_size']
    | Shaped[np.ndarray, '*batch_shape k 2*half_tree_size']
    | Shaped[np.ndarray, '*batch_shape k k 2*half_tree_size'],
) -> (
    Shaped[np.ndarray, '*batch_shape half_tree_size']
    | Shaped[np.ndarray, '*batch_shape k half_tree_size']
    | Shaped[np.ndarray, '*batch_shape k k half_tree_size']
):
    """Pack values defined on the leaves of trees into arrays half as long.

    The value of the leaf at index i goes to the index obtained shifting i
    right until its lowest set bit is dropped, the other elements are zero.
    """
    *batch_shape, half_tree_size = split_tree.shape
    leaf_ndim = leaf_tree.ndim - split_tree.ndim
    split_tree = split_tree.reshape(*batch_shape, *(1,) * leaf_ndim, half_tree_size)

    # an element takes the nodes reached going left from a right child or the
    # root; visit them top-down, moving on to a node if it is below a decision
    # node
    out = leaf_tree[..., 1::2].copy()
    stride = 2
    while stride <= half_tree_size:
        parent_is_decision = split_tree[..., stride // 2 :: stride] != 0
        *_, size = parent_is_decision.shape
        nodes = leaf_tree[..., stride :: 2 * stride]
        out[..., :size] = np.where(parent_is_decision, nodes, out[..., :size])
        stride *= 2

    # the node at the top of element i is the right child of node i, or the root
    top_in_tree = split_tree != 0
    top_in_tree[..., 0] = True
    return np.where(top_in_tree, out, np.zeros((), out.dtype))


def compress_leaf_indices(
    leaf_indices: UInt[np.ndarray, '*shape'], half_tree_size: int
) -> UInt[np.ndarray, '*shape']:
    """Convert the indices of leaves to those of their values after `compress_leaf_tree`."""
    lowest_set_bit = leaf_indices & (~leaf_indices + 1)
    out = leaf_indices // np.maximum(lowest_set_bit, 1) >> 1
    dtypes = (np.uint8, np.uint16, np.uint32, np.uint64)
    return out.astype(
        next(dtype for dtype in dtypes if half_tree_size - 1 <= np.iinfo(dtype).max)
    )


# the classes with arrays over the leaves in bartz 0.14: the traces, with the
# trees alone, and the one with also the datapoints
WITH_LEAVES = ('MainTrace', 'MainTraceWithTrainPred')
WITH_LEAVES_AND_DATA = 'Forest'


def compress_leaves(old: Arrays, fields: dict[str, Skeleton]) -> Arrays:
    """Convert the arrays over the leaves and datapoints of a forest or a trace.

    In a forest, they follow the trees without the pending prunes applied.
    """
    split_tree = old[fields['split_tree']['key']]
    *_, half_tree_size = split_tree.shape
    new: Arrays = {}
    if 'leaf_indices' in fields:
        key = fields['leaf_indices']['key']
        new[key] = compress_leaf_indices(old[key], half_tree_size)
        to_prune = old[fields['to_prune']['key']]
        move_node = old[fields['move_node']['key']]
        is_move_node = np.arange(half_tree_size) == move_node[..., None]
        split_tree = np.where(to_prune[..., None] & is_move_node, 1, split_tree)
    for name in ('leaf_tree', 'count_tree', 'prec_tree'):
        node = fields.get(name)
        if node is not None:
            new[node['key']] = compress_leaf_tree(split_tree, old[node['key']])
    return new


def upgrade_leaf_layout(root: Skeleton, old: Arrays) -> tuple[Skeleton, Arrays]:
    """Halve the length of the arrays over the leaves.

    These are `leaf_tree`, and in `Forest` also `count_tree` and `prec_tree`,
    with `leaf_indices` pointing into them.

    Only the classes of bartz, recognized by name, are converted. Any other
    class with a `leaf_tree` field raises `ValueError`.
    """
    new: Arrays = {}

    def walk(node: Skeleton) -> Skeleton:
        if type(node) is list:
            return [walk(v) for v in node]
        elif type(node) is not dict or node['type'] == 'array':
            return node
        elif node['type'] == 'module':
            fields = {k: walk(v) for k, v in node['fields'].items()}
            if node['cls'] in (*WITH_LEAVES, WITH_LEAVES_AND_DATA):
                new.update(compress_leaves(old, fields))
            elif 'leaf_tree' in fields:
                raise custom_class_error(node, 'leaf_tree')
            return module(node['cls'], **fields)
        elif type(node['items']) is dict:
            return dict(node, items={k: walk(v) for k, v in node['items'].items()})
        else:
            return dict(node, items=[walk(v) for v in node['items']])

    return walk(root), {**old, **new}


def upgrade_bartz_0_14(root: Skeleton, old: Arrays) -> tuple[Skeleton, Arrays]:
    """Convert an archive written by bartz 0.14."""
    steps = (
        upgrade_used_trees,
        upgrade_sum_trees_prior,
        upgrade_peff_binary_mask,
        upgrade_leaf_layout,
    )
    for step in steps:
        root, old = step(root, old)
    return root, old


# Step k converts format k to k+1. Steps are frozen once released. Use only
# json/numpy, never bartz classes. Fill new fields with the value that
# reproduces the old behavior, not the current default; if that is impossible,
# convert to the prediction-only form, with `None` in all the fields not needed
# to predict; in cases that don't fall in the prediction-only paradigm, decide
# a fill strategy which may not be `None` if the field is not already optional.
# Recognize the classes of bartz by name, and raise on anything else that would
# need converting, e.g., a custom class with a field that changes, instead of
# guessing what it is.
UPGRADES: tuple[Callable[[Skeleton, Arrays], tuple[Skeleton, Arrays]], ...] = (
    upgrade_bartz_0_13_bcf,
    upgrade_bartz_0_14,
)
