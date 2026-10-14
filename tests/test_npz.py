# bartz/tests/test_npz.py
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

"""Test the npz archives written by `Bart.save_npz` and `bcf.save_npz`.

``tests/npz/`` holds an archive per model and format version, written by
``python -m tests.test_npz`` for the current format, together with the
predictions of the model that wrote it. See `test_schema_matches_archive`.
"""

import json
import re
import sys
from argparse import ArgumentParser
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, TypedDict
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from jax import numpy as jnp
from jax import random
from jaxtyping import Array, Float32

from bartz import Bart
from bartz._jaxext import Module, split
from bartz._npz._npz import (
    FORMAT_VERSION,
    META_KEY,
    Json,
    load_npz,
    save_npz,
    serializable,
)
from bartz.bcf import bcf
from tests.util import assert_array_equal, assert_close_matrices

ARCHIVE_DIR = Path(__file__).parent / 'npz'

# sha256 of the archives of released formats, which must not change
FROZEN = {
    'bcf_v0.npz': '57b3ade0791bf13dde38b65d41cc1bdec34bda33a132a9196b5c9466bd6e1123',
    'bcf_v0.outputs.npz': 'db73ea736b194edb8d499e394b8a60d75a14b544c002d47e9d1e5dbd1165e25e',
}

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:  # WORKAROUND(python<3.11): typing.NotRequired was added in 3.11
    from typing_extensions import NotRequired


class Inputs(TypedDict):
    """The test points of the predictions saved next to an archive."""

    x: Float32[np.ndarray, 'n p']
    pihat: NotRequired[Float32[np.ndarray, ' n']]


def make_bart() -> tuple[Bart, Inputs]:
    """Fit a small `Bart` on a dataframe, deterministically."""
    keys = split(random.key(0), 3)
    n, p = 20, 2
    x = random.normal(keys.pop(), (n, p))
    y = x[:, 0] + random.normal(keys.pop(), (n,))
    x_train = pd.DataFrame(np.asarray(x), columns=['a', 'b'])
    model = Bart(
        x_train,
        y,
        num_trees=3,
        maxdepth=3,
        n_save=2,
        n_burn=2,
        num_chains=2,
        printevery=None,
        seed=keys.pop(),
    )
    return model, Inputs(x=np.asarray(x))


def predict_bart(model: Bart, inputs: Inputs) -> dict[str, Array]:
    """Compute the predictions saved next to an archive to check loading it."""
    x = pd.DataFrame(inputs['x'], columns=['a', 'b'])
    return dict(yhat=model.predict(x, kind='mean_samples'))


def make_bcf() -> tuple[bcf, Inputs]:
    """Fit a small `bcf` with test predictions, deterministically."""
    keys = split(random.key(0), 5)
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
    return model, Inputs(x=np.asarray(x), pihat=np.asarray(pihat))


def predict_bcf(model: bcf, inputs: Inputs) -> dict[str, Array]:
    """Compute the predictions saved next to an archive to check loading it."""
    pred = model.predict(inputs['x'], pihat_test=inputs['pihat'])
    return dict(mu=pred['mu'], tau=pred['tau'], sigma=model.sigma_trace)


@dataclass(frozen=True)
class ModelSpec:
    """How to make and check a model type in the archive tests."""

    cls: type[Bart] | type[bcf]
    make: Callable[[], tuple[Any, Inputs]]
    predict: Callable[[Any, Inputs], dict[str, Array]]


MODELS = dict(
    bart=ModelSpec(Bart, make_bart, predict_bart),
    bcf=ModelSpec(bcf, make_bcf, predict_bcf),
)

ARCHIVE_NAME = re.compile(r'(?P<model>\w+)_v(?P<version>\d+)\.npz')

ARCHIVES = sorted(
    path.name for path in ARCHIVE_DIR.iterdir() if ARCHIVE_NAME.fullmatch(path.name)
)


def archive_path(name: str) -> Path:
    """Return the path of the committed archive of model `name` in the current format."""
    return ARCHIVE_DIR / f'{name}_v{FORMAT_VERSION}.npz'


def outputs_path(archive: Path) -> Path:
    """Return the path of the outputs saved together with `archive`."""
    return archive.with_suffix('.outputs.npz')


def read_archive(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Read the metadata, without the bartz version, and the arrays of an archive."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data[META_KEY])
        arrays = {k: data[k] for k in data.files if k != META_KEY}
    del meta['bartz_version']
    return meta, arrays


def schema(node: Json, arrays: dict[str, np.ndarray]) -> Json:
    """Replace the arrays with their dtype and shape, and the scalars with their type."""
    if type(node) is list:
        return [schema(v, arrays) for v in node]
    elif type(node) is dict and node.get('type') == 'array':
        array = arrays[node['key']]
        return dict(type='array', dtype=str(array.dtype), shape=list(array.shape))
    elif type(node) is dict:
        return {k: schema(v, arrays) for k, v in node.items()}
    else:
        return type(node).__name__


@pytest.mark.parametrize('name', MODELS)
def test_schema_matches_archive(name: str, tmp_path: Path) -> None:
    """Check a freshly saved model has the same schema as the committed archive.

    If this fails, the file format changed:

    - if the current archives are not in `FROZEN` (format unreleased),
      regenerate them with ``python -m tests.test_npz``
    - else add a step to `bartz._npz._npz_upgrades.UPGRADES` (this bumps the format
      version) and regenerate

    In both cases, update the last step so the older archives still pass. A
    field changing meaning is a format change even if this test passes.
    """
    path = tmp_path / 'model.npz'
    model, _ = MODELS[name].make()
    model.save_npz(path)
    meta, arrays = read_archive(path)
    committed_meta, committed_arrays = read_archive(archive_path(name))
    assert schema(meta, arrays) == schema(committed_meta, committed_arrays)


@pytest.mark.parametrize('name', MODELS)
def test_resave_archive(name: str, tmp_path: Path) -> None:
    """Check loading and saving again the committed archive reproduces it."""
    path = tmp_path / 'model.npz'
    MODELS[name].cls.load_npz(archive_path(name)).save_npz(path)
    meta, arrays = read_archive(path)
    committed_meta, committed_arrays = read_archive(archive_path(name))
    assert meta == committed_meta
    assert arrays.keys() == committed_arrays.keys()
    for key, array in arrays.items():
        assert_array_equal(array, committed_arrays[key], err_msg=key)


@pytest.mark.parametrize('archive', ARCHIVES)
def test_archive_outputs(archive: str, tmp_path: Path) -> None:
    """Check the archives of all formats predict as the model that wrote them.

    Also check after saving again in the current format.
    """
    path = ARCHIVE_DIR / archive
    match = ARCHIVE_NAME.fullmatch(archive)
    assert match is not None
    spec = MODELS[match['model']]
    model = spec.cls.load_npz(path)
    resaved = tmp_path / 'model.npz'
    model.save_npz(resaved)
    with np.load(outputs_path(path), allow_pickle=False) as data:
        expected = dict(data)
    inputs = Inputs(x=expected.pop('x'))
    if 'pihat' in expected:
        inputs['pihat'] = expected.pop('pihat')
    for loaded in model, spec.cls.load_npz(resaved):
        outputs = spec.predict(loaded, inputs)
        assert outputs.keys() == expected.keys()
        for key, value in outputs.items():
            assert_close_matrices(
                value,
                expected[key],
                rtol=1e-6,
                err_msg=f'{key}, if the format changed see test_schema_matches_archive',
            )


def unfrozen_archives() -> Iterator[str]:
    """Yield the committed archives and outputs that are not in `FROZEN`."""
    for archive in ARCHIVES:
        for name in archive, outputs_path(Path(archive)).name:
            if name not in FROZEN:
                yield name


def test_frozen() -> None:
    """Check the archives of released formats are unchanged and still there.

    The archives of formats older than the current one must be frozen.
    """
    for name, digest in FROZEN.items():
        content = (ARCHIVE_DIR / name).read_bytes()
        assert sha256(content).hexdigest() == digest, (
            f'{name} is released and must not change: to change the format, '
            'add a step to bartz._npz._npz_upgrades.UPGRADES instead'
        )
    for name in unfrozen_archives():
        match = ARCHIVE_NAME.fullmatch(name.replace('.outputs', ''))
        assert match is not None
        assert int(match['version']) == FORMAT_VERSION, f'{name} is not in FROZEN'


@serializable
class Box(Module):
    """Holder for test values."""

    value: Any


def test_roundtrip(tmp_path: Path) -> None:
    """Check all supported value types survive saving and loading."""
    value: dict = dict(
        scalars=[None, True, 1, 1.5, 'a'],
        nested=dict(box=Box(jnp.arange(3, dtype=jnp.uint8))),
        array=jnp.ones((2, 3)),
    )
    path = tmp_path / 'box.npz'
    save_npz(path, Box(value))
    box = load_npz(path)
    assert isinstance(box, Box)
    loaded = box.value
    assert loaded['scalars'] == value['scalars']
    for key in 'scalars', 'nested', 'array':
        assert type(loaded[key]) is type(value[key])
    assert_array_equal(loaded['nested']['box'].value, value['nested']['box'].value)
    assert_array_equal(loaded['array'], value['array'])


def test_path_as_given(tmp_path: Path) -> None:
    """Check the path is used as given, without appending `.npz`."""
    path = tmp_path / 'box'
    save_npz(path, Box(None))
    assert path.exists()
    assert load_npz(path) == Box(None)


def test_failed_save_keeps_file(tmp_path: Path) -> None:
    """Check a save failing midway leaves an existing archive untouched."""
    path = tmp_path / 'box.npz'
    save_npz(path, Box(1))
    deleted = jnp.zeros(3)
    deleted.delete()
    with pytest.raises(RuntimeError, match='deleted'):
        save_npz(path, Box(deleted))
    assert load_npz(path) == Box(1)
    assert list(tmp_path.iterdir()) == [path]


def make_reloadable() -> type[Module]:
    """Define a new class each time with the same name, like a module reload."""

    @serializable
    class Reloadable(Module):
        value: Any

    return Reloadable


def test_reregister(tmp_path: Path) -> None:
    """Check registering again a class replaces it, and a homonym raises."""
    old = make_reloadable()
    new = make_reloadable()
    assert old is not new
    path = tmp_path / 'reloadable.npz'
    save_npz(path, old(1))  # ty: ignore[too-many-positional-arguments]
    assert type(load_npz(path)) is new

    with pytest.raises(ValueError, match="'Box' already registered by"):

        @serializable
        class Box(Module):
            pass


class Unregistered(Module):
    """Module not registered for npz archives."""


@pytest.mark.parametrize(
    ('value', 'match'),
    [
        ((1, 2), 'unsupported type tuple'),
        (np.zeros(3), 'unsupported type ndarray'),
        (Unregistered(), 'unsupported type Unregistered'),
        ({1: 2}, 'dict key 1'),
        ({'a/b': 2}, "dict key 'a/b'"),
        (jnp.asarray(1.0), 'weakly typed'),
    ],
)
def test_unsupported(value: Any, match: str, tmp_path: Path) -> None:  # noqa: ANN401
    """Check unsupported values are rejected at save time."""
    with pytest.raises(TypeError, match=match):
        save_npz(tmp_path / 'box.npz', Box(dict(x=value)))


def edit_meta(path: Path, edit: Callable[[dict], None]) -> None:
    """Modify in place the metadata of an archive."""
    with ZipFile(path) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    meta = json.loads(members[META_KEY])
    edit(meta)
    members[META_KEY] = json.dumps(meta).encode()
    with ZipFile(path, 'w') as zf:
        for name, content in members.items():
            zf.writestr(name, content)


@pytest.mark.parametrize(
    ('edit', 'match'),
    [
        (lambda meta: meta.update(format_version=FORMAT_VERSION + 1), 'upgrade bartz'),
        (
            lambda meta: meta['root'].update(cls='Nonexistent'),
            "unknown class 'Nonexistent'",
        ),
        (
            lambda meta: meta['root']['fields'].update(extra=None),
            r"unexpected \['extra'\]",
        ),
        (
            lambda meta: meta['root']['fields'].pop('value'),
            r"missing in the archive \['value'\]",
        ),
        (
            lambda meta: meta['root']['fields'].update(value=dict(type='foo')),
            "unknown node type 'foo'",
        ),
        (
            lambda meta: meta['root']['fields'].update(value=None),
            "unused arrays \\['value'\\]",
        ),
    ],
)
def test_invalid_archive(
    edit: Callable[[dict], None], match: str, tmp_path: Path
) -> None:
    """Check malformed or mismatched archives are rejected at load time."""
    path = tmp_path / 'box.npz'
    save_npz(path, Box(jnp.zeros(3)))
    edit_meta(path, edit)
    with pytest.raises(ValueError, match=match):
        load_npz(path)


@pytest.mark.parametrize(('cls', 'other'), [(Bart, 'bcf'), (bcf, 'bart')])
def test_wrong_class(cls: type[Bart] | type[bcf], other: str) -> None:
    """Check loading an archive of a different model fails."""
    match = f'contains a {MODELS[other].cls.__name__}, not a {cls.__name__}'
    with pytest.raises(TypeError, match=match):
        cls.load_npz(archive_path(other))


def test_not_bartz_archive(tmp_path: Path) -> None:
    """Check loading an npz archive not written by bartz fails."""
    path = tmp_path / 'other.npz'
    np.savez(path, x=1)
    with pytest.raises(ValueError, match='not an archive written by bartz'):
        load_npz(path)


def write_archives() -> None:
    """Write the archives of the current format, if not frozen."""
    for name, spec in MODELS.items():
        path = archive_path(name)
        if path.name in FROZEN:
            msg = f'{path.name} is frozen, see test_schema_matches_archive'
            raise RuntimeError(msg)
        model, inputs = spec.make()
        model.save_npz(path)
        arrays: dict = dict(**inputs, **spec.predict(model, inputs))
        np.savez(outputs_path(path), **arrays)


def check_frozen() -> None:
    """Exit with an error if some archive is not frozen, printing its hash."""
    unfrozen = list(unfrozen_archives())
    if unfrozen:
        print(
            'Release the current file format by adding to FROZEN in tests/test_npz.py:'
        )
        for name in unfrozen:
            digest = sha256((ARCHIVE_DIR / name).read_bytes()).hexdigest()
            print(f'    {name!r}: {digest!r},')
        sys.exit(1)


if __name__ == '__main__':
    parser = ArgumentParser(description='Write the archives of the current format.')
    parser.add_argument(
        '--check-frozen',
        action='store_true',
        help='instead, check all archives are frozen, before a release',
    )
    if parser.parse_args().check_frozen:
        check_frozen()
    else:
        write_archives()
