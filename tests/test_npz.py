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

The archives of the current format version committed in ``tests/npz/`` pin
the format. Regenerate them with ``python -m tests.test_npz`` after bumping
`bartz._npz.FORMAT_VERSION`.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from jax import numpy as jnp
from jax import random

from bartz import Bart
from bartz._jaxext import Module, split
from bartz._npz import FORMAT_VERSION, META_KEY, Json, load_npz, save_npz, serializable
from bartz.bcf import bcf
from tests.util import assert_array_equal

ARCHIVE_DIR = Path(__file__).parent / 'npz'


def make_bart() -> Bart:
    """Fit a small `Bart` on a dataframe, deterministically."""
    keys = split(random.key(0), 3)
    n, p = 20, 2
    x = random.normal(keys.pop(), (n, p))
    y = x[:, 0] + random.normal(keys.pop(), (n,))
    x_train = pd.DataFrame(np.asarray(x), columns=['a', 'b'])
    return Bart(
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


def make_bcf() -> bcf:
    """Fit a small `bcf` with test predictions, deterministically."""
    keys = split(random.key(0), 5)
    n, p = 20, 2
    x = random.normal(keys.pop(), (n, p))
    z = random.bernoulli(keys.pop(), 0.5, (n,)).astype(float)
    pihat = random.uniform(keys.pop(), (n,), minval=0.2, maxval=0.8)
    y = x[:, 0] + z + random.normal(keys.pop(), (n,))
    return bcf(
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


MODELS: dict[str, tuple[type[Bart] | type[bcf], Callable[[], Bart | bcf]]] = dict(
    bart=(Bart, make_bart), bcf=(bcf, make_bcf)
)


def archive_path(name: str) -> Path:
    """Return the path of the committed archive of model `name`."""
    return ARCHIVE_DIR / f'{name}_v{FORMAT_VERSION}.npz'


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

    If this fails, the format changed: bump `bartz._npz.FORMAT_VERSION` and
    regenerate the archives. Bump the version also on changes this test can
    not detect, e.g., a field changing meaning.
    """
    _, make = MODELS[name]
    path = tmp_path / 'model.npz'
    make().save_npz(path)
    meta, arrays = read_archive(path)
    committed_meta, committed_arrays = read_archive(archive_path(name))
    assert schema(meta, arrays) == schema(committed_meta, committed_arrays)


@pytest.mark.parametrize('name', MODELS)
def test_resave_archive(name: str, tmp_path: Path) -> None:
    """Check loading and saving again the committed archive reproduces it."""
    cls, _ = MODELS[name]
    path = tmp_path / 'model.npz'
    cls.load_npz(archive_path(name)).save_npz(path)
    meta, arrays = read_archive(path)
    committed_meta, committed_arrays = read_archive(archive_path(name))
    assert meta == committed_meta
    assert arrays.keys() == committed_arrays.keys()
    for key, array in arrays.items():
        assert_array_equal(array, committed_arrays[key], err_msg=key)


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
        (lambda meta: meta.update(format_version=FORMAT_VERSION + 1), 'format version'),
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
    match = f'contains a {MODELS[other][0].__name__}, not a {cls.__name__}'
    with pytest.raises(TypeError, match=match):
        cls.load_npz(archive_path(other))


def test_bartz_0_13_bcf_archive(tmp_path: Path) -> None:
    """Check the archives written by `bcf.save_npz` in bartz 0.13 are rejected."""
    path = tmp_path / 'old.npz'
    np.savez(path, schema_version=1)
    with pytest.raises(ValueError, match=r'not an archive written by bartz>0\.13'):
        bcf.load_npz(path)


if __name__ == '__main__':
    ARCHIVE_DIR.mkdir(exist_ok=True)
    for name, (_, make) in MODELS.items():
        make().save_npz(archive_path(name))
