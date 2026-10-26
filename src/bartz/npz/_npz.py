# bartz/src/bartz/npz/_npz.py
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

"""Serialize trees of registered dataclasses to npz archives.

The archive contains one ``.npy`` member per array, named by its path in the
tree, and a ``meta.json`` member with the format version and a skeleton of the
tree. The skeleton is made of JSON scalars and lists, plus objects tagged by
``type``: ``module`` (a registered dataclass), ``dict`` (with `str` keys),
``tuple``, and ``array`` (a jax array, stored in the member ``key``).
"""

import json
from collections.abc import Mapping
from dataclasses import fields
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias, TypeVar
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from jax import Array, device_get, device_put
from numpy.lib.npyio import NpzFile

from bartz._version import __version__
from bartz.npz._npz_upgrades import UPGRADES

FORMAT_VERSION = len(UPGRADES)

META_KEY = 'meta.json'

_REGISTRY: dict[str, type] = {}

# a JSON-serializable value, like the skeleton of an object produced by `encode`
Json: TypeAlias = Any

T = TypeVar('T')


def qualified_name(cls: type) -> str:
    """Return the name of `cls` including its module."""
    return f'{cls.__module__}.{cls.__qualname__}'


def is_registered(cls: type) -> bool:
    """Check if `cls` is registered, allowing for its module being reloaded."""
    registered = _REGISTRY.get(cls.__name__)
    return registered is not None and qualified_name(registered) == qualified_name(cls)


def serializable(cls: type[T]) -> type[T]:
    """
    Register a dataclass to allow its instances in npz archives.

    Use it as a class decorator, e.g., on a custom `~bartz.mcmcloop.Callback`,
    to save a `~bartz.Bart` fitted with it.

    Parameters
    ----------
    cls
        The dataclass to register, e.g., an `equinox.Module`. Its fields may
        hold `None`, `bool`, `int`, `float`, `str`, lists, tuples, dicts with
        `str` keys without slashes, jax arrays that are not weakly typed, and
        instances of registered classes.

    Returns
    -------
    `cls` itself.

    Raises
    ------
    ValueError
        If a different class with the same name is already registered,
        including the classes of bartz.

    Notes
    -----
    Classes are identified by name alone, so registered classes must have
    unique names. Registering again the same class, e.g., after reloading its
    module, replaces it. Registration is not inherited: register each subclass
    explicitly.

    To load an archive, its classes must be registered, i.e., their modules
    imported, with the same fields as when saving. Loading sets the fields
    directly, without calling ``__init__``.
    """
    # the fields of the bartz classes are the archive format, check changes
    # with tests/test_npz.py
    tag = cls.__name__
    if tag in _REGISTRY and not is_registered(cls):
        msg = (
            f'class name {tag!r} already registered by {qualified_name(_REGISTRY[tag])}'
        )
        raise ValueError(msg)
    _REGISTRY[tag] = cls
    return cls


def join(path: str, name: str) -> str:
    """Append `name` to the archive path `path`."""
    return f'{path}/{name}' if path else name


def encode(obj: object, path: str, arrays: dict[str, Array]) -> Json:
    """Convert `obj` to a JSON skeleton, moving the arrays to `arrays`."""
    if obj is None or type(obj) in (bool, int, float, str):
        return obj
    elif type(obj) is list:
        return [encode(v, join(path, str(i)), arrays) for i, v in enumerate(obj)]
    elif type(obj) is tuple:
        items = [encode(v, join(path, str(i)), arrays) for i, v in enumerate(obj)]
        return dict(type='tuple', items=items)
    elif type(obj) is dict:
        items = {}
        for k, v in obj.items():
            if not isinstance(k, str) or '/' in k:
                msg = f'{path}: dict key {k!r} is not a str without slashes'
                raise TypeError(msg)
            items[k] = encode(v, join(path, k), arrays)
        return dict(type='dict', items=items)
    elif isinstance(obj, Array):
        if obj.weak_type:
            msg = f'{path}: weakly typed arrays are not supported'
            raise TypeError(msg)
        arrays[path] = obj
        return dict(type='array', key=path)
    elif is_registered(type(obj)):
        values = {
            f.name: encode(getattr(obj, f.name), join(path, f.name), arrays)
            for f in fields(obj)  # ty: ignore[invalid-argument-type]
        }
        return dict(type='module', cls=type(obj).__name__, fields=values)
    else:
        msg = f'{path}: unsupported type {type(obj).__qualname__}'
        raise TypeError(msg)


def decode(
    node: Json, data: Mapping[str, np.ndarray | bytes], used: set[str]
) -> object:
    """Rebuild the object encoded by `encode`, reading the arrays from `data`."""
    if node is None or type(node) in (bool, int, float, str):
        return node
    elif type(node) is list:
        return [decode(v, data, used) for v in node]
    elif node['type'] == 'tuple':
        return tuple(decode(v, data, used) for v in node['items'])
    elif node['type'] == 'dict':
        return {k: decode(v, data, used) for k, v in node['items'].items()}
    elif node['type'] == 'array':
        key = node['key']
        used.add(key)
        # `device_put` wraps aligned host arrays without copying
        return device_put(data[key])
    elif node['type'] == 'module':
        cls = _REGISTRY.get(node['cls'])
        if cls is None:
            msg = f'unknown class {node["cls"]!r}'
            raise ValueError(msg)
        names = {f.name for f in fields(cls)}  # ty: ignore[invalid-argument-type]
        saved = set(node['fields'])
        if saved != names:
            msg = (
                f'fields of {cls.__name__} do not match: missing in the archive '
                f'{sorted(names - saved)}, unexpected {sorted(saved - names)}'
            )
            raise ValueError(msg)
        # bypass `__init__`, which may for example run the MCMC
        obj = object.__new__(cls)
        for name, value in node['fields'].items():
            object.__setattr__(obj, name, decode(value, data, used))
        return obj
    else:
        msg = f'unknown node type {node["type"]!r}'
        raise ValueError(msg)


def save_npz(path: str | PathLike, obj: object, *, compresslevel: int = 3) -> None:
    """Save a tree of registered dataclasses to an npz archive."""
    arrays: dict[str, Array] = {}
    root = encode(obj, '', arrays)
    meta = dict(format_version=FORMAT_VERSION, bartz_version=__version__, root=root)

    # write to a temporary file and then rename it, so a failure does not
    # destroy an existing file at `path`
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        with (
            tmp.open('xb') as file,
            ZipFile(
                file, 'w', compression=ZIP_DEFLATED, compresslevel=compresslevel
            ) as zf,
        ):
            # an explicit `ZipInfo` gets a fixed date like the arrays below,
            # instead of the current time, so the archive is reproducible;
            # then compression must be set explicitly
            zf.writestr(
                ZipInfo(META_KEY),
                json.dumps(meta, indent=1),
                compress_type=ZIP_DEFLATED,
                compresslevel=compresslevel,
            )
            # `device_get` copies all the arrays to the host in parallel, and
            # does not copy on cpu
            for key, value in device_get(arrays).items():
                with zf.open(f'{key}.npy', 'w', force_zip64=True) as member:
                    np.lib.format.write_array(member, value, allow_pickle=False)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def check_class(obj: object, cls: type[T], path: str | PathLike) -> T:
    """Check the object loaded from `path` is an instance of `cls`."""
    if not isinstance(obj, cls):
        msg = f'{path} contains a {type(obj).__name__}, not a {cls.__name__}'
        raise TypeError(msg)
    return obj


def read_meta(path: str | PathLike, data: NpzFile) -> tuple[int, Json]:
    """Return the format version and skeleton of the archive `data` read from `path`."""
    if META_KEY in data.files:
        meta = json.loads(data[META_KEY])
        version = meta['format_version']
        if version > FORMAT_VERSION:
            msg = (
                f'{path} has format version {version}, written by bartz '
                f'{meta["bartz_version"]}, this bartz reads up to version '
                f'{FORMAT_VERSION}: upgrade bartz'
            )
            raise ValueError(msg)
        return version, meta['root']
    elif 'schema_version' in data.files:  # `bcf` archive written by bartz 0.13
        return 0, None
    else:
        msg = f'{path} is not an archive written by bartz'
        raise ValueError(msg)


def load_npz(path: str | PathLike) -> object:
    """Load an object saved with `save_npz`, or with an older format."""
    with np.load(path, allow_pickle=False) as data:
        version, root = read_meta(path, data)
        arrays: Mapping[str, np.ndarray] = data
        if version < FORMAT_VERSION:
            arrays = {k: data[k] for k in data.files}
            for upgrade in UPGRADES[version:]:
                root, arrays = upgrade(root, arrays)
        used: set[str] = set()
        obj = decode(root, arrays, used)
        unused = set(arrays) - used - {META_KEY}
        if unused:
            msg = f'{path} contains unused arrays {sorted(unused)}'
            raise ValueError(msg)
    return obj
