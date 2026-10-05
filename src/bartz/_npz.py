# bartz/src/bartz/_npz.py
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
``type``: ``module`` (a registered dataclass), ``dict`` (with `str` keys), and
``array`` (a jax array, stored in the member ``key``).
"""

import json
from collections.abc import Mapping
from dataclasses import fields
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias, TypeVar
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from jax import Array, device_put

from bartz._version import __version__

FORMAT_VERSION = 1

META_KEY = 'meta.json'

_REGISTRY: dict[str, type] = {}

# a JSON-serializable value, like the skeleton of an object produced by `encode`
Json: TypeAlias = Any

C = TypeVar('C', bound=type)
T = TypeVar('T')


def qualified_name(cls: type) -> str:
    """Return the name of `cls` including its module."""
    return f'{cls.__module__}.{cls.__qualname__}'


def is_registered(cls: type) -> bool:
    """Check if `cls` is registered, allowing for its module being reloaded."""
    registered = _REGISTRY.get(cls.__name__)
    return registered is not None and qualified_name(registered) == qualified_name(cls)


def serializable(cls: C) -> C:
    """Register a dataclass as allowed in npz archives, under its name.

    Registering again the same class, e.g., after reloading its module,
    replaces it.
    """
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
            zf.writestr(META_KEY, json.dumps(meta, indent=1))
            for key, value in arrays.items():
                with zf.open(f'{key}.npy', 'w', force_zip64=True) as member:
                    # `asarray` does not copy on cpu
                    np.lib.format.write_array(
                        member, np.asarray(value), allow_pickle=False
                    )
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


def load_npz(path: str | PathLike) -> object:
    """Load an object saved with `save_npz`."""
    with np.load(path, allow_pickle=False) as data:
        if META_KEY not in data.files:
            msg = f'{path} is not an archive written by bartz>0.13'
            raise ValueError(msg)
        meta = json.loads(data[META_KEY])
        version = meta['format_version']
        if version != FORMAT_VERSION:
            msg = (
                f'{path} has format version {version} (written by bartz '
                f'{meta["bartz_version"]}), this bartz reads version {FORMAT_VERSION}'
            )
            raise ValueError(msg)
        used: set[str] = set()
        obj = decode(meta['root'], data, used)
        unused = set(data.files) - used - {META_KEY}
        if unused:
            msg = f'{path} contains unused arrays {sorted(unused)}'
            raise ValueError(msg)
    return obj
