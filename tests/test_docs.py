# bartz/tests/test_docs.py
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

"""Check the curated API listings in the docs cover the whole public API.

Each public module's docstring contains autosummary tables organized by topic
(scipy-style), and `docs/reference/index.rst` lists the top-level objects and
the public modules. Both are written by hand, so check they stay in sync with
the actual public names.
"""

import os
from importlib import import_module
from inspect import ismodule
from pathlib import Path
from pkgutil import iter_modules
from types import ModuleType

import pytest

import bartz

PUBLIC_MODULES = (
    'bartz.BART',
    'bartz.debug',
    'bartz.grove',
    'bartz.mcmcloop',
    'bartz.mcmcstep',
    'bartz.prepcovars',
    'bartz.stochtree',
    'bartz.testing',
)


def autosummary_entries(text: str) -> list[str]:
    """Extract the entries of all autosummary directives in rst text."""
    entries = []
    in_directive = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('.. autosummary::'):
            in_directive = True
        elif in_directive and (not stripped or stripped.startswith(':')):
            pass  # blank line or directive option
        elif in_directive and line[:1].isspace():
            # drop the leading `~` (rst "show last component only") so the
            # entry is the bare name / importable path callers expect
            entries.append(stripped.lstrip('~'))
        else:
            in_directive = False
    return entries


def public_names(module: ModuleType) -> set[str]:
    """Return the non-module public names in a module's namespace."""
    return {
        name
        for name, obj in vars(module).items()
        if not name.startswith('_') and not ismodule(obj)
    }


def test_public_modules_complete() -> None:
    """Check `PUBLIC_MODULES` matches the public submodules of bartz."""
    found = {
        f'bartz.{info.name}'
        for info in iter_modules(bartz.__path__)
        if not info.name.startswith('_')
    }
    assert found == set(PUBLIC_MODULES)


@pytest.mark.parametrize('module_name', PUBLIC_MODULES)
def test_module_docstring_lists_public_api(module_name: str) -> None:
    """Check the docstring autosummary tables match the public names exactly."""
    module = import_module(module_name)
    assert module.__doc__ is not None
    entries = autosummary_entries(module.__doc__)
    assert len(entries) == len(set(entries)), 'duplicate autosummary entries'
    assert set(entries) == public_names(module)


def resolve_entry(entry: str) -> object:
    """Import the object referenced by an autosummary entry."""
    try:
        return import_module(entry)
    except ModuleNotFoundError:
        module_name, _, attr = entry.rpartition('.')
        return getattr(import_module(module_name), attr)


def test_reference_index_lists_top_level_api() -> None:
    """Check docs/reference/index.rst covers the top-level public objects."""
    # `resolve_entry` tells a class (e.g. `bartz.Bart`) from a same-spelled,
    # different-case module (`bartz.BART`) by relying on case-sensitive imports.
    # On case-insensitive filesystems CPython enforces case only when
    # PYTHONCASEOK is unset, so guard against it.
    assert 'PYTHONCASEOK' not in os.environ
    index = Path(__file__).parent.parent / 'docs' / 'reference' / 'index.rst'
    entries = autosummary_entries(index.read_text())
    assert len(entries) == len(set(entries)), 'duplicate autosummary entries'
    documented = {resolve_entry(entry) for entry in entries}
    expected = {getattr(bartz, name) for name in public_names(bartz)}
    expected |= {import_module(name) for name in PUBLIC_MODULES}
    assert documented == expected
