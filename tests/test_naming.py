# bartz/tests/test_naming.py
#
# Copyright (c) 2024-2026, The Bartz Contributors
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

"""Check the naming convention of module-level globals.

Module-level data globals (anything that is not a function, class, module or
type alias) must use ``UPPER_CASE`` names, since ruff has no rule for this. This
holds for private globals too: the casing signals global (rather than local)
semantics when reading the code, regardless of visibility.
"""

import ast
import importlib
import inspect
import pkgutil
import sys
from types import ModuleType
from typing import Any, Literal, TypeVar, get_origin

import pytest

import bartz


def all_modules() -> list[ModuleType]:
    """Import and return every module in the `bartz` package."""
    submodules = [
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(bartz.__path__, f'{bartz.__name__}.')
    ]
    return [bartz, *submodules]


# WORKAROUND(python<3.12): use `ast.TypeAlias` directly
_TYPE_ALIAS_STMT = getattr(ast, 'TypeAlias', ())  # PEP 695 `type X = ...`


def is_type_alias_annotation(node: ast.expr) -> bool:
    """Whether an annotation is `TypeAlias` (e.g. ``Foo: TypeAlias = ...``)."""
    return (isinstance(node, ast.Name) and node.id == 'TypeAlias') or (
        isinstance(node, ast.Attribute) and node.attr == 'TypeAlias'
    )


def assigned_names_in_source(source: str) -> set[str]:
    """Names bound by top-level data assignments (not import/def/class/alias)."""
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, _TYPE_ALIAS_STMT):
            continue  # `type X = ...` (PEP 695) is a type declaration
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if is_type_alias_annotation(node.annotation):
                continue  # `Foo: TypeAlias = ...` is a type declaration
            targets = [node.target]
        else:
            continue  # imports, def/class
        for target in targets:
            names.update(
                sub.id
                for sub in ast.walk(target)
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store)
            )
    return names


def assigned_names(module: ModuleType) -> set[str]:
    """Names bound by top-level data assignments in `module`."""
    return assigned_names_in_source(inspect.getsource(module))


def is_type_alias(obj: object) -> bool:
    """Coarse test for a type-alias-like object that is not callable."""
    # `X | Y`, `Union[...]`, `Literal[...]`, `list[int]`, ... have a typing
    # origin; bare special forms (`Any`, `TypeVar`, ...) live in `typing`.
    return get_origin(obj) is not None or type(obj).__module__ == 'typing'


def name_ok(name: str, obj: object) -> bool:
    """Whether a module-level binding satisfies the naming convention."""
    return (
        (name.startswith('__') and name.endswith('__'))  # dunder
        or inspect.ismodule(obj)
        or callable(obj)  # function, class, most type aliases
        or name.isupper()  # constant or TypeVar (underscores ignored)
        or is_type_alias(obj)  # remaining aliases, e.g. `X | Y` unions
    )


@pytest.mark.parametrize('module', all_modules(), ids=lambda m: m.__name__)
def test_global_data_uppercase(module: ModuleType) -> None:
    """Check that module-level data globals are `UPPER_CASE`."""
    bad = sorted(
        name
        for name in assigned_names(module)
        if not name_ok(name, getattr(module, name, None))
    )
    assert not bad, f'non-uppercase data globals in {module.__name__}: {bad}'


def test_name_ok_classifies_offenders() -> None:
    """Spot-check `name_ok` on representative bindings."""
    # accepted: constants, dunders, callables, and genuine type aliases
    assert name_ok('MAX_ITER', 20)
    assert name_ok('_PRIVATE_CONST', 'x')
    assert name_ok('__version__', '1.0')
    assert name_ok('helper', lambda: None)
    assert name_ok('MyClass', type('MyClass', (), {}))
    assert name_ok('ArrayLike', int | str)
    assert name_ok('IntList', list[int])
    assert name_ok('Kind', Literal['a', 'b'])
    assert name_ok('Anything', Any)
    assert name_ok('KeyType', TypeVar('KeyType'))

    # rejected: CapWords or lowercase names bound to plain data
    assert not name_ok('Config', {'a': 1})
    assert not name_ok('_Registry', {})
    assert not name_ok('Defaults', [1, 2, 3])
    assert not name_ok('_counter', 0)


def test_type_aliases_are_exempt() -> None:
    """Explicit type aliases are not collected as data globals."""
    src = 'from typing import TypeAlias\nFoo: TypeAlias = int\nBAR = 1\n'
    assert assigned_names_in_source(src) == {'BAR'}

    if sys.version_info >= (3, 12):  # PEP 695 `type` statement
        assert assigned_names_in_source('type Baz = int\nBAR = 1\n') == {'BAR'}
