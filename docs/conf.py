# bartz/docs/conf.py
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

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import contextlib
import datetime
import importlib
import pathlib
import pkgutil
import re
import sys
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from enum import Enum
from functools import cached_property
from inspect import getsourcefile, getsourcelines, isclass, unwrap
from os import getenv

import git
from equinox import Module
from sphinx.ext.autodoc._dynamic._preserve_defaults import update_default_value

# -- Version info ------------------------------------------------------------

REPO = git.Repo(search_parent_directories=True)

COMMIT = REPO.head.commit.hexsha
UNCOMMITTED_STUFF = REPO.is_dirty()

# Check if current commit has a version tag (vX.Y.Z)
version = None
for tag in REPO.tags:
    if tag.commit == REPO.head.commit:
        MATCH = re.match(r'^v(\d+\.\d+\.\d+)$', tag.name)
        if MATCH:
            version = MATCH.group(1)
            break

if version is None:
    version = f'{COMMIT[:7]}{"+" if UNCOMMITTED_STUFF else ""}'

import bartz

# -- Project information -----------------------------------------------------

project = f'bartz {version}'
author = 'The Bartz Contributors'

NOW = datetime.datetime.now(tz=datetime.timezone.utc)
YEAR = '2024'
if NOW.year > int(YEAR):
    YEAR += '-' + str(NOW.year)
copyright = YEAR + ', ' + author  # noqa: A001, because sphinx uses this variable

release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',  # generate per-object pages and index tables
    'sphinx_autodoc_typehints',  # (!) keep after napoleon
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',  # link to other documentations automatically
    'myst_nb',  # markdown + jupyter notebook support
]

# WORKAROUND(sphinx-autodoc-typehints<3.10.0): on Python 3.14 Union type aliases
# (like jax.typing.DTypeLike = str | ...) have __module__='typing' and
# __qualname__='Union', so build_type_mapping() creates a bogus mapping
# typing.Union -> jax.typing.DTypeLike, rendering every Union as DTypeLike[...].
# See https://github.com/tox-dev/sphinx-autodoc-typehints/issues/677.
if sys.version_info >= (3, 14):
    import importlib as _importlib
    import types as _types

    def _resolves_to_union_instance(dotted_path) -> bool:  # noqa: ANN001
        """Check whether *dotted_path* points at a Union instance."""
        mod_path, _, attr = dotted_path.rpartition('.')
        if not mod_path:
            return False
        try:
            obj = getattr(_importlib.import_module(mod_path), attr)
            return isinstance(obj, _types.UnionType) and not isinstance(obj, type)
        except Exception:  # noqa: BLE001
            return False

    def _remove_union_aliases_from_mapping(app, _env, _docnames) -> None:  # noqa: ANN001
        mapping = getattr(app.config, '_intersphinx_type_mapping', None)
        if mapping:
            app.config._intersphinx_type_mapping = {  # noqa: SLF001
                k: v for k, v in mapping.items() if not _resolves_to_union_instance(v)
            }


# equinox.Module's metaclass installs a `__signature__` descriptor on every
# subclass. Sphinx's autodoc takes that short-circuit (see
# `_get_object_for_signature` in `sphinx.ext.autodoc._dynamic._signatures`)
# and never applies `autodoc_preserve_defaults`, so defaults that are class
# objects (e.g. `binner=UniqueQuantileBinner`) end up rendered via `repr()` as
# `<class 'pkg.mod.Name'>` in both the signature and the parameter list. Apply
# `update_default_value` to every equinox `Module` `__init__` we ship so the
# source-text defaults propagate through the class signature too.
def _apply_preserve_defaults_to_equinox_modules() -> None:
    """Apply autodoc preserve_defaults to every equinox.Module subclass in bartz."""
    seen: set[type] = set()
    for _finder, mod_name, _ispkg in pkgutil.walk_packages(
        bartz.__path__, prefix=f'{bartz.__name__}.'
    ):
        try:
            module = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001, S112
            continue
        for attr in vars(module).values():
            if not isinstance(attr, type) or attr in seen:
                continue
            seen.add(attr)
            if attr is Module or not issubclass(attr, Module):
                continue
            init = attr.__dict__.get('__init__')
            if init is None:
                continue
            with contextlib.suppress(Exception):
                update_default_value(init, bound_method=True)


_apply_preserve_defaults_to_equinox_modules()


# Render only the implementation signature of overloaded functions/methods.
# Otherwise autodoc emits one signature per `@overload`, and those go through a
# text path that mangles the type hints (straight quotes become typographic
# quotes) and merely duplicates the Parameters section below. Clearing the
# analyzer's overload table makes autodoc fall back to the real signature.
from sphinx.pycode import ModuleAnalyzer

_orig_module_analyze = ModuleAnalyzer.analyze


def _analyze_without_overloads(self) -> None:  # noqa: ANN001
    _orig_module_analyze(self)
    self.overloads = {}


ModuleAnalyzer.analyze = _analyze_without_overloads  # ty: ignore[invalid-assignment]


def setup(app) -> None:  # noqa: ANN001
    if sys.version_info >= (3, 14):
        # priority 501 runs after validate_config (default 500) which populates
        # the mapping
        app.connect(
            'env-before-read-docs', _remove_union_aliases_from_mapping, priority=501
        )


# decide whether to use viewcode or linkcode extension
EXT = 'viewcode'  # copy source code in static website
if getenv('BARTZ_FORCE_LINKCODE'):
    EXT = 'linkcode'  # links to code on github
elif not UNCOMMITTED_STUFF:
    BRANCHES = REPO.git.branch('--remotes', '--contains', COMMIT)
    COMMIT_ON_GITHUB = bool(BRANCHES.strip())
    if COMMIT_ON_GITHUB:
        EXT = 'linkcode'  # links to code on github
extensions.append(f'sphinx.ext.{EXT}')

myst_enable_extensions = [
    # "amsmath",
    'dollarmath'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_inventory', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

html_title = f'{project} documentation'

html_theme_options = dict(
    description='Super-fast BART (Bayesian Additive Regression Trees) in Python',
    fixed_sidebar=True,
    github_button=True,
    github_type='star',
    github_repo='bartz',
    github_user='bartz-org',
    show_relbars=True,
)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

master_doc = 'index'

# -- Other options -------------------------------------------------

default_role = 'py:obj'

# autodoc
autoclass_content = 'class'
# default arguments are printed as in source instead of being evaluated
autodoc_preserve_defaults = True
autodoc_default_options = {'member-order': 'bysource'}
# WORKAROUND(sphinx-autodoc-typehints<99): napoleon escapes the trailing
# underscore of parameter names (e.g. `lambda_`) only when
# strip_signature_backslash is on, but sphinx-autodoc-typehints always escapes
# it when looking up the `:param:` line to attach the type to, so with the
# default off the names never match and such parameters lose their type and
# default in the rendered docs. Keep this on until the upstream fix (see
# https://github.com/tox-dev/sphinx-autodoc-typehints/issues) is released.
strip_signature_backslash = True

# autosummary
# generate the per-object stub pages at build time
autosummary_generate = True
# public modules use an _src-like layout: they re-export the public API from
# private `_*` submodules, so members' `__module__` is the private submodule,
# not the public one. Documenting imported members is therefore required. The
# members are listed by hand in autosummary tables in the module docstrings;
# tests/test_docs.py checks the tables match the public namespaces.
autosummary_imported_members = True

# autodoc-typehints
typehints_use_rtype = False
typehints_document_rtype = True
always_use_bars_union = True
typehints_defaults = 'comma'

# napoleon
napoleon_google_docstring = False
napoleon_use_ivar = True
napoleon_use_rtype = False

# intersphinx
# stochtree's and equinox's docs (quarto/quartodoc and mkdocs/mkdocstrings,
# respectively) don't publish a Sphinx objects.inv, so we point intersphinx at
# vendored inventories scraped from their API references (see
# docs/_inventory/make_inventories.py).
intersphinx_mapping = dict(
    python=('https://docs.python.org/3', None),
    scipy=('https://docs.scipy.org/doc/scipy', None),
    numpy=('https://numpy.org/doc/stable', None),
    jax=('https://docs.jax.dev/en/latest', None),
    stochtree=(
        'https://stochtree.ai',
        str(pathlib.Path(__file__).parent / '_inventory' / 'stochtree.inv'),
    ),
    equinox=(
        'https://docs.kidger.site/equinox',
        str(pathlib.Path(__file__).parent / '_inventory' / 'equinox.inv'),
    ),
)

# myst_nb
nb_execution_mode = 'off'

# viewcode
viewcode_line_numbers = True


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """
    Determine the URL corresponding to Python object, for extension linkcode.

    Adapted from scipy/doc/release/conf.py.
    """
    assert domain == 'py'

    modname = info['module']
    assert modname.startswith('bartz')
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    assert submod

    obj = submod
    for part in fullname.split('.'):
        if isclass(obj) and (
            (is_dataclass(obj) and any(f.name == part for f in dataclass_fields(obj)))
            or any(
                part in getattr(klass, '__annotations__', {}) for klass in obj.__mro__
            )
        ):
            # a class data attribute (an annotation or a dataclass/equinox
            # field, possibly inherited); there is no source line to link to
            return None
        else:
            obj = getattr(obj, part)

    if isinstance(obj, cached_property):
        obj = obj.func
    elif isinstance(obj, property):
        obj = obj.fget
    elif isinstance(obj, Enum):
        obj = type(obj)
    if callable(obj):
        obj = unwrap(obj)

    fn = getsourcefile(obj)
    assert fn

    source, lineno = getsourcelines(obj)
    assert lineno
    linespec = f'#L{lineno}-L{lineno + len(source) - 1}'

    prefix = 'https://github.com/bartz-org/bartz/blob'
    root = pathlib.Path(bartz.__file__).parent
    fn_path = pathlib.Path(fn)
    if not fn_path.is_relative_to(root):
        # re-exported foreign symbol (e.g. a jaxtyping parametric type alias
        # assigned at module scope); no in-repo source to link to
        return None
    path = fn_path.relative_to(root).as_posix()
    return f'{prefix}/{COMMIT}/src/bartz/{path}{linespec}'
