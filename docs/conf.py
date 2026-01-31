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

import datetime
import pathlib
import re
import sys
from functools import cached_property
from inspect import getsourcefile, getsourcelines, isclass, unwrap
from os import getenv

import git

# -- Version info ------------------------------------------------------------

repo = git.Repo(search_parent_directories=True)

commit = repo.head.commit.hexsha
uncommitted_stuff = repo.is_dirty()

# Check if current commit has a version tag (vX.Y.Z)
version = None
for tag in repo.tags:
    if tag.commit == repo.head.commit:
        match = re.match(r'^v(\d+\.\d+\.\d+)$', tag.name)
        if match:
            version = match.group(1)
            break

if version is None:
    version = f'{commit[:7]}{"+" if uncommitted_stuff else ""}'

import bartz

# -- Project information -----------------------------------------------------

project = f'bartz {version}'
author = 'The Bartz Contributors'

now = datetime.datetime.now(tz=datetime.timezone.utc)
year = '2024'
if now.year > int(year):
    year += '-' + str(now.year)
copyright = year + ', ' + author  # noqa: A001, because sphinx uses this variable

release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',  # (!) keep after napoleon
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',  # link to other documentations automatically
    'myst_nb',  # markdown + jupyter notebook support
]

# decide whether to use viewcode or linkcode extension
ext = 'viewcode'  # copy source code in static website
if getenv('BARTZ_FORCE_LINKCODE'):
    ext = 'linkcode'  # links to code on github
elif not uncommitted_stuff:
    commit = repo.head.commit.hexsha
    branches = repo.git.branch('--remotes', '--contains', commit)
    commit_on_github = bool(branches.strip())
    if commit_on_github:
        ext = 'linkcode'  # links to code on github
extensions.append(f'sphinx.ext.{ext}')

myst_enable_extensions = [
    # "amsmath",
    'dollarmath'
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates'] # noqa: ERA001

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


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
autoclass_content = 'both'  # concatenate the class and __init__ docstrings
# default arguments are printed as in source instead of being evaluated
autodoc_preserve_defaults = True
autodoc_default_options = {'member-order': 'bysource'}

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
intersphinx_mapping = dict(
    scipy=('https://docs.scipy.org/doc/scipy', None),
    numpy=('https://numpy.org/doc/stable', None),
    jax=('https://docs.jax.dev/en/latest', None),
)

# myst_nb
nb_execution_mode = 'off'

# viewcode
viewcode_line_numbers = True


def linkcode_resolve(domain, info):
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
        if isclass(obj) and part in obj.__annotations__:
            # this is a class attribute and it does not make much sense to
            # create the source link
            return None
        else:
            obj = getattr(obj, part)

    if isinstance(obj, cached_property):
        obj = obj.func
    elif isinstance(obj, property):
        obj = obj.fget
    obj = unwrap(obj)

    fn = getsourcefile(obj)
    assert fn

    source, lineno = getsourcelines(obj)
    assert lineno
    linespec = f'#L{lineno}-L{lineno + len(source) - 1}'

    prefix = 'https://github.com/bartz-org/bartz/blob'
    root = pathlib.Path(bartz.__file__).parent
    path = pathlib.Path(fn).relative_to(root).as_posix()
    return f'{prefix}/{commit}/src/bartz/{path}{linespec}'
