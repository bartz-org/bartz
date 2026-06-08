# bartz/docs/_inventory/make_inventories.py
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

"""Generate the vendored intersphinx inventories for stochtree and equinox.

Neither documentation site publishes a Sphinx ``objects.inv``: stochtree
(https://stochtree.ai) is built with Quarto / quartodoc, equinox
(https://docs.kidger.site/equinox) with mkdocs / mkdocstrings. Both, however,
use real Python-domain ids as element anchors (e.g.
``stochtree.bart.BARTModel``, ``equinox.tree_at``), so this script scrapes
their API reference pages and writes Sphinx v2 inventories next to it
(``stochtree.inv``, ``equinox.inv``) that ``docs/conf.py`` feeds to intersphinx
as local inventories. stochtree re-exports the documented symbols at top
level, so for it we also emit the short aliases bartz uses in its docstrings
(e.g. ``stochtree.BARTModel``); equinox anchors are already the public paths.

Re-run manually whenever stochtree's or equinox's public API changes::

    uv run python docs/_inventory/make_inventories.py
"""

from __future__ import annotations

import importlib
import inspect
import re
import urllib.request
import zlib
from importlib import metadata
from pathlib import Path

OUTDIR = Path(__file__).parent


def fetch(url: str) -> str:
    """Return the decoded body of url (some hosts want a browser UA)."""
    request = urllib.request.Request(  # noqa: S310
        url, headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read().decode()


def resolve(fullname: str) -> object | None:
    """Import and return the object named fullname, or None if absent."""
    parts = fullname.split('.')
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module('.'.join(parts[:i]))
        except ImportError:
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return None
        return obj
    return None


def role_of(fullname: str) -> str:
    """Best-effort Python-domain role for fullname."""
    obj = resolve(fullname)
    parent = resolve(fullname.rsplit('.', 1)[0]) if '.' in fullname else None
    if inspect.isclass(obj):
        return 'class'
    if isinstance(obj, property):
        return 'property'
    if inspect.isroutine(obj):
        return 'method' if inspect.isclass(parent) else 'function'
    return 'attribute'


def page_anchors(html: str, package: str) -> list[str]:
    """List the ``package.*`` element ids documented in a reference page."""
    # element ids are Python-domain dotted paths, so we match only `.`-joined
    # word characters; an id with any other character would be skipped, but real
    # Python identifiers never contain one.
    return sorted(set(re.findall(rf'id="({re.escape(package)}(?:\.\w+)+)"', html)))


def build_stochtree_entries() -> dict[str, tuple[str, str]]:
    """Map inventory name -> (role, uri) for every documented stochtree symbol."""
    base = 'https://stochtree.ai'
    reference = 'python-api/reference'

    def toplevel_alias(fullname: str) -> str | None:
        """Short ``stochtree.<name>`` alias for fullname, if it is re-exported."""
        parts = fullname.split('.')
        if len(parts) < 3 or parts[0] != 'stochtree':
            return None
        head = parts[2]
        package = importlib.import_module('stochtree')
        if getattr(package, head, None) is resolve(f'stochtree.{parts[1]}.{head}'):
            return 'stochtree.' + '.'.join(parts[2:])
        return None

    entries: dict[str, tuple[str, str]] = {
        # bare ``stochtree`` links to the Python API reference landing page
        'stochtree': ('module', f'{reference}/index.html')
    }
    sitemap = fetch(f'{base}/sitemap.xml')
    pages = set(re.findall(rf'{re.escape(reference)}/([^<]+\.html)', sitemap))
    pages.discard('index.html')
    for page in sorted(pages):
        html = fetch(f'{base}/{reference}/{page}')
        for anchor in page_anchors(html, 'stochtree'):
            role = role_of(anchor)
            entries[anchor] = (role, f'{reference}/{page}#$')
            alias = toplevel_alias(anchor)
            if alias and alias not in entries:
                entries[alias] = (role, f'{reference}/{page}#{anchor}')
    return entries


def build_equinox_entries() -> dict[str, tuple[str, str]]:
    """Map inventory name -> (role, uri) for every documented equinox symbol."""
    base = 'https://docs.kidger.site/equinox'
    sitemap = fetch(f'{base}/sitemap.xml')
    pages = set(re.findall(rf'{re.escape(base)}/(api/[^<]+)', sitemap))
    entries: dict[str, tuple[str, str]] = {}
    for page in sorted(pages):
        html = fetch(f'{base}/{page}')
        for anchor in page_anchors(html, 'equinox'):
            entries[anchor] = (role_of(anchor), f'{page}#$')
    return entries


def dump(entries: dict[str, tuple[str, str]], project: str, version: str) -> bytes:
    """Serialize entries as a Sphinx v2 ``objects.inv`` byte string."""
    body = ''.join(
        f'{name} py:{role} 1 {uri} -\n' for name, (role, uri) in sorted(entries.items())
    )
    header = (
        '# Sphinx inventory version 2\n'
        f'# Project: {project}\n'
        f'# Version: {version}\n'
        '# The remainder of this file is compressed using zlib.\n'
    )
    return header.encode() + zlib.compress(body.encode())


def main() -> None:
    """Build and write the inventories."""
    builders = dict(stochtree=build_stochtree_entries, equinox=build_equinox_entries)
    for project, build in builders.items():
        entries = build()
        out = OUTDIR / f'{project}.inv'
        out.write_bytes(dump(entries, project, metadata.version(project)))
        print(f'wrote {len(entries)} entries to {out}')


if __name__ == '__main__':
    main()
