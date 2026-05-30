# bartz/docs/_inventory/make_stochtree_inventory.py
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

"""Generate the vendored intersphinx inventory for stochtree.

stochtree's documentation site (https://stochtree.ai) is built with Quarto /
quartodoc and does *not* publish a Sphinx ``objects.inv``, so plain intersphinx
cannot link to it. This script scrapes the quartodoc API reference and writes a
Sphinx v2 inventory next to it (``stochtree.inv``) that ``docs/conf.py`` feeds to
intersphinx as a local inventory. The quartodoc anchors are real Python-domain
ids (e.g. ``stochtree.bart.BARTModel``), and stochtree re-exports the documented
symbols at top level, so we also emit the short aliases bartz uses in its
docstrings (e.g. ``stochtree.BARTModel``).

Re-run manually whenever stochtree's public API changes::

    uv run python docs/_inventory/make_stochtree_inventory.py
"""

from __future__ import annotations

import importlib
import inspect
import re
import urllib.request
import zlib
from importlib import metadata
from pathlib import Path

BASE = 'https://stochtree.ai'
REFERENCE = 'python-api/reference'
OUT = Path(__file__).parent / 'stochtree.inv'


def fetch(url: str) -> str:
    """Return the decoded body of url (quartodoc's host wants a browser UA)."""
    request = urllib.request.Request(  # noqa: S310
        url, headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read().decode()


def reference_pages() -> list[str]:
    """List the quartodoc reference page filenames, from the sitemap."""
    sitemap = fetch(f'{BASE}/sitemap.xml')
    pages = set(re.findall(rf'{REFERENCE}/([^<]+\.html)', sitemap))
    pages.discard('index.html')
    return sorted(pages)


def page_anchors(page: str) -> list[str]:
    """List the ``stochtree.*`` element ids documented on a reference page."""
    html = fetch(f'{BASE}/{REFERENCE}/{page}')
    return sorted(set(re.findall(r'id="(stochtree\.[^"]+)"', html)))


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


def build_entries() -> dict[str, tuple[str, str]]:
    """Map inventory name -> (role, uri) for every documented stochtree symbol."""
    entries: dict[str, tuple[str, str]] = {
        # bare ``stochtree`` links to the Python API reference landing page
        'stochtree': ('module', f'{REFERENCE}/index.html')
    }
    for page in reference_pages():
        for anchor in page_anchors(page):
            role = role_of(anchor)
            entries[anchor] = (role, f'{REFERENCE}/{page}#$')
            alias = toplevel_alias(anchor)
            if alias and alias not in entries:
                entries[alias] = (role, f'{REFERENCE}/{page}#{anchor}')
    return entries


def dump(entries: dict[str, tuple[str, str]], version: str) -> bytes:
    """Serialize entries as a Sphinx v2 ``objects.inv`` byte string."""
    body = ''.join(
        f'{name} py:{role} 1 {uri} -\n' for name, (role, uri) in sorted(entries.items())
    )
    header = (
        '# Sphinx inventory version 2\n'
        '# Project: stochtree\n'
        f'# Version: {version}\n'
        '# The remainder of this file is compressed using zlib.\n'
    )
    return header.encode() + zlib.compress(body.encode())


def main() -> None:
    """Build and write the inventory."""
    version = metadata.version('stochtree')
    entries = build_entries()
    OUT.write_bytes(dump(entries, version))
    print(f'wrote {len(entries)} entries to {OUT}')


if __name__ == '__main__':
    main()
