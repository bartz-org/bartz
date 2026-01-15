# bartz/benchmarks/_vendor_latest_bartz.py
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

"""
Vendor (copy and patch) the latest bartz source for use in benchmarks.

This module copies the bartz package from src/bartz to benchmarks/latest_bartz
and rewrites all absolute imports from 'bartz.' to 'benchmarks.latest_bartz.'.

This allows benchmark code to use utilities from the latest development version
of bartz while ASV tests against older installed versions of the package.

Usage:
    In benchmarks/__init__.py:
        from benchmarks._vendor_latest_bartz import ensure_vendored
        ensure_vendored()

    Then in benchmark files:
        from benchmarks.latest_bartz.testing import gen_data
"""

import hashlib
import re
import shutil
from pathlib import Path

# Paths relative to this file
_THIS_DIR = Path(__file__).parent
_SOURCE_DIR = _THIS_DIR.parent / 'src' / 'bartz'
_VENDOR_DIR = _THIS_DIR / 'latest_bartz'
_HASH_FILE = _VENDOR_DIR / '.source_hash'

# Import rewriting pattern: matches 'from bartz.' or 'import bartz.'
# but not 'from benchmarks.latest_bartz.' (already rewritten)
_IMPORT_PATTERN = re.compile(r'^(\s*)(from|import)(\s+)bartz\b', re.MULTILINE)


def _compute_source_hash() -> str:
    """Compute a hash of all Python files in the source directory."""
    hasher = hashlib.sha256()

    # Sort for deterministic ordering
    for path in sorted(_SOURCE_DIR.rglob('*.py')):
        # Include relative path in hash so renames are detected
        rel_path = path.relative_to(_SOURCE_DIR)
        hasher.update(str(rel_path).encode())
        hasher.update(path.read_bytes())

    return hasher.hexdigest()


def _rewrite_imports(content: str) -> str:
    """Rewrite 'from bartz.' and 'import bartz.' to use 'benchmarks.latest_bartz.'."""

    def replacer(match: re.Match) -> str:
        indent = match.group(1)
        keyword = match.group(2)  # 'from' or 'import'
        space = match.group(3)
        return f'{indent}{keyword}{space}benchmarks.latest_bartz'

    return _IMPORT_PATTERN.sub(replacer, content)


def _copy_and_patch() -> None:
    """Copy the source directory and patch all imports."""
    # Remove existing vendor directory if it exists
    if _VENDOR_DIR.exists():
        shutil.rmtree(_VENDOR_DIR)

    # Create vendor directory
    _VENDOR_DIR.mkdir(parents=True)

    # Copy and patch all files
    for src_path in _SOURCE_DIR.rglob('*'):
        rel_path = src_path.relative_to(_SOURCE_DIR)
        dst_path = _VENDOR_DIR / rel_path

        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
        elif src_path.suffix == '.py':
            # Read, patch imports, write
            content = src_path.read_text(encoding='utf-8')
            patched = _rewrite_imports(content)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            dst_path.write_text(patched, encoding='utf-8')
        else:
            # Copy non-Python files as-is
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)

    # Write hash file
    source_hash = _compute_source_hash()
    _HASH_FILE.write_text(source_hash, encoding='utf-8')


def _is_up_to_date() -> bool:
    """Check if the vendored copy is up-to-date with the source."""
    if not _VENDOR_DIR.exists() or not _HASH_FILE.exists():
        return False

    cached_hash = _HASH_FILE.read_text(encoding='utf-8').strip()
    current_hash = _compute_source_hash()
    return cached_hash == current_hash


def ensure_vendored() -> None:
    """Ensure the vendored latest_bartz is up-to-date."""
    if not _is_up_to_date():
        _copy_and_patch()


if __name__ == '__main__':
    # Allow running directly to force re-vendoring
    print(f'Source: {_SOURCE_DIR}')
    print(f'Vendor: {_VENDOR_DIR}')
    print(f'Up-to-date: {_is_up_to_date()}')
    print('Re-vendoring...')
    _copy_and_patch()
    print('Done.')
