# bartz/config/check_path_headers.py
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

"""Check the `bartz/<path>` comment atop each source file matches its location.

Source files open with a comment line `bartz/<repo-relative-path>` just above the
MIT license header. The comment marker varies by language (`#`, `..`, `{#`, `/*`,
or none inside an HTML comment); only the path is checked. This catches a header
left stale after a file is moved or renamed.

Files without the bartz license header are out of scope and skipped. Pass the
files to check as command-line arguments, as pre-commit does.
"""

import re
import sys
from pathlib import Path

# Present in every per-file license header but not in the project LICENSE, so it
# scopes the check to files carrying the header.
ANCHOR_RE = re.compile(r'This file is part of bartz\.')
# The path line, stripped of an optional comment marker, is `bartz/<path>`.
PATH_RE = re.compile(
    r'^\s*(?:#|\.\.|\{#|/\*|<!--|\*)?\s*bartz/(?P<path>.+?)(?:\s*(?:\*/|\#\}|-->))?\s*$'
)
# Lines scanned for the header, enough to clear an optional shebang or `<!--`.
WINDOW = 10


def check(path: Path) -> str | None:
    """Return an error message if `path`'s header is wrong, else None."""
    try:
        lines = path.read_text(encoding='utf-8').splitlines()[:WINDOW]
    except (UnicodeDecodeError, FileNotFoundError):
        return None  # binary or deleted file: nothing to check
    anchor = next((i for i, line in enumerate(lines) if ANCHOR_RE.search(line)), None)
    if anchor is None:
        return None  # no license header: out of scope
    for line in lines[:anchor]:
        match = PATH_RE.match(line)
        if match is not None:
            got = match.group('path')
            expected = path.as_posix()
            if got != expected:
                return f'path header is `bartz/{got}`, expected `bartz/{expected}`'
            return None
    return 'license header present but no `bartz/<path>` line above it'


def main(argv: list[str]) -> int:
    failed = False
    for arg in argv:
        message = check(Path(arg))
        if message is not None:
            print(f'{arg}: {message}', file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
