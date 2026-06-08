# bartz/config/util.py
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

import datetime
import re
import subprocess
import sys
from pathlib import Path

CHANGELOG_PATH = Path('docs/changelog.md')


def get_version() -> str:
    """Read the release version from the topmost changelog section."""
    for line in CHANGELOG_PATH.read_text().splitlines():
        if line.startswith('## '):
            version, _title, _date = _parse_changelog_header(line)
            return version
    msg = f'No release header found in {CHANGELOG_PATH}'
    raise ValueError(msg)


def _parse_changelog_header(line: str) -> tuple[str, str, str]:
    """Parse a ``## VERSION TITLE (YYYY-MM-DD)`` line, error if malformed."""
    m = re.fullmatch(r'## (\S+) (.+) \((\d{4}-\d{2}-\d{2})\)', line)
    if m is None:
        msg = f'Cannot parse changelog header: {line!r}'
        raise ValueError(msg)
    return m[1], m[2], m[3]


def _read_changelog_section() -> tuple[str, str, str, str]:
    """Read and validate the topmost changelog section.

    Returns ``(version, title, date, body)``. Raises ValueError if the
    changelog cannot be parsed or the date is not today.
    """
    lines = CHANGELOG_PATH.read_text().splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith('## ')]
    if len(headers) < 2:
        msg = f'Expected at least 2 release headers in {CHANGELOG_PATH}, found {len(headers)}'
        raise ValueError(msg)
    first, second = headers[0], headers[1]
    version, title, date = _parse_changelog_header(lines[first])
    _parse_changelog_header(lines[second])  # validate boundary header
    today = datetime.datetime.now(tz=datetime.timezone.utc).date().isoformat()
    if date != today:
        msg = f'Changelog date {date} does not match today {today}'
        raise ValueError(msg)
    body = '\n'.join(lines[first + 1 : second]).strip('\n')
    return version, title, date, body


def check_changelog() -> None:
    """Validate the topmost changelog section is parseable and dated today."""
    _read_changelog_section()


def gh_release() -> None:
    """Create a draft GitHub release from the topmost changelog section."""
    version, title, _date, body = _read_changelog_section()
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            'gh',
            'release',
            'create',
            f'v{version}',
            '--draft',
            '--verify-tag',
            '--title',
            title,
            '--notes-file',
            '-',
        ],
        input=body,
        text=True,
        check=True,
    )


def main() -> None:
    command = sys.argv[1]
    if command == 'get_version':
        print(get_version())
    elif command == 'check_changelog':
        check_changelog()
    elif command == 'gh_release':
        gh_release()
    else:
        raise ValueError(command)


if __name__ == '__main__':
    main()
