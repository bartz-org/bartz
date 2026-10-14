# bartz/config/update_cooldown.py
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

"""Pin the dependency cooldown in pyproject.toml to an absolute instant.

uv's `[tool.uv] exclude-newer` accepts a relative span (e.g. `1 week`), but a
span re-resolves against wall-clock time, so `make release` and the uv-lock
hook would compute different cutoffs and churn the lockfile. Pin it
to `today - cooldown_days` as an RFC 3339 UTC instant instead: an absolute
value both agree on, anchored to the UTC day so same-day runs are reproducible.
"""

import argparse
import datetime
import re
import sys
from pathlib import Path

from config._common import format_utc, parse_datetime


def pin_exclude_newer(pyproject_path: Path, cutoff: datetime.datetime) -> bool:
    """Rewrite the `exclude-newer` value in pyproject.toml; True if changed."""
    text = pyproject_path.read_text()
    pattern = re.compile(r'^exclude-newer = .*$', re.MULTILINE)
    new_text, n = pattern.subn(f"exclude-newer = '{format_utc(cutoff)}'", text)
    if n != 1:
        msg = f'expected one exclude-newer line in {pyproject_path}, found {n}'
        raise RuntimeError(msg)
    if new_text == text:
        return False
    pyproject_path.write_text(new_text)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--today',
        required=True,
        type=parse_datetime,
        help='RFC 3339 timestamp (with timezone); reference instant for the policy.',
    )
    parser.add_argument(
        '--cooldown-days',
        required=True,
        type=int,
        help='Days to hold back from today (the dependency cooldown).',
    )
    parser.add_argument('--pyproject', type=Path, default=Path('pyproject.toml'))
    args = parser.parse_args()

    cutoff = args.today - datetime.timedelta(days=args.cooldown_days)
    changed = pin_exclude_newer(args.pyproject, cutoff)
    print(
        f'exclude-newer -> {format_utc(cutoff)} ({"updated" if changed else "unchanged"})'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
