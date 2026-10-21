# bartz/config/_common.py
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

"""Helpers shared by the config scripts."""

import argparse
import datetime

from packaging.specifiers import SpecifierSet
from packaging.version import Version


def parse_datetime(s: str) -> datetime.datetime:
    """Parse an RFC 3339 timestamp, requiring an explicit timezone offset."""
    dt = datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        msg = f'timestamp {s!r} must include a timezone offset'
        raise argparse.ArgumentTypeError(msg)
    return dt


def format_utc(dt: datetime.datetime) -> str:
    """Format an instant as an RFC 3339 UTC timestamp with a trailing `Z`."""
    return dt.astimezone(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')


def lower_bound(spec: SpecifierSet) -> Version:
    """Return the version floor set by the single `>=` clause in `spec`."""
    floors = [Version(s.version) for s in spec if s.operator == '>=']
    others = [s for s in spec if s.operator not in ('>=', '<', '<=', '!=')]
    if len(floors) != 1 or others:
        msg = f'expected exactly one `>=` clause plus `<`, `<=`, `!=`, got {spec!r}'
        raise ValueError(msg)
    return floors[0]
