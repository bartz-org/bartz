# bartz/tests/test_config.py
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

"""Test the development scripts in `config/`."""

import datetime
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import pytest
from packaging.version import Version

from config.refs_for_asv import (
    MIN_AGE,
    MIN_RELEASES_BACK,
    MIN_VERSION,
    NUM_LAST,
    NUM_SPREAD,
    Tag,
    benchmarked_version_tags,
    oldest_benchmarked_version,
    select_tags,
)

NUM_SELECTED = 1 + NUM_SPREAD + NUM_LAST
T0 = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)


def make_tags(days: Sequence[int]) -> tuple[Tag, ...]:
    """Make tags released at the given days after `T0`."""
    return tuple(
        Tag(date=T0 + datetime.timedelta(days=d), name=f'v0.{i}.0')
        for i, d in enumerate(days)
    )


def check_selection(tags: Sequence[Tag], today: datetime.datetime) -> tuple[Tag, ...]:
    """Check the invariants of `select_tags` and return its selection."""
    selected = select_tags(tags, today)
    assert selected == tuple(sorted(selected))
    assert len(set(selected)) == len(selected)
    assert set(selected) <= set(tags)
    assert len(selected) == min(len(tags), NUM_SELECTED)
    assert selected[-NUM_LAST:] == tuple(sorted(tags))[-NUM_LAST:]
    return selected


def test_tag() -> None:
    """Check tags are ordered by date and parse their version."""
    early, late = make_tags((10, 0))
    assert late < early
    assert early.version == Version('0.0.0')


@pytest.mark.parametrize('num', range(NUM_SELECTED + 1))
def test_few_tags(num: int) -> None:
    """Check all tags are selected if there are not more than needed."""
    tags = make_tags(range(0, 30 * num, 30))
    today = T0 + datetime.timedelta(days=30 * num)
    assert check_selection(tags[::-1], today) == tags


@pytest.mark.parametrize('period', [1, 7, 30])
def test_frequent_releases(period: int) -> None:
    """Check the oldest tag is the first in `MIN_AGE` if releasing frequently."""
    tags = make_tags(range(0, 1000, period))
    today = tags[-1].date + datetime.timedelta(days=1)
    selected = check_selection(tags, today)
    first_in_window = min(tag for tag in tags if tag.date >= today - MIN_AGE)
    assert selected[0] == first_in_window


@pytest.mark.parametrize('period', [60, 91, 365])
def test_infrequent_releases(period: int) -> None:
    """Check the oldest tag is `MIN_RELEASES_BACK` back if releasing slowly."""
    tags = make_tags(range(0, 5000, period))
    today = tags[-1].date + datetime.timedelta(days=1)
    selected = check_selection(tags, today)
    assert selected[0] == tags[-MIN_RELEASES_BACK]


def test_no_recent_releases() -> None:
    """Check the selection when there are no tags in the last `MIN_AGE`."""
    tags = make_tags(range(0, 600, 30))
    today = tags[-1].date + 2 * MIN_AGE
    selected = check_selection(tags, today)
    assert selected[0] == tags[-MIN_RELEASES_BACK]


def test_spread_in_time() -> None:
    """Check the middle tags are the closest to evenly spaced dates."""
    tags = make_tags(range(365))
    today = tags[-1].date
    selected = check_selection(tags, today)
    gaps = {(b.date - a.date).days for a, b in pairwise(selected[: -NUM_LAST + 1])}
    assert max(gaps) - min(gaps) <= 1


def test_clustered_releases() -> None:
    """Check bursts of releases still yield distinct tags."""
    tags = make_tags((0, 1, 2, 3, 4, 5, 300, 301, 302, 303, 304, 305))
    today = tags[-1].date
    selected = check_selection(tags, today)
    assert selected[0] == tags[0]


def test_benchmarked_version_tags() -> None:
    """Check the selection on the actual repository."""
    root = Path(__file__).parent.parent
    tags = benchmarked_version_tags(root)
    assert len(tags) == NUM_SELECTED
    assert all(tag.version >= MIN_VERSION for tag in tags)
    assert oldest_benchmarked_version(root) == tags[0].version
