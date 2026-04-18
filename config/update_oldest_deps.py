# bartz/config/update_oldest_deps.py
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

"""Refresh oldest-supported dependency pins per the OLD_DATE policy.

Computes a new OLD_DATE as max(today - delay_days, min_old_date), queries PyPI
for each dependency declared in pyproject.toml, picks the latest final release
whose earliest file upload is strictly before OLD_DATE, and rewrites the
lower bounds in pyproject.toml and the OLD_DATE line in the Makefile.
Bounds only move forward: per-dep, the chosen version is max(pypi_candidate,
current lower bound).
"""

import argparse
import datetime
import json
import re
import sys
import urllib.request
from pathlib import Path

import tomli
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

PYPI_URL = 'https://pypi.org/pypi/{name}/json'


def parse_date(s: str) -> datetime.date:
    return (
        datetime.datetime.strptime(s, '%Y-%m-%d')
        .replace(tzinfo=datetime.timezone.utc)
        .date()
    )


def compute_old_date(min_old_date: datetime.date, delay_days: int) -> datetime.date:
    today = datetime.datetime.now(tz=datetime.timezone.utc).date()
    return max(today - datetime.timedelta(days=delay_days), min_old_date)


def collect_requirements(pyproject: dict) -> list[str]:
    """Return every requirement string declared in pyproject.toml."""
    reqs: list[str] = []
    reqs.extend(pyproject.get('project', {}).get('dependencies', []))
    for extra_reqs in (
        pyproject.get('project', {}).get('optional-dependencies', {}).values()
    ):
        reqs.extend(extra_reqs)
    for group_reqs in pyproject.get('dependency-groups', {}).values():
        reqs.extend(group_reqs)
    return reqs


def current_lower_bound(spec: SpecifierSet) -> Version | None:
    for s in spec:
        if s.operator in ('>=', '==', '~='):
            try:
                return Version(s.version)
            except InvalidVersion:
                return None
    return None


def fetch_pypi(name: str) -> dict:
    url = PYPI_URL.format(name=name)
    if not url.startswith('https://'):
        msg = f'refusing non-https URL: {url}'
        raise ValueError(msg)
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
        return json.load(resp)


def earliest_upload(files: list[dict]) -> datetime.datetime | None:
    times = []
    for f in files:
        if f.get('yanked'):
            continue
        t = f.get('upload_time_iso_8601') or f.get('upload_time')
        if not t:
            continue
        # strip trailing 'Z' and optional fractional seconds handled by fromisoformat in 3.11+
        t = t.replace('Z', '+00:00')
        try:
            dt = datetime.datetime.fromisoformat(t)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        times.append(dt)
    return min(times) if times else None


def release_supports_python(files: list[dict], oldest_python: Version) -> bool:
    """Check Python compatibility of a release.

    Return True iff any non-yanked file's `requires_python` accepts
    `oldest_python`. Files with no `requires_python` metadata are treated as
    universally compatible (common for old releases predating PEP 345).
    """
    any_non_yanked = False
    for f in files:
        if f.get('yanked'):
            continue
        any_non_yanked = True
        spec_str = f.get('requires_python')
        if not spec_str:
            return True
        try:
            spec = SpecifierSet(spec_str)
        except InvalidSpecifier:
            return True
        if spec.contains(str(oldest_python), prereleases=True):
            return True
    return not any_non_yanked


def latest_before(
    name: str, cutoff: datetime.date, oldest_python: Version
) -> Version | None:
    """Return the latest final release of `name` matching the constraints.

    The release must have earliest upload strictly before `cutoff` and must
    support `oldest_python`.
    """
    data = fetch_pypi(name)
    cutoff_dt = datetime.datetime.combine(
        cutoff, datetime.time.min, tzinfo=datetime.timezone.utc
    )
    candidates: list[Version] = []
    for vstr, files in data.get('releases', {}).items():
        try:
            v = Version(vstr)
        except InvalidVersion:
            continue
        if v.is_prerelease or v.is_devrelease:
            continue
        if not files:
            continue
        upload = earliest_upload(files)
        if upload is None:
            continue
        if upload >= cutoff_dt:
            continue
        if not release_supports_python(files, oldest_python):
            continue
        candidates.append(v)
    return max(candidates) if candidates else None


def format_spec(req: Requirement, version: Version) -> str:
    extras = f'[{",".join(sorted(req.extras))}]' if req.extras else ''
    return f'{req.name}{extras}>={version}'


def replace_requirement_line(
    text: str, old_req_str: str, new_req_str: str
) -> tuple[str, bool]:
    """Replace a quoted requirement literal in pyproject.toml text.

    Matches either single- or double-quoted form of the exact old string.
    """
    if old_req_str == new_req_str:
        return text, False
    replaced = False
    for quote in ('"', "'"):
        old_lit = f'{quote}{old_req_str}{quote}'
        new_lit = f'{quote}{new_req_str}{quote}'
        if old_lit in text:
            text = text.replace(old_lit, new_lit, 1)
            replaced = True
            break
    return text, replaced


def update_pyproject(pyproject_path: Path, updates: list[tuple[str, str]]) -> int:
    text = pyproject_path.read_text()
    changed = 0
    for old, new in updates:
        text, did = replace_requirement_line(text, old, new)
        if did:
            changed += 1
        elif old != new:
            print(
                f'warning: could not locate {old!r} in pyproject.toml', file=sys.stderr
            )
    pyproject_path.write_text(text)
    return changed


def update_makefile_old_date(makefile_path: Path, new_date: datetime.date) -> bool:
    text = makefile_path.read_text()
    pattern = re.compile(r'^OLD_DATE\s*=\s*\S+\s*$', re.MULTILINE)
    new_line = f'OLD_DATE = {new_date.isoformat()}'
    new_text, n = pattern.subn(new_line, text, count=1)
    if n == 0:
        msg = 'OLD_DATE assignment not found in Makefile'
        raise RuntimeError(msg)
    if new_text == text:
        return False
    makefile_path.write_text(new_text)
    return True


def main() -> int:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--min-old-date',
        required=True,
        type=parse_date,
        help='YYYY-MM-DD; floor for the new OLD_DATE.',
    )
    parser.add_argument(
        '--delay-days', required=True, type=int, help='Target lag behind today in days.'
    )
    parser.add_argument('--pyproject', type=Path, default=Path('pyproject.toml'))
    parser.add_argument('--makefile', type=Path, default=Path('Makefile'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    new_old_date = compute_old_date(args.min_old_date, args.delay_days)
    print(f'OLD_DATE: {args.min_old_date.isoformat()} -> {new_old_date.isoformat()}')

    with args.pyproject.open('rb') as f:
        pyproject = tomli.load(f)

    requires_python = pyproject.get('project', {}).get('requires-python')
    if not requires_python:
        msg = 'project.requires-python missing from pyproject.toml'
        raise RuntimeError(msg)
    rp_spec = SpecifierSet(requires_python)
    oldest_python = next(
        (Version(s.version) for s in rp_spec if s.operator in ('>=', '==', '~=')), None
    )
    if oldest_python is None:
        msg = f'cannot derive oldest Python from requires-python={requires_python!r}'
        raise RuntimeError(msg)
    print(f'oldest supported Python: {oldest_python}')

    raw_reqs = collect_requirements(pyproject)
    # Deduplicate while preserving order (a dep may appear in multiple groups).
    seen: set[str] = set()
    reqs: list[str] = []
    for r in raw_reqs:
        if r not in seen:
            seen.add(r)
            reqs.append(r)

    updates: list[tuple[str, str]] = []
    rows: list[tuple[str, str, str]] = []
    for req_str in reqs:
        req = Requirement(req_str)
        if req.url or req.marker:
            rows.append((req.name, 'skipped', 'has url/marker'))
            continue
        floor = current_lower_bound(req.specifier)
        try:
            pypi_cand = latest_before(req.name, new_old_date, oldest_python)
        except Exception as e:  # noqa: BLE001
            rows.append((req.name, 'error', str(e)))
            continue
        if pypi_cand is None and floor is None:
            rows.append(
                (req.name, 'skipped', 'no eligible version and no existing floor')
            )
            continue
        chosen = max(v for v in (pypi_cand, floor) if v is not None)
        new_req_str = format_spec(req, chosen)
        if new_req_str != req_str:
            updates.append((req_str, new_req_str))
        rows.append((req.name, str(floor) if floor else '-', str(chosen)))

    col1 = max((len(r[0]) for r in rows), default=4)
    col2 = max((len(r[1]) for r in rows), default=4)
    print()
    print(f'{"dep".ljust(col1)}  {"from".ljust(col2)}  to')
    print(f'{"-" * col1}  {"-" * col2}  --')
    for name, old, new in rows:
        print(f'{name.ljust(col1)}  {old.ljust(col2)}  {new}')
    print()

    if args.dry_run:
        print(
            f'dry-run: {len(updates)} pyproject edit(s), OLD_DATE target {new_old_date.isoformat()}'
        )
        return 0

    n_py = update_pyproject(args.pyproject, updates)
    mk_changed = update_makefile_old_date(args.makefile, new_old_date)
    print(f'pyproject.toml: {n_py} line(s) updated')
    print(f'Makefile OLD_DATE: {"updated" if mk_changed else "unchanged"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
