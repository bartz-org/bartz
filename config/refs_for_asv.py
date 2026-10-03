# bartz/config/refs_for_asv.py
#
# Copyright (c) 2025-2026, The Bartz Contributors
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
Print a git rev-list range spec for ASV benchmarking.

The output covers:
1. A fixed number of version tags on the default branch, see `select_tags`
2. The HEAD of the default branch

The output is one space-separated line, prefixed with `--no-walk`, suitable
for passing directly as the positional `range` argument to `asv run`. asv
shlex-splits it and feeds it to `git rev-list --first-parent` — which errors
out on unknown refs, so an unresolvable ref aborts the whole run instead of
being silently skipped (as `asv run HASHFILE:-` does).

The helpers in this module are also used by `check_workarounds.py` to derive
the floor for `bartz` itself (= oldest benchmarked version).
"""

import datetime
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from git import Commit, Repo
from git.exc import BadName, GitCommandError
from packaging.version import Version

# Configuration
MIN_VERSION = Version('0.4.1')  # oldest version supported by the benchmarks
MIN_AGE = datetime.timedelta(days=365)
MIN_RELEASES_BACK = 11
NUM_LAST = 3
NUM_SPREAD = 3


@dataclass(frozen=True, order=True)
class Tag:
    """A version tag, ordered by date."""

    date: datetime.datetime
    """Commit date of the tagged commit."""

    name: str
    """Tag name, e.g., `v0.1.0`."""

    @property
    def version(self) -> Version:
        """The version parsed from the tag name."""
        return Version(self.name)


def get_default_branch_name(repo: Repo) -> str:
    try:
        return repo.git.symbolic_ref('refs/remotes/origin/HEAD', short=True).split('/')[
            -1
        ]
    except GitCommandError:
        pass
    # Ask the remote directly. Works even when refs/remotes/origin/HEAD
    # was never set locally (e.g. origin added after the initial clone).
    try:
        output = repo.git.ls_remote('--symref', 'origin', 'HEAD')
        assert isinstance(output, str)  # type narrowing
    except GitCommandError:
        output = ''
    for line in output.splitlines():
        if line.startswith('ref:'):
            return line.split()[1].split('/')[-1]
    # Last resort: pick the first conventional name that exists as a local
    # branch. Hits the asv-on-VM case where origin is unreachable but the
    # default branch was pushed in by the host-setup step.
    local = {h.name for h in repo.heads}
    for candidate in ('main', 'master'):
        if candidate in local:
            return candidate
    msg = 'could not determine default branch of origin'
    raise RuntimeError(msg)


def _resolve_commit(repo: Repo, ref: str) -> Commit | None:
    try:
        return repo.commit(ref)
    except (GitCommandError, BadName):
        return None


def default_branch_commit(repo: Repo) -> Commit:
    """Resolve the default branch to a commit (local head or remote-tracking).

    In CI the default branch is often present only as `origin/<name>` (the
    checkout is a detached HEAD), so a bare-name lookup in `repo.refs` misses it.
    """
    name = get_default_branch_name(repo)
    commit = _resolve_commit(repo, name) or _resolve_commit(repo, f'origin/{name}')
    if commit is None:
        msg = f'could not resolve default branch {name!r} to a commit'
        raise RuntimeError(msg)
    return commit


def select_tags(tags: Sequence[Tag], today: datetime.datetime) -> tuple[Tag, ...]:
    """Select the tags to benchmark.

    The selection comprises:
    - the oldest tag, i.e., the older of the first tag in the last `MIN_AGE`
      and the `MIN_RELEASES_BACK`-th to last tag
    - `NUM_SPREAD` tags evenly spaced in time between the oldest and the
      `NUM_LAST` last tags
    - the `NUM_LAST` last tags
    """
    tags = tuple(sorted(tags))
    if len(tags) <= 1 + NUM_SPREAD + NUM_LAST:
        return tags

    in_window = tuple(i for i, tag in enumerate(tags) if tag.date >= today - MIN_AGE)
    first = min((*in_window[:1], max(0, len(tags) - MIN_RELEASES_BACK)))
    oldest = tags[first]
    last = tags[-NUM_LAST:]
    candidates = tags[first + 1 : -NUM_LAST]

    step = (last[0].date - oldest.date) / (NUM_SPREAD + 1)
    spread: tuple[Tag, ...] = ()
    for k in range(1, NUM_SPREAD + 1):
        target = oldest.date + k * step
        unused = tuple(tag for tag in candidates if tag not in spread)
        spread += (min(unused, key=lambda tag: abs(tag.date - target)),)

    return tuple(sorted((oldest, *spread, *last)))


def benchmarked_version_tags(repo_path: Path | str = '.') -> tuple[Tag, ...]:
    """Return the tags benchmarked by ASV, sorted oldest first.

    Considers tags that start with `v`, are reachable from the default
    branch, and are at least `MIN_VERSION`, selected by `select_tags`.
    """
    repo = Repo(repo_path)
    head_commit = default_branch_commit(repo)
    tags = (
        Tag(
            date=datetime.datetime.fromtimestamp(
                tag.commit.committed_date, tz=datetime.timezone.utc
            ),
            name=tag.name,
        )
        for tag in repo.tags
        if tag.name.startswith('v')
        and Version(tag.name) >= MIN_VERSION
        and repo.is_ancestor(tag.commit, head_commit)
    )
    return select_tags(tuple(tags), datetime.datetime.now(datetime.timezone.utc))


def oldest_benchmarked_version(repo_path: Path | str = '.') -> Version:
    """Return the oldest bartz version benchmarked by ASV."""
    return benchmarked_version_tags(repo_path)[0].version


def main() -> None:
    repo = Repo('.')
    default_branch_name = get_default_branch_name(repo)
    tags = benchmarked_version_tags('.')
    print('--no-walk', *(tag.name for tag in tags), default_branch_name)


if __name__ == '__main__':
    main()
