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


def select_tags(
    tags: list[tuple[datetime.datetime, str]], today: datetime.datetime
) -> list[tuple[datetime.datetime, str]]:
    """Select the tags to benchmark.

    The selection comprises:
    - the oldest tag, i.e., the older of the first tag in the last `MIN_AGE`
      and the `MIN_RELEASES_BACK`-th to last tag
    - `NUM_SPREAD` tags evenly spaced in time between the oldest and the
      `NUM_LAST` last tags
    - the `NUM_LAST` last tags
    """
    tags = sorted(tags)
    if len(tags) <= 1 + NUM_SPREAD + NUM_LAST:
        return tags

    in_window = [i for i, (date, _) in enumerate(tags) if date >= today - MIN_AGE]
    first = min((*in_window[:1], max(0, len(tags) - MIN_RELEASES_BACK)))
    oldest = tags[first]
    last = tags[-NUM_LAST:]
    candidates = tags[first + 1 : -NUM_LAST]

    end = last[0][0]
    step = (end - oldest[0]) / (NUM_SPREAD + 1)
    spread = []
    for k in range(1, NUM_SPREAD + 1):
        target = oldest[0] + k * step
        unused = [tag for tag in candidates if tag not in spread]
        spread.append(min(unused, key=lambda tag: abs(tag[0] - target)))

    return sorted((oldest, *spread, *last))


def benchmarked_version_tags(
    repo_path: Path | str = '.',
) -> list[tuple[datetime.datetime, str]]:
    """Return `[(commit_date, tag_name), ...]` of tags benchmarked by ASV.

    Considers tags that start with `v`, are reachable from the default
    branch, and are at least `MIN_VERSION`, selected by `select_tags`.
    Sorted oldest first.
    """
    repo = Repo(repo_path)
    head_commit = default_branch_commit(repo)
    tags: list[tuple[datetime.datetime, str]] = []
    for tag in repo.tags:
        commit = tag.commit
        if (
            not tag.name.startswith('v')
            or Version(tag.name) < MIN_VERSION
            or not repo.is_ancestor(commit, head_commit)
        ):
            continue
        commit_date = datetime.datetime.fromtimestamp(
            commit.committed_date, tz=datetime.timezone.utc
        )
        tags.append((commit_date, tag.name))
    return select_tags(tags, datetime.datetime.now(datetime.timezone.utc))


def oldest_benchmarked_version(repo_path: Path | str = '.') -> Version:
    """Return the oldest bartz version benchmarked by ASV."""
    return Version(benchmarked_version_tags(repo_path)[0][1])


def main() -> None:
    repo = Repo('.')
    default_branch_name = get_default_branch_name(repo)
    refs = [tag_name for _, tag_name in benchmarked_version_tags('.')]
    refs.append(default_branch_name)
    print('--no-walk', *refs)


if __name__ == '__main__':
    main()
