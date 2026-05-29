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
1. All version tags on the default branch with commit dates after CUTOFF_DATE
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

from git import Repo
from git.exc import GitCommandError
from packaging.version import Version

# Configuration
CUTOFF_DATE = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)


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


def benchmarked_version_tags(
    repo_path: Path | str = '.',
) -> list[tuple[datetime.datetime, str]]:
    """Return `[(commit_date, tag_name), ...]` of tags benchmarked by ASV.

    A tag is included iff it starts with `v`, is reachable from the default
    branch, and points at a commit on/after `CUTOFF_DATE`. Sorted oldest first.
    """
    repo = Repo(repo_path)
    main_branch = repo.refs[get_default_branch_name(repo)]
    tags: list[tuple[datetime.datetime, str]] = []
    for tag in repo.tags:
        commit = tag.commit
        if not repo.is_ancestor(commit, main_branch.commit):
            continue
        commit_date = datetime.datetime.fromtimestamp(
            commit.committed_date, tz=datetime.timezone.utc
        )
        if commit_date >= CUTOFF_DATE and tag.name.startswith('v'):
            tags.append((commit_date, tag.name))
    tags.sort()
    return tags


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
