# bartz/config/refs-for-asv.py
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
Print a list of git refs for ASV benchmarking.

This script outputs:
1. All version tags on the default branch with commit dates after CUTOFF_DATE
2. The HEAD of the default branch

The output format is one ref per line, suitable for piping to `asv run HASHFILE:-`
"""

import datetime

from git import Repo

# Configuration
CUTOFF_DATE = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)


def main() -> None:
    repo = Repo('.')

    # Get the default branch name from git
    # This queries the symbolic-ref for the remote's HEAD
    default_branch_name = repo.git.symbolic_ref(
        'refs/remotes/origin/HEAD', short=True
    ).split('/')[-1]

    # Get the default branch
    main_branch = repo.refs[default_branch_name]

    # Collect tags that are reachable from main and after cutoff date
    tags_to_include = []

    for tag in repo.tags:
        # Get the commit the tag points to
        commit = tag.commit

        # Check if this tag is reachable from main
        if not repo.is_ancestor(commit, main_branch.commit):
            continue

        # Check if commit date is after cutoff
        commit_date = datetime.datetime.fromtimestamp(
            commit.committed_date, tz=datetime.timezone.utc
        )

        if commit_date >= CUTOFF_DATE and tag.name.startswith('v'):
            tags_to_include.append((commit_date, tag.name))

    # Sort tags by commit date
    tags_to_include.sort()

    # Print tags
    for _, tag_name in tags_to_include:
        print(tag_name)

    # Print default branch ref
    print(default_branch_name)


if __name__ == '__main__':
    main()
