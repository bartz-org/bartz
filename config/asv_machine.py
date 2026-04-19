# bartz/config/asv_machine.py
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

"""Ensure ``~/.asv-machine.json`` carries a human-readable machine id.

Run as a script: if the asv machine file already has a machine entry, do
nothing. Otherwise let asv generate the default entry, detect whether an
NVIDIA GPU is present, and rename the entry to ``<prefix>-<original>`` where
``<prefix>`` is a short GPU slug (e.g. ``3090``, ``a4000``) or ``cpu``.
"""

import json
import subprocess
import sys
from pathlib import Path

MACHINE_FILE = Path.home() / '.asv-machine.json'


def load() -> dict:
    """Return the parsed machine file, or an empty dict if absent."""
    if not MACHINE_FILE.exists():
        return {}
    return json.loads(MACHINE_FILE.read_text())


def has_machine(data: dict) -> bool:
    """Tell whether ``data`` already contains a machine entry."""
    return any(k != 'version' for k in data)


def gpu_slug() -> str | None:
    """Return a short slug for the first NVIDIA GPU, or None if none found."""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines:
        return None
    raw = lines[0]
    drop = {'nvidia', 'corporation', 'geforce', 'rtx'}
    tokens = [t for t in raw.lower().split() if t not in drop]
    slug = ''.join(c for c in ''.join(tokens) if c.isalnum())
    if not slug:
        msg = f'could not derive gpu slug from {raw!r}'
        raise RuntimeError(msg)
    return slug


def run_asv_machine() -> None:
    """Invoke ``asv machine --yes`` to populate the default entry."""
    subprocess.run([sys.executable, '-m', 'asv', 'machine', '--yes'], check=True)


def rename_entry(data: dict, prefix: str) -> tuple[dict, str, str]:
    """Return ``(new_data, old_name, new_name)`` with the entry renamed."""
    keys = [k for k in data if k != 'version']
    if len(keys) != 1:
        msg = f'expected exactly one machine entry, got {keys!r}'
        raise RuntimeError(msg)
    old_name = keys[0]
    new_name = f'{prefix}-{old_name}'
    entry = dict(data[old_name])
    entry['machine'] = new_name
    new_data = {k: v for k, v in data.items() if k != old_name}
    new_data[new_name] = entry
    return new_data, old_name, new_name


def main() -> None:
    data = load()
    if has_machine(data):
        print(f'asv machine file already configured at {MACHINE_FILE}')
        return
    run_asv_machine()
    data = load()
    slug = gpu_slug()
    prefix = slug if slug is not None else 'cpu'
    new_data, old_name, new_name = rename_entry(data, prefix)
    MACHINE_FILE.write_text(json.dumps(new_data, indent=4) + '\n')
    print(f'renamed asv machine {old_name!r} -> {new_name!r}')


if __name__ == '__main__':
    main()
