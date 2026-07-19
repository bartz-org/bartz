# bartz/src/bartz/_workarounds.py
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

"""Workarounds for upstream bugs, applied when bartz is imported.

Set the environment variable ``BARTZ_SKIP_XLA_WORKAROUND=1`` to disable them.
"""

import os
import re
from importlib.util import find_spec

import jax

FTZ_ATOMICS_OPTION = '-nvptx-allow-ftz-atomics'


def parse_version(version: str) -> tuple[int, int, int]:
    """Extract the leading numeric triplet of a version string."""
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)', version)
    assert match is not None, version
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def add_backend_extra_option(xla_flags: str, option: str) -> str:
    """Return `xla_flags` with `option` added to ``--xla_backend_extra_options``."""
    prefix = '--xla_backend_extra_options='
    tokens = xla_flags.split()
    for i, token in enumerate(tokens):
        if token.startswith(prefix):
            value = token.removeprefix(prefix)
            tokens[i] = prefix + (f'{value},{option}' if value else option)
            break
    else:
        tokens.append(prefix + option)
    return ' '.join(tokens)


def option_in_parsed_flags(option: str) -> bool:
    """Check `option` is in the backend extra options XLA parsed from ``XLA_FLAGS``.

    XLA reads ``XLA_FLAGS`` only once per process, on the first use of a jax
    backend; later changes to the environment variable are silently ignored.
    Constructing a `CompileOptions` triggers that one-time read, so calling
    this function right after modifying ``XLA_FLAGS`` both locks the change in
    (if XLA had not read the variable yet) and reports whether it was read.
    """
    # import locally so that a future jaxlib dropping this internal module
    # can not break `import bartz` on jax versions that don't need the fix
    from jaxlib import xla_client  # noqa: PLC0415

    serialized = xla_client.CompileOptions().SerializeAsString()
    return option.encode() in serialized


def cuda_plugin_installed() -> bool:
    """Check if a jax cuda plugin package is installed."""
    return any(find_spec(f'jax_cuda{v}_plugin') is not None for v in (12, 13))


def cuda_devices_available() -> bool:
    """Check if jax can actually use cuda devices."""
    try:
        devices = jax.devices('cuda')
    except RuntimeError:
        return False
    else:
        return bool(devices)


def fix_gpu_scatter_performance() -> None:
    """Restore native f32 atomics in gpu scatters on affected jax versions."""
    # WORKAROUND(jax<=0.11.0): jax 0.10.2 and 0.11.0 ship an LLVM that lowers
    # the f32 atomic adds in gpu scatters to CAS loops, catastrophically slow
    # under index contention; this hidden LLVM option restores the native
    # atomics (and pre-0.10.2 numerics). See
    # https://github.com/jax-ml/jax/issues/38806. If the next jax release
    # contains the LLVM fix, delete this whole file and its uses.
    if not ((0, 10, 2) <= parse_version(jax.__version__) <= (0, 11, 0)):
        return
    if not cuda_plugin_installed():
        return
    xla_flags = os.environ.get('XLA_FLAGS', '')
    if FTZ_ATOMICS_OPTION in xla_flags or '--xla_gpu_ftz=true' in xla_flags.lower():
        return  # the user already took care of it
    os.environ['XLA_FLAGS'] = add_backend_extra_option(xla_flags, FTZ_ATOMICS_OPTION)
    if not option_in_parsed_flags(FTZ_ATOMICS_OPTION) and cuda_devices_available():
        msg = (
            f'jax {jax.__version__} has a severe performance regression in gpu '
            'scatter operations (https://github.com/jax-ml/jax/issues/38806). '
            f'bartz works around it by adding {FTZ_ATOMICS_OPTION} to the '
            'XLA_FLAGS environment '
            'variable, but XLA has already read XLA_FLAGS (that happens on the '
            'first use of a jax backend), so the workaround is ineffective. '
            'Either import bartz before using jax, or set '
            f"XLA_FLAGS='--xla_backend_extra_options={FTZ_ATOMICS_OPTION}' "
            'before starting Python, or use a jax version other than '
            '0.10.2/0.11.0. Set BARTZ_SKIP_XLA_WORKAROUND=1 to ignore this '
            'error and run with slow gpu scatters.'
        )
        raise RuntimeError(msg)


def apply_workarounds() -> None:
    """Apply all workarounds; invoked on ``import bartz``."""
    if os.environ.get('BARTZ_SKIP_XLA_WORKAROUND', '') not in ('', '0'):
        return
    fix_gpu_scatter_performance()
