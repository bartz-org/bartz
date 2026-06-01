# bartz/tests/conftest.py
#
# Copyright (c) 2024-2026, The Bartz Contributors
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

"""Pytest configuration."""

import sys
from contextlib import nullcontext
from pathlib import Path
from re import fullmatch

import jax
import numpy as np
import pytest

# WORKAROUND(python<3.11): use stdlib tomllib instead of tomli
import tomli
from jax import config, random
from jaxtyping import install_import_hook

# Turn on runtime type checking of the whole library for the test suite. This is
# done here, before the first `bartz` import, rather than via the jaxtyping
# pytest plugin's `--jaxtyping-packages` option, because that option is not
# honored when set through `addopts` in `pyproject.toml` (the plugin reads it
# too early to see the ini config). The hook stays installed for the session.
install_import_hook('bartz', 'beartype.beartype')

from bartz._jaxext import get_default_device, get_device_count, split


def get_old_python_version() -> tuple[int, int]:
    """Return the minimum Python version required by pyproject.toml as (major, minor)."""
    pyproject = Path(__file__).parent.parent / 'pyproject.toml'
    with pyproject.open('rb') as f:
        data = tomli.load(f)
    spec = data['project']['requires-python'].strip()
    match = fullmatch(r'>=\s*(\d+)\.(\d+)', spec)
    return (int(match.group(1)), int(match.group(2)))


INVASIVE_DEBUG_CHECKS = False
if INVASIVE_DEBUG_CHECKS:
    # they make the tests 10% slower, disable buffer donation, and yield some
    # false positive, so we don't keep them on by default
    config.update('jax_debug_key_reuse', True)
    config.update('jax_debug_nans', True)
    config.update('jax_debug_infs', True)
config.update('jax_legacy_prng_key', 'error')
# WORKAROUND(jax<0.8.0): jax_explicit_x64_dtypes config option added in 0.8.0
if jax.__version_info__ >= (0, 8, 0):
    config.update('jax_explicit_x64_dtypes', 'error')
# WORKAROUND(jax<0.8.2): jax_check_static_indices config option added in 0.8.2
if jax.__version_info__ >= (0, 8, 2):
    config.update('jax_check_static_indices', True)

# enable logging arrays destroyed by the gc
config.update('jax_array_garbage_collection_guard', 'log')

# enable compilation cache
if sys.version_info[:2] > get_old_python_version():
    # enable only on latest config because `make tests-old` fails if there is a
    # cache created with a newer jax version
    config.update('jax_compilation_cache_dir', 'config/jax_cache')
    config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    config.update('jax_persistent_cache_min_compile_time_secs', 0.1)


@pytest.fixture
def keys(request: pytest.FixtureRequest) -> split:
    """
    Return a deterministic per-test-case list of jax random keys.

    To use a key, do `keys.pop()`. If consumed this way, this list of keys can
    be safely used by multiple fixtures involved in the test case.
    """
    nodeid = request.node.nodeid
    # exclude xdist_group suffixes because they are active only under xdist
    match = fullmatch(r'(.+?\.py::.+?(\[.+?\])?)(@.+)?', nodeid)
    nodeid = match.group(1)
    seed = np.array([nodeid], np.bytes_).view(np.uint8)
    rng = np.random.default_rng(seed)
    seed = np.array(rng.bytes(4)).view(np.uint32)
    key = random.key(seed)
    return split(key, 128)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options."""
    parser.addoption(
        '--platform',
        choices=['cpu', 'gpu', 'auto'],
        default='auto',
        help='JAX platform to use: cpu, gpu, or auto (default: auto)',
    )
    parser.addoption(
        '--num-cpu-devices',
        type=int,
        default=10,
        help='Number of virtual jax cpu devices to create (default: 10)',
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Customizable jax setup."""
    setup_jax_num_cpu_devices(session)
    setup_jax_platform(session)


def setup_jax_num_cpu_devices(session: pytest.Session) -> None:
    """Configure the number of virtual jax cpu devices."""
    num_cpu_devices = session.config.getoption('--num-cpu-devices')
    config.update('jax_num_cpu_devices', num_cpu_devices)


def setup_jax_platform(session: pytest.Session) -> None:
    """Configure, check, and log the default jax platform."""
    # Get the platform option
    platform = session.config.getoption('--platform')

    # Set the default JAX device if not auto
    if platform != 'auto':
        current_platform = get_default_device().platform
        if current_platform != platform:
            config.update('jax_default_device', jax.devices(platform)[0])
        assert get_default_device().platform == platform

    # Get the capture manager plugin
    capman = session.config.pluginmanager.get_plugin('capturemanager')

    # Suspend capturing temporarily
    if capman:
        ctx = capman.global_and_fixture_disabled()
    else:
        ctx = nullcontext()

    with ctx:
        dd = get_default_device()
        num_devices = get_device_count()
        print(f'jax default device: {dd.device_kind}, num devices: {num_devices}')
