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

from contextlib import nullcontext
from re import fullmatch

import jax
import numpy as np
import pytest
from jax import config, random

from bartz.jaxext import get_default_device, split

# enable debug checks; some of these slow down unit tests
config.update('jax_debug_key_reuse', True)
config.update('jax_debug_nans', True)
config.update('jax_debug_infs', True)
config.update('jax_legacy_prng_key', 'error')
if jax.__version_info__ >= (0, 9, 0):
    config.update('jax_check_static_indices', True)
    config.update('jax_explicit_x64_dtypes', 'error')

# enable virtual cpu devices to do multi-device testing on cpu
config.update('jax_num_cpu_devices', 10)  # 2 * 5

# enable compilation cache
if jax.__version_info__ >= (0, 9, 0):
    # enable only on latest jax because `make tests-old` fails if there is a
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


def pytest_sessionstart(session: pytest.Session) -> None:
    """Configure and print the jax device."""
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
        device_kind = get_default_device().device_kind
        print(f'jax default device: {device_kind}')
