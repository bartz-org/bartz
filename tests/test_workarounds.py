# bartz/tests/test_workarounds.py
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

"""Test the XLA_FLAGS workaround for the jax gpu scatter regression."""

import os
import subprocess
import sys
from textwrap import dedent

import jax
import pytest

from bartz import _workarounds
from bartz._jaxext import get_default_device


def test_parse_version() -> None:
    """Check version strings are reduced to numeric triplets."""
    assert _workarounds.parse_version('0.11.0') == (0, 11, 0)
    assert _workarounds.parse_version('0.12.0.dev20260718') == (0, 12, 0)
    with pytest.raises(AssertionError):
        _workarounds.parse_version('garbage')
    with pytest.raises(AssertionError):
        _workarounds.parse_version('garbage0.11.0')


def test_add_backend_extra_option() -> None:
    """Check the option is merged into XLA_FLAGS without clobbering it."""
    option = _workarounds.FTZ_ATOMICS_OPTION
    add = _workarounds.add_backend_extra_option
    prefix = '--xla_backend_extra_options='
    assert add('', option) == f'{prefix}{option}'
    assert (
        add('--xla_dump_to=/tmp/x', option) == f'--xla_dump_to=/tmp/x {prefix}{option}'
    )
    assert add(f'{prefix}-foo --xla_gpu_autotune_level=0', option) == (
        f'{prefix}-foo,{option} --xla_gpu_autotune_level=0'
    )
    assert add(prefix, option) == f'{prefix}{option}'


def test_cuda_detection() -> None:
    """Check the cuda detection helpers against the test platform."""
    if get_default_device().platform == 'gpu':  # pragma: no cover, needs gpu
        assert _workarounds.cuda_plugin_installed()
        assert _workarounds.cuda_devices_available()
    elif _workarounds.cuda_devices_available():  # pragma: no cover, needs gpu
        # tests forced to cpu on a gpu machine: the plugin must still be there
        assert _workarounds.cuda_plugin_installed()


def test_cuda_devices_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check gpu detection with a stubbed `jax.devices`."""
    monkeypatch.setattr(jax, 'devices', lambda _backend: ['gpu0'])
    assert _workarounds.cuda_devices_available()

    def no_cuda(_backend: str) -> list:
        msg = 'Unknown backend'
        raise RuntimeError(msg)

    monkeypatch.setattr(jax, 'devices', no_cuda)
    assert not _workarounds.cuda_devices_available()


def test_option_in_parsed_flags() -> None:
    """Check a probe option is not found in the flags XLA parsed at startup."""
    assert not _workarounds.option_in_parsed_flags('-definitely-not-a-real-option')


def test_raises_when_too_late(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the error raised when the workaround can't be applied on gpu."""
    monkeypatch.setattr(jax, '__version__', '0.11.0')
    monkeypatch.setattr(_workarounds, 'cuda_plugin_installed', lambda: True)
    monkeypatch.setattr(_workarounds, 'cuda_devices_available', lambda: True)
    monkeypatch.setattr(_workarounds, 'option_in_parsed_flags', lambda _option: False)
    monkeypatch.setenv('XLA_FLAGS', '')
    with pytest.raises(RuntimeError, match='38806'):
        _workarounds.fix_gpu_scatter_performance()


def test_user_already_set_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the workaround defers to user-provided XLA flags."""
    monkeypatch.setattr(jax, '__version__', '0.11.0')
    monkeypatch.setattr(_workarounds, 'cuda_plugin_installed', lambda: True)
    for flags in (
        f'--xla_backend_extra_options={_workarounds.FTZ_ATOMICS_OPTION}',
        '--XLA_GPU_FTZ=true',
    ):
        monkeypatch.setenv('XLA_FLAGS', flags)
        _workarounds.fix_gpu_scatter_performance()
        assert os.environ['XLA_FLAGS'] == flags


def test_skip_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check BARTZ_SKIP_XLA_WORKAROUND disables the workarounds."""
    monkeypatch.setenv('BARTZ_SKIP_XLA_WORKAROUND', '1')
    monkeypatch.setattr(
        _workarounds,
        'fix_gpu_scatter_performance',
        lambda: pytest.fail('workaround not skipped'),
    )
    _workarounds.apply_workarounds()


def run_in_subprocess(code: str) -> subprocess.CompletedProcess:
    """Run python code in a fresh interpreter with pristine XLA state."""
    env = dict(os.environ)
    env.pop('XLA_FLAGS', None)
    env.pop('BARTZ_SKIP_XLA_WORKAROUND', None)
    return subprocess.run(  # noqa: S603, the code is a trusted literal
        [sys.executable, '-c', dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )


def test_applied_in_time() -> None:
    """Check applying the workaround before any jax backend use locks it in."""
    result = run_in_subprocess("""
        import os

        import jax

        from bartz import _workarounds as w

        jax.__version__ = '0.11.0'
        w.cuda_plugin_installed = lambda: True
        w.fix_gpu_scatter_performance()
        assert w.FTZ_ATOMICS_OPTION in os.environ['XLA_FLAGS']
        assert w.option_in_parsed_flags(w.FTZ_ATOMICS_OPTION)
        """)
    assert result.returncode == 0, result.stderr


def test_too_late() -> None:
    """Check what happens if a jax backend is used before applying.

    The workaround is ineffective: silently if there is no gpu, with an error
    otherwise.
    """
    result = run_in_subprocess("""
        import os

        os.environ['BARTZ_SKIP_XLA_WORKAROUND'] = '1'

        import jax

        jax.devices()  # initialize backends, locking XLA_FLAGS

        from bartz import _workarounds as w

        jax.__version__ = '0.11.0'
        w.cuda_plugin_installed = lambda: True

        w.cuda_devices_available = lambda: False
        w.fix_gpu_scatter_performance()  # no error without gpus
        assert w.FTZ_ATOMICS_OPTION in os.environ['XLA_FLAGS']
        assert not w.option_in_parsed_flags(w.FTZ_ATOMICS_OPTION)

        os.environ['XLA_FLAGS'] = ''
        w.cuda_devices_available = lambda: True
        w.fix_gpu_scatter_performance()  # error with gpus
        """)
    assert result.returncode != 0
    assert 'RuntimeError' in result.stderr
    assert '38806' in result.stderr
