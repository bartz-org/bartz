# bartz/tests/test_init.py
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

"""Test what ``import bartz`` loads."""

import subprocess
import sys
from textwrap import dedent

import pytest


# WORKAROUND(python<3.15): drop the skip when we require Python 3.15
@pytest.mark.skipif(sys.version_info < (3, 15), reason='needs PEP 810 lazy imports')
def test_lazy_submodules() -> None:
    """Check the compatibility wrappers are loaded only on first access."""
    code = """
        import sys

        import bartz

        lazy = ('bartz.BART', 'bartz.bcf', 'bartz.stochtree')
        loaded = [m for m in lazy if m in sys.modules]
        assert not loaded, loaded

        bartz.BART.mc_gbart
        bartz.stochtree.BARTModel
        from bartz import bcf
        bcf.bcf

        assert all(m in sys.modules for m in lazy)
        """
    result = subprocess.run(  # noqa: S603, the code is a trusted literal
        [sys.executable, '-c', dedent(code)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
