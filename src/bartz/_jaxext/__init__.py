# bartz/src/bartz/_jaxext/__init__.py
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

"""Additions to jax and equinox."""

# ruff: noqa: F401

from bartz._jaxext._autobatch import autobatch
from bartz._jaxext._equinox import Module, field, project
from bartz._jaxext._jaxext import (
    equal_shards,
    float32_matmuls,
    get_default_device,
    get_default_devices,
    get_device_count,
    is_key,
    jaxtyping_disabled,
    jit_active,
    minimal_unsigned_dtype,
    sliced_map,
    split,
    truncated_normal_onesided,
    unique,
    vmap_nodoc,
)
from bartz._jaxext._jit import jit
