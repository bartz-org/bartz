#!/bin/bash
# bartz/config/asv-install.sh
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

# Helper script for ASV to install the bartz wheel with conditional CUDA support.
# Usage: asv-install.sh <bartz_wheel> <venv_dir>

set -e

WHEEL_FILE="$1"
ENV_DIR="$2"

if [ -z "$WHEEL_FILE" ]; then
    echo "Error: No wheel file specified" >&2
    exit 1
fi

if [ -z "$ENV_DIR" ]; then
    echo "Error: No environment directory specified" >&2
    exit 1
fi

# Detect CUDA version (same logic as Makefile)
CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]*' | cut -d' ' -f3 || echo "")

# Determine the extra to install
if [ -z "$CUDA_VERSION" ]; then
    EXTRA=""
    echo "No CUDA detected, installing without CUDA extra"
elif [ "$CUDA_VERSION" = "12" ] || [ "$CUDA_VERSION" = "13" ]; then
    EXTRA="[cuda${CUDA_VERSION}]"
    echo "Detected CUDA $CUDA_VERSION, installing with $EXTRA extra"
else
    echo "Error: Unsupported CUDA version $CUDA_VERSION (only 12 and 13 are supported)" >&2
    exit 1
fi

# Install the wheel
# python -m pip install "${WHEEL_FILE}${EXTRA}"
uv pip install --python="$ENV_DIR" "${WHEEL_FILE}${EXTRA}"
