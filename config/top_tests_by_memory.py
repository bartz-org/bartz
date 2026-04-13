# bartz/config/top_tests_by_memory.py
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

"""Print the top 20 tests by memory usage from a test resource usage CSV."""

import argparse

import polars as pl

parser = argparse.ArgumentParser(description='Show top 20 tests by peak memory usage.')
parser.add_argument('csv', help='Path to the test resource usage CSV file.')
args = parser.parse_args()

df = (
    pl.read_csv(args.csv)
    .with_columns(
        peak_memory_bytes=pl.max_horizontal('peak_rss_bytes', 'peak_footprint_bytes')
    )
    .sort('peak_memory_bytes', descending=True)
    .head(20)
    .with_columns(
        peak_memory_gb=(pl.col('peak_memory_bytes') / 1e9).round(1),
        duration_min=(pl.col('duration_s') / 60).round(1),
    )
    .select('test', 'peak_memory_gb', 'duration_min')
)

with pl.Config(tbl_width_chars=200, fmt_str_lengths=120, set_tbl_rows=20):
    print(df)
