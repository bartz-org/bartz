# bartz/tests/_memtrace.py
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

R"""
Opt-in pytest plugin that traces the memory usage of a test session.

Load it with ``-p tests._memtrace``; it does nothing unless ``--memlog`` is
passed. It writes one csv row per test with the memory occupation around the
test and the number of xla compilations it triggered, to tell apart memory that
accumulates over the session from memory a single test allocates and releases.

Example
-------
::

    JAX_PLATFORMS=cpu uv run python -m pytest -p tests._memtrace \
        --memlog=memory.csv --mem-dump-regions tests/test_mcmcloop.py
"""

import ctypes
import os
import sys
from collections.abc import Callable, Generator
from csv import DictWriter
from pathlib import Path
from subprocess import DEVNULL, run
from threading import Event, Thread
from time import perf_counter

import jax
import psutil
import pytest
from jax import monitoring

# jax wraps this event around `compile_or_get_cached`, so it fires for every
# executable that enters the process, whether compiled or read from the disk
# cache, and independently of whether the disk cache is enabled
COMPILE_EVENT = '/jax/core/compile/backend_compile_duration'

GB = 1 << 30
FIELDS = (
    'nodeid',
    'module',
    'duration',
    'peak',
    'before',
    'after',
    'growth',
    'compiles',
    'executables',
    'rss',
)

# offset of ri_phys_footprint in `struct rusage_info_v2` (sys/resource.h)
FOOTPRINT_OFFSET = 16 + 7 * 8


class Memory:
    """Read the memory occupation of the current process."""

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._libc = (
            ctypes.CDLL('/usr/lib/libSystem.B.dylib') if self.on_macos else None
        )
        self._buffer = (ctypes.c_uint8 * 1024)()

    @property
    def on_macos(self) -> bool:
        """Whether the physical footprint is the metric in use."""
        return sys.platform == 'darwin'

    def rss(self) -> int:
        """Return the resident set size in bytes."""
        return self._process.memory_info().rss

    def footprint(self) -> int:
        """Return the memory occupation in bytes, as the os accounts for it.

        On macOS this is the physical footprint, i.e., what Activity Monitor
        shows. It is read through a private `libSystem` call because the rss
        excludes dirty pages that the kernel has compressed, and so understates
        the memory by a large factor (4x in a test session). Elsewhere it falls
        back to the rss, which does not have the problem.
        """
        if self._libc is None:
            return self.rss()
        else:
            failed = self._libc.proc_pid_rusage(
                ctypes.c_int(self._process.pid),
                ctypes.c_int(4),  # RUSAGE_INFO_V4
                ctypes.byref(self._buffer),
            )
            if failed:
                return self.rss()
            else:
                return ctypes.c_uint64.from_buffer(self._buffer, FOOTPRINT_OFFSET).value


class PeakSampler(Thread):
    """Poll the memory occupation in the background to catch peaks within a test."""

    def __init__(self, memory: Memory, interval: float = 0.2) -> None:
        super().__init__(daemon=True)
        self._memory = memory
        self._interval = interval
        self._stop = Event()
        self.peak = 0

    def reset(self) -> None:
        """Restart the peak accumulator from the current occupation."""
        self.peak = self._memory.footprint()

    def run(self) -> None:
        """Sample the memory until stopped."""
        while not self._stop.wait(self._interval):
            self.peak = max(self.peak, self._memory.footprint())

    def stop(self) -> None:
        """Terminate the sampling loop."""
        self._stop.set()


class MemTrace:
    """Write per-test memory statistics to a csv file."""

    def __init__(self, path: Path, dump_regions: bool) -> None:
        worker = os.environ.get('PYTEST_XDIST_WORKER')
        if worker:
            path = path.with_name(f'{path.stem}-{worker}{path.suffix}')
        self._path = path
        self._dump_regions = dump_regions
        self._compiles = 0
        monitoring.register_event_duration_secs_listener(self.count_compile)
        self._memory = Memory()
        self._sampler = PeakSampler(self._memory)
        self._file = path.open('w', newline='')
        self._writer = DictWriter(self._file, FIELDS)
        self._writer.writeheader()
        self._initial = self._memory.footprint()
        self._previous = self._initial
        self._peak = self._initial
        self._rows: list[dict] = []
        self._sampler.start()

    def count_compile(
        self,
        event: str,
        duration_secs: float,  # noqa: ARG002, signature fixed by jax
        **kwargs: str | int,  # noqa: ARG002, signature fixed by jax
    ) -> None:
        """Count executables compiled or read from the disk cache."""
        if event == COMPILE_EVENT:
            self._compiles += 1

    def count_executables(self) -> int:
        """Return how many compiled executables the backend keeps alive."""
        return len(jax.devices()[0].client.live_executables())

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(
        self, item: pytest.Item
    ) -> Generator[None, object, object]:
        """Measure memory around the whole setup/call/teardown cycle of a test."""
        self._sampler.reset()
        start = perf_counter()
        try:
            return (yield)
        finally:
            duration = perf_counter() - start
            compiles, self._compiles = self._compiles, 0
            after = self._memory.footprint()
            peak = max(self._sampler.peak, after)
            row = dict(
                nodeid=item.nodeid,
                module=item.nodeid.split('::')[0],
                duration=round(duration, 3),
                peak=peak,
                before=self._previous,
                after=after,
                growth=after - self._previous,
                compiles=compiles,
                executables=self.count_executables(),
                rss=self._memory.rss(),
            )
            self._writer.writerow(row)
            self._file.flush()
            self._rows.append(row)
            self._previous = after
            self._peak = max(self._peak, peak)

    def dump_regions(self, write: Callable[[str], None]) -> None:
        """Save a breakdown of the memory in use to files next to the csv."""
        for tool in ('heap', 'vmmap'):
            dump = self._path.with_suffix(f'.{tool}.txt')
            flags = ['-summary'] if tool == 'vmmap' else []
            with dump.open('w') as file:
                run(  # noqa: S603, the command line is a literal
                    [tool, *flags, str(os.getpid())],
                    stdout=file,
                    stderr=DEVNULL,
                    check=False,
                )
            write(f'  {tool} output written to {dump}')

    def pytest_terminal_summary(
        self, terminalreporter: pytest.TerminalReporter
    ) -> None:
        """Print a summary of the trace."""
        self._sampler.stop()
        self._file.close()
        write = terminalreporter.write_line
        write(f'memory trace written to {self._path}')
        metric = 'footprint' if self._memory.on_macos else 'rss'
        write(
            f'{metric}: start {self._initial / GB:.2f} GB, '
            f'end {self._previous / GB:.2f} GB, '
            f'peak {self._peak / GB:.2f} GB'
        )
        compiles = sum(row['compiles'] for row in self._rows)
        write(f'xla compilations: {compiles}')
        jax.clear_caches()
        write(f'after clearing jax caches: {self._memory.footprint() / GB:.2f} GB')
        if self._dump_regions:
            self.dump_regions(write)
        write('largest growth steps:')
        for row in sorted(self._rows, key=lambda row: -row['growth'])[:10]:
            write(f'  {row["growth"] / GB:+.2f} GB  {row["nodeid"]}')


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the memory tracing options."""
    group = parser.getgroup('memtrace')
    group.addoption(
        '--memlog', default=None, help='write per-test memory stats to this csv file'
    )
    group.addoption(
        '--mem-dump-regions',
        action='store_true',
        help='at the end of the session, dump the macOS `heap` and `vmmap` breakdowns',
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the tracer if requested."""
    memlog = config.getoption('--memlog')
    if memlog:
        plugin = MemTrace(Path(memlog), config.getoption('--mem-dump-regions'))
        config.pluginmanager.register(plugin)
