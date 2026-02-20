.. bartz/docs/development.rst
..
.. Copyright (c) 2024-2026, The Bartz Contributors
..
.. This file is part of bartz.
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.

Development
===========

Initial setup
-------------

`Fork <https://github.com/bartz-org/bartz/fork>`_ the repository on Github, then clone the fork:

.. code-block:: shell

    git clone git@github.com:YourGithubUserName/bartz.git
    cd bartz

Install `R <https://cran.r-project.org>`_ and `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ (for example, with `Homebrew <https://brew.sh>`_ do :literal:`brew install r uv`). Then run

.. code-block:: shell

    make setup

to set up the Python and R environments. (Note: at the time of writing, the `R installation instructions for ubuntu <https://cran.r-project.org/bin/linux/ubuntu>`_ miss a :code:`sudo apt install r-base-dev` at the end.)

The Python environment is managed by uv. To run commands that involve the Python installation, do :literal:`uv run <command>`. For example, to start an IPython shell, do :literal:`uv run ipython`. Alternatively, do :literal:`source .venv/bin/activate` to activate the virtual environment in the current shell.

The R environment is automatically active when you use :literal:`R` in the project directory.

We don't support using conda's R, though it might work.

Contributing
------------

To contribute code changes to the main repository, create a `pull request <https://github.com/bartz-org/bartz/pulls>`_ from your fork to the main repo.

Pre-defined commands
--------------------

Development commands are defined in a makefile. Run :literal:`make` without arguments to list the targets. All commands that simply consist in invoking a tool with the right command line arguments use the :literal:`ARGS` variable to add extra arguments, for example:

.. code-block:: shell

    make tests ARGS='-k test_pigs_fly'

will invoke something like

.. code-block:: shell

    uv run pytest --foo=1 --bar=128 --etc-etc -k test_pigs_fly

Documentation
-------------

To build the documentation for the current working copy, do

.. code-block:: shell

    make docs

To build the documentation for the latest release tag, do

.. code-block:: shell

    make docs-latest

To debug the documentation build, do

.. code-block:: shell

    make docs SPHINXOPTS='--fresh-env --pdb'

Unit tests
----------

The typical workflow to debug new changes is to first run all tests with

.. code-block:: shell

    make tests

Then, if some tests fail, use :literal:`pytest` directly to run and debug only the relevant tests, e.g., with

.. code-block:: shell

    uv run pytest --lf --sw --pdb

Where :code:`--lf` selects only the tests that failed, :code:`--sw` stops on the first failed test, starting again from it on the next run, and :code:`--pdb` opens the python debugger at the point where the test failed. Another useful option is :code:`-k <pattern>`, which selects only tests whose name matches <pattern>.

Debugging dependencies
----------------------

To debug tests that fail with old versions of dependencies, it's convenient to piggyback on the predefined make target using :code:`ARGS`:

.. code-block:: shell

    make tests-old ARGS='-n0 -k test_pigs_fly'

Where :code:`-n0` disables test parallelization.

For more fine-grained control, it's useful to invoke directly :code:`uv` with the :code:`--with` option, e.g., the following command will start an IPython shell equipped with specific versions of python and jax:

.. code-block:: shell

    uv run --with='jax<0.7,jaxlib<0.7' --isolated --python=3.11 --dev python -m IPython

Benchmarks
----------

The benchmarks are managed with `asv <https://asv.readthedocs.io/en/latest>`_. The basic asv workflow is:

.. code-block:: shell

    uv run asv run      # run and save benchmarks on main branch
    uv run asv publish  # create html report
    uv run asv preview  # start a local server to view the report

:literal:`asv run` writes the results into files saved in :literal:`./benchmarks`. These files are tracked by git; consider deliberately not committing all results generated while developing.

There are a few make targets for common asv commands. The most useful command during development is

.. code-block:: shell

    make asv-quick ARGS='--bench <pattern>'

This runs only benchmarks whose name matches <pattern>, only once, within the working copy and current Python environment.

Profiling
---------

Use the `JAX profiling utilities <https://docs.jax.dev/en/latest/profiling.html>`_ to profile `bartz`. By default the MCMC loop is compiled all at once, which makes it quite opaque to profiling. There are two ways to understand what's going on inside in more detail: 1) inspect the individual operations and use intuition to understand to what piece of code they correspond to, 2) turn on bartz's profile mode. Basic workflow:

.. code-block:: python

    from jax.profiler import trace, ProfileOptions
    from bartz.BART import gbart
    from bartz import profile_mode

    traceopt = ProfileOptions()

    # this setting makes Python function calls show up in the trace
    traceopt.python_tracer_level = 1

    # on cpu, this makes the trace detailed enough to understand what's going on
    # even within compiled functions
    traceopt.host_tracer_level = 2

    with trace('./trace_results', profiler_options=traceopt), profile_mode(True):
        bart = gbart(...)

On the first run, the trace will show compilation operations, while subsequent runs (within the same Python shell) will be warmed-up. Start a xprof server to visualize the results:

.. code-block:: shell

    $ uvx --python 3.13 xprof ./trace_results
    [...]
    XProf at http://localhost:8791/ (Press CTRL+C to quit)

Open the provided URL in a browser. In the sidebar, select the tool "Trace Viewer".

In "profile mode", the MCMC loop is split into a few chunks that are compiled separately, allowing to see at a glance how much time each phase of the MCMC cycle takes. This causes some overhead, so the timings are not equivalent to the normal mode ones. On some specific example on CPU, Bartz was 20% slower in profile mode with one chain, and 2x slower with multiple chains.
