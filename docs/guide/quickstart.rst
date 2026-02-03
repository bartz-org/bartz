.. bartz/docs/guide/quickstart.rst
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

Quickstart
==========

Basics
------

Import and use the `bartz.BART.gbart` class:

.. code-block:: python

    from bartz.BART import gbart
    bart = gbart(X, y, ...)
    y_pred = bart.predict(X_test)

The interface hews to the R package `BART3 <https://github.com/rsparapa/bnptools>`_, with a few differences explained in the documentation of `bartz.BART.gbart`. This interface has the longest ancestry, descending from the original BART implementation.

There is also a new interface where new features are added, `bartz.Bart`. However it is unstable and continuously evolving.

Getting a GPU
-------------

`bartz` works decently on CPU, but it shines on GPU where it can be up to 200x faster. Your personal computer may already have a nvidia GPU. The integrated GPUs on Apple Silicon Macs are not supported and won't ever be. If you don't have a GPU, you can rent one on the cloud, starting from about 0.10 $/hour, e.g., on `vast.ai <https://vast.ai>`_. If you are not familiar with connecting to remote machines, you can use `Google colab <https://colab.research.google.com>`_.

JAX
---

`bartz` is implemented using `jax`, a Google library for machine learning. It allows to run the code on GPU or TPU and do various other things.

For basic usage, JAX is just an alternative implementation of `numpy`. The arrays returned by `~bartz.BART.gbart` are "jax arrays" instead of "numpy arrays", but there is no perceived difference in their functionality. If you pass numpy arrays to `bartz`, they will be converted automatically. You don't have to deal with `jax` in any way.

For advanced usage, refer to the `jax documentation <https://docs.jax.dev>`_.

Advanced
--------

`bartz` exposes the various functions that implement the MCMC of BART. You can use those yourself to try to make your own variant of BART. See the rest of the documentation for reference; the main entry points are `bartz.mcmcstep.init` and `bartz.mcmcloop.run_mcmc`.
