# bartz/tests/rbartpackages/_base.py
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

from collections.abc import Callable
from functools import wraps
from re import fullmatch, match

import numpy as np
from rpy2 import robjects
from rpy2.robjects import BoolVector, conversion, numpy2ri
from rpy2.robjects.help import Package
from rpy2.robjects.methods import RS4

# converter for pandas
pandas_converter = conversion.Converter('pandas')
try:
    from rpy2.robjects import pandas2ri
except ImportError:
    pass
else:
    pandas_converter = pandas2ri.converter

# converter for polars
polars_converter = conversion.Converter('polars')
try:
    import polars
    from rpy2.robjects import pandas2ri
except ImportError:
    pass
else:

    def polars_to_r(df):
        df = df.to_pandas()
        return pandas2ri.py2rpy(df)

    polars_converter.py2rpy.register(polars.DataFrame, polars_to_r)
    polars_converter.py2rpy.register(polars.Series, polars_to_r)

# converter for jax
jax_converter = conversion.Converter('jax')
try:
    import jax
except ImportError:
    pass
else:

    def jax_to_r(x):
        x = np.asarray(x)
        if x.ndim == 0:
            x = x[()]
        return numpy2ri.py2rpy(x)

    jax_converter.py2rpy.register(jax.Array, jax_to_r)

# converter for numpy
numpy_converter = numpy2ri.converter


# converter for BoolVector (why isn't it in the numpy converter?)
def bool_vector_to_python(x):
    return np.array(x, bool)


bool_vector_converter = conversion.Converter('bool_vector')
bool_vector_converter.rpy2py.register(BoolVector, bool_vector_to_python)


# converter for python dictionaries
dict_converter = conversion.Converter('dict')


def dict_to_r(x):
    return robjects.ListVector(x)


dict_converter.py2rpy.register(dict, dict_to_r)

R_IDENTIFIER = r'(?:[a-zA-Z]|\.(?![0-9]))[a-zA-Z0-9._]*'


class RObjectBase:
    """
    Base class for Python wrappers of R objects creators.

    Subclasses should define the class attribute `_rfuncname`, and declare
    stub methods decorated with `rmethod`.

    _rfuncname : str
        An R function in the format ``'<package>::<function>``. The function is
        called with the initialization arguments, converted to R objects, and is
        expected to return an R object. The attributes of the R object are
        converted to equivalent Python values and set as attributes of the
        Python object. The R object itself is assigned to the member `_robject`.
    """

    _converter = (
        robjects.default_converter
        + pandas_converter
        + polars_converter
        + numpy_converter
        + bool_vector_converter
        + jax_converter
        + dict_converter
    )
    _convctx = conversion.localconverter(_converter)

    def _py2r(self, x):
        if isinstance(x, __class__):
            return x._robject
        with self._convctx:
            return self._converter.py2rpy(x)

    def _r2py(self, x):
        with self._convctx:
            return self._converter.rpy2py(x)

    def _args2r(self, args):
        return tuple(map(self._py2r, args))

    def _kw2r(self, kw):
        return {key: self._py2r(value) for key, value in kw.items()}

    _rfuncname: str = NotImplemented

    @property
    def _library(self) -> str:
        """Parse `_rfuncname` to get the library. Also checks `_rfuncname` is valid."""
        pattern = rf'^({R_IDENTIFIER})::({R_IDENTIFIER})$'
        m = match(pattern, self._rfuncname)
        if m is None:
            msg = f'Invalid _rfuncname: {self._rfuncname}.'
            raise ValueError(msg)
        return m.group(1)

    def __init__(self, *args, **kw) -> None:
        robjects.r(f'loadNamespace("{self._library}")')
        func = robjects.r(self._rfuncname)
        obj = func(*self._args2r(args), **self._kw2r(kw))
        self._robject = obj
        if hasattr(obj, 'items'):
            for s, v in obj.items():
                setattr(self, s.replace('.', '_'), self._r2py(v))

    def __init_subclass__(cls, **kw):
        """Automatically add R documentation to subclasses."""
        library, name = cls._rfuncname.split('::')
        page = Package(library).fetch(name)
        if cls.__doc__ is None:
            cls.__doc__ = ''
        cls.__doc__ += 'R documentation:\n' + page.to_docstring()


def rmethod(meth: Callable, *, rname: str | None = None) -> Callable:
    """Automatically implement a method using the correspoding R method.

    Parameters
    ----------
    meth
        A method in a subclass of `RObjectBase`.
    rname
        The name of the method in R. If not specified, use the name of `meth`.

    Returns
    -------
    methimpl
        An implementation of the method that calls the R method. The original
        implementation of meth is completely discarded.

    Examples
    --------
    >>> class MyRObject(RObjectBase):
    ...     _rfuncname = 'mypackage::myfunction'
    ...     @partial(rmethod, rname='my.method')
    ...     def my_method(self, arg1: int, arg2: str):
    ...         ...
    """
    if rname is None:
        rname = meth.__name__

    # I can't automatically add a docstring to the method because the R class
    # can be determined at runtime

    @wraps(meth)
    def impl(self, *args, **kw):
        if isinstance(self._robject, RS4):
            func = robjects.r['$'](self._robject, rname)
            out = func(*self._args2r(args), **self._kw2r(kw))

        else:
            if not fullmatch(R_IDENTIFIER, rname):
                msg = f'Invalid R method name: {rname}'
                raise ValueError(msg)
            rclass = self._robject.rclass[0]
            func = robjects.r(
                f'getS3method("{rname}", "{rclass}", envir = asNamespace("{self._library}"))'
            )
            out = func(self._robject, *self._args2r(args), **self._kw2r(kw))

        return self._r2py(out)

    return impl
