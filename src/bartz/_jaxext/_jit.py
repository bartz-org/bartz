# bartz/src/bartz/_jaxext/_jit.py
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

"""Signature-preserving `jax.jit` wrapper."""

from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

from jax import ShapeDtypeStruct
from jax import jit as _jax_jit
from jax.stages import Lowered, Traced
from jaxtyping import PyTree

_P = ParamSpec('_P')
_R = TypeVar('_R')
_R_co = TypeVar('_R_co', covariant=True)


@runtime_checkable
class JitWrapped(Protocol[_P, _R_co]):
    """Static type of a jitted function: the wrapped signature plus jit methods."""

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ...

    def clear_cache(self) -> None: ...

    def eval_shape(
        self, *args: _P.args, **kwargs: _P.kwargs
    ) -> PyTree[ShapeDtypeStruct]: ...

    def lower(self, *args: _P.args, **kwargs: _P.kwargs) -> Lowered: ...

    def trace(self, *args: _P.args, **kwargs: _P.kwargs) -> Traced: ...

    if not TYPE_CHECKING:
        # WORKAROUND(beartype<99): beartype chokes on ParamSpec-subscripted
        # generics, and the jaxtyping import hook used by the test suite makes
        # it process the `JitWrapped[_P, _R]` hints in `jit`'s overloads. Erase
        # the subscript at runtime so beartype sees the plain runtime-checkable
        # protocol, which jitted functions genuinely satisfy. `99` is a
        # placeholder for the beartype release gaining PEP 612 generics support.
        def __class_getitem__(cls, item: object) -> type:
            return cls


# WORKAROUND(jax<99): `jax.jit` is typed to return `JitWrapped`, which erases the
# wrapped function's signature, so static checkers can't validate calls to jitted
# functions. This shim recovers the signature via `ParamSpec`, declaring our own
# `JitWrapped` protocol that combines it with the jit-specific methods (including
# `clear_cache`, which jax adds to the jitted callable at runtime and omits from
# its own static `JitWrapped` type). Tracked upstream at jax-ml/jax#23719; the
# jax maintainers are blocked on migrating internal Google code to a type checker
# that understands `ParamSpec` (jax itself has moved to pyrefly). Once `jax.jit`
# preserves the signature natively, this whole module can go and `jit` can be
# imported straight from jax. `99` is a placeholder for that unknown future jax
# release.
@overload
def jit(
    fun: Callable[_P, _R],
    /,
    *,
    static_argnums: int | Sequence[int] | None = ...,
    static_argnames: str | Sequence[str] | None = ...,
    donate_argnums: int | Sequence[int] | None = ...,
    **kwargs: Any,
) -> JitWrapped[_P, _R]: ...


@overload
def jit(
    fun: None = ...,
    /,
    *,
    static_argnums: int | Sequence[int] | None = ...,
    static_argnames: str | Sequence[str] | None = ...,
    donate_argnums: int | Sequence[int] | None = ...,
    **kwargs: Any,
) -> Callable[[Callable[_P, _R]], JitWrapped[_P, _R]]: ...
def jit(fun: Any = None, /, **kwargs: Any) -> Any:
    """Wrap `jax.jit` preserving the wrapped function's static type signature.

    `jax.jit` is typed to return an opaque ``JitWrapped`` callable, which erases
    the wrapped signature; static checkers then treat every call to a jitted
    function as returning an unknown type, cascading into false positives. This
    shim is typed with a `ParamSpec` so jitted calls keep their real signature
    and argument checking, while at runtime it just defers to `jax.jit`.

    Use it as a drop-in for both decorator forms, ``@jit`` and ``@jit(...)``.

    Parameters
    ----------
    fun
        The function to compile, or `None` to use the keyword-only form.
    **kwargs
        Keyword arguments forwarded to `jax.jit` (e.g. `static_argnums`,
        `static_argnames`, `donate_argnums`).

    Returns
    -------
    The jitted function, or a decorator if `fun` is `None`.
    """
    # WORKAROUND(jax<0.8.1): jax gained native `@jit(...)` two-stage decorator
    # support in 0.8.1. Once the floor reaches 0.8.1 the runtime fallback could
    # defer to jax's native form, but keep the shim regardless, because jax's
    # own overloads still return `JitWrapped` and erase the signature; the
    # ParamSpec typing here is the whole point.
    if fun is None:
        return lambda f: _jax_jit(f, **kwargs)
    else:
        return _jax_jit(fun, **kwargs)
