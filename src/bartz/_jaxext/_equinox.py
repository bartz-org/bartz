# bartz/src/bartz/_jaxext/_equinox.py
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

"""The bartz `Module` base and its `field` helper.

`Module` is an `equinox.Module` whose metaclass registers bartz's `field` as a
dataclass *field specifier* (PEP 681), in addition to `dataclasses.field` and
`equinox.field`. Static type checkers then understand that ``x: T = field(...)``
declares no default, so a later required field is not misread as a required
field following a defaulted one. Subclass `Module` (rather than
`equinox.Module`) in any dataclass that assigns the `field` helper below.

The chain/data/sample axis markers attached by `field` are consumed by the
resolvers in `bartz.mcmcstep._axes`.
"""

import sys
from dataclasses import field as dataclasses_field
from dataclasses import fields
from typing import Any, TypeVar

from equinox import Module as EquinoxModule
from equinox import field as eqx_field

if sys.version_info >= (3, 11):
    from typing import dataclass_transform
else:  # WORKAROUND(python<3.11): typing.dataclass_transform was added in 3.11
    from typing_extensions import dataclass_transform


# No return annotation (hence the noqa): annotating `-> Field[Any]` makes ty model
# a defaulted field's value as `Field[Field[Any]]` and reject it against the field's
# declared array type. The inferred `Any` return keeps every call site clean.
def field(  # noqa: ANN202
    *,
    chains: int | None = None,
    data: int | None = None,
    samples: int | None = None,
    **kwargs: Any,
):
    """Extend `equinox.field` with chain/data/sample axis markers.

    Parameters
    ----------
    chains
        Index of the chain axis for the field's arrays, or `None` if the field
        has no chain axis. Any int is accepted, including negative indices with
        the usual numpy semantics (e.g. ``-1`` for the last axis); the index is
        normalized per-leaf against the leaf's ``ndim`` by `chain_vmap_axes`.
    data
    samples
        Indices of the data/sample axes for the field's arrays, declared in the
        chain-less "core" layout. `None` if the field has no data/sample axis.
        The index is normalized per-leaf against the core ``ndim`` (the leaf's
        ``ndim`` minus 1 when a chain axis is present, else the leaf's
        ``ndim``); the chain axis, if any, is treated as inserted after the
        data/sample axis exists, so `data_vmap_axes`/`trace_sample_axes` shift
        the returned sample index up by 1 when the chain position is at or
        before the core data/sample index.
    **kwargs
        Other parameters passed to `equinox.field`.

    Returns
    -------
    A dataclass field descriptor with the axis indices in the metadata, unset if `None`.
    """
    metadata = dict(kwargs.pop('metadata', {}))
    assert 'chains' not in metadata
    assert 'data' not in metadata
    assert 'samples' not in metadata
    for name, value in (('chains', chains), ('data', data), ('samples', samples)):
        # bool is a subclass of int; reject it so a boolean value does not
        # silently mean axis 0 or 1.
        assert not isinstance(value, bool), (
            f'{name!r} marker must be an int axis index or None, not bool'
        )
        if value is not None:
            metadata[name] = value
    return eqx_field(metadata=metadata, **kwargs)


# `field_specifiers` is read off the metaclass (where equinox anchors its own
# `dataclass_transform`), so registering `field` here must happen on a metaclass.
# `type(EquinoxModule)` avoids importing equinox's private metaclass, whose name
# differs across versions.
@dataclass_transform(field_specifiers=(dataclasses_field, eqx_field, field))
class _ModuleMeta(type(EquinoxModule)):
    pass


class Module(EquinoxModule, metaclass=_ModuleMeta):
    """`equinox.Module` that registers bartz's `field` as a field specifier."""


T = TypeVar('T', bound=Module)


def project(cls: type[T], source: object) -> T:
    """Build a `cls` instance by copying each of its fields from `source`.

    `source` only needs to expose `cls`'s field names as attributes; any extra
    attributes (e.g. when `source` is a wider dataclass) are ignored.
    """
    return cls(**{f.name: getattr(source, f.name) for f in fields(cls)})
