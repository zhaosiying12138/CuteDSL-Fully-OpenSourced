"""cutlass.cute.struct — @struct decorator + MemRange/Align (compat).

`cute.struct` is BOTH the decorator (@cute.struct class ...) and the
namespace (cute.struct.MemRange[...]) — realized as a callable proxy.
"""
from __future__ import annotations

import sys as _sys


class _MemRangeMeta(type):
    def __getitem__(cls, args):
        dtype, count = args
        from cutlass.utils import MemRangeSpec
        return MemRangeSpec(dtype, count)


class MemRange(metaclass=_MemRangeMeta):
    pass


class _AlignMeta(type):
    def __getitem__(cls, args):
        inner, alignment = args
        from cutlass.utils import AlignSpec
        return AlignSpec(inner, alignment)


class Align(metaclass=_AlignMeta):
    pass


class _StructNamespace:
    """Callable: @cute.struct — collect MemRange/Align fields; attribute
    access (cute.struct.MemRange) resolves to this module's names."""

    def __call__(self, cls):
        from cutlass.utils import _struct_decorator
        return _struct_decorator(cls)

    def __getattr__(self, name):
        return getattr(_sys.modules[__name__], name)
