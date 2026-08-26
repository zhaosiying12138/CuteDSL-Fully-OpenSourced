"""cutlass.cute.struct — @struct / MemRange / Align (compat)."""
from __future__ import annotations


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


def struct(cls):
    from cutlass.utils import _struct_decorator
    return _struct_decorator(cls)
