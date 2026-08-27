"""dtype markers and Constexpr wrapper for the compat surface."""
from __future__ import annotations


class _DType:
    """A scalar dtype marker. `mlir` gives the MLIR type text used in IR."""

    def __init__(self, name: str, mlir: str, bits: int):
        self.name, self.mlir, self.bits = name, mlir, bits

    @property
    def is_integer(self) -> bool:
        return self.name.startswith(("Int", "Uint", "UInt"))

    @property
    def is_float(self) -> bool:
        return self.name.startswith("Float")

    @property
    def width(self) -> int:
        return self.bits

    def __repr__(self):
        return self.name

    def __call__(self, value):
        # dtype instances double as conversion functions (cutlass.Int32(7));
        # official support code relies on the result being a typed scalar
        # carrying SSA / literal + ir_value()
        try:
            from cutlass import cutlass_dsl as _cdsl
            return _cdsl._typed_from_dtype(self, value=value)
        except ImportError:
            return TypedValue(value, self)

    def __str__(self):
        return self.name


class TypedValue:
    def __init__(self, value, dtype: _DType):
        self.value, self.dtype = value, dtype


Int8 = _DType("Int8", "i8", 8)
Int16 = _DType("Int16", "i16", 16)
Int32 = _DType("Int32", "i32", 32)
Int64 = _DType("Int64", "i64", 64)
UInt32 = _DType("UInt32", "i32", 32)
Uint8 = _DType("Uint8", "i8", 8)
Uint16 = _DType("Uint16", "i16", 16)
Uint32 = _DType("Uint32", "i32", 32)
Uint64 = _DType("Uint64", "i64", 64)
Float32 = _DType("Float32", "f32", 32)
Float16 = _DType("Float16", "f16", 16)
BFloat16 = _DType("BFloat16", "bf16", 16)
Float8E4M3FN = _DType("Float8E4M3FN", "!cute.f8e4m3fn", 8)
Float8E5M2 = _DType("Float8E5M2", "!cute.f8e5m2", 8)
Float8E8M0FNU = _DType("Float8E8M0FNU", "!cute.f8e8m0fnu", 8)
Float4E2M1FN = _DType("Float4E2M1FN", "!cute.f4e2m1fn", 4)

Numeric = _DType("Numeric", "i32", 32)
Boolean = _DType("Boolean", "i1", 1)

_BY_NAME = {d.name: d for d in
            (Int8, Int32, Int64, UInt32, Uint8, Uint16, Uint32, Uint64,
             Float32, Float16, BFloat16,
             Float8E4M3FN, Float8E5M2, Float8E8M0FNU, Float4E2M1FN)}
# unsigned aliases share the i32/i64 MLIR text (signedness is nominal here)
_BY_NAME["UInt64"] = Int64
_BY_NAME["UInt8"] = Uint8
_BY_NAME["UInt16"] = Uint16


def dtype(name: str) -> _DType:
    """cutlass.dtype('Float16') -> Float16 (argparse type hook)."""
    if isinstance(name, _DType):
        return name
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"can't interpret {name} as data type")


class Constexpr:
    """Marks a parameter as compile-time constant (specialization key)."""

    def __init__(self, value=None):
        self.value = value

    # Constexpr[X] annotation form (typing-style __class_getitem__)
    def __class_getitem__(cls, item):
        return ConstexprAnnotation(item)

    def __call__(self, value):
        return Constexpr(value)


class ConstexprAnnotation:
    """cutlass.Constexpr[cutlass.Int32] annotation."""

    def __init__(self, dtype):
        self.dtype = dtype
