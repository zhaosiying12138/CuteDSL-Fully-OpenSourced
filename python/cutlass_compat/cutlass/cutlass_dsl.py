"""cutlass.cutlass_dsl bridge: T (type-token factory), dsl_user_op, and the
typed-scalar constructors (Int32(x), Uint32(x), Float32(x), ...) used by
official CuTe-DSL support code.

`T` mirrors the official typing namespace: T.i32() / T.f32() / ... return
ir.Type tokens whose .text is the MLIR type used by the textual emitter.
"""
from __future__ import annotations

from ._bridge_helpers import TypedScalar, TypedBool  # noqa: F401
from ._mlir import ir as _ir
from . import dtypes as _dt


class _T:
    @staticmethod
    def i1():
        return _ir.Type("i1")

    @staticmethod
    def i8():
        return _ir.Type("i8")

    @staticmethod
    def i16():
        return _ir.Type("i16")

    @staticmethod
    def i32():
        return _ir.Type("i32")

    @staticmethod
    def i64():
        return _ir.Type("i64")

    @staticmethod
    def f16():
        return _ir.Type("f16")

    @staticmethod
    def f32():
        return _ir.Type("f32")

    @staticmethod
    def f64():
        return _ir.Type("f64")


T = _T()


def dsl_user_op(fn):
    """Official decorator marking a function as a device-side op. On the
    self stack these run inside the AST interpreter as plain calls."""
    fn._dsl_user_op = True
    return fn


# ---------------------------------------------------------------------------
# typed scalar classes bound to _DType instances
# ---------------------------------------------------------------------------
_CLASS_BY_DTYPE = {}


def _typed_from_dtype(dtype, ssa=None, value=None):
    import cutlass  # resolve through package exports
    cls = _CLASS_BY_DTYPE.get(dtype.name)
    if cls is None:
        cls = type(f"_{dtype.name}Scalar", (TypedScalar,), {})
        _CLASS_BY_DTYPE[dtype.name] = cls
        setattr(cutlass.cutlass_dsl, dtype.name, cls)
    v = value
    if isinstance(v, TypedScalar):
        if ssa is None:
            ssa = v.ssa
        if value is None or not isinstance(value, (int, float)):
            v = v.value
    elif hasattr(v, "name") and hasattr(v, "type") and ssa is None:
        # raw emitter SSA handle: adopt it directly
        return cls(value=None, dtype=dtype, ssa=v)
    return cls(value=v, dtype=dtype, ssa=ssa)
