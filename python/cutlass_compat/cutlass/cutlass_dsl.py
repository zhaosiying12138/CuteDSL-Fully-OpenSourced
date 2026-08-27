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

# Official modules import scalar constructors from both ``cutlass`` and
# ``cutlass.cutlass_dsl``. The dtype objects are callable constructors and
# carry the MLIR type metadata consumed by the self frontend.
Int32 = _dt.Int32
Int64 = _dt.Int64
Uint8 = _dt.Uint8
Uint32 = _dt.Uint32
Uint64 = _dt.Uint64
Integer = TypedScalar


def dsl_user_op(fn):
    """Official decorator marking a function as a device-side op. On the
    self stack these run inside the AST interpreter as plain calls."""
    fn._dsl_user_op = True
    return fn


def extract_mlir_values(fn):
    return fn


def new_from_mlir_values(fn):
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
    source_dtype = None
    if isinstance(v, TypedScalar):
        source_dtype = v.dtype
        if ssa is None:
            ssa = v.ssa
        if value is None or not isinstance(value, (int, float)):
            v = v.value
    elif hasattr(v, "name") and hasattr(v, "type") and ssa is None:
        # raw emitter SSA handle: adopt it directly
        ssa = v
        v = None
    if ssa is not None and ssa.type != dtype.mlir:
        ssa = _cast_scalar_ssa(ssa, source_dtype, dtype)
    return cls(value=v, dtype=dtype, ssa=ssa)


def _cast_scalar_ssa(ssa, source_dtype, target_dtype):
    """Materialize official typed-scalar conversions in textual MLIR."""
    from ._bridge_helpers import _emitter

    e = _emitter()
    src, dst = ssa.type, target_dtype.mlir
    source_name = getattr(source_dtype, "name", "")
    source_unsigned = source_name.startswith(("Uint", "UInt"))
    if src == "index" and dst.startswith("i"):
        return e.ssa(dst, f"arith.index_cast {ssa.name} : index to {dst}")
    if src.startswith("i") and dst == "index":
        return e.ssa(dst, f"arith.index_cast {ssa.name} : {src} to index")
    if src.startswith("i") and dst.startswith("i"):
        src_bits, dst_bits = int(src[1:]), int(dst[1:])
        if src_bits < dst_bits:
            op = "arith.extui" if source_unsigned else "arith.extsi"
        elif src_bits > dst_bits:
            op = "arith.trunci"
        else:
            return ssa
        return e.ssa(dst, f"{op} {ssa.name} : {src} to {dst}")
    if src == "f16" and dst == "f32":
        return e.ssa(dst, f"llvm.fpext {ssa.name} : f16 to f32")
    if src == "bf16" and dst == "f32":
        return e.ssa(dst, f"arith.extf {ssa.name} : bf16 to f32")
    if src == "f32" and dst == "f16":
        return e.ssa(dst, f"arith.truncf {ssa.name} : f32 to f16")
    if src == "f32" and dst == "bf16":
        return e.ssa(dst, f"arith.truncf {ssa.name} : f32 to bf16")
    if src.startswith("i") and dst.startswith("f"):
        op = "arith.uitofp" if source_unsigned else "arith.sitofp"
        return e.ssa(dst, f"{op} {ssa.name} : {src} to {dst}")
    if src.startswith("f") and dst.startswith("i"):
        target_unsigned = target_dtype.name.startswith(("Uint", "UInt"))
        op = "arith.fptoui" if target_unsigned else "arith.fptosi"
        return e.ssa(dst, f"{op} {ssa.name} : {src} to {dst}")
    raise TypeError(f"unsupported typed scalar conversion {src} -> {dst}")
