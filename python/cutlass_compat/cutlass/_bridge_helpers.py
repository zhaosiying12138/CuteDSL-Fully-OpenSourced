"""Shared bridge helpers for official-API emulation over the self stack."""
from __future__ import annotations


def _emitter():
    from self_cutedsl.frontend import builtins as _b
    e = getattr(_b._active, "emitter", None)
    if e is None:
        raise RuntimeError(
            "emitter requested outside an active @cute.jit trace")
    return e


def _interp():
    from self_cutedsl.frontend import builtins as _b
    it = getattr(_b, "_active", None)
    if it is None:
        raise RuntimeError("no active @cute.jit trace")
    return it


class TypedScalar:
    """Base for official typed scalars (Int32/Uint32/Float32/...) that carry
    an SSA value through arithmetic while staying usable as Python values at
    trace time. `dtype` is a cutlass_compat _DType."""

    __slots__ = ("ssa", "value", "dtype")

    def __init__(self, value=None, dtype=None, ssa=None):
        self.ssa = ssa
        self.value = value if not hasattr(value, "ssa") else None
        self.dtype = dtype
        if ssa is None and isinstance(value, TypedScalar):
            self.ssa = value.ssa
            self.dtype = dtype or value.dtype
            self.value = value.value
        elif ssa is None and hasattr(value, "ssa") and not hasattr(value, "dtype"):
            # IrValue wrapper from the _mlir bridge
            self.ssa = value.ssa
            self.value = None
        elif ssa is None and hasattr(value, "name") and hasattr(value, "type"):
            # raw emitter SSA handle
            self.ssa = value
            self.value = None

    # -- official API -----------------------------------------------------
    def ir_value(self, loc=None, ip=None):
        lit = None
        if self.ssa is None:
            e = _emitter()
            if isinstance(self.value, (int, float)) \
                    and not isinstance(self.value, bool):
                lit = self.value
            if isinstance(self.value, int):
                ty = self.dtype.mlir if self.dtype else "i32"
                self.ssa = e.ssa(ty, f"arith.constant {int(self.value)} : {ty}")
            elif isinstance(self.value, float):
                self.ssa = e.ssa(
                    "f32",
                    "arith.constant " + f"{float(self.value):.10e}".replace("+", "") + " : f32")
            else:
                raise TypeError(f"no SSA for {self!r}")
        iv = __import__("cutlass._mlir.ir", fromlist=["IrValue"]).IrValue(self.ssa)
        if lit is not None:
            iv.const_val = lit
        return iv

    # -- arithmetic (traced when both sides carry SSA) --------------------
    def _bin(self, other, op_int, op_flt):
        e = _emitter()
        a = self._coerce(e)
        if hasattr(other, "name") and hasattr(other, "type"):
            b = other  # raw SSA
        else:
            b = other._coerce(e) if isinstance(other, TypedScalar) else \
                _const_like(e, a, other)
        if a.type != b.type and "f" not in (a.type[0], b.type[0]):
            if b.type == "i32" and a.type == "index":
                b = e.ssa("index", f"arith.index_cast {b.name} : i32 to index")
            elif a.type == "i32" and b.type == "index":
                a = e.ssa("index", f"arith.index_cast {a.name} : i32 to index")
            elif b.type != a.type and "i64" in (a.type, b.type):
                wide = "i64"
                if a.type != wide:
                    ext = "arith.extui" if _is_unsigned(self.dtype) \
                        else "arith.extsi"
                    a = e.ssa(wide, f"{ext} {a.name} : {a.type} to {wide}")
                if b.type != wide:
                    other_dtype = other.dtype \
                        if isinstance(other, TypedScalar) else None
                    ext = "arith.extui" if _is_unsigned(other_dtype) \
                        else "arith.extsi"
                    b = e.ssa(wide, f"{ext} {b.name} : {b.type} to {wide}")
        isf = a.type.startswith("f")
        op = op_flt if isf else op_int
        return _make_typed(self.dtype, ssa=e.ssa(a.type, f"{op} {a.name}, {b.name} : {a.type}"))

    def _un(self, op_int):
        e = _emitter()
        a = self._coerce(e)
        return _make_typed(self.dtype, ssa=e.ssa(a.type, f"{op_int} {a.name} : {a.type}"))

    def _coerce(self, e):
        if self.ssa is None:
            self.ir_value()
        return self.ssa

    def __add__(self, o):
        return self._bin(o, "arith.addi", "arith.addf")

    def __radd__(self, o):
        return self._bin(o, "arith.addi", "arith.addf")

    def __sub__(self, o):
        return self._bin(o, "arith.subi", "arith.subf")

    def __rsub__(self, o):
        e = _emitter()
        a = _const_like(e, self._coerce(e), o)
        b = self._coerce(e)
        isf = a.type.startswith("f")
        op = "arith.subf" if isf else "arith.subi"
        return _make_typed(self.dtype, ssa=e.ssa(a.type, f"{op} {a.name}, {b.name} : {a.type}"))

    def __mul__(self, o):
        return self._bin(o, "arith.muli", "arith.mulf")

    def __rmul__(self, o):
        return self._bin(o, "arith.muli", "arith.mulf")

    def __floordiv__(self, o):
        return self._bin(o, "arith.divsi", "arith.divf")

    def __truediv__(self, o):
        return self._bin(o, "arith.divsi", "arith.divf")

    def __mod__(self, o):
        return self._bin(o, "arith.remsi", "arith.divf")

    def __and__(self, o):
        return self._bin(o, "arith.andi", "arith.andi")

    def __or__(self, o):
        return self._bin(o, "arith.ori", "arith.ori")

    def __xor__(self, o):
        return self._bin(o, "arith.xori", "arith.xori")

    def __lshift__(self, o):
        return self._bin(o, "arith.shli", "arith.shli")

    def __rshift__(self, o):
        return self._bin(o, "arith.shrsi", "arith.shrsi")

    def __neg__(self):
        return self._un("arith.neg")

    def __invert__(self):
        return self._un("arith.xori")  # caller supplies -1 via ~ semantics

    def _cmp(self, o, int_pred, flt_pred):
        e = _emitter()
        a = self._coerce(e)
        b = o._coerce(e) if isinstance(o, TypedScalar) else _const_like(e, a, o)
        isf = a.type.startswith("f")
        pred = flt_pred if isf else int_pred
        op = "arith.cmpf" if isf else "arith.cmpi"
        return e.ssa("i1", f"{op} {pred}, {a.name}, {b.name} : {a.type}")

    def __lt__(self, o):
        return TypedBool(self._cmp(o, "slt", "olt"))

    def __le__(self, o):
        return TypedBool(self._cmp(o, "sle", "ole"))

    def __gt__(self, o):
        return TypedBool(self._cmp(o, "sgt", "ogt"))

    def __ge__(self, o):
        return TypedBool(self._cmp(o, "sge", "oge"))

    def __eq__(self, o):
        if isinstance(o, TypedScalar):
            return TypedBool(self._cmp(o, "eq", "oeq"))
        return NotImplemented

    def __ne__(self, o):
        if isinstance(o, TypedScalar):
            return TypedBool(self._cmp(o, "ne", "one"))
        return NotImplemented

    def __bool__(self):
        if self.value is not None and self.ssa is None:
            return bool(self.value)
        raise TypeError("dynamic value in boolean context; use select")

    def __repr__(self):
        n = getattr(self.ssa, "name", None)
        return f"<{self.dtype.name if self.dtype else '?'} {n or self.value!r}>"


class TypedBool(TypedScalar):
    """i1 typed scalar."""

    def __init__(self, ssa=None, value=None):
        super().__init__(value=value, dtype=None, ssa=ssa)


def _is_unsigned(dtype) -> bool:
    return getattr(dtype, "name", "").startswith(("Uint", "UInt"))


def _const_like(e, like, v):
    t = like.type
    if t.startswith("f"):
        return e.ssa(t, f"arith.constant {float(v)!r} : {t}")
    return e.ssa(t, f"arith.constant {int(v)} : {t}")


def _make_typed(dtype, ssa=None, value=None):
    from cutlass import cutlass_dsl as _cdsl
    return _cdsl._typed_from_dtype(dtype, ssa=ssa, value=value)
