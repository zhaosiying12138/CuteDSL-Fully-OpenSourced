"""llvm dialect bridge: inline_asm / extractvalue over the textual emitter.

The official signature used by flashinfer's fp4_common:

    llvm.inline_asm(result_type_or_None, [operands...], asm, constraints,
                    has_side_effects=..., is_align_stack=...,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=..., ip=...)

Operands are IrValue-wrapped SSAs (or Python ints folded as i32/f32/i64
constants by constraint letter). Result types are ir.Type tokens or
llvm.StructType literals. Mirrors the nvvm.inline_ptx encoding already
proven by the blockscaled mxf4nvf4 mma path in the self stack.
"""
from __future__ import annotations

from .. import ir as _ir
from ..._bridge_helpers import _emitter


class AsmDialect:
    AD_ATT = 0
    AD_INTEL = 1


class StructType:
    @staticmethod
    def get_literal(members):
        return _ir.Type("!" + "llvm.struct<(" +
                        ", ".join(m.text for m in members) + ")>")


def _fold_operand(v, constraint):
    """Python int/float -> SSA constant; IrValue -> inner SSA."""
    ssa = getattr(v, "ssa", None)
    if ssa is not None:
        return ssa
    e = _emitter()
    c = constraint.strip()
    if isinstance(v, int):
        if "l" in c:  # 64-bit operand class
            return e.ssa("i64", f"arith.constant {int(v)} : i64")
        return e.ssa("i32", f"arith.constant {int(v)} : i32")
    if isinstance(v, float):
        return e.ssa("f32", f"arith.constant {float(v)!r} : f32")
    raise TypeError(f"cannot fold inline_asm operand {v!r}")


def inline_asm(result_type, operands, asm, constraints,
               has_side_effects=True, is_align_stack=False,
               asm_dialect=AsmDialect.AD_ATT, loc=None, ip=None, **kw):
    e = _emitter()
    ops = [_fold_operand(o, "") for o in operands]
    # result type: single ir.Type, None (side-effecting, no result), or a
    # StructType literal (multiple results -> one struct SSA)
    if result_type is None:
        # void form: ro-bundle (+ optional predicate folded into the string
        # by callers that need it)
        names = ", ".join(o.name for o in ops)
        tys = ", ".join(o.type for o in ops)
        e.raw(f'nvvm.inline_ptx "{_esc(asm)}" ro ({names} : {tys})')
        return None
    return e.ssa(result_type.text, _inline_line(asm, constraints, ops,
                                                result_type, e))


def _inline_line(asm, constraints, ops, rtype, e):
    ro = ", ".join(o.name for o in ops)
    ot = ", ".join(o.type for o in ops)
    head = "nvvm.inline_ptx"
    if rtype is not None:
        return (f'nvvm.inline_ptx "{_esc(asm)}" '
                f'ro ({ro} : {ot}) -> {rtype.text}')
    return f'nvvm.inline_ptx "{_esc(asm)}" ({ro} : {ot})'


def _esc(asm: str) -> str:
    return asm.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\0a")


def extractvalue(type_token, struct_ssa, idx, loc=None, ip=None, **kw):
    e = _emitter()
    return e.ssa(type_token.text,
                 f"llvm.extractvalue {struct_ssa.name}[{int(idx[0] if isinstance(idx, (list, tuple)) else idx)}] : "
                 f"{struct_ssa.type}")


def ptrtoint(type_token, value, loc=None, ip=None, **kw):
    e = _emitter()
    v = getattr(value, "ssa", None) or value
    if hasattr(v, "ssa"):
        v = v.ssa
    return e.ssa(type_token.text, f"llvm.ptrtoint {v.name} : {v.type} to {type_token.text}")
