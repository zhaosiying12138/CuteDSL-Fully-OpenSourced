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


import struct as _struct


def _ptx_f32_literal(v: float) -> str:
    return "0f" + _struct.pack(">f", float(v)).hex().upper()


def _ptx_f64_literal(v: float) -> str:
    return "0d" + _struct.pack(">d", float(v)).hex().upper()


def _const_substitutable(v):
    """Literal constants get baked into the asm text as PTX immediates in
    a temp register — feeding arith.constant into inline_ptx makes the
    lowering emit an "n" (immediate) constraint that breaks float asm."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    if hasattr(v, "ssa") and getattr(v, "ssa", None) is None \
            and isinstance(getattr(v, "value", None), (int, float)) \
            and not isinstance(v.value, bool):
        return v.value
    cv = getattr(v, "const_val", None)
    if isinstance(cv, (int, float)) and not isinstance(cv, bool):
        return cv
    return None


def inline_asm(result_type, operands, asm, constraints,
               has_side_effects=True, is_align_stack=False,
               asm_dialect=AsmDialect.AD_ATT, loc=None, ip=None, **kw):
    e = _emitter()
    # Literal-constant operands are baked into the asm body as
    # mov-to-temp-register statements: feeding arith.constant into
    # nvvm.inline_ptx makes the lowering pick an "n" immediate constraint,
    # which is invalid for float asm operands. The vacated operand slot
    # is filled with a duplicate of a runtime register (unused in the
    # text) so the remaining $N positions do not shift.
    n_results = 0 if result_type is None else 1  # single or struct result

    consts = {i: _const_substitutable(o) for i, o in enumerate(operands)}
    runtime_idx = [i for i, c in consts.items() if c is None]
    decls, subs = [], {}
    for i, cv in consts.items():
        if cv is None:
            continue
        tname = f"%fk{i}"
        if isinstance(cv, float):
            decls.append(f".reg .f32 {tname};")
            decls.append(f"mov.f32 {tname}, {_ptx_f32_literal(cv)};")
        else:
            decls.append(f".reg .s32 {tname};")
            decls.append(f"mov.s32 {tname}, {int(cv)};")
        subs[f"${n_results + i}"] = tname
    if subs:
        for k, v in subs.items():
            asm = asm.replace(k, v)
        asm = "{\n" + "\n".join(decls) + "\n" + asm + "\n}"
        # duplicate a runtime operand into each vacated slot
        if runtime_idx:
            donor = operands[runtime_idx[0]]
            operands = [o if consts[i] is None else donor
                        for i, o in enumerate(operands)]
        else:
            # all-constant: emit with no operands at all
            ops0 = []
            if result_type is None:
                e.raw(f'nvvm.inline_ptx "{_esc(asm)}"')
                return None
            r = e.ssa(result_type.text,
                      f'nvvm.inline_ptx "{_esc(asm)}" -> {result_type.text}')
            from cutlass._mlir.ir import IrValue
            return IrValue(r)
    ops = [_fold_operand(o, "") for o in operands]
    if result_type is None:
        names = ", ".join(o.name for o in ops)
        tys = ", ".join(o.type for o in ops)
        e.raw(f'nvvm.inline_ptx "{_esc(asm)}" ro ({names} : {tys})')
        return None
    r = e.ssa(result_type.text, _inline_line(asm, constraints, ops,
                                             result_type, e))
    from cutlass._mlir.ir import IrValue
    return IrValue(r)


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
    v = getattr(struct_ssa, "ssa", struct_ssa)
    i = int(idx[0] if isinstance(idx, (list, tuple)) else idx)
    return e.ssa(type_token.text,
                 f"llvm.extractvalue {v.name}[{i}] : {v.type}")


def ptrtoint(type_token, value, loc=None, ip=None, **kw):
    e = _emitter()
    v = getattr(value, "ssa", None) or value
    if hasattr(v, "ssa"):
        v = v.ssa
    return e.ssa(type_token.text, f"llvm.ptrtoint {v.name} : {v.type} to {type_token.text}")


def _fold_operand(v, constraint=""):
    """Python int/float -> SSA constant; IrValue -> inner SSA."""
    ssa = getattr(v, "ssa", None)
    if ssa is not None:
        return ssa
    e = _emitter()
    if isinstance(v, int):
        return e.ssa("i32", f"arith.constant {int(v)} : i32")
    if isinstance(v, float):
        return e.ssa("f32", f"arith.constant {float(v):.10e}".replace("+", "") + " : f32")
    raise TypeError(f"cannot fold inline_asm operand {v!r}")
