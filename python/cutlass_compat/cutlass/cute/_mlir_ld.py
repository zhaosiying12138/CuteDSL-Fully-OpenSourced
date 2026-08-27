"""Probe helper: ld.global.v4.u32 through the llvm bridge, results -> y."""
from cutlass._bridge_helpers import _emitter
from cutlass._mlir.dialects import llvm
from cutlass._mlir import ir as _ir
from cutlass import Int64, Uint32
from cutlass.cutlass_dsl import T


def ld_v4_probe(mX, mY, byte_off):
    e = _emitter()
    base = mX.iterator + Int64(0)
    ptr_i = llvm.ptrtoint(T.i64(), base.llvm_ptr)
    addr = Int64(ptr_i) + Int64(byte_off)
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(addr).ir_value()],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    v0 = Uint32(llvm.extractvalue(T.i32(), res, [0])).ir_value().ssa
    # write the four u32s as raw words into y[3..6]
    for k in range(4):
        vk = Uint32(llvm.extractvalue(T.i32(), res, [k])).ir_value().ssa
        fv = e.ssa("f32", f"llvm.bitcast {vk.name} : i32 to f32")
        off = e.ssa("index", f"arith.constant {3 + k} : index")
        off64 = e.ssa("i64", f"arith.index_cast {off.name} : index to i64")
        p = e.ssa("!llvm.ptr<1>",
                  f"llvm.getelementptr {mY.ptr.name}[{off64.name}] : "
                  "(!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32")
        e.raw(f"llvm.store {fv.name}, {p.name} : f32, !llvm.ptr<1>")
