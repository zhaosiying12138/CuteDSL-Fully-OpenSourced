// selfcute.sm120 dialect — concrete SM120a hardware ops: ldmatrix/stmatrix,
// warp mma.sync (FP16/BF16 and block-scaled kinds), TMA + mbarrier, setmaxnreg.
// Skeleton (M1): dialect + seed ops with full type/layout verifiers to come
// in M3-M7; parse/print/verify only for now.

#ifndef SELF_CUTE_SM120_OPS
#define SELF_CUTE_SM120_OPS

include "mlir/IR/OpBase.td"

def SelfCute_SM120_Dialect : Dialect {
  let name = "selfcute_sm120";
  let summary = "SM120a (GeForce Blackwell) hardware ops";
  let description = [{
    The selfcute.sm120 dialect is the single place where SM120a-specific
    instructions are modeled: ldmatrix/stmatrix SMEM<->RMEM fragments,
    mma.sync.aligned FP16/BF16 and narrow/block-scaled kinds, cp.async.bulk
    .tensor (TMA) transfers, mbarrier ops, and setmaxnreg. Lowering targets
    LLVM/NVVM ops plus a tightly-audited inline-PTX adapter for encodings
    NVVM lacks. The only target is sm_120a; other chips are rejected.
  }];
  let cppNamespace = "::mlir::selfcute::sm120";
}

class SelfCute_SM120_Op<string mnemonic, list<Trait> traits = []> :
    Op<SelfCute_SM120_Dialect, mnemonic, traits>;

def SelfCute_SM120_LdMatrixOp : SelfCute_SM120_Op<"ldmatrix"> {
  let summary = "warp-collective ldmatrix SMEM -> RMEM fragment load";
  let arguments = (ins AnyType:$smem_ptr, AnyType:$layout,
                   BoolAttr:$transpose, I32Attr:$num_matrices);
  let results = (outs AnyType:$fragment);
}

def SelfCute_SM120_MmaF16BF16Op : SelfCute_SM120_Op<"mma_f16bf16"> {
  let summary = "mma.sync.aligned.m16n8k16 (FP16/BF16, f32 accumulate)";
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$c);
  let results = (outs AnyType:$d);
}

def SelfCute_SM120_SetMaxRegOp : SelfCute_SM120_Op<"setmaxnreg"> {
  let summary = "setmaxnreg.inc/dec warp-group register reallocation";
  let arguments = (ins I32Attr:$value, BoolAttr:$is_increase);
  let results = (outs);
}

#endif // SELF_CUTE_SM120_OPS
