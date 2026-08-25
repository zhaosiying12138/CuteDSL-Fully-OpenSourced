// selfcute.kernel dialect — Tensor/Fragment memory space ops, tiling and
// partitioning meta-ops, shared-memory allocation, kernel ABI.
// Skeleton (M1): dialect + seed ops, parse/print/verify only.

#ifndef SELF_CUTE_KERNEL_OPS
#define SELF_CUTE_KERNEL_OPS

include "mlir/IR/OpBase.td"

def SelfCute_Kernel_Dialect : Dialect {
  let name = "selfcute_kernel";
  let summary = "CuTeDSL kernel-level meta-ops: tensors, fragments, tiling, SMEM";
  let description = [{
    The selfcute.kernel dialect carries CuTeDSL kernel semantics that sit
    between the public cute layout algebra and SM120 lowering: tensor views
    over GMEM/SMEM/RMEM, CopyAtom/TiledCopy and MmaAtom/TiledMma material-
    ization (local_tile/partition/retile), shared-memory planning, and the
    kernel ABI boundary.
  }];
  let cppNamespace = "::mlir::selfcute::kernel";
}

class SelfCute_Kernel_Op<string mnemonic, list<Trait> traits = []> :
    Op<SelfCute_Kernel_Dialect, mnemonic, traits>;

def SelfCute_Kernel_SharedAllocOp : SelfCute_Kernel_Op<"shared_alloc"> {
  let summary = "Allocate dynamic shared memory with alignment";
  let description = [{
    Plans a shared-memory allocation of `bytes` bytes with the given
    alignment (min 128 for SM120 bank-conflict-free tiles). Materialized
    by the selfcute-plan-shared-storage pass into an llvm alloca-like SMEM
    base pointer; the SM120 runtime passes dynamic_smem_bytes at launch.
  }];
  let arguments = (ins I64Attr:$bytes, I32Attr:$alignment);
  let results = (outs AnyType:$ptr);
}

def SelfCute_Kernel_LocalTileOp : SelfCute_Kernel_Op<"local_tile"> {
  let summary = "Tile a tensor view by a static tile shape (meta-op)";
  let description = [{
    Partitions a tensor into a flat view of tiles (the CuTe local_tile).
    `tile_shape` and `tile_layout` are cute layout values materialized by
    the frontend; lowering rewrites this into cute algebra + views.
  }];
  let arguments = (ins AnyType:$tensor, AnyType:$tile_shape);
  let results = (outs AnyType:$tiled);
}

#endif // SELF_CUTE_KERNEL_OPS
