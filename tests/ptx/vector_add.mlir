// M1 backend-spine test: hand-written vector-add kernel (raw-pointer ABI,
// mirroring how CuTeDSL kernels see memory after tensor lowering).
// Pipeline: cutlass-compiler --one-shot-convert-to-llvm
//                            --attach-nvvm-target=chip=sm_120a
//           mlir-translate --mlir-to-llvmir
//           llc -march=nvptx64 -mcpu=sm_120a
// Expected: textual PTX with .target sm_120a, executed via Driver JIT.

module attributes {gpu.container_module} {
  gpu.module @vector_add_kernel {
    gpu.func @add(%a : !llvm.ptr<1>, %b : !llvm.ptr<1>, %c : !llvm.ptr<1>, %n : i32) kernel {
      %tx  = gpu.thread_id  x
      %bdx = gpu.block_dim  x
      %bx  = gpu.block_id   x
      %off = arith.muli %bx, %bdx : index
      %i   = arith.addi %off, %tx : index
      %i64 = arith.index_cast %i : index to i64
      %n_i = arith.extsi %n : i32 to i64
      %inb = arith.cmpi slt, %i64, %n_i : i64
      scf.if %inb {
        %pa = llvm.getelementptr %a[%i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
        %pb = llvm.getelementptr %b[%i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
        %xa = llvm.load %pa : !llvm.ptr<1> -> f32
        %xb = llvm.load %pb : !llvm.ptr<1> -> f32
        %xc = arith.addf %xa, %xb : f32
        %pc = llvm.getelementptr %c[%i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
        llvm.store %xc, %pc : f32, !llvm.ptr<1>
      }
      gpu.return
    }
  }
}
