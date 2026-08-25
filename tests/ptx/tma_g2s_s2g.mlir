// M5 vertical slice: TMA G2S + mbarrier completion + TMA S2G on sm_120a.
// One CTA (32 threads): TMA-load a 4x8 f32 tile (box 8x4) into SMEM via
// mbarrier complete_tx, wait on phase parity, plain-copy SMEM->out for
// golden check, then TMA-store an incremented tile back to another tensor.

module attributes {gpu.container_module} {
  gpu.module @tma_kernel {
    llvm.mlir.global internal @smem_tile() {addr_space = 3 : i32, alignment = 128 : i64} : !llvm.array<32 x f32>
    llvm.mlir.global internal @smem_bar() {addr_space = 3 : i32, alignment = 8 : i64} : i64

    gpu.func @tma_roundtrip(%tma_in : !llvm.ptr, %tma_out : !llvm.ptr,
                            %gmem_out : !llvm.ptr<1>) kernel {
      %tid = gpu.thread_id x
      %tid32 = arith.index_cast %tid : index to i32
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c128 = arith.constant 128 : i32
      %ticks = arith.constant 100000000 : i32
      %is0 = arith.cmpi eq, %tid32, %c0 : i32

      %smem = llvm.mlir.addressof @smem_tile : !llvm.ptr<3>
      %bar = llvm.mlir.addressof @smem_bar : !llvm.ptr<3>

      // ---- init mbarrier (1 arrival) + make it visible
      scf.if %is0 {
        nvvm.mbarrier.init %bar, %c1 : !llvm.ptr<3>, i32
      }
      nvvm.fence.mbarrier.init
      gpu.barrier

      // ---- producer: expect 128 tx-bytes then TMA-load box (0,0)
      scf.if %is0 {
        %tok = nvvm.mbarrier.arrive.expect_tx %bar, %c128 : !llvm.ptr<3>, i32 -> i64
        nvvm.cp.async.bulk.tensor.shared.cluster.global %smem, %tma_in, %bar, box[%c0, %c0] {isCTAOnly = true} : !llvm.ptr<3>, !llvm.ptr
      }

      // ---- consumer: wait for phase 0 completion
      nvvm.mbarrier.try_wait.parity %bar, %c0, %ticks : !llvm.ptr<3>, i32, i32
      gpu.barrier

      // ---- golden path 1: plain copy SMEM -> gmem_out
      %tid64 = arith.index_cast %tid : index to i64
      %sp = llvm.getelementptr %smem[%tid64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
      %v = llvm.load %sp {alignment = 4 : i64} : !llvm.ptr<3> -> f32
      %gp = llvm.getelementptr %gmem_out[%tid64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %v, %gp : f32, !llvm.ptr<1>

      // ---- golden path 2: increment tile in SMEM, TMA-store to tma_out
      %one = arith.constant 100.0 : f32
      %v2 = arith.addf %v, %one : f32
      llvm.store %v2, %sp : f32, !llvm.ptr<3>
      gpu.barrier
      scf.if %is0 {
        nvvm.cp.async.bulk.tensor.global.shared.cta %tma_out, %smem, box[%c0, %c0] : !llvm.ptr, !llvm.ptr<3>
        nvvm.cp.async.bulk.commit.group
        nvvm.cp.async.bulk.wait_group 0
      }
      gpu.barrier
      gpu.return
    }
  }
}
