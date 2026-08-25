// M5: 2-stage TMA pipeline with phase-parity rollover.
// Loads "tile" k = the 4x8 tensor windowed at row k (rows beyond 3 are
// OOB zero-fill), stage = k % 2, consumer waits parity floor(k/2) % 2,
// producer re-arms the same stage with tile k+2. Accumulates per-element
// sums across all tiles for the golden check.

module attributes {gpu.container_module} {
  gpu.module @tma_pipeline {
    llvm.mlir.global internal @smem_s0() {addr_space = 3 : i32, alignment = 128 : i64} : !llvm.array<32 x f32>
    llvm.mlir.global internal @smem_s1() {addr_space = 3 : i32, alignment = 128 : i64} : !llvm.array<32 x f32>
    llvm.mlir.global internal @smem_b0() {addr_space = 3 : i32, alignment = 8 : i64} : i64
    llvm.mlir.global internal @smem_b1() {addr_space = 3 : i32, alignment = 8 : i64} : i64

    gpu.func @tma_pipe(%tma : !llvm.ptr, %result : !llvm.ptr<1>, %n_tiles : i32) kernel {
      %tid = gpu.thread_id x
      %tid32 = arith.index_cast %tid : index to i32
      %tid64 = arith.index_cast %tid : index to i64
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %c128 = arith.constant 128 : i32
      %ticks = arith.constant 100000000 : i32
      %i1_64 = arith.constant 1 : i64
      %i4 = arith.constant 4 : i64
      %i2_64 = arith.constant 2 : i64
      %i3_64 = arith.constant 3 : i64
      %is0 = arith.cmpi eq, %tid32, %c0 : i32

      %s0 = llvm.mlir.addressof @smem_s0 : !llvm.ptr<3>
      %s1 = llvm.mlir.addressof @smem_s1 : !llvm.ptr<3>
      %b0 = llvm.mlir.addressof @smem_b0 : !llvm.ptr<3>
      %b1 = llvm.mlir.addressof @smem_b1 : !llvm.ptr<3>

      scf.if %is0 {
        nvvm.mbarrier.init %b0, %c1 : !llvm.ptr<3>, i32
        nvvm.mbarrier.init %b1, %c1 : !llvm.ptr<3>, i32
      }
      nvvm.fence.mbarrier.init
      gpu.barrier

      // ---- prologue: tile 0 -> s0, tile 1 -> s1
      scf.if %is0 {
        %t0 = nvvm.mbarrier.arrive.expect_tx %b0, %c128 : !llvm.ptr<3>, i32 -> i64
        nvvm.cp.async.bulk.tensor.shared.cluster.global %s0, %tma, %b0, box[%c0, %c0] {isCTAOnly = true} : !llvm.ptr<3>, !llvm.ptr
        %t1 = nvvm.mbarrier.arrive.expect_tx %b1, %c128 : !llvm.ptr<3>, i32 -> i64
        %row1 = arith.constant 1 : i32
        nvvm.cp.async.bulk.tensor.shared.cluster.global %s1, %tma, %b1, box[%c0, %row1] {isCTAOnly = true} : !llvm.ptr<3>, !llvm.ptr
      }

      // ---- mainloop
      %zero = arith.constant 0.0 : f32
      %c0_idx = arith.constant 0 : index
      %c1_idx = arith.constant 1 : index
      %nt_idx = arith.index_cast %n_tiles : i32 to index
      %acc0 = scf.for %k = %c0_idx to %nt_idx step %c1_idx
          iter_args(%a0 = %zero) -> (f32) {
        %k32 = arith.index_cast %k : index to i32
        %stage = arith.remsi %k32, %c2 : i32
        %nwrap = arith.divsi %k32, %c2 : i32
        %parity = arith.remsi %nwrap, %c2 : i32
        %isStage0 = arith.cmpi eq, %stage, %c0 : i32

        // wait stage's current phase
        scf.if %isStage0 {
          nvvm.mbarrier.try_wait.parity %b0, %parity, %ticks : !llvm.ptr<3>, i32, i32
        } else {
          nvvm.mbarrier.try_wait.parity %b1, %parity, %ticks : !llvm.ptr<3>, i32, i32
        }
        gpu.barrier

        // consume: thread tid accumulates its element of the stage
        %sel = scf.if %isStage0 -> (!llvm.ptr<3>) {
          scf.yield %s0 : !llvm.ptr<3>
        } else {
          scf.yield %s1 : !llvm.ptr<3>
        }
        %pc = llvm.getelementptr %sel[%tid64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
        %vc = llvm.load %pc : !llvm.ptr<3> -> f32
        %na0 = arith.addf %a0, %vc : f32
        gpu.barrier

        // produce: refill this stage with tile k+2
        %k2 = arith.addi %k32, %c2 : i32
        scf.if %is0 {
          scf.if %isStage0 {
            %tP = nvvm.mbarrier.arrive.expect_tx %b0, %c128 : !llvm.ptr<3>, i32 -> i64
            nvvm.cp.async.bulk.tensor.shared.cluster.global %s0, %tma, %b0, box[%c0, %k2] {isCTAOnly = true} : !llvm.ptr<3>, !llvm.ptr
          } else {
            %tP1 = nvvm.mbarrier.arrive.expect_tx %b1, %c128 : !llvm.ptr<3>, i32 -> i64
            nvvm.cp.async.bulk.tensor.shared.cluster.global %s1, %tma, %b1, box[%c0, %k2] {isCTAOnly = true} : !llvm.ptr<3>, !llvm.ptr
          }
        }
        scf.yield %na0 : f32
      }

      // ---- write per-thread accumulation
      %q0 = llvm.getelementptr %result[%tid64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %acc0, %q0 : f32, !llvm.ptr<1>
      gpu.return
    }
  }
}
