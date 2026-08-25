// M4: warp mma.sync m16n8k16 f16->f32 via the centralized inline-PTX adapter
// (llvm.inline_asm). Register layout follows the PTX ISA documentation
// directly; this is the sanctioned path for anything NVVM conventions
// obscure. One warp computes D = A@B.
// A: 16x16 f16 row-major; B: 16x8 f16 [k][n]; D: 16x8 f32 row-major.

module attributes {gpu.container_module} {
  gpu.module @mma_kernel {
    gpu.func @mma16816(%a : !llvm.ptr<1>, %b : !llvm.ptr<1>, %d : !llvm.ptr<1>) kernel {
      %tid = gpu.thread_id x
      %tid64 = arith.index_cast %tid : index to i64

      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c4 = arith.constant 4 : i64
      %c8 = arith.constant 8 : i64
      %c16 = arith.constant 16 : i64
      %group = arith.divsi %tid64, %c4 : i64
      %tig = arith.remsi %tid64, %c4 : i64
      %t2 = arith.muli %tig, %c2 : i64
      %t2p1 = arith.addi %t2, %c1 : i64
      %t2p8 = arith.addi %t2, %c8 : i64
      %t2p9 = arith.addi %t2p8, %c1 : i64
      %gp8 = arith.addi %group, %c8 : i64

      // ---- A fragment (PTX doc): a0..a3, rows {group, group+8}, cols
      //      {t2, t2+1} and {t2+8, t2+9}, each f16x2 = one b32
      %rowA0 = arith.muli %group, %c16 : i64
      %rowA1 = arith.muli %gp8, %c16 : i64

      %fa0 = arith.addi %rowA0, %t2 : i64
      %fa1 = arith.addi %rowA1, %t2 : i64
      %fa2 = arith.addi %rowA0, %t2p8 : i64
      %fa3 = arith.addi %rowA1, %t2p8 : i64

      %pa0 = llvm.getelementptr %a[%fa0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %va0 = llvm.load %pa0 : !llvm.ptr<1> -> vector<2xf16>
      %pa1 = llvm.getelementptr %a[%fa1] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %va1 = llvm.load %pa1 : !llvm.ptr<1> -> vector<2xf16>
      %pa2 = llvm.getelementptr %a[%fa2] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %va2 = llvm.load %pa2 : !llvm.ptr<1> -> vector<2xf16>
      %pa3 = llvm.getelementptr %a[%fa3] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %va3 = llvm.load %pa3 : !llvm.ptr<1> -> vector<2xf16>

      // ---- B fragment: b0 {B[t2][g], B[t2+1][g]}, b1 {B[t2+8][g], B[t2+9][g]}
      // B flat = k*8 + n
      %kb0e0 = arith.muli %t2, %c8 : i64
      %fb0e0 = arith.addi %kb0e0, %group : i64
      %pb0e0 = llvm.getelementptr %b[%fb0e0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %b0e0 = llvm.load %pb0e0 : !llvm.ptr<1> -> f16
      %kb0e1 = arith.muli %t2p1, %c8 : i64
      %fb0e1 = arith.addi %kb0e1, %group : i64
      %pb0e1 = llvm.getelementptr %b[%fb0e1] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %b0e1 = llvm.load %pb0e1 : !llvm.ptr<1> -> f16
      %kb1e0 = arith.muli %t2p8, %c8 : i64
      %fb1e0 = arith.addi %kb1e0, %group : i64
      %pb1e0 = llvm.getelementptr %b[%fb1e0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %b1e0 = llvm.load %pb1e0 : !llvm.ptr<1> -> f16
      %kb1e1 = arith.muli %t2p9, %c8 : i64
      %fb1e1 = arith.addi %kb1e1, %group : i64
      %pb1e1 = llvm.getelementptr %b[%fb1e1] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %b1e1 = llvm.load %pb1e1 : !llvm.ptr<1> -> f16

      // pack pairs: element0 = first k, element1 = second k (low, high)
      %ci0 = arith.constant 0 : i32
      %ci1 = arith.constant 1 : i32
      %u = llvm.mlir.undef : vector<2xf16>
      %v0 = llvm.insertelement %b0e0, %u[%ci0 : i32] : vector<2xf16>
      %b0 = llvm.insertelement %b0e1, %v0[%ci1 : i32] : vector<2xf16>
      %v1 = llvm.insertelement %b1e0, %u[%ci0 : i32] : vector<2xf16>
      %b1 = llvm.insertelement %b1e1, %v1[%ci1 : i32] : vector<2xf16>

      // ---- bitcast fragments to b32 for inline asm "r" constraints
      %a0 = llvm.bitcast %va0 : vector<2xf16> to i32
      %a1 = llvm.bitcast %va1 : vector<2xf16> to i32
      %a2 = llvm.bitcast %va2 : vector<2xf16> to i32
      %a3 = llvm.bitcast %va3 : vector<2xf16> to i32
      %bb0 = llvm.bitcast %b0 : vector<2xf16> to i32
      %bb1 = llvm.bitcast %b1 : vector<2xf16> to i32

      // ---- C = 0 (f32 registers for the f32 accumulator variant)
      %zc = arith.constant 0.0 : f32

      // ---- inline_asm mma.sync (exact PTX ISA semantics)
      %res = llvm.inline_asm "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13}", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f" %a0, %a1, %a2, %a3, %bb0, %bb1, %zc, %zc, %zc, %zc : (i32, i32, i32, i32, i32, i32, f32, f32, f32, f32) -> !llvm.struct<(f32, f32, f32, f32)>

      // ---- store D: c0/c1 (group, t2 / t2+1); c2/c3 (group+8, ...)
      %d0 = llvm.extractvalue %res[0] : !llvm.struct<(f32, f32, f32, f32)>
      %d1 = llvm.extractvalue %res[1] : !llvm.struct<(f32, f32, f32, f32)>
      %d2 = llvm.extractvalue %res[2] : !llvm.struct<(f32, f32, f32, f32)>
      %d3 = llvm.extractvalue %res[3] : !llvm.struct<(f32, f32, f32, f32)>

      %rowD0 = arith.muli %group, %c8 : i64
      %rowD1 = arith.muli %gp8, %c8 : i64
      %fd0 = arith.addi %rowD0, %t2 : i64
      %fd1 = arith.addi %fd0, %c1 : i64
      %fd2 = arith.addi %rowD1, %t2 : i64
      %fd3 = arith.addi %fd2, %c1 : i64

      %pd0 = llvm.getelementptr %d[%fd0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %d0, %pd0 : f32, !llvm.ptr<1>
      %pd1 = llvm.getelementptr %d[%fd1] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %d1, %pd1 : f32, !llvm.ptr<1>
      %pd2 = llvm.getelementptr %d[%fd2] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %d2, %pd2 : f32, !llvm.ptr<1>
      %pd3 = llvm.getelementptr %d[%fd3] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %d3, %pd3 : f32, !llvm.ptr<1>
      gpu.return
    }
  }
}
