// M4: warp mma.sync m16n8k16 f16->f32 GEMM, operands loaded straight from
// gmem in the PTX-documented register layout. One warp computes D = A@B.
// A: 16x16 f16 row-major (flat = row*16 + col)
// B: 16x8  f16 [k][n]     (flat = k*8 + n)
// D: 16x8  f32 row-major  (flat = row*8 + col)

module attributes {gpu.container_module} {
  gpu.module @mma_kernel {
    gpu.func @mma16816(%a : !llvm.ptr<1>, %b : !llvm.ptr<1>, %d : !llvm.ptr<1>) kernel {
      %tid  = gpu.thread_id x
      %tid64 = arith.index_cast %tid : index to i64

      // groupID = tid >> 2 ; tid_in_group = tid % 4
      %c2 = arith.constant 2 : i64
      %c4 = arith.constant 4 : i64
      %c8 = arith.constant 8 : i64
      %c16 = arith.constant 16 : i64
      %group = arith.divsi %tid64, %c4 : i64
      %tig = arith.remsi %tid64, %c4 : i64

      // -------- A fragment: 4x f16x2 (row, col) pairs
      //   a0 (g, t2)   a1 (g+8, t2)   a2 (g, t2+8)   a3 (g+8, t2+8)
      %t2 = arith.muli %tig, %c2 : i64
      %t2p8 = arith.addi %t2, %c8 : i64
      %gp8 = arith.addi %group, %c8 : i64
      %rowA0 = arith.muli %group, %c16 : i64
      %rowA1 = arith.muli %gp8, %c16 : i64

      %fa0 = arith.addi %rowA0, %t2 : i64
      %fa1 = arith.addi %rowA1, %t2 : i64
      %fa2 = arith.addi %rowA0, %t2p8 : i64
      %fa3 = arith.addi %rowA1, %t2p8 : i64

      %pa0 = llvm.getelementptr %a[%fa0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %a0 = llvm.load %pa0 : !llvm.ptr<1> -> vector<2xf16>
      %pa1 = llvm.getelementptr %a[%fa1] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %a1 = llvm.load %pa1 : !llvm.ptr<1> -> vector<2xf16>
      %pa2 = llvm.getelementptr %a[%fa2] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %a2 = llvm.load %pa2 : !llvm.ptr<1> -> vector<2xf16>
      %pa3 = llvm.getelementptr %a[%fa3] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %a3 = llvm.load %pa3 : !llvm.ptr<1> -> vector<2xf16>

      // -------- B fragment: 2x f16x2, pairs (k, n=group):
      //   b0 { (t2, g), (t2+1, g) }  b1 { (t2+8, g), (t2+9, g) }
      // B flat = k*8 + n -> element0 = t2*8+g, element1 = (t2+1)*8+g
      
      %kb0 = arith.muli %t2, %c8 : i64
      %kb0b = arith.addi %kb0, %group : i64
      %pb0e0 = llvm.getelementptr %b[%kb0b] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %b0e0 = llvm.load %pb0e0 : !llvm.ptr<1> -> f16
      %kb0c = arith.addi %kb0b, %c8 : i64
      %pb0e1 = llvm.getelementptr %b[%kb0c] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %b0e1 = llvm.load %pb0e1 : !llvm.ptr<1> -> f16
      %u0 = llvm.mlir.undef : vector<2xf16>
      %ci0 = arith.constant 0 : i32
      %ci1 = arith.constant 1 : i32
      %v0 = llvm.insertelement %b0e0, %u0[%ci0 : i32] : vector<2xf16>
      %b0 = llvm.insertelement %b0e1, %v0[%ci1 : i32] : vector<2xf16>

      %kb1 = arith.addi %kb0, %c8 : i64
      %kb1b = arith.addi %kb1, %group : i64
      %pb1e0 = llvm.getelementptr %b[%kb1b] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %b1e0 = llvm.load %pb1e0 : !llvm.ptr<1> -> f16
      %kb1c = arith.addi %kb1b, %c8 : i64
      %pb1e1 = llvm.getelementptr %b[%kb1c] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
      %b1e1 = llvm.load %pb1e1 : !llvm.ptr<1> -> f16
      %u1 = llvm.mlir.undef : vector<2xf16>
      %cj0 = arith.constant 0 : i32
      %cj1 = arith.constant 1 : i32
      %w0 = llvm.insertelement %b1e0, %u1[%cj0 : i32] : vector<2xf16>
      %b1 = llvm.insertelement %b1e1, %w0[%cj1 : i32] : vector<2xf16>

      // -------- mma.sync
      %z0 = arith.constant 0.0 : f32
      %r = nvvm.mma.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%z0, %z0, %z0, %z0]
           {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
            shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}}
           : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>

      // -------- store D: c0/c1 (g, t2 / +1); c2/c3 (g+8, ...)
      %d0 = llvm.extractvalue %r[0] : !llvm.struct<(f32, f32, f32, f32)>
      %d1 = llvm.extractvalue %r[1] : !llvm.struct<(f32, f32, f32, f32)>
      %d2 = llvm.extractvalue %r[2] : !llvm.struct<(f32, f32, f32, f32)>
      %d3 = llvm.extractvalue %r[3] : !llvm.struct<(f32, f32, f32, f32)>

      %rowD0 = arith.muli %group, %c8 : i64
      %rowD1 = arith.muli %gp8, %c8 : i64
      %fd0 = arith.addi %rowD0, %t2 : i64
      %fd1 = arith.addi %fd0, %c2 : i64
      %fd2 = arith.addi %rowD1, %t2 : i64
      %fd3 = arith.addi %fd2, %c2 : i64

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
