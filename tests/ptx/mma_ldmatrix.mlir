// M4: warp mma.sync m16n8k16 f16->f32 via SMEM staging + ldmatrix.
// Canonical fragment path: gmem -> smem (plain copies) -> ldmatrix.x4/.x2
// (hardware-defined fragment layout) -> nvvm.mma.sync -> f32 stores.
// A: 16x16 f16 row-major gmem; B: 16x8 f16 [k][n] gmem; D: 16x8 f32.

module attributes {gpu.container_module} {
  gpu.module @mma_kernel {
    llvm.mlir.global internal @smemA() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<256 x f16>
    llvm.mlir.global internal @smemB() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<128 x f16>

    gpu.func @mma16816(%a : !llvm.ptr<1>, %b : !llvm.ptr<1>, %d : !llvm.ptr<1>) kernel {
      %tid = gpu.thread_id x
      %tid64 = arith.index_cast %tid : index to i64

      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c4 = arith.constant 4 : i64
      %c8 = arith.constant 8 : i64
      %c16 = arith.constant 16 : i64

      %smemA = llvm.mlir.addressof @smemA : !llvm.ptr<3>
      %smemB = llvm.mlir.addressof @smemB : !llvm.ptr<3>

      // ---- stage A: 8 f16 per thread (32*8 = 256 elements)
      %t16 = arith.muli %tid64, %c8 : i64
      %i0 = arith.constant 0 : index
      %i1v = arith.constant 1 : index
      %c16i = arith.constant 8 : index
      scf.for %ia = %i0 to %c16i step %i1v {
        %ia64 = arith.index_cast %ia : index to i64
        %offa = arith.addi %t16, %ia64 : i64
        %ga = llvm.getelementptr %a[%offa] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
        %va = llvm.load %ga : !llvm.ptr<1> -> f16
        %sa = llvm.getelementptr %smemA[%offa] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
        llvm.store %va, %sa : f16, !llvm.ptr<3>
      }

      // ---- stage B: 4 f16 per thread (flat)
      %t4 = arith.muli %tid64, %c4 : i64
      %c4i = arith.constant 4 : index
      scf.for %ib = %i0 to %c4i step %i1v {
        %ib64 = arith.index_cast %ib : index to i64
        %offb = arith.addi %t4, %ib64 : i64
        %gb = llvm.getelementptr %b[%offb] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
        %vb = llvm.load %gb : !llvm.ptr<1> -> f16
        %sb = llvm.getelementptr %smemB[%offb] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
        llvm.store %vb, %sb : f16, !llvm.ptr<3>
      }

      gpu.barrier

      // ---- ldmatrix.x4 for A: lane l -> row l%16, col-half l/16
      %lane = arith.remsi %tid64, %c16 : i64
      %half = arith.divsi %tid64, %c16 : i64
      %lrow = arith.muli %lane, %c16 : i64
      %hcol = arith.muli %half, %c8 : i64
      %aec = arith.addi %lrow, %hcol : i64
      %pa = llvm.getelementptr %smemA[%aec] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
      %fa = nvvm.ldmatrix %pa {num = 4 : i32, layout = #nvvm.mma_layout<row>, eltType = #nvvm.ld_st_matrix_elt_type<b16>, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>

      // ---- ldmatrix.x2 for B (col layout): lanes -> k rows
      %brow = arith.muli %lane, %c8 : i64
      %pb = llvm.getelementptr %smemB[%brow] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f16
      %fb = nvvm.ldmatrix %pb {num = 2 : i32, layout = #nvvm.mma_layout<col>, eltType = #nvvm.ld_st_matrix_elt_type<b16>, shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32)>

      %a0 = llvm.extractvalue %fa[0] : !llvm.struct<(i32, i32, i32, i32)>
      %a1 = llvm.extractvalue %fa[1] : !llvm.struct<(i32, i32, i32, i32)>
      %a2 = llvm.extractvalue %fa[2] : !llvm.struct<(i32, i32, i32, i32)>
      %a3 = llvm.extractvalue %fa[3] : !llvm.struct<(i32, i32, i32, i32)>
      %b0 = llvm.extractvalue %fb[0] : !llvm.struct<(i32, i32)>
      %b1 = llvm.extractvalue %fb[1] : !llvm.struct<(i32, i32)>

      // ---- mma.sync f16 m16n8k16 -> f32 (operands as f16x2 vectors)
      %va0 = llvm.bitcast %a0 : i32 to vector<2xf16>
      %va1 = llvm.bitcast %a1 : i32 to vector<2xf16>
      %va2 = llvm.bitcast %a2 : i32 to vector<2xf16>
      %va3 = llvm.bitcast %a3 : i32 to vector<2xf16>
      %vb0 = llvm.bitcast %b0 : i32 to vector<2xf16>
      %vb1 = llvm.bitcast %b1 : i32 to vector<2xf16>
      %zc = arith.constant 0.0 : f32
      %r = nvvm.mma.sync A[%va0, %va1, %va2, %va3] B[%vb0, %vb1] C[%zc, %zc, %zc, %zc] {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>

      // ---- store D (PTX C layout): c0/c1 (group, t2 / +1); c2/c3 (+8 row)
      %d0 = llvm.extractvalue %r[0] : !llvm.struct<(f32, f32, f32, f32)>
      %d1 = llvm.extractvalue %r[1] : !llvm.struct<(f32, f32, f32, f32)>
      %d2 = llvm.extractvalue %r[2] : !llvm.struct<(f32, f32, f32, f32)>
      %d3 = llvm.extractvalue %r[3] : !llvm.struct<(f32, f32, f32, f32)>

      %group = arith.divsi %tid64, %c4 : i64
      %tig = arith.remsi %tid64, %c4 : i64
      %t2c = arith.muli %tig, %c2 : i64
      %gp8 = arith.addi %group, %c8 : i64
      %rowD0 = arith.muli %group, %c8 : i64
      %rowD1 = arith.muli %gp8, %c8 : i64
      %fd0 = arith.addi %rowD0, %t2c : i64
      %fd1 = arith.addi %fd0, %c1 : i64
      %fd2 = arith.addi %rowD1, %t2c : i64
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
