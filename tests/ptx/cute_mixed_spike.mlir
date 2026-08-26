// Spike (object-model plan): a gpu kernel with ONE dynamic leaf feeding
// cute layout algebra (make_shape/make_stride/make_layout/zipped_divide/
// composition), the result consumed by crd2idx → arith → nvvm op, proving
// the three cute passes lower the algebra to base inside a gpu.module and
// the module still reaches sm_120a PTX.

module attributes {gpu.container_module} {
  gpu.module @spike {
    gpu.func @mixed(%gptr : !llvm.ptr<1>, %n : i32) kernel {
      // dynamic leaf: shape (n, 8) — ONE dynamic operand for the ? leaf
      %sh = cute.make_shape (%n) : (i32) -> !cute.shape<"(?,8)">
      %st = cute.make_stride (%n) : (i32) -> !cute.stride<"(1,?)">
      %lay = cute.make_layout (%sh, %st)
           : (!cute.shape<"(?,8)">, !cute.stride<"(1,?)">) -> !cute.layout<"(?,8):(1,?)">

      // static tiler (shape form) + zipped_divide
      %tiler = cute.static : !cute.shape<"(4,8)">
      %zd = cute.zipped_divide(%lay, %tiler)
          : (!cute.layout<"(?,8):(1,?)">, !cute.shape<"(4,8)">)
         -> !cute.layout<"((4,8),(?,1)):((1,?),(4,?))">

      // layout_eval: coord (1,3) through the divided layout -> index
      %c1i = arith.constant 1 : i32
      %c3i = arith.constant 3 : i32
      %crd = cute.make_coord (%c1i, %c3i) : (i32, i32) -> !cute.coord<"(?,?)">
      %idx = cute.layout_eval(%crd, %zd)
          : (!cute.coord<"(?,?)">, !cute.layout<"((4,8),(?,1)):((1,?),(4,?))">)
         -> !cute.int_tuple<"?">

      // feed the folded index into a base op (store) — mixed lowering
      %z = arith.constant 0.0 : f32
      %i32v = "cute.get_scalars"(%idx) : (!cute.int_tuple<"?">) -> i32
      %i64v = arith.extsi %i32v : i32 to i64
      %p = llvm.getelementptr %gptr[%i64v] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %z, %p : f32, !llvm.ptr<1>
      gpu.return
    }
  }
}
