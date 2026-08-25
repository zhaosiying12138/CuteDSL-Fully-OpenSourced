// M1 skeleton DoD: all three selfcute dialects parse/print/verify.
// RUN: %selfcute_opt %s | %FileCheck %s

// CHECK-LABEL: func @kernel_skeleton(
func.func @kernel_skeleton(%smem : !llvm.ptr<3>, %stage : i32, %phase : i1,
                           %frag_a : !llvm.array<4 x f32>) {
  // CHECK: %[[A0:.*]] = "selfcute_kernel.shared_alloc"
  %sm = "selfcute_kernel.shared_alloc"() <{bytes = 65536 : i64, alignment = 128 : i32}> : () -> !llvm.ptr<3>

  // CHECK: "selfcute_kernel.local_tile"
  %tiled = "selfcute_kernel.local_tile"(%sm, %stage) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>

  // CHECK: "selfcute_pipeline.acquire"
  %s1 = "selfcute_pipeline.acquire"(%stage, %stage, %phase) : (i32, i32, i1) -> i32
  // CHECK: "selfcute_pipeline.release"
  %s2 = "selfcute_pipeline.release"(%s1, %stage, %phase) : (i32, i32, i1) -> i32

  // CHECK: "selfcute_sm120.ldmatrix"
  %f = "selfcute_sm120.ldmatrix"(%sm, %stage) <{num_matrices = 4 : i32, transpose = false}> : (!llvm.ptr<3>, i32) -> !llvm.array<4 x f32>
  // CHECK: "selfcute_sm120.mma_f16bf16"
  %d = "selfcute_sm120.mma_f16bf16"(%frag_a, %frag_a, %f) : (!llvm.array<4 x f32>, !llvm.array<4 x f32>, !llvm.array<4 x f32>) -> !llvm.array<4 x f32>
  // CHECK: "selfcute_sm120.setmaxnreg"
  "selfcute_sm120.setmaxnreg"() <{is_increase = true, value = 232 : i32}> : () -> ()

  // CHECK: return
  return
}
