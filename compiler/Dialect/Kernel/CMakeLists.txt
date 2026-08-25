set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/include/SelfCute/Kernel/IR/KernelOps.td)
mlir_tablegen(KernelOps.h.inc -gen-op-decls)
mlir_tablegen(KernelOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRSelfCuteKernelOpsIncGen)

add_subdirectory(IR)
