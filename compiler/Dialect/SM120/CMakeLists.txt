set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/include/SelfCute/SM120/IR/SM120Ops.td)
mlir_tablegen(SM120Ops.h.inc -gen-op-decls)
mlir_tablegen(SM120Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRSelfCuteSM120OpsIncGen)

add_subdirectory(IR)
