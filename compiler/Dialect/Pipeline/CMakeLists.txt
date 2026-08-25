set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/include/SelfCute/Pipeline/IR/PipelineOps.td)
mlir_tablegen(PipelineOps.h.inc -gen-op-decls)
mlir_tablegen(PipelineOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRSelfCutePipelineOpsIncGen)

add_subdirectory(IR)
