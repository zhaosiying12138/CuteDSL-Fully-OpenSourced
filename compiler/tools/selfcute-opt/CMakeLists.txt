# selfcute-opt links against the cutlass_compiler registration libraries so
# cute + base + selfcute round-trip in one tool.
set(SELF_CUTE_CUTLASS_COMPILER_ROOT "${SELF_CUTE_CUTLASS_COMPILER_ROOT}" CACHE PATH
    "Path to the built cutlass_compiler tree (build-compiler)")

set(_cc_src ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/cutlass/cutlass_compiler)
set(_cc_bin ${SELF_CUTE_CUTLASS_COMPILER_ROOT})

add_llvm_executable(selfcute-opt selfcute-opt.cpp)
llvm_update_compile_flags(selfcute-opt)
target_link_libraries(selfcute-opt PRIVATE
  SelfCuteRegistration
  MLIRMlirOptMain
  MLIRPass
  MLIRTransforms
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRNVVMDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRUBDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  ${_cc_bin}/cute_ir/lib/Registration/libMLIRCuteRegistration.a
  ${_cc_bin}/base/lib/Registration/libMLIRBaseRegistration.a
  ${_cc_bin}/cute_ir/lib/Dialect/IR/libMLIRCuteDialect.a
  ${_cc_bin}/cute_ir/lib/Dialect/Transforms/libMLIRCuteTransforms.a
  ${_cc_bin}/cute_ir/lib/Conversion/CuteToBase/libMLIRCuteConversion.a
  ${_cc_bin}/base/lib/Conversion/BaseToTargets/libMLIRBaseToTargets.a
)
target_include_directories(selfcute-opt PRIVATE
  ${_cc_src}/cute_ir/include
  ${_cc_src}/base/include
  ${_cc_bin}/cute_ir/include
  ${_cc_bin}/base/include
)
