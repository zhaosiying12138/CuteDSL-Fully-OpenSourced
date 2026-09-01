# selfcute-opt links against the cutlass_compiler registration libraries so
# cute + base + selfcute round-trip in one tool.
set(SELF_CUTE_CUTLASS_COMPILER_ROOT "${SELF_CUTE_CUTLASS_COMPILER_ROOT}" CACHE PATH
    "Path to the built cutlass_compiler tree (build-compiler)")

set(_cc_src "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/cutlass/cutlass_compiler")
set(_cc_bin "${SELF_CUTE_CUTLASS_COMPILER_ROOT}")

if(NOT _cc_bin)
  message(FATAL_ERROR
    "SELF_CUTE_CUTLASS_COMPILER_ROOT must point at build-compiler")
endif()

# The standalone cutlass_compiler build does not export CMake targets into
# this independent build tree.  Do not spell the Unix .a filenames here:
# MSVC produces .lib files (and may use a different import/archive prefix).
function(_find_cutlass_compiler_library out_var name relative_dir)
  find_library(${out_var}
    NAMES ${name} lib${name}
    PATHS "${_cc_bin}/${relative_dir}"
    PATH_SUFFIXES Release
    NO_DEFAULT_PATH)
  if(NOT ${out_var})
    message(FATAL_ERROR
      "Could not find cutlass_compiler library ${name} under "
      "${_cc_bin}/${relative_dir}")
  endif()
  set(${out_var} "${${out_var}}" PARENT_SCOPE)
endfunction()

_find_cutlass_compiler_library(_cute_registration MLIRCuteRegistration
  cute_ir/lib/Registration)
_find_cutlass_compiler_library(_base_registration MLIRBaseRegistration
  base/lib/Registration)
_find_cutlass_compiler_library(_cute_dialect MLIRCuteDialect
  cute_ir/lib/Dialect/IR)
_find_cutlass_compiler_library(_cute_transforms MLIRCuteTransforms
  cute_ir/lib/Dialect/Transforms)
_find_cutlass_compiler_library(_cute_conversion MLIRCuteConversion
  cute_ir/lib/Conversion/CuteToBase)
_find_cutlass_compiler_library(_base_conversion MLIRBaseToTargets
  base/lib/Conversion/BaseToTargets)

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
  ${_cute_registration}
  ${_base_registration}
  ${_cute_dialect}
  ${_cute_transforms}
  ${_cute_conversion}
  ${_base_conversion}
)
target_include_directories(selfcute-opt PRIVATE
  "${_cc_src}/cute_ir/include"
  "${_cc_src}/base/include"
  "${_cc_bin}/cute_ir/include"
  "${_cc_bin}/base/include"
)
