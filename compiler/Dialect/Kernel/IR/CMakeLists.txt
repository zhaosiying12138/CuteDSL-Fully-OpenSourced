add_mlir_dialect_library(SelfCuteKernelDialect
  KernelDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/SelfCute/Kernel

  DEPENDS
  MLIRSelfCuteKernelOpsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
