add_mlir_library(SelfCuteRegistration
  RegisterSelfCute.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/SelfCute/Registration

  LINK_LIBS PUBLIC
  SelfCuteKernelDialect
  SelfCutePipelineDialect
  SelfCuteSM120Dialect
  MLIRIR
)
