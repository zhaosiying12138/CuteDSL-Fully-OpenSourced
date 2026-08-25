add_mlir_dialect_library(SelfCutePipelineDialect
  PipelineDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/SelfCute/Pipeline

  DEPENDS
  MLIRSelfCutePipelineOpsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
