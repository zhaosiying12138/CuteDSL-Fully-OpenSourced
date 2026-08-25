add_mlir_dialect_library(SelfCuteSM120Dialect
  SM120Dialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/SelfCute/SM120

  DEPENDS
  MLIRSelfCuteSM120OpsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
