add_mlir_dialect_library(MLIRGoogleOps
  GoogleOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Google/IR

  DEPENDS
  MLIRGoogleOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)