
add_mlir_library(MLIRGoogleToArithTransforms
  GoogleToArith.cpp
  GoogleToArith.h
  GoogletoTosa.cpp
  GoogletoTosa.h

  DEPENDS
  MLIRArithDialect

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRTensorDialect
  MLIRTosaDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
)

target_include_directories(MLIRGoogleToArithTransforms PRIVATE
  ${PROJECT_SOURCE_DIR}/include/Google/Translation
)