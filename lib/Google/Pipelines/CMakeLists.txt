add_mlir_library(MLIRGooglePipelines
  Pipelines.cpp

  LINK_LIBS PUBLIC
  MLIRGoogleOps
  MLIRGoogleToArithTransforms
  MLIRAffineTransforms
  MLIRArithTransforms
  MLIRBufferizationTransforms
  MLIRBufferizationToMemRef
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIRSCFTransforms
  MLIRTransformDialect
  MLIRPass
  MLIRTransforms
)
