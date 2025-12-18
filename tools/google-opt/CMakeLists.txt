get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(google-opt
  google-opt.cpp
)

llvm_update_compile_flags(google-opt)

target_link_libraries(google-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRGoogleOps
  MLIRGoogleToArithTransforms 
  MLIROptLib
)