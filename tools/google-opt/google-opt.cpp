// google-opt.cpp - 
#include "Google/IR/GoogleOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "Google/Translation/GoogleToArith.h"
#include "Google/Translation/GoogletoTosa.h"

// namespace mlir {
// namespace google {
// #define GEN_PASS_REGISTRATION
// #include "Compiler/Transforms/Passes.h.inc"
// } 
// }

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  
  mlir::DialectRegistry registry;
  registry.insert<mlir::google::GoogleDialect>();
  mlir::registerAllDialects(registry);

  mlir::google::registerGoogleToArithLoweringPass();
  mlir::google::registertranslationtoTosa();

  
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Google dialect optimizer\n", registry));
}