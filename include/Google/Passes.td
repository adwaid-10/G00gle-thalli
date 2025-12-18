#ifndef TILING_PASS_
#define TILING_PASS_

include "mlir/Pass/PassBase.td"

def GoogleTileAndFuse : Pass<"google-tile-and-fuse", "func::FuncOp"> {
  let summary = "";
  let dependentDialects = ["linalg::LinalgDialect", "scf::SCFDialect"];
  let options = [
    Option<"tilingLevel", "tiling-level", "google::TilingLevel",
    "/*default=*/google::TilingLevel::Parallel",
    "",
    [{llvm::cl::values(
      clEnumValN(google::TilingLevel::Parallel, "parallel", "scf.forall"),
      clEnumValN(google::TilingLevel::Reduction, "Reduction", "scf.for")
    )}]>
  ];
}

#endif