#ifndef GOOGLE_COMPILER_PASSES_H_
#define GOOGLE_COMPILER_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir::google {

enum class TilingLevel { Parallel, Reduction };

#define GEN_PASS_DECL
#include "Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

}  // namespace mlir::google

#endif  // GOOGLE_COMPILER_PASSES_H_