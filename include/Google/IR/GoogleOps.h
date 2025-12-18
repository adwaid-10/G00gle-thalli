//===- GoogleOps.h - Google dialect ops ------------------------*- C++ -*-===//
//
// Google Dialect Operations Header
//
//===----------------------------------------------------------------------===//

#ifndef GOOGLE_IR_GOOGLEOPS_H
#define GOOGLE_IR_GOOGLEOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"


//===----------------------------------------------------------------------===//
// Google Dialect Declaration
//===----------------------------------------------------------------------===//

#include "Google/IR/GoogleOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Google Enums
//===----------------------------------------------------------------------===//

#include "Google/IR/GoogleOpsEnums.h.inc"

//===----------------------------------------------------------------------===//
// Google Attributes
//===----------------------------------------------------------------------===//

#include "Google/IR/GoogleOpsAttributes.h.inc"

//===----------------------------------------------------------------------===//
// Google Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Google/IR/GoogleOps.h.inc"

#endif // GOOGLE_IR_GOOGLEOPS_H