//===- GoogleOps.cpp - Google dialect ops ----------------------*- C++ -*-===//
//
// Google Dialect Operations Implementation
//
//===----------------------------------------------------------------------===//

#include "Google/IR/GoogleOps.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::google;

//===----------------------------------------------------------------------===//
// Google Dialect
//===----------------------------------------------------------------------===//

#include "Google/IR/GoogleOpsDialect.cpp.inc"

void GoogleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Google/IR/GoogleOps.cpp.inc"
      >();
}

#include "Google/IR/GoogleOpsAttributes.cpp.inc"
#include "Google/IR/GoogleOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Google Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Google/IR/GoogleOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstantOp Verification
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  auto valueAttr = getValue();
  auto outputType = getOutput().getType();
  
  // Check if value attribute type matches output type
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
    if (denseAttr.getType() != outputType) {
      return emitOpError("value attribute type ")
             << denseAttr.getType() << " does not match output type "
             << outputType;
    }
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Binary Operations Verification
//===----------------------------------------------------------------------===//

// Generic verifier for all binary operations
template<typename OpTy>
static LogicalResult verifyBinaryOp(OpTy op) {
  auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(op.getRhs().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
  
  if (!lhsType || !rhsType || !resultType) {
    return op.emitOpError("operands and result must be ranked tensors");
  }
  
  // Check shapes match
  if (lhsType.getShape() != rhsType.getShape()) {
    return op.emitOpError("operand shapes must match, got ")
           << lhsType << " and " << rhsType;
  }
  
  if (lhsType.getShape() != resultType.getShape()) {
    return op.emitOpError("result shape must match operand shapes, got ")
           << resultType << " for operands " << lhsType;
  }
  
  // Check element types match
  if (lhsType.getElementType() != rhsType.getElementType()) {
    return op.emitOpError("operand element types must match, got ")
           << lhsType.getElementType() << " and " << rhsType.getElementType();
  }
  
  return success();
}

LogicalResult AddOp::verify() { return verifyBinaryOp(*this); }
LogicalResult MaxOp::verify() { return verifyBinaryOp(*this); }
LogicalResult MinOp::verify() { return verifyBinaryOp(*this); }

//===----------------------------------------------------------------------===//
// ReduceOp Builder and Verification
//===----------------------------------------------------------------------===//

void ReduceOp::build(OpBuilder &builder, OperationState &state,
                     ReductionKind kind, Value input, ArrayRef<int64_t> axes,
                     bool keepdims, Type resultType) {
  state.addOperands(input);
  state.addAttribute("kind", builder.getI32IntegerAttr(static_cast<int32_t>(kind)));
  
  if (!axes.empty()) {
    state.addAttribute("axes", builder.getI64ArrayAttr(axes));
  }
  
  if (keepdims) {
    state.addAttribute("keepdims", builder.getBoolAttr(keepdims));
  }
  
  state.addTypes(resultType);
}

LogicalResult ReduceOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType) {
    return emitOpError("input must be a ranked tensor");
  }
    
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!outputType) {
    return emitOpError("output must be a ranked tensor");
  }
  
  int64_t inputRank = inputType.getRank();
  
  // Verify axes are valid
  if (auto axesAttr = getAxesAttr()) {
    llvm::SmallVector<int64_t> axesVec;
    for (auto axis : axesAttr.getAsValueRange<IntegerAttr>()) {
      int64_t axisVal = axis.getZExtValue();
      if (axisVal < 0 || axisVal >= inputRank) {
        return emitOpError("axis ") << axisVal << " is out of range [0, " 
                                     << inputRank << ")";
      }
      axesVec.push_back(axisVal);
    }
    
    // Check for duplicate axes
    llvm::SmallSet<int64_t, 4> uniqueAxes(axesVec.begin(), axesVec.end());
    if (uniqueAxes.size() != axesVec.size()) {
      return emitOpError("duplicate axis found in axes attribute");
    }
  }
  
  // For argmax/argmin, output element type should be integer
  auto kind = getKind();
  if (kind == ReductionKind::ARGMAX || kind == ReductionKind::ARGMIN) {
    if (!outputType.getElementType().isInteger(64)) {
      return emitOpError("argmax/argmin output must have i64 element type, got ")
             << outputType.getElementType();
    }
  } else {
    // For other reductions, element types should match
    if (inputType.getElementType() != outputType.getElementType()) {
      return emitOpError("input and output element types must match for non-arg reductions, got input ")
             << inputType.getElementType() << " and output " 
             << outputType.getElementType();
    }
  }
  
  // Verify output shape matches expected reduced shape
  llvm::SmallVector<int64_t> expectedShape;
  if (auto axesAttr = getAxesAttr()) {
    llvm::SmallSet<int64_t, 4> reducedAxes;
    for (auto axis : axesAttr.getAsValueRange<IntegerAttr>()) {
      reducedAxes.insert(axis.getZExtValue());
    }
    
    for (int64_t i = 0; i < inputRank; ++i) {
      if (reducedAxes.contains(i)) {
        if (getKeepdims()) {
          expectedShape.push_back(1);
        }
      } else {
        expectedShape.push_back(inputType.getDimSize(i));
      }
    }
  } else {
    // No axes specified - reduce all dimensions
    if (getKeepdims()) {
      expectedShape.assign(inputRank, 1);
    }
    // else expectedShape is empty (scalar result)
  }
  
  auto outputShape = outputType.getShape();
  if (outputShape.size() != expectedShape.size()) {
    return emitOpError("output rank does not match expected rank, expected ")
           << expectedShape.size() << " got " << outputShape.size();
  }
  
  for (size_t i = 0; i < expectedShape.size(); ++i) {
    if (expectedShape[i] != outputShape[i] && expectedShape[i] != ShapedType::kDynamic) {
      return emitOpError("output shape mismatch at dimension ") << i
             << ", expected " << expectedShape[i] << " got " << outputShape[i];
    }
  }
  
  return success();
}

SmallVector<Range> DequantOp::getIterationDomain(OpBuilder &b) {
  int64_t rank = getInput().getType().getRank();
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);

  SmallVector<OpFoldResult> sizes =
      tensor::getMixedSizes(b, getLoc(), getInput());

  SmallVector<Range> loopBounds(rank);
  for (auto dim : llvm::seq<int64_t>(rank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = sizes[dim];
    loopBounds[dim].stride = one;
  }

  return loopBounds;
}

SmallVector<utils::IteratorType> DequantOp::getLoopIteratorTypes() {
  int64_t rank = getInput().getType().getRank();
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

LogicalResult DequantOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets = llvm::to_vector(offsets);
  resultSizes = llvm::to_vector(sizes);
  return success();
}

FailureOr<TilingResult> DequantOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();
  int64_t rank = getInput().getType().getRank();
  SmallVector<OpFoldResult> strides(rank, b.getI64IntegerAttr(1));

  auto inputTile = b.create<tensor::ExtractSliceOp>(loc, getInput(), offsets,
                                                    sizes, strides);
  auto scaleTile = b.create<tensor::ExtractSliceOp>(loc, getScale(), offsets,
                                                    sizes, strides);

  Type resultType = inputTile.getResultType();

  Operation *tiledOp =
      mlir::clone(b, getOperation(), {resultType}, {inputTile, scaleTile});

  return TilingResult{{tiledOp},
                      SmallVector<Value>(tiledOp->getResults()),
                      {inputTile, scaleTile}};
}

LogicalResult DequantOp::getIterationDomainTileFromResultTile(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
    SmallVectorImpl<OpFoldResult> &iterDomainSizes) {
  iterDomainOffsets = llvm::to_vector(offsets);
  iterDomainSizes = llvm::to_vector(sizes);
  return success();
}

FailureOr<TilingResult> DequantOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
  if (failed(getIterationDomainTileFromResultTile(
          b, resultNumber, offsets, sizes, mappedOffsets, mappedSizes))) {
    return failure();
  }
  return getTiledImplementation(b, mappedOffsets, mappedSizes);
}
