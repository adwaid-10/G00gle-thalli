//===- GoogleToLinalg.cpp - Google to Linalg conversion --------*- C++ -*-===//
//
// This file implements the lowering of Google dialect operations to
// Linalg, Tensor, Arith, and Math dialects following a Linalg-first strategy.
//
//===----------------------------------------------------------------------===//

#include "Google/Translation/GoogleToLinalg.h"
#include "Google/IR/GoogleOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace google {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Create an identity affine map for the given number of dimensions
static AffineMap getIdentityMap(unsigned numDims, MLIRContext *context) {
  return AffineMap::getMultiDimIdentityMap(numDims, context);
}

/// Create a linalg.generic operation for elementwise operations
static Value createElementwiseLinalgGeneric(
    Location loc, ValueRange inputs, Value output,
    function_ref<Value(OpBuilder &, Location, ValueRange)> bodyBuilder,
    OpBuilder &rewriter) {
  
  auto outputType = cast<RankedTensorType>(output.getType());
  unsigned rank = outputType.getRank();
  
  // Create identity affine maps for all inputs and output
  SmallVector<AffineMap> indexingMaps;
  auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
  for (unsigned i = 0; i < inputs.size(); ++i) {
    indexingMaps.push_back(identityMap);
  }
  indexingMaps.push_back(identityMap); // for output
  
  // All dimensions are parallel for elementwise ops
  SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
  
  // Create linalg.generic operation
  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc,
      /*resultTensorTypes=*/TypeRange{outputType},
      /*inputs=*/inputs,
      /*outputs=*/ValueRange{output},
      /*indexingMaps=*/indexingMaps,
      /*iteratorTypes=*/iteratorTypes,
      /*bodyBuild=*/[&](OpBuilder &b, Location loc, ValueRange blockArgs) {
        // blockArgs = [input0, input1, ..., output_init]
        // We only pass inputs to the bodyBuilder
        ValueRange inputArgs = blockArgs.drop_back(1);
        Value result = bodyBuilder(b, loc, inputArgs);
        b.create<linalg::YieldOp>(loc, result);
      });
  
  return genericOp.getResult(0);
}

/// Create an empty tensor for output
static Value createEmptyTensor(Location loc, RankedTensorType type,
                                OpBuilder &rewriter) {
  SmallVector<Value> dynamicDims;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    if (type.isDynamicDim(i)) {
      // Would need to extract dynamic dimensions from input
      // For now, assume static shapes
    }
  }
  return rewriter.create<tensor::EmptyOp>(loc, type.getShape(),
                                           type.getElementType(), dynamicDims);
}

//===----------------------------------------------------------------------===//
// Binary Operations Lowering
//===----------------------------------------------------------------------===//

// Add operation - inline implementation for debugging
struct AddOpLowering : public OpConversionPattern<AddOp> {
  using OpConversionPattern<AddOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    // Create empty output tensor
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    // Create identity affine maps
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap); // lhs
    maps.push_back(identityMap); // rhs
    maps.push_back(identityMap); // output
    
    // Create iterator types (all parallel)
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    // Create linalg.generic
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{resultType},
        /*inputs=*/ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        /*outputs=*/ValueRange{init},
        /*indexingMaps=*/maps,
        /*iteratorTypes=*/iterTypes,
        /*bodyBuild=*/[&](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = lhs element, args[1] = rhs element, args[2] = output element
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Sub operation
struct SubOpLowering : public OpConversionPattern<SubOp> {
  using OpConversionPattern<SubOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      SubOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::SubFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Mul operation
struct MulOpLowering : public OpConversionPattern<MulOp> {
  using OpConversionPattern<MulOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::MulFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Div operation
struct DivOpLowering : public OpConversionPattern<DivOp> {
  using OpConversionPattern<DivOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      DivOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::DivFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Pow operation
struct PowOpLowering : public OpConversionPattern<PowOp> {
  using OpConversionPattern<PowOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      PowOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<math::PowFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Max operation
struct MaxOpLowering : public OpConversionPattern<MaxOp> {
  using OpConversionPattern<MaxOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      MaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::MaximumFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Min operation
struct MinOpLowering : public OpConversionPattern<MinOp> {
  using OpConversionPattern<MinOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      MinOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::MinimumFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Unary Operations Lowering
//===----------------------------------------------------------------------===//

// Neg operation
struct NegOpLowering : public OpConversionPattern<NegOp> {
  using OpConversionPattern<NegOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      NegOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap); // input
    maps.push_back(identityMap); // output
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getInput()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::NegFOp>(loc, args[0]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Abs operation
struct AbsOpLowering : public OpConversionPattern<AbsOp> {
  using OpConversionPattern<AbsOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      AbsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getInput()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<math::AbsFOp>(loc, args[0]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Sqrt operation
struct SqrtOpLowering : public OpConversionPattern<SqrtOp> {
  using OpConversionPattern<SqrtOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      SqrtOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getInput()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<math::SqrtOp>(loc, args[0]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Rsqrt operation
struct RsqrtOpLowering : public OpConversionPattern<RsqrtOp> {
  using OpConversionPattern<RsqrtOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      RsqrtOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getInput()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<math::RsqrtOp>(loc, args[0]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Exp operation
struct ExpOpLowering : public OpConversionPattern<ExpOp> {
  using OpConversionPattern<ExpOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      ExpOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getInput()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<math::ExpOp>(loc, args[0]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Log operation
struct LogOpLowering : public OpConversionPattern<LogOp> {
  using OpConversionPattern<LogOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      LogOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getInput()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<math::LogOp>(loc, args[0]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Ceil operation
struct CeilOpLowering : public OpConversionPattern<CeilOp> {
  using OpConversionPattern<CeilOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      CeilOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getInput()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<math::CeilOp>(loc, args[0]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Tanh operation
struct TanhOpLowering : public OpConversionPattern<TanhOp> {
  using OpConversionPattern<TanhOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      TanhOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getInput()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<math::TanhOp>(loc, args[0]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Activation Functions Lowering
//===----------------------------------------------------------------------===//

// ReLU: max(x, 0)
struct ReluOpLowering : public OpConversionPattern<ReluOp> {
  using OpConversionPattern<ReluOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      ReluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getInput()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto elementType = resultType.getElementType();
          Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(elementType));
          Value result = b.create<arith::MaximumFOp>(loc, args[0], zero);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Sigmoid: 1 / (1 + exp(-x))
struct SigmoidOpLowering : public OpConversionPattern<SigmoidOp> {
  using OpConversionPattern<SigmoidOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      SigmoidOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getInput()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto elementType = resultType.getElementType();
          Value one = b.create<arith::ConstantOp>(loc, b.getFloatAttr(elementType, 1.0));
          Value negX = b.create<arith::NegFOp>(loc, args[0]);
          Value expNegX = b.create<math::ExpOp>(loc, negX);
          Value onePlusExp = b.create<arith::AddFOp>(loc, one, expNegX);
          Value result = b.create<arith::DivFOp>(loc, one, onePlusExp);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
struct GeluOpLowering : public OpConversionPattern<GeluOp> {
  using OpConversionPattern<GeluOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      GeluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getInput()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto elementType = resultType.getElementType();
          Value half = b.create<arith::ConstantOp>(loc, b.getFloatAttr(elementType, 0.5));
          Value one = b.create<arith::ConstantOp>(loc, b.getFloatAttr(elementType, 1.0));
          Value sqrt2 = b.create<arith::ConstantOp>(loc, b.getFloatAttr(elementType, 1.4142135623730951));
          
          Value xDivSqrt2 = b.create<arith::DivFOp>(loc, args[0], sqrt2);
          Value erf = b.create<math::ErfOp>(loc, xDivSqrt2);
          Value onePlusErf = b.create<arith::AddFOp>(loc, one, erf);
          Value halfOnePlusErf = b.create<arith::MulFOp>(loc, half, onePlusErf);
          Value result = b.create<arith::MulFOp>(loc, args[0], halfOnePlusErf);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Softmax: Use linalg.softmax (named operation!)
struct SoftmaxOpLowering : public OpConversionPattern<SoftmaxOp> {
  using OpConversionPattern<SoftmaxOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      SoftmaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t axis = op.getAxis();
    
    Value init = createEmptyTensor(loc, resultType, rewriter);
    
    // Create linalg.softmax operation
    auto softmaxOp = rewriter.create<linalg::SoftmaxOp>(
        loc, resultType, adaptor.getInput(), init, axis);
    
    rewriter.replaceOp(op, softmaxOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MatMul Lowering
//===----------------------------------------------------------------------===//

struct MatMulOpLowering : public OpConversionPattern<MatMulOp> {
  using OpConversionPattern<MatMulOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      MatMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    
    Value init = createEmptyTensor(loc, resultType, rewriter);
    
    // Create linalg.matmul operation
    auto matmulOp = rewriter.create<linalg::MatmulOp>(
        loc, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{init});
    
    rewriter.replaceOp(op, matmulOp.getResults()[0]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Reduce Lowering
//===----------------------------------------------------------------------===//

struct ReduceOpLowering : public OpConversionPattern<ReduceOp> {
  using OpConversionPattern<ReduceOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      ReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    auto kind = op.getKind();
    
    Value init = createEmptyTensor(loc, resultType, rewriter);
    
    // Get reduction dimensions
    SmallVector<int64_t> dimensions;
    if (auto axesAttr = op.getAxesAttr()) {
      for (auto axis : axesAttr.getAsValueRange<IntegerAttr>()) {
        dimensions.push_back(axis.getZExtValue());
      }
    } else {
      // Reduce all dimensions
      auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
      for (int64_t i = 0; i < inputType.getRank(); ++i) {
        dimensions.push_back(i);
      }
    }
    
    // Create linalg.reduce operation with appropriate combiner
    auto reduceOp = rewriter.create<linalg::ReduceOp>(
        loc, adaptor.getInput(), init, dimensions,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result;
          switch (kind) {
            case ReductionKind::MAX:
              result = b.create<arith::MaximumFOp>(loc, args[0], args[1]);
              break;
            case ReductionKind::MIN:
              result = b.create<arith::MinimumFOp>(loc, args[0], args[1]);
              break;
            case ReductionKind::SUM:
              result = b.create<arith::AddFOp>(loc, args[0], args[1]);
              break;
            case ReductionKind::PRODUCT:
              result = b.create<arith::MulFOp>(loc, args[0], args[1]);
              break;
            case ReductionKind::MEAN:
              // For mean, we do sum first, then divide by count later
              result = b.create<arith::AddFOp>(loc, args[0], args[1]);
              break;
            case ReductionKind::ARGMAX:
            case ReductionKind::ARGMIN:
              // TODO: Implement argmax/argmin with custom combiner
              // For now, just use max as placeholder
              result = b.create<arith::MaximumFOp>(loc, args[0], args[1]);
              break;
          }
          b.create<linalg::YieldOp>(loc, result);
        });
    
    Value finalResult = reduceOp.getResults()[0];
    
    // For MEAN, divide by the number of elements reduced
    if (kind == ReductionKind::MEAN) {
      auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
      int64_t numElements = 1;
      for (int64_t dim : dimensions) {
        numElements *= inputType.getDimSize(dim);
      }
      
      auto elementType = resultType.getElementType();
      Value count = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(elementType, static_cast<double>(numElements)));
      
      // Divide each element by count
      Value initDiv = createEmptyTensor(loc, resultType, rewriter);
      finalResult = createElementwiseLinalgGeneric(
          loc, ValueRange{finalResult}, initDiv,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            return b.create<arith::DivFOp>(loc, args[0], count);
          },
          rewriter);
    }
    
    rewriter.replaceOp(op, finalResult);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Shape Operations Lowering
//===----------------------------------------------------------------------===//

struct ReshapeOpLowering : public OpConversionPattern<ReshapeOp> {
  using OpConversionPattern<ReshapeOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      ReshapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    
    // Get target shape from attribute
    auto shapeAttr = op.getShape();
    SmallVector<int64_t> targetShape;
    for (auto dim : shapeAttr.getAsValueRange<IntegerAttr>()) {
      targetShape.push_back(dim.getZExtValue());
    }
    
    // Create shape tensor as 1D tensor of indices
    auto shapeType = RankedTensorType::get({static_cast<int64_t>(targetShape.size())}, 
                                           rewriter.getIndexType());
    
    SmallVector<Value> shapeValues;
    for (int64_t dim : targetShape) {
      shapeValues.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
    }
    
    Value shapeTensor = rewriter.create<tensor::FromElementsOp>(loc, shapeType, shapeValues);
    
    // Use tensor.reshape with shape operand
    auto reshapeOp = rewriter.create<tensor::ReshapeOp>(
        loc, resultType, adaptor.getInput(), shapeTensor);
    
    rewriter.replaceOp(op, reshapeOp.getResult());
    return success();
  }
};

struct TransposeOpLowering : public OpConversionPattern<TransposeOp> {
  using OpConversionPattern<TransposeOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      TransposeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    
    Value init = createEmptyTensor(loc, resultType, rewriter);
    
    // Get permutation
    SmallVector<int64_t> perm;
    for (auto p : op.getPerm().getAsValueRange<IntegerAttr>()) {
      perm.push_back(p.getZExtValue());
    }
    
    // Create linalg.transpose
    auto transposeOp = rewriter.create<linalg::TransposeOp>(
        loc, adaptor.getInput(), init, perm);
    
    rewriter.replaceOp(op, transposeOp.getResults()[0]);
    return success();
  }
};

struct ConcatOpLowering : public OpConversionPattern<ConcatOp> {
  using OpConversionPattern<ConcatOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      ConcatOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t axis = op.getAxis();
    
    // Use tensor.concat
    auto concatOp = rewriter.create<tensor::ConcatOp>(
        op.getLoc(), resultType, axis, adaptor.getInputs());
    
    rewriter.replaceOp(op, concatOp.getResult());
    return success();
  }
};

struct SliceOpLowering : public OpConversionPattern<SliceOp> {
  using OpConversionPattern<SliceOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      SliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    
    // Convert start/end arrays to OpFoldResult
    SmallVector<OpFoldResult> offsets, sizes, strides;
    auto startAttr = op.getStart();
    auto endAttr = op.getEnd();
    
    int i = 0;
    for (auto [s, e] : llvm::zip(startAttr.getAsValueRange<IntegerAttr>(),
                                   endAttr.getAsValueRange<IntegerAttr>())) {
      int64_t start = s.getZExtValue();
      int64_t end = e.getZExtValue();
      offsets.push_back(rewriter.getIndexAttr(start));
      sizes.push_back(rewriter.getIndexAttr(end - start));
      strides.push_back(rewriter.getIndexAttr(1));
      i++;
    }
    
    // Use tensor.extract_slice
    auto extractOp = rewriter.create<tensor::ExtractSliceOp>(
        loc, resultType, adaptor.getInput(), offsets, sizes, strides);
    
    rewriter.replaceOp(op, extractOp.getResult());
    return success();
  }
};

struct BroadcastOpLowering : public OpConversionPattern<BroadcastOp> {
  using OpConversionPattern<BroadcastOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      BroadcastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    
    Value init = createEmptyTensor(loc, resultType, rewriter);
    
    // Create linalg.broadcast
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    SmallVector<int64_t> dimensions;
    
    // Determine which dimensions to broadcast
    int64_t inputRank = inputType.getRank();
    int64_t outputRank = resultType.getRank();
    for (int64_t i = 0; i < outputRank - inputRank; ++i) {
      dimensions.push_back(i);
    }
    
    auto broadcastOp = rewriter.create<linalg::BroadcastOp>(
        loc, adaptor.getInput(), init, dimensions);
    
    rewriter.replaceOp(op, broadcastOp.getResults()[0]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Selection Operations Lowering
//===----------------------------------------------------------------------===// Select operation
struct SelectOpLowering : public OpConversionPattern<SelectOp> {
  using OpConversionPattern<SelectOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      SelectOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap); // condition
    maps.push_back(identityMap); // true_value
    maps.push_back(identityMap); // false_value
    maps.push_back(identityMap); // output
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType},
        ValueRange{adaptor.getCondition(), adaptor.getTrueVal(), adaptor.getFalseVal()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::SelectOp>(loc, args[0], args[1], args[2]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Clamp operation: clamp(x, min, max) = min(max(x, min), max)
struct ClampOpLowering : public OpConversionPattern<ClampOp> {
  using OpConversionPattern<ClampOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      ClampOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap); // input
    maps.push_back(identityMap); // min
    maps.push_back(identityMap); // max
    maps.push_back(identityMap); // output
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType},
        ValueRange{adaptor.getInput(), adaptor.getMin(), adaptor.getMax()},
        ValueRange{init}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value clamped = b.create<arith::MaximumFOp>(loc, args[0], args[1]);
          Value result = b.create<arith::MinimumFOp>(loc, clamped, args[2]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Specialized Operations Lowering
//===----------------------------------------------------------------------===//

// Dequantization: convert quantized int to float and scale
struct DequantOpLowering : public OpConversionPattern<DequantOp> {
  using OpConversionPattern<DequantOp>::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      DequantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    unsigned rank = resultType.getRank();
    
    // Step 1: Convert int to float
    Value init1 = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps;
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    maps.push_back(identityMap);
    maps.push_back(identityMap);
    
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    
    // Convert int to float
    auto convertOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getInput()},
        ValueRange{init1}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value floatVal = b.create<arith::SIToFPOp>(loc, resultType.getElementType(), args[0]);
          b.create<linalg::YieldOp>(loc, floatVal);
        });
    
    // Step 2: Multiply by scale (scale is a tensor)
    Value init2 = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    
    SmallVector<AffineMap> maps2;
    maps2.push_back(identityMap); // converted float
    maps2.push_back(identityMap); // scale
    maps2.push_back(identityMap); // output
    
    auto scaleOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, 
        ValueRange{convertOp.getResult(0), adaptor.getScale()},
        ValueRange{init2}, maps2, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::MulFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        });
    
    rewriter.replaceOp(op, scaleOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct GoogleToLinalgLoweringPass
    : public PassWrapper<GoogleToLinalgLoweringPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GoogleToLinalgLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect, math::MathDialect,
                    func::FuncDialect>();
  }

  StringRef getArgument() const final { return "convert-google-to-linalg"; }
  
  StringRef getDescription() const final {
    return "Lower Google dialect operations to Linalg + Arith/Math/Tensor";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    ConversionTarget target(getContext());
    
    // Mark target dialects as legal
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect,
                           arith::ArithDialect, math::MathDialect,
                           func::FuncDialect>();
    
    // Mark specific Google ops as illegal (to be converted)
    // Keep ConstantOp legal for now as it's used internally
    target.addIllegalOp<AddOp, SubOp, MulOp, DivOp, PowOp, MaxOp, MinOp>();
    target.addIllegalOp<NegOp, AbsOp, SqrtOp, RsqrtOp, ExpOp, LogOp, CeilOp>();
    target.addIllegalOp<ReluOp, SigmoidOp, GeluOp, TanhOp, SoftmaxOp>();
    target.addIllegalOp<MatMulOp, ReduceOp>();
    target.addIllegalOp<ReshapeOp, TransposeOp, ConcatOp, SliceOp, BroadcastOp>();
    target.addIllegalOp<SelectOp, ClampOp>();
    target.addIllegalOp<DequantOp>();
    
    // Allow ConstantOp and unknown ops to pass through
    target.addLegalOp<ConstantOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(&getContext());
    populateGoogleToLinalgConversionPatterns(patterns);
    
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateGoogleToLinalgConversionPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  
  // Binary operations
  patterns.add<AddOpLowering, SubOpLowering, MulOpLowering, DivOpLowering,
               PowOpLowering, MaxOpLowering, MinOpLowering>(context);
  
  // Unary operations
  patterns.add<NegOpLowering, AbsOpLowering, SqrtOpLowering, RsqrtOpLowering,
               ExpOpLowering, LogOpLowering, CeilOpLowering>(context);
  
  // Activations
  patterns.add<ReluOpLowering, SigmoidOpLowering, GeluOpLowering,
               TanhOpLowering, SoftmaxOpLowering>(context);
  
  // Structured operations
  patterns.add<MatMulOpLowering, ReduceOpLowering>(context);
  
  // Shape operations
  patterns.add<ReshapeOpLowering, TransposeOpLowering, ConcatOpLowering,
               SliceOpLowering, BroadcastOpLowering>(context);
  
  // Selection operations
  patterns.add<SelectOpLowering, ClampOpLowering>(context);
  
  // Specialized operations
  patterns.add<DequantOpLowering>(context);
}

std::unique_ptr<Pass> createGoogleToLinalgLoweringPass() {
  return std::make_unique<GoogleToLinalgLoweringPass>();
}

void registerGoogleToLinalgLoweringPass() {
  PassRegistration<GoogleToLinalgLoweringPass>();
}

} // namespace google
} // namespace mlir
