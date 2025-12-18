#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"

// Assuming these headers exist in your project structure
#include "Google/Translation/GoogleToArith.h"
#include "Google/IR/GoogleOps.h"

// namespace mlir {
// namespace google {

// //--- Constant Operation Lowering (Idiomatic MLIR approach) ---//

// /**
//  * @brief Defines a pattern to convert google::ConstantOp to arith::ConstantOp.
//  * 
//  * This is the preferred way in modern MLIR over complex templates 
//  * because it clearly specifies the source and target operations.
//  */
// struct GoogleConstantOpLowering 
//     : public OpConversionPattern<google::ConstantOp> {
//   using OpConversionPattern<google::ConstantOp>::OpConversionPattern;

//   LogicalResult matchAndRewrite(
//       google::ConstantOp op, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {
    
//     // google::ConstantOp value is already a TypedAttr via its definition.
//     // We can directly access the value attribute and the result type.
//     auto valueAttr = op.getValue();
//     auto resultType = op.getType();

//     // 2. Cast the generic Attribute to a TypedAttr, which arith::ConstantOp requires.
//     auto typedValueAttr = dyn_cast<TypedAttr>(valueAttr);

//     if (!typedValueAttr) {
//         return rewriter.notifyMatchFailure(
//             op, "Constant value attribute must be a TypedAttr");
//     }

    
//     // Replace the old google op with the new Arith op using the existing attributes.
//     rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, typedValueAttr);
    
//     return success();
//   }
// };


// // Pass Definition

// namespace {
// /**
//  * @brief The main pass structure, defined using the modern PassWrapper.
//  */
// struct GoogleToArithLoweringPass
//     : public PassWrapper<GoogleToArithLoweringPass, OperationPass<ModuleOp>> {
  
//   // Use the modern type ID mechanism.
//   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GoogleToArithLoweringPass)

//   void getDependentDialects(DialectRegistry &registry) const override {
//     // This pass needs arith and func dialects available in the context.
//     registry.insert<arith::ArithDialect, func::FuncDialect>();
//   }

//   StringRef getArgument() const final { return "convert-google-to-arith"; }
  
//   StringRef getDescription() const final {
//     return "Lower google dialect operations to Arith dialect";
//   }

//   void runOnOperation() override {
//     ModuleOp module = getOperation();
    
//     // The ConversionTarget dictates which ops are legal after conversion.
//     ConversionTarget target(getContext());
//     target.addLegalDialect<arith::ArithDialect, func::FuncDialect>();
    
//     // Mark the source google ops as illegal.
//     target.addIllegalOp<google::ConstantOp>();
//     // If you add google.add, google.sub later, you add them here too.

//     // Allow ops not in Arith/Func to pass through if no pattern matches them (e.g. standard control flow).
//     target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });


//     // Collect the patterns needed for this conversion.
//     RewritePatternSet patterns(&getContext());
//     populateGoogleToArithConversionPatterns(patterns);
    
//     // Apply the conversion to the module.
//     if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
//       signalPassFailure();
//       return;
//     }
//   }
// };

// } // namespace

// //--- Registration and Population Functions ---//

// void populateGoogleToArithConversionPatterns(RewritePatternSet &patterns) {
//   // Add the specific lowering pattern we defined above to the set.
//   patterns.add<GoogleConstantOpLowering>(patterns.getContext());
// }

// std::unique_ptr<Pass> createGoogleToArithLoweringPass() {
//   return std::make_unique<GoogleToArithLoweringPass>();
// }

// // Register the pass so it can be used by mlir-opt or in C++ code via the PassManager.
// void registerGoogleToArithLoweringPass() {
//   // PassRegistration takes the Pass class as its template argument.
//   PassRegistration<GoogleToArithLoweringPass>();
// }

// } // namespace google
// } // namespace mlir



//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


namespace mlir {
namespace google {

struct Googletoarith {
    template <typename OpTy>
    static Value MapOp(OpTy op, Type resulttype, ValueRange input, OpBuilder* builder){
        return mappingarith(op, resulttype, input, builder);
    }

    private : 
    template <typename opTy>
    static Value mappingarith(opTy op, Type resulttype, ValueRange input, OpBuilder* builder){
        return nullptr;
    }

    static Value mappingarith(google::ConstantOp op, Type resulttype, ValueRange input, OpBuilder* builder){
        auto typedValueAttr = dyn_cast<TypedAttr>(op.getValue());
        return builder -> create<arith::ConstantOp>(op.getLoc(), resulttype, typedValueAttr);
    }
};

template <typename operation>
class ToAirthconversionpattern : public OpConversionPattern<operation>{
    public : 
    using OpConversionPattern<operation> :: OpConversionPattern;
    using opAdaptor = typename operation::Adaptor;

     LogicalResult matchAndRewrite(operation op,opAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    ValueRange operands = adaptor.getOperands();

    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if(!resultType)
    return rewriter.notifyMatchFailure(op,"Needs Tensor Type");

    Value result = Googletoarith::MapOp(op,resultType,operands,&rewriter);
    if (!result)
      return rewriter.notifyMatchFailure(op, "Mapping to Arith failed or returned null");
    rewriter.replaceOp(op,result);
    return success();
   }
};

namespace {
struct GoogleToArithLoweringPass
    : public PassWrapper<GoogleToArithLoweringPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GoogleToArithLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
  }

  StringRef getArgument() const final { return "convert-google-to-arith"; }
  
  StringRef getDescription() const final {
    return "Lower Google dialect operations to Arith dialect";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    ConversionTarget target(getContext());
    
    target.addLegalDialect<arith::ArithDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalOp<google::ConstantOp>();

    RewritePatternSet patterns(&getContext());

    
    populateGoogleToArithConversionPatterns(patterns);
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

void populateGoogleToArithConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ToAirthconversionpattern<google::ConstantOp>>(
    patterns.getContext()
  );
}

std::unique_ptr<Pass> createGoogleToArithLoweringPass() {
  return std::make_unique<GoogleToArithLoweringPass>();
}

// Register the pass
void registerGoogleToArithLoweringPass() {
  PassRegistration<GoogleToArithLoweringPass>();
}

}

}