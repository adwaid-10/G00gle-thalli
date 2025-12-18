#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"

#include "Google/Translation/GoogletoTosa.h"
#include "Google/IR/GoogleOps.h"

namespace mlir {
    namespace google {
        struct Helper {
            template <typename opty>
            static Value Mapop(opty op, Type resulttype, ValueRange input, OpBuilder *builder) {
                return mappingtosa(op, resulttype, input, builder);
            }

            private :
            template <typename opty>
            static Value mappingtosa(opty op, Type resulttype, ValueRange input, OpBuilder *builder) {
                return nullptr;
            }

            static Value mappingtosa(google::MaxOp op, Type resulttype, ValueRange input, OpBuilder *builder) {
                auto resultype = dyn_cast<mlir::TensorType>(resulttype);
                // auto x = builder->create<tosa::CastOp>(op.getLoc(), resultype, input[0]);
                // auto y = builder->create<tosa::CastOp>(op.getLoc(), resultype, input[1]);

                return builder->create<tosa::MaximumOp>(op.getLoc(),resulttype, input[0], input[1]);
            }

            static Value mappingtosa(google::MinOp op, Type resulttype, ValueRange input, OpBuilder *builder) {
                auto resultype = dyn_cast<mlir::TensorType>(resulttype);
                // auto x = builder->create<tosa::CastOp>(op.getLoc(), resultype, input[0]);
                // auto y = builder->create<tosa::CastOp>(op.getLoc(), resultype, input[1]);

                return builder->create<tosa::MinimumOp>(op.getLoc(),resulttype, input[0], input[1]);
            }
            
        };


        template <typename operation>
        class GoogletoTosaLow : public OpConversionPattern<operation> {
            public : 
            using OpConversionPattern<operation>::OpConversionPattern;
            using OpAdaptor = typename operation::Adaptor;
            LogicalResult matchAndRewrite(operation op, OpAdaptor adaptor, ConversionPatternRewriter &rewrite) const override {
                ValueRange operands = adaptor.getOperands();

                if (operands.empty()) {
                    return rewrite.notifyMatchFailure(op, " expected tosa lower ops");
                }

                auto resulttype = op.getResult().getType();

                Value result = Helper::Mapop(op, resulttype, operands, &rewrite);

                if (!result){
                    return rewrite.notifyMatchFailure(op, "failed to map to TOSA operation");
                }

                rewrite.replaceOp(op, result);
                return success();

            }
        };

        namespace {
            struct TranslatetoTOSA : public PassWrapper<TranslatetoTOSA, OperationPass<ModuleOp>> {
                MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TranslatetoTOSA)

                void getDependentDialects(DialectRegistry &registry) const override {
                    registry.insert<tosa::TosaDialect>();
                    registry.insert<func::FuncDialect>();
                    registry.insert<google::GoogleDialect>();
                }

                StringRef getArgument() const final{ return "convert-google-to-tosa";}
                StringRef getDescription() const final {
                    return "translate google to TOSA";
                }

                void runOnOperation() override {
                    ModuleOp module = getOperation();
                    ConversionTarget target(getContext());

                    target.addLegalDialect<tosa::TosaDialect, func::FuncDialect>();
                    target.addIllegalOp<google::MaxOp>();
                    target.addIllegalOp<google::MinOp>();
                    // target.addIllegalOp<nova::AndOp>();
                    // target.addIllegalOp<nova::OrOp>();
                    // target.addIllegalOp<nova::XorOp>();
                    // target.addIllegalOp<nova::NegOp>();
                    // target.addIllegalOp<nova::NotOp>();
                    target.markUnknownOpDynamicallyLegal([](Operation *) { return "true";});
                    
                    // TypeConverter typeConverter;
                    // typeConverter.addConversion([](Type type) { return type; });
                    RewritePatternSet patterns(&getContext());
                    populatewithconversionpatterns(patterns);

                    if(failed(applyPartialConversion(module, target, std::move(patterns)))){
                        signalPassFailure();
                        return;
                    }
                }
            };
        }

        void populatewithconversionpatterns (RewritePatternSet &patterns){           
            patterns.add<GoogletoTosaLow<google::MaxOp>,
                        GoogletoTosaLow<google::MinOp>>(
                patterns.getContext()
            );
        }

        std::unique_ptr<Pass> translationtoTosa() {
            return std::make_unique<TranslatetoTOSA>();
        }

        void registertranslationtoTosa(){
            PassRegistration<TranslatetoTOSA>();
        }
    }
}
