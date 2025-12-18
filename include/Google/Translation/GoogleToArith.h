#ifndef MLIR_CONVERSION_GOOGLETOARITH_H
#define MLIR_CONVERSION_GOOGLETOARITH_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

    class Pass;
    class RewritePatternSet;
    class TypeConverter;

    
namespace google {

std::unique_ptr<Pass> createGoogleToArithLoweringPass();

void registerGoogleToArithLoweringPass();

void populateGoogleToArithConversionPatterns(RewritePatternSet &patterns);

} // namespace google
}

#endif // MLIR_CONVERSION_GOOGLETOARITH_H

