#ifndef GOOGLETOTOSA
#define GOOGLETOTOSA

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
    class Pass;
    class RewritePatternSet;
    class TypeConverter;

    namespace google {
        std::unique_ptr<Pass> translationtoTosa ();
        void registertranslationtoTosa ();
        void populatewithconversionpatterns (RewritePatternSet &patterns);
    }
}

#endif 