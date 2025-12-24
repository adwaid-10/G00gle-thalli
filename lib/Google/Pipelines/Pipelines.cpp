//===- Pipelines.cpp - Google Optimization Pipelines ---------------------===//
//
// This file implements optimization pipeline configurations for the Google
// dialect using existing MLIR passes.
//
//===----------------------------------------------------------------------===//

#include "Google/Pipelines/Pipelines.h"
#include "Google/IR/GoogleOps.h"
#include "Google/Translation/GoogleToLinalg.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
namespace google {

//===----------------------------------------------------------------------===//
// Basic Pipeline - Fast Compilation
//===----------------------------------------------------------------------===//

void registerBasicPipeline() {
  PassPipelineRegistration<>(
    "google-basic-pipeline",
    "Basic Google optimization pipeline for fast compilation",
    [](OpPassManager &pm) {
      // Phase 1: Lower Google to Linalg
      pm.addPass(createGoogleToLinalgLoweringPass());
      
      // Phase 2: Basic fusion
      pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
      
      // Phase 3: Bufferization
      pm.addPass(bufferization::createOneShotBufferizePass());
      
      // Phase 4: Lower to loops
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
      
      // Phase 5: Lower to LLVM (using pass pipeline string)
      // Note: Some passes are better invoked via pipeline strings
      pm.addPass(createConvertFuncToLLVMPass());
      pm.addPass(createArithToLLVMConversionPass());
      pm.addPass(createFinalizeMemRefToLLVMConversionPass());
      pm.addPass(createReconcileUnrealizedCastsPass());
    });
}

//===----------------------------------------------------------------------===//
// Optimized Pipeline - Balanced Performance
//===----------------------------------------------------------------------===//

void registerOptimizedPipeline() {
  PassPipelineRegistration<>(
    "google-optimized-pipeline",
    "Optimized Google pipeline with fusion and affine optimizations",
    [](OpPassManager &pm) {
      // Phase 1: Lower Google to Linalg
      pm.addPass(createGoogleToLinalgLoweringPass());
      
      // Phase 2: Linalg optimizations
      pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
      pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
      
      // Phase 3: Bufferization
      pm.addPass(bufferization::createOneShotBufferizePass());
      
      // Phase 4: Lower to affine for better loop optimization
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToAffineLoopsPass());
      
      // Affine optimizations
      pm.addNestedPass<func::FuncOp>(affine::createLoopFusionPass());
      pm.addNestedPass<func::FuncOp>(affine::createLoopCoalescingPass());
      
      // Lower affine
      pm.addPass(createLowerAffinePass());
      
      // Phase 5: Lower to LLVM
      pm.addPass(createConvertFuncToLLVMPass());
      pm.addPass(createArithToLLVMConversionPass());
      pm.addPass(createFinalizeMemRefToLLVMConversionPass());
      pm.addPass(createReconcileUnrealizedCastsPass());
    });
}

//===----------------------------------------------------------------------===//
// Extreme Pipeline L1 - L1 Cache Tiling with Transform Dialect
//===----------------------------------------------------------------------===//

void registerExtremePipelineL1() {
  PassPipelineRegistration<>(
    "google-extreme-l1",
    "Extreme pipeline with L1 cache tiling (16x16x16) - MINIMAL for testing",
    [](OpPassManager &pm) {
      // Phase 1: Lower Google to Linalg
      pm.addPass(createGoogleToLinalgLoweringPass());
      
      // Phase 2: FUSION
      pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
      
      // Phase 3: TILING via Transform Dialect
      // Stop here to verify tiling works
      pm.addPass(mlir::transform::createInterpreterPass());
      
      // STOP HERE - no bufferization, no LLVM lowering
      // This allows us to see the tiled linalg operations
    });
}

//===----------------------------------------------------------------------===//
// Extreme Pipeline L2 - L1+L2 Cache Tiling with Transform Dialect
//===----------------------------------------------------------------------===//

void registerExtremePipelineL2() {
  PassPipelineRegistration<>(
    "google-extreme-l2",
    "Extreme pipeline with L1+L2 cache tiling (L2: 64x64x64, L1: 16x16x16)",
    [](OpPassManager &pm) {
      // Phase 1: Lower Google to Linalg
      pm.addPass(createGoogleToLinalgLoweringPass());
      
      // Phase 2: FUSION (critical - must happen before tiling)
      pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
      
      // Phase 3: L1+L2 TILING via Transform Dialect
      // The transform script should be embedded in the input MLIR file
      pm.addPass(mlir::transform::createInterpreterPass());
      
      // STOP HERE - minimal pipeline to verify 2-level tiling
    });
}

//===----------------------------------------------------------------------===//
// Extreme Pipeline L2 Full - Complete L1+L2 Tiling with LLVM Lowering
//===----------------------------------------------------------------------===//

void registerExtremePipelineL2Full() {
  PassPipelineRegistration<>(
    "google-extreme-l2-full",
    "Complete extreme pipeline with L1+L2 tiling and LLVM lowering",
    [](OpPassManager &pm) {
      // Phase 1: Lower Google to Linalg
      pm.addPass(createGoogleToLinalgLoweringPass());
      
      // Phase 2: FUSION (critical - must happen before tiling)
      pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
      
      // Phase 3: L1+L2 TILING via Transform Dialect
      pm.addPass(mlir::transform::createInterpreterPass());
      
      // Phase 4: Generalize named ops for further optimization
      pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
      
      // Phase 5: Bufferization
      pm.addPass(bufferization::createOneShotBufferizePass());
      
      // Phase 6: Lower to affine for loop optimization
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToAffineLoopsPass());
      
      // Affine optimizations
      pm.addNestedPass<func::FuncOp>(affine::createLoopFusionPass());
      pm.addNestedPass<func::FuncOp>(affine::createLoopCoalescingPass());
      
      // Lower affine
      pm.addPass(createLowerAffinePass());
      
      // Phase 7: Lower to LLVM
      pm.addPass(createConvertFuncToLLVMPass());
      pm.addPass(createArithToLLVMConversionPass());
      pm.addPass(createFinalizeMemRefToLLVMConversionPass());
      pm.addPass(createReconcileUnrealizedCastsPass());
    });
}

//===----------------------------------------------------------------------===//
// Extreme Pipeline L3 - L1+L2+L3 Cache Tiling (Minimal)
//===----------------------------------------------------------------------===//

void registerExtremePipelineL3() {
  PassPipelineRegistration<>(
    "google-extreme-l3",
    "Extreme pipeline with L1+L2+L3 cache tiling (L3: 256, L2: 64, L1: 16)",
    [](OpPassManager &pm) {
      // Phase 1: Lower Google to Linalg
      pm.addPass(createGoogleToLinalgLoweringPass());
      
      // Phase 2: FUSION (critical - must happen before tiling)
      pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
      
      // Phase 3: L1+L2+L3 TILING via Transform Dialect
      pm.addPass(mlir::transform::createInterpreterPass());
      
      // STOP HERE - minimal pipeline to verify 3-level tiling
    });
}

//===----------------------------------------------------------------------===//
// Extreme Pipeline L3 Full - Complete L1+L2+L3 Tiling with LLVM Lowering
//===----------------------------------------------------------------------===//

void registerExtremePipelineL3Full() {
  PassPipelineRegistration<>(
    "google-extreme-l3-full",
    "Complete extreme pipeline with L1+L2+L3 tiling and LLVM lowering",
    [](OpPassManager &pm) {
      // Phase 1: Lower Google to Linalg
      pm.addPass(createGoogleToLinalgLoweringPass());
      
      // Phase 2: FUSION (critical - must happen before tiling)
      pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
      
      // Phase 3: L1+L2+L3 TILING via Transform Dialect
      pm.addPass(mlir::transform::createInterpreterPass());
      
      // Phase 4: Generalize named ops for further optimization
      pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
      
      // Phase 5: Bufferization
      pm.addPass(bufferization::createOneShotBufferizePass());
      
      // Phase 6: Lower to affine for loop optimization
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToAffineLoopsPass());
      
      // Affine optimizations
      pm.addNestedPass<func::FuncOp>(affine::createLoopFusionPass());
      pm.addNestedPass<func::FuncOp>(affine::createLoopCoalescingPass());
      
      // Lower affine
      pm.addPass(createLowerAffinePass());
      
      // Phase 7: Lower to LLVM
      pm.addPass(createConvertFuncToLLVMPass());
      pm.addPass(createArithToLLVMConversionPass());
      pm.addPass(createFinalizeMemRefToLLVMConversionPass());
      pm.addPass(createReconcileUnrealizedCastsPass());
    });
}

//===----------------------------------------------------------------------===//
// Extreme Pipeline - Maximum Performance with L3 Tiling
//===----------------------------------------------------------------------===//

void registerExtremePipeline() {
  PassPipelineRegistration<>(
    "google-extreme-pipeline",
    "Extreme optimization with L1+L2+L3 tiling and aggressive optimizations",
    [](OpPassManager &pm) {
      // Phase 1: Lower Google to Linalg
      pm.addPass(createGoogleToLinalgLoweringPass());
      
      // Phase 2: FUSION (critical - must happen before tiling)
      pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
      
      // Phase 3: L1+L2+L3 TILING via Transform Dialect
      // This applies 3-level cache hierarchy tiling for maximum performance
      pm.addPass(mlir::transform::createInterpreterPass());
      
      // Phase 4: Generalize named ops for further optimization
      pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
      
      // Phase 5: Bufferization
      bufferization::OneShotBufferizePassOptions options;
      options.bufferizeFunctionBoundaries = true;
      pm.addPass(bufferization::createOneShotBufferizePass(options));
      
      // Phase 5.5: Cleanup bufferization ops
      pm.addPass(mlir::createConvertBufferizationToMemRefPass());

      
      // Phase 6: Lower Linalg to Affine for optimization opportunities
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToAffineLoopsPass());
      
      // Phase 6.5: Affine optimizations (loop fusion and coalescing)
      pm.addNestedPass<func::FuncOp>(affine::createLoopFusionPass());
      pm.addNestedPass<func::FuncOp>(affine::createLoopCoalescingPass());
      
      // Phase 7: Lower Affine to SCF (this properly lowers affine.for to scf.for)
      pm.addPass(createLowerAffinePass());
      
      // Phase 7.5: Lower SCF to ControlFlow
      pm.addPass(createSCFToControlFlowPass());
      
      // Phase 8: Lower to LLVM
      pm.addPass(memref::createExpandStridedMetadataPass());  // Expand memref.subview
      pm.addPass(createLowerAffinePass());  // Lower affine.apply from expansion
      pm.addPass(createConvertFuncToLLVMPass());
      pm.addPass(createArithToLLVMConversionPass());
      pm.addPass(createConvertControlFlowToLLVMPass());
      pm.addPass(createFinalizeMemRefToLLVMConversionPass());
      pm.addPass(createReconcileUnrealizedCastsPass());
    });
}

//===----------------------------------------------------------------------===//
// Pipeline Registration
//===----------------------------------------------------------------------===//

void registerGooglePipelines() {
  registerBasicPipeline();
  registerOptimizedPipeline();
  registerExtremePipeline();
  registerExtremePipelineL1();
  registerExtremePipelineL2();
  registerExtremePipelineL2Full();
  registerExtremePipelineL3();
  registerExtremePipelineL3Full();
}

} // namespace google
} // namespace mlir
