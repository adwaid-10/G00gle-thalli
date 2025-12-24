# MLIR Execution Engine - Practical Examples

## 📚 Quick Reference

This document provides ready-to-use code examples for MLIR's execution engine.

---

## 🚀 Example 1: Basic mlir-cpu-runner Usage

### Simple MatMul Test

**File**: `examples/01_simple_matmul_runner.mlir`

```mlir
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    
    // Create 256x256 matrices filled with 1.0
    %A = arith.constant dense<1.0> : tensor<256x256xf32>
    %B = arith.constant dense<1.0> : tensor<256x256xf32>
    %C_init = tensor.empty() : tensor<256x256xf32>
    
    // Initialize C with zeros
    %cst = arith.constant 0.0 : f32
    %C = linalg.fill ins(%cst : f32) outs(%C_init : tensor<256x256xf32>) 
                      -> tensor<256x256xf32>
    
    // Perform matrix multiplication
    %result = linalg.matmul ins(%A, %B : tensor<256x256xf32>, tensor<256x256xf32>)
                             outs(%C : tensor<256x256xf32>) 
                             -> tensor<256x256xf32>
    
    // Extract and print result[0,0] (should be 256.0)
    %elem = tensor.extract %result[%c0, %c0] : tensor<256x256xf32>
    vector.print %elem : f32
    
    return
  }
}
```

**Run Command**:
```bash
# Process with pipeline
google-opt examples/01_simple_matmul_runner.mlir \
  --google-optimized-pipeline \
  -o output/matmul_processed.mlir

# Execute with CPU runner
mlir-cpu-runner output/matmul_processed.mlir \
  --entry-point-result=void \
  --shared-libs=$LLVM_BUILD/lib/libmlir_c_runner_utils.so \
  --shared-libs=$LLVM_BUILD/lib/libmlir_runner_utils.so
```

**Expected Output**:
```
256.0
```

---

## 🔬 Example 2: Benchmarking Script

### PowerShell Benchmark Script

**File**: `scripts/benchmark_mlir.ps1`

```powershell
# MLIR Pipeline Benchmarking Script

param(
    [string]$TestFile = "test\test_extreme_l3_e2e.mlir",
    [int]$Iterations = 10
)

Write-Host "`n=== MLIR Pipeline Benchmark ===`n" -ForegroundColor Cyan

# Function to run pipeline and measure time
function Benchmark-Pipeline {
    param($PipelineName, $PipelineFlag)
    
    Write-Host "Benchmarking $PipelineName..." -ForegroundColor Yellow
    $times = @()
    
    for ($i = 1; $i -le $Iterations; $i++) {
        $start = Get-Date
        .\build\bin\google-opt.exe $TestFile $PipelineFlag `
            -o output\temp.mlir 2>&1 | Out-Null
        $end = Get-Date
        
        $ms = ($end - $start).TotalMilliseconds
        $times += $ms
        Write-Host "  Run $i : $([math]::Round($ms, 2))ms" -ForegroundColor Gray
    }
    
    $avg = ($times | Measure-Object -Average).Average
    $min = ($times | Measure-Object -Minimum).Minimum
    $max = ($times | Measure-Object -Maximum).Maximum
    
    return @{
        Name = $PipelineName
        Average = $avg
        Min = $min
        Max = $max
    }
}

# Benchmark pipelines
$optimized = Benchmark-Pipeline "Optimized" "--google-optimized-pipeline"
$extreme = Benchmark-Pipeline "Extreme" "--google-extreme-pipeline"

# Display results
Write-Host "`n=== Results ===`n" -ForegroundColor Cyan

Write-Host "Optimized Pipeline:" -ForegroundColor Green
Write-Host "  Average: $([math]::Round($optimized.Average, 2))ms"
Write-Host "  Range: $([math]::Round($optimized.Min, 2))ms - $([math]::Round($optimized.Max, 2))ms"

Write-Host "`nExtreme Pipeline:" -ForegroundColor Green
Write-Host "  Average: $([math]::Round($extreme.Average, 2))ms"
Write-Host "  Range: $([math]::Round($extreme.Min, 2))ms - $([math]::Round($extreme.Max, 2))ms"

$overhead = (($extreme.Average - $optimized.Average) / $optimized.Average) * 100
Write-Host "`nOverhead: +$([math]::Round($overhead, 1))%" -ForegroundColor Yellow
```

**Usage**:
```powershell
.\scripts\benchmark_mlir.ps1 -TestFile "test\my_test.mlir" -Iterations 20
```

---

## 💻 Example 3: C++ ExecutionEngine Integration

### Complete C++ Example

**File**: `examples/execution_engine_example.cpp`

```cpp
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <mlir-file>\n";
    return 1;
  }
  
  // Initialize LLVM
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  
  // Create MLIR context
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  mlir::registerLLVMDialectTranslation(context);
  
  // Parse MLIR file
  std::cout << "Parsing " << argv[1] << "...\n";
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  
  if (!module) {
    std::cerr << "Failed to parse MLIR file\n";
    return 1;
  }
  
  // Create execution engine
  std::cout << "Creating execution engine...\n";
  
  mlir::ExecutionEngineOptions options;
  options.transformer = mlir::makeOptimizingTransformer(
    /*optLevel=*/3, 
    /*sizeLevel=*/0,
    /*targetMachine=*/nullptr
  );
  
  // Add runtime libraries
  options.sharedLibPaths = {
    "libmlir_c_runner_utils.so",
    "libmlir_runner_utils.so"
  };
  
  auto maybeEngine = mlir::ExecutionEngine::create(module.get(), options);
  
  if (!maybeEngine) {
    std::cerr << "Failed to create execution engine\n";
    return 1;
  }
  
  auto& engine = maybeEngine.get();
  
  // Find the function to execute
  std::string funcName = "matmul_l3_test";
  std::cout << "Looking up function: " << funcName << "\n";
  
  auto expectedFPtr = engine->lookup(funcName);
  if (!expectedFPtr) {
    std::cerr << "Function not found: " << funcName << "\n";
    return 1;
  }
  
  // Warm-up run
  std::cout << "Warm-up run...\n";
  auto error = engine->invokePacked(funcName);
  if (error) {
    std::cerr << "Warm-up execution failed\n";
    return 1;
  }
  
  // Benchmark
  const int iterations = 10;
  std::cout << "Benchmarking " << iterations << " iterations...\n";
  
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < iterations; i++) {
    auto error = engine->invokePacked(funcName);
    if (error) {
      std::cerr << "Execution failed at iteration " << i << "\n";
      return 1;
    }
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    end - start).count();
  
  double avgTime = static_cast<double>(duration) / iterations;
  
  std::cout << "\nResults:\n";
  std::cout << "  Total time: " << duration << " ms\n";
  std::cout << "  Average time: " << avgTime << " ms\n";
  std::cout << "  Throughput: " << (1000.0 / avgTime) << " ops/sec\n";
  
  return 0;
}
```

**CMakeLists.txt**:
```cmake
add_executable(mlir_execution_example
  examples/execution_engine_example.cpp
)

target_link_libraries(mlir_execution_example
  PRIVATE
  MLIRExecutionEngine
  MLIRIR
  MLIRParser
  MLIRTargetLLVMIRExport
  MLIRSupport
  ${llvm_libs}
)
```

**Build and Run**:
```bash
cmake --build build --target mlir_execution_example
./build/mlir_execution_example output/matmul_extreme.mlir
```

---

## 📊 Example 4: Performance Analysis Script

### Analyze LLVM IR Characteristics

**File**: `scripts/analyze_llvm_ir.ps1`

```powershell
param([string]$LLVMFile)

Write-Host "`n=== LLVM IR Analysis ===`n" -ForegroundColor Cyan

$content = Get-Content $LLVMFile
$lines = $content.Count

# Count operations
$branches = (Select-String "\sbr " $LLVMFile).Count
$phis = (Select-String "phi " $LLVMFile).Count
$loads = (Select-String "load " $LLVMFile).Count
$stores = (Select-String "store " $LLVMFile).Count
$fmuls = (Select-String "fmul " $LLVMFile).Count
$fadds = (Select-String "fadd " $LLVMFile).Count
$calls = (Select-String "call " $LLVMFile).Count

Write-Host "File: $LLVMFile" -ForegroundColor Yellow
Write-Host "Total lines: $lines`n" -ForegroundColor White

Write-Host "Control Flow:" -ForegroundColor Green
Write-Host "  Branch instructions: $branches"
Write-Host "  PHI nodes: $phis"
Write-Host "  Estimated loops: $([math]::Round($branches / 4, 0))`n"

Write-Host "Memory Operations:" -ForegroundColor Green
Write-Host "  Loads: $loads"
Write-Host "  Stores: $stores"
Write-Host "  Total: $($loads + $stores)`n"

Write-Host "Arithmetic:" -ForegroundColor Green
Write-Host "  FP Multiplications: $fmuls"
Write-Host "  FP Additions: $fadds"
Write-Host "  Total FLOPs: $($fmuls + $fadds)`n"

Write-Host "Function Calls: $calls`n" -ForegroundColor Green

if (($loads + $stores) -gt 0) {
    $intensity = ($fmuls + $fadds) / ($loads + $stores)
    Write-Host "Arithmetic Intensity: $([math]::Round($intensity, 3))" -ForegroundColor Yellow
    
    if ($intensity -lt 0.1) {
        Write-Host "  → Memory-optimized (good for cache tiling)" -ForegroundColor Green
    } elseif ($intensity -lt 1.0) {
        Write-Host "  → Balanced" -ForegroundColor Yellow
    } else {
        Write-Host "  → Compute-intensive" -ForegroundColor Cyan
    }
}

Write-Host ""
```

**Usage**:
```powershell
.\scripts\analyze_llvm_ir.ps1 output\matmul_extreme.ll
```

---

## 🎯 Example 5: Complete Workflow

### End-to-End Testing Workflow

**File**: `scripts/test_pipeline_e2e.ps1`

```powershell
# End-to-End Pipeline Testing

param(
    [string]$TestFile = "test\test_extreme_l3_e2e.mlir",
    [string]$Pipeline = "--google-extreme-pipeline"
)

Write-Host "`n=== End-to-End Pipeline Test ===`n" -ForegroundColor Cyan

# Step 1: Compile with pipeline
Write-Host "Step 1: Running pipeline..." -ForegroundColor Yellow
$start = Get-Date
.\build\bin\google-opt.exe $TestFile $Pipeline -o output\processed.mlir 2>&1 | Out-Null
$compileTime = (Get-Date - $start).TotalMilliseconds

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Pipeline succeeded ($([math]::Round($compileTime, 2))ms)" -ForegroundColor Green
} else {
    Write-Host "  ✗ Pipeline failed" -ForegroundColor Red
    exit 1
}

# Step 2: Filter Transform dialect
Write-Host "`nStep 2: Filtering Transform dialect..." -ForegroundColor Yellow
$lines = Get-Content output\processed.mlir
$inModule = $false
$moduleDepth = 0
$output = @()

for ($i = 0; $i -lt $lines.Count; $i++) {
    $line = $lines[$i]
    if ($line -match "^module") {
        $output += "module {"
        $inModule = $true
        $moduleDepth = 1
        continue
    }
    if ($inModule) {
        if ($line -match "^\s*transform\.") {
            $depth = 1
            while ($depth -gt 0 -and $i -lt $lines.Count) {
                $i++
                if ($lines[$i] -match "\{") { $depth++ }
                if ($lines[$i] -match "\}") { $depth-- }
            }
            continue
        }
        if ($line -match "^\}$") {
            $moduleDepth--
            if ($moduleDepth -eq 0) {
                $output += "}"
                break
            }
        }
        if ($line -match "\{") { $moduleDepth++ }
        $output += $line
    }
}

$output | Set-Content output\processed_clean.mlir
Write-Host "  ✓ Transform dialect removed" -ForegroundColor Green

# Step 3: Translate to LLVM IR
Write-Host "`nStep 3: Translating to LLVM IR..." -ForegroundColor Yellow
$mlirTranslate = "C:\Users\Asus\Desktop\llvm-project\build\bin\mlir-translate.exe"
& $mlirTranslate --mlir-to-llvmir output\processed_clean.mlir -o output\result.ll 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    $llFile = Get-Item output\result.ll
    Write-Host "  ✓ LLVM IR generated ($([math]::Round($llFile.Length/1KB, 2)) KB)" -ForegroundColor Green
} else {
    Write-Host "  ✗ Translation failed" -ForegroundColor Red
    exit 1
}

# Step 4: Analyze LLVM IR
Write-Host "`nStep 4: Analyzing LLVM IR..." -ForegroundColor Yellow
.\scripts\analyze_llvm_ir.ps1 output\result.ll

# Step 5: Verify syntax
Write-Host "`nStep 5: Verifying LLVM IR syntax..." -ForegroundColor Yellow
$llvmAs = "C:\Users\Asus\Desktop\llvm-project\build\bin\llvm-as.exe"
& $llvmAs output\result.ll -o output\result.bc 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    $bcFile = Get-Item output\result.bc
    Write-Host "  ✓ Syntax valid ($([math]::Round($bcFile.Length/1KB, 2)) KB bitcode)" -ForegroundColor Green
} else {
    Write-Host "  ✗ Syntax errors found" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== All Tests Passed! ===`n" -ForegroundColor Green
```

**Usage**:
```powershell
.\scripts\test_pipeline_e2e.ps1 -TestFile "test\my_test.mlir" -Pipeline "--google-extreme-pipeline"
```

---

## 📚 Quick Command Reference

### Common Commands

```bash
# Run with optimized pipeline
google-opt input.mlir --google-optimized-pipeline -o output.mlir

# Run with extreme pipeline
google-opt input.mlir --google-extreme-pipeline -o output.mlir

# Translate to LLVM IR
mlir-translate --mlir-to-llvmir input.mlir -o output.ll

# Verify LLVM IR syntax
llvm-as output.ll -o output.bc

# Execute with CPU runner
mlir-cpu-runner input.mlir \
  --entry-point-result=void \
  --shared-libs=libmlir_c_runner_utils.so

# Analyze with llvm-mca (machine code analyzer)
llvm-mca output.ll --timeline --iterations=100
```

### Environment Setup

```powershell
# Set LLVM paths
$env:LLVM_BUILD = "C:\Users\Asus\Desktop\llvm-project\build"
$env:PATH += ";$env:LLVM_BUILD\bin"

# Set library paths
$env:LD_LIBRARY_PATH = "$env:LLVM_BUILD\lib"
```

---

## ✅ Summary

This document provides:
- ✅ Ready-to-use MLIR examples
- ✅ Benchmarking scripts
- ✅ C++ ExecutionEngine integration
- ✅ Analysis tools
- ✅ Complete workflows

**Next Steps**:
1. Try the examples
2. Modify for your use case
3. Benchmark your optimizations
4. Analyze the results

**Remember**:
- Start simple, then optimize
- Measure before and after
- Use the right tool for the job
- MLIR ExecutionEngine makes testing easy! 🚀
