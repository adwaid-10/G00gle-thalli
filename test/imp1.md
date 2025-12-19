# Implementing MatMul with TilingInterface in Google Dialect

This guide provides step-by-step instructions to add a `google.matmul` operation with full TilingInterface support.

## Overview

**What we're building:**
- A `google.matmul` operation that performs matrix multiplication: `C[M, N] = A[M, K] * B[K, N]`
- Full TilingInterface implementation to enable tiling via Transform Dialect
- Verification tests to ensure correctness

**Key Concepts:**
- **TilingInterface**: Defines how an operation can be split into smaller tiles
- **Transform Dialect**: Provides high-level scripting to trigger tiling transformations
- **Iterator Types**: Parallel (can run concurrently) vs Reduction (sequential accumulation)

---

## Step 1: Define the Operation in TableGen

**File:** `include/Google/IR/GoogleOps.td`

Add the MatMul operation definition after the existing operations:

```tablegen
def Google_MatMulOp : Google_Op<"matmul",
  [Pure,
   DeclareOpInterfaceMethods<TilingInterface,
    ["getIterationDomain",
     "getLoopIteratorTypes",
     "getResultTilePosition",
     "getTiledImplementation",
     "generateResultTileValue",
     "getIterationDomainTileFromResultTile"]>
  ]> {
  let summary = "Matrix multiplication operation";
  let description = [{
    Performs matrix multiplication: C = A * B
    
    Where:
    - A has shape [M, K]
    - B has shape [K, N]
    - C has shape [M, N]
    
    Example:
    ```mlir
    %result = google.matmul %A, %B : tensor<4x8xf32>, tensor<8x16xf32> -> tensor<4x16xf32>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs
  );
  
  let results = (outs
    AnyRankedTensor:$output
  );
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
  
  let hasVerifier = 1;
}
```

**Explanation:**
- `Pure`: Operation has no side effects
- `DeclareOpInterfaceMethods<TilingInterface, [...]>`: Declares we'll implement these specific interface methods
- `AnyRankedTensor`: Accepts any ranked tensor (must have known rank, not just `tensor<*xf32>`)
- `hasVerifier = 1`: We'll add verification logic to check shapes match

---

## Step 2: Implement Verification Logic

**File:** `lib/Google/IR/GoogleOps.cpp`

Add verification to ensure matrix dimensions are compatible:

```cpp
LogicalResult MatMulOp::verify() {
  auto lhsType = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(getRhs().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  
  if (!lhsType || !rhsType || !outputType) {
    return emitOpError("operands and result must be ranked tensors");
  }
  
  // Check ranks are exactly 2
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || outputType.getRank() != 2) {
    return emitOpError("all operands must be 2D tensors (matrices)");
  }
  
  int64_t M = lhsType.getDimSize(0);
  int64_t K_lhs = lhsType.getDimSize(1);
  int64_t K_rhs = rhsType.getDimSize(0);
  int64_t N = rhsType.getDimSize(1);
  
  // Check contraction dimension matches
  if (K_lhs != K_rhs && K_lhs != ShapedType::kDynamic && K_rhs != ShapedType::kDynamic) {
    return emitOpError("contraction dimension mismatch: lhs has ")
           << K_lhs << " but rhs has " << K_rhs;
  }
  
  // Check output shape
  int64_t out_M = outputType.getDimSize(0);
  int64_t out_N = outputType.getDimSize(1);
  
  if (M != out_M && M != ShapedType::kDynamic && out_M != ShapedType::kDynamic) {
    return emitOpError("output dimension 0 mismatch: expected ")
           << M << " but got " << out_M;
  }
  
  if (N != out_N && N != ShapedType::kDynamic && out_N != ShapedType::kDynamic) {
    return emitOpError("output dimension 1 mismatch: expected ")
           << N << " but got " << out_N;
  }
  
  // Check element types match
  if (lhsType.getElementType() != rhsType.getElementType() ||
      lhsType.getElementType() != outputType.getElementType()) {
    return emitOpError("element types must match across all operands");
  }
  
  return success();
}
```

**Explanation:**
- Validates that LHS[M, K] × RHS[K, N] = Output[M, N]
- Handles dynamic dimensions (`ShapedType::kDynamic`) gracefully
- Ensures element types are consistent

---

## Step 3: Implement TilingInterface Methods

**File:** `lib/Google/IR/GoogleOps.cpp`

### 3.1 Define Iteration Domain

This tells the compiler the loop bounds for each dimension:

```cpp
SmallVector<Range> MatMulOp::getIterationDomain(OpBuilder &b) {
  Location loc = getLoc();
  Value lhs = getLhs();
  Value rhs = getRhs();
  
  // Get M, N, K dimensions
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);
  
  SmallVector<OpFoldResult> lhsSizes = tensor::getMixedSizes(b, loc, lhs);
  SmallVector<OpFoldResult> rhsSizes = tensor::getMixedSizes(b, loc, rhs);
  
  OpFoldResult M = lhsSizes[0];  // Rows of LHS
  OpFoldResult N = rhsSizes[1];  // Cols of RHS
  OpFoldResult K = lhsSizes[1];  // Contraction dimension
  
  // Return 3 ranges: [M, N, K]
  // M and N are parallel, K is reduction
  SmallVector<Range> loopBounds(3);
  loopBounds[0] = Range{zero, M, one};  // M dimension
  loopBounds[1] = Range{zero, N, one};  // N dimension
  loopBounds[2] = Range{zero, K, one};  // K dimension (reduction)
  
  return loopBounds;
}
```

**Explanation:**
- MatMul has 3 logical loops: `for m in [0, M)`, `for n in [0, N)`, `for k in [0, K)`
- `Range{offset, size, stride}` defines each loop's bounds
- We return all 3 dimensions even though output is only 2D (K is the reduction dimension)

### 3.2 Define Iterator Types

This specifies which loops are parallel vs reduction:

```cpp
SmallVector<utils::IteratorType> MatMulOp::getLoopIteratorTypes() {
  // M and N are parallel (can be tiled independently)
  // K is reduction (must accumulate sequentially)
  return {
    utils::IteratorType::parallel,   // M
    utils::IteratorType::parallel,   // N
    utils::IteratorType::reduction   // K
  };
}
```

**Explanation:**
- **Parallel**: M and N loops can run in any order or concurrently
- **Reduction**: K loop must accumulate results (like a sum)

### 3.3 Map Result Tile to Iteration Domain

```cpp
LogicalResult MatMulOp::getIterationDomainTileFromResultTile(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
    SmallVectorImpl<OpFoldResult> &iterDomainSizes) {
  
  // Result has shape [M, N]
  // Iteration domain is [M, N, K]
  // When tiling result at [m_offset, n_offset] with size [m_size, n_size],
  // we tile iteration domain at [m_offset, n_offset, 0] with size [m_size, n_size, K]
  
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult K = tensor::getMixedSizes(b, getLoc(), getLhs())[1];
  
  iterDomainOffsets = {offsets[0], offsets[1], zero};
  iterDomainSizes = {sizes[0], sizes[1], K};
  
  return success();
}
```

**Explanation:**
- When we tile the output at position `[m, n]`, we need the full K dimension
- This maps 2D output tiles to 3D iteration space

### 3.4 Map Iteration Domain Tile to Result Position

```cpp
LogicalResult MatMulOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  
  // Iteration domain is [M, N, K]
  // Result is [M, N]
  // Just drop the K dimension
  resultOffsets = {offsets[0], offsets[1]};
  resultSizes = {sizes[0], sizes[1]};
  
  return success();
}
```

**Explanation:**
- Inverse of previous function
- Projects 3D iteration tile to 2D output tile

### 3.5 Generate Tiled Implementation

This is the core method that generates the IR for one tile:

```cpp
FailureOr<TilingResult> MatMulOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  
  Location loc = getLoc();
  
  // offsets = [m_offset, n_offset, k_offset]
  // sizes = [m_size, n_size, k_size]
  
  // Extract tile from LHS: A[m_offset:m_offset+m_size, k_offset:k_offset+k_size]
  SmallVector<OpFoldResult> lhsOffsets = {offsets[0], offsets[2]};
  SmallVector<OpFoldResult> lhsSizes = {sizes[0], sizes[2]};
  SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
  
  auto lhsTile = b.create<tensor::ExtractSliceOp>(
      loc, getLhs(), lhsOffsets, lhsSizes, strides);
  
  // Extract tile from RHS: B[k_offset:k_offset+k_size, n_offset:n_offset+n_size]
  SmallVector<OpFoldResult> rhsOffsets = {offsets[2], offsets[1]};
  SmallVector<OpFoldResult> rhsSizes = {sizes[2], sizes[1]};
  
  auto rhsTile = b.create<tensor::ExtractSliceOp>(
      loc, getRhs(), rhsOffsets, rhsSizes, strides);
  
  // Compute result type for the tile
  auto lhsTileType = lhsTile.getType();
  auto rhsTileType = rhsTile.getType();
  
  SmallVector<int64_t> resultShape = {
    lhsTileType.getDimSize(0),  // m_size
    rhsTileType.getDimSize(1)   // n_size
  };
  
  auto resultTileType = RankedTensorType::get(
      resultShape, lhsTileType.getElementType());
  
  // Create tiled matmul operation
  Operation *tiledOp = mlir::clone(b, getOperation(), {resultTileType},
                                   {lhsTile, rhsTile});
  
  return TilingResult{
    {tiledOp},                                    // Tiled operations
    SmallVector<Value>(tiledOp->getResults()),   // Results
    {lhsTile, rhsTile}                            // Extracted slices
  };
}
```

**Explanation:**
- Extracts the appropriate slices from LHS and RHS based on tile position
- Creates a new `google.matmul` operating on the smaller tiles
- Returns the tiled operation and the slice operations

### 3.6 Generate Result Tile Value

```cpp
FailureOr<TilingResult> MatMulOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  
  SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
  if (failed(getIterationDomainTileFromResultTile(
          b, resultNumber, offsets, sizes, mappedOffsets, mappedSizes))) {
    return failure();
  }
  
  return getTiledImplementation(b, mappedOffsets, mappedSizes);
}
```

**Explanation:**
- Convenience method that combines the mapping and tiling steps
- Used by some Transform Dialect operations

---

## Step 4: Create Test Cases

### 4.1 Basic MatMul Test

**File:** `test/matmul_basic.mlir`

```mlir
// RUN: google-opt %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @test_matmul
  func.func @test_matmul(%A: tensor<4x8xf32>, %B: tensor<8x16xf32>) -> tensor<4x16xf32> {
    // CHECK: google.matmul
    %C = google.matmul %A, %B : tensor<4x8xf32>, tensor<8x16xf32> -> tensor<4x16xf32>
    return %C : tensor<4x16xf32>
  }
}
```

### 4.2 Tiling Test with Transform Dialect

**File:** `test/matmul_tiling.mlir`

```mlir
// RUN: google-opt %s --transform-interpreter | FileCheck %s

module {
  // CHECK-LABEL: func.func @test_matmul_tiling
  func.func @test_matmul_tiling(%A: tensor<128x256xf32>, %B: tensor<256x512xf32>) -> tensor<128x512xf32> {
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     tensor.extract_slice
    // CHECK:     tensor.extract_slice
    // CHECK:     google.matmul
    // CHECK:     tensor.insert_slice
    %C = google.matmul %A, %B : tensor<128x256xf32>, tensor<256x512xf32> -> tensor<128x512xf32>
    return %C : tensor<128x512xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["google.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    
    // Tile M and N dimensions with size 32x64
    %tiled, %loops:2 = transform.structured.tile_using_for %matmul tile_sizes [32, 64, 0] 
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    
    transform.yield
  }
}
```

**Explanation:**
- `tile_sizes [32, 64, 0]`: Tile M by 32, N by 64, don't tile K (0 means full dimension)
- This creates nested `scf.for` loops for M and N
- Inside the loops, smaller matmuls operate on 32×K and K×64 tiles

---

## Step 5: Build and Test

### 5.1 Rebuild the Project

```bash
cmake --build c:\Users\Asus\Desktop\google\build --config Release --target google-opt
```

### 5.2 Run Tests

```bash
# Basic verification
c:\Users\Asus\Desktop\google\build\bin\google-opt.exe test/matmul_basic.mlir

# Tiling test
c:\Users\Asus\Desktop\google\build\bin\google-opt.exe test/matmul_tiling.mlir --transform-interpreter
```

---

## Understanding the Tiling Flow

When you run the transform script with `tile_sizes [32, 64, 0]`:

1. **Match Phase**: Transform Dialect finds `google.matmul`
2. **Query Phase**: Calls `getIterationDomain()` → gets `[M=128, N=512, K=256]`
3. **Loop Generation**: Creates nested loops:
   ```
   for m = 0 to 128 step 32:
     for n = 0 to 512 step 64:
       // Tile computation here
   ```
4. **Tile Computation**: For each iteration, calls `getTiledImplementation([m, n, 0], [32, 64, 256])`:
   - Extracts `A[m:m+32, 0:256]`
   - Extracts `B[0:256, n:n+64]`
   - Computes `C_tile = matmul(A_tile, B_tile)` → shape `[32, 64]`
   - Inserts `C_tile` into `C[m:m+32, n:n+64]`

---

## Key Takeaways

1. **TilingInterface is the engine**: It defines the semantics of how your operation tiles
2. **Transform Dialect is the driver**: It provides user-friendly scripting to trigger tiling
3. **Iterator types matter**: Parallel vs reduction affects how loops can be optimized
4. **Verification is crucial**: Catch shape mismatches early with good verifiers
5. **Test incrementally**: Start with basic tests, then add tiling tests

This pattern can be applied to any structured operation (convolution, pooling, etc.) in your custom dialect!
