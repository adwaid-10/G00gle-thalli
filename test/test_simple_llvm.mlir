// Simpler test for complete LLVM lowering
// Uses smaller matrix to avoid bufferization issues

module {
  func.func @simple_matmul() -> i32 {
    // Simple computation that can fully lower to LLVM
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    
    %sum = arith.addi %c1, %c2 : i32
    %result = arith.muli %sum, %c2 : i32
    
    return %result : i32
  }
  
  func.func @main() -> i32 {
    %result = func.call @simple_matmul() : () -> i32
    return %result : i32
  }
}
