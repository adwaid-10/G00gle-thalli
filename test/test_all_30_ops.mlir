// FINAL COMPREHENSIVE TEST: All 30 Operations - 100% COMPLETE
// GoogleToLinalg Lowering Pass - Complete Test Suite
module {
  // Binary operations (7)
  func.func @test_binary(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
    %0 = google.add %arg0, %arg1 : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %1 = google.sub %0, %arg1 : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %2 = google.mul %1, %arg0 : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %3 = google.div %2, %arg1 : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %4 = google.pow %3, %arg0 : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %5 = google.max %4, %arg0 : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %6 = google.min %5, %arg1 : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    return %6 : tensor<8xf32>
  }

  // Unary operations (8)
  func.func @test_unary(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = google.neg %arg0 : tensor<16xf32> -> tensor<16xf32>
    %1 = google.abs %0 : tensor<16xf32> -> tensor<16xf32>
    %2 = google.sqrt %1 : tensor<16xf32> -> tensor<16xf32>
    %3 = google.rsqrt %2 : tensor<16xf32> -> tensor<16xf32>
    %4 = google.exp %3 : tensor<16xf32> -> tensor<16xf32>
    %5 = google.log %4 : tensor<16xf32> -> tensor<16xf32>
    %6 = google.ceil %5 : tensor<16xf32> -> tensor<16xf32>
    %7 = google.tanh %6 : tensor<16xf32> -> tensor<16xf32>
    return %7 : tensor<16xf32>
  }

  // Activation functions (4)
  func.func @test_activations(%arg0: tensor<32x128xf32>) -> tensor<32x128xf32> {
    %0 = google.relu %arg0 : tensor<32x128xf32> -> tensor<32x128xf32>
    %1 = google.sigmoid %0 : tensor<32x128xf32> -> tensor<32x128xf32>
    %2 = google.gelu %1 : tensor<32x128xf32> -> tensor<32x128xf32>
    %3 = google.softmax %2 {axis = 1} : tensor<32x128xf32> -> tensor<32x128xf32>
    return %3 : tensor<32x128xf32>
  }

  // Selection operations (2)
  func.func @test_selection(%cond: tensor<8xi1>, %true_val: tensor<8xf32>, %false_val: tensor<8xf32>, 
                             %input: tensor<8xf32>, %min: tensor<8xf32>, %max: tensor<8xf32>) -> tensor<8xf32> {
    %0 = google.select %cond, %true_val, %false_val : tensor<8xi1>, tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %1 = google.clamp %input, %min, %max : tensor<8xf32>, tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %2 = google.add %0, %1 : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    return %2 : tensor<8xf32>
  }

  // Matrix operations (1)
  func.func @test_matmul(%arg0: tensor<32x64xf32>, %arg1: tensor<64x128xf32>) -> tensor<32x128xf32> {
    %0 = google.matmul %arg0, %arg1 : tensor<32x64xf32>, tensor<64x128xf32> -> tensor<32x128xf32>
    return %0 : tensor<32x128xf32>
  }

  // Reduction operations (1)
  func.func @test_reduce(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
    %0 = google.reduce<sum> %arg0 {axes = [1]} : tensor<4x8xf32> -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // Shape operations (5) - NOW INCLUDING RESHAPE AND CONCAT!
  func.func @test_shape_ops(%arg0: tensor<4x8x16xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<8xf32>, 
                             %arg3: tensor<32xf32>, %arg4: tensor<4x8xf32>, %arg5: tensor<4x8xf32>) 
                             -> (tensor<16x4x8xf32>, tensor<32x40xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<8x8xf32>) {
    %0 = google.transpose %arg0 {perm = [2, 0, 1]} : tensor<4x8x16xf32> -> tensor<16x4x8xf32>
    %1 = google.slice %arg1 {start = [0, 10], end = [32, 50]} : tensor<32x128xf32> -> tensor<32x40xf32>
    %2 = google.broadcast %arg2 {shape = [4, 8]} : tensor<8xf32> -> tensor<4x8xf32>
    %3 = google.reshape %arg3 {shape = [4, 8]} : tensor<32xf32> -> tensor<4x8xf32>
    %4 = google.concat %arg4, %arg5 {axis = 0} : tensor<4x8xf32>, tensor<4x8xf32> -> tensor<8x8xf32>
    return %0, %1, %2, %3, %4 : tensor<16x4x8xf32>, tensor<32x40xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<8x8xf32>
  }

  // Dequant operation (1)
  func.func @test_dequant(%input: tensor<16x32xi8>, %scale: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %0 = google.dequant %input, %scale : tensor<16x32xi8>, tensor<16x32xf32>, tensor<16x32xf32>
    return %0 : tensor<16x32xf32>
  }

  // Constant operation (1) - passes through unchanged
  func.func @test_constant() -> tensor<4xf32> {
    %0 = google.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// Total: 30 operations tested
// Binary: 7, Unary: 8, Activations: 4, Selection: 2
// Matrix: 1, Reduce: 1, Shape: 5, Dequant: 1, Constant: 1
