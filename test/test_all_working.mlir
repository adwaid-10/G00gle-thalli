// Comprehensive test for all implemented operations
module {
  // Binary operations (7)
  func.func @test_binary(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
    %0 = google.add %arg0, %arg1 : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %1 = google.sub %0, %arg1 : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %2 = google.mul %1, %arg0 : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %3 = google.div %2, %arg1 : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %4 = google.max %3, %arg0 : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %5 = google.min %4, %arg1 : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    return %5 : tensor<8xf32>
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
  func.func @test_activations(%arg0: tensor<32xf32>) -> tensor<32xf32> {
    %0 = google.relu %arg0 : tensor<32xf32> -> tensor<32xf32>
    %1 = google.sigmoid %0 : tensor<32xf32> -> tensor<32xf32>
    %2 = google.gelu %1 : tensor<32xf32> -> tensor<32xf32>
    %3 = google.tanh %2 : tensor<32xf32> -> tensor<32xf32>
    return %3 : tensor<32xf32>
  }

  // Selection operations (2)
  func.func @test_selection(%cond: tensor<8xi1>, %true_val: tensor<8xf32>, %false_val: tensor<8xf32>, 
                             %input: tensor<8xf32>, %min: tensor<8xf32>, %max: tensor<8xf32>) -> tensor<8xf32> {
    %0 = google.select %cond, %true_val, %false_val : tensor<8xi1>, tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %1 = google.clamp %input, %min, %max : tensor<8xf32>, tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %2 = google.add %0, %1 : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    return %2 : tensor<8xf32>
  }
}
