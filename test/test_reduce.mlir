// Test Reduce operation
module {
  func.func @test_reduce_sum(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
    %0 = google.reduce<sum> %arg0 {axes = [1]} : tensor<4x8xf32> -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @test_reduce_max(%arg0: tensor<8x16xf32>) -> tensor<8xf32> {
    %0 = google.reduce<max> %arg0 {axes = [1]} : tensor<8x16xf32> -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
