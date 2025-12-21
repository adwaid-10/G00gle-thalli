// Test shape operations
module {
  func.func @test_transpose(%arg0: tensor<4x8x16xf32>) -> tensor<16x4x8xf32> {
    %0 = google.transpose %arg0 {perm = [2, 0, 1]} : tensor<4x8x16xf32> -> tensor<16x4x8xf32>
    return %0 : tensor<16x4x8xf32>
  }

  func.func @test_slice(%arg0: tensor<32x128xf32>) -> tensor<32x40xf32> {
    %0 = google.slice %arg0 {start = [0, 10], end = [32, 50]} : tensor<32x128xf32> -> tensor<32x40xf32>
    return %0 : tensor<32x40xf32>
  }

  func.func @test_broadcast(%arg0: tensor<8xf32>) -> tensor<4x8xf32> {
    %0 = google.broadcast %arg0 {shape = [4, 8]} : tensor<8xf32> -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}
