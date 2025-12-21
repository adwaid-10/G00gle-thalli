// Test Reshape operation - comprehensive
module {
  // 2D to 1D
  func.func @test_reshape_2d_to_1d(%arg0: tensor<4x8xf32>) -> tensor<32xf32> {
    %0 = google.reshape %arg0 {shape = [32]} : tensor<4x8xf32> -> tensor<32xf32>
    return %0 : tensor<32xf32>
  }

  // 1D to 2D
  func.func @test_reshape_1d_to_2d(%arg0: tensor<32xf32>) -> tensor<4x8xf32> {
    %0 = google.reshape %arg0 {shape = [4, 8]} : tensor<32xf32> -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }

  // 3D to 2D
  func.func @test_reshape_3d_to_2d(%arg0: tensor<2x4x8xf32>) -> tensor<8x8xf32> {
    %0 = google.reshape %arg0 {shape = [8, 8]} : tensor<2x4x8xf32> -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  // 2D to 3D
  func.func @test_reshape_2d_to_3d(%arg0: tensor<8x8xf32>) -> tensor<2x4x8xf32> {
    %0 = google.reshape %arg0 {shape = [2, 4, 8]} : tensor<8x8xf32> -> tensor<2x4x8xf32>
    return %0 : tensor<2x4x8xf32>
  }
}
