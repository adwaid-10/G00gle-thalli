// Test MatMul operation
module {
  func.func @test_matmul(%arg0: tensor<32x64xf32>, %arg1: tensor<64x128xf32>) -> tensor<32x128xf32> {
    %0 = google.matmul %arg0, %arg1 : tensor<32x64xf32>, tensor<64x128xf32> -> tensor<32x128xf32>
    return %0 : tensor<32x128xf32>
  }
}
