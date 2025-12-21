// Test Softmax operation
module {
  func.func @test_softmax(%arg0: tensor<32x128xf32>) -> tensor<32x128xf32> {
    %0 = google.softmax %arg0 {axis = 1} : tensor<32x128xf32> -> tensor<32x128xf32>
    return %0 : tensor<32x128xf32>
  }
}
