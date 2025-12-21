// Test Dequant operation
module {
  func.func @test_dequant(%input: tensor<32x128xi8>, %scale: tensor<32x128xf32>) -> tensor<32x128xf32> {
    %0 = google.dequant %input, %scale : tensor<32x128xi8>, tensor<32x128xf32>, tensor<32x128xf32>
    return %0 : tensor<32x128xf32>
  }
}
