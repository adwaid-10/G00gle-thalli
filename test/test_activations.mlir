// Test activation functions
module {
  func.func @test_activations(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = google.relu %arg0 : tensor<16xf32> -> tensor<16xf32>
    %1 = google.sigmoid %0 : tensor<16xf32> -> tensor<16xf32>
    %2 = google.gelu %1 : tensor<16xf32> -> tensor<16xf32>
    %3 = google.tanh %2 : tensor<16xf32> -> tensor<16xf32>
    return %3 : tensor<16xf32>
  }
}
