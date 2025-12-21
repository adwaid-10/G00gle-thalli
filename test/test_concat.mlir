// Test Concat operation - comprehensive
module {
  // Concat 2 tensors along axis 0
  func.func @test_concat_2_axis0(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<8x8xf32> {
    %0 = google.concat %arg0, %arg1 {axis = 0} : tensor<4x8xf32>, tensor<4x8xf32> -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  // Concat 2 tensors along axis 1
  func.func @test_concat_2_axis1(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x16xf32> {
    %0 = google.concat %arg0, %arg1 {axis = 1} : tensor<4x8xf32>, tensor<4x8xf32> -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }

  // Concat 3 tensors
  func.func @test_concat_3(%arg0: tensor<2x8xf32>, %arg1: tensor<2x8xf32>, %arg2: tensor<2x8xf32>) -> tensor<6x8xf32> {
    %0 = google.concat %arg0, %arg1, %arg2 {axis = 0} : tensor<2x8xf32>, tensor<2x8xf32>, tensor<2x8xf32> -> tensor<6x8xf32>
    return %0 : tensor<6x8xf32>
  }

  // Concat 4 tensors
  func.func @test_concat_4(%arg0: tensor<1x8xf32>, %arg1: tensor<1x8xf32>, %arg2: tensor<1x8xf32>, %arg3: tensor<1x8xf32>) -> tensor<4x8xf32> {
    %0 = google.concat %arg0, %arg1, %arg2, %arg3 {axis = 0} : tensor<1x8xf32>, tensor<1x8xf32>, tensor<1x8xf32>, tensor<1x8xf32> -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}
