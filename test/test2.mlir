
module {
  // CHECK-LABEL: func.func @test_constants_and_add
  func.func @test_constants_and_add() -> tensor<5xf32> {
    // Create constant tensors
    %cst1 = google.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : tensor<4xf32>
    %cst2 = google.constant {value = dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>} : tensor<4xf32>

    // Perform element-wise addition
    // CHECK: google.add
    %add_result = google.add %cst1, %cst2 : tensor<4xf32>, tensor<4xf32> -> tensor<5xf32>

    return %add_result : tensor<5xf32>
  }

}