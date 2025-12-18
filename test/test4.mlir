// RUN: google-opt %s | FileCheck %s
// This file contains only VALID operations that should pass verification

module {
  
  // CHECK-LABEL: func.func @test_max_min
  func.func @test_max_min(%arg0: tensor<4x8xi32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
    // CHECK: google.max
    %0 = google.max %arg0, %arg1 : tensor<4x8xi32>, tensor<4x8xf32> -> tensor<4x8xf32>
    // CHECK: google.min
    %1 = google.min %0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32> -> tensor<4x8xf32>
    return %1 : tensor<4x8xf32>
  }

  
}