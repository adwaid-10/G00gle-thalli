// RUN: google-opt %s | FileCheck %s
// This file contains only VALID operations that should pass verification

module {
  // ========== Valid Constant Operations ==========
  
  // CHECK-LABEL: func.func @test_constant_f32
  func.func @test_constant_f32() -> tensor<4xf32> {
    // CHECK: google.constant
    %0 = google.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @test_constant_i32
  func.func @test_constant_i32() -> tensor<3xi32> {
    // CHECK: google.constant
    %0 = google.constant {value = dense<[10, 20, 30]> : tensor<3xi32>} : tensor<3xi32>
    return %0 : tensor<3xi32>
  }

  // CHECK-LABEL: func.func @test_constant_2d
  func.func @test_constant_2d() -> tensor<2x3xf32> {
    // CHECK: google.constant
    %0 = google.constant {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>} : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  // ========== Valid Binary Operations ==========
  
  // CHECK-LABEL: func.func @test_add
  func.func @test_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: google.add
    %0 = google.add %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @test_max_min
  func.func @test_max_min(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
    // CHECK: google.max
    %0 = google.max %arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32> -> tensor<4x8xf32>
    // CHECK: google.min
    %1 = google.min %0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32> -> tensor<4x8xf32>
    return %1 : tensor<4x8xf32>
  }

  // ========== Valid Reduction Operations ==========
  
  // CHECK-LABEL: func.func @test_reduce_max_all
  func.func @test_reduce_max_all(%arg0: tensor<4x8xf32>) -> tensor<f32> {
    // Reduce all dimensions to scalar
    // CHECK: google.reduce<max>
    %0 = google.reduce<max> %arg0 : tensor<4x8xf32> -> tensor<f32>
    return %0 : tensor<f32>
  }

  // CHECK-LABEL: func.func @test_reduce_sum_axis0
  func.func @test_reduce_sum_axis0(%arg0: tensor<4x8xf32>) -> tensor<8xf32> {
    // Reduce along axis 0
    // CHECK: google.reduce<sum> %{{.*}} axes = [0]
    %0 = google.reduce<sum> %arg0 axes = [0] : tensor<4x8xf32> -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  // CHECK-LABEL: func.func @test_reduce_mean_axis1
  func.func @test_reduce_mean_axis1(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
    // Reduce along axis 1
    // CHECK: google.reduce<mean> %{{.*}} axes = [1]
    %0 = google.reduce<mean> %arg0 axes = [1] : tensor<4x8xf32> -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @test_reduce_keepdims
  func.func @test_reduce_keepdims(%arg0: tensor<4x8xf32>) -> tensor<4x1xf32> {
    // Reduce with keepdims = true
    // CHECK: google.reduce<min> %{{.*}} axes = [1] keepdims = true
    %0 = google.reduce<min> %arg0 axes = [1] keepdims = true : tensor<4x8xf32> -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }

  // CHECK-LABEL: func.func @test_reduce_product
  func.func @test_reduce_product(%arg0: tensor<4x8xf32>) -> tensor<f32> {
    // CHECK: google.reduce<product>
    %0 = google.reduce<product> %arg0 : tensor<4x8xf32> -> tensor<f32>
    return %0 : tensor<f32>
  }

  // CHECK-LABEL: func.func @test_reduce_argmax
  func.func @test_reduce_argmax(%arg0: tensor<4x8xf32>) -> tensor<4xi64> {
    // Argmax returns indices (i64)
    // CHECK: google.reduce<argmax> %{{.*}} axes = [1]
    %0 = google.reduce<argmax> %arg0 axes = [1] : tensor<4x8xf32> -> tensor<4xi64>
    return %0 : tensor<4xi64>
  }

  // CHECK-LABEL: func.func @test_reduce_argmin
  func.func @test_reduce_argmin(%arg0: tensor<4x8xf32>) -> tensor<8xi64> {
    // Argmin returns indices (i64)
    // CHECK: google.reduce<argmin> %{{.*}} axes = [0]
    %0 = google.reduce<argmin> %arg0 axes = [0] : tensor<4x8xf32> -> tensor<8xi64>
    return %0 : tensor<8xi64>
  }

  // ========== Valid 3D Tensor Operations ==========
  
  // CHECK-LABEL: func.func @test_3d_reduce_single_axis
  func.func @test_3d_reduce_single_axis(%arg0: tensor<2x4x8xf32>) -> tensor<2x8xf32> {
    // Reduce middle dimension
    // CHECK: google.reduce<max> %{{.*}} axes = [1]
    %0 = google.reduce<max> %arg0 axes = [1] : tensor<2x4x8xf32> -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @test_3d_reduce_multiple_axes
  func.func @test_3d_reduce_multiple_axes(%arg0: tensor<2x4x8xf32>) -> tensor<8xf32> {
    // Reduce first two dimensions
    // CHECK: google.reduce<sum> %{{.*}} axes = [0, 1]
    %0 = google.reduce<sum> %arg0 axes = [0, 1] : tensor<2x4x8xf32> -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  // CHECK-LABEL: func.func @test_3d_reduce_keepdims
  func.func @test_3d_reduce_keepdims(%arg0: tensor<2x4x8xf32>) -> tensor<2x1x8xf32> {
    // Reduce with keepdims on 3D tensor
    // CHECK: google.reduce<mean> %{{.*}} axes = [1] keepdims = true
    %0 = google.reduce<mean> %arg0 axes = [1] keepdims = true : tensor<2x4x8xf32> -> tensor<2x1x8xf32>
    return %0 : tensor<2x1x8xf32>
  }

  // CHECK-LABEL: func.func @test_3d_reduce_all_with_keepdims
  func.func @test_3d_reduce_all_with_keepdims(%arg0: tensor<2x4x8xf32>) -> tensor<1x1x1xf32> {
    // Reduce all dimensions with keepdims
    // CHECK: google.reduce<max> %{{.*}} keepdims = true
    %0 = google.reduce<max> %arg0 keepdims = true : tensor<2x4x8xf32> -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }

  // ========== Valid Complex Workflows ==========
  
  // CHECK-LABEL: func.func @test_workflow_with_constants
  func.func @test_workflow_with_constants() -> tensor<f32> {
    // CHECK: google.constant
    %0 = google.constant {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>} : tensor<2x2xf32>
    // CHECK: google.constant
    %1 = google.constant {value = dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>} : tensor<2x2xf32>
    // CHECK: google.add
    %2 = google.add %0, %1 : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    // CHECK: google.reduce<sum>
    %3 = google.reduce<sum> %2 : tensor<2x2xf32> -> tensor<f32>
    return %3 : tensor<f32>
  }

  // CHECK-LABEL: func.func @test_chained_reductions
  func.func @test_chained_reductions(%arg0: tensor<4x8xf32>) -> tensor<f32> {
    // First reduce along one axis
    // CHECK: google.reduce<max> %{{.*}} axes = [0]
    %0 = google.reduce<max> %arg0 axes = [0] : tensor<4x8xf32> -> tensor<8xf32>
    // Then reduce remaining dimension
    // CHECK: google.reduce<mean>
    %1 = google.reduce<mean> %0 : tensor<8xf32> -> tensor<f32>
    return %1 : tensor<f32>
  }

  // CHECK-LABEL: func.func @test_ml_loss_computation
  func.func @test_ml_loss_computation(%predictions: tensor<32x10xf32>, %targets: tensor<32x10xf32>) -> tensor<f32> {
    // Mean squared error: mean((predictions - targets)^2)
    // CHECK: google.constant
    %cst = google.constant {value = dense<2.0> : tensor<32x10xf32>} : tensor<32x10xf32>
    // CHECK: google.max
    %diff = google.max %predictions, %targets : tensor<32x10xf32>, tensor<32x10xf32> -> tensor<32x10xf32>
    // CHECK: google.reduce<mean>
    %loss = google.reduce<mean> %diff : tensor<32x10xf32> -> tensor<f32>
    return %loss : tensor<f32>
  }
}