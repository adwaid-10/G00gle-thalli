// RUN: google-opt %s -verify-diagnostics -split-input-file
// This file contains INVALID operations that should FAIL verification
// Each test is separated by 

// ========== Invalid Constant Operations ==========

// -----
// Test: Constant value type doesn't match output type
func.func @invalid_constant_type_mismatch() -> tensor<4xf32> {
  %0 = google.constant {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// Test: Constant shape mismatch
func.func @invalid_constant_shape_mismatch() -> tensor<4xf32> {
  // expected-error@+1 {{value attribute type 'tensor<3xf32>' does not match output type 'tensor<4xf32>'}}
  %0 = google.constant {value = dense<[1.0, 2.0, 3.0]> : tensor<3xf32>} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// ========== Invalid Reduction Operations ==========

// -----
// Test: Axis out of range (too high)
func.func @invalid_reduce_axis_too_high(%arg0: tensor<4x8xf32>) -> tensor<f32> {
  // expected-error@+1 {{axis 5 is out of range [0, 2)}}
  %0 = google.reduce<max> %arg0 axes = [1] : tensor<4x8xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----
// Test: Axis out of range (negative)
func.func @invalid_reduce_axis_negative(%arg0: tensor<4x8xf32>) -> tensor<f32> {
  // expected-error@+1 {{axis -1 is out of range [0, 2)}}
  %0 = google.reduce<sum> %arg0 axes = [1] : tensor<4x8xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----
// Test: Duplicate axes
func.func @invalid_reduce_duplicate_axes(%arg0: tensor<4x8xf32>) -> tensor<f32> {
  // expected-error@+1 {{duplicate axis found in axes attribute}}
  %0 = google.reduce<max> %arg0 axes = [0, 0] : tensor<4x8xf32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----
// Test: Duplicate axes (different order)
func.func @invalid_reduce_duplicate_axes_2(%arg0: tensor<2x4x8xf32>) -> tensor<8xf32> {
  // expected-error@+1 {{duplicate axis found in axes attribute}}
  %0 = google.reduce<sum> %arg0 axes = [0, 1, 0] : tensor<2x4x8xf32> -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
// Test: Argmax with wrong output element type (f32 instead of i64)
func.func @invalid_argmax_wrong_output_type(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{argmax/argmin output must have i64 element type, got 'f32'}}
  %0 = google.reduce<argmax> %arg0 axes = [1] : tensor<4x8xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// Test: Argmin with wrong output element type (i32 instead of i64)
func.func @invalid_argmin_wrong_output_type(%arg0: tensor<4x8xf32>) -> tensor<8xi32> {
  // expected-error@+1 {{argmax/argmin output must have i64 element type, got 'i32'}}
  %0 = google.reduce<argmin> %arg0 axes = [0] : tensor<4x8xf32> -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// -----
// Test: Element type mismatch for non-arg reduction
func.func @invalid_reduce_element_type_mismatch(%arg0: tensor<4x8xf32>) -> tensor<f64> {
  // expected-error@+1 {{input and output element types must match for non-arg reductions, got input 'f32' and output 'f64'}}
  %0 = google.reduce<max> %arg0 : tensor<4x8xf32> -> tensor<f64>
  return %0 : tensor<f64>
}

// -----
// Test: Wrong output shape (expected 8, got 4)
func.func @invalid_reduce_wrong_output_shape(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{output shape mismatch at dimension 0, expected 8 got 4}}
  %0 = google.reduce<sum> %arg0 axes = [0] : tensor<4x8xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// Test: Wrong output rank (should be scalar)
func.func @invalid_reduce_wrong_output_rank(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{output rank does not match expected rank, expected 0 got 1}}
  %0 = google.reduce<max> %arg0 : tensor<4x8xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// Test: Keepdims with wrong shape
func.func @invalid_reduce_keepdims_wrong_shape(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error@+1 {{output shape mismatch at dimension 1, expected 1 got 8}}
  %0 = google.reduce<mean> %arg0 axes = [1] keepdims = true : tensor<4x8xf32> -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----
// Test: Reduce all with keepdims but wrong output shape
func.func @invalid_reduce_all_keepdims_wrong_shape(%arg0: tensor<4x8xf32>) -> tensor<1xf32> {
  // expected-error@+1 {{output rank does not match expected rank, expected 2 got 1}}
  %0 = google.reduce<sum> %arg0 keepdims = true : tensor<4x8xf32> -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// ========== Invalid 3D Tensor Reductions ==========

// -----
// Test: 3D tensor with axis out of range
func.func @invalid_3d_axis_out_of_range(%arg0: tensor<2x4x8xf32>) -> tensor<2x8xf32> {
  // expected-error@+1 {{axis  is out of range [0, 3)}}
  %0 = google.reduce<max> %arg0 axes = [3] : tensor<2x4x8xf32> -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

// -----
// Test: 3D tensor with wrong output shape after multi-axis reduction
func.func @invalid_3d_wrong_output_shape(%arg0: tensor<2x4x8xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{output shape mismatch at dimension 0, expected 8 got 4}}
  %0 = google.reduce<sum> %arg0 axes = [0, 1] : tensor<2x4x8xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// Test: 3D tensor keepdims with wrong shape
func.func @invalid_3d_keepdims_wrong_shape(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x1xf32> {
  // expected-error@+1 {{output shape mismatch at dimension 1, expected 1 got 4}}
  %0 = google.reduce<max> %arg0 axes = [1] keepdims = true : tensor<2x4x8xf32> -> tensor<2x4x1xf32>
  return %0 : tensor<2x4x1xf32>
}

// ========== Invalid Binary Operations (if verifiers exist) ==========

// -----
// Test: Shape mismatch in binary operation
func.func @invalid_add_shape_mismatch(%arg0: tensor<4xf32>, %arg1: tensor<8xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{operand shapes must match}}
  %0 = google.add %arg0, %arg1 : tensor<4xf32>, tensor<8xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// Test: Element type mismatch in binary operation
func.func @invalid_max_type_mismatch(%arg0: tensor<4xf32>, %arg1: tensor<4xi32>) -> tensor<4xf32> {
  // expected-error@+1 {{operand element types must match}}
  %0 = google.max %arg0, %arg1 : tensor<4xf32>, tensor<4xi32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// Test: Result shape doesn't match operand shapes
func.func @invalid_min_result_shape(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<8xf32> {
  // expected-error@+1 {{result shape must match operand shapes}}
  %0 = google.min %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// ========== Complex Invalid Cases ==========

// -----
// Test: Chained operations with type error
func.func @invalid_chained_type_error(%arg0: tensor<4x8xf32>) -> tensor<f64> {
  // Reduce to tensor<8xf32>
  %0 = google.reduce<max> %arg0 axes = [0] : tensor<4x8xf32> -> tensor<8xf32>
  // expected-error@+1 {{input and output element types must match for non-arg reductions, got input 'f32' and output 'f64'}}
  %1 = google.reduce<sum> %0 : tensor<8xf32> -> tensor<f64>
  return %1 : tensor<f64>
}

// -----
// Test: Multiple errors (only first will be caught)
func.func @invalid_multiple_errors(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{axis 10 is out of range [0, 2)}}
  %0 = google.reduce<max> %arg0 axes = [10] : tensor<4x8xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}