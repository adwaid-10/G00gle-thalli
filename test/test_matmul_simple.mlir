// Simple MatMul test without Transform dialect for end-to-end verification
module {
  func.func @matmul_simple(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = tensor.empty() : tensor<256x256xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<256x256xf32>, tensor<256x256xf32>)
                        outs(%1 : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %2 : tensor<256x256xf32>
  }
}
