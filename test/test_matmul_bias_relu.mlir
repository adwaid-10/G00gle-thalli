// MatMul + Bias + ReLU Examples - Common Neural Network Pattern
// Testing with different dimensions and data types

module {
  //===----------------------------------------------------------------------===//
  // Example 1: Simple Linear Layer (f32)
  // Input: [batch=32, in_features=128]
  // Weight: [in_features=128, out_features=64]
  // Bias: [out_features=64]
  // Output: [batch=32, out_features=64]
  //===----------------------------------------------------------------------===//
  
  func.func @linear_layer_f32(%input: tensor<32x128xf32>, 
                               %weight: tensor<128x64xf32>, 
                               %bias: tensor<64xf32>) -> tensor<32x64xf32> {
    // MatMul: [32x128] @ [128x64] = [32x64]
    %matmul = google.matmul %input, %weight : tensor<32x128xf32>, tensor<128x64xf32> -> tensor<32x64xf32>
    
    // Broadcast bias: [64] -> [32x64]
    %bias_broadcast = google.broadcast %bias {shape = [32, 64]} : tensor<64xf32> -> tensor<32x64xf32>
    
    // Add bias: [32x64] + [32x64] = [32x64]
    %biased = google.add %matmul, %bias_broadcast : tensor<32x64xf32>, tensor<32x64xf32> -> tensor<32x64xf32>
    
    // ReLU activation: max(x, 0)
    %output = google.relu %biased : tensor<32x64xf32> -> tensor<32x64xf32>
    
    return %output : tensor<32x64xf32>
  }

  //===----------------------------------------------------------------------===//
  // Example 2: Large Batch Linear Layer (f32)
  // Input: [batch=256, in_features=512]
  // Weight: [in_features=512, out_features=256]
  // Bias: [out_features=256]
  // Output: [batch=256, out_features=256]
  //===----------------------------------------------------------------------===//
  
  func.func @large_linear_layer_f32(%input: tensor<256x512xf32>, 
                                     %weight: tensor<512x256xf32>, 
                                     %bias: tensor<256xf32>) -> tensor<256x256xf32> {
    %matmul = google.matmul %input, %weight : tensor<256x512xf32>, tensor<512x256xf32> -> tensor<256x256xf32>
    %bias_broadcast = google.broadcast %bias {shape = [256, 256]} : tensor<256xf32> -> tensor<256x256xf32>
    %biased = google.add %matmul, %bias_broadcast : tensor<256x256xf32>, tensor<256x256xf32> -> tensor<256x256xf32>
    %output = google.relu %biased : tensor<256x256xf32> -> tensor<256x256xf32>
    return %output : tensor<256x256xf32>
  }

  //===----------------------------------------------------------------------===//
  // Example 3: Small Batch Linear Layer (f32)
  // Input: [batch=1, in_features=784] (single image, e.g., MNIST)
  // Weight: [in_features=784, out_features=128]
  // Bias: [out_features=128]
  // Output: [batch=1, out_features=128]
  //===----------------------------------------------------------------------===//
  
  func.func @small_batch_linear_f32(%input: tensor<1x784xf32>, 
                                     %weight: tensor<784x128xf32>, 
                                     %bias: tensor<128xf32>) -> tensor<1x128xf32> {
    %matmul = google.matmul %input, %weight : tensor<1x784xf32>, tensor<784x128xf32> -> tensor<1x128xf32>
    %bias_broadcast = google.broadcast %bias {shape = [1, 128]} : tensor<128xf32> -> tensor<1x128xf32>
    %biased = google.add %matmul, %bias_broadcast : tensor<1x128xf32>, tensor<1x128xf32> -> tensor<1x128xf32>
    %output = google.relu %biased : tensor<1x128xf32> -> tensor<1x128xf32>
    return %output : tensor<1x128xf32>
  }

  //===----------------------------------------------------------------------===//
  // Example 4: Wide Linear Layer (f32)
  // Input: [batch=64, in_features=256]
  // Weight: [in_features=256, out_features=1024] (expansion)
  // Bias: [out_features=1024]
  // Output: [batch=64, out_features=1024]
  //===----------------------------------------------------------------------===//
  
  func.func @wide_linear_layer_f32(%input: tensor<64x256xf32>, 
                                    %weight: tensor<256x1024xf32>, 
                                    %bias: tensor<1024xf32>) -> tensor<64x1024xf32> {
    %matmul = google.matmul %input, %weight : tensor<64x256xf32>, tensor<256x1024xf32> -> tensor<64x1024xf32>
    %bias_broadcast = google.broadcast %bias {shape = [64, 1024]} : tensor<1024xf32> -> tensor<64x1024xf32>
    %biased = google.add %matmul, %bias_broadcast : tensor<64x1024xf32>, tensor<64x1024xf32> -> tensor<64x1024xf32>
    %output = google.relu %biased : tensor<64x1024xf32> -> tensor<64x1024xf32>
    return %output : tensor<64x1024xf32>
  }

  //===----------------------------------------------------------------------===//
  // Example 5: Narrow Linear Layer (f32)
  // Input: [batch=128, in_features=1024]
  // Weight: [in_features=1024, out_features=64] (compression)
  // Bias: [out_features=64]
  // Output: [batch=128, out_features=64]
  //===----------------------------------------------------------------------===//
  
  func.func @narrow_linear_layer_f32(%input: tensor<128x1024xf32>, 
                                      %weight: tensor<1024x64xf32>, 
                                      %bias: tensor<64xf32>) -> tensor<128x64xf32> {
    %matmul = google.matmul %input, %weight : tensor<128x1024xf32>, tensor<1024x64xf32> -> tensor<128x64xf32>
    %bias_broadcast = google.broadcast %bias {shape = [128, 64]} : tensor<64xf32> -> tensor<128x64xf32>
    %biased = google.add %matmul, %bias_broadcast : tensor<128x64xf32>, tensor<128x64xf32> -> tensor<128x64xf32>
    %output = google.relu %biased : tensor<128x64xf32> -> tensor<128x64xf32>
    return %output : tensor<128x64xf32>
  }

  //===----------------------------------------------------------------------===//
  // Example 6: Multi-Layer Network (f32)
  // Demonstrates chaining multiple linear layers
  //===----------------------------------------------------------------------===//
  
  func.func @multi_layer_network(%input: tensor<32x128xf32>, 
                                  %w1: tensor<128x256xf32>, %b1: tensor<256xf32>,
                                  %w2: tensor<256x128xf32>, %b2: tensor<128xf32>,
                                  %w3: tensor<128x10xf32>, %b3: tensor<10xf32>) -> tensor<32x10xf32> {
    // Layer 1: 128 -> 256
    %mm1 = google.matmul %input, %w1 : tensor<32x128xf32>, tensor<128x256xf32> -> tensor<32x256xf32>
    %bb1 = google.broadcast %b1 {shape = [32, 256]} : tensor<256xf32> -> tensor<32x256xf32>
    %add1 = google.add %mm1, %bb1 : tensor<32x256xf32>, tensor<32x256xf32> -> tensor<32x256xf32>
    %relu1 = google.relu %add1 : tensor<32x256xf32> -> tensor<32x256xf32>
    
    // Layer 2: 256 -> 128
    %mm2 = google.matmul %relu1, %w2 : tensor<32x256xf32>, tensor<256x128xf32> -> tensor<32x128xf32>
    %bb2 = google.broadcast %b2 {shape = [32, 128]} : tensor<128xf32> -> tensor<32x128xf32>
    %add2 = google.add %mm2, %bb2 : tensor<32x128xf32>, tensor<32x128xf32> -> tensor<32x128xf32>
    %relu2 = google.relu %add2 : tensor<32x128xf32> -> tensor<32x128xf32>
    
    // Layer 3: 128 -> 10 (output layer, no ReLU)
    %mm3 = google.matmul %relu2, %w3 : tensor<32x128xf32>, tensor<128x10xf32> -> tensor<32x10xf32>
    %bb3 = google.broadcast %b3 {shape = [32, 10]} : tensor<10xf32> -> tensor<32x10xf32>
    %output = google.add %mm3, %bb3 : tensor<32x10xf32>, tensor<32x10xf32> -> tensor<32x10xf32>
    
    return %output : tensor<32x10xf32>
  }

  //===----------------------------------------------------------------------===//
  // Example 7: Irregular Dimensions (f32)
  // Testing with non-power-of-2 dimensions
  //===----------------------------------------------------------------------===//
  
  func.func @irregular_linear_layer(%input: tensor<17x97xf32>, 
                                     %weight: tensor<97x53xf32>, 
                                     %bias: tensor<53xf32>) -> tensor<17x53xf32> {
    %matmul = google.matmul %input, %weight : tensor<17x97xf32>, tensor<97x53xf32> -> tensor<17x53xf32>
    %bias_broadcast = google.broadcast %bias {shape = [17, 53]} : tensor<53xf32> -> tensor<17x53xf32>
    %biased = google.add %matmul, %bias_broadcast : tensor<17x53xf32>, tensor<17x53xf32> -> tensor<17x53xf32>
    %output = google.relu %biased : tensor<17x53xf32> -> tensor<17x53xf32>
    return %output : tensor<17x53xf32>
  }

  //===----------------------------------------------------------------------===//
  // Example 8: With Other Activations (f32)
  // Using different activation functions
  //===----------------------------------------------------------------------===//
  
  func.func @linear_with_sigmoid(%input: tensor<32x128xf32>, 
                                  %weight: tensor<128x64xf32>, 
                                  %bias: tensor<64xf32>) -> tensor<32x64xf32> {
    %matmul = google.matmul %input, %weight : tensor<32x128xf32>, tensor<128x64xf32> -> tensor<32x64xf32>
    %bias_broadcast = google.broadcast %bias {shape = [32, 64]} : tensor<64xf32> -> tensor<32x64xf32>
    %biased = google.add %matmul, %bias_broadcast : tensor<32x64xf32>, tensor<32x64xf32> -> tensor<32x64xf32>
    %output = google.sigmoid %biased : tensor<32x64xf32> -> tensor<32x64xf32>
    return %output : tensor<32x64xf32>
  }

  func.func @linear_with_gelu(%input: tensor<32x128xf32>, 
                               %weight: tensor<128x64xf32>, 
                               %bias: tensor<64xf32>) -> tensor<32x64xf32> {
    %matmul = google.matmul %input, %weight : tensor<32x128xf32>, tensor<128x64xf32> -> tensor<32x64xf32>
    %bias_broadcast = google.broadcast %bias {shape = [32, 64]} : tensor<64xf32> -> tensor<32x64xf32>
    %biased = google.add %matmul, %bias_broadcast : tensor<32x64xf32>, tensor<32x64xf32> -> tensor<32x64xf32>
    %output = google.gelu %biased : tensor<32x64xf32> -> tensor<32x64xf32>
    return %output : tensor<32x64xf32>
  }

  func.func @linear_with_tanh(%input: tensor<32x128xf32>, 
                               %weight: tensor<128x64xf32>, 
                               %bias: tensor<64xf32>) -> tensor<32x64xf32> {
    %matmul = google.matmul %input, %weight : tensor<32x128xf32>, tensor<128x64xf32> -> tensor<32x64xf32>
    %bias_broadcast = google.broadcast %bias {shape = [32, 64]} : tensor<64xf32> -> tensor<32x64xf32>
    %biased = google.add %matmul, %bias_broadcast : tensor<32x64xf32>, tensor<32x64xf32> -> tensor<32x64xf32>
    %output = google.tanh %biased : tensor<32x64xf32> -> tensor<32x64xf32>
    return %output : tensor<32x64xf32>
  }
}

// Summary:
// - Example 1: Standard linear layer (32x128 -> 32x64)
// - Example 2: Large batch (256x512 -> 256x256)
// - Example 3: Single sample (1x784 -> 1x128)
// - Example 4: Wide expansion (64x256 -> 64x1024)
// - Example 5: Narrow compression (128x1024 -> 128x64)
// - Example 6: Multi-layer network (3 layers)
// - Example 7: Irregular dimensions (17x97 -> 17x53)
// - Example 8: Different activations (sigmoid, gelu, tanh)
