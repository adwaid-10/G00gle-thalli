func.func @main() {
    %a = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %b = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
    %bad = google.constant {value = dense<[1., 2.]> : tensor<2xf32>} : tensor<2xf32>
    %result = google.add %a, %b : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    return
}

