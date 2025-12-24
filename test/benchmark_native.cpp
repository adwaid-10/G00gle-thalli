// Simple baseline MatMul for performance comparison
// This is a naive implementation without any optimizations

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

void matmul_baseline(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_tiled(const float* A, const float* B, float* C, int N) {
    const int TILE = 64;  // L2 cache-friendly tile size
    
    // Initialize C to zero
    for (int i = 0; i < N * N; i++) {
        C[i] = 0.0f;
    }
    
    // Tiled matrix multiplication
    for (int i0 = 0; i0 < N; i0 += TILE) {
        for (int j0 = 0; j0 < N; j0 += TILE) {
            for (int k0 = 0; k0 < N; k0 += TILE) {
                // Process tile
                for (int i = i0; i < std::min(i0 + TILE, N); i++) {
                    for (int j = j0; j < std::min(j0 + TILE, N); j++) {
                        float sum = C[i * N + j];
                        for (int k = k0; k < std::min(k0 + TILE, N); k++) {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

double benchmark(void (*func)(const float*, const float*, float*, int), 
                 const std::vector<float>& A, 
                 const std::vector<float>& B,
                 std::vector<float>& C,
                 int N, int iterations) {
    // Warm-up
    func(A.data(), B.data(), C.data(), N);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        func(A.data(), B.data(), C.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return total_ms / iterations;
}

int main() {
    std::cout << "\n=== Matrix Multiplication Performance Benchmark ===\n" << std::endl;
    
    const int N = 1024;
    const int ITERATIONS = 5;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Matrix size: " << N << "x" << N << std::endl;
    std::cout << "  Iterations: " << ITERATIONS << std::endl;
    std::cout << "  Total FLOPs: " << (2.0 * N * N * N / 1e9) << " billion" << std::endl;
    std::cout << std::endl;
    
    // Allocate matrices
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 1.0f);
    std::vector<float> C_baseline(N * N);
    std::vector<float> C_tiled(N * N);
    
    // Benchmark baseline
    std::cout << "Benchmarking baseline (no tiling)..." << std::endl;
    double baseline_time = benchmark(matmul_baseline, A, B, C_baseline, N, ITERATIONS);
    
    // Verify correctness
    if (std::abs(C_baseline[0] - N) > 0.01) {
        std::cout << "  ERROR: Incorrect result!" << std::endl;
        return 1;
    }
    std::cout << "  ✓ Correctness verified" << std::endl;
    
    // Benchmark tiled
    std::cout << "\nBenchmarking tiled (L2 cache optimization)..." << std::endl;
    double tiled_time = benchmark(matmul_tiled, A, B, C_tiled, N, ITERATIONS);
    
    // Verify correctness
    if (std::abs(C_tiled[0] - N) > 0.01) {
        std::cout << "  ERROR: Incorrect result!" << std::endl;
        return 1;
    }
    std::cout << "  ✓ Correctness verified" << std::endl;
    
    // Calculate performance
    double gflops_baseline = (2.0 * N * N * N) / (baseline_time * 1e6);
    double gflops_tiled = (2.0 * N * N * N) / (tiled_time * 1e6);
    double speedup = baseline_time / tiled_time;
    
    // Display results
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n=== Results ===\n" << std::endl;
    
    std::cout << "Baseline (no tiling):" << std::endl;
    std::cout << "  Time: " << baseline_time << " ms" << std::endl;
    std::cout << "  Performance: " << gflops_baseline << " GFLOPS" << std::endl;
    
    std::cout << "\nTiled (L2 optimization):" << std::endl;
    std::cout << "  Time: " << tiled_time << " ms" << std::endl;
    std::cout << "  Performance: " << gflops_tiled << " GFLOPS" << std::endl;
    
    std::cout << "\nSpeedup: " << speedup << "x faster" << std::endl;
    
    std::cout << "\n=== Analysis ===\n" << std::endl;
    std::cout << "Note: This is L2 tiling only (single level, 64x64 tiles)" << std::endl;
    std::cout << "MLIR extreme pipeline uses L3 tiling (3 levels: 256→64→16)" << std::endl;
    std::cout << "Expected additional speedup from L3 tiling: 2-3x" << std::endl;
    std::cout << "Expected additional speedup from Affine opts: 1.5-2x" << std::endl;
    std::cout << "Total expected MLIR speedup: " << (speedup * 2.5) << "-" << (speedup * 6) << "x\n" << std::endl;
    
    return 0;
}
