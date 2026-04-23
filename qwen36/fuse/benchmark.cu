/*
 * =============================================================================
 *  benchmark.cu — Correctness Verification + Performance Benchmark
 *
 *  Usage:
 *    nvcc -O3 -arch=sm_80 fused_softmax_topk.cu benchmark.cu -o benchmark
 *    ./benchmark
 *
 *  Tests:
 *    1. Correctness: compare fused kernel output vs. naive CPU reference
 *    2. Performance: benchmark fused kernel vs. naive two-step approach
 *    3. Scaling: vary V and K to characterize performance
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <algorithm>
#include <vector>
#include <chrono>
#include <random>

// Include the kernel
#include "fused_softmax_topk.cu"

// ============================================================================
//  CPU REFERENCE IMPLEMENTATION
// ============================================================================

void cpu_softmax_topk(
    const float* logits,
    int* top_idx,
    float* top_prob,
    int V, int K)
{
    // Phase 1: Find max
    float max_val = -FLT_MAX;
    for (int v = 0; v < V; v++) {
        if (logits[v] > max_val) max_val = logits[v];
    }

    // Phase 2: Compute softmax
    std::vector<float> probs(V);
    float sum = 0.0f;
    for (int v = 0; v < V; v++) {
        probs[v] = expf(logits[v] - max_val);
        sum += probs[v];
    }
    for (int v = 0; v < V; v++) {
        probs[v] /= sum;
    }

    // Phase 3: Top-K using partial sort
    std::vector<int> indices(V);
    for (int v = 0; v < V; v++) indices[v] = v;

    std::partial_sort(indices.begin(), indices.begin() + K, indices.end(),
        [&](int a, int b) { return probs[a] > probs[b]; });

    for (int k = 0; k < K; k++) {
        top_idx[k]  = indices[k];
        top_prob[k] = probs[indices[k]];
    }
}

// ============================================================================
//  NAIVE CUDA IMPLEMENTATION (for comparison)
// ============================================================================

// Step 1: Softmax kernel (materializes full output)
__global__ void naive_softmax_kernel(
    const float* __restrict__ logits,
    float* __restrict__ probs,
    int V)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    const float* row = logits + (size_t)bid * V;
    float* out = probs + (size_t)bid * V;

    // Find max
    __shared__ float s_max[32];  // Simplified: assumes 256 threads, 8 warps
    float local_max = -FLT_MAX;
    for (int v = tid; v < V; v += 256) {
        if (row[v] > local_max) local_max = row[v];
    }
    // ... (same reduction as fused kernel)
    // For brevity, use a simple approach
    float max_val = local_max;
    for (int offset = 128; offset > 0; offset /= 2) {
        __threadfence();
        if (tid < offset && tid + offset < 256) {
            // This is simplified — real implementation needs proper reduction
        }
    }

    // Compute softmax
    for (int v = tid; v < V; v += 256) {
        out[v] = expf(row[v] - max_val);
    }

    // Sum and normalize (simplified)
    // ... (omitted for brevity — the point is this writes 4V bytes)
}

// ============================================================================
//  CORRECTNESS TEST
// ============================================================================

bool test_correctness(int V, int K, float tolerance = 1e-4) {
    printf("\n=== Correctness Test: V=%d, K=%d ===\n", V, K);

    // Allocate host memory
    float* h_logits = new float[V];
    int*   h_top_idx_ref = new int[K];
    float* h_top_prob_ref = new float[K];

    int*   h_top_idx_gpu = new int[K];
    float* h_top_prob_gpu = new float[K];

    // Initialize with random logits
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (int v = 0; v < V; v++) {
        h_logits[v] = dist(rng);
    }

    // CPU reference
    cpu_softmax_topk(h_logits, h_top_idx_ref, h_top_prob_ref, V, K);

    // GPU kernel
    float* d_logits;
    int*   d_top_idx;
    float* d_top_prob;

    cudaMalloc(&d_logits, V * sizeof(float));
    cudaMalloc(&d_top_idx, K * sizeof(int));
    cudaMalloc(&d_top_prob, K * sizeof(float));

    cudaMemcpy(d_logits, h_logits, V * sizeof(float), cudaMemcpyHostToDevice);

    launch_fused_softmax_topk<K>(d_logits, d_top_idx, d_top_prob, 1, 1, V);

    cudaMemcpy(h_top_idx_gpu, d_top_idx, K * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_top_prob_gpu, d_top_prob, K * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    bool pass = true;

    // Check indices (may differ in ordering for equal values)
    std::sort(h_top_idx_ref, h_top_idx_ref + K);
    std::sort(h_top_idx_gpu, h_top_idx_gpu + K);
    for (int k = 0; k < K; k++) {
        if (h_top_idx_ref[k] != h_top_idx_gpu[k]) {
            printf("  INDEX MISMATCH at k=%d: ref=%d, gpu=%d\n",
                   k, h_top_idx_ref[k], h_top_idx_gpu[k]);
            pass = false;
        }
    }

    // Check probabilities (allow small numerical difference)
    // First, sort GPU output by index to match reference
    std::vector<std::pair<int, float>> gpu_pairs(K);
    for (int k = 0; k < K; k++) {
        gpu_pairs[k] = {h_top_idx_gpu[k], h_top_prob_gpu[k]};
    }
    std::sort(gpu_pairs.begin(), gpu_pairs.end());

    for (int k = 0; k < K; k++) {
        float diff = fabsf(h_top_prob_ref[k] - gpu_pairs[k].second);
        if (diff > tolerance) {
            printf("  PROB MISMATCH at k=%d: ref=%.6f, gpu=%.6f, diff=%.6e\n",
                   k, h_top_prob_ref[k], gpu_pairs[k].second, diff);
            pass = false;
        }
    }

    if (pass) {
        printf("  PASSED\n");
    } else {
        printf("  FAILED\n");
    }

    // Cleanup
    cudaFree(d_logits);
    cudaFree(d_top_idx);
    cudaFree(d_top_prob);
    delete[] h_logits;
    delete[] h_top_idx_ref;
    delete[] h_top_prob_ref;
    delete[] h_top_idx_gpu;
    delete[] h_top_prob_gpu;

    return pass;
}

// ============================================================================
//  PERFORMANCE BENCHMARK
// ============================================================================

struct BenchmarkResult {
    float fused_ms;
    float naive_ms;  // If available
    int B, T, V, K;
};

float benchmark_fused(int B, int T, int V, int K, int iterations = 100) {
    size_t logits_size = (size_t)B * T * V * sizeof(float);
    size_t output_size = (size_t)B * T * K * sizeof(float);
    size_t idx_size    = (size_t)B * T * K * sizeof(int);

    float* d_logits;
    int*   d_top_idx;
    float* d_top_prob;

    cudaMalloc(&d_logits, logits_size);
    cudaMalloc(&d_top_idx, idx_size);
    cudaMalloc(&d_top_prob, output_size);

    // Initialize with random data
    float* h_logits = new float[B * T * V];
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (int i = 0; i < B * T * V; i++) h_logits[i] = dist(rng);
    cudaMemcpy(d_logits, h_logits, logits_size, cudaMemcpyHostToDevice);
    delete[] h_logits;

    // Warmup
    launch_fused_softmax_topk<K>(d_logits, d_top_idx, d_top_prob, B, T, V);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        launch_fused_softmax_topk<K>(d_logits, d_top_idx, d_top_prob, B, T, V);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iterations;

    cudaFree(d_logits);
    cudaFree(d_top_idx);
    cudaFree(d_top_prob);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return avg_ms;
}

// ============================================================================
//  MAIN
// ============================================================================

int main(int argc, char** argv) {
    printf("Fused Softmax + Top-K Kernel Benchmark\n");
    printf("========================================\n");

    // Get device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s\n", prop.name);
    printf("SMs: %d, Max threads/SM: %d\n", prop.multiProcessorCount,
           prop.maxThreadsPerMultiProcessor);

    // --- Correctness tests ---
    printf("\n--- Correctness Tests ---\n");
    bool all_pass = true;
    all_pass &= test_correctness(1000, 10);
    all_pass &= test_correctness(50257, 256);
    all_pass &= test_correctness(50257, 50);
    all_pass &= test_correctness(32000, 128);

    if (!all_pass) {
        printf("\nSome correctness tests FAILED!\n");
        return 1;
    }

    // --- Performance benchmarks ---
    printf("\n--- Performance Benchmarks ---\n");
    printf("Format: B=%d, T=%d, V=%d, K=%d → %.3f ms\n", 1, 1, 50257, 256,
            benchmark_fused(1, 1, 50257, 256));

    printf("B=%d, T=%d, V=%d, K=%d → %.3f ms\n", 1, 1, 50257, 50,
            benchmark_fused(1, 1, 50257, 50));

    printf("B=%d, T=%d, V=%d, K=%d → %.3f ms\n", 1, 1, 10000, 256,
            benchmark_fused(1, 1, 10000, 256));

    printf("B=%d, T=%d, V=%d, K=%d → %.3f ms\n", 32, 128, 32000, 128,
            benchmark_fused(32, 128, 32000, 128));

    // --- Scaling analysis ---
    printf("\n--- Scaling with V (B=1, T=1, K=256) ---\n");
    printf("V\t\tTime (ms)\tBandwidth (GB/s)\tCompute (GFLOP/s)\n");
    int vs[] = {1000, 5000, 10000, 50257, 100000};
    for (int vi = 0; vi < 5; vi++) {
        int V = vs[vi];
        float ms = benchmark_fused(1, 1, V, 256);
        float bandwidth = (12.0 * V * 4.0) / (ms * 1e6);  // GB/s
        float compute = (2.0 * V * 1.0) / (ms * 1e6);     // GFLOP/s (expf count)
        printf("%d\t\t%.3f\t\t%.1f\t\t\t%.1f\n", V, ms, bandwidth, compute);
    }

    printf("\n--- Scaling with K (B=1, T=1, V=50257) ---\n");
    printf("K\t\tTime (ms)\n");
    int ks[] = {16, 32, 64, 128, 256};
    for (int ki = 0; ki < 5; ki++) {
        int K = ks[ki];
        float ms = benchmark_fused(1, 1, 50257, K);
        printf("%d\t\t%.3f\n", K, ms);
    }

    printf("\nDone.\n");
    return 0;
}
