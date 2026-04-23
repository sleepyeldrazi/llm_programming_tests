// =============================================================================
// Test / Benchmark: Fused Softmax + Top-K
// =============================================================================
// Compile:
//   nvcc -O3 -arch=sm_80 -o test_fused test_fused.cu fused_softmax_topk.cuh
//
// Run:
//   ./test_fused
// =============================================================================

#include "fused_softmax_topk.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

// ---------- CPU reference implementation ----------

void cpu_softmax_topk(const float* logits, int* indices, float* probs,
                      int B, int T, int V, int K) {
    for (int bt = 0; bt < B * T; bt++) {
        const float* row = logits + bt * V;
        int*   out_idx   = indices + bt * K;
        float* out_prob  = probs   + bt * K;

        // Numerically stable softmax
        float max_val = *std::max_element(row, row + V);
        float sum = 0.0f;
        std::vector<float> exp_vals(V);
        for (int v = 0; v < V; v++) {
            exp_vals[v] = expf(row[v] - max_val);
            sum += exp_vals[v];
        }
        float inv_sum = 1.0f / sum;
        for (int v = 0; v < V; v++) {
            exp_vals[v] *= inv_sum;
        }

        // Top-K by sorting (simple but correct)
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                          [&](int a, int b) { return exp_vals[a] > exp_vals[b]; });

        for (int k = 0; k < K; k++) {
            out_idx[k]  = idx[k];
            out_prob[k] = exp_vals[idx[k]];
        }
    }
}

// ---------- Verification ----------

bool verify(const float* ref_probs, const int* ref_idx,
            const float* gpu_probs, const int* gpu_idx,
            int B, int T, int K, float tol = 1e-4f) {
    bool ok = true;
    int failures = 0;
    for (int bt = 0; bt < B * T && failures < 10; bt++) {
        for (int k = 0; k < K; k++) {
            int ri = ref_idx[bt * K + k];
            int gi = gpu_idx[bt * K + k];
            float rp = ref_probs[bt * K + k];
            float gp = gpu_probs[bt * K + k];

            // Index must match (probabilities might have ties, but for random data they won't)
            if (ri != gi) {
                // Check if probability is close (might be a tie)
                if (fabsf(rp - gp) > tol) {
                    printf("FAIL [bt=%d, k=%d]: ref_idx=%d gpu_idx=%d  ref_prob=%.8f gpu_prob=%.8f\n",
                           bt, k, ri, gi, rp, gp);
                    ok = false;
                    failures++;
                }
            }

            // Probability must match
            if (fabsf(rp - gp) > tol) {
                printf("FAIL [bt=%d, k=%d]: idx=%d  ref_prob=%.8f gpu_prob=%.8f  diff=%.2e\n",
                       bt, k, gi, rp, gp, fabsf(rp - gp));
                ok = false;
                failures++;
            }
        }
    }
    return ok;
}

// ---------- Main ----------

int main() {
    constexpr int B = 4;
    constexpr int T = 8;
    constexpr int V = 1024;    // manageable for CPU verification
    constexpr int K = 10;
    constexpr int N = B * T;

    printf("=== Fused Softmax + Top-K Test ===\n");
    printf("Shape: [B=%d, T=%d, V=%d], K=%d\n\n", B, T, V, K);

    // Allocate and initialize
    size_t logits_bytes = (size_t)N * V * sizeof(float);
    size_t idx_bytes    = (size_t)N * K * sizeof(int);
    size_t prob_bytes   = (size_t)N * K * sizeof(float);

    std::vector<float> h_logits(N * V);
    std::vector<int>   h_idx_gpu(N * K);
    std::vector<float> h_prob_gpu(N * K);
    std::vector<int>   h_idx_ref(N * K);
    std::vector<float> h_prob_ref(N * K);

    // Random logits with large range to stress numerical stability
    srand(42);
    for (auto& x : h_logits) {
        x = ((float)rand() / RAND_MAX - 0.5f) * 40.0f;  // range [-20, 20]
    }

    // GPU allocation
    float *d_logits, *d_probs;
    int   *d_indices;
    cudaMalloc(&d_logits,  logits_bytes);
    cudaMalloc(&d_indices, idx_bytes);
    cudaMalloc(&d_probs,   prob_bytes);

    cudaMemcpy(d_logits, h_logits.data(), logits_bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Launching fused kernel...\n");
    cudaEventRecord(start);
    launch_fused_softmax_topk<K>(d_logits, d_indices, d_probs, B, T, V);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %.3f ms\n\n", ms);

    // Copy results back
    cudaMemcpy(h_idx_gpu.data(),  d_indices, idx_bytes,  cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prob_gpu.data(), d_probs,   prob_bytes, cudaMemcpyDeviceToHost);

    // CPU reference
    printf("Running CPU reference...\n");
    cpu_softmax_topk(h_logits.data(), h_idx_ref.data(), h_prob_ref.data(),
                     B, T, V, K);

    // Verify
    printf("Verifying...\n");
    bool pass = verify(h_prob_ref.data(), h_idx_ref.data(),
                       h_prob_gpu.data(), h_idx_gpu.data(),
                       B, T, K);

    printf("\n%s\n", pass ? "✓ ALL TESTS PASSED" : "✗ TESTS FAILED");

    // Print a sample row
    int row = 0;
    printf("\nSample output (row %d):\n", row);
    printf("  %-6s %-12s %-12s %-12s\n", "k", "Index", "GPU Prob", "Ref Prob");
    printf("  %-6s %-12s %-12s %-12s\n", "---", "-----", "--------", "--------");
    for (int k = 0; k < K; k++) {
        printf("  %-6d %-12d %-12.8f %-12.8f\n", k,
               h_idx_gpu[row * K + k],
               h_prob_gpu[row * K + k],
               h_prob_ref[row * K + k]);
    }

    // Check probability sums
    float sum_gpu = 0, sum_ref = 0;
    for (int k = 0; k < K; k++) {
        sum_gpu += h_prob_gpu[row * K + k];
        sum_ref += h_prob_ref[row * K + k];
    }
    printf("\n  Sum of top-%d probs: GPU=%.8f  Ref=%.8f\n", K, sum_gpu, sum_ref);
    printf("  (Note: sum < 1.0 because K << V; these should match)\n");

    // Bandwidth estimate
    size_t total_read  = logits_bytes;
    size_t total_write = idx_bytes + prob_bytes;
    double bw = (total_read + total_write) / (ms * 1e-3) / 1e9;
    printf("\nEstimated effective bandwidth: %.1f GB/s\n", bw);
    printf("  Reads:  %zu bytes (%.1f KB)\n", total_read, total_read / 1024.0);
    printf("  Writes: %zu bytes (%.1f KB)\n", total_write, total_write / 1024.0);

    // Cleanup
    cudaFree(d_logits);
    cudaFree(d_indices);
    cudaFree(d_probs);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return pass ? 0 : 1;
}
