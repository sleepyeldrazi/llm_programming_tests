/*
 * =============================================================================
 *  fused_softmax_topk.cu
 *  High-Performance Fused Softmax + Top-K Kernel for LLM Sampling
 *
 *  Input:  logits[B, T, V]   (row-major, float32)
 *  Output: top_idx[B, T, K], top_prob[B, T, K]
 *
 *  Key properties:
 *    • ZERO global memory writes for intermediate softmax values
 *    • Numerically stable via log-sum-exp (max subtraction)
 *    • Warp-level shuffle reductions (no shared memory for reductions)
 *    • Shared-memory min-heap for top-K selection
 *    • Grid-stride loops handle V up to millions
 *    • Dynamic shared memory staging for warp-to-warp merge
 *
 *  Typical usage: B=1, T=1, V=50257 (LLaMA), K=256
 *    → 1 block, 256 threads, ~200 iterations of grid-stride loop
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
//  §1  CONFIGURATION
// ============================================================================

constexpr int BLOCK_THREADS = 256;
constexpr int WARP_SIZE     = 32;
constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE;  // 8

// Per-thread local top-K buffer size.
// Constraint: LOCAL_K * BLOCK_THREADS >= K (enough candidates for merge).
// For K=256: LOCAL_K=16 → 4096 candidates, plenty of oversampling.
constexpr int LOCAL_K = 16;

// ============================================================================
//  §2  WARP-LEVEL PRIMITIVES
//
 *  All use __shfl_xor_sync / __shfl_up_sync — zero shared memory,
 *  zero global memory. Pure register operations within a warp.
 *
 *  Butterfly (xor) reduction pattern:
 *    Step 0:  [0↔16, 1↔17, ..., 15↔31, 32↔48, ...]
 *    Step 1:  [0↔8,  1↔9,  ..., 7↔15,  ...]
 *    Step 2:  [0↔4,  1↔5,  ..., 3↔7,   ...]
 *    Step 3:  [0↔2,  1↔3,  ..., 5↔7,   ...]
 *    Step 4:  [0↔1,  2↔3,  ..., 6↔7,   ...]
 *
 *  5 steps for 32 lanes = log2(32) = optimal.
 * ============================================================================

__device__ __forceinline__ float warp_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ============================================================================
//  §3  REGISTER-RESIDENT LOCAL TOP-K
//
 *  Each thread processes V / BLOCK_THREADS elements and keeps the
 *  LOCAL_K largest softmax values in registers.
 *
 *  Insertion strategy: linear scan for minimum (eviction candidate).
 *  For LOCAL_K=16, this is 16 comparisons — fast in registers.
 *
 *  Alternative for larger LOCAL_K: maintain a small register heap,
 *  but linear scan wins for LOCAL_K <= 32 due to branch prediction.
 * ============================================================================

template <int LK>
struct LocalTopK {
    float vals[LK];
    int   idxs[LK];
    int   count;

    __device__ __forceinline__ LocalTopK() : count(0) {
        #pragma unroll
        for (int i = 0; i < LK; i++) vals[i] = -FLT_MAX;
    }

    __device__ __forceinline__ void insert(float val, int idx) {
        if (count < LK) {
            vals[count] = val;
            idxs[count] = idx;
            count++;
            return;
        }
        // Find minimum (eviction candidate)
        float min_val = vals[0];
        int   min_pos = 0;
        #pragma unroll
        for (int i = 1; i < LK; i++) {
            if (vals[i] < min_val) { min_val = vals[i]; min_pos = i; }
        }
        if (val > min_val) {
            vals[min_pos] = val;
            idxs[min_pos] = idx;
        }
    }
};

// ============================================================================
//  §4  SHARED-MEMORY MIN-HEAP (size K)
//
 *  Layout: heap_vals[0] is the SMALLEST of the K kept values.
 *  New values > heap_vals[0] replace root and sift down.
 *
 *  Sift-down: O(log K) comparisons, all in shared memory (L1-like latency).
 * ============================================================================

template <int K>
__device__ __forceinline__ void heap_sift_down(
    float* __restrict__ vals, int* __restrict__ idxs, int root)
{
    int child = 2 * root + 1;
    float val = vals[root];
    int   idx = idxs[root];

    while (child < K) {
        int right = child + 1;
        if (right < K && vals[right] < vals[child]) child = right;
        if (val <= vals[child]) break;

        vals[child] = val; idxs[child] = idx;
        vals[root]  = vals[child]; idxs[root]  = idxs[child];

        root = child; child = 2 * root + 1;
    }
    vals[root] = val; idxs[root] = idx;
}

// ============================================================================
//  §5  MAIN KERNEL
//
 *  Block assignment: 1 block per (b, t) position.
 *  Thread assignment: grid-stride loop over V.
 *
 *  Shared memory layout (static + dynamic):
 *    Static:
 *      s_warp_max[8]       : 32 B    — per-warp max from phase 1
 *      s_warp_sum[8]       : 32 B    — per-warp sum from phase 2
 *      s_heap_vals[K]      : 4K B    — shared min-heap values
 *      s_heap_idxs[K]      : 4K B    — shared min-heap indices
 *    Dynamic (extern __shared__):
 *      s_stage_vals[512]   : 2048 B  — per-warp staging values
 *      s_stage_idxs[512]   : 2048 B  — per-warp staging indices
 *
 *    Total for K=256: 32+32+1024+1024+2048+2048 = 6208 B
 *    (well within 48 KB shared memory limit)
 * ============================================================================

template <int K>
__global__ void fused_softmax_topk_kernel(
    const float* __restrict__ logits,   // [B, T, V]
    int*         __restrict__ top_idx,   // [B, T, K]
    float*       __restrict__ top_prob,  // [B, T, K]
    int B, int T, int V)
{
    // ------------------------------------------------------------------
    //  Static shared memory
    // ------------------------------------------------------------------
    __shared__ float s_warp_max[WARPS_PER_BLOCK];
    __shared__ float s_warp_sum[WARPS_PER_BLOCK];
    __shared__ float s_heap_vals[K];
    __shared__ int   s_heap_idxs[K];

    // Dynamic shared memory (staging buffer for warp merge)
    extern __shared__ float s_shared[];
    float* s_stage_vals = s_shared;
    int*   s_stage_idxs = reinterpret_cast<int*>(
        s_shared + (WARP_SIZE * LOCAL_K));

    // ------------------------------------------------------------------
    //  Thread/block indexing
    // ------------------------------------------------------------------
    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int bid = blockIdx.x;
    int b = bid / T;
    int t = bid % T;

    const float* __restrict__ row =
        logits + ((size_t)b * T * V + (size_t)t * V);

    int* __restrict__ out_idx  =
        top_idx  + ((size_t)b * T * K + (size_t)t * K);
    float* __restrict__ out_prob =
        top_prob + ((size_t)b * T * K + (size_t)t * K);

    // ==================================================================
    //  PHASE 1: Max reduction (numerical stability)
    //
    //  Each thread scans its grid-stride chunk of V, finds local max.
    //  Warp-level shuffle reduction → warp leader writes to shared mem.
    //  Warp 0 reads all warp results → block max.
    //
    //  Memory accesses: V reads (coalesced across threads in first iter)
    //  Compute: V comparisons
    // ==================================================================
    float local_max = -FLT_MAX;
    for (int v = tid; v < V; v += BLOCK_THREADS) {
        float val = row[v];
        if (val > local_max) local_max = val;
    }

    local_max = warp_max(local_max);
    if (lane_id == 0) s_warp_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float block_max = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            block_max = fmaxf(block_max, s_warp_max[w]);
        }
        block_max = warp_max(block_max);
        if (lane_id == 0) s_warp_max[0] = block_max;
    }
    __syncthreads();
    float max_val = s_warp_max[0];

    // ==================================================================
    //  PHASE 2: Log-sum-exp denominator
    //
    //  sum(exp(x_i - max)) for all i. Same reduction pattern as phase 1.
    //
    //  Memory accesses: V reads (coalesced)
    //  Compute: V expf() + V additions
    // ==================================================================
    float local_sum = 0.0f;
    for (int v = tid; v < V; v += BLOCK_THREADS) {
        local_sum += expf(row[v] - max_val);
    }

    local_sum = warp_sum(local_sum);
    if (lane_id == 0) s_warp_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            block_sum += s_warp_sum[w];
        }
        block_sum = warp_sum(block_sum);
        if (lane_id == 0) s_warp_sum[0] = block_sum;
    }
    __syncthreads();

    float inv_sum = 1.0f / s_warp_sum[0];

    // ==================================================================
    //  PHASE 3: Softmax + local top-K collection
    //
    //  Each thread computes softmax values and maintains a local
    //  top-K buffer in registers. No global memory writes yet.
    //
    //  Memory accesses: V reads (coalesced)
    //  Compute: V expf() + V multiplications + V * LOCAL_K comparisons
    // ==================================================================
    LocalTopK<LOCAL_K> local_topk;

    for (int v = tid; v < V; v += BLOCK_THREADS) {
        float prob = expf(row[v] - max_val) * inv_sum;
        local_topk.insert(prob, v);
    }

    // ==================================================================
    //  PHASE 4: Merge local buffers → shared heap
    //
    //  Strategy: process one warp at a time.
    //    1. Active warp writes LOCAL_K entries per thread to staging.
    //    2. Warp 0, thread 0 merges staging into shared heap.
    //    3. __syncthreads() before next warp.
    //
    //  This serializes the merge across warps but avoids any concurrent
    //  heap mutation. Total: WARPS_PER_BLOCK rounds, each with 2 barriers.
    //
    //  Heap insertions: WARP_SIZE * LOCAL_K = 512 per round.
    //  Total heap insertions: 8 * 512 = 4096.
    //  Each insertion: O(log K) = O(8) shared memory ops.
    //  Total: ~32K shared memory ops (negligible vs global memory).
    // ==================================================================
    for (int i = tid; i < K; i += BLOCK_THREADS) {
        s_heap_vals[i] = -FLT_MAX;
        s_heap_idxs[i] = -1;
    }
    __syncthreads();

    for (int w = 0; w < WARPS_PER_BLOCK; w++) {
        // Active warp writes to staging
        if (warp_id == w) {
            #pragma unroll
            for (int i = 0; i < LOCAL_K; i++) {
                int pos = lane_id * LOCAL_K + i;
                s_stage_vals[pos] = local_topk.vals[i];
                s_stage_idxs[pos] = local_topk.idxs[i];
            }
        }
        __syncthreads();

        // Warp 0, thread 0 merges into shared heap
        if (tid == 0) {
            for (int i = 0; i < WARP_SIZE * LOCAL_K; i++) {
                float val = s_stage_vals[i];
                int   idx = s_stage_idxs[i];
                if (val > s_heap_vals[0]) {
                    s_heap_vals[0] = val;
                    s_heap_idxs[0] = idx;
                    heap_sift_down<K>(s_heap_vals, s_heap_idxs, 0);
                }
            }
        }
        __syncthreads();
    }

    // ==================================================================
    //  PHASE 5: Sort and write-back
    //
    //  The shared heap contains the top-K values (as a min-heap).
    //  Thread 0 sorts in descending order and writes to global memory.
    //
    //  Sort: selection sort O(K²) = O(65536) for K=256.
    //  This is done once per block, so it's negligible.
    //  Alternative: heap-extract O(K log K) = O(2048) — faster.
    // ==================================================================
    if (tid == 0) {
        // Heap-extract: repeatedly remove max, write to output.
        // The max is NOT at the root (min-heap). We find it by scanning.
        // Better: convert to max-heap first, or just scan.

        // Selection sort (simple, correct, fast enough for K=256)
        for (int i = 0; i < K; i++) {
            // Find max in s_heap_vals[i..K-1]
            int max_pos = i;
            float max_v = s_heap_vals[i];
            for (int j = i + 1; j < K; j++) {
                if (s_heap_vals[j] > max_v) {
                    max_v = s_heap_vals[j];
                    max_pos = j;
                }
            }
            // Swap to position i
            float tmp_v = s_heap_vals[i];
            int   tmp_i = s_heap_idxs[i];
            s_heap_vals[i] = s_heap_vals[max_pos];
            s_heap_idxs[i] = s_heap_idxs[max_pos];
            s_heap_vals[max_pos] = tmp_v;
            s_heap_idxs[max_pos] = tmp_i;

            // Write to global memory
            out_idx[i]  = s_heap_idxs[i];
            out_prob[i] = s_heap_vals[i];
        }
    }
}

// ============================================================================
//  §6  LAUNCHER
// ============================================================================

template <int K>
cudaError_t launch_fused_softmax_topk(
    const float* d_logits,
    int*         d_top_idx,
    float*       d_top_prob,
    int B, int T, int V)
{
    dim3 grid(B * T);
    dim3 block(BLOCK_THREADS);

    // Dynamic shared memory: staging buffer
    // vals: WARP_SIZE * LOCAL_K * sizeof(float) = 32 * 16 * 4 = 2048 B
    // idxs: WARP_SIZE * LOCAL_K * sizeof(int)   = 32 * 16 * 4 = 2048 B
    size_t dsm_bytes = 2 * WARP_SIZE * LOCAL_K * sizeof(float);

    fused_softmax_topk_kernel<K><<<grid, block, dsm_bytes>>>(
        d_logits, d_top_idx, d_top_prob, B, T, V);

    return cudaGetLastError();
}

// Explicit template instantiations
template cudaError_t launch_fused_softmax_topk<16>(
    const float*, int*, float*, int, int, int);
template cudaError_t launch_fused_softmax_topk<32>(
    const float*, int*, float*, int, int, int);
template cudaError_t launch_fused_softmax_topk<64>(
    const float*, int*, float*, int, int, int);
template cudaError_t launch_fused_softmax_topk<128>(
    const float*, int*, float*, int, int, int);
template cudaError_t launch_fused_softmax_topk<256>(
    const float*, int*, float*, int, int, int);
