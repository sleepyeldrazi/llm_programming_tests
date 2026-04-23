// =============================================================================
// Fused Softmax + Top-K Kernel
// =============================================================================
// Input:  logits  [B, T, V]   (row-major, fp32 or fp16)
// Output: indices [B, T, K]   (int32)
//         probs   [B, T, K]   (fp32, softmax probabilities of top-K)
//
// Key insight: we never materialize the full V-length softmax vector.
// We compute the softmax in a single forward pass (online softmax) while
// simultaneously maintaining a min-heap of the top-K values seen so far.
// =============================================================================

#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// --------------- tunable parameters ---------------
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef HEAP_K
// Max K we support; must be a power of 2 for warp-reduce simplicity.
// For K <= 32 we keep the heap entirely in registers per warp.
#define HEAP_K 32
#endif

// We launch one warp per (b, t) row. Each warp processes V/WARP_SIZE
// elements, accumulating partial softmax statistics and a local top-K heap,
// then merges heaps across warps in shared memory.
//
// Block layout:  WARPS_PER_BLOCK warps, each handling one row.
// Grid  layout:  ceil(B*T / WARPS_PER_BLOCK) blocks.
// Total threads: B * T * WARP_SIZE  (every thread in a warp works on one row)

#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 8
#endif
#define BLOCK_SIZE (WARPS_PER_BLOCK * WARP_SIZE)

// =============================================================================
// Min-heap utilities (keeps top-K largest values)
// =============================================================================
// We store a small sorted array of size K (K <= 32) in registers.
// This is faster than a tree-based heap for small K because:
//   - Insertion is just a single compare + conditional shift
//   - Cache/coherence is trivial (all registers)
//   - No pointer chasing

template <int K>
struct TopKHeap {
    float vals[K];   // sorted ascending (vals[0] is the minimum)
    int   idxs[K];

    __device__ __forceinline__
    void init() {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            vals[i] = -FLT_MAX;
            idxs[i] = 0;
        }
    }

    // Insert if value > current minimum (the K-th largest so far).
    __device__ __forceinline__
    void insert(float val, int idx) {
        if (val <= vals[0]) return;          // not in top-K, skip
        // Linear scan to find insertion point (small K → branch predictor loves it).
        // For K=32 this is ~5 compares on average, cheaper than binary search overhead.
        int pos = 0;
        #pragma unroll
        for (int i = 1; i < K; i++) {
            if (val > vals[i]) pos = i;      // find last position where val > vals[i]
            else break;
        }
        // Shift elements down: vals[0..pos-1] ← vals[1..pos]
        #pragma unroll
        for (int i = 0; i < pos; i++) {
            vals[i] = vals[i + 1];
            idxs[i] = idxs[i + 1];
        }
        vals[pos] = val;
        idxs[pos] = idx;
    }
};

// =============================================================================
// Warp-level primitives
// =============================================================================

__device__ __forceinline__
float warp_reduce_max(float val) {
    // Butterfly reduction across 32 lanes
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val,  8));
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val,  4));
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val,  2));
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val,  1));
    return val;
}

__device__ __forceinline__
float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
    val += __shfl_xor_sync(0xFFFFFFFF, val,  8);
    val += __shfl_xor_sync(0xFFFFFFFF, val,  4);
    val += __shfl_xor_sync(0xFFFFFFFF, val,  2);
    val += __shfl_xor_sync(0xFFFFFFFF, val,  1);
    return val;
}

// =============================================================================
// Shared memory layout (one per block)
// =============================================================================

struct SharedStorage {
    // Per-warp partial results for cross-warp merge
    float warp_max[WARPS_PER_BLOCK];       // partial max
    float warp_sum[WARPS_PER_BLOCK];       // partial sum of exps
    // Heap merge buffer: each warp writes its local top-K here
    float heap_buf[WARPS_PER_BLOCK][HEAP_K];
    int   idx_buf [WARPS_PER_BLOCK][HEAP_K];
    // Synchronization
    int   barrier_count;
};

// =============================================================================
// Phase 1 — Per-warp local pass over V/WARPS_PER_BLOCK chunks
// =============================================================================
// Each lane j in warp w processes logits at indices:
//   j, j + WARP_SIZE, j + 2*WARP_SIZE, ...
// covering a strided subset of the V-dimension.
//
// Online softmax recurrence (per lane):
//   m_j ← max(m_j, x_j)          (local max)
//   d_j ← d_j * exp(m_old - m_j) + exp(x_j - m_j)
//
// After the loop we do a warp-all-reduce to get the global max m and sum d
// for this row. Then each lane rescales its accumulated exp-sum and
// inserts its local top-K candidates into a heap scaled by 1/d.

template <int K>
__device__ __forceinline__
void local_pass(
    const float* __restrict__ logits_row,  // pointer to row of length V
    int V,
    float& out_max,                        // warp-reduced max
    float& out_sum,                        // warp-reduced sum of exps
    TopKHeap<K>& heap)                     // per-lane local top-K
{
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    // Strided loop: lane i processes indices i, i+32, i+64, ...
    // This gives coalesced global reads because consecutive lanes read
    // consecutive addresses.
    for (int v = lane; v < V; v += WARP_SIZE) {
        float x = logits_row[v];
        float old_max = local_max;
        local_max = fmaxf(local_max, x);
        // Rescale running sum to new max
        local_sum *= expf(old_max - local_max);
        local_sum += expf(x - local_max);

        // Track top-K in the original logit space (before exp).
        // We will rescale to probabilities later using the final max & sum.
        heap.insert(x, v);
    }

    // ---- Warp-level reduction for max and sum ----
    float warp_max = warp_reduce_max(local_max);
    // Rescale all lane sums to the common warp_max
    local_sum *= expf(local_max - warp_max);
    float warp_sum = warp_reduce_sum(local_sum);

    out_max = warp_max;
    out_sum = warp_sum;
}

// =============================================================================
// Phase 2 — Cross-warp heap merge in shared memory
// =============================================================================
// When WARPS_PER_BLOCK > 1, each warp has its own local top-K heap.
// We merge by:
//   1. Each warp writes its heap to shared memory
//   2. __syncthreads()
//   3. Lane 0 of warp 0 does a serial K-way merge (K is small, typically 5-50)
//      over WARPS_PER_BLOCK heaps → global top-K
//   4. Rescale values: prob_i = exp(val_i - global_max) / global_sum
//
// For WARPS_PER_BLOCK == 1 this phase is a no-op (single warp = single row).

template <int K>
__device__ __forceinline__
void cross_warp_merge(
    SharedStorage& smem,
    float global_max,
    float global_sum,
    TopKHeap<K>& heap,
    int warp_id,
    int lane_id,
    float* out_probs,                      // [K] output
    int*   out_idxs)                       // [K] output
{
    // Each warp writes its local heap to shared memory
    if (lane_id < K) {
        smem.heap_buf[warp_id][lane_id] = heap.vals[K - 1 - lane_id]; // descending
        smem.idx_buf [warp_id][lane_id] = heap.idxs[K - 1 - lane_id];
    }
    __syncthreads();

    // Warp 0 merges all heaps
    if (warp_id == 0) {
        // Build the global top-K by scanning all warp heaps
        TopKHeap<K> global_heap;
        global_heap.init();

        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            #pragma unroll
            for (int i = 0; i < K; i++) {
                float v = smem.heap_buf[w][i];
                int   j = smem.idx_buf [w][i];
                global_heap.insert(v, j);
            }
        }

        // Lane 0 writes the final result (rescaled to probabilities)
        if (lane_id == 0) {
            float inv_sum = 1.0f / global_sum;
            #pragma unroll
            for (int i = 0; i < K; i++) {
                // vals are sorted ascending; reverse for output (descending prob)
                int ki = K - 1 - i;
                out_probs[i] = expf(global_heap.vals[ki] - global_max) * inv_sum;
                out_idxs [i] = global_heap.idxs[ki];
            }
        }
    }
}

// =============================================================================
// Main kernel
// =============================================================================

template <int K>
__global__ void fused_softmax_topk_kernel(
    const float* __restrict__ logits,      // [B, T, V]
    int*   __restrict__ out_indices,        // [B, T, K]
    float* __restrict__ out_probs,          // [B, T, K]
    int B, int T, int V)
{
    // One block processes WARPS_PER_BLOCK rows.
    // Each warp handles one row.
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // Map warp → (b, t) row index
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= B * T) return;

    int b = row / T;
    int t = row % T;

    // Pointers for this row
    const float* logits_row  = logits       + (size_t)row * V;
    int*   row_out_indices   = out_indices  + (size_t)row * K;
    float* row_out_probs     = out_probs    + (size_t)row * K;

    // Shared memory
    __shared__ __align__(16) SharedStorage smem;

    // Phase 1: local pass over logits
    TopKHeap<K> heap;
    heap.init();

    float warp_max, warp_sum;
    local_pass<K>(logits_row, V, warp_max, warp_sum, heap);

    // Store partials in shared memory for cross-warp merge
    if (lane_id == 0) {
        smem.warp_max[warp_id] = warp_max;
        smem.warp_sum[warp_id] = warp_sum;
    }
    __syncthreads();

    // Compute global max and sum across warps (lane 0 does it)
    float global_max = -FLT_MAX;
    float global_sum = 0.0f;
    if (lane_id == 0) {
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            if (blockIdx.x * WARPS_PER_BLOCK + w < B * T) {
                float wm = smem.warp_max[w];
                float ws = smem.warp_sum[w];
                float old_max = global_max;
                global_max = fmaxf(global_max, wm);
                global_sum *= expf(old_max - global_max);
                global_sum += ws * expf(wm - global_max);
            }
        }
        smem.warp_max[0] = global_max;  // reuse for broadcast
        smem.warp_sum[0] = global_sum;
    }
    __syncthreads();

    global_max = smem.warp_max[0];
    global_sum = smem.warp_sum[0];

    // Phase 2: cross-warp heap merge + write output
    cross_warp_merge<K>(smem, global_max, global_sum,
                        heap, warp_id, lane_id,
                        row_out_probs, row_out_indices);
}

// =============================================================================
// Host launch wrapper
// =============================================================================

template <int K>
void launch_fused_softmax_topk(
    const float* d_logits,
    int*   d_indices,
    float* d_probs,
    int B, int T, int V,
    cudaStream_t stream = 0)
{
    int total_rows = B * T;
    int grid = (total_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    size_t smem_bytes = sizeof(SharedStorage);

    fused_softmax_topk_kernel<K>
        <<<grid, BLOCK_SIZE, smem_bytes, stream>>>(
            d_logits, d_indices, d_probs, B, T, V);
}

// Explicit instantiation for common K values
template void launch_fused_softmax_topk<5>(const float*, int*, float*, int, int, int, cudaStream_t);
template void launch_fused_softmax_topk<10>(const float*, int*, float*, int, int, int, cudaStream_t);
template void launch_fused_softmax_topk<20>(const float*, int*, float*, int, int, int, cudaStream_t);
template void launch_fused_softmax_topk<32>(const float*, int*, float*, int, int, int, cudaStream_t);
