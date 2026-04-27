// Fused Online-Softmax + Top-K Kernel
// -------------------------------------
// Input : logits  [B, T, V]   (fp32 or fp16, fp32 shown here)
// Output: topk_idx [B, T, K]   (int32)
//         topk_prob[B, T, K]   (fp32, post-softmax probabilities)
//
// One block processes one (b, t) row of length V.
// Single pass over global memory: streams logits exactly once.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

constexpr int WARP = 32;

// -----------------------------------------------------------------------------
// Per-thread top-K register buffer (insertion-sorted, descending by value).
// Kept tiny (K <= 32 typical) so it lives in registers.
// -----------------------------------------------------------------------------
template <int K>
struct TopK {
    float val[K];
    int   idx[K];

    __device__ __forceinline__ void init() {
        #pragma unroll
        for (int i = 0; i < K; ++i) { val[i] = -FLT_MAX; idx[i] = -1; }
    }

    // Insert (v, i) if v beats the current min (val[K-1]).
    __device__ __forceinline__ void push(float v, int i) {
        if (v <= val[K-1]) return;
        int p = K - 1;
        while (p > 0 && val[p-1] < v) {
            val[p] = val[p-1];
            idx[p] = idx[p-1];
            --p;
        }
        val[p] = v;
        idx[p] = i;
    }
};

// -----------------------------------------------------------------------------
// Online softmax reduction primitive (Milakov & Gimelshein, 2018).
// Combines two partial states (m_a, s_a) and (m_b, s_b) into one:
//   m = max(m_a, m_b)
//   s = s_a * exp(m_a - m) + s_b * exp(m_b - m)
// Numerically stable; associative => valid for tree/warp reductions.
// -----------------------------------------------------------------------------
struct MS { float m; float s; };

__device__ __forceinline__ MS ms_combine(MS a, MS b) {
    float m = fmaxf(a.m, b.m);
    float s = a.s * __expf(a.m - m) + b.s * __expf(b.m - m);
    return {m, s};
}

__device__ __forceinline__ MS warp_reduce_ms(MS x) {
    #pragma unroll
    for (int o = WARP/2; o > 0; o >>= 1) {
        MS y;
        y.m = __shfl_xor_sync(0xffffffff, x.m, o);
        y.s = __shfl_xor_sync(0xffffffff, x.s, o);
        x = ms_combine(x, y);
    }
    return x;
}

// -----------------------------------------------------------------------------
// Merge two TopK<K> buffers held by threads `lane` and `lane^offset`.
// Each thread ends with the merged top-K. Implemented via XOR-shuffle
// on the K (val, idx) pairs and a K+K -> K linear merge.
// -----------------------------------------------------------------------------
template <int K>
__device__ __forceinline__ void warp_merge_topk(TopK<K>& a, int offset) {
    TopK<K> b;
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        b.val[i] = __shfl_xor_sync(0xffffffff, a.val[i], offset);
        b.idx[i] = __shfl_xor_sync(0xffffffff, a.idx[i], offset);
    }
    // Merge two descending lists of length K -> length K.
    float ov[K]; int oi[K];
    int ia = 0, ib = 0;
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        bool take_a = (ia < K) && (ib >= K || a.val[ia] >= b.val[ib]);
        ov[i] = take_a ? a.val[ia] : b.val[ib];
        oi[i] = take_a ? a.idx[ia] : b.idx[ib];
        ia += take_a; ib += !take_a;
    }
    #pragma unroll
    for (int i = 0; i < K; ++i) { a.val[i] = ov[i]; a.idx[i] = oi[i]; }
}

template <int K>
__device__ __forceinline__ void warp_reduce_topk(TopK<K>& a) {
    #pragma unroll
    for (int o = WARP/2; o > 0; o >>= 1) warp_merge_topk<K>(a, o);
}

// =============================================================================
// Kernel: one block per (b, t) row.
//   blockDim.x = BLOCK (multiple of 32, e.g. 256 or 512)
//   gridDim    = (B * T)
// =============================================================================
template <int K, int BLOCK>
__global__ void fused_softmax_topk_kernel(
    const float* __restrict__ logits,  // [B*T, V]
    int* __restrict__         topk_idx,  // [B*T, K]
    float* __restrict__       topk_prob, // [B*T, K]
    int V)
{
    static_assert(BLOCK % WARP == 0, "BLOCK must be a multiple of 32");
    constexpr int WARPS = BLOCK / WARP;

    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int warp = tid >> 5;

    const float* row_logits = logits + (size_t)row * V;

    // -- Pass 1 (the only pass over V): online-softmax state + top-K -----------
    MS  ms{-FLT_MAX, 0.f};
    TopK<K> tk; tk.init();

    // Coalesced strided read: thread `tid` of block reads V[tid], V[tid+BLOCK], ...
    // Each warp reads 32 contiguous floats per step => 128B transactions.
    for (int i = tid; i < V; i += BLOCK) {
        float x = row_logits[i];
        // Online-softmax update
        float m_new = fmaxf(ms.m, x);
        ms.s = ms.s * __expf(ms.m - m_new) + __expf(x - m_new);
        ms.m = m_new;
        // Top-K update (cheap: usually no swap needed once tk is populated)
        tk.push(x, i);
    }

    // -- Warp-level reductions -------------------------------------------------
    ms = warp_reduce_ms(ms);
    warp_reduce_topk<K>(tk);  // every lane in the warp now holds warp's top-K

    // -- Cross-warp via shared memory -----------------------------------------
    __shared__ MS         smem_ms[WARPS];
    __shared__ float      smem_tk_v[WARPS * K];
    __shared__ int        smem_tk_i[WARPS * K];

    if (lane == 0) {
        smem_ms[warp] = ms;
        #pragma unroll
        for (int j = 0; j < K; ++j) {
            smem_tk_v[warp * K + j] = tk.val[j];
            smem_tk_i[warp * K + j] = tk.idx[j];
        }
    }
    __syncthreads();

    // Final reduction performed by warp 0 only.
    if (warp == 0) {
        // Load each warp's partial into one lane (lanes 0..WARPS-1).
        MS my_ms = (lane < WARPS) ? smem_ms[lane] : MS{-FLT_MAX, 0.f};
        TopK<K> my_tk;
        if (lane < WARPS) {
            #pragma unroll
            for (int j = 0; j < K; ++j) {
                my_tk.val[j] = smem_tk_v[lane * K + j];
                my_tk.idx[j] = smem_tk_i[lane * K + j];
            }
        } else {
            my_tk.init();
        }

        // Reduce across the (up to 32) warp partials in this single warp.
        my_ms = warp_reduce_ms(my_ms);
        warp_reduce_topk<K>(my_tk);

        if (lane == 0) {
            const float m_final = my_ms.m;
            const float inv_s   = 1.0f / my_ms.s;
            int*   out_idx = topk_idx  + (size_t)row * K;
            float* out_prob= topk_prob + (size_t)row * K;
            #pragma unroll
            for (int j = 0; j < K; ++j) {
                out_idx[j]  = my_tk.idx[j];
                // softmax(x_j) = exp(x_j - m) / s
                out_prob[j] = __expf(my_tk.val[j] - m_final) * inv_s;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Host launcher
// -----------------------------------------------------------------------------
template <int K>
void launch_fused_softmax_topk(
    const float* d_logits, int* d_topk_idx, float* d_topk_prob,
    int B, int T, int V, cudaStream_t stream = 0)
{
    constexpr int BLOCK = 256;            // 8 warps; tune per arch (256/512)
    dim3 grid(B * T);
    dim3 block(BLOCK);
    fused_softmax_topk_kernel<K, BLOCK>
        <<<grid, block, 0, stream>>>(d_logits, d_topk_idx, d_topk_prob, V);
}

// Explicit instantiations for common K
template void launch_fused_softmax_topk<8 >(const float*, int*, float*, int, int, int, cudaStream_t);
template void launch_fused_softmax_topk<16>(const float*, int*, float*, int, int, int, cudaStream_t);
template void launch_fused_softmax_topk<32>(const float*, int*, float*, int, int, int, cudaStream_t);
