# Fused Softmax + Top-K Kernel — Design Analysis

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Memory Access Pattern](#2-memory-access-pattern)
3. [Warp-Level Optimization Strategy](#3-warp-level-optimization-strategy)
4. [Complexity Analysis](#4-complexity-analysis)
5. [Comparison to Naive Implementation](#5-comparison-to-naive-implementation)
6. [Further Optimizations](#6-further-optimizations)

---

## 1. Architecture Overview

### Block Assignment
```
Grid:  B × T blocks (one block per (b, t) position)
Block: 256 threads per block
```

### Three-Phase Pipeline (per block)
```
Phase 1:  Find max(logits[b,t,:])          → numerical stability anchor
Phase 2:  Compute Σexp(xᵢ - max)           → log-sum-exp denominator
Phase 3:  Compute softmax + collect top-K   → register-local buffers
Phase 4:  Merge local buffers → shared heap → global top-K
Phase 5:  Sort + write-back                 → output [B,T,K]
```

### Why Three Passes Over V?
You might wonder why we don't do this in one pass. The answer is **numerical stability**:

```
softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
```

Without knowing the max first, `exp(xᵢ)` can overflow for large logits. The standard
trick is:

```
softmax(xᵢ) = exp(xᵢ - max) / Σⱼ exp(xⱼ - max)
```

This requires knowing `max` before computing any softmax values, hence two passes
(max reduction, then softmax computation).

**Could we do it in one pass?** Yes, with an online algorithm that tracks a running
max and re-normalizes, but this adds complexity and potential numerical issues. The
two-pass approach is simpler, correct, and the extra V reads are coalesced.

---

## 2. Memory Access Pattern

### Global Memory Reads

| Phase | Access Pattern | Bytes Read | Coalesced? |
|-------|---------------|------------|------------|
| Phase 1 | `row[tid], row[tid+256], ...` | 4V | ✅ First iteration |
| Phase 2 | `row[tid], row[tid+256], ...` | 4V | ✅ First iteration |
| Phase 3 | `row[tid], row[tid+256], ...` | 4V | ✅ First iteration |
| **Total** | | **12V** | |

For V=50257: **12 × 50257 × 4B ≈ 2.4 MB read per (b,t)**.

**Coalescing analysis:**
- First iteration: threads 0-255 read `row[0]` through `row[255]` → perfectly coalesced
  into ~8-16 128-byte transactions (depending on alignment).
- Subsequent iterations: threads read `row[256]` through `row[511]`, etc. → also coalesced.
- Stride within a thread (256 elements apart) doesn't affect coalescing — coalescing
  is about **consecutive threads accessing consecutive addresses**.

### Global Memory Writes

| Output | Bytes Written |
|--------|--------------|
| `top_idx[B,T,K]` | 4BK |
| `top_prob[B,T,K]` | 4BK |
| **Total** | **8BK** |

For B=1, T=1, K=256: **8 × 256 = 2048 B** (negligible).

### Shared Memory Usage

| Buffer | Size (K=256) | Access Pattern |
|--------|-------------|----------------|
| `s_warp_max[8]` | 32 B | Write: 8 threads, Read: warp 0 |
| `s_warp_sum[8]` | 32 B | Write: 8 threads, Read: warp 0 |
| `s_heap_vals[256]` | 1024 B | Write: all (init), Read/Write: thread 0 |
| `s_heap_idxs[256]` | 1024 B | Write: all (init), Read/Write: thread 0 |
| `s_stage_vals[512]` | 2048 B | Write: active warp, Read: thread 0 |
| `s_stage_idxs[512]` | 2048 B | Write: active warp, Read: thread 0 |
| **Total** | **6208 B** | |

Well within the 48 KB shared memory limit per SM.

### Register Usage (per thread)

| Variable | Count |
|----------|-------|
| `LocalTopK<16>::vals` | 16 floats = 64 B |
| `LocalTopK<16>::idxs` | 16 ints = 64 B |
| Loop counters, temporaries | ~10 registers |
| **Total** | **~40 registers** |

With 256 threads/block and 40 registers/thread: 10,240 registers per block.
On Ampere (64K registers/SM): fits 6 blocks → 1536 threads → good occupancy.

---

## 3. Warp-Level Optimization Strategy

### 3.1 Shuffle-Based Reductions

**Problem:** Traditional reductions use shared memory + sync barriers.

**Our approach:** `__shfl_xor_sync` (warp shuffle) — data moves directly between
thread registers within a warp, zero shared memory, zero global memory.

```
warp_max(val):
    for offset in [16, 8, 4, 2, 1]:
        other = __shfl_xor_sync(mask, val, offset)
        val = max(val, other)
    return val
```

**Latency:** 5 shuffle operations × ~3 cycles = ~15 cycles per reduction.
**vs. shared memory:** ~5 cycles per access + barrier overhead = ~30+ cycles.

### 3.2 Warp-Level Merge Strategy

The merge of local top-K buffers into the shared heap uses a **warp-by-warp** strategy:

```
for each warp w in [0, 7]:
    if warp_id == w:
        write LOCAL_K entries to staging buffer
    __syncthreads()
    if tid == 0:
        merge staging into shared heap
    __syncthreads()
```

**Why not all threads merge concurrently?** Concurrent heap mutations require
atomics or locks, which serialize anyway and add overhead. The warp-by-warp
approach:
- Uses only 2 barriers per warp (16 total)
- Thread 0 does all heap operations (no contention)
- Other threads are idle during merge (but this is a small fraction of total work)

**Alternative: warp-level merge within each warp.** Each warp could merge its 32
threads' LOCAL_K entries into a warp-local top-K using shuffle operations, then
only 8 warp leaders contribute to the shared heap. This reduces heap insertions
from 4096 to 8×K = 2048. **This is a valid optimization** (see §6).

### 3.3 Grid-Stride Loop for Large V

```cuda
for (int v = tid; v < V; v += BLOCK_THREADS) {
    // process row[v]
}
```

For V=50257, BLOCK_THREADS=256: each thread processes ⌈50257/256⌉ = 197 elements.

**Benefits:**
- Works for any V (no template parameter needed)
- Good load balancing (threads process nearly equal elements)
- First iteration is coalesced; subsequent iterations are also coalesced

**Trade-off:** Strided access within a thread means poor L2 cache reuse.
However, for V=50K, the entire row fits in L2 (200 KB on Ampere), so
re-reading across phases benefits from L2 cache.

---

## 4. Complexity Analysis

### 4.1 Bandwidth vs. Compute Bound

**Parameters:** B=1, T=1, V=50257, K=256

| Metric | Value |
|--------|-------|
| Global memory reads | 12 × 50257 × 4B = **2.41 MB** |
| Global memory writes | 8 × 256 = **2.05 KB** |
| Shared memory ops | ~32K (heap) + ~4K (staging) = **~36K** |
| expf() calls | 2 × 50257 = **100,514** |
| Comparisons | 50257 × LOCAL_K × 256 ≈ **163M** (local top-K inserts) |
| Heap sifts | 4096 × log₂(256) = **32,768** |

**Bandwidth requirement:** 2.41 MB per (b,t).
On H100 (3.35 TB/s): 2.41 MB / 3.35 TB/s = **0.72 μs** (theoretical minimum).

**Compute requirement:** 100,514 expf() calls.
On H100 (194 TFLOPS FP32): expf ≈ 50 cycles → 5.0M cycles / 1.5 GHz = **3.3 μs**.

**Verdict: COMPUTE-BOUND.** The kernel is limited by expf() throughput, not memory bandwidth.

### 4.2 Scaling with V

| V | Global Reads | expf() calls | Bandwidth (μs) | Compute (μs) | Bound |
|---|-------------|-------------|----------------|---------------|-------|
| 10K | 480 KB | 20K | 0.14 | 0.67 | Compute |
| 50K | 2.41 MB | 100K | 0.72 | 3.3 | Compute |
| 100K | 4.82 MB | 200K | 1.44 | 6.6 | Compute |
| 500K | 24.1 MB | 1M | 7.2 | 33 | Compute |
| 1M | 48.2 MB | 2M | 14.4 | 66 | Compute |

The kernel remains compute-bound across all practical V values.

### 4.3 Scaling with K

| K | Heap ops | Sort ops | Impact |
|---|----------|----------|--------|
| 16 | 512 × 4 = 2K | 256 | Negligible |
| 64 | 4096 × 6 = 25K | 4K | Small |
| 256 | 4096 × 8 = 33K | 66K | Moderate |
| 1024 | 4096 × 10 = 41K | 1M | Significant |

For K > 256, the heap operations and sort become noticeable. Consider:
- Increasing LOCAL_K to maintain oversampling ratio
- Using a more efficient merge (warp-level top-K within each warp)
- Parallel sort (bitonic sort across threads)

---

## 5. Comparison to Naive Implementation

### Naive Approach
```python
# Python pseudocode
probs = softmax(logits)           # Materialize [B, T, V] in global memory
top_idx, top_prob = topk(probs, K)  # Read [B, T, V], write [B, T, K]
```

### Comparison Table

| Metric | Naive | Fused Kernel | Speedup |
|--------|-------|-------------|---------|
| **Global reads** | 4V (logits) + 4V (probs) = **8V** | **12V** (logits × 3) | 0.67× |
| **Global writes** | 4V (probs) + 8K (output) | **8K** (output only) | **V/K ×** |
| **Peak memory** | 4V + 8K | 8K | **V/K ×** |
| **expf() calls** | V (softmax) | 2V (phase 2 + 3) | 0.5× |
| **Numerical stability** | Depends on softmax impl | Guaranteed (max subtraction) | — |

### Key Insight: Memory Savings Dominate

For V=50257, K=256:
- **Naive:** writes 4 × 50257 = **201 KB** of softmax probabilities to global memory
- **Fused:** writes only 8 × 256 = **2 KB** of output

The fused kernel reads 50% more (12V vs 8V) but **avoids writing the entire softmax
matrix**. For large V, the write savings dominate:

```
Naive bandwidth:  8V + 8K = 8V(1 + K/V) ≈ 8V
Fused bandwidth:  12V + 8K = 12V(1 + K/(3V)) ≈ 12V

Ratio: 12V / 8V = 1.5× more reads, but 0 writes vs 4V writes.
Net: fused saves 4V - 8K = 4V(1 - 2K/V) bytes.
```

For V=50257, K=256: saves **4 × 50257 - 8 × 256 = 192 KB** per (b,t).

### When Naive Wins

The naive approach can be faster when:
1. **V is small** (V < 1024): the overhead of 3 passes isn't worth it
2. **You need the full softmax** for other operations (e.g., KL divergence)
3. **Hardware has very high bandwidth** relative to compute (e.g., HBM3)

### When Fused Wins

The fused kernel dominates when:
1. **V is large** (V > 10K): memory savings are significant
2. **Memory is the bottleneck** (e.g., mobile, edge devices)
3. **You only need top-K** (common in LLM sampling)
4. **Batch size is small** (B=1): one block per (b,t) means no inter-block sync

---

## 6. Further Optimizations

### 6.1 Warp-Level Top-K Merge (Recommended)

Instead of merging all 4096 candidates through a single thread, each warp
merges its 32 threads' LOCAL_K entries into a warp-local top-K using shuffle:

```cuda
// Each warp: 32 threads × LOCAL_K = 512 entries → top-K within warp
// Use warp shuffle to find top-K in O(K × WARP_SIZE) operations
// Then only 8 warp leaders contribute to shared heap
```

**Benefit:** Reduces heap insertions from 4096 to 8 × K = 2048.
**Complexity:** Moderate — requires warp-level selection algorithm.

### 6.2 Float16/BFloat16 Support

For LLM workloads, logits are often in FP16/BF16:

```cuda
// Use __hexp2() for half-precision exp
// Use __shfl_xor_sync with half-precision values
// Promote to FP32 only for final softmax computation
```

**Benefit:** 2× less global memory bandwidth, 2× more throughput.
**Trade-off:** Slight numerical precision loss (acceptable for top-K).

### 6.3 Vectorized Memory Access

```cuda
// Read 4 floats at once (128-bit load)
float4 val = reinterpret_cast<const float4*>(&row[v])[0];
```

**Benefit:** 4× fewer memory instructions, better utilization of memory bandwidth.
**Constraint:** V must be divisible by 4, BLOCK_THREADS must be divisible by 4.

### 6.4 Persistent Blocks for Large B×T

For large B×T, launch fewer blocks and have each block process multiple (b,t):

```cuda
int bid = blockIdx.x * GRID_STRIDE + threadIdx.x;
while (bid < B * T) {
    process(bid);
    bid += GRID_STRIDE * BLOCK_THREADS;
}
```

**Benefit:** Better occupancy, hides memory latency.

### 6.5 Asynchronous Copy (Hopper+)

On H100+, use `ld.global.nc.v4.f32` (non-coherent load) for the logits reads:

```cuda
// Compiler hint: these values won't be modified
#pragma unroll
for (int v = tid; v < V; v += BLOCK_THREADS) {
    float val = __ldg(&row[v]);  // cacheable load
    // ...
}
```

**Benefit:** Better L2 cache utilization across the three passes.

### 6.6 Single-Pass Online Algorithm

Track a running max and re-normalize:

```cuda
float local_max = -FLT_MAX;
float local_sum = 0.0f;
LocalTopK<LOCAL_K> local_topk;

for (int v = tid; v < V; v += BLOCK_THREADS) {
    float x = row[v];
    if (x > local_max) {
        // Re-normalize all previous values
        float old_max = local_max;
        local_max = x;
        local_sum = 0.0f;
        // Re-insert all local_topk entries with new normalization
        // ... (complex)
    }
    float prob = expf(x - local_max);
    local_sum += prob;
    // ...
}
```

**Benefit:** Single pass over V (4V reads instead of 12V).
**Trade-off:** Complex, potential numerical issues, re-normalization overhead.
**Verdict:** Not recommended unless V is extremely large (>1M).

---

## Appendix: Kernel Instantiation

```cuda
// Launch for LLaMA (V=50257, K=256)
launch_fused_softmax_topk<256>(d_logits, d_top_idx, d_top_prob, 1, 1, 50257);

// Launch for GPT-2 (V=50257, K=50)
launch_fused_softmax_topk<50>(d_logits, d_top_idx, d_top_prob, 1, 1, 50257);

// Launch for batched inference (B=32, T=128, V=32000, K=128)
launch_fused_softmax_topk<128>(d_logits, d_top_idx, d_top_prob, 32, 128, 32000);
```
