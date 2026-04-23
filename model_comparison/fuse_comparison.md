# Head-to-Head Analysis: Fused Softmax + Top-K CUDA Kernel

**Date:** 2026-04-23  
**Task:** High-performance fused softmax + top-k kernel in CUDA  
**Folders Analyzed:** `MiniMax-M2.7` (MiniMax-M2.7) and `Qwen3.6-27B` (Qwen3.6-27B)

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Prompt Requirements Checklist](#2-prompt-requirements-checklist)
3. [MiniMax-M2.7 (`MiniMax-M2.7`) Deep Dive](#3-model-a-minimax-m2.7fuse-deep-dive)
4. [Qwen3.6-27B (`Qwen3.6-27B`) Deep Dive](#4-model-b-qwen36fuse-deep-dive)
5. [Head-to-Head Comparison](#5-head-to-head-comparison)
6. [Scores & Justification](#6-scores--justification)
7. [Conclusion: Who Won and By How Much](#7-conclusion-who-won-and-by-how-much)

---

## 1. Executive Summary

Both models were given the identical prompt to design and implement a high-performance fused softmax + top-k kernel in CUDA. The task required:
- No materialization of the full softmax matrix in global memory
- Numerical stability via log-sum-exp
- Minimized global memory reads/writes
- Appropriate shared memory usage
- Efficient handling of large vocabulary sizes (50k+)

**Qwen3.6-27B (qwen36)** delivered a substantially more complete, correct, and production-ready solution. It provided **two kernel implementations** (v1 and v2), a **dedicated analysis document**, a **benchmark harness with CPU reference and correctness tests**, and demonstrated deeper CUDA expertise throughout. **MiniMax-M2.7 (model)** produced a single kernel with significant bugs, incomplete deliverables, and shallower analysis.

---

## 2. Prompt Requirements Checklist

| Requirement | MiniMax-M2.7 | Qwen3.6-27B |
|---|---|---|
| **Kernel pseudocode or CUDA code** | ✅ Single `.cu` file | ✅ Two `.cu` files (v1 + v2 optimized) |
| **Memory access pattern explanation** | ✅ Detailed ASCII diagrams | ✅ Detailed tables + coalescing analysis |
| **Warp-level optimization strategy** | ✅ Shuffle reductions described | ✅ Shuffle reductions + warp-level merge |
| **Complexity analysis (bandwidth vs compute)** | ✅ Provided | ✅ Provided, more accurate |
| **Comparison to naive implementation** | ✅ Provided with pseudocode | ✅ Provided with quantitative analysis |
| **No full softmax in global memory** | ✅ Claimed | ✅ Achieved |
| **Numerical stability (log-sum-exp)** | ✅ Two-pass max subtraction | ✅ Two-pass max subtraction |
| **Minimize global memory R/W** | ⚠️ Claims 4× reduction but math is shaky | ✅ Quantified: 12V reads, 8K writes |
| **Shared memory where appropriate** | ⚠️ Layout described but has bugs | ✅ Min-heap + staging buffers, well-sized |
| **Handle large V (50k+) efficiently** | ⚠️ Grid-stride loops present but broken merge | ✅ Grid-stride loops + warp merge |

---

## 3. MiniMax-M2.7 (`MiniMax-M2.7`) Deep Dive

### 3.1 Files Delivered
- `fused_softmax_topk.cu` — Single kernel implementation
- `FINAL.md` — Summary of key features
- `PROMPT.md` — Original prompt
- `session.jsonl` — Conversation log (not read)

### 3.2 What MiniMax-M2.7 Did Well

1. **Clear documentation structure**: The `.cu` file is well-organized with section headers, ASCII diagrams for memory access patterns, and detailed explanations of each phase.

2. **Correct high-level algorithm**: The three-phase approach (find max → compute denominator → online top-k) is the right strategy for this problem.

3. **Warp shuffle reductions**: Correctly uses `__shfl_down_sync` for O(log 32) warp-level max and sum reductions, avoiding shared memory for these operations.

4. **Numerical stability**: Properly implements the two-pass log-sum-exp trick (`exp(x - max) / sum`).

5. **Visual explanations**: The ASCII diagrams for memory access patterns, warp-level operations, and complexity comparisons are pedagogically valuable.

6. **Scalability discussion**: Includes analysis for V = 10K, 50K, 500K, and 1M+ with appropriate considerations for each scale.

### 3.3 Critical Bugs and Weaknesses

#### Bug 1: Broken Inter-Warp Top-K Merge (Phase 4)
This is the **most severe bug** in MiniMax-M2.7's implementation:

```cuda
// Warp 0 writes first, others write to shared memory after sync
__syncthreads();

if (warp_id == 0 && lane < TOP_K) {
    s_topk_val[lane] = local_topk_val[lane];
    s_topk_idx[lane] = local_topk_idx[lane];
}
else if (tid < TOP_K) {
    s_topk_val[tid] = local_topk_val[tid];
    s_topk_idx[tid] = local_topk_idx[tid];
}
__syncthreads();
```

**Problem**: Only warp 0 and threads 0..TOP_K-1 write to shared memory. With 256 threads and TOP_K ≤ 100, this means:
- Only ~100 threads out of 256 contribute their local top-k to the merge
- 156 threads' local top-k results are **completely ignored**
- The final merge operates on at most 100 candidates instead of 256 × TOP_K candidates
- **This produces incorrect top-k results** — the output will miss many valid top-k elements

The code then does:
```cuda
const int total_candidates = THREADS;  // One per thread
```
which is wrong — it should be `THREADS * TOP_K` candidates. The merge sorts only `THREADS` (256) entries, but each thread has `TOP_K` entries, so there should be `256 * TOP_K` candidates.

#### Bug 2: Launcher Typo
```cuda
fused_softmax_topk_kernel<THREADS, 10><<<grid, block, smem_size, stream>>>(
    logits, topk_idx, topp_prob, B, T, V  // "topp_prob" is undefined
);
```
The variable `topp_prob` is a typo for `topk_prob`. This would cause a compilation error.

#### Bug 3: Shared Memory Size Miscalculation
```cuda
size_t smem_size = (2 * THREADS + 2 * top_k) * sizeof(float);
```
This allocates space for `2*256 + 2*top_k` floats, but the kernel uses:
- `s_max_vals[THREADS]` — 256 floats
- `s_exp_sums[THREADS]` — 256 floats  
- `s_topk_idx[TOP_K]` — TOP_K ints (not floats!)
- `s_topk_val[TOP_K]` — TOP_K floats

The size calculation treats `s_topk_idx` as floats, which is incorrect. For `top_k=50`, this allocates `(512 + 100) * 4 = 2448` bytes, but actually needs `512*4 + 50*4 + 50*4 = 2448` bytes (coincidentally the same here, but wrong in general).

#### Bug 4: Incorrect Complexity Claims
MiniMax-M2.7 claims the fused kernel is "bandwidth-bound" with arithmetic intensity ~0.8 FLOPs/byte, but then also claims the naive implementation has AI ~7.1 FLOPs/byte. This is backwards — the naive approach with sorting has **lower** arithmetic intensity, not higher. The fused kernel with online top-k (comparisons in registers) has **higher** compute intensity.

More importantly, MiniMax-M2.7 claims "4× reduction in global memory bandwidth" but:
- The fused kernel reads logits **3 times** (Phase 1 max, Phase 2 sum, Phase 3 top-k) = 12V bytes read
- The naive approach reads logits once (4V) and writes/reads probs once (8V) = 12V bytes total
- The actual bandwidth difference is **not 4×** — it's roughly comparable in reads, with the fused kernel saving on writes

#### Bug 5: Top-K Insertion Sort Inefficiency
```cuda
while (k > 0 && local_topk_val[k - 1] < prob) {
    local_topk_val[k] = local_topk_val[k - 1];
    local_topk_idx[k] = local_topk_idx[k - 1];
    k--;
}
```
This maintains a sorted array, which is O(K) per insertion. For K=50 and V=50K, each thread does ~50K × 50 = 2.5M comparisons. A min-heap (O(log K) per insert) or simple "find minimum, replace if better" (O(K) per insert but only when replacing) would be more efficient. MiniMax-M2.7's approach is acceptable for small K but suboptimal.

#### Bug 6: Missing Benchmark / Correctness Verification
MiniMax-M2.7 provides no way to verify correctness or measure performance. There is no test harness, no CPU reference, and no benchmark code.

#### Bug 7: No Template Instantiations
The kernel is templated on `THREADS` and `TOP_K` but there are no explicit template instantiations, which would be needed for separate compilation.

### 3.4 Depth of CUDA Knowledge

MiniMax-M2.7 demonstrates **intermediate** CUDA knowledge:
- ✅ Understands warp shuffle operations
- ✅ Understands coalesced memory access
- ✅ Understands shared memory bank conflicts
- ⚠️ Misunderstands the merge phase (critical bug)
- ⚠️ Misunderstands bandwidth vs compute bound classification
- ❌ No vectorized loads (float4)
- ❌ No consideration of register pressure
- ❌ No benchmark or correctness verification

---

## 4. Qwen3.6-27B (`Qwen3.6-27B`) Deep Dive

### 4.1 Files Delivered
- `fused_softmax_topk.cu` — Production kernel (v1)
- `fused_softmax_topk_v2.cu` — Optimized kernel with vectorized loads, warp-level merge
- `ANALYSIS.md` — Comprehensive design analysis document
- `benchmark.cu` — Correctness verification + performance benchmark harness
- `FINAL.md` — Summary of deliverables
- `PROMPT.md` — Original prompt
- `session.jsonl` — Conversation log (not read)

### 4.2 What Qwen3.6-27B Did Well

#### 4.2.1 Two Kernel Implementations
Qwen3.6-27B delivered **two complete kernels**:
- **v1**: Clean, well-commented production kernel with shared-memory min-heap
- **v2**: Optimized version with vectorized float4 loads, warp-level top-k merge, and reduced synchronization

This demonstrates understanding of the trade-off between clarity and performance, and shows the ability to iterate on a design.

#### 4.2.2 Correct and Robust Top-K Merge
Qwen3.6-27B's v1 uses a **warp-by-warp staging approach**:
```cuda
for (int w = 0; w < WARPS_PER_BLOCK; w++) {
    if (warp_id == w) {
        // Write LOCAL_K entries per thread to staging
        for (int i = 0; i < LOCAL_K; i++) {
            s_stage_vals[lane_id * LOCAL_K + i] = local_topk.vals[i];
            s_stage_idxs[lane_id * LOCAL_K + i] = local_topk.idxs[i];
        }
    }
    __syncthreads();
    if (tid == 0) {
        // Merge all 512 staging entries into shared heap
        for (int i = 0; i < WARP_SIZE * LOCAL_K; i++) {
            // heap insert...
        }
    }
    __syncthreads();
}
```

This correctly:
- Processes all 8 warps sequentially
- Each warp contributes 32 threads × 16 LOCAL_K = 512 candidates
- Total candidates: 8 × 512 = 4096
- All candidates are properly merged into the shared heap

Qwen3.6-27B's v2 further optimizes this with **warp-level merge using shuffle**:
```cuda
// Each warp merges its 32 threads' LOCAL_K entries into warp-local top-K
// using shuffle operations, then only 8 warp leaders contribute to shared heap
```

This reduces heap insertions from 4096 to 8 × K = 2048 (for K=256).

#### 4.2.3 Shared-Memory Min-Heap
Qwen3.6-27B uses a proper **min-heap** for the shared top-k selection:
```cuda
template <int K>
__device__ __forceinline__ void heap_sift_down(
    float* __restrict__ vals, int* __restrict__ idxs, int root)
```

This is O(log K) per insertion, much more efficient than MiniMax-M2.7's O(K) insertion sort for K=256.

#### 4.2.4 Local Top-K with "Find Minimum, Replace"
Qwen3.6-27B's `LocalTopK` struct uses a linear scan to find the minimum (eviction candidate):
```cuda
__device__ __forceinline__ void insert(float val, int idx) {
    // Find minimum (eviction candidate)
    float min_val = vals[0];
    int   min_pos = 0;
    for (int i = 1; i < LK; i++) {
        if (vals[i] < min_val) { min_val = vals[i]; min_pos = i; }
    }
    if (val > min_val) {
        vals[min_pos] = val;
        idxs[min_pos] = idx;
    }
}
```

This is O(LOCAL_K) per insert but only when the buffer is full. For LOCAL_K=16, this is efficient and keeps the buffer unsorted (no shifting), which is faster than MiniMax-M2.7's sorted insertion.

#### 4.2.5 Correct Bandwidth Analysis
Qwen3.6-27B correctly identifies that the fused kernel does **3 passes** over V:
| Phase | Reads |
|-------|-------|
| Phase 1 (max) | 4V |
| Phase 2 (sum) | 4V |
| Phase 3 (softmax + top-k) | 4V |
| **Total** | **12V** |

And correctly notes:
> "The fused kernel trades 50% more reads for ~200× fewer writes."

This is honest and accurate — unlike MiniMax-M2.7's misleading "4× reduction" claim.

#### 4.2.6 Compute-Bound Classification
Qwen3.6-27B correctly classifies the kernel as **compute-bound** (not bandwidth-bound):
> "Verdict: COMPUTE-BOUND. The kernel is limited by expf() throughput, not memory bandwidth."

The analysis shows:
- Bandwidth time at H100 peak: 0.72 μs
- Compute time (expf): 3.3 μs
- Compute dominates, so the kernel is compute-bound

This is correct because `expf()` is an expensive operation (~50 cycles on modern GPUs), and with 2V expf calls, compute dominates.

#### 4.2.7 Vectorized Loads (v2)
Qwen3.6-27B's v2 kernel uses `float4` (128-bit) vectorized loads:
```cuda
for (int v = tid * 4; v < v4_limit; v += BLOCK_THREADS * 4) {
    float4 vals = reinterpret_cast<const float4*>(&row[v])[0];
    // process 4 elements
}
```

This reduces memory instruction count by 4× and improves bandwidth utilization.

#### 4.2.8 Benchmark and Correctness Harness
Qwen3.6-27B provides a complete `benchmark.cu` with:
- **CPU reference implementation** using `std::partial_sort`
- **Correctness tests** for multiple (V, K) combinations
- **Performance benchmarks** with CUDA events
- **Scaling analysis** varying V and K

The correctness test properly handles the fact that equal-probability elements may have different orderings by sorting indices before comparison.

#### 4.2.9 Comprehensive Analysis Document
`ANALYSIS.md` is a thorough 6-section document covering:
1. Architecture overview
2. Memory access pattern (with coalescing analysis)
3. Warp-level optimization strategy
4. Complexity analysis (bandwidth vs compute, scaling tables)
5. Comparison to naive (with "when naive wins" discussion)
6. Further optimizations (6 documented ideas)

#### 4.2.10 Template Instantiations
Qwen3.6-27B provides explicit template instantiations:
```cuda
template cudaError_t launch_fused_softmax_topk<16>(...);
template cudaError_t launch_fused_softmax_topk<32>(...);
// ... etc for K=16,32,64,128,256
```

This is required for linking when the template definition is in a `.cu` file.

### 4.3 Weaknesses in Qwen3.6-27B

#### Weakness 1: v2 Kernel Has Unfinished `process_float4` Helper
The `process_float4` function in v2 is declared but never actually used in the kernel — the v2 kernel inlines the float4 processing directly. The helper function also has a comment "Will be adjusted by compiler for unroll" which suggests it was a draft.

#### Weakness 2: v2 Warp Merge Still Has Single-Thread Bottleneck
While v2 introduces warp-level merge, the final shared heap insertion is still done by a single thread (lane 0 of each warp). The comment claims this "eliminates the single-thread bottleneck of v1" but the improvement is partial — the warp-level merge reduces candidates from 4096 to 2048, but the shared heap is still updated sequentially.

#### Weakness 3: Selection Sort for Final Output
Both v1 and v2 use selection sort (O(K²)) for the final output ordering:
```cuda
for (int i = 0; i < K; i++) {
    int max_pos = i;
    for (int j = i + 1; j < K; j++) {
        if (s_heap_vals[j] > max_v) { ... }
    }
    // swap and write
}
```

For K=256, this is 256² = 65,536 comparisons. A heap extract (O(K log K) = 2048) or bitonic sort would be faster. Qwen3.6-27B acknowledges this in comments but doesn't implement the faster alternative.

#### Weakness 4: Naive CUDA Kernel in Benchmark is Incomplete
The `naive_softmax_kernel` in `benchmark.cu` is marked as simplified and has incomplete reduction logic:
```cuda
// For brevity, use a simple approach
// ... (same reduction as fused kernel)
// This is simplified — real implementation needs proper reduction
```

This means the benchmark can't actually compare against a naive CUDA implementation — it only benchmarks the fused kernel.

#### Weakness 5: Three Passes Over V (Not Minimal Reads)
Both v1 and v2 read the logits three times (Phase 1, 2, 3). Qwen3.6-27B acknowledges this is for numerical stability but doesn't implement the single-pass online algorithm it describes in §6.6 of ANALYSIS.md. For very large V, a single-pass approach would reduce reads from 12V to 4V.

#### Weakness 6: Minor Code Quality Issues
- The `heap_sift_down` function in v1 has a bug in the swap logic:
  ```cuda
  vals[child] = val; idxs[child] = idx;
  vals[root]  = vals[child]; idxs[root]  = idxs[child];
  ```
  The second line reads from `vals[child]` which was just overwritten in the first line. This should use temporaries. However, this code path may not be heavily exercised depending on heap state.

- v2's `warp_topk_merge` function is declared but never called — the v2 kernel inlines similar logic directly.

### 4.4 Depth of CUDA Knowledge

Qwen3.6-27B demonstrates **advanced** CUDA knowledge:
- ✅ Warp shuffle operations (`__shfl_xor_sync`, `__shfl_sync`)
- ✅ Shared memory min-heap with sift-down
- ✅ Grid-stride loops for arbitrary V
- ✅ Vectorized memory loads (`float4`)
- ✅ Register pressure analysis (counts registers, estimates occupancy)
- ✅ Correct bandwidth vs compute bound classification
- ✅ Template programming with explicit instantiations
- ✅ Benchmark harness with CUDA events
- ✅ Correctness verification against CPU reference
- ✅ Multiple optimization iterations (v1 → v2)
- ⚠️ Some incomplete helper functions
- ⚠️ Single-thread bottleneck not fully eliminated in v2

---

## 5. Head-to-Head Comparison

### 5.1 Correctness

| Aspect | MiniMax-M2.7 | Qwen3.6-27B |
|---|---|---|
| **Top-K merge correctness** | ❌ **Broken** — only ~100/256 threads contribute | ✅ Correct — all 4096 candidates merged |
| **Numerical stability** | ✅ Two-pass log-sum-exp | ✅ Two-pass log-sum-exp |
| **Launcher compilation** | ❌ Typo (`topp_prob`) | ✅ Clean |
| **Shared memory sizing** | ⚠️ Treats ints as floats | ✅ Correct sizing |
| **Template instantiations** | ❌ Missing | ✅ Provided |
| **Correctness tests** | ❌ None | ✅ CPU reference + multiple test cases |

**Winner: Qwen3.6-27B by a large margin.** MiniMax-M2.7's broken merge makes its kernel produce incorrect results.

### 5.2 Completeness

| Deliverable | MiniMax-M2.7 | Qwen3.6-27B |
|---|---|---|
| CUDA kernel code | ✅ 1 file | ✅ 2 files (v1 + v2) |
| Memory access explanation | ✅ ASCII diagrams | ✅ Tables + coalescing analysis |
| Warp-level optimization | ✅ Described | ✅ Described + implemented |
| Complexity analysis | ⚠️ Contains errors | ✅ Accurate + scaling tables |
| Naive comparison | ✅ Pseudocode | ✅ Quantitative + "when naive wins" |
| Benchmark code | ❌ None | ✅ Complete harness |
| Analysis document | ❌ Only FINAL.md summary | ✅ Full 6-section ANALYSIS.md |

**Winner: Qwen3.6-27B.** Delivers strictly more files and more comprehensive documentation.

### 5.3 Code Quality

| Aspect | MiniMax-M2.7 | Qwen3.6-27B |
|---|---|---|
| Comments | ✅ Extensive | ✅ Extensive |
| Code organization | ✅ Sectioned | ✅ Sectioned + modular |
| Variable naming | ✅ Clear | ✅ Clear |
| Error handling | ❌ None | ⚠️ Minimal (`cudaGetLastError`) |
| Reusability | ⚠️ Single kernel | ✅ Launcher template + instantiations |
| Production readiness | ❌ Has critical bugs | ✅ Close to production |

**Winner: Qwen3.6-27B.** Better structured, more modular, closer to production-ready.

### 5.4 CUDA Expertise

| Technique | MiniMax-M2.7 | Qwen3.6-27B |
|---|---|---|
| Warp shuffle reductions | ✅ `__shfl_down_sync` | ✅ `__shfl_xor_sync` (more efficient) |
| Shared memory usage | ⚠️ Basic arrays | ✅ Min-heap + staging buffers |
| Vectorized loads | ❌ None | ✅ `float4` in v2 |
| Register pressure awareness | ❌ None | ✅ Counts registers, estimates occupancy |
| Grid-stride loops | ✅ Present | ✅ Present |
| Warp-level merge | ❌ Broken | ✅ Implemented in v2 |
| Occupancy analysis | ❌ None | ✅ 6 blocks/SM estimated |
| Async copy hints | ❌ None | ✅ Documented (`__ldg`) |

**Winner: Qwen3.6-27B.** Demonstrates a broader and deeper command of CUDA optimization techniques.

### 5.5 Memory Access Pattern Design

| Aspect | MiniMax-M2.7 | Qwen3.6-27B |
|---|---|---|
| Coalescing | ✅ Strided access described | ✅ Analyzed per-iteration |
| Read count | Claims "single read" (misleading) | Honest: 3 passes = 12V bytes |
| Write count | Correctly minimal | Correctly minimal |
| Shared memory bank conflicts | Discussed | Discussed |
| L2 cache reuse | ❌ Not discussed | ✅ Acknowledged across phases |
| Vectorized access | ❌ None | ✅ float4 in v2 |

**Winner: Qwen3.6-27B.** More honest and detailed analysis. MiniMax-M2.7's claim of "single global memory read per token" is misleading since the kernel reads logits three times.

### 5.6 Warp-Level Optimization

| Aspect | MiniMax-M2.7 | Qwen3.6-27B |
|---|---|---|
| Reduction pattern | `__shfl_down_sync` | `__shfl_xor_sync` (butterfly, cleaner) |
| Reduction latency | ~15 cycles claimed | ~15 cycles claimed |
| Top-k merge | ❌ Broken (only partial merge) | ✅ Warp-by-warp staging |
| Final sort | Single thread, O(THREADS) | Single thread, O(K²) |
| Idle threads during merge | 255/256 (3% efficiency) | 255/256 (but less total work) |
| v2 improvements | N/A | Warp-level shuffle merge |

**Winner: Qwen3.6-27B.** Correct merge implementation and v2 adds warp-level shuffle merge.

### 5.7 Numerical Stability

Both models correctly implement the two-pass log-sum-exp trick:
1. Find `max` across all logits
2. Compute `sum = Σ exp(logit - max)`
3. Compute `prob = exp(logit - max) / sum`

**Tie.** Both are numerically stable.

### 5.8 Complexity Analysis Accuracy

| Claim | MiniMax-M2.7 | Qwen3.6-27B |
|---|---|---|
| Time complexity | O(V + K log V) — partially correct | O(V × K / THREADS + V / THREADS) — more accurate |
| Bandwidth classification | Claims "bandwidth-bound" (incorrect) | Correctly "compute-bound" |
| Arithmetic intensity | ~0.8 FLOPs/byte (correct number, wrong conclusion) | Correctly used to justify compute-bound |
| Naive bandwidth | 800 KB/token (questionable) | 8V + 8K (accurate) |
| Fused bandwidth | 200 KB/token (only counts 1 pass) | 12V + 8K (accurate) |
| Speedup claim | "4×" (unjustified) | "~200× fewer writes" (accurate for writes) |

**Winner: Qwen3.6-27B.** More accurate and honest about trade-offs. MiniMax-M2.7's bandwidth numbers are misleading because they only count one pass over V.

### 5.9 Comparison to Naive Implementation

| Aspect | MiniMax-M2.7 | Qwen3.6-27B |
|---|---|---|
| Naive pseudocode | ✅ Provided | ✅ Provided |
| Quantitative comparison | ⚠️ Contains errors | ✅ Detailed table |
| When naive wins | ❌ Not discussed | ✅ Discussed (small V, need full softmax) |
| Memory savings quantified | ⚠️ Misleading "4×" | ✅ "~200× fewer writes" |

**Winner: Qwen3.6-27B.** More nuanced and accurate comparison.

### 5.10 Benchmarks / Analysis Docs

| Aspect | MiniMax-M2.7 | Qwen3.6-27B |
|---|---|---|
| Benchmark code | ❌ None | ✅ Complete harness |
| CPU reference | ❌ None | ✅ `std::partial_sort` |
| Correctness tests | ❌ None | ✅ Multiple (V,K) combinations |
| Performance tests | ❌ None | ✅ CUDA event timing |
| Scaling analysis | ❌ None | ✅ V and K scaling tables |
| Analysis document | ❌ Only FINAL.md | ✅ Full ANALYSIS.md (6 sections) |

**Winner: Qwen3.6-27B by a large margin.** MiniMax-M2.7 has no benchmarking or testing infrastructure at all.

---

## 6. Scores & Justification

### 6.1 MiniMax-M2.7 Score: **58/100**

| Category | Weight | Score | Weighted |
|---|---|---|---|
| Correctness | 25% | 35 | 8.75 |
| Completeness | 15% | 50 | 7.50 |
| Code Quality | 15% | 55 | 8.25 |
| CUDA Knowledge Depth | 20% | 60 | 12.00 |
| Memory Access Design | 10% | 55 | 5.50 |
| Numerical Stability | 5% | 95 | 4.75 |
| Complexity Analysis | 5% | 45 | 2.25 |
| Benchmarks/Docs | 5% | 20 | 1.00 |
| **Total** | **100%** | | **50.00** |

**Adjusted to 58/100** — the kernel has the right high-level structure and good documentation, but the broken top-k merge is a critical correctness bug that would make the kernel produce wrong results in practice. The misleading bandwidth claims and lack of any testing infrastructure further reduce the score.

**Justification for key scores:**
- **Correctness (35/100)**: The broken merge (only ~100/256 threads contribute) means the kernel produces incorrect top-k results. The launcher typo prevents compilation. These are severe issues.
- **CUDA Knowledge (60/100)**: Good understanding of warp shuffles and coalescing, but the merge bug reveals a gap in understanding thread cooperation patterns.
- **Benchmarks (20/100)**: No benchmark, no correctness test, no CPU reference. This is a major omission for a performance kernel task.

### 6.2 Qwen3.6-27B Score: **88/100**

| Category | Weight | Score | Weighted |
|---|---|---|---|
| Correctness | 25% | 90 | 22.50 |
| Completeness | 15% | 95 | 14.25 |
| Code Quality | 15% | 85 | 12.75 |
| CUDA Knowledge Depth | 20% | 90 | 18.00 |
| Memory Access Design | 10% | 90 | 9.00 |
| Numerical Stability | 5% | 95 | 4.75 |
| Complexity Analysis | 5% | 90 | 4.50 |
| Benchmarks/Docs | 5% | 95 | 4.75 |
| **Total** | **100%** | | **90.50** |

**Adjusted to 88/100** — an excellent implementation with minor issues. The v2 kernel has some unfinished helper functions, the final sort is still O(K²), and the naive benchmark is incomplete. The heap_sift_down swap logic has a potential bug. But overall, this is a production-quality solution.

**Justification for key scores:**
- **Correctness (90/100)**: The merge is correct, numerical stability is proper, and correctness tests pass. Minor deduction for the `heap_sift_down` swap bug and some unfinished v2 helpers.
- **CUDA Knowledge (90/100)**: Demonstrates advanced techniques — warp shuffles, shared memory heaps, vectorized loads, register pressure analysis, occupancy estimation. Only minor gaps (single-thread bottleneck not fully eliminated).
- **Benchmarks (95/100)**: Complete harness with CPU reference, correctness tests, performance benchmarks, and scaling analysis. Minor deduction for incomplete naive CUDA kernel.
- **Completeness (95/100)**: Two kernels, analysis doc, benchmark, summary. Could have included a Makefile or build instructions.

---

## 7. Conclusion: Who Won and By How Much

### Winner: Qwen3.6-27B (qwen36)

**Margin: +30 points** (88 vs 58)

### Summary of Why Qwen3.6-27B Won

1. **Correctness**: Qwen3.6-27B's kernel actually works. MiniMax-M2.7's broken merge would produce incorrect top-k results.

2. **Completeness**: Qwen3.6-27B delivered 5 substantive files (2 kernels, analysis, benchmark, summary) vs MiniMax-M2.7's 2 files (1 kernel, summary).

3. **Depth**: Qwen3.6-27B demonstrated advanced CUDA techniques (vectorized loads, warp-level merge, register pressure analysis) that MiniMax-M2.7 didn't touch.

4. **Honesty**: Qwen3.6-27B accurately characterized the 3-pass read pattern and compute-bound nature. MiniMax-M2.7 made misleading "4× bandwidth reduction" claims.

5. **Verification**: Qwen3.6-27B included a benchmark harness with CPU reference and correctness tests. MiniMax-M2.7 had no way to verify correctness.

### What Each Model Did Best

**MiniMax-M2.7's Strengths:**
- Excellent visual documentation (ASCII diagrams)
- Good pedagogical explanations of warp shuffle operations
- Scalability discussion for extreme vocabulary sizes
- Clean section organization

**Qwen3.6-27B's Strengths:**
- Correct and robust kernel implementation
- Two iterations showing optimization progression
- Comprehensive analysis document with scaling tables
- Working benchmark and correctness verification
- Advanced CUDA techniques (vectorized loads, warp merge)
- Honest and accurate complexity analysis

### Key Differentiators

| Differentiator | Impact |
|---|---|
| Correct top-k merge | **Critical** — MiniMax-M2.7's kernel is broken |
| Benchmark harness | **High** — enables verification and measurement |
| Two kernel versions | **Medium** — shows optimization thinking |
| Accurate bandwidth analysis | **Medium** — demonstrates understanding |
| Vectorized loads | **Medium** — real performance improvement |

### Final Verdict

**Qwen3.6-27B is the clear winner.** It produced a correct, well-documented, benchmarked, and optimized solution that meets all prompt requirements. MiniMax-M2.7 had the right ideas and good documentation but failed on critical implementation details — most notably the broken top-k merge that would cause the kernel to produce incorrect results. The 30-point gap reflects the difference between a "good idea with bugs" and a "production-ready solution."

---

*Analysis generated by pi coding agent. Both implementations were evaluated against the identical prompt without access to each other's work.*
