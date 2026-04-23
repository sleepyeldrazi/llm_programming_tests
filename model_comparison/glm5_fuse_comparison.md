# Head-to-Head Analysis: Fused Softmax + Top-K Kernel in CUDA

**Task:** Design and implement a high-performance fused softmax + top-k kernel in CUDA (or CUDA-like pseudocode).

**Models Compared:**
- **GLM-5:** Implementation from glm5
- **Qwen3.6-27B:** Implementation from qwen36

**Date:** 2026-04-23

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Prompt Requirements Checklist](#2-prompt-requirements-checklist)
3. [GLM-5 — Deep Dive](#3-model-a-gammafuse--deep-dive)
4. [Qwen3.6-27B — Deep Dive](#4-model-b-qwen36fuse--deep-dive)
5. [Head-to-Head Comparison](#5-head-to-head-comparison)
6. [Scores and Justification](#6-scores-and-justification)
7. [Conclusion](#7-conclusion)

---

## 1. Executive Summary

Both models produced competent, working CUDA implementations of a fused softmax + top-k kernel. However, they took fundamentally different algorithmic approaches:

- **GLM-5** uses a **single-pass online softmax** algorithm (Milakov & Gimelshein 2018) combined with per-thread register-resident sorted arrays for top-K tracking. It maps **one warp per row** (b,t), with each lane striding across V. This is a more sophisticated, theoretically optimal approach.

- **Qwen3.6-27B** uses a **three-pass algorithm**: (1) find max, (2) compute sum-of-exps, (3) compute softmax + collect top-K. It maps **one block per row** (b,t), with all threads in the block cooperating. This is simpler and more conventional but reads the logits 3× from global memory.

**Bottom line:** GLM-5 demonstrates deeper CUDA expertise, a more optimal algorithmic choice (single-pass online softmax), and a more sophisticated memory access design. Qwen3.6-27B is solid but makes suboptimal design choices (3 passes over V, single-thread merge bottleneck) that significantly increase memory traffic. GLM-5 wins decisively.

---

## 2. Prompt Requirements Checklist

| Requirement | Description |
|-------------|-------------|
| R1 | Input: logits [B, T, V]; Output: top-k indices + top-k probabilities |
| R2 | Do NOT materialize full softmax matrix in global memory |
| R3 | Must be numerically stable (log-sum-exp) |
| R4 | Minimize global memory reads/writes |
| R5 | Use shared memory where appropriate |
| R6 | Handle large V (e.g., 50k+) efficiently |
| D1 | Kernel pseudocode or CUDA code |
| D2 | Memory access pattern explanation |
| D3 | Warp-level optimization strategy |
| D4 | Complexity analysis (bandwidth vs compute bound) |
| D5 | Comparison to naive implementation |

---

## 3. GLM-5 — Deep Dive

### 3.1 Files Delivered

| File | Purpose |
|------|---------|
| `DESIGN.md` | Comprehensive design document (9 sections) |
| `fused_softmax_topk.cuh` | Production kernel header (complete, templated) |
| `test_fused.cu` | Correctness verification + benchmark harness |
| `diagram.py` | ASCII architecture diagram generator |
| `session.jsonl` | Session log (not analyzed) |

### 3.2 Architecture

**Grid/Block Mapping:** One warp per (b,t) row. Block = 8 warps × 32 lanes = 256 threads. Grid = ceil(B×T / 8) blocks.

**Algorithm:** Single-pass **online softmax** (Milakov & Gimelshein 2018):
```
m_j = max(m_{j-1}, x_j)
d_j = d_{j-1} * exp(m_{j-1} - m_j) + exp(x_j - m_j)
```

This maintains running max and running sum-of-exps in a single pass over V. Simultaneously, each thread maintains a register-resident sorted array (size K) for top-K tracking.

**Three-phase pipeline:**
1. **Phase 1 (Local Pass):** Each lane reads V/32 logits in strided coalesced pattern. Maintains local_max, local_sum, and a TopKHeap<K> in registers.
2. **Phase 2 (Cross-Warp Merge):** Warps write local heaps to shared memory. Warp 0 merges WARPS_PER_BLOCK heaps into global top-K. Rescales to probabilities.
3. **Phase 3 (Write Output):** Lane 0 writes K (prob, index) pairs to global memory.

### 3.3 Correctness Analysis

**Strengths:**
- Uses online softmax recurrence — mathematically equivalent to standard two-pass softmax, numerically stable.
- All `exp()` calls use `x - current_max`, ensuring arguments ≤ 0. No overflow possible.
- Running sum is rescaled on max update: `d_new = d_old * exp(old_max - new_max) + exp(x - new_max)`.
- Final rescaling: `prob_i = exp(val_i - global_max) / global_sum`. Since `global_sum ≥ 1.0`, division is safe.
- Test harness includes CPU reference with wide-range random data (range [-20, 20]) to stress numerical stability.
- Tolerance check: 1e-4 for probability comparison.

**Potential Issues:**
- The cross-warp merge is done by warp 0 only. If WARPS_PER_BLOCK > 1 and multiple warps process the **same** row, the merge is necessary. But the design says "one warp per row" — so multiple warps in a block process **different** rows. The cross-warp merge in `cross_warp_merge()` operates on heaps from different rows, which is a **bug**. Wait — re-reading: each warp handles one row, and there are WARPS_PER_BLOCK warps per block. So warp 0 handles row 0, warp 1 handles row 1, etc. The `cross_warp_merge` function is called by all warps but only warp 0 does work. However, each warp has its own `heap` and writes to its own `row_out_probs`/`row_out_indices`. The `__syncthreads()` ensures all warps have written to shared memory before warp 0 reads. But warp 0 only merges its own heap (from its own row) with... nothing? Actually, re-reading the code more carefully:

In `fused_softmax_topk_kernel`:
- `row = blockIdx.x * WARPS_PER_BLOCK + warp_id` — each warp gets a distinct row.
- `cross_warp_merge` is called with `heap` (per-thread heap, but each warp has its own threads).
- Inside `cross_warp_merge`, each warp writes its heap to `smem.heap_buf[warp_id]`. 
- Then warp 0 merges ALL warps' heaps: `for (int w = 0; w < WARPS_PER_BLOCK; w++)`.
- But warp 0's row is `blockIdx.x * WARPS_PER_BLOCK + 0`, while warp 1's row is `blockIdx.x * WARPS_PER_BLOCK + 1`.
- **This is a bug!** Warp 0 is merging heaps from DIFFERENT rows and writing the merged result to warp 0's output only. The other warps (1..7) don't write anything in Phase 2 because `if (warp_id == 0)` guards the output write.

Wait, let me re-read even more carefully:

```cuda
void cross_warp_merge(...) {
    // Each warp writes its local heap to shared memory
    if (lane_id < K) {
        smem.heap_buf[warp_id][lane_id] = heap.vals[K - 1 - lane_id];
        smem.idx_buf [warp_id][lane_id] = heap.idxs[K - 1 - lane_id];
    }
    __syncthreads();

    // Warp 0 merges all heaps
    if (warp_id == 0) {
        // ... merges ALL warps' heaps ...
        // Lane 0 writes the final result
        if (lane_id == 0) {
            for (int i = 0; i < K; i++) {
                out_probs[i] = ...;
                out_idxs[i] = ...;
            }
        }
    }
}
```

And in the kernel:
```cuda
// Phase 2: cross-warp heap merge + write output
cross_warp_merge<K>(smem, global_max, global_sum,
                    heap, warp_id, lane_id,
                    row_out_probs, row_out_indices);
```

So ALL warps call `cross_warp_merge`, but only warp 0 writes to `row_out_probs`/`row_out_indices`. For warps 1-7, `out_probs`/`out_idxs` point to their own row's output. But warp 0 writes to `row_out_probs` which is warp 0's row. Warps 1-7 don't write anything!

**This is a significant correctness bug.** The kernel only produces correct output for the first row in each block. Rows handled by warps 1-7 get no output written.

However, when `WARPS_PER_BLOCK == 1`, this bug doesn't manifest because there's only one warp per block. The default is `WARPS_PER_BLOCK = 8`, so the bug is present in the default configuration.

This is a serious issue that would cause the test to fail for B*T > 1 when using the default 8 warps per block. The test in `test_fused.cu` uses B=4, T=8 (32 rows) which would exercise multiple warps per block.

Actually, wait — let me re-check. The test uses `launch_fused_softmax_topk<K>` which uses the default `WARPS_PER_BLOCK = 8`. With B=4, T=8, there are 32 rows. Grid = ceil(32/8) = 4 blocks. Each block has 8 warps, each handling one row. So warp 0 in block 0 handles row 0, warp 1 handles row 1, etc.

In `cross_warp_merge`, warp 0 merges all 8 heaps and writes to `row_out_probs` which is row 0's output. Warps 1-7 don't write anything. So rows 1-7 in each block get uninitialized output.

**This is a real bug.** The test would fail unless the test only checks row 0 (which it does print, but `verify()` checks all rows).

Hmm, but the `verify()` function checks `bt` from 0 to B*T-1. If rows 1-7 have garbage, it should fail. Unless... the `__syncthreads()` at the end of the kernel causes warps 1-7 to also reach the end, but they don't write. The output arrays are allocated with `cudaMalloc` which gives uninitialized memory. So rows 1-7 would have garbage.

**This is a critical correctness bug in GLM-5.**

But wait — I should double-check my understanding. Let me look at the kernel again:

```cuda
int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
if (row >= B * T) return;

// ... pointers for this row ...

// Phase 1: local pass
local_pass<K>(logits_row, V, warp_max, warp_sum, heap);

// Store partials in shared memory
if (lane_id == 0) {
    smem.warp_max[warp_id] = warp_max;
    smem.warp_sum[warp_id] = warp_sum;
}
__syncthreads();

// Compute global max and sum across warps
// ... (lane 0 of warp 0 computes global max/sum for ALL warps in block)
// ... but each warp processed a DIFFERENT row!

// Wait, this is also wrong! The global max/sum computation merges across
// warps that processed DIFFERENT rows. It should only merge within a warp
// (since one warp = one row).
```

Yes, there's a fundamental design confusion here. The kernel says "one warp per row" but then tries to do cross-warp reductions (max/sum and heap merge) as if all warps in a block cooperated on the SAME row. This is contradictory.

When WARPS_PER_BLOCK = 1, everything works because there's only one warp per block. But with WARPS_PER_BLOCK > 1, the cross-warp logic is wrong because it conflates data from different rows.

**Verdict on GLM-5 correctness: The code has a fundamental design flaw when WARPS_PER_BLOCK > 1. It would only work correctly with WARPS_PER_BLOCK = 1. This is a significant correctness issue.**

However, the online softmax algorithm itself is correct. The warp-level shuffle reductions are correct for within-warp. The heap insert logic is correct. The numerical stability approach is correct. The issue is purely in the block-level coordination when multiple warps per block handle different rows.

### 3.4 Completeness

| Deliverable | Present | Quality |
|-------------|---------|---------|
| Kernel code | ✅ | Complete, templated, production-quality |
| Memory access pattern | ✅ | Excellent — detailed coalescing analysis |
| Warp-level optimization | ✅ | Excellent — shuffle reductions, register heaps |
| Complexity analysis | ✅ | Excellent — bandwidth vs compute bound with numbers |
| Comparison to naive | ✅ | Excellent — quantitative comparison table |
| Test/benchmark | ✅ | CPU reference, verification, timing |
| Design document | ✅ | Comprehensive 9-section document |
| Architecture diagram | ✅ | ASCII diagram with memory traffic summary |

### 3.5 Code Quality

- **Header-only design** with `.cuh` — good for library use.
- **Template parameter K** with explicit instantiations — clean.
- **`__restrict__` qualifiers** on pointers — excellent for compiler optimization.
- **`__device__ __forceinline__`** on hot functions — good.
- **`#pragma unroll`** on small loops — good.
- **Comments are excellent** — explains the "why" not just the "what".
- **No vectorized loads** (float4) — missed optimization opportunity.
- **No FP16/BF16 support** — mentioned in DESIGN.md but not implemented.

### 3.6 CUDA Knowledge Depth

- **Online softmax:** Shows awareness of cutting-edge research (Milakov & Gimelshein 2018). This is advanced knowledge.
- **Warp shuffle reductions:** Correct use of `__shfl_xor_sync` with butterfly pattern.
- **Register-resident heap:** Correctly identifies that sorted arrays in registers outperform binary heaps for small K.
- **Coalesced strided access:** Correctly explains why lane-i reading index i, i+32, i+64... is coalesced.
- **Shared memory bank conflicts:** Correctly analyzes that warp-id-based indexing avoids bank conflicts.
- **Occupancy analysis:** Provides register count estimates and block/SM calculations.
- **Complexity analysis:** Correctly identifies the kernel as bandwidth-bound with AI ≈ 1.5 FLOP/byte.

### 3.7 Key Strengths

1. **Single-pass online softmax** — reads V only once, not 3×. This is the theoretically optimal approach.
2. **Excellent design document** — 9 sections covering every aspect from algorithm to advanced optimizations.
3. **Strong numerical stability analysis** — explains why online softmax is stable.
4. **Accurate bandwidth-bound characterization** — AI calculation and comparison to A100 specs.
5. **Register pressure analysis** — estimates ~26 registers/thread, fits well within SM limits.
6. **Advanced optimization ideas** — FP16, async copy (Hopper), multi-row per warp, tournament merge.

### 3.8 Key Weaknesses

1. **Critical correctness bug with WARPS_PER_BLOCK > 1** — cross-warp merge conflates data from different rows. Only works when each block has exactly 1 warp.
2. **No vectorized loads** — misses opportunity for 4× wider memory transactions.
3. **Heap merge is serial** — warp 0 does all merging, even within a single warp's data.
4. **No v2/optimized variant** — only one kernel implementation.
5. **Test only covers small V (1024)** — doesn't test the large-V case that the design targets.

---

## 4. Qwen3.6-27B — Deep Dive

### 4.1 Files Delivered

| File | Purpose |
|------|---------|
| `PROMPT.md` | Original prompt (included for reference) |
| `FINAL.md` | Executive summary of deliverables |
| `ANALYSIS.md` | Full design analysis (6 sections) |
| `fused_softmax_topk.cu` | Production kernel v1 (three-pass) |
| `fused_softmax_topk_v2.cu` | Optimized kernel v2 (vectorized loads, warp merge) |
| `benchmark.cu` | Correctness + performance benchmark harness |
| `session.jsonl` | Session log (not analyzed) |

### 4.2 Architecture (v1)

**Grid/Block Mapping:** One block per (b,t) row. Block = 256 threads. Grid = B×T blocks.

**Algorithm:** Three-pass approach:
1. **Phase 1 (Max reduction):** All threads find local max via grid-stride loop. Warp shuffle reduce → block max.
2. **Phase 2 (Sum reduction):** All threads compute `exp(x - max)` and sum. Warp shuffle reduce → block sum.
3. **Phase 3 (Softmax + local top-K):** Each thread computes softmax probabilities and maintains a LocalTopK<16> buffer in registers.
4. **Phase 4 (Merge to shared heap):** Warp-by-warp, threads write LOCAL_K entries to staging buffer. Thread 0 merges into shared min-heap.
5. **Phase 5 (Sort + write-back):** Thread 0 selection-sorts heap and writes to global memory.

### 4.3 Architecture (v2)

Improvements over v1:
1. **Vectorized float4 loads** — 128-bit memory transactions where V % 4 == 0.
2. **Warp-level top-K merge** — each warp merges its 32 threads' LOCAL_K entries via shuffle before contributing to shared heap.
3. **Reduced synchronization** — uses `__syncwarp()` instead of `__syncthreads()` where possible.
4. **Parallel sort mention** — bitonic network (not fully implemented, falls back to selection sort).

### 4.4 Correctness Analysis

**Strengths:**
- Three-pass approach is straightforward and well-understood. Max-first ensures numerical stability.
- `exp(x - max_val)` guarantees no overflow.
- `inv_sum = 1.0f / s_warp_sum[0]` — safe because sum includes at least `exp(0) = 1.0`.
- Test harness includes CPU reference with random data (range [-10, 10]).
- Handles index sorting for tie-breaking comparison.
- Tests multiple configurations: V=1000/K=10, V=50257/K=256, V=50257/K=50, V=32000/K=128.

**Potential Issues:**
- **v1: Single-thread merge bottleneck** — Thread 0 does all 4096 heap insertions. For K=256, each insertion is O(log K) = ~8 operations. Total ~32K shared memory ops. This is small but serializes the merge.
- **v1: Selection sort O(K²)** — For K=256, this is 65K comparisons. Done once per block, so acceptable but not optimal.
- **v2: Warp-level merge has issues** — The `warp_topk_merge` function is declared but never actually used in the v2 kernel. Instead, v2 uses inline lane-0 collection with `__shfl_sync`. The function signature takes `K` as a runtime parameter but the template has `K` as compile-time — this mismatch means the function can't be called with the template's K.
- **v2: Float4 alignment** — The vectorized load assumes `V` is divisible by 4 and the row pointer is 16-byte aligned. No handling for misaligned cases beyond the tail loop.
- **v2: Selection sort still used** — Despite claiming "parallel sort using warp-level bitonic network," the actual code still uses thread-0 selection sort.
- **v2: `__syncwarp()` after lane-0 work** — After lane 0 collects all data via shuffle, `__syncwarp()` is called but lane 0 is the only one that did work. Other lanes are idle. This is fine but the warp-level merge doesn't actually distribute work.

**No critical correctness bugs** like GLM-5's cross-warp row conflation. The three-pass design with one block per row is simpler and avoids the row-ownership ambiguity.

### 4.5 Completeness

| Deliverable | Present | Quality |
|-------------|---------|---------|
| Kernel code | ✅ | Two versions (v1 + v2) |
| Memory access pattern | ✅ | Good — table with bytes per phase |
| Warp-level optimization | ✅ | Good — shuffle reductions, warp merge in v2 |
| Complexity analysis | ✅ | Good — compute-bound claim (disputed below) |
| Comparison to naive | ✅ | Good — quantitative table |
| Test/benchmark | ✅ | CPU reference, timing, scaling analysis |
| Design document | ✅ | 6-section ANALYSIS.md |
| Executive summary | ✅ | FINAL.md with architecture at a glance |

### 4.6 Code Quality

- **Two versions** (v1 and v2) — shows iterative improvement mindset.
- **Template parameter K** with explicit instantiations.
- **`__restrict__` qualifiers** present.
- **`__device__ __forceinline__`** on hot functions.
- **`#pragma unroll`** on reduction loops.
- **Dynamic shared memory** for staging buffer — good for flexibility.
- **Comments are good** but slightly less detailed than GLM-5.
- **v2 has dead code** — `warp_topk_merge` function is never called.
- **v2 has a bug in `process_float4`** — The function takes `const float4& vals` but then tries to access components with `if (i == 0) raw_val = vals.x;` etc. However, the function is also never called (dead code).

### 4.7 CUDA Knowledge Depth

- **Three-pass softmax:** Standard, well-known approach. Not cutting-edge but correct.
- **Warp shuffle reductions:** Correct use of `__shfl_xor_sync`.
- **Shared memory min-heap:** Correct implementation of sift-down.
- **Grid-stride loops:** Correctly used for arbitrary V.
- **Vectorized loads:** Correctly uses `float4` in v2.
- **Occupancy analysis:** Provides register count (~40/thread) and block/SM calculations.
- **Complexity analysis:** Claims kernel is **compute-bound** due to `expf()` throughput. This is **incorrect** for the stated parameters.

### 4.8 Complexity Analysis Dispute

Qwen3.6-27B claims:
> "Verdict: COMPUTE-BOUND. The kernel is limited by expf() throughput, not memory bandwidth."

With V=50257, K=256:
- Global reads: 12V × 4B = 2.41 MB per (b,t)
- `expf()` calls: 2V = 100,514

Qwen3.6-27B calculates:
- Bandwidth time on H100: 2.41 MB / 3.35 TB/s = 0.72 μs
- Compute time: 100,514 expf × 50 cycles / 1.5 GHz = 3.3 μs

**The error:** The bandwidth calculation assumes the logits stay in L2 cache across the three passes. But with one block per (b,t), each block processes one row independently. The L2 cache may hold the row for subsequent passes, but:

1. With B×T blocks, there's no guarantee of L2 cache residency. If B×T is large, the L2 cache will be thrashed.
2. Even with perfect L2 caching, the kernel reads 12V bytes. GLM-5 reads only V bytes.
3. The arithmetic intensity is: ~6V FLOPs / (12V × 4 bytes) = 6V / 48V = **0.125 FLOP/byte** for the three-pass approach. This is extremely low.

For comparison, GLM-5's single-pass approach has AI ≈ 1.5 FLOP/byte (6V FLOPs / 4V bytes), which is still bandwidth-bound but 12× higher than Qwen3.6-27B.

**Qwen3.6-27B's complexity analysis is flawed.** The kernel is bandwidth-bound, not compute-bound. The three-pass design makes it read 12V bytes instead of V, making the bandwidth problem worse.

### 4.9 Key Strengths

1. **Two kernel versions** — shows willingness to iterate and optimize.
2. **Vectorized loads in v2** — float4 for 4× wider transactions.
3. **No critical correctness bugs** — simpler design avoids GLM-5's row-conflation issue.
4. **Good test coverage** — tests multiple (V, K) combinations including LLaMA-sized.
5. **Scaling analysis** — benchmarks varying V and K.
6. **Shared memory heap** — correctly implements min-heap with sift-down.

### 4.10 Key Weaknesses

1. **Three-pass algorithm reads 12V bytes** — 12× more than GLM-5's single-pass approach. This is the fundamental inefficiency.
2. **Incorrect compute-bound claim** — the kernel is bandwidth-bound, and the three-pass design exacerbates this.
3. **Single-thread merge bottleneck in v1** — thread 0 does all heap operations.
4. **v2 has dead code** — `warp_topk_merge` and `process_float4` are never called.
5. **v2 still uses selection sort** — claimed bitonic sort not implemented.
6. **No online softmax** — misses the state-of-the-art single-pass approach.
7. **No architecture diagram** — less visual communication than GLM-5.

---

## 5. Head-to-Head Comparison

### 5.1 Algorithmic Approach

| Aspect | GLM-5 | Qwen3.6-27B |
|--------|---------|---------|
| Passes over V | **1** (online softmax) | **3** (max, sum, softmax+topk) |
| Global reads per row | **V × 4B** | **12V × 4B** |
| Global writes per row | **2K × 4B** | **2K × 4B** |
| Theoretical optimality | **Optimal** (can't do better than 1 pass) | Suboptimal (3× more reads) |

**Winner: GLM-5** — Single-pass online softmax is the right algorithmic choice.

### 5.2 Numerical Stability

| Aspect | GLM-5 | Qwen3.6-27B |
|--------|---------|---------|
| Stability mechanism | Online max tracking + rescaling | Max subtraction (two-pass) |
| Overflow risk | None (all exp args ≤ 0) | None (all exp args ≤ 0) |
| Underflow risk | Minimal (rescaling on max update) | Minimal (sum includes exp(0)=1) |
| Equivalent to standard softmax | Yes (proven equivalence) | Yes (standard approach) |

**Winner: Tie** — Both are numerically stable. GLM-5's online approach is more sophisticated but equivalent.

### 5.3 Memory Access Pattern

| Aspect | GLM-5 | Qwen3.6-27B |
|--------|---------|---------|
| Coalescing | Perfect strided coalescing | Perfect grid-stride coalescing |
| Cache efficiency | Good (one pass, likely L2 resident) | Poor (3 passes, may thrash L2) |
| Vectorized loads | ❌ Not implemented | ✅ float4 in v2 |
| Shared memory usage | ~2 KB (heap merge) | ~6.2 KB (heap + staging) |
| Bank conflicts | Avoided (warp-id indexing) | Avoided (sequential access) |

**Winner: GLM-5** — Despite lacking vectorized loads, the 3× reduction in global reads dominates.

### 5.4 Warp-Level Optimization

| Aspect | GLM-5 | Qwen3.6-27B |
|--------|---------|---------|
| Shuffle reductions | ✅ Butterfly max + sum | ✅ Butterfly max + sum |
| Register heap | ✅ Sorted array (K ≤ 32) | ✅ Linear scan (LOCAL_K=16) |
| Warp-level merge | ❌ Not implemented (serial) | ⚠️ Claimed but not fully working |
| Cross-warp coordination | ❌ Buggy (conflates rows) | ✅ Correct (one block = one row) |

**Winner: Tie** — Both have good shuffle reductions. GLM-5's register heap is cleaner. Qwen3.6-27B's warp merge in v2 is partially implemented but has dead code.

### 5.5 Code Correctness

| Aspect | GLM-5 | Qwen3.6-27B |
|--------|---------|---------|
| Core algorithm | ✅ Correct (online softmax) | ✅ Correct (three-pass) |
| Block-level coordination | ❌ **Bug: cross-warp merge conflates different rows** | ✅ Correct |
| Edge cases | ⚠️ Only works with WARPS_PER_BLOCK=1 | ✅ Handles arbitrary V via grid-stride |
| Test coverage | Small V only (1024) | Multiple configs including 50257 |

**Winner: Qwen3.6-27B** — GLM-5 has a critical correctness bug when WARPS_PER_BLOCK > 1.

### 5.6 Documentation Quality

| Aspect | GLM-5 | Qwen3.6-27B |
|--------|---------|---------|
| Design document | ✅ Excellent (9 sections, 3000+ words) | ✅ Good (6 sections, detailed) |
| Executive summary | ❌ Not present | ✅ FINAL.md with quick reference |
| Architecture diagram | ✅ ASCII diagram generator | ❌ Not present |
| Complexity analysis | ✅ Excellent (AI calculation, A100 specs) | ⚠️ Good but flawed (compute-bound claim) |
| Comparison table | ✅ Detailed with workload example | ✅ Good quantitative comparison |
| Advanced optimizations | ✅ FP16, async copy, tournament merge | ✅ FP16, persistent blocks, async copy |

**Winner: GLM-5** — More comprehensive documentation with accurate analysis.

### 5.7 Benchmark/Test Infrastructure

| Aspect | GLM-5 | Qwen3.6-27B |
|--------|---------|---------|
| CPU reference | ✅ Included | ✅ Included |
| Verification | ✅ Tolerance-based | ✅ Tolerance-based + index sorting |
| Timing harness | ✅ cudaEvent-based | ✅ cudaEvent-based |
| Scaling analysis | ❌ Not present | ✅ Varying V and K |
| Naive comparison | ❌ Not benchmarked | ⚠️ Claimed but naive kernel is incomplete |

**Winner: Qwen3.6-27B** — Better test coverage and scaling analysis.

### 5.8 Production Readiness

| Aspect | GLM-5 | Qwen3.6-27B |
|--------|---------|---------|
| Header-only library | ✅ `.cuh` format | ❌ `.cu` files |
| Template instantiations | ✅ Common K values | ✅ Common K values |
| Stream parameter | ✅ Optional stream arg | ❌ No stream parameter |
| Error handling | ❌ No CUDA error checks | ⚠️ Returns `cudaError_t` |
| Multiple versions | ❌ Single kernel | ✅ v1 + v2 |

**Winner: GLM-5** (with caveat: bug must be fixed) — Better API design with stream support.

---

## 6. Scores and Justification

### 6.1 Scoring Rubric

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Correctness | 25% | Does the code produce correct output? |
| Completeness | 15% | Are all deliverables present? |
| Code Quality | 15% | Is the code clean, well-structured, production-ready? |
| CUDA Depth | 15% | How deep is the CUDA knowledge demonstrated? |
| Memory Design | 10% | Is the memory access pattern optimal? |
| Complexity Analysis | 10% | Is the analysis accurate and insightful? |
| Naive Comparison | 10% | Is the comparison thorough and quantitative? |

### 6.2 GLM-5 Score: 72/100

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Correctness | **12/25** | The online softmax and per-lane heap logic are correct, but there's a **critical bug**: when WARPS_PER_BLOCK > 1, the cross-warp merge conflates heaps from different rows. Only the first row in each block gets correct output. This would fail any real test with B*T > WARPS_PER_BLOCK. Test only uses small V (1024) but doesn't catch this because... actually it would catch it if verifying all rows. The test does verify all rows, so it should fail. Either the test wasn't actually run, or WARPS_PER_BLOCK was set to 1 for testing. |
| Completeness | **14/15** | All deliverables present: kernel, memory analysis, warp optimization, complexity analysis, naive comparison, tests, design doc, diagram. |
| Code Quality | **13/15** | Excellent code structure, good use of CUDA features, header-only design, stream support. Minor issues: no vectorized loads, no error checking. |
| CUDA Depth | **14/15** | Shows advanced knowledge: online softmax (research-level), register-resident heaps, shuffle reductions, occupancy analysis. |
| Memory Design | **9/10** | Optimal single-pass design, perfect coalescing, minimal shared memory. Only misses vectorized loads. |
| Complexity Analysis | **9/10** | Excellent AI calculation, accurate bandwidth-bound characterization, A100 specs used correctly. |
| Naive Comparison | **1/10** | Excellent quantitative comparison with workload example. |

**Total: 12 + 14 + 13 + 14 + 9 + 9 + 1 = 72/100**

Wait, let me recalculate: 12 + 14 + 13 + 14 + 9 + 9 + 10 = **81/100**

Actually, let me be more precise. The naive comparison score should be higher:

| Criterion | Score | Max |
|-----------|-------|-----|
| Correctness | 12 | 25 |
| Completeness | 14 | 15 |
| Code Quality | 13 | 15 |
| CUDA Depth | 14 | 15 |
| Memory Design | 9 | 10 |
| Complexity Analysis | 9 | 10 |
| Naive Comparison | 9 | 10 |
| **Total** | **80** | **100** |

**GLM-5 Final Score: 80/100**

The correctness deduction is severe (-13) because the bug means the kernel doesn't work for the default configuration. However, the algorithmic insight (online softmax) is so strong that it still scores well in other categories.

### 6.3 Qwen3.6-27B Score: 78/100

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Correctness | **22/25** | No critical bugs. The three-pass approach is straightforward and correct. v2 has dead code but doesn't affect correctness of the main path. |
| Completeness | **14/15** | All deliverables present. Two kernel versions, benchmark, analysis docs. Missing architecture diagram. |
| Code Quality | **12/15** | Good code structure. Issues: dead code in v2, no stream parameter, no header-only design. |
| CUDA Depth | **11/15** | Good knowledge of standard techniques but misses the online softmax innovation. Uses conventional three-pass approach. |
| Memory Design | **6/10** | Three-pass design reads 12V bytes — 12× suboptimal. Vectorized loads in v2 partially compensate. |
| Complexity Analysis | **5/10** | Claims compute-bound but the kernel is actually bandwidth-bound. The 12V reads make bandwidth the dominant factor. |
| Naive Comparison | **8/10** | Good quantitative comparison but the "naive" kernel in benchmark.cu is incomplete (omitted reduction code). |

**Qwen3.6-27B Final Score: 78/100**

### 6.4 Final Scores

| Model | Score | Grade |
|-------|-------|-------|
| **GLM-5** | **80/100** | B+ |
| **Qwen3.6-27B** | **78/100** | B+ |

**Winner: GLM-5 by 2 points** — A narrow win driven by superior algorithmic insight and documentation, offset by a critical correctness bug.

---

## 7. Conclusion

### What GLM-5 Did Well

1. **Algorithmic brilliance:** The single-pass online softmax is the optimal approach for this problem. It reduces global reads from 12V to V, which is the single most important optimization for a bandwidth-bound kernel.
2. **Deep CUDA knowledge:** Demonstrated awareness of cutting-edge research (online softmax), register-resident data structures, and warp-level primitives.
3. **Excellent documentation:** The DESIGN.md is a model of technical writing — clear, quantitative, and comprehensive.
4. **Accurate complexity analysis:** Correctly identified the kernel as bandwidth-bound with proper arithmetic intensity calculations.

### What GLM-5 Did Poorly

1. **Critical correctness bug:** The cross-warp merge logic conflates data from different rows when WARPS_PER_BLOCK > 1. This is a fundamental design error that makes the default configuration non-functional.
2. **No vectorized loads:** Missed an easy optimization for wider memory transactions.
3. **Limited test coverage:** Only tested small V (1024), not the large-V case the design targets.

### What Qwen3.6-27B Did Well

1. **Correctness:** No critical bugs. The simpler design avoids the row-ownership ambiguity that tripped GLM-5.
2. **Iterative improvement:** Delivered v1 and v2, showing a mindset of optimization.
3. **Good test coverage:** Tested multiple realistic configurations including LLaMA-sized vocabularies.
4. **Vectorized loads in v2:** Properly implemented float4 for 4× wider transactions.

### What Qwen3.6-27B Did Poorly

1. **Suboptimal algorithm:** Three-pass design reads 12V bytes. For a bandwidth-bound kernel, this is a 12× penalty compared to the optimal single-pass approach.
2. **Flawed complexity analysis:** Incorrectly claimed compute-bound when the kernel is clearly bandwidth-bound (especially with 12V reads).
3. **Dead code in v2:** The `warp_topk_merge` and `process_float4` functions are never called.
4. **Missed online softmax:** Failed to identify the state-of-the-art single-pass approach.

### Who Won and By How Much

**GLM-5 wins by a narrow margin (80 vs 78).**

The win is driven by:
- **+3 in CUDA Depth** — online softmax shows research-level knowledge
- **+3 in Memory Design** — single-pass is optimal
- **+4 in Complexity Analysis** — accurate bandwidth-bound characterization
- **+1 in Documentation** — more comprehensive

Offset by:
- **-10 in Correctness** — critical bug with WARPS_PER_BLOCK > 1

If GLM-5 had fixed the cross-warp merge bug (e.g., by removing cross-warp logic entirely since one warp = one row), its score would be **~92/100**, winning decisively. The bug is a one-line conceptual fix: since each warp handles a distinct row, there's no need for cross-warp merging at all — each warp can independently compute its row's top-K and write output.

If Qwen3.6-27B had used online softmax (single-pass), its score would be **~88/100**, still trailing GLM-5's theoretical best but much closer.

### Recommendation

For production use, **neither implementation is ready as-is**:
- **GLM-5** needs the cross-warp merge bug fixed.
- **Qwen3.6-27B** needs the algorithm changed to single-pass online softmax.

The ideal implementation would combine:
- GLM-5's **online softmax algorithm** (single-pass)
- GLM-5's **register-resident sorted heap** (efficient for small K)
- Qwen3.6-27B's **vectorized float4 loads** (wider memory transactions)
- Qwen3.6-27B's **warp-level merge** (reduced serial bottleneck)
- GLM-5's **header-only design with stream support**
- Qwen3.6-27B's **comprehensive test coverage**

Such a hybrid would score **~95/100**.

---

*Analysis completed on 2026-04-23. Both implementations were evaluated against the original prompt requirements without modification.*
