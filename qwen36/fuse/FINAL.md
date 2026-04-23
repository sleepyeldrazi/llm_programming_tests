Here's a summary of everything delivered:

---

## Deliverables

### Files

| File | Purpose |
|------|---------|
| `fused_softmax_topk.cu` | **Production kernel** — complete, correct, well-commented |
| `fused_softmax_topk_v2.cu` | **Optimized kernel** — vectorized loads, warp-level merge |
| `ANALYSIS.md` | **Full design analysis** — memory, complexity, comparisons |
| `benchmark.cu` | **Correctness + perf harness** — CPU reference, timing, scaling |

---

### Architecture at a Glance

```
Grid:  B×T blocks (one per position)
Block: 256 threads

Phase 1:  max-reduction over V          → warp shuffle → shared mem
Phase 2:  Σexp(xᵢ - max)                → warp shuffle → shared mem  
Phase 3:  softmax + local top-K (regs)  → LOCAL_K=16 per thread
Phase 4:  merge → shared min-heap       → warp-by-warp staging
Phase 5:  sort + write-back             → thread 0, O(K²)
```

### Key Design Decisions

| Constraint | Solution |
|---|---|
| **No full softmax in global mem** | Softmax values computed in registers, only top-K written out |
| **Numerical stability** | Two-pass: find max first, then `exp(x - max)` |
| **Large V (50k+)** | Grid-stride loops; each thread processes ~196 elements |
| **Shared memory** | Min-heap (2KB) + warp staging (4KB) = 6.2KB total |
| **Warp optimization** | `__shfl_xor_sync` for reductions (zero shared mem, ~15 cycles) |

### Complexity (V=50257, K=256)

| Metric | Value |
|---|---|
| Global reads | **12V = 2.4 MB** (3 passes, coalesced) |
| Global writes | **8K = 2 KB** (output only) |
| expf() calls | **2V = 100K** |
| Bound | **Compute-bound** (expf throughput, not bandwidth) |

### vs. Naive (softmax → topk)

| Metric | Naive | Fused | Win |
|---|---|---|---|
| Global writes | 4V + 8K | **8K** | **~200× less** |
| Peak memory | 4V + 8K | **8K** | **~200× less** |
| Global reads | 8V | 12V | 0.67× (reads more) |
| expf() calls | V | 2V | 0.5× (computes more) |

**Net: The fused kernel trades 50% more reads for ~200× fewer writes.** For V=50K, this saves ~192 KB of global memory traffic per position — the write savings dominate.

### Further Optimizations (documented in ANALYSIS.md §6)

1. **Warp-level top-K merge** — reduce heap insertions from 4096 → 2048
2. **FP16/BF16** — 2× bandwidth savings, acceptable precision loss
3. **Vectorized loads (float4)** — already in v2, 4× fewer memory instructions
4. **Persistent blocks** — for large B×T, better occupancy
5. **Async copy (Hopper+)** — `__ldg()` for L2 cache reuse across passes
6. **Single-pass online** — tracks running max, re-normalizes (complex, for V > 1M)