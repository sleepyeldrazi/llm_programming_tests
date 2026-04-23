# Fused Softmax + Top-K: Design Document

## 1. Problem Statement

Given logits `[B, T, V]` (e.g., batch=64, seq_len=128, vocab=50257), produce:
- **indices** `[B, T, K]` — the K highest-probability token indices per row
- **probs** `[B, T, K]` — their softmax probabilities

**Constraint:** Never write the full V-length softmax vector to global memory.

---

## 2. Algorithm: Online Softmax + Register Min-Heap

### 2.1 Core Idea

We fuse three operations — **softmax computation**, **top-K selection**, and **probability rescaling** — into a single pass over the logits. This is an instance of the *online softmax* algorithm (Milakov & Gimelshein, 2018) extended with a streaming top-K heap.

### 2.2 Online Softmax Recurrence

Standard softmax requires two passes: one for the max, one for the sum-of-exps. The online variant maintains running statistics:

```
m_j = max(x_0, ..., x_j)        // running maximum
d_j = Σ_{i≤j} exp(x_i - m_j)    // running sum, always relative to current max
```

Update rule for each new element `x_j`:
```
m_{j}   = max(m_{j-1}, x_j)
d_{j}   = d_{j-1} * exp(m_{j-1} - m_{j}) + exp(x_j - m_{j})
```

This is **numerically stable** — all exponentials use `x - m_j` where `m_j` is the running max, so no term exceeds `exp(0) = 1`.

### 2.3 Streaming Top-K Heap

Simultaneously, each thread maintains a sorted array of size K in registers:

```
insert(value, index):
    if value <= heap[0]:       // heap[0] = K-th largest seen so far
        return                 // reject — not in top-K
    find position via linear scan (K ≤ 32, so ~5 compares average)
    shift lower elements down
    place new element
```

For K ≤ 32 this register-resident sorted array outperforms a binary heap because:
- No indirection / pointer chasing
- The GPU's branch predictor handles the predictable comparison pattern well
- Register access is ~0 latency vs. shared memory's ~20 cycle latency

---

## 3. Kernel Architecture

### 3.1 Mapping: One Warp per Row

```
Grid:   ceil(B*T / WARPS_PER_BLOCK)  blocks
Block:  WARPS_PER_BLOCK × WARP_SIZE   threads (default: 8 × 32 = 256)
Warp:   one (b,t) row
```

Each warp cooperatively processes one row of length V. Lane `j` (0..31) processes elements at indices `j, j+32, j+64, ...`.

### 3.2 Three-Phase Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Local Pass (per-warp, parallel across lanes)          │
│                                                                 │
│   Each lane reads V/32 logits in a coalesced strided pattern    │
│   Each lane maintains:                                          │
│     • local_max, local_sum  (online softmax statistics)         │
│     • TopKHeap<K>           (K best logits seen by this lane)   │
│                                                                 │
│   Warp reduce → warp_max, warp_sum                              │
├─────────────────────────────────────────────────────────────────┤
│ Phase 2: Cross-Warp Merge (shared memory)                       │
│                                                                 │
│   Only needed when WARPS_PER_BLOCK > 1 (i.e., multiple warps   │
│   process different rows — they still need to sync for shared   │
│   memory reuse). Within a single warp, Phase 2 is trivial.     │
│                                                                 │
│   • Warp 0 reduces global max/sum from all warps               │
│   • Each warp writes its local top-K heap to shared memory      │
│   • Warp 0 merges WARPS_PER_BLOCK heaps → global top-K         │
│   • Rescale: prob_i = exp(val_i - global_max) / global_sum     │
├─────────────────────────────────────────────────────────────────┤
│ Phase 3: Write Output                                           │
│                                                                 │
│   Lane 0 of warp 0 writes K (prob, index) pairs to global mem  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Data Flow Diagram

```
Global Memory (logits [B,T,V])
    │
    ▼  coalesced reads, V/32 per lane
┌───────────────┐
│   Registers   │  Lane 0  Lane 1  ...  Lane 31
│               │  [heap]  [heap]      [heap]
│               │  [lmax]  [lmax]      [lmax]
│               │  [lsum]  [lsum]      [lsum]
└──────┬────────┘
       │ warp shuffle (reduce_max, reduce_sum)
       ▼
┌───────────────┐
│   Warp-level  │  warp_max, warp_sum (broadcast)
│   consensus   │  merged heap via shared memory
└──────┬────────┘
       │
       ▼
Global Memory (probs [B,T,K], indices [B,T,K])
```

---

## 4. Memory Access Pattern

### 4.1 Global Memory Reads (Logits)

**Pattern: Strided coalesced access**

```
Warp for row r reads logits[r*V + 0], logits[r*V + 1], ..., logits[r*V + V-1]

Lane 0: reads indices 0, 32, 64, 96, ...
Lane 1: reads indices 1, 33, 65, 97, ...
...
Lane 31: reads indices 31, 63, 95, 127, ...
```

Consecutive lanes read consecutive addresses → **perfectly coalesced** 128-byte transactions. Each 128-byte cache line is fully utilized by 32 `float` values.

**Memory efficiency:** V reads per row, 100% coalesced. No redundant loads.

### 4.2 Global Memory Writes (Output)

Each row writes exactly `2K` values (K probabilities + K indices). For K=10, that's 80 bytes — negligible compared to reading V×4 bytes (200KB for V=50k).

**Writes are coalesced within a warp** because consecutive warps write consecutive rows, and lane 0 handles the output for its row.

### 4.3 Shared Memory

Used for cross-warp heap merge. Total footprint per block:

```
float warp_max[8]           =   32 bytes
float warp_sum[8]           =   32 bytes
float heap_buf[8][32]       = 1024 bytes
int   idx_buf[8][32]        = 1024 bytes
                                ─────────
                          Total ≈ 2 KB
```

Well within the 48KB shared memory limit. **No bank conflicts** because each warp writes to a different row of `heap_buf[warp_id][...]`, and during the merge phase only warp 0 reads (sequentially, from its own perspective).

### 4.4 Register Usage

Per thread:
- Online softmax state: 2 floats (8 bytes)
- TopKHeap<K=10>: 10 floats + 10 ints (80 bytes)
- Loop variables: ~4 floats (16 bytes)
- **Total: ~104 bytes/thread**

For a 256-thread block: ~26 KB of register file usage. Comfortably fits modern GPU register files (64KB–256KB per SM).

---

## 5. Warp-Level Optimization Strategy

### 5.1 Shuffle-Based Reductions

The max and sum reductions use `__shfl_xor_sync` (butterfly pattern):

```
Step 1: exchange with lane ^ 16  →  16 pairs
Step 2: exchange with lane ^ 8   →  8 quads
Step 3: exchange with lane ^ 4   →  4 groups of 8
Step 4: exchange with lane ^ 2   →  2 groups of 16
Step 5: exchange with lane ^ 1   →  1 group of 32
```

5 steps × 2 ops (max + sum) = **10 shuffle instructions total**. No shared memory, no synchronization needed within a warp.

### 5.2 Why Not One Warp Per Row with Vector Loads?

Alternative: use a wider type (`float4`) to read 4 values per lane, reducing the loop iterations by 4×. This is beneficial when V is very large:

```
// Vectorized load variant (Phase 1 inner loop)
float4 vec = reinterpret_cast<const float4*>(logits_row)[v];
float x0 = vec.x, x1 = vec.y, x2 = vec.z, x3 = vec.w;
// Process 4 elements per iteration
```

**Trade-off:** Increases register pressure (4× more values live at once) but reduces loop overhead and improves memory throughput via wider transactions. Recommended when V > 10K.

### 5.3 Occupancy Considerations

| Parameter          | Value   |
|--------------------|---------|
| Threads/block      | 256     |
| Registers/thread   | ~26     |
| Shared memory/block| ~2 KB   |
| Blocks/SM (A100)   | 16–20   |
| Rows in flight/SM  | 128–160 |

The kernel is **not register-heavy** and uses minimal shared memory, allowing high occupancy and effective latency hiding.

---

## 6. Complexity Analysis

### 6.1 Per-Row Work

| Operation                | Reads    | Writes  | Compute           |
|--------------------------|----------|---------|-------------------|
| Read logits              | V        | 0       | 0                 |
| Online max/sum           | 0*       | 0       | V × (1 max + 1 exp + 2 FMAs) |
| Top-K heap insert        | 0*       | 0       | V × ~5 compares + ~2.5 moves avg |
| Warp reduce              | 0        | 0       | 10 shuffles       |
| Final rescale (K values) | 0*       | 2K      | K × (1 exp + 1 mul) |
| **Total**                | **V**    | **2K**  | **~6V + 10 + 2K FLOPs** |

*All intermediate values are in registers.

### 6.2 Bandwidth vs Compute Bound Analysis

For V = 50,257 and K = 10:

**Memory traffic per row:**
```
Reads:  V × 4 bytes = 201 KB
Writes: K × 8 bytes = 80 bytes
Total:  ~201 KB
```

**Compute per row:**
```
~6 × 50,257 = 301,542 FLOPs (approximate)
```

**Arithmetic intensity:**
```
AI = 301,542 FLOPs / 201,028 bytes ≈ 1.5 FLOP/byte
```

**NVIDIA A100 specs:**
```
Peak bandwidth:  2039 GB/s  →  compute/bw ratio = 19.5 TFLOPS / 2039 GB/s ≈ 9.6 FLOP/byte
Peak FP32:       19.5 TFLOPS
```

**Conclusion: AI (1.5) << ratio (9.6) → kernel is BANDWIDTH BOUND.**

This means:
1. **The bottleneck is reading V logits from global memory**, not compute.
2. Optimizations should focus on memory access patterns (coalescing, caching) not arithmetic.
3. The fusion saves one full write+read of the `[B,T,V]` tensor (~201 KB/row), directly translating to ~2× end-to-end speedup vs. separate softmax + top-K.

### 6.3 Comparison to Naive Implementation

```
Naive (separate kernels):
  Kernel 1: softmax
    Read  V logits  →  Write V probabilities     (201 KB + 201 KB = 402 KB I/O)
  Kernel 2: top-k
    Read  V probabilities  →  Write K results     (201 KB + 80 bytes = 201 KB I/O)
  Total I/O: ~603 KB/row
  Kernel launch overhead: 2× 

Fused (this kernel):
  Read V logits  →  Write K results                (201 KB + 80 bytes = 201 KB I/O)
  Total I/O: ~201 KB/row
  Kernel launch overhead: 1×

Savings:
  Memory I/O:  3× reduction (603 KB → 201 KB per row)
  Kernel launches: 2× reduction
  Effective speedup: ~2.5–3× (bandwidth-bound, so I/O directly maps to time)
```

For a real workload (B=64, T=128, V=50257):
```
Naive:    64 × 128 × 603 KB = 4.7 GB global memory traffic
Fused:    64 × 128 × 201 KB = 1.6 GB global memory traffic
Savings:  3.1 GB avoided

At A100 bandwidth (2039 GB/s):
  Naive time:  ~2.3 ms
  Fused time:  ~0.8 ms
  Speedup:     2.9×
```

---

## 7. Advanced Optimizations

### 7.1 FP16 Input with FP32 Accumulation

For mixed-precision workloads (logits stored as `__half`):

```cuda
// Read 2 values per load, accumulate in FP32
__half2 h2 = reinterpret_cast<const __half2*>(logits_row)[v];
float x0 = __half2float(h2.x);
float x1 = __half2float(h2.y);
```

This halves memory traffic (V × 2 bytes instead of V × 4 bytes), doubling throughput for bandwidth-bound workloads.

### 7.2 Multi-Row Per Warp (for Small V)

When V < 1024, each warp has spare bandwidth. Assign multiple rows per warp:

```
for (int row_offset = 0; row_offset < ROWS_PER_WARP; row_offset++) {
    int row = base_row + row_offset;
    // ... process row ...
}
```

This amortizes warp-management overhead and improves occupancy for small-V cases.

### 7.3 Async Copy (Hopper/Ada Lovelace)

```cuda
// Pipeline loads with cp.async to overlap compute and memory
cp.async.ca.shared.global [smem_ptr], [gmem_ptr], 16;
```

Overlaps the next chunk's load with the current chunk's heap insertions. Beneficial when V > 10K and the compute path has enough latency to hide.

### 7.4 Warp-Level Heap Merge for Large WARPS_PER_BLOCK

When using many warps per block, the serial merge by warp 0 becomes a bottleneck. Alternative:

```
1. Each warp writes its K values to shared memory
2. Tournament merge using warp shuffles:
   - Round 1: warp 0 vs warp 1, warp 2 vs warp 3, ...
   - Round 2: winners merge
   - Final: one warp produces global top-K
3. Each round uses warp-cooperative merge of two sorted arrays
```

This reduces merge complexity from O(WARPS × K) to O(K × log(WARPS)).

---

## 8. Correctness: Numerical Stability

The algorithm maintains numerical stability through three mechanisms:

1. **Subtract running max before exp:** All calls to `expf()` use `x - current_max`, ensuring the argument is ≤ 0. No overflow possible.

2. **Rescaling on max update:** When `current_max` increases, we multiply the running sum by `exp(old_max - new_max)`, which is in (0, 1]. No overflow; minimal underflow risk.

3. **Final rescaling:** `prob_i = exp(val_i - global_max) / global_sum`. Since `global_sum ≥ 1.0` (it includes `exp(global_max - global_max) = 1.0`), division is safe.

**Comparison with log-sum-exp:**
The online algorithm computes exactly `Σ exp(x_i - max(x))` which is equivalent to `exp(logsumexp(x) - max(x))`. The final probabilities are identical to standard numerically-stable softmax to within floating-point rounding.

---

## 9. Summary Table

| Metric                    | Naive (separate)  | Fused (this work)  | Improvement    |
|---------------------------|--------------------|---------------------|----------------|
| Global memory reads       | 2V per row         | V per row           | 2×             |
| Global memory writes      | V + 2K per row     | 2K per row          | ~V/(2K)×       |
| Total I/O per row         | ~3V                | ~V                  | 3×             |
| Kernel launches           | 2                  | 1                   | 2×             |
| Intermediate tensor       | V floats/row       | 0 (registers)       | ∞              |
| Numerically stable        | Yes                | Yes                 | —              |
| Register pressure         | Low                | Moderate (~26 regs) | Acceptable     |
| Shared memory             | None               | ~2 KB               | Minimal        |
| Bandwidth utilization     | Wastes BW on intermediate | Optimal       | —              |
| Effective speedup         | Baseline           | **2.5–3×**          | —              |
