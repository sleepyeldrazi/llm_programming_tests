# Fused Softmax + Top-K — Design Notes

Implementation: `fused_softmax_topk.cu`.

## 1. Algorithmic Idea

A standard softmax + top-k requires **three passes** over `V`:

1. `m = max(x)`                    — for numerical stability
2. `s = Σ exp(x_i − m)`             — denominator
3. `p_i = exp(x_i − m) / s` then top-k on `p`

Two reductions can be collapsed to **one** with the **online-softmax** recurrence
(Milakov & Gimelshein, 2018). For each new element `x`:

```
m_new = max(m, x)
s_new = s · exp(m − m_new) + exp(x − m_new)
```

The pair `(m, s)` is associative under the `combine` operator above, so it
reduces in a tree across threads/warps just like a sum.

A second observation: **softmax is monotonic**, so the top-k indices on
`logits` equal the top-k indices on probabilities. We therefore track top-k
on raw logits during the same streaming pass, and only at the end normalize
the K winning logits with the global `(m, s)`:

```
p_j = exp(logit_j − m) / s        for j in top-k
```

So the *full* softmax matrix is never written — only `K` probabilities per row
ever leave the SM.

## 2. Kernel Layout

- One CUDA block ↔ one `(b, t)` row of length `V`.
- Grid: `B*T` blocks. Block: 256 threads (8 warps).
- Each thread strides through `V` with stride `BLOCK`, maintaining:
  - `(m, s)` online-softmax state in two registers,
  - a register-resident sorted top-K buffer (`val[K]`, `idx[K]`).

After streaming, partials are merged via:

1. **Warp reduction** — `__shfl_xor_sync` butterfly
   - `(m, s)` reduced with `ms_combine`.
   - Top-K reduced by exchanging full K-arrays via shfl and doing a `2K → K`
     linear merge in registers.
2. **Cross-warp** — warp leaders dump their partials to shared memory
   (`8 × (MS + 2·K floats/ints)` ≈ a few hundred bytes).
3. Warp 0 loads them, does the same shuffle reduction once more, and lane 0
   writes the K outputs.

## 3. Memory Access Pattern

- `logits` is read **exactly once**, with fully coalesced 128-byte
  transactions (warp of 32 threads × 4 bytes contiguous per step).
- No intermediate writes to global memory. The full softmax is never
  materialized — constraint (1) satisfied.
- Outputs: `2·K` values per row (typ. K ≤ 32 → ≤ 256 B/row), negligible.
- Shared memory footprint: `WARPS·(8 + 8K)` bytes ≈ 1 KB for K=16, well
  inside L1/SMEM, so occupancy is bounded by registers, not SMEM.

## 4. Warp-Level Optimization

| Reduction          | Mechanism                                         |
|--------------------|---------------------------------------------------|
| `(m, s)` across 32 | 5-stage `__shfl_xor_sync` butterfly, no SMEM      |
| Top-K across 32    | 5-stage shfl + register-resident merge of 2K→K    |
| Cross-warp         | One shared-mem hand-off, then a final warp shuffle|
| Sync barriers      | A single `__syncthreads()` for the SMEM hand-off  |

Per-thread top-K update is a tight insertion sort. Once the buffer is full
(after the first `K/BLOCK` iterations), the common path is one compare against
`val[K-1]` and a fall-through, which is essentially free relative to the
`__expf` next to it. The sort itself is unrolled (`#pragma unroll`) so the K
comparisons live in registers with no branches on K.

`__expf` is the fast intrinsic; for the final normalization we use
`1.0f / s` and a single multiply per output to avoid K divisions.

## 5. Complexity

Let `N = B·T`.

- **Compute**: `O(N·V)` — one fused pass; `2 fmax + 2 expf + 2 fma` per
  element plus an amortized O(1) top-K compare. Reductions add `O(N·log W)`
  where `W = BLOCK` (negligible).
- **Global memory**: read `N·V` floats, write `2·N·K` words. With `K ≪ V`
  this is `≈ N·V` bytes·4 — the absolute lower bound for any algorithm that
  must look at every logit.

### Bandwidth vs compute

- A100 HBM2e: ~1.5 TB/s. RTX 4090: ~1 TB/s.
- A100 fp32: ~19.5 TFLOP/s. The kernel does ~6–8 flops/element (incl. the
  cost amortized per `expf`). Arithmetic intensity ≈ 8 flops / 4 bytes =
  **2 flops/byte**, which sits well under the machine balance (~13
  flops/byte on A100). **The kernel is memory-bandwidth bound** — exactly
  where we want to be: we are paying only the cost of one HBM read of the
  logits, which is unavoidable.

For typical LLM logits (`V=50257`, `B·T=4096`): one row = 196 KB, total
≈ 800 MB. On A100 that's **~0.5 ms of HBM read time**, which is the floor
this kernel approaches.

## 6. Comparison to Naïve Implementation

| Aspect                         | Naïve (3-pass)                | Fused kernel                         |
|--------------------------------|-------------------------------|--------------------------------------|
| Global reads of `logits`       | 3·N·V floats                  | **1·N·V floats**                     |
| Global writes of softmax       | N·V floats                    | **0** (never materialized)           |
| Global memory traffic          | ≈ 4·N·V·4 bytes               | **≈ N·V·4 bytes** — 4× less         |
| Kernel launches                | 3 (max, sum/normalize, topk)  | **1**                                |
| Numerical stability            | Yes (with explicit max)        | Yes (online-softmax recurrence)      |
| Top-K stage cost               | extra N·V read + sort/select  | **folded into pass 1**               |
| Peak SMEM / row                | ~`V` floats (if cached)        | ~1 KB regardless of `V`              |
| Scales to `V = 50k+`?          | Yes but slow                   | **Yes; no per-V SMEM growth**        |

End-to-end speedup vs the 3-kernel naïve path is ≈ **3–4× on bandwidth-
bound workloads** (typical LLM sampling), bounded above by the read-once
HBM floor.

## 7. Knobs / Variants

- **K range**: register top-K is ideal for `K ≤ 32`. For larger K
  (e.g. K=128) switch the per-thread buffer to shared memory and use a
  bitonic top-K, or use a two-stage radix-select.
- **fp16 / bf16 logits**: cast to fp32 in the inner loop (`__half2float`);
  the reduction state must stay fp32 for numerical stability.
- **Small V (V < 1024)**: drop to one warp per row to raise occupancy.
- **Very large V (≥ 256k)**: split the row across multiple blocks, write
  per-block partial `(m, s, topK)` to global, then a tiny merge kernel.
  Still <2× the global traffic of the single-pass version.
- **Sampling integration**: extend the final stage with a Gumbel-max draw
  from the K probabilities to fuse softmax+topk+sample into one launch.
