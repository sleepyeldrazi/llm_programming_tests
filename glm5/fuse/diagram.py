#!/usr/bin/env python3
"""
Generate an ASCII architecture diagram for the fused softmax + top-k kernel.
Run: python3 diagram.py
"""

print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    FUSED SOFTMAX + TOP-K KERNEL ARCHITECTURE                ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Global Memory Layout                                                        ║
║  ┌──────────────────────────────────────────────┐                            ║
║  │  logits [B, T, V]    (read-only)             │                            ║
║  │  indices [B, T, K]   (write-only)            │                            ║
║  │  probs [B, T, K]     (write-only)            │                            ║
║  └──────────────────────────────────────────────┘                            ║
║                                                                              ║
║  Thread Mapping: 1 warp = 1 row (b, t)                                       ║
║  ┌────────────────────────────────────────────────────────────────────────┐  ║
║  │  Block (256 threads = 8 warps)                                         │  ║
║  │  ┌──────────┐ ┌──────────┐       ┌──────────┐                         │  ║
║  │  │ Warp 0   │ │ Warp 1   │  ...  │ Warp 7   │                         │  ║
║  │  │ row=0    │ │ row=1    │       │ row=7    │                         │  ║
║  │  │ 32 lanes │ │ 32 lanes │       │ 32 lanes │                         │  ║
║  │  └────┬─────┘ └────┬─────┘       └────┬─────┘                         │  ║
║  │       │             │                  │                                │  ║
║  │  ┌────▼─────────────▼──────────────────▼──────────────────────────┐    │  ║
║  │  │  Shared Memory (~2 KB)                                         │    │  ║
║  │  │  • warp_max[8], warp_sum[8]   (32+32 bytes)                   │    │  ║
║  │  │  • heap_buf[8][K], idx_buf[8][K]  (2×8×K × 4 bytes)          │    │  ║
║  │  └───────────────────────────────────────────────────────────────┘    │  ║
║  └────────────────────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║  Single Warp Detail (processing row r, V=50257):                             ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │ Lane 0    Lane 1    Lane 2    ...    Lane 31                         │   ║
║  │                                                                          │   ║
║  │ READ:   logits[r*V + {0,1,2,...,31}]     ← 1 coalesced 128B load     │   ║
║  │         logits[r*V + {32,33,...,63}]      ← next coalesced load       │   ║
║  │         ...                                                             │   ║
║  │         logits[r*V + {50224,...,50255}]   ← last load                 │   ║
║  │                                                                          │   ║
║  │ Each lane processes ~V/32 ≈ 1571 elements:                             │   ║
║  │                                                                          │   ║
║  │ ┌─────────────────────────────────────────────────────────┐            │   ║
║  │ │ Per-Lane Computation (in REGISTERS):                    │            │   ║
║  │ │                                                         │            │   ║
║  │ │  local_max = -∞, local_sum = 0                          │            │   ║
║  │ │  heap = {(-∞, 0), ..., (-∞, 0)}  // K entries          │            │   ║
║  │ │                                                         │            │   ║
║  │ │  for each element x_j at index j:                       │            │   ║
║  │ │    old_max = local_max                                  │            │   ║
║  │ │    local_max = max(local_max, x_j)                      │            │   ║
║  │ │    local_sum *= exp(old_max - local_max)  // rescale   │            │   ║
║  │ │    local_sum += exp(x_j - local_max)       // add new  │            │   ║
║  │ │    heap.insert(x_j, j)    // O(K) compare+shift        │            │   ║
║  │ └─────────────────────────────────────────────────────────┘            │   ║
║  │                          │                                              │   ║
║  │                          ▼  Warp Shuffle Reduction                      │   ║
║  │                                                                          │   ║
║  │ ┌─────────────────────────────────────────────────────────┐            │   ║
║  │ │  warp_max = reduce_max(local_max) across 32 lanes       │            │   ║
║  │ │  warp_sum = reduce_sum(local_sum * exp(local_max -      │            │   ║
║  │ │                        warp_max)) across 32 lanes       │            │   ║
║  │ │                                                         │            │   ║
║  │ │  5 butterfly steps using __shfl_xor_sync:               │            │   ║
║  │ │  Step 1: ⊕ 16  ── 16↔16 pairs merge                    │            │   ║
║  │ │  Step 2: ⊕ 8   ── 8 groups of 4 merge                  │            │   ║
║  │ │  Step 3: ⊕ 4   ── 4 groups of 8 merge                  │            │   ║
║  │ │  Step 4: ⊕ 2   ── 2 groups of 16 merge                 │            │   ║
║  │ │  Step 5: ⊕ 1   ── final 32-lane consensus               │            │   ║
║  │ └─────────────────────────────────────────────────────────┘            │   ║
║  │                          │                                              │   ║
║  │                          ▼  Cross-Warp Merge (Phase 2)                  │   ║
║  │                                                                          │   ║
║  │ ┌─────────────────────────────────────────────────────────┐            │   ║
║  │ │  1. Each warp writes its K heap entries → shared memory │            │   ║
║  │ │  2. __syncthreads()                                     │            │   ║
║  │ │  3. Warp 0 merges 8 heaps → global top-K:              │            │   ║
║  │ │     • Scan 8×K=80 candidates                           │            │   ║
║  │ │     • Keep top K=10 via sorted insertion                │            │   ║
║  │ │  4. Rescale to probabilities:                           │            │   ║
║  │ │     prob_i = exp(val_i - global_max) / global_sum      │            │   ║
║  │ │  5. Write K × (prob, index) to global memory           │            │   ║
║  │ └─────────────────────────────────────────────────────────┘            │   ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  MEMORY TRAFFIC SUMMARY (per row, V=50257, K=10)                             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌───────────────────┬──────────────┬──────────────┬─────────────┐          ║
║  │ Implementation    │ Reads        │ Writes       │ Total I/O   │          ║
║  ├───────────────────┼──────────────┼──────────────┼─────────────┤          ║
║  │ Naive (separate)  │ 2V = 402 KB  │ V+2K = 201KB │ 603 KB      │          ║
║  │ Fused (this work) │ V = 201 KB   │ 2K = 80 B    │ 201 KB      │          ║
║  │ Speedup           │ 2×           │ 2500×        │ 3×          │          ║
║  └───────────────────┴──────────────┴──────────────┴─────────────┘          ║
║                                                                              ║
║  Full workload (B=64, T=128, V=50257, K=10):                                ║
║  ┌───────────────────┬─────────────┬──────────┬────────────────────┐        ║
║  │ Implementation    │ Total I/O   │ Time*    │ Intermediate Tensor│        ║
║  ├───────────────────┼─────────────┼──────────┼────────────────────┤        ║
║  │ Naive             │ 4.7 GB      │ ~2.3 ms  │ 201 MB (full sm)   │        ║
║  │ Fused             │ 1.6 GB      │ ~0.8 ms  │ 0 MB (registers)   │        ║
║  └───────────────────┴─────────────┴──────────┴────────────────────┘        ║
║  * Estimated on NVIDIA A100 at peak bandwidth (2039 GB/s)                    ║
║                                                                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")
