Implement the BACKWARD pass of tiled (Flash) attention using online softmax
recomputation, from scratch in NumPy.

You must also write (or include) a minimal forward pass. The forward pass MUST
store only these intermediates per (B, H) head for the backward pass:
  - O:    (N, D)  — attention output
  - L:    (N,)    — logsumexp per query row: L_i = m_i + log(l_i)
    where m_i is the final row max and l_i is the final row sum of exps.
  - Q, K, V: the original inputs (needed for recomputation).

The forward MUST NOT store the full (N, N) attention matrix or softmax matrix.
It MUST process Q and K/V in tiles of size T using the online softmax recurrence.

BACKWARD PASS REQUIREMENTS:

1. RECOMPUTATION:
   Given dO (upstream gradient, same shape as O), Q, K, V, O, and L, compute:
     dQ: (B, H, N, D) — gradient w.r.t. queries
     dK: (B, H, N, D) — gradient w.r.t. keys
     dV: (B, H, N, D) — gradient w.r.t. values

   The backward pass must NOT materialize the full (N, N) attention matrix
   either. It recomputes softmax probabilities P on-the-fly from the stored
   L and locally recomputed S = Q_tile @ K_tile^T * scale.

2. GRADIENT FORMULAS (for a single tile interaction):
   Let scale = 1/sqrt(D). For each (Q_tile, KV_tile) pair:
   
   a) Recompute local attention scores: S = Q_tile @ K_tile^T * scale
      Shape: S is (T_q, T_kv) where T_q and T_kv are tile lengths.
   b) Recompute local softmax:
        P = exp(S - L_query[:, None])
      L_query are the logsumexp values for the query rows in this tile,
      broadcast against the key dimension. Shape: P is (T_q, T_kv).
   c) Compute local dV contribution and ACCUMULATE:
        dV_tile += P^T @ dO_tile
   d) Compute local dP:
        dP = dO_tile @ V_tile^T     Shape: (T_q, T_kv)
   e) Compute local dS via the softmax gradient:
        rowsum_PdP = (P * dP).sum(axis=-1, keepdims=True)   # shape (T_q, 1)
        dS = P * (dP - rowsum_PdP)
      This is the dsoftmax formula. The rowsum is over the KEY axis (last axis).
      The subtraction broadcasts rowsum_PdP from (T_q, 1) to (T_q, T_kv).
      The elementwise multiply by P is the FINAL step.
   f) Compute local dQ contribution and ACCUMULATE:
        dQ_tile += dS @ K_tile
   g) Compute local dK contribution and ACCUMULATE:
        dK_tile += dS^T @ Q_tile

   IMPORTANT: dQ, dK, dV contributions must be ACCUMULATED (added) across all
   KV tiles within a Q tile, not overwritten.

3. TILING:
   The backward pass uses the same tiling pattern as forward:
   - Outer loop over Q tiles (query blocks)
   - Inner loop over KV tiles (key/value blocks)
   - For causal attention, skip (Q_tile, KV_tile) pairs that are entirely
     above the diagonal (all key positions > all query positions)
   - Within each Q tile, initialize dQ_tile, dK_tile, dV_tile accumulators
     and accumulate contributions from each KV tile

4. BATCHING:
   Handle (B, H, N, D) tensors. You may loop over (b, h) or use batched
   operations — either is acceptable.

5. CAUSAL MASKING IN BACKWARD:
   When causal=True, the backward pass must apply the same masking pattern
   as the forward pass. For each (Q_tile, KV_tile) pair:
   - If the entire block is above the diagonal, SKIP it (no contribution
     to any gradient)
   - If partially masked, apply the causal mask to S before computing P:
       S = S + causal_mask  (masked positions = -inf)
     Then exp(S - L) gives 0 for masked positions, which correctly
     zeros out their contribution to dV, dS, dQ, and dK.

6. NUMERICAL STABILITY:
   - L already incorporates the row max from forward, so exp(S - L[:, None])
     has arguments ≤ 0, which is stable (no overflow).
   - The dsoftmax formula computes (dP - rowsum(P*dP)). Both dP and rowsum
     can be large, but the subtraction is benign because the result is
     multiplied by P (which sums to 1 per row), keeping dS bounded.
   - Use float64 for intermediate reductions if possible.

Deliver:
- Function flash_attention_fwd(Q, K, V, tile_size, causal=True)
  → returns (O, cache) where cache = {'O': O, 'L': L, 'Q': Q, 'K': K, 'V': V}
  and L has shape (B, H, N)
- Function flash_attention_bwd(dO, cache, tile_size, causal=True)
  → returns (dQ, dK, dV) each of shape (B, H, N, D)
- Test 1 (gradient check): (B=1, H=1, N=64, D=32, T=16, causal=True)
  → Compare dV against central finite differences across ALL elements
  → Spot-check dQ and dK at 10 random positions
  → Assert relative error < 1e-5 for all
- Test 2 (vs naive backward): (B=2, H=4, N=256, D=64, T=64, causal=True)
  → Compare dQ, dK, dV against naive full-materialized backward
  → Assert max relative error < 1e-4
- Test 3 (memory): (B=1, H=1, N=4096, D=64, T=128, causal=True)
  → Use tracemalloc to verify peak memory is less than 20% of the
    memory required for a single (N, N) matrix

Use only NumPy. No PyTorch, JAX, TensorFlow, or autograd.
