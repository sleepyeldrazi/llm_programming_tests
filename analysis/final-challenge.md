# Final Challenge: Flash Attention Backward Pass (Tiled, Recompute)

## Why this challenge

The forward pass of Flash Attention has been implemented correctly by all models tested
so far. The backward pass is the real test — 5-10x harder, with subtle interactions
between tiling, recomputation, and the softmax gradient. PyTorch's own autograd gets
this wrong without careful `torch.compile` handling. Three of the five major open-source
Flash Attention ports (xformers early, vLLM's first kernel, and llama.cpp's first
attempt) shipped with gradient bugs that passed forward correctness but failed backward.

This challenge:
- Runs on your M4 MacBook Pro (~200-400 MB, not GB)
- Takes ~5-10 seconds for the gradient check
- Catches incorrect implementations that "look right" in the forward
- Is directly relevant to LLM training (every training framework uses Flash Attention)
- Tests the exact capability gap between your local model and frontier models

The key trap: the `dsoftmax` formula is `dS = P * (dP - rowsum(P * dP))`. The rowsum
is over the KEY dimension, and P must be the recomputed softmax from the stored
logsumexp. Getting ANY of these details wrong produces gradients that look plausible
but fail finite-difference verification.

## The prompt

```
Implement the BACKWARD pass of tiled (Flash) attention using online softmax
recomputation, from scratch in NumPy.

You already have a forward pass (include it or write a minimal one). The forward
pass MUST store only these intermediates per (B, H) head:
  - O:    (N, D)  — attention output
  - L:    (N,)    — logsumexp per query row: L_i = m_i + log(l_i)
    where m_i is the final row max and l_i is the final row sum of exps
  - Q, K, V: the original inputs (required for recomputation)

The forward MUST NOT store the full (N, N) attention matrix or softmax matrix.
It MAY process Q and K/V in tiles of size T and use the online softmax recurrence.

BACKWARD PASS REQUIREMENTS:

1. RECOMPUTATION:
   Given dO (upstream gradient, same shape as O), Q, K, V, O, and L, compute:
     dQ: (N, D) — gradient w.r.t. queries
     dK: (N, D) — gradient w.r.t. keys
     dV: (N, D) — gradient w.r.t. values
   
   The backward pass must NOT materialize the full (N, N) attention or
   softmax matrix either. It recomputes softmax probabilities P on-the-fly
   from the stored L and locally recomputed S = Q @ K^T / sqrt(D).

2. GRADIENT FORMULAS (for a single N×D head, no batching yet):
   Let scale = 1/sqrt(D). For each tile interaction between Q_tile and K_tile:
   
   a) Recompute local attention scores: S = Q_tile @ K_tile^T * scale
   b) Recompute local softmax: P = exp(S - L_query[:, None])
      (L_query are the logsumexp values for the query rows in this tile,
       broadcast against the key dimension)
   c) Compute local dV contribution: dV += P^T @ dO_tile
   d) Compute local dP: dP = dO_tile @ V_tile^T
   e) Compute local dS via the softmax gradient:
        dS = P * (dP - rowsum(P * dP))   where rowsum is over the KEY axis
      IMPORTANT: P * dP is elementwise. rowsum sums over the last axis (keys).
      The subtraction broadcasts: rowsum(P*dP) has shape (T_q, 1), subtracted
      from dP which is (T_q, T_kv), then multiplied elementwise by P.
   f) Compute local dQ contribution: dQ += dS @ K_tile
   g) Compute local dK contribution: dK += dS^T @ Q_tile

3. TILING:
   The backward pass should also use tiling to avoid materializing full matrices.
   Process Q in tiles, and for each Q tile, iterate over KV tiles to recompute
   P, dP, dS and accumulate dQ, dK, dV. This mirrors the forward pass structure.

4. BATCHING:
   Extend the above to handle (B, H, N, D) tensors. The L tensor becomes
   (B, H, N). The tile loops can be per-(b,h) or batched — either is acceptable.

5. NUMERICAL STABILITY:
   - The stored L values already incorporate the row max, so P = exp(S - L)
     is numerically stable (arguments ≤ 0).
   - The dsoftmax formula involves computing (dP - rowsum(P * dP)). If dP has
     large values, the subtraction can cause cancellation, but this is inherent
     to softmax and handled by the upcast to float64 for the rowsum operation.
   - Ensure no division by zero or log of negative numbers.

6. CORRECTNESS VERIFICATION:
   Compare your backward pass output against numerical gradients (central
   finite differences) for a small test case (N=64, D=32, tile_size=16).
   Also compare against the naive full-materialized backward (which computes
   the full attention matrix).

Deliver:
- Function flash_attention_fwd(Q, K, V, tile_size, causal=True)
  → returns O (B,H,N,D) and cache dict with {'O': O, 'L': L, 'Q': Q, 'K': K, 'V': V}
- Function flash_attention_bwd(dO, cache, tile_size, causal=True)
  → returns dQ, dK, dV, each (B,H,N,D)
- Gradient check test: (B=1, H=1, N=64, D=32, T=16, causal=True)
  → compare bwd output vs central finite differences, assert relative error < 1e-5
- Correctness test: (B=2, H=4, N=256, D=64, T=64, causal=True)
  → compare bwd output vs naive full-materialized backward, assert rel error < 1e-4
- Memory test: (B=1, H=1, N=4096, D=64, T=128, causal=True)
  → verify peak memory is well below N² (use tracemalloc)

Use only NumPy. No PyTorch, JAX, TensorFlow, or autograd.
```

## How the trap works

The dsoftmax formula in Step 2e is where 80% of implementations fail:

```python
# CORRECT (what you should write):
dS = P * (dP - (P * dP).sum(axis=-1, keepdims=True))

# WRONG (very common — wrong axis):
dS = P * (dP - (P * dP).sum(axis=-2, keepdims=True))

# WRONG (forgets to multiply by P):
dS = dP - (P * dP).sum(axis=-1, keepdims=True)

# WRONG (divides instead of subtracts):
dS = P * dP / (P * dP).sum(axis=-1, keepdims=True)

# WRONG (uses dO instead of dP):
dS = P * (dP - (P * dO).sum(axis=-1, keepdims=True))
```

All of these produce dQ, dK, dV values that "look like gradients" — they have
reasonable magnitudes and shapes — but fail finite-difference verification.

## Additional trap: the stored L format

The forward pass stores `L = m + log(l)`. To recompute P:
```python
P = exp(S - L[:, None])  # S is (T_q, T_kv), L is (T_q,)
```

If the forward accidentally stores `l` (sum of exps) instead of `L` (logsumexp),
the backward would need `P = exp(S - log(l[:, None]))` which is a different
computation. The test catches this because the `exp(S - wrong_value)` produces
incorrect P, which cascades to incorrect dV, dP, dS, etc.

## Reference implementation skeleton

```python
def flash_attention_fwd(Q, K, V, tile_size, causal=True):
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    T = tile_size
    
    O = np.zeros_like(Q)
    L = np.full((B, H, N), -np.inf)
    
    for b in range(B):
        for h in range(H):
            # ... standard tiled forward with online softmax ...
            # At the end of processing all KV tiles for a Q tile:
            #   O[b, h, q_s:q_e, :] = O_acc / l[:, None]
            #   L[b, h, q_s:q_e] = m + np.log(l)
    
    cache = {'O': O, 'L': L, 'Q': Q, 'K': K, 'V': V}
    return O, cache


def flash_attention_bwd(dO, cache, tile_size, causal=True):
    O = cache['O']
    L = cache['L']
    Q = cache['Q']
    K = cache['K']
    V = cache['V']
    
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    T = tile_size
    
    dQ = np.zeros_like(Q)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)
    
    for b in range(B):
        for h in range(H):
            # ... tiled backward pass ...
            # For each Q_tile (q_s:q_e) × KV_tile (k_s:k_e):
            #   S = Q_tile @ K_tile^T * scale
            #   P = exp(S - L_query[:, None])
            #   dV_tile += P^T @ dO_tile
            #   dP = dO_tile @ V_tile^T
            #   dS = P * (dP - (P * dP).sum(axis=-1, keepdims=True))
            #   dQ_tile += dS @ K_tile
            #   dK_tile += dS^T @ Q_tile
    
    return dQ, dK, dV
```

## Test code that catches the bugs

```python
def test_gradient_check():
    """Compare backward against central finite differences."""
    np.random.seed(42)
    B, H, N, D = 1, 1, 64, 32
    T = 16
    
    Q = np.random.randn(B, H, N, D).astype(np.float64)
    K = np.random.randn(B, H, N, D).astype(np.float64)
    V = np.random.randn(B, H, N, D).astype(np.float64)
    dO = np.random.randn(B, H, N, D).astype(np.float64)
    
    # Forward + backward
    O, cache = flash_attention_fwd(Q, K, V, T, causal=True)
    dQ, dK, dV = flash_attention_bwd(dO, cache, T, causal=True)
    
    # Finite difference check for dV (dQ and dK are more expensive)
    eps = 1e-5
    dV_fd = np.zeros_like(V)
    for b in range(B):
        for h in range(H):
            for i in range(N):
                for j in range(D):
                    V_plus = V.copy()
                    V_minus = V.copy()
                    V_plus[b, h, i, j] += eps
                    V_minus[b, h, i, j] -= eps
                    O_plus, _ = flash_attention_fwd(Q, K, V_plus, T, causal=True)
                    O_minus, _ = flash_attention_fwd(Q, K, V_minus, T, causal=True)
                    loss_plus = (dO * O_plus).sum()
                    loss_minus = (dO * O_minus).sum()
                    dV_fd[b, h, i, j] = (loss_plus - loss_minus) / (2 * eps)
    
    rel_err = np.abs(dV - dV_fd).max() / np.abs(dV_fd).max()
    print(f"dV relative error vs finite diff: {rel_err:.2e}")
    assert rel_err < 1e-5, f"dV gradient check FAILED: {rel_err:.2e}"
    
    # Spot-check dQ and dK at a few random positions
    for name, grad, tensor in [('dQ', dQ, Q), ('dK', dK, K)]:
        b, h, i, j = np.random.randint(0, B), np.random.randint(0, H), \
                      np.random.randint(0, N), np.random.randint(0, D)
        tensor_plus = tensor.copy()
        tensor_minus = tensor.copy()
        tensor_plus[b, h, i, j] += eps
        tensor_minus[b, h, i, j] -= eps
        O_plus, _ = flash_attention_fwd(
            Q if name != 'dQ' else tensor_plus,
            K if name != 'dK' else tensor_plus, V, T, causal=True
        )
        O_minus, _ = flash_attention_fwd(
            Q if name != 'dQ' else tensor_minus,
            K if name != 'dK' else tensor_minus, V, T, causal=True
        )
        loss_plus = (dO * O_plus).sum()
        loss_minus = (dO * O_minus).sum()
        fd_val = (loss_plus - loss_minus) / (2 * eps)
        rel = abs(grad[b, h, i, j] - fd_val) / (abs(fd_val) + 1e-10)
        print(f"{name}[{b},{h},{i},{j}] rel error: {rel:.2e}")
        assert rel < 1e-5, f"{name} gradient check FAILED at [{b},{h},{i},{j}]: {rel:.2e}"
    
    print("Gradient check PASSED\n")
```

## Why this will separate models

| Aspect | What good models do | What weak models do |
|--------|-------------------|-------------------|
| dsoftmax axis | sum over last axis (keys) | sum over wrong axis, or forget keepdims |
| dsoftmax formula | P * (dP - rowsum(P*dP)) | Forget to multiply by P, or use dO instead of dP |
| Stored intermediate | Store L = m + log(l) for stable recomputation | Store wrong intermediate, causing P recomputation errors |
| Tile accumulation | Accumulate dQ, dK, dV ACROSS tiles | Overwrite instead of accumulating |
| Causal mask in bwd | Skip entirely masked Q tile × KV tile pairs | Include masked tiles → incorrect dK from -inf scores |
| Memory | Never materialize (N,N) in backward either | Allocate (N,N) dS array |
| Gradient check | Passes at 1e-5 | Fails — the gradients "look right" but are wrong |

## Grading rubric

| Check | Weight | What it catches |
|-------|--------|----------------|
| dV matches finite differences at 1e-5 | 30% | Basic backward correctness |
| dQ spot-check matches finite diff at 1e-5 | 25% | Correct dS and dQ accumulation |
| dK spot-check matches finite diff at 1e-5 | 25% | Correct dS transpose and dK accumulation |
| Large N=4096 test: peak memory < N² | 10% | No full matrix materialized in backward |
| Causal masking handled correctly in bwd | 10% | Fully masked tile pairs are skipped |
