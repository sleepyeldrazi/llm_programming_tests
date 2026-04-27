All gradients pass the finite-difference check with relative errors ~1e-10. Here's a summary of what's in `layer_norm.py`:

---

## Key Design Decisions

### 1. Simplified `dx` formula (avoids redundant recomputation)

Instead of computing `dvar` and `dmean` separately and then combining them, I substitute them into a single closed-form expression:

```
dx = rstd · [ dxhat  -  x̂ · mean(dxhat · x̂)  -  mean(dxhat) ]
```

This requires only **two reductions** over the D dimension (the `proj` and `sum` terms), and everything else is elementwise. The intermediates `x̂` and `rstd` are reused directly from the forward pass cache.

### 2. Numerical stability — where things go wrong

| Danger zone | Why | Mitigation |
|---|---|---|
| `var ≈ 0` → `1/σ̂ → ∞` | All elements identical | `eps = 1e-5` caps `rstd ≤ 1/√ε ≈ 316` |
| `x - mean` cancellation | Large x, small σ | Two-pass variance (already used); Welford's for extreme cases |
| `xc²` overflow | Float16/32 for large x | Upcast to float32 for computation |
| Gradient explosion | `dx ∝ 1/σ̂` | Gradient clipping upstream; eps bounds the scale |

### 3. Complexity

- **Time:** `O(B·T·D)` — each element is touched a constant number of times in both forward and backward.
- **Memory:** Forward stores `xhat (B,T,D)` + `rstd (B,T,1)`. Backward allocates one temporary `(B,T,D)` for `dxhat`, which can be fused in-place on GPU.

### 4. GPU fusion into one kernel

The strategy is **one thread-block per (B,T) row** of length D:

1. **Load** `dy` and `xhat` → compute `dxhat = dy * gamma` into shared memory
2. **Reduce** across the block to get `Σ(dxhat · x̂)` and `Σ(dxhat)` via warp shuffles
3. **Compute** `dx_d = rstd * (dxhat_d - x̂_d * proj/D - sum/D)` per thread
4. **Atomic add** contributions to global `dgamma` and `dbeta`

This reads each element **once** from global memory (~3D traffic per row) versus ~10D+ for an unfused chain of separate kernels. For D ≤ 1024, shared memory (~12 KB per block in float32) is well within GPU limits.