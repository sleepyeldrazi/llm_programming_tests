Everything passes. Here's a summary of what was built:

## Files

| File | Purpose |
|---|---|
| `layer_norm_backward.py` | Core implementation + gradient check + complexity analysis + GPU fusion discussion |
| `test_layer_norm.py` | Edge cases (zero input, D=1, large D, large mean, scale invariance) |
| `benchmark_layer_norm.py` | Variance stability demo, performance benchmarks, alternative derivation cross-check |

## Key results

**Gradient accuracy** — all three gradients pass with relative error ~10⁻¹¹:
```
dx        relative error: 5.04e-11  ✓
dgamma    relative error: 1.75e-11  ✓
dbeta     relative error: 1.46e-11  ✓
```

**Numerical stability** — the two-pass variance completely avoids catastrophic cancellation:
```
offset=1e10:  naive=0.000000  stable=2.000000  true=2.000000
```
The naive `E[x²] - E[x]²` formula loses all precision at offset 1e10, while the two-pass algorithm stays exact.

**Backward formula** — the compact projection form:
```
dx = σ⁻¹ · [ g - mean(g) - x_hat · mean(g · x_hat) ]
```
where `g = γ · dy`. This requires only **two reductions** over D and **zero recomputation** of forward intermediates.

**Memory** — the cache stores exactly 3 tensors: `x_hat` (B·T·D), `std_inv` (B·T), `gamma` (D). No `x_centered`, no `var`, no `x` — the backward formula is self-contained from these.

**GPU fusion** — the fused kernel reduces global memory traffic from ~12 accesses/element (naive separate kernels) to 4 (forward) and 5 (backward), a 2–3× speedup since layer norm is memory-bandwidth bound.