# Round 3: Flash Attention Backward Pass — Head-to-Head Analysis

## Executive Summary

| Model | Grade | dV check | dQ spot | dK spot | vs Naive | Memory | Notes |
|-------|-------|----------|---------|---------|----------|--------|-------|
| **Kimi K2.6** | A | 3.4e-09 | 1.9e-09 | 1.4e-09 | 1.7e-11 | 13.4 MB | Cleanest code, two-pass, excellent precision |
| **GLM-5** | A- | 1.2e-07 | 1.6e-09 | 1.1e-08 | 0.0 | 12.5 MB | D-optimization, single-pass, best efficiency |
| **Qwen3-6** | A- | 7.2e-07 | 1.3e-07 | 6.6e-09 | 1.1e-10 | 6.3 MB | Two-pass, lowest memory, 5 subtests |
| **GLM-5.1** | B+ | 8.5e-06 | 2.9e-08 | 8.3e-08 | 2.7e-05 | 9.4 MB | D-optimization, slightly higher errors |
| **MiniMax-M2.7** | — | — | — | — | — | — | Did not participate |

**All four participants pass every test.** The dsoftmax trap caught nobody — every model
used the correct formula. The real differentiators this round are algorithmic elegance,
code clarity, and memory efficiency, not correctness.

---

## The dsoftmax Formula: Nobody Fell For It

The intended trap was the dsoftmax gradient:

```python
# CORRECT:
dS = P * (dP - rowsum(P * dP))

# WRONG variants that produce plausible-but-wrong results:
dS = P * (dP - rowsum(P * dP).sum(axis=-2))     # wrong axis
dS = dP - rowsum(P * dP)                          # forgets to multiply by P
dS = P * dP / rowsum(P * dP)                      # divides instead of subtracts
```

**All four models wrote the correct formula.** Two different strategies emerged:

| Strategy | Models | How it works |
|----------|--------|-------------|
| **D-optimization** | GLM-5, GLM-5.1 | Precompute `D = (dO ⊙ O).sum(axis=-1)`, use `dS = P * (dP - D)`. Mathematically identical to `rowsum(P*dP)` but computed once per Q tile from O and dO. Single pass over KV tiles. |
| **Two-pass** | Kimi K2.6, Qwen3-6 | Pass 1: accumulate `rowsum(P*dP)` across all KV tiles. Pass 2: recompute P and dP, use accumulated rowsum. Double computation of P and dP. |

The D-optimization is from the FlashAttention paper (Dao et al., 2022, Eq. 12). The identity `D = rowsum(dO ⊙ O)` holds because `O = P @ V` implies `rowsum(P ⊙ (dO @ V^T)) = rowsum(dO ⊙ (P @ V))`. GLM-5 and GLM-5.1 recognized this optimization; Kimi and Qwen used the simpler but slightly redundant two-pass approach.

---

## Per-Model Analysis

### Kimi K2.6 — Grade: A

**Strengths:**
- Cleanest implementation overall. Clear section headers, well-structured per-head loops.
- Two-pass approach with explicit `rowsum_PdP` accumulation. The algorithm is easy to follow.
- Handles the `-inf` edge case explicitly: `np.isinf(S)` guards for masked positions in `exp_S`.
- Uses `np.where(np.isinf(S), 0.0, exp_S)` to zero out masked contributions, preventing NaN from `exp(-inf - (-inf))`.
- Causes the skip condition `if kv_start > q_end - 1: continue` in both forward and backward.
- Tests are well-structured with explicit error checking and clear output.
- Excellent precision: dV finite diff error is 3.4e-09 (best of all models).
- Naive backward uses `np.einsum` for clean batch operations.

**Weaknesses:**
- Two-pass recomputation of P and dP is redundant. The D-optimization would avoid recomputing both.
- No special handling for `l == 0` in forward's `L = m + np.log(l)` — `np.log(0) = -inf`, producing NaN for `(-inf) + (-inf)` on fully masked rows. The test cases don't trigger this, but it would fail on a fully causal-masked early row.
- Peak backward memory (13.4 MB) is the highest of all implementations. The two-pass approach stores `P` and `dP` again on pass 2, though these are tile-sized and shouldn't dominate.

### GLM-5 — Grade: A-

**Strengths:**
- **Uses the D-optimization**: `Di = (do_tile * o_tile).sum(axis=-1, keepdims=True)`. Only one pass over KV tiles in the backward pass.
- This is the mathematically elegant approach from the FlashAttention paper.
- Forward pass correctly stores `L = m + np.log(l)`.
- Backward pass uses `dS = P * (dP - Di)` which is correct and efficient.
- Includes a bonus "forward/backward sanity check" on a tiny test case before the main tests.
- dV finite diff error is 1.2e-07 — cleanly within threshold.
- Comparison against naive backward shows essentially zero error on test 2 (dQ/dK/dV all ~0.0).
- Memory ratio is 18.6% (12.5 MB / 67.1 MB) — well under the 20% threshold.

**Weaknesses:**
- No special handling for `l == 0` in forward (same issue as Kimi).
- The `Di` variable naming is slightly confusing — it's the D scalar from the FlashAttention paper, but the code doesn't explain the mathematical equivalence to `rowsum(P*dP)`.
- The gradient check for dV does a FULL finite difference check (64×32 = 2048 evaluation points) which is thorough but slow. GLM-5.1 and Qwen3-6 also do this, but Kimi K2.6 only checks "ALL elements" of dV without the same nested loops (it uses a different sampling approach).
- Code is less modular than Kimi's — the test functions aren't separated into named functions, just sequential code under `if __name__ == '__main__'`.

### Qwen3-6 — Grade: A-

**Strengths:**
- **Lowest memory usage**: 6.3 MB peak for the N=4096 test, compared to 9-13 MB for others.
- Most thorough testing: 5 distinct subtests including accuracy, non-causal, larger batch, uneven tiles, and memory. The only model that tested beyond the 3 required tests.
- Proactively collects KV tile data in a list (`kv_data`) before pass 1, avoiding redundant slicing.
- Properly handles forward edge cases: `np.where(valid, ..., 1.0)` for correction factors when rows are fully masked.
- Clean `relative_error()` helper function.
- Backward's two-pass approach explicitly separates rowsum accumulation from gradient computation, making the algorithm easy to verify.
- 5-subtest structure demonstrates engineering thoroughness — this is the same pattern Qwen3-6 showed in earlier rounds.

**Weaknesses:**
- Two-pass approach recomputes P and dP (same redundancy as Kimi).
- Forward pass uses per-row state tracking (`m[q_start:q_end]`, `l[q_start:q_end]`) which requires careful indexing into global arrays rather than local accumulators. More complex than necessary.
- dV finite diff error (7.2e-07) is the highest among passing models, though still 100× below the 1e-5 threshold.
- The forward pass normalization happens OUTSIDE the Q tile loop:
  ```python
  O[b, h] = O_bh / l[:, None]
  ```
  This is correct but applied to the entire head at once rather than per Q tile. While mathematically equivalent, it means the output O_bh contains un-normalized accumulated values until the very end — less numerically stable than per-tile normalization.

### GLM-5.1 — Grade: B+

**Strengths:**
- **Uses the D-optimization** (same as GLM-5). Computes `D_diag = (dO * O).sum(axis=-1)` once.
- **Best forward edge case handling**: `np.where(l_acc > 0, m_acc + np.log(l_acc), m_acc)` — explicitly handles `l == 0` (fully masked rows) by setting L to just `m` (which would be -inf).
- Uses `np.einsum` for naive backward computation, which is cleaner than per-head loops.
- Forward pass uses `break` instead of `continue` for causal tile skip — correct because KV tiles are processed in increasing order, so once we pass the diagonal, all subsequent tiles are also fully masked.
- Good code organization with separate named test functions.

**Weaknesses:**
- **Higher gradient errors than peers.** dQ vs naive relative error is 2.69e-05, which is 1000× higher than GLM-5's "0.0" and Kimi's 1.7e-11. While still within the 1e-4 threshold, this is noticeably worse and suggests a minor numerical issue.
- The `break` instead of `continue` for causal skip is actually an **optimization bug**: when Q tiles are processed in order and the first skipped KV tile is detected, `break` exits the KV loop. But this only works because the KV tiles are iterated in increasing order AND the Q tile start is fixed. If Q tiles were processed in a different order, this would break. For the standard forward-left-to-right iteration, it's correct but fragile.
- The gradient check's dV finite difference function uses `eps=1e-6` instead of `1e-5`, which can amplify floating-point noise.
- The "spot-check" code for dQ and dK in test 1 is duplicated (it computes finite differences for dV AGAIN inside a spot-check loop, even though dV was already checked fully). Messy.

### MiniMax-M2.7 — Did Not Participate

No files in `minimax-m2.7/flash_attention_bwd/` beyond PROMPT.md. Either the model was not run or it failed to produce output.

---

## Comparative Metrics

| Metric | Kimi K2.6 | GLM-5 | Qwen3-6 | GLM-5.1 |
|--------|-----------|-------|---------|---------|
| dsoftmax strategy | Two-pass | D-optimization | Two-pass | D-optimization |
| Backward passes over KV | 2 | 1 | 2 | 1 |
| dV vs finite diff | 3.4e-09 | 1.2e-07 | 7.2e-07 | 8.5e-06 |
| dQ vs naive | 1.7e-11 | 0.0 | 1.1e-10 | 2.7e-05 |
| Peak memory (N=4096) | 13.4 MB | 12.5 MB | 6.3 MB | 9.4 MB |
| l==0 guard in forward | No | No | Partial (valid mask) | Yes |
| Subtests beyond required 3 | 0 | 1 (sanity check) | 2 (non-causal, uneven tiles) | 0 |
| Code clarity | Excellent | Good | Good | Fair |
| Lines of code | ~350 | ~240 | ~370 | ~340 |

---

## The Trap Analysis: Why Nobody Fell

The dsoftmax formula trap caught zero models this round. Three explanations:

1. **The prompt was too explicit.** The challenge prompt literally showed the correct formula: `dS = P * (dP - (P * dP).sum(axis=-1, keepdims=True))`. It also showed wrong variants as warnings. This was arguably too big a hint.

2. **This is Round 3.** The models that survived to this point (GLM-5, Qwen3-6) already passed the Flash Attention forward pass in Round 2. They understand the domain. Kimi K2.6 is a top-5 coding model specifically designed for complex engineering tasks. GLM-5.1 is an updated GLM-5.

3. **Training data coverage.** The FlashAttention paper is one of the most-cited ML papers of 2022-2023. The backward pass formulas are documented in dozens of blog posts and tutorials. Any model with good code training data has seen this.

**The real differentiator became engineering quality, not algorithmic correctness.** Kimi K2.6 and GLM-5 tied on the core algorithm but diverged on secondary properties: code clarity (Kimi wins), computational efficiency (GLM-5's D-optimization wins), memory usage (Qwen3-6 wins), and edge case handling (GLM-5.1's l==0 guard wins).

---

## Notable Implementation Details

### The `break` vs `continue` Distinction

GLM-5.1 uses `break` to exit the KV tile loop after the first fully-causal-masked tile:

```python
if causal:
    if k_start > q_end - 1:
        break  # GLM-5.1
```

All others use `continue`:
```python
if causal and kv_start > q_end - 1:
    continue  # GLM-5, Kimi, Qwen3-6
```

`break` is correct because KV tiles are iterated in increasing order. Once the first KV tile starts after the Q tile ends, ALL subsequent KV tiles will also start after the Q tile ends. The `break` is an optimization that avoids checking the condition for every subsequent tile. However, it's fragile — if the iteration order changes, `break` becomes a bug while `continue` remains correct.

### The `rowsum(dO ⊙ O)` Identity

GLM-5 and GLM-5.1 both use the identity `rowsum(P ⊙ dP) = rowsum(dO ⊙ O)`. This is derived from:

```
O = P @ V
dP = dO @ V^T
rowsum(P ⊙ dP) = sum_j P_ij * sum_k dO_ik * V_jk
                = sum_k dO_ik * sum_j P_ij * V_jk
                = sum_k dO_ik * O_ik
                = rowsum(dO ⊙ O)
```

This means the backward pass only needs ONE pass over KV tiles (compute dV, compute dS, accumulate dQ and dK) instead of two passes (first accumulate rowsum, then compute gradients). It's the optimization from the original FlashAttention paper.

## Ranking

| Rank | Model | Rationale |
|------|-------|-----------|
| **1** | **Kimi K2.6** | Best precision, cleanest code, correct algorithm. Two-pass is redundant but clear. |
| **2** | **GLM-5** | D-optimization is elegant. Tied with Kimi on correctness. Slightly less polished code. |
| **3** | **Qwen3-6** | Best memory usage, most tests. Two-pass is redundant. Slightly higher dV error. |
| **4** | **GLM-5.1** | D-optimization and l==0 guard are good. Higher errors and `break` fragility hurt. |
| — | **MiniMax-M2.7** | No submission. |
