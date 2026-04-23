# Overall Summary: All Model Comparisons

## Complete Scoreboard

### Round 1: MiniMax-M2.7 vs Qwen3.6-27B

| Task | MiniMax-M2.7 | Qwen3.6-27B | Winner | Margin |
|------|--------|---------|--------|--------|
| KV Cache | **64** | **91** | qwen36 | +27 |
| Backwards Pass | **76** | **92** | qwen36 | +16 |
| Fused Softmax+TopK | **58** | **88** | qwen36 | +30 |
| **Average** | **66** | **90** | **qwen36** | **+24** |

### Round 2: GLM-5 vs Qwen3.6-27B

| Task | GLM-5 | Qwen3.6-27B | Winner | Margin |
|------|--------|---------|--------|--------|
| KV Cache | **82** | **94** | qwen36 | +12 |
| Backwards Pass | **82** | **93** | qwen36 | +11 |
| Fused Softmax+TopK | **80** | **78** | **glm5** | **+2** |
| **Average** | **81** | **88** | **qwen36** | **+7** |

---

## Final Rankings

| Rank | Model | Average Score | Best Task | Worst Task | Notes |
|------|-------|--------------|-----------|------------|-------|
| 🥇 | **Qwen3.6-27B** | **89** | KV (92 avg) | Fuse (78) | Won 5/6 matchups. Correct, comprehensive, quantitative. |
| 🥈 | **GLM-5** | **81** | KV / Backwards (82) | Fuse (80) | Correct, concise, well-engineered. Won fuse task. |
| 🥉 | **MiniMax-M2.7** | **66** | Backwards (76) | Fuse (58) | Critical bugs in all 3 tasks. No tests. |

---

## Task-by-Task Breakdown

### KV Cache
- **Qwen3.6-27B (91, 94)** — Consistently dominant. 10 demos, modular architecture, real model comparisons, GQA, arithmetic intensity analysis.
- **GLM-5 (82)** — Correct, good tests, excellent docs, INT4 quantization. Lost on missing MLP/causal masking and less systems depth.
- **MiniMax-M2.7 (64)** — Inverted causal mask, broken batched caching, no tests, 1,720-line monolith.

### Backwards Pass
- **Qwen3.6-27B (92, 93)** — Minimal cache, concrete stability demo, 3-file separation, 5 edge-case tests, cross-check derivation.
- **GLM-5 (82)** — Excellent conciseness (280 lines), minimal cache, safe gradient check. Lost on no edge-case tests and no stability demo.
- **MiniMax-M2.7 (76)** — Over-cached (10 items), no edge-case tests, fragile in-place gradient check, monolithic.

### Fused Softmax+TopK
- **GLM-5 (80)** — Single-pass online softmax (research-level), 1× global reads, register heaps. Won narrowly (+2) but has cross-warp merge bug when WARPS_PER_BLOCK > 1.
- **Qwen3.6-27B (88, 78)** — Two kernel versions, correct merge, vectorized loads, benchmark harness. Lost on fuse due to suboptimal 3-pass algorithm (12V reads vs 4V).
- **MiniMax-M2.7 (58)** — Broken inter-warp merge (156 threads ignored), compilation typo, zero tests.

---

## Key Patterns

### What Separates the Tiers

| Dimension | MiniMax-M2.7 | GLM-5 | Qwen3.6-27B |
|-----------|--------|--------|---------|
| **Correctness** | ❌ Buggy in all 3 | ✅ Correct (1 minor bug) | ✅ Correct in all 3 |
| **Testing** | ❌ None | ⚠️ Basic assertions | ✅ Comprehensive suites |
| **Analysis depth** | ⚠️ High-level / conceptual | ✅ Good | ✅ Quantitative + real models |
| **Code quality** | ❌ Bloated monoliths | ✅ Concise & focused | ✅ Modular & production-grade |
| **Algorithmic sophistication** | ⚠️ Claims many, delivers few | ✅ Online softmax, INT4 | ✅ Solid, well-validated |
| **Engineering rigor** | ❌ Untested claims | ✅ Clean & minimal | ✅ Every claim validated |

### The Decisive Factors

1. **Testing is everything**: Qwen3.6-27B's comprehensive test suites caught issues that GLM-5 and MiniMax-M2.7 missed. glm5's fuse bug (cross-warp merge) would have been caught by a multi-row test. MiniMax-M2.7's causal mask bug would have been caught by any numerical validation.

2. **Concrete > theoretical**: Qwen3.6-27B demonstrated numerical stability problems with actual numbers; MiniMax-M2.7 and GLM-5 only described them. This pattern repeated across all tasks.

3. **Minimal cache wins**: Both Qwen3.6-27B and GLM-5 used minimal caches (3-4 items), while MiniMax-M2.7 over-cached (10 items). The backward pass is particularly sensitive to this — the compact projection formula eliminates most intermediates.

4. **Algorithmic sophistication has tradeoffs**: GLM-5's online softmax was theoretically optimal but harder to get right (the cross-warp bug). Qwen3.6-27B's 3-pass approach was simpler and correct but suboptimal in memory traffic. The ideal is glm5's algorithm + qwen36's testing.

---

## The Ideal Hybrid

Combining the best of each model would score ~95/100 on each task:

| Task | Best Algorithm | Best Testing | Best Analysis |
|------|---------------|-------------|---------------|
| **KV Cache** | Qwen3.6-27B (full transformer, GQA) | Qwen3.6-27B (10 demos) | Qwen3.6-27B (arithmetic intensity, real GPUs) |
| **Backwards** | Qwen3.6-27B or GLM-5 (both minimal cache) | Qwen3.6-27B (edge cases, cross-check) | Qwen3.6-27B (concrete stability demo) |
| **Fuse** | GLM-5 (online softmax, 1× reads) | Qwen3.6-27B (benchmark harness, CPU ref) | GLM-5 (accurate bandwidth analysis) |

---

## Files in This Folder

| File | Matchup | Size |
|------|---------|------|
| `kv_comparison.md` | MiniMax-M2.7kv vs Qwen3.6-27Bkv | 20KB |
| `backwards_comparison.md` | MiniMax-M2.7backwards vs Qwen3.6-27Bbackwards | 11KB |
| `fuse_comparison.md` | MiniMax-M2.7fuse vs Qwen3.6-27Bfuse | 28KB |
| `glm5_kv_comparison.md` | GLM-5kv vs Qwen3.6-27Bkv | 21KB |
| `glm5_backwards_comparison.md` | GLM-5backwards vs Qwen3.6-27Bbackwards | 10KB |
| `glm5_fuse_comparison.md` | GLM-5fuse vs Qwen3.6-27Bfuse | 35KB |
| `model_vs_qwen36_summary.md` | Round 1 summary | This file's sibling |
| `glm5_vs_qwen36_summary.md` | Round 2 summary | This file's sibling |
| `overall_summary.md` | This file | — |
