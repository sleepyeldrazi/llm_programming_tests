# Round 1 Summary: MiniMax-M2.7 vs Qwen3.6-27B

## Overall Scoreboard

| Task | MiniMax-M2.7 | Qwen3.6-27B | Winner | Margin |
|------|--------|---------|--------|--------|
| **KV Cache** | **64/100** | **91/100** | qwen36 | +27 |
| **Backwards Pass** | **76/100** | **92/100** | qwen36 | +16 |
| **Fused Softmax+TopK** | **58/100** | **88/100** | qwen36 | +30 |
| **Average** | **66** | **90** | **qwen36** | **+24** |

**Clear winner: Qwen3.6-27B — dominant across all 3 tasks.**

---

## Task 1: KV Cache System

| Dimension | MiniMax-M2.7 | Qwen3.6-27B |
|-----------|--------|---------|
| Correctness | 55 | 92 |
| Completeness | 75 | 95 |
| Code Quality | 60 | 88 |
| Depth of Analysis | 78 | 90 |
| Optimizations | 72 | 90 |
| GPU Mapping | 75 | 88 |
| Tests/Demos | 30 | 95 |
| **Overall** | **64** | **91** |

### MiniMax-M2.7 Critical Issues
- **Inverted causal mask** — masks the wrong triangle, allowing attention to future tokens
- **Broken batched caching** — all batch elements share the same `kv_cache` dict keyed only by layer, not by batch item
- **Prefill doesn't store KV** — prefill KV tensors never stored in persistent cache
- **No tests** — only a 3-step hardcoded demo with zero assertions
- **1,720-line monolith** — everything crammed into one file

### Qwen3.6-27B Strengths
- **10 passing demos** with numerical validation (cached attention diff < 1e-5, chunked prefill diff = 4.56e-10)
- **Modular 7-file architecture** — clean separation of concerns
- **Correct variable-length batching** — proper causal + length masks
- **3 working optimizations** — paged attention, int8 quantization, chunked prefill (all tested)
- **Quantitative analysis** — arithmetic intensity calculations, per-GPU context limits, real model comparisons (Llama, Mistral, GPT-4)

---

## Task 2: Layer Norm Backward Pass

| Dimension | MiniMax-M2.7 | Qwen3.6-27B |
|-----------|--------|---------|
| Correctness | 85 | 95 |
| Completeness | 80 | 95 |
| Code Quality | 70 | 90 |
| Numerical Stability | 75 | 95 |
| Gradient Check | 80 | 90 |
| Complexity Analysis | 80 | 90 |
| GPU Fusion | 85 | 85 |
| Tests/Benchmarks | 60 | 95 |
| **Overall** | **76** | **92** |

### MiniMax-M2.7 Weaknesses
- **Over-caching**: Stores 10 cache items when only 3 tensors are needed
- **No edge-case tests**: No tests for zero input, D=1, large offsets
- **No concrete stability demo**: Discusses catastrophic cancellation but never demonstrates it
- **Monolithic 750-line file**: Everything mixed together
- **Fragile gradient check**: Modifies input in-place without a copy

### Qwen3.6-27B Strengths
- **Minimal cache**: Only 4 items (x_hat, std_inv, glm5, D) — exactly what's needed
- **Concrete stability demo**: Shows naive variance fails at offset=1e8 while two-pass stays exact
- **3-file separation**: Core + tests + benchmarks
- **Edge-case tests**: Zero input, D=1, large D (1024), large mean, scale invariance
- **Alternative derivation cross-check**: Independent step-by-step chain rule verifies compact formula (<1e-10 error)

---

## Task 3: Fused Softmax + TopK CUDA

| Dimension | MiniMax-M2.7 | Qwen3.6-27B |
|-----------|--------|---------|
| Correctness | 40 | 95 |
| Completeness | 65 | 90 |
| Code Quality | 60 | 85 |
| CUDA Depth | 65 | 92 |
| Memory Design | 55 | 90 |
| Complexity Analysis | 60 | 88 |
| Naive Comparison | 55 | 88 |
| **Overall** | **58** | **88** |

### MiniMax-M2.7 Critical Issues
- **Broken inter-warp top-k merge**: Only ~100 of 256 threads contribute to final merge; 156 threads' results silently discarded → **produces incorrect top-k**
- **Compilation-stopping typo**: `topp_prob` instead of `topk_prob`
- **Misleading bandwidth claims**: Claims "4× reduction" but only counts one of three passes
- **Zero testing infrastructure**: No benchmark harness, no CPU reference, no correctness verification

### Qwen3.6-27B Strengths
- **Two kernel versions** (v1 + optimized v2 with vectorized float4 loads)
- **Correct warp-by-warp merge** — properly collects all 4096 candidates
- **Shared-memory min-heap** for O(log K) insertions
- **Complete benchmark harness** with CPU reference and correctness tests
- **Honest 3-pass bandwidth analysis** — correctly identifies kernel as compute-bound (expf throughput)

---

## What Separated These Two

| Factor | MiniMax-M2.7 | Qwen3.6-27B |
|--------|--------|---------|
| **Correctness** | Buggy in all 3 tasks | Correct in all 3 |
| **Testing** | None / minimal | Comprehensive with assertions |
| **Analysis depth** | High-level / conceptual | Quantitative with real numbers |
| **Code organization** | Monolithic | Modular and focused |
| **Engineering rigor** | Claims untested | Every claim validated |

**The decisive pattern**: MiniMax-M2.7 was conceptually broad but executionally weak — it mentioned many optimizations and ideas but delivered buggy, untested code. Qwen3.6-27B was narrower in scope but flawlessly executed — every claim backed by working, validated code.
