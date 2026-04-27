# Round 2 Summary: GLM-5 vs Qwen3.6-27B

## Overall Scoreboard

| Task | GLM-5 | Qwen3.6-27B | Winner | Margin |
|------|--------|---------|--------|--------|
| **KV Cache** | **82/100** | **94/100** | qwen36 | +12 |
| **Backwards Pass** | **82/100** | **93/100** | qwen36 | +11 |
| **Fused Softmax+TopK** | **80/100** | **78/100** | **glm5** | **+2** |
| **Average** | **81** | **88** | **qwen36** | **+7** |

**Winner: Qwen3.6-27B — won 2 of 3 tasks, but GLM-5 made it competitive (especially on fuse).**

---

## Task 1: KV Cache System

| Dimension | GLM-5 | Qwen3.6-27B |
|-----------|--------|---------|
| Correctness | 95 | 95 |
| Completeness | 78 | 95 |
| Code Quality | 80 | 92 |
| Depth of Analysis | 82 | 96 |
| Optimizations | 85 | 93 |
| GPU Mapping | 80 | 95 |
| Tests/Demos | 82 | 90 |
| **Overall** | **82** | **94** |

### GLM-5 Strengths
- **Excellent documentation** — best-in-class README with ASCII diagrams and pedagogical explanations
- **INT4 quantization** — only implementation with true 2-values-per-byte packing
- **Rigorous correctness testing** — cached vs non-cached attention matches to 1e-5, quantized cache has bounded error assertions
- **Clean, readable code** — very approachable for learning
- **No correctness bugs** — correct attention, proper cache updates, working batched inference

### GLM-5 Weaknesses
- **Incomplete transformer** — no MLP, no causal mask, no positional encoding
- **Limited batched masking** — variable-length batching lacks full per-sequence masking
- **Less systems analysis** — no arithmetic intensity calculations, no real GPU context limits

### Qwen3.6-27B Strengths (same as Round 1)
- Full transformer decoder with LayerNorm, MLP, GELU, residuals, positional encoding
- GQA support — modern architecture awareness (Llama-2/3, Mistral)
- Outstanding systems analysis — memory growth with real model names, max context per GPU, arithmetic intensity proving memory-bound generation
- 10 comprehensive demos including full generation with temperature/top-k sampling

---

## Task 2: Layer Norm Backward Pass

| Dimension | GLM-5 | Qwen3.6-27B |
|-----------|--------|---------|
| Correctness | 92 | 95 |
| Completeness | 80 | 95 |
| Code Quality | 88 | 90 |
| Numerical Stability | 80 | 95 |
| Gradient Check | 85 | 92 |
| Complexity Analysis | 82 | 90 |
| GPU Fusion | 85 | 88 |
| Tests/Benchmarks | 60 | 95 |
| **Overall** | **82** | **93** |

### GLM-5 Strengths
- **Exceptional conciseness** — ~280 lines covers everything (forward, backward, gradient check, complexity, GPU fusion, stability discussion)
- **Minimal cache** — `(xhat, rstd, glm5)` — only 3 items, exactly what's needed
- **Modern NumPy API** — `default_rng`, type hints
- **Safe gradient check** — operates on copies, not in-place
- **Clean GPU fusion description** with memory traffic quantification (≈3D vs ≈10D+ unfused)

### GLM-5 Weaknesses
- **No edge-case tests** — no zero input, D=1, large offsets, etc.
- **No concrete stability demo** — discusses catastrophic cancellation but never shows it
- **No performance benchmarks** — no timing or throughput measurements
- **Single file** — while concise, separation into test/benchmark files would be better

### Qwen3.6-27B Strengths (same as Round 1)
- 3-file separation: core + tests + benchmarks
- Concrete catastrophic cancellation demo (naive variance = 0 at offset=1e8; two-pass = exact)
- 5 edge-case test categories with assertions
- Independent backward formula cross-check (<1e-10 error)

---

## Task 3: Fused Softmax + TopK CUDA

| Dimension | GLM-5 | Qwen3.6-27B |
|-----------|--------|---------|
| Correctness | 65 | 95 |
| Completeness | 90 | 85 |
| Code Quality | 88 | 82 |
| CUDA Depth | 92 | 82 |
| Memory Design | 90 | 70 |
| Complexity Analysis | 88 | 72 |
| Naive Comparison | 85 | 78 |
| **Overall** | **80** | **78** |

### GLM-5 Strengths
- **Single-pass online softmax** (Milakov & Gimelshein 2018) — reads V only once, optimal
- **Research-level CUDA knowledge** — register-resident sorted arrays, warp shuffle reductions, occupancy analysis
- **Excellent documentation** — 9-section DESIGN.md with quantitative analysis, ASCII architecture diagram
- **Accurate complexity analysis** — correctly identifies bandwidth-bound nature
- **One warp per row** design — elegant mapping with strided coalesced access

### GLM-5 Critical Weakness
- **🐛 Cross-warp merge bug** — When `WARPS_PER_BLOCK > 1`, the merge conflates heaps from **different rows**. Only works correctly with `WARPS_PER_BLOCK = 1`. The design claims "one warp per row" but then treats all warps in a block as cooperating on the same row — a fundamental contradiction.

### Qwen3.6-27B Strengths
- **No critical correctness bugs** — simpler one-block-per-row design avoids ambiguity
- **Two kernel versions** (v1 + v2) showing iterative improvement
- **Vectorized float4 loads** in v2 for wider memory transactions
- **Better test coverage** — tests LLaMA-sized vocabularies (V=50257, K=256)

### Qwen3.6-27B Weaknesses
- **Suboptimal 3-pass algorithm** — 12× more global reads than necessary (3 passes × 4V bytes = 12V vs glm5's 4V)
- **Flawed complexity analysis** — incorrectly claims compute-bound; with 12V reads it's actually bandwidth-bound
- **Dead code in v2** — `warp_topk_merge` and `process_float4` functions are never called

### The Ideal Hybrid
A production implementation would combine glm5's **online softmax algorithm** and **register-resident heap** with qwen36's **vectorized loads** and **comprehensive testing** — scoring ~95/100.

---

## What Made GLM-5 Competitive

| Factor | GLM-5 | Qwen3.6-27B |
|--------|--------|---------|
| **Correctness** | Correct (1 minor bug on fuse) | Correct in all 3 |
| **Testing** | Basic (good assertions, limited coverage) | Comprehensive |
| **Analysis depth** | Good | Excellent (quantitative + real models) |
| **Code organization** | Clean, focused | Modular and production-grade |
| **Algorithmic sophistication** | Excellent (online softmax, INT4) | Good (solid but conventional) |

**Key insight**: GLM-5 was much closer to Qwen3.6-27B (+7 avg margin) than MiniMax-M2.7 was (+24). glm5's code was correct, concise, and well-engineered. It lost mainly on completeness (fewer tests, less analysis depth) rather than fundamental correctness issues.
