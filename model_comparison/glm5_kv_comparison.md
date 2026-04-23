# Head-to-Head Analysis: KV-Cache System for Autoregressive Transformer Inference

**Date:** 2026-04-23  
**Task:** Implement an efficient KV-cache system for autoregressive transformer inference from scratch.  
**GLM-5:** GLM-5 KV/  
**Qwen3.6-27B:** Qwen3.6-27B KV/

---

## Executive Summary

Both implementations successfully address the core KV-cache problem with pure NumPy, no frameworks. Both provide:
- Core KV-cache data structures with pre-allocated memory
- Incremental decoding (one token at a time)
- Multi-head attention using cached keys/values
- Memory growth analysis
- Multiple optimizations (paged attention, quantization, chunked prefill)
- GPU execution mapping explanations

However, **Qwen3.6-27B (Qwen3.6-27B KV/) is the clear winner** by a substantial margin. It delivers a more complete, production-oriented architecture with significantly deeper analysis, cleaner separation of concerns, richer GPU mapping, and a more comprehensive demo suite. GLM-5 is solid and correct but narrower in scope and less polished in its architectural layering.

| Criterion | GLM-5 (glm5) | Qwen3.6-27B (qwen36) |
|-----------|----------------|------------------|
| **Correctness** | 95/100 | 95/100 |
| **Completeness** | 78/100 | 95/100 |
| **Code Quality** | 80/100 | 92/100 |
| **Depth of Analysis** | 82/100 | 96/100 |
| **Optimizations** | 85/100 | 93/100 |
| **GPU Mapping** | 80/100 | 95/100 |
| **Tests/Demos** | 82/100 | 90/100 |
| **Overall** | **82/100** | **94/100** |

**Winner: Qwen3.6-27B by ~12 points.**

---

## 1. Correctness (Both: 95/100)

### GLM-5
- All 8 tests pass cleanly.
- Cached attention output matches non-cached (full recomputation) to within `1e-5`.
- Paged cache correctly allocates, writes, reads, and frees blocks.
- Quantized cache (INT8/INT4) round-trips with bounded error.
- Variable sequence lengths are handled via per-batch `seq_lens` tracking.
- **Minor issue:** The `multi_head_attention_batched` function is essentially identical to `multi_head_attention_with_cache` and does not actually demonstrate true batched masking in a single tensor operation—it still loops per batch element. The mask-building logic exists but isn't exercised in a meaningful batched GEMM path.

### Qwen3.6-27B
- All 10 demos run to completion (no crashes, no assertion failures).
- Cached attention matches manual computation to `1e-5`.
- Chunked prefill matches full attention to `4.56e-10`.
- Paged attention correctly manages physical page allocation and retrieval.
- Quantized cache round-trips with acknowledged per-position scale overhead.
- Variable-length batching works via `lengths` arrays and explicit causal + length masks.
- **Minor issue:** Demo 6 (quantized cache) shows a **very high max absolute error (~5.1)** and **max relative error (~1.7)** for one token. This is acknowledged in the printout ("per-position quantization has high overhead"), but the demo still exposes a real numerical weakness in the per-position scale approach. The code comments correctly note that production should use shared per-channel scales.

### Verdict
Both are fundamentally correct. Qwen3.6-27B's quantized cache has a documented weakness; GLM-5's "batched" function is a bit of a misnomer. Tie.

---

## 2. Completeness (GLM-5: 78/100, Qwen3.6-27B: 95/100)

### Prompt Requirements Checklist

| Requirement | GLM-5 | Qwen3.6-27B |
|------------|---------|---------|
| 1. Incremental decoding (one token at a time) | ✅ `IncrementalDecoder.forward_step` | ✅ `TransformerDecoder.generate_step` |
| 2. Avoid recomputing attention for past tokens | ✅ Cache read in `multi_head_attention_with_cache` | ✅ `cached_attention()` reads from cache |
| 3. Multi-head attention | ✅ | ✅ |
| 3. Batching with variable sequence lengths | ⚠️ Partial (per-batch loop, no true batched tensor masking) | ✅ `build_variable_length_mask`, `cached_attention_with_mask` |
| 4. Data structure layout (memory format) | ✅ Excellent README + docstrings | ✅ Excellent README + `CacheConfig` dataclass |
| 4. Update logic per step | ✅ `KVCache.update()` | ✅ `KVCache.update()` |
| 4. Attention computation using cached K/V | ✅ | ✅ |
| Memory growth analysis | ✅ Table + `memory_analysis()` | ✅ Comprehensive `memory_analysis.py` with model specs |
| At least two optimizations | ✅ 3 optimizations (Paged, Chunked, Quantized) | ✅ 3 optimizations + hybrid (Paged, Quantized, Chunked, Hybrid) |
| GPU execution mapping | ✅ Good (FlashAttention, memory hierarchy, CUDA pseudocode) | ✅ Excellent (Tensor Core analysis, arithmetic intensity, multi-GPU, tuning guide) |

### GLM-5 Gaps
1. **No full transformer layer implementation.** GLM-5 stops at the attention level. It has an `IncrementalDecoder` that does LayerNorm + Attention + residual, but there is **no MLP/feed-forward network**, no proper pre-norm/post-norm architecture, and no complete transformer block. The `forward_step` is more of a skeleton than a real layer.
2. **No positional encoding.** The decoder uses raw embeddings without position information.
3. **No causal mask construction.** The prompt prefill in GLM-5 does not apply a causal mask—it relies on the fact that the cache only contains past tokens during decode, but the prefill phase itself lacks causal masking in the code.
4. **Limited batched masking.** The `multi_head_attention_batched` function claims to handle variable lengths but doesn't actually construct or apply a mask in the demonstrated path.
5. **No GQA/MQA variants.** GLM-5 only implements standard MHA.

### Qwen3.6-27B Strengths
1. **Full transformer decoder.** `TransformerDecoderLayer` includes LayerNorm, QKV projection, cached attention, output projection, MLP with GELU, and residual connections. `TransformerDecoder` orchestrates prefill + generation with positional encoding and weight tying.
2. **Grouped-Query Attention (GQA).** `attention.py` includes `cached_attention_gqa()`, demonstrating awareness of modern attention variants (Llama-2/3, Mistral).
3. **Explicit causal masking.** `build_causal_mask()` and `build_variable_length_mask()` are fully implemented and used in `prompt_attention()`.
4. **Rich configuration system.** `CacheConfig` and `PageConfig` dataclasses make the code more maintainable and self-documenting.
5. **Hybrid optimization.** `HybridKVCache` combines paged + quantized, showing systems thinking.
6. **Multi-GPU strategies.** `gpu_mapping.py` covers tensor, pipeline, sequence, and expert parallelism.

### Verdict
Qwen3.6-27B is substantially more complete. It builds a nearly production-grade transformer inference stack, while GLM-5 is more of a focused KV-cache + attention demonstration.

---

## 3. Code Quality (GLM-5: 80/100, Qwen3.6-27B: 92/100)

### GLM-5
- **Strengths:** Very clean docstrings, excellent ASCII diagrams in README, consistent naming, good type hints.
- **Weaknesses:**
  - `multi_head_attention_with_cache` and `multi_head_attention_batched` are nearly identical (DRY violation).
  - `IncrementalDecoder.forward_step` conflates prefill and decode in a single function with an `is_prefill` flag, making the control flow less clear.
  - The `optimizations.py` `ChunkedPrefillCache.prefill()` has a hacky "fake q_new" using `np.random.randn`—this is acknowledged as a simplification but is still a code smell.
  - No dataclasses or config objects; parameters are passed as raw ints.
  - The `memory_analysis` functions are standalone utilities, not integrated into the cache classes.

### Qwen3.6-27B
- **Strengths:**
  - Excellent separation of concerns: `kv_cache.py` (data), `attention.py` (compute), `transformer.py` (model), `optimizations.py` (strategies), `memory_analysis.py` (analysis), `gpu_mapping.py` (hardware).
  - Dataclasses (`CacheConfig`, `PageConfig`, `ModelSpec`) make the API clean and extensible.
  - `TransformerDecoderLayer` cleanly separates `forward_prefill` and `forward_generate`.
  - `BatchedKVCache` provides a natural multi-layer coordinator.
  - Consistent use of properties (`memory_used_bytes`, `memory_allocated_bytes`).
- **Weaknesses:**
  - `QuantizedKVCache` uses per-position scales, which is inefficient and leads to the high error shown in Demo 6. The code comments acknowledge this, but the implementation still does it.
  - `PagedKVCache.append_token` requires the caller to compute `logical_block` and `offset_in_block` manually, which is error-prone. A higher-level `update()` method that hides block arithmetic would be cleaner.
  - Some functions in `gpu_mapping.py` return large dicts of strings rather than structured data.

### Verdict
Qwen3.6-27B has superior architectural layering, cleaner APIs, and better abstraction boundaries. GLM-5 is readable but less modular.

---

## 4. Depth of Analysis (GLM-5: 82/100, Qwen3.6-27B: 96/100)

### GLM-5
- Provides a memory growth table with concrete numbers for GPT-4-class models.
- FLOPs comparison (cached vs uncached) with a 109× speedup claim.
- Three optimizations are well-explained with ASCII diagrams.
- GPU mapping covers memory hierarchy, FlashAttention fusion, and CUDA pseudocode for paged attention.
- **Gaps:** No analysis of arithmetic intensity, no Tensor Core discussion, no multi-GPU strategies, no analysis of model parameter memory vs KV-cache memory, no per-token cost breakdown.

### Qwen3.6-27B
- **Memory analysis is outstanding:**
  - `memory_analysis.py` computes model parameter memory, KV-cache memory, total system memory, and KV fraction.
  - Compares 6 real-world models (Llama-2-7B/13B/70B, Llama-3-8B, Mistral-7B, GPT-4-class).
  - Computes **max context length per GPU** (RTX 4090, A100-40GB, A100-80GB, H100-80GB, H100-96GB) accounting for model weights + activations + KV cache.
  - Batch size impact analysis.
  - Per-token memory cost breakdown.
- **GPU analysis is outstanding:**
  - Arithmetic intensity calculation showing cached attention is **memory-bound** (~1.0 FLOPs/byte).
  - Tensor Core utilization analysis with compute-bound vs memory-bound time estimates.
  - FlashAttention-style cached kernel description.
  - Multi-GPU strategy comparison table.
  - Practical GPU tuning guide (streaming KV cache, small-batch optimization, continuous batching, CUDA graphs).
- **Optimization comparison:** `compare_strategies()` provides a quantitative side-by-side of naive FP16, FP32, quantized INT8, paged, and paged+quantized.

### Verdict
Qwen3.6-27B's analysis is deeper, more quantitative, and more systems-oriented. It connects the KV-cache problem to real hardware constraints and production deployment concerns.

---

## 5. Optimizations Proposed (GLM-5: 85/100, Qwen3.6-27B: 93/100)

### GLM-5
1. **Paged Attention:** Well-implemented with free-list allocation, block gathering, and page table indirection. Includes CUDA pseudocode.
2. **Chunked Prefill:** Implemented as a wrapper around `KVCache`. Reduces peak attention memory from O(S²) to O(C×S). The implementation has a hacky fake query but the concept is correct.
3. **Cache Quantization (INT8/INT4):** Implements per-token quantization with scale + zero-point. Supports INT4 packing (2 values per byte). Good demonstration of the concept.

### Qwen3.6-27B
1. **Paged Attention:** Implemented with `PageConfig` dataclass, physical page pool, page tables, and utilization tracking. Slightly more structured than GLM-5.
2. **Quantization:** Per-channel INT8 with affine transform (`x ≈ scale * q + zero`). Acknowledges the overhead of per-position scales and notes that production should use shared scales.
3. **Chunked Prefill:** Computes causal attention in chunks with explicit causal masking per chunk. Includes `peak_memory_comparison()` function.
4. **Hybrid (Paged + Quantized):** `HybridKVCache` combines both strategies, showing systems-level thinking about composing optimizations.
5. **Optimization comparison table:** Quantitative comparison of all strategies with per-layer and total memory numbers.

### Comparison
- **GLM-5's quantization is more sophisticated** (supports INT4 packing, per-token scales + zero-points). Qwen3.6-27B only does INT8 and admits its per-position approach is inefficient.
- **Qwen3.6-27B's chunked prefill is more rigorous** (explicit causal mask per chunk, peak memory comparison function).
- **Qwen3.6-27B wins on systems thinking** with the hybrid cache and the quantitative comparison framework.
- Both meet the "at least two optimizations" requirement comfortably.

### Verdict
Qwen3.6-27B edges ahead due to the hybrid approach and quantitative comparison framework, though GLM-5's INT4 support is a nice touch.

---

## 6. GPU Mapping Explanation (GLM-5: 80/100, Qwen3.6-27B: 95/100)

### GLM-5
- Memory hierarchy diagram (registers → shared memory → HBM).
- Kernel mapping table (CPU op → GPU kernel).
- FlashAttention fusion explanation with online softmax algorithm.
- CUDA pseudocode for paged attention kernel.
- Good but somewhat high-level; lacks concrete performance numbers.

### Qwen3.6-27B
- **Memory hierarchy** with concrete sizes and latencies (H100: 166 KB shared mem, 50 MB L2, 80 GB HBM, 3.35 TB/s bandwidth).
- **Cached attention kernel design** with grid/block dimensions, shared memory usage breakdown, and optimization strategies.
- **Tensor Core analysis** with actual FLOPs, memory traffic, arithmetic intensity, compute-bound time, memory-bound time, and bottleneck classification.
- **FlashAttention-style cached kernel** description with online softmax and HBM traffic reduction claims.
- **Multi-GPU strategies** with detailed descriptions of tensor/pipeline/sequence/expert parallelism and their KV-cache implications.
- **Practical GPU tuning guide** covering streaming KV cache, small-batch optimization, continuous batching, KV-cache quantization on GPU, and CUDA graphs.
- Key insight: **"Generation is memory-bound"** — 1.0 FLOPs/byte intensity, bottleneck is HBM bandwidth.

### Verdict
Qwen3.6-27B's GPU mapping is significantly more detailed, quantitative, and actionable. It reads like a systems performance analysis rather than a conceptual mapping.

---

## 7. Tests and Demos (GLM-5: 82/100, Qwen3.6-27B: 90/100)

### GLM-5
- **8 tests**, all passing:
  1. Basic cache update/retrieval
  2. Attention correctness (cached vs non-cached)
  3. Variable sequence lengths
  4. Incremental decoder end-to-end
  5. Paged cache
  6. Quantized cache (INT8 + INT4)
  7. Memory growth analysis
  8. FLOPs analysis
- Tests use `assert` and `np.testing.assert_allclose`.
- Good coverage of core functionality.
- **Weakness:** No demo of the full transformer in action (prefill + multi-step generation with sampling). Test 4 does a minimal decode loop but without causal masking or real sampling.

### Qwen3.6-27B
- **10 demos**, all completing:
  1. Basic KV cache operations
  2. Cached attention computation
  3. Full transformer (prefill + generation with temperature/top-k sampling)
  4. Variable-length batching
  5. Paged attention
  6. Quantized cache
  7. Chunked prefill (with correctness check against full attention)
  8. Optimization comparison (quantitative table)
  9. Memory analysis (model comparison, growth curves, GPU limits)
  10. GPU Tensor Core analysis (arithmetic intensity, bound classification)
- Demo 3 is particularly strong: it shows a full transformer prefill + 5-step generation with temperature scaling and top-k filtering.
- Demo 9 prints a comprehensive memory report with real model names and GPU limits.
- **Weakness:** Demo 6 exposes high quantization error without a clear assertion boundary. The demo completes but prints a concerning error value.

### Verdict
Qwen3.6-27B has more demos, broader coverage, and more impressive end-to-end demonstrations. GLM-5's tests are more rigorous in their assertions (especially the quantized cache), but narrower in scope.

---

## 8. Head-to-Head: What Each Did Well

### GLM-5 (GLM-5 KV/) — Strengths
1. **Excellent documentation.** The README.md is outstanding—clear ASCII diagrams, well-structured sections, and pedagogical explanations of the BHSD layout, update logic, and attention computation.
2. **INT4 quantization.** GLM-5 is the only one to implement INT4 packing (2 values per byte), showing attention to extreme compression scenarios.
3. **Clean pedagogical style.** The code is very readable and well-commented, making it easy to follow for someone learning KV-caching.
4. **Strong correctness testing.** The attention correctness test (cached vs non-cached) is rigorous, and the quantized cache has bounded error assertions.
5. **FLOPs analysis.** The explicit FLOPs comparison with speedup factor is a nice touch.

### GLM-5 — Weaknesses
1. **Incomplete transformer.** No MLP, no positional encoding, no causal masking in prefill.
2. **Limited batched masking.** The "batched" attention function doesn't actually demonstrate true batched tensor masking.
3. **Less quantitative analysis.** No arithmetic intensity, no Tensor Core discussion, no per-GPU context limits.
4. **Simpler GPU mapping.** Good conceptual coverage but lacks concrete numbers and actionable tuning advice.
5. **Code duplication.** The two attention functions are nearly identical.

### Qwen3.6-27B (Qwen3.6-27B KV/) — Strengths
1. **Full transformer implementation.** Complete decoder with LayerNorm, MLP, residuals, positional encoding, and weight tying. This is a huge completeness win.
2. **GQA support.** Includes grouped-query attention, showing awareness of modern architectures.
3. **Outstanding systems analysis.** Memory growth with real models, max context per GPU, arithmetic intensity, Tensor Core analysis, multi-GPU strategies, and a practical tuning guide.
4. **Quantitative optimization comparison.** Side-by-side memory costs for all strategies.
5. **Clean architecture.** Excellent separation of concerns with dataclasses and dedicated modules.
6. **Rich demo suite.** 10 demos covering every component, including a full generation loop with sampling.
7. **Hybrid optimization.** Combines paged + quantized, demonstrating systems-level thinking.

### Qwen3.6-27B — Weaknesses
1. **Quantized cache error.** Demo 6 shows a max absolute error of ~5.1 and relative error of ~1.7 for one token. While acknowledged, this is a real implementation weakness.
2. **Per-position scales in quantization.** The `QuantizedKVCache` uses per-position scales, which is inefficient. The code comments note this but the implementation doesn't fix it.
3. **Paged cache API is low-level.** `append_token` requires manual block/offset calculation. A higher-level `update()` would be more ergonomic.
4. **Some GPU mapping functions return string dicts.** `describe_cached_attention_kernel()` returns a large nested dict of strings rather than structured data, making it less useful for programmatic analysis.

---

## 9. Final Scores and Justification

### GLM-5 (GLM-5 KV/): 82/100

GLM-5 is a **solid, well-documented, pedagogical implementation** of KV-caching. It gets the core concepts right, provides three meaningful optimizations, and has good test coverage. However, it falls short on completeness—there is no full transformer layer, no causal masking, no positional encoding, and limited batched masking. The analysis is good but not as deep or quantitative as Qwen3.6-27B. The GPU mapping is conceptual rather than actionable. This is a good "learning" implementation but not a production-oriented one.

**Breakdown:**
- Correctness: 95/100
- Completeness: 78/100
- Code Quality: 80/100
- Depth of Analysis: 82/100
- Optimizations: 85/100
- GPU Mapping: 80/100
- Tests/Demos: 82/100
- **Overall: 82/100**

### Qwen3.6-27B (Qwen3.6-27B KV/): 94/100

Qwen3.6-27B is a **near-production-grade implementation** of a KV-cache system for transformer inference. It provides a complete transformer decoder, supports GQA, delivers outstanding quantitative analysis (memory growth, GPU limits, arithmetic intensity, Tensor Core utilization), and includes a comprehensive GPU tuning guide. The demo suite is rich and covers every component. The architecture is clean and modular. The main weaknesses are the high quantization error in Demo 6 (acknowledged but not fixed) and some API rough edges in the paged cache. These are relatively minor issues in an otherwise exceptional implementation.

**Breakdown:**
- Correctness: 95/100
- Completeness: 95/100
- Code Quality: 92/100
- Depth of Analysis: 96/100
- Optimizations: 93/100
- GPU Mapping: 95/100
- Tests/Demos: 90/100
- **Overall: 94/100**

---

## 10. Winner and Margin

**Winner: Qwen3.6-27B (Qwen3.6-27B KV/)**

**Margin: ~12 points** (94 vs 82)

Qwen3.6-27B wins decisively on **completeness**, **depth of analysis**, and **GPU mapping**. It builds a full transformer, analyzes real hardware constraints, and provides actionable tuning guidance. GLM-5 is a worthy competitor with excellent documentation and a nice INT4 quantization implementation, but it is narrower in scope and less systems-oriented. The gap is primarily in architectural completeness and analytical depth, not in fundamental correctness.

---

*Analysis conducted by reading all source files, READMEs, PROMPT.md, FINAL.md, and running all tests/demos in both folders. No files in the original folders were modified.*
