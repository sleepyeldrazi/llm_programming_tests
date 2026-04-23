# Head-to-Head Analysis: KV-Cache System for Autoregressive Transformer Inference

**Task:** Implement an efficient KV-cache system for autoregressive transformer inference from scratch.
**Date:** 2026-04-23
**Analyst:** pi coding agent

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [MiniMax-M2.7: `MiniMax-M2.7`](#2-model-a-minimax-m2.7kv)
3. [Qwen3.6-27B: `Qwen3.6-27B`](#3-model-b-qwen36kv)
4. [Detailed Scoring](#4-detailed-scoring)
5. [Head-to-Head Comparison](#5-head-to-head-comparison)
6. [Final Verdict](#6-final-verdict)

---

## 1. Executive Summary

Both implementations satisfy the core requirements of the prompt: incremental decoding, KV-cache reuse, multi-head attention, batching support, memory analysis, optimization proposals, and GPU execution mapping. However, **Qwen3.6-27B (`Qwen3.6-27B`) is the clear winner** with a decisive margin. It delivers a **modular, well-tested, and rigorously validated codebase** with 10 passing end-to-end demos, precise numerical correctness checks, and production-grade analysis. MiniMax-M2.7 is a **single-file monolith** with broader conceptual scope but weaker execution, no automated tests, and several correctness issues in its attention masking and batching logic.

| Dimension | MiniMax-M2.7 Score | Qwen3.6-27B Score |
|-----------|--------------|---------------|
| Correctness | 55 | 92 |
| Completeness | 75 | 95 |
| Code Quality | 60 | 88 |
| Depth of Analysis | 78 | 90 |
| Optimizations Proposed | 72 | 90 |
| GPU Mapping Explanation | 75 | 88 |
| Tests / Demos | 30 | 95 |
| **Overall** | **64** | **91** |

---

## 2. MiniMax-M2.7: `MiniMax-M2.7`

### 2.1 Files
- `kv_cache.py` — Single 1,720-line monolithic file containing everything
- `FINAL.md` — Summary document
- `PROMPT.md` — Identical prompt

### 2.2 What It Does Well

1. **Conceptual Breadth:** MiniMax-M2.7 covers an impressive range of topics in one file:
   - Multiple memory formats (BHSD, BSHD, PAGED, HBSD)
   - Both paged (`PagedKVCache`) and flat (`FlatKVCache`) cache implementations
   - Full transformer block with pre-norm, FFN, and residual connections
   - Batched inference engine with `BatchElement` tracking
   - Memory analyzer with formulas and latency estimates
   - GPU execution mapper with CUDA kernel pseudocode
   - Five optimization strategies (paged attention, chunked attention, quantization, sparse KV, speculative decoding)

2. **Data Structure Variety:** It implements two distinct cache backends (paged and flat), which shows understanding of trade-offs.

3. **Extensive ASCII Diagrams:** The code is heavily annotated with visual diagrams explaining memory layouts, execution pipelines, and GPU hierarchies.

4. **GPU Kernel Pseudocode:** Includes actual CUDA-style pseudocode for `kvcache_update` and `attention_with_cache` kernels.

### 2.3 Weaknesses

1. **No Automated Tests:** The only "test" is a 3-step hardcoded decode in `run_demo()` with no assertions, no numerical validation, and no edge-case coverage. There is no way to verify correctness systematically.

2. **Attention Masking Bug:** The causal mask construction is incorrect:
   ```python
   mask = np.triu(np.ones((seq_len, total_len), dtype=np.float32), k=1 - seq_len)
   ```
   This produces a mask where the lower-left triangle is 1s (masked) and upper-right is 0s (unmasked) — the **opposite** of causal masking. The correct causal mask should mask the **upper triangle** (future positions). This is a critical correctness bug.

3. **KV Cache Update Bug in Batched Setting:** In `BatchedInferenceEngine.step_inference()`, the engine iterates over batch elements one at a time and calls `self.model.forward()` with `batch_idx=elem.batch_idx`, but `TransformerBlockStack.forward()` ignores `batch_idx` entirely — it always uses the same shared `self.kv_cache` dictionary keyed by `layer_idx`, not by batch element. This means **all batch elements share the same KV cache**, which is fundamentally broken for batched inference with different sequences.

4. **No Variable-Length Masking:** While the prompt requires "batching with variable sequence lengths," MiniMax-M2.7 does not implement per-sequence length masking in its attention computation. The `BatchElement` class tracks lengths but they are never used to mask padded positions.

5. **Monolithic Architecture:** Everything is crammed into a single 1,720-line file. This hurts readability, maintainability, and makes it impossible to import components independently.

6. **Prefill Does Not Store KV Cache Correctly:** In `KVCacheAwareGenerator.prefill()`, the model forward is called but the returned KV tensors are never stored into the `FlatKVCache` or `PagedKVCache` data structures. The prefill only populates the in-memory `self.kv_cache` dict inside `TransformerBlockStack`, not the persistent cache.

7. **Weak Quantization Analysis:** The quantization demo only shows format comparisons (FP32→FP16→INT8) without any actual quantization/dequantization implementation or error analysis.

8. **Chunked Attention Is Only Described, Not Implemented:** The "chunked attention" optimization is documented in comments with no runnable code.

9. **Memory Analysis Is High-Level:** The memory analyzer provides formulas and tables but lacks concrete model comparisons (e.g., Llama-7B vs GPT-4) and GPU-specific context limits.

10. **GPU Mapping Is Mostly Descriptive:** While it includes CUDA pseudocode, the analysis lacks quantitative metrics like arithmetic intensity, memory-bound vs compute-bound classification, or concrete kernel tiling parameters.

---

## 3. Qwen3.6-27B: `Qwen3.6-27B`

### 3.1 Files
- `kv_cache.py` — Core data structures (`KVCache`, `BatchedKVCache`)
- `attention.py` — Attention computation (standard, cached, masked, GQA)
- `transformer.py` — Full transformer decoder with prefill + generation
- `optimizations.py` — Paged attention, quantization, chunked prefill
- `memory_analysis.py` — Memory growth formulas, model comparisons, GPU limits
- `gpu_mapping.py` — GPU kernel design, Tensor Core analysis, multi-GPU strategies
- `demo.py` — 10 end-to-end demos with assertions
- `README.md` — Comprehensive documentation
- `FINAL.md` — Summary of passing demos

### 3.2 What It Does Well

1. **Modular Architecture:** Seven focused files, each with a single responsibility. Clean imports, clear separation of concerns. This is production-quality structure.

2. **10 Passing End-to-End Demos:** Every component is exercised and validated:
   - Demo 1: Basic cache ops with shape assertions
   - Demo 2: Cached attention **numerically verified** against manual computation (`diff < 1e-5`)
   - Demo 3: Full transformer prefill + generation with variable-length batching
   - Demo 4: Variable-length batching with per-sequence attention
   - Demo 5: Paged attention with block allocation and page table verification
   - Demo 6: Quantized cache with error measurement
   - Demo 7: Chunked prefill **numerically verified** against full attention (`diff = 4.56e-10`)
   - Demo 8: Side-by-side optimization comparison
   - Demo 9: Memory analysis with real model specs (Llama-2/3, Mistral, GPT-4)
   - Demo 10: GPU Tensor Core analysis with arithmetic intensity and bound classification

3. **Correct Attention Implementation:**
   - `build_causal_mask()` correctly masks the upper triangle with `-inf`
   - `build_variable_length_mask()` handles per-batch-item lengths with both causal and length masking
   - `cached_attention()` correctly notes that causality is implicit during generation (cache only contains past tokens)
   - `prompt_attention()` correctly applies causal masking during prefill

4. **Proper Prefill/Decode Separation:**
   - `TransformerDecoderLayer.forward_prefill()` processes full prompts, stores K/V in cache, and applies causal masking
   - `TransformerDecoderLayer.forward_generate()` processes single tokens, appends K/V to cache, and uses cached attention
   - `TransformerDecoder.prefill()` and `.generate_step()` orchestrate the phases cleanly

5. **Variable-Length Batching Is Real:** The `lengths` parameter is threaded through prefill and generation, and `build_variable_length_mask()` creates proper combined causal + length masks.

6. **Working Quantization Implementation:** `QuantizedKVCache` implements actual per-channel int8 quantization with affine transform (`x ≈ scale * q + zero`). It honestly reports that per-position scales have high overhead and suggests shared per-channel scales for production.

7. **Working Chunked Prefill:** `ChunkedPrefill.compute_attention_chunked()` is a real implementation that processes prompts in chunks, applies causal masks per chunk, and accumulates results. It is numerically verified to match full attention.

8. **Working Paged Attention:** `PagedKVCache` implements page tables, free lists, physical page pools, and on-demand allocation. Demo 5 verifies block allocation and memory utilization.

9. **Rich Memory Analysis:**
   - Compares 6 real model architectures (Llama-2 7B/13B/70B, Llama-3 8B, Mistral-7B, GPT-4-class)
   - Computes max context lengths per GPU (RTX 4090, A100-40/80GB, H100-80/96GB)
   - Shows KV cache fraction of total memory at different sequence lengths
   - Analyzes batch size impact with concrete numbers

10. **Quantitative GPU Mapping:**
    - Computes arithmetic intensity (FLOPs/byte) for different configs
    - Classifies all configs as **memory-bound** (critical insight)
    - Describes kernel tiling with concrete sizes (BLOCK=32, shared memory = ~16-20 KB)
    - Includes FlashAttention-style online softmax algorithm
    - Covers multi-GPU strategies (tensor, pipeline, sequence, expert parallelism)
    - Provides practical tuning guide (CUDA graphs, continuous batching, INT8 Tensor Cores)

11. **Group Query Attention (GQA):** Implements `cached_attention_gqa()` showing awareness of modern optimizations beyond standard MHA.

12. **Honest Self-Critique:** The quantization demo explicitly notes that its per-position scale approach has high overhead and suggests the production approach (shared per-channel scales). This shows intellectual honesty.

### 3.3 Weaknesses

1. **Quantized Cache Has Negative Memory Savings in Demo:** Due to per-position scales stored in fp16, the `QuantizedKVCache` actually uses **more** memory than fp16 in the demo. The code acknowledges this and explains the production fix, but the implementation itself is not optimized.

2. **Paged Attention Gather Is Inefficient:** `PagedKVCache.get_sequence()` iterates over blocks and copies them one at a time. In a real GPU kernel, this would be a gather operation, but the NumPy implementation is O(num_blocks) with Python-level looping.

3. **No Speculative Decoding:** While MiniMax-M2.7 at least mentions speculative decoding in its optimization list, Qwen3.6-27B does not cover it at all.

4. **No Sliding Window Attention:** Qwen3.6-27B implements GQA but does not implement sliding window attention (a key optimization for very long contexts in models like Mistral).

5. **GQA Is Not Integrated into Transformer:** The `cached_attention_gqa()` function exists in `attention.py` but is not used in `TransformerDecoderLayer` or `TransformerDecoder`.

---

## 4. Detailed Scoring

### 4.1 Correctness (0-100)

| Aspect | MiniMax-M2.7 | Qwen3.6-27B |
|--------|---------|---------|
| Attention masking | **Buggy** — causal mask is inverted | **Correct** — proper causal + length masks |
| KV cache update | **Buggy** — batched cache is shared across all elements | **Correct** — per-layer, per-batch caches |
| Prefill cache storage | **Buggy** — prefill KV not stored in persistent cache | **Correct** — `prompt_attention()` stores all tokens |
| Numerical validation | None | 10 demos with assertions |
| Variable-length batching | Described but not correctly implemented | Fully working with masks |
| **Score** | **55** | **92** |

**MiniMax-M2.7 loses 45 points** due to the inverted causal mask (critical), shared batched cache (critical), and missing prefill cache storage (major). These are not edge cases — they are fundamental to the task.

**Qwen3.6-27B loses 8 points** for the quantized cache overhead issue (minor, acknowledged) and the lack of GQA integration (minor).

### 4.2 Completeness (0-100)

| Requirement | MiniMax-M2.7 | Qwen3.6-27B |
|-------------|---------|---------|
| Incremental decoding | ✓ | ✓ |
| Avoid recomputing attention | ✓ (conceptually) | ✓ (working) |
| Multi-head attention | ✓ | ✓ |
| Batching with variable lengths | Partial (broken) | ✓ |
| Data structure layout | ✓ (4 formats) | ✓ (clearly documented) |
| Update logic per step | ✓ | ✓ |
| Attention computation with cache | ✓ (buggy mask) | ✓ |
| Memory growth analysis | ✓ (formulas) | ✓ (formulas + models + GPUs) |
| ≥2 optimizations proposed | ✓ (5 listed, 2 implemented) | ✓ (3 implemented + comparisons) |
| GPU execution mapping | ✓ (descriptive) | ✓ (quantitative + kernel design) |
| **Score** | **75** | **95** |

MiniMax-M2.7 is incomplete on variable-length batching (the requirement is not met due to the shared cache bug) and its optimizations are partially documented rather than implemented.

### 4.3 Code Quality (0-100)

| Aspect | MiniMax-M2.7 | Qwen3.6-27B |
|--------|---------|---------|
| Modularity | Single 1,720-line file | 7 focused files |
| Readability | Dense, diagram-heavy | Clean, well-commented |
| Type hints | Present but inconsistent | Consistent and thorough |
| Naming | Generally good | Excellent |
| Docstrings | Extensive | Concise and precise |
| Reusability | Poor (monolith) | Good (modular imports) |
| **Score** | **60** | **88** |

MiniMax-M2.7's single-file approach makes it difficult to navigate and impossible to import components selectively. Qwen3.6-27B's modular structure is a clear best practice.

### 4.4 Depth of Analysis (0-100)

| Aspect | MiniMax-M2.7 | Qwen3.6-27B |
|--------|---------|---------|
| Memory formulas | ✓ | ✓ (more detailed) |
| Model-specific analysis | None | 6 real models |
| GPU-specific limits | Generic | Per-GPU context limits |
| Arithmetic intensity | Not computed | Computed and classified |
| Multi-GPU strategies | Listed | Detailed with KV cache impact |
| Practical tuning | Limited | Comprehensive guide |
| **Score** | **78** | **90** |

Both provide good analysis, but Qwen3.6-27B grounds everything in concrete numbers (real models, real GPUs, real FLOPs/byte ratios).

### 4.5 Optimizations Proposed (0-100)

| Optimization | MiniMax-M2.7 | Qwen3.6-27B |
|--------------|---------|---------|
| Paged attention | Described + partial implementation | Fully implemented + tested |
| Quantization | Described only | Implemented + error measured |
| Chunked attention | Described only | Implemented + numerically verified |
| Sparse KV / token selection | Described | Not covered |
| Speculative decoding | Described | Not covered |
| GQA | Not covered | Implemented (not integrated) |
| Side-by-side comparison | No | Yes (5 strategies) |
| **Score** | **72** | **90** |

MiniMax-M2.7 covers more optimization *ideas* (5 vs 3) but only implements 1 (paged) partially. Qwen3.6-27B implements 3 fully with tests and comparisons. Quality over quantity.

### 4.6 GPU Mapping Explanation (0-100)

| Aspect | MiniMax-M2.7 | Qwen3.6-27B |
|--------|---------|---------|
| Memory hierarchy | ✓ (ASCII diagram) | ✓ (table) |
| CUDA kernel pseudocode | ✓ | ✓ (more detailed) |
| Thread block design | Brief | Detailed with sizes |
| Tensor Core analysis | Mentioned | Quantified (FLOPs, intensity, bounds) |
| FlashAttention adaptation | Mentioned | Algorithm described |
| Multi-GPU strategies | Listed | Detailed per-strategy |
| Practical tuning | Limited | 5 concrete recommendations |
| **Score** | **75** | **88** |

MiniMax-M2.7 has CUDA pseudocode; Qwen3.6-27B has quantitative analysis. Both are good, but Qwen3.6-27B's arithmetic intensity analysis and bound classification are more insightful.

### 4.7 Tests / Demos (0-100)

| Aspect | MiniMax-M2.7 | Qwen3.6-27B |
|--------|---------|---------|
| Number of demos | 1 (hardcoded) | 10 (comprehensive) |
| Assertions / validation | None | Numerical diff checks |
| Edge cases covered | None | Variable lengths, padding, quantization error |
| Integration test | Partial | Full prefill → generate pipeline |
| **Score** | **30** | **95** |

This is the biggest gap. Qwen3.6-27B's 10 passing demos with numerical validation provide confidence that the system works. MiniMax-M2.7 has no systematic validation.

---

## 5. Head-to-Head Comparison

### What Each Did Well

**MiniMax-M2.7 Strengths:**
- Broader conceptual coverage (5 optimization ideas vs 3)
- Multiple memory format enums (BHSD, BSHD, PAGED, HBSD)
- Both paged and flat cache implementations in one file
- Includes speculative decoding in optimization list
- CUDA kernel pseudocode is more extensive
- `MemoryFormat` enum shows awareness of layout trade-offs

**Qwen3.6-27B Strengths:**
- Everything is tested and numerically validated
- Modular, maintainable codebase
- Correct attention masking (causal + variable length)
- Proper prefill/decode phase separation
- Working implementations of 3 optimizations (paged, quantized, chunked)
- Concrete model and GPU analysis with real numbers
- Quantitative GPU performance characterization (memory-bound classification)
- GQA implementation (modern architecture awareness)
- Honest self-critique of quantization overhead
- Excellent documentation (README.md is comprehensive)

### Weaknesses Comparison

**MiniMax-M2.7 Critical Issues:**
1. **Inverted causal mask** — attention attends to future tokens instead of past
2. **Shared batched KV cache** — all batch elements overwrite each other's cache
3. **No systematic testing** — correctness is assumed, not verified
4. **Monolithic file** — unmaintainable at scale

**Qwen3.6-27B Minor Issues:**
1. Quantized cache has overhead in current implementation (acknowledged)
2. GQA is not wired into the transformer
3. No speculative decoding coverage
4. No sliding window attention

### Who Won and By How Much

**Qwen3.6-27B wins decisively.**

| Metric | MiniMax-M2.7 | Qwen3.6-27B | Delta |
|--------|---------|---------|-------|
| Overall Score | 64 | 91 | **+27** |

The margin is large and justified:
- Qwen3.6-27B is **correct** where MiniMax-M2.7 has fundamental bugs
- Qwen3.6-27B is **tested** where MiniMax-M2.7 has no validation
- Qwen3.6-27B is **modular** where MiniMax-M2.7 is a monolith
- Qwen3.6-27B's analysis is **quantitative** where MiniMax-M2.7's is descriptive

MiniMax-M2.7 shows broader *familiarity* with concepts (more optimization ideas, more memory formats) but Qwen3.6-27B demonstrates deeper *understanding* and *execution* (working code, passing tests, numerical validation). In engineering, correctness and validation trump conceptual breadth.

---

## 6. Final Verdict

### MiniMax-M2.7: 64/100 — "Conceptually Broad, Executionally Weak"

MiniMax-M2.7 demonstrates familiarity with a wide range of KV-cache concepts and writes extensive documentation. However, it suffers from critical correctness bugs (inverted causal mask, broken batched caching), lacks any systematic testing, and crams everything into an unmaintainable monolith. The implementation does not reliably meet the prompt's requirements for correct incremental decoding or variable-length batching. It reads like a knowledgeable engineer's first draft — full of good ideas but not yet debugged or validated.

### Qwen3.6-27B: 91/100 — "Production-Grade, Rigorously Validated"

Qwen3.6-27B delivers a modular, correct, and thoroughly tested KV-cache system. Every component has a dedicated file, every demo passes with numerical validation, and the analysis is grounded in real models and GPUs. The attention masking is correct, the prefill/decode separation is clean, and the optimizations are actually implemented and verified. The README alone is a better technical document than MiniMax-M2.7's entire output. This is the work of an engineer who understands that **correctness and testing are not optional**.

### Recommendation

If you need a KV-cache system to study, extend, or adapt: **use Qwen3.6-27B**. It is correct, tested, modular, and well-documented. MiniMax-M2.7 may be useful as a supplementary reference for additional optimization ideas (speculative decoding, sliding window, sparse KV), but its code should not be used without significant bug fixes.

---

*Analysis completed by pi coding agent. Both implementations were read in full, executed, and evaluated against the original prompt requirements.*
