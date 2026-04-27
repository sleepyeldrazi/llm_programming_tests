# 2-Way Head-to-Head Comparisons

## GLM-5 vs MiniMax-M2.7

### Task 1: Backward Layer Norm

| Criteria | GLM-5 | MiniMax-M2.7 | Edge |
|----------|-------|-------------|------|
| Lines of code | 275 | 1148 | GLM (more concise) |
| Gradient correctness | PASS (~1e-10 rel) | PASS (~1e-10 rel) | Tie |
| Cache efficiency | 3 items | 12 items (9 redundant) | **GLM** |
| Numerical stability discussion | 5 failure modes | Buried in code comments | **GLM** |
| GPU fusion detail | Backward only, 4 steps | Forward + backward, full CUDA pseudocode | **MiniMax** |
| Edge case testing | None | None (spot-check only) | Tie |
| Benchmark | None | 4 shape configs | **MiniMax** |
| Spot-check for large tensors | No | Yes (>100k elements) | **MiniMax** |

**Winner: GLM-5** (cleaner, more correct cache design; MiniMax's GPU pseudocode is better but the cache bloat is a fundamental flaw)

### Task 2: Fused Softmax+Top-K

| Criteria | GLM-5 | MiniMax-M2.7 | Edge |
|----------|-------|-------------|------|
| Algorithm | Online softmax (single pass) | 2-pass (max → sum → topk) | **GLM** |
| CUDA correctness | Compilable, correct | **Has bugs** (launch bounds, shared mem layout, stack overflow) | **GLM** |
| K limit | ≤32 | ≤100 | MiniMax |
| Warp-level | Butterfly shuffle reductions | Butterfly shuffle reductions | Tie |
| Top-K data structure | Register sorted array | Register sorted array | Tie |
| Cross-warp merge | Shared memory, serial | Shared memory, thread 0 only | Tie |
| Documentation | DESIGN.md (9 sections) | Inline ASCII diagrams (comprehensive) | **GLM** |
| Bandwidth analysis | AI=1.5, 3× speedup | AI=0.8, 4× speedup | Tie (both correct) |
| Production readiness | Medium | Low (bugs) | **GLM** |

**Winner: GLM-5** (MiniMax's CUDA has real bugs that prevent compilation/correctness; GLM's online algorithm is genuinely superior)

### Task 3: KV-Cache

| Criteria | GLM-5 | MiniMax-M2.7 | Edge |
|----------|-------|-------------|------|
| Core cache design | Clean, correct | Over-complicated, format mismatch | **GLM** |
| Memory layout | BHSD (good) | Multiple formats (good concept, messy impl) | Tie |
| Variable-length batching | Working | Attempted but flawed | **GLM** |
| Paged attention | Working, free-list | Working, block allocator | Tie |
| Quantization | INT8/INT4 working | Not implemented separately | **GLM** |
| Chunked prefill | Implemented (partial) | Mentioned but not implemented | **GLM** |
| Tests | 8 tests, ALL PASS | 0 tests | **GLM** |
| Memory analysis | Tables + FLOPs comparison | MemoryAnalyzer class (estimated latency) | Tie |
| Code organization | 3 files (core + opt + test) | 1 monolithic 1720-line file | **GLM** |
| Architecture issues | None significant | Format mismatch between stack and attention | **GLM** |

**Winner: GLM-5** (MiniMax's implementation has a critical format mismatch bug and no tests; GLM's is correct and well-tested)

### GLM-5 vs MiniMax-M2.7 Overall: **GLM-5 wins 3-0**

---

## MiniMax-M2.7 vs Qwen3-6

### Task 1: Backward Layer Norm

| Criteria | MiniMax-M2.7 | Qwen3-6 | Edge |
|----------|-------------|---------|------|
| Lines of code | 1148 (monolithic) | 294 + 113 + 150 = 557 (3 files) | **Qwen** |
| Gradient check | PASS | PASS (5× lower rel error) | **Qwen** |
| Cache minimality | 12 items (bloated) | 4 items (optimal) | **Qwen** |
| Edge cases | None | 5 distinct edge cases | **Qwen** |
| Cross-verification | None | Alternative derivation check | **Qwen** |
| Stability demo | None | Two-pass vs naive variance demo | **Qwen** |
| GPU fusion | Full CUDA pseudocode | Both forward/backward, memory traffic table | **Qwen** |
| Benchmark | 4 configs | 8 configs + stability demo | **Qwen** |
| Memory analysis | Per-operation FLOPs table | N-based FLOPs estimate | Tie |

**Winner: Qwen3-6** (decisive — better in every dimension)

### Task 2: Fused Softmax+Top-K

| Criteria | MiniMax-M2.7 | Qwen3-6 | Edge |
|----------|-------------|---------|------|
| CUDA correctness | Has bugs | Both v1 and v2 compilable | **Qwen** |
| Algorithm | 2-pass | 2-pass (v1), semi-online (v2) | Tie |
| K support | ≤100 (if/else chain) | ≤256 (template, 5 instantiations) | **Qwen** |
| Vectorized loads | No | float4 in v2 | **Qwen** |
| Top-K structure | Register array | Shared heap (O(log K) insert) | **Qwen** |
| Warp merge | Thread 0 serial | Warp-leader serial + barriers | **Qwen** |
| Cross-warp merge | Shared mem, thread 0 | Warp-level staging → shared heap | **Qwen** |
| Documentation quality | Excellent ASCII diagrams | ANALYSIS.md + inline comments | Tie |
| Benchmark harness | None | benchmark.cu | **Qwen** |
| Multiple versions | No | v1 + v2 optimized | **Qwen** |

**Winner: Qwen3-6** (MiniMax has bugs; Qwen has two correct kernels with optimization)

### Task 3: KV-Cache

| Criteria | MiniMax-M2.7 | Qwen3-6 | Edge |
|----------|-------------|---------|------|
| File count | 1 | 8 | **Qwen** |
| Lines of code | 1720 (monolithic) | 205 + 234 + 390 + ... = ~1200 (modular) | **Qwen** |
| Architecture bugs | Format mismatch in attn/cache stack | None significant | **Qwen** |
| Tests/Demos | 0 | 10 demos, ALL PASS | **Qwen** |
| Variable-length batching | Broken (engine logic error) | Working, 4 different lengths | **Qwen** |
| Paged attention | Working but fragmented | Working with page tables | Tie |
| Quantization | Not implemented | Implemented, notes overhead honestly | **Qwen** |
| Memory analysis | MemoryAnalyzer class | ModelSpec + find_max_context + 6 real models | **Qwen** |
| Attention variants | Standard only | Standard + GQA + MQA | **Qwen** |
| GPU mapping | Basic | Dedicated gpu_mapping.py with Tensor Cores | **Qwen** |
| Chunked prefill | Mentioned | Full implementation, matches full attn to 4.5e-10 | **Qwen** |
| Model specs | None | Llama-2-7B/13B/70B, Llama-3-8B, Mistral-7B, GPT-4-class | **Qwen** |
| Max context calculator | Estimated latency only | Per-GPU max context (RTX 4090→H100) | **Qwen** |

**Winner: Qwen3-6** (decisive — functionally correct where MiniMax has bugs, 10× more thorough)

### MiniMax-M2.7 vs Qwen3-6 Overall: **Qwen3-6 wins 3-0**

---

## Qwen3-6 vs GLM-5

This is the closest matchup. Both are correct and well-engineered.

### Task 1: Backward Layer Norm

| Criteria | Qwen3-6 | GLM-5 | Edge |
|----------|---------|-------|------|
| Code size | 557 lines, 3 files | 275 lines, 1 file | **GLM** (more concise) |
| Gradient precision | 5.04e-11 (dx) | 9.74e-11 (dx) | **Qwen** (2× better) |
| Cache items | 4 (x_hat, std_inv, gamma, D) | 3 (xhat, rstd, gamma) | **GLM** (one less!) |
| Edge cases | 5 tested (zero, large mean, D=1, D=1024, norm sanity) | 0 tested | **Qwen** |
| Formula cross-verify | Alternative derivation: matches to 1e-10 | Not done | **Qwen** |
| Stability demo | 2-pass vs naive variance (offset 1e10) | Prose discussion only | **Qwen** |
| GPU fusion scope | Forward + backward kernels, memory traffic | Backward kernel only, shared mem layout | **Qwen** |
| Complexity format | Concise formula (N-based) | Prose-based | Tie |
| Derivations | Shown in docstring | Shown in docstring | Tie |
| Speed (full grad check) | Very slow (element-wise, no spot-check) | Very slow (element-wise, no spot-check) | Tie |

**Winner: Qwen3-6** (slightly better precision, edge cases, cross-verification, broader GPU fusion scope)

### Task 2: Fused Softmax+Top-K

| Criteria | Qwen3-6 | GLM-5 | Edge |
|----------|---------|-------|------|
| Algorithm elegance | 2-pass (practical) | Online single-pass (elegant) | **GLM** |
| Memory reads | 3 × V (max+sum+softmax) | 1 × V (online pass) | **GLM** |
| K support | Up to 256 | Up to 32 | **Qwen** |
| Top-K structure | Shared heap (O(log K)) | Register array (O(K)) | **Qwen** (for K>32) |
| Vectorization | float4 in v2 | None | **Qwen** |
| Multiple versions | v1 + v2 | Single version | **Qwen** |
| Benchmark harness | benchmark.cu | test_fused.cu | Tie |
| Design doc | ANALYSIS.md | DESIGN.md (9 sections) | Tie |
| Numerical stability | Log-sum-exp (2-pass) | Online max tracking | Tie (both correct) |
| I/O efficiency | 3 reads, 1 write (v1) | 1 read, 1 write | **GLM** |
| Production readiness | Higher (v2, float4, K=256) | Medium (K=32 limit) | **Qwen** |

This one is genuinely a split decision:
- **For algorithmic elegance and the specific constraint ("do NOT materialize"), GLM-5 wins.**
- **For production readiness, vectorization, and K scalability, Qwen3-6 wins.**

**Winner: Split — GLM-5 on algorithm, Qwen3-6 on production readiness**

### Task 3: KV-Cache

| Criteria | Qwen3-6 | GLM-5 | Edge |
|----------|---------|-------|------|
| Files | 8 modular files | 3 files (core + opt + test) | **Qwen** |
| Core cache design | Clean, minimal | Clean, minimal | Tie |
| Memory layout | BHSD | BHSD | Tie |
| Abstractions | KVCache + BatchedKVCache | KVCache only | **Qwen** |
| Attention variants | Standard + GQA + MQA | Standard only | **Qwen** |
| Tests/Demos | 10 demos (comprehensive) | 8 tests (comprehensive) | **Qwen** (2 more) |
| Variable-length batching | Working, 4 lengths demo | Working, 3 lengths test | Tie |
| Paged attention | Page tables + free list | Block pool + free list | Tie |
| Quantization | INT8 with honest overhead notes | INT8/INT4 with reliable error measurement | **GLM** (INT4 support) |
| Chunked prefill | Full impl, verified to 4.5e-10 | Partial impl (uses random Q) | **Qwen** |
| Memory analysis | 6 real models, max context per GPU | 2 model configs, growth tables | **Qwen** |
| GPU mapping | Dedicated file, Tensor Cores | README-level discussion | **Qwen** |
| Model integration | Full transformer with RoPE | IncrementalDecoder (simplified) | **Qwen** |
| Code quality | Dataclasses, type hints | Clean but simpler | Tie |
| Optimizations | Paged + Quant + Chunked + Hybrid | Paged + Quant + Chunked | **Qwen** (Hybrid) |

**Winner: Qwen3-6** (modular architecture, broader scope including attention variants, GPU mapping, and hybrid optimizations, more demos)

### Qwen3-6 vs GLM-5 Overall: **Qwen3-6 wins 2.5-0.5**

Qwen3-6 takes backwards and KV-cache clearly. The fuse task is split — GLM-5's online softmax is algorithmically superior, but Qwen3-6's implementation is more production-ready with float4 vectorization and support for K up to 256.

---

## Summary Matrix

| Matchup | Backwards | Fuse | KV-Cache | Overall |
|---------|-----------|------|----------|---------|
| **GLM-5 vs MiniMax** | GLM | GLM | GLM | **GLM 3-0** |
| **MiniMax vs Qwen3-6** | Qwen | Qwen | Qwen | **Qwen 3-0** |
| **Qwen3-6 vs GLM-5** | Qwen | Split | Qwen | **Qwen 2.5-0.5** |

### Final Rankings (from 2-way analysis)

1. **Qwen3-6** — Best breadth, correctness, and production readiness
2. **GLM-5** — Best algorithm design, clean code; limited scope
3. **MiniMax-M2.7** — Ambitious but buggy; over-engineered yet under-delivered

### Key Takeaways

1. **Qwen3-6** is the most "engineering-mature" model — it writes modular code with separate test files, handles edge cases, cross-verifies formulas, and thinks about production deployment (GPU limits, real model specs).

2. **GLM-5** is the most "algorithmically clever" model — its online softmax kernel is the only genuinely single-pass implementation, and its backward pass caches the fewest intermediates. It values elegance over exhaustiveness.

3. **MiniMax-M2.7** is the most "verbose but inconsistent" model — it writes the most code but has the most bugs. The ambition is there (multiple memory formats, full transformer implementation) but execution falls short (format mismatches, incorrect CUDA syntax, no tests).

4. **Common failure mode**: All three models struggle with efficient numerical gradient checking — they all use Python element-by-element loops instead of batched finite differences, making gradient checks impractical for realistic tensor sizes. MiniMax has the best mitigation (spot-check for >100k elements) but doesn't apply it uniformly.

5. **KV-cache is the most differentiating task**: The complexity of designing a correct, efficient KV-cache system with variable-length batching, paged attention, and quantization reveals the largest quality gap between models. Qwen3-6's 8-file architecture vs MiniMax's monolithic buggy implementation is the clearest illustration.
