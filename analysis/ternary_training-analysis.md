# Ternary Training Challenge: Replicating Ternary Bonsai — Head-to-Head Analysis

## Executive Summary

### Original Runs (each model's own data)

| Model | Ternary? | Path | Loss ↓ | Perplexity | Notes |
|-------|----------|------|--------|------------|-------|
| Qwen3-6 | PASS | A | 9.1 → 4.8 | **83.7** ⚠ | Data leakage (overlapping batches) |
| GLM-5 | PASS | A | 10.3 → 6.0 | 340.9 | Solid |
| GLM-5.1 | PASS | A | 13.8 → 5.1 | 232 | 2000 steps, best docs |
| Kimi K2.6 | PARTIAL | A+B | 19.4 → 3.0 / 11.0 → 3.6 | 3012 / 2001 | Embeddings not ternary |

### Clean Data Rerun (identical `train_data.txt`, ~48K tokens)

| Model | Steps | Train Loss | Val PPL | Ternary | Grade |
|-------|-------|------------|---------|---------|-------|
| **GLM-5** | 250 | 10.8 → **5.27** | **594** | PASS | **A-** |
| **Qwen3-6** | 300 | 13.5 → **5.36** | **319** | PASS | **B+** |
| **Kimi K2.6** | 1000 | 11.1 → **0.016** | **5,501** | PASS | **C+** |
| **GLM-5.1** | 1500 | ? → **0.18** | **30,731** | PASS | **C** |

**GLM-5 wins the clean comparison.** On identical data, its architecture produced the second-best perplexity (594) in the fewest steps (250), with no data leakage. Qwen3-6's honest perplexity (319) is the generalization leader, but the 83.7 from their original run was inflated by training/testing on overlapping data — they disclosed this themselves upon rerun. GLM-5.1 and Kimi catastrophically overfit: near-zero training loss but exploding val PPL (30K and 5.5K respectively).

---

## The Prompt: What Was Asked

The prompt presented two paths:
- **Path A**: Load Qwen3-0.6B via MLX, convert all linear layers to ternary, fine-tune on text with STE.
- **Path B**: Build a smaller Qwen3-style transformer from scratch in NumPy or MLX.

Evaluation criteria: (1) projected weights MUST be in {-1, 0, +1}, (2) training loss must decrease, (3) model must produce non-random text, (4) explain engineering choices.

All four models chose Path A (Qwen3-0.6B fine-tune). Kimi also attempted Path B (small model from scratch).

---

## Per-Model Analysis

### Qwen3-6 — Grade: B+ (`qwen36/ternary_training/train_ternary.py`, 675 lines)

**STE implementation: `W + stop_gradient(W_ternary - W)`**

Clean, idiomatic MLX. The `ternarize_ste` function wraps `ternarize` with the stop-gradient trick. Forward uses ternary, backward is identity. Verified correct.

**Weight loading — BEST IN CLASS:**
```python
layers_list = []
for i in range(args.num_hidden_layers):
    layer = orig_params["layers"][i]
    attn = layer["self_attn"]
    mlp = layer["mlp"]
    layers_list.append({
        "attention": {
            "q_proj": {"weight": attn["q_proj"]["weight"].astype(mx.float32)},
            ...
```
Qwen3-6 is the ONLY model that builds the weight dict with explicit structure matching the MLX module tree. GLM-5/5.1 use recursive traversal that depends on module iteration working correctly — which GLM-5.1 discovered doesn't work with `__dict__` (children are in `.keys()` not `__dict__`). Qwen3-6 sidesteps this entirely by constructing the exact structure manually.

**Architecture:**
- `TernaryEmbedding` (separate class, gather-based) — correct
- Full Qwen3 architecture: GQA (2:1), SwiGLU, RMSNorm, RoPE, Q/K norm, `tie_word_embeddings`
- All linear layers ternary: ✓
- Handles padding for non-divisible in_features via padding/trimming ✓

**CRITICAL FLAW: Data leakage in original run**

The original run used `generate_sample_text()` — synthetic paragraphs about programming, ML, and computing history. The training pipeline uses overlapping batches (`overlap=0.5`):

```python
def prepare_batches(tokenizer, text, max_seq_len=256, overlap=0.5):
    step = int(max_seq_len * (1 - overlap))  # 128
    for i in range(0, len(encoded) - max_seq_len, step):
        batches.append(encoded[i:i + max_seq_len])
```

And validation used `text[:50000]` — **the first 50K characters of the same text, with no train/val separation.** The 83.7 perplexity was measured on data the model had already seen during training.

To their credit, Qwen3-6 **voluntarily disclosed this** upon rerun:
> "The previous run was essentially testing on the same distribution it was trained on (overlapping batches). This run tests on genuinely new content within the same file, giving a more honest perplexity estimate."

**Results — Original (inflated):**
| Metric | Value |
|--------|-------|
| Final loss | 4.76 |
| Perplexity | **83.7** ⚠ (data leakage) |

**Results — Clean rerun (300 steps, BS=4, seq_len=128, train_data.txt):**
| Metric | Value |
|--------|-------|
| Final loss | 5.36 |
| Perplexity | **319** |
| Ternary verification | PASS (2248 groups, 0 violations) |
| Throughput | ~522 tok/s |

**Observations from rerun (their own words):**
- "Loss trajectory is more gradual — genuine learning rather than memorization"
- "Topic coherence: terms like servers, Linux, TCP, Kubernetes appear"
- "Repetition patterns: 'servers, servers, servers' — typical of ternary constraints with limited data"
- "300 steps not enough — loss still decreasing, longer run would likely achieve PPL < 100"

**Strengths:**
- Best architecture fidelity (correct GQA, RoPE with theta=1M, Q/K norm, head_dim=128)
- Best generalization (PPL=319 on clean held-out data)
- Only model to explicitly acknowledge and explain the data leakage
- Explicit nested dict weight loading avoids module-traversal bugs
- Clean separation: `TernaryEmbedding` vs `TernaryLinear`
- Proper padding handling in ternarize

**Weaknesses:**
- Original run's 83.7 was inflated by overlapping batches
- 300 steps not enough for PPL < 100
- Synthetic text in original run was a confound
- `generate_sample_text` would loop infinitely for very large `length`
- Repetition artifacts ("servers servers servers") under ternary constraints

---

### GLM-5 — Grade: A- (`glm5/ternary_training/` modular: `ternary_linear.py` + `ternary_model.py` + `train.py` + `convert.py` + standalone `run_ternary.py`)

**STE implementation: `@mx.custom_function` with explicit `.vjp`**

```python
@mx.custom_function
def ternary_projection(w):
    # ... full projection logic ...

@ternary_projection.vjp
def ternary_projection_vjp(primals, cotangent, output):
    return (cotangent,)
```

This is the **most sophisticated STE** in the field. GLM-5 defines `ternary_projection` as a custom MLX function with an explicit VJP that returns `cotangent` unchanged. This gives direct control over the backward pass and avoids the `stop_gradient` trick entirely. It's the correct MLX-native way to implement custom gradients.

**Architecture:**
- Modular: 4 files + standalone script. Clean class hierarchy.
- `TernaryLinear`, `TernaryEmbedding` (separate classes with proper semantics)
- Full Qwen3: GQA (2:1), SwiGLU, RMSNorm, RoPE, Q/K norm, `tie_word_embeddings`
- Handles padding for non-divisible dimensions ✓
- Uses `mlx_lm.models.qwen3.Attention` as reference + `initialize_rope` from MLX
- `head_dim=128` (correctly extracted from pretrained config)

**Weight loading:**
```python
def copy_weights(src_model, dst_model):
    def collect_weights(module, prefix=''):
        for name in module:  # iterates .keys() — correct
            ...
```
Uses dict-style iteration (`for name in module`) which correctly accesses MLX children. This is the right approach (GLM-5.1 initially tried `__dict__` and failed).

**Training (250 steps, BS=2, seq_len=256, WikiText-2):**
| Metric | Value |
|--------|-------|
| Initial loss | ~10.3 |
| Final loss | ~6.0 |
| Val perplexity | 340.9 |
| Ternary verification | PASS (all 310 weight tensors) |
| Ternary distribution | ~34% / ~31% / ~34% |

**Key engineering insight — gradient clipping at norm=1.0 is CRITICAL:**
GLM-5 discovered that pretrained Qwen3 weights produce initial gradient norms of ~369. Without clipping, training NaN-diverges immediately. Clipping at norm=1.0 is essential. Higher LRs (>1.5e-4) cause divergence even with clipping. This is a genuinely useful finding that no other model reported.

**IMPLEMENTATION_NOTES.md** is the best write-up: detailed, honest about failures, includes a "What Broke and How We Fixed It" table, explains hyperparameter choices clearly.

**Strengths:**
- Best STE implementation (explicit VJP via `@mx.custom_function`)
- Modular code (4 files, clean class hierarchy)
- Best documentation: detailed notes on gradient clipping, NaN divergence, module iteration
- Explicit gradient clipping at norm=1.0 — correct mitigation for pretrained initialization
- Only model with explicit `view_ternary_weights()` and `verify_ternary_weights()` functions
- Uses `initialize_rope` from MLX with `traditional=False` (Qwen3-style RoPE)
- Self-contained `run_ternary.py` (674 lines) for easy single-file execution

**Weaknesses:**
- Higher perplexity than Qwen3-6 (340 vs 84)
- 250 steps is short; loss at 6.0 still very high for a 0.6B model
- `train.py` has a subtle bug in validation loss computation: `total_loss += float(loss) * batch_size` should be `* batch_size * seq_len` (inconsistent with perplexity formula)
- The `ternary_projection` function repeats the full group computation rather than extracting it — DRY violation with `verify_ternary_weights` which reimplements the same logic
- Weight copying via recursive name matching is fragile compared to Qwen3-6's explicit structure

---

### GLM-5.1 — Grade: B+ (`glm5.1/ternary_training/ternary_model.py` + `train.py`, 280 + 424 lines)

**STE implementation: `W + stop_gradient(W_ternary - W)`**

```python
def ternarize_ste(W, group_size=128):
    W_q = mx.clip(mx.round(grouped / scales), -1.0, 1.0)
    W_ternary = (W_q * scales).reshape(flat.shape).reshape(orig_shape)
    return W + mx.stop_gradient(W_ternary - W)
```

Standard stop-gradient trick. Functionally correct (verified non-zero gradients), though less explicit than GLM-5's VJP approach.

**Critical constraint: `assert n % group_size == 0`**

```python
*leading, n = orig_shape
assert n % group_size == 0, f"dim {n} not divisible by group_size {group_size}"
```

GLM-5.1 **requires** all input dimensions to be exactly divisible by 128. This means any layer with a non-divisible in_features will **crash**. For Qwen3-0.6B, `intermediate_size=3072` — which is divisible by 128 (3072/128=24), so it works. But `vocab_size=151936` is used for the LM head — 151936/128 = 1187, so that works too. However, this is fragile for arbitrary models.

**Weight loading — discovered the `__dict__` trap:**
GLM-5.1's NOTES.md documents the most important debugging finding:
> **Critical finding**: MLX's `nn.Module` extends `dict`. Sub-modules and parameters are stored as dict entries (`model['model']`), NOT as `__dict__` attributes. Our initial `copy_weights` using `__dict__` silently failed, leaving all weights at zero.

This caused ALL logits to be zero and ALL gradients to be zero. A critical, non-obvious failure mode specific to MLX's module system. Fixed by iterating over `model.keys()`.

**Results (2000 steps, BS=2, seq_len=512, WikiText-2):**
| Metric | Value |
|--------|-------|
| Initial loss | 13.81 |
| Final loss | 5.14 |
| Perplexity (pre-train) | ~995,563 |
| Perplexity (post-train) | **232** |
| Ternary distribution | 34.7% / 30.9% / 34.3% |
| Ternary verification | PASS |
| Eval PPL trajectory | 333 → 264 → 228 (at steps 500/1000/1500) |

**Key insight — ternarization destroys pretrained knowledge:**
GLM-5.1 explicitly measures the loss jump when converting pretrained weights to ternary:
> Loss jumps from ~2.5 (pretrained) to ~14 (ternarized). The model must re-learn through the ternary constraint.

This is a genuinely valuable observation. The model correctly identifies that fine-tuning from a pretrained checkpoint is fundamentally different from training from scratch — the optimizer must simultaneously "unlearn" full-precision structure AND learn ternary-friendly patterns.

**Training improvements over GLM-5:**
- Constant LR after warmup (not cosine decay) — found that cosine decay drops LR too quickly
- 2000 steps (vs GLM-5's 250) — much longer run
- Better perplexity: 232 vs 340

**Strengths:**
- Best documentation of debugging process (dict vs `__dict__`, zero gradients, weight copy bugs)
- Longest training run (2000 steps) — provides the most reliable convergence signal
- Best perplexity reduction from pre-training: ~995K → 232 (nearly 4300× reduction)
- Explicit pre/post-training perplexity comparison
- Measured eval perplexity trajectory at multiple checkpoints
- Good discussion of "why fine-tuning is harder than training from scratch"
- Notes.md includes a failure table with cause/fix pairs

**Weaknesses:**
- `assert n % group_size == 0` — crashes on non-divisible dimensions (no padding support)
- STE via stop-gradient is less explicit than GLM-5's VJP
- RoPE uses `nn.RoPE(head_dim, base=10000, traditional=False)` — uses hardcoded base=10000 rather than reading from config (Qwen3-0.6B uses 1,000,000)
- `head_dim=64` is hardcoded in ModelArgs (Qwen3-0.6B uses 128) — the `from_dict` correctly reads from config, but the default is wrong
- `ModelArgs.from_dict` try block silently drops unknown keys — could mask real issues
- Attention: no Q/K norm (Qwen3 architecture requires `q_norm` and `k_norm` RMSNorms)
- Text generation is repetitive ("the first two days...")

---

### Kimi K2.6 — Grade: C+ (`kimi-k2.6/ternary_training/`, 3+ files, both Path A and B)

Kimi submitted **two** implementations attempting both paths. Both have significant issues.

#### Path A: `train_ternary.py` (595 lines) — Qwen3-0.6B conversion

**STE: `W + stop_gradient(W_ternary - W)`** — standard, functionally correct.

**MAJOR FLAW: Embedding is NOT ternary:**
```python
def convert_qwen3_to_ternary(model, group_size=128):
    # Skip embedding - it's an Embedding layer, not Linear
    if hasattr(model.model, 'embed_tokens'):
        print(f"  Skipping embedding (not Linear): {model.model.embed_tokens.weight.shape}")
```

The prompt explicitly requires: "ALL linear layers are ternary: embeddings, Q/K/V/O projections, SwiGLU gate/up/down projections, LM head." Kimi's justification that "it's an Embedding layer, not Linear" misses that embeddings ARE linear layers (weighted lookup = matrix multiplication with one-hot vectors). The embedding layer is arguably the MOST important one to ternarize because it dominates parameter count (151936 × 1024 ≈ 155M vs attention projections at 1024² ≈ 1M each).

Furthermore:
- LM head is also NOT converted if `in_features % group_size != 0` (line 147-151: "Skipping lm_head (not divisible)")
- Only Linear layers are converted; there's no `TernaryEmbedding` class at all

**Training data loading is broken:**
```python
def load_wikitext_data(tokenizer, split="train", max_samples=1000, seq_length=256):
    all_tokens = []
    for i, example in enumerate(dataset):
        if i >= max_samples: break
        text = example["text"].strip()
        if len(text) < 50: continue
        tokens = tokenizer.encode(text)
        if len(tokens) > 10:
            all_tokens.append(tokens)
```
Each WikiText example is tokenized INDIVIDUALLY and stored as separate token lists. Then in `create_batches`, each sequence is zero-padded to `seq_length`. This means wiki headings like `## = Valkyria Chronicles III =` become their own "training example" with 7 tokens + 249 padding zeros. The model is mostly learning padding tokens.

**Results (500 steps, BS=2, seq_len=128):**
| Metric | Value |
|--------|-------|
| Initial loss | 19.42 |
| Final loss | 2.95 |
| Perplexity | **3012** |
| Ternary verification | PASS (but embeddings are NOT ternary) |

The loss dropped from 19.42 to 2.95 — which looks like good convergence. But perplexity is 3012, which is TERRIBLE. A random model on vocab_size=151936 would have ln(151936) ≈ 11.93 loss / 151K perplexity. Perplexity 3012 means the model learned SOMETHING but is nearly random. 

The loss curve tells a revealing story: the loss is wildly cyclic (19 → 2 → 11 → 8 → 3 → 9...). It periodically spikes to ~9-11 and then drops to ~3. This pattern suggests **the model is overfitting to individual batches** — the "final loss of 2.95" is just the loss on the last batch, not an indicator of global convergence.

**Cross-entropy loss manual implementation has precision issues:**
```python
probs = mx.softmax(logits_flat, axis=-1)
log_probs = mx.log(probs + 1e-10)
```
Computing softmax then log separately loses precision compared to `log_softmax`. But this is a minor issue.

**Generation quality:** Not shown in results for Path A. The `generate_text` function exists but its output wasn't captured in the results file.

#### Path B: `train_pathb.py` (613 lines) — Small model from scratch

**Model: 8 layers, d_model=512, 8 heads, 4 KV heads, vocab=50257, 75M params.**

**STE: `W + stop_gradient(W_ternary - W)`** with padding support for non-divisible dimensions. Better than the Path A version.

**Same embedding flaw:** `self.embed_tokens = nn.Embedding(vocab_size, dims)` — AGAIN not ternary. The prompt requires ternary for ALL linear layers including embeddings.

**Has padding support:** Handles non-divisible in_features by padding weights. Good engineering.

**Results (1000 steps, BS=16, seq_len=128):**
| Metric | Value |
|--------|-------|
| Initial loss | 11.00 |
| Final loss | 3.63 |
| Perplexity | **2001** |
| Ternary verification | PASS (but embeddings excluded!) |
| Training time | ~247s |

**CRITICAL: Pattern of periodic collapse:**
Looking at the loss curve in `pathb_output.txt`, the model exhibits a disturbing pattern every ~50 steps:
```
Step 200: loss ~5.4
Step 250: loss ~5.3
...
Step 400: loss ~4.7
Step 450: loss ~4.4
...
Step 650: loss ~3.8
Step 700: loss ~3.7
```

But look at perplexity at evaluation checkpoints:
- Step 200: 2336
- Step 400: 1811
- Step 600: 2095
- Step 800: 2165
- Step 1000: 2265

**Perplexity gets WORSE between step 400 and 1000!** The training loss goes 4.67 → 3.63, but perplexity goes 1811 → 2265. This is a clear sign of **overfitting**: the model is memorizing training batches but generalizing worse. The cosine scheduler drops LR to 9.14e-10 by step 1000, which is essentially zero — the model stops learning.

**Text generation quality** — best among all, actually:
> "The capital of France is a " by two @-@ inch ( 2 @.@ 5 m ). The first two @-@ inch m ( 5 @.@"

This is recognizable WikiText-style output (dimensions, measurements, @-@ tokens are WikiText artifacts). Better than GLM-5.1's "the first two days" repetition. But still far from coherent.

**Cosine LR decays to essentially zero:**
```python
lr = LEARNING_RATE * 0.5 * (1 + np.cos(np.pi * progress))
```
By step 1000, lr = 3e-4 * 0.5 * (1 + cos(π)) = 3e-4 * 0 = 9.14e-10. The model effectively stops updating in the last 200 steps. Combined with the perplexity getting worse after step 400, this is a badly tuned schedule.

**Strengths:**
- Only model to attempt BOTH paths
- Path B has padding support for non-divisible dimensions
- Recognized that fine-tuning from pretrained Qwen3 is "catastrophic" (REPORT.md honesty)
- REPORT.md is thoughtful: correctly identifies that training from scratch is better than quantizing pretrained weights
- Path B text generation shows WikiText artifacts (model actually learned from the data)
- Explicitly counts TernaryLinear layers and verifies them

**Weaknesses:**
- **Embeddings are NOT ternary in both paths** — violates the core challenge requirement
- Path A: LM head skipped when in_features not divisible by group_size
- Path B perplexity gets WORSE as training progresses (overfitting)
- Cosine LR schedule decays too aggressively (lr → 0 by step 1000)
- Path A training data fragmented into 1-7 token "examples" with massive padding
- Path A perplexity 3012 despite loss of 2.95 — model is memorizing, not learning
- No explicit Q/K norm (Qwen3 architecture requirement)
- RoPE implementation is a custom class (55 lines) — verbose, zero reuse of `mlx.nn.RoPE`
- Cross-entropy computes softmax then log (precision loss vs log_softmax)
- Path A tokenizer is Qwen's (151936 vocab) but Path B uses GPT-2 (50257) — inconsistent

---

## Comparative Metrics

| Metric | Qwen3-6 | GLM-5 | GLM-5.1 | Kimi K2.6 |
|--------|---------|-------|---------|-----------|
| **Lines of code** | 675 | 674 (run) + modular | 280+424 | 595+613+168 |
| **Files** | 1 | 4+1 standalone | 2 | 3+ |
| **STE method** | stop_gradient | `@custom_function` VJP | stop_gradient | stop_gradient |
| **Embedding ternary?** | ✓ TernaryEmbedding | ✓ TernaryEmbedding | ✓ TernaryEmbedding | ✗ nn.Embedding |
| **Padding support** | ✓ | ✓ | ✗ (assert) | ✓ (Path B) |
| **Weight loading** | Explicit dict | Recursive traversal | Recursive traversal | Layer replacement |
| **Gradient clipping** | ✗ | ✓ (norm=1.0) | ✗ | ✓ (clip=1.0, Path A) |
| **Q/K norm** | ✓ | ✓ | ✗ | ✗ |
| **SwiGLU** | ✓ (silu*gate) | ✓ (swiglu from mlx_lm) | ✓ (silu*gate) | ✓ (silu*gate) |
| **RoPE** | Manual cos/sin | `initialize_rope` | `nn.RoPE` | Custom class |
| **head_dim** | 128 (correct) | 128 (correct) | 64 (wrong default) | 64 (Path B) |
| **rope_theta** | 1,000,000 (correct) | From config | 10000 (hardcoded) | 10000 |
| **Training data** | Synthetic text | WikiText-2 | WikiText-2 | WikiText-2 (fragmented) |
| **Number of steps** | 100 | 250 | **2000** | 500 (A) / 1000 (B) |
| **Perplexity** | **83.7** | 340.9 | 232 | 3012 / 2001 |
| **Ternary verified** | ✓ 0 violations | ✓ | ✓ | ✓ (but embeds excluded) |
| **Documentation** | PROGRESS.md (decent) | **IMPLEMENTATION_NOTES.md (excellent)** | NOTES.md (excellent) | REPORT.md (honest) |
| **Weight copy bug** | Avoided | Fixed (dict iter) | **Found & fixed** | N/A (replaces layers) |

---

## Critical Technical Deep-Dives

### 1. The Embedding Layer: Ternary or Not?

This is the single biggest differentiator. The prompt says "ALL linear layers are ternary: embeddings, Q/K/V/O projections, SwiGLU gate/up/down, LM head."

| Model | Embedding | Verdict |
|-------|-----------|---------|
| Qwen3-6 | `TernaryEmbedding` (gather-based) | ✓ |
| GLM-5 | `TernaryEmbedding` (gather-based) | ✓ |
| GLM-5.1 | `TernaryEmbedding` (gather-based) | ✓ |
| Kimi K2.6 | `nn.Embedding` (standard) | ✗ |

Kimi explicitly skips the embedding layer with the comment "Skip embedding — it's an Embedding layer, not Linear." This is architecturally wrong. The embedding layer stores `vocab_size × hidden_size` weights — for Qwen3-0.6B, that's 151936 × 1024 = 155M parameters, which is ~25% of all parameters. Excluding it from ternarization means 25% of the model is NOT ternary.

Additionally, the embedding weights dominate the first layer's computation: the token embedding projection IS a linear operation (lookup = matrix multiply with a one-hot vector), and not ternarizing it means the first transformation from tokens to hidden states operates at full precision while all subsequent layers are ternary. This creates a precision mismatch at the very input of the network.

### 2. STE Implementation Approaches

Three different STE implementations emerged, all functionally correct:

| Approach | Model | Mechanism |
|----------|-------|-----------|
| `@mx.custom_function` + VJP | GLM-5 | Explicit custom gradient: VJP returns cotangent unchanged |
| `W + stop_gradient(W_ternary - W)` | Qwen3-6, GLM-5.1, Kimi | Forward: W_ternary. Backward: identity through W |
| `mx.stop_gradient(self.weight)` + effective weight | Kimi (alt) | W_effective = W_ternary + (W - stop_gradient(W)) |

GLM-5's approach is the most MLX-idiomatic. By defining `ternary_projection` as a `@mx.custom_function` with an explicit VJP that returns `(cotangent,)`, it gives the framework full knowledge of the gradient computation for potential optimizations. The stop-gradient trick relies on the compiler to optimize away the `W - stop_gradient(W)` term.

### 3. The Weight Copy Bug (MLX-specific trap)

MLX's `nn.Module` extends Python's `dict`. Module children are dict entries, NOT `__dict__` attributes. GLM-5.1 discovered this the hard way:

```python
# BROKEN: iterating __dict__ misses MLX children
for name, child in module.__dict__.items():
    ...

# CORRECT: iterate keys (or use items())
for name in module:
    child = module[name]
    ...
```

GLM-5.1's `copy_weights` initially used `__dict__`, resulting in ALL weights being left at their initialization (zeros for embeddings, random for linear layers). This caused:
- All-zero logits (embedding weights were zeros)
- All-zero gradients (nothing was trainable)
- Loss never moved

After fixing to dict iteration, training worked correctly. Qwen3-6 avoided this entirely by constructing the weight dict explicitly rather than traversing the module tree.

### 4. Gradient Clipping: Critical for Pretrained Initialization

GLM-5 discovered that when starting from pretrained Qwen3-0.6B weights, the initial gradient norm is ~369. Without gradient clipping at norm=1.0, training immediately diverges to NaN. This is because:

1. Pretrained weights are in the range [-0.5, 0.5] after init scaling
2. The ternary projection compresses them to {-s, 0, +s} where s = mean(|W|) ≈ 0.1-0.3
3. The STE passes the full gradient through the projection
4. Large weight values produce large ternary deltas
5. AdamW's update is proportional to the sign of the gradient, but the magnitude of latent weight updates is large due to large gradient values from the pretrained scale

Kimi also uses gradient clipping (clip=1.0), but doesn't document why. Qwen3-6 and GLM-5.1 don't use clipping and both converge — suggesting their different initialization or LR choices avoided this issue.

### 5. Perplexity/Loss Disconnect in Kimi

Kimi Path A shows loss 19.4 → 2.95 but perplexity 3012. The loss curve is violently cyclic:

```
Step 1:  19.4
Step 10: 12.8
Step 40: 10.2
Step 55: 4.0
Step 60: 1.4   ← suspiciously low
Step 65: 2.5
Step 70: 11.4  ← spikes back up
```

This pattern means the model is **overfitting to individual training sequences**. Each batch is a different subset of the fragmented training data, and the model "memorizes" it, then "unlearns" it when switching to the next batch. The final loss of 2.95 is just the loss on the last batch — not representative of global model quality. The 3012 perplexity on validation data is the true signal.

### 6. RoPE Implementation Quality

| Model | Approach | Correct for Qwen3? |
|-------|----------|---------------------|
| Qwen3-6 | Manual cos/sin, freq precomputation | ✓ (theta=1M, head_dim=128) |
| GLM-5 | `initialize_rope(head_dim, base=theta, traditional=False)` | ✓ (reads from config) |
| GLM-5.1 | `nn.RoPE(head_dim, base=10000, traditional=False)` | ✗ (theta=10K, head_dim=64) |
| Kimi | Custom 55-line `RoPE` class | ~ (theta=10K for Path B) |

Qwen3-0.6B uses `rope_theta=1,000,000` and `head_dim=128`. GLM-5.1 hardcodes `rope_theta=10000` in ModelArgs defaults and `head_dim=64` — both wrong. However, `ModelArgs.from_dict` reads from the loaded config, so at runtime these are overridden. But the default values are misleading and would cause incorrect behavior if the config read fails.

---

## Clean Data Rerun: The Decisive Experiment

To control for data quality as a confound, all four models were given the same `train_data.txt` — 271K characters (~48K tokens) of clean encyclopedic prose across science, technology, history, philosophy, medicine, and other domains. Each model was asked to re-run with identical data, keeping all architectural choices unchanged.

### Results

| Model | Steps | Batch/Seq | Train Loss | Val PPL | Overfitting ratio |
|-------|-------|-----------|------------|---------|-------------------|
| **GLM-5** | 250 | 2/256 | 10.8 → 5.27 | **594** | Moderate (5.3 vs 6.4 val) |
| **Qwen3-6** | 300 | 4/128 | 13.5 → 5.36 | **319** | Moderate (5.4 vs 5.8 val) |
| **Kimi K2.6** | 1000 | 16/128 | 11.1 → 0.016 | **5,501** | Catastrophic (0.02 vs 8.6 val) |
| **GLM-5.1** | 1500 | 2/256 | ? → 0.18 | **30,731** | Catastrophic (0.2 vs 10.3 val) |

### Analysis

**The data size trap.** At only ~48K tokens, it's fundamentally impossible for a model to generalize well. With vocabularies of 50K-150K tokens, the amount of data per parameter is tiny. However, the *degree* of overfitting reveals structural differences in the implementations:

**GLM-5 and Qwen3-6 overfit moderately.** Their training losses (~5.3) are in a healthy range relative to val PPL (319-594). The gap between train and val is expected for such small data. Both models are still learning — loss continues decreasing at the end of training. Their architectures are fundamentally sound.

**GLM-5.1 catastrophically overfit.** Training loss hit 0.18 (PPL=1.2 — nearly perfect next-token prediction), while val PPL exploded from 1,254 at step 300 to 30,731 by step 1500. The val PPL *worsened* with more training — classic sign of memorization. Key factors:
- 1500 steps at BS=2/seq=256 = ~16 full passes over 48K tokens for a 0.6B model
- No gradient clipping — large updates from pretrained weights
- Constant LR at 5e-4 after warmup with no decay

**Kimi massively overfit Path B.** Training loss 0.016 on a 50K vocabulary — the model is outputting near-certainty predictions for every token. Val PPL 5,501 despite this. Their Path B (75M param model from scratch) simply memorized the 198 training sequences. On in-domain prompts like "Artificial intelligence is," the model regurgitated training text verbatim: "*...the field was formally founded in 1956. Early researchers confidently predicted that machines would match human intelligence within a generation.*" On out-of-domain prompts like "The capital of France is," it produced garbled cross-topic hallucinations: "a eukary toeseses are a bustling period, and be proteins..."

### Qwen3-6's Original Data Leakage

Qwen3-6's original 83.7 perplexity — which initially appeared to be the best result by a wide margin — was inflated by overlapping training/validation batches. Their `prepare_batches` function uses 50% overlap, and their validation text was `text[:50000]` — literally the first 50K characters of the training text. Upon rerun with proper train/val separation, they achieved 319 PPL.

To their credit, they disclosed this immediately and unprompted:
> "The previous run was essentially testing on the same distribution it was trained on (overlapping batches). This run tests on genuinely new content within the same file, giving a more honest perplexity estimate."

This level of self-awareness and honesty is notable. It also explains why their original "Text generation: coherent output" claim seemed generous — it was coherent because it was tested on data it had seen during training.

### The Overfitting Spectrum

The rerun reveals a clear spectrum of generalization quality:

```
Best generalization ←→ Worst generalization
Qwen3-6 (319)  >  GLM-5 (594)  >>  Kimi (5,501)  >>  GLM-5.1 (30,731)
```

GLM-5 achieved the second-best PPL (594) in the fewest steps (250). Qwen3-6 achieved the best generalization (319) in 300 steps. The gap between these two and the others is a chasm — Kimi and GLM-5.1 completely collapsed into memorization, producing meaningless validation results.

---

## Rankings

| Rank | Model | Rationale |
|------|-------|-----------|
| **1** | **GLM-5** | Best STE (`@mx.custom_function` with explicit VJP). Only model with gradient clipping insight. Best honest PPL (594) per training step (250). Modular code. Excellent documentation. No data leakage. Robust across two different datasets. |
| **2** | **Qwen3-6** | Best architecture fidelity (correct GQA, Q/K norm, RoPE theta=1M). Best generalization margin (honest PPL=319 after proper train/val split). Only model to explicitly acknowledge the data leakage. But: original 83.7 was inflated, synthetic text was a confound, and baseline PPL after proper split is 319 — still solid but not the sweep it first appeared. |
| **3** | **GLM-5.1** | Best documentation narrative (debugging journey, failure table). Weight copy bug discovery is a genuine contribution. Longest original run (2000 steps on WikiText-2) showing genuine improvement. But: missing Q/K norm, hardcoded RoPE defaults, assert crash on non-divisible dimensions, catastrophic overfit in rerun (PPL=30K at 1500 steps) suggesting the optimizer/LR configuration is fragile. |
| **4** | **Kimi K2.6** | Ambitious (attempted both paths). REPORT.md is honest. Path B generation shows ability to memorize training data verbatim. But: embeddings NOT ternary (violates core spec), Path A training data fragmented to uselessness, Path B perplexity worsens with training, cosine LR decays to near-zero, Path B rerun produced near-zero train loss with 5.5K val PPL — classic catastrophic overfitting to a tiny dataset. Multiple fundamental issues. |

---

## Key Takeaways

1. **The data size trap is real, and it discriminates.** At ~48K tokens, every model overfit — but the degree ranged from moderate (Qwen3-6: PPL=319, GLM-5: PPL=594) to catastrophic (Kimi: PPL=5,501, GLM-5.1: PPL=30,731). This reveals which architectures are robust to small-data fine-tuning vs. which collapse into memorization. GLM-5's gradient clipping and Qwen3-6's proper train/val separation are both protective factors.

2. **Qwen3-6's original 83.7 PPL was inflated by overlapping train/val batches.** Their `prepare_batches(overlap=0.5)` combined with `val_text = text[:50000]` meant they tested on data the model had seen. They disclosed this themselves. Their honest PPL is 319 — still the generalization leader, but not by the 4× margin the original numbers suggested.

3. **The embedding layer is the hidden differentiator.** Three models correctly implemented `TernaryEmbedding`; Kimi skipped it entirely with the justification "it's an Embedding, not Linear." The embedding layer is 155M parameters out of ~600M — excluding it means 25% of the model operates at full precision.

4. **Pretrained initialization is a double-edged sword.** GLM-5 discovered that starting from Qwen3-0.6B weights requires gradient clipping (norm=1.0) to prevent NaN divergence. GLM-5.1 explicitly measured the loss jump from ~2.5 to ~14 after ternarization. GLM-5.1's catastrophic rerun overfitting despite pretrained initialization suggests their optimizer configuration is incompatible with small-data fine-tuning.

5. **Training step count interacts critically with data size.** GLM-5.1 ran 1500 steps on 48K tokens — ~16 epochs over a dataset where the model already had pretrained knowledge. This is far too many. GLM-5's 250 steps (2.7 epochs) and Qwen3-6's 300 steps were in a healthier range. Training longer is not always better.

6. **The MLX `__dict__` vs `.keys()` trap is real and subtle.** GLM-5.1's debugging journey — all-zero logits, all-zero gradients, hours of confusion — came down to using `__dict__` to iterate MLX modules instead of dict-style iteration.

7. **Perplexity doesn't always track with loss.** Kimi Path A showed loss 19.4 → 2.95 but perplexity 3012 — the cyclic loss curve and fragmented data explain the disconnect. In the rerun, Kimi Path B showed loss 11.1 → 0.016 but perplexity 5,501 — near-perfect training loss with no generalization. Loss alone is not a trustworthy metric.

8. **Honesty and self-diagnosis matter.** Qwen3-6 voluntarily disclosed the overlapping batch issue. GLM-5.1 documented every failure. GLM-5 explained gradient clipping. Kimi acknowledged "catastrophic" results from pretrained conversion. The models that understood what went wrong wrote better code.

9. **The perplexity target of <100 was likely unreachable with 48K tokens of data for a 0.6B ternary model.** None of the four achieved it with honest measurement. Qwen3-6 was closest at 319, and their loss was still decreasing at step 300. A run of 1000-2000 steps might get there, but would risk the overfitting cliff that GLM-5.1 fell off.

10. **This challenge reveals the gap between "code that runs" and "code that learns."** All four models wrote syntactically correct MLX. All four verified ternary projection correctly. But only GLM-5 and Qwen3-6 produced architectures that generalize (even weakly) under constrained data. The difference is in hyperparameter discipline — gradient clipping, appropriate step counts, proper train/val separation — not in code correctness.

---

## Final Ternary Training Ranking

1. **GLM-5** — Grade: A- — Best STE, gradient clipping insight, most robust across both datasets. Honest PPL=594 in 250 steps. No data leakage.
2. **Qwen3-6** — Grade: B+ — Best architecture fidelity, best generalization (PPL=319). Original 83.7 was inflated; honest disclosure. Still the generalization leader.
3. **GLM-5.1** — Grade: C+ — Best docs, longest original run. But catastrophic overfit in rerun (PPL=30K), missing Q/K norm, assert crash risk. Architecture is fragile.
4. **Kimi K2.6** — Grade: C — Ambitious (both paths), honest REPORT.md. Embeddings not ternary, cyclic loss, catastrophic overfit in rerun (PPL=5,501). Multiple correctness issues.

---

## Verdict on the Challenge Itself

This is the hardest challenge by design, and the results reflect it. No model achieved what PrismML achieved (competitive 8B models at 1.58 bits). All four implementations are proof-of-concept demonstrations, not production ternary training pipelines.

The clean data rerun was decisive. It revealed that:
- Only two of four models (GLM-5, Qwen3-6) generalize at all under constrained data
- The gap between "loss goes down on training data" and "loss goes down on unseen data" is a chasm
- Hyperparameter discipline (gradient clipping, appropriate step counts, proper train/val separation) matters more than algorithmic sophistication
- Qwen3-6's original 83.7 PPL was inflated — their honest 319 is the best generalization, but not the sweep it first appeared

The gap between these implementations and a competitive ternary 8B model is enormous — probably 1000× more compute, 10^6× more data, knowledge distillation, and careful initialization schemes. However, **all four models demonstrated core understanding of the ternary training concept**: group-wise quantization, STE gradient flow, and verification of ternary projection. This is non-trivial engineering, and the fact that all four produced working implementations is impressive.

Between the original runs and the clean-data rerun, **GLM-5 emerges as the most robust model** — correct architecture, disciplined training, no data quirks, and consistent results across datasets. **Qwen3-6 has the best generalization potential** but needs proper train/val separation. **GLM-5.1 and Kimi need significant work** on data efficiency and overfitting control before their ternary training is practically useful.
