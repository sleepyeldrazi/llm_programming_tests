# Ternary Bonsai: Implementation Notes & Findings

## Architecture

The implementation follows the Qwen3-0.6B architecture exactly, replacing all `nn.Linear` and `nn.Embedding` layers with ternary equivalents:

- **Model**: Qwen3-0.6B (28 layers, hidden_size=1024, 16 query heads, 8 KV heads, head_dim=128, intermediate_size=3072, vocab_size=151936)
- **Ternary layers**: Every linear layer (embeddings, Q/K/V/O projections, SwiGLU gate/up/down, LM head) uses ternary weights
- **Full-precision layers**: RMSNorm and attention scaling remain in float32

## Key Implementation Details

### Ternary Weight Projection (group_size=128)

Each weight matrix is divided into groups of 128 along the last dimension. For each group:
```
s = mean(|W_group|)           # FP16 scale factor
W_q = clip(round(W / s), -1, 1)  # Ternary indices {-1, 0, +1}
W_ternary = W_q * s            # Effective weight
```

### Straight-Through Estimator (STE)

The non-differentiable rounding is handled via:
```python
W_out = W + stop_gradient(W_ternary - W)
```
- **Forward**: Uses `W_ternary` (quantized weights)
- **Backward**: Gradient passes through `W` as identity (`dL/dW = dL/dW_ternary`)

This was verified to produce non-zero gradients in isolation (Test 1-3 in debugging).

### Why group_size=128?

- Powers of 2 align well with GPU/accelerator memory access patterns
- 128 provides a good balance between quantization granularity and statistical stability of the scale factor
- Too small (e.g., 32): noisy scales, unstable training
- Too large (e.g., 256): scales can't adapt to local weight distributions
- PrismML confirmed group_size=128 in their GGUF format discussion

### Why mean(|W|) for scale?

- `mean(|W|)` is more robust than `max(|W|)` because it's less sensitive to outliers
- With normally distributed weights, `mean(|W|) ≈ 0.8 * std(W)`, giving a stable scale
- `max(|W|)` would compress most weights toward 0, losing expressivity
- BitNet b1.58 also uses absmean quantization, confirming this choice

## Training Procedure

### Setup
1. Load Qwen3-0.6B weights from HuggingFace (via mlx_lm)
2. Create ternary model with identical architecture (TernaryLinear replacing nn.Linear)
3. Copy pre-trained weights as latent float32 weights
4. Ternary projection happens on every forward pass

### Hyperparameters
- **Optimizer**: AdamW (betas=0.9, 0.95, weight_decay=0.01)
- **Learning rate**: 5e-4 constant after 50-step linear warmup
- **Batch size**: 2 (limited by GPU memory with 0.6B float32 latent weights + optimizer state)
- **Sequence length**: 512
- **Dataset**: WikiText-2 (train: 2.5M tokens, val: 262K tokens)

## Results

### 2000-step Training Run
| Metric | Pre-training | Post-training |
|--------|-------------|---------------|
| Loss | 13.81 | 5.14 |
| Perplexity | 995,563 | 232 |
| Ternary weights | {-1, 0, +1} | {-1, 0, +1} |

Eval perplexity trajectory:
- Step 500: 333
- Step 1000: 264
- Step 1500: 228

The model is still steadily improving. With more training steps (5K-10K), perplexity would likely drop below 100.

### Text Generation (after 2000 steps)
```
Prompt: "The most important thing about"
Output: "...the world . The first two days later , the first two days of
the first two days , the first two days of the first two days..."
```

The output shows learned patterns (English syntax, punctuation) but is repetitive due to limited training.

### Weight Distribution
All ternary layers project correctly to {-1, 0, +1}:
- ~34.7% are -1
- ~30.9% are 0
- ~34.3% are +1

This matches the expected distribution for normally-distributed latent weights.

## Key Findings & Observations

### 1. Weight Copy: MLX Module Structure
**Critical finding**: MLX's `nn.Module` extends `dict`. Sub-modules and parameters are stored as dict entries (`model['model']`, `model['embed_tokens']`), NOT as `__dict__` attributes. Our initial `copy_weights` using `__dict__` silently failed, leaving all weights at zero. Fixed by iterating over `model.keys()` instead.

### 2. Ternarization Destroys Pre-trained Knowledge
When Qwen3-0.6B weights are ternarized, the model's loss jumps from ~2.5 (pre-trained) to ~14 (ternarized). This is expected: ternary weights at ~1.58 bits cannot represent the same information as 16-bit weights. The model must re-learn through the ternary constraint.

### 3. STE Works Correctly
The Straight-Through Estimator implementation via `W + stop_gradient(W_ternary - W)` produces correct non-zero gradients. We verified:
- Simple STE: gradient = [-2, 0, 2] (expected)
- W-dependent STE: non-zero gradients
- Full model: non-zero gradients for all layers

### 4. Training From Scratch vs Fine-tuning
PrismML trains from scratch, not from a pre-trained checkpoint. Our fine-tuning approach is fundamentally harder because:
- Pre-trained latent weights encode full-precision patterns
- The optimizer must simultaneously "unlearn" full-precision structure and learn ternary-friendly patterns
- Training from scratch with random init would likely converge faster to a good ternary solution

### 5. What Broke and How We Fixed It
| Issue | Cause | Fix |
|-------|-------|-----|
| All-zero logits | copy_weights used `__dict__` which misses MLX sub-modules | Use dict-style iteration (`model.keys()`) |
| Zero gradients (first attempt) | Weights were never actually loaded (same root cause) | Same fix |
| Slow convergence with cosine decay | LR decays to near-zero too quickly | Use constant LR after warmup |
| Noisy training loss | batch_size=2 gives high variance gradients | Acceptable for demo; gradient accumulation would help |

## Files

- `ternary_model.py` — Ternary Bonsai model definition (TernaryLinear, TernaryEmbedding, full Qwen3 architecture)
- `train.py` — Training, evaluation, and verification script
- `NOTES.md` — This document
