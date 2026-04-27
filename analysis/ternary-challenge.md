# Ternary Training Challenge: "Make It Work"

## What this is

Ternary Bonsai (PrismML, April 2026) is a family of language models trained natively
with ternary weights {-1, 0, +1} from scratch. The group-wise quantization scheme
stores one FP16 scale factor per 128 weights. The result: 8B params in 1.75 GB,
running at 82 tok/s on M4 Pro, competitive with full-precision 8B models.

The exact training recipe is semi-public — PrismML's whitepaper and blog posts give
hints (BitNet b1.58 lineage, group-wise scales, straight-through estimator, no
high-precision escape hatches), but the full procedure is not open-sourced.

This challenge asks models to synthesize the known information, fill in the gaps,
and produce a working ternary training implementation from scratch.

## What's known (give this to the model)

Publicly available facts about Ternary Bonsai's training:
- Based on BitNet b1.58 (Microsoft Research, 2024): ternary weights {-1, 0, +1}
  with the straight-through estimator for gradient propagation
- Group-wise quantization: groups of 128 weights share one FP16 scale factor
- During training: weights are stored in FP32/FP16, projected to ternary on the
  forward pass, gradients flow through the STE on the backward pass
- All layers are ternary: embeddings, attention projections, MLP, LM head
- No high-precision "escape hatches" — the entire network operates at 1.58 bits
- The scale factor per group is typically computed as the mean absolute value
  of weights in that group: s = mean(|W_group|)
- The ternary projection: W_ternary = s * round_clip(W / s, -1, 0, 1)
  where round_clip maps each element to the nearest of {-1, 0, 1}
- Training uses the STE: forward pass uses W_ternary, backward pass computes
  gradients w.r.t. the full-precision latent weights W_latent
- Latent weights are kept in FP32 and only projected to ternary for the forward pass
- The gradient through round_clip is treated as identity (STE)

## The prompt

```
Implement NATIVE TERNARY TRAINING for a small transformer language model
from scratch in NumPy.

BACKGROUND:
Ternary Bonsai (PrismML, 2026) showed that language models trained with
ternary weights {-1, 0, +1} from scratch can match full-precision 8B models
while using 9x less memory. The key technique: group-wise ternary projection
with the straight-through estimator (STE), applied to ALL layers.

WHAT TO BUILD:

1. TERNARY LINEAR LAYER:
   Instead of a standard Linear(W @ x + b), implement TernaryLinear where:
   
   a) The layer stores LATENT weights W_latent of shape (out_dim, in_dim)
      in full precision (float32).
   b) On the forward pass:
      - Reshape W_latent into groups of GROUP_SIZE=128 along the in_dim.
        If in_dim is not divisible by 128, pad the last group.
      - For each group g:
          s_g = mean(|W_latent[g]|)           # scale factor
          W_ternary[g] = s_g * round_clip(W_latent[g] / s_g)
        where round_clip(x) maps each element to the nearest of {-1, 0, +1}.
        Ties: values in [-0.5, 0.5] → 0, values > 0.5 → 1, values < -0.5 → -1.
      - output = x @ W_ternary^T  (use ternary weights for the forward pass)
   c) On the backward pass (for gradient computation):
      - Gradients flow through the ternary projection via the straight-through
        estimator (STE): ∂L/∂W_latent = ∂L/∂W_ternary
        (The rounding operation's gradient is treated as identity.)
      - The scale factor s_g is treated as constant w.r.t. W_latent for
        gradient purposes (stop_gradient on s_g).
      - ∂L/∂x = ∂L/∂output @ W_ternary  (use ternary weights for the VJP too)
   d) Weight decay or gradient clipping is optional but recommended.

2. TERNARY TRANSFORMER:
   Build a minimal transformer where ALL linear layers use TernaryLinear:
   - Token embedding projection (TernaryLinear, no bias)
   - Query, Key, Value projections (TernaryLinear, no bias)
   - Output projection (TernaryLinear, no bias)
   - FFN up-projection (TernaryLinear, no bias)
   - FFN down-projection (TernaryLinear, no bias)
   - LM head (TernaryLinear, no bias)
   
   Use standard attention (non-ternary softmax is fine — attention scores
   are computed, not stored as weights). RMSNorm or LayerNorm can remain
   in FP32 (normalization has few parameters).

   Architecture: 2 layers, d_model=128, n_heads=4, d_ff=512, vocab_size=256.

3. TRAINING LOOP:
   Train on a synthetic COPY TASK:
   - Input: random token sequences of length 16 (tokens 0..255)
   - Target: identical sequence (model must learn to copy)
   - Loss: cross-entropy on each position
   - Optimizer: AdamW or SGD with momentum (your choice)
   - Train for 500 steps, batch_size=32
   - Learning rate: tune it (start at 3e-4 and adjust if needed)

4. CORRECTNESS CHECKS:
   After training, verify:
   a) TRAIN LOSS: final loss < 0.3 (model actually learned something)
   b) TERNARITY: inspect W_latent of any TernaryLinear layer.
      After projecting to ternary (dividing by group scales and rounding),
      ALL values must be in {-1, 0, +1}. No exceptions.
      Test: compute projected = round_clip(W_latent / s_per_group).
      assert all values in projected are -1, 0, or 1.
   c) SCALE FACTORS: each group of 128 has exactly one scale factor.
      Verify group 0 uses s[0], group 1 uses s[1], etc.
   d) COPY ACCURACY: on 10 held-out sequences, the model should copy
      >80% of tokens correctly.

5. DELIVERABLES:
   - Class TernaryLinear with forward/backward (manual gradients, no autograd)
   - Class TernaryTransformer (2 layers, 128 dim)
   - Training loop that produces decreasing loss
   - Test 1: After training, assert all projected weights are in {-1, 0, +1}
   - Test 2: Train loss < 0.3
   - Test 3: Copy accuracy > 80% on held-out data
   - Comments explaining:
     * Why the STE works for ternary training
     * Why group-wise scales are needed (not one global scale)
     * What happens if you don't use group-wise scales

Use only NumPy. No PyTorch, JAX, TensorFlow, or autograd.
```

## Why this is the hardest challenge

1. **No public reference implementation exists.** PrismML has demonstrated the approach works; models must reason from their published results and first principles.

2. **It's a full training pipeline.** Requires correct forward pass, manual backward pass, ternary projection, STE gradient handling, optimizer integration, and debugging why the loss doesn't go down.

3. **The scale factor computation has a footgun.** If you compute `s = mean(|W|)` but then divide by it before rounding, the ternary values are `round(W/s)`. If `s` is very small (near-zero weights at initialization), `W/s` explodes and all weights round to ±1, losing the zero state. Good initialization and proper scale computation are critical.

4. **The STE is simple but the interaction with weight decay is not.** Standard weight decay pulls all weights toward zero. But ternary training WANTS weights at -1, 0, or +1 — weight decay fights the ternary projection. Models need to recognize this tension and handle it.

5. **Testing is clean and objective.** Either loss goes down (or it doesn't), weights are ternary (or they're not), copy accuracy is high (or it's not).

## How to test it

```bash
cd <model>/ternary_training
python ternary_train.py
# Should print:
# Step 0: loss=5.234
# Step 100: loss=1.234
# ...
# Step 500: loss=0.123
# 
# Ternary check: all weights in {-1, 0, +1}: PASS
# Copy accuracy: 94.2%: PASS
```
