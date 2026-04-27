You are attempting to replicate Ternary Bonsai (PrismML, April 2026) — a family of
language models natively trained with ternary weights {-1, 0, +1} that achieve
competitive benchmark scores at 1/9th the memory of full-precision models.

This is an active research area. PrismML has demonstrated it works with Ternary Bonsai.
What follows is everything the public knows. Your job is to fill in the gaps
and produce a working ternary training procedure.

================================================================================
WHAT IS KNOWN
================================================================================

Architecture:
- Ternary Bonsai uses the EXACT Qwen3 architecture (confirmed by HF model card,
  config.json, and multiple community sources).
- Qwen3 features: Grouped Query Attention (2:1 ratio), SwiGLU MLP, RoPE
  positional embeddings, RMSNorm, no bias in linear layers.
- Qwen3-0.6B: 28 layers, hidden_size=1024, 16 query heads, 8 KV heads,
  intermediate_size=3072, vocab_size=151936, max_position_embeddings=32768.
- ALL linear layers are ternary: embeddings, Q/K/V/O projections, SwiGLU
  gate/up/down projections, LM head. No high-precision escape hatches.
- RMSNorm and other normalization layers remain in FP16/FP32 (few params).

Ternary weight format:
- Group-wise quantization: groups of 128 weights share one FP16 scale factor `s`.
- Each weight in a group is {-s, 0, +s}, stored as {-1, 0, +1} (2 bits each).
- The scale factor per group is computed as: s = mean(|W_group|).
  Some BitNet variants use max(|W_group|) or a learned scale — the community
  believes PrismML uses mean absolute value based on ablation studies.
- Q2_0 is the GGUF packing format: 2 bits per weight, 4 code points where
  q=0 → -s, q=1 → 0, q=2 → +s, q=3 → reserved/unused.

Training procedure (from BitNet b1.58 lineage + PrismML hints):
- Weights are stored in full precision (FP32/FP16) as LATENT weights.
- On the FORWARD pass: project latent weights to ternary using group-wise
  scales, then use the ternary weights for computation.
- On the BACKWARD pass: use the Straight-Through Estimator (STE).
  The gradient through the rounding operation is treated as identity.
  dL/dW_latent = dL/dW_ternary. The scale factor is treated as constant.
- Training is done FROM SCRATCH (not post-training quantization of an
  existing model). However, the architecture is identical to Qwen3.
- The initialization likely follows BitNet: weights initialized with
  a normal distribution scaled by (fan_in)^(-0.5), then the ternary
  projection is applied from step 0.
- Optimizer: likely AdamW with weight decay. BitNet uses a specific
  learning rate schedule with warmup.
- Training data: unknown, but PrismML claims the models are competitive
  with Qwen3-8B, suggesting similar-scale pretraining data.

Key references to consult (web search recommended):
1. "BitNet b1.58" paper (Microsoft Research, 2024) — the foundation
2. PrismML blog: https://prismml.com/news/ternary-bonsai
3. PrismML GitHub: https://github.com/PrismML-Eng/Bonsai-demo
4. PrismML whitepaper (PDF in Bonsai-demo repo): ternary-bonsai-8b-whitepaper.pdf
5. HF model card: https://huggingface.co/prism-ml/Ternary-Bonsai-8B-mlx-2bit
6. llama.cpp Q2_0 kernel implementation (for packing format reference)
7. Bankai: https://github.com/... (post-training ternary adaptation method,
   different approach but relevant)

================================================================================
YOUR TASK
================================================================================

Implement ternary training and apply it to produce a working ternary model.
You have TWO paths — choose the one you can complete successfully:

PATH A (Recommended — Real Scale):
1. Use MLX (Apple's ML framework, native on this Mac) to load the Qwen3-0.6B
   checkpoint. MLX is pre-installed. Import it as `import mlx.core as mx`
   and `import mlx.nn as nn`. MLX tensors are NumPy-compatible.
2. Implement the ternary linear layer as an MLX module that:
   - Stores latent weights in float32
   - Projects to ternary on forward pass using group_size=128
   - Uses STE for gradient propagation
   - Handles the scale factor computation: s = mean(|W|) per group
3. Convert the loaded Qwen3-0.6B model to use ternary linear layers.
   Keep RMSNorm in float16. Keep the attention mechanism unchanged (it
   operates on activations, not stored weights).
4. Fine-tune the ternary model on a small text dataset for at least 200 steps.
   Use cross-entropy loss. Show that loss decreases.
5. After training, verify:
   a) ALL weights in ternary linear layers project to {-1, 0, +1} (× scales)
   b) The model can generate coherent text (qualitative check)
   c) Perplexity on a held-out set is not astronomical (< 100)
6. Explain your training procedure, hyperparameters chosen, and any
   observations about what worked and what didn't.

PATH B (NumPy-only, smaller scale):
1. Using only NumPy, implement a Qwen3-style transformer with the SAME
   architecture features (GQA 2:1, SwiGLU, RMSNorm, RoPE) but at a smaller
   scale: 6-8 layers, d_model=512-768, at least 4 attention heads.
2. Implement the ternary linear layer with group_size=128 and STE.
3. Train from scratch on a text corpus (WikiText-2 or similar) for at
   least 1000 steps. Use batch_size >= 16.
4. Verify ternary projection and measure perplexity improvement.
5. Explain your procedure and hyperparameters.

================================================================================
EVALUATION CRITERIA
================================================================================

Your solution will be judged on:
1. CORRECTNESS: After training, projected weights MUST be in {-1, 0, +1}.
   This is non-negotiable. Check with: abs(round(W/s) - {-1,0,+1}) < 1e-5.

2. CONVERGENCE: Training loss must decrease. If loss stays flat or increases,
   your STE implementation or learning rate is wrong.

3. FUNCTIONALITY: The model must produce non-random text. Even if quality is
   low, it must demonstrate it learned SOMETHING from the data.

4. ENGINEERING JUDGMENT: Explain your choices. Why group_size=128 and not 256?
   Why mean(|W|) for scale and not max(|W|)? What learning rate worked? What
   broke and how did you fix it?

================================================================================
RESOURCES ON THIS MACHINE
================================================================================

- MLX is available: `import mlx.core as mx`, `import mlx.nn as nn`
- NumPy is available
- GPU: Apple M4 with unified memory (use MLX for GPU acceleration)
- Qwen3-0.6B weights may be downloaded via:
  `from mlx_lm import load; model, tokenizer = load("Qwen/Qwen3-0.6B")`
  or from HuggingFace: Qwen/Qwen3-0.6B
- WikiText-2 is available via `from datasets import load_dataset` or
  can be downloaded as raw text
- Web search is available if you need to check paper details or APIs

================================================================================
NOTE
================================================================================

This is a genuinely open-ended challenge. PrismML has demonstrated success with Ternary Bonsai.
The BitNet b1.58 paper describes the concept but not the exact recipe for
training a competitive 8B model. Your implementation may not match PrismML's
exactly — that's expected. The goal is to produce a working ternary training
procedure and learn what works. Document your findings.
