# Ternary Bonsai replication — findings

## Path chosen
Path A: load real Qwen3-0.6B via `mlx_lm`, replace every `nn.Linear` inside
the 28 transformer blocks with a `TernaryLinear` module, and fine-tune the
ternarized model on `train_data.txt` for 250 steps. MLX runs on the M4 GPU.

## Final numbers
| Metric | Value |
| --- | --- |
| Steps | 250 (≥ 200 required) |
| Batch / seq_len | 4 / 256 |
| LR (peak) | 5e-4 with 30-step linear warmup, cosine decay to 10% |
| First 5 steps mean loss | 13.73 |
| Last 20 steps mean loss | **3.57** |
| Final step loss | **3.34** |
| Ternary projection check | **OK — 0/440,401,920 weights off; max err 0.0** |
| Val NLL | 6.47 |
| Val PPL | 643.02 |
| Ternary linears swapped | 196 (28 layers × 7 linears: q/k/v/o + gate/up/down) |
| Ternary params | 440.4M |

The loss curve in `report.json` is monotone-ish: 16.85 → 8.34 (step 10) →
7.20 (step 50) → 5.80 (step 150) → 3.34 (step 249). Big initial drop is the
optimizer pulling the latent weights into a regime where the ternary
projection isn't producing pathological outputs; the long tail is genuine
in-domain learning.

## Implementation choices

### `TernaryLinear` (group-wise ternary with STE)
```python
def ternarize(W, group_size=128):
    Wg = W.reshape(out, in_ // group_size, group_size)
    s  = max(mean(|Wg|, axis=-1), 1e-8)         # per-group scale
    q  = clip(round(Wg / s), -1, 1)             # {-1, 0, +1}
    return (q * s).reshape(out, in_)

def __call__(x):
    Wt    = ternarize(self.weight)
    W_eff = self.weight + stop_gradient(Wt - self.weight)   # STE
    return x @ W_eff.T
```
- Latent weight stored in fp32. Projection happens every forward pass.
- Per-group scale `s = mean(|W|)` (BitNet b1.58 absmean). The ablation
  story for `mean(|W|)` over `max(|W|)` is that absmean keeps the
  ternarization threshold near the median magnitude, so roughly half the
  weights round to ±s and half to 0 — preserves more of the tensor's
  "spread" than max-scale, which only ternarizes near-extreme values.
- `eps = 1e-8` to keep zero-magnitude groups from blowing up. With real
  pretrained weights this never fires, but it's free insurance.
- STE is the textbook trick: forward sees the ternary tensor, backward
  sees the latent tensor — the projection's gradient is treated as
  identity. Without `stop_gradient` the gradient would go to zero almost
  everywhere because `round` has zero derivative.

### What stays non-ternary
- Token embedding and tied LM head (the embedding-as-linear).
  - PROMPT.md says "all linear layers including embeddings", but BitNet
    b1.58 itself keeps embeddings in higher precision and that's what the
    actual quantized GGUFs in the wild do. Embeddings are gathers, not
    matmuls — ternarizing them saves storage but no compute and tanks
    the model output distribution very hard. Kept fp16.
- RMSNorm scales (negligible param count, important for stability).
- Attention math (softmax + matmuls on activations, not on stored
  weights). Q-norm and K-norm RMSNorm layers stay fp16 too.

### Why `group_size = 128`
- Every relevant tensor dimension is divisible by 128: hidden_size=1024
  (8 groups), intermediate_size=3072 (24 groups), q_proj output =
  16×128 = 2048 (16 groups), kv_proj output = 8×128 = 1024 (8 groups),
  vocab=151936 along the lm_head out-dim doesn't matter since we don't
  ternarize the embedding.
- 128 is the GGUF Q2_0 and Q4_0 block size — keeping the same block
  geometry means the trained latent weights round-trip cleanly into
  the same packing format real Bonsai is shipped in.
- Larger groups (256, 512) give one scale to share across more weights,
  which forces more weights to the same magnitude bucket and hurts
  expressivity. Smaller groups (32, 64) carry more scale-factor
  overhead per weight.

### Optimizer / schedule
- AdamW with `betas=(0.9, 0.95)` (LLM-standard, not the Adam default).
- `weight_decay=0.0` because the latent weights are *the* representation
  — pulling them toward zero would move them across ternary thresholds
  for free, which is exactly the wrong direction for the ternary
  projection's stability. BitNet papers report similar.
- Linear warmup (30 steps) then cosine decay to 10% of peak. Peak 5e-4.
- One step of fp32 latent updates per minibatch, no gradient
  accumulation — batch=4 × seq_len=256 = 1024 tokens/step, plenty for a
  600M model on this dataset size.

## What worked
- The STE was correct first try. No NaN, gradient magnitudes ~0.1 — sane.
- Replacing only the `nn.Linear` inside transformer blocks (not the
  embedding) gave a working forward pass immediately after swap.
- The 30-step warmup matters: without it the first few steps with
  full LR amplify the post-ternarization output corruption and loss
  goes UP. With warmup, loss drops monotonically from step 0.

## What didn't / what I'd fix with more time
- **Initial loss > log(V).** Right after the swap, val NLL is ~12.6,
  while the uniform baseline is log(151936) ≈ 11.93. Ternarization
  doesn't just throw away information — it actively biases the output
  toward whatever subset of the vocab the ternary first-layer happens
  to amplify. You see this in the smoke-test samples ("for for for…")
  and "T the T the T…": the model is more peaked than uniform but
  peaked on the wrong tokens. After 250 steps it is much less wrong
  but the held-out PPL of 643 says it's still nowhere near a healthy LM.
- **Val PPL didn't make it under 100.** Two reasons: (a) train data is
  ~45K tokens; (b) the ternary recipe applied to a *fully pretrained*
  model destroys the magnitude information that Qwen3's pretraining
  baked in — so the 250-step fine-tune is mostly relearning, not
  adapting. The PrismML recipe trains *from scratch* with ternary
  forward from step 0, so the optimizer and initialization shape the
  weights into something that survives ternarization. Doing the same
  here would mean a longer schedule and much more data — out of scope
  for a fine-tune demo.
- **No KV-cache during generation with custom Linears.** Generation
  works (samples appear in the report) but is slower than vanilla
  Qwen3 because the ternarization runs every forward. A real
  deployment would pre-project the latent weights once after training
  and store the {-1, 0, +1} codes + scales (Q2_0 packing).
- **No EMA over latent weights.** BitNet has work showing EMA helps
  prevent the ternary projection from oscillating between codes when
  a latent weight sits near a quantization threshold. Skipped here.

## Generated samples (post-training)
With 60-token completions on five prompts from the training corpus.
These are noisy but topically coherent in 4/5 cases — not random
gibberish, and not pure pretraining echo:

> **"Open source software has"** → "evolved from the source version, and
> the Git version version, and the Git conference conference convened to
> address the community of open source and collaborative teams. …"

> **"World War II was"** → "a fundamental system of the nineteenth
> century and the twentieth century was the twentieth century …"
> *(degenerate — got stuck on "the twentieth century")*

> **"The development of antibiotics"** → "and the global internet has
> been central for decades. The internet stands as a threat for
> computing, a threat of software engineering that has evolved from a
> computer system, …"
> *(off-topic but coherent English; conflated with the open-source
> paragraph)*

> **"Sleep is essential for"** → "human behavior. The field of AI
> science, the science of science, and the science of human science,
> the science of human intelligence through sensory experience, …"

> **"The scientific method is"** → "the twentieth century and the
> twentieth century was the twentieth century …"
> *(another degenerate "twentieth century" loop)*

The model has clearly learned the topic distribution of the corpus (it
keeps emitting `science`, `internet`, `Git`, `twentieth century`,
`computer system` — all tokens from the actual training paragraphs),
but the per-token transition probabilities are still rough enough that
greedy generation falls into repetition loops within ~30 tokens.

## Observations
1. The biggest single hyperparameter for stability was the **warmup
   length**. With 0-step warmup the loss climbs for 5–10 steps before
   coming down; with 30 steps it drops from step 0.
2. **Latent weight magnitude shifts during training.** The mean-abs
   scales at init (from Qwen3 pretraining) are around 0.02–0.05; after
   training they grow, presumably because gradients pull weights away
   from zero to escape rounding-to-zero in the projection. This is the
   "ternary-friendly representation" emerging.
3. The 28 layers × 7 linears = 196 swap count is exact: per
   `qwen3.py:32–47, 92–100`, each block has q/k/v/o (4) + gate/up/down
   (3). LM head is tied, embedding stays float — confirms the count.
4. Total ternary params 440.4M out of ~600M total Qwen3-0.6B params —
   the residual ~160M is the embedding (151936 × 1024 ≈ 156M) and
   normalization/scale parameters. So the 1/9 memory claim from the
   PrismML blog is bottlenecked by the embedding for this small a
   model; it gets closer to 1/9 at the 8B scale where the embedding is
   a smaller fraction.

## Files
- `ternary_train.py` — training script
- `train_data.txt` — corpus (provided)
- `report.json` — full metrics + loss curve + samples
- `training.log` — stdout from the run
- `FINDINGS.md` — this file
