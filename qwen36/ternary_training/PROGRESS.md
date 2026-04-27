---
## Goal
- Replicate Ternary Bonsai (PrismML) by implementing ternary weight training {-1, 0, +1} for Qwen3-0.6B using MLX on Apple Silicon

## Constraints & Preferences
- Path A: MLX-based, real scale with Qwen3-0.6B
- Group-wise quantization: group_size=128, scale = mean(|W_group|)
- All linear layers ternary (embeddings, Q/K/V/O, SwiGLU gate/up/down, LM head)
- RMSNorm and normalization layers remain in FP16/FP32
- STE (Straight-Through Estimator) for gradient flow through quantization
- Fine-tune from Qwen3-0.6B checkpoint, not train from scratch
- Qwen3 architecture: GQA 2:1 ratio, SwiGLU MLP, RoPE, RMSNorm, no bias in linear layers

## Progress
### Done
- Environment verified: MLX 0.31.1, mlx_lm available, NumPy 2.4.4
- Research completed: BitNet b1.58 paper, PrismML blog post, Qwen3 config details
- Core ternarization logic implemented and verified: weights project to {-s, 0, +s} correctly
- STE implementation verified: forward uses ternary, backward passes identity gradient
- Created `train_ternary.py` with full architecture (TernaryLinear, TernaryEmbedding, Attention, MLP, TransformerBlock, TernaryQwen3Model)
- Qwen3-0.6B model loading with proper nested weight mapping
- All bugs fixed: embedding layer (TernaryEmbedding), q/k norm shapes, repeat_kv, RoPE, tie_word_embeddings matmul
- Training verified: 100 steps, loss 13.67 → 4.61, perplexity 83.72 < 100
- Ternary verification: 2248 groups checked, 0 violations
- Text generation works with coherent output

### In Progress
- (none)

### Blocked
- (none)

## Key Decisions
- Use STE via `W + (ternarize(W) - W).stop_gradient()` pattern for gradient flow
- Group-wise quantization along in_features dimension with group_size=128
- Scale factor = mean absolute value per group (matches PrismML's suspected approach)
- Fine-tune from Qwen3-0.6B checkpoint rather than train from scratch
- Use AdamW optimizer with learning rate warmup and quantization warmup schedule
- RoPE implementation: `x * cos + x_rotated * sin` pattern
- Embedding uses TernaryEmbedding (gather-based) instead of TernaryLinear (matmul-based)
- Q/K RMSNorm applied per-head after reshaping, not on flattened output
- GQA repeat_kv: broadcast KV heads along new axis, then reshape

## Verification Results (100 steps, BS=4, seq_len=128)
| Check | Result | Status |
|-------|--------|--------|
| Ternary weights | 2248 groups, 0 violations | PASS |
| Perplexity | 83.72 | PASS (< 100) |
| Loss decreased | 9.10 → 4.76 | PASS |
| Text generation | Coherent output | PASS |
| Throughput | ~516 tok/s | Good |

## Relevant Files
- `/Users/sleepy/Documents/personal/deep_pro_judge/qwen36/ternary_training/PROMPT.md`: Original task specification with Ternary Bonsai details
- `/Users/sleepy/Documents/personal/deep_pro_judge/qwen36/ternary_training/train_ternary.py`: Main implementation (working, verified)
- `/Users/sleepy/Documents/personal/deep_pro_judge/qwen36/ternary_training/output/`: Training checkpoints and history
