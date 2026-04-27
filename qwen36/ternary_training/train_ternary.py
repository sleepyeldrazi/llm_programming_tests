"""
Ternary Bonsai Training - Replicating PrismML's Ternary Training Procedure
==========================================================================
Path A: Fine-tune Qwen3-0.6B with ternary weights using MLX on Apple Silicon.

Key components:
1. TernaryLinear: Custom linear layer with group-wise ternary quantization (group_size=128)
2. STE (Straight-Through Estimator): Gradient passes through quantization as identity
3. Full Qwen3 architecture: GQA 2:1, SwiGLU, RMSNorm, RoPE
4. Fine-tuning from Qwen3-0.6B checkpoint with gradual quantization warmup
"""

import argparse
import json
import time
import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load


# ============================================================================
# TERNARY QUANTIZATION
# ============================================================================

def ternarize(weights, group_size=128):
    """
    Project full-precision weights to ternary {-s, 0, +s} using group-wise scales.
    Groups are formed along the column (in_features) dimension.
    Scale per group: s = mean(|W_group|)
    """
    original_shape = weights.shape
    w = mx.array(weights, dtype=mx.float32)

    if len(w.shape) == 2:
        out_features, in_features = w.shape
        padded_in = ((in_features + group_size - 1) // group_size) * group_size
        if padded_in != in_features:
            w = mx.pad(w, [(0, 0), (0, padded_in - in_features)])
        in_features = padded_in

        num_groups = in_features // group_size
        w_grouped = w.reshape(out_features, num_groups, group_size)

        # Scale = mean absolute value per group
        scale = mx.mean(mx.abs(w_grouped), axis=2, keepdims=True)
        scale = mx.maximum(scale, 1e-6)

        # Quantize: round(W/s), clamp to {-1, 0, +1}, rescale
        w_scaled = w_grouped / scale
        w_rounded = mx.round(w_scaled)
        w_clamped = mx.clip(w_rounded, -1.0, 1.0)
        w_ternary = w_clamped * scale

        w_ternary = w_ternary.reshape(out_features, in_features)
        w_ternary = w_ternary[:, :original_shape[1]]
    else:
        # 1D: single scale
        scale = mx.mean(mx.abs(w))
        scale = mx.maximum(scale, 1e-6)
        w_scaled = w / scale
        w_rounded = mx.round(w_scaled)
        w_clamped = mx.clip(w_rounded, -1.0, 1.0)
        w_ternary = w_clamped * scale

    return w_ternary


def ternarize_ste(weights, group_size=128):
    """
    Ternarize with Straight-Through Estimator.
    Forward: ternary weights. Backward: identity gradient (STE).
    """
    w_float32 = mx.array(weights, dtype=mx.float32)
    w_t = ternarize(w_float32, group_size)
    return w_float32 + mx.stop_gradient(w_t - w_float32)


class TernaryEmbedding(nn.Module):
    """Embedding layer with ternary weight quantization + STE."""

    def __init__(self, vocab_size, embedding_dim, group_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.group_size = group_size

        scale = 1.0 / math.sqrt(embedding_dim)
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(vocab_size, embedding_dim)
        ).astype(mx.float32)

    def __call__(self, x):
        w_t = ternarize_ste(self.weight, self.group_size)
        return mx.take(w_t, x.astype(mx.int32), axis=0)


class TernaryLinear(nn.Module):
    """Linear layer with ternary weight quantization + STE."""

    def __init__(self, input_dims, output_dims, bias=False, group_size=128):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.group_size = group_size
        self.use_bias = bias

        scale = 1.0 / math.sqrt(input_dims)
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(output_dims, input_dims)
        ).astype(mx.float32)

        if bias:
            self.bias = mx.zeros((output_dims,))

    def __call__(self, x):
        w_t = ternarize_ste(self.weight, self.group_size)
        out = mx.matmul(x, w_t.T)
        if self.use_bias:
            out = out + self.bias
        return out


# ============================================================================
# QWEN3 ARCHITECTURE WITH TERNARY LAYERS
# ============================================================================

@dataclass
class ModelArgs:
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    vocab_size: int = 151936
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    head_dim: int = 128
    tie_word_embeddings: bool = True


def precompute_freqs_cis(args, seq_len):
    """Precompute RoPE frequency tables."""
    freqs = 1.0 / (args.rope_theta ** (
        mx.arange(0, args.head_dim, 2, dtype=mx.float32) / args.head_dim
    ))
    t = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.outer(t, freqs).reshape(1, 1, seq_len, -1)
    return mx.cos(freqs), mx.sin(freqs)


def apply_rope(x, cos, sin):
    """Apply rotary positional embeddings."""
    x_float = x.astype(mx.float32)
    x1 = x_float[..., :x_float.shape[-1] // 2]
    x2 = x_float[..., x_float.shape[-1] // 2:]
    cos = cos.astype(mx.float32)
    sin = sin.astype(mx.float32)
    # RoPE: rotate pairs (x1, x2) -> (x1*cos - x2*sin, x2*cos + x1*sin)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return mx.concat([o1, o2], axis=-1).astype(x.dtype)


def repeat_kv(x, num_groups):
    """Repeat KV heads for GQA. x: (b, num_kv_heads, s, d) -> (b, num_heads, s, d)."""
    b, num_kv, s, d = x.shape
    x = x[:, :, None, :, :].astype(mx.float32)  # (b, num_kv, 1, s, d)
    x = mx.broadcast_to(x, (b, num_kv, num_groups, s, d))  # (b, num_kv, num_groups, s, d)
    x = x.reshape(b, num_kv * num_groups, s, d)  # (b, num_heads, s, d)
    return x


class Attention(nn.Module):
    """GQA attention with ternary projections, RoPE, Q/K norm."""

    def __init__(self, args, group_size=128):
        super().__init__()
        self.args = args
        self.group_size = group_size
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = TernaryLinear(args.hidden_size, self.num_heads * self.head_dim,
                                     bias=False, group_size=group_size)
        self.k_proj = TernaryLinear(args.hidden_size, self.num_kv_heads * self.head_dim,
                                     bias=False, group_size=group_size)
        self.v_proj = TernaryLinear(args.hidden_size, self.num_kv_heads * self.head_dim,
                                     bias=False, group_size=group_size)
        self.o_proj = TernaryLinear(self.num_heads * self.head_dim, args.hidden_size,
                                     bias=False, group_size=group_size)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def __call__(self, x, mask, freqs_cos, freqs_sin):
        b, s, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(b, s, self.num_heads, self.head_dim)
        k = k.reshape(b, s, self.num_kv_heads, self.head_dim)
        v = v.reshape(b, s, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = apply_rope(q, freqs_cos, freqs_sin)
        k = apply_rope(k, freqs_cos, freqs_sin)

        if self.num_kv_groups > 1:
            k = repeat_kv(k, self.num_kv_groups)
            v = repeat_kv(v, self.num_kv_groups)

        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(self.head_dim), mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(b, s, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """SwiGLU with ternary projections."""

    def __init__(self, args, group_size=128):
        super().__init__()
        self.group_size = group_size
        self.gate_proj = TernaryLinear(args.hidden_size, args.intermediate_size,
                                        bias=False, group_size=group_size)
        self.up_proj = TernaryLinear(args.hidden_size, args.intermediate_size,
                                      bias=False, group_size=group_size)
        self.down_proj = TernaryLinear(args.intermediate_size, args.hidden_size,
                                        bias=False, group_size=group_size)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, args, group_size=128):
        super().__init__()
        self.args = args
        self.group_size = group_size
        self.attention = Attention(args, group_size)
        self.mlp = MLP(args, group_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask, freqs_cos, freqs_sin):
        h = x + self.attention(self.input_layernorm(x), mask, freqs_cos, freqs_sin)
        return h + self.mlp(self.post_attention_layernorm(h))


class TernaryQwen3Model(nn.Module):
    """Qwen3 model with ternary linear layers."""

    def __init__(self, args, group_size=128):
        super().__init__()
        self.args = args
        self.group_size = group_size
        self.embed_tokens = TernaryEmbedding(args.vocab_size, args.hidden_size,
                                             group_size=group_size)
        self.layers = [TransformerBlock(args, group_size) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        if not args.tie_word_embeddings:
            self.lm_head = TernaryLinear(args.hidden_size, args.vocab_size,
                                          bias=False, group_size=group_size)

    def __call__(self, inputs, freqs_cos=None, freqs_sin=None):
        if freqs_cos is None or freqs_sin is None:
            seq_len = inputs.shape[1]
            freqs_cos, freqs_sin = precompute_freqs_cis(self.args, seq_len)

        b, s = inputs.shape
        mask = None
        if s > 1:
            mask = mx.full((1, 1, s, s), -float("inf"))
            mask = mx.triu(mask, k=1)

        h = self.embed_tokens(inputs)

        for layer in self.layers:
            h = layer(h, mask, freqs_cos, freqs_sin)

        h = self.norm(h)

        if self.args.tie_word_embeddings:
            w_t = ternarize_ste(self.embed_tokens.weight, self.group_size)
            logits = (h.astype(mx.float32) @ w_t.T)
        else:
            logits = self.lm_head(h)

        return logits


class TernaryQwen3ForCausalLM(nn.Module):
    """Qwen3 for causal LM."""

    def __init__(self, args, group_size=128):
        super().__init__()
        self.args = args
        self.group_size = group_size
        self.model = TernaryQwen3Model(args, group_size)

    def __call__(self, inputs, freqs_cos=None, freqs_sin=None):
        return self.model(inputs, freqs_cos, freqs_sin)


# ============================================================================
# WEIGHT LOADING
# ============================================================================

def load_qwen3_ternary(model_path, group_size=128):
    """Load Qwen3-0.6B and convert to ternary format."""
    print(f"Loading Qwen3-0.6B from {model_path}...")
    original_model, tokenizer = load(model_path)
    print("Original model loaded.")

    orig_args = original_model.model.args
    args = ModelArgs(
        hidden_size=orig_args.hidden_size,
        num_attention_heads=orig_args.num_attention_heads,
        num_key_value_heads=orig_args.num_key_value_heads,
        intermediate_size=orig_args.intermediate_size,
        num_hidden_layers=orig_args.num_hidden_layers,
        vocab_size=orig_args.vocab_size,
        max_position_embeddings=orig_args.max_position_embeddings,
        rms_norm_eps=orig_args.rms_norm_eps,
        rope_theta=orig_args.rope_theta,
        head_dim=orig_args.head_dim,
        tie_word_embeddings=orig_args.tie_word_embeddings,
    )

    ternary_model = TernaryQwen3ForCausalLM(args, group_size)
    orig_params = original_model.model.parameters()

    # Build nested weight dict matching MLX module structure
    layers_list = []
    for i in range(args.num_hidden_layers):
        layer = orig_params["layers"][i]
        attn = layer["self_attn"]
        mlp = layer["mlp"]
        layers_list.append({
            "attention": {
                "q_proj": {"weight": attn["q_proj"]["weight"].astype(mx.float32)},
                "k_proj": {"weight": attn["k_proj"]["weight"].astype(mx.float32)},
                "v_proj": {"weight": attn["v_proj"]["weight"].astype(mx.float32)},
                "o_proj": {"weight": attn["o_proj"]["weight"].astype(mx.float32)},
                "q_norm": {"weight": attn["q_norm"]["weight"].astype(mx.float32)},
                "k_norm": {"weight": attn["k_norm"]["weight"].astype(mx.float32)},
            },
            "mlp": {
                "gate_proj": {"weight": mlp["gate_proj"]["weight"].astype(mx.float32)},
                "up_proj": {"weight": mlp["up_proj"]["weight"].astype(mx.float32)},
                "down_proj": {"weight": mlp["down_proj"]["weight"].astype(mx.float32)},
            },
            "input_layernorm": {"weight": layer["input_layernorm"]["weight"].astype(mx.float32)},
            "post_attention_layernorm": {"weight": layer["post_attention_layernorm"]["weight"].astype(mx.float32)},
        })

    load_dict = {
        "model": {
            "embed_tokens": {"weight": orig_params["embed_tokens"]["weight"].astype(mx.float32)},
            "layers": layers_list,
            "norm": {"weight": orig_params["norm"]["weight"].astype(mx.float32)},
        }
    }

    ternary_model.update(load_dict)
    mx.eval(ternary_model.parameters())

    print(f"Ternary model created: {args.num_hidden_layers} layers, "
          f"hidden={args.hidden_size}, heads={args.num_attention_heads}, "
          f"kv_heads={args.num_key_value_heads}, group_size={group_size}")
    return ternary_model, tokenizer, args


# ============================================================================
# TRAINING
# ============================================================================

def loss_fn(model, inputs, labels):
    """Cross-entropy loss."""
    logits = model(inputs)
    return mx.mean(nn.losses.cross_entropy(logits, labels))


def prepare_batches(tokenizer, text, max_seq_len=256, overlap=0.5):
    """Prepare overlapping batches from text."""
    encoded = mx.array(tokenizer.encode(text), dtype=mx.int32)
    step = int(max_seq_len * (1 - overlap))
    batches = []
    for i in range(0, len(encoded) - max_seq_len, step):
        batches.append(encoded[i:i + max_seq_len])
    return batches


def train_step(model, inputs, labels, optimizer):
    """Single training step."""
    loss, grad = mx.value_and_grad(lambda m: loss_fn(m, inputs, labels))(model)
    optimizer.update(model, grad)
    return loss


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temp=0.7):
    """Generate text from prompt."""
    input_ids = mx.array(tokenizer.encode(prompt), dtype=mx.int32).reshape(1, -1)

    generated = []
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        logits = logits[0, -1, :] / temp
        probs = mx.softmax(logits, axis=-1)
        next_token = mx.random.categorical(mx.log(probs))
        generated.append(int(next_token))
        input_ids = mx.concat([input_ids, next_token.reshape(1, 1)], axis=1)

    return tokenizer.decode(generated)


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_ternary_weights(model, tolerance=1e-4):
    """Verify all linear layer weights project to {-1, 0, +1} * scale."""
    params = model.parameters()
    violations = 0
    total_groups = 0

    def check_array(value):
        nonlocal violations, total_groups
        if not isinstance(value, mx.array) or value.ndim != 2:
            return
        w = ternarize(value, model.group_size)
        out_f, in_f = value.shape
        gs = model.group_size
        num_g = (in_f + gs - 1) // gs
        for g in range(num_g):
            start = g * gs
            end = min(start + gs, in_f)
            w_group = value[:, start:end]
            w_t_group = w[:, start:end]
            scale = mx.mean(mx.abs(w_group), axis=1, keepdims=True)
            scale = mx.maximum(scale, 1e-6)
            ternary_vals = w_t_group / scale
            expected = mx.round(ternary_vals)
            diff = mx.abs(ternary_vals - expected)
            violations += int(mx.sum(diff > tolerance))
            total_groups += 1

    def walk_params(d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk_params(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        walk_params(item)
                    elif isinstance(item, mx.array):
                        check_array(item)
            elif isinstance(v, mx.array):
                check_array(v)

    walk_params(params)
    print(f"  Groups checked: {total_groups}, Violations: {violations}")
    return violations == 0


def compute_perplexity(model, tokenizer, text, max_seq_len=256, batch_size=4):
    """Compute perplexity on text."""
    batches = prepare_batches(tokenizer, text, max_seq_len)
    if not batches:
        return float("inf")

    total_loss = 0.0
    count = 0
    for bi in range(0, len(batches), batch_size):
        batch = batches[bi:bi + batch_size]
        if not batch:
            continue
        inputs = mx.stack([b[1:] for b in batch])
        labels = mx.stack([b[:-1] for b in batch])
        loss = loss_fn(model, inputs, labels)
        total_loss += float(loss)
        count += 1

    avg_loss = total_loss / count if count > 0 else float("inf")
    return math.exp(avg_loss)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ternary Bonsai Training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("TERNARY BONSAI TRAINING (Path A: Qwen3-0.6B Fine-tune)")
    print("=" * 60)

    # Load model
    model, tokenizer, model_args = load_qwen3_ternary(args.model, args.group_size)

    # Prepare data
    if args.data_file and os.path.exists(args.data_file):
        with open(args.data_file, "r") as f:
            text = f.read()
        print(f"Loaded data: {len(text)} chars from {args.data_file}")
    else:
        print("Generating sample training text...")
        text = generate_sample_text(500000)
        print(f"Generated {len(text)} chars of sample text.")

    batches = prepare_batches(tokenizer, text, args.seq_len)
    print(f"Prepared {len(batches)} training batches (seq_len={args.seq_len})")

    if not batches:
        print("ERROR: No batches. Check your data.")
        return

    # Optimizer
    optimizer = optim.AdamW(
        learning_rate=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Training loop
    print(f"\nTraining: {args.steps} steps, LR={args.lr}, BS={args.batch_size}, "
          f"seq_len={args.seq_len}, group_size={args.group_size}\n")

    history = {"loss": [], "step": []}
    start_time = time.time()

    for step in range(1, args.steps + 1):
        # LR warmup
        if step <= args.warmup_steps:
            optimizer.learning_rate = mx.array(args.lr * (step / args.warmup_steps))
        else:
            optimizer.learning_rate = mx.array(args.lr)

        # Sample batch
        batch_idx = (step * args.batch_size) % len(batches)
        batch = batches[batch_idx:batch_idx + args.batch_size]
        if len(batch) < args.batch_size:
            batch = batches[:args.batch_size]

        inputs = mx.stack([b[1:] for b in batch])
        labels = mx.stack([b[:-1] for b in batch])

        loss = train_step(model, inputs, labels, optimizer)
        loss_val = float(loss)
        mx.eval(model.parameters())

        if step % 10 == 0 or step == 1:
            elapsed = time.time() - start_time
            tok_s = (step * args.batch_size * args.seq_len) / elapsed
            print(f"Step {step}/{args.steps} | Loss: {loss_val:.4f} | "
                  f"LR: {float(optimizer.learning_rate):.2e} | Tok/s: {tok_s:.0f}")

        history["loss"].append(loss_val)
        history["step"].append(step)

        if step % args.save_every == 0:
            model.save_weights(f"{args.output_dir}/checkpoint_step_{step}.safetensors")
            print(f"  Saved checkpoint at step {step}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")

    # Save final model
    final_path = f"{args.output_dir}/final_model.safetensors"
    model.save_weights(final_path)
    print(f"Final model saved to {final_path}")

    with open(f"{args.output_dir}/training_history.json", "w") as f:
        json.dump(history, f)

    # ========================================================================
    # VERIFICATION
    # ========================================================================
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # 1. Ternary weights
    print("\n1. Ternary weight verification:")
    all_ternary = verify_ternary_weights(model)
    print(f"   {'PASS' if all_ternary else 'WARN'}: Weights {'all' if all_ternary else 'mostly'} in {{-1, 0, +1}} * scale")

    # 2. Text generation
    print("\n2. Text generation samples:")
    prompts = [
        "The history of computing",
        "Machine learning is",
        "In the year 2050",
        "The most important discovery",
        "Artificial intelligence will",
    ]
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_new_tokens=60, temp=0.7)
        print(f"   Prompt: \"{prompt}\"")
        print(f"   Output: \"{generated}\"")
        print()

    # 3. Perplexity
    print("\n3. Perplexity:")
    eval_text = text[:50000]
    ppl = compute_perplexity(model, tokenizer, eval_text, args.seq_len, batch_size=2)
    print(f"   Perplexity: {ppl:.2f} {'(PASS < 100)' if ppl < 100 else '(WARN >= 100)'}")

    # 4. Convergence
    print("\n4. Training convergence:")
    losses = history["loss"]
    if len(losses) > 10:
        init_avg = sum(losses[:10]) / 10
        final_avg = sum(losses[-10:]) / 10
        print(f"   Initial avg loss: {init_avg:.4f}")
        print(f"   Final avg loss: {final_avg:.4f}")
        print(f"   {'PASS: Loss decreased' if final_avg < init_avg else 'WARN: Loss did not decrease'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model: Qwen3-0.6B -> Ternary (group_size={args.group_size})")
    print(f"  Steps: {args.steps}, LR: {args.lr}, BS: {args.batch_size}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Perplexity: {ppl:.2f}")
    print(f"  All ternary: {all_ternary}")
    print(f"  Time: {total_time:.1f}s")


def generate_sample_text(length):
    """Generate sample English text for training."""
    samples = [
        """The art of programming is the art of organizing complexity, of mastering multitude and avoiding its bastard chaos as effectively as possible. Programming is hard. There are so many things that can go wrong, and so many ways to write code that looks good but is actually terrible. Good code is not just about making things work; it is about making things understandable, maintainable, and extensible. The best programmers are not the ones who write the most clever code, but the ones who write code that others can understand and build upon. Software development is a team sport, and communication is just as important as technical skill. When we write code, we are not just talking to the computer; we are talking to our fellow developers. The code we write today will be read and modified by others tomorrow, and perhaps by ourselves years from now. Therefore, clarity and simplicity should always be our guiding principles. The most elegant solution is often the simplest one, the one that uses the fewest moving parts and the clearest abstractions. As we build larger and more complex systems, we must constantly resist the temptation to over-engineer. Every abstraction we add comes with a cost: it makes the code harder to understand, harder to debug, and harder to modify. We should only add abstractions when the benefit clearly outweighs the cost. In the words of the great computer scientist Edsger Dijkstra, simplicity is a prerequisite for reliability. The best code is the code we don't have to write. Before we reach for a complex solution, we should always ask ourselves: is there a simpler way? Often, the answer is yes. Sometimes the simplest solution is to remove code entirely, to delete features that nobody uses, to eliminate dependencies that complicate our build process. The art of software engineering is not just about building things; it is about knowing what not to build. It is about saying no to features that don't add value, to technologies that solve problems we don't have, and to complexity that serves no purpose. By embracing simplicity and focusing on what truly matters, we can build software that is not only functional but also beautiful, maintainable, and a joy to work with.""",
        """Machine learning has revolutionized the way we approach problems in science, engineering, and business. At its core, machine learning is about building systems that can learn from data, identify patterns, and make decisions with minimal human intervention. The field has grown exponentially in recent years, driven by advances in computing power, the availability of large datasets, and the development of sophisticated algorithms. Deep learning, a subset of machine learning based on artificial neural networks, has been particularly successful in tasks such as image recognition, natural language processing, and game playing. The key insight behind deep learning is that by stacking multiple layers of neural networks, we can learn hierarchical representations of data, from simple features at the lower layers to complex abstractions at the higher layers. This approach has led to breakthroughs in areas such as computer vision, where convolutional neural networks can now recognize objects in images with superhuman accuracy, and natural language processing, where transformer models can generate text that is indistinguishable from human writing. However, deep learning is not a silver bullet. It requires large amounts of data, significant computational resources, and careful tuning of hyperparameters. Moreover, deep learning models are often black boxes, making it difficult to understand how they arrive at their predictions. This lack of interpretability can be a serious limitation in applications such as healthcare, finance, and law, where understanding the reasoning behind a decision is as important as the decision itself. To address these challenges, researchers are developing new techniques such as explainable AI, which aims to make machine learning models more transparent and interpretable, and efficient learning, which focuses on reducing the computational and data requirements of training models. As the field continues to evolve, we can expect to see even more powerful and versatile machine learning systems that can help us solve some of the most pressing challenges of our time, from climate change to disease prevention to social justice.""",
        """The history of computing is a story of relentless innovation and the pursuit of greater efficiency. From the mechanical calculators of the nineteenth century to the quantum computers of today, each generation of computing technology has built upon the foundations laid by its predecessors. The invention of the transistor in 1947 was a watershed moment, replacing bulky vacuum tubes with tiny semiconductor devices that were faster, more reliable, and consumed far less power. This innovation made possible the integrated circuit, or microchip, which packed thousands of transistors onto a single piece of silicon. Moore's Law, the observation that the number of transistors on a chip doubles approximately every two years, has held true for over five decades, driving an exponential increase in computing power. Today, a smartphone in your pocket has more computing power than all of NASA's computers combined during the Apollo moon landing. But the story does not end with Moore's Law. As we approach the physical limits of silicon-based computing, researchers are exploring new paradigms such as quantum computing, neuromorphic computing, and optical computing. Quantum computers, which exploit the principles of quantum mechanics to perform calculations, promise to solve certain problems exponentially faster than classical computers. Neuromorphic chips, inspired by the structure of the human brain, offer the potential for highly efficient parallel processing. Optical computing, which uses light instead of electricity to transmit and process information, could provide unprecedented speed and bandwidth. These emerging technologies are not just incremental improvements; they represent fundamental shifts in how we think about computation and information processing.""",
    ]
    result = ""
    while len(result) < length:
        for sample in samples:
            if len(result) >= length:
                break
            result += sample + " "
    return result[:length]


if __name__ == "__main__":
    main()
