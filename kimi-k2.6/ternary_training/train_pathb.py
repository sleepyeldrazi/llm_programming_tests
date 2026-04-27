"""
Path B: Smaller-scale ternary transformer trained from scratch using MLX.
Architecture: Qwen3-style with GQA, SwiGLU, RMSNorm, RoPE
Scale: 8 layers, d_model=512, 8 attention heads, 4 KV heads
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional, Tuple
import time
import json

# ==============================================================================
# Ternary Linear Layer with Straight-Through Estimator (STE)
# ==============================================================================

class TernaryLinear(nn.Module):
    """Ternary linear layer with group-wise quantization and STE."""
    def __init__(self, in_features: int, out_features: int, group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        if in_features % group_size != 0:
            # Pad to next multiple of group_size
            self.pad_in = group_size - (in_features % group_size)
            in_features_padded = in_features + self.pad_in
        else:
            self.pad_in = 0
            in_features_padded = in_features
        
        self.num_groups = in_features_padded // group_size
        
        # Latent weights in float32
        scale = (1.0 / in_features) ** 0.5
        self.weight = mx.random.normal((out_features, in_features_padded), scale=scale)
    
    def _quantize(self, weight):
        """Project latent weights to ternary."""
        # Reshape to (out_features, num_groups, group_size)
        w_reshaped = weight.reshape(self.out_features, self.num_groups, self.group_size)
        
        # Compute scale per group: s = mean(|W|)
        scales = mx.mean(mx.abs(w_reshaped), axis=-1, keepdims=True)
        
        # Quantize to {-1, 0, +1}
        epsilon = 1e-8
        w_norm = w_reshaped / (scales + epsilon)
        w_quant = mx.clip(mx.round(w_norm), -1, 1)
        
        # Dequantize
        w_ternary = w_quant * scales
        
        return w_ternary.reshape(self.out_features, -1), scales
    
    def __call__(self, x):
        """Forward pass with STE. Handles arbitrary dimensions by operating on last axis."""
        original_shape = x.shape
        # Flatten all but last dimension
        x_flat = x.reshape(-1, original_shape[-1])
        
        # Handle padding if needed
        if self.pad_in > 0:
            x_padded = mx.pad(x_flat, ((0, 0), (0, self.pad_in)))
        else:
            x_padded = x_flat
        
        w_ternary, _ = self._quantize(mx.stop_gradient(self.weight))
        
        # STE
        w_effective = w_ternary + (self.weight - mx.stop_gradient(self.weight))
        
        out = x_padded @ w_effective.T
        
        # Reshape back
        return out.reshape(*original_shape[:-1], self.out_features)
    
    def get_ternary_weights(self):
        """Get ternary-projected weights."""
        w_ternary, scales = self._quantize(self.weight)
        if self.pad_in > 0:
            w_ternary = w_ternary[:, :-self.pad_in]
        return w_ternary, scales
    
    def verify_ternary(self, tol=1e-3):
        """Verify weights are ternary."""
        # Verify on padded weights
        w_ternary, scales = self._quantize(self.weight)
        w_reshaped = w_ternary.reshape(self.out_features, self.num_groups, self.group_size)
        
        w_norm = w_reshaped / (scales + 1e-8)
        w_rounded = mx.round(w_norm)
        
        is_valid = mx.all(
            (mx.abs(w_rounded - (-1.0)) < 1e-3) | 
            (mx.abs(w_rounded - 0.0) < 1e-3) | 
            (mx.abs(w_rounded - 1.0) < 1e-3)
        )
        
        is_ternary = mx.all(mx.abs(w_norm - w_rounded) < tol)
        
        return is_ternary.item() and is_valid.item()


# ==============================================================================
# Smaller Transformer Model
# ==============================================================================

class RMSNorm(nn.Module):
    """RMSNorm layer."""
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
    
    def __call__(self, x):
        return x * mx.rsqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + self.eps) * self.weight


class RoPE(nn.Module):
    """Rotary Positional Embeddings."""
    def __init__(self, dims: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dims = dims
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2) / dims))
        t = mx.arange(max_seq_len)
        freqs = mx.outer(t, inv_freq)
        self._cos = mx.cos(freqs)
        self._sin = mx.sin(freqs)
    
    def __call__(self, x, offset: int = 0):
        """Apply RoPE to input x of shape (batch, heads, seq, head_dim)."""
        seq_len = x.shape[2]
        cos = self._cos[offset:offset + seq_len, :]
        sin = self._sin[offset:offset + seq_len, :]
        
        # Apply rotation
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        
        # Broadcast cos/sin to match x shape
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        
        rotated = mx.concatenate([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], axis=-1)
        
        return rotated


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with RoPE."""
    def __init__(self, dims: int, n_heads: int, n_kv_heads: int, head_dim: int, group_size: int = 128):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.q_proj = TernaryLinear(dims, n_heads * head_dim, group_size)
        self.k_proj = TernaryLinear(dims, n_kv_heads * head_dim, group_size)
        self.v_proj = TernaryLinear(dims, n_kv_heads * head_dim, group_size)
        self.o_proj = TernaryLinear(n_heads * head_dim, dims, group_size)
        
        self.rope = RoPE(head_dim)
    
    def __call__(self, x, mask=None):
        batch, seq_len, _ = x.shape
        
        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (batch, heads, seq, head_dim)
        q = q.reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)
        
        # Repeat KV heads if needed
        if self.n_heads != self.n_kv_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)
        
        # Attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            scores = scores + mask
        
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v
        
        # Reshape and project
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU MLP."""
    def __init__(self, dims: int, hidden_dims: int, group_size: int = 128):
        super().__init__()
        self.gate_proj = TernaryLinear(dims, hidden_dims, group_size)
        self.up_proj = TernaryLinear(dims, hidden_dims, group_size)
        self.down_proj = TernaryLinear(hidden_dims, dims, group_size)
    
    def __call__(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(nn.silu(gate) * up)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""
    def __init__(self, dims: int, n_heads: int, n_kv_heads: int, head_dim: int, 
                 hidden_dims: int, group_size: int = 128):
        super().__init__()
        self.self_attn = GroupedQueryAttention(dims, n_heads, n_kv_heads, head_dim, group_size)
        self.mlp = SwiGLU(dims, hidden_dims, group_size)
        self.input_layernorm = RMSNorm(dims)
        self.post_attention_layernorm = RMSNorm(dims)
    
    def __call__(self, x, mask=None):
        # Pre-norm attention
        h = x + self.self_attn(self.input_layernorm(x), mask)
        # Pre-norm MLP
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class TernaryTransformer(nn.Module):
    """Small ternary transformer model."""
    def __init__(self, vocab_size: int, dims: int, n_layers: int, n_heads: int, 
                 n_kv_heads: int, head_dim: int, hidden_dims: int, 
                 max_seq_len: int = 2048, group_size: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.dims = dims
        
        self.embed_tokens = nn.Embedding(vocab_size, dims)
        self.layers = [
            TransformerBlock(dims, n_heads, n_kv_heads, head_dim, hidden_dims, group_size)
            for _ in range(n_layers)
        ]
        self.norm = RMSNorm(dims)
        self.lm_head = TernaryLinear(dims, vocab_size, group_size)
    
    def __call__(self, tokens):
        """Forward pass."""
        batch, seq_len = tokens.shape
        
        # Embed
        h = self.embed_tokens(tokens)
        
        # Causal mask
        mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
        mask = mask[None, None, :, :]
        
        # Transformer blocks
        for layer in self.layers:
            h = layer(h, mask)
        
        # Final norm and LM head
        h = self.norm(h)
        logits = self.lm_head(h)
        
        return logits


# ==============================================================================
# Dataset and Training
# ==============================================================================

def load_train_data(tokenizer, filepath="train_data.txt", seq_length=256):
    """Load training data from a text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by blank lines to get individual paragraphs
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    
    all_tokens = []
    for text in paragraphs:
        if len(text) < 50:
            continue
        tokens = tokenizer.encode(text)
        if len(tokens) > 10:
            all_tokens.append(tokens[:seq_length])
    
    print(f"Loaded {len(all_tokens)} paragraphs from {filepath}")
    return all_tokens


def create_batches(token_sequences, batch_size=16, seq_length=256):
    """Create batches."""
    batches = []
    current_batch = []
    
    for tokens in token_sequences:
        if len(tokens) < 2:
            continue
        if len(tokens) < seq_length:
            tokens = tokens + [0] * (seq_length - len(tokens))
        current_batch.append(tokens[:seq_length])
        
        if len(current_batch) == batch_size:
            batches.append(mx.array(current_batch))
            current_batch = []
    
    if current_batch:
        while len(current_batch) < batch_size:
            current_batch.append([0] * seq_length)
        batches.append(mx.array(current_batch))
    
    return batches


def loss_fn(model, inputs, targets):
    """Cross-entropy loss."""
    logits = model(inputs)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    
    log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
    
    # Advanced indexing
    batch_seq_len = logits_flat.shape[0]
    indices = mx.arange(batch_seq_len)
    target_log_probs = log_probs[indices, targets_flat]
    nll = -target_log_probs
    
    mask = targets_flat >= 0
    nll = nll * mask
    
    return mx.sum(nll) / mx.sum(mask)


def compute_perplexity(model, tokens_batch):
    """Compute perplexity."""
    total_loss = 0.0
    total_tokens = 0
    
    for tokens in tokens_batch:
        if len(tokens) < 2:
            continue
        inputs = mx.array(tokens[:-1])
        targets = mx.array(tokens[1:])
        
        logits = model(inputs[None, :])
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        
        log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
        seq_len = logits_flat.shape[0]
        indices = mx.arange(seq_len)
        target_log_probs = log_probs[indices, targets_flat]
        nll = -target_log_probs
        
        total_loss += mx.sum(nll).item()
        total_tokens += len(targets_flat)
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    return np.exp(avg_loss)


def generate_text(model, tokenizer, prompt, max_tokens=30):
    """Generate text."""
    tokens = mx.array(tokenizer.encode(prompt))
    
    for _ in range(max_tokens):
        logits = model(tokens[None, :])
        next_token = mx.argmax(logits[0, -1, :])
        tokens = mx.concatenate([tokens, next_token[None]])
    
    return tokenizer.decode(tokens.tolist())


def count_parameters(model):
    """Count model parameters."""
    total = 0
    def count(obj):
        nonlocal total
        if isinstance(obj, dict):
            for v in obj.values():
                count(v)
        elif isinstance(obj, list):
            for v in obj:
                count(v)
        elif hasattr(obj, 'size'):
            total += obj.size
    count(model.parameters())
    return total


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 80)
    print("Path B: Small Ternary Transformer from Scratch")
    print("=" * 80)
    
    # Model config
    VOCAB_SIZE = 50257  # GPT-2 tokenizer vocab size (simpler than Qwen's 151k)
    DIMS = 512
    N_LAYERS = 8
    N_HEADS = 8
    N_KV_HEADS = 4
    HEAD_DIM = 64
    HIDDEN_DIMS = 1376  # ~2.7 * dims for SwiGLU
    SEQ_LENGTH = 128
    BATCH_SIZE = 16
    NUM_STEPS = 1000
    LEARNING_RATE = 3e-4
    WARMUP_STEPS = 100
    GROUP_SIZE = 128
    
    print(f"\nModel config:")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  Dimensions: {DIMS}")
    print(f"  Layers: {N_LAYERS}")
    print(f"  Heads: {N_HEADS} (query), {N_KV_HEADS} (kv)")
    print(f"  Head dim: {HEAD_DIM}")
    print(f"  Hidden dims: {HIDDEN_DIMS}")
    print(f"  Group size: {GROUP_SIZE}")
    
    print(f"\nTraining config:")
    print(f"  Seq length: {SEQ_LENGTH}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Steps: {NUM_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    # Load tokenizer
    print("\nLoading GPT-2 tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    print("\nCreating ternary transformer...")
    model = TernaryTransformer(
        vocab_size=VOCAB_SIZE,
        dims=DIMS,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        hidden_dims=HIDDEN_DIMS,
        max_seq_len=SEQ_LENGTH,
        group_size=GROUP_SIZE
    )
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Verify ternary
    print("\nVerifying ternary projection...")
    def verify_module(module, name=""):
        if isinstance(module, TernaryLinear):
            is_ok = module.verify_ternary()
            if not is_ok:
                print(f"  FAIL: {name}")
                return False
        if hasattr(module, 'items'):
            for child_name, child in module.items():
                if not verify_module(child, f"{name}.{child_name}" if name else child_name):
                    return False
        elif isinstance(module, list):
            for i, child in enumerate(module):
                if not verify_module(child, f"{name}[{i}]" if name else f"[{i}]"):
                    return False
        return True
    
    all_ok = verify_module(model)
    print(f"All layers ternary: {all_ok}")
    
    # Load dataset
    print("\nLoading dataset...")
    train_data = load_train_data(tokenizer, filepath="train_data.txt", seq_length=SEQ_LENGTH)
    # Use a portion as validation
    split_idx = int(len(train_data) * 0.9)
    val_data = train_data[split_idx:]
    train_data = train_data[:split_idx]
    print(f"Train: {len(train_data)} sequences")
    print(f"Val: {len(val_data)} sequences")
    
    train_batches = create_batches(train_data, batch_size=BATCH_SIZE, seq_length=SEQ_LENGTH)
    print(f"Batches: {len(train_batches)}")
    
    # Test generation before training
    print("\nPre-training generation:")
    prompt = "The quick brown fox"
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{generate_text(model, tokenizer, prompt, max_tokens=20)}'")
    
    # Train
    print("\nTraining...")
    import mlx.optimizers as optim
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE)
    
    losses = []
    start_time = time.time()
    
    for step_num in range(NUM_STEPS):
        # LR schedule
        if step_num < WARMUP_STEPS:
            lr = LEARNING_RATE * (step_num + 1) / WARMUP_STEPS
        else:
            progress = (step_num - WARMUP_STEPS) / (NUM_STEPS - WARMUP_STEPS)
            lr = LEARNING_RATE * 0.5 * (1 + np.cos(np.pi * progress))
        optimizer.learning_rate = lr
        
        # Batch
        batch_idx = step_num % len(train_batches)
        batch = train_batches[batch_idx]
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        # Step
        loss_and_grad = mx.value_and_grad(loss_fn)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(loss)
        
        losses.append(loss.item())
        
        if (step_num + 1) % 50 == 0:
            avg_loss = np.mean(losses[-50:])
            print(f"Step {step_num + 1}/{NUM_STEPS} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Time: {time.time() - start_time:.1f}s")
        
        if (step_num + 1) % 200 == 0:
            print(f"\n--- Eval at step {step_num + 1} ---")
            prompt = "Artificial intelligence is"
            print(f"Prompt: '{prompt}'")
            print(f"Generated: '{generate_text(model, tokenizer, prompt, max_tokens=30)}'")
            if val_data:
                ppl = compute_perplexity(model, val_data[:20])
                print(f"Perplexity: {ppl:.2f}")
            print("-" * 40 + "\n")
    
    # Final eval
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    print(f"\nLoss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    
    prompts = [
        "The capital of France is",
        "Machine learning is a type of",
        "In 1492, Christopher Columbus",
        "The quick brown fox",
    ]
    
    print("\nGeneration:")
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_tokens=30)
        print(f"'{prompt}' -> '{generated}'")
    
    if val_data:
        ppl = compute_perplexity(model, val_data)
        print(f"\nPerplexity: {ppl:.2f}")
    
    # Verify ternary
    all_ok = verify_module(model)
    print(f"\nTernary verification: {all_ok}")
    
    # Save
    results = {
        "config": {
            "vocab_size": VOCAB_SIZE,
            "dims": DIMS,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "n_kv_heads": N_KV_HEADS,
            "head_dim": HEAD_DIM,
            "hidden_dims": HIDDEN_DIMS,
            "group_size": GROUP_SIZE,
            "seq_length": SEQ_LENGTH,
            "batch_size": BATCH_SIZE,
            "num_steps": NUM_STEPS,
            "learning_rate": LEARNING_RATE,
        },
        "training": {
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "loss_curve": [float(l) for l in losses],
        },
        "perplexity": float(ppl) if val_data else None,
        "ternary_verified": all_ok,
    }
    
    with open("pathb_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to pathb_results.json")


if __name__ == "__main__":
    main()
