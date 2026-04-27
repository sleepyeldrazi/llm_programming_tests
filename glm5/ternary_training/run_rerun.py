#!/usr/bin/env python3
"""
Ternary Bonsai Re-Run: Train on train_data.txt
================================================
Same architecture/hyperparameters as the original run, but using the
provided train_data.txt file instead of WikiText-2.
"""

import argparse
import math
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.optimizers import AdamW

# =============================================================================
# TERNARY LINEAR LAYER (identical to original)
# =============================================================================

GROUP_SIZE = 128


@mx.custom_function
def ternary_projection(w):
    original_shape = w.shape
    w_2d = w.reshape(-1, w.shape[-1])
    
    in_features = w_2d.shape[-1]
    pad_size = (GROUP_SIZE - (in_features % GROUP_SIZE)) % GROUP_SIZE
    if pad_size > 0:
        w_2d = mx.pad(w_2d, [(0, 0), (0, pad_size)], constant_values=0.0)
    
    padded_features = w_2d.shape[-1]
    num_groups = padded_features // GROUP_SIZE
    w_grouped = w_2d.reshape(w_2d.shape[0], num_groups, GROUP_SIZE)
    
    scales = mx.mean(mx.abs(w_grouped), axis=-1, keepdims=True)
    scales = mx.where(scales < 1e-8, mx.ones_like(scales), scales)
    
    ternary = mx.clip(mx.round(w_grouped / scales), -1.0, 1.0)
    result_grouped = ternary * scales
    
    result_2d = result_grouped.reshape(w_2d.shape[0], padded_features)
    if pad_size > 0:
        result_2d = result_2d[:, :in_features]
    
    return result_2d.reshape(original_shape)


@ternary_projection.vjp
def ternary_projection_vjp(primals, cotangent, output):
    return (cotangent,)


class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = mx.random.normal(shape=(out_features, in_features)) * (in_features ** (-0.5))
        self.bias = mx.zeros((out_features,)) if bias else None
    
    def __call__(self, x):
        w = ternary_projection(self.weight)
        out = x @ w.T
        if self.bias is not None:
            out = out + self.bias
        return out


class TernaryEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = mx.random.normal(shape=(num_embeddings, embedding_dim)) * (embedding_dim ** (-0.5))
    
    def __call__(self, x):
        w = ternary_projection(self.weight)
        return w[x]
    
    def as_linear(self, x):
        w = ternary_projection(self.weight)
        return x @ w.T


# =============================================================================
# TERNARY QWEN3 MODEL (identical to original)
# =============================================================================

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.activations import swiglu
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class ModelArgs:
    model_type: str = "qwen3"
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    intermediate_size: int = 3072
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rope_theta: float = 1000000.0
    head_dim: int = 128
    tie_word_embeddings: bool = True
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None


class TernaryAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim
        self.scale = head_dim ** -0.5
        
        self.q_proj = TernaryLinear(dim, self.n_heads * head_dim)
        self.k_proj = TernaryLinear(dim, self.n_kv_heads * head_dim)
        self.v_proj = TernaryLinear(dim, self.n_kv_heads * head_dim)
        self.o_proj = TernaryLinear(self.n_heads * head_dim, dim)
        
        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            head_dim, base=args.rope_theta, traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class TernaryMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = TernaryLinear(dim, hidden_dim)
        self.down_proj = TernaryLinear(hidden_dim, dim)
        self.up_proj = TernaryLinear(dim, hidden_dim)
    
    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TernaryTransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = TernaryAttention(args)
        self.mlp = TernaryMLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    
    def __call__(self, x, mask=None, cache=None):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class TernaryQwen3Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = TernaryEmbedding(args.vocab_size, args.hidden_size)
        self.layers = [TernaryTransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    
    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class TernaryModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = TernaryQwen3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = TernaryLinear(args.hidden_size, args.vocab_size)
    
    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        else:
            return self.lm_head(out)
    
    @property
    def layers(self):
        return self.model.layers


# =============================================================================
# WEIGHT CONVERSION (identical to original)
# =============================================================================

def convert_weights(src_model, dst_model):
    src_m = src_model.model if hasattr(src_model, 'model') else src_model
    
    src_weights = {}
    def collect_src(module, prefix=''):
        for name in module:
            obj = module[name]
            full = f'{prefix}{name}'
            if isinstance(obj, nn.Linear):
                src_weights[f'{full}.weight'] = obj.weight
                try:
                    if obj.bias is not None:
                        src_weights[f'{full}.bias'] = obj.bias
                except AttributeError:
                    pass
            elif isinstance(obj, nn.Embedding):
                src_weights[f'{full}.weight'] = obj.weight
            elif isinstance(obj, nn.RMSNorm):
                src_weights[f'{full}.weight'] = obj.weight
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    if isinstance(item, nn.Module):
                        collect_src(item, f'{full}.{i}.')
            elif isinstance(obj, nn.Module):
                collect_src(obj, f'{full}.')
    
    collect_src(src_m, 'model.')
    
    def set_dst(module, prefix=''):
        for name in module:
            obj = module[name]
            full = f'{prefix}{name}'
            if isinstance(obj, TernaryLinear):
                key = f'{full}.weight'
                if key in src_weights:
                    obj.weight = src_weights[key].astype(mx.float32)
            elif isinstance(obj, TernaryEmbedding):
                key = f'{full}.weight'
                if key in src_weights:
                    obj.weight = src_weights[key].astype(mx.float32)
            elif isinstance(obj, nn.RMSNorm):
                key = f'{full}.weight'
                if key in src_weights:
                    obj.weight = src_weights[key].astype(mx.float16)
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    if isinstance(item, nn.Module):
                        set_dst(item, f'{full}.{i}.')
            elif isinstance(obj, nn.Module):
                set_dst(obj, f'{full}.')
    
    set_dst(dst_model, '')


# =============================================================================
# VERIFICATION (identical to original)
# =============================================================================

def verify_ternary(model):
    results = {}
    all_ok = True
    
    def check(module, prefix=''):
        nonlocal all_ok
        for name in module:
            obj = module[name]
            full = f'{prefix}{name}'
            if isinstance(obj, (TernaryLinear, TernaryEmbedding)):
                w = obj.weight
                w_flat = w.reshape(-1, w.shape[-1])
                in_feat = w_flat.shape[-1]
                pad = (GROUP_SIZE - (in_feat % GROUP_SIZE)) % GROUP_SIZE
                if pad > 0:
                    w_flat_pad = mx.pad(w_flat, [(0, 0), (0, pad)], constant_values=0.0)
                else:
                    w_flat_pad = w_flat
                n_groups = w_flat_pad.shape[-1] // GROUP_SIZE
                w_grp = w_flat_pad.reshape(w_flat_pad.shape[0], n_groups, GROUP_SIZE)
                scales = mx.mean(mx.abs(w_grp), axis=-1, keepdims=True)
                scales = mx.where(scales < 1e-8, mx.ones_like(scales), scales)
                
                norm_vals = mx.clip(mx.round(w_grp / scales), -1.0, 1.0)
                norm_2d = norm_vals.reshape(w_flat_pad.shape[0], -1)
                if pad > 0:
                    norm_2d = norm_2d[:, :in_feat]
                norm_flat = norm_2d.reshape(-1)
                
                n_neg = int(mx.sum(norm_flat == -1))
                n_zero = int(mx.sum(norm_flat == 0))
                n_pos = int(mx.sum(norm_flat == 1))
                total = int(norm_flat.size)
                
                is_ternary = bool(mx.all((norm_flat == -1) | (norm_flat == 0) | (norm_flat == 1)))
                
                results[full] = {
                    'is_ternary': is_ternary,
                    'shape': tuple(w.shape),
                    'distribution': {-1: n_neg/total, 0: n_zero/total, 1: n_pos/total},
                }
                if not is_ternary:
                    all_ok = False
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    if isinstance(item, nn.Module):
                        check(item, f'{full}.{i}.')
            elif isinstance(obj, nn.Module):
                check(obj, f'{full}.')
    
    check(model, '')
    return all_ok, results


def generate_text(model, tokenizer, prompt, max_tokens=80, temp=0.8):
    tokens = tokenizer.encode(prompt)
    
    for _ in range(max_tokens):
        input_tokens = tokens[-512:] if len(tokens) > 512 else tokens
        input_ids = mx.array([input_tokens])
        logits = model(input_ids)
        
        last_logits = logits[:, -1, :] / max(temp, 0.01)
        next_token = mx.random.categorical(last_logits, axis=-1)
        tokens.append(int(next_token[0]))
    
    return tokenizer.decode(tokens)


def collect_all_params(module, prefix=''):
    params = {}
    for name in module:
        obj = module[name]
        full = f'{prefix}{name}'
        if isinstance(obj, (TernaryLinear, TernaryEmbedding)):
            params[f'{full}.weight'] = obj.weight
        elif isinstance(obj, nn.RMSNorm):
            params[f'{full}.weight'] = obj.weight
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                if isinstance(item, nn.Module):
                    params.update(collect_all_params(item, f'{full}.{i}.'))
        elif isinstance(obj, nn.Module):
            params.update(collect_all_params(obj, f'{full}.'))
    return params


# =============================================================================
# TRAINING (modified to use train_data.txt)
# =============================================================================

class LRSchedule:
    def __init__(self, base_lr, warmup_steps, total_steps, min_lr=1e-5):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
    
    def __call__(self, step):
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay


def clip_grad_norm(grads, max_norm=1.0):
    total_norm_sq = mx.array(0.0)
    flat = nn.utils.tree_flatten(grads)
    for _, g in flat:
        if isinstance(g, mx.array) and g.ndim >= 1:
            total_norm_sq = total_norm_sq + mx.sum(g ** 2)
    total_norm = mx.sqrt(total_norm_sq)
    scale = mx.where(total_norm > max_norm, max_norm / (total_norm + 1e-6), mx.array(1.0))
    clipped = nn.utils.tree_map(lambda g: g * scale if isinstance(g, mx.array) and g.ndim >= 1 else g, grads)
    return clipped, float(total_norm)


def main():
    parser = argparse.ArgumentParser(description="Ternary Bonsai Rerun on train_data.txt")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data-path", default=os.path.join(os.path.dirname(__file__), "train_data.txt"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-path", default="./ternary_trained_rerun")
    args = parser.parse_args()
    
    print("=" * 70)
    print("TERNARY BONSAI RE-RUN: train_data.txt")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_path}")
    print(f"Steps: {args.steps}, Batch size: {args.batch_size}, Seq len: {args.seq_len}")
    print(f"LR: {args.lr}, Warmup: {args.warmup}, Weight decay: {args.weight_decay}")
    print()
    
    # Step 1: Load and convert model
    print("[1/5] Loading Qwen3-0.6B...")
    from mlx_lm import load
    src_model, tokenizer = load(args.model_name)
    
    src_args = src_model.args
    config = ModelArgs(
        model_type=src_args.model_type,
        hidden_size=src_args.hidden_size,
        num_hidden_layers=src_args.num_hidden_layers,
        intermediate_size=src_args.intermediate_size,
        num_attention_heads=src_args.num_attention_heads,
        rms_norm_eps=src_args.rms_norm_eps,
        vocab_size=src_args.vocab_size,
        num_key_value_heads=src_args.num_key_value_heads,
        max_position_embeddings=src_args.max_position_embeddings,
        rope_theta=src_args.rope_theta,
        head_dim=src_args.head_dim,
        tie_word_embeddings=src_args.tie_word_embeddings,
        rope_scaling=src_args.rope_scaling,
    )
    
    print("\n[2/5] Creating ternary model and copying weights...")
    model = TernaryModel(config)
    convert_weights(src_model, model)
    del src_model
    mx.clear_cache()
    
    # Verify ternary projection before training
    print("\n[3/5] Pre-training ternary check...")
    all_ok, results = verify_ternary(model)
    print(f"  All weights ternary: {all_ok}")
    if all_ok:
        for name, r in list(results.items())[:3]:
            d = r['distribution']
            print(f"  {name}: shape={r['shape']}, "
                  f"-1:{d[-1]:.3f}, 0:{d[0]:.3f}, +1:{d[1]:.3f}")
    
    # Load training data from train_data.txt
    print(f"\n[4/5] Loading train_data.txt from {args.data_path}...")
    with open(args.data_path, 'r') as f:
        text = f.read()
    
    print(f"  Text length: {len(text):,} characters")
    
    # Tokenize - use 90% for train, 10% for validation
    all_tokens = tokenizer.encode(text)
    print(f"  Total tokens: {len(all_tokens):,}")
    
    split_point = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:split_point]
    val_tokens = all_tokens[split_point:]
    print(f"  Train tokens: {len(train_tokens):,}")
    print(f"  Val tokens: {len(val_tokens):,}")
    
    seq_len = args.seq_len
    train_sequences = []
    for i in range(0, len(train_tokens) - seq_len - 1, seq_len + 1):
        train_sequences.append(train_tokens[i:i + seq_len + 1])
    
    val_sequences = []
    for i in range(0, len(val_tokens) - seq_len - 1, seq_len + 1):
        val_sequences.append(val_tokens[i:i + seq_len + 1])
    
    n_train = len(train_sequences)
    n_val = len(val_sequences)
    print(f"  Train sequences: {n_train:,} (seq_len={seq_len})")
    print(f"  Val sequences: {n_val:,}")
    
    if n_train == 0:
        print("ERROR: No training sequences! Data is too short for seq_len={seq_len}")
        return
    
    # Training loop
    print(f"\n[5/5] Training for {args.steps} steps...\n")
    
    lr_schedule = LRSchedule(args.lr, args.warmup, args.steps, args.min_lr)
    optimizer = AdamW(learning_rate=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    
    def loss_fn(model, batch):
        input_ids = mx.array(batch[:, :-1])
        targets = mx.array(batch[:, 1:])
        logits = model(input_ids)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")
    
    step = 0
    losses = []
    start_time = time.time()
    
    for epoch in range(100):
        if step >= args.steps:
            break
        
        indices = np.random.permutation(n_train)
        
        for i in range(0, n_train, args.batch_size):
            if step >= args.steps:
                break
            
            batch_indices = indices[i:i + args.batch_size]
            if len(batch_indices) < args.batch_size:
                # Pad by wrapping around
                extra = np.random.choice(n_train, size=args.batch_size - len(batch_indices), replace=False)
                batch_indices = np.concatenate([batch_indices, extra])
            
            batch = np.array([train_sequences[j] for j in batch_indices])
            
            current_lr = lr_schedule(step)
            optimizer.learning_rate = current_lr
            
            loss, grads = nn.value_and_grad(model, lambda m: loss_fn(m, batch))(model)
            grads, grad_norm = clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            mx.eval(loss)
            
            losses.append(float(loss))
            step += 1
            
            if step % args.log_every == 0:
                recent = losses[-args.log_every:]
                avg_loss = np.mean(recent)
                elapsed = time.time() - start_time
                toks_per_sec = args.log_every * args.batch_size * seq_len / max(elapsed, 0.001)
                print(f"  Step {step:4d}/{args.steps} | Loss: {avg_loss:.4f} | "
                      f"GradNorm: {grad_norm:.1f} | LR: {current_lr:.2e} | Tok/s: {toks_per_sec:.0f}")
                start_time = time.time()
            
            if step % args.eval_every == 0 and step > 0:
                val_indices = np.random.choice(n_val, size=min(args.batch_size, n_val), replace=False)
                val_batch = np.array([val_sequences[j] for j in val_indices])
                val_loss = loss_fn(model, val_batch)
                mx.eval(val_loss)
                val_ppl = math.exp(min(float(val_loss), 20))
                print(f"  >> Eval at step {step}: val_loss={float(val_loss):.4f}, val_ppl={val_ppl:.1f}")
                
                all_ok, _ = verify_ternary(model)
                print(f"     Ternary check: {'PASS' if all_ok else 'FAIL'}")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    all_ok, results = verify_ternary(model)
    print(f"\n1. TERNARY VERIFICATION: {'PASS' if all_ok else 'FAIL'}")
    for name, r in sorted(results.items())[:10]:
        d = r['distribution']
        status = "OK" if r['is_ternary'] else "FAIL"
        print(f"  [{status}] {name}: shape={r['shape']}, "
              f"-1:{d[-1]:.3f}, 0:{d[0]:.3f}, +1:{d[1]:.3f}")
    if len(results) > 10:
        print(f"  ... ({len(results) - 10} more layers)")
    
    # Validation perplexity
    print("\n2. PERPLEXITY EVALUATION:")
    eval_batch_size = min(4, n_val)
    val_losses_list = []
    for i in range(0, min(n_val, 20), eval_batch_size):
        batch = np.array(val_sequences[i:i + eval_batch_size])
        if len(batch) < eval_batch_size:
            continue
        vl = loss_fn(model, batch)
        mx.eval(vl)
        val_losses_list.append(float(vl))
    
    avg_val_loss = np.mean(val_losses_list) if val_losses_list else float('inf')
    vocab_size = config.vocab_size
    random_loss = math.log(vocab_size)
    print(f"  Train loss (last 50): {np.mean(losses[-50:]):.4f}")
    print(f"  Val loss: {avg_val_loss:.4f}")
    print(f"  Val perplexity: {math.exp(min(avg_val_loss, 20)):.1f}")
    print(f"  Random baseline: perplexity={vocab_size} (loss={random_loss:.2f})")
    
    # Text generation
    print("\n3. TEXT GENERATION:")
    prompts = [
        "The history of the United States",
        "Open source software",
        "The development of computers",
        "World War II",
        "The philosophy of mind",
    ]
    for prompt in prompts:
        try:
            generated = generate_text(model, tokenizer, prompt, max_tokens=80, temp=0.7)
            print(f"  Prompt: {prompt}")
            print(f"  Output: {generated[:250]}")
            print()
        except Exception as e:
            print(f"  Generation failed for '{prompt}': {e}")
    
    # Save
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        print(f"\nSaving model to {args.save_path}...")
        
        params = collect_all_params(model)
        if params:
            mx.save_safetensors(
                os.path.join(args.save_path, "weights.safetensors"),
                params
            )
            print(f"Saved {len(params)} weight tensors.")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Ternary projection verified: {all_ok}")
    print(f"Final training loss: {np.mean(losses[-50:]):.4f}")
    print(f"Validation perplexity: {math.exp(min(avg_val_loss, 20)):.1f}")
    print(f"(Random baseline: {vocab_size})")
    print()
    print("Comparison with previous WikiText-2 run:")
    print(f"  Previous: val_ppl=340.9, train_loss=6.13 (WikiText-2, 250 steps)")
    print(f"  This run: val_ppl={math.exp(min(avg_val_loss, 20)):.1f}, train_loss={np.mean(losses[-50:]):.4f} (train_data.txt, {args.steps} steps)")


if __name__ == "__main__":
    main()