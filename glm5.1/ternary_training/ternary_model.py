"""
Ternary Bonsai: Qwen3 architecture with ternary weights {-1, 0, +1}.

Group-wise quantization with group_size=128, STE for gradient propagation.
All linear layers (embeddings, Q/K/V/O, SwiGLU gate/up/down, LM head) are ternary.
RMSNorm layers remain in full precision.
"""

from dataclasses import dataclass, fields
from typing import Optional, Dict, Union

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelArgs:
    model_type: str = ""
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    intermediate_size: int = 3072
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    head_dim: int = 64
    tie_word_embeddings: bool = True
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    @classmethod
    def from_dict(cls, config):
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in field_names})


def ternarize_ste(W: mx.array, group_size: int = 128) -> mx.array:
    """
    Project weights to ternary {-s, 0, +s} with Straight-Through Estimator.

    Forward:  W -> clip(round(W / mean(|W_group|)), -1, 1) * mean(|W_group|)
    Backward: gradient passes through as identity (STE).
    """
    orig_shape = W.shape
    *leading, n = orig_shape
    assert n % group_size == 0, f"dim {n} not divisible by group_size {group_size}"

    flat = W.reshape(-1, n)
    num_groups = n // group_size
    grouped = flat.reshape(flat.shape[0], num_groups, group_size)

    scales = mx.mean(mx.abs(grouped), axis=-1, keepdims=True)
    scales = mx.maximum(scales, 1e-5)

    W_q = mx.clip(mx.round(grouped / scales), -1.0, 1.0)
    W_ternary = (W_q * scales).reshape(flat.shape).reshape(orig_shape)

    return W + mx.stop_gradient(W_ternary - W)


def project_ternary(W: mx.array, group_size: int = 128):
    """Project weights to ternary indices (inference/verification only)."""
    orig_shape = W.shape
    *_, n = orig_shape
    flat = W.reshape(-1, n)
    num_groups = n // group_size
    grouped = flat.reshape(flat.shape[0], num_groups, group_size)

    scales = mx.mean(mx.abs(grouped), axis=-1, keepdims=True)
    scales = mx.maximum(scales, 1e-5)
    W_q = mx.clip(mx.round(grouped / scales), -1.0, 1.0)
    return W_q.reshape(orig_shape), scales.squeeze(-1)


class TernaryLinear(nn.Module):
    """Linear layer whose weights are projected to ternary on every forward pass."""

    def __init__(self, in_features: int, out_features: int, group_size: int = 128):
        super().__init__()
        self.weight = mx.random.normal((out_features, in_features)) * (in_features ** -0.5)
        self.group_size = group_size

    def __call__(self, x: mx.array) -> mx.array:
        W = ternarize_ste(self.weight, self.group_size)
        return x @ W.T


class TernaryEmbedding(nn.Module):
    """Embedding layer with ternary weights."""

    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int = 128):
        super().__init__()
        self.weight = mx.zeros((num_embeddings, embedding_dim))
        self.group_size = group_size

    def __call__(self, ids: mx.array) -> mx.array:
        W = ternarize_ste(self.weight, self.group_size)
        return W[ids]

    def as_linear(self, x: mx.array) -> mx.array:
        W = ternarize_ste(self.weight, self.group_size)
        return x @ W.T


def _repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    if n_rep == 1:
        return x
    B, H, L, D = x.shape
    return mx.broadcast_to(x[:, :, None, :, :], (B, H, n_rep, L, D)).reshape(
        B, H * n_rep, L, D
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, group_size: int = 128):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        head_dim = args.head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = TernaryLinear(dim, self.n_heads * head_dim, group_size)
        self.k_proj = TernaryLinear(dim, self.n_kv_heads * head_dim, group_size)
        self.v_proj = TernaryLinear(dim, self.n_kv_heads * head_dim, group_size)
        self.o_proj = TernaryLinear(self.n_heads * head_dim, dim, group_size)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.rope = nn.RoPE(head_dim, base=args.rope_theta, traditional=False)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, -1)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1)

        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)

        scores = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, group_size: int = 128):
        super().__init__()
        self.gate_proj = TernaryLinear(dim, hidden_dim, group_size)
        self.down_proj = TernaryLinear(hidden_dim, dim, group_size)
        self.up_proj = TernaryLinear(dim, hidden_dim, group_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, group_size: int = 128):
        super().__init__()
        self.self_attn = Attention(args, group_size)
        self.mlp = MLP(args.hidden_size, args.intermediate_size, group_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen3TernaryBody(nn.Module):
    """Inner model holding embed, layers, norm — mirrors original Qwen3Model."""

    def __init__(self, args: ModelArgs, group_size: int = 128):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.embed_tokens = TernaryEmbedding(
            args.vocab_size, args.hidden_size, group_size
        )
        self.layers = [
            TransformerBlock(args, group_size) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        L = h.shape[1]
        if cache[0] is None:
            mask = mx.triu(
                mx.full((L, L), -1e9, dtype=h.dtype), k=1
            )[None, None, :, :]
        else:
            offset = cache[0].offset
            mask = mx.triu(
                mx.full((L, L + offset), -1e9, dtype=h.dtype), k=1 + offset
            )[None, None, :, :]

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    """Ternary Bonsai model — Qwen3 architecture with ternary weights.

    Structure matches the original Qwen3 Model so copy_weights works:
      self.model.embed_tokens / self.model.layers / self.model.norm
      self.lm_head  (only if tie_word_embeddings=False)
    """

    def __init__(self, args: ModelArgs, group_size: int = 128):
        super().__init__()
        self.args = args
        self.group_size = group_size
        self.model_type = args.model_type

        self.model = Qwen3TernaryBody(args, group_size)

        if not args.tie_word_embeddings:
            self.lm_head = TernaryLinear(
                args.hidden_size, args.vocab_size, group_size
            )

    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights


def copy_weights(src, dst):
    """Recursively copy weight arrays from src model to dst model (float32).

    MLX nn.Module extends dict, so children/params live as dict items.
    """
    for name in src.keys():
        if name not in dst:
            continue
        sv = src[name]
        dv = dst[name]
        if isinstance(sv, mx.array) and isinstance(dv, mx.array):
            dst[name] = sv.astype(mx.float32)
        elif isinstance(sv, nn.Module) and isinstance(dv, nn.Module):
            copy_weights(sv, dv)
        elif isinstance(sv, list) and isinstance(dv, list):
            for s, d in zip(sv, dv):
                if isinstance(s, nn.Module) and isinstance(d, nn.Module):
                    copy_weights(s, d)
