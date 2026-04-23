"""
Transformer Layer with KV-Cache Integration

Implements a complete decoder transformer layer that:
  - Computes Q, K, V projections
  - Stores K, V in the cache
  - Performs cached attention
  - Applies MLP with residual connections and layer norm
"""

import numpy as np
from typing import Optional, Tuple, List
from kv_cache import KVCache, CacheConfig, BatchedKVCache
from attention import (
    cached_attention,
    cached_attention_with_mask,
    prompt_attention,
)


class Linear:
    """Simple linear layer (no framework)."""

    def __init__(self, in_features: int, out_features: int,
                 dtype=np.float32, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        # Kaiming initialization
        scale = np.sqrt(2.0 / in_features)
        self.weight = np.random.randn(out_features, in_features).astype(dtype) * scale
        self.bias = np.zeros(out_features, dtype=dtype)
        self.dtype = dtype

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (x @ self.weight.T + self.bias).astype(self.dtype)


class LayerNorm:
    """Layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-5, dtype=np.float32):
        self.dim = dim
        self.eps = eps
        self.weight = np.ones(dim, dtype=dtype)
        self.bias = np.zeros(dim, dtype=dtype)
        self.dtype = dtype

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_f = x.astype(np.float32)
        mean = np.mean(x_f, axis=-1, keepdims=True)
        var = np.var(x_f, axis=-1, keepdims=True)
        x_norm = (x_f - mean) / np.sqrt(var + self.eps)
        return (x_norm * self.weight + self.bias).astype(self.dtype)


class MLP:
    """Feed-forward network: linear -> activation -> linear."""

    def __init__(self, dim: int, hidden_dim: int, dtype=np.float32, seed: int = None):
        self.fc1 = Linear(dim, hidden_dim, dtype=dtype, seed=seed)
        self.fc2 = Linear(hidden_dim, dim, dtype=dtype, seed=seed + 1 if seed else None)
        self.dtype = dtype

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.fc1.forward(x)
        # GELU approximation
        h = h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h ** 3))) * 0.5
        return self.fc2.forward(h)


class TransformerDecoderLayer:
    """
    Single decoder transformer layer with KV-cache support.

    Architecture:
        x -> LayerNorm -> Self-Attention -> Residual -> LayerNorm -> MLP -> Residual

    Pre-norm variant (used by most modern models).
    """

    def __init__(self, dim: int, num_heads: int, mlp_hidden: int,
                 dtype=np.float32, seed: int = None):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.dtype = dtype

        # Q, K, V projections
        self.wq = Linear(dim, dim, dtype=dtype, seed=seed)
        self.wk = Linear(dim, dim, dtype=dtype, seed=seed + 1 if seed else None)
        self.wv = Linear(dim, dim, dtype=dtype, seed=seed + 2 if seed else None)

        # Output projection
        self.wo = Linear(dim, dim, dtype=dtype, seed=seed + 3 if seed else None)

        # Normalizations
        self.norm1 = LayerNorm(dim, dtype=dtype)
        self.norm2 = LayerNorm(dim, dtype=dtype)

        # MLP
        self.mlp = MLP(dim, mlp_hidden, dtype=dtype, seed=seed + 4 if seed else None)

    def _to_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape (batch, seq, dim) -> (batch, seq, heads, head_dim)."""
        batch, seq, _ = x.shape
        return x.reshape(batch, seq, self.num_heads, self.head_dim)

    def _from_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape (batch, seq, heads, head_dim) -> (batch, seq, dim)."""
        batch, seq, _, _ = x.shape
        return x.reshape(batch, seq, self.dim)

    def forward_prefill(
        self,
        x: np.ndarray,
        cache: KVCache,
        lengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Process the full prompt (prefill phase).

        Args:
            x: (batch, prompt_len, dim)
            cache: KVCache to populate with K, V
            lengths: optional per-batch-item prompt lengths

        Returns:
            output: (batch, prompt_len, dim)
        """
        batch, seq_len, _ = x.shape

        # Self-attention with residual
        residual = x
        x_norm = self.norm1.forward(x)

        # Project to Q, K, V
        q = self.wq.forward(x_norm)  # (batch, seq, dim)
        k = self.wk.forward(x_norm)
        v = self.wv.forward(x_norm)

        # Reshape to multi-head
        q = self._to_heads(q).transpose(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
        k = self._to_heads(k).transpose(0, 2, 1, 3)
        v = self._to_heads(v).transpose(0, 2, 1, 3)

        # Cached attention (stores K, V in cache)
        attn_out, _, _ = prompt_attention(
            q, k, v, cache, self.scale, lengths=lengths
        )
        # (batch, heads, seq, head_dim)

        # Reshape and project output
        attn_out = attn_out.transpose(0, 2, 1, 3)  # (batch, seq, heads, head_dim)
        attn_out = self._from_heads(attn_out)       # (batch, seq, dim)
        attn_out = self.wo.forward(attn_out)

        x = residual + attn_out

        # MLP with residual
        residual = x
        x_norm = self.norm2.forward(x)
        mlp_out = self.mlp.forward(x_norm)
        x = residual + mlp_out

        return x

    def forward_generate(
        self,
        x: np.ndarray,
        cache: KVCache,
        lengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Process one token (generation phase).

        Args:
            x: (batch, 1, dim) — single token
            cache: KVCache with previous K, V
            lengths: optional per-batch-item sequence lengths

        Returns:
            output: (batch, 1, dim)
        """
        # Self-attention with residual
        residual = x
        x_norm = self.norm1.forward(x)

        # Project to Q, K, V
        q = self.wq.forward(x_norm)  # (batch, 1, dim)
        k = self.wk.forward(x_norm)
        v = self.wv.forward(x_norm)

        # Reshape to multi-head
        q = self._to_heads(q).transpose(0, 2, 1, 3)  # (batch, heads, 1, head_dim)
        k = self._to_heads(k).transpose(0, 2, 1, 3)
        v = self._to_heads(v).transpose(0, 2, 1, 3)

        # Store K, V in cache
        cache.update(k, v)

        # Cached attention
        if lengths is not None:
            attn_out = cached_attention_with_mask(
                q, cache, self.scale, lengths=lengths
            )
        else:
            attn_out = cached_attention(q, cache, self.scale)
        # (batch, heads, 1, head_dim)

        # Reshape and project output
        attn_out = attn_out.transpose(0, 2, 1, 3)  # (batch, 1, heads, head_dim)
        attn_out = self._from_heads(attn_out)       # (batch, 1, dim)
        attn_out = self.wo.forward(attn_out)

        x = residual + attn_out

        # MLP with residual
        residual = x
        x_norm = self.norm2.forward(x)
        mlp_out = self.mlp.forward(x_norm)
        x = residual + mlp_out

        return x


class TransformerDecoder:
    """
    Full transformer decoder with KV-cache management.

    Orchestrates prefill and generation across all layers.
    """

    def __init__(self, num_layers: int, dim: int, num_heads: int,
                 mlp_hidden: int, vocab_size: int, max_seq_len: int,
                 batch_size: int = 1, dtype=np.float32, seed: int = 42):
        self.num_layers = num_layers
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.vocab_size = vocab_size
        self.dtype = dtype

        # Embedding
        self.embedding = np.random.randn(vocab_size, dim).astype(dtype) * 0.02

        # Positional encoding (learnable)
        self.pos_embedding = np.random.randn(max_seq_len, dim).astype(dtype) * 0.02

        # Layers
        self.layers = [
            TransformerDecoderLayer(dim, num_heads, mlp_hidden,
                                    dtype=dtype, seed=seed + i * 100)
            for i in range(num_layers)
        ]

        # Final normalization and LM head
        self.final_norm = LayerNorm(dim, dtype=dtype)
        self.lm_head_weight = self.embedding.T  # weight tying

        # KV cache
        cache_config = CacheConfig(
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            dtype=dtype,
        )
        self.cache = BatchedKVCache(num_layers, cache_config)

    def _add_positional_encoding(self, x: np.ndarray, start_pos: int = 0) -> np.ndarray:
        """Add positional encoding to input embeddings."""
        batch, seq, _ = x.shape
        pos_enc = self.pos_embedding[start_pos:start_pos + seq]
        return (x + pos_enc[None, :, :]).astype(self.dtype)

    def prefill(self, token_ids: np.ndarray,
                lengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process the full prompt.

        Args:
            token_ids: (batch, prompt_len) integer token IDs
            lengths: optional (batch,) actual lengths per batch item

        Returns:
            hidden: (batch, prompt_len, dim) — hidden states after all layers
        """
        batch, prompt_len = token_ids.shape

        # Embed + positional encoding
        x = self.embedding[token_ids]  # (batch, prompt_len, dim)
        x = self._add_positional_encoding(x, start_pos=0)

        # Through all layers
        for i, layer in enumerate(self.layers):
            x = layer.forward_prefill(x, self.cache.caches[i], lengths=lengths)

        return x

    def generate_step(
        self,
        token_ids: np.ndarray,
        lengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate one token.

        Args:
            token_ids: (batch, 1) — the token to process
            lengths: optional (batch,) current sequence lengths

        Returns:
            logits: (batch, vocab_size) — output logits for next token
        """
        batch = token_ids.shape[0]
        current_pos = self.cache.caches[0].write_pos - 1  # position of this token

        # Embed + positional encoding
        x = self.embedding[token_ids]  # (batch, 1, dim)
        x = self._add_positional_encoding(x, start_pos=current_pos)

        # Through all layers
        for i, layer in enumerate(self.layers):
            x = layer.forward_generate(x, self.cache.caches[i], lengths=lengths)

        # Final norm + LM head
        x = self.final_norm.forward(x)  # (batch, 1, dim)
        logits = x @ self.lm_head_weight  # (batch, 1, vocab_size)
        return logits[:, 0, :]  # (batch, vocab_size)

    def generate(self, prompt_ids: np.ndarray, num_tokens: int,
                 temperature: float = 1.0, top_k: int = None,
                 lengths: Optional[np.ndarray] = None) -> List[int]:
        """
        Full generation loop.

        Args:
            prompt_ids: (batch, prompt_len) prompt token IDs
            num_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
            lengths: optional per-batch-item prompt lengths

        Returns:
            generated_ids: list of (batch,) token arrays
        """
        # Reset cache
        self.cache.reset()

        # Prefill
        self.prefill(prompt_ids, lengths=lengths)

        # Get last token from prefill
        batch = prompt_ids.shape[0]
        last_tokens = prompt_ids[:, -1:]  # (batch, 1)

        # Track current lengths (start from prompt lengths)
        if lengths is not None:
            cur_lengths = lengths.copy()
        else:
            cur_lengths = np.full(batch, prompt_ids.shape[1], dtype=np.int32)

        generated = []
        for step in range(num_tokens):
            logits = self.generate_step(last_tokens, lengths=cur_lengths)

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                top_k_values = np.sort(logits, axis=-1)[:, -top_k:]
                mask = logits < top_k_values[:, -1:]
                logits = np.where(mask, -np.inf, logits)

            # Softmax + sample
            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)

            # Sample
            sampled = np.array([
                np.random.choice(len(probs[b]), p=probs[b] / probs[b].sum())
                for b in range(batch)
            ])

            generated.append(sampled)
            last_tokens = sampled[:, None]  # (batch, 1)

            # Update lengths
            cur_lengths = cur_lengths + 1

        return generated

    def memory_report(self) -> dict:
        """Get memory usage report."""
        return self.cache.memory_report()
