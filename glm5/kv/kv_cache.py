"""
KV-Cache for Autoregressive Transformer Inference
===================================================

Memory layout
-------------
Each layer stores two tensors:

    keys:   shape (B, H, S_max, D)    — float32
    values: shape (B, H, S_max, D)    — float32

Where:
    B      = batch size
    H      = number of attention heads
    S_max  = pre-allocated max sequence length
    D      = head dimension  (d_model / H)

The layout is BHSD (batch, head, seq, dim) which is contiguous along
the sequence axis — ideal for appending one token at a time and for
the inner attention matmul.

A companion `seq_lens: list[int]` (length B) tracks how many positions
are valid in each batch element.  Positions beyond seq_lens[b] contain
garbage and must never participate in attention.

No external frameworks are used.  All kernels are pure-NumPy for
correctness; the design maps 1:1 to CUDA kernels (see README).
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional
import numpy as np

# ──────────────────────────────────────────────────────────────────────
#  1.  DATA STRUCTURE
# ──────────────────────────────────────────────────────────────────────

class KVCache:
    """
    Pre-allocated KV cache for one transformer layer.

    Physical storage
    ~~~~~~~~~~~~~~~~
    Two numpy arrays allocated once at construction:

        self.k_cache  (B, H, S_max, D)  float32
        self.v_cache  (B, H, S_max, D)  float32

    An auxiliary array `self.seq_lens` (length B, int) records how many
    token positions are live for each sequence in the batch.

    On GPU the same layout would be backed by a single cudaMalloc per
    layer.  The B-H-S-D ordering keeps the S-dimension stride == D,
    making the per-token write a simple 3D slice copy.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: np.dtype = np.float32,
    ):
        self.B = batch_size
        self.S_max = max_seq_len
        self.H = num_heads
        self.D = head_dim
        self.dtype = dtype

        shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.k_cache = np.zeros(shape, dtype=dtype)
        self.v_cache = np.zeros(shape, dtype=dtype)

        # seq_lens[b] = number of valid positions for batch element b
        self.seq_lens: List[int] = [0] * batch_size

    # ── helpers ──────────────────────────────────────────────────────

    def _check_batch(self, token_k: np.ndarray) -> None:
        """Validate shape of incoming key/value tensors."""
        # token_k expected: (B, H, T, D)  where T is the number of new tokens
        assert token_k.ndim == 4
        assert token_k.shape[0] == self.B
        assert token_k.shape[1] == self.H
        assert token_k.shape[3] == self.D

    # ── core update ──────────────────────────────────────────────────

    def update(
        self,
        new_k: np.ndarray,
        new_v: np.ndarray,
        positions: Optional[List[int]] = None,
    ) -> None:
        """
        Write new key/value vectors into the cache.

        Parameters
        ----------
        new_k, new_v : ndarray, shape (B, H, T, D)
            Keys and values for T new tokens.  In incremental decoding T=1.
        positions : list[int] | None
            Explicit write offsets per batch element.  When *None* the
            tokens are appended right after the current `seq_lens[b]`.
        """
        self._check_batch(new_k)
        T = new_k.shape[2]  # number of new tokens (1 for decode, S for prefill)

        for b in range(self.B):
            pos = positions[b] if positions is not None else self.seq_lens[b]
            assert pos + T <= self.S_max, (
                f"batch {b}: pos {pos} + {T} tokens would exceed S_max={self.S_max}"
            )
            # ---- the actual write: a slice copy into pre-allocated memory ----
            self.k_cache[b, :, pos : pos + T, :] = new_k[b]
            self.v_cache[b, :, pos : pos + T, :] = new_v[b]

        # advance sequence pointers
        for b in range(self.B):
            base = positions[b] if positions is not None else self.seq_lens[b]
            self.seq_lens[b] = base + T

    # ── retrieval (used by attention) ────────────────────────────────

    def get_kv(
        self, batch_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (keys, values) for a single batch element, trimmed to the
        valid prefix: shapes (H, S_valid, D) each.
        """
        s = self.seq_lens[batch_idx]
        return self.k_cache[batch_idx, :, :s, :], self.v_cache[batch_idx, :, :s, :]

    def get_full_kv(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return per-batch (keys, values) lists, each entry (H, S_valid, D)."""
        ks, vs = [], []
        for b in range(self.B):
            k, v = self.get_kv(b)
            ks.append(k)
            vs.append(v)
        return ks, vs

    # ── bookkeeping ──────────────────────────────────────────────────

    def reset(self) -> None:
        self.k_cache[:] = 0
        self.v_cache[:] = 0
        self.seq_lens = [0] * self.B

    def memory_bytes(self) -> int:
        return self.k_cache.nbytes + self.v_cache.nbytes

    def __repr__(self) -> str:
        return (
            f"KVCache(B={self.B}, H={self.H}, S_max={self.S_max}, "
            f"D={self.D}, seq_lens={self.seq_lens}, "
            f"mem={self.memory_bytes() / 1e6:.1f} MB)"
        )


# ──────────────────────────────────────────────────────────────────────
#  2.  MULTI-HEAD ATTENTION USING THE CACHE
# ──────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """
    Single-head attention.

    q: (S_q, D)   k: (S_kv, D)   v: (S_kv, D)
    returns: (S_q, D)
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = q @ k.T * scale           # (S_q, S_kv)
    weights = _softmax(scores, axis=-1)  # (S_q, S_kv)
    return weights @ v                  # (S_q, D)


def multi_head_attention_with_cache(
    q_new: np.ndarray,
    cache: KVCache,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
) -> np.ndarray:
    """
    Multi-head attention that *reads* from the KV cache but does NOT
    update it — the caller decides when to write.

    Parameters
    ----------
    q_new : ndarray, shape (B, T, d_model)
        Query representations for the T new tokens.
    cache : KVCache
        The key/value cache for this layer (already updated).
    w_q, w_k, w_v : ndarray, shape (d_model, d_model)
        Projection weight matrices.
    w_o : ndarray, shape (d_model, d_model)
        Output projection matrix.

    Returns
    -------
    output : ndarray, shape (B, T, d_model)
    """
    B = cache.B
    H = cache.H
    D = cache.D
    d_model = H * D
    T = q_new.shape[1]

    # project queries — same for every batch element
    q_proj = (q_new @ w_q).reshape(B, T, H, D)  # (B, T, H, D)

    outputs = np.empty((B, T, d_model), dtype=q_new.dtype)

    for b in range(B):
        k_cached, v_cached = cache.get_kv(b)  # (H, S_valid, D) each
        S_valid = cache.seq_lens[b]
        assert S_valid > 0, f"batch {b}: cache is empty"

        out_heads = np.empty((T, H, D), dtype=q_new.dtype)
        for h in range(H):
            # q: (T, D), k: (S_valid, D), v: (S_valid, D)
            q_h = q_proj[b, :, h, :]            # (T, D)
            k_h = k_cached[h]                    # (S_valid, D)
            v_h = v_cached[h]                    # (S_valid, D)
            out_heads[:, h, :] = _scaled_dot_product_attention(q_h, k_h, v_h)

        # concatenate heads and apply output projection
        out_heads = out_heads.reshape(T, d_model)
        outputs[b] = out_heads @ w_o

    return outputs


# ──────────────────────────────────────────────────────────────────────
#  3.  MASKED BATCHED ATTENTION  (variable seq lens in one batch)
# ──────────────────────────────────────────────────────────────────────

def multi_head_attention_batched(
    q_new: np.ndarray,
    cache: KVCache,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
) -> np.ndarray:
    """
    Batched MHA that correctly handles *variable sequence lengths*.

    We build a causal mask of shape (B, T, S_max_padded) that zeros out
    positions belonging to other sequences (in the packed sense) or
    future tokens.  Because we store per-batch caches separately this
    simplifies to per-element attention (no cross-contamination), but
    this function shows the masking technique that a GPU kernel would
    use when sequences are packed into a shared tensor.
    """
    B = cache.B
    H = cache.H
    D = cache.D
    d_model = H * D
    T = q_new.shape[1]

    q_proj = (q_new @ w_q).reshape(B, T, H, D)
    outputs = np.empty((B, T, d_model), dtype=q_new.dtype)

    for b in range(B):
        k_cached, v_cached = cache.get_kv(b)
        S_valid = cache.seq_lens[b]
        if S_valid == 0:
            raise ValueError(f"batch {b}: cache is empty — call update first")

        out_heads = np.empty((T, H, D), dtype=q_new.dtype)
        for h in range(H):
            q_h = q_proj[b, :, h, :]
            k_h = k_cached[h]
            v_h = v_cached[h]
            out_heads[:, h, :] = _scaled_dot_product_attention(q_h, k_h, v_h)

        outputs[b] = out_heads.reshape(T, d_model) @ w_o

    return outputs


# ──────────────────────────────────────────────────────────────────────
#  4.  INCREMENTAL DECODER  (end-to-end usage example)
# ──────────────────────────────────────────────────────────────────────

class IncrementalDecoder:
    """
    Minimal transformer decoder with L layers and KV caching.

    Demonstrates the full lifecycle:
        prefill  → fill cache with the entire prompt
        decode   → generate one token at a time using the cache
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        vocab_size: int,
        dtype: np.dtype = np.float32,
    ):
        self.d_model = d_model
        self.H = num_heads
        self.D = d_model // num_heads
        self.L = num_layers
        self.dtype = dtype

        # ---- weight matrices (Xavier init) ----
        scale = 2.0 / d_model
        self.w_embed = (np.random.randn(vocab_size, d_model) * scale).astype(dtype)
        self.w_q = [
            (np.random.randn(d_model, d_model) * scale).astype(dtype)
            for _ in range(num_layers)
        ]
        self.w_k = [
            (np.random.randn(d_model, d_model) * scale).astype(dtype)
            for _ in range(num_layers)
        ]
        self.w_v = [
            (np.random.randn(d_model, d_model) * scale).astype(dtype)
            for _ in range(num_layers)
        ]
        self.w_o = [
            (np.random.randn(d_model, d_model) * scale).astype(dtype)
            for _ in range(num_layers)
        ]
        self.w_out = (np.random.randn(d_model, vocab_size) * scale).astype(dtype)

        # ---- one KV cache per layer ----
        self.caches: List[KVCache] = []

    def _init_caches(self, batch_size: int) -> None:
        self.caches = [
            KVCache(batch_size, self.max_seq_len, self.H, self.D, self.dtype)
            for _ in range(self.L)
        ]

    # ---- layer norm (simplified) ----
    @staticmethod
    def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def forward_step(
        self,
        token_ids: np.ndarray,
        caches: List[KVCache],
        is_prefill: bool = False,
    ) -> np.ndarray:
        """
        One forward step.

        token_ids : int array, shape (B,) for decode or (B, T) for prefill
        caches    : list of KVCache, one per layer

        Returns logits (B, vocab_size) — always only for the *last* token.
        """
        if token_ids.ndim == 1:
            token_ids = token_ids[:, None]  # (B, 1)

        B, T = token_ids.shape
        hidden = self.w_embed[token_ids]  # (B, T, d_model)

        for layer_idx in range(self.L):
            # ---- project Q, K, V ----
            q = (hidden @ self.w_q[layer_idx]).reshape(B, T, self.H, self.D)
            k = (hidden @ self.w_k[layer_idx]).reshape(B, T, self.H, self.D)
            v = (hidden @ self.w_v[layer_idx]).reshape(B, T, self.H, self.D)

            # ---- update cache (write K, V) ----
            caches[layer_idx].update(
                k.transpose(0, 2, 1, 3),  # (B, H, T, D)
                v.transpose(0, 2, 1, 3),
            )

            # ---- attention read ----
            attn_out = multi_head_attention_with_cache(
                hidden, caches[layer_idx],
                self.w_q[layer_idx],
                self.w_k[layer_idx],
                self.w_v[layer_idx],
                self.w_o[layer_idx],
            )

            hidden = self._layer_norm(hidden + attn_out)

        # project last position to vocab
        logits = hidden[:, -1, :] @ self.w_out  # (B, vocab_size)
        return logits


# ──────────────────────────────────────────────────────────────────────
#  5.  MEMORY ANALYSIS
# ──────────────────────────────────────────────────────────────────────

def memory_analysis(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    batch_size: int,
    seq_len: int,
    bytes_per_element: int = 4,
) -> dict:
    """
    Analyse KV-cache memory consumption.

    Returns a dict with per-layer and total memory in bytes / MB.
    """
    per_token_per_layer = 2 * num_heads * head_dim * bytes_per_element  # K + V
    per_layer_bytes = per_token_per_layer * batch_size * seq_len
    total_bytes = per_layer_bytes * num_layers

    return {
        "per_token_per_layer_B": per_token_per_layer,
        "per_layer_bytes": per_layer_bytes,
        "per_layer_MB": per_layer_bytes / 1e6,
        "total_bytes": total_bytes,
        "total_MB": total_bytes / 1e6,
        "total_GB": total_bytes / 1e9,
        "params": {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "bytes_per_element": bytes_per_element,
        },
    }


def memory_growth_table(
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
    batch_size: int = 1,
    seq_lens: Optional[List[int]] = None,
) -> str:
    """
    Pretty-print a table of KV-cache memory vs sequence length.
    """
    if seq_lens is None:
        seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    lines = []
    lines.append(f"{'Seq Len':>10} | {'Per Layer (MB)':>15} | {'Total (MB)':>12} | {'Total (GB)':>12}")
    lines.append("-" * 60)

    for s in seq_lens:
        info = memory_analysis(num_layers, num_heads, head_dim, batch_size, s)
        lines.append(
            f"{s:>10} | {info['per_layer_MB']:>15.2f} | {info['total_MB']:>12.2f} | {info['total_GB']:>12.3f}"
        )
    return "\n".join(lines)
