"""
Attention Computation with KV-Cache

Implements:
  1. Standard scaled dot-product attention (no cache)
  2. Cached attention for incremental decoding
  3. Masked attention for variable-length batches
  4. Multi-query and grouped-query attention variants
"""

import numpy as np
from typing import Optional, Tuple
from kv_cache import KVCache, CacheConfig


def softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    scale: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Standard scaled dot-product attention (no caching).

    Args:
        q: (batch, num_heads, seq_q, head_dim)
        k: (batch, num_heads, seq_k, head_dim)
        v: (batch, num_heads, seq_k, head_dim)
        scale: typically 1 / sqrt(head_dim)
        mask: (batch, 1, 1, seq_k) or broadcastable — values masked to -inf

    Returns:
        output: (batch, num_heads, seq_q, head_dim)
    """
    # Q @ K^T: (batch, heads, seq_q, head_dim) @ (batch, heads, head_dim, seq_k)
    # -> (batch, heads, seq_q, seq_k)
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) * scale

    if mask is not None:
        scores = scores + mask  # mask has -inf for masked positions

    attn_weights = softmax_stable(scores, axis=-1)

    # Attn @ V: (batch, heads, seq_q, seq_k) @ (batch, heads, seq_k, head_dim)
    # -> (batch, heads, seq_q, head_dim)
    output = np.einsum("bhqk,bhkd->bhqd", attn_weights, v)
    return output


def build_causal_mask(seq_len: int, dtype=np.float32) -> np.ndarray:
    """
    Build a causal (triangular) mask for a sequence.

    Returns (seq_len, seq_len) where upper triangle is -inf.
    Position i can attend to positions j where j <= i.
    """
    indices = np.arange(seq_len)
    # Mask positions where key_pos > query_pos (future positions)
    mask = np.where(indices[None, :] > indices[:, None], -np.inf, 0.0)
    return mask.astype(dtype)


def build_variable_length_mask(
    lengths: np.ndarray,
    query_len: int,
    max_key_len: int = None,
    dtype=np.float32,
) -> np.ndarray:
    """
    Build a mask for variable-length batches.

    For each batch item, positions beyond its actual length are masked.
    Also applies causal masking (only attend to positions <= query position).

    Args:
        lengths: (batch,) actual sequence lengths per batch item
        query_len: number of query positions (usually 1 for generation)
        max_key_len: override for key dimension (defaults to max(lengths))

    Returns:
        mask: (batch, 1, query_len, max_key_len)
    """
    batch_size = len(lengths)
    if max_key_len is None:
        max_key_len = int(np.max(lengths))

    # Key positions: 0 .. max_key_len-1
    key_positions = np.arange(max_key_len)  # (max_key_len,)

    # Query positions: 0 .. query_len-1 (relative to each sequence)
    query_positions = np.arange(query_len)  # (query_len,)

    # Causal: key_pos <= query_pos is allowed (attend to past)
    causal = (key_positions[None, :] <= query_positions[:, None]).astype(dtype)
    # (query_len, max_key_len)

    # Length mask: key_pos < length[b] is allowed
    length_mask = (key_positions[None, None, None, :] < lengths[:, None, None, None]).astype(dtype)
    # (batch, 1, 1, max_key_len)

    # Combined: both causal and within length
    # causal: (query_len, max_key_len) -> (1, 1, query_len, max_key_len)
    combined = causal[None, None, :, :] * length_mask  # broadcast
    # (batch, 1, query_len, max_key_len)

    # Convert 0/1 to 0/-inf
    mask = np.where(combined > 0, 0.0, -np.inf)
    return mask.astype(dtype)


def cached_attention(
    q: np.ndarray,
    cache: KVCache,
    scale: float,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Attention using cached K and V.

    During generation, q is (batch, heads, 1, head_dim) — just the current token.
    The cache holds all previous K and V.

    Steps:
      1. Retrieve cached K, V from the cache
      2. Compute Q @ K^T with the full history
      3. Apply softmax and @ V

    This avoids recomputing K and V for past tokens.

    Args:
        q: (batch, num_heads, 1, head_dim) — current query
        cache: KVCache with previously stored K and V
        scale: 1 / sqrt(head_dim)

    Returns:
        output: (batch, num_heads, 1, head_dim)
    """
    # Retrieve all cached keys and values
    cached_k, cached_v = cache.get_all()
    # (batch, num_heads, seq_so_far, head_dim)

    # Cast to computation dtype for numerical stability
    q_f = q.astype(dtype)
    k_f = cached_k.astype(dtype)
    v_f = cached_v.astype(dtype)

    # Q @ K^T: (batch, heads, 1, head_dim) @ (batch, heads, head_dim, seq)
    # -> (batch, heads, 1, seq)
    scores = np.einsum("bhqd,bhkd->bhqk", q_f, k_f) * scale

    # No mask needed during generation (causal is implicit: we only have
    # past keys, no future keys exist in the cache)
    attn_weights = softmax_stable(scores, axis=-1)

    # Attn @ V: (batch, heads, 1, seq) @ (batch, heads, seq, head_dim)
    # -> (batch, heads, 1, head_dim)
    output = np.einsum("bhqk,bhkd->bhqd", attn_weights, v_f)

    return output.astype(q.dtype)


def cached_attention_with_mask(
    q: np.ndarray,
    cache: KVCache,
    scale: float,
    lengths: Optional[np.ndarray] = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Cached attention with variable-length masking.

    Handles batches where sequences have different lengths (some may have
    finished generation and are padded).
    """
    cached_k, cached_v = cache.get_all()
    seq_len = cached_k.shape[2]

    q_f = q.astype(dtype)
    k_f = cached_k.astype(dtype)
    v_f = cached_v.astype(dtype)

    scores = np.einsum("bhqd,bhkd->bhqk", q_f, k_f) * scale

    # Build mask if variable lengths
    if lengths is not None:
        # During generation, lengths should reflect current cache position
        # Clamp lengths to not exceed cache size
        effective_lengths = np.minimum(lengths, seq_len)
        mask = build_variable_length_mask(effective_lengths, query_len=1,
                                          max_key_len=seq_len, dtype=dtype)
        scores = scores + mask

    attn_weights = softmax_stable(scores, axis=-1)
    output = np.einsum("bhqk,bhkd->bhqd", attn_weights, v_f)

    return output.astype(q.dtype)


def prompt_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    cache: KVCache,
    scale: float,
    lengths: Optional[np.ndarray] = None,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process the initial prompt (prefill phase).

    During prefill, we compute Q, K, V for all prompt tokens at once,
    store K and V in the cache, and compute attention with causal masking.

    Args:
        q: (batch, heads, prompt_len, head_dim)
        k: (batch, heads, prompt_len, head_dim)
        v: (batch, heads, prompt_len, head_dim)
        cache: KVCache to populate
        scale: 1 / sqrt(head_dim)

    Returns:
        output, k, v (k and v are returned for the caller to use)
    """
    batch_size = q.shape[0]
    prompt_len = q.shape[2]

    # Store all prompt tokens in cache
    for pos in range(prompt_len):
        k_slice = k[:, :, pos:pos+1, :]  # (batch, heads, 1, head_dim)
        v_slice = v[:, :, pos:pos+1, :]
        cache.update(k_slice, v_slice, seqlen_offset=pos)

    # Causal attention over the full prompt
    q_f = q.astype(dtype)
    k_f = k.astype(dtype)
    v_f = v.astype(dtype)

    scores = np.einsum("bhqd,bhkd->bhqk", q_f, k_f) * scale

    # Causal mask
    causal = build_causal_mask(prompt_len, dtype=dtype)
    scores = scores + causal[None, None, :, :]  # broadcast over batch, heads

    # Variable length mask
    if lengths is not None:
        mask = build_variable_length_mask(lengths, query_len=prompt_len, dtype=dtype)
        scores = scores + mask

    attn_weights = softmax_stable(scores, axis=-1)
    output = np.einsum("bhqk,bhkd->bhqd", attn_weights, v_f)

    return output.astype(q.dtype), k, v


# ---------------------------------------------------------------------------
# Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
# ---------------------------------------------------------------------------

def cached_attention_gqa(
    q: np.ndarray,
    cache_k: np.ndarray,
    cache_v: np.ndarray,
    num_query_groups: int,
    scale: float,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Grouped-query attention with cached K/V.

    In GQA, multiple query heads share one key-value head.
    q: (batch, num_heads, 1, head_dim)
    cache_k, cache_v: (batch, num_kv_heads, seq, head_dim)
    num_query_groups = num_heads / num_kv_heads

    Each group of query heads attends to the same K/V head.
    """
    batch, num_heads, _, head_dim = q.shape
    num_kv_heads = cache_k.shape[1]
    assert num_heads % num_kv_heads == 0
    queries_per_group = num_heads // num_kv_heads

    q_f = q.astype(dtype)
    k_f = cache_k.astype(dtype)
    v_f = cache_v.astype(dtype)

    # Expand K and V to match query heads
    # k_f: (batch, num_kv_heads, 1, seq, head_dim)
    k_expanded = k_f[:, None, :, :, :]
    v_expanded = v_f[:, None, :, :, :]

    # q_f: (batch, num_kv_heads, queries_per_group, 1, head_dim)
    q_reshaped = q_f.reshape(batch, num_kv_heads, queries_per_group, 1, head_dim)

    # Q @ K^T per group
    # (batch, kv_heads, q_per_group, 1, head_dim) @ (batch, kv_heads, head_dim, seq)
    scores = np.einsum("bhgqd,bhkd->bhgqk", q_reshaped, k_f) * scale

    attn_weights = softmax_stable(scores, axis=-1)

    # Attn @ V
    output = np.einsum("bhgqk,bhkd->bhgqd", attn_weights, v_f)

    # Reshape back: (batch, num_heads, 1, head_dim)
    output = output.reshape(batch, num_heads, 1, head_dim)
    return output.astype(q.dtype)
