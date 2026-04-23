"""
KV-Cache Data Structures for Autoregressive Transformer Inference

Core memory layout:
    cache_k[batch, head, seq_len, head_dim]
    cache_v[batch, head, seq_len, head_dim]

This layout enables O(1) append per token and contiguous memory access
during attention computation (Q @ K^T scans along seq_len).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CacheConfig:
    """Configuration for a single layer's KV cache."""
    batch_size: int
    num_heads: int
    head_dim: int
    max_seq_len: int
    dtype: np.dtype = np.float16

    @property
    def cache_bytes_per_layer(self) -> int:
        """Bytes for one layer's K + V cache."""
        elem_bytes = np.dtype(self.dtype).itemsize
        one_side = self.batch_size * self.num_heads * self.max_seq_len * self.head_dim
        return 2 * one_side * elem_bytes  # K + V

    @property
    def cache_bytes_per_layer_per_token(self) -> int:
        """Bytes consumed per generated token per layer."""
        elem_bytes = np.dtype(self.dtype).itemsize
        return 2 * self.num_heads * self.head_dim * elem_bytes


class KVCache:
    """
    Standard contiguous KV cache for one transformer layer.

    Memory layout (row-major / C-contiguous):
        cache_k: (batch, num_heads, max_seq_len, head_dim)
        cache_v: (batch, num_heads, max_seq_len, head_dim)

    Why this layout:
    - batch first: enables batched GEMM on GPU
    - head second: allows parallel head computation
    - seq_len third: contiguous scan for Q @ K^T
    - head_dim last: inner product dimension

    The cache is pre-allocated to max_seq_len. A `lengths` array tracks
    actual sequence lengths per batch item (for variable-length batching).
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.batch_size = config.batch_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.dtype = config.dtype

        # Pre-allocate full buffers (zero-initialized)
        shape = (self.batch_size, self.num_heads, self.max_seq_len, self.head_dim)
        self.cache_k = np.zeros(shape, dtype=self.dtype)
        self.cache_v = np.zeros(shape, dtype=self.dtype)

        # Per-batch-item current sequence length
        self.lengths = np.zeros(self.batch_size, dtype=np.int32)

        # Write pointer: next position to write into
        self.write_pos = 0

    def reset(self):
        """Clear the cache for a new generation."""
        self.cache_k[...] = 0
        self.cache_v[...] = 0
        self.lengths[...] = 0
        self.write_pos = 0

    def update(self, keys: np.ndarray, values: np.ndarray,
               seqlen_offset: int = None) -> None:
        """
        Append newly computed K and V to the cache.

        Args:
            keys:     (batch, num_heads, 1, head_dim) — current step's K
            values:   (batch, num_heads, 1, head_dim) — current step's V
            seqlen_offset: optional explicit write position (defaults to self.write_pos)

        The write position advances by 1 each call during generation.
        For the initial prompt, seqlen_offset=0 and we write all prompt tokens.
        """
        if seqlen_offset is None:
            seqlen_offset = self.write_pos

        pos = seqlen_offset
        self.cache_k[:, :, pos, :] = keys[:, :, 0, :]
        self.cache_v[:, :, pos, :] = values[:, :, 0, :]

        # Update per-batch-item lengths
        for b in range(self.batch_size):
            self.lengths[b] = pos + 1

        self.write_pos = pos + 1

    def get(self, start: int = 0, end: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve cached K and V slices.

        Returns:
            k: (batch, num_heads, end-start, head_dim)
            v: (batch, num_heads, end-start, head_dim)
        """
        if end is None:
            end = self.write_pos
        return (
            self.cache_k[:, :, start:end, :],
            self.cache_v[:, :, start:end, :],
        )

    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all cached tokens so far (up to write_pos)."""
        return self.get(0, self.write_pos)

    @property
    def memory_used_bytes(self) -> int:
        """Actual bytes used (based on write_pos, not max allocation)."""
        elem_bytes = np.dtype(self.dtype).itemsize
        tokens = self.write_pos
        return 2 * self.batch_size * self.num_heads * tokens * self.head_dim * elem_bytes

    @property
    def memory_allocated_bytes(self) -> int:
        """Total pre-allocated bytes."""
        return self.config.cache_bytes_per_layer


class BatchedKVCache:
    """
    Manages KV caches across all layers of a transformer.

    In a real model with L layers, we need L separate KV caches.
    This class coordinates them and handles variable-length batching.
    """

    def __init__(self, num_layers: int, config: CacheConfig):
        self.num_layers = num_layers
        self.config = config
        self.caches = [KVCache(config) for _ in range(num_layers)]

    def reset(self):
        for cache in self.caches:
            cache.reset()

    def update(self, layer_idx: int, keys: np.ndarray, values: np.ndarray,
               seqlen_offset: int = None):
        self.caches[layer_idx].update(keys, values, seqlen_offset)

    def get(self, layer_idx: int, start: int = 0, end: int = None):
        return self.caches[layer_idx].get(start, end)

    @property
    def total_memory_allocated_bytes(self) -> int:
        return sum(c.memory_allocated_bytes for c in self.caches)

    @property
    def total_memory_used_bytes(self) -> int:
        return sum(c.memory_used_bytes for c in self.caches)

    def memory_report(self) -> dict:
        """Detailed memory breakdown."""
        elem_bytes = self.config.dtype.itemsize
        tokens = self.caches[0].write_pos if self.caches else 0
        per_layer = self.config.cache_bytes_per_layer
        per_token_per_layer = self.config.cache_bytes_per_layer_per_token

        return {
            "num_layers": self.num_layers,
            "batch_size": self.config.batch_size,
            "num_heads": self.config.num_heads,
            "head_dim": self.config.head_dim,
            "max_seq_len": self.config.max_seq_len,
            "dtype": str(self.config.dtype),
            "tokens_generated": tokens,
            "per_layer_allocated_mb": per_layer / (1024 * 1024),
            "total_allocated_mb": self.total_memory_allocated_bytes / (1024 * 1024),
            "total_used_mb": self.total_memory_used_bytes / (1024 * 1024),
            "growth_per_token_mb": (per_token_per_layer * self.num_layers) / (1024 * 1024),
        }
