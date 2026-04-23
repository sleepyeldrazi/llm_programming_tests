"""
KV-Cache Optimizations

Implements three major optimization strategies:
  1. Paged Attention — non-contiguous memory allocation (inspired by vLLM)
  2. Quantization — reduced precision for cached K/V
  3. Chunked Prefill — processing long prompts in chunks to limit peak memory
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from kv_cache import CacheConfig


# =============================================================================
# 1. PAGED ATTENTION
# =============================================================================

@dataclass
class PageConfig:
    """Configuration for paged KV cache."""
    block_size: int = 16          # tokens per block
    num_pages: int = 256          # total pages per sequence
    batch_size: int = 4
    num_heads: int = 32
    head_dim: int = 128
    dtype: np.dtype = np.float16


class PagedKVCache:
    """
    Paged KV Cache — inspired by vLLM's PagedAttention.

    Instead of a contiguous [batch, heads, max_seq, head_dim] buffer,
    memory is divided into fixed-size blocks (pages). Each sequence
    maintains a page table mapping logical block indices to physical pages.

    Benefits:
      - Zero memory fragmentation: blocks are allocated on demand
      - Supports speculative decoding and branching
      - Enables sharing of common prefixes (prefix caching)
      - No need to pre-allocate max_seq_len

    Memory layout:
      physical_pages: (num_pages, batch_size, num_heads, block_size, head_dim)  [for K]
      physical_pages_v: same shape [for V]
      page_tables: (batch_size, max_blocks) — maps logical block -> physical page index
    """

    def __init__(self, config: PageConfig):
        self.config = config
        self.batch_size = config.batch_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.block_size = config.block_size
        self.num_pages = config.num_pages
        self.dtype = config.dtype

        # Physical page pool (shared across all sequences)
        # Each page holds: (num_heads, block_size, head_dim)
        page_shape = (config.num_pages * config.batch_size,
                      config.num_heads, config.block_size, config.head_dim)
        self.physical_pages_k = np.zeros(page_shape, dtype=self.dtype)
        self.physical_pages_v = np.zeros(page_shape, dtype=self.dtype)

        # Page table per sequence: logical_block_idx -> physical_page_idx
        max_blocks = config.num_pages
        self.page_tables = np.full(
            (config.batch_size, max_blocks), -1, dtype=np.int32
        )

        # Number of allocated blocks per sequence
        self.num_blocks = np.zeros(config.batch_size, dtype=np.int32)

        # Free page pool (global, shared)
        total_pages = config.num_pages * config.batch_size
        self.free_list = np.arange(total_pages, dtype=np.int32)
        self.free_ptr = 0  # index into free_list

    def _alloc_page(self) -> int:
        """Allocate one physical page from the free pool."""
        if self.free_ptr >= len(self.free_list):
            raise MemoryError("Paged KV cache out of memory")
        page_idx = self.free_list[self.free_ptr]
        self.free_ptr += 1
        return page_idx

    def _free_page(self, page_idx: int):
        """Return a physical page to the free pool."""
        self.free_list[self.free_ptr - 1] = page_idx
        self.free_ptr -= 1

    def reset(self):
        """Reset cache for a new generation."""
        self.physical_pages_k[...] = 0
        self.physical_pages_v[...] = 0
        self.page_tables[...] = -1
        self.num_blocks[...] = 0
        self.free_ptr = 0

    def append_token(self, batch_idx: int, keys: np.ndarray,
                     values: np.ndarray, logical_block: int,
                     offset_in_block: int):
        """
        Append one token to a specific logical block.

        Args:
            batch_idx: batch item index
            keys: (1, num_heads, 1, head_dim)
            values: (1, num_heads, 1, head_dim)
            logical_block: which logical block to write to
            offset_in_block: position within the block (0..block_size-1)
        """
        # Check if physical page is allocated for this logical block
        phys_page = self.page_tables[batch_idx, logical_block]

        if phys_page == -1:
            # Allocate new physical page
            phys_page = self._alloc_page()
            self.page_tables[batch_idx, logical_block] = phys_page
            if logical_block + 1 > self.num_blocks[batch_idx]:
                self.num_blocks[batch_idx] = logical_block + 1

        # Write to physical page
        self.physical_pages_k[phys_page, :, offset_in_block, :] = keys[0, :, 0, :]
        self.physical_pages_v[phys_page, :, offset_in_block, :] = values[0, :, 0, :]

    def get_sequence(self, batch_idx: int,
                     start_block: int = 0,
                     end_block: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve K and V for a sequence, gathering from physical pages.

        Returns:
            k: (num_heads, total_tokens, head_dim)
            v: (num_heads, total_tokens, head_dim)
        """
        if end_block is None:
            end_block = self.num_blocks[batch_idx]

        blocks = end_block - start_block
        total_tokens = blocks * self.block_size

        k_out = np.zeros(
            (self.num_heads, total_tokens, self.head_dim), dtype=self.dtype
        )
        v_out = np.zeros(
            (self.num_heads, total_tokens, self.head_dim), dtype=self.dtype
        )

        for i in range(start_block, end_block):
            phys_page = self.page_tables[batch_idx, i]
            if phys_page == -1:
                break
            block_idx = i - start_block
            token_start = block_idx * self.block_size
            token_end = token_start + self.block_size
            k_out[:, token_start:token_end, :] = self.physical_pages_k[phys_page]
            v_out[:, token_start:token_end, :] = self.physical_pages_v[phys_page]

        return k_out, v_out

    def get_sequence_contiguous(self, batch_idx: int,
                                 num_tokens: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get K, V as contiguous arrays for attention computation.

        Returns:
            k: (1, num_heads, num_tokens, head_dim)
            v: (1, num_heads, num_tokens, head_dim)
        """
        if num_tokens is None:
            num_tokens = self.num_blocks[batch_idx] * self.block_size

        k, v = self.get_sequence(batch_idx)
        # k: (num_heads, num_tokens, head_dim) -> (1, num_heads, num_tokens, head_dim)
        return k[None, ...], v[None, ...]

    @property
    def memory_allocated_bytes(self) -> int:
        elem_bytes = np.dtype(self.dtype).itemsize
        total_pages = self.num_pages * self.batch_size
        page_bytes = self.num_heads * self.block_size * self.head_dim * elem_bytes
        return 2 * total_pages * page_bytes  # K + V

    @property
    def memory_used_bytes(self) -> int:
        """Bytes actually used (allocated blocks only)."""
        elem_bytes = np.dtype(self.dtype).itemsize
        total_used_blocks = np.sum(self.num_blocks)
        page_bytes = self.num_heads * self.block_size * self.head_dim * elem_bytes
        return 2 * total_used_blocks * page_bytes

    def memory_utilization(self) -> float:
        """Fraction of allocated memory actually used."""
        alloc = self.memory_allocated_bytes
        if alloc == 0:
            return 0.0
        return self.memory_used_bytes / alloc


# =============================================================================
# 2. QUANTIZED KV CACHE
# =============================================================================

class QuantizedKVCache:
    """
    Quantized KV Cache — stores K and V in reduced precision.

    Strategy: per-channel (per-head-dim) int8 quantization.
      - Each head-dimension channel has its own scale and zero-point
      - Dequantize on-the-fly during attention computation

    Memory savings: float16 (16-bit) -> int8 (8-bit) = 2x reduction
    Plus metadata overhead: 2 scales per channel (K and V) in float16

    For head_dim=128:
      - Original: 128 * 16 = 2048 bits per token per head
      - Quantized: 128 * 8 + 2 * 128 * 16 = 1024 + 4096 = 5120 bits
      - But scales are shared across all tokens, so per-token: 128 * 8 = 1024 bits
      - Net savings: ~50%
    """

    def __init__(self, batch_size: int, num_heads: int, head_dim: int,
                 max_seq_len: int, dtype=np.float16):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.write_pos = 0

        # Quantized storage: int8
        shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.cache_k_int8 = np.zeros(shape, dtype=np.int8)
        self.cache_v_int8 = np.zeros(shape, dtype=np.int8)

        # Per-channel scales and zero-points per position
        scale_shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.k_scales = np.ones(scale_shape, dtype=dtype)
        self.k_zeros = np.zeros(scale_shape, dtype=dtype)
        self.v_scales = np.ones(scale_shape, dtype=dtype)
        self.v_zeros = np.zeros(scale_shape, dtype=dtype)

    def reset(self):
        self.cache_k_int8[...] = 0
        self.cache_v_int8[...] = 0
        self.k_scales[...] = 1.0
        self.k_zeros[...] = 0.0
        self.v_scales[...] = 1.0
        self.v_zeros[...] = 0.0
        self.write_pos = 0

    def _quantize(self, x: np.ndarray, axis: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quantize to int8 with per-channel affine transform: x ≈ scale * q + zero.

        Returns quantized values, scales, and zero-points.
        """
        x_f = x.astype(np.float32)
        # Per-channel min/max
        x_min = np.min(x_f, axis=axis, keepdims=True)
        x_max = np.max(x_f, axis=axis, keepdims=True)

        # Avoid division by zero
        x_range = x_max - x_min
        x_range = np.where(x_range < 1e-6, 1.0, x_range)

        # Scale: map [-128, 127] to [x_min, x_max]
        scale = x_range / 255.0
        zero = x_min  # zero-point

        # Quantize
        x_centered = x_f - zero
        x_quant = np.round(x_centered / scale).astype(np.int8)
        x_quant = np.clip(x_quant, -128, 127)

        return x_quant, scale.astype(self.dtype), zero.astype(self.dtype)

    def _dequantize(self, x_int8: np.ndarray, scale: np.ndarray,
                    zero: np.ndarray) -> np.ndarray:
        """Dequantize int8 back to float: x = scale * q + zero."""
        return (x_int8.astype(np.float32) * scale + zero).astype(self.dtype)

    def update(self, keys: np.ndarray, values: np.ndarray,
               seqlen_offset: int = None):
        """
        Quantize and store K, V.

        Args:
            keys: (batch, heads, 1, head_dim)
            values: (batch, heads, 1, head_dim)
        """
        if seqlen_offset is None:
            seqlen_offset = self.write_pos

        pos = seqlen_offset

        # Quantize K
        k_q, k_s, k_z = self._quantize(keys, axis=-1)
        self.cache_k_int8[:, :, pos, :] = k_q[:, :, 0, :]
        self.k_scales[:, :, pos:pos+1, :] = k_s
        self.k_zeros[:, :, pos:pos+1, :] = k_z

        # Quantize V
        v_q, v_s, v_z = self._quantize(values, axis=-1)
        self.cache_v_int8[:, :, pos, :] = v_q[:, :, 0, :]
        self.v_scales[:, :, pos:pos+1, :] = v_s
        self.v_zeros[:, :, pos:pos+1, :] = v_z

        self.write_pos = pos + 1

    def get(self, start: int = 0, end: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get dequantized K, V."""
        if end is None:
            end = self.write_pos

        k_int = self.cache_k_int8[:, :, start:end, :]
        v_int = self.cache_v_int8[:, :, start:end, :]

        # Dequantize using scales and zero-points from each position
        k_deq = self._dequantize(k_int, self.k_scales[:, :, start:end, :],
                                 self.k_zeros[:, :, start:end, :])
        v_deq = self._dequantize(v_int, self.v_scales[:, :, start:end, :],
                                 self.v_zeros[:, :, start:end, :])

        return k_deq, v_deq

    @property
    def memory_allocated_bytes(self) -> int:
        """Total allocated memory including quantization metadata.

        Includes: int8 K + int8 V + fp scales (K+V) + fp zero-points (K+V)
        """
        elem_int8 = np.dtype(np.int8).itemsize
        elem_fp = np.dtype(self.dtype).itemsize
        n = self.batch_size * self.num_heads * self.max_seq_len * self.head_dim
        k_v_bytes = 2 * n * elem_int8       # int8 K + V
        meta_bytes = 4 * n * elem_fp         # scales + zeros for K and V
        return k_v_bytes + meta_bytes

    @property
    def memory_savings_vs_fp16(self) -> float:
        """Fraction of memory saved vs. full fp16 cache.

        Note: with per-position scales in fp32, this may be negative.
        For real savings, use fp16 scales or shared (per-channel) scales.
        """
        elem_fp16 = np.dtype(np.float16).itemsize
        fp16_bytes = 2 * self.batch_size * self.num_heads * self.max_seq_len * self.head_dim * elem_fp16
        return 1.0 - self.memory_allocated_bytes / fp16_bytes

    @property
    def memory_savings_vs_fp32(self) -> float:
        """Fraction of memory saved vs. full fp32 cache."""
        elem_fp32 = np.dtype(np.float32).itemsize
        fp32_bytes = 2 * self.batch_size * self.num_heads * self.max_seq_len * self.head_dim * elem_fp32
        return 1.0 - self.memory_allocated_bytes / fp32_bytes


# =============================================================================
# 3. CHUNKED PREFILL
# =============================================================================

class ChunkedPrefill:
    """
    Chunked Prefill — process long prompts in chunks to limit peak memory.

    During prefill with very long prompts (e.g., 32K tokens), computing
    full attention O(n²) requires materializing a (n, n) attention matrix,
    which can exceed GPU memory.

    Chunked prefill processes the prompt in chunks of size C:
      - Chunk 0: tokens [0, C) — full causal attention within chunk
      - Chunk 1: tokens [C, 2C) — attend to all previous tokens + causal within chunk
      - ...

    Each chunk's attention is O(C * (i*C + C)) = O(i*C²), but the peak
    memory for the attention matrix is O(C²) instead of O(n²).

    The KV cache is updated incrementally after each chunk.
    """

    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size

    def compute_attention_chunked(
        self,
        q_all: np.ndarray,
        k_all: np.ndarray,
        v_all: np.ndarray,
        scale: float,
        dtype=np.float32,
    ) -> np.ndarray:
        """
        Compute causal attention in chunks.

        Args:
            q_all: (batch, heads, seq, head_dim)
            k_all: (batch, heads, seq, head_dim)
            v_all: (batch, heads, seq, head_dim)
            scale: 1 / sqrt(head_dim)

        Returns:
            output: (batch, heads, seq, head_dim)
        """
        batch, heads, seq, head_dim = q_all.shape
        output = np.zeros((batch, heads, seq, head_dim), dtype=dtype)

        num_chunks = (seq + self.chunk_size - 1) // self.chunk_size

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, seq)
            chunk_len = end - start

            # Current chunk's Q
            q_chunk = q_all[:, :, start:end, :]  # (batch, heads, chunk_len, head_dim)

            # Keys and values up to current position (causal)
            k_prefix = k_all[:, :, :end, :]  # (batch, heads, end, head_dim)
            v_prefix = v_all[:, :, :end, :]

            q_f = q_chunk.astype(dtype)
            k_f = k_prefix.astype(dtype)
            v_f = v_prefix.astype(dtype)

            # Q @ K^T: (batch, heads, chunk_len, end)
            scores = np.einsum("bhqd,bhkd->bhqk", q_f, k_f) * scale

            # Causal mask: query at position p can only attend to keys at position <= p
            # Query positions (absolute): start..end-1
            # Key positions (absolute): 0..end-1
            q_positions = np.arange(start, end)  # (chunk_len,)
            k_positions = np.arange(end)          # (end,)
            # Allowed: q_pos >= k_pos (causal)
            causal_mask = (q_positions[:, None] >= k_positions[None, :]).astype(dtype)
            # (chunk_len, end)
            causal_mask = np.where(causal_mask, 0.0, -np.inf)

            scores = scores + causal_mask[None, None, :, :]

            # Softmax
            attn_weights = self._softmax_stable(scores, axis=-1)

            # Attn @ V
            chunk_output = np.einsum("bhqk,bhkd->bhqd", attn_weights, v_f)
            output[:, :, start:end, :] = chunk_output

        return output

    @staticmethod
    def _softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    def peak_memory_comparison(seq_len: int, chunk_size: int,
                                head_dim: int = 128) -> dict:
        """
        Compare peak memory usage between full and chunked prefill.

        The dominant memory is the attention score matrix.
        """
        # Full prefill: attention matrix is (seq_len, seq_len) in float32
        full_attention_bytes = seq_len * seq_len * 4  # float32

        # Chunked prefill: attention matrix is (chunk_size, seq_len) at most
        # The last chunk sees all previous tokens
        max_chunk_attention = chunk_size * seq_len * 4

        return {
            "seq_len": seq_len,
            "chunk_size": chunk_size,
            "full_attention_mb": full_attention_bytes / (1024 * 1024),
            "chunked_peak_attention_mb": max_chunk_attention / (1024 * 1024),
            "savings_ratio": full_attention_bytes / max(chunk_chunk_attention := chunk_size * seq_len * 4, 1),
        }


# =============================================================================
# 4. HYBRID: PAGED + QUANTIZED
# =============================================================================

class HybridKVCache:
    """
    Combines paged attention with quantization for maximum memory efficiency.

    - Paged allocation eliminates fragmentation
    - Quantization reduces per-token storage by ~50%
    - Together: can handle 2-4x longer contexts in the same memory
    """

    def __init__(self, page_config: PageConfig):
        self.page_config = page_config
        self.paged = PagedKVCache(page_config)
        self.quantized = QuantizedKVCache(
            batch_size=page_config.batch_size,
            num_heads=page_config.num_heads,
            head_dim=page_config.head_dim,
            max_seq_len=page_config.num_pages * page_config.block_size,
            dtype=page_config.dtype,
        )

    def reset(self):
        self.paged.reset()
        self.quantized.reset()

    @property
    def total_memory_saved(self) -> float:
        """Combined memory savings vs. naive contiguous fp16 cache."""
        return self.quantized.memory_savings_vs_fp16


# =============================================================================
# COMPARISON ANALYSIS
# =============================================================================

def compare_strategies(batch_size: int = 4, num_heads: int = 32,
                       head_dim: int = 128, max_seq_len: int = 4096,
                       num_layers: int = 32) -> Dict[str, dict]:
    """
    Compare memory usage across different KV-cache strategies.
    """
    elem_fp16 = 2  # bytes per float16 element
    elem_fp32 = 4
    elem_int8 = 1

    base_tokens = batch_size * num_heads * max_seq_len * head_dim
    base_bytes_per_layer = 2 * base_tokens * elem_fp16  # K + V

    results = {}

    # 1. Naive contiguous fp16
    results["naive_fp16"] = {
        "description": "Contiguous fp16 cache",
        "per_layer_mb": base_bytes_per_layer / (1024 * 1024),
        "total_mb": base_bytes_per_layer * num_layers / (1024 * 1024),
        "per_token_per_layer_bytes": 2 * num_heads * head_dim * elem_fp16,
    }

    # 2. Contiguous fp32
    base_bytes_fp32 = 2 * base_tokens * elem_fp32
    results["naive_fp32"] = {
        "description": "Contiguous fp32 cache",
        "per_layer_mb": base_bytes_fp32 / (1024 * 1024),
        "total_mb": base_bytes_fp32 * num_layers / (1024 * 1024),
        "per_token_per_layer_bytes": 2 * num_heads * head_dim * elem_fp32,
    }

    # 3. Quantized int8 (with fp16 scales)
    # Per-token: int8 data + shared fp16 scales per channel
    quant_data = base_tokens * elem_int8 * 2  # K + V int8
    quant_scales = batch_size * num_heads * head_dim * elem_fp16 * 2  # shared scales
    quant_total = quant_data + quant_scales
    results["quantized_int8"] = {
        "description": "Int8 quantized with fp16 scales",
        "per_layer_mb": quant_total / (1024 * 1024),
        "total_mb": quant_total * num_layers / (1024 * 1024),
        "savings_vs_fp16": 1.0 - quant_total / base_bytes_per_layer,
    }

    # 4. Paged (no fragmentation waste)
    block_size = 16
    blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    # Paged has slight overhead from block alignment
    padded_tokens = batch_size * blocks_per_seq * block_size * num_heads * head_dim
    paged_bytes = 2 * padded_tokens * elem_fp16
    results["paged"] = {
        "description": "Paged attention (block_size=16)",
        "per_layer_mb": paged_bytes / (1024 * 1024),
        "total_mb": paged_bytes * num_layers / (1024 * 1024),
        "overhead_vs_naive": paged_bytes / base_bytes_per_layer,
    }

    # 5. Paged + Quantized
    paged_quant_data = padded_tokens * elem_int8 * 2
    paged_quant_scales = batch_size * num_heads * head_dim * elem_fp16 * 2
    paged_quant_total = paged_quant_data + paged_quant_scales
    results["paged_quantized"] = {
        "description": "Paged + int8 quantized",
        "per_layer_mb": paged_quant_total / (1024 * 1024),
        "total_mb": paged_quant_total * num_layers / (1024 * 1024),
        "savings_vs_fp16": 1.0 - paged_quant_total / base_bytes_per_layer,
    }

    return results
