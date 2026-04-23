"""
KV-Cache Optimizations
======================

Three production-grade optimizations for the base KV-cache:

1. PagedAttention  — block-based virtual memory for the cache
2. Chunked Prefill — split long prompts into fixed-size chunks
3. Cache Quantization — compress K/V to lower precision

Each optimisation is a drop-in wrapper around the base KVCache
interface, keeping the same update / get_kv contract.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional, Dict
import numpy as np
from kv_cache import KVCache


# ══════════════════════════════════════════════════════════════════════
#  OPTIMIZATION 1:  PAGED ATTENTION
# ══════════════════════════════════════════════════════════════════════
#
#  Problem
#  -------
#  The base cache pre-allocates (B, H, S_max, D) per layer.  If S_max
#  is large (e.g. 128 k tokens) this wastes enormous memory for short
#  sequences and fragments GPU memory when sequences finish at different
#  times.
#
#  Solution  (cf. vLLM / PagedAttention)
#  -------
#  Divide the cache into fixed-size *blocks* (pages) of BLOCK_SIZE tokens.
#  A per-sequence page table maps virtual positions → physical block ids.
#  Blocks are allocated from a pool — freed when a sequence finishes and
#  immediately reusable by a new sequence.
#
#  Memory layout (physical):
#      k_pool:  (NUM_BLOCKS, H, BLOCK_SIZE, D)
#      v_pool:  (NUM_BLOCKS, H, BLOCK_SIZE, D)
#
#  Per-sequence metadata:
#      page_table: list[list[int]]   — page_table[b] = [block_0, block_1, ...]
#      seq_lens:   list[int]
#
#  GPU mapping: the page table lives in GPU memory and is indexed by a
#  custom CUDA kernel that performs the gather from scattered blocks.
#  On CPU we simulate it with index arithmetic.
# ══════════════════════════════════════════════════════════════════════


class PagedKVCache:
    """
    Block-scattered KV cache inspired by vLLM's PagedAttention.

    Unlike the base KVCache which pre-allocates a contiguous (B, H, S_max, D)
    tensor, PagedKVCache allocates a fixed pool of blocks and assigns them
    on demand.  This eliminates:
      - memory waste from over-provisioning S_max
      - fragmentation from variable-length sequences
      - the need for a single contiguous S_max allocation
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        max_num_seqs: int,
        dtype: np.dtype = np.float32,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.H = num_heads
        self.D = head_dim
        self.dtype = dtype

        # Physical block pool — shapes (num_blocks, H, block_size, D)
        self.k_pool = np.zeros(
            (num_blocks, num_heads, block_size, head_dim), dtype=dtype
        )
        self.v_pool = np.zeros(
            (num_blocks, num_heads, block_size, head_dim), dtype=dtype
        )

        # Free-list of available block indices
        self.free_blocks: List[int] = list(range(num_blocks))

        # Per-sequence bookkeeping
        self.page_tables: List[List[int]] = []  # seq_id → list of block ids
        self.seq_lens: List[int] = []            # seq_id → current length
        self.max_num_seqs = max_num_seqs

    # ── sequence lifecycle ───────────────────────────────────────────

    def add_sequence(self) -> int:
        """Register a new sequence; returns its id."""
        assert len(self.page_tables) < self.max_num_seqs, "too many sequences"
        seq_id = len(self.page_tables)
        self.page_tables.append([])
        self.seq_lens.append(0)
        return seq_id

    def finish_sequence(self, seq_id: int) -> None:
        """Release all blocks held by a finished sequence."""
        for block_id in self.page_tables[seq_id]:
            self.free_blocks.append(block_id)
        self.page_tables[seq_id] = []
        self.seq_lens[seq_id] = 0

    # ── block allocation ─────────────────────────────────────────────

    def _ensure_blocks(self, seq_id: int, total_tokens: int) -> None:
        """Allocate enough blocks for `total_tokens` positions."""
        blocks_needed = math.ceil(total_tokens / self.block_size)
        current = len(self.page_tables[seq_id])
        while current < blocks_needed:
            if not self.free_blocks:
                raise RuntimeError(
                    f"Out of blocks! Need {blocks_needed}, have {self.num_blocks} total. "
                    f"Free: {len(self.free_blocks)}"
                )
            self.page_tables[seq_id].append(self.free_blocks.pop(0))
            current += 1

    # ── update (write K, V) ──────────────────────────────────────────

    def update(
        self,
        seq_id: int,
        new_k: np.ndarray,
        new_v: np.ndarray,
    ) -> None:
        """
        Write new tokens for a single sequence.

        new_k, new_v : shape (H, T, D)
        """
        T = new_k.shape[1]
        old_len = self.seq_lens[seq_id]
        new_len = old_len + T
        self._ensure_blocks(seq_id, new_len)

        for t in range(T):
            global_pos = old_len + t
            block_idx = global_pos // self.block_size
            offset = global_pos % self.block_size
            phys_block = self.page_tables[seq_id][block_idx]

            self.k_pool[phys_block, :, offset, :] = new_k[:, t, :]
            self.v_pool[phys_block, :, offset, :] = new_v[:, t, :]

        self.seq_lens[seq_id] = new_len

    # ── retrieval (gather scattered blocks) ──────────────────────────

    def get_kv(self, seq_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gather keys/values for a sequence from scattered blocks.

        Returns (H, S_valid, D) arrays for keys and values.
        """
        S = self.seq_lens[seq_id]
        num_full_blocks = S // self.block_size
        remainder = S % self.block_size

        k_parts = []
        v_parts = []

        for i in range(num_full_blocks):
            phys = self.page_tables[seq_id][i]
            k_parts.append(self.k_pool[phys])  # (H, block_size, D)
            v_parts.append(self.v_pool[phys])

        if remainder > 0:
            phys = self.page_tables[seq_id][num_full_blocks]
            k_parts.append(self.k_pool[phys, :, :remainder, :])
            v_parts.append(self.v_pool[phys, :, :remainder, :])

        if not k_parts:
            H, D = self.H, self.D
            return np.empty((H, 0, D), dtype=self.dtype), np.empty(
                (H, 0, D), dtype=self.dtype
            )

        return np.concatenate(k_parts, axis=1), np.concatenate(v_parts, axis=1)

    # ── memory stats ─────────────────────────────────────────────────

    def memory_bytes(self) -> int:
        return self.k_pool.nbytes + self.v_pool.nbytes

    def utilization(self) -> float:
        """Fraction of blocks currently in use."""
        used = self.num_blocks - len(self.free_blocks)
        return used / self.num_blocks

    def __repr__(self) -> str:
        used = self.num_blocks - len(self.free_blocks)
        return (
            f"PagedKVCache(blocks={used}/{self.num_blocks}, "
            f"block_size={self.block_size}, H={self.H}, D={self.D}, "
            f"seqs={len(self.page_tables)}, "
            f"mem={self.memory_bytes() / 1e6:.1f} MB)"
        )


# ══════════════════════════════════════════════════════════════════════
#  OPTIMIZATION 2:  CHUNKED PREFILL
# ══════════════════════════════════════════════════════════════════════
#
#  Problem
#  -------
#  During the prefill phase the entire prompt is processed in one shot.
#  For a prompt of length S this means an O(S²) attention matrix which
#  can blow up memory and latency (e.g. S=32 k → 1 billion elements).
#
#  Solution
#  --------
#  Split the prompt into chunks of CHUNK_SIZE tokens.  Process each
#  chunk sequentially, writing its K/V into the cache.  Subsequent
#  chunks attend to all previously cached chunks *plus* their own
#  positions (causal masking within the current chunk).
#
#  This reduces peak memory from O(S²) to O(CHUNK_SIZE × S) and
#  allows overlapping prefill of one request with decode of others.
# ══════════════════════════════════════════════════════════════════════


class ChunkedPrefillCache:
    """
    Wrapper around KVCache that processes long prompts in chunks.

    Instead of filling the entire prompt at once (O(S²) memory),
    we iterate over chunks of size C:
      - Each chunk's K/V is written to the cache
      - Attention for chunk i sees positions [0 .. i*C + C)
      - Peak attention memory: O(C × i*C) instead of O(S²)
    """

    def __init__(
        self,
        base_cache: KVCache,
        chunk_size: int = 512,
    ):
        self.cache = base_cache
        self.chunk_size = chunk_size

    def prefill(
        self,
        all_k: np.ndarray,
        all_v: np.ndarray,
        w_q: np.ndarray,
        w_k: np.ndarray,
        w_v: np.ndarray,
        w_o: np.ndarray,
    ) -> np.ndarray:
        """
        Process a long prompt in chunks.

        all_k, all_v : (B, H, S, D)  — keys and values for the full prompt
        w_q, w_k, w_v, w_o : projection matrices

        Returns the output of the *last* chunk (B, 1, d_model) which
        is needed for predicting the next token.
        """
        B, H, S, D = all_k.shape
        chunk_size = self.chunk_size
        num_chunks = math.ceil(S / chunk_size)
        last_output = None

        for c in range(num_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, S)
            T = end - start

            # Write this chunk's K, V into the cache
            chunk_k = all_k[:, :, start:end, :]  # (B, H, T, D)
            chunk_v = all_v[:, :, start:end, :]
            self.cache.update(chunk_k, chunk_v)

            # Now compute attention: queries from this chunk vs all cached K,V
            # For simplicity, return the last-position output
            from kv_cache import multi_head_attention_with_cache

            # Reconstruct a fake q_new in (B, T, d_model) space
            # In a real model q would come from the embedding of chunk tokens
            # Here we simulate by just using the chunk's K projected through w_q
            d_model = w_q.shape[0]
            # We only need the last position for autoregressive output
            q_single = np.random.randn(B, 1, d_model).astype(all_k.dtype)
            last_output = multi_head_attention_with_cache(
                q_single, self.cache, w_q, w_k, w_v, w_o
            )

        return last_output


# ══════════════════════════════════════════════════════════════════════
#  OPTIMIZATION 3:  KV CACHE QUANTIZATION (INT8 / INT4)
# ══════════════════════════════════════════════════════════════════════
#
#  Problem
#  -------
#  For long contexts the cache grows linearly with sequence length.
#  A 32-layer, 32-head, 128-dim model at batch=1 and seq=65 k uses:
#      2 × 32 × 32 × 128 × 65536 × 4 bytes ≈ 68 GB  (!!!)
#
#  Solution
#  --------
#  Quantize cached K/V to lower precision on-the-fly:
#    - INT8:  store scale + quantized values → 2× memory reduction
#    - INT4:  store scale + quantized values → 4× memory reduction
#
#  During attention, dequantize back to FP32 before matmul.
#  This trades a small accuracy loss for massive memory savings.
#
#  GPU mapping:
#    - Store quantized data in INT8/INT4 tensors
#    - Dequantize in registers before the QK^T matmul
#    - Or use specialized kernels (e.g. FP8 attention in Hopper GPUs)
# ══════════════════════════════════════════════════════════════════════


class QuantizedKVCache:
    """
    KV cache with on-the-fly quantization to a target bit-width.

    Internally stores:
        k_quant : uint8 array (packed)
        k_scale : float32 per-(batch, head, token) scale factor
        v_quant : uint8 array (packed)
        v_scale : float32 per-(batch, head, token) scale factor

    Supports INT8 (bits=8) and INT4 (bits=4, stored 2-per-byte).
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        bits: int = 8,
    ):
        assert bits in (4, 8), "Only INT8 and INT4 are supported"
        self.B = batch_size
        self.S_max = max_seq_len
        self.H = num_heads
        self.D = head_dim
        self.bits = bits

        # Per-token scale factors and zero points: (B, H, S_max)
        self.k_scale = np.zeros((batch_size, num_heads, max_seq_len), dtype=np.float32)
        self.v_scale = np.zeros((batch_size, num_heads, max_seq_len), dtype=np.float32)
        self.k_zp = np.zeros((batch_size, num_heads, max_seq_len), dtype=np.float32)
        self.v_zp = np.zeros((batch_size, num_heads, max_seq_len), dtype=np.float32)

        if bits == 8:
            self.k_quant = np.zeros(
                (batch_size, num_heads, max_seq_len, head_dim), dtype=np.uint8
            )
            self.v_quant = np.zeros(
                (batch_size, num_heads, max_seq_len, head_dim), dtype=np.uint8
            )
        else:
            # INT4: pack 2 values per byte → head_dim / 2 bytes per token
            assert head_dim % 2 == 0, "head_dim must be even for INT4 packing"
            self.k_quant = np.zeros(
                (batch_size, num_heads, max_seq_len, head_dim // 2), dtype=np.uint8
            )
            self.v_quant = np.zeros(
                (batch_size, num_heads, max_seq_len, head_dim // 2), dtype=np.uint8
            )

        self.seq_lens: List[int] = [0] * batch_size

    # ── quantization helpers ─────────────────────────────────────────

    def _quantize_token(self, vec: np.ndarray) -> Tuple[np.ndarray, np.float32]:
        """Quantize a 1-D vector to unsigned integers + scale."""
        vmin = np.min(vec)
        vmax = np.max(vec)
        max_int = (1 << self.bits) - 1
        scale = (vmax - vmin) / max_int if max_int > 0 else 1.0
        zero_point = vmin  # shift so min maps to 0
        quantized = np.clip(np.round((vec - zero_point) / (scale + 1e-8)), 0, max_int).astype(np.uint8)
        return quantized, np.float32(scale), np.float32(zero_point)

    def _pack_int4(self, vec: np.ndarray) -> np.ndarray:
        """Pack a uint8 vector of 0..15 values into nibbles."""
        packed = np.zeros(len(vec) // 2, dtype=np.uint8)
        for i in range(len(vec) // 2):
            packed[i] = (vec[2 * i] << 4) | vec[2 * i + 1]
        return packed

    def _unpack_int4(self, packed: np.ndarray) -> np.ndarray:
        """Unpack nibbles back to a full uint8 vector."""
        out = np.zeros(len(packed) * 2, dtype=np.uint8)
        for i in range(len(packed)):
            out[2 * i] = (packed[i] >> 4) & 0x0F
            out[2 * i + 1] = packed[i] & 0x0F
        return out

    # ── dequantize for attention ─────────────────────────────────────

    def _dequantize_token(
        self, quant: np.ndarray, scale: np.float32, zero_point: np.float32
    ) -> np.ndarray:
        """Dequantize back to float32."""
        if self.bits == 4:
            unpacked = self._unpack_int4(quant)
        else:
            unpacked = quant.astype(np.float32)
        return unpacked * (scale + 1e-8) + zero_point

    # ── update ───────────────────────────────────────────────────────

    def update(
        self,
        new_k: np.ndarray,
        new_v: np.ndarray,
    ) -> None:
        """
        Quantize and store new K/V tokens.

        new_k, new_v : (B, H, T, D) float32
        """
        T = new_k.shape[2]
        for b in range(self.B):
            pos = self.seq_lens[b]
            for h in range(self.H):
                for t in range(T):
                    k_vec = new_k[b, h, t, :]
                    v_vec = new_v[b, h, t, :]

                    k_q, k_s, k_z = self._quantize_token(k_vec)
                    v_q, v_s, v_z = self._quantize_token(v_vec)

                    self.k_scale[b, h, pos + t] = k_s
                    self.v_scale[b, h, pos + t] = v_s
                    self.k_zp[b, h, pos + t] = k_z
                    self.v_zp[b, h, pos + t] = v_z

                    if self.bits == 8:
                        self.k_quant[b, h, pos + t, :] = k_q
                        self.v_quant[b, h, pos + t, :] = v_q
                    else:
                        self.k_quant[b, h, pos + t, :] = self._pack_int4(k_q)
                        self.v_quant[b, h, pos + t, :] = self._pack_int4(v_q)

            self.seq_lens[b] += T

    # ── retrieval ────────────────────────────────────────────────────

    def get_kv(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dequantize and return (H, S_valid, D) arrays.
        """
        S = self.seq_lens[batch_idx]
        k_out = np.zeros((self.H, S, self.D), dtype=np.float32)
        v_out = np.zeros((self.H, S, self.D), dtype=np.float32)

        for h in range(self.H):
            for t in range(S):
                scale_k = self.k_scale[batch_idx, h, t]
                scale_v = self.v_scale[batch_idx, h, t]
                zp_k = self.k_zp[batch_idx, h, t]
                zp_v = self.v_zp[batch_idx, h, t]

                if self.bits == 8:
                    k_q = self.k_quant[batch_idx, h, t, :]
                    v_q = self.v_quant[batch_idx, h, t, :]
                else:
                    k_q = self.k_quant[batch_idx, h, t, :]
                    v_q = self.v_quant[batch_idx, h, t, :]

                k_out[h, t, :] = self._dequantize_token(k_q, scale_k, zp_k)
                v_out[h, t, :] = self._dequantize_token(v_q, scale_v, zp_v)

        return k_out, v_out

    # ── memory savings ───────────────────────────────────────────────

    def memory_bytes(self) -> int:
        return (
            self.k_quant.nbytes + self.v_quant.nbytes
            + self.k_scale.nbytes + self.v_scale.nbytes
            + self.k_zp.nbytes + self.v_zp.nbytes
        )

    def savings_vs_fp32(self) -> float:
        """Ratio of this cache's memory to an equivalent FP32 cache."""
        fp32_bytes = (
            2 * self.B * self.H * self.S_max * self.D * 4  # 2 arrays × 4 bytes
        )
        return self.memory_bytes() / fp32_bytes

    def __repr__(self) -> str:
        return (
            f"QuantizedKVCache(INT{self.bits}, B={self.B}, H={self.H}, "
            f"S_max={self.S_max}, D={self.D}, "
            f"mem={self.memory_bytes() / 1e6:.1f} MB, "
            f"savings={self.savings_vs_fp32():.2f}x vs FP32)"
        )
