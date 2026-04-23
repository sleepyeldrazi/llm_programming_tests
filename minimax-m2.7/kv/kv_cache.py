"""
Efficient KV-Cache System for Autoregressive Transformer Inference

This module implements a complete KV-cache system from scratch, including:
- Data structures for cached keys and values
- Multi-head attention with cached KV pairs
- Batching support with variable sequence lengths
- Memory analysis and optimization strategies
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import math


# =============================================================================
# Part 1: Core Data Structures and Memory Layout
# =============================================================================

class MemoryFormat(Enum):
    """
    Memory layout options for KV-cache storage.
    
    Different layouts have different trade-offs for memory contiguity,
    access patterns, and cache efficiency.
    """
    # [batch, heads, seq_len, dim] - Standard format
    # Good for prefill, less efficient for decode attention
    BHSD = "batch_heads_seq_dim"
    
    # [batch, seq_len, heads, dim] - PyTorch default for attention
    BSHD = "batch_seq_heads_dim"
    
    # [batch, seq_len, heads, dim] with physical memory pages
    # Optimized for paged attention
    PAGED = "paged"
    
    # [heads, batch, seq_len, dim] - Good for sequence parallelism
    HBSD = "heads_batch_seq_dim"


@dataclass
class CacheConfig:
    """Configuration for KV-cache behavior."""
    max_batch_size: int = 32
    max_seq_len: int = 4096
    num_heads: int = 32
    head_dim: int = 128
    num_layers: int = 32
    dtype_bytes: int = 2  # FP16
    
    # Memory layout
    memory_format: MemoryFormat = MemoryFormat.BSHD
    
    # Block size for paged attention (must be power of 2)
    block_size: int = 16
    
    # Pre-allocation strategy
    pre_allocate: bool = True
    
    def memory_per_layer(self) -> int:
        """Bytes needed per layer for full sequence."""
        return 2 * self.num_heads * self.max_seq_len * self.head_dim * self.dtype_bytes
    
    def total_memory(self) -> int:
        """Total bytes for all layers."""
        return 2 * self.num_layers * self.memory_per_layer()


class KVCacheBlock:
    """
    A single block in the KV-cache.
    
    Memory Layout (BSHD format):
    ┌─────────────────────────────────────────────────┐
    │  k_cache: [max_seq_len, num_heads, head_dim]     │
    │  v_cache: [max_seq_len, num_heads, head_dim]     │
    └─────────────────────────────────────────────────┘
    
    Each block stores keys and values for a contiguous sequence segment.
    """
    
    def __init__(self, config: CacheConfig, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        self.block_size = config.block_size
        
        # Physical memory allocation
        self.k_data = np.zeros(
            (self.block_size, config.num_heads, config.head_dim),
            dtype=np.float16
        )
        self.v_data = np.zeros(
            (self.block_size, config.num_heads, config.head_dim),
            dtype=np.float16
        )
        
        # Logical state
        self.seq_offset: int = 0  # Starting position in full sequence
        self.seq_length: int = 0  # Current length (0 to block_size)
        self.is_full: bool = False
        
        # Access tracking for LRU/replacement
        self.last_access_step: int = 0
        self.access_count: int = 0
    
    @property
    def physical_size(self) -> Tuple[int, int, int]:
        """Return the shape of stored tensors."""
        return self.k_data.shape
    
    def write(self, k_chunk: np.ndarray, v_chunk: np.ndarray, position: int) -> None:
        """
        Write a chunk of keys and values to this block.
        
        Args:
            k_chunk: [chunk_len, num_heads, head_dim]
            v_chunk: [chunk_len, num_heads, head_dim]
            position: Position in block to write to (0-indexed)
        """
        chunk_len = k_chunk.shape[0]
        assert chunk_len <= self.block_size - position, "Chunk exceeds block capacity"
        
        self.k_data[position:position + chunk_len] = k_chunk
        self.v_data[position:position + chunk_len] = v_chunk
        self.seq_length = max(self.seq_length, position + chunk_len)
        
    def read(self, start: int, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Read a chunk from this block."""
        return (
            self.k_data[start:start + length].copy(),
            self.v_data[start:start + length].copy()
        )
    
    def clear(self) -> None:
        """Reset the block."""
        self.k_data.fill(0)
        self.v_data.fill(0)
        self.seq_length = 0
        self.is_full = False


class PagedKVCache:
    """
    Paged KV-Cache with block-wise management.
    
    Memory Layout:
    ┌────────────────────────────────────────────────────────────┐
    │                    Virtual Sequence                         │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
    │  │ Block 0  │ │ Block 1  │ │ Block 2  │ │ Block 3  │ ...  │
    │  │ (16 tok) │ │ (16 tok) │ │ (16 tok) │ │ (16 tok) │      │
    │  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
    └────────────────────────────────────────────────────────────┘
    │                    Physical Memory                          │
    │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                │
    │  │ GPU    │ │ GPU    │ │ GPU    │ │ CPU    │  ...           │
    │  │ Block  │ │ Block  │ │ Block  │ │ Block  │                │
    │  └────────┘ └────────┘ └────────┘ └────────┘                │
    └────────────────────────────────────────────────────────────┘
    
    Benefits:
    - No wasted memory from max_seq_len pre-allocation
    - Enables KV-cache offloading to CPU/disk
    - Supports heterogeneous sequence lengths in batch
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.blocks: List[KVCacheBlock] = []
        self.block_allocator: Dict[int, List[int]] = {}  # batch_idx -> list of block indices
        
        # Pre-allocate some blocks
        if config.pre_allocate:
            self._preallocate_blocks(num_blocks=config.max_batch_size * 2)
    
    def _preallocate_blocks(self, num_blocks: int) -> None:
        """Pre-allocate blocks for efficiency."""
        for _ in range(num_blocks):
            # We don't know layer_idx yet; use -1 as placeholder
            block = KVCacheBlock(self.config, layer_idx=-1)
            self.blocks.append(block)
    
    def allocate_block(self, layer_idx: int) -> int:
        """Allocate a new block, return its index."""
        # Find an unused block or create new one
        for idx, block in enumerate(self.blocks):
            if block.seq_length == 0 and block.layer_idx == -1:
                block.layer_idx = layer_idx
                return idx
        
        # No free block, create new one
        block = KVCacheBlock(self.config, layer_idx=layer_idx)
        self.blocks.append(block)
        return len(self.blocks) - 1
    
    def append(self, layer_idx: int, batch_idx: int,
               k_new: np.ndarray, v_new: np.ndarray) -> None:
        """
        Append new key/value to the cache for a specific batch element.
        
        Args:
            layer_idx: Which layer this cache belongs to
            batch_idx: Which element in the batch
            k_new: [1, num_heads, head_dim] - new key for single token
            v_new: [1, num_heads, head_dim] - new value for single token
        """
        if batch_idx not in self.block_allocator:
            self.block_allocator[batch_idx] = []
        
        block_indices = self.block_allocator[batch_idx]
        
        # Find the last block or allocate new one
        if block_indices:
            last_block_idx = block_indices[-1]
            last_block = self.blocks[last_block_idx]
            
            if not last_block.is_full:
                # Write to existing block
                pos = last_block.seq_length
                last_block.write(k_new, v_new, pos)
                return
        
        # Need new block
        new_block_idx = self.allocate_block(layer_idx)
        self.blocks[new_block_idx].write(k_new, v_new, position=0)
        block_indices.append(new_block_idx)
    
    def get_all_cached(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve all cached keys and values for a batch element.
        
        Returns:
            k_all: [seq_len, num_heads, head_dim]
            v_all: [seq_len, num_heads, head_dim]
        """
        if batch_idx not in self.block_allocator:
            return np.zeros((0, self.config.num_heads, self.config.head_dim)), \
                   np.zeros((0, self.config.num_heads, self.config.head_dim))
        
        # Calculate total sequence length
        total_len = 0
        for block_idx in self.block_allocator[batch_idx]:
            total_len += self.blocks[block_idx].seq_length
        
        # Gather all blocks
        k_concat = []
        v_concat = []
        
        for block_idx in self.block_allocator[batch_idx]:
            block = self.blocks[block_idx]
            k_concat.append(block.k_data[:block.seq_length])
            v_concat.append(block.v_data[:block.seq_length])
        
        if k_concat:
            return np.concatenate(k_concat, axis=0), np.concatenate(v_concat, axis=0)
        return np.zeros((0, self.config.num_heads, self.config.head_dim)), \
               np.zeros((0, self.config.num_heads, self.config.head_dim))


class FlatKVCache:
    """
    Flat pre-allocated KV-cache for maximum performance.
    
    Memory Layout:
    ┌──────────────────────────────────────────────────────────────────┐
    │  Shape: [num_layers, max_batch, max_seq_len, 2, num_heads, head_dim] │
    │                                                                      │
    │  ┌─ Layer 0 ─────────────────────────────────────────────────┐    │
    │  │  ┌─ Batch 0 ──────────────────────────────────────────┐  │    │
    │  │  │  K: [max_seq_len, num_heads, head_dim]               │  │    │
    │  │  │  V: [max_seq_len, num_heads, head_dim]               │  │    │
    │  │  └──────────────────────────────────────────────────────┘  │    │
    │  │  ┌─ Batch 1 ──────────────────────────────────────────┐  │    │
    │  │  │  ...                                                 │  │    │
    │  │  └──────────────────────────────────────────────────────┘  │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    │  ┌─ Layer 1 ─────────────────────────────────────────────────┐    │
    │  │  ...                                                       │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────────────────┘
    
    Use this when:
    - Sequence lengths are known or bounded
    - Memory is sufficient
    - Maximum throughput is required
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Flat allocation: [layers, batch, seq, 2, heads, dim]
        shape = (
            config.num_layers,
            config.max_batch_size,
            config.max_seq_len,
            2,  # K and V
            config.num_heads,
            config.head_dim
        )
        
        self.kv_data = np.zeros(shape, dtype=np.float16)
        self.seq_lengths = np.zeros((config.num_layers, config.max_batch_size), dtype=np.int32)
        
    def update(self, layer_idx: int, batch_idx: int, 
               seq_pos: int, k_new: np.ndarray, v_new: np.ndarray) -> None:
        """
        Update cache with new token at specific position.
        
        Args:
            layer_idx: Which layer
            batch_idx: Which batch element
            seq_pos: Position in sequence (0-indexed)
            k_new: [num_heads, head_dim]
            v_new: [num_heads, head_dim]
        """
        self.kv_data[layer_idx, batch_idx, seq_pos, 0] = k_new
        self.kv_data[layer_idx, batch_idx, seq_pos, 1] = v_new
        self.seq_lengths[layer_idx, batch_idx] = seq_pos + 1
    
    def get_kv_slice(self, layer_idx: int, batch_idx: int, 
                     start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get cached KV for a position range."""
        return (
            self.kv_data[layer_idx, batch_idx, start:end, 0].copy(),
            self.kv_data[layer_idx, batch_idx, start:end, 1].copy()
        )
    
    def get_full_kv(self, layer_idx: int, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get all cached KV for a batch element."""
        seq_len = self.seq_lengths[layer_idx, batch_idx]
        return (
            self.kv_data[layer_idx, batch_idx, :seq_len, 0].copy(),
            self.kv_data[layer_idx, batch_idx, :seq_len, 1].copy()
        )


# =============================================================================
# Part 2: Multi-Head Attention with KV-Cache
# =============================================================================

class AttentionPattern(Enum):
    """Attention computation patterns."""
    FULL = "full"           # All tokens attend to all (prefill)
    CAUSAL = "causal"       # Causal masking (autoregressive)
    SLIDING_WINDOW = "window"  # Local window attention


class MultiHeadAttention:
    """
    Multi-Head Attention with integrated KV-cache support.
    
    Architecture:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                         Multi-Head Attention                         │
    │                                                                       │
    │  Input: [batch, seq_len, d_model]                                    │
    │      │                                                                │
    │      ▼                                                                │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │  Q = x @ W_q    K = x @ W_k    V = x @ W_v    (Linear proj)    │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │      │                    │                    │                      │
    │      ▼                    ▼                    ▼                      │
    │  ┌────────────┐    ┌────────────┐        ┌────────────┐              │
    │  │ Q_heads    │    │ K + CacheK │        │ V + CacheV │              │
    │  │ [B,H,L,D]  │    │ [B,H,L',D] │        │ [B,H,L',D] │              │
    │  └────────────┘    └────────────┘        └────────────┘              │
    │      │                    │                                        │
    │      ▼                    ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │                    Attention Score                              │  │
    │  │         S = Q @ K^T / sqrt(d_k) + Mask                         │  │
    │  │                                                               │  │
    │  │         [B, H, L, L'] where L' includes cached tokens           │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │      │                                                              │
    │      ▼                                                              │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │                    Softmax                                     │  │
    │  │         A = softmax(S, dim=-1)                                 │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │      │                                                              │
    │      ▼                                                              │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │                    Output                                       │  │
    │  │         O = A @ V                                               │  │
    │  │         [B, H, L, D]                                            │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │      │                                                              │
    │      ▼                                                              │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │  out = O @ W_o    (Output projection)                          │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
        attention_pattern: AttentionPattern = AttentionPattern.CAUSAL
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.pattern = attention_pattern
        
        # Initialize weights (simplified - normally from trained model)
        self.W_q = np.random.randn(d_model, num_heads * head_dim).astype(np.float32) * 0.02
        self.W_k = np.random.randn(d_model, num_heads * head_dim).astype(np.float32) * 0.02
        self.W_v = np.random.randn(d_model, num_heads * head_dim).astype(np.float32) * 0.02
        self.W_o = np.random.randn(num_heads * head_dim, d_model).astype(np.float32) * 0.02
        
        # Bias
        if bias:
            self.b_q = np.zeros(num_heads * head_dim, dtype=np.float32)
            self.b_k = np.zeros(num_heads * head_dim, dtype=np.float32)
            self.b_v = np.zeros(num_heads * head_dim, dtype=np.float32)
            self.b_o = np.zeros(d_model, dtype=np.float32)
        else:
            self.b_q = self.b_k = self.b_v = self.b_o = None
    
    def _project(self, x: np.ndarray, W: np.ndarray, b: Optional[np.ndarray]) -> np.ndarray:
        """Project input to Q/K/V space."""
        # x: [batch, seq, d_model]
        result = np.matmul(x, W)
        if b is not None:
            result += b
        # Reshape to [batch, seq, heads, dim]
        batch, seq, _ = result.shape
        result = result.reshape(batch, seq, self.num_heads, self.head_dim)
        # Transpose to [batch, heads, seq, dim] for efficient attention
        return result.transpose(0, 2, 1, 3)
    
    def _create_causal_mask(self, seq_len: int, cache_len: int) -> np.ndarray:
        """Create causal mask for autoregressive generation."""
        total_len = seq_len + cache_len
        # [1, 1, seq, total_len] - broadcastable
        mask = np.triu(np.ones((seq_len, total_len), dtype=np.float32), k=1 - seq_len)
        return mask[np.newaxis, np.newaxis]
    
    def _apply_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Core attention computation: O = softmax(QK^T / sqrt(d_k))V
        
        Args:
            Q: [batch, heads, q_seq, dim]
            K: [batch, heads, k_seq, dim]
            V: [batch, heads, k_seq, dim]
            mask: Optional attention mask
            
        Returns:
            output: [batch, heads, q_seq, dim]
        """
        # Compute attention scores
        # Q @ K^T: [batch, heads, q_seq, k_seq]
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply mask
        if mask is not None:
            scores = scores + mask
        
        # Softmax
        # Subtract max for numerical stability
        scores_max = scores.max(axis=-1, keepdims=True)
        scores = np.exp(scores - scores_max)
        scores_sum = scores.sum(axis=-1, keepdims=True)
        attn_weights = scores / (scores_sum + 1e-8)
        
        # Apply to values
        output = np.matmul(attn_weights, V)
        
        return output
    
    def forward(
        self,
        x: np.ndarray,
        kv_cache: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
        layer_idx: int = 0,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Forward pass with optional KV-cache.
        
        Args:
            x: [batch, seq_len, d_model] - input tensor
            kv_cache: Dict mapping layer_idx -> (k_cache, v_cache) from previous layers
            layer_idx: Current layer index
            use_cache: Whether to read from/write to cache
            
        Returns:
            output: [batch, seq_len, d_model]
            kv_output: (k_current, v_current) for caching
        """
        batch, q_seq, _ = x.shape
        
        # Project to Q, K, V
        Q = self._project(x, self.W_q, self.b_q)  # [B, H, Q_seq, D]
        K = self._project(x, self.W_k, self.b_k)  # [B, H, Q_seq, D]
        V = self._project(x, self.W_v, self.b_v)  # [B, H, Q_seq, D]
        
        # Combine with cached K, V if available
        if use_cache and kv_cache and layer_idx in kv_cache:
            k_cache, v_cache = kv_cache[layer_idx]
            # k_cache: [B, H, cache_seq, D]
            K_full = np.concatenate([k_cache, K], axis=2)
            V_full = np.concatenate([v_cache, V], axis=2)
        else:
            K_full = K
            V_full = V
        
        # Create mask
        cache_len = 0
        if use_cache and kv_cache and layer_idx in kv_cache:
            cache_len = kv_cache[layer_idx][0].shape[2]
        
        mask = None
        if self.pattern == AttentionPattern.CAUSAL:
            mask = self._create_causal_mask(q_seq, cache_len)
            # Broadcast mask to batch and heads
            mask = np.broadcast_to(mask, (batch, self.num_heads, q_seq, q_seq + cache_len))
        
        # Compute attention
        attn_output = self._apply_attention(Q, K_full, V_full, mask)
        
        # Reshape and project output
        # [B, H, Q_seq, D] -> [B, Q_seq, H, D] -> [B, Q_seq, H*D]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, q_seq, -1)
        output = np.matmul(attn_output, self.W_o)
        if self.b_o is not None:
            output += self.b_o
            
        # Return new K, V for caching
        kv_output = (K, V) if use_cache else None
        
        return output, kv_output


# =============================================================================
# Part 3: Batching with Variable Sequence Lengths
# =============================================================================

@dataclass
class BatchElement:
    """
    Represents a single element in a batch.
    
    For autoregressive decoding, each element has:
    - Its own position in the sequence
    - Its own KV-cache
    - Potentially different sequence history
    """
    batch_idx: int
    prompt_len: int  # Length of initial prompt (prefill)
    current_len: int  # Current sequence length
    generated_tokens: List[int] = field(default_factory=list)
    logits: Optional[np.ndarray] = None
    finished: bool = False
    kv_cache_refs: Dict[int, int] = field(default_factory=dict)  # layer -> block indices


class BatchedInferenceEngine:
    """
    Manages batched inference with variable-length sequences.
    
    Key challenges:
    1. Different sequences have different lengths at each step
    2. We only compute attention for valid (non-padding) positions
    3. KV-caches must be indexed per-sequence
    4. Finished sequences should stop consuming compute
    
    Optimization strategies:
    - Padding to longest sequence in batch
    - Packing multiple sequences into one "pseudo-sequence"
    - Dynamic batch scheduling based on length
    """
    
    def __init__(self, config: CacheConfig, model: 'TransformerBlockStack'):
        self.config = config
        self.model = model
        self.batch: List[BatchElement] = []
        self.step: int = 0
        
        # Initialize KV-caches
        if config.pre_allocate:
            self.kv_cache = FlatKVCache(config)
        else:
            self.kv_cache = PagedKVCache(config)
    
    def add_to_batch(self, batch_idx: int, prompt_len: int) -> BatchElement:
        """Add an element to the batch."""
        elem = BatchElement(
            batch_idx=batch_idx,
            prompt_len=prompt_len,
            current_len=prompt_len
        )
        self.batch.append(elem)
        return elem
    
    def remove_from_batch(self, batch_idx: int) -> None:
        """Remove finished element from batch."""
        self.batch = [e for e in self.batch if e.batch_idx != batch_idx]
    
    def get_active_batch_size(self) -> int:
        """Get number of active (non-finished) elements."""
        return sum(1 for e in self.batch if not e.finished)
    
    def get_max_seq_len_in_batch(self) -> int:
        """Get maximum sequence length among active elements."""
        return max(e.current_len for e in self.batch if not e.finished)
    
    def _create_packed_input(
        self,
        input_tokens: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pack multiple sequences into one for efficient computation.
        
        Uses position encoding to prevent attention between sequences.
        
        Returns:
            packed_input: [1, total_packed_len, d_model]
            position_ids: [1, total_packed_len] - absolute positions
        """
        # In practice, this would concatenate embeddings with position info
        # For now, simplified version
        max_len = max(t.shape[0] for t in input_tokens)
        batch_size = len(input_tokens)
        
        # Pad sequences
        padded = []
        for tokens in input_tokens:
            if tokens.shape[0] < max_len:
                pad = np.zeros((max_len - tokens.shape[0], tokens.shape[1]))
                tokens = np.concatenate([tokens, pad], axis=0)
            padded.append(tokens)
        
        return np.stack(padded, axis=0), np.arange(max_len)[np.newaxis]
    
    def step_inference(
        self,
        input_embeddings: np.ndarray,  # [batch, 1, d_model]
        use_kv_cache: bool = True
    ) -> List[np.ndarray]:
        """
        Run one step of batched inference.
        
        Args:
            input_embeddings: [active_batch, 1, d_model] for new tokens
            use_kv_cache: Whether to use KV-caching
            
        Returns:
            List of output logits per batch element
        """
        batch_size = len(self.batch)
        outputs = []
        
        for i, elem in enumerate(self.batch):
            if elem.finished:
                outputs.append(np.zeros((self.config.max_seq_len, 1)))
                continue
            
            # Run model forward
            output = self.model.forward(
                input_embeddings[i:i+1],
                use_kv_cache=use_kv_cache,
                batch_idx=elem.batch_idx
            )
            
            elem.logits = output
            elem.current_len += 1
            outputs.append(output)
        
        self.step += 1
        return outputs


# =============================================================================
# Part 4: Complete Transformer Layer Integration
# =============================================================================

class TransformerBlock:
    """
    Single transformer block with KV-cache support.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                      Input x                                 │
    │                       │                                      │
    │                       ▼                                      │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │              Pre-Norm: x + LayerNorm(x)               │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                       │                                      │
    │                       ▼                                      │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │           Multi-Head Self-Attention                  │   │
    │  │            + KV-Cache (on decode)                     │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                       │                                      │
    │                       ▼                                      │
    │                 ┌─────────────┐                              │
    │                 │ Residual: + │                              │
    │                 └─────────────┘                              │
    │                       │                                      │
    │                       ▼                                      │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │              Pre-Norm: + LayerNorm(+)                 │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                       │                                      │
    │                       ▼                                      │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │           FFN (SwiGLU or FFN2)                        │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                       │                                      │
    │                       ▼                                      │
    │                 ┌─────────────┐                              │
    │                 │ Residual: + │                              │
    │                 └─────────────┘                              │
    │                       │                                      │
    │                       ▼                                      │
    │                      Output                                 │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        hidden_dim: int,
        layer_idx: int
    ):
        self.layer_idx = layer_idx
        
        # Attention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim
        )
        
        # Layer norms (simplified RMSNorm)
        self.norm1_weight = np.ones(d_model, dtype=np.float32)
        self.norm2_weight = np.ones(d_model, dtype=np.float32)
        
        # FFN
        self.W_up = np.random.randn(d_model, hidden_dim).astype(np.float32) * 0.02
        self.W_down = np.random.randn(hidden_dim, d_model).astype(np.float32) * 0.02
        
    def forward(
        self,
        x: np.ndarray,
        kv_cache: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forward pass through transformer block.
        
        Returns:
            output: Same shape as input
            kv_output: (K, V) tensors for caching
        """
        # Pre-norm for attention
        x_normed = x * self.norm1_weight / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        
        # Attention with cache
        attn_out, kv_output = self.attention.forward(
            x_normed, kv_cache, self.layer_idx, use_cache
        )
        
        # Residual
        x = x + attn_out
        
        # Pre-norm for FFN
        x_normed = x * self.norm2_weight / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        
        # FFN
        ffn_out = np.matmul(np.matmul(x_normed, self.W_up), self.W_down)
        
        # Residual
        x = x + ffn_out
        
        return x, kv_output


class TransformerBlockStack:
    """
    Stack of transformer blocks with shared KV-cache.
    
    Manages:
    - Layer-by-layer forward pass
    - KV-cache sharing across layers
    - Efficient batched inference
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        head_dim: int,
        hidden_dim: int
    ):
        self.num_layers = num_layers
        self.d_model = d_model
        
        # Create layers
        self.layers: List[TransformerBlock] = []
        for i in range(num_layers):
            self.layers.append(
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    hidden_dim=hidden_dim,
                    layer_idx=i
                )
            )
        
        # KV-cache store
        self.kv_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        
        # Final layer norm
        self.final_norm = np.ones(d_model, dtype=np.float32)
        
    def forward(
        self,
        x: np.ndarray,
        use_kv_cache: bool = True,
        batch_idx: int = 0
    ) -> np.ndarray:
        """
        Forward pass through all layers.
        
        Args:
            x: [batch, seq, d_model]
            use_kv_cache: Whether to read/write KV cache
            batch_idx: Which batch element to cache for
            
        Returns:
            output: [batch, seq, d_model]
        """
        for layer_idx, layer in enumerate(self.layers):
            # Prepare KV cache for this layer
            layer_cache = None
            if use_kv_cache and layer_idx in self.kv_cache:
                layer_cache = {layer_idx: self.kv_cache[layer_idx]}
            
            # Forward through layer
            x, kv_output = layer.forward(x, layer_cache, use_kv_cache)
            
            # Store KV for caching
            if use_kv_cache and kv_output is not None:
                self.kv_cache[layer_idx] = kv_output
        
        # Final norm
        x = x * self.final_norm / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        
        return x
    
    def clear_cache(self) -> None:
        """Clear all cached KV pairs."""
        self.kv_cache = {}


# =============================================================================
# Part 5: Autoregressive Generation Loop
# =============================================================================

class KVCacheAwareGenerator:
    """
    Autoregressive generator with KV-cache optimization.
    
    Generation flow:
    ┌────────────────────────────────────────────────────────────────────┐
    │                         Prefill Phase                              │
    │  ┌──────────────────────────────────────────────────────────────┐ │
    │  │  Prompt: "The cat"                                            │ │
    │  │  Process full sequence with attention                         │ │
    │  │  Cache all K, V for each layer                                │ │
    │  └──────────────────────────────────────────────────────────────┘ │
    │                            │                                      │
    │                            ▼                                      │
    │  ┌──────────────────────────────────────────────────────────────┐ │
    │  │                    Decode Phase                                │ │
    │  │                                                                 │ │
    │  │  Step 1:                                                     │ │
    │  │    Input: [token_id=456]  (single token)                       │ │
    │  │    Cache: [K_cache, V_cache] from prompt                       │ │
    │  │    Compute: Q(new) @ concat(K_cache, K_new)                   │ │
    │  │    Output: next token logits                                   │ │
    │  │    Update: Append K_new, V_new to cache                        │ │
    │  │                                                                 │ │
    │  │  Step 2:                                                     │ │
    │  │    Input: [new_token]                                          │ │
    │  │    Cache: Includes tokens from Step 1                          │ │
    │  │    Compute: ...                                               │ │
    │  │    ... repeat until EOS or max_len                            │ │
    │  └──────────────────────────────────────────────────────────────┘ │
    └────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Create model
        self.model = TransformerBlockStack(
            num_layers=config.num_layers,
            d_model=config.num_heads * config.head_dim,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            hidden_dim=4 * config.num_heads * config.head_dim
        )
        
        # Token embeddings (simplified)
        self.vocab_size = 32000
        self.embed_dim = config.num_heads * config.head_dim
        self.token_embedding = np.random.randn(self.vocab_size, self.embed_dim).astype(np.float32) * 0.02
        
        # Output projection
        self.lm_head = np.random.randn(self.embed_dim, self.vocab_size).astype(np.float32) * 0.02
        
        # Inference engine
        self.engine = BatchedInferenceEngine(config, self.model)
        
    def _embed_tokens(self, token_ids: np.ndarray) -> np.ndarray:
        """Convert token IDs to embeddings."""
        # token_ids: [batch, seq_len]
        # output: [batch, seq_len, embed_dim]
        return self.token_embedding[token_ids]
    
    def _sample_token(self, logits: np.ndarray, temperature: float = 1.0) -> int:
        """Sample next token from logits."""
        if temperature == 0:
            return np.argmax(logits)
        
        # Apply temperature
        logits = logits / temperature
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Sample
        return np.random.choice(len(logits), p=probs)
    
    def prefill(self, prompt_tokens: np.ndarray) -> None:
        """
        Prefill phase: process full prompt and populate KV cache.
        
        Args:
            prompt_tokens: [batch, prompt_len] token IDs
        """
        # Embed
        embeddings = self._embed_tokens(prompt_tokens)
        
        # Forward with cache population
        self.model.forward(embeddings, use_kv_cache=True)
        
    def decode_step(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Single decode step with cached attention.
        
        Args:
            token_ids: [batch, 1] - single token IDs
            
        Returns:
            next_token_logits: [batch, vocab_size]
        """
        # Embed new tokens
        embeddings = self._embed_tokens(token_ids)
        
        # Forward with cache (only computes Q for new token)
        hidden = self.model.forward(embeddings, use_kv_cache=True)
        
        # Project to vocabulary
        logits = np.matmul(hidden, self.lm_head)
        
        return logits.squeeze(1)  # [batch, vocab_size]
    
    def generate(
        self,
        prompt_tokens: np.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        eos_token: int = 2
    ) -> List[List[int]]:
        """
        Generate tokens autoregressively with KV-cache.
        
        Args:
            prompt_tokens: [batch, prompt_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            eos_token: End-of-sequence token ID
            
        Returns:
            List of generated sequences (excluding prompts)
        """
        batch_size = prompt_tokens.shape[0]
        
        # Prefill phase
        self.prefill(prompt_tokens)
        
        # Decode phase
        generated = [[] for _ in range(batch_size)]
        current_tokens = prompt_tokens[:, -1:]  # Last token
        
        for step in range(max_new_tokens):
            # Get logits
            logits = self.decode_step(current_tokens)
            
            # Sample tokens
            next_tokens = [self._sample_token(logits[i], temperature) 
                          for i in range(batch_size)]
            
            # Store generated
            for i in range(batch_size):
                generated[i].append(next_tokens[i])
            
            # Update for next step
            current_tokens = np.array(next_tokens)[:, np.newaxis]
            
            # Check for EOS
            if all(t == eos_token for t in next_tokens):
                break
        
        # Clear cache after generation
        self.model.clear_cache()
        
        return generated


# =============================================================================
# Part 6: Memory Analysis
# =============================================================================

class MemoryAnalyzer:
    """
    Analyzes memory growth patterns in KV-cache.
    
    Memory Growth Model:
    
    For a transformer with L layers, H heads, and D head dimension:
    
    Key memory formulas:
    
    1. Single Sequence Memory:
       M_seq = 2 * L * S * H * D * bytes_per_element
       
       where:
       - 2 = K and V tensors
       - S = sequence length
       - L = number of layers
       - H = number of attention heads
       - D = head dimension
    
    2. Per-token Memory Addition:
       M_token = 2 * L * H * D * bytes_per_element
       
       Each new token adds constant memory (O(1) per token)
       
    3. Attention Complexity Growth:
       - Prefill: O(S^2) for full attention
       - Decode step: O(S) for cached attention
       - This is where KV-cache provides exponential speedup
    
    4. Memory for Batch with Variable Lengths:
       M_batch = sum(M_seq_i for i in batch) + M_padding
       
    5. Long Sequence Analysis:
       At S=4096, L=32, H=32, D=128, FP16:
       M_seq = 2 * 32 * 4096 * 32 * 128 * 2 bytes
             ≈ 2 GB per sequence!
             
       Without cache: each decode step recomputes attention over 4096 tokens
       With cache: each decode step only computes Q for new token
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
    
    def memory_per_layer(self, seq_len: int) -> int:
        """Calculate memory for one layer at given sequence length."""
        return 2 * seq_len * self.config.num_heads * self.config.head_dim * self.config.dtype_bytes
    
    def memory_full_sequence(self) -> int:
        """Calculate memory for full max sequence."""
        return self.config.memory_per_layer() * self.config.num_layers
    
    def memory_growth_rate(self) -> int:
        """Memory added per new token."""
        return self.memory_per_layer(seq_len=1)
    
    def estimate_latency(self, seq_len: int, is_decode: bool = False) -> float:
        """
        Estimate latency based on sequence length.
        
        Args:
            seq_len: Current sequence length
            is_decode: True if this is a decode step (vs prefill)
            
        Returns:
            Estimated relative latency
        """
        if is_decode:
            # Decode: O(S) per step due to cache
            return seq_len
        else:
            # Prefill: O(S^2) for full attention
            return seq_len * seq_len
    
    def print_memory_analysis(self) -> None:
        """Print comprehensive memory analysis."""
        print("=" * 70)
        print("KV-CACHE MEMORY ANALYSIS")
        print("=" * 70)
        
        config = self.config
        
        print(f"\nConfiguration:")
        print(f"  - Num layers: {config.num_layers}")
        print(f"  - Num heads: {config.num_heads}")
        print(f"  - Head dim: {config.head_dim}")
        print(f"  - Max seq len: {config.max_seq_len}")
        print(f"  - Batch size: {config.max_batch_size}")
        print(f"  - Dtype: FP{config.dtype_bytes * 8}")
        
        print(f"\nMemory per layer at max seq:")
        for seq_len in [512, 1024, 2048, 4096, 8192, 16384]:
            mem = self.memory_per_layer(seq_len)
            print(f"  seq_len={seq_len:>6}: {mem / 1024**2:>8.2f} MB")
        
        print(f"\nMemory per token (growth rate):")
        per_token = self.memory_growth_rate()
        print(f"  {per_token / 1024**2:.4f} MB per token")
        
        print(f"\nTotal memory for full configuration:")
        total = self.memory_full_sequence()
        print(f"  Single sequence: {total / 1024**3:.2f} GB")
        print(f"  Full batch ({config.max_batch_size}): {total * config.max_batch_size / 1024**3:.2f} GB")
        
        print(f"\nLatency comparison (relative units):")
        print(f"  {'Seq Len':>10} | {'Prefill':>10} | {'Decode (cached)':>15} | {'Speedup':>10}")
        print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*15}-+-{'-'*10}")
        
        for seq_len in [128, 512, 1024, 2048, 4096]:
            prefill = self.estimate_latency(seq_len, is_decode=False)
            decode = self.estimate_latency(seq_len, is_decode=True)
            speedup = prefill / decode if decode > 0 else 0
            print(f"  {seq_len:>10} | {prefill:>10.0f} | {decode:>15.0f} | {speedup:>10.1f}x")
        
        print("\n" + "=" * 70)


# =============================================================================
# Part 7: Optimization Strategies
# =============================================================================

class OptimizationStrategies:
    """
    Optimization strategies for KV-cache.
    
    This module documents and implements key optimizations:
    
    1. PAGED ATTENTION
       ─────────────────────────────────────────────────────────────────────
       Problem: Pre-allocating max_seq_len for each sequence wastes memory
                when actual sequences are much shorter.
       
       Solution: Split KV-cache into fixed-size blocks (pages).
       
       ┌──────────────────────────────────────────────────────────────┐
       │  Instead of:                                                 │
       │  ┌──────────────────────────────────────────────────────┐    │
       │  │  [max_seq_len=4096, ...] (most unused!)               │    │
       │  └──────────────────────────────────────────────────────┘    │
       │                                                               │
       │  Use blocks:                                                  │
       │  ┌────┐ ┌────┐ ┌────┐ ┌────┐                                │
       │  │ 16 │ │ 16 │ │ 16 │ │ 16 │  ...  (allocate on demand)      │
       │  └────┘ └────┘ └────┘ └────┘                                │
       └──────────────────────────────────────────────────────────────┘
       
       Benefits:
       - 50-80% memory reduction for variable-length batches
       - Enables efficient KV offloading
       - Supports speculative decoding with variable-length candidates
       
       Implementation: See PagedKVCache class above
       
       ─────────────────────────────────────────────────────────────────────
       
    2. CHUNKED ATTENTION / FLASH ATTENTION
       ─────────────────────────────────────────────────────────────────────
       Problem: Attention score matrix S = Q @ K^T grows as O(S^2) in memory.
                For long sequences, this doesn't fit in SRAM/registers.
       
       Solution: Process attention in chunks that fit in fast memory.
       
       ┌──────────────────────────────────────────────────────────────┐
       │  Chunked attention algorithm:                                 │
       │                                                               │
       │  for chunk_q in Q_chunks:                                     │
       │      for chunk_k in K_chunks:                                 │
       │          S_chunk = chunk_q @ chunk_k.T                        │
       │          P_chunk = softmax(S_chunk)                          │
       │          O_chunk = P_chunk @ chunk_v                          │
       │          accumulate(O_chunk)                                  │
       │                                                               │
       │  Key insight: Only need O(chunk_size^2) intermediate memory   │
       └──────────────────────────────────────────────────────────────┘
       
       Benefits:
       - Enables processing of arbitrarily long sequences
       - ~2-4x speedup from better memory locality
       - Flash Attention: IO-aware chunking for GPU memory hierarchy
       
       ─────────────────────────────────────────────────────────────────────
       
    3. KV CACHE COMPRESSION
       ─────────────────────────────────────────────────────────────────────
       Problem: Storing full precision KV-cache for all tokens is expensive.
       
       Solutions:
       
       a) Key-Value Quantization:
          - Store K,V in INT8/FP8 instead of FP16/BF16
          - Use per-channel or per-token scaling
          - Typical compression: 2x (FP16→INT8)
          
       b) Sparse KV-cache:
          - Store only "important" tokens (high attention score)
          - Use locality-sensitive hashing for attention grouping
          - Mythomedia: keep only top-k per head per token
          
       c) Token Selection:
          - Heavy Head Hypothesis: Some attention heads are key
          - H2O: Keep tokens with highest attention flow
          - StreamingLLM: Keep recent + "sink" tokens
          
       d) Temporal Compression:
          - Differential coding: store delta from previous
          - Run-length encoding for repeated patterns
          
       ─────────────────────────────────────────────────────────────────────
       
    4. PARALLEL EXECUTION STRATEGIES
       ─────────────────────────────────────────────────────────────────────
       a) Tensor Parallelism:
          - Split heads across GPUs: Q, K, V distributed
          - All-reduce for attention output
          
       b) Pipeline Parallelism:
          - Different layers on different devices
          - Micro-batching to reduce pipeline bubbles
          - KV-cache passed between stages
          
       c) Sequence Parallelism:
          - Split sequence across attention heads
          - Ring attention for long sequences
          - Each GPU handles subset of heads
          
       ─────────────────────────────────────────────────────────────────────
       
    5. SPECULATIVE DECODING
       ─────────────────────────────────────────────────────────────────────
       Problem: Autoregressive decoding is sequential (N steps for N tokens).
       
       Solution: Draft multiple tokens speculatively, verify in parallel.
       
       ┌──────────────────────────────────────────────────────────────┐
       │  Draft model (smaller):                                       │
       │    Generates K candidate tokens in parallel                   │
       │                                                               │
       │  Target model (larger):                                      │
       │    Verifies all K tokens in ONE forward pass                 │
       │    Accept/reject based on logits                             │
       │                                                               │
       │  Speedup: ~2-4x when drafts are good                         │
       │  Key: KV-cache is reused for verification                     │
       └──────────────────────────────────────────────────────────────┘
    """
    
    @staticmethod
    def demonstrate_paged_attention() -> None:
        """Demonstrate memory savings from paged attention."""
        print("\n" + "=" * 70)
        print("PAGED ATTENTION DEMONSTRATION")
        print("=" * 70)
        
        # Example batch with variable lengths
        batch_sequences = [127, 256, 512, 1024, 2048, 4096]
        block_size = 16
        num_heads = 32
        head_dim = 128
        
        print(f"\nBlock size: {block_size} tokens")
        print(f"Heads: {num_heads}, Head dim: {head_dim}")
        print(f"\n{'Sequence':>10} | {'Tokens':>6} | {'Blocks':>6} | {'Paged Mem':>12} | {'Flat Mem':>12} | {'Savings':>10}")
        print("-" * 70)
        
        for seq_len in batch_sequences:
            # Paged: ceiling division for blocks
            num_blocks = (seq_len + block_size - 1) // block_size
            paged_mem = num_blocks * block_size * num_heads * head_dim * 2 * 2  # bytes (FP16)
            flat_mem = seq_len * num_heads * head_dim * 2 * 2
            savings = (1 - paged_mem / flat_mem) * 100 if flat_mem > 0 else 0
            
            print(f"{seq_len:>10} | {seq_len:>6} | {num_blocks:>6} | "
                  f"{paged_mem/1024**2:>10.2f}MB | {flat_mem/1024**2:>10.2f}MB | {savings:>9.1f}%")
        
        # Total comparison
        total_tokens = sum(batch_sequences)
        total_paged = sum((s + block_size - 1) // block_size * block_size for s in batch_sequences)
        total_flat = sum(batch_sequences)
        
        print(f"\n{'TOTAL':>10} | {total_tokens:>6} | {'':<6} | "
              f"{total_paged*num_heads*head_dim*2*2/1024**2:>10.2f}MB | "
              f"{total_flat*num_heads*head_dim*2*2/1024**2:>10.2f}MB | "
              f"{(1-total_paged/total_flat)*100:>9.1f}%")
    
    @staticmethod
    def demonstrate_quantization() -> None:
        """Demonstrate KV-cache quantization savings."""
        print("\n" + "=" * 70)
        print("KV-CACHE QUANTIZATION DEMONSTRATION")
        print("=" * 70)
        
        config = CacheConfig(
            max_batch_size=32,
            max_seq_len=4096,
            num_heads=32,
            head_dim=128,
            num_layers=32,
            dtype_bytes=2  # FP16
        )
        
        print(f"\nSequence length: {config.max_seq_len}")
        print(f"Layers: {config.num_layers}, Heads: {config.num_heads}, Head dim: {config.head_dim}")
        print(f"\n{'Format':>15} | {'Bytes/Element':>15} | {'Total Memory':>15} | {'Compression':>12}")
        print("-" * 70)
        
        formats = [
            ("FP32", 4),
            ("FP16", 2),
            ("BF16", 2),
            ("INT8", 1),
            ("INT4", 0.5),
            ("INT2", 0.25),
        ]
        
        base_mem = 2 * config.num_layers * config.max_seq_len * config.num_heads * config.head_dim * 2  # FP16 base
        
        for name, bytes_per_elem in formats:
            mem = 2 * config.num_layers * config.max_seq_len * config.num_heads * config.head_dim * bytes_per_elem
            ratio = base_mem / mem if mem > 0 else 0
            print(f"{name:>15} | {bytes_per_elem:>15.2f} | {mem/1024**3:>13.2f} GB | {ratio:>10.1f}x")
    
    @staticmethod
    def demonstrate_chunked_attention() -> None:
        """Demonstrate chunked attention computation."""
        print("\n" + "=" * 70)
        print("CHUNKED ATTENTION (FLASH ATTENTION STYLE)")
        print("=" * 70)
        
        # Simulate chunked attention
        seq_len = 4096
        head_dim = 128
        chunk_size = 256
        
        print(f"\nSequence length: {seq_len}")
        print(f"Head dimension: {head_dim}")
        print(f"Chunk size: {chunk_size}")
        print(f"\nMemory comparison:")
        
        # Full attention: S x S matrix
        full_attention_mem = seq_len * seq_len * 4  # 32-bit floats for scores
        print(f"  Full attention: {full_attention_mem/1024**2:.2f} MB for attention scores")
        
        # Chunked attention: only chunk_size^2 per chunk
        chunk_attention_mem = chunk_size * chunk_size * 4
        print(f"  Chunked attention: {chunk_attention_mem/1024**2:.2f} MB per chunk")
        print(f"  Memory reduction: {full_attention_mem/chunk_attention_mem:.0f}x")
        
        print(f"\n  Number of chunks: {seq_len // chunk_size}")
        print(f"  Processing: chunks processed sequentially, results accumulated")
        print(f"  Note: Total compute is same, but peak memory is reduced")


# =============================================================================
# Part 8: GPU Execution Mapping
# =============================================================================

class GPUExecutionMapper:
    """
    Maps KV-cache operations to GPU execution model.
    
    GPU Memory Hierarchy:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        HBM (High Bandwidth Memory)              │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │  KV-cache (main storage)                                  │  │
    │  │  - Flat format for current active sequences               │  │
    │  │  - Offloaded to CPU/disk for finished sequences           │  │
    │  │  Capacity: 80GB H100, 24-80GB A100                       │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘
    │                              │                                  │
    │                              │ ( PCIe / NVLink )                │
    │                              ▼                                  │
    ┌─────────────────────────────────────────────────────────────────┐
    │                        SM (Streaming Multiprocessor)           │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │  Shared Memory / L1 Cache                                │  │
    │  │  - Chunk of KV for current computation                   │  │
    │  │  - Tile size: 16-64KB per SM                             │  │
    │  │  Capacity: 192KB A100, 228KB H100                        │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │  Registers                                               │  │
    │  │  - Q, K, V tiles for current warp                        │  │
    │  │  - Per-thread: 255 registers A100, 65536 H100           │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘
    
    CUDA Kernel Design for KV-Cache Attention:
    
    1. KERNEL: kvcache_update
       ─────────────────────────────────────────────────────────────────
       Purpose: Store new K, V to global memory
       
       Thread grid: (batch * num_heads, head_dim / warp_size)
       Each thread writes one element
       
       __global__ void kvcache_update(
           const half* k_new,    // [batch, heads, 1, dim]
           const half* v_new,    // [batch, heads, 1, dim]
           half* k_cache,        // [batch, heads, max_seq, dim]
           half* v_cache,
           int* seq_lengths,    // Current lengths per batch
           int batch_idx,
           int layer_idx
       ) {
           int h = blockIdx.x % num_heads;
           int d = threadIdx.x;
           
           int pos = atomicAdd(&seq_lengths[layer_idx, batch_idx], 1);
           k_cache[batch_idx, h, pos, d] = k_new[batch_idx, h, 0, d];
           v_cache[batch_idx, h, pos, d] = v_new[batch_idx, h, 0, d];
       }
       
    2. KERNEL: attention_with_cache
       ─────────────────────────────────────────────────────────────────
       Purpose: Compute attention using cached K, V
       
       Thread grid: (batch, num_heads, num_chunks)
       Uses shared memory for tile storage
       
       Key optimizations:
       - Load K, V tiles from global to shared memory
       - Compute QK^T in shared memory
       - Apply causal mask inline
       - Softmax with warp-level reduction
       - Store results to output tensor
       
    3. KERNEL: attention_batch_update
       ─────────────────────────────────────────────────────────────────
       Purpose: Prefill phase - process multiple tokens at once
       
       Different from decode: processes S tokens instead of 1
       No causal masking needed (can compute full attention)
       
    GPU Memory Access Patterns:
    
    ┌──────────────────────────────────────────────────────────────────┐
    │  KV-Cache Access Pattern During Decode                          │
    │                                                                   │
    │  Q (new):    k_cache[b, h, pos, :]  →  strided access, pos = S-1 │
    │  K (cached): k_cache[b, h, 0:S, :] →  strided, S reads          │
    │  V (cached): v_cache[b, h, 0:S, :] →  strided, S reads          │
    │                                                                   │
    │  For batch B, heads H, dim D:                                   │
    │  - Q load: B * H * D elements (1 position)                      │
    │  - K load: B * H * S * D elements (full cache)                  │
    │  - V load: B * H * S * D elements (full cache)                  │
    │                                                                   │
    │  Memory bandwidth: O(B * H * S * D) per decode step              │
    │  This is the bottleneck!                                         │
    └──────────────────────────────────────────────────────────────────┘
    
    Optimization Strategies for GPU:
    
    1. Memory Coalescing:
       - Organize cache as [batch, heads, seq, dim] for coalesced access
       - Each thread accesses consecutive elements
       
    2. Tensor Core Usage:
       - Use WMMA (Warp Matrix Multiply Accumulate) for attention
       - Gemm operations: Q @ K^T, attention @ V
       
    3. Asynchronous Execution:
       - Overlap KV-cache updates with next forward pass
       - Use CUDA streams for independent operations
       
    4. Persistent Kernels:
       - Keep kernel resident for low-latency decode
       - Reduces kernel launch overhead
    """
    
    @staticmethod
    def print_gpu_execution_analysis() -> None:
        """Print detailed GPU execution analysis."""
        print("\n" + "=" * 70)
        print("GPU EXECUTION MAPPING FOR KV-CACHE")
        print("=" * 70)
        
        # GPU specifications
        gpus = {
            "A100": {"memory_gb": 80, "bandwidth_gbps": 2039, "sm_count": 108},
            "H100": {"memory_gb": 80, "bandwidth_gbps": 3350, "sm_count": 132},
            "L40S": {"memory_gb": 48, "bandwidth_gbps": 900, "sm_count": 84},
        }
        
        config = CacheConfig(
            max_batch_size=32,
            max_seq_len=4096,
            num_heads=32,
            head_dim=128,
            num_layers=32
        )
        
        tokens_per_layer = config.max_seq_len * config.num_heads * config.head_dim
        bytes_per_layer = tokens_per_layer * config.dtype_bytes * 2  # K and V
        total_bytes = bytes_per_layer * config.num_layers
        
        print(f"\nKV-cache size (full config): {total_bytes / 1024**3:.2f} GB")
        print(f"\nGPU Specifications:")
        
        for gpu_name, specs in gpus.items():
            cache_fit = "YES" if total_bytes < specs["memory_gb"] * 1024**3 else "NO"
            bw_time = total_bytes / (specs["bandwidth_gbps"] * 1e9) * 1000
            
            print(f"\n  {gpu_name}:")
            print(f"    Memory: {specs['memory_gb']} GB")
            print(f"    Bandwidth: {specs['bandwidth_gbps']} GB/s")
            print(f"    SMs: {specs['sm_count']}")
            print(f"    KV-cache fits: {cache_fit}")
            print(f"    Time to read full cache: {bw_time:.2f} ms")
        
        print("\n" + "-" * 70)
        print("EXECUTION PIPELINE")
        print("-" * 70)
        
        print("""
┌──────────────────────────────────────────────────────────────────────┐
│                      DECODE STEP EXECUTION                           │
│                                                                      │
│  Step N:                                                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 1. Load Q (current token) from hidden states                 │   │
│  │    ← Compute previous layer                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 2. Load KV cache from HBM                                    │   │
│  │    [batch * heads * seq * dim] ← main memory                │   │
│  │    Latency: ~1-2 μs per access (H100)                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 3. Compute Q @ K^T (attention scores)                        │   │
│  │    Tensor cores: 256x128x128 GEMM per batch element          │   │
│  │    Output: [batch, heads, 1, seq]                           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 4. Softmax with causal mask                                  │   │
│  │    Warp-level reduction                                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 5. Compute softmax(QK^T) @ V                                 │   │
│  │    Second GEMM: [batch, heads, 1, dim]                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 6. Update KV cache (write new K, V)                          │   │
│  │    → HBM async write                                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 7. Output projection + FFN                                   │   │
│  │    Standard transformer layers                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│                      Step N+1 ready                                │
└──────────────────────────────────────────────────────────────────────┘

        """)
        
        print("-" * 70)
        print("CRITICAL BOTTLENECKS AND SOLUTIONS")
        print("-" * 70)
        
        bottlenecks = [
            ("Memory Bandwidth", 
             "Full KV cache read every decode step",
             "Paged attention + async copy, KV compression"),
            ("Attention O(S²)",
             "Score matrix grows with sequence",
             "Flash Attention chunking, sparse attention"),
            ("Synchronization",
             "Wait for KV update before next step",
             "Lookahead decoding, speculative execution"),
            ("Memory Capacity",
             "Can't cache all sequences indefinitely",
             "KV offloading, selective caching"),
        ]
        
        for name, problem, solution in bottlenecks:
            print(f"\n{name}:")
            print(f"  Problem: {problem}")
            print(f"  Solution: {solution}")


# =============================================================================
# Part 9: Demonstration and Testing
# =============================================================================

def run_demo():
    """Run comprehensive demonstration of KV-cache system."""
    
    print("=" * 70)
    print("KV-CACHE SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Configuration
    config = CacheConfig(
        max_batch_size=8,
        max_seq_len=512,  # Reduced for demo
        num_heads=4,      # Reduced for demo
        head_dim=32,     # Reduced for demo
        num_layers=2,    # Reduced for demo
        dtype_bytes=2
    )
    
    # Memory analysis
    analyzer = MemoryAnalyzer(config)
    analyzer.print_memory_analysis()
    
    # Optimizations
    OptimizationStrategies.demonstrate_paged_attention()
    OptimizationStrategies.demonstrate_quantization()
    OptimizationStrategies.demonstrate_chunked_attention()
    
    # GPU execution
    GPUExecutionMapper.print_gpu_execution_analysis()
    
    # Functional test
    print("\n" + "=" * 70)
    print("FUNCTIONAL TEST")
    print("=" * 70)
    
    # Create small model for testing
    test_config = CacheConfig(
        max_batch_size=4,
        max_seq_len=64,
        num_heads=2,
        head_dim=16,
        num_layers=2,
        dtype_bytes=2
    )
    
    generator = KVCacheAwareGenerator(test_config)
    
    # Simple test
    prompt = np.array([[1, 2, 3, 4, 5]])  # Batch of 1, seq of 5
    print(f"\nInput prompt: shape {prompt.shape}")
    
    # Prefill
    generator.prefill(prompt)
    print("Prefill complete - KV cache populated")
    
    # Decode steps
    print("\nDecode steps:")
    for i in range(3):
        next_token = np.array([[prompt[0, -1] + i + 1]])  # Simulated next token
        logits = generator.decode_step(next_token)
        print(f"  Step {i+1}: output shape {logits.shape}, max logit at position {np.argmax(logits)}")
    
    print("\nTest completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
