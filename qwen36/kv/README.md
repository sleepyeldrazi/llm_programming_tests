# KV-Cache System for Autoregressive Transformer Inference

Pure NumPy implementation — no frameworks. Demonstrates the complete KV-cache pipeline from data structures through GPU mapping.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                            │
│                                                                 │
│  Prompt ──→ [Prefill] ──→ KV Cache populated ──→ [Generate]   │
│              O(n²) attn       O(1) per token      O(seq) attn   │
│                                                                 │
│  Per generation step:                                           │
│    1. Embed + positional encoding                               │
│    2. For each layer:                                           │
│       a. LayerNorm → QKV projection                             │
│       b. Store K,V in cache (append at write_pos)               │
│       c. Cached attention: Q @ K_cache^T → softmax → @ V_cache │
│       d. Output projection → MLP → residual                     │
│    3. LM head → logits → sample next token                      │
└─────────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `kv_cache.py` | Core KV-cache data structures (`KVCache`, `BatchedKVCache`) |
| `attention.py` | Attention computation (standard, cached, GQA, masked) |
| `transformer.py` | Full transformer decoder layer + model with KV-cache integration |
| `optimizations.py` | Paged attention, quantization, chunked prefill |
| `memory_analysis.py` | Memory growth formulas, model size comparisons, GPU limits |
| `gpu_mapping.py` | GPU kernel design, Tensor Core analysis, multi-GPU strategies |
| `demo.py` | 10 end-to-end demos exercising every component |

## 1. Data Structure Layout

### Memory Format

```
cache_k[batch, num_heads, max_seq_len, head_dim]    # float16
cache_v[batch, num_heads, max_seq_len, head_dim]    # float16
lengths[batch]                                      # int32 (actual seq len per item)
write_pos                                           # int (global write pointer)
```

**Why this layout:**
- `batch` first → enables batched GEMM on GPU
- `heads` second → parallel head computation
- `seq_len` third → contiguous scan for Q @ K^T
- `head_dim` last → inner product dimension, coalesced access

### Per-Token Memory Cost

For a 7B model (32 layers, 32 heads, head_dim=128, fp16):

```
Per token per layer: 2 × 32 × 128 × 2 bytes = 16 KB
Per token (all layers): 16 KB × 32 = 512 KB
At 32K context: 512 KB × 32,768 = 16 GB
```

## 2. Update Logic Per Step

```python
# Each generation step:
pos = cache.write_pos
cache.cache_k[:, :, pos, :] = new_k[:, :, 0, :]   # O(1) write
cache.cache_v[:, :, pos, :] = new_v[:, :, 0, :]   # O(1) write
cache.write_pos += 1
```

The write is a simple memory copy — no computation needed. The cache grows by exactly `2 × heads × head_dim × elem_bytes` per token per layer.

## 3. Attention Computation Using Cache

```python
# Retrieve all cached K, V
cached_k, cached_v = cache.get_all()  # (batch, heads, seq_so_far, head_dim)

# Q @ K^T: (batch, heads, 1, head_dim) × (batch, heads, head_dim, seq)
scores = einsum("bhqd,bhkd->bhqk", q, cached_k) / sqrt(head_dim)

# Softmax (no mask needed — cache only has past tokens)
attn = softmax(scores, axis=-1)

# Attn @ V: (batch, heads, 1, seq) × (batch, heads, seq, head_dim)
output = einsum("bhqk,bhkd->bhqd", attn, cached_v)
```

**Key insight:** During generation, the cache naturally enforces causality — it only contains past tokens, so no explicit mask is needed.

## 4. Memory Growth Analysis

### Linear Growth Formula

```
KV_cache(bytes) = 2 × batch × layers × heads × seq_len × head_dim × elem_bytes
```

### 7B Model (batch=1, fp16)

| Context | KV Cache | Total (params + KV) | KV Fraction |
|---------|----------|---------------------|-------------|
| 256     | 0.12 GB  | 7.04 GB             | 1.8%        |
| 4,096   | 2.00 GB  | 8.91 GB             | 22.4%       |
| 8,192   | 4.00 GB  | 10.91 GB            | 36.7%       |
| 32,768  | 16.00 GB | 22.91 GB            | 69.8%       |

### Maximum Context by GPU (7B model, batch=1)

| GPU | Max Context |
|-----|-------------|
| RTX 4090 (24 GB) | 6,690 tokens |
| A100-40GB | 39,458 tokens |
| A100-80GB / H100-80GB | 121,378 tokens |

### Batch Size Impact

KV cache scales linearly with batch size. At batch=4, the 7B model on an A100-80GB can only handle ~30K context instead of 121K.

## 5. Optimizations

### Optimization 1: Paged Attention (vLLM-style)

**Problem:** Contiguous allocation wastes memory when sequences have variable lengths. A batch with one 32K sequence and three 100-token sequences still allocates 32K for all.

**Solution:** Divide memory into fixed-size blocks (pages). Each sequence maintains a page table mapping logical blocks to physical pages.

```
Physical page pool: (total_pages, heads, block_size, head_dim)
Page table: (batch, max_blocks) → logical → physical mapping
```

**Benefits:**
- Zero memory fragmentation
- Supports speculative decoding and branching
- Enables prefix caching (share common prefixes)
- No need to pre-allocate max_seq_len

**Trade-off:** Page table indirection adds complexity to the attention kernel (gather from non-contiguous pages).

### Optimization 2: Quantization

**Problem:** fp16 KV cache dominates memory for long contexts.

**Solution:** Store K/V in int8 with per-channel affine dequantization: `x ≈ scale × q + zero`

```
int8 data: 1 byte per element (vs 2 for fp16)
fp16 scales + zeros: shared per channel (not per token)
Net savings: ~50% memory with <1% accuracy loss
```

**Production approach:** Shared per-channel scales (not per-position) stored in fp16. The per-position approach in this codebase is for correctness demonstration but has higher overhead.

### Optimization 3: Chunked Prefill

**Problem:** Processing a 32K prompt requires materializing a 32K × 32K attention matrix (4 GB in fp32).

**Solution:** Process the prompt in chunks of size C. Each chunk attends to all previous tokens + causal within chunk.

```
Peak memory: O(C × seq_len) instead of O(seq_len²)
For C=512, seq=4096: 8 MB vs 64 MB (8× savings)
```

### Combined: Paged + Quantized

Together these give 2-4× memory reduction, enabling 2-4× longer contexts in the same GPU memory.

## 6. GPU Execution Mapping

### Memory Hierarchy

| Level | Size | Latency | Usage |
|-------|------|---------|-------|
| Registers | 64 KB/SM | 1 cycle | Thread-local, warp computation |
| Shared memory | 166 KB/SM (H100) | 1-3 cycles | Tiling, softmax intermediates |
| L2 cache | 50 MB (H100) | ~20 cycles | Automatic global memory caching |
| HBM | 80 GB (H100) | ~300-400 cycles | Model weights, KV cache, activations |

### Cached Attention Kernel Design

```
Grid: (batch_size, num_heads, 1)
Block: (32, 32) = 1024 threads

Shared memory per block (~16-20 KB):
  - Q tile: 1 × head_dim (512 bytes fp16)
  - K tile: 32 × head_dim (8 KB fp16)
  - Score tile: 32 × 32 (4 KB fp16)
```

**Optimization strategies:**
1. Coalesced global memory access (warp-level consecutive addresses)
2. Tiled GEMM with shared memory
3. Persistent kernels (keep blocks alive until all tiles processed)
4. Async copy (H100 `cp.async`) to overlap memory transfer with computation
5. Tensor Cores (`mma.sync`) for matmul operations
6. Fusion: merge softmax with attention score computation

### Arithmetic Intensity

For single-token generation (batch=1, heads=32, seq=4096):
- **FLOPs:** 0.02 GFLOPs
- **Memory traffic:** 16.79 MB
- **Arithmetic intensity:** 1.0 FLOPs/byte
- **→ Memory-bound** (H100 peak: 1,970 TFLOPS, 3.35 TB/s)

The cached attention is fundamentally memory-bound — the bottleneck is reading the KV cache from HBM, not computation. This is why bandwidth-optimized kernels (FlashAttention-style tiling) matter more than compute optimization.

### Multi-GPU Strategies

| Strategy | KV Cache Impact | Best For |
|----------|----------------|----------|
| Tensor parallelism | Split K/V by head_dim across GPUs | Large models |
| Pipeline parallelism | Each GPU holds its layer shard's K/V | Very large models |
| Sequence parallelism | Split K/V by sequence dimension | Long context prefill |
| Expert parallelism | KV cache shared; only MLP experts sharded | MoE models |

## Running

```bash
# Run all 10 demos
python demo.py

# Run memory analysis standalone
python memory_analysis.py

# Run GPU mapping report
python gpu_mapping.py
```

## Key Takeaways

1. **KV cache grows linearly** with sequence length and batch size — it's the dominant memory cost for long-context inference.

2. **Generation is memory-bound** — reading the KV cache from HBM dominates latency, not the attention computation itself.

3. **Paged attention eliminates fragmentation** — critical for serving variable-length sequences efficiently.

4. **Quantization gives ~50% savings** with minimal accuracy loss when using shared per-channel scales.

5. **FlashAttention-style tiling** reduces HBM traffic by processing K/V in tiles that fit in shared memory, cutting memory bandwidth requirements by 2-4×.
