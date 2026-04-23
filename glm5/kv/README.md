# KV-Cache for Autoregressive Transformer Inference

A complete, framework-free implementation of KV-caching for autoregressive
transformer inference, built from scratch in Python/NumPy.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Transformer Layer                         │
│                                                             │
│  Token IDs ──► Embedding ──► Q,K,V Projections             │
│                                  │                          │
│                    ┌─────────────┼──────────────┐           │
│                    ▼             ▼              ▼           │
│                   Q_new  ──► K_new, V_new ──► Cache Write  │
│                    │                          │             │
│                    │    ┌─────────────────────┘             │
│                    ▼    ▼                                   │
│              ┌──────────────┐                               │
│              │  Attention   │  Q_new × (K_cached + K_new)   │
│              │  Computation │  ──────────────────────────►  │
│              │  (read-only) │  weights × (V_cached + V_new) │
│              └──────┬───────┘                               │
│                     ▼                                       │
│            Output Projection ──► LayerNorm ──► next layer   │
└─────────────────────────────────────────────────────────────┘
```

## Data Structure Layout

### Memory Format

Each layer maintains two pre-allocated tensors:

```
keys:   (B, H, S_max, D)   float32
values: (B, H, S_max, D)   float32
```

| Symbol | Meaning                        | Example (GPT-4 class) |
|--------|--------------------------------|----------------------|
| B      | Batch size                     | 1–64                 |
| H      | Number of attention heads      | 32                   |
| S_max  | Maximum sequence length        | 8192–131072          |
| D      | Head dimension (d_model / H)   | 128                  |

**Why BHSD layout?**

The dimensions are ordered so that the sequence axis (S) is stride-D
contiguous. This means:

1. **Append is a simple slice copy** — `cache[b, :, pos, :] = new_kv`
   writes D×H floats to a contiguous region.
2. **Attention matmul is efficient** — the inner `Q @ K^T` reads K along
   the S dimension, which is stride-D contiguous.
3. **GPU-friendly** — maps directly to a CUDA tensor with no transposition
   needed between the write and read paths.

### Auxiliary State

```
seq_lens: int[B]  — valid prefix length per batch element
```

Positions `[..., :seq_lens[b], :]` contain valid data. Everything beyond
is garbage and must be masked out during attention.

## Update Logic Per Step

### Prefill Phase (processing the full prompt)

```
Input:  prompt tokens of length S
Output: cache filled with S key-value pairs

for each layer:
    Q, K, V = project(prompt_embeddings)          # (B, S, d_model) → 3× (B, S, d_model)
    K = reshape(K, (B, H, S, D))                  # split into heads
    V = reshape(V, (B, H, S, D))
    cache.write(positions=[0, 1, ..., S-1], K, V)  # bulk write

    # Self-attention within the prompt (causal mask)
    attn_output = attention(Q, cache.read())        # O(S²) — one-time cost
```

### Decode Phase (one token at a time)

```
Input:  single new token
Output: logits for next token prediction

for each layer:
    q_new, k_new, v_new = project(token_embedding)  # each (B, 1, d_model)
    k_new = reshape(k_new, (B, H, 1, D))
    v_new = reshape(v_new, (B, H, 1, D))

    # ── CACHE UPDATE: O(H·D) — write 1 token ──
    cache[pos] = (k_new, v_new)                      # 2 × H × D floats

    # ── ATTENTION: O(S·H·D) — query vs ALL cached keys ──
    K_all, V_all = cache.read()                       # (B, H, S+1, D)
    scores = q_new @ K_all.T / √D                     # (B, H, 1, S+1)
    weights = softmax(scores)
    output = weights @ V_all                           # (B, H, 1, D)
```

**Key insight**: Without caching, each decode step would require O(S²) work
(recomputing attention for all S previous tokens). With caching, it's only
O(S) — the new query attends against the cached keys/values.

## Attention Computation Using Cached Keys/Values

```
┌───────────┐     ┌───────────────────────────────────┐
│  Q_new    │     │  Cached K (all past tokens)        │
│  (1, D)   │  ×  │  (S_valid, D)                     │
│           │     │                                   │
│           │     │  [k₀] [k₁] [k₂] ... [k_{S-1}]    │
└─────┬─────┘     └───────────────────────────────────┘
      │                         │
      ▼                         ▼
  ┌────────────────────────────────────┐
  │  scores = Q · K^T / √D            │  → (1, S_valid)
  │  weights = softmax(scores)         │  → (1, S_valid)
  │  output = weights · V              │  → (1, D)
  └────────────────────────────────────┘
```

This is performed independently for each head H and batch element B.

## Memory Growth Analysis

### Linear Growth

The cache grows **linearly** with sequence length:

```
Memory per layer = 2 × B × H × S × D × sizeof(dtype)
                 = 2 × B × d_model × S × sizeof(dtype)
```

For a GPT-4-class model (32 layers, d_model=4096, FP32):

| Seq Length | Per Layer (MB) | Total (MB) | Total (GB) |
|-----------|---------------|-----------|-----------|
| 128       | 0.67          | 21.47     | 0.021     |
| 1,024     | 5.37          | 171.79    | 0.172     |
| 4,096     | 21.47         | 687.19    | 0.687     |
| 16,384    | 85.89         | 2,748.77  | 2.749     |
| 65,536    | 343.59        | 10,995.08 | 10.995    |
| 131,072   | 687.19        | 21,990.16 | 21.990    |

**Observation**: At 128K context with batch=1, you need **~22 GB** just for
the KV cache — before accounting for model weights, activations, or
gradients.

### FLOPs Savings

| Scenario | Without Cache | With Cache | Speedup |
|----------|--------------|-----------|---------|
| 1024 prompt + 100 decode | 4.2e14 | 2.0e12 | ~200× |

The speedup grows quadratically with sequence length.

## Optimizations

### 1. Paged Attention (Virtual Memory for KV Cache)

**Problem**: Pre-allocating `(B, H, S_max, D)` wastes memory for short
sequences and causes fragmentation when sequences finish at different
times.

**Solution**: Divide the cache into fixed-size blocks (pages):

```
Physical Memory:
┌────────┬────────┬────────┬────────┬────────┬────────┐
│ Block 0│ Block 1│ Block 2│ Block 3│ Block 4│  ...   │
│(H,B,D) │(H,B,D) │(H,B,D) │(H,B,D) │(H,B,D) │        │
└────────┴────────┴────────┴────────┴────────┴────────┘

Page Tables:
  Seq 0: [0] → [3] → [1]       (3 blocks = 3 × BLOCK_SIZE tokens)
  Seq 1: [2] → [4]             (2 blocks = 2 × BLOCK_SIZE tokens)
  Seq 2: [5]                   (1 block)
  Free:  [6, 7, 8, ...]
```

**Benefits**:
- Memory allocated only as needed (no S_max pre-allocation)
- Finished sequences free blocks immediately → higher throughput
- No external fragmentation
- Enables sharing of KV blocks across sequences (e.g., prefix caching)

**Implementation**: See `PagedKVCache` in `optimizations.py`.

### 2. Chunked Prefill

**Problem**: Processing a 32K-token prompt requires a 32K×32K attention
matrix (1 billion floats = 4 GB) just for the prefill.

**Solution**: Split the prompt into chunks of C tokens:

```
Prompt: [t₀, t₁, t₂, ..., t_{S-1}]   (S = 32K)

Chunk 0: [t₀..t_{C-1}]    → cache write → attention vs cache (0..C)
Chunk 1: [t_C..t_{2C-1}]  → cache write → attention vs cache (0..2C)
Chunk 2: [t_{2C}..t_{3C-1}] → cache write → attention vs cache (0..3C)
...
```

Peak attention memory: O(C × S) instead of O(S²).

**Benefits**:
- Bounded peak memory regardless of prompt length
- Can interleave prefill chunks with decode steps from other sequences
- Better GPU utilization (uniform work items)

### 3. Cache Quantization (INT8 / INT4)

**Problem**: 22 GB for a 128K context is unsustainable.

**Solution**: Quantize cached K/V to lower precision:

| Precision | Bytes/Element | Memory Savings | Typical Quality Loss |
|-----------|-------------|---------------|---------------------|
| FP32      | 4           | 1× (baseline) | 0%                  |
| FP16      | 2           | 2×            | <0.1%               |
| INT8      | 1           | 4×            | <0.5%               |
| INT4      | 0.5         | 8×            | 1-3%                |

Quantization is per-token: `scale[b,h,t] = max(|K[b,h,t,:]|) / (2^bits - 1)`.

```
Storage:
  k_quant: uint8 (B, H, S, D) or packed uint8 (B, H, S, D/2) for INT4
  k_scale: float32 (B, H, S)    — one scalar per token per head

Dequantize during attention:
  K_float = k_quant * k_scale    — in registers before matmul
```

**Benefits**:
- 4-8× memory reduction → longer contexts or larger batches
- Minimal quality loss for most tasks
- Hardware support on modern GPUs (FP8 on Hopper, INT8 on Ampere)

## GPU Execution Mapping

### Memory Hierarchy

```
┌──────────────────────────────────────────────┐
│  HBM (High Bandwidth Memory)                 │
│  ┌──────────────────────────────────────┐    │
│  │ KV Cache: (B, H, S, D) per layer     │    │
│  │ ~10-70 GB for long contexts           │    │
│  └──────────────────────────────────────┘    │
│  ┌──────────────────────────────────────┐    │
│  │ Model Weights                        │    │
│  └──────────────────────────────────────┘    │
└──────────────────────┬───────────────────────┘
                       │  ~2-3 TB/s bandwidth
                       ▼
┌──────────────────────────────────────────────┐
│  Shared Memory (per SM)                      │
│  ┌──────────────────────────────────────┐    │
│  │ Q tile: (block_B, H, tile_S, D)      │    │
│  │ K tile: (block_B, H, tile_S, D)      │    │
│  │ V tile: (block_B, H, tile_S, D)      │    │
│  │ Score tile: (block_B, H, tile_S²)    │    │
│  └──────────────────────────────────────┘    │
│  ~48-164 KB per SM                          │
└──────────────────────┬───────────────────────┘
                       │  ~19 TB/s bandwidth
                       ▼
┌──────────────────────────────────────────────┐
│  Registers (per thread block)                │
│  accumulator for QK^T, softmax, etc.        │
│  ~255 registers/thread                      │
└──────────────────────────────────────────────┘
```

### Kernel Mapping

| Operation | CPU (this impl) | GPU Kernel |
|-----------|----------------|------------|
| Cache write | `cache[b,:,pos,:] = new_kv` | `cudaMemcpyAsync` or block-level scatter |
| Q×K^T | `q @ k.T` | Batched GEMM (cuBLAS) or FlashAttention |
| Softmax | `_softmax(scores)` | Online softmax (FlashAttention) |
| Weights×V | `weights @ v` | GEMM (part of FlashAttention fused kernel) |
| Quantize | `_quantize_token()` | Block-reduce + scale + convert |

### FlashAttention Integration

The attention computation in this codebase performs the naive:

```
S = Q × K^T          # materialize full (S_q, S_kv) matrix
A = softmax(S)        # another (S_q, S_kv) matrix
O = A × V             # output
```

On GPU, **FlashAttention** fuses these three operations:

```
for each tile of Q:
    init: O = 0, m = -∞, l = 0
    for each tile of K, V:
        S_tile = Q_tile × K_tile^T       # in SRAM
        m_new = max(m, max(S_tile))
        P_tile = exp(S_tile - m_new)      # in SRAM
        l_new = l + sum(P_tile)
        O = O * (l/l_new) + P_tile × V_tile  # accumulate
        m, l = m_new, l_new
    O = O / l
```

This keeps the O(S²) attention matrix entirely in SRAM, avoiding
HBM reads/writes. The KV cache is read tile-by-tile from HBM.

### Paged Attention on GPU

The `PagedKVCache` page table translates to a GPU indirection:

```cuda
// CUDA pseudocode for paged attention
__global__ void paged_attention(
    float* Q,           // (B, H, 1, D) — new query
    float* K_pool,      // (num_blocks, H, BLOCK_SIZE, D)
    float* V_pool,
    int*   page_table,  // (B, max_pages_per_seq)
    int*   seq_lens,    // (B,)
    float* output       // (B, H, 1, D)
) {
    int b = blockIdx.y;
    int h = blockIdx.x;
    int S = seq_lens[b];

    // Load query into registers
    float q[D];
    load_query(q, Q, b, h);

    // Iterate over pages
    float score[S_MAX_LOCAL];
    for (int page = 0; page < ceil(S / BLOCK_SIZE); page++) {
        int phys_block = page_table[b * max_pages + page];
        // Gather K/V from scattered physical blocks
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float k = K_pool[phys_block * H * BLOCK_SIZE * D
                            + h * BLOCK_SIZE * D + i * D + d];
            score[page * BLOCK_SIZE + i] = dot(q, k) / sqrt(D);
        }
    }
    // ... softmax, multiply by V, write output
}
```

## File Structure

```
kv/
├── README.md            ← you are here
├── kv_cache.py          ← core data structures + attention
├── optimizations.py     ← paged attention, chunked prefill, quantization
└── test_kv_cache.py     ← comprehensive test suite
```

## Running

```bash
python test_kv_cache.py
```

All tests run without any external dependencies beyond NumPy.

## Key Design Decisions

1. **Pre-allocation**: The base `KVCache` pre-allocates to `S_max` to
   avoid GPU memory allocation during inference (malloc is expensive).
   The `PagedKVCache` trades this for on-demand block allocation.

2. **No cross-contamination**: Each batch element maintains its own
   valid prefix via `seq_lens`. Attention never attends to garbage
   positions from other sequences.

3. **Separation of concerns**: Cache update (write) and attention
   (read) are decoupled. The caller controls when each happens,
   enabling chunked prefill and prefix sharing.

4. **Quantization at cache boundary**: K/V are computed in FP32,
   quantized on write, dequantized on read. This keeps the attention
   computation unchanged while reducing memory.
