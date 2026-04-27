# KV-cache: design notes, memory growth, optimizations, GPU mapping

## What's in the repo

- `kv_cache.py` — `KVCache` data structure plus a `MultiHeadAttention` layer
  that reads/writes it. Pure Python, no frameworks.
- `demo.py` — exercises prefill, lockstep decoding, and variable-length /
  early-stop batching, and verifies bit-for-bit (modulo float epsilon) that
  the cached path matches a no-cache recompute.

## Memory layout (recap)

Per layer we keep two flat float buffers of length `B * H * S_max * D` for K
and V, with index `((b * H + h) * S_max + t) * D + d`. That's the row-major
encoding of logical shape `[B, H, S_max, D]`. Head-dim is the fastest-varying
axis, so reading row `(b, h, t)` is `D` contiguous floats — one or two
cache-line loads, friendly to GPU coalescing.

Variable-length batching is handled with a per-sequence `lengths[b]` counter
plus an `active` mask on `decode_step`. Inactive sequences neither write into
their slot nor advance, so finished sequences don't pollute attention scores
or waste compute. Slots are preallocated to `S_max`, so appending is O(D)
with no realloc.

## Memory growth

The total footprint is

    2 * L * B * H * S * D * dtype_bytes        (factor 2 = K and V)

It is **linear** in every factor, including sequence length `S`. Concrete
numbers from `demo.py` for a Llama-class config (L=32, H=32, D=128, fp16):

| B   | S      | KV cache |
|-----|--------|----------|
| 1   | 4096   | 2 GiB    |
| 8   | 4096   | 16 GiB   |
| 32  | 8192   | 128 GiB  |
| 128 | 32768  | 2048 GiB |

Two consequences worth flagging:

1. At long context the cache, not the weights, dominates HBM. A 7B model is
   ~14 GiB in fp16 — a single B=32 / S=8192 cache is already 9× that.
2. Bandwidth is the bottleneck during decode, not flops. Each step reads the
   entire cache (`O(S)` per token per head per layer) and produces one new
   token. The arithmetic intensity is roughly `D / (D + 1) ≈ 1` flop per byte
   read, so a decode step on an H100 (~3 TB/s HBM) is bound by how fast the
   cache streams in, not by the tensor cores.

Practical implication: any optimization that shrinks the cache (or reads less
of it per step) buys decode latency directly.

## Optimizations

### 1. Paged KV cache (vLLM-style)

The flat `[B, H, S_max, D]` buffer assumes a worst-case `S_max` per slot. If
half the sequences are short, half that memory is wasted, and a new request
can't fit even when total used memory is small — classic external
fragmentation.

Fix: split the cache into fixed-size **pages** (e.g. 16 tokens × H × D each)
and replace the per-sequence contiguous slot with a **block table** —
`page_table[b]` is a list of page IDs in logical token order. Allocation
becomes a free-list pop; deallocation is a free-list push; memory utilization
goes from "max across batch" to "sum across batch". Attention kernels gain
one indirection (`page = page_table[b][t // page_size]; offset = t % page_size`)
which is essentially free on a GPU because the page table is tiny and
register-resident. Same trick lets us share prompt prefixes across requests
by sharing pages — copy-on-write only when a sequence diverges.

### 2. Multi-Query / Grouped-Query Attention (MQA / GQA)

Standard MHA stores `H` separate K/V heads. MQA keeps `H` query heads but
**one** shared K/V head; GQA keeps `G < H` K/V groups. The cache shrinks by
`H` (MQA) or `H/G` (GQA), and decode bandwidth shrinks by the same factor —
a free latency win on top of the memory win, with very small quality loss
(used by Llama-2-70B, Mistral, etc.). In our layout this is one parameter
change: `cache.H = G` while attention still iterates `H` query heads,
broadcasting reads from the shared group.

### 3. Quantization (INT8 / INT4 / FP8)

The cache is read-mostly during decode and tolerant of low precision because
softmax is invariant to additive shifts and forgiving of noise. Storing K, V
in INT8 with per-token scales halves bandwidth; INT4 or FP8 quarters it.
Combine with on-the-fly dequant in the matmul kernel — the dequant cost is
hidden behind the HBM read.

### 4. Sliding-window / chunked attention

For some workloads (long-context reading where recent tokens dominate), we
can cap effective context: keep only the last `W` tokens in the cache, or
mix a small dense window with a sparse global "sink". Memory and decode cost
become O(W) instead of O(S). Mistral 7B uses W=4096; Longformer-style models
add a few persistent global tokens.

The four optimizations compose: GQA + paged cache + INT8 is the typical
production recipe.

## GPU mapping

The reference Python loop is the wrong shape for a GPU; here's how the same
algorithm executes on hardware.

**Prefill** (many query tokens, all K/V already known): one large fused
attention kernel — FlashAttention. Each thread block owns a tile of the
output `(b, h, q_block)`. It loads `Q` once, then streams `K` and `V` tiles
through SRAM, accumulating softmax in a numerically stable online form
(`m_i`, `l_i` running max/sumexp). The `[B, H, S, D]` layout means each
`(b, h)` is its own contiguous matrix, so the kernel just maps each block to
a `(b, h)` and tiles along `S` — coalesced loads fall out of the layout for
free.

**Decode** (one query token): the single-token-Q matmul is bandwidth-bound,
so the relevant kernel is **FlashDecoding** — split the K/V sequence axis
across thread blocks (`split-K`), have each block compute partial
softmax-weighted sums, then a tiny reduction kernel combines them. Without
the split, one SM is reading the whole cache for that `(b, h)` and the rest
of the GPU is idle; with it, all SMs are active and HBM is saturated.

**Cache append.** A `decode_step` writes one `(D,)` row per `(b, h)` per
layer. On a GPU this is `B * H * L` independent stores of D contiguous fp16,
issued from the QKV projection kernel directly (write `K`, `V` projections
into the cache slot at `lengths[b]` instead of into a temporary). No extra
launch.

**Block tables for paging.** With paged cache, the attention kernel takes an
extra input — the page table for each `(b)`. Inside the kernel, the inner
loop over `t` becomes "compute page index + offset, gather K/V tile". The
page table is small (`ceil(S/page_size)` ints per sequence), broadcast into
shared memory once per block, and the indirection adds no measurable
overhead because the actual K/V loads are still big contiguous tiles.

**Variable lengths in a batch.** The `active` mask becomes a per-sequence
length tensor. FlashAttention already handles ragged batches via cumulative
sequence offsets (`cu_seqlens`), so the kernel iterates `[0, lengths[b])`
per `(b, h)` rather than `[0, S_max)` — no padding work, no masked-out
flops. This is critical: with paging + ragged decode, an SM never spends a
cycle on a token that doesn't exist.
