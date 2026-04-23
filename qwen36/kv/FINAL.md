All 10 demos pass cleanly. Here's a summary of what was built:

## Project Structure

```
kv/
├── kv_cache.py          # Core KV-cache data structures
├── attention.py         # Attention computation (standard, cached, GQA, masked)
├── transformer.py       # Full transformer decoder with KV-cache integration
├── optimizations.py     # Paged attention, quantization, chunked prefill
├── memory_analysis.py   # Memory growth formulas, GPU limits
├── gpu_mapping.py       # GPU kernel design, Tensor Core analysis
├── demo.py              # 10 end-to-end demos
└── README.md            # Comprehensive documentation
```

## What Each Demo Validates

| # | Demo | Validates |
|---|------|-----------|
| 1 | Basic KV Cache | Data structure layout, O(1) append, memory tracking |
| 2 | Cached Attention | Q @ K^T @ V using cache, matches manual computation |
| 3 | Full Transformer | Prefill + generation loop, variable-length batching |
| 4 | Variable-Length Batching | Per-sequence attention with different lengths |
| 5 | Paged Attention | Block allocation, page tables, non-contiguous memory |
| 6 | Quantized Cache | int8 quantization/dequantization with affine transform |
| 7 | Chunked Prefill | Matches full attention (4.56e-10 diff), 8× memory savings |
| 8 | Optimization Comparison | Side-by-side memory costs of all strategies |
| 9 | Memory Analysis | Model sizes, growth curves, max context per GPU |
| 10 | GPU Tensor Cores | Arithmetic intensity → all configs are memory-bound |

## Key Findings

- **KV cache is linear**: `2 × batch × layers × heads × seq × head_dim × elem_bytes`
- **7B model on H100-80GB**: max ~121K tokens (batch=1), ~30K (batch=4)
- **Generation is memory-bound**: 1.0 FLOPs/byte intensity — bottleneck is HBM bandwidth, not compute
- **Paged + quantized**: 2-4× memory reduction, enabling proportionally longer contexts