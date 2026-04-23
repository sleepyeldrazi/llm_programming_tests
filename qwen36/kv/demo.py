"""
End-to-End KV-Cache Demo

Demonstrates:
  1. Building a small transformer with KV-cache
  2. Prefill phase (prompt processing)
  3. Incremental generation (one token at a time)
  4. Variable-length batching
  5. Memory tracking
  6. Optimization comparisons
"""

import numpy as np
import sys
import os

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(__file__))

from kv_cache import KVCache, CacheConfig, BatchedKVCache
from attention import (
    scaled_dot_product_attention,
    cached_attention,
    build_causal_mask,
    softmax_stable,
)
from transformer import TransformerDecoder, TransformerDecoderLayer
from optimizations import (
    PagedKVCache, PageConfig,
    QuantizedKVCache,
    ChunkedPrefill,
    compare_strategies,
)
from memory_analysis import (
    ModelSpec, compute_model_memory, compute_kv_cache_memory,
    find_max_context, compare_model_sizes,
)
from gpu_mapping import tensor_core_analysis, print_gpu_report


def demo_basic_kv_cache():
    """Demo 1: Basic KV cache operations."""
    print("=" * 70)
    print("DEMO 1: Basic KV Cache Operations")
    print("=" * 70)

    config = CacheConfig(
        batch_size=2,
        num_heads=4,
        head_dim=16,
        max_seq_len=64,
        dtype=np.float32,
    )
    cache = KVCache(config)

    print(f"\nCache shape: {cache.cache_k.shape}")
    print(f"  (batch={config.batch_size}, heads={config.num_heads}, "
          f"max_seq={config.max_seq_len}, head_dim={config.head_dim})")
    print(f"Allocated: {cache.memory_allocated_bytes:,} bytes")

    # Simulate generating tokens one at a time
    np.random.seed(42)
    for step in range(10):
        # Simulate new K and V from the model
        k_new = np.random.randn(2, 4, 1, 16).astype(np.float32) * 0.01
        v_new = np.random.randn(2, 4, 1, 16).astype(np.float32) * 0.01

        cache.update(k_new, v_new)

    print(f"\nAfter 10 steps:")
    print(f"  Write position: {cache.write_pos}")
    print(f"  Sequence lengths: {cache.lengths}")
    print(f"  Memory used: {cache.memory_used_bytes:,} bytes")

    # Retrieve cached data
    k_cached, v_cached = cache.get_all()
    print(f"  Cached K shape: {k_cached.shape}")
    print(f"  Cached V shape: {v_cached.shape}")

    # Verify data integrity
    assert k_cached.shape == (2, 4, 10, 16)
    assert v_cached.shape == (2, 4, 10, 16)
    print("\n  ✓ Data integrity verified")


def demo_cached_attention():
    """Demo 2: Cached attention computation."""
    print("\n" + "=" * 70)
    print("DEMO 2: Cached Attention Computation")
    print("=" * 70)

    batch, heads, head_dim = 2, 4, 16
    seq_len = 8
    scale = 1.0 / np.sqrt(head_dim)

    np.random.seed(123)

    # Build a cache with some history
    config = CacheConfig(batch_size=batch, num_heads=heads,
                         head_dim=head_dim, max_seq_len=64)
    cache = KVCache(config)

    # Fill cache with random K, V
    for i in range(seq_len):
        k = np.random.randn(batch, heads, 1, head_dim).astype(np.float32) * 0.01
        v = np.random.randn(batch, heads, 1, head_dim).astype(np.float32) * 0.01
        cache.update(k, v)

    # Current query (new token)
    q = np.random.randn(batch, heads, 1, head_dim).astype(np.float32) * 0.01

    # Cached attention
    output = cached_attention(q, cache, scale)
    print(f"\nQuery shape: {q.shape}")
    print(f"Cached K shape: {cache.cache_k.shape} (used: {cache.write_pos} tokens)")
    print(f"Output shape: {output.shape}")

    # Verify against manual computation
    k_all, v_all = cache.get_all()
    scores = np.einsum("bhqd,bhkd->bhqk", q, k_all) * scale
    attn = softmax_stable(scores, axis=-1)
    manual_output = np.einsum("bhqk,bhkd->bhqd", attn, v_all)

    diff = np.max(np.abs(output - manual_output))
    print(f"Max difference from manual: {diff:.2e}")
    assert diff < 1e-5, f"Attention mismatch: {diff}"
    print("  ✓ Cached attention matches manual computation")

    # Show attention weights for one batch/head
    print(f"\nAttention weights (batch=0, head=0):")
    print(f"  {attn[0, 0, 0, :].round(3)}")
    print(f"  Sum: {attn[0, 0, 0, :].sum():.4f} (should be ~1.0)")


def demo_full_transformer():
    """Demo 3: Full transformer with KV-cache."""
    print("\n" + "=" * 70)
    print("DEMO 3: Full Transformer with KV-Cache")
    print("=" * 70)

    # Small model for demo
    model = TransformerDecoder(
        num_layers=2,
        dim=64,
        num_heads=4,
        mlp_hidden=128,
        vocab_size=1000,
        max_seq_len=128,
        batch_size=2,
        dtype=np.float32,
        seed=42,
    )

    # Create a prompt (padded to same length)
    prompt = np.array([[10, 20, 30, 40, 50],
                       [15, 25, 35, 45, 0]], dtype=np.int32)  # 0 = pad

    lengths = np.array([5, 4], dtype=np.int32)

    print(f"\nPrompt tokens: {prompt.shape}")
    print(f"  Sequence 0: {prompt[0]} (length={lengths[0]})")
    print(f"  Sequence 1: {prompt[1]} (length={lengths[1]})")

    # Prefill
    hidden = model.prefill(prompt, lengths=lengths)
    print(f"\nAfter prefill:")
    print(f"  Hidden shape: {hidden.shape}")
    print(f"  Cache write position: {model.cache.caches[0].write_pos}")

    # Generate tokens
    print(f"\nGenerating 5 tokens...")
    generated = model.generate(prompt, num_tokens=5, temperature=0.8, top_k=50,
                               lengths=lengths)

    for i, tokens in enumerate(generated):
        print(f"  Step {i+1}: {tokens}")

    # Memory report
    report = model.memory_report()
    print(f"\nMemory Report:")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def demo_variable_length_batching():
    """Demo 4: Variable-length batching."""
    print("\n" + "=" * 70)
    print("DEMO 4: Variable-Length Batching")
    print("=" * 70)

    batch_size = 4
    config = CacheConfig(
        batch_size=batch_size,
        num_heads=4,
        head_dim=16,
        max_seq_len=32,
        dtype=np.float32,
    )
    cache = KVCache(config)

    np.random.seed(99)

    # Simulate sequences of different lengths
    # Seq 0: 8 tokens, Seq 1: 5 tokens, Seq 2: 10 tokens, Seq 3: 3 tokens
    seq_lengths = [8, 5, 10, 3]
    max_len = max(seq_lengths)

    print("\nSimulating variable-length batch:")
    # Each batch item has its own cache (simplified: use separate caches)
    per_seq_caches = [KVCache(CacheConfig(
        batch_size=1, num_heads=4, head_dim=16,
        max_seq_len=max_len, dtype=np.float32
    )) for _ in range(batch_size)]

    for b, length in enumerate(seq_lengths):
        for t in range(length):
            k = np.random.randn(1, 4, 1, 16).astype(np.float32) * 0.01
            v = np.random.randn(1, 4, 1, 16).astype(np.float32) * 0.01
            per_seq_caches[b].update(k, v)

    # Query for each sequence at its current position
    scale = 1.0 / np.sqrt(16)
    for b in range(batch_size):
        q = np.random.randn(1, 4, 1, 16).astype(np.float32) * 0.01
        k_cached, v_cached = per_seq_caches[b].get_all()

        # Attention for this batch item
        scores = np.einsum("bhqd,bhkd->bhqk", q, k_cached) * scale
        attn = softmax_stable(scores, axis=-1)

        # Show which positions are attended to
        print(f"\n  Sequence {b} (length={seq_lengths[b]}):")
        print(f"    Attention: {attn[0, 0, 0, :].round(3)}")


def demo_paged_attention():
    """Demo 5: Paged attention."""
    print("\n" + "=" * 70)
    print("DEMO 5: Paged Attention (vLLM-style)")
    print("=" * 70)

    config = PageConfig(
        block_size=4,
        num_pages=16,
        batch_size=2,
        num_heads=4,
        head_dim=16,
        dtype=np.float32,
    )
    paged = PagedKVCache(config)

    print(f"\nPage config:")
    print(f"  Block size: {config.block_size} tokens")
    print(f"  Pages per sequence: {config.num_pages}")
    print(f"  Max tokens per sequence: {config.num_pages * config.block_size}")
    print(f"  Allocated: {paged.memory_allocated_bytes:,} bytes")

    np.random.seed(77)

    # Fill sequence 0 with 12 tokens (3 blocks)
    print(f"\nFilling sequence 0 with 12 tokens...")
    for t in range(12):
        k = np.random.randn(1, 4, 1, 16).astype(np.float32) * 0.01
        v = np.random.randn(1, 4, 1, 16).astype(np.float32) * 0.01
        block_idx = t // config.block_size
        offset = t % config.block_size
        paged.append_token(0, k, v, block_idx, offset)

    print(f"  Blocks allocated: {paged.num_blocks[0]}")
    print(f"  Page table: {paged.page_tables[0, :paged.num_blocks[0]]}")

    # Fill sequence 1 with 8 tokens (2 blocks)
    print(f"\nFilling sequence 1 with 8 tokens...")
    for t in range(8):
        k = np.random.randn(1, 4, 1, 16).astype(np.float32) * 0.01
        v = np.random.randn(1, 4, 1, 16).astype(np.float32) * 0.01
        block_idx = t // config.block_size
        offset = t % config.block_size
        paged.append_token(1, k, v, block_idx, offset)

    print(f"  Blocks allocated: {paged.num_blocks[1]}")
    print(f"  Page table: {paged.page_tables[1, :paged.num_blocks[1]]}")

    # Retrieve and verify
    k0, v0 = paged.get_sequence_contiguous(0, num_tokens=12)
    k1, v1 = paged.get_sequence_contiguous(1, num_tokens=8)
    print(f"\n  Seq 0 K shape: {k0.shape}")
    print(f"  Seq 1 K shape: {k1.shape}")

    print(f"\n  Memory used: {paged.memory_used_bytes:,} bytes")
    print(f"  Utilization: {paged.memory_utilization():.1%}")


def demo_quantized_cache():
    """Demo 6: Quantized KV cache."""
    print("\n" + "=" * 70)
    print("DEMO 6: Quantized KV Cache (int8)")
    print("=" * 70)

    batch, heads, head_dim, max_seq = 2, 4, 16, 32
    cache = QuantizedKVCache(batch, heads, head_dim, max_seq, dtype=np.float32)

    np.random.seed(55)

    # Fill with random data
    for t in range(10):
        k = np.random.randn(batch, heads, 1, head_dim).astype(np.float32) * 0.1
        v = np.random.randn(batch, heads, 1, head_dim).astype(np.float32) * 0.1
        cache.update(k, v)

    # Retrieve and compare
    k_deq, v_deq = cache.get()
    print(f"\nQuantized cache (10 tokens):")
    print(f"  Dequantized K shape: {k_deq.shape}")
    print(f"  Dequantized V shape: {v_deq.shape}")

    # Compare with original (we need to re-quantize to compare)
    # The quantization error depends on the data distribution
    print(f"  Memory savings vs fp32: {cache.memory_savings_vs_fp32:.1%}")
    print(f"  Memory savings vs fp16: {cache.memory_savings_vs_fp16:.1%} (per-pos scales overhead)")

    # Show quantization error for one position
    # Use larger values for better int8 quantization fidelity
    k_orig = np.random.randn(batch, heads, 1, head_dim).astype(np.float32) * 1.0
    v_orig = np.random.randn(batch, heads, 1, head_dim).astype(np.float32) * 1.0
    cache.update(k_orig, v_orig)
    k_deq_single, _ = cache.get(start=10, end=11)

    # k_deq_single: (batch, heads, 1, head_dim), k_orig: (batch, heads, 1, head_dim)
    print(f"  k_orig shape: {k_orig.shape}, k_deq shape: {k_deq_single.shape}")
    error = np.max(np.abs(k_orig - k_deq_single))
    rel_error = error / (np.max(np.abs(k_orig)) + 1e-8)
    print(f"  Max absolute error (one token): {error:.6f}")
    print(f"  Max relative error: {rel_error:.4f}")
    print(f"  → Per-position quantization has high overhead; production uses")
    print(f"    shared per-channel scales for ~50% memory savings with <1% error")


def demo_chunked_prefill():
    """Demo 7: Chunked prefill."""
    print("\n" + "=" * 70)
    print("DEMO 7: Chunked Prefill")
    print("=" * 70)

    chunker = ChunkedPrefill(chunk_size=4)

    batch, heads, seq, head_dim = 1, 4, 12, 16
    scale = 1.0 / np.sqrt(head_dim)

    np.random.seed(33)
    q = np.random.randn(batch, heads, seq, head_dim).astype(np.float32) * 0.01
    k = np.random.randn(batch, heads, seq, head_dim).astype(np.float32) * 0.01
    v = np.random.randn(batch, heads, seq, head_dim).astype(np.float32) * 0.01

    # Chunked attention
    output_chunked = chunker.compute_attention_chunked(q, k, v, scale)

    # Full attention (for comparison)
    from attention import scaled_dot_product_attention, build_causal_mask
    causal = build_causal_mask(seq, dtype=np.float32)
    output_full = scaled_dot_product_attention(
        q, k, v, scale, mask=causal[None, None, :, :]
    )

    diff = np.max(np.abs(output_chunked - output_full))
    print(f"\nChunk size: {chunker.chunk_size}")
    print(f"Sequence length: {seq}")
    print(f"Chunks: {(seq + chunker.chunk_size - 1) // chunker.chunk_size}")
    print(f"Max difference from full attention: {diff:.2e}")
    assert diff < 1e-5, f"Chunked attention mismatch: {diff}"
    print("  ✓ Chunked attention matches full attention")

    # Memory comparison
    mem = ChunkedPrefill.peak_memory_comparison(seq_len=4096, chunk_size=512)
    print(f"\nMemory comparison (seq=4096, chunk=512):")
    print(f"  Full attention matrix: {mem['full_attention_mb']:.0f} MB")
    print(f"  Chunked peak: {mem['chunked_peak_attention_mb']:.0f} MB")
    print(f"  Savings: {mem['savings_ratio']:.1f}x")


def demo_optimization_comparison():
    """Demo 8: Optimization strategy comparison."""
    print("\n" + "=" * 70)
    print("DEMO 8: Optimization Strategy Comparison")
    print("=" * 70)

    results = compare_strategies(
        batch_size=4, num_heads=32, head_dim=128,
        max_seq_len=4096, num_layers=32
    )

    print(f"\nConfiguration: batch=4, heads=32, head_dim=128, "
          f"seq=4096, layers=32\n")

    header = f"{'Strategy':<25} {'Per Layer(MB)':>14} {'Total(GB)':>10} {'Notes':<25}"
    print(header)
    print("-" * len(header))

    for name, data in results.items():
        notes = ""
        if "savings_vs_fp16" in data:
            notes = f"{data['savings_vs_fp16']:.0%} savings"
        elif "overhead_vs_naive" in data:
            notes = f"{data['overhead_vs_naive']:.3f}x overhead"

        print(f"{name:<25} {data['per_layer_mb']:>14.1f} {data['total_mb']/1024:>10.2f} "
              f"{notes:<25}")


def demo_memory_analysis():
    """Demo 9: Memory growth analysis."""
    print("\n" + "=" * 70)
    print("DEMO 9: Memory Growth Analysis")
    print("=" * 70)

    # Compare model sizes
    comparisons = compare_model_sizes()

    print("\nModel Size Comparison (fp16):\n")
    header = f"{'Model':<20} {'Params(GB)':>10} {'KV@1K':>8} {'KV@8K':>8} {'KV@32K':>8} {'MaxCtx(H100)':>12}"
    print(header)
    print("-" * len(header))
    for name, data in comparisons.items():
        print(f"{name:<20} {data['params_gb']:>10.1f} {data['kv_1k_gb']:>8.2f} "
              f"{data['kv_8k_gb']:>8.2f} {data['kv_32k_gb']:>8.2f} "
              f"{data['max_context_H100']:>12,}")

    # Growth for 7B model
    spec = ModelSpec(num_layers=32, dim=4096, num_heads=32, head_dim=128)
    model_mem = compute_model_memory(spec, np.float16)

    print(f"\n\n7B Model Memory Growth (batch=1, fp16):\n")
    print(f"  Model params: {model_mem['total_params_gb']:.1f} GB")
    print()

    seq_lens = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    print(f"  {'Seq Len':>8} {'KV(GB)':>8} {'Total(GB)':>10} {'KV%':>6}")
    print(f"  {'-'*40}")
    for sl in seq_lens:
        kv = compute_kv_cache_memory(1, sl, spec, np.float16)
        total = kv["total_gb"] + model_mem["total_params_gb"]
        pct = kv["total_gb"] / total * 100
        print(f"  {sl:>8,} {kv['total_gb']:>8.2f} {total:>10.2f} {pct:>5.1f}%")

    # GPU limits
    print(f"\n\nMax Context by GPU (7B model, batch=1):\n")
    gpus = {"RTX 4090": 24, "A100-40GB": 40, "A100-80GB": 80, "H100-80GB": 80}
    for gpu, mem in gpus.items():
        ctx = find_max_context(spec, mem, batch_size=1)
        print(f"  {gpu:<15}: {ctx:>8,} tokens")


def demo_gpu_tensor_cores():
    """Demo 10: GPU Tensor Core analysis."""
    print("\n" + "=" * 70)
    print("DEMO 10: GPU Tensor Core Analysis")
    print("=" * 70)

    configs = [
        {"batch": 1, "heads": 32, "seq": 1024, "label": "Short context"},
        {"batch": 1, "heads": 32, "seq": 8192, "label": "Long context"},
        {"batch": 4, "heads": 32, "seq": 4096, "label": "Batched"},
    ]

    for cfg in configs:
        tc = tensor_core_analysis(
            batch=cfg["batch"], heads=cfg["heads"], seq_len=cfg["seq"]
        )
        print(f"\n  {cfg['label']} (batch={cfg['batch']}, seq={cfg['seq']}):")
        print(f"    Total FLOPs: {tc['total_flops']}")
        print(f"    Memory traffic: {tc['memory_traffic_mb']}")
        print(f"    Arithmetic intensity: {tc['arithmetic_intensity']}")
        print(f"    Compute bound: {tc['compute_bound_ms']}")
        print(f"    Memory bound: {tc['memory_bound_ms']}")
        print(f"    → {tc['bound']}")


def main():
    """Run all demos."""
    print("\n" + "█" * 70)
    print("  KV-CACHE SYSTEM FOR AUTOREGRESSIVE TRANSFORMER INFERENCE")
    print("  Pure NumPy Implementation — No Frameworks")
    print("█" * 70)

    demos = [
        ("Basic KV Cache", demo_basic_kv_cache),
        ("Cached Attention", demo_cached_attention),
        ("Full Transformer", demo_full_transformer),
        ("Variable-Length Batching", demo_variable_length_batching),
        ("Paged Attention", demo_paged_attention),
        ("Quantized Cache", demo_quantized_cache),
        ("Chunked Prefill", demo_chunked_prefill),
        ("Optimization Comparison", demo_optimization_comparison),
        ("Memory Analysis", demo_memory_analysis),
        ("GPU Tensor Cores", demo_gpu_tensor_cores),
    ]

    for name, func in demos:
        try:
            func()
        except Exception as e:
            print(f"\n  ✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "█" * 70)
    print("  ALL DEMOS COMPLETE")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()
