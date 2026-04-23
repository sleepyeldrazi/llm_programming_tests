"""
End-to-end tests and demonstrations for the KV-cache system.

Run with:  python test_kv_cache.py
"""

import numpy as np
from kv_cache import (
    KVCache,
    multi_head_attention_with_cache,
    memory_growth_table,
    memory_analysis,
    IncrementalDecoder,
)
from optimizations import PagedKVCache, QuantizedKVCache


# ══════════════════════════════════════════════════════════════════════
#  TEST 1:  Basic KV-cache update & retrieval
# ══════════════════════════════════════════════════════════════════════

def test_basic_cache():
    print("=" * 70)
    print("TEST 1: Basic KV-cache update and retrieval")
    print("=" * 70)

    B, H, S_max, D = 2, 4, 16, 8
    cache = KVCache(B, S_max, H, D)
    print(f"Initial: {cache}")

    # Prefill: write 5 tokens for batch 0, 3 tokens for batch 1
    # (In practice, the full batch gets the same number, but we test
    #  the update logic by writing per-batch via positions)
    new_k = np.random.randn(B, H, 5, D).astype(np.float32)
    new_v = np.random.randn(B, H, 5, D).astype(np.float32)
    cache.update(new_k, new_v)
    print(f"After prefill (5 tokens): seq_lens={cache.seq_lens}")

    # Decode: write 1 token at a time
    for step in range(3):
        one_k = np.random.randn(B, H, 1, D).astype(np.float32)
        one_v = np.random.randn(B, H, 1, D).astype(np.float32)
        cache.update(one_k, one_v)
        print(f"  Decode step {step}: seq_lens={cache.seq_lens}")

    # Verify retrieval
    k0, v0 = cache.get_kv(0)
    print(f"\nBatch 0: retrieved K shape={k0.shape}, expected (4, 8, 8)")
    assert k0.shape == (H, 8, D), f"Wrong shape: {k0.shape}"

    k1, v1 = cache.get_kv(1)
    print(f"Batch 1: retrieved K shape={k1.shape}, expected (4, 8, 8)")
    assert k1.shape == (H, 8, D), f"Wrong shape: {k1.shape}"

    # Verify the written values match
    np.testing.assert_allclose(cache.k_cache[0, :, 7, :], one_k[0, :, 0, :])
    np.testing.assert_allclose(cache.v_cache[1, :, 7, :], one_v[1, :, 0, :])
    print("✓ All assertions passed.\n")


# ══════════════════════════════════════════════════════════════════════
#  TEST 2:  Attention with cache vs without (correctness check)
# ══════════════════════════════════════════════════════════════════════

def test_attention_correctness():
    print("=" * 70)
    print("TEST 2: Cached attention matches non-cached attention")
    print("=" * 70)

    np.random.seed(42)
    B, H, D = 1, 2, 4
    d_model = H * D
    S = 6  # sequence length
    T = 1  # decode step

    # Random projection matrices
    w_q = np.random.randn(d_model, d_model).astype(np.float32)
    w_k = np.random.randn(d_model, d_model).astype(np.float32)
    w_v = np.random.randn(d_model, d_model).astype(np.float32)
    w_o = np.random.randn(d_model, d_model).astype(np.float32)

    # Simulate embeddings for S+T tokens
    all_tokens = np.random.randn(B, S + T, d_model).astype(np.float32)

    # --- METHOD A: Non-cached (full recomputation) ---
    from kv_cache import _scaled_dot_product_attention, _softmax

    q_full = (all_tokens @ w_q).reshape(B, S + T, H, D)
    k_full = (all_tokens @ w_k).reshape(B, S + T, H, D)
    v_full = (all_tokens @ w_v).reshape(B, S + T, H, D)

    # Compute attention for the LAST position only (autoregressive)
    out_heads_a = np.empty((T, H, D), dtype=np.float32)
    for h in range(H):
        q_h = q_full[0, S:, h, :]   # (1, D)
        k_h = k_full[0, :, h, :]    # (S+T, D)
        v_h = v_full[0, :, h, :]    # (S+T, D)
        out_heads_a[:, h, :] = _scaled_dot_product_attention(q_h, k_h, v_h)
    result_a = out_heads_a.reshape(T, d_model) @ w_o

    # --- METHOD B: Cached (prefill S tokens, then decode 1) ---
    cache = KVCache(B, S + T, H, D)

    # Prefill: write K, V for first S tokens
    k_prefill = k_full[:, :S, :, :].transpose(0, 2, 1, 3)  # (B, H, S, D)
    v_prefill = v_full[:, :S, :, :].transpose(0, 2, 1, 3)
    cache.update(k_prefill, v_prefill)

    # Decode: write K, V for the new token
    k_decode = k_full[:, S:, :, :].transpose(0, 2, 1, 3)  # (B, H, 1, D)
    v_decode = v_full[:, S:, :, :].transpose(0, 2, 1, 3)
    cache.update(k_decode, v_decode)

    # Now compute attention for the new token using the cache
    q_new = all_tokens[:, S:, :]  # (B, 1, d_model)
    result_b = multi_head_attention_with_cache(q_new, cache, w_q, w_k, w_v, w_o)

    np.testing.assert_allclose(result_a, result_b[0], atol=1e-5)
    print(f"Non-cached output: {result_a.flatten()[:4]}")
    print(f"Cached output:     {result_b.flatten()[:4]}")
    print("✓ Cached and non-cached outputs match.\n")


# ══════════════════════════════════════════════════════════════════════
#  TEST 3:  Multi-batch with variable sequence lengths
# ══════════════════════════════════════════════════════════════════════

def test_variable_seq_lens():
    print("=" * 70)
    print("TEST 3: Multi-batch with variable sequence lengths")
    print("=" * 70)

    np.random.seed(123)
    B, H, D = 3, 4, 8
    S_max = 32

    cache = KVCache(B, S_max, H, D)

    # --- Prefill each batch element with a different prompt length ---
    # We bypass the batched update() and write each element directly
    # into the underlying cache arrays.  This simulates the real
    # scenario where different requests arrive with different prompt
    # lengths and are packed into the same batch.
    prompt_lens = [5, 12, 3]

    original_k = {}
    original_v = {}

    for b in range(B):
        L = prompt_lens[b]
        k = np.random.randn(H, L, D).astype(np.float32)
        v = np.random.randn(H, L, D).astype(np.float32)
        cache.k_cache[b, :, :L, :] = k
        cache.v_cache[b, :, :L, :] = v
        cache.seq_lens[b] = L
        original_k[b] = k
        original_v[b] = v

    print(f"After prefill: seq_lens={cache.seq_lens}")
    assert cache.seq_lens == prompt_lens

    # --- Verify prefill retrieval ---
    for b in range(B):
        k_ret, v_ret = cache.get_kv(b)
        np.testing.assert_allclose(k_ret, original_k[b])
        np.testing.assert_allclose(v_ret, original_v[b])
        print(f"  Batch {b}: ✓ prefill data verified (len={prompt_lens[b]})")

    # --- Decode: all batch elements advance together (normal decode) ---
    for step in range(4):
        one_k = np.random.randn(B, H, 1, D).astype(np.float32)
        one_v = np.random.randn(B, H, 1, D).astype(np.float32)
        cache.update(one_k, one_v)
        print(f"  Decode step {step}: seq_lens={cache.seq_lens}")

    # Verify each batch element has the right length
    expected = [l + 4 for l in prompt_lens]
    for b in range(B):
        k_b, v_b = cache.get_kv(b)
        print(f"  Batch {b}: expected len={expected[b]}, got K shape seq dim={k_b.shape[1]}")
        assert k_b.shape[1] == expected[b]

    print("✓ Variable sequence lengths handled correctly.\n")


# ══════════════════════════════════════════════════════════════════════
#  TEST 4:  Incremental decoder end-to-end
# ══════════════════════════════════════════════════════════════════════

def test_incremental_decoder():
    print("=" * 70)
    print("TEST 4: Incremental decoder (prefill + autoregressive decode)")
    print("=" * 70)

    np.random.seed(7)
    d_model = 32
    num_heads = 4
    num_layers = 2
    max_seq_len = 64
    vocab_size = 100
    B = 1

    decoder = IncrementalDecoder(d_model, num_heads, num_layers, max_seq_len, vocab_size)
    decoder.max_seq_len = max_seq_len
    decoder._init_caches(B)

    # Prefill with a prompt of 8 tokens
    prompt = np.array([[1, 5, 10, 15, 20, 25, 30, 35]], dtype=np.int64)  # (1, 8)
    logits = decoder.forward_step(prompt, decoder.caches, is_prefill=True)
    print(f"After prefill (8 tokens):")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Cache seq_lens: {[c.seq_lens for c in decoder.caches]}")

    # Autoregressive decode: generate 5 more tokens
    generated = []
    next_token = logits.argmax(axis=-1)  # (1,)
    generated.append(next_token[0])

    for step in range(5):
        logits = decoder.forward_step(next_token, decoder.caches)
        next_token = logits.argmax(axis=-1)
        generated.append(next_token[0])
        print(
            f"  Decode step {step}: seq_lens={decoder.caches[0].seq_lens}, "
            f"token={next_token[0]}"
        )

    assert decoder.caches[0].seq_lens[0] == 8 + 5, "Should have 13 tokens cached"
    print(f"Generated tokens: {generated}")
    print("✓ Incremental decoder works.\n")


# ══════════════════════════════════════════════════════════════════════
#  TEST 5:  Paged KV-cache
# ══════════════════════════════════════════════════════════════════════

def test_paged_cache():
    print("=" * 70)
    print("TEST 5: Paged KV-cache (block-based allocation)")
    print("=" * 70)

    np.random.seed(99)
    num_blocks = 20
    block_size = 4
    H, D = 4, 8
    max_seqs = 4

    paged = PagedKVCache(num_blocks, block_size, H, D, max_seqs)
    print(f"Initial: {paged}")

    # Start 3 sequences with different lengths
    seq_ids = []
    for _ in range(3):
        sid = paged.add_sequence()
        seq_ids.append(sid)

    # Write different amounts to each
    lengths = [6, 11, 3]
    original_data_k = {}
    original_data_v = {}

    for i, sid in enumerate(seq_ids):
        L = lengths[i]
        k = np.random.randn(H, L, D).astype(np.float32)
        v = np.random.randn(H, L, D).astype(np.float32)
        paged.update(sid, k, v)
        original_data_k[sid] = k
        original_data_v[sid] = v
        print(f"  Seq {sid}: wrote {L} tokens, seq_len={paged.seq_lens[sid]}")

    print(f"After writes: {paged}")

    # Verify retrieval
    for i, sid in enumerate(seq_ids):
        k_ret, v_ret = paged.get_kv(sid)
        L = lengths[i]
        assert k_ret.shape == (H, L, D), f"Seq {sid}: expected ({H}, {L}, {D}), got {k_ret.shape}"
        np.testing.assert_allclose(k_ret, original_data_k[sid], atol=1e-6)
        np.testing.assert_allclose(v_ret, original_data_v[sid], atol=1e-6)
        print(f"  Seq {sid}: ✓ retrieved data matches original")

    # Finish sequence 1 and verify blocks are freed
    paged.finish_sequence(seq_ids[1])
    print(f"After finishing seq {seq_ids[1]}: {paged}")

    # Allocate a new sequence — should reuse freed blocks
    new_sid = paged.add_sequence()
    k_new = np.random.randn(H, 8, D).astype(np.float32)
    v_new = np.random.randn(H, 8, D).astype(np.float32)
    paged.update(new_sid, k_new, v_new)
    print(f"New seq {new_sid} with 8 tokens: {paged}")

    # Verify new sequence data
    k_new_ret, v_new_ret = paged.get_kv(new_sid)
    np.testing.assert_allclose(k_new_ret, k_new, atol=1e-6)
    print("✓ Paged KV-cache works correctly.\n")


# ══════════════════════════════════════════════════════════════════════
#  TEST 6:  Quantized KV-cache
# ══════════════════════════════════════════════════════════════════════

def test_quantized_cache():
    print("=" * 70)
    print("TEST 6: Quantized KV-cache (INT8 and INT4)")
    print("=" * 70)

    np.random.seed(42)
    B, H, D, S_max = 1, 2, 8, 32

    for bits in [8, 4]:
        print(f"\n--- INT{bits} ---")
        qcache = QuantizedKVCache(B, S_max, H, D, bits=bits)
        print(f"  {qcache}")

        # Write some tokens
        T = 10
        k_orig = np.random.randn(B, H, T, D).astype(np.float32) * 2
        v_orig = np.random.randn(B, H, T, D).astype(np.float32) * 2
        qcache.update(k_orig, v_orig)

        # Retrieve and measure error
        k_ret, v_ret = qcache.get_kv(0)
        assert k_ret.shape == (H, T, D)

        k_error = np.mean(np.abs(k_ret - k_orig[0]))
        v_error = np.mean(np.abs(v_ret - v_orig[0]))
        print(f"  Mean absolute error (K): {k_error:.6f}")
        print(f"  Mean absolute error (V): {v_error:.6f}")
        print(f"  Memory savings vs FP32: {qcache.savings_vs_fp32():.3f}x")
        print(f"  Actual memory: {qcache.memory_bytes() / 1e3:.1f} KB")

        # For INT8, error should be small; for INT4, larger but bounded
        # Scale factor ≈ (max-min) / 255 for INT8, so error ≈ scale/2 per element
        max_expected_error = {8: 0.1, 4: 0.5}
        assert k_error < max_expected_error[bits], f"INT{bits} quantization error too large: {k_error}"

    print("\n✓ Quantized cache works.\n")


# ══════════════════════════════════════════════════════════════════════
#  TEST 7:  Memory growth analysis
# ══════════════════════════════════════════════════════════════════════

def test_memory_analysis():
    print("=" * 70)
    print("TEST 7: Memory growth analysis")
    print("=" * 70)

    # GPT-4 class model: 32 layers, 32 heads, dim 128
    print("\nKV-Cache Memory vs Sequence Length (GPT-4-class model)")
    print("Model: 32 layers, 32 heads, head_dim=128, batch=1, FP32")
    print(memory_growth_table())

    # Llama-2 70B class
    print("\nKV-Cache Memory vs Sequence Length (Llama-2 70B class)")
    print("Model: 80 layers, 64 heads, head_dim=128, batch=1, FP32")
    print(memory_growth_table(num_layers=80, num_heads=64, head_dim=128))

    # Batch scaling
    print("\nMemory scaling with batch size (seq_len=4096):")
    print(f"{'Batch':>8} | {'Total (GB)':>12}")
    print("-" * 28)
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        info = memory_analysis(32, 32, 128, bs, 4096)
        print(f"{bs:>8} | {info['total_GB']:>12.3f}")

    print()


# ══════════════════════════════════════════════════════════════════════
#  TEST 8:  FLOPs comparison — cached vs uncached
# ══════════════════════════════════════════════════════════════════════

def test_flops_analysis():
    print("=" * 70)
    print("TEST 8: FLOPs saved by KV-caching")
    print("=" * 70)

    d_model = 4096
    H = 32
    D = d_model // H
    prompt_len = 1024
    decode_steps = 100

    # Without cache: each decode step recomputes attention for ALL positions
    # FLOPs per attention step = 2 * S * d_model (Q projection)
    #                          + 2 * S * d_model * S (attention scores) -- O(S²)
    #                          + 2 * S * d_model * S (weighted sum)
    #                          ≈ 4 * S² * d_model per layer

    # With cache: each decode step only computes for 1 new token
    # FLOPs = 2 * d_model (Q projection for 1 token)
    #       + 2 * S * d_model (Q * K^T for 1 query vs S keys)
    #       + 2 * S * d_model (attention weights * V)
    #       ≈ 4 * S * d_model per layer

    flops_no_cache = 4 * decode_steps * (prompt_len + decode_steps) ** 2 * d_model
    flops_cached = (
        # Prefill: O(S² * d_model)
        4 * prompt_len**2 * d_model
        # Decode: O(S * d_model) per step
        + sum(4 * (prompt_len + t) * d_model for t in range(decode_steps))
    )

    print(f"Model d_model={d_model}, H={H}, prompt={prompt_len}, decode={decode_steps}")
    print(f"  Without cache: {flops_no_cache:.3e} FLOPs")
    print(f"  With cache:    {flops_cached:.3e} FLOPs")
    print(f"  Speedup:       {flops_no_cache / flops_cached:.1f}x")
    print()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_basic_cache()
    test_attention_correctness()
    test_variable_seq_lens()
    test_incremental_decoder()
    test_paged_cache()
    test_quantized_cache()
    test_memory_analysis()
    test_flops_analysis()

    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
