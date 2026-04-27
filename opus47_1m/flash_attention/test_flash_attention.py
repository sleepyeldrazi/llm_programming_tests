"""Tests for flash_attention_fwd."""

import tracemalloc

import numpy as np

from flash_attention import flash_attention_fwd, naive_attention


def test_correctness_small():
    """(B=1, H=1, N=256, D=64), tile_size=64, causal=True — match naive within 1e-4."""
    rng = np.random.default_rng(0)
    B, H, N, D = 1, 1, 256, 64
    Q = rng.standard_normal((B, H, N, D)).astype(np.float64)
    K = rng.standard_normal((B, H, N, D)).astype(np.float64)
    V = rng.standard_normal((B, H, N, D)).astype(np.float64)

    O_flash = flash_attention_fwd(Q, K, V, tile_size=64, causal=True)
    O_naive = naive_attention(Q, K, V, causal=True)

    rel_err = np.linalg.norm(O_flash - O_naive) / np.linalg.norm(O_naive)
    print(f"[small] relative error vs naive: {rel_err:.3e}")
    assert rel_err < 1e-4, f"relative error {rel_err} exceeds 1e-4"


def test_correctness_noncausal():
    """Non-causal sanity check at a different tile shape."""
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((2, 4, 130, 32)).astype(np.float64)
    K = rng.standard_normal((2, 4, 130, 32)).astype(np.float64)
    V = rng.standard_normal((2, 4, 130, 32)).astype(np.float64)

    O_flash = flash_attention_fwd(Q, K, V, tile_size=37, causal=False)
    O_naive = naive_attention(Q, K, V, causal=False)
    rel_err = np.linalg.norm(O_flash - O_naive) / np.linalg.norm(O_naive)
    print(f"[noncausal, ragged tiles] relative error: {rel_err:.3e}")
    assert rel_err < 1e-4


def test_memory_no_NN_allocation():
    """(B=2, H=8, N=4096, D=64). Verify peak alloc << B*H*N*N*itemsize."""
    B, H, N, D = 2, 8, 4096, 64
    tile_size = 128
    dtype = np.float32

    rng = np.random.default_rng(2)
    Q = rng.standard_normal((B, H, N, D)).astype(dtype)
    K = rng.standard_normal((B, H, N, D)).astype(dtype)
    V = rng.standard_normal((B, H, N, D)).astype(dtype)

    # An (N, N) tensor (per batch/head) would be N*N*itemsize bytes.
    # A full B*H*N*N tensor would be that times B*H.
    nn_per_bh = N * N * np.dtype(dtype).itemsize
    nn_full = B * H * nn_per_bh
    print(f"[memory] N*N*itemsize = {nn_per_bh / 1e6:.1f} MB per (B,H)")
    print(f"[memory] B*H*N*N*itemsize = {nn_full / 1e6:.1f} MB full")

    tracemalloc.start()
    tracemalloc.reset_peak()
    O = flash_attention_fwd(Q, K, V, tile_size=tile_size, causal=True)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"[memory] tracemalloc peak during flash attention: {peak / 1e6:.2f} MB")
    print(f"[memory] output shape: {O.shape}")

    # The output array itself (B, H, N, D) is unavoidable and accounts for
    # B*H*N*D*itemsize = 16 MB here. The score tile is only (B, H, T, T) =
    # 2 MB. A full (N, N) score matrix per (B, H) would be ~67 MB; the full
    # (B, H, N, N) score tensor would be ~1.07 GB. We assert peak stays
    # below a *single* (N, N) allocation, which proves no (N, N) — let
    # alone (B, H, N, N) — score matrix was ever materialized.
    output_bytes = B * H * N * D * np.dtype(dtype).itemsize
    assert peak < nn_per_bh, (
        f"peak {peak} bytes >= one (N,N) allocation ({nn_per_bh} bytes); "
        f"flash attention should stay well below that"
    )
    # Tighter sanity: peak should be roughly output + a few tile buffers.
    assert peak < output_bytes + 16 * 1024 * 1024, (
        f"peak {peak} bytes is more than output + 16 MB of tile work"
    )


def test_causal_first_row_sanity():
    """Row 0 with causal masking must equal V[0] regardless of K (only j=0 attends)."""
    rng = np.random.default_rng(3)
    Q = rng.standard_normal((1, 1, 64, 16)).astype(np.float64)
    K = rng.standard_normal((1, 1, 64, 16)).astype(np.float64)
    V = rng.standard_normal((1, 1, 64, 16)).astype(np.float64)

    O = flash_attention_fwd(Q, K, V, tile_size=16, causal=True)
    np.testing.assert_allclose(O[0, 0, 0], V[0, 0, 0], rtol=1e-12, atol=1e-12)
    print("[causal row 0] O[0] == V[0] ✓")


if __name__ == "__main__":
    test_correctness_small()
    test_correctness_noncausal()
    test_causal_first_row_sanity()
    test_memory_no_NN_allocation()
    print("\nAll tests passed.")
