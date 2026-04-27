import numpy as np
import tracemalloc


def naive_attention(Q, K, V, causal=False):
    """Standard full-softmax attention for verification."""
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    # Full (N, N) attention matrix
    S = np.einsum("bhnd,bhmd->bhnm", Q, K) * scale
    if causal:
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        S[:, :, mask] = -np.inf
    # Stable softmax
    row_max = S.max(axis=-1, keepdims=True)
    exp_S = np.exp(S - row_max)
    probs = exp_S / exp_S.sum(axis=-1, keepdims=True)
    out = np.einsum("bhnm,bhmd->bhnd", probs, V)
    return out


def flash_attention_fwd(Q, K, V, tile_size, causal=False):
    """
    Tiled (Flash) attention forward pass using online softmax.

    Processes Q and K/V in tiles of size `tile_size`. Never materializes
    the full (N, N) attention matrix.

    Online softmax rescaling recurrence:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For each query row, we maintain running max `m`, running sum `l`,
    and accumulated output `O`. For each new KV tile with scores S_tile:

        m_new = max(m_old, max(S_tile))
        correction = exp(m_old - m_new)
        O_new = O_old * correction + exp(S_tile - m_new) @ V_tile
        l_new = l_old * correction + sum(exp(S_tile - m_new))

    Final output = O_new / l_new

    Why exp(m_old - m_new) and NOT exp(m_new - m_old):
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Suppose we have already computed exp(S_old - m_old) for previous tiles
    and now found m_new >= m_old. We need to re-express the old terms under
    the new max m_new:

        exp(S_old - m_new) = exp(S_old - m_old) * exp(m_old - m_new)

    Since m_new >= m_old, the correction factor exp(m_old - m_new) <= 1.
    Using exp(m_new - m_old) would give a factor >= 1, which is the
    reciprocal and would cause the old terms to blow up rather than
    shrink. The direction matters because we're shifting the subtracted
    constant from m_old to the larger m_new, making every old exponent
    smaller by exactly (m_new - m_old).

    Numerical stability hazard at causal tile boundaries:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When a query row's first KV tile is fully masked (all positions in
    that KV tile are beyond the query position), every score in S_tile
    is -inf. After subtracting m_new, exp(S_tile - m_new) = 0 for all
    entries. This means:
        - m_new = max(-inf, -inf) = -inf  (row max of all -inf scores)
        - l_new = l_old * exp(-inf - (-inf)) + 0

    The issue: if m_old = -inf and m_new = -inf, then
    exp(m_old - m_new) = exp(-inf - (-inf)) = exp(nan) = nan.
    This propagates nan through O and l, corrupting the result.

    Fix: When a tile is fully masked, we must skip the update entirely
    (no rescaling, no accumulation). The running m, l, O remain
    unchanged. We detect this by checking if the row max of S_tile is
    -inf (or a very negative sentinel), and only updating rows that
    have at least one valid (unmasked) score.
    """
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)

    # Precompute all Q tiles
    num_q_tiles = int(np.ceil(N / tile_size))
    Q_tiles = [Q[:, :, i * tile_size:(i + 1) * tile_size, :]
               for i in range(num_q_tiles)]

    # Precompute all KV tiles
    num_kv_tiles = int(np.ceil(N / tile_size))
    K_tiles = [K[:, :, i * tile_size:(i + 1) * tile_size, :]
               for i in range(num_kv_tiles)]
    V_tiles = [V[:, :, i * tile_size:(i + 1) * tile_size, :]
               for i in range(num_kv_tiles)]

    # Output accumulator
    out = np.zeros_like(Q)

    # Process each Q tile independently
    for qi, Q_t in enumerate(Q_tiles):
        q_tile_start = qi * tile_size
        q_tile_end = min(q_tile_start + tile_size, N)
        q_actual_size = q_tile_end - q_tile_start  # last tile may be smaller

        # Initialize online softmax state for this Q tile
        # m: running max per (B, H, q_row) — shape (B, H, q_actual_size)
        # l: running sum per (B, H, q_row)
        # O: accumulated output per (B, H, q_actual_size, D)
        m = np.full((B, H, q_actual_size), -np.inf)
        l = np.zeros((B, H, q_actual_size))
        O = np.zeros((B, H, q_actual_size, D))

        for kj, (K_t, V_t) in enumerate(zip(K_tiles, V_tiles)):
            kv_tile_start = kj * tile_size
            kv_tile_end = min(kv_tile_start + tile_size, N)
            kv_actual_size = kv_tile_end - kv_tile_start

            # Check if this (Q_tile, KV_tile) block is entirely masked
            # under causal: query at q_tile_start can only attend to
            # keys up to position q_tile_start. If kv_tile_start >
            # q_tile_end - 1, every query in this tile is before every
            # key, so the entire block is masked.
            if causal and kv_tile_start >= q_tile_end:
                # Entire block is above the diagonal — skip completely
                continue

            # Compute local attention scores: (B, H, q_actual_size, kv_actual_size)
            S = np.einsum("bhqd,bhkd->bhqk", Q_t, K_t) * scale

            if causal:
                # Build causal mask for this block.
                # Query row r (global pos = q_tile_start + r) can attend
                # to key col c (global pos = kv_tile_start + c) only if
                # kv_tile_start + c <= q_tile_start + r
                # => c - r <= q_tile_start - kv_tile_start
                q_indices = np.arange(q_actual_size)  # shape (q_actual_size,)
                kv_indices = np.arange(kv_actual_size)  # shape (kv_actual_size,)
                # Global positions
                q_global = q_tile_start + q_indices
                kv_global = kv_tile_start + kv_indices
                # Mask: True where kv_global > q_global (i.e., j > i, must mask)
                causal_mask = (kv_global > q_global[:, None])  # (q_actual, kv_actual)
                S[:, :, causal_mask] = -np.inf

            # Row-wise max of S: shape (B, H, q_actual_size)
            row_max = np.max(S, axis=-1)

            # Detect rows where ALL scores are -inf (fully masked row)
            # These rows should not participate in the update at all.
            row_valid = row_max > -np.inf  # (B, H, q_actual_size)

            # m_new = max(m_old, row_max)
            # For invalid rows, keep m_old unchanged.
            m_new = np.where(row_valid, np.maximum(m, row_max), m)

            # Correction factor: exp(m_old - m_new)
            # For rows where m stays unchanged (m_new == m), correction = 1.
            # For invalid rows, correction = 1 (no rescaling).
            # We must handle the -inf - (-inf) = nan case for invalid rows.
            correction = np.exp(np.where(row_valid, m - m_new, 0.0))
            # Shape: (B, H, q_actual_size)

            # Rescale accumulated output
            O = O * correction[:, :, :, np.newaxis]

            # Rescale running sum
            l = l * correction

            # Compute stable probabilities: exp(S - m_new[:, :, :, None])
            # For invalid rows, S is all -inf, so exp(-inf - anything) = 0.
            # But m_new for invalid rows is -inf, so we get exp(-inf - (-inf)) = nan.
            # Fix: use 0 for invalid rows explicitly.
            S_shifted = S - m_new[:, :, :, np.newaxis]
            P = np.exp(S_shifted)
            # Zero out invalid rows to avoid nan from -inf - (-inf)
            P = np.where(row_valid[:, :, :, np.newaxis], P, 0.0)

            # Update running sum
            l = l + np.sum(P, axis=-1)

            # Accumulate output
            O = O + P @ V_t

            # Update m for next iteration
            m = m_new

        # Final normalization: output = O / l
        # Handle rows where l == 0 (all KV tiles were masked) — output is 0
        l_safe = np.where(l > 0, l, 1.0)
        tile_out = O / l_safe[:, :, :, np.newaxis]

        out[:, :, q_tile_start:q_tile_end, :] = tile_out

    return out


def test_accuracy():
    """
    Test with (B=1, H=1, N=256, D=64), tile_size=64, causal=True.
    Compare against naive full-softmax attention. Assert relative error < 1e-4.
    """
    print("=" * 60)
    print("TEST 1: Accuracy (N=256, D=64, tile_size=64, causal=True)")
    print("=" * 60)

    np.random.seed(42)
    B, H, N, D = 1, 1, 256, 64
    tile_size = 64

    Q = np.random.randn(B, H, N, D).astype(np.float32)
    K = np.random.randn(B, H, N, D).astype(np.float32)
    V = np.random.randn(B, H, N, D).astype(np.float32)

    # Naive reference
    naive_out = naive_attention(Q, K, V, causal=True)

    # Tiled implementation
    tiled_out = flash_attention_fwd(Q, K, V, tile_size, causal=True)

    # Compute relative error
    diff = np.abs(naive_out - tiled_out)
    rel_error = diff / (np.abs(naive_out) + 1e-10)
    max_rel_error = np.max(rel_error)
    mean_rel_error = np.mean(rel_error)

    print(f"Max relative error: {max_rel_error:.2e}")
    print(f"Mean relative error: {mean_rel_error:.2e}")
    print(f"Max absolute diff: {np.max(diff):.2e}")

    assert max_rel_error < 1e-4, (
        f"Relative error {max_rel_error:.2e} exceeds threshold 1e-4"
    )
    print("PASSED: Relative error < 1e-4\n")


def test_memory():
    """
    Test with (B=2, H=8, N=4096, D=64), tile_size=128, causal=True.
    Verify via tracemalloc that no (N, N) tensor is ever allocated.
    An (N, N) float32 tensor would be 4096*4096*4 = 64 MiB.
    We check that peak memory stays well below that threshold.
    """
    print("=" * 60)
    print("TEST 2: Memory (N=4096, D=64, tile_size=128, causal=True)")
    print("=" * 60)

    np.random.seed(42)
    B, H, N, D = 2, 8, 4096, 64
    tile_size = 128

    Q = np.random.randn(B, H, N, D).astype(np.float32)
    K = np.random.randn(B, H, N, D).astype(np.float32)
    V = np.random.randn(B, H, N, D).astype(np.float32)

    # Size of a full (N, N) float32 matrix in bytes
    nn_size_bytes = N * N * 4  # 67,108,864 bytes = 64 MiB
    nn_size_mb = nn_size_bytes / (1024 * 1024)

    # Maximum tile score matrix: (B, H, T, T) = 2 * 8 * 128 * 128 * 4 = 1 MiB
    max_tile_bytes = B * H * tile_size * tile_size * 4
    max_tile_mb = max_tile_bytes / (1024 * 1024)

    print(f"Input size: {Q.nbytes / 1e6:.1f} MB each (Q, K, V)")
    print(f"Full (N,N) matrix would be: {nn_size_mb:.0f} MB")
    print(f"Max tile score matrix: {max_tile_mb:.1f} MB")

    tracemalloc.start()
    tiled_out = flash_attention_fwd(Q, K, V, tile_size, causal=True)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    print(f"\nPeak memory during tiled attention: {peak_mb:.1f} MB")
    print(f"Threshold (N*N matrix): {nn_size_mb:.0f} MB")

    # Peak memory should be well below the (N, N) matrix size.
    # The inputs themselves are ~16 MB each, so total input is ~48 MB.
    # Peak should be inputs + working memory, but no single allocation
    # should approach 64 MiB for an (N, N) matrix.
    # We allow for input storage + some overhead, but cap at half of N*N.
    assert peak < nn_size_bytes * 0.5, (
        f"Peak memory {peak_mb:.1f} MB is suspiciously close to "
        f"{nn_size_mb:.0f} MB (N*N matrix). May be materializing full matrix."
    )
    print("PASSED: No (N, N) tensor allocated\n")


def test_non_causal():
    """
    Test non-causal mode against naive attention.
    """
    print("=" * 60)
    print("TEST 3: Non-causal (N=256, D=64, tile_size=64, causal=False)")
    print("=" * 60)

    np.random.seed(42)
    B, H, N, D = 1, 1, 256, 64
    tile_size = 64

    Q = np.random.randn(B, H, N, D).astype(np.float32)
    K = np.random.randn(B, H, N, D).astype(np.float32)
    V = np.random.randn(B, H, N, D).astype(np.float32)

    naive_out = naive_attention(Q, K, V, causal=False)
    tiled_out = flash_attention_fwd(Q, K, V, tile_size, causal=False)

    diff = np.abs(naive_out - tiled_out)
    rel_error = diff / (np.abs(naive_out) + 1e-10)
    max_rel_error = np.max(rel_error)

    print(f"Max relative error: {max_rel_error:.2e}")
    assert max_rel_error < 1e-4, (
        f"Relative error {max_rel_error:.2e} exceeds threshold 1e-4"
    )
    print("PASSED: Relative error < 1e-4\n")


def test_large_batch():
    """
    Test with larger batch and head dimensions.
    """
    print("=" * 60)
    print("TEST 4: Larger batch (B=2, H=8, N=512, D=64, tile_size=128)")
    print("=" * 60)

    np.random.seed(42)
    B, H, N, D = 2, 8, 512, 64
    tile_size = 128

    Q = np.random.randn(B, H, N, D).astype(np.float32)
    K = np.random.randn(B, H, N, D).astype(np.float32)
    V = np.random.randn(B, H, N, D).astype(np.float32)

    naive_out = naive_attention(Q, K, V, causal=True)
    tiled_out = flash_attention_fwd(Q, K, V, tile_size, causal=True)

    diff = np.abs(naive_out - tiled_out)
    rel_error = diff / (np.abs(naive_out) + 1e-10)
    max_rel_error = np.max(rel_error)
    mean_rel_error = np.mean(rel_error)

    print(f"Max relative error: {max_rel_error:.2e}")
    print(f"Mean relative error: {mean_rel_error:.2e}")
    assert max_rel_error < 1e-4, (
        f"Relative error {max_rel_error:.2e} exceeds threshold 1e-4"
    )
    print("PASSED: Relative error < 1e-4\n")


def test_uneven_tiles():
    """
    Test with N not evenly divisible by tile_size.
    """
    print("=" * 60)
    print("TEST 5: Uneven tiles (N=300, D=64, tile_size=97, causal=True)")
    print("=" * 60)

    np.random.seed(42)
    B, H, N, D = 1, 1, 300, 64
    tile_size = 97

    Q = np.random.randn(B, H, N, D).astype(np.float32)
    K = np.random.randn(B, H, N, D).astype(np.float32)
    V = np.random.randn(B, H, N, D).astype(np.float32)

    naive_out = naive_attention(Q, K, V, causal=True)
    tiled_out = flash_attention_fwd(Q, K, V, tile_size, causal=True)

    diff = np.abs(naive_out - tiled_out)
    rel_error = diff / (np.abs(naive_out) + 1e-10)
    max_rel_error = np.max(rel_error)

    print(f"Max relative error: {max_rel_error:.2e}")
    assert max_rel_error < 1e-4, (
        f"Relative error {max_rel_error:.2e} exceeds threshold 1e-4"
    )
    print("PASSED: Relative error < 1e-4\n")


if __name__ == "__main__":
    test_accuracy()
    test_non_causal()
    test_large_batch()
    test_uneven_tiles()
    test_memory()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
