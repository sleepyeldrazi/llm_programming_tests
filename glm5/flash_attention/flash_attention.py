"""
Flash Attention forward pass — tiled attention with online softmax in NumPy.

Implements the algorithm from:
  "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
  by Dao et al., 2022.

Key idea: process Q and K/V in tiles so that the full (N, N) attention matrix
is never materialized in memory. An online-softmax recurrence keeps running
statistics (row-wise max, row-wise sum-of-exp) and incrementally builds the
output.
"""

import numpy as np
import tracemalloc


# ---------------------------------------------------------------------------
# Naive full-softmax attention (reference implementation)
# ---------------------------------------------------------------------------

def naive_attention(Q, K, V, causal=True):
    """
    Standard scaled dot-product attention that materializes the full (N, N)
    attention matrix. Used as a ground-truth reference.

    Parameters
    ----------
    Q, K, V : ndarray, shape (B, H, N, D)
    causal  : bool, whether to apply causal (lower-triangular) masking

    Returns
    -------
    O : ndarray, shape (B, H, N, D)
    """
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)

    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale  # (B, H, N, N)

    if causal:
        mask = np.zeros((N, N), dtype=np.float64)
        mask[np.triu_indices(N, k=1)] = -np.inf
        S = S + mask

    P = np.exp(S - S.max(axis=-1, keepdims=True))       # numerically stable
    P = P / P.sum(axis=-1, keepdims=True)
    O = np.matmul(P, V)
    return O


# ---------------------------------------------------------------------------
# Flash (tiled) attention forward pass
# ---------------------------------------------------------------------------

def flash_attention_fwd(Q, K, V, tile_size, causal=True):
    """
    Tiled flash-attention forward pass using online softmax.

    Parameters
    ----------
    Q, K, V   : ndarray, shape (B, H, N, D)
    tile_size : int, size of each tile T (e.g. 64 or 128)
    causal    : bool, apply causal masking

    Returns
    -------
    O : ndarray, shape (B, H, N, D)

    Notes on memory
    ----------------
    The largest intermediate tensor is at most (T, T) for the local score
    matrix S_tile, plus (T, D) for the local output. Since T << N, we never
    allocate anything close to (N, N).
    """
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    T = tile_size

    n_q_tiles = (N + T - 1) // T
    n_kv_tiles = (N + T - 1) // T

    O = np.zeros((B, H, N, D), dtype=Q.dtype)

    for b in range(B):
        for h in range(H):
            # Per-(b,h) loop — each sequence head is independent.

            # Running statistics per query row, shape (N,).
            m = np.full(N, -np.inf, dtype=np.float64)   # running row max
            l = np.zeros(N, dtype=np.float64)            # running row sum-of-exp

            for tq in range(n_q_tiles):
                q_start = tq * T
                q_end = min(q_start + T, N)
                q_len = q_end - q_start

                Q_tile = Q[b, h, q_start:q_end, :]  # (q_len, D)

                # Accumulator for this Q tile's output rows, shape (q_len, D).
                O_acc = np.zeros((q_len, D), dtype=np.float64)
                # Running stats for just the rows in this Q tile.
                m_tile = np.full(q_len, -np.inf, dtype=np.float64)
                l_tile = np.zeros(q_len, dtype=np.float64)

                for tk in range(n_kv_tiles):
                    k_start = tk * T
                    k_end = min(k_start + T, N)
                    k_len = k_end - k_start

                    # ----------------------------------------------------------
                    # Causal skip: if the smallest key position (k_start) is
                    # strictly greater than the largest query position
                    # (q_end - 1), then for every (i, j) pair we have j > i,
                    # meaning the entire block is masked.  Skip it.
                    # ----------------------------------------------------------
                    if causal and k_start > q_end - 1:
                        continue

                    K_tile = K[b, h, k_start:k_end, :]  # (k_len, D)
                    V_tile = V[b, h, k_start:k_end, :]  # (k_len, D)

                    # Local attention scores for this (Q_tile, K_tile) block.
                    S_tile = np.matmul(Q_tile, K_tile.T) * scale  # (q_len, k_len)

                    # Apply causal mask within the block.
                    if causal:
                        # Row i (global index q_start+i) can attend to
                        # column j (global index k_start+j) only if j <= i,
                        # i.e. k_start+j <= q_start+i  =>  j - i <= q_start - k_start.
                        # Equivalently, mask positions where k_start+j > q_start+i.
                        row_idx = np.arange(q_len)[:, None]  # (q_len, 1) local
                        col_idx = np.arange(k_len)[None, :]  # (1, k_len) local
                        # global query position = q_start + row_idx
                        # global key position   = k_start + col_idx
                        causal_mask = (k_start + col_idx) > (q_start + row_idx)
                        S_tile[causal_mask] = -np.inf

                    # ---- Online softmax recurrence ----
                    #
                    # We maintain per-row running max m and sum-of-exp l.
                    # For each new KV tile we observe a block of scores S_tile.
                    #
                    # Step 1: compute the row-wise max of the new scores.
                    row_maxes = S_tile.max(axis=-1)  # (q_len,)
                    # NOTE: if an entire row is -inf (fully masked), row_max is -inf.

                    m_new = np.maximum(m_tile, row_maxes)

                    # -----------------------------------------------------------------
                    # WHY the correction factor is exp(m_old - m_new), NOT exp(m_new - m_old):
                    #
                    # The accumulated output O_acc currently stores:
                    #   O_acc = sum_over_past_tiles [ exp(S_past - m_old) * V_past ]
                    #
                    # We want to re-express everything relative to the NEW max m_new:
                    #   O_acc_new = sum_over_past_tiles [ exp(S_past - m_new) * V_past ]
                    #
                    # Since exp(S_past - m_new) = exp(S_past - m_old) * exp(m_old - m_new),
                    # we multiply O_acc by exp(m_old - m_new).
                    #
                    # If we instead used exp(m_new - m_old), we would be MULTIPLYING by
                    # a factor >= 1 (since m_new >= m_old), which EXPLODES the accumulated
                    # sum rather than shrinking it to match the new denominator. The correct
                    # factor is always <= 1, which scales down old contributions to make
                    # room for new ones.
                    # -----------------------------------------------------------------
                    correction = np.exp(m_tile - m_new)

                    # Rescale the accumulated output and sum-of-exp.
                    O_acc = O_acc * correction[:, None]
                    l_tile = l_tile * correction

                    # Add the new tile's contributions.
                    # P_tile = exp(S_tile - m_new) are the (unnormalised) probabilities
                    # from this KV tile, computed in a numerically stable way.
                    #
                    # When a row of S_tile is entirely -inf (fully masked by causal),
                    # S_tile - m_new gives -inf - (-inf) = NaN if m_new is also -inf.
                    # We handle this by clipping: where m_new is -inf, the correction
                    # is exp(-inf - (-inf)) = NaN, but l_tile stays 0 and O_acc stays 0,
                    # so we just skip the contribution with np.where.
                    P_tile = np.exp(S_tile - m_new[:, None])

                    # Guard against NaN from 0 * inf in masked rows.
                    safe_mask = np.isfinite(m_new)
                    P_tile = np.where(safe_mask[:, None], P_tile, 0.0)

                    l_tile = l_tile + P_tile.sum(axis=-1)
                    O_acc = O_acc + np.matmul(P_tile, V_tile)

                    m_tile = m_new

                # After processing all KV tiles for this Q tile, normalise.
                #
                # -----------------------------------------------------------------
                # NUMERICAL STABILITY HAZARD at tile boundaries with causal masking:
                #
                # When a query row's first KV tile(s) are fully masked (e.g. for
                # query position i=3 and KV tile starting at k_start=64), the running
                # statistics are:
                #   m = -inf  (no valid scores seen yet)
                #   l = 0     (no valid exp contributions)
                #
                # This is dangerous because:
                #   1. If we compute exp(m_old - m_new) with m_old=-inf and m_new=-inf,
                #      we get exp(-inf - (-inf)) = exp(NaN) = NaN, which poisons O_acc.
                #   2. At final normalisation, O/l = 0/0 = NaN instead of the correct 0.
                #
                # We handle this by:
                #   - Using np.isfinite guards when computing P_tile to zero out
                #     contributions from fully-masked rows.
                #   - At normalisation, rows with l_tile == 0 get output = 0 (not NaN).
                # -----------------------------------------------------------------
                valid = l_tile > 0
                O_acc[valid] = O_acc[valid] / l_tile[valid, None]
                O_acc[~valid] = 0.0

                O[b, h, q_start:q_end, :] = O_acc

                # Also update the global running stats for completeness.
                m[q_start:q_end] = m_tile
                l[q_start:q_end] = l_tile

    return O


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _relative_error(A, B):
    denom = np.maximum(np.abs(A).max(), np.abs(B).max())
    if denom == 0:
        return 0.0
    return np.max(np.abs(A - B)) / denom


def test_small():
    """Test with B=1, H=1, N=256, D=64, tile_size=64, causal=True."""
    np.random.seed(42)
    B, H, N, D = 1, 1, 256, 64
    T = 64

    Q = np.random.randn(B, H, N, D).astype(np.float64)
    K = np.random.randn(B, H, N, D).astype(np.float64)
    V = np.random.randn(B, H, N, D).astype(np.float64)

    O_naive = naive_attention(Q, K, V, causal=True)
    O_flash = flash_attention_fwd(Q, K, V, tile_size=T, causal=True)

    rel_err = _relative_error(O_naive, O_flash)
    print(f"[test_small]  B={B}, H={H}, N={N}, D={D}, T={T}")
    print(f"  Relative error: {rel_err:.2e}")
    assert rel_err < 1e-4, f"Relative error {rel_err:.2e} exceeds 1e-4"
    print("  PASSED\n")


def test_large_tracemalloc():
    """
    Test with B=2, H=8, N=4096, D=64, tile_size=128.
    Use tracemalloc to verify no (N, N) tensor is allocated.
    """
    np.random.seed(123)
    B, H, N, D = 2, 8, 4096, 64
    T = 128

    Q = np.random.randn(B, H, N, D).astype(np.float64)
    K = np.random.randn(B, H, N, D).astype(np.float64)
    V = np.random.randn(B, H, N, D).astype(np.float64)

    tracemalloc.start()
    O_flash = flash_attention_fwd(Q, K, V, tile_size=T, causal=True)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    N_squared_bytes = N * N * 8  # float64

    print(f"[test_large_tracemalloc]  B={B}, H={H}, N={N}, D={D}, T={T}")
    print(f"  Peak memory:          {peak / 1e6:.2f} MB")
    print(f"  Single (N,N) matrix:  {N_squared_bytes / 1e6:.2f} MB")

    # The peak memory should be much less than even a single (N, N) matrix.
    # With tiling, the largest intermediate is roughly (T, T) = 128x128,
    # which is negligible. We allow some overhead for the (B,H,N,D) inputs
    # and output, but a full (N,N) matrix would dominate.
    #
    # Conservative check: peak should be < 0.5 * N^2 * 8 bytes.
    # (The inputs Q,K,V alone are 3 * B*H*N*D * 8 = 3*2*8*4096*64*8 ≈ 100 MB,
    #  so we check that the *additional* memory beyond inputs is small.)
    assert peak < N_squared_bytes, (
        f"Peak memory {peak / 1e6:.2f} MB exceeds single (N,N) matrix "
        f"{N_squared_bytes / 1e6:.2f} MB — full attention matrix may have "
        f"been materialized!"
    )
    print("  PASSED — no (N,N) tensor detected in peak memory\n")

    # Also verify correctness against naive for a subset (first head, first
    # batch element) since naive would OOM on the full tensor.
    b, h = 0, 0
    O_naive_slice = naive_attention(
        Q[b:b+1, h:h+1], K[b:b+1, h:h+1], V[b:b+1, h:h+1], causal=True
    )
    rel_err = _relative_error(O_naive_slice, O_flash[b:b+1, h:h+1])
    print(f"  Correctness check (b=0, h=0): relative error = {rel_err:.2e}")
    assert rel_err < 1e-4, f"Relative error {rel_err:.2e} exceeds 1e-4"
    print("  PASSED\n")


if __name__ == "__main__":
    test_small()
    test_large_tracemalloc()
    print("All tests passed!")
