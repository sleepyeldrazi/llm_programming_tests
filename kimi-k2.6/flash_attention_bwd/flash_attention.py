"""
Flash Attention Forward + Backward Implementation in NumPy.

Follows the tiled online softmax algorithm with recomputation in backward.
No full (N, N) attention matrix is ever materialized in forward or backward.
"""

import numpy as np
import tracemalloc


def flash_attention_fwd(Q, K, V, tile_size, causal=True):
    """
    Forward pass of Flash Attention with online softmax.

    Parameters
    ----------
    Q, K, V : np.ndarray, shape (B, H, N, D)
    tile_size : int
        Tile size T for Q and KV blocks.
    causal : bool
        If True, apply causal (lower-triangular) masking.

    Returns
    -------
    O : np.ndarray, shape (B, H, N, D)
        Attention output.
    cache : dict
        {'O': O, 'L': L, 'Q': Q, 'K': K, 'V': V}
        L has shape (B, H, N).
    """
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)

    O = np.zeros_like(Q)
    L = np.empty((B, H, N), dtype=Q.dtype)

    num_tiles = (N + tile_size - 1) // tile_size

    for b in range(B):
        for h in range(H):
            # Per-head accumulators
            O_bh = np.zeros((N, D), dtype=Q.dtype)
            L_bh = np.empty(N, dtype=Q.dtype)

            for qi in range(num_tiles):
                q_start = qi * tile_size
                q_end = min(q_start + tile_size, N)
                T_q = q_end - q_start
                Q_tile = Q[b, h, q_start:q_end, :]  # (T_q, D)

                # Online softmax accumulators for this Q tile
                m = np.full(T_q, -np.inf, dtype=np.float64)  # running max
                l = np.zeros(T_q, dtype=np.float64)          # running sum
                acc = np.zeros((T_q, D), dtype=np.float64)   # running output

                for kvj in range(num_tiles):
                    kv_start = kvj * tile_size
                    kv_end = min(kv_start + tile_size, N)

                    # Causal skip: if the entire KV block is after the Q block, skip
                    if causal and kv_start > q_end - 1:
                        continue

                    K_tile = K[b, h, kv_start:kv_end, :]  # (T_kv, D)
                    V_tile = V[b, h, kv_start:kv_end, :]  # (T_kv, D)

                    S = Q_tile @ K_tile.T * scale  # (T_q, T_kv)

                    if causal:
                        # Build causal mask: S positions where col > row (within global indices) get -inf
                        q_idx = np.arange(q_start, q_end)[:, None]  # (T_q, 1)
                        k_idx = np.arange(kv_start, kv_end)[None, :]  # (1, T_kv)
                        mask = k_idx > q_idx
                        S = np.where(mask, -np.inf, S)

                    # Online softmax update
                    m_new = np.maximum(m, np.max(S, axis=1, where=~np.isinf(S), initial=-np.inf))

                    # Compute exp(S - m_new[:, None]) safely
                    # For rows where all S are -inf, m_new stays -inf; those positions are masked out
                    exp_S = np.exp(S - m_new[:, None])
                    # Zero out -inf positions
                    exp_S = np.where(np.isinf(S), 0.0, exp_S)

                    l_new = l * np.exp(m - m_new) + np.sum(exp_S, axis=1)

                    # Update output accumulator
                    acc = acc * np.exp(m - m_new)[:, None] + exp_S @ V_tile

                    m = m_new
                    l = l_new

                # Write tile results
                O_bh[q_start:q_end, :] = acc / l[:, None]
                L_bh[q_start:q_end] = m + np.log(l)

            O[b, h, :, :] = O_bh
            L[b, h, :] = L_bh

    cache = {'O': O, 'L': L, 'Q': Q, 'K': K, 'V': V}
    return O, cache


def flash_attention_bwd(dO, cache, tile_size, causal=True):
    """
    Backward pass of Flash Attention with recomputation.

    Parameters
    ----------
    dO : np.ndarray, shape (B, H, N, D)
        Upstream gradient w.r.t. O.
    cache : dict
        From forward: {'O': O, 'L': L, 'Q': Q, 'K': K, 'V': V}
    tile_size : int
        Tile size T.
    causal : bool
        Same as forward.

    Returns
    -------
    dQ, dK, dV : np.ndarray, shape (B, H, N, D)
    """
    Q = cache['Q']
    K = cache['K']
    V = cache['V']
    L = cache['L']

    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)

    dQ = np.zeros_like(Q)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)

    num_tiles = (N + tile_size - 1) // tile_size

    for b in range(B):
        for h in range(H):
            dQ_bh = np.zeros((N, D), dtype=np.float64)
            dK_bh = np.zeros((N, D), dtype=np.float64)
            dV_bh = np.zeros((N, D), dtype=np.float64)

            for qi in range(num_tiles):
                q_start = qi * tile_size
                q_end = min(q_start + tile_size, N)
                T_q = q_end - q_start
                Q_tile = Q[b, h, q_start:q_end, :]      # (T_q, D)
                dO_tile = dO[b, h, q_start:q_end, :]    # (T_q, D)
                L_query = L[b, h, q_start:q_end]        # (T_q,)

                # -----------------------------------------------------------------
                # Pass 1: accumulate rowsum_PdP over ALL KV tiles for this Q tile
                # -----------------------------------------------------------------
                rowsum_PdP = np.zeros((T_q, 1), dtype=np.float64)
                for kvj in range(num_tiles):
                    kv_start = kvj * tile_size
                    kv_end = min(kv_start + tile_size, N)

                    if causal and kv_start > q_end - 1:
                        continue

                    K_tile = K[b, h, kv_start:kv_end, :]  # (T_kv, D)
                    V_tile = V[b, h, kv_start:kv_end, :]  # (T_kv, D)

                    S = (Q_tile @ K_tile.T) * scale       # (T_q, T_kv)

                    if causal:
                        q_idx = np.arange(q_start, q_end)[:, None]
                        k_idx = np.arange(kv_start, kv_end)[None, :]
                        mask = k_idx > q_idx
                        S = np.where(mask, -np.inf, S)

                    P = np.exp(S - L_query[:, None])      # (T_q, T_kv)
                    P = np.where(np.isinf(S), 0.0, P)

                    dP = dO_tile @ V_tile.T               # (T_q, T_kv)
                    rowsum_PdP += np.sum(P * dP, axis=-1, keepdims=True)

                # -----------------------------------------------------------------
                # Pass 2: compute dS and accumulate dQ, dK, dV
                # -----------------------------------------------------------------
                for kvj in range(num_tiles):
                    kv_start = kvj * tile_size
                    kv_end = min(kv_start + tile_size, N)

                    if causal and kv_start > q_end - 1:
                        continue

                    K_tile = K[b, h, kv_start:kv_end, :]  # (T_kv, D)
                    V_tile = V[b, h, kv_start:kv_end, :]  # (T_kv, D)

                    S = (Q_tile @ K_tile.T) * scale       # (T_q, T_kv)

                    if causal:
                        q_idx = np.arange(q_start, q_end)[:, None]
                        k_idx = np.arange(kv_start, kv_end)[None, :]
                        mask = k_idx > q_idx
                        S = np.where(mask, -np.inf, S)

                    P = np.exp(S - L_query[:, None])      # (T_q, T_kv)
                    P = np.where(np.isinf(S), 0.0, P)

                    # dV contribution: P^T @ dO_tile
                    dV_bh[kv_start:kv_end, :] += P.T @ dO_tile

                    dP = dO_tile @ V_tile.T               # (T_q, T_kv)
                    dS = P * (dP - rowsum_PdP)            # (T_q, T_kv)

                    # dQ contribution
                    dQ_bh[q_start:q_end, :] += dS @ K_tile * scale

                    # dK contribution
                    dK_bh[kv_start:kv_end, :] += dS.T @ Q_tile * scale

            dQ[b, h, :, :] = dQ_bh
            dK[b, h, :, :] = dK_bh
            dV[b, h, :, :] = dV_bh

    return dQ, dK, dV


# =============================================================================
# Helper: naive attention for testing
# =============================================================================

def naive_attention(Q, K, V, causal=True):
    """
    Naive attention for reference: materializes full (N, N) matrix.
    """
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    S = np.einsum('bhqd,bhkd->bhqk', Q, K) * scale  # (B, H, N, N)
    if causal:
        mask = np.triu(np.ones((N, N)), k=1).astype(bool)
        S = np.where(mask[None, None, :, :], -np.inf, S)
    # Softmax
    S_max = np.max(S, axis=-1, keepdims=True)
    exp_S = np.exp(S - S_max)
    sum_exp = np.sum(exp_S, axis=-1, keepdims=True)
    P = exp_S / sum_exp
    O = np.einsum('bhqk,bhkd->bhqd', P, V)
    return O, P


def naive_attention_bwd(dO, Q, K, V, causal=True):
    """
    Naive backward by materializing P and using standard formulas.
    """
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    S = np.einsum('bhqd,bhkd->bhqk', Q, K) * scale
    if causal:
        mask = np.triu(np.ones((N, N)), k=1).astype(bool)
        S = np.where(mask[None, None, :, :], -np.inf, S)
    S_max = np.max(S, axis=-1, keepdims=True)
    exp_S = np.exp(S - S_max)
    sum_exp = np.sum(exp_S, axis=-1, keepdims=True)
    P = exp_S / sum_exp

    dV = np.einsum('bhqk,bhqd->bhkd', P, dO)
    dP = np.einsum('bhqd,bhkd->bhqk', dO, V)
    rowsum_PdP = np.sum(P * dP, axis=-1, keepdims=True)
    dS = P * (dP - rowsum_PdP)
    dQ = np.einsum('bhqk,bhkd->bhqd', dS, K) * scale
    dK = np.einsum('bhqk,bhqd->bhkd', dS, Q) * scale
    return dQ, dK, dV


# =============================================================================
# Test 1: Gradient check with central finite differences
# =============================================================================

def test1_gradient_check():
    print("=" * 60)
    print("TEST 1: Gradient Check (central finite differences)")
    print("=" * 60)

    B, H, N, D = 1, 1, 64, 32
    T = 16
    causal = True
    eps = 1e-4
    tol = 1e-5

    np.random.seed(42)
    Q = np.random.randn(B, H, N, D).astype(np.float64)
    K = np.random.randn(B, H, N, D).astype(np.float64)
    V = np.random.randn(B, H, N, D).astype(np.float64)
    dO = np.random.randn(B, H, N, D).astype(np.float64)

    O, cache = flash_attention_fwd(Q, K, V, tile_size=T, causal=causal)
    dQ, dK, dV = flash_attention_bwd(dO, cache, tile_size=T, causal=causal)

    # Check dV across ALL elements
    errors_v = []
    for i in range(N):
        for j in range(D):
            V_plus = V.copy()
            V_minus = V.copy()
            V_plus[0, 0, i, j] += eps
            V_minus[0, 0, i, j] -= eps
            O_plus, _ = flash_attention_fwd(Q, K, V_plus, tile_size=T, causal=causal)
            O_minus, _ = flash_attention_fwd(Q, K, V_minus, tile_size=T, causal=causal)
            fd = np.sum(dO * (O_plus - O_minus) / (2 * eps))
            ana = dV[0, 0, i, j]
            rel_err = abs(fd - ana) / (abs(ana) + 1e-8)
            errors_v.append(rel_err)

    max_err_v = max(errors_v)
    print(f"  dV max relative error across ALL elements: {max_err_v:.3e}")
    assert max_err_v < tol, f"dV gradient check failed: {max_err_v:.3e} >= {tol}"

    # Spot-check dQ at 10 random positions
    rng = np.random.RandomState(123)
    idxs_q = rng.randint(0, N, size=10)
    idxs_qd = rng.randint(0, D, size=10)
    max_err_q = 0.0
    for i, d in zip(idxs_q, idxs_qd):
        Q_plus = Q.copy()
        Q_minus = Q.copy()
        Q_plus[0, 0, i, d] += eps
        Q_minus[0, 0, i, d] -= eps
        O_plus, _ = flash_attention_fwd(Q_plus, K, V, tile_size=T, causal=causal)
        O_minus, _ = flash_attention_fwd(Q_minus, K, V, tile_size=T, causal=causal)
        fd = np.sum(dO * (O_plus - O_minus) / (2 * eps))
        ana = dQ[0, 0, i, d]
        rel_err = abs(fd - ana) / (abs(ana) + 1e-8)
        max_err_q = max(max_err_q, rel_err)

    print(f"  dQ spot-check max relative error (10 random): {max_err_q:.3e}")
    assert max_err_q < tol, f"dQ gradient check failed: {max_err_q:.3e} >= {tol}"

    # Spot-check dK at 10 random positions
    idxs_k = rng.randint(0, N, size=10)
    idxs_kd = rng.randint(0, D, size=10)
    max_err_k = 0.0
    for i, d in zip(idxs_k, idxs_kd):
        K_plus = K.copy()
        K_minus = K.copy()
        K_plus[0, 0, i, d] += eps
        K_minus[0, 0, i, d] -= eps
        O_plus, _ = flash_attention_fwd(Q, K_plus, V, tile_size=T, causal=causal)
        O_minus, _ = flash_attention_fwd(Q, K_minus, V, tile_size=T, causal=causal)
        fd = np.sum(dO * (O_plus - O_minus) / (2 * eps))
        ana = dK[0, 0, i, d]
        rel_err = abs(fd - ana) / (abs(ana) + 1e-8)
        max_err_k = max(max_err_k, rel_err)

    print(f"  dK spot-check max relative error (10 random): {max_err_k:.3e}")
    assert max_err_k < tol, f"dK gradient check failed: {max_err_k:.3e} >= {tol}"

    print("  PASSED")


# =============================================================================
# Test 2: Compare against naive backward
# =============================================================================

def test2_vs_naive():
    print("=" * 60)
    print("TEST 2: Compare vs Naive Backward")
    print("=" * 60)

    B, H, N, D = 2, 4, 256, 64
    T = 64
    causal = True
    tol = 1e-4

    np.random.seed(7)
    Q = np.random.randn(B, H, N, D).astype(np.float64)
    K = np.random.randn(B, H, N, D).astype(np.float64)
    V = np.random.randn(B, H, N, D).astype(np.float64)
    dO = np.random.randn(B, H, N, D).astype(np.float64)

    O, cache = flash_attention_fwd(Q, K, V, tile_size=T, causal=causal)
    dQ_f, dK_f, dV_f = flash_attention_bwd(dO, cache, tile_size=T, causal=causal)

    dQ_n, dK_n, dV_n = naive_attention_bwd(dO, Q, K, V, causal=causal)

    def rel_err(a, b):
        return np.max(np.abs(a - b) / (np.abs(b) + 1e-8))

    err_dq = rel_err(dQ_f, dQ_n)
    err_dk = rel_err(dK_f, dK_n)
    err_dv = rel_err(dV_f, dV_n)

    print(f"  dQ max relative error: {err_dq:.3e}")
    print(f"  dK max relative error: {err_dk:.3e}")
    print(f"  dV max relative error: {err_dv:.3e}")

    assert err_dq < tol, f"dQ mismatch: {err_dq:.3e} >= {tol}"
    assert err_dk < tol, f"dK mismatch: {err_dk:.3e} >= {tol}"
    assert err_dv < tol, f"dV mismatch: {err_dv:.3e} >= {tol}"

    print("  PASSED")


# =============================================================================
# Test 3: Memory test
# =============================================================================

def test3_memory():
    print("=" * 60)
    print("TEST 3: Memory Test")
    print("=" * 60)

    B, H, N, D = 1, 1, 4096, 64
    T = 128
    causal = True

    np.random.seed(99)
    Q = np.random.randn(B, H, N, D).astype(np.float64)
    K = np.random.randn(B, H, N, D).astype(np.float64)
    V = np.random.randn(B, H, N, D).astype(np.float64)
    dO = np.random.randn(B, H, N, D).astype(np.float64)

    # Forward memory
    tracemalloc.start()
    O, cache = flash_attention_fwd(Q, K, V, tile_size=T, causal=causal)
    current_fwd, peak_fwd = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Backward memory
    tracemalloc.start()
    dQ, dK, dV = flash_attention_bwd(dO, cache, tile_size=T, causal=causal)
    current_bwd, peak_bwd = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Full (N, N) matrix memory
    nn_bytes = N * N * 8  # float64
    threshold = 0.20 * nn_bytes

    print(f"  Peak forward memory:  {peak_fwd / 1e6:.2f} MB")
    print(f"  Peak backward memory: {peak_bwd / 1e6:.2f} MB")
    print(f"  Full (N,N) matrix:    {nn_bytes / 1e6:.2f} MB")
    print(f"  Threshold (20%):      {threshold / 1e6:.2f} MB")

    assert peak_fwd < threshold, f"Forward memory {peak_fwd} >= threshold {threshold}"
    assert peak_bwd < threshold, f"Backward memory {peak_bwd} >= threshold {threshold}"

    print("  PASSED")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    test1_gradient_check()
    test2_vs_naive()
    test3_memory()
    print("\nALL TESTS PASSED")
