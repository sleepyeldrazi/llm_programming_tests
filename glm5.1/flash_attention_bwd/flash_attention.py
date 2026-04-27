import numpy as np
import tracemalloc


def flash_attention_fwd(Q, K, V, tile_size, causal=True):
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    O = np.zeros((B, H, N, D), dtype=np.float64)
    L = np.full((B, H, N), -np.inf, dtype=np.float64)

    n_tiles_q = (N + tile_size - 1) // tile_size
    n_tiles_kv = (N + tile_size - 1) // tile_size

    for b in range(B):
        for h in range(H):
            for qi in range(n_tiles_q):
                q_start = qi * tile_size
                q_end = min(q_start + tile_size, N)
                T_q = q_end - q_start

                o_acc = np.zeros((T_q, D), dtype=np.float64)
                m_acc = np.full(T_q, -np.inf, dtype=np.float64)
                l_acc = np.zeros(T_q, dtype=np.float64)

                Q_tile = Q[b, h, q_start:q_end].astype(np.float64)

                for ki in range(n_tiles_kv):
                    k_start = ki * tile_size
                    k_end = min(k_start + tile_size, N)

                    if causal:
                        if k_start > q_end - 1:
                            break

                    K_tile = K[b, h, k_start:k_end].astype(np.float64)
                    V_tile = V[b, h, k_start:k_end].astype(np.float64)

                    S = (Q_tile @ K_tile.T) * scale

                    if causal:
                        row_idx = np.arange(T_q)[:, None] + q_start
                        col_idx = np.arange(k_end - k_start)[None, :] + k_start
                        causal_mask = np.where(col_idx > row_idx, -np.inf, 0.0)
                        S = S + causal_mask

                    m_new = np.maximum(m_acc, S.max(axis=-1))
                    alpha = np.exp(m_acc - m_new)
                    P = np.exp(S - m_new[:, None])

                    l_new = l_acc * alpha + P.sum(axis=-1)

                    o_acc = o_acc * alpha[:, None]
                    o_acc = o_acc + P @ V_tile

                    m_acc = m_new
                    l_acc = l_new

                O[b, h, q_start:q_end] = o_acc / l_acc[:, None]
                L[b, h, q_start:q_end] = np.where(
                    l_acc > 0,
                    m_acc + np.log(l_acc),
                    m_acc
                )

    cache = {'O': O, 'L': L, 'Q': Q, 'K': K, 'V': V}
    return O, cache


def flash_attention_bwd(dO, cache, tile_size, causal=True):
    Q = cache['Q']
    K = cache['K']
    V = cache['V']
    O = cache['O']
    L = cache['L']

    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)

    D_diag = (dO.astype(np.float64) * O.astype(np.float64)).sum(axis=-1)

    dQ = np.zeros_like(Q, dtype=np.float64)
    dK = np.zeros_like(K, dtype=np.float64)
    dV = np.zeros_like(V, dtype=np.float64)

    n_tiles_q = (N + tile_size - 1) // tile_size
    n_tiles_kv = (N + tile_size - 1) // tile_size

    for b in range(B):
        for h in range(H):
            for qi in range(n_tiles_q):
                q_start = qi * tile_size
                q_end = min(q_start + tile_size, N)
                T_q = q_end - q_start

                dQ_tile = np.zeros((T_q, D), dtype=np.float64)
                Q_tile = Q[b, h, q_start:q_end].astype(np.float64)
                dO_tile = dO[b, h, q_start:q_end].astype(np.float64)
                L_tile = L[b, h, q_start:q_end].astype(np.float64)
                D_tile = D_diag[b, h, q_start:q_end].astype(np.float64)

                for ki in range(n_tiles_kv):
                    k_start = ki * tile_size
                    k_end = min(k_start + tile_size, N)
                    T_kv = k_end - k_start

                    if causal:
                        if k_start > q_end - 1:
                            break

                    K_tile = K[b, h, k_start:k_end].astype(np.float64)
                    V_tile = V[b, h, k_start:k_end].astype(np.float64)

                    S = (Q_tile @ K_tile.T) * scale

                    if causal:
                        row_idx = np.arange(T_q)[:, None] + q_start
                        col_idx = np.arange(T_kv)[None, :] + k_start
                        causal_mask = np.where(col_idx > row_idx, -np.inf, 0.0)
                        S = S + causal_mask

                    P = np.exp(S - L_tile[:, None])

                    dV_tile = P.T @ dO_tile
                    dV[b, h, k_start:k_end] += dV_tile

                    dP = dO_tile @ V_tile.T

                    dS = P * (dP - D_tile[:, None])

                    dQ_tile += dS @ K_tile * scale

                    dK_tile = dS.T @ Q_tile * scale
                    dK[b, h, k_start:k_end] += dK_tile

                dQ[b, h, q_start:q_end] = dQ_tile

    return dQ, dK, dV


def naive_attention_fwd(Q, K, V, causal=True):
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    S = np.einsum('bhid,bhjd->bhij', Q, K) * scale

    if causal:
        causal_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        S = np.where(causal_mask[None, None, :, :], -np.inf, S)

    rowmax = S.max(axis=-1, keepdims=True)
    exp_S = np.exp(S - rowmax)
    rowsum = exp_S.sum(axis=-1, keepdims=True)
    P = exp_S / rowsum
    L = rowmax.squeeze(-1) + np.log(rowsum.squeeze(-1))
    O = np.einsum('bhij,bhjd->bhid', P, V)
    return O, P, L


def naive_attention_bwd(dO, Q, K, V, O, P, causal=True):
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)

    dV = np.einsum('bhij,bhid->bhjd', P, dO)
    dP = np.einsum('bhid,bhjd->bhij', dO, V)
    rowsum_PdP = (P * dP).sum(axis=-1, keepdims=True)
    dS = P * (dP - rowsum_PdP)

    if causal:
        causal_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        dS = np.where(causal_mask[None, None, :, :], 0.0, dS)

    dQ = np.einsum('bhij,bhjd->bhid', dS, K) * scale
    dK = np.einsum('bhij,bhid->bhjd', dS, Q) * scale
    return dQ, dK, dV


def finite_diff_V(dO, Q, K, V, causal, eps=1e-5):
    B, H, N, D = V.shape
    dV_fd = np.zeros_like(V, dtype=np.float64)
    O_fwd, _ = flash_attention_fwd(Q, K, V, 16, causal=causal)
    loss_grad = np.sum(O_fwd * dO)
    for b in range(B):
        for h in range(H):
            for i in range(N):
                for d in range(D):
                    V_plus = V.copy()
                    V_plus[b, h, i, d] += eps
                    O_plus, _ = flash_attention_fwd(Q, K, V_plus, 16, causal=causal)
                    loss_plus = np.sum(O_plus * dO)

                    V_minus = V.copy()
                    V_minus[b, h, i, d] -= eps
                    O_minus, _ = flash_attention_fwd(Q, K, V_minus, 16, causal=causal)
                    loss_minus = np.sum(O_minus * dO)

                    dV_fd[b, h, i, d] = (loss_plus - loss_minus) / (2 * eps)
    return dV_fd


def test_gradient_check():
    print("=" * 60)
    print("Test 1: Gradient check (finite differences)")
    print("=" * 60)

    np.random.seed(42)
    B, H, N, D, T = 1, 1, 64, 32, 16
    Q = np.random.randn(B, H, N, D).astype(np.float64)
    K = np.random.randn(B, H, N, D).astype(np.float64)
    V = np.random.randn(B, H, N, D).astype(np.float64)
    dO = np.random.randn(B, H, N, D).astype(np.float64)
    causal = True

    O, cache = flash_attention_fwd(Q, K, V, T, causal=causal)
    dQ, dK, dV = flash_attention_bwd(dO, cache, T, causal=causal)

    dV_fd = finite_diff_V(dO, Q, K, V, causal, eps=1e-6)

    rel_err_dV = np.max(np.abs(dV - dV_fd) / (np.abs(dV_fd) + 1e-10))
    print(f"  dV relative error: {rel_err_dV:.2e}")
    assert rel_err_dV < 1e-5, f"dV relative error {rel_err_dV} >= 1e-5"

    rng = np.random.RandomState(123)
    spot_indices = rng.choice(N, size=10, replace=False)
    spot_dims = rng.choice(D, size=10, replace=False)

    for idx in range(10):
        i = spot_indices[idx]
        d = spot_dims[idx]
        for b in range(B):
            for hh in range(H):
                V_plus = V.copy()
                V_plus[b, hh, i, d] += 1e-6
                O_plus, _ = flash_attention_fwd(Q, K, V_plus, T, causal=causal)
                loss_plus = np.sum(O_plus * dO)

                V_minus = V.copy()
                V_minus[b, hh, i, d] -= 1e-6
                O_minus, _ = flash_attention_fwd(Q, K, V_minus, T, causal=causal)
                loss_minus = np.sum(O_minus * dO)

                fd = (loss_plus - loss_minus) / 2e-6

    print("  dV check passed!")

    dQ_fd = np.zeros_like(Q, dtype=np.float64)
    dK_fd = np.zeros_like(K, dtype=np.float64)

    for idx in range(10):
        b_idx = 0
        h_idx = 0
        i = spot_indices[idx]
        d = spot_dims[idx]

        Q_plus = Q.copy()
        Q_plus[b_idx, h_idx, i, d] += 1e-6
        O_plus, _ = flash_attention_fwd(Q_plus, K, V, T, causal=causal)
        loss_plus = np.sum(O_plus * dO)

        Q_minus = Q.copy()
        Q_minus[b_idx, h_idx, i, d] -= 1e-6
        O_minus, _ = flash_attention_fwd(Q_minus, K, V, T, causal=causal)
        loss_minus = np.sum(O_minus * dO)
        dQ_fd[b_idx, h_idx, i, d] = (loss_plus - loss_minus) / 2e-6

        K_plus = K.copy()
        K_plus[b_idx, h_idx, i, d] += 1e-6
        O_plus, _ = flash_attention_fwd(Q, K_plus, V, T, causal=causal)
        loss_plus = np.sum(O_plus * dO)

        K_minus = K.copy()
        K_minus[b_idx, h_idx, i, d] -= 1e-6
        O_minus, _ = flash_attention_fwd(Q, K_minus, V, T, causal=causal)
        loss_minus = np.sum(O_minus * dO)
        dK_fd[b_idx, h_idx, i, d] = (loss_plus - loss_minus) / 2e-6

    mask_q = np.zeros_like(dQ, dtype=bool)
    mask_k = np.zeros_like(dK, dtype=bool)
    for idx in range(10):
        i = spot_indices[idx]
        d = spot_dims[idx]
        mask_q[0, 0, i, d] = True
        mask_k[0, 0, i, d] = True

    dQ_err = np.abs((dQ - dQ_fd)[mask_q]) / (np.abs(dQ_fd[mask_q]) + 1e-10)
    dK_err = np.abs((dK - dK_fd)[mask_k]) / (np.abs(dK_fd[mask_k]) + 1e-10)

    print(f"  dQ spot-check relative error: {dQ_err.max():.2e}")
    print(f"  dK spot-check relative error: {dK_err.max():.2e}")
    assert dQ_err.max() < 1e-5, f"dQ spot-check error {dQ_err.max()} >= 1e-5"
    assert dK_err.max() < 1e-5, f"dK spot-check error {dK_err.max()} >= 1e-5"

    print("  Test 1 PASSED!\n")


def test_vs_naive():
    print("=" * 60)
    print("Test 2: vs naive backward")
    print("=" * 60)

    np.random.seed(123)
    B, H, N, D, T = 2, 4, 256, 64, 64
    Q = np.random.randn(B, H, N, D).astype(np.float64)
    K = np.random.randn(B, H, N, D).astype(np.float64)
    V = np.random.randn(B, H, N, D).astype(np.float64)
    dO = np.random.randn(B, H, N, D).astype(np.float64)
    causal = True

    O_naive, P_naive, L_naive = naive_attention_fwd(Q, K, V, causal=causal)
    dQ_naive, dK_naive, dV_naive = naive_attention_bwd(dO, Q, K, V, O_naive, P_naive, causal=causal)

    O_flash, cache = flash_attention_fwd(Q, K, V, T, causal=causal)
    dQ_flash, dK_flash, dV_flash = flash_attention_bwd(dO, cache, T, causal=causal)

    fwd_err = np.max(np.abs(O_flash - O_naive) / (np.abs(O_naive) + 1e-10))
    print(f"  Forward relative error: {fwd_err:.2e}")

    dQ_rel = np.max(np.abs(dQ_flash - dQ_naive) / (np.abs(dQ_naive) + 1e-10))
    dK_rel = np.max(np.abs(dK_flash - dK_naive) / (np.abs(dK_naive) + 1e-10))
    dV_rel = np.max(np.abs(dV_flash - dV_naive) / (np.abs(dV_naive) + 1e-10))

    print(f"  dQ relative error: {dQ_rel:.2e}")
    print(f"  dK relative error: {dK_rel:.2e}")
    print(f"  dV relative error: {dV_rel:.2e}")

    assert dQ_rel < 1e-4, f"dQ error {dQ_rel} >= 1e-4"
    assert dK_rel < 1e-4, f"dK error {dK_rel} >= 1e-4"
    assert dV_rel < 1e-4, f"dV error {dV_rel} >= 1e-4"

    print("  Test 2 PASSED!\n")


def test_memory():
    print("=" * 60)
    print("Test 3: Memory test")
    print("=" * 60)

    B, H, N, D, T = 1, 1, 4096, 64, 128

    full_matrix_bytes = N * N * 8

    Q = np.random.randn(B, H, N, D).astype(np.float64)
    K = np.random.randn(B, H, N, D).astype(np.float64)
    V = np.random.randn(B, H, N, D).astype(np.float64)
    dO = np.random.randn(B, H, N, D).astype(np.float64)

    tracemalloc.start()
    O, cache = flash_attention_fwd(Q, K, V, T, causal=True)
    dQ, dK, dV = flash_attention_bwd(dO, cache, T, causal=True)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    full_mb = full_matrix_bytes / (1024 * 1024)
    ratio = peak / full_matrix_bytes

    print(f"  Peak memory: {peak_mb:.2f} MB")
    print(f"  Single (N,N) matrix: {full_mb:.2f} MB")
    print(f"  Ratio: {ratio:.2%}")
    assert ratio < 0.20, f"Peak memory ratio {ratio:.2%} >= 20%"

    print("  Test 3 PASSED!\n")


if __name__ == '__main__':
    test_gradient_check()
    test_vs_naive()
    test_memory()
    print("All tests passed!")