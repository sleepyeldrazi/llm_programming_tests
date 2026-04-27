import numpy as np


def flash_attention_fwd(Q, K, V, tile_size, causal=True):
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    dtype = Q.dtype
    O = np.zeros((B, H, N, D), dtype=np.float64)
    m = np.full((B, H, N), -np.inf, dtype=np.float64)
    l = np.zeros((B, H, N), dtype=np.float64)

    for b in range(B):
        for h in range(H):
            Q_bh = Q[b, h].astype(np.float64)
            K_bh = K[b, h].astype(np.float64)
            V_bh = V[b, h].astype(np.float64)

            for i in range(0, N, tile_size):
                i_end = min(i + tile_size, N)
                T_q = i_end - i
                q_tile = Q_bh[i:i_end]

                m_row = np.full(T_q, -np.inf, dtype=np.float64)
                l_row = np.zeros(T_q, dtype=np.float64)
                o_acc = np.zeros((T_q, D), dtype=np.float64)

                for j in range(0, N, tile_size):
                    j_end = min(j + tile_size, N)

                    if causal and j >= i_end:
                        continue

                    k_tile = K_bh[j:j_end]
                    v_tile = V_bh[j:j_end]

                    S = (q_tile @ k_tile.T) * scale

                    if causal:
                        causal_mask = np.arange(j, j_end)[None, :] > np.arange(i, i_end)[:, None]
                        S = np.where(causal_mask, -np.inf, S)

                    m_new = np.maximum(m_row, S.max(axis=-1))
                    rescale = np.exp(m_row - m_new)
                    P = np.exp(S - m_new[:, None])

                    if causal:
                        P = np.where(causal_mask, 0.0, P)

                    l_new = rescale * l_row + P.sum(axis=-1)
                    o_acc = rescale[:, None] * o_acc + P @ v_tile

                    m_row = m_new
                    l_row = l_new

                o_acc = o_acc / l_row[:, None]
                O[b, h, i:i_end] = o_acc
                m[b, h, i:i_end] = m_row
                l[b, h, i:i_end] = l_row

    L = m + np.log(l)
    O_out = O.astype(dtype)
    cache = {'O': O_out, 'L': L, 'Q': Q, 'K': K, 'V': V}
    return O_out, cache


def flash_attention_bwd(dO, cache, tile_size, causal=True):
    Q = cache['Q']
    K = cache['K']
    V = cache['V']
    O = cache['O']
    L = cache['L']

    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    dtype = Q.dtype

    dQ = np.zeros((B, H, N, D), dtype=np.float64)
    dK = np.zeros((B, H, N, D), dtype=np.float64)
    dV = np.zeros((B, H, N, D), dtype=np.float64)

    for b in range(B):
        for h in range(H):
            for i in range(0, N, tile_size):
                i_end = min(i + tile_size, N)
                T_q = i_end - i

                q_tile = Q[b, h, i:i_end].astype(np.float64)
                do_tile = dO[b, h, i:i_end].astype(np.float64)
                o_tile = O[b, h, i:i_end].astype(np.float64)
                l_tile = L[b, h, i:i_end].astype(np.float64)

                Di = (do_tile * o_tile).sum(axis=-1, keepdims=True)

                dq_tile = np.zeros((T_q, D), dtype=np.float64)

                for j in range(0, N, tile_size):
                    j_end = min(j + tile_size, N)
                    T_kv = j_end - j

                    if causal and j >= i_end:
                        continue

                    k_tile = K[b, h, j:j_end].astype(np.float64)
                    v_tile = V[b, h, j:j_end].astype(np.float64)

                    S = (q_tile @ k_tile.T) * scale

                    if causal:
                        causal_mask = np.arange(j, j_end)[None, :] > np.arange(i, i_end)[:, None]
                        S = np.where(causal_mask, -np.inf, S)

                    P = np.exp(S - l_tile[:, None])

                    if causal:
                        P = np.where(causal_mask, 0.0, P)

                    dV[b, h, j:j_end] += P.T @ do_tile

                    dP = do_tile @ v_tile.T

                    dS = P * (dP - Di)

                    if causal:
                        dS = np.where(causal_mask, 0.0, dS)

                    dq_tile += dS @ k_tile * scale
                    dK[b, h, j:j_end] += dS.T @ q_tile * scale

                dQ[b, h, i:i_end] = dq_tile

    return dQ.astype(dtype), dK.astype(dtype), dV.astype(dtype)


def naive_attention_fwd(Q, K, V, causal=True):
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    O = np.zeros((B, H, N, D), dtype=np.float64)

    for b in range(B):
        for h in range(H):
            S = (Q[b, h].astype(np.float64) @ K[b, h].astype(np.float64).T) * scale
            if causal:
                causal_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                S = np.where(causal_mask, -np.inf, S)
            S_max = S.max(axis=-1, keepdims=True)
            P = np.exp(S - S_max)
            P = P / P.sum(axis=-1, keepdims=True)
            if causal:
                P = np.where(causal_mask, 0.0, P)
            O[b, h] = P @ V[b, h].astype(np.float64)

    return O.astype(Q.dtype)


def naive_attention_bwd(Q, K, V, dO, causal=True):
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    Q64 = Q.astype(np.float64)
    K64 = K.astype(np.float64)
    V64 = V.astype(np.float64)
    dO64 = dO.astype(np.float64)
    O64 = naive_attention_fwd(Q, K, V, causal=causal).astype(np.float64)

    dQ = np.zeros((B, H, N, D), dtype=np.float64)
    dK = np.zeros((B, H, N, D), dtype=np.float64)
    dV = np.zeros((B, H, N, D), dtype=np.float64)

    for b in range(B):
        for h in range(H):
            S = (Q64[b, h] @ K64[b, h].T) * scale
            if causal:
                causal_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                S = np.where(causal_mask, -np.inf, S)
            S_max = S.max(axis=-1, keepdims=True)
            P = np.exp(S - S_max)
            P = P / P.sum(axis=-1, keepdims=True)
            if causal:
                P = np.where(causal_mask, 0.0, P)

            dV[b, h] = P.T @ dO64[b, h]
            dP = dO64[b, h] @ V64[b, h].T
            Di = (dO64[b, h] * O64[b, h]).sum(axis=-1, keepdims=True)
            dS = P * (dP - Di)
            if causal:
                dS = np.where(causal_mask, 0.0, dS)
            dQ[b, h] = dS @ K64[b, h] * scale
            dK[b, h] = dS.T @ Q64[b, h] * scale

    return dQ.astype(Q.dtype), dK.astype(Q.dtype), dV.astype(Q.dtype)


if __name__ == '__main__':
    np.random.seed(42)

    # Quick forward + backward sanity check (small)
    print("=== Forward/backward sanity check (small) ===")
    B_c, H_c, N_c, D_c, T_c = 1, 1, 8, 4, 4
    Qc = np.random.randn(B_c, H_c, N_c, D_c)
    Kc = np.random.randn(B_c, H_c, N_c, D_c)
    Vc = np.random.randn(B_c, H_c, N_c, D_c)
    Of, cf = flash_attention_fwd(Qc, Kc, Vc, T_c, causal=True)
    On = naive_attention_fwd(Qc, Kc, Vc, causal=True)
    fwd_err = np.max(np.abs(Of - On) / (np.abs(On) + 1e-8))
    print(f"Forward max relative error: {fwd_err:.2e}")

    dOc = np.random.randn(*Of.shape)
    dQf, dKf, dVf = flash_attention_bwd(dOc, cf, T_c, causal=True)
    dQn, dKn, dVn = naive_attention_bwd(Qc, Kc, Vc, dOc, causal=True)
    for name, fg, ng in [('dQ', dQf, dQn), ('dK', dKf, dKn), ('dV', dVf, dVn)]:
        rel = np.max(np.abs(fg - ng) / (np.abs(ng) + 1e-8))
        print(f"  {name} max rel error vs naive: {rel:.2e}")
    print()

    # ── Test 1: Gradient check with finite differences ──
    print("Test 1: Gradient check (B=1, H=1, N=64, D=32, T=16, causal=True)")
    B1, H1, N1, D1, T1 = 1, 1, 64, 32, 16
    Q1 = np.random.randn(B1, H1, N1, D1)
    K1 = np.random.randn(B1, H1, N1, D1)
    V1 = np.random.randn(B1, H1, N1, D1)

    O1, cache1 = flash_attention_fwd(Q1, K1, V1, T1, causal=True)
    dO1 = np.random.randn(*O1.shape)
    dQ1, dK1, dV1 = flash_attention_bwd(dO1, cache1, T1, causal=True)

    dQ1n, dK1n, dV1n = naive_attention_bwd(Q1, K1, V1, dO1, causal=True)
    for name, fg, ng in [('dQ', dQ1, dQ1n), ('dK', dK1, dK1n), ('dV', dV1, dV1n)]:
        rel = np.max(np.abs(fg.astype(np.float64) - ng.astype(np.float64)) / (np.abs(ng.astype(np.float64)) + 1e-8))
        print(f"  {name} flash vs naive: {rel:.2e}")

    eps = 1e-5
    print("  Computing dV via finite differences...")
    dV_fd = np.zeros_like(V1, dtype=np.float64)
    for idx in np.ndindex(V1.shape):
        V_plus = V1.copy(); V_plus[idx] += eps
        V_minus = V1.copy(); V_minus[idx] -= eps
        O_plus, _ = flash_attention_fwd(Q1, K1, V_plus, T1, causal=True)
        O_minus, _ = flash_attention_fwd(Q1, K1, V_minus, T1, causal=True)
        dV_fd[idx] = ((O_plus - O_minus) * dO1).sum() / (2 * eps)

    rel_err_dV = np.max(np.abs(dV1.astype(np.float64) - dV_fd) / (np.abs(dV_fd) + 1e-8))
    print(f"  dV max relative error vs finite differences: {rel_err_dV:.2e}")
    assert rel_err_dV < 1e-5, f"dV relative error {rel_err_dV} >= 1e-5"

    print("  Computing dQ via finite differences (spot-check)...")
    rand_idx_Q = np.random.choice(N1 * D1, size=10, replace=False)
    dQ_fd = np.zeros_like(Q1, dtype=np.float64)
    for flat_idx in rand_idx_Q:
        idx = np.unravel_index(flat_idx, Q1.shape)
        Q_plus = Q1.copy(); Q_plus[idx] += eps
        Q_minus = Q1.copy(); Q_minus[idx] -= eps
        O_p, _ = flash_attention_fwd(Q_plus, K1, V1, T1, causal=True)
        O_m, _ = flash_attention_fwd(Q_minus, K1, V1, T1, causal=True)
        dQ_fd[idx] = ((O_p - O_m) * dO1).sum() / (2 * eps)
    rel_err_dQ = np.max(np.abs(dQ1.astype(np.float64).ravel()[rand_idx_Q] - dQ_fd.ravel()[rand_idx_Q]) /
                         (np.abs(dQ_fd.ravel()[rand_idx_Q]) + 1e-8))
    print(f"  dQ spot-check relative error: {rel_err_dQ:.2e}")
    assert rel_err_dQ < 1e-5, f"dQ relative error {rel_err_dQ} >= 1e-5"

    print("  Computing dK via finite differences (spot-check)...")
    rand_idx_K = np.random.choice(N1 * D1, size=10, replace=False)
    dK_fd = np.zeros_like(K1, dtype=np.float64)
    for flat_idx in rand_idx_K:
        idx = np.unravel_index(flat_idx, K1.shape)
        K_plus = K1.copy(); K_plus[idx] += eps
        K_minus = K1.copy(); K_minus[idx] -= eps
        O_p, _ = flash_attention_fwd(Q1, K_plus, V1, T1, causal=True)
        O_m, _ = flash_attention_fwd(Q1, K_minus, V1, T1, causal=True)
        dK_fd[idx] = ((O_p - O_m) * dO1).sum() / (2 * eps)
    rel_err_dK = np.max(np.abs(dK1.astype(np.float64).ravel()[rand_idx_K] - dK_fd.ravel()[rand_idx_K]) /
                         (np.abs(dK_fd.ravel()[rand_idx_K]) + 1e-8))
    print(f"  dK spot-check relative error: {rel_err_dK:.2e}")
    assert rel_err_dK < 1e-5, f"dK relative error {rel_err_dK} >= 1e-5"

    print("  PASSED\n")

    # ── Test 2: Vs naive backward ──
    print("Test 2: Vs naive backward (B=2, H=4, N=256, D=64, T=64, causal=True)")
    B2, H2, N2, D2, T2 = 2, 4, 256, 64, 64
    Q2 = np.random.randn(B2, H2, N2, D2).astype(np.float32)
    K2 = np.random.randn(B2, H2, N2, D2).astype(np.float32)
    V2 = np.random.randn(B2, H2, N2, D2).astype(np.float32)

    O2, cache2 = flash_attention_fwd(Q2, K2, V2, T2, causal=True)
    O2_naive = naive_attention_fwd(Q2, K2, V2, causal=True)
    fwd_err = np.max(np.abs(O2 - O2_naive) / (np.abs(O2_naive) + 1e-6))
    print(f"  Forward max relative error: {fwd_err:.2e}")

    dO2 = np.random.randn(*O2.shape).astype(np.float32)
    dQ2, dK2, dV2 = flash_attention_bwd(dO2, cache2, T2, causal=True)
    dQ2n, dK2n, dV2n = naive_attention_bwd(Q2, K2, V2, dO2, causal=True)

    for name, flash_grad, naive_grad in [('dQ', dQ2, dQ2n), ('dK', dK2, dK2n), ('dV', dV2, dV2n)]:
        rel = np.max(np.abs(flash_grad.astype(np.float64) - naive_grad.astype(np.float64)) /
                     (np.abs(naive_grad.astype(np.float64)) + 1e-6))
        print(f"  {name} max relative error vs naive: {rel:.2e}")
        assert rel < 1e-4, f"{name} relative error {rel} >= 1e-4"

    print("  PASSED\n")

    # ── Test 3: Memory ──
    print("Test 3: Memory (B=1, H=1, N=4096, D=64, T=128, causal=True)")
    import tracemalloc
    import gc

    B3, H3, N3, D3, T3 = 1, 1, 4096, 64, 128
    Q3 = np.random.randn(B3, H3, N3, D3).astype(np.float32)
    K3 = np.random.randn(B3, H3, N3, D3).astype(np.float32)
    V3 = np.random.randn(B3, H3, N3, D3).astype(np.float32)

    gc.collect()
    tracemalloc.start()
    O3, cache3 = flash_attention_fwd(Q3, K3, V3, T3, causal=True)
    dO3 = np.ones_like(O3)
    dQ3, dK3, dV3 = flash_attention_bwd(dO3, cache3, T3, causal=True)
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    full_attn_mem = N3 * N3 * 4
    ratio = peak / full_attn_mem
    print(f"  Current memory: {current / 1e6:.2f} MB")
    print(f"  Peak memory: {peak / 1e6:.2f} MB")
    print(f"  Full (N,N) matrix would be: {full_attn_mem / 1e6:.2f} MB")
    print(f"  Ratio: {ratio:.2%}")
    assert ratio < 0.20, f"Memory ratio {ratio:.2%} >= 20%"
    print("  PASSED\n")

    print("All tests passed!")