import numpy as np


def flash_attention_fwd(Q, K, V, tile_size, causal=True):
    """Tiled forward pass with online softmax. Stores only O, L, Q, K, V."""
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    T = tile_size

    O = np.zeros_like(Q)
    L = np.zeros((B, H, N), dtype=Q.dtype)

    for b in range(B):
        for h in range(H):
            Qbh, Kbh, Vbh = Q[b, h], K[b, h], V[b, h]

            for q_start in range(0, N, T):
                q_end = min(q_start + T, N)
                Tq = q_end - q_start
                Q_tile = Qbh[q_start:q_end]

                m_i = np.full((Tq,), -np.inf, dtype=Q.dtype)
                l_i = np.zeros((Tq,), dtype=Q.dtype)
                O_i = np.zeros((Tq, D), dtype=Q.dtype)

                for k_start in range(0, N, T):
                    k_end = min(k_start + T, N)

                    if causal and k_start >= q_end:
                        continue

                    K_tile = Kbh[k_start:k_end]
                    V_tile = Vbh[k_start:k_end]

                    S = (Q_tile @ K_tile.T) * scale

                    if causal and k_end > q_start + 1:
                        q_idx = np.arange(q_start, q_end)[:, None]
                        k_idx = np.arange(k_start, k_end)[None, :]
                        mask = k_idx > q_idx
                        if mask.any():
                            S = np.where(mask, -np.inf, S)

                    m_block = S.max(axis=-1)
                    m_new = np.maximum(m_i, m_block)

                    with np.errstate(invalid='ignore'):
                        m_new_safe = np.where(np.isneginf(m_new), 0.0, m_new)
                        P = np.exp(S - m_new_safe[:, None])
                        m_i_safe = np.where(np.isneginf(m_i), 0.0, m_i)
                        alpha = np.where(
                            np.isneginf(m_i), 0.0,
                            np.exp(m_i_safe - m_new_safe),
                        )

                    l_i = alpha * l_i + P.sum(axis=-1)
                    O_i = alpha[:, None] * O_i + P @ V_tile
                    m_i = m_new

                O[b, h, q_start:q_end] = O_i / l_i[:, None]
                L[b, h, q_start:q_end] = m_i + np.log(l_i)

    cache = {'O': O, 'L': L, 'Q': Q, 'K': K, 'V': V}
    return O, cache


def flash_attention_bwd(dO, cache, tile_size, causal=True):
    """Tiled backward pass with on-the-fly softmax recomputation from L."""
    Q, K, V, O, L = cache['Q'], cache['K'], cache['V'], cache['O'], cache['L']
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    T = tile_size

    dQ = np.zeros_like(Q)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)

    # Precompute D_i = sum_d O[i,d] * dO[i,d] — equals sum_k P[i,k] * dP[i,k]
    # over ALL keys, which is what the softmax-gradient row sum requires.
    D_row = (O * dO).sum(axis=-1)  # (B, H, N)

    for b in range(B):
        for h in range(H):
            Qbh, Kbh, Vbh = Q[b, h], K[b, h], V[b, h]
            Lbh = L[b, h]
            dObh = dO[b, h]
            Dbh = D_row[b, h]

            for q_start in range(0, N, T):
                q_end = min(q_start + T, N)
                Q_tile = Qbh[q_start:q_end]
                dO_tile = dObh[q_start:q_end]
                L_q = Lbh[q_start:q_end]
                D_q = Dbh[q_start:q_end]

                dQ_tile = np.zeros_like(Q_tile)

                for k_start in range(0, N, T):
                    k_end = min(k_start + T, N)

                    if causal and k_start >= q_end:
                        continue

                    K_tile = Kbh[k_start:k_end]
                    V_tile = Vbh[k_start:k_end]

                    S = (Q_tile @ K_tile.T) * scale

                    if causal and k_end > q_start + 1:
                        q_idx = np.arange(q_start, q_end)[:, None]
                        k_idx = np.arange(k_start, k_end)[None, :]
                        mask = k_idx > q_idx
                        if mask.any():
                            S = np.where(mask, -np.inf, S)

                    with np.errstate(invalid='ignore'):
                        P = np.exp(S - L_q[:, None])

                    dV[b, h, k_start:k_end] += P.T @ dO_tile

                    dP = dO_tile @ V_tile.T
                    dS = P * (dP - D_q[:, None])

                    dQ_tile += (dS @ K_tile) * scale
                    dK[b, h, k_start:k_end] += (dS.T @ Q_tile) * scale

                dQ[b, h, q_start:q_end] = dQ_tile

    return dQ, dK, dV


# ----- Naive reference implementations for testing -----

def naive_forward(Q, K, V, causal=True):
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    S = np.einsum('bhid,bhjd->bhij', Q, K) * scale
    if causal:
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        S = np.where(mask[None, None], -np.inf, S)
    S_max = S.max(axis=-1, keepdims=True)
    P = np.exp(S - S_max)
    P = P / P.sum(axis=-1, keepdims=True)
    O = np.einsum('bhij,bhjd->bhid', P, V)
    return O, P


def naive_backward(Q, K, V, dO, P):
    _, _, _, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    dV = np.einsum('bhij,bhid->bhjd', P, dO)
    dP = np.einsum('bhid,bhjd->bhij', dO, V)
    rowsum = (P * dP).sum(axis=-1, keepdims=True)
    dS = P * (dP - rowsum)
    dQ = np.einsum('bhij,bhjd->bhid', dS, K) * scale
    dK = np.einsum('bhij,bhid->bhjd', dS, Q) * scale
    return dQ, dK, dV
