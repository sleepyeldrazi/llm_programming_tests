"""
Tiled (Flash) attention forward pass with online softmax, in NumPy.

The core idea: instead of materializing the full (N, N) score matrix and
softmaxing it row-wise, we stream over K/V in tiles and maintain per-row
running statistics (max m, exp-sum l, weighted output O) so we can fuse
the softmax with the matmul against V. Memory drops from O(N^2) to
O(N * D + B * H * T * T) where T is the tile size.

Online softmax rescaling derivation
------------------------------------
For a single row, true softmax-attention is:
    o = sum_j  exp(s_j - m*) / Z  *  v_j
where m* = max_j s_j and Z = sum_j exp(s_j - m*).

If we have only seen scores s_1..s_k so far with running max m_old and have
accumulated  O_old = sum_{j<=k} exp(s_j - m_old) * v_j   and
             l_old = sum_{j<=k} exp(s_j - m_old),
then on seeing more scores s_{k+1}..s_{k+t} with local max r, we update:
    m_new = max(m_old, r)
    O_new = exp(m_old - m_new) * O_old + sum_{j>k} exp(s_j - m_new) * v_j
    l_new = exp(m_old - m_new) * l_old + sum_{j>k} exp(s_j - m_new)
The rescale factor is exp(m_old - m_new), NOT exp(m_new - m_old):
m_new >= m_old, so m_old - m_new <= 0 and the factor lies in (0, 1].
Using exp(m_new - m_old) would blow up to >= 1 and produce overflow, since
the existing partial sums were already normalized against the smaller m_old
and re-normalizing to the larger m_new requires *shrinking* them.

After all tiles we divide once: o = O / l. Algebraically this equals the
standard softmax answer because both numerator and denominator have been
rescaled by the same factor at every step.

Causal-masking + tiling hazard
------------------------------
At the start of a Q tile, the running state is m = -inf, l = 0, O = 0.
If the very first KV tile that touches this Q tile happens to be fully
masked for some row (every key index j satisfies j > i for that row), the
local row-max would be -inf. Then:
    m_new = max(-inf, -inf) = -inf
    correction = exp(m_old - m_new) = exp(-inf - (-inf)) = exp(NaN) = NaN
which poisons O and l forever. The cure is to *skip* any (Q_tile, KV_tile)
block that is entirely above the diagonal — those have no valid entries
for any row, so processing them serves no purpose and produces NaNs.
For partially-masked tiles, individual rows whose entries are all masked
are safe as long as m_old is already finite: row_max = -inf gives
m_new = m_old, correction = 1, P = 0 — a no-op update.
Because we always process KV tiles left-to-right and skip those with
kv_start >= q_end, the first non-skipped KV tile for a Q tile always
contains the diagonal entry for at least one row (in particular, the row
i = q_start sees key j = q_start), so m for that Q tile becomes finite
on its first update. Subsequent fully-masked rows within partial tiles
are then fine.
"""

import numpy as np


def flash_attention_fwd(Q, K, V, tile_size, causal=True):
    """Tiled attention with online softmax.

    Q, K, V: arrays of shape (B, H, N, D).
    Returns O of shape (B, H, N, D).
    """
    B, H, N, D = Q.shape
    assert K.shape == (B, H, N, D)
    assert V.shape == (B, H, N, D)

    scale = np.float64(1.0) / np.sqrt(np.float64(D))
    out = np.empty_like(Q)
    T = tile_size

    for q_start in range(0, N, T):
        q_end = min(q_start + T, N)
        Tq = q_end - q_start
        Q_tile = Q[:, :, q_start:q_end, :]  # (B, H, Tq, D)

        # Per-row online state for this Q tile.
        m = np.full((B, H, Tq, 1), -np.inf, dtype=Q.dtype)
        l = np.zeros((B, H, Tq, 1), dtype=Q.dtype)
        O_tile = np.zeros((B, H, Tq, D), dtype=Q.dtype)

        q_idx = np.arange(q_start, q_end)[:, None]  # (Tq, 1) absolute row indices

        for kv_start in range(0, N, T):
            kv_end = min(kv_start + T, N)

            # Skip tiles entirely above the diagonal: every (i, j) in this
            # block has j >= kv_start > i (since i < q_end <= kv_start).
            # See the "causal-masking + tiling hazard" docstring above.
            if causal and kv_start >= q_end:
                break  # remaining tiles are even further right

            K_tile = K[:, :, kv_start:kv_end, :]
            V_tile = V[:, :, kv_start:kv_end, :]

            # Local scores S = (Q_tile @ K_tile^T) / sqrt(D).
            # Shape: (B, H, Tq, Tk). This is the only "big" intermediate
            # and it is bounded by T*T per (B, H), never N*N.
            S = np.matmul(Q_tile, np.swapaxes(K_tile, -2, -1)) * scale

            if causal and kv_end > q_start:
                # Only the diagonal-straddling tile needs a per-element mask.
                k_idx = np.arange(kv_start, kv_end)[None, :]  # (1, Tk)
                mask = k_idx > q_idx  # (Tq, Tk), True where j > i
                if mask.any():
                    S = np.where(mask, -np.inf, S)

            # Online softmax update.
            row_max = S.max(axis=-1, keepdims=True)  # (B, H, Tq, 1)
            m_new = np.maximum(m, row_max)
            # exp(m - m_new) is in (0, 1]; rescales the partial sums from
            # being normalized against m_old to being normalized against m_new.
            correction = np.exp(m - m_new)
            P = np.exp(S - m_new)  # stable: S - m_new <= 0 entrywise
            l = l * correction + P.sum(axis=-1, keepdims=True)
            O_tile = O_tile * correction + np.matmul(P, V_tile)
            m = m_new

        out[:, :, q_start:q_end, :] = O_tile / l

    return out


def naive_attention(Q, K, V, causal=True):
    """Reference: full-softmax attention. Materializes the (N, N) score matrix."""
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    S = np.matmul(Q, np.swapaxes(K, -2, -1)) * scale  # (B, H, N, N)
    if causal:
        i = np.arange(N)[:, None]
        j = np.arange(N)[None, :]
        S = np.where(j > i, -np.inf, S)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(axis=-1, keepdims=True)
    return np.matmul(P, V)
