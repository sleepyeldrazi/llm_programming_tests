import numpy as np
import tracemalloc


def flash_attention_fwd(Q, K, V, tile_size, causal=True):
    """
    Tiled forward pass with online softmax.

    Q, K, V: (B, H, N, D)
    Returns: (O, cache) where O is (B, H, N, D) and cache = {'O': O, 'L': L, 'Q': Q, 'K': K, 'V': V}
    L has shape (B, H, N)
    """
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    T = tile_size

    O = np.zeros((B, H, N, D), dtype=np.float64)
    L = np.zeros((B, H, N), dtype=np.float64)

    for b in range(B):
        for h in range(H):
            Q_bh = Q[b, h]  # (N, D)
            K_bh = K[b, h]  # (N, D)
            V_bh = V[b, h]  # (N, D)

            # Online softmax state per query row
            m = -np.inf * np.ones(N, dtype=np.float64)  # running max
            l = np.zeros(N, dtype=np.float64)            # running sum
            O_bh = np.zeros((N, D), dtype=np.float64)    # accumulated output

            n_q_tiles = (N + T - 1) // T
            n_kv_tiles = (N + T - 1) // T

            for ti in range(n_q_tiles):
                q_start = ti * T
                q_end = min(q_start + T, N)
                Q_tile = Q_bh[q_start:q_end]  # (T_q, D)

                for tj in range(n_kv_tiles):
                    kv_start = tj * T
                    kv_end = min(kv_start + T, N)
                    K_tile = K_bh[kv_start:kv_end]  # (T_kv, D)
                    V_tile = V_bh[kv_start:kv_end]  # (T_kv, D)

                    # Skip entirely masked blocks
                    if causal and kv_start >= q_end:
                        continue

                    S = Q_tile @ K_tile.T * scale  # (T_q, T_kv)

                    if causal:
                        q_pos = np.arange(q_start, q_end)[:, None]   # (T_q, 1)
                        k_pos = np.arange(kv_start, kv_end)[None, :]  # (1, T_kv)
                        mask = q_pos < k_pos
                        S = np.where(mask, -np.inf, S)

                    # Online softmax update
                    new_max = np.max(S, axis=-1, keepdims=True)  # (T_q, 1)
                    old_m = m[q_start:q_end, None]               # (T_q, 1)

                    # Handle rows where all values are -inf (fully masked by causal)
                    valid = np.isfinite(new_max)  # True where row has at least one finite value

                    # For valid rows: normal update. For invalid rows: no change.
                    alpha = np.where(valid, np.exp(old_m - new_max), 1.0)  # (T_q, 1)
                    exp_S = np.where(valid, np.exp(S - new_max), 0.0)       # (T_q, T_kv)
                    new_l = alpha * l[q_start:q_end, None] + np.sum(exp_S, axis=-1, keepdims=True)  # (T_q, 1)

                    # Update output
                    O_bh[q_start:q_end] = O_bh[q_start:q_end] * alpha + exp_S @ V_tile
                    m[q_start:q_end] = np.where(np.squeeze(valid, axis=-1),
                                                  np.squeeze(new_max, axis=-1),
                                                  m[q_start:q_end])
                    l[q_start:q_end] = np.squeeze(new_l, axis=-1)

                # End of KV tiles for this Q tile

             # Normalize accumulated output by l
            O[b, h] = O_bh / l[:, None]
            L[b, h] = m + np.log(l)

    cache = {'O': O, 'L': L, 'Q': Q, 'K': K, 'V': V}
    return O, cache


def flash_attention_bwd(dO, cache, tile_size, causal=True):
    """
    Tiled backward pass with online softmax recomputation.

    dO: (B, H, N, D)
    cache: {'O': O, 'L': L, 'Q': Q, 'K': K, 'V': V}
    Returns: (dQ, dK, dV) each (B, H, N, D)
    """
    B, H, N, D = dO.shape
    scale = 1.0 / np.sqrt(D)
    T = tile_size

    Q = cache['Q']
    K = cache['K']
    V = cache['V']
    L = cache['L']

    dQ = np.zeros((B, H, N, D), dtype=np.float64)
    dK = np.zeros((B, H, N, D), dtype=np.float64)
    dV = np.zeros((B, H, N, D), dtype=np.float64)

    n_q_tiles = (N + T - 1) // T
    n_kv_tiles = (N + T - 1) // T

    for b in range(B):
        for h in range(H):
            Q_bh = Q[b, h]
            K_bh = K[b, h]
            V_bh = V[b, h]
            L_bh = L[b, h]
            dO_bh = dO[b, h]

            dQ_bh = np.zeros((N, D), dtype=np.float64)
            dK_bh = np.zeros((N, D), dtype=np.float64)
            dV_bh = np.zeros((N, D), dtype=np.float64)

            for ti in range(n_q_tiles):
                q_start = ti * T
                q_end = min(q_start + T, N)
                Q_tile = Q_bh[q_start:q_end]
                L_query = L_bh[q_start:q_end]
                dO_tile = dO_bh[q_start:q_end]
                T_q = q_end - q_start

                # Collect list of valid KV tiles and their data
                kv_data = []  # list of (kv_start, kv_end, K_tile, V_tile, P, dP)
                for tj in range(n_kv_tiles):
                    kv_start = tj * T
                    kv_end = min(kv_start + T, N)
                    K_tile = K_bh[kv_start:kv_end]
                    V_tile = V_bh[kv_start:kv_end]

                    if causal:
                        q_max_pos = q_end - 1
                        k_min_pos = kv_start
                        if k_min_pos > q_max_pos:
                            continue

                    S = Q_tile @ K_tile.T * scale

                    if causal:
                        q_pos = np.arange(q_start, q_end)[:, None]
                        k_pos = np.arange(kv_start, kv_end)[None, :]
                        mask = q_pos < k_pos
                        S = S.copy()
                        S[mask] = -np.inf

                    P = np.exp(S - L_query[:, None])
                    dP = dO_tile @ V_tile.T
                    kv_data.append((kv_start, kv_end, K_tile, V_tile, P, dP))

                # Pass 1: accumulate rowsum_PdP over all KV tiles
                rowsum_PdP = np.zeros((T_q, 1), dtype=np.float64)
                for kv_start, kv_end, K_tile, V_tile, P, dP in kv_data:
                    rowsum_PdP += (P * dP).sum(axis=-1, keepdims=True)

                # Pass 2: compute dS, dQ, dK, dV with full rowsum
                dQ_tile_acc = np.zeros((T_q, D), dtype=np.float64)
                for kv_start, kv_end, K_tile, V_tile, P, dP in kv_data:
                    dV_bh[kv_start:kv_end] += P.T @ dO_tile
                    dS = P * (dP - rowsum_PdP)
                    dQ_tile_acc += dS @ K_tile * scale
                    dK_bh[kv_start:kv_end] += dS.T @ Q_tile * scale

                dQ_bh[q_start:q_end] += dQ_tile_acc

            dQ[b, h] = dQ_bh
            dK[b, h] = dK_bh
            dV[b, h] = dV_bh

    return dQ, dK, dV


def naive_attention_fwd(Q, K, V, causal=True):
    """Naive full-materialized attention for comparison."""
    B, H, N, D = Q.shape
    scale = 1.0 / np.sqrt(D)

    S = np.einsum('bhnd,bhmd->bhnm', Q, K) * scale  # (B, H, N, N)

    if causal:
        mask = np.arange(N)[:, None] < np.arange(N)[None, :]  # (N, N)
        S = S.copy()
        S[:, :, mask] = -np.inf

    # Softmax
    m = np.max(S, axis=-1)  # (B, H, N)
    exp_S = np.exp(S - m[..., None])  # (B, H, N, N)
    l = np.sum(exp_S, axis=-1)  # (B, H, N)
    P = exp_S / l[..., None]  # (B, H, N, N)

    O = np.einsum('bhnm,bhmd->bhnd', P, V)  # (B, H, N, D)
    L = m + np.log(l)  # (B, H, N)

    return O, P, L


def naive_attention_bwd(dO, Q, K, V, P, causal=True):
    """Naive backward with full attention matrix."""
    B, H, N, D = dO.shape
    scale = 1.0 / np.sqrt(D)

    # dV = P^T @ dO
    dV = np.einsum('bhnm,bhnd->bhmd', P, dO)

    # dP = dO @ V^T
    dP = np.einsum('bhnd,bhmd->bhnm', dO, V)

    # dS via softmax gradient
    rowsum_PdP = (P * dP).sum(axis=-1, keepdims=True)  # (B, H, N, 1)
    dS = P * (dP - rowsum_PdP)

    if causal:
        mask = np.arange(N)[:, None] < np.arange(N)[None, :]  # (N, N)
        dS = dS.copy()
        dS[:, :, mask] = 0.0

    # dQ = dS @ K * scale
    dQ = np.einsum('bhnm,bhmd->bhnd', dS, K) * scale

    # dK = dS^T @ Q * scale
    dK = np.einsum('bhnm,bhnd->bhmd', dS, Q) * scale

    return dQ, dK, dV


def finite_diff_dV(Q, K, V, dO, eps=1e-5, tile_size=16, causal=True):
    """Compute dV via central finite differences."""
    B, H, N, D = Q.shape
    dV_fd = np.zeros((B, H, N, D), dtype=np.float64)

    for b in range(B):
        for h in range(H):
            for n in range(N):
                for d in range(D):
                    V_plus = V.copy()
                    V_minus = V.copy()
                    V_plus[b, h, n, d] += eps
                    V_minus[b, h, n, d] -= eps

                    O_plus, _ = flash_attention_fwd(Q, K, V_plus, tile_size, causal)
                    O_minus, _ = flash_attention_fwd(Q, K, V_minus, tile_size, causal)

                    loss_plus = np.sum(O_plus * dO)
                    loss_minus = np.sum(O_minus * dO)

                    dV_fd[b, h, n, d] = (loss_plus - loss_minus) / (2 * eps)

    return dV_fd


def finite_diff_at_point(Q, K, V, dO, tensor, grad_name, idx, eps=1e-5, tile_size=16, causal=True):
    """Compute gradient at a single point via central finite differences."""
    B, H, N, D = Q.shape
    plus_val = tensor[idx] + eps
    minus_val = tensor[idx] - eps

    tensor_plus = tensor.copy()
    tensor_minus = tensor.copy()
    tensor_plus[idx] = plus_val
    tensor_minus[idx] = minus_val

    if grad_name == 'Q':
        O_plus, _ = flash_attention_fwd(tensor_plus, K, V, tile_size, causal)
        O_minus, _ = flash_attention_fwd(tensor_minus, K, V, tile_size, causal)
    elif grad_name == 'K':
        O_plus, _ = flash_attention_fwd(Q, tensor_plus, V, tile_size, causal)
        O_minus, _ = flash_attention_fwd(Q, tensor_minus, V, tile_size, causal)
    else:
        raise ValueError(f"Unknown grad_name: {grad_name}")

    loss_plus = np.sum(O_plus * dO)
    loss_minus = np.sum(O_minus * dO)

    return (loss_plus - loss_minus) / (2 * eps)


def relative_error(a, b):
    """Compute max relative error, handling near-zero values."""
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.maximum(denom, 1e-10)
    return np.max(np.abs(a - b) / denom)


def test_gradient_check():
    """Test 1: Gradient check vs central finite differences."""
    print("=" * 60)
    print("Test 1: Gradient check (finite differences)")
    print("=" * 60)

    np.random.seed(42)
    B, H, N, D = 1, 1, 64, 32
    T = 16
    causal = True

    Q = np.random.randn(B, H, N, D).astype(np.float64)
    K = np.random.randn(B, H, N, D).astype(np.float64)
    V = np.random.randn(B, H, N, D).astype(np.float64)
    dO = np.random.randn(B, H, N, D).astype(np.float64)

    _, cache = flash_attention_fwd(Q, K, V, T, causal)
    dQ, dK, dV = flash_attention_bwd(dO, cache, T, causal)

    # Full dV check
    print("Computing finite differences for dV (all elements)...")
    dV_fd = finite_diff_dV(Q, K, V, dO, eps=1e-5, tile_size=T, causal=causal)
    err_dV = relative_error(dV, dV_fd)
    print(f"  dV max relative error: {err_dV:.2e}")
    assert err_dV < 1e-5, f"dV gradient check FAILED: rel error {err_dV:.2e} >= 1e-5"
    print("  dV: PASSED")

    # Spot-check dQ and dK at 10 random positions
    for grad_name, grad_tensor in [('Q', dQ), ('K', dK)]:
        src = Q if grad_name == 'Q' else K
        errors = []
        indices = []
        for _ in range(10):
            idx = tuple(np.random.randint(0, s, size=(4,)) for s in [B, H, N, D])[0]
            idx = (np.random.randint(B), np.random.randint(H), np.random.randint(N), np.random.randint(D))
            grad_fd = finite_diff_at_point(Q, K, V, dO, src, grad_name, idx, eps=1e-5, tile_size=T, causal=causal)
            grad_actual = grad_tensor[idx]
            err = np.abs(grad_actual - grad_fd) / max(abs(grad_actual), abs(grad_fd), 1e-10)
            errors.append(err)
            indices.append(idx)

        max_err = max(errors)
        print(f"  d{grad_name} spot-check (10 positions) max relative error: {max_err:.2e}")
        assert max_err < 1e-5, f"d{grad_name} gradient check FAILED: max rel error {max_err:.2e} >= 1e-5"
        print(f"  d{grad_name}: PASSED")

    print("Test 1: ALL PASSED\n")


def test_vs_naive():
    """Test 2: Compare against naive full-materialized backward."""
    print("=" * 60)
    print("Test 2: vs naive full-materialized backward")
    print("=" * 60)

    np.random.seed(42)
    B, H, N, D = 2, 4, 256, 64
    T = 64
    causal = True

    Q = np.random.randn(B, H, N, D).astype(np.float64)
    K = np.random.randn(B, H, N, D).astype(np.float64)
    V = np.random.randn(B, H, N, D).astype(np.float64)
    dO = np.random.randn(B, H, N, D).astype(np.float64)

    # Tiled forward + backward
    _, cache = flash_attention_fwd(Q, K, V, T, causal)
    dQ_tiled, dK_tiled, dV_tiled = flash_attention_bwd(dO, cache, T, causal)

    # Naive forward + backward
    _, P, _ = naive_attention_fwd(Q, K, V, causal)
    dQ_naive, dK_naive, dV_naive = naive_attention_bwd(dO, Q, K, V, P, causal)

    # Also check forward outputs match
    O_tiled = cache['O']
    O_naive, _, _ = naive_attention_fwd(Q, K, V, causal)
    err_O = relative_error(O_tiled, O_naive)
    print(f"  Forward O max relative error: {err_O:.2e}")

    err_dQ = relative_error(dQ_tiled, dQ_naive)
    err_dK = relative_error(dK_tiled, dK_naive)
    err_dV = relative_error(dV_tiled, dV_naive)

    print(f"  dQ max relative error: {err_dQ:.2e}")
    print(f"  dK max relative error: {err_dK:.2e}")
    print(f"  dV max relative error: {err_dV:.2e}")

    assert err_dQ < 1e-4, f"dQ FAILED: max rel error {err_dQ:.2e} >= 1e-4"
    assert err_dK < 1e-4, f"dK FAILED: max rel error {err_dK:.2e} >= 1e-4"
    assert err_dV < 1e-4, f"dV FAILED: max rel error {err_dV:.2e} >= 1e-4"

    print("Test 2: ALL PASSED\n")


def test_memory():
    """Test 3: Memory constraint with tracemalloc."""
    print("=" * 60)
    print("Test 3: Memory constraint")
    print("=" * 60)

    np.random.seed(42)
    B, H, N, D = 1, 1, 4096, 64
    T = 128
    causal = True

    Q = np.random.randn(B, H, N, D).astype(np.float64)
    K = np.random.randn(B, H, N, D).astype(np.float64)
    V = np.random.randn(B, H, N, D).astype(np.float64)
    dO = np.random.randn(B, H, N, D).astype(np.float64)

    # Memory for a single (N, N) matrix in float64
    nn_matrix_bytes = N * N * 8  # float64 = 8 bytes
    threshold = 0.20 * nn_matrix_bytes

    print(f"  N={N}, single (N,N) matrix = {nn_matrix_bytes / 1e6:.1f} MB")
    print(f"  20% threshold = {threshold / 1e6:.1f} MB")

    # Measure forward pass peak memory
    tracemalloc.start()
    _, cache = flash_attention_fwd(Q, K, V, T, causal)
    fwd_peak, fwd_curr = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Forward peak memory: {fwd_peak / 1e6:.1f} MB")
    fwd_pass = fwd_peak < threshold
    print(f"  Forward under threshold: {'YES' if fwd_pass else 'NO'}")

    # Measure backward pass peak memory
    tracemalloc.start()
    dQ, dK, dV = flash_attention_bwd(dO, cache, T, causal)
    bwd_peak, bwd_curr = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Backward peak memory: {bwd_peak / 1e6:.1f} MB")
    bwd_pass = bwd_peak < threshold
    print(f"  Backward under threshold: {'YES' if bwd_pass else 'NO'}")

    # Combined peak (should be max of fwd/bwd since they're sequential)
    combined_peak = max(fwd_peak, bwd_peak)
    print(f"  Combined peak: {combined_peak / 1e6:.1f} MB")

    assert fwd_pass, f"Forward peak {fwd_peak / 1e6:.1f} MB >= threshold {threshold / 1e6:.1f} MB"
    assert bwd_pass, f"Backward peak {bwd_peak / 1e6:.1f} MB >= threshold {threshold / 1e6:.1f} MB"

    print("Test 3: ALL PASSED\n")


if __name__ == '__main__':
    test_gradient_check()
    test_vs_naive()
    test_memory()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
