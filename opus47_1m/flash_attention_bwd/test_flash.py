import numpy as np
import tracemalloc

from flash_attention import (
    flash_attention_fwd,
    flash_attention_bwd,
    naive_forward,
    naive_backward,
)


def rel_err(a, b):
    num = np.abs(a - b).max()
    den = max(np.abs(a).max(), np.abs(b).max(), 1e-12)
    return num / den


def test1_grad_check():
    """Finite difference gradient check."""
    print("=" * 60)
    print("Test 1: Gradient check (finite differences)")
    print("=" * 60)
    rng = np.random.default_rng(0)
    B, H, N, D, T = 1, 1, 64, 32, 16
    causal = True

    Q = rng.standard_normal((B, H, N, D))
    K = rng.standard_normal((B, H, N, D))
    V = rng.standard_normal((B, H, N, D))

    O, cache = flash_attention_fwd(Q, K, V, T, causal=causal)
    dO = np.ones_like(O)
    dQ, dK, dV = flash_attention_bwd(dO, cache, T, causal=causal)

    eps = 1e-6

    def loss(Qx, Kx, Vx):
        Ox, _ = flash_attention_fwd(Qx, Kx, Vx, T, causal=causal)
        return Ox.sum()

    # dV across ALL elements
    dV_fd = np.zeros_like(V)
    for idx in np.ndindex(*V.shape):
        Vp = V.copy(); Vm = V.copy()
        Vp[idx] += eps; Vm[idx] -= eps
        dV_fd[idx] = (loss(Q, K, Vp) - loss(Q, K, Vm)) / (2 * eps)

    err_dV = rel_err(dV, dV_fd)
    print(f"  dV (all elements) rel error: {err_dV:.3e}")
    assert err_dV < 1e-5, f"dV mismatch: {err_dV}"

    # dQ at 10 random positions
    n_spot = 10
    rng2 = np.random.default_rng(1)
    spots_Q = [tuple(rng2.integers(s) for s in Q.shape) for _ in range(n_spot)]
    for idx in spots_Q:
        Qp = Q.copy(); Qm = Q.copy()
        Qp[idx] += eps; Qm[idx] -= eps
        fd = (loss(Qp, K, V) - loss(Qm, K, V)) / (2 * eps)
        an = dQ[idx]
        e = abs(fd - an) / max(abs(fd), abs(an), 1e-12)
        assert e < 1e-5, f"dQ mismatch at {idx}: an={an}, fd={fd}, rel={e}"
    print(f"  dQ (10 spots) max rel error OK")

    # dK at 10 random positions
    spots_K = [tuple(rng2.integers(s) for s in K.shape) for _ in range(n_spot)]
    for idx in spots_K:
        Kp = K.copy(); Km = K.copy()
        Kp[idx] += eps; Km[idx] -= eps
        fd = (loss(Q, Kp, V) - loss(Q, Km, V)) / (2 * eps)
        an = dK[idx]
        e = abs(fd - an) / max(abs(fd), abs(an), 1e-12)
        assert e < 1e-5, f"dK mismatch at {idx}: an={an}, fd={fd}, rel={e}"
    print(f"  dK (10 spots) max rel error OK")
    print("Test 1 PASSED\n")


def test2_vs_naive():
    """Compare against full-materialized naive backward."""
    print("=" * 60)
    print("Test 2: vs naive backward")
    print("=" * 60)
    rng = np.random.default_rng(2)
    B, H, N, D, T = 2, 4, 256, 64, 64
    causal = True

    Q = rng.standard_normal((B, H, N, D))
    K = rng.standard_normal((B, H, N, D))
    V = rng.standard_normal((B, H, N, D))
    dO = rng.standard_normal((B, H, N, D))

    # Flash
    O_flash, cache = flash_attention_fwd(Q, K, V, T, causal=causal)
    dQ_f, dK_f, dV_f = flash_attention_bwd(dO, cache, T, causal=causal)

    # Naive
    O_naive, P = naive_forward(Q, K, V, causal=causal)
    dQ_n, dK_n, dV_n = naive_backward(Q, K, V, dO, P)

    err_O = rel_err(O_flash, O_naive)
    err_dQ = rel_err(dQ_f, dQ_n)
    err_dK = rel_err(dK_f, dK_n)
    err_dV = rel_err(dV_f, dV_n)
    print(f"  O  rel err: {err_O:.3e}")
    print(f"  dQ rel err: {err_dQ:.3e}")
    print(f"  dK rel err: {err_dK:.3e}")
    print(f"  dV rel err: {err_dV:.3e}")
    assert err_O < 1e-4
    assert err_dQ < 1e-4
    assert err_dK < 1e-4
    assert err_dV < 1e-4
    print("Test 2 PASSED\n")


def test3_memory():
    """Verify peak memory < 20% of (N,N) matrix size."""
    print("=" * 60)
    print("Test 3: Memory check")
    print("=" * 60)
    B, H, N, D, T = 1, 1, 4096, 64, 128
    causal = True

    nn_bytes = N * N * 8  # float64 (N,N) matrix
    budget = 0.2 * nn_bytes
    print(f"  (N,N) matrix size: {nn_bytes / 1e6:.2f} MB")
    print(f"  Budget (20%):      {budget / 1e6:.2f} MB")

    rng = np.random.default_rng(3)
    Q = rng.standard_normal((B, H, N, D))
    K = rng.standard_normal((B, H, N, D))
    V = rng.standard_normal((B, H, N, D))

    tracemalloc.start()
    tracemalloc.reset_peak()

    O, cache = flash_attention_fwd(Q, K, V, T, causal=causal)
    dO = np.ones_like(O)
    dQ, dK, dV = flash_attention_bwd(dO, cache, T, causal=causal)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Peak traced bytes: {peak / 1e6:.2f} MB")
    assert peak < budget, f"Peak {peak} >= budget {budget}"
    print("Test 3 PASSED\n")


if __name__ == '__main__':
    test1_grad_check()
    test2_vs_naive()
    test3_memory()
    print("All tests PASSED")
