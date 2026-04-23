"""
Benchmark and numerical stability comparison for layer_norm_backward.py
"""

import numpy as np
import time
from layer_norm_backward import layer_norm_forward, layer_norm_backward


# ---------------------------------------------------------------------------
# 1. Numerical stability: two-pass vs naive variance
# ---------------------------------------------------------------------------

def naive_variance(x, axis=-1):
    """Naive one-pass variance: E[x²] - E[x]² — prone to cancellation."""
    return np.mean(x ** 2, axis=axis) - np.mean(x, axis=axis) ** 2


def two_pass_variance(x, axis=-1):
    """Two-pass variance: center first, then compute — numerically stable."""
    mu = np.mean(x, axis=axis, keepdims=True)
    return np.mean((x - mu) ** 2, axis=axis)


def demo_variance_stability():
    print("=" * 70)
    print("NUMERICAL STABILITY: TWO-PASS vs NAIVE VARIANCE")
    print("=" * 70)
    print()
    print("When mean² ≫ var, the naive formula E[x²] - E[x]² suffers from")
    print("catastrophic cancellation. The two-pass algorithm avoids this.")
    print()

    # Construct a pathological case: large offset, tiny variance
    offset = 1e8
    true_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    true_var = np.var(true_values)  # 2.0

    x_shifted = true_values + offset

    naive_var = naive_variance(x_shifted[np.newaxis, np.newaxis, :])
    stable_var = two_pass_variance(x_shifted[np.newaxis, np.newaxis, :])

    print(f"  True values:       {true_values}")
    print(f"  True variance:     {true_var:.15f}")
    print(f"  Offset:            {offset:.0e}")
    print(f"  Shifted values:    {x_shifted}")
    print()
    print(f"  Naive (E[x²]-E[x]²):   {naive_var[0,0]:.15f}  (error: {abs(naive_var[0,0] - true_var):.2e})")
    print(f"  Two-pass (centered):   {stable_var[0,0]:.15f}  (error: {abs(stable_var[0,0] - true_var):.2e})")
    print()

    # Show how it gets worse with larger offsets
    print("  Worsening with larger offsets:")
    for exp in range(4, 16, 2):
        offset = 10 ** exp
        x = true_values + offset
        nv = naive_variance(x[np.newaxis, np.newaxis, :])[0, 0]
        sv = two_pass_variance(x[np.newaxis, np.newaxis, :])[0, 0]
        print(f"    offset=1e{exp:2d}:  naive={nv:15.6f}  stable={sv:15.6f}  true=2.000000")

    print()


# ---------------------------------------------------------------------------
# 2. Performance benchmark
# ---------------------------------------------------------------------------

def benchmark(B, T, D, n_warmup=5, n_iter=50):
    """Benchmark forward + backward throughput."""
    x = np.random.randn(B, T, D).astype(np.float32)
    gamma = np.random.randn(D).astype(np.float32)
    beta = np.random.randn(D).astype(np.float32)
    dy = np.random.randn(B, T, D).astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        y, cache = layer_norm_forward(x, gamma, beta)
        dx, dgamma, dbeta = layer_norm_backward(dy, cache)

    # Benchmark forward
    times_fwd = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        y, cache = layer_norm_forward(x, gamma, beta)
        times_fwd.append(time.perf_counter() - t0)

    # Benchmark backward
    times_bwd = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        dx, dgamma, dbeta = layer_norm_backward(dy, cache)
        times_bwd.append(time.perf_counter() - t0)

    N = B * T * D
    fwd_ms = np.median(times_fwd) * 1000
    bwd_ms = np.median(times_bwd) * 1000
    fwd_tflops = (6 * N) / (fwd_ms * 1e-3) / 1e12
    bwd_tflops = (9 * N) / (bwd_ms * 1e-3) / 1e12

    return {
        "shape": f"({B}, {T}, {D})",
        "N": N,
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "fwd_tflops": fwd_tflops,
        "bwd_tflops": bwd_tflops,
    }


def run_benchmarks():
    print("=" * 70)
    print("PERFORMANCE BENCHMARK (NumPy, single CPU core)")
    print("=" * 70)
    print()
    print(f"{'Shape':<20} {'Elements':>10} {'Fwd (ms)':>10} {'Bwd (ms)':>10} {'Fwd TF/s':>10} {'Bwd TF/s':>10}")
    print("-" * 72)

    configs = [
        (1, 1, 64),
        (1, 1, 1024),
        (1, 1, 4096),
        (2, 128, 64),
        (2, 128, 1024),
        (2, 128, 4096),
        (4, 512, 1024),
        (4, 512, 4096),
    ]

    for B, T, D in configs:
        result = benchmark(B, T, D)
        print(
            f"{result['shape']:<20} {result['N']:>10,} "
            f"{result['fwd_ms']:>10.4f} {result['bwd_ms']:>10.4f} "
            f"{result['fwd_tflops']:>10.4f} {result['bwd_tflops']:>10.4f}"
        )

    print()
    print("  Note: NumPy is multithreaded for large arrays (BLAS).")
    print("  These numbers are memory-bandwidth bound, not compute bound.")


# ---------------------------------------------------------------------------
# 3. Backward formula verification: alternative derivation
# ---------------------------------------------------------------------------

def verify_backward_alternative():
    """
    Verify the backward formula using an alternative derivation path.

    Alternative: compute dx by explicitly differentiating through each step
    (mean → centered → normalized → affine) rather than using the compact
    projection formula. This serves as a cross-check.
    """
    print("=" * 70)
    print("BACKWARD CROSS-CHECK: ALTERNATIVE DERIVATION")
    print("=" * 70)
    print()

    B, T, D = 3, 5, 8
    x = np.random.randn(B, T, D).astype(np.float64)
    gamma = np.random.randn(D).astype(np.float64)
    beta = np.random.randn(D).astype(np.float64)
    dy = np.random.randn(B, T, D).astype(np.float64)

    # Forward
    mu = x.mean(axis=-1, keepdims=True)        # (B, T, 1)
    x_c = x - mu                                 # (B, T, D)
    var = np.mean(x_c ** 2, axis=-1, keepdims=True)  # (B, T, 1)
    std = np.sqrt(var + 1e-5)                    # (B, T, 1)
    x_hat = x_c / std                            # (B, T, D)
    y = gamma * x_hat + beta

    # --- Alternative backward: step-by-step chain rule ---
    # Step 4: y = γ·x_hat + β  →  ∂L/∂x_hat = γ·dy
    dx_hat = gamma[np.newaxis, np.newaxis, :] * dy  # (B, T, D)

    # Step 3: x_hat = x_c / std
    #   ∂x_hat_i/∂x_c_j = δ_ij/std - x_c_i·(Σ_k x_c_k·∂x_c_k/∂x_c_j)/(D·std³)
    #   But since std depends on x_c, we need the full derivative.
    #   ∂x_hat_i/∂x_c_j = (δ_ij·std - x_hat_i·x_hat_j/std) / std
    #                    = (δ_ij - x_hat_i·x_hat_j) / std
    #   Wait, that's not quite right. Let me be more careful.
    #
    #   x_hat_i = x_c_i / σ where σ = sqrt(mean(x_c²) + ε)
    #   ∂σ/∂x_c_j = x_c_j / (D·σ)
    #   ∂x_hat_i/∂x_c_j = (δ_ij·σ - x_c_i·∂σ/∂x_c_j) / σ²
    #                    = (δ_ij·σ - x_c_i·x_c_j/(D·σ)) / σ²
    #                    = δ_ij/σ - x_hat_i·x_hat_j/(D·σ)
    #                    = (1/σ) · (δ_ij - x_hat_i·x_hat_j/D)
    #
    #   So: ∂L/∂x_c_j = Σ_i dx_hat_i · (1/σ) · (δ_ij - x_hat_i·x_hat_j/D)
    #                    = (1/σ) · [dx_hat_j - (1/D)·Σ_i(dx_hat_i·x_hat_i)·x_hat_j]

    std_inv = 1.0 / std[..., 0]  # (B, T)
    dx_hat_sum_xhat = np.sum(dx_hat * x_hat, axis=-1, keepdims=True)  # (B, T, 1)
    dx_c = std_inv[..., np.newaxis] * (dx_hat - dx_hat_sum_xhat * x_hat / D)

    # Step 2: x_c = x - μ
    #   ∂x_c_i/∂x_j = δ_ij - 1/D
    #   ∂L/∂x_j = Σ_i ∂L/∂x_c_i · (δ_ij - 1/D)
    #            = dx_c_j - (1/D)·Σ_i dx_c_i
    dx_c_sum = np.sum(dx_c, axis=-1, keepdims=True)  # (B, T, 1)
    dx_alt = dx_c - dx_c_sum / D

    # --- Our compact backward ---
    y2, cache = layer_norm_forward(x, gamma, beta)
    dx_ours, _, _ = layer_norm_backward(dy, cache)

    # Compare
    rel_err = np.max(np.abs(dx_alt - dx_ours)) / np.max(np.abs(dx_alt) + np.abs(dx_ours) + 1e-30)
    print(f"  Alternative derivation (step-by-step chain rule)")
    print(f"  Compact derivation (projection formula)")
    print(f"  Relative error: {rel_err:.2e}")
    print(f"  {'✓ MATCH' if rel_err < 1e-10 else '✗ MISMATCH'}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    demo_variance_stability()
    run_benchmarks()
    verify_backward_alternative()
