"""
Numerically Stable Layer Normalization Backward Pass — From Scratch in NumPy

Forward:
    μ       = mean(x, axis=-1)                          # (B, T)
    σ²      = var(x, axis=-1)                           # (B, T)
    x_hat   = (x - μ) / sqrt(σ² + ε)                    # (B, T, D)
    y       = γ · x_hat + β                             # (B, T, D)

Backward (given ∂L/∂y ≡ dy of shape (B, T, D)):
    dγ = sum(dy · x_hat, axis=(0,1))                    # (D,)
    dβ = sum(dy, axis=(0,1))                            # (D,)
    dx = (1/N) · (σ²+ε)^(-1/2) · [
            N·dy
          - sum(dy, axis=-1)
          - x_hat · sum(dy·x_hat, axis=-1)
        ]                                                # (B, T, D)

    where N = D (feature dimension).

Derivation sketch (see comments in code for full detail):
    The normalization map x ↦ x_hat is a projection onto the unit sphere
    (per position). Its Jacobian has the form:
        ∂x_hat_i / ∂x_j = (1/σ) · (δ_ij - 1/N - x_hat_i · x_hat_j / N)
    Contracting with dy gives the compact formula above.

Numerical stability notes:
    1. Variance computation: use the two-pass (Welford-style) algorithm
       instead of E[x²] - E[x]² to avoid catastrophic cancellation.
    2. The backward formula reuses x_hat (already computed in forward),
       avoiding recomputing (x - μ) / σ.
    3. All divisions go through σ = sqrt(σ² + ε) with ε > 0, so no
       division-by-zero.
    4. The term (σ²+ε)^(-1/2) is precomputed once and broadcast.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def layer_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Layer normalization forward pass.

    Parameters
    ----------
    x     : (B, T, D) — input
    gamma : (D,)      — scale
    beta  : (D,)      — shift
    eps   : float     — numerical stability constant

    Returns
    -------
    y      : (B, T, D) — output
    cache  : dict       — intermediates for backward
    """
    B, T, D = x.shape

    # --- mean (B, T) ---
    mu = x.mean(axis=-1)                          # (B, T)

    # --- variance via two-pass (numerically stable) ---
    # Pass 1: centered values
    x_centered = x - mu[..., np.newaxis]           # (B, T, D)
    # Pass 2: variance of centered values
    var = np.mean(x_centered ** 2, axis=-1)        # (B, T)

    # --- normalization ---
    std_inv = 1.0 / np.sqrt(var + eps)             # (B, T)
    x_hat = x_centered * std_inv[..., np.newaxis]  # (B, T, D)

    # --- affine ---
    y = gamma[np.newaxis, np.newaxis, :] * x_hat + beta[np.newaxis, np.newaxis, :]

    # Cache only what the backward pass needs — minimal memory footprint.
    # The backward formula uses x_hat, std_inv, and gamma. Nothing else.
    cache = {
        "x_hat": x_hat,           # (B, T, D) — normalized input
        "std_inv": std_inv,       # (B, T)   — 1/sqrt(var + eps)
        "gamma": gamma,           # (D,)     — scale parameter
        "D": D,                   # scalar   — feature dimension
    }

    return y, cache


# ---------------------------------------------------------------------------
# Backward pass — numerically stable, no redundant recomputation
# ---------------------------------------------------------------------------

def layer_norm_backward(dy, cache):
    """
    Layer normalization backward pass.

    Given dy = ∂L/∂y of shape (B, T, D), compute gradients w.r.t.
    x, gamma, and beta.

    The key insight for numerical stability is to express dx entirely in
    terms of quantities already cached from the forward pass (x_hat,
    std_inv), avoiding any recomputation of (x - μ) or sqrt(var + ε).

    Parameters
    ----------
    dy    : (B, T, D) — upstream gradient
    cache : dict       — from forward pass

    Returns
    -------
    dx    : (B, T, D)
    dgamma: (D,)
    dbeta : (D,)
    """
    x_hat = cache["x_hat"]           # (B, T, D)
    std_inv = cache["std_inv"]       # (B, T)
    gamma = cache["gamma"]           # (D,)
    D = cache["D"]                   # scalar

    B, T, _ = dy.shape

    # --- gradient w.r.t. gamma and beta (trivial) ---
    dgamma = np.sum(dy * x_hat, axis=(0, 1))   # (D,)
    dbeta = np.sum(dy, axis=(0, 1))             # (D,)

    # --- gradient w.r.t. x (the non-trivial part) ---
    #
    # Full derivation:
    #   y = γ · x_hat + β
    #   ∂L/∂x_hat = γ · dy
    #
    #   x_hat_i = (x_i - μ) / σ,  where σ = sqrt(var + ε)
    #
    #   ∂x_hat_i / ∂x_j = (1/σ) · (δ_ij - 1/D - x_hat_i · x_hat_j / D)
    #
    #   Therefore:
    #   ∂L/∂x_j = Σ_i (∂L/∂x_hat_i) · ∂x_hat_i / ∂x_j
    #            = (1/σ) · [ Σ_i (γ·dy)_i · (δ_ij - 1/D - x_hat_i·x_hat_j/D) ]
    #            = (1/σ) · [ (γ·dy)_j - (1/D)·Σ_i(γ·dy)_i - x_hat_j·(1/D)·Σ_i(γ·dy)_i·x_hat_i ]
    #
    #   Let g = γ · dy  (elementwise)
    #   dx = (1/σ) · [ g - mean(g) - x_hat · mean(g · x_hat) ]
    #
    # This is the compact, numerically stable form. All terms are O(1) per
    # element after the two reductions (mean over D).

    g = gamma[np.newaxis, np.newaxis, :] * dy    # (B, T, D)

    # Two reductions over the feature dimension D
    g_mean = g.mean(axis=-1, keepdims=True)       # (B, T, 1)
    gx_mean = (g * x_hat).mean(axis=-1, keepdims=True)  # (B, T, 1)

    # Combine — std_inv broadcasts from (B, T) to (B, T, D)
    dx = std_inv[..., np.newaxis] * (g - g_mean - x_hat * gx_mean)

    return dx, dgamma, dbeta


# ---------------------------------------------------------------------------
# Gradient check — finite differences
# ---------------------------------------------------------------------------

def numerical_gradient(f, param, delta=1e-5, **fixed_kwargs):
    """
    Compute numerical gradient of scalar function f w.r.t. param using
    central finite differences.

    f should take param as its first positional argument and return a scalar.
    """
    grad = np.zeros_like(param)
    flat_param = param.ravel()
    flat_grad = grad.ravel()

    for i in range(len(flat_param)):
        old_val = flat_param[i]

        flat_param[i] = old_val + delta
        f_plus = f(param.reshape(param.shape), **fixed_kwargs)

        flat_param[i] = old_val - delta
        f_minus = f(param.reshape(param.shape), **fixed_kwargs)

        flat_grad[i] = (f_plus - f_minus) / (2 * delta)
        flat_param[i] = old_val

    return grad


def gradient_check(gamma, beta, x, eps=1e-5, delta=1e-5):
    """
    Verify analytical gradients against finite-difference numerical gradients.

    Returns a dict with relative errors for each parameter.
    """
    # Random upstream gradient
    dy = np.random.randn(*x.shape)

    # --- Analytical gradients ---
    y, cache = layer_norm_forward(x, gamma, beta, eps=eps)
    dx_analytical, dgamma_analytical, dbeta_analytical = layer_norm_backward(dy, cache)

    # --- Numerical gradients ---
    def loss_wrt_x(x_arg):
        y_arg, _ = layer_norm_forward(x_arg, gamma, beta, eps=eps)
        return np.sum(y_arg * dy)

    def loss_wrt_gamma(gamma_arg):
        y_arg, _ = layer_norm_forward(x, gamma_arg, beta, eps=eps)
        return np.sum(y_arg * dy)

    def loss_wrt_beta(beta_arg):
        y_arg, _ = layer_norm_forward(x, gamma_arg=gamma, beta_arg=beta_arg, eps=eps)
        return np.sum(y_arg * dy)

    # Fix the kwargs properly
    def loss_x(x_arg):
        y_arg, _ = layer_norm_forward(x_arg, gamma, beta, eps=eps)
        return np.sum(y_arg * dy)

    def loss_gamma(gamma_arg):
        y_arg, _ = layer_norm_forward(x, gamma_arg, beta, eps=eps)
        return np.sum(y_arg * dy)

    def loss_beta(beta_arg):
        y_arg, _ = layer_norm_forward(x, gamma, beta_arg, eps=eps)
        return np.sum(y_arg * dy)

    dx_numerical = numerical_gradient(loss_x, x, delta=delta)
    dgamma_numerical = numerical_gradient(loss_gamma, gamma, delta=delta)
    dbeta_numerical = numerical_gradient(loss_beta, beta, delta=delta)

    # --- Relative errors ---
    def rel_error(a, b):
        denom = np.max(np.abs(a) + np.abs(b))
        if denom < 1e-12:
            return 0.0
        return np.max(np.abs(a - b)) / denom

    errors = {
        "dx": rel_error(dx_analytical, dx_numerical),
        "dgamma": rel_error(dgamma_analytical, dgamma_numerical),
        "dbeta": rel_error(dbeta_analytical, dbeta_numerical),
    }

    return errors


# ---------------------------------------------------------------------------
# Complexity analysis
# ---------------------------------------------------------------------------

def print_complexity_analysis(B, T, D):
    """
    Time and memory complexity of layer norm forward + backward.

    Notation: N = B·T·D (total elements), D = feature dim.

    FORWARD:
    ┌──────────────────────────────────────────────────────────────────┐
    │ Operation                  │ FLOPs              │ Memory (extra)  │
    ├──────────────────────────────────────────────────────────────────┤
    │ mean(x, axis=-1)           │ N                  │ B·T             │
    │ x_centered = x - μ         │ N                  │ B·T·D           │
    │ var = mean(x_centered²)    │ 2N                 │ B·T             │
    │ std_inv = 1/sqrt(var+ε)    │ B·T                │ B·T             │
    │ x_hat = x_centered * σ⁻¹   │ N                  │ B·T·D           │
    │ y = γ·x_hat + β            │ 2N                 │ B·T·D (output)  │
    ├──────────────────────────────────────────────────────────────────┤
    │ Total                      │ ~6N                │ ~3·B·T·D        │
    └──────────────────────────────────────────────────────────────────┘

    BACKWARD:
    ┌──────────────────────────────────────────────────────────────────┐
    │ Operation                  │ FLOPs              │ Memory (extra)  │
    ├──────────────────────────────────────────────────────────────────┤
    │ g = γ · dy                 │ N                  │ B·T·D           │
    │ g_mean = mean(g, axis=-1)  │ N                  │ B·T             │
    │ gx_mean = mean(g·x_hat)    │ 2N                 │ B·T             │
    │ dx = σ⁻¹·(g - g_mean - …) │ 3N                 │ B·T·D           │
    │ dgamma = sum(dy·x_hat)     │ 2N                 │ D               │
    │ dbeta = sum(dy)            │ N                  │ D               │
    ├──────────────────────────────────────────────────────────────────┤
    │ Total                      │ ~9N                │ ~B·T·D          │
    └──────────────────────────────────────────────────────────────────┘

    OVERALL:
      Time:    O(N) = O(B·T·D)  — linear in total elements
      Memory:  O(B·T·D)         — dominated by cached x_hat

    KEY OBSERVATIONS:
      • The backward pass is ~1.5× the forward pass in FLOPs.
      • Memory is dominated by caching x_hat (B·T·D floats).
      • The two-pass variance is O(N) extra FLOPs but essential for
        numerical stability — the naive E[x²]-E[x]² formula can lose
        15+ digits of precision when var ≪ mean².
    """
    N = B * T * D
    print(f"Complexity Analysis for B={B}, T={T}, D={D} (N={N:,} total elements)")
    print(f"  Forward FLOPs:  ~{6*N:,}")
    print(f"  Backward FLOPs: ~{9*N:,}")
    print(f"  Total FLOPs:    ~{15*N:,}")
    print(f"  Extra memory:   ~{3*N * 4 / 1024 / 1024:.1f} MB (forward cache)")
    print(f"  Time complexity: O(B·T·D)")
    print(f"  Space complexity: O(B·T·D)")


# ---------------------------------------------------------------------------
# GPU kernel fusion discussion
# ---------------------------------------------------------------------------

GPU_FUSION_DISCUSSION = """
GPU KERNEL FUSION FOR LAYER NORM
=================================

1. FORWARD KERNEL (single kernel, no intermediate global memory writes):

   Thread block: one block per (b, t) position, D threads per block.
   Each thread handles one feature dimension d.

   Pseudocode (CUDA-style):
   ```
   __global__ void layer_norm_fwd(const float* __restrict__ x,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ y,
                                   int B, int T, int D, float eps) {
       int bt = blockIdx.x;  // flattened (b, t)
       int d  = threadIdx.x; // feature dimension
       int stride = gridDim.x;

       // --- Parallel reduce: mean ---
       float sum = 0.0f;
       for (int i = d; i < D; i += blockDim.x)
           sum += x[bt * D + i];
       float mu = blockReduceSum(sum) / D;

       // --- Parallel reduce: variance (two-pass) ---
       float sum2 = 0.0f;
       for (int i = d; i < D; i += blockDim.x) {
           float diff = x[bt * D + i] - mu;
           sum2 += diff * diff;
       }
       float var = blockReduceSum(sum2) / D;
       float std_inv = rsqrtf(var + eps);  // hardware reciprocal sqrt

       // --- Write output ---
       float x_hat = (x[bt * D + d] - mu) * std_inv;
       y[bt * D + d] = gamma[d] * x_hat + beta[d];

       // --- Cache x_hat for backward (write to pre-allocated buffer) ---
       // This is the ONLY intermediate that must survive to backward.
       // All other intermediates (mu, var, std_inv) are register-local.
   }
   ```

   Key fusion benefits:
   • x is read ONCE from global memory (not twice as in separate mean/var).
   • mu, var, std_inv live in registers/shared memory — zero global writes.
   • x_hat is written once to the cache buffer.
   • rsqrtf is a single hardware instruction on NVIDIA GPUs.

2. BACKWARD KERNEL (single kernel):

   Thread block: one block per (b, t), D threads per block.

   ```
   __global__ void layer_norm_bwd(const float* __restrict__ dy,
                                   const float* __restrict__ x_hat,
                                   const float* __restrict__ gamma,
                                   float std_inv,  // passed as param or loaded
                                   float* __restrict__ dx,
                                   float* __restrict__ dgamma,
                                   float* __restrict__ dbeta,
                                   int D) {
       int bt = blockIdx.x;
       int d  = threadIdx.x;

       float g = gamma[d] * dy[bt * D + d];

       // --- Parallel reduce: mean(g) and mean(g * x_hat) ---
       float g_sum = 0.0f, gx_sum = 0.0f;
       for (int i = d; i < D; i += blockDim.x) {
           g_sum  += gamma[i] * dy[bt * D + i];
           gx_sum += gamma[i] * dy[bt * D + i] * x_hat[bt * D + i];
       }
       float g_mean  = blockReduceSum(g_sum)  / D;
       float gx_mean = blockReduceSum(gx_sum) / D;

       // --- Compute dx ---
       float x_hat_d = x_hat[bt * D + d];
       dx[bt * D + d] = std_inv * (g - g_mean - x_hat_d * gx_mean);

       // --- Atomic adds for dgamma, dbeta ---
       float dy_d = dy[bt * D + d];
       atomicAdd(&dgamma[bt * D_stride + d], dy_d * x_hat_d);
       atomicAdd(&dbeta[bt * D_stride + d], dy_d);
   }
   ```

   Key fusion benefits:
   • dy and x_hat are read ONCE each.
   • The two reductions (g_mean, gx_mean) share the same loop — one pass.
   • dx is computed and written in the same thread that computed g.
   • dgamma/dbeta use atomicAdd (D is typically small enough that contention
     is manageable; alternatively, use a two-phase reduce).

3. MEMORY TRAFFIC COMPARISON:

   Naive (separate kernels):
     Forward:  read x (1×), write mu (1×), read x+mu (2×), write var (1×),
               read x+mu+var (3×), write x_hat (1×), read x_hat+γ+β (3×),
               write y (1×)  →  ~12 global memory accesses per element
     Backward: similar explosion

   Fused:
     Forward:  read x (1×), read γ+β (1×), write x_hat (1×), write y (1×)
               →  4 global memory accesses per element
     Backward: read dy (1×), read x_hat (1×), read γ (1×), write dx (1×),
               atomic dgamma+dbeta (1×)  →  5 global memory accesses per element

   The fused approach is ~2-3× faster in practice because memory bandwidth
   is the bottleneck for layer norm (it's an O(N) algorithm with O(N) memory).

4. SHARED MEMORY OPTIMIZATION:

   For small D (≤ 1024), load the entire (b,t) slice into shared memory:
   ```
   __shared__ float s_x[1024], s_dy[1024], s_xhat[1024];
   // Cooperative load
   s_x[d] = x[bt * D + d];
   __syncthreads();
   // All subsequent ops use shared memory (L1-equivalent speed)
   ```
   This cuts global memory reads from 3 to 1 per kernel launch.

5. TENSOR CORE / WARP LEVEL:

   Layer norm doesn't benefit from tensor cores (no GEMM), but warp-level
   primitives (__shfl_down_sync) can replace shared memory for the parallel
   reductions when D ≤ 32, eliminating synchronization overhead entirely.
"""


# ---------------------------------------------------------------------------
# Main — run gradient check and analysis
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    B, T, D = 4, 8, 16
    x = np.random.randn(B, T, D).astype(np.float64)
    gamma = np.random.randn(D).astype(np.float64)
    beta = np.random.randn(D).astype(np.float64)

    print("=" * 70)
    print("LAYER NORMALIZATION — BACKWARD PASS (NUMPY, FROM SCRATCH)")
    print("=" * 70)

    # --- Forward ---
    y, cache = layer_norm_forward(x, gamma, beta)
    print(f"\nForward: x({x.shape}) → y({y.shape})")
    print(f"  y[0,0,:4] = {y[0, 0, :4]}")

    # --- Backward ---
    dy = np.random.randn(B, T, D).astype(np.float64)
    dx, dgamma, dbeta = layer_norm_backward(dy, cache)
    print(f"\nBackward: dy({dy.shape}) → dx({dx.shape}), dγ({dgamma.shape}), dβ({dbeta.shape})")

    # --- Gradient check ---
    print("\n" + "-" * 70)
    print("GRADIENT CHECK (central finite differences, δ=1e-5)")
    print("-" * 70)
    errors = gradient_check(gamma, beta, x)
    for name, err in errors.items():
        status = "✓ PASS" if err < 1e-6 else "✗ FAIL"
        print(f"  {name:8s}  relative error: {err:.2e}  {status}")

    # --- Complexity ---
    print("\n" + "-" * 70)
    print("COMPLEXITY ANALYSIS")
    print("-" * 70)
    print_complexity_analysis(B, T, D)

    # --- GPU fusion discussion ---
    print("\n" + "-" * 70)
    print("GPU KERNEL FUSION STRATEGY")
    print("-" * 70)
    print(GPU_FUSION_DISCUSSION)

    # --- Numerical stability demo ---
    print("\n" + "-" * 70)
    print("NUMERICAL STABILITY DEMONSTRATION")
    print("-" * 70)
    print("""
    Where instability can occur and how we handle it:

    1. VARIANCE COMPUTATION
       Problem: E[x²] - E[x]² loses precision when var ≪ mean²
       (catastrophic cancellation). Example: x = [1e8, 1e8+1, 1e8+2]
       Solution: TWO-PASS algorithm — center first, then compute variance.
       This is what we do: x_centered = x - μ, then var = mean(x_centered²).

    2. DIVISION BY ZERO
       Problem: var could be exactly 0 (all features identical).
       Solution: add ε (default 1e-5) inside sqrt: sqrt(var + ε).
       This is standard and numerically safe.

    3. BACKWARD OVERFLOW
       Problem: if std_inv is very large (var ≈ 0), dx could overflow.
       Solution: ε prevents std_inv from exceeding 1/sqrt(ε) ≈ 447.
       With float32, this is well within range.

    4. ACCUMULATION ERROR IN REDUCTIONS
       Problem: summing D values can accumulate floating-point error.
       Solution: NumPy uses pairwise summation (O(log D) error growth
       instead of O(D)). For extreme cases, use Kahan summation.

    5. RECOMPUTATION AVOIDANCE
       Problem: naive backward recomputes (x - μ) and sqrt(var + ε).
       Solution: cache x_hat and std_inv from forward. The backward
       formula uses ONLY these cached values + dy — zero recomputation.
    """)


if __name__ == "__main__":
    main()
