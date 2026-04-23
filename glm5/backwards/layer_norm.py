"""
Numerically stable forward & backward pass for Layer Normalization.

Forward:
    y = gamma * (x - mean) / sqrt(var + eps) + beta

where mean and var are computed over the last dimension (D) independently
for each (b, t) position.

Reference derivation
--------------------
Let:
    μ   = (1/D) Σ_d x_d                          (mean over D)
    σ²  = (1/D) Σ_d (x_d - μ)²                   (variance over D)
    σ̂   = sqrt(σ² + ε)                           (std with epsilon)
    x̂   = (x - μ) / σ̂                            (normalized x)
    y   = γ · x̂ + β                              (output)

Backward (upstream gradient dy arrives):
    dβ   = Σ_{b,t} dy                     (sum over batch & time)
    dγ   = Σ_{b,t} dy · x̂                (sum over batch & time)

    For dx we chain through x̂, μ, σ̂:
        dx̂    = dy · γ                           (element-wise)
        dσ²   = Σ_d [dx̂_d · (x_d - μ)] · (-½)(σ² + ε)^{-3/2}
              = Σ_d [dx̂_d · (x_d - μ)] · (-1 / σ̂³)
        dμ     = Σ_d dx̂_d · (-1/σ̂)  +  dσ² · (-2/D) Σ_d (x_d - μ)
               = Σ_d dx̂_d · (-1/σ̂)              (second term = 0)
        dx_d   = dx̂_d / σ̂ + dσ² · (2/D)(x_d - μ) + dμ / D

    After substitution and simplification (see analysis below):
        dx = (1/σ̂) · [ dx̂  -  (1/D)(x̂ · Σ_d dx̂_d · x̂_d  +  Σ_d dx̂_d) ]

Time complexity : O(B·T·D)  — one pass over all elements
Memory complexity: O(B·T·D) for the output; intermediates σ̂ and x̂
                    are O(B·T) and O(B·T·D) respectively.
"""

import numpy as np


# ──────────────────────────────────────────────────────────────
# Forward pass — returns cache needed for backward
# ──────────────────────────────────────────────────────────────
def layer_norm_forward(x: np.ndarray,
                       gamma: np.ndarray,
                       beta: np.ndarray,
                       eps: float = 1e-5):
    """
    x     : (B, T, D)
    gamma : (D,)
    beta  : (D,)
    """
    # --- compute statistics over last dim ---
    # keepdims=True avoids a broadcast copy later
    mean = x.mean(axis=-1, keepdims=True)          # (B, T, 1)
    xc   = x - mean                                 # (B, T, D)  centered x

    # Numerical stability note: var is always >= 0 by construction
    # because xc = x - mean, so (xc)**2 >= 0.  The eps guards against
    # the degenerate case where all elements in a row are identical.
    var  = (xc ** 2).mean(axis=-1, keepdims=True)   # (B, T, 1)

    # rsqrt is more stable than 1/sqrt for very small arguments because
    # it avoids the intermediate sqrt → division round-off.
    # NumPy has no native rsqrt, so we compute 1/sqrt carefully.
    rstd = 1.0 / np.sqrt(var + eps)                  # (B, T, 1)  reciprocal std

    xhat = xc * rstd                                 # (B, T, D)  normalized
    y    = gamma * xhat + beta                        # (B, T, D)

    # Cache everything needed for backward
    cache = (xhat, rstd, gamma)
    return y, cache


# ──────────────────────────────────────────────────────────────
# Backward pass — manually derived
# ──────────────────────────────────────────────────────────────
def layer_norm_backward(dy: np.ndarray, cache: tuple):
    """
    dy    : (B, T, D)  upstream gradient
    cache : (xhat, rstd, gamma) from forward

    Returns: dx (B,T,D), dgamma (D,), dbeta (D,)
    """
    xhat, rstd, gamma = cache
    D = xhat.shape[-1]

    # ── dgamma, dbeta ──────────────────────────────────────────
    #    Sum over all batch & time positions; pointwise over D.
    dbeta  = dy.sum(axis=(0, 1))                     # (D,)
    dgamma = (dy * xhat).sum(axis=(0, 1))            # (D,)

    # ── dx ─────────────────────────────────────────────────────
    #  Chain through y = gamma * xhat + beta:
    #      dxhat = dy * gamma
    dxhat = dy * gamma                                # (B, T, D)

    #  Direct implementation of the simplified formula:
    #
    #      dx = (1/σ̂) [ dxhat  -  xhat · mean(dxhat · xhat)  -  mean(dxhat) ]
    #
    #  where the means are over the D dimension.
    #
    #  Derivation:
    #      dxhat_d / σ̂  +  (dvar)(2/D)(x_d-μ)  +  dμ/D
    #      = dxhat_d/σ̂  -  (x̂_d/D) Σ_j dxhat_j x̂_j  -  (1/D) Σ_j dxhat_j
    #      = (1/σ̂)[ dxhat_d  -  x̂_d · (1/D)Σ_j dxhat_j x̂_j  -  (1/D)Σ_j dxhat_j ]
    #
    #  This avoids forming σ² + ε separately and reuses xhat directly.

    #  Inner products over D — these are O(B·T·D) but touch each element once
    proj   = (dxhat * xhat).sum(axis=-1, keepdims=True)   # (B, T, 1)
    dxhat_sum = dxhat.sum(axis=-1, keepdims=True)         # (B, T, 1)

    dx = rstd * (dxhat
                 - xhat * proj / D
                 - dxhat_sum / D)                         # (B, T, D)

    return dx, dgamma, dbeta


# ──────────────────────────────────────────────────────────────
# Gradient check via finite differences
# ──────────────────────────────────────────────────────────────
def gradient_check(B=2, T=3, D=8, eps_fd=1e-5, tol=1e-4, seed=42):
    """Central finite-difference check on x, gamma, beta."""
    rng = np.random.default_rng(seed)
    x     = rng.standard_normal((B, T, D))
    gamma = rng.standard_normal(D) * 0.5 + 1.0
    beta  = rng.standard_normal(D) * 0.1

    eps = 1e-5

    # ── analytic backward ──
    y, cache = layer_norm_forward(x, gamma, beta, eps=eps)
    dy = rng.standard_normal(y.shape)                       # random upstream grad
    dx, dgamma, dbeta = layer_norm_backward(dy, cache)

    # ── helper: scalar loss = sum(dy * y)  ──
    def loss_fn(x_, g_, b_):
        y_, _ = layer_norm_forward(x_, g_, b_, eps=eps)
        return np.sum(dy * y_)

    # ── finite-difference for x ──
    dx_fd = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_plus = x.copy();  x_plus[idx] += eps_fd
        x_minus = x.copy(); x_minus[idx] -= eps_fd
        dx_fd[idx] = (loss_fn(x_plus, gamma, beta)
                      - loss_fn(x_minus, gamma, beta)) / (2 * eps_fd)

    err_x = np.max(np.abs(dx - dx_fd))
    rel_x = err_x / (np.max(np.abs(dx)) + 1e-12)

    # ── finite-difference for gamma ──
    dg_fd = np.zeros_like(gamma)
    for i in range(D):
        g_plus = gamma.copy();  g_plus[i] += eps_fd
        g_minus = gamma.copy(); g_minus[i] -= eps_fd
        dg_fd[i] = (loss_fn(x, g_plus, beta)
                    - loss_fn(x, g_minus, beta)) / (2 * eps_fd)

    err_g = np.max(np.abs(dgamma - dg_fd))
    rel_g = err_g / (np.max(np.abs(dgamma)) + 1e-12)

    # ── finite-difference for beta ──
    db_fd = np.zeros_like(beta)
    for i in range(D):
        b_plus = beta.copy();  b_plus[i] += eps_fd
        b_minus = beta.copy(); b_minus[i] -= eps_fd
        db_fd[i] = (loss_fn(x, gamma, b_plus)
                    - loss_fn(x, gamma, b_minus)) / (2 * eps_fd)

    err_b = np.max(np.abs(dbeta - db_fd))
    rel_b = err_b / (np.max(np.abs(dbeta)) + 1e-12)

    # ── report ──
    print("=" * 60)
    print("Gradient check  (central finite differences, h={})".format(eps_fd))
    print("=" * 60)
    for name, abs_err, rel_err in [("dx", err_x, rel_x),
                                    ("dgamma", err_g, rel_g),
                                    ("dbeta", err_b, rel_b)]:
        ok = "PASS" if rel_err < tol else "FAIL"
        print(f"  {name:>6s}:  max|err| = {abs_err:.2e}   "
              f"rel = {rel_err:.2e}   [{ok}]")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────
# Complexity analysis & GPU fusion discussion
# ──────────────────────────────────────────────────────────────
def main():
    B, T, D = 4, 16, 64
    rng = np.random.default_rng(0)

    x     = rng.standard_normal((B, T, D)).astype(np.float64)
    gamma = rng.standard_normal(D).astype(np.float64) * 0.5 + 1.0
    beta  = rng.standard_normal(D).astype(np.float64) * 0.1

    y, cache = layer_norm_forward(x, gamma, beta)
    dy = rng.standard_normal(y.shape)
    dx, dg, db = layer_norm_backward(dy, cache)

    print(f"Forward  output shape : {y.shape}")
    print(f"Backward dx shape     : {dx.shape}")
    print(f"Backward dgamma shape : {dg.shape}")
    print(f"Backward dbeta  shape : {db.shape}")
    print()

    # Gradient check
    gradient_check(B=3, T=5, D=32)

    print()
    print_complexity_and_fusion(B, T, D)


def print_complexity_and_fusion(B, T, D):
    N = B * T  # number of independent normalizations
    M = N * D  # total elements

    print("─" * 60)
    print("COMPLEXITY ANALYSIS")
    print("─" * 60)
    print(f"  Problem size: B={B}, T={T}, D={D}  →  {N} vectors of dim {D}")
    print()
    print("  Forward pass:")
    print(f"    • mean  : O(M)  ({N} reductions of size {D})")
    print(f"    • var   : O(M)")
    print(f"    • rstd  : O(N)  (one rsqrt per vector)")
    print(f"    • xhat  : O(M)")
    print(f"    • y     : O(M)")
    print(f"    Total time  : O(M) = O(B·T·D)")
    print(f"    Extra memory: O(M) for xhat + O(N) for rstd")
    print()
    print("  Backward pass:")
    print(f"    • dbeta  : O(M)  (sum reduction)")
    print(f"    • dgamma : O(M)  (sum reduction)")
    print(f"    • dx     : O(M)  (two D-wide reductions + elementwise)")
    print(f"    Total time  : O(M) = O(B·T·D)")
    print(f"    Extra memory: O(M) for dxhat (can be fused in-place)")
    print()

    print("─" * 60)
    print("NUMERICAL STABILITY DISCUSSION")
    print("─" * 60)
    print("""
  1. Division by near-zero σ̂:
     When all elements in a vector are identical, var = 0 and σ̂ = √ε.
     The epsilon (typically 1e-5) prevents division by zero.  Using
     double precision (float64) for the gradient check gives ~1e-10
     residual; in float32 the residual is ~1e-4, which is acceptable.

  2. Catastrophic cancellation in xc = x - mean:
     If x values are large but close together (e.g., x ≈ 1e6 with
     σ ≈ 1e-3), then xc = x - mean loses relative precision in float32.
     Remedy: the two-pass algorithm (compute mean first, then centered
     sum of squares) is already used here, which is the standard
     approach.  For extreme cases, a compensated (Kahan) summation
     or Welford's online algorithm can be used.

  3. Overflow in xc² or var:
     For very large values, squaring xc can overflow float16 or float32.
     The standard fix is to compute in float32 for float16 inputs, or
     use a scaled variant.

  4. Gradient explosion when σ̂ is very small:
     dx ∝ 1/σ̂, so tiny variance → large gradients.  This is inherent
     to the operation and is typically handled by gradient clipping
     upstream.  The epsilon also bounds 1/σ̂ ≤ 1/√ε.

  5. rstd computation:
     We compute 1/sqrt(var + eps) directly rather than forming
     sqrt(var + eps) and then dividing.  On GPU, the rsqrt instruction
     is a single hardware instruction with correct rounding.
    """)

    print("─" * 60)
    print("GPU FUSION STRATEGY")
    print("─" * 60)
    print("""
  Goal: Fuse the entire backward pass into a single CUDA kernel that
  loads each (B,T) row exactly once from global memory.

  Kernel design (one thread-block per row of length D):
  ───────────────────────────────────────────────────────────────

  Shared memory (per block, size ≈ 3·D·4 bytes for float32):
      smem_xhat[D]     — the normalized input
      smem_dxhat[D]    — dy * gamma
      smem_proj[1]     — scalar  Σ dxhat_d · xhat_d
      smem_sum[1]      — scalar  Σ dxhat_d

  Steps inside the kernel (no globalmem round-trips between steps):

    1. Each thread loads one (or more) element(s) of dy and xhat.
       Compute dxhat_d = dy_d * gamma_d  (gamma in constant mem or smem).
       Store dxhat_d and xhat_d into shared memory.

    2. Cooperative reduction across the D threads of the block:
          proj  += dxhat_d * xhat_d
          sum   += dxhat_d
       Two warp-level reductions (or one Blelloch scan) give us the
       two scalars in O(log D) steps.

    3. Each thread computes:
          dx_d = rstd * (dxhat_d  -  xhat_d * proj / D  -  sum / D)
       and writes the result to global memory.

    4. Atomic adds to global dgamma[d] += dy_d * xhat_d
                          dbeta[d]  += dy_d
       (one per element; can be deferred to a second pass or done
        with block-level reduction + single atomic per block).

  Memory traffic per row:
      Reads : dy (D) + xhat (D) + rstd (1) = 2D + 1 elements
      Writes: dx (D) + dgamma accumulator + dbeta accumulator
      Total : ≈ 3D elements  vs.  ≈ 10D+ for an unfused implementation
               (which would read/write intermediates to global memory
                between each of the 4–5 separate kernel launches).

  Additional optimizations:
    • Use warp-level shuffles (__shfl_down_sync) instead of shared
      memory for the reductions when D ≤ 32 (or D ≤ warpSize).
    • Vectorized loads (float4 / float2) to improve memory throughput.
    • For D values that don't divide evenly into warpSize, use a
      grid-stride loop with cooperative groups.
    • Fuse with the preceding or following elementwise op (GELU,
      residual add) to eliminate another global memory round-trip.
    """)


if __name__ == "__main__":
    main()
