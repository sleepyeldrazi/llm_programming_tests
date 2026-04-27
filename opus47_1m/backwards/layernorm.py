"""
Numerically-stable LayerNorm forward + manual backward in pure NumPy.

Forward (per row of length D, last axis):
    mean   = (1/D) * sum(x)
    xc     = x - mean
    var    = (1/D) * sum(xc**2)
    inv_std = 1 / sqrt(var + eps)
    xhat   = xc * inv_std
    y      = gamma * xhat + beta

Backward derivation (one row; vectorize over the leading (B, T) axes):
Let g_i = dL/dy_i, ghat_i = dL/dxhat_i = g_i * gamma_i.
xhat depends on every x_j in the row through (mean, var), so we apply the chain rule
to mean and var:

    dL/dvar  = sum_i ghat_i * xc_i * (-1/2) * (var + eps)^(-3/2)
    dL/dmean = sum_i ghat_i * (-inv_std)
               + (dL/dvar) * (1/D) * sum_i (-2 * xc_i)
             = -inv_std * sum_i ghat_i      # second term is 0 since sum xc = 0

    dL/dx_i  = ghat_i * inv_std
               + (dL/dvar) * (2/D) * xc_i
               + (dL/dmean) * (1/D)

Substituting and simplifying using xhat_i = xc_i * inv_std collapses to the
common compact form:

    dL/dx_i = (inv_std / D) * (D * ghat_i - sum_j ghat_j - xhat_i * sum_j (ghat_j * xhat_j))

Parameter grads (sum over all rows, i.e. axes (0, 1) for (B, T, D) input):
    dL/dgamma_i = sum g_i * xhat_i
    dL/dbeta_i  = sum g_i

Numerical stability notes:
- Compute the variance from the centered values (xc**2) rather than E[x^2] - E[x]^2,
  which suffers catastrophic cancellation for large means.
- Use rsqrt of (var + eps) instead of dividing by sqrt(var) to avoid div-by-zero
  on a constant row, and to fold eps in before the sqrt.
- The compact dx form avoids forming D * sum(...) intermediates per element and
  keeps everything in O(D) reductions per row.
- Cast to float32/float64 as needed; mixed precision should accumulate the row
  reductions (sum, sum of squares, sum(ghat), sum(ghat*xhat)) in float32 even
  when storage is float16/bfloat16.

Time / memory complexity for input of shape (B, T, D) with N = B*T rows:
- Forward:  O(N * D) time, O(N * D) memory for y. Cache mean (N), inv_std (N),
  and xhat (N*D) for the backward — total backward-cache memory ~ N*D + 2N.
- Backward: O(N * D) time. Working memory O(N*D) for ghat and dx; the per-row
  reductions sum(ghat), sum(ghat*xhat) are O(N) extra.
"""

from __future__ import annotations

import numpy as np


def layernorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    """LayerNorm over the last axis. Returns (y, cache)."""
    mean = x.mean(axis=-1, keepdims=True)
    xc = x - mean
    var = (xc * xc).mean(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    xhat = xc * inv_std
    y = gamma * xhat + beta
    cache = (xhat, inv_std, gamma)
    return y, cache


def layernorm_backward(dy: np.ndarray, cache):
    """Manual backward. Returns (dx, dgamma, dbeta)."""
    xhat, inv_std, gamma = cache
    D = xhat.shape[-1]

    # Param grads: reduce over all leading axes.
    reduce_axes = tuple(range(dy.ndim - 1))
    dbeta = dy.sum(axis=reduce_axes)
    dgamma = (dy * xhat).sum(axis=reduce_axes)

    # dx via the compact form. Two row-wise reductions only.
    ghat = dy * gamma                                    # dL/dxhat
    sum_ghat = ghat.sum(axis=-1, keepdims=True)          # sum_j ghat_j
    sum_ghat_xhat = (ghat * xhat).sum(axis=-1, keepdims=True)  # sum_j ghat_j * xhat_j
    dx = (inv_std / D) * (D * ghat - sum_ghat - xhat * sum_ghat_xhat)
    return dx, dgamma, dbeta


# --------------------------------------------------------------------------------------
# Gradient check via centered finite differences.
# --------------------------------------------------------------------------------------

def _scalar_loss(y: np.ndarray, w: np.ndarray) -> float:
    """A simple scalar loss L = sum(w * y); dL/dy = w. Lets us seed an arbitrary dy."""
    return float((w * y).sum())


def _numeric_grad(f, param: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Centered finite differences over `param` (modifies in place then restores)."""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]
        param[idx] = original + h
        f_plus = f()
        param[idx] = original - h
        f_minus = f()
        param[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * h)
        it.iternext()
    return grad


def gradient_check(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    B, T, D = 2, 3, 5
    x = rng.standard_normal((B, T, D)).astype(np.float64)
    gamma = rng.standard_normal(D).astype(np.float64)
    beta = rng.standard_normal(D).astype(np.float64)
    w = rng.standard_normal((B, T, D)).astype(np.float64)  # arbitrary upstream dy
    eps = 1e-5

    y, cache = layernorm_forward(x, gamma, beta, eps)
    # dL/dy = w because L = sum(w * y).
    dx, dgamma, dbeta = layernorm_backward(w, cache)

    def loss_x():
        y_, _ = layernorm_forward(x, gamma, beta, eps)
        return _scalar_loss(y_, w)

    def loss_gamma():
        y_, _ = layernorm_forward(x, gamma, beta, eps)
        return _scalar_loss(y_, w)

    def loss_beta():
        y_, _ = layernorm_forward(x, gamma, beta, eps)
        return _scalar_loss(y_, w)

    dx_num = _numeric_grad(loss_x, x)
    dgamma_num = _numeric_grad(loss_gamma, gamma)
    dbeta_num = _numeric_grad(loss_beta, beta)

    def rel_err(a, b):
        return np.max(np.abs(a - b) / np.maximum(1e-12, np.abs(a) + np.abs(b)))

    print(f"max rel err dx     = {rel_err(dx, dx_num):.3e}")
    print(f"max rel err dgamma = {rel_err(dgamma, dgamma_num):.3e}")
    print(f"max rel err dbeta  = {rel_err(dbeta, dbeta_num):.3e}")

    assert rel_err(dx, dx_num) < 1e-7
    assert rel_err(dgamma, dgamma_num) < 1e-7
    assert rel_err(dbeta, dbeta_num) < 1e-7
    print("gradient check passed.")


# --------------------------------------------------------------------------------------
# GPU fusion sketch (text only — no CUDA here).
# --------------------------------------------------------------------------------------
GPU_FUSION_NOTES = """
Fused GPU kernel sketch
=======================

Layout: launch one thread block per row (i.e. per (b, t) pair); D elements per row
are processed cooperatively by the block's threads. With D up to a few thousand,
one row fits in shared memory and registers.

Forward kernel (single pass over the row, plus a small reduction):
  1. Each thread loads its slice of x into registers.
  2. Block-wide reduction (warp shuffles + shared memory) for sum and sum-of-squares.
     Use Welford's online algorithm or the two-pass mean-then-var; Welford avoids
     the catastrophic-cancellation form E[x^2] - E[x]^2.
  3. Thread 0 computes inv_std = rsqrtf(var + eps) and broadcasts via shared mem.
  4. Each thread writes y = gamma * (x - mean) * inv_std + beta back to global mem.
     Optionally also writes xhat for the backward, or recomputes it there.
  5. Cache mean and inv_std (one float per row) for the backward pass.

Backward kernel (single fused pass, two block-wide reductions):
  1. Each thread loads its (x or xhat), gamma, dy slice.
     If only mean and inv_std were saved, recompute xhat = (x - mean) * inv_std.
  2. Compute ghat = dy * gamma per element.
  3. Block reduction for sum_ghat and sum_ghat_xhat. Two reductions can be fused
     into one pass over the row by carrying a float2 accumulator.
  4. Each thread writes
        dx = inv_std / D * (D * ghat - sum_ghat - xhat * sum_ghat_xhat).
  5. dgamma and dbeta accumulate across rows: each thread holds a per-D-slot
     register accumulator that's reduced across the grid via either
       - a second small kernel that sums per-block partial buffers, or
       - atomicAdd into global dgamma/dbeta when D is modest.

Why fusing wins:
  - Activations are read once and never re-materialized.
  - mean, inv_std, sum_ghat, sum_ghat_xhat live in registers / shared memory,
    so the kernel is purely memory-bandwidth bound at ~3*N*D bytes (read x,
    read dy, write dx) plus a tiny amount for params and stats.
  - Mixed precision: keep x/dy in fp16 or bf16, accumulate reductions in fp32,
    do rsqrt in fp32, write outputs back in the storage dtype.

Edge cases worth handling:
  - D > block-size: each thread handles a strided chunk of the row.
  - D very small: pad the block-wide reduction or fall back to a kernel that
    processes multiple rows per block to keep occupancy.
  - Constant rows (var = 0): eps inside the sqrt prevents NaNs; rsqrt is finite.
"""


if __name__ == "__main__":
    gradient_check()
    print()
    print(GPU_FUSION_NOTES)
