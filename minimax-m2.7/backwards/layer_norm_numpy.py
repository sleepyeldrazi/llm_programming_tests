"""
Layer Normalization from scratch in NumPy
==========================================
Numerically stable backward pass implementation with gradient checking.

Author: pi coding assistant
Date: 2026-04-22
"""

import numpy as np
from typing import Tuple, Dict, Optional
import time
import copy


# =============================================================================
# Numerical constants
# =============================================================================
DEFAULT_EPS = 1e-8


# =============================================================================
# Helper functions
# =============================================================================

def logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = True) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    max_x = np.max(x, axis=axis, keepdims=True)
    return max_x + np.log(np.exp(x - max_x).sum(axis=axis, keepdims=True))


# =============================================================================
# Layer Normalization Forward Pass
# =============================================================================

def layer_norm_forward(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = DEFAULT_EPS
) -> Tuple[np.ndarray, Dict]:
    """
    Layer Norm forward pass.
    
    y = gamma * (x - mean) / sqrt(var + eps) + beta
    
    Args:
        x: Input tensor of shape (B, T, D)
        gamma: Scale parameter of shape (D,)
        beta: Bias parameter of shape (D,)
        eps: Small constant for numerical stability
    
    Returns:
        y: Normalized output of shape (B, T, D)
        cache: Dictionary of intermediates for backward pass
    """
    B, T, D = x.shape
    
    # Compute mean over feature dimension
    # mean[b, t] = (1/D) * sum_d x[b, t, d]
    mean = np.mean(x, axis=-1, keepdims=True)  # (B, T, 1)
    
    # Compute variance over feature dimension
    # var[b, t] = (1/D) * sum_d (x[b, t, d] - mean[b, t])^2
    x_centered = x - mean  # (B, T, D)
    var = np.mean(x_centered ** 2, axis=-1, keepdims=True)  # (B, T, 1)
    
    # Compute standard deviation with eps for numerical stability
    # std >= sqrt(eps) > 0, preventing division by zero
    std = np.sqrt(var + eps)  # (B, T, 1)
    
    # Normalize
    x_norm = x_centered / std  # (B, T, D)
    
    # Scale and shift
    y = gamma * x_norm + beta  # (B, T, D)
    
    # Cache intermediates for backward pass
    # We store only what we need to avoid recomputation
    cache = {
        'x': x,              # Original input (needed for gradient check)
        'x_centered': x_centered,
        'x_norm': x_norm,    # Normalized values (needed for d_gamma)
        'mean': mean,        # (B, T, 1)
        'var': var,          # (B, T, 1)
        'std': std,          # (B, T, 1)
        'gamma': gamma,      # Needed for gradient computation
        'beta': beta,        # Needed for gradient check
        'eps': eps,
        'B': B,
        'T': T,
        'D': D
    }
    
    return y, cache


# =============================================================================
# Layer Normalization Backward Pass
# =============================================================================

def layer_norm_backward(
    dy: np.ndarray,
    cache: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Layer Norm backward pass.
    
    Derivation:
    -----------
    Let:
        μ = mean[x]  (over axis=-1)
        σ² = var[x]  (over axis=-1)
        σ = sqrt(σ² + eps)
        x̄ = (x - μ) / σ  (normalized)
        y = γ * x̄ + β
    
    Then:
        ∂L/∂γ = sum(dy * x̄) over (B, T)
        ∂L/∂β = sum(dy) over (B, T)
        
    For ∂L/∂x:
        ∂L/∂x_i = (∂L/∂x̄_i / σ) 
                 - (∂L/∂x̄_i / D) * (1/σ)
                 - (x̄_i / D) * (∂L/∂σ²) * (2/σ)
                  
        where ∂L/∂σ² = -0.5 * sum_i(∂L/∂x̄_i * x̄_i) / σ³
                  
    Consolidating:
        ∂L/∂x_i = (γ_i / σ) * [∂L/∂y_i 
                               - mean(∂L/∂y) 
                               - x̄_i * mean(∂L/∂y * x̄)]
    
    Args:
        dy: Upstream gradient of shape (B, T, D)
        cache: Forward pass intermediates
    
    Returns:
        dx: Gradient w.r.t. input x, shape (B, T, D)
        d_gamma: Gradient w.r.t. gamma, shape (D,)
        d_beta: Gradient w.r.t. beta, shape (D,)
    """
    x = cache['x']
    x_centered = cache['x_centered']
    x_norm = cache['x_norm']
    mean = cache['mean']
    std = cache['std']
    gamma = cache['gamma']
    eps = cache['eps']
    B, T, D = cache['B'], cache['T'], cache['D']
    
    # -------------------------------------------------------------------------
    # 1. Compute gradients w.r.t. gamma and beta
    # -------------------------------------------------------------------------
    # d_gamma[d] = sum_{b,t} dy[b,t,d] * x_norm[b,t,d]
    d_gamma = np.sum(dy * x_norm, axis=(0, 1))  # (D,)
    
    # d_beta[d] = sum_{b,t} dy[b,t,d]
    d_beta = np.sum(dy, axis=(0, 1))  # (D,)
    
    # -------------------------------------------------------------------------
    # 2. Compute gradient w.r.t. normalized input
    # -------------------------------------------------------------------------
    # dz = dy * gamma  (chain rule: y = gamma * x_norm + beta)
    # Note: We can compute this and reuse in dx computation
    dz = dy * gamma  # (B, T, D)
    
    # -------------------------------------------------------------------------
    # 3. Compute gradient w.r.t. x
    # -------------------------------------------------------------------------
    # 
    # From the derivation:
    # dx = (dz - mean(dz) - x_norm * mean(dz * x_norm)) / std
    #
    # This comes from applying the chain rule considering:
    # - Direct dependence of x on x_norm through (x - mean) / std
    # - Indirect dependence through mean and std
    #
    # Key insight: We compute the two reduction terms efficiently:
    # - mean(dz) = (1/D) * sum(dz, axis=-1, keepdims=True)
    # - mean(dz * x_norm) = (1/D) * sum(dz * x_norm, axis=-1, keepdims=True)
    #
    
    # Compute reduction terms (these are O(BTD) each)
    sum_dz = np.sum(dz, axis=-1, keepdims=True)  # (B, T, 1)
    sum_dz_xnorm = np.sum(dz * x_norm, axis=-1, keepdims=True)  # (B, T, 1)
    
    # Compute dx using the consolidated formula
    # dx = (dz - (sum_dz / D) - x_norm * (sum_dz_xnorm / D)) / std
    dx = (dz - sum_dz / D - x_norm * sum_dz_xnorm / D) / std
    
    # -------------------------------------------------------------------------
    # Numerical stability analysis:
    # ---------------------------
    # 1. Division by std: We use std = sqrt(var + eps), so std >= sqrt(eps) > 0
    #    Example: eps = 1e-8 => std >= 1e-4, so no division by zero
    #
    # 2. Division by D: D is typically 512-4096, so this is stable
    #
    # 3. The formula (dz - mean(dz) - x_norm * mean(dz * x_norm)) / std
    #    When std is very small, the gradient can be large, but this is
    #    mathematically correct - small std means large normalization effect
    #
    # 4. For extreme stability, we could use the two-pass formula:
    #    dx = (dz - mean(dz) - x_norm * mean(dz * x_norm))
    #    dx = dx / std
    #    This avoids any intermediate overflow/underflow in the subtraction
    #
    # 5. Alternative numerically stable computation using centering trick:
    #    temp = dz / std
    #    dx = temp - x_norm * (sum(temp * x_norm) / D)
    #    dx = dx - sum(dx) / D  (but this is less efficient)
    #
    # 6. Catastrophic cancellation can occur in: (dz - mean(dz))
    #    When dz is roughly constant across D, mean(dz) ≈ dz, causing
    #    cancellation. However, this is exactly when dx should be small,
    #    so the cancellation is benign (relative error is small).
    #
    # 7. The x_norm * mean(dz * x_norm) term can also suffer from cancellation
    #    when mean(dz * x_norm) ≈ 0, but again this is when the term is small.
    #
    # 8. For fp16 or extreme cases, consider pairwise summation for reductions
    #    and/or higher precision accumulators.
    #
    # -------------------------------------------------------------------------
    
    return dx, d_gamma, d_beta


# =============================================================================
# Layer Norm Module (combines forward and backward)
# =============================================================================

class LayerNorm:
    """
    Layer Normalization module with manual gradient computation.
    
    Forward: y = gamma * (x - mean) / sqrt(var + eps) + beta
    Backward: Computes gradients w.r.t. x, gamma, beta
    """
    
    def __init__(self, normalized_shape: int, eps: float = DEFAULT_EPS):
        """
        Args:
            normalized_shape: Dimension D of the feature space
            eps: Epsilon for numerical stability in sqrt(var + eps)
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Initialize gamma (scale) and beta (shift) parameters
        # Xavier initialization for gamma to keep variance stable
        self.gamma = np.ones(normalized_shape)  # Scale initialized to 1
        self.beta = np.zeros(normalized_shape)   # Shift initialized to 0
        
        # Storage for gradients
        self.d_gamma = None
        self.d_beta = None
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Forward pass."""
        assert x.shape[-1] == self.normalized_shape, \
            f"Expected last dimension {self.normalized_shape}, got {x.shape[-1]}"
        return layer_norm_forward(x, self.gamma, self.beta, self.eps)
    
    def backward(self, dy: np.ndarray, cache: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass.
        
        Args:
            dy: Upstream gradient of shape (B, T, D)
            cache: Forward pass cache
        
        Returns:
            dx: Gradient w.r.t. input x
            d_gamma: Gradient w.r.t. gamma
            d_beta: Gradient w.r.t. beta
        """
        dx, d_gamma, d_beta = layer_norm_backward(dy, cache)
        self.d_gamma = d_gamma
        self.d_beta = d_beta
        return dx, d_gamma, d_beta
    
    def parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (gamma, beta)."""
        return self.gamma, self.beta
    
    def gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (d_gamma, d_beta)."""
        return self.d_gamma, self.d_beta


# =============================================================================
# Gradient Checking via Finite Differences
# =============================================================================

def compute_numerical_gradient_gamma(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    dy: np.ndarray,
    h: float = 1e-5
) -> np.ndarray:
    """
    Compute numerical gradient for gamma using finite differences.
    
    Args:
        x: Input tensor (B, T, D)
        gamma: Scale parameter (D,)
        beta: Bias parameter (D,)
        dy: Upstream gradient (B, T, D)
        h: Step size
    
    Returns:
        Numerical gradient for gamma (D,)
    """
    D = len(gamma)
    num_grad = np.zeros(D)
    
    for i in range(D):
        # Save original value
        original = gamma[i]
        
        # f(gamma + h)
        gamma[i] = original + h
        y_plus, _ = layer_norm_forward(x, gamma, beta)
        loss_plus = np.sum(y_plus * dy)
        
        # f(gamma - h)
        gamma[i] = original - h
        y_minus, _ = layer_norm_forward(x, gamma, beta)
        loss_minus = np.sum(y_minus * dy)
        
        # Central difference
        num_grad[i] = (loss_plus - loss_minus) / (2 * h)
        
        # Restore original
        gamma[i] = original
    
    return num_grad


def compute_numerical_gradient_beta(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    dy: np.ndarray,
    h: float = 1e-5
) -> np.ndarray:
    """
    Compute numerical gradient for beta using finite differences.
    
    Args:
        x: Input tensor (B, T, D)
        gamma: Scale parameter (D,)
        beta: Bias parameter (D,)
        dy: Upstream gradient (B, T, D)
        h: Step size
    
    Returns:
        Numerical gradient for beta (D,)
    """
    D = len(beta)
    num_grad = np.zeros(D)
    
    for i in range(D):
        # Save original value
        original = beta[i]
        
        # f(beta + h)
        beta[i] = original + h
        y_plus, _ = layer_norm_forward(x, gamma, beta)
        loss_plus = np.sum(y_plus * dy)
        
        # f(beta - h)
        beta[i] = original - h
        y_minus, _ = layer_norm_forward(x, gamma, beta)
        loss_minus = np.sum(y_minus * dy)
        
        # Central difference
        num_grad[i] = (loss_plus - loss_minus) / (2 * h)
        
        # Restore original
        beta[i] = original
    
    return num_grad


def compute_numerical_gradient_x(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    dy: np.ndarray,
    h: float = 1e-5,
    max_elements: int = 100000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute numerical gradient for x using finite differences.
    For large tensors, uses spot check.
    
    Returns both the numerical gradient AND the numerical gradient for spot-checked elements.
    
    Args:
        x: Input tensor (B, T, D) - will be restored to original values
        gamma: Scale parameter (D,)
        beta: Bias parameter (D,)
        dy: Upstream gradient (B, T, D)
        h: Step size
        max_elements: Maximum elements to check (spot check if larger)
    
    Returns:
        Tuple of (num_grad, spot_check_mask) where spot_check_mask marks checked elements
    """
    B, T, D = x.shape
    total_elements = B * T * D
    
    # Save original x values
    orig_x = x.copy()
    
    if total_elements <= max_elements:
        # Full gradient check
        num_grad = np.zeros_like(x)
        # Use reshape to get a view, not a copy
        x_flat = x.reshape(-1)
        num_grad_flat = num_grad.reshape(-1)
        
        for i in range(total_elements):
            if (i + 1) % 10000 == 0:
                print(f"    Progress: {i+1}/{total_elements}")
            
            original = x_flat[i]
            
            # f(x + h)
            x_flat[i] = original + h
            y_plus, _ = layer_norm_forward(x, gamma, beta)
            loss_plus = np.sum(y_plus * dy)
            
            # f(x - h)
            x_flat[i] = original - h
            y_minus, _ = layer_norm_forward(x, gamma, beta)
            loss_minus = np.sum(y_minus * dy)
            
            # Central difference
            num_grad_flat[i] = (loss_plus - loss_minus) / (2 * h)
            
            # Restore
            x_flat[i] = original
        
        # Restore x to original values
        x[:] = orig_x
        
        return num_grad, np.ones((B, T, D), dtype=bool)  # All elements checked
    else:
        # Spot check
        print(f"  Spot checking {max_elements} random elements...")
        n_samples = max_elements
        num_grad = np.zeros_like(x)
        spot_checked = np.zeros((B, T, D), dtype=bool)
        
        indices = [tuple(np.random.randint(b) for b in (B, T, D)) for _ in range(n_samples)]
        
        for idx in indices:
            bi, ti, di = idx
            original = x[bi, ti, di]
            spot_checked[bi, ti, di] = True
            
            # f(x + h)
            x[bi, ti, di] = original + h
            y_plus, _ = layer_norm_forward(x, gamma, beta)
            loss_plus = np.sum(y_plus * dy)
            
            # f(x - h)
            x[bi, ti, di] = original - h
            y_minus, _ = layer_norm_forward(x, gamma, beta)
            loss_minus = np.sum(y_minus * dy)
            
            num_grad[bi, ti, di] = (loss_plus - loss_minus) / (2 * h)
            
            # Restore
            x[bi, ti, di] = original
        
        # Restore x to original values
        x[:] = orig_x
        
        return num_grad, spot_checked


def gradient_check(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    dy: np.ndarray,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    verbose: bool = True
) -> Dict[str, bool]:
    """
    Perform gradient check for all parameters.
    
    Checks: |analytical - numerical| <= atol + rtol * |numerical|
    
    Args:
        x: Input tensor
        gamma: Scale parameter
        beta: Bias parameter
        dy: Upstream gradient
        rtol: Relative tolerance
        atol: Absolute tolerance
        verbose: Print detailed results
    
    Returns:
        Dictionary of pass/fail for each parameter
    """
    results = {}
    
    # Store originals
    orig_gamma = gamma.copy()
    orig_beta = beta.copy()
    orig_x = x.copy()
    
    # -------------------------------------------------------------------------
    # Forward pass to get analytical gradients
    # -------------------------------------------------------------------------
    y, cache = layer_norm_forward(x, gamma, beta)
    dx_analytical, d_gamma_analytical, d_beta_analytical = layer_norm_backward(dy, cache)
    
    # -------------------------------------------------------------------------
    # Check gradient w.r.t. gamma
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "="*60)
        print("GRADIENT CHECK: gamma")
        print("="*60)
    
    # Reset gamma to original
    gamma[:] = orig_gamma
    
    d_gamma_numerical = compute_numerical_gradient_gamma(x, gamma, beta, dy)
    
    # Compare
    diff = np.abs(d_gamma_analytical - d_gamma_numerical)
    tolerance = atol + rtol * np.abs(d_gamma_numerical)
    passed = np.all(diff <= tolerance)
    
    if verbose:
        print(f"Analytical gradient shape: {d_gamma_analytical.shape}")
        print(f"Numerical gradient shape: {d_gamma_numerical.shape}")
        print(f"Max absolute difference: {np.max(diff):.2e}")
        print(f"Max relative tolerance: {np.max(tolerance):.2e}")
        print(f"Mean analytical: {np.mean(np.abs(d_gamma_analytical)):.6e}")
        print(f"Mean numerical: {np.mean(np.abs(d_gamma_numerical)):.6e}")
        print(f"\nGradient check: {'PASSED ✓' if passed else 'FAILED ✗'}")
    
    results['gamma'] = passed
    
    # -------------------------------------------------------------------------
    # Check gradient w.r.t. beta
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "="*60)
        print("GRADIENT CHECK: beta")
        print("="*60)
    
    # Reset beta to original
    beta[:] = orig_beta
    
    d_beta_numerical = compute_numerical_gradient_beta(x, gamma, beta, dy)
    
    diff = np.abs(d_beta_analytical - d_beta_numerical)
    tolerance = atol + rtol * np.abs(d_beta_numerical)
    passed = np.all(diff <= tolerance)
    
    if verbose:
        print(f"Analytical gradient shape: {d_beta_analytical.shape}")
        print(f"Numerical gradient shape: {d_beta_numerical.shape}")
        print(f"Max absolute difference: {np.max(diff):.2e}")
        print(f"Max relative tolerance: {np.max(tolerance):.2e}")
        print(f"Mean analytical: {np.mean(np.abs(d_beta_analytical)):.6e}")
        print(f"Mean numerical: {np.mean(np.abs(d_beta_numerical)):.6e}")
        print(f"\nGradient check: {'PASSED ✓' if passed else 'FAILED ✗'}")
    
    results['beta'] = passed
    
    # -------------------------------------------------------------------------
    # Check gradient w.r.t. x
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "="*60)
        print("GRADIENT CHECK: x (input)")
        print("="*60)
    
    # Reset x to original
    x[:] = orig_x
    
    d_x_numerical, spot_checked = compute_numerical_gradient_x(x, gamma, beta, dy)
    
    # Only check elements that were numerically computed
    diff = np.abs(dx_analytical[spot_checked] - d_x_numerical[spot_checked])
    tolerance = atol + rtol * np.abs(d_x_numerical[spot_checked])
    passed = np.all(diff <= tolerance)
    
    if verbose:
        print(f"Analytical gradient shape: {dx_analytical.shape}")
        print(f"Numerical gradient shape: {d_x_numerical.shape}")
        print(f"Elements checked: {np.sum(spot_checked)} / {spot_checked.size}")
        if np.any(spot_checked):
            print(f"Max absolute difference: {np.max(diff):.2e}")
            print(f"Max relative tolerance: {np.max(tolerance):.2e}")
            print(f"Mean analytical (checked): {np.mean(np.abs(dx_analytical[spot_checked])):.6e}")
            print(f"Mean numerical (checked): {np.mean(np.abs(d_x_numerical[spot_checked])):.6e}")
        print(f"\nGradient check: {'PASSED ✓' if passed else 'FAILED ✗'}")
    
    results['x'] = passed
    
    # Restore originals
    gamma[:] = orig_gamma
    beta[:] = orig_beta
    x[:] = orig_x
    
    return results


# =============================================================================
# Complexity Analysis
# =============================================================================

def analyze_complexity():
    """Print complexity analysis for layer norm forward and backward."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COMPLEXITY ANALYSIS                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Input: x ∈ ℝ^(B×T×D)                                                        ║
║  Parameters: γ, β ∈ ℝ^D                                                      ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FORWARD PASS                                                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Operation              │ Work (FLOPs)      │ Memory                        ║
║  ────────────────────────────────────────────────────────────────────────── ║
║  mean (reduction)       │ O(BTD)            │ -                             ║
║  x - mean (broadcast)   │ O(BTD)            │ O(BTD)                        ║
║  var (reduction)        │ O(BTD)            │ -                             ║
║  sqrt(var + eps)        │ O(BT)             │ -                             ║
║  divide by std          │ O(BTD)            │ -                             ║
║  gamma * x_norm         │ O(BTD)            │ -                             ║
║  add beta               │ O(BTD)            │ O(BTD) output                 ║
║  ────────────────────────────────────────────────────────────────────────── ║
║  TOTAL                    │ 5×O(BTD)          │ O(BTD)                       ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  BACKWARD PASS                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Operation              │ Work (FLOPs)      │ Memory                        ║
║  ────────────────────────────────────────────────────────────────────────── ║
║  d_gamma = sum(dy*x_norm)│ O(BTD)           │ O(D)                          ║
║  d_beta = sum(dy)       │ O(BTD)            │ O(D)                          ║
║  dz = dy * gamma        │ O(BTD)            │ O(BTD) (can be avoided)        ║
║  sum(dz)                │ O(BTD)            │ -                             ║
║  sum(dz * x_norm)       │ O(BTD)            │ -                             ║
║  dx computation         │ O(BTD)            │ O(BTD) output                 ║
║  ────────────────────────────────────────────────────────────────────────── ║
║  TOTAL                    │ 5×O(BTD)          │ O(BTD) + O(D)                ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SUMMARY                                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Time Complexity:  O(BTD) for both forward and backward                      ║
║  Space Complexity: O(BTD) for storing activations (during training)          ║
║                   O(BTD) during inference (no need to store)                 ║
║                                                                              ║
║  Cache efficiency: We store x_centered, x_norm, mean, std                    ║
║                   These are O(BTD) total, reused across all gradient comps  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# GPU Kernel Fusion Design
# =============================================================================

def explain_gpu_fusion():
    """Explain GPU kernel fusion for layer normalization."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║               GPU KERNEL FUSION DESIGN                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  CURRENT SEPARATE KERNEL APPROACH:                                           ║
║  ────────────────────────────────────                                        ║
║  Kernel 1: Compute mean (reduction over D)                                   ║
║  Kernel 2: Compute variance (reduction over D)                               ║
║  Kernel 3: Normalize (element-wise)                                          ║
║  Kernel 4: Scale and shift (element-wise)                                    ║
║  Kernel 5: Backward kernels (x, gamma, beta)                                 ║
║                                                                              ║
║  Issues with separate kernels:                                               ║
║  • Multiple kernel launches (overhead)                                       ║
║  • Data movement between global memory passes                                 ║
║  • Can't use persistent threads for reduction efficiency                      ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FUSED KERNEL DESIGN (Forward):                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Grid: (B × T) blocks, each handling one (b,t) position                      ║
║  Block: 256-512 threads                                                      ║
║                                                                              ║
║  PHASE 1: Load and compute local sum                                         ║
║  ─────────────────────────────────                                           ║
║  • Each thread loads x[b,t,d] into shared memory                             ║
║  • Compute partial sum using warp-level reduction                            ║
║  • Single thread writes mean to __shared__                                    ║
║                                                                              ║
║  PHASE 2: Compute variance locally                                           ║
║  ─────────────────────────────────                                           ║
║  • Re-load x with loaded mean                                                ║
║  • Compute (x-mean)² and partial variance                                    ║
║  • Reduce to get variance                                                   ║
║                                                                              ║
║  PHASE 3: Normalize and output                                               ║
║  ─────────────────────────────────                                           ║
║  • All threads compute: y = gamma * (x-mean) / sqrt(var+eps) + beta          ║
║  • Write to output (fully coalesced)                                         ║
║                                                                              ║
║  MEMORY ACCESS PATTERN:                                                      ║
║  • Each block reads contiguous D elements (coalesced)                        ║
║  • Use shared memory for intermediate results                                ║
║  • Output writes are also coalesced                                          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FUSED KERNEL DESIGN (Backward):                                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Key insight: dz = dy * gamma can be merged into the computation              ║
║                                                                              ║
║  Grid: (B × T) blocks                                                       ║
║  Block: 256-512 threads                                                      ║
║                                                                              ║
║  SHARED MEMORY STRUCTURE:                                                    ║
║  [x_norm_0, x_norm_1, ..., x_norm_{D-1}]                                     ║
║  [dz_0, dz_1, ..., dz_{D-1}]                                                 ║
║                                                                              ║
║  ALGORITHM:                                                                  ║
║  1. Load x_norm and compute local dz = dy * gamma                            ║
║  2. Reduce to get sum(dz) and sum(dz * x_norm)                              ║
║  3. Second pass to compute dx using the formula:                             ║
║     dx = (dz - mean(dz) - x_norm * mean(dz*x_norm)) / std                    ║
║                                                                              ║
║  REDUCTION OPTIMIZATIONS:                                                    ║
║  • Warp-level shuffle reductions (no shared memory needed)                    ║
║  • Block-level using shared memory with tree reduction                        ║
║  • Use block-level primitives for final reduction                            ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  BENEFITS OF FUSION:                                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ✓ Reduced kernel launch overhead (1 kernel vs 4-5)                           ║
║  ✓ Better memory bandwidth utilization (single read, single write)          ║
║  ✓ Improved cache locality (data stays in registers/shared mem)              ║
║  ✓ Only loads x once, computes mean and var from same data                   ║
║  ✓ Backward can reuse cached values from forward (if memory allows)          ║
║  ✓ Lower register pressure allows for larger block sizes                     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CUDA KERNEL SKETCH (Pseudo-code):                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  __global__ void layer_norm_fwd(float* y, const float* x,                    ║
║                                  const float* gamma, const float* beta,      ║
║                                  int B, int T, int D, float eps) {          ║
║                                                                              ║
║      __shared__ float mean_smem[256]; // block-level mean                     ║
║      __shared__ float var_smem[256];  // block-level variance                 ║
║      __shared__ float std_smem[256]; // block-level std                      ║
║                                                                              ║
║      int tid = threadIdx.x;                                                  ║
║      int bid = blockIdx.x;                                                   ║
║      int D_blk = (D + blockDim.x - 1) / blockDim.x;                          ║
║                                                                              ║
║      // Phase 1: Load and compute mean                                        ║
║      float sum = 0.0;                                                        ║
║      for (int i = 0; i < D_blk; i++) {                                       ║
║          int idx = bid * D + i * blockDim.x + tid;                           ║
║          sum += x[idx];                                                      ║
║      }                                                                       ║
║      sum = warpReduceSum(sum);                                               ║
║      if (tid % 32 == 0) mean_smem[tid / 32] = sum;                           ║
║      __syncthreads();                                                        ║
║                                                                              ║
║      float mean = mean_smem[0] / D;                                         ║
║                                                                              ║
║      // Phase 2: Compute variance                                            ║
║      sum = 0.0;                                                             ║
║      for (int i = 0; i < D_blk; i++) {                                       ║
║          int idx = bid * D + i * blockDim.x + tid;                           ║
║          float diff = x[idx] - mean;                                         ║
║          sum += diff * diff;                                                 ║
║      }                                                                       ║
║      sum = warpReduceSum(sum);                                               ║
║      if (tid % 32 == 0) var_smem[tid / 32] = sum;                           ║
║      __syncthreads();                                                       ║
║                                                                              ║
║      float var = var_smem[0] / D;                                          ║
║      float std = sqrt(var + eps);                                           ║
║                                                                              ║
║      // Phase 3: Normalize and output                                        ║
║      for (int i = 0; i < D_blk; i++) {                                       ║
║          int idx = bid * D + i * blockDim.x + tid;                            ║
║          float x_norm = (x[idx] - mean) / std;                               ║
║          y[idx] = gamma[idx % D] * x_norm + beta[idx % D];                  ║
║      }                                                                       ║
║  }                                                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# Benchmark and Tests
# =============================================================================

def benchmark():
    """Benchmark forward and backward passes."""
    print("\n" + "="*70)
    print("BENCHMARKING LAYER NORMALIZATION")
    print("="*70)
    
    # Test different shapes
    shapes = [
        (32, 128, 256),    # Small
        (64, 128, 512),    # Medium (BERT-base hidden)
        (32, 512, 768),    # BERT-base
        (16, 512, 1024),   # Larger
    ]
    
    results = []
    
    for B, T, D in shapes:
        print(f"\nShape: (B={B}, T={T}, D={D})")
        print("-" * 40)
        
        # Create random inputs
        np.random.seed(42)
        x = np.random.randn(B, T, D).astype(np.float64)
        gamma = np.random.randn(D).astype(np.float64)
        beta = np.random.randn(D).astype(np.float64)
        dy = np.random.randn(B, T, D).astype(np.float64)
        
        # Forward benchmark
        n_iters = 100
        times = []
        for _ in range(n_iters):
            start = time.perf_counter()
            y, cache = layer_norm_forward(x, gamma, beta)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        fwd_time = np.mean(times)
        fwd_std = np.std(times)
        
        # Backward benchmark
        times = []
        for _ in range(n_iters):
            start = time.perf_counter()
            dx, d_gamma, d_beta = layer_norm_backward(dy, cache)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        bwd_time = np.mean(times)
        bwd_std = np.std(times)
        
        # Throughput
        elements = B * T * D
        fwd_throughput = elements / (fwd_time / 1000) / 1e9  # GB/s
        bwd_throughput = elements / (bwd_time / 1000) / 1e9
        
        print(f"Forward:  {fwd_time:.3f} ± {fwd_std:.3f} ms ({fwd_throughput:.1f} GB/s)")
        print(f"Backward: {bwd_time:.3f} ± {bwd_std:.3f} ms ({bwd_throughput:.1f} GB/s)")
        print(f"Total:    {fwd_time + bwd_time:.3f} ms")
        
        results.append({
            'shape': (B, T, D),
            'fwd_time': fwd_time,
            'bwd_time': bwd_time,
            'elements': elements
        })
    
    return results


def run_gradient_checks():
    """Run gradient checks on various shapes."""
    print("\n" + "="*70)
    print("RUNNING GRADIENT CHECKS")
    print("="*70)
    
    shapes = [
        (2, 4, 8),     # Tiny
        (4, 8, 16),    # Small
        (8, 16, 32),   # Medium-small
    ]
    
    all_passed = True
    
    for B, T, D in shapes:
        print(f"\n{'='*60}")
        print(f"Testing shape (B={B}, T={T}, D={D})")
        print(f"{'='*60}")
        
        # Create inputs
        np.random.seed(123)
        x = np.random.randn(B, T, D).astype(np.float64)
        gamma = np.random.randn(D).astype(np.float64)
        beta = np.random.randn(D).astype(np.float64)
        dy = np.random.randn(B, T, D).astype(np.float64)
        
        # Run gradient check
        results = gradient_check(x, gamma, beta, dy, verbose=True)
        
        for name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {name:>8}: {status}")
            all_passed = all_passed and passed
    
    print("\n" + "="*70)
    if all_passed:
        print("ALL GRADIENT CHECKS PASSED ✓")
    else:
        print("SOME GRADIENT CHECKS FAILED ✗")
    print("="*70)
    
    return all_passed


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LAYER NORMALIZATION: FORWARD AND BACKWARD FROM SCRATCH")
    print("="*70)
    
    # Print complexity analysis
    analyze_complexity()
    
    # Explain GPU fusion
    explain_gpu_fusion()
    
    # Run gradient checks
    run_gradient_checks()
    
    # Benchmark
    benchmark()