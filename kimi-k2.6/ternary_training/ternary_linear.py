import mlx.core as mx
import mlx.nn as nn
import numpy as np

# ==============================================================================
# Ternary Linear Layer with Straight-Through Estimator
# ==============================================================================

class TernaryLinear(nn.Module):
    """
    Ternary linear layer: weights are projected to {-1, 0, +1} * scale
    during forward pass, with STE for backward pass.
    
    Group-wise quantization: groups of 128 weights share one FP16 scale factor.
    Scale factor: s = mean(|W_group|)
    """
    def __init__(self, in_features: int, out_features: int, group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        # Validate that in_features is divisible by group_size
        if in_features % group_size != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by group_size ({group_size})")
        
        self.num_groups = in_features // group_size
        
        # Latent weights in float32 (trainable)
        scale = (1.0 / in_features) ** 0.5
        self.weight = mx.random.normal((out_features, in_features), scale=scale)
    
    def _quantize(self, weight):
        """
        Project latent weights to ternary using group-wise scales.
        Returns ternary weights and scales (for verification).
        """
        # Reshape to (out_features, num_groups, group_size)
        w_reshaped = weight.reshape(self.out_features, self.num_groups, self.group_size)
        
        # Compute scale per group: s = mean(|W|)
        scales = mx.mean(mx.abs(w_reshaped), axis=-1, keepdims=True)  # (out, num_groups, 1)
        
        # Quantize to {-1, 0, +1}
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        w_norm = w_reshaped / (scales + epsilon)
        
        # Round to nearest in {-1, 0, +1}
        w_quant = mx.clip(mx.round(w_norm), -1, 1)
        
        # Dequantize back
        w_ternary = w_quant * scales
        
        return w_ternary.reshape(self.out_features, self.in_features), scales
    
    def __call__(self, x):
        """
        Forward pass with STE:
        - Compute ternary weights (no gradient through rounding)
        - Use ternary weights for matmul
        - STE: gradient flows straight through to latent weights
        """
        # Get ternary weights (stop gradient on quantization)
        w_ternary, _ = self._quantize(mx.stop_gradient(self.weight))
        
        # STE: y = x @ w_ternary, but gradient goes to self.weight
        # In MLX, we can use custom_vjp or the stop_gradient trick
        # Standard STE: output uses quantized weights, but gradient is identity
        
        # For STE: y = w_ternary @ x.T, grad_w = grad_y @ x
        # We want grad to flow to latent weight, so we do:
        # y = x @ w_ternary^T + 0 * self.weight (so gradient also flows to self.weight)
        
        # Actually, the simplest STE in MLX:
        # w_effective = w_ternary + (self.weight - mx.stop_gradient(self.weight))
        # This way forward uses w_ternary, backward uses self.weight
        
        w_effective = w_ternary + (self.weight - mx.stop_gradient(self.weight))
        
        return x @ w_effective.T
    
    def get_ternary_weights(self):
        """Get the actual ternary-projected weights (for verification)."""
        w_ternary, scales = self._quantize(self.weight)
        return w_ternary, scales
    
    def verify_ternary(self, tol=1e-3):
        """Verify that weights project cleanly to {-1, 0, +1} * scale."""
        w_ternary, scales = self.get_ternary_weights()
        w_reshaped = w_ternary.reshape(self.out_features, self.num_groups, self.group_size)
        
        # Check values are in {-scale, 0, +scale}
        # Use a more robust check: see if w_norm is close to integers after dividing by scale
        w_norm = w_reshaped / (scales + 1e-8)
        w_rounded = mx.round(w_norm)
        
        # Should be -1, 0, or 1 (check that rounded values are exactly these)
        is_valid_value = mx.all(
            (mx.abs(w_rounded - (-1.0)) < 1e-3) | 
            (mx.abs(w_rounded - 0.0) < 1e-3) | 
            (mx.abs(w_rounded - 1.0) < 1e-3)
        )
        
        # And rounding should not change much
        is_ternary = mx.all(mx.abs(w_norm - w_rounded) < tol)
        
        return is_ternary.item() and is_valid_value.item()


# ==============================================================================
# Test TernaryLinear
# ==============================================================================
if __name__ == "__main__":
    print("Testing TernaryLinear implementation...")
    
    # Test 1: Basic forward pass
    layer = TernaryLinear(256, 128, group_size=128)
    x = mx.random.normal((4, 256))
    y = layer(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    assert y.shape == (4, 128), "Output shape mismatch"
    
    # Debug: Check actual weight distribution
    w_ternary, scales = layer.get_ternary_weights()
    w_reshaped = w_ternary.reshape(layer.out_features, layer.num_groups, layer.group_size)
    w_norm = w_reshaped / (scales + 1e-8)
    
    print(f"\nWeight statistics:")
    print(f"  Original weight range: [{mx.min(layer.weight).item():.4f}, {mx.max(layer.weight).item():.4f}]")
    print(f"  Scale range: [{mx.min(scales).item():.6f}, {mx.max(scales).item():.6f}]")
    print(f"  Normalized weight range: [{mx.min(w_norm).item():.4f}, {mx.max(w_norm).item():.4f}]")
    print(f"  Unique normalized values: {len(np.unique(np.round(w_norm.flatten(), 6)))}")
    
    # Check unique values
    w_flat = w_norm.flatten()
    print(f"  Sample normalized values: {w_flat[:20].tolist()}")
    
    # Test 2: Verify ternary projection
    is_ternary = layer.verify_ternary()
    print(f"\nTernary verification: {'PASS' if is_ternary else 'FAIL'}")
    
    # Manual check
    w_rounded = mx.round(w_norm)
    diff = mx.abs(w_norm - w_rounded)
    print(f"  Max diff from rounded: {mx.max(diff).item():.6f}")
    print(f"  All values close to -1, 0, or 1: {mx.all(diff < 1e-3).item()}")
    
    # Check scale match
    computed_scales = mx.mean(mx.abs(w_reshaped), axis=-1, keepdims=True)
    scale_diff = mx.abs(scales - computed_scales)
    print(f"  Max scale diff: {mx.max(scale_diff).item():.8f}")
    print(f"  Scale match: {mx.all(scale_diff < 1e-3).item()}")
    
    assert is_ternary, "Weights are not ternary!"
    
    # Test 3: Check gradient flow
    def loss_fn(layer, x):
        y = layer(x)
        return mx.sum(y ** 2)
    
    loss, grads = mx.value_and_grad(loss_fn)(layer, x)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Weight grad shape: {grads['weight'].shape}")
    print(f"Weight grad norm: {mx.linalg.norm(grads['weight']).item():.4f}")
    assert grads['weight'] is not None, "No gradient flowing to weight!"
    
    print("\nAll tests passed!")
