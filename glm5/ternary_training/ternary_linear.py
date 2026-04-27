"""
Group-wise ternary linear layer with Straight-Through Estimator (STE).

Implements the core building block for Ternary Bonsai training:
- Latent (full-precision) weights stored in float32
- Forward pass: project latent weights to ternary {-s, 0, +s} using group-wise scales
- Backward pass: STE — gradient flows through as identity
- Group size = 128, scale s = mean(|W_group|) per group
"""

import mlx.core as mx
import mlx.nn as nn


GROUP_SIZE = 128


@mx.custom_function
def ternary_projection(w):
    """
    Project latent weights to ternary values using group-wise quantization.
    
    Args:
        w: latent weight tensor of any shape
    
    Returns:
        Ternary weights with same shape as input, values in {-s, 0, +s}
        where s is computed per group of GROUP_SIZE elements.
    
    VJP (STE): gradient passes through as identity — dL/dW_latent = dL/dW_ternary
    """
    original_shape = w.shape
    
    # Flatten to 2D: (num_groups, GROUP_SIZE) for group processing
    # We group along the last dimension (input features for weight矩阵)
    # For a weight matrix of shape (out_features, in_features), we treat
    # each row as being split into groups of GROUP_SIZE along in_features
    w_2d = w.reshape(-1, w.shape[-1])
    
    # Pad the last dimension to be divisible by GROUP_SIZE
    in_features = w_2d.shape[-1]
    pad_size = (GROUP_SIZE - (in_features % GROUP_SIZE)) % GROUP_SIZE
    if pad_size > 0:
        w_2d = mx.pad(w_2d, [(0, 0), (0, pad_size)], constant_values=0.0)
    
    padded_features = w_2d.shape[-1]
    num_groups = padded_features // GROUP_SIZE
    
    # Reshape to (flat_batch, num_groups, GROUP_SIZE)
    w_grouped = w_2d.reshape(w_2d.shape[0], num_groups, GROUP_SIZE)
    
    # Compute scale per group: s = mean(|W_group|)
    scales = mx.mean(mx.abs(w_grouped), axis=-1, keepdims=True)
    
    # Avoid division by zero: if scale is 0, set to 1 (group is all zeros)
    scales = mx.where(scales == 0, mx.ones_like(scales), scales)
    
    # Project to ternary: round(W / s) * s, where round gives {-1, 0, +1}
    # We use a clip to ensure we only get -1, 0, +1
    ternary = w_grouped / scales
    ternary = mx.clip(mx.round(ternary), -1.0, 1.0)
    
    # Scale back up
    result_grouped = ternary * scales
    
    # Reshape back to padded 2D
    result_2d = result_grouped.reshape(w_2d.shape[0], padded_features)
    
    # Remove padding
    if pad_size > 0:
        result_2d = result_2d[:, :in_features]
    
    # Reshape back to original shape
    result = result_2d.reshape(original_shape)
    
    return result


@ternary_projection.vjp
def ternary_projection_vjp(primals, cotangent, output):
    """
    Straight-Through Estimator: gradient passes through as identity.
    The scale factor is treated as constant w.r.t. the latent weights.
    """
    return (cotangent,)


class TernaryLinear(nn.Module):
    """
    Linear layer with ternary weight projection.
    
    Stores latent (full-precision) weights and projects to ternary
    on the forward pass. Gradients flow through via STE.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize latent weights with Kaiming-like init
        # scaled by fan_in^(-0.5) as per BitNet
        scale = in_features ** (-0.5)
        self.weight = mx.random.normal(
            shape=(out_features, in_features)
        ) * scale
        
        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None
    
    def __call__(self, x: mx.array) -> mx.array:
        # Project latent weights to ternary
        w_ternary = ternary_projection(self.weight)
        
        # Standard linear: y = x @ W^T + b
        # MLX Linear convention: weight is (out_features, in_features)
        output = x @ w_ternary.T
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class TernaryEmbedding(nn.Module):
    """
    Embedding layer with ternary weight projection.
    
    Same as standard embedding but projects weights to ternary on forward pass.
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize with small values
        self.weight = mx.random.normal(
            shape=(num_embeddings, embedding_dim)
        ) * (embedding_dim ** (-0.5))
    
    def __call__(self, x: mx.array) -> mx.array:
        # Project embedding weights to ternary
        w_ternary = ternary_projection(self.weight)
        return w_ternary[x]
    
    def as_linear(self, x: mx.array) -> mx.array:
        """Use embedding as linear layer (for tied weights)."""
        w_ternary = ternary_projection(self.weight)
        return x @ w_ternary.T


def verify_ternary_weights(model, tol=1e-5):
    """
    Verify that all ternary layers project to valid ternary weights.
    Returns dict of layer_name -> {is_ternary, fraction_valid, sample_values}.
    """
    results = {}
    
    def check_module(module, prefix=''):
        for name in module.keys() if hasattr(module, 'keys') else []:
            obj = module[name]
            full_name = f'{prefix}{name}'
            if isinstance(obj, (TernaryLinear, TernaryEmbedding)):
                w = obj.weight
                w_ternary = ternary_projection(w)
                # Check if projected weights are truly ternary
                # Compute scales per group
                w_flat = w.reshape(-1, w.shape[-1])
                in_features = w_flat.shape[-1]
                pad_size = (GROUP_SIZE - (in_features % GROUP_SIZE)) % GROUP_SIZE
                if pad_size > 0:
                    w_flat_padded = mx.pad(w_flat, [(0, 0), (0, pad_size)], constant_values=0.0)
                else:
                    w_flat_padded = w_flat
                
                padded_features = w_flat_padded.shape[-1]
                num_groups = padded_features // GROUP_SIZE
                w_grouped = w_flat_padded.reshape(w_flat_padded.shape[0], num_groups, GROUP_SIZE)
                scales = mx.mean(mx.abs(w_grouped), axis=-1, keepdims=True)
                scales = mx.where(scales == 0, mx.ones_like(scales), scales)
                
                # Normalized values should be in {-1, 0, +1}
                normalized = w_ternary.reshape(w_flat_padded.shape[0], num_groups, GROUP_SIZE) / scales
                diff = mx.abs(normalized - mx.round(normalized))
                max_diff = float(mx.max(diff))
                
                # Also check that rounding gives only {-1, 0, 1}
                rounded = mx.round(normalized)
                is_ternary = bool(mx.all((rounded == -1) | (rounded == 0) | (rounded == 1)))
                
                results[full_name] = {
                    'is_ternary': is_ternary,
                    'max_round_error': max_diff,
                    'shape': tuple(w.shape),
                }
            elif isinstance(obj, nn.Module):
                check_module(obj, f'{full_name}.')
    
    check_module(model)
    return results