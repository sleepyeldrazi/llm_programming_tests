Implement a numerically stable backward pass for layer normalization from scratch in NumPy.

Constraints:
- Input: x of shape (B, T, D)
- Parameters: gamma, beta of shape (D,)
- Forward:
    y = gamma * (x - mean) / sqrt(var + eps) + beta

Requirements:
1. Derive and implement gradients w.r.t. x, gamma, beta manually (no autodiff).
2. Avoid redundant recomputation — reuse intermediates where possible.
3. Ensure numerical stability (discuss where instability can occur).
4. Provide a gradient check using finite differences.
5. Analyze time and memory complexity.
6. Explain how you would fuse this into a single kernel for GPU execution.

Do not use PyTorch, TensorFlow, JAX, or autograd.
