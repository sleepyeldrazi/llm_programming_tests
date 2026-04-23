"""
Memory Growth Analysis for KV-Cache Systems

Analyzes how memory consumption scales with:
  - Sequence length
  - Batch size
  - Number of heads
  - Model dimension
  - Number of layers

Provides formulas, visualizations, and practical limits.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ModelSpec:
    """Specification of a transformer model."""
    num_layers: int
    dim: int
    num_heads: int
    head_dim: int
    vocab_size: int = 32000
    mlp_hidden_mult: float = 4.0 / 3  # GPT-style


def compute_model_memory(spec: ModelSpec, dtype=np.float16) -> Dict[str, float]:
    """
    Compute total model parameter memory.

    Per layer:
      - Wq, Wk, Wv: 3 * dim * dim
      - Wo: dim * dim
      - MLP fc1: dim * hidden
      - MLP fc2: hidden * dim
      - LayerNorm: 2 * dim (weight + bias)
      - Embedding: vocab_size * dim (shared with LM head)

    Total per layer (excluding shared embedding):
      4 * dim² + 2 * dim * hidden + 2 * dim
    """
    elem = np.dtype(dtype).itemsize
    hidden = int(spec.dim * spec.mlp_hidden_mult)

    per_layer = (
        4 * spec.dim * spec.dim +           # Wq, Wk, Wv, Wo
        2 * spec.dim * hidden +              # MLP fc1, fc2
        2 * spec.dim                          # LayerNorm params
    ) * elem

    embedding = spec.vocab_size * spec.dim * elem

    return {
        "per_layer_bytes": per_layer,
        "per_layer_mb": per_layer / (1024 * 1024),
        "embedding_mb": embedding / (1024 * 1024),
        "total_params_mb": (per_layer * spec.num_layers + embedding) / (1024 * 1024),
        "total_params_gb": (per_layer * spec.num_layers + embedding) / (1024 ** 3),
    }


def compute_kv_cache_memory(
    batch_size: int,
    seq_len: int,
    spec: ModelSpec,
    dtype=np.float16,
) -> Dict[str, float]:
    """
    Compute KV cache memory for a given batch and sequence length.

    Per layer: 2 * batch * heads * seq * head_dim * elem_bytes
    (factor of 2 for K and V)
    """
    elem = np.dtype(dtype).itemsize
    per_layer = 2 * batch_size * spec.num_heads * seq_len * spec.head_dim * elem
    total = per_layer * spec.num_layers

    return {
        "per_layer_bytes": per_layer,
        "per_layer_mb": per_layer / (1024 * 1024),
        "total_bytes": total,
        "total_mb": total / (1024 * 1024),
        "total_gb": total / (1024 ** 3),
        "per_token_per_layer_bytes": 2 * spec.num_heads * spec.head_dim * elem,
        "growth_rate_mb_per_token": (
            2 * batch_size * spec.num_heads * spec.head_dim * elem * spec.num_layers
        ) / (1024 * 1024),
    }


def analyze_memory_growth(spec: ModelSpec, batch_sizes: List[int] = None,
                           seq_lengths: List[int] = None,
                           dtype=np.float16) -> Dict:
    """
    Comprehensive memory growth analysis.

    Returns analysis for various batch sizes and sequence lengths.
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    model_mem = compute_model_memory(spec, dtype)

    results = {
        "model": model_mem,
        "spec": {
            "num_layers": spec.num_layers,
            "dim": spec.dim,
            "num_heads": spec.num_heads,
            "head_dim": spec.head_dim,
            "dtype": str(dtype),
        },
        "kv_cache": {},
    }

    for bs in batch_sizes:
        for sl in seq_lengths:
            kv = compute_kv_cache_memory(bs, sl, spec, dtype)
            key = f"bs{bs}_sl{sl}"
            results["kv_cache"][key] = {
                "batch_size": bs,
                "seq_len": sl,
                "kv_cache_gb": kv["total_gb"],
                "total_system_gb": kv["total_gb"] + model_mem["total_params_gb"],
                "kv_fraction": kv["total_gb"] / (kv["total_gb"] + model_mem["total_params_gb"]),
            }

    return results


def find_max_context(spec: ModelSpec, gpu_memory_gb: float = 80,
                      batch_size: int = 1, dtype=np.float16) -> int:
    """
    Find the maximum context length that fits in GPU memory.

    GPU memory = model_params + kv_cache + activation_overhead

    We estimate activation overhead as ~2x model params (conservative).
    """
    model_mem = compute_model_memory(spec, dtype)
    model_gb = model_mem["total_params_gb"]

    # Reserve for activations and other overhead (~2x model params)
    activation_gb = model_gb * 2

    # Remaining for KV cache
    kv_budget_gb = gpu_memory_gb - model_gb - activation_gb
    if kv_budget_gb <= 0:
        return 0

    elem = np.dtype(dtype).itemsize
    bytes_per_token = (2 * batch_size * spec.num_heads * spec.head_dim * elem *
                       spec.num_layers)

    max_tokens = int(kv_budget_gb * (1024 ** 3) / bytes_per_token)
    return max_tokens


def compare_model_sizes() -> Dict[str, dict]:
    """
    Analyze memory for several well-known model sizes.
    """
    models = {
        "Llama-2-7B": ModelSpec(num_layers=32, dim=4096, num_heads=32, head_dim=128),
        "Llama-2-13B": ModelSpec(num_layers=40, dim=5120, num_heads=40, head_dim=128),
        "Llama-2-70B": ModelSpec(num_layers=80, dim=8192, num_heads=64, head_dim=128),
        "Llama-3-8B": ModelSpec(num_layers=32, dim=4096, num_heads=32, head_dim=128),
        "Mistral-7B": ModelSpec(num_layers=32, dim=4096, num_heads=32, head_dim=128),
        "GPT-4-class": ModelSpec(num_layers=100, dim=12288, num_heads=96, head_dim=128),
    }

    results = {}
    for name, spec in models.items():
        model_mem = compute_model_memory(spec, np.float16)

        # KV cache for batch=1, various lengths
        kv_1k = compute_kv_cache_memory(1, 1024, spec, np.float16)
        kv_8k = compute_kv_cache_memory(1, 8192, spec, np.float16)
        kv_32k = compute_kv_cache_memory(1, 32768, spec, np.float16)

        results[name] = {
            "params_gb": model_mem["total_params_gb"],
            "kv_1k_gb": kv_1k["total_gb"],
            "kv_8k_gb": kv_8k["total_gb"],
            "kv_32k_gb": kv_32k["total_gb"],
            "max_context_H100": find_max_context(spec, gpu_memory_gb=80, batch_size=1),
            "max_context_A100_40": find_max_context(spec, gpu_memory_gb=40, batch_size=1),
            "max_context_A100_80": find_max_context(spec, gpu_memory_gb=80, batch_size=1),
        }

    return results


def print_analysis():
    """Print a comprehensive memory analysis report."""
    print("=" * 80)
    print("KV-CACHE MEMORY GROWTH ANALYSIS")
    print("=" * 80)

    # Model size comparison
    print("\n--- Model Size Comparison (fp16) ---\n")
    comparisons = compare_model_sizes()
    header = f"{'Model':<20} {'Params(GB)':>10} {'KV@1K':>10} {'KV@8K':>10} {'KV@32K':>10} {'MaxCtx(H100)':>12}"
    print(header)
    print("-" * len(header))
    for name, data in comparisons.items():
        print(f"{name:<20} {data['params_gb']:>10.1f} {data['kv_1k_gb']:>10.2f} "
              f"{data['kv_8k_gb']:>10.2f} {data['kv_32k_gb']:>10.2f} "
              f"{data['max_context_H100']:>12,d}")

    # Growth analysis for a 7B model
    print("\n\n--- Detailed Growth: 7B Model (batch=1, fp16) ---\n")
    spec_7b = ModelSpec(num_layers=32, dim=4096, num_heads=32, head_dim=128)
    model_mem = compute_model_memory(spec_7b, np.float16)

    seq_lens = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    print(f"{'Seq Len':>10} {'KV Cache(GB)':>14} {'Total(GB)':>12} {'KV Fraction':>12}")
    print("-" * 52)
    for sl in seq_lens:
        kv = compute_kv_cache_memory(1, sl, spec_7b, np.float16)
        total = kv["total_gb"] + model_mem["total_params_gb"]
        frac = kv["total_gb"] / total
        print(f"{sl:>10,} {kv['total_gb']:>14.2f} {total:>12.2f} {frac:>12.1%}")

    # Batch size impact
    print("\n\n--- Batch Size Impact (seq_len=4096, fp16) ---\n")
    batch_sizes = [1, 2, 4, 8, 16, 32]
    print(f"{'Batch':>6} {'KV Cache(GB)':>14} {'Growth/Token(MB)':>18}")
    print("-" * 40)
    for bs in batch_sizes:
        kv = compute_kv_cache_memory(bs, 4096, spec_7b, np.float16)
        print(f"{bs:>6} {kv['total_gb']:>14.2f} {kv['growth_rate_mb_per_token']:>18.4f}")

    # Per-token cost
    print("\n\n--- Per-Token Memory Cost ---\n")
    kv_one = compute_kv_cache_memory(1, 1, spec_7b, np.float16)
    per_token = kv_one["total_bytes"]
    print(f"  Per token (all layers): {per_token:,} bytes = {per_token/1024:.1f} KB")
    print(f"  Per token per layer: {kv_one['per_token_per_layer_bytes']:,} bytes")
    print(f"  At 32K context: {per_token * 32768 / (1024**3):.2f} GB")

    # GPU memory limits
    print("\n\n--- Maximum Context Lengths by GPU ---\n")
    gpus = {
        "RTX 4090": 24,
        "A100-40GB": 40,
        "A100-80GB": 80,
        "H100-80GB": 80,
        "H100-96GB (SXM)": 96,
    }
    print(f"{'GPU':<20} {'Max Context (bs=1)':>20} {'Max Context (bs=4)':>20}")
    print("-" * 62)
    for gpu, mem in gpus.items():
        ctx_1 = find_max_context(spec_7b, mem, batch_size=1)
        ctx_4 = find_max_context(spec_7b, mem, batch_size=4)
        print(f"{gpu:<20} {ctx_1:>20,} {ctx_4:>20,}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_analysis()
