"""Demo + correctness check for kv_cache.py.

Three things are exercised:

1. Prefill with variable prompt lengths across the batch.
2. Decoding new tokens one at a time, advancing each sequence independently.
3. Equivalence vs. a from-scratch (no-cache) recomputation: for any sequence,
   running the same projections + causal attention over the full token list
   must produce exactly the same outputs as the cached path.
"""

import math
import random

from kv_cache import KVCache, MultiHeadAttention, cache_memory_bytes, _matvec, _softmax


def recompute_no_cache(mha, tokens):
    """Reference attention over `tokens` (length T) with causal mask, no cache."""
    H, D = mha.H, mha.D
    Qs, Ks, Vs = [], [], []
    for x in tokens:
        q, k, v = mha._project_qkv(x)
        Qs.append(mha._split(q))
        Ks.append(mha._split(k))
        Vs.append(mha._split(v))
    scale = 1.0 / math.sqrt(D)
    outs = []
    for i in range(len(tokens)):
        head_outs = []
        for h in range(H):
            scores = [sum(Qs[i][h][d] * Ks[j][h][d] for d in range(D)) * scale
                      for j in range(i + 1)]
            w = _softmax(scores)
            ctx = [0.0] * D
            for j in range(i + 1):
                for d in range(D):
                    ctx[d] += w[j] * Vs[j][h][d]
            head_outs.extend(ctx)
        outs.append(_matvec(mha.Wo, head_outs))
    return outs


def max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def main():
    rng = random.Random(42)
    d_model, num_heads, num_layers = 16, 4, 2
    B, S_max = 3, 32
    cache = KVCache(num_layers, B, num_heads, d_model // num_heads, S_max)
    layers = [MultiHeadAttention(d_model, num_heads, l, seed=7) for l in range(num_layers)]

    # Build three prompts of different lengths to exercise variable-length batching.
    prompt_lens = [5, 8, 3]
    prompts = [[[rng.gauss(0, 1) for _ in range(d_model)] for _ in range(L)]
               for L in prompt_lens]

    # Prefill each sequence independently. Only layer 0 is checked against the
    # reference here; the same logic applies layer-by-layer in a real stack.
    print("== prefill ==")
    for b, prompt in enumerate(prompts):
        cached_outs = layers[0].prefill(prompt, cache, b)
        ref_outs = recompute_no_cache(layers[0], prompt)
        diffs = [max_abs_diff(c, r) for c, r in zip(cached_outs, ref_outs)]
        print(f"  batch {b}: prompt_len={len(prompt)} "
              f"cache.length={cache.lengths[b]} max|cache-ref|={max(diffs):.2e}")
        assert max(diffs) < 1e-9, "cache vs. no-cache mismatch"

    # Decode 4 more tokens for each sequence in lockstep. We also keep the full
    # token history in `histories` so we can re-verify against the no-cache path.
    print("== decode ==")
    histories = [list(p) for p in prompts]
    for step in range(4):
        new_tokens = [[rng.gauss(0, 1) for _ in range(d_model)] for _ in range(B)]
        for b in range(B):
            histories[b].append(new_tokens[b])
        outs = layers[0].decode_step(new_tokens, cache)
        for b in range(B):
            ref_full = recompute_no_cache(layers[0], histories[b])
            assert max_abs_diff(outs[b], ref_full[-1]) < 1e-9
        print(f"  step {step}: lengths={cache.lengths} (all match no-cache reference)")

    # Variable-length / early-stop: deactivate batch item 0 and decode again.
    # Its length must not advance and its slot must not be written.
    print("== early-stop (active mask) ==")
    snapshot_len = cache.lengths[0]
    new_tokens = [[rng.gauss(0, 1) for _ in range(d_model)] for _ in range(B)]
    layers[0].decode_step(new_tokens, cache, active=[False, True, True])
    assert cache.lengths[0] == snapshot_len, "inactive sequence advanced"
    print(f"  lengths={cache.lengths} (batch 0 frozen at {snapshot_len})")

    # Memory accounting for a realistic config (analytic, no allocation).
    print("== memory model ==")
    for B_, S_ in [(1, 4096), (8, 4096), (32, 8192), (128, 32768)]:
        gb = cache_memory_bytes(num_layers=32, B=B_, H=32, S=S_, D=128, dtype_bytes=2) / (1024 ** 3)
        print(f"  L=32 H=32 D=128 fp16  B={B_:>3} S={S_:>5}: {gb:8.2f} GiB")


if __name__ == "__main__":
    main()
