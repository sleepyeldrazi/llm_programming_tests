"""KV-cache for autoregressive transformer inference. Pure-Python, no frameworks.

Memory layout
-------------
For each transformer layer we keep two flat float buffers (K and V) of length

    B * H * S_max * D

with index ((b * H + h) * S_max + t) * D + d.

This corresponds to logical tensor shape [B, H, S_max, D] in row-major order:
head_dim is the fastest-varying axis, then token position, then heads, then
batch. That matches the layout PyTorch / FlashAttention use for KV tensors of
shape (B, H, S, D); reading a single (b, h, t) row touches D contiguous floats,
which on a GPU coalesces into one or two cache-line loads.

Per-sequence lengths are tracked in `lengths[b]`, so each batch item can be at
its own position in the buffer (variable-length batching). The buffers are
preallocated to S_max so appending a new token is O(D) — no reallocation,
no copy.
"""

import math
import random


class KVCache:
    def __init__(self, num_layers, batch_size, num_heads, head_dim, max_seq_len):
        self.L = num_layers
        self.B = batch_size
        self.H = num_heads
        self.D = head_dim
        self.S = max_seq_len
        n = batch_size * num_heads * max_seq_len * head_dim
        self.K = [[0.0] * n for _ in range(num_layers)]
        self.V = [[0.0] * n for _ in range(num_layers)]
        self.lengths = [0] * batch_size

    def base(self, b, h, t):
        return ((b * self.H + h) * self.S + t) * self.D

    def write(self, layer, b, h, t, k_vec, v_vec):
        Kb, Vb = self.K[layer], self.V[layer]
        off = self.base(b, h, t)
        for d in range(self.D):
            Kb[off + d] = k_vec[d]
            Vb[off + d] = v_vec[d]

    def memory_bytes(self, dtype_bytes=2):
        return cache_memory_bytes(self.L, self.B, self.H, self.S, self.D, dtype_bytes)


def cache_memory_bytes(num_layers, B, H, S, D, dtype_bytes=2):
    """Total cache footprint in bytes (K + V across all layers). fp16 -> 2."""
    return 2 * num_layers * B * H * S * D * dtype_bytes


def _matvec(M, v):
    return [sum(mi * vi for mi, vi in zip(row, v)) for row in M]


def _softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


class MultiHeadAttention:
    """Single decoder-style MHA layer that reads/writes a KVCache.

    Provides two entry points:
      - prefill(prompt_b, cache, b): process a variable-length prompt for one
        batch item, populating cache rows [0, len(prompt)).
      - decode_step(x_batch, cache, active): append one new token per active
        batch item and compute attention against everything cached so far.

    No FFN / LayerNorm — those are orthogonal to the cache and the prompt only
    asked for the attention path.
    """

    def __init__(self, d_model, num_heads, layer_idx, seed=0):
        assert d_model % num_heads == 0
        self.d = d_model
        self.H = num_heads
        self.D = d_model // num_heads
        self.layer = layer_idx
        rng = random.Random(seed + layer_idx)
        scale = 1.0 / math.sqrt(d_model)

        def W():
            return [[rng.gauss(0, 1) * scale for _ in range(d_model)] for _ in range(d_model)]

        self.Wq, self.Wk, self.Wv, self.Wo = W(), W(), W(), W()

    def _split(self, vec):
        return [vec[h * self.D : (h + 1) * self.D] for h in range(self.H)]

    def _project_qkv(self, x):
        return _matvec(self.Wq, x), _matvec(self.Wk, x), _matvec(self.Wv, x)

    def _attend_one(self, qh, cache, b, t_end):
        """Attention for batch item b: query qh[h] against cache rows [0, t_end)."""
        scale = 1.0 / math.sqrt(self.D)
        Kbuf, Vbuf = cache.K[self.layer], cache.V[self.layer]
        head_outs = []
        for h in range(self.H):
            q = qh[h]
            scores = [0.0] * t_end
            for t in range(t_end):
                off = cache.base(b, h, t)
                s = 0.0
                for d in range(self.D):
                    s += q[d] * Kbuf[off + d]
                scores[t] = s * scale
            w = _softmax(scores)
            ctx = [0.0] * self.D
            for t in range(t_end):
                off = cache.base(b, h, t)
                wt = w[t]
                for d in range(self.D):
                    ctx[d] += wt * Vbuf[off + d]
            head_outs.extend(ctx)
        return _matvec(self.Wo, head_outs)

    def prefill(self, prompt, cache, b):
        """Process a prompt (list of token hidden states) for batch item b.

        Writes K/V for every position and computes the per-position output with
        causal masking — i.e., position t attends to [0, t]. Returns the list
        of output vectors. After this call, cache.lengths[b] == len(prompt).
        """
        outs = []
        for x in prompt:
            q, k, v = self._project_qkv(x)
            qh, kh, vh = self._split(q), self._split(k), self._split(v)
            t = cache.lengths[b]
            for h in range(self.H):
                cache.write(self.layer, b, h, t, kh[h], vh[h])
            outs.append(self._attend_one(qh, cache, b, t + 1))
            cache.lengths[b] = t + 1
        return outs

    def decode_step(self, x_batch, cache, active=None):
        """One decode step. x_batch[b] is the new token's hidden state.

        active[b] = False skips that batch item (no write, no length advance) —
        this is how variable-length / early-stop batches stay correct without
        padding work onto the cache.
        """
        B = len(x_batch)
        if active is None:
            active = [True] * B
        out = [None] * B
        for b in range(B):
            if not active[b]:
                continue
            x = x_batch[b]
            q, k, v = self._project_qkv(x)
            qh, kh, vh = self._split(q), self._split(k), self._split(v)
            t = cache.lengths[b]
            for h in range(self.H):
                cache.write(self.layer, b, h, t, kh[h], vh[h])
            out[b] = self._attend_one(qh, cache, b, t + 1)
            cache.lengths[b] = t + 1
        return out
