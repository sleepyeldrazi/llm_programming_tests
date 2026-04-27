Implement the forward pass of tiled (Flash) attention using online softmax
from scratch in NumPy.

Input:  Q — (B, H, N, D)   queries
        K — (B, H, N, D)   keys
        V — (B, H, N, D)   values
        tile_size T  (e.g., 128)

Algorithm: process Q in tiles of size T, and K/V in tiles of size T.
For each (Q_tile, KV_tile) pair, compute local attention scores, update
online statistics, and accumulate output. Never materialize the full
(N, N) attention matrix.

Requirements:
1. Implement the ONLINE softmax rescaling recurrence:
   - Track running max m and running exp-sum l per query row within the
     current Q tile. These start as m = -inf, l = 0, O = 0.
   - For each KV tile processed:
       S = Q_tile @ K_tile^T / sqrt(D)          # local scores
       m_new = maximum(m_old, row_maxes_from_S)  # update running max
       correction = exp(m_old - m_new)            # RESCALE factor
       O = O * correction                         # rescale accumulated output
       l = l * correction + sum(exp(S - m_new))  # rescale sum, add new
       P = exp(S - m_new)                         # stable probabilities
       O = O + P @ V_tile                         # accumulate weighted V
       m_old = m_new
   - After all KV tiles: output = O / l

2. Support causal masking: query position i can attend only to key positions
   j where j <= i. Handle the interaction between causal masking and tiling
   correctly — some (Q_tile, KV_tile) blocks are entirely above the diagonal
   and must be skipped (all masked).

3. Match the naive full-softmax attention output to within 1e-4 relative error.

4. Verify memory: for a large N (e.g., 4096), the implementation must never
   allocate an (N, N) tensor. Demonstrate this with tracemalloc or similar,
   or at minimum explain why no such allocation occurs.

5. Explain in comments:
   - Why the rescaling factor is exp(m_old - m_new) and NOT exp(m_new - m_old)
   - What happens at tile boundaries when a query row's first KV tile is
     fully masked (causal) — what are m and l at that point, and why is
     this a numerical stability hazard?

Deliver:
- A working function `flash_attention_fwd(Q, K, V, tile_size, causal=True)`
  that returns the attention output of shape (B, H, N, D)
- A test with (B=1, H=1, N=256, D=64), tile_size=64, causal=True, comparing
  against naive full-softmax attention. Assert relative error < 1e-4.
- A test with (B=2, H=8, N=4096, D=64), tile_size=128, causal=True.
  Verify via tracemalloc that no (N, N) tensor is ever allocated.
- Comments explaining the online softmax rescaling math and the two
  numerical stability hazards identified above.

Use only NumPy. No PyTorch, JAX, TensorFlow, or autograd.
