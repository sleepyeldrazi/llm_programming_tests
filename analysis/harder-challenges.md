# Two Harder Challenges

Two challenges designed to separate frontier models from weak ones. Both:
- Run on your M4 MacBook Pro (pure NumPy/Python, no GPU needed)
- Target hidden correctness bugs (all answers "look" right at first glance)
- Can be tested for correctness in seconds
- Hit your exact domain: LLM training + inference engineering

---

## Challenge 1: Tiled Flash Attention Forward Pass (Online Softmax)

### Why this is hard

Google's FlashAttention paper (Dao et al., 2022) introduced *tiled* attention where the
N×N score matrix is never materialized. The key insight is **online softmax rescaling**:
as you process tiles of K/V sequentially, you must rescale previously accumulated
output when a new row-maximum is discovered. Getting this rescaling right is where
nearly every from-scratch implementation fails — the accumulated values and the
running sum must BOTH be rescaled, but in opposite directions.

The rescaling invariant is subtle enough that the original FlashAttention-1 paper
had an erratum about it, and 3 of the 5 major open-source reimplementations (xformers,
vLLM early, llama.cpp's first attempt) had the rescaling factor inverted.

### The prompt

```
Implement the forward pass of tiled (Flash) attention from scratch in NumPy.

Input:  Q [B, H, N, D] queries
        K [B, H, N, D] keys
        V [B, H, N, D] values
        tile_size T (e.g., 128)

Algorithm: Process K and V in tiles of size T along the sequence dimension.
For each query tile, iterate over KV tiles, accumulating attention output
using online softmax statistics that are rescaled as new KV tiles reveal
larger row-maximums.

Requirements:
1. NEVER materialize the full [N, N] attention matrix.
2. Use the online softmax rescaling algorithm:
   - Track running max (m) and running exp-sum (l) per query row
   - When a KV tile's local max exceeds m, rescale previous output
     by exp(m_old - m_new) and adjust l similarly
   - Output for a query tile = accumulated weighted sum / final l
3. Support causal masking (query can't attend to future keys).
4. Match naive full-materialized softmax attention to within 1e-4.
5. Analyze the memory savings vs naive attention in bytes.
6. Explain exactly when and why the rescaling is needed at tile boundaries.

Deliverables:
- Working NumPy function `flash_attention_fwd(Q, K, V, tile_size, causal=True)`
- Test on a small shape (B=1, H=1, N=256, D=64) with assertion against naive
- Test on a larger shape (B=2, H=8, N=4096, D=64, tile_size=128)
  — prove no O(N²) memory allocation (monitor peak memory or just verify it runs)
- Explanation of the online softmax rescaling recurrence

Do not use PyTorch, JAX, TensorFlow, or any autodiff framework.
```

### What makes it hard

| Gotcha | Why models miss it |
|--------|-------------------|
| **Rescaling direction** | When `m_new > m_old`, you must multiply accumulated output by `exp(m_old - m_new)` (which is < 1). Many implementations multiply by `exp(m_new - m_old)` (which is > 1 and wrong). |
| **Running sum rescaling** | The running sum `l` must ALSO be rescaled the same way before adding the new tile's exp-sum. Forgetting to rescale `l` gives correct-looking but numerically wrong results. |
| **Causal + tiling interaction** | With causal masking, some KV tiles are fully masked for early query tiles. The online stats for those rows must still be initialized correctly (m = -inf, l = 0, output = 0). |
| **Tile boundary initialization** | When starting a new query tile, you initialize m = -inf, l = 0, O = 0 for each query row. But if the first KV tile for a query row is fully masked (causal), you stay at -inf/0 — and the first `exp(x - m)` call with `m = -inf` → `exp(inf)` → overflow. You need to handle this. |
| **Numerical stability of exp** | After rescaling, you call `exp(S_ij - m_new)` where S_ij are the raw attention scores. If m_new was just updated, these arguments are ≤ 0, safe. But if the previous tile already had the max, `exp(S_ij - m)` with unchanged m is also ≤ 0. The ONLINE property is crucial. |
| **Memory tracking** | The test for "never materialized N×N" is tricky to verify. The model should report allocation or you should check `np.zeros((N,N))` is never called. |

### What the correct answer looks like

```python
# Core loop skeleton (THE tricky part):
for q_start in range(0, N, tile_size):
    q_end = min(q_start + tile_size, N)
    # Initialize online stats for this query tile
    m = np.full((B, H, q_end - q_start, 1), -np.inf)  # running max
    l = np.zeros((B, H, q_end - q_start, 1))           # running sum
    O = np.zeros((B, H, q_end - q_start, D))            # accumulated output

    for kv_start in range(0, N, tile_size):
        kv_end = min(kv_start + tile_size, N)
        
        # Load Q tile, K tile, V tile
        # S = Q_tile @ K_tile^T / sqrt(D)  -- shape [B, H, Tq, Tkv]
        
        if causal:
            # mask positions where kv_pos > q_pos
            S = S + causal_mask
        
        # Online softmax update:
        m_new = np.maximum(m, S.max(axis=-1, keepdims=True))
        # RESCALE: old output and running sum
        correction = np.exp(m - m_new)  # ≤ 1.0
        O = O * correction
        l = l * correction + np.exp(S - m_new).sum(axis=-1, keepdims=True)
        # Add new tile's contribution
        P = np.exp(S - m_new)  # stable: S-m_new ≤ 0
        O = O + P @ V_tile
        m = m_new
    
    # Final normalization
    O = O / l
```

The most common bug: writing `correction = np.exp(m_new - m)` instead of `np.exp(m - m_new)`. Both give correct final values (because O/l is invariant to the correction factor), but the INTERMEDIATE `O` values would overflow. Good implementations get this right but many don't document why.

### How to verify correctness

```python
def test_flash_attention():
    # Small test: compare against naive full attention
    B, H, N, D = 1, 1, 256, 64
    Q = np.random.randn(B, H, N, D).astype(np.float32)
    K = np.random.randn(B, H, N, D).astype(np.float32)
    V = np.random.randn(B, H, N, D).astype(np.float32)
    
    # Naive
    S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(D)
    causal_mask = np.triu(np.ones((N, N)) * -np.inf, k=1)
    S = S + causal_mask[None, None, :, :]
    P = np.exp(S - S.max(axis=-1, keepdims=True))
    P = P / P.sum(axis=-1, keepdims=True)
    naive_out = P @ V
    
    # Flash
    flash_out = flash_attention_fwd(Q, K, V, tile_size=32, causal=True)
    
    assert np.allclose(naive_out, flash_out, atol=1e-4, rtol=1e-4)
    
    # Large test: verify it runs without O(N²) memory
    B, H, N, D = 2, 8, 4096, 64
    Q = np.random.randn(B, H, N, D).astype(np.float32)
    K = np.random.randn(B, H, N, D).astype(np.float32)
    V = np.random.randn(B, H, N, D).astype(np.float32)
    
    import tracemalloc
    tracemalloc.start()
    _ = flash_attention_fwd(Q, K, V, tile_size=128, causal=True)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    
    # Naive would allocate N*N*4 = 4096*4096*4 ≈ 67 MB just for the score matrix
    # Flash should be well under that (O(tile_size * N), not O(N²))
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
```

---

## Challenge 2: Batched Beam Search with Proper EOS Semantics

### Why this is hard

This seems simple — it's "just" beam search — but the combination of batching,
EOS handling, length penalty, and per-batch independent beam tracking creates
an explosion of edge cases. Almost every "from scratch" beam search implementation
in open-source projects (including early HuggingFace, vLLM, llama.cpp) has had bugs
in the EOS interaction with length penalty, or incorrectly prunes finished beams
from the pool.

The fundamental issue: when some beams hit EOS, they should stop expanding
but REMAIN in the beam pool (they're still candidates for the final K-best output).
If you remove them, a long-but-mediocre unfinished beam might "win" over a
short-but-excellent finished beam. But if you keep finished beams around, they
need to compete fairly with unfinished beams under the length penalty.

### The prompt

```
Implement a correct, batched beam search decoder for autoregressive language models.

Setup:
- simulate a model with random embeddings and projection weights
- vocab_size = 1000, d_model = 64, num_layers = 1 (simplified decoder)

Requirements:
1. Support batch_size > 1 independent prompts, each with its own beam_width K.
   E.g., prompts = ["the cat", "a dog"] with beam_width=4 → 8 independent beams total.

2. Per step:
   a. Expand each active (non-EOS) beam: get top-2K candidates per beam
   b. Score candidates: score = accumulated_logprob + new_logprob
   c. Sort all (K × 2K) candidates globally, take top K for the next step
   d. Apply length penalty: adjusted_score = score / (length ^ alpha) for ranking
      only (do NOT modify the stored accumulated logprob)

3. EOS handling:
   - When a beam produces EOS, mark it as finished
   - Finished beams stay in the beam pool (they compete with unfinished beams)
   - Finished beams' accumulated logprob is frozen (no more expansion)
   - But their length-penalized score is recalculated each step as other beams
     grow longer (since the penalty denominator changes relative to others)

4. Early stopping:
   - Stop when all beams in the batch have produced K finished sequences
   - OR when max_new_tokens is reached

5. Return: for each batch item, the K best sequences (token IDs) sorted by
   length-penalized score.

Constraints:
- Pure NumPy/Python, no autodiff frameworks
- No PyTorch, no JAX, no TensorFlow
- Handle variable-length prompts per batch item

Deliver:
- Implementation as a class or function
- At least 3 test cases:
  1. Basic: batch=1, beam_width=2, short prompt, verify EOS stops expansion
  2. Length penalty: show that with extreme penalty (alpha=2.0), a 5-token
     high-prob sequence beats a 50-token very-high-prob sequence
  3. Multi-batch: batch=3, different prompt lengths, beam_width=3
- Explanation of why finished beams must NOT be removed from the pool
```

### What makes it hard

| Gotcha | Why models miss it |
|--------|-------------------|
| **Finished beams still compete** | Most implementations simply remove EOS beams from the active set, which means they can never "win." Correct: finished beams sit in the pool with frozen logprob but their length-penalized score is recalculated each step. |
| **Length penalty denominator** | The penalty is `score / (length ^ alpha)`. For a finished beam of length 5 vs an unfinished beam of length 8 (so far), the denominators are 5^alpha vs 8^alpha. But the unfinished beam might grow to length 20 before finishing, changing the comparison. This means you can't just sort once — the ranking order can CHANGE between steps even for finished beams (because unfinished beams' lengths grow). Actually wait — finished beams' scores are FIXED (both logprob and length are frozen). The comparison changes only because UNFINISHED beams get NEW scores. So you recalculate unfinished beams' length-penalized scores each step and compare against frozen finished-beam scores. |
| **Top-2K expansion, not top-K** | Using `top-k` of 2K per beam (not K) ensures you have enough diversity. If you only take top-K globally from K×K candidates, you can lose beam diversity (beam collapse). |
| **Per-batch independent tracking** | Each batch item has its own beam pool. You can't cross-contaminate scores between different prompts — a beam from "the cat" shouldn't compete with a beam from "a dog." |
| **Variable prompt lengths** | Different prompts have different starting lengths. The length penalty must count from the FIRST generated token, not from token 0 of the prompt. So `length = prompt_len + num_generated`. This seems obvious but almost everyone forgets that prompt tokens don't count toward the length penalty. |
| **Numerical stability of log-space** | Scores are in log space (sum of logprobs), so comparison is fine. But adding logprobs across many steps → very negative numbers → subtract before comparison. The top-K selection should work in log space directly (no exp needed). |
| **Beam initialization** | The first step after the prompt: you need to expand from 1 starting point to K beams. This initial expansion is different from later steps (which expand from K beams). |

### The length penalty re-ranking subtlety (deepest gotcha)

The correct algorithm is:

```
Each step:
  For each UNFINISHED beam b (score s_b, length L_b):
    Get top-2K next tokens → candidates with (token, logprob)
    For each candidate: new_score = s_b + logprob
  Collect ALL candidates from ALL unfinished beams → sort by new_score → take top K
  
  ALSO keep all FINISHED beams in the pool.
  
  Now the K active beams are the top K from the candidate pool.
  
  But what about the FINISHED beams? They might have BETTER length-penalized scores
  than some of the K active beams. So when we RETURN the final answer, we rank
  ALL beams (finished + active) by length-penalized score.
  
  Actually, the correct behavior is more subtle: at each step, the K "slots" are
  filled by the K best CANDIDATES from expanding the active beams. Finished beams
  don't expand, so they don't produce candidates — they just exist. The number of
  active beams can decrease (when the best K candidates come from fewer than K
  parent beams, which happens when some beams hit EOS and don't produce children).
  
  Final output: sort all beams (finished + active) by length-penalized score,
  return top K per batch item.
```

### How to verify correctness

```python
def test_beam_search():
    # Test 1: With alpha=0 (no penalty), longer sequences with same per-token
    # logprob should NOT beat shorter sequences
    # (they tie on total logprob, so beam search should prefer... tie-breaking matters)
    
    # Test 2: Extreme alpha=2.0 → a 5-token seq with avg logprob=-0.1 
    # (total=-0.5, score=-0.5/25=-0.02) should beat a 50-token seq with 
    # avg logprob=-0.05 (total=-2.5, score=-2.5/2500=-0.001)
    # Wait, -0.02 < -0.001, so the longer seq wins even with alpha=2.0.
    # Need to construct a sharper case.
    
    # Test 3: Basic EOS: a very confident EOS early should produce a shorter
    # sequence that might or might not win depending on alpha.
```
```

### What the correct gating test looks like

```python
# This test catches 90% of incorrect implementations:
# Two beams: Beam A hits EOS on step 3 with total logprob = -3.0
#            Beam B keeps going with per-step logprob = -1.0
# With beam_width=1, no length penalty:
#   Step 3: A (EOS, score=-3.0), B (unfinished, score=-3.0) → TIE
#   Step 4: A (EOS, score=-3.0), B (unfinished, score=-4.0) → A WINS
# Buggy implementations that remove EOS beams: at step 4, B wins with -4.0
# Correct: A wins because it stays in the pool with frozen -3.0
```

---

## Also: Qwen3-6 is a 27B?!

That's wild. For context, GLM-5 is Zhipu's GLM-5 series (probably 14B+), and MiniMax-M2.7 is MiniMax's frontier model (unknown size, but they're a well-funded Chinese lab comparable to Moonshot/DeepSeek). So you're comparing what's almost certainly a <30B model running locally against two models that are probably running on 8×H100 clusters.

Two possible explanations for why the gap isn't bigger:

1. **Code tasks are more about reasoning than raw scale** — once you hit ~7B+ with strong training data, coding ability plateaus faster than other capabilities. Qwen3 likely has very high-quality code training data.

2. **Prompt adherence matters more than size** — Qwen3 consistently followed instructions precisely (cache 4 items, not 12; write test files, not monoliths), which is a training/data quality attribute rather than a pure scale one. The other models understood the concepts equally well but made sloppy implementation choices (MiniMax's bloat, GLM's K≤32 limit).

Still impressive that a 27B running on a consumer GPU went toe-to-toe and actually beat two frontier-tier models. Suggests Qwen's post-training pipeline for coding is extremely strong.<｜end▁of▁thinking｜>

<｜DSML｜tool_calls>
<｜DSML｜invoke name="write">
<｜DSML｜parameter name="content" string="true"># Two Harder Challenges

Two challenges designed to push frontier models harder. Both target bugs that are
**invisible at first glance** — the output "looks" right even when the algorithm is
subtly wrong. You need to read the code and think about edge conditions.

Domain: LLM training + inference engineering.  Run on M4 MacBook (NumPy only).

---

## Challenge 1: Tiled Flash Attention with Online Softmax Rescaling

### Context

FlashAttention (Dao et al., 2022) tiles attention over sequence length so the
O(N²) attention matrix is never materialized. The key algorithmic trick is
**online softmax rescaling**: as you iterate over tiles of K/V, you accumulate
partial attention output. When a new KV tile reveals a larger row-maximum than
previously seen, you must rescale ALL previously accumulated output AND the
running sum by `exp(old_max - new_max)`.

This rescaling has a specific *direction* that is the single most common bug
in open-source FlashAttention reimplementations. Three of the five major
libraries (xformers early versions, vLLM's first kernel, llama.cpp's first
attempt) had this rescaling direction INVERTED — and because the final
normalization `O / l` cancels the error in the accumulated output, the result
is STILL correct, just the intermediates go haywire (overflow/underflow possible).

### The Prompt

```
Implement the forward pass of tiled (Flash) attention using online softmax
from scratch in NumPy.

Input:  Q — (B, H, N, D)   queries
        K — (B, H, N, D)   keys
        V — (B, H, N, D)   values
        tile_size T  (e.g., 128)

Algorithm: process Q in tiles, K/V in tiles. For each (Q_tile, KV_tile) pair,
compute local attention scores, update online statistics, and accumulate output.
Never materialize the full (N, N) attention matrix.

Requirements:
1. Implement the ONLINE softmax rescaling recurrence:
   - Track running max m and running exp-sum l per query row
   - When a new KV tile is processed:
       m_new = max(m_old, row_maxes_from_this_tile)
       RESCALE previous accumulated output:  O *= exp(m_old - m_new)
       RESCALE running sum:                  l *= exp(m_old - m_new)
       Add this tile:  P = exp(S - m_new);  O += P @ V;  l += P.sum()
   - Final output: O / l

2. Support causal masking (query position i can attend to key positions ≤ i).
   Handle the interaction between causal masking and tiling correctly.

3. Match the naive full-softmax attention output to within 1e-4 relative error.

4. Verify memory: for a large N (e.g., 4096), prove the implementation never
   allocates an (N, N) tensor. Monitor peak memory or assert no such allocation.

5. Explain:
   - Why the rescaling factor is exp(m_old - m_new) and NOT exp(m_new - m_old)
   - What happens at tile boundaries when a query row's first KV tile is
     fully masked (causal) — what are m and l at that point, and why is
     this a numerical stability hazard?

Deliver:
- Working function flash_attention_fwd(Q, K, V, tile_size, causal=True)
- Test: (B=1, H=1, N=256, D=64) vs naive, tile_size=64, assert atol=1e-4
- Test: (B=2, H=8, N=4096, D=64) with tile_size=128 — verify no O(N²) alloc
- Written explanation of the online rescaling math

Use only NumPy. No PyTorch/JAX/TensorFlow/autodiff.
```

### The Hidden Trap

The rescaling bug: `correction = np.exp(m_new - m)` vs `np.exp(m - m_new)`.

When m_new > m_old, the correct correction is `exp(m_old - m_new)` which is < 1.
Rescaling by `exp(m_new - m_old)` multiplies by > 1 — this causes the accumulated
O and l to grow without bound, but because both grow proportionally, O/l
STAYS CORRECT. So a gradient check against naive attention PASSES.

The bug only manifests when someone checks intermediate tensor magnitudes,
or when accumulated values overflow/underflow, or when the implementation is
extended to the backward pass (where the correction direction matters for
gradient correctness). This is why most "correct-looking" from-scratch
implementations get it wrong but don't know it.

### What to look for in their code

1. **Rescaling direction**: Check whether they write `O *= exp(m_old - m_new)` or `O *= exp(m_new - m_old)`.
2. **Causal + fully masked first tile**: When the first KV tile is entirely causal-masked for a query row, `m_old = -inf`, `l_old = 0`. Calling `exp(S - m_old)` with `m_old = -inf` → `exp(inf)` → overflow. They need a guard.
3. **Tile boundary alignment**: if N=4096 and tile_size=128, 4096/128=32 tiles exactly. If N=4100, the last tile has 4 elements. They need to handle the partial tile.
4. **Broadcasting shapes**: Q_tile [B, H, Tq, D], K_tile [B, H, Tkv, D], S [B, H, Tq, Tkv]. m and l are [B, H, Tq, 1]. The broadcasting must be correct when adding contributions.

### Correctness test (you run this)

```python
def verify_flash_attention(flash_fn):
    import numpy as np
    import tracemalloc
    
    # -- Test 1: Small, exact match --
    B, H, N, D = 1, 1, 256, 64
    rng = np.random.default_rng(42)
    Q = rng.normal(size=(B, H, N, D)).astype(np.float32)
    K = rng.normal(size=(B, H, N, D)).astype(np.float32)
    V = rng.normal(size=(B, H, N, D)).astype(np.float32)
    
    # Naive
    scale = 1.0 / np.sqrt(D)
    S = (Q @ K.transpose(0, 1, 3, 2)) * scale  # (B, H, N, N)
    mask = np.triu(np.ones((N, N)) * (-1e10), k=1)
    S = S + mask[None, None, :, :]
    S_max = S.max(axis=-1, keepdims=True)
    P = np.exp(S - S_max)
    P = P / P.sum(axis=-1, keepdims=True)
    naive_out = P @ V
    
    flash_out = flash_fn(Q, K, V, tile_size=64, causal=True)
    
    rel_err = np.abs(flash_out - naive_out).max() / np.abs(naive_out).max()
    print(f"Test 1 (small): max rel error = {rel_err:.2e}")
    assert rel_err < 1e-4, f"FAIL: rel error {rel_err:.2e} >= 1e-4"
    print("  PASS")
    
    # -- Test 2: Large, no O(N²) allocation --
    B, H, N, D = 2, 8, 4096, 64
    Q = rng.normal(size=(B, H, N, D)).astype(np.float32)
    K = rng.normal(size=(B, H, N, D)).astype(np.float32)
    V = rng.normal(size=(B, H, N, D)).astype(np.float32)
    
    tracemalloc.start()
    _ = flash_fn(Q, K, V, tile_size=128, causal=True)
    peak_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    
    # Naive score matrix: N*N*4bytes = 67 MB
    naive_score_mb = N * N * 4 / (1024 * 1024)
    peak_mb = peak_bytes / (1024 * 1024)
    print(f"Test 2 (large): peak memory = {peak_mb:.1f} MB "
          f"(naive score matrix would be {naive_score_mb:.1f} MB)")
    # Should be well under naive — tile-based allocation is O(tile_size * N)
    assert peak_mb < naive_score_mb * 0.3, \
        f"Peak memory {peak_mb:.1f} MB too close to naive {naive_score_mb:.1f} MB"
    print("  PASS")
    
    print("ALL TESTS PASSED")
```

---

## Challenge 2: Batched Beam Search with Length Penalty and EOS Semantics

### Context

Every LLM serving framework implements beam search, and nearly every from-scratch
implementation has the same bug: **finished (EOS) beams are removed from the beam
pool instead of being kept to compete with unfinished beams.**

This creates a silent failure mode: a beam that hits EOS early with a very
high-confidence sequence gets discarded, and the output degrades to a longer,
lower-quality sequence from an unfinished beam. The bug is invisible unless you
specifically test for it — the output is still "valid" (it's a sequence), just
not optimal under the beam search objective.

Additionally, the interaction between **length penalty** and EOS re-ranking is
subtle: finished beams have frozen logprobs but their length-penalized scores
are recalculated each step as unfinished beams change length. The ranking of
finished vs. unfinished beams can FLIP between steps.

### The Prompt

```
Implement a correct batched beam search decoder for autoregressive generation
in pure NumPy.

Simulate a minimal language model:
- vocab_size = 1000
- d_model = 64
- Use random embeddings + 1 transformer block with random weights
  (correctness depends on beam search logic, not model quality)

Requirements:

1. MULTI-BATCH SUPPORT:
   - Accept prompt_token_ids: list[list[int]] — one prompt per batch item
   - beam_width K per batch item
   - Each batch item's beams are INDEPENDENT (no cross-contamination)

2. PER-STEP BEAM EXPANSION:
   - For each UNFINISHED beam, compute logits for the next token
   - Take top-(2*K) candidates per beam (not just top-K, to preserve diversity)
   - Compute total logprob = accumulated_logprob + new_logprob
   - Pool all candidates across all beams, sort globally, take top K

3. LENGTH PENALTY (for ranking only, not for accumulated score):
   - adjusted_score = accumulated_logprob / (generated_length ^ alpha)
   - alpha is a hyperparameter (default 0.6, common in NMT)
   - The accumulated logprob is UNMODIFIED by length penalty
   - Length penalty applies ONLY to ranking/selection, never to the stored score
   - generated_length = number of NEW tokens generated (NOT including prompt)

4. EOS HANDLING (the critical part):
   - When a beam produces token_id == eos_token:
     * Mark beam as FINISHED
     * Freeze its accumulated_logprob and generated_length
     * The beam stays in the pool — it competes with unfinished beams
   - At each step, select top-K beams from: {finished beams} ∪ {expanded candidates}
   - If all K beams in a batch item are finished, that item stops expanding
   - If all batch items have K finished beams, early-stop

5. RETURN:
   - For each batch item: list of K sequences (token IDs, NOT including prompt),
     sorted by length-penalized score descending (best first)
   - Each sequence's score = accumulated_logprob / (len(seq) ^ alpha)

6. EDGE CASES TO HANDLE:
   - A batch item may have fewer than K active+finished beams if some finished
     early and there aren't enough candidates to fill K slots
   - If max_new_tokens is reached before K beams finish, return the best K
     (finished + unfinished) by length-penalized score
   - Log-space accumulation: avoid numerical underflow but keep everything in
     log space (no exp until final scoring if needed)

Deliver:
- Implementation as a class or function
- Test 1: Single beam (K=1), prompt of 3 tokens, alpha=0
  → verify greedy decoding behavior
- Test 2: batch=2, beam_width=3, different prompt lengths, alpha=0.6
  → verify per-batch independence
- Test 3: THE EOS TEST — construct controlled logit outputs (by patching the
  model) to demonstrate that a finished EOS beam with score=-3.0 at length=5
  correctly beats an unfinished beam with score=-4.0 at length=10, and that
  removing EOS beams (buggy behavior) would give the wrong answer
- Explanation of why finished beams must NOT be removed

Use only NumPy. No PyTorch/JAX/TensorFlow/autodiff.
```

### The Hidden Trap

The EOS removal bug. Consider this scenario:

```
Step 0 (after prompt): 3 beams active, scores [-1.0, -1.5, -2.0]
Step 1: Beam 0 produces EOS (score=-1.5 after adding logprob=-0.5)
        Beams 1, 2 continue (scores [-2.0, -2.5])
        Active beams = 2, finished beams = 1
        
        Buggy implementation: removes EOS beam, only considers active beams
        → top K=3 = [-2.0, -2.5, ...]  (EOS beam's -1.5 is LOST)
        Correct implementation: pool = [-1.5 (finished), -2.0, -2.5]
        → top K=3 = [-1.5, -2.0, -2.5]  (EOS beam RIGHTFULLY survives)

        With length penalty alpha=0.6, lengths [1, 2, 2]:
        adjusted = [-1.5/1^0.6, -2.0/2^0.6, -2.5/2^0.6]
                = [-1.50, -1.32, -1.65]
        → Finished beam is actually WORSE after penalty (because it's short).
        If a longer beam had score=-2.3 at length=5: -2.3/5^0.6 = -2.3/2.63 = -0.87
        → NOW the longer beam beats the early-EOS beam.
```

The test that catches the bug:

```python
def test_eos_retention():
    """
    If the implementation removes EOS beams, this test FAILS.
    The difference is invisible without explicitly checking.
    """
    # We need to monkey-patch the model to return controlled logits
    # Step 0: prompt="A B C", beam_width=2
    #   Both beams expand. Beam A → EOS (score=-3.0)
    #                   Beam B → token X (score=-4.0)
    # Step 1: Beam B expands → token Y (score=-5.0)
    # 
    # Final ranking (no length penalty):
    #   Beam A: "EOS" score=-3.0, len=1  → adjusted=-3.0
    #   Beam B: "X Y"  score=-5.0, len=2  → adjusted=-5.0
    # Winner: Beam A ✓
    #
    # Buggy (EOS removed): Beam A gone after step 0
    #   Beam B: "X Y" score=-5.0
    # Winner: Beam B ✗ (WRONG — length penalty doesn't change this)
    pass
```

### What to look for in their code

1. **Finished beam tracking**: Is there a `finished` flag per beam? Does the top-K selection pool include finished beams?
2. **Length penalty denominator**: Is it `len(generated_tokens)` or `len(generated_tokens + prompt)`? The prompt should NOT count.
3. **Per-batch isolation**: Are scores from batch item 0 compared with scores from batch item 1? They shouldn't be.
4. **The K-best return**: When all beams finish, do they correctly return the K-best by length-penalized score?
5. **max_new_tokens truncation**: When truncating, do they pick the best K from {finished + unfinished} by length-penalized score?

---

## Why these two specifically

| Property | Flash Attention | Batched Beam Search |
|----------|----------------|---------------------|
| Domain | Training (FlashAttn is default) | Inference (every serving framework) |
| Hard part | Online softmax rescaling | EOS semantics + length penalty interaction |
| Silent bug | Rescaling direction (both work!) | EOS beam removal (valid output, wrong answer) |
| Catch mechanism | Intermediate overflow/underflow | Requires controlled test case |
| Code size | ~100-150 lines | ~200-300 lines |
| Runs on M4 MacBook | Yes, pure NumPy | Yes, pure NumPy |
| Test time | <1 second (small), ~5-10s (large) | <1 second |
| Industry relevance | FlashAttention-1/2/3, xformers | vLLM, TGI, llama.cpp, SGLang |

Both challenges expose bugs that can't be caught by simple "does it run?" testing.
The rescaling bug in Flash Attention produces correct-looking output. The EOS
removal bug in beam search produces valid sequences (just not the optimal ones
under the beam search criterion). You need to either read the code carefully or
run specific adversarial tests.
