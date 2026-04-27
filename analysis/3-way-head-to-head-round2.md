# 3-Way Head-to-Head: Harder Challenges

## Executive Summary

| Dimension | GLM-5 | MiniMax-M2.7 | Qwen3-6 |
|-----------|-------|-------------|---------|
| **Flash Attention Grade** | A- | B | A |
| **Beam Search Grade** | B+ | B- | A |
| **Overall Grade** | B+ | B- | A |
| **All tests pass?** | ✓ | ✓ | ✓ |
| **Silent bugs?** | 0 | 1 (cosmetic) | 0 |

**Context note:** These runs used **opencode** (full LSP/tooling harness) instead of the
minimal pi-mono harness used in the first round. The system prompt is substantially
larger and the environment has more tooling. This may affect verbosity and
architectural choices but shouldn't impact algorithmic correctness.

---

## Challenge 1: Tiled Flash Attention (Online Softmax)

### The Hidden Trap — Rescaling Direction

The flash attention online softmax recurrence requires:
```
correction = exp(m_old - m_new)    # ≤ 1, correct
```

The most common bug in open-source implementations is writing this as
`exp(m_new - m_old)` instead. Both produce identical final output (because
O/l is invariant to the correction factor magnitude), but with the wrong
direction, intermediate O and l values grow without bound, causing
overflow/underflow in the backward pass or with fp16. The bug is invisible
in forward-only correctness tests.

**All three models got this right in their code.**

### GLM-5 (`glm5/flash_attention/flash_attention.py`, ~215 lines)

**Grade: A-**

**Strengths:**
- Correct online softmax with `correction = np.exp(m_tile - m_new)`
- Clean per-(b,h) loop structure that mirrors how GPU kernels are organized
- Handles the fully-masked first tile NaN hazard with `np.isfinite` guards:
  ```python
  safe_mask = np.isfinite(m_new)
  P_tile = np.where(safe_mask[:, None], P_tile, 0.0)
  ```
- Also handles l==0 at normalization time (masked rows get output 0)
- Causal skip optimization: `if causal and k_start > q_end - 1: continue`
- Uses float64 → relative error 1.85e-16 (near machine epsilon)
- Peak memory 34.24 MB vs 134.22 MB for naive (N=4096, D=64) — well under
- Comments clearly explain WHY correction is `exp(m_old - m_new)` and NOT `exp(m_new - m_old)` with a concrete derivation
- Two tests delivered as specified

**Weaknesses:**
- Per-(b,h) Python loops make it slow for large B,H (the matmul within tiles is vectorized, but the outer loops are serial). In a GPU kernel this structure is correct, but for NumPy testing, a fully batched `einsum` would be faster.
- Only tests causal=True. Doesn't test non-causal mode
- No test for uneven tile sizes (N not divisible by tile_size)
- No multi-head test (H>1) separately — the large test does H=8 but doesn't verify correctness on each head

### MiniMax-M2.7 (`minimax-m2.7/flash_attention/flash_attention.py`, ~370 lines)

**Grade: B**

**Strengths:**
- Code produces correct results (rel error 1.25e-10, passes all tests)
- Correctly implements online softmax with `exp(m[valid_corr_mask] - m_new_flat[valid_corr_mask])`
- Explicit handling of `m_old == -inf` edge case with boolean masks:
  ```python
  m_old_is_neg_inf = m == -np.inf
  need_correction = ~(m_old_is_neg_inf & m_new_is_neg_inf)
  ```
- Causal skip condition is correct: `if kv_tile_start >= q_tile_end: continue`
- Peak memory 32.59 MB vs 128 MB naive — good
- Third test on N=512 for additional correctness verification
- Handles `l==0` at normalization (if all KV tiles were masked)

**Weaknesses:**
- **The explanation of the rescaling factor IS WRONG in the docstring.** The docstring contains a ~80-line monologue where the model argues with itself about the correction direction, initially claiming it should be `exp(m_new - m_old)` and "This is WRONG!" before eventually talking itself into the right answer. But in one line it says:
  ```
  O_new = O_old * exp(m_new - m_old)  # This is WRONG! Unless...
  ```
  Then later contradicts and gets it right. The confusion in the explanation is a red flag — it suggests the model is parroting a learned pattern rather than reasoning from first principles. The code happens to be correct, but the explanation reveals uncertainty.
- Per-(b,h) loops with per-row Python for-loop for exp computation:
  ```python
  for i in range(S.shape[0]):
      if not np.isinf(m_new_flat[i]):
          exp_S_minus_m_new[i] = np.exp(S[i] - m_new_flat[i])
  ```
  This Python for-loop over query positions within each tile is **extremely slow** for realistic tile sizes. It should be a vectorized operation.
- The docstring is 3× longer than the actual code — mostly confused self-dialogue about the rescaling
- Uses `np.matmul` but then loops per-row for exp — inconsistent vectorization strategy
- No tracemalloc in the actual large test (just printed analysis, which is fine but less rigorous)

### Qwen3-6 (`qwen36/flash_attention/flash_attention.py`, ~310 lines)

**Grade: A**

**Strengths:**
- **Most efficient implementation**: uses `np.einsum` for batched score computation, avoiding per-(b,h) loops entirely
- Precomputes Q_tiles, K_tiles, V_tiles as lists — clean separation of tiling from computation
- Proper `row_valid` masking to handle the -inf - (-inf) = NaN problem:
  ```python
  row_valid = row_max > -np.inf
  correction = np.exp(np.where(row_valid, m - m_new, 0.0))
  P = np.where(row_valid[:, :, :, np.newaxis], P, 0.0)
  ```
- Handles `l==0` gracefully at normalization with `l_safe = np.where(l > 0, l, 1.0)`
- **5 tests**: accuracy (causal), non-causal, larger batch (B=2,H=8), uneven tiles (N=300, tile_size=97), memory
- Peak memory 27.2 MB (float32 → smaller absolute numbers, still well under 64 MB naive)
- Relative error 5.93e-08 — slightly higher than GLM/MiniMax but only because float32, not float64. Still well within 1e-4 threshold
- Comments explain the correction factor concisely and correctly
- Test 5 (uneven tiles) specifically catches off-by-one tile boundary bugs

**Weaknesses:**
- Precomputing all tiles as a list duplicates memory (stores all tiles at once). In a production GPU kernel you'd stream them, but for NumPy testing this is acceptable
- Uses float32 instead of float64 (slightly less precision, but the threshold is 1e-4 so it's fine)
- The non-causal test is Test 3 (out of order) which is slightly confusing labeling
- The `correct = np.exp(np.where(...))` pattern wastes a tiny amount of compute on invalid rows (computes `exp(0.0) = 1.0` then discards via masking). Not a real issue but slightly inefficient

### Flash Attention Winner: **Qwen3-6** (narrowly over GLM-5)

Qwen3-6 wins on breadth (5 tests vs 2), efficiency (batched einsum vs per-head loops), and edge case handling (uneven tiles). GLM-5's implementation is equally correct in the core algorithm and has better (float64) precision and cleaner comments. MiniMax's implementation is correct but the confused docstring and per-row Python loops for exp are concerning.

| Metric | GLM-5 | MiniMax | Qwen3-6 |
|--------|-------|---------|---------|
| Correct rescaling | ✓ exp(m_old-m_new) | ✓ exp(m_old-m_new) | ✓ exp(m_old-m_new) |
| NaN handling | ✓ isfinite guard | ✓ -inf mask | ✓ row_valid mask |
| Causal skip | ✓ per-tile check | ✓ per-tile check | ✓ per-tile check |
| Uneven tiles | Not tested | Not tested | ✓ tested (N=300, T=97) |
| Non-causal | Not tested | Not tested | ✓ tested |
| Multi-head test | H=8 in large test | H=8 in large test | ✓ separate test |
| Peak memory (N=4096) | 34 MB (float64) | 33 MB (float64) | 27 MB (float32) |
| Vectorization | Per-(b,h) loops | Per-(b,h) + per-row exp | Batched einsum |
| Num tests | 2 | 3 (1 extra) | 5 |
| Doc quality | Clear, correct | Confused self-dialogue | Clear, concise |

---

## Challenge 2: Batched Beam Search with EOS Semantics

### The Hidden Trap — EOS Beam Removal

When a beam produces EOS, it represents a complete high-confidence candidate.
If you remove it from the pool, a longer lower-confidence unfinished beam
might "win" simply because it hasn't stopped yet. The correct behavior:
finished beams stay in the pool and compete via their frozen logprob +
length-penalized score against unfinished beams' growing scores.

**All three models correctly retain EOS beams in the pool. None commit the removal bug.**

### GLM-5 (`glm5/beam_search/beam_search.py`, ~190 lines)

**Grade: B+**

**Strengths:**
- Most concise implementation — just 190 lines including tests and mock model
- Beam tracking is clean: each beam is `(tokens, acc_logprob, finished)`
- Correctly pools candidates + finished beams at each step:
  ```python
  pool = candidates + finished
  pool.sort(key=lambda b: penalized_score(b[1], len(b[0])), reverse=True)
  beams = pool[:K]
  ```
- `_make_logits` helper correctly constructs logits that produce exact logprobs after softmax — this is important for the EOS test
- `_MockModel` class is simple and reusable
- EOS test verifies exact score (-3.0) for the EOS beam
- Length penalty `alpha=0.0` test verifies greedy equivalence
- Comments clearly explain why finished beams must stay in the pool

**Weaknesses:**
- Sequences are lists of ints stored in tuples — mutable but tracked via tuple identity, which works but is slightly fragile
- `_make_logits` spreads remaining probability mass uniformly — this means non-EOS tokens have non-zero probability even when not explicitly set, potentially affecting beam exploration. In the EOS test this is fine because K=2
- Per-batch sequential execution (for loop over prompts) — not truly "batched" in the simultaneous sense
- Only 3 tests
- No test for the length penalty interaction case (alpha > 0 with two EOS beams at different lengths)
- The beam representation `(tokens, acc_logprob, finished)` uses a 3-tuple with implicit field ordering — easy to confuse

### MiniMax-M2.7 (`minimax-m2.7/beam_search/beam_search.py`, ~360 lines)

**Grade: B-**

**Strengths:**
- Full `MinimalLanguageModel` with multi-head attention built in (overkill but complete)
- Proper variable-length batching with padded sequences and per-beam tracking
- Uses `np.argpartition` for efficient top-k selection
- Per-batch candidate filtering: `batch_candidates = [c for c in all_candidates if c['batch_idx'] == batch_idx]`
- EOS test correctly identifies that the EOS beam wins
- Comments explain EOS retention rationale

**Weaknesses:**
- **EOS logprob vs logit confusion in the test**. The EOS test uses pre-softmax logits (`5.0` for EOS, `3.0` for continue) and relies on softmax conversion. The model then converts logits → probs → logprobs via `np.log(token_prob)`. This means the actual accumulated logprobs are softmax-transformed values, not the controlled values the test thinks. The test prints `score=0.0000` for the EOS beam — that's because 5.0 is a very high logit, and after softmax the probability is near 1.0, logprob near 0.0. The test "passes" but the scores aren't the exact values specified.
- **Finished beams are added to `finished_results` immediately** but also **removed from `active_beams`**:
  ```python
  if c['finished']:
      finished_results[c['batch_idx']].append({...})
  else:
      new_active_beams.append({...})
  ```
  This means finished beams move to `finished_results` and don't compete in subsequent steps' candidate selection! Wait, let me re-read...
  
  Actually, looking more carefully: at the end of each step, finished beams go to `finished_results` and unfinished beams go to `active_beams`. Then at the START of the next step, ALL finished beams (already moved to finished_results) are NOT re-added to the active pool. So they don't compete.
  
  But wait — there's a loop that adds accumulated finished beams back:
  ```python
  if all(beam['finished'] for beam in beams):
      for beam in beams:
          finished_results[batch_idx].append({...})
      active_beams[batch_idx] = []
  ```
  But this only triggers when ALL beams are finished, not when SOME are finished.
  
  When SOME beams are finished: the finished ones go to `finished_results` and the unfinished ones continue as `active_beams`. At the next step, only `active_beams` are expanded. Finished beams in `finished_results` are NOT re-added to the candidate pool for the next step's top-K selection.
  
  **THIS IS THE EOS REMOVAL BUG!** The implementation only keeps finished beams in the pool within a single step (via `all_candidates` which excludes them — actually wait, `all_candidates` only contains candidates from EXPANDING beams, not finished beams from previous steps).
  
  Let me re-read one more time... The candidate collection at step N:
  ```python
  for beam in beams:
      if beam['finished']: continue  # skip finished beams
      # expand...
  ```
  So finished beams don't produce candidates. The pool for top-K at step N is ONLY `all_candidates` (which excludes finished beams from previous steps because they were moved to `finished_results`).
  
  But the test passes because... the EOS test has beam_width=3 and only PATCHES step 1. At step 1, the EOS beam is created as a CANDIDATE (from expanding an unfinished beam). This candidate is then selected into the active set if its length-penalized score is good enough. Then it's moved to `finished_results` and removed from `active_beams`.
  At step 2, only the remaining unfinished beam is expanded. Its candidates compete with... only themselves (finished beams from step 1 are in `finished_results`, not in `all_candidates`). But beam_width=3 and there's only 1 unfinished beam producing 2*K=6 candidates, so 3 are selected, all from the continuing beam.
  
  Wait, but then how does the EOS beam WIN in the final output? Because at the end, `finished_results` contains all finished beams and `active_beams` contains unfinished ones, and BOTH are merged into `results`:
  ```python
  results.append(scored_results[:beam_width])
  ```
  where `scored_results` combines both sources. So the EOS beam DOES appear in the final output and CAN win.
  
  But the issue is: during step 2 selection, the EOS beam doesn't compete against the continuing beam's candidates for the K active slots. The continuing beam fills all K slots with its own candidates. So at step 3, all K active beams are children of the continuing beam — the EOS beam is not "active" anymore. This means:
  - If beam_width=1: at step 1, EOS finishes. Step 2: only the unfinished beam produces candidates. The unfinished beam fills the 1 slot. Step 3: same. Final output: EOS beam (score=-3.0) vs unfinished beam (score=-5.0). EOS wins. ✓
  - If beam_width=2: same as above, the unfinished beam fills 2 slots. ✓
  - **The bug appears when beam_width is small and there are more unfinished beams than slots**: the EOS beam might be pushed out of the top-K because only candidates from unfinished beams are being considered, not past finished beams.
  
  Actually, the final output merges ALL finished + unfinished beams, sorts by score, and takes top-K. So the EOS beam from step 1 is in `finished_results` and WILL appear. The only case where this fails is if there are K+1 unfinished beams that all have better scores than the EOS beam, AND the EOS beam would have been in the top-K if it had been allowed to compete at step selection time. But that's the correct behavior — if K other beams are truly better, they should win.
  
  The real EOS removal bug is when finished beams are DISCARDED entirely, not when they're moved to a separate list but still included in final ranking. MiniMax includes them in final ranking. So this is actually CORRECT behavior, just implemented in a roundabout way.
  
  OK, I was over-analyzing. The implementation is functionally correct for the test case. Let me reconsider the grade.

- The implementation is very verbose and the control flow is hard to follow: `active_beams` ↔ `finished_results` ↔ `all_candidates` with per-batch indices scattered throughout
- Dictionary-based beam representation with string keys is fragile and slow
- The `all_candidates` list grows unboundedly across batches per step — not per-batch filtered until selection time
- No test for the length penalty + two EOS beams case

**Re-graded to B-** primarily for the complexity and confusing architecture, not for a correctness bug.

### Qwen3-6 (`qwen36/beam_search/beam_search.py` + `model.py` + `test_beam_search.py`, ~380 lines total)

**Grade: A**

**Strengths:**
- **Best architecture**: 3 separate files — `model.py` (MinimalLM), `beam_search.py` (algorithm), `test_beam_search.py` (4 tests)
- `Beam` class with `__slots__` for memory efficiency and clean `length_penalized_score()` method
- Clear separation: `accumulated_logprob` (never modified by penalty) vs `length_penalized_score()` (used only for ranking)
- `MockModel` class with `set_log_probs()` for precise control: returns EXACT log probs (not pre-softmax logits), so test assertions on exact scores work perfectly
- **4 tests**, including the critical "Test 3b" that verifies length penalty interaction:
  - Greedy equivalence (K=1, alpha=0)
  - Batch independence with cross-validation against solo runs
  - EOS retention: verifies exact score -3.0 for EOS beam, -6.0 for continuing
  - EOS retention + length penalty: longer beam (-1.0 at len=2) beats shorter beam (-2.0 at len=1) because -1.0/2^0.6 ≈ -0.66 > -2.0/1^0.6 = -2.0
- Uses `np.argpartition` for efficient top-k selection
- `finished_beams` list is maintained alongside `beams` list — both participate in ranking
- Comments comprehensively explain the EOS retention rationale

**Weaknesses:**
- `MinimalLM.forward()` only returns logits for the last token position (not the full sequence). This is correct for beam search but inconsistent with the docstring
- The `get_log_probs` method recomputes forward for every candidate — could be batched, but for a correctness test this is fine
- Per-batch sequential execution (for loop over batch items) — the same as GLM
- The `Beam` class uses `__slots__` which is a Python optimization detail that doesn't matter for a toy implementation (but shows the model is thinking about memory)

### Beam Search Winner: **Qwen3-6** (decisive)

Qwen3-6 wins on every dimension: architecture (3 files, Beam class), testing (4 tests including the critical length penalty interaction), mocking precision (exact logprobs), and code clarity. GLM-5's implementation is correct and concise but has fewer tests. MiniMax's implementation is architecturally confusing with dictionary-based beams and indirect finished beam tracking.

| Metric | GLM-5 | MiniMax | Qwen3-6 |
|--------|-------|---------|----------|
| Beam data structure | 3-tuple (tokens, lprob, finished) | dict with string keys | Beam class with __slots__ |
| EOS retention | ✓ pooled + sorted | ✓ merged at final | ✓ finished_beams list |
| EOS test precision | Exact logprobs (-3.0, -4.0) | Logit-based (scores vary) | Exact logprobs (-3.0, -6.0) |
| Length penalty test | 0.0 only (greedy) | 0.6 (basic) | 0.0 + 0.6 with two EOS beams |
| Batch independence | ✓ solo comparison | ✓ token overlap check | ✓ solo comparison + score check |
| Model simulation | `_make_logits` + MockModel | Full transformer inlined | Separate `MinimalLM` + MockModel |
| Tests | 3 | 3 | 4 |
| Code clarity | Clean, concise | Verbose, hard to follow | Clean, well-separated |
| Vectors | Lists of tuples | Lists of dicts | List of Beam objects |

---

## Cross-Challenge Patterns

### Code Quality & Architecture (OpenCode vs pi-mono)

The switch from pi-mono to opencode appears to have had these effects:

| Aspect | pi-mono (Round 1) | opencode (Round 2) |
|--------|-------------------|---------------------|
| File organization | Mixed (1 file to 8 files) | All single-file except Qwen (3 files) |
| Verbosity | Moderate | Higher (MiniMax docstring is 3× code) |
| Comments/explanation | More concise | More verbose, sometimes confused |
| Test quality | Good | Generally still good |
| Architecture decisions | More opinionated | More "safe" (single file, less modular) |

The larger system prompt in opencode doesn't seem to have hurt correctness — all implementations pass. But it may have encouraged more verbosity (MiniMax's 80-line confused self-dialogue) and less confident architectural choices (flatter file structures).

### The Critical Tests

Both challenges had adversarial tests designed to catch specific bugs:

| Bug | Test | GLM-5 | MiniMax | Qwen3-6 |
|-----|------|-------|---------|----------|
| Rescaling direction (flash attn) | Check `exp(m_old - m_new)` vs `exp(m_new - m_old)` in code | ✓ Correct | ✓ Correct | ✓ Correct |
| NaN from -inf - (-inf) (flash attn) | Causal with fully masked first tile | ✓ isfinite guard | ✓ -inf mask | ✓ row_valid mask |
| EOS beam removal (beam search) | EOS at step 1 with score -3.0 vs cont at -5.0 | ✓ EOS wins | ✓ EOS wins | ✓ EOS wins, exact score |
| Length penalty interaction | Two EOS beams at different lengths | Not tested | Not tested | ✓ tested (alpha=0.6) |

The most interesting finding: **MiniMax's flash attention docstring gets the rescaling explanation wrong** (it argues with itself about the direction) but the code is correct. This suggests the model has pattern-matched the correct code but doesn't have deep understanding of why it's correct. In a variant of the problem (e.g., changing the recurrence slightly), this model would likely produce buggy code because it's reciting rather than reasoning.

### The NaN Hazard

All three models correctly identified and handled the NaN hazard from `exp(-inf - (-inf))` when the first KV tile is fully masked under causal attention. The approaches differ:

- **GLM-5**: `np.isfinite(m_new)` guard → zeros out P for invalid rows
- **MiniMax**: `m_old_is_neg_inf & m_new_is_neg_inf` boolean mask → sets correction to 1, zeros out P
- **Qwen3-6**: `row_valid = row_max > -np.inf` → zeros out P, correction = exp(0) = 1 for invalid rows

All three are correct. Qwen3-6's approach is cleanest because it detects validity at the source (are there any valid scores in this row?) rather than checking secondary conditions.

---

## Overall Rankings (Harder Challenges)

### 1st Place: **Qwen3-6** (A)

Wins both tasks. Flash attention: most efficient (batched einsum), 5 tests including uneven tiles and non-causal. Beam search: best architecture (3 files, Beam class, MockModel), 4 tests including the length penalty interaction case, exact score verification. The only model that separated the language model from the beam search algorithm into different files — exactly the right engineering instinct.

### 2nd Place: **GLM-5** (B+)

Solid, correct implementations with concise code. Flash attention: clean online softmax, good NaN handling, float64 precision. Beam search: correct EOS retention, clean tuple-based tracking. Weaknesses are scope: fewer tests, missing edge cases (no uneven tiles, no length penalty interaction), no non-causal flash attention mode.

### 3rd Place: **MiniMax-M2.7** (B-)

All tests pass but the implementations have concerning properties. Flash attention: correct code but the docstring shows confusion about the rescaling direction — the model clearly doesn't understand why the formula works. Beam search: architecturally confusing with dictionary-based beams and indirect EOS tracking. The code works but the reasoning and structure are both weaker than competitors. The 80-line self-dialogue in the flash attention docstring is a red flag for frontier model capability.

---

## Combined Rankings (Both Rounds)

| Model | Round 1 Grade | Round 2 Grade | Combined |
|-------|--------------|--------------|----------|
| **Qwen3-6** | A- | A | **A/A-** |
| **GLM-5** | B+ | B+ | **B+** |
| **MiniMax-M2.7** | B | B- | **B/B-** |

### Final Takeaways

1. **Qwen3-6 (27B, local) continues to outperform two frontier-class models.** The gap actually **widened** in round 2 — Qwen3-6 got STRONGER on harder tasks while the others stayed flat or regressed slightly. This is the opposite of what you'd expect if scale was the primary driver.

2. **The rescaling direction test didn't catch anyone** — all three models wrote `exp(m_old - m_new)` correctly in code. This might mean the bug is well-known enough to be in training data, or the models genuinely understand the math. MiniMax's confused docstring argues for the former.

3. **The EOS retention test also didn't catch anyone** — all three correctly keep finished beams. This is encouraging since the EOS removal bug is common in production frameworks.

4. **The opencode harness may have been a slight negative.** The implementations in round 2 are less modular (more single-file, fewer test files) than round 1. Qwen3-6 maintained modularity (3 files for beam search). GLM-5 went from 3 files (KV-cache) to single files. MiniMax was already monolithic.

5. **The most differentiating factor remains engineering discipline, not algorithmic knowledge.** All three models understand flash attention and beam search. What separates them is testing thoroughness (5 tests vs 2), edge case handling (uneven tiles, non-causal), mock precision (exact logprobs vs logits), and code organization (Beam class vs tuple vs dict).
