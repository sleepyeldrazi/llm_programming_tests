# 2-Way Head-to-Head: Harder Challenges (Round 2)

These pairwise comparisons cover the two harder challenges: Flash Attention and Beam Search.
All implementations were run under opencode (full LSP tooling harness) rather than the
minimal pi-mono harness from Round 1.

---

## GLM-5 vs MiniMax-M2.7

### Flash Attention

| Criteria | GLM-5 | MiniMax-M2.7 | Edge |
|----------|-------|-------------|------|
| Lines of code | ~215 | ~370 | **GLM** (more concise) |
| Correct rescaling | ✓ exp(m_old - m_new) | ✓ exp(m_old - m_new) | Tie |
| NaN handling | np.isfinite guard | -inf boolean mask | Tie (both correct) |
| Causal skip | ✓ per-tile | ✓ per-tile | Tie |
| Rel error (N=256) | 1.85e-16 (float64) | 1.25e-10 (float64) | **GLM** (1000× better) |
| Peak memory (N=4096) | 34.2 MB | 32.6 MB | Tie |
| Tests | 2 (specified) | 3 (extra N=512 check) | MiniMax |
| Vectorization | Per-(b,h) loops | Per-(b,h) + per-row Python for-loop for exp | **GLM** (no per-row loops) |
| Doc quality | Clear, correct derivation | **Confused 80-line self-dialogue** that initially claims wrong answer | **GLM** |
| Extra modes | Causal only | Causal only | Tie |

**Winner: GLM-5** — The critical differentiator is MiniMax's docstring which **argues with itself about the rescaling direction**, initially claiming the wrong answer before correcting. The code is right but the reasoning is shaky. GLM-5's code is also faster (no per-row Python exp loop inside the tile loop).

### Beam Search

| Criteria | GLM-5 | MiniMax-M2.7 | Edge |
|----------|-------|-------------|------|
| Lines of code | ~190 | ~360 | **GLM** |
| Beam representation | 3-tuple (tokens, lprob, finished) | dict with 7 string keys | **GLM** |
| EOS retention | ✓ pool = candidates + finished | ✓ merged at final ranking | Tie |
| Mocking precision | Exact logprobs via _make_logits | Pre-softmax logits → scores vary | **GLM** |
| EOS test score | -3.0 (exact) | 0.0 (transformed by softmax) | **GLM** |
| Length penalty test | alpha=0.0 only | alpha=0.6 basic | **MiniMax** |
| Batch independence | Solo comparison | Token-overlap + solo | Tie |
| Tests | 3 | 3 | Tie |
| Code clarity | Clean, easy to follow | Verbose, hard to trace control flow | **GLM** |
| Model integration | MockModel class + MinimalTransformer | Full transformer inlined in test | **GLM** |

**Winner: GLM-5** — Both are correct, but GLM-5's implementation is dramatically cleaner. MiniMax uses dictionaries-with-string-keys for beams and has complex control flow between `active_beams`, `finished_results`, and `all_candidates`. The mocking precision is also better in GLM-5 (exact logprob control vs logit-based).

### GLM-5 vs MiniMax-M2.7 Overall: **GLM-5 wins 2-0**

---

## MiniMax-M2.7 vs Qwen3-6

### Flash Attention

| Criteria | MiniMax-M2.7 | Qwen3-6 | Edge |
|----------|-------------|---------|------|
| Correct rescaling | ✓ | ✓ | Tie |
| NaN handling | -inf boolean mask | row_valid = row_max > -inf | **Qwen** (cleaner) |
| Tests | 3 (causal ×2, N=512 extra) | 5 (causal, non-causal, multi-head, uneven tiles, memory) | **Qwen** |
| Non-causal support | Not tested | ✓ tested | **Qwen** |
| Uneven tiles | Not tested | ✓ N=300, T=97 | **Qwen** |
| Rel error | 1.25e-10 (float64) | 5.93e-08 (float32) | **MiniMax** (dtype artifact) |
| Peak memory | 32.6 MB (float64) | 27.2 MB (float32) | Tie (both proportional to dtype) |
| Vectorization | Per-(b,h) + per-row Python exp loop | Batched einsum, no per-row loops | **Qwen** |
| Doc quality | Confused self-dialogue | Clear, concise | **Qwen** |
| Multi-head | H=8 in large test only | H=8 with separate correctness check | **Qwen** |

**Winner: Qwen3-6** — Decisive. More tests, cleaner code, better vectorization, correct doc. MiniMax's confused docstring is a major red flag even though the code works.

### Beam Search

| Criteria | MiniMax-M2.7 | Qwen3-6 | Edge |
|----------|-------------|---------|------|
| Files | 1 (monolithic) | 3 (model.py + beam_search.py + test) | **Qwen** |
| Beam representation | dict with 7 string keys | Beam class with __slots__ | **Qwen** |
| EOS retention | ✓ (indirect, via final merge) | ✓ (direct, finished_beams list in pool) | **Qwen** |
| Mocking precision | Logit-based (can't control exact scores) | Exact logprobs via MockModel.set_log_probs | **Qwen** |
| Tests | 3 | 4 (includes length penalty + two EOS beams) | **Qwen** |
| Length penalty interaction | Only basic alpha=0.6 | Test 3b: alpha=0.6, two EOS beams at different lengths, verifies longer beam wins correctly | **Qwen** |
| Batch independence verification | Token-overlap check (weak) | Solo comparison + exact score match | **Qwen** |
| Code clarity | Hard to follow control flow | Clean, well-separated concerns | **Qwen** |

**Winner: Qwen3-6** — Decisive on every dimension. The length penalty interaction test (3b) is the strongest differentiator — only Qwen3-6 tested that two EOS beams at different lengths interact correctly with the penalty.

### MiniMax-M2.7 vs Qwen3-6 Overall: **Qwen3-6 wins 2-0**

---

## Qwen3-6 vs GLM-5

### Flash Attention

| Criteria | Qwen3-6 | GLM-5 | Edge |
|----------|---------|-------|------|
| Correct rescaling | ✓ | ✓ | Tie |
| NaN handling | row_valid = row_max > -inf (cleaner) | np.isfinite(m_new) (also correct) | Tie |
| Tests | 5 tests | 2 tests | **Qwen** |
| Non-causal | ✓ tested | Not tested | **Qwen** |
| Uneven tiles | ✓ N=300, T=97 | Not tested | **Qwen** |
| Multi-head B,H>1 | ✓ separate test | H=8 in large test only | **Qwen** |
| Vectorization | Batched einsum (fast) | Per-(b,h) loops (correct but slower) | **Qwen** |
| Precision | 5.93e-08 (float32) | 1.85e-16 (float64) | **GLM** (dtype choice) |
| Peak memory | 27.2 MB (float32) | 34.2 MB (float64) | Tie |
| Code clarity | Clear, concise | Clear, concise | Tie |
| Doc quality | Concise and correct | Concise and correct with derivation | **GLM** (slightly better derivation) |

**Winner: Qwen3-6** — GLM-5 has slightly better precision by using float64 (but both are well within 1e-4). Qwen3-6 wins on breadth: 5 tests covering modes GLM-5 didn't attempt (non-causal, uneven tiles), and batched einsum is more efficient. This is a close call — GLM-5's core algorithm is equally correct.

### Beam Search

| Criteria | Qwen3-6 | GLM-5 | Edge |
|----------|---------|-------|------|
| Files | 3 (model, beam_search, test) | 1 (monolithic) | **Qwen** |
| Beam representation | Beam class with __slots__ | 3-tuple (tokens, lprob, finished) | **Qwen** (more readable) |
| EOS retention | ✓ finished_beams list | ✓ pool = candidates + finished | Tie |
| Mocking precision | Exact logprobs via MockModel | Exact logprobs via _make_logits | Tie |
| Tests | 4 tests | 3 tests | **Qwen** |
| Length penalty + two EOS beams | ✓ Test 3b (alpha=0.6) | Not tested | **Qwen** |
| Greedy equivalence | ✓ K=1, alpha=0 | ✓ K=1, alpha=0 | Tie |
| Batch independence | ✓ Solo comparison + score match | ✓ Solo comparison | Tie |
| Code organization | Model separated from algorithm | Model inlined (MinimalTransformer) | **Qwen** |
| Score verification precision | Exact (-3.0, -6.0) | Exact (-3.0) | **Qwen** |

**Winner: Qwen3-6** — This is close. Both correctly retain EOS beams and use exact logprob control for testing. Qwen3-6 edges ahead with: (a) the length penalty + two EOS beams test (Test 3b), which verifies that the penalty correctly flips the ranking when a longer sequence has higher confidence; (b) better code organization with separate model file; (c) Beam class vs raw tuple for readability.

### Qwen3-6 vs GLM-5 Overall: **Qwen3-6 wins 2-0**

---

## Summary Matrix (Round 2)

| Matchup | Flash Attention | Beam Search | Overall |
|---------|----------------|-------------|---------|
| **GLM-5 vs MiniMax** | GLM | GLM | **GLM 2-0** |
| **MiniMax vs Qwen3-6** | Qwen | Qwen | **Qwen 2-0** |
| **Qwen3-6 vs GLM-5** | Qwen | Qwen | **Qwen 2-0** |

---

## Combined Rankings (Both Rounds)

| Matchup | Round 1 | Round 2 | Trend |
|---------|---------|---------|-------|
| **GLM-5 vs MiniMax** | GLM 3-0 | GLM 2-0 | GLM dominance holds |
| **MiniMax vs Qwen3-6** | Qwen 3-0 | Qwen 2-0 | Qwen dominance holds |
| **Qwen3-6 vs GLM-5** | Qwen 2.5-0.5 | Qwen 2-0 | Qwen gap WIDENS |

### Overall Model Rankings

| Rank | Model | Round 1 Grade | Round 2 Grade | Notes |
|------|-------|--------------|--------------|-------|
| 1 | **Qwen3-6** | A- | A | Gap over #2 widened. Only model that maintained multi-file architecture in opencode |
| 2 | **GLM-5** | B+ | B+ | Consistent. Strong algorithms, clean code, limited test coverage vs Qwen |
| 3 | **MiniMax-M2.7** | B | B- | Regressed slightly. Confused docstring in flash attn, architectural mess in beam search |

### Key Observations from Round 2

1. **The EOS retention trap didn't catch anyone.** All three models correctly kept finished beams in the pool. This suggests either the bug is well-represented in training data (many blog posts about "the beam search EOS bug") or the prompt's explicit warning ("Do NOT remove finished beams from the pool") was too heavy a hint.

2. **The rescaling direction trap also didn't catch anyone** in code, but MiniMax's docstring reveals shaky understanding. The model wrote correct code but couldn't explain why — classic pattern-matching behavior. If you modify the recurrence slightly (e.g., use a different normalization), MiniMax would likely produce buggy code because it's reciting rather than reasoning.

3. **The strongest differentiating test was length penalty interaction** (Qwen3-6 Test 3b). Neither GLM-5 nor MiniMax tested that two EOS beams at different lengths interact correctly with the penalty. This is a subtle bug that would pass basic "does EOS stay?" tests.

4. **OpenCode vs pi-mono impact:** The richer harness seemed to encourage verbosity (MiniMax's 80-line self-dialogue) and slightly less modular code (GLM-5 went from multi-file to single-file). Qwen3-6 was unaffected — maintained 3-file architecture.

5. **Qwen3-6's local 27B advantage over frontier models is real.** The gap widened in round 2, suggesting the model's strength is genuine reasoning/engineering discipline rather than luck on the specific tasks. The pattern of doing MORE than asked (extra tests, extra modes, extra files) while maintaining correctness is consistent.
