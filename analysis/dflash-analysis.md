# DFlash Challenge: Tree Attention Verification — Head-to-Head Analysis

## Executive Summary

| Model | All tests pass? | Logits extraction | Branching support | Subtree invalidation test | Overall |
|-------|---------|-------------------|-------------------|---------------------------|---------|
| **GLM-5** | ✓ | Parent-indexed ✓ | ✓ correct | ✓ actually verifies skipped | **A** |
| **GLM-5.1** | ✓ | Parent-indexed ✓ | ✓ correct | Partial (chain only) | **B+** |
| **Kimi K2.6** | ✓ | Positional (chain only) ✗ | ✗ broken | Illusory (break hides bug) | **B-** |
| **Qwen3-6** | ✓ | Depth-based (broken for branching) ✗ | ✗ broken | Illusory (break hides bug) | **B-** |
| **MiniMax-M2.7** | — | — | — | — | No submission |

**All four participants pass every test.** The difference is in how they extract the
verification logits — and whether their tests would catch branching-tree failures.

---

## The Central Hidden Trap: Logits Extraction

The prompt's pseudo-code says:
```python
tree_logits = logits[len(generated_tokens):]  # logits[P : P+N]
```

This is **wrong for branching trees**. In a standard autoregressive transformer,
`logits[j]` predicts the token at position `j+1`. To verify tree node i:

- **Root node** (parent = -1): the model should predict based on the full prompt.
  Correct source: `logits[P-1]` (the prompt's last position).
- **Non-root node** with parent p: the model should predict based on the prefix
  including parent p. Correct source: `logits[P+p]` (the parent's position).

Using `logits[P+i]` for node i (as the pseudo-code implies) would verify root nodes
against `logits[P]` — which is the prediction AFTER node 0, not the prediction for
node 0. This is fundamentally wrong.

The implementations had to **detect and correct** the misleading pseudo-code.

---

## Per-Model Analysis

### GLM-5 — Grade: A (`glm5/dflash_verify/dflash.py`, ~356 lines)

Equivalent to GLM-5.1's more complete submission (path-based variant).

**Logits extraction (CORRECT):**
```python
logit_pos = (P - 1) if parent == -1 else (P + parent)
target_pred = int(np.argmax(logits[logit_pos]))
```
Each node is verified against its parent's logits. This is the only approach
that generalizes to arbitrary tree topologies.

**Acceptance strategy: Path-based**
```python
on_path = parent == path_end
if tree_tokens[i] == target_pred:
    if on_path:
        accepted.append(tree_tokens[i])
        path_end = i
else:
    if on_path:
        accepted.append(target_pred)
        return accepted  # stop cycle
```
Only one path through the tree is followed. Off-path matches are recorded but
don't affect output. At cycle end, a bonus token is emitted from the last
position on the path.

**Subtree invalidation test (ACTUALLY TESTS IT):**
```python
tree_tokens = [t0, t1, wrong_root, t1_given_wrong]
tree_parents = [-1, 0, 0, 2]  # root0→child0, wrong_root→child_of_wrong
```
Constructs a tree where `root0` (node 0, correct) has child `t1` (node 1, correct),
and `wrong_root` (node 2, WRONG) has child `t1_given_wrong` (node 3, WOULD match
if processed independently). Verifies that:
1. Node 2 is rejected ✓
2. Node 3 is in `skipped_by_ancestor` set ✓
3. Output matches autoregressive ✓

**Tests:** 5 (mask correctness, basic, subtree invalidation ×4 configs,
multi-step ×5 configs, golden ×60 configs)

**Strengths:**
- Only implementation with correctly parent-indexed logits extraction
- Actual subtree invalidation testing (exposes the branching bug)
- Path-based approach elegantly handles off-path matches
- Bonus token at cycle end (correct per DFlash spec)
- 60-config golden test
- Clean class-based architecture (LayerNorm, Linear, TransformerBlock)
- GELU activation (more realistic than ReLU)

**Weaknesses:**
- Path-based approach only extracts ONE path per cycle (spec says accept ALL
  matching nodes in topological order). In practice, this is a design choice
  — both approaches converge for greedy decoding.
- The test's `_make_draft_fn` helper uses oracle (autoregressive greedy) to
  generate draft tokens — not a real draft model mock

---

### GLM-5.1 — Grade: B+ (`glm5.1/dflash_verify/dflash_verify.py`, ~263 lines)

**Logits extraction (CORRECT):**
```python
if tree_parents[i] == -1:
    parent_logit_idx = prompt_len - 1
else:
    parent_logit_idx = prompt_len + tree_parents[i]
```
Parent-indexed, same core correctness as GLM-5.

**Acceptance strategy: All-matching + break**
```python
if tree_tokens[i] == target_greedy:
    accepted.append(tree_tokens[i])
else:
    accepted.append(target_greedy)
    rejected_ancestors.add(i)
    break
```
Simpler than GLM-5's path approach: accepts ALL matching nodes in order,
breaks on first rejection. This correctly handles the case where multiple
roots would all be accepted.

**Tests:** 3 (basic, subtree invalidation, multi-step)

**Strengths:**
- Correct parent-indexed logits extraction
- Simpler acceptance logic (easier to verify)
- Clean, concise implementation
- Model uses vocab_size=100 (smaller, faster)

**Weaknesses:**
- Subtree invalidation test uses a CHAIN (`tree_parents = [-1, 0, 1]`), not
  a branching tree. The test verifies that a rejected node's descendant is
  skipped, but doesn't test that different branches are handled correctly.
- Fewer tests than GLM-5 (3 vs 5+60)
- No bonus token at cycle end — the fallback to causal mask generation
  works but is slightly different from the DFlash spec

---

### Kimi K2.6 — Grade: B- (`kimi-k2.6/dflash_verify/tree_attention.py`, ~200 lines)

**Logits extraction (BROKEN for branching):**
```python
tree_logits = logits[prompt_len - 1:prompt_len + n_nodes - 1]
```
Maps node 0 → logits[P-1], node 1 → logits[P], node 2 → logits[P+1], etc.

This treats the tree as a **linear chain** where node i is always at depth i+1
and the parent of node i is always node i-1. It works for chains but fails for
any tree with branching.

For a tree `parents = [-1, -1, 0, 0]` (two roots, each with one child):
- Node 0 (root): logits[P-1] ✓
- Node 1 (root): logits[P] ✗ (should be logits[P-1] — it's also a root!)
- Node 2 (child of 0): logits[P+1] ✗ (should be logits[P+0] — parent's logits!)
- Node 3 (child of 0): logits[P+2] ✗ (should be logits[P+0])

**Why tests still pass:** The subtree invalidation test constructs a branching
tree but the first node checked is a WRONG root, so the algorithm breaks immediately
before processing any depth-2 nodes. The bug in node 1's logits extraction is
never exercised.

**Tests:** 3 (basic, subtree invalidation, multi-step)

**Strengths:**
- Most concise implementation (~200 lines)
- Clean, readable code
- Proper MinimalLM with multi-head attention
- Clear docstring

**Weaknesses:**
- Logits extraction is fundamentally wrong for branching trees
- Subtree invalidation test is broken: uses 5 nodes but the algorithm stops
  at node 0 (first rejection), so depth-2 nodes are never reached
  ```python
  tree_tokens = [wrong_root0, expected[1], expected[2], expected[3], expected[4]]
  tree_parents = [-1, -1, -1, 0, 1]
  ```
  Node 0 (wrong_root0) is rejected → algorithm breaks. Node 3 (child of
  wrong_root0) is never checked. The test only asserts `len(accepted) ==
  len(auto_tokens[:P+1])`, which happens to pass because the replacement token
  matches.
- No mask correctness test
- No golden test

---

### Qwen3-6 — Grade: B- (`qwen36/dflash_verify/dflash_verify.py`, ~470 lines)

**Logits extraction (BROKEN for branching):**
```python
depths = _compute_depths(tree_parents)
tree_logits = np.stack([logits[P + d - 2] for d in depths])
```
Uses **depth-based** extraction: depth 1 → logits[P-1], depth 2 → logits[P],
depth 3 → logits[P+1], etc.

This means ALL nodes at the same depth share the same logits source, regardless
of which parent they belong to. For branching trees:
- Nodes at depth 2 with different parents get the same logits[P]
- logits[P] only captures the prediction after node 0 — it's wrong for children
  of nodes 1, 2, etc.

For a tree `parents = [-1, -1, 0, 1]`:
- Node 0 (root, depth 1): logits[P-1] ✓
- Node 1 (root, depth 1): logits[P-1] ✓
- Node 2 (child of 0, depth 2): logits[P] ✓ (parent=0, position=P+0)
- Node 3 (child of 1, depth 2): logits[P] ✗ (should be logits[P+1], parent=1's logits!)

The depth-based approach **accidentally works** when all depth-2 nodes are
children of node 0 — which is the common case in simple test trees.

**Why tests still pass:** Same reason as Kimi — the controlled subtree
invalidation test breaks at the first wrong root before reaching depth-2
nodes from different parents.

**Tests:** 7 (mask correctness, logits consistency, basic, subtree
invalidation, multi-step, golden, correct-draft bonus)

**Strengths:**
- Most tests (7, including golden)
- Mask correctness test (bonus)
- Logits consistency test verifies AR vs tree logits match (bonus)
- Good controlled subtree invalidation with printed analysis
- Clean MinimalLM with einsum for efficient attention
- Accepts ALL matching nodes (not just path-based)

**Weaknesses:**
- Depth-based logits extraction is wrong for branching trees
- Subtree invalidation test has the same structural flaw as Kimi's —
  the test tree `[-1, -1, -1, 0, 0, 1, 1]` has nodes 0,1,2 as roots
  and nodes 3,4 as children of 0, nodes 5,6 as children of 1.
  Node 0 (999) is rejected → algorithm breaks. Nodes 5 and 6 (children
  of node 1, a DIFFERENT root) are never reached, so their incorrect
  logits extraction is never tested.
- Verify_and_accept doesn't have a fallback for empty accepted list
  (speculative_generate handles it but the function itself doesn't)

---

## The Logits Extraction Trap: Detailed Breakdown

The DFlash challenge prompt contains a deliberately misleading pseudo-code:

```python
# Prompt pseudo-code (WRONG for branching):
tree_logits = logits[len(generated_tokens):]  # logits[P:P+N]
accepted = accept_reject(tree_tokens, tree_parents, tree_logits, ...)
```

This would mean tree node i is verified against `logits[P+i]`. In a standard
transformer, `logits[j]` predicts token at position j+1, so `logits[P+i]`
predicts the token AFTER tree node i — not tree node i itself.

Three schools of thought emerged:

| Approach | Models | Correct? | Behavior |
|----------|--------|----------|----------|
| **Parent-indexed** | GLM-5, GLM-5.1 | ✓ | `logits[P-1]` for roots, `logits[P+parent]` for children |
| **Positional** | Kimi K2.6 | ✗ | `logits[P-1+i]` — assumes chain topology |
| **Depth-based** | Qwen3-6 | ✗ | `logits[P+depth-2]` — assumes all nodes at depth d share parent |

Parent-indexed is the only correct approach because:
1. It follows the tree topology exactly
2. It correctly handles multiple roots (all checked against prompt's last logits)
3. It correctly handles children of different parents

Positional and depth-based both fail for trees with branching at different
levels where siblings have different parents.

---

## Why Nobody Caught the Branching Logits Bug

Three factors:

1. **The test trees are adversarially insufficient.** All three models'
   subtree invalidation tests construct trees where the first node is
   WRONG and triggers immediate rejection. The branching logits error
   only manifests when MULTIPLE branches are processed — but the
   break-on-reject prevents reaching those branches.

2. **The prompt's pseudo-code misled models.** The pseudo-code showing
   `logits[len(generated_tokens):]` directly encouraged the positional
   and depth-based approaches. GLM-5 and GLM-5.1 were the only models
   to recognize this was wrong and use parent-indexed logits.

3. **Basic chain tests pass with either approach.** The basic test
   uses a chain (`parents = [-1, 0, 1]`), where positional, depth-based,
   and parent-indexed all produce identical results. So the bug lurks
   in branching cases that aren't tested.

**To catch the bug, you'd need a test like:**
```python
# Tree with two roots, both CORRECT, each with children
# root0: correct (matches target) → child0 verified against logits[P+0]
# root1: correct (matches target) → child1 verified against logits[P+1] (NOT logits[P]!)
tree_tokens = [t0, t1, child_of_0, child_of_1]
tree_parents = [-1, -1, 0, 1]
# Positional: child_of_1 gets logits[P+2] (should be logits[P+1]) → WRONG
# Depth-based: child_of_1 gets logits[P] (should be logits[P+1]) → WRONG
# Parent-indexed: child_of_1 gets logits[P+1] ✓
```

---

## Comparative Metrics

| Metric | GLM-5 | GLM-5.1 | Kimi K2.6 | Qwen3-6 |
|--------|-------|---------|-----------|---------|
| Lines of code | 356 | 263 | 200 | 470 |
| Logits extraction | Parent-indexed ✓ | Parent-indexed ✓ | Positional ✗ | Depth-based ✗ |
| Branching correctness | ✓ | ✓ | ✗ | ✗ |
| Subtree invalidation tests branching | ✓ | ✗ | ✗ | ✗ |
| Acceptance strategy | Path-based + bonus | All-matching + break | All-matching + break | All-matching + break |
| Tests | 5 (incl. 60 gold) | 3 | 3 | 7 |
| Model architecture | GELU, class-based | vocab=100, simpler | MHA, ReLU | einsum, MHA, ReLU |
| Golden test | ✓ 60 configs | ✗ | ✗ | ✓ 1 config |
| Mask test | ✓ | ✗ | ✗ | ✓ |
| Bonus: logits consistency | ✗ | ✗ | ✗ | ✓ |

---

## Detailed Subtree Invalidation Showdown

This is the single most important test — it distinguishes understanding from
pattern-matching.

| Aspect | GLM-5 | GLM-5.1 | Kimi K2.6 | Qwen3-6 |
|--------|-------|---------|-----------|---------|
| Tree shape | Branching (2 roots) | Chain | Branching but only 1 active root | Branching, 3 roots |
| Wrong node depth | depth 1 (root1) | depth 2 (child of root) | depth 1 (root0) | depth 1 (root0) |
| Child would match? | ✓ verified | ✓ verified | Not verified (break hides) | Not verified (break hides) |
| Skip detection | `3 in skipped` ✓ | `assert len(accepted)==2` | Only checks len match | Only checks len match |
| Tests branching correctness | YES | No (chain) | No (breaks before) | No (breaks before) |

Only GLM-5's test actually exercises the scenario where a WRONG branch is
rejected and its children are skipped BY ANCESTOR INVALIDATION (not by
break-after-rejection). GLM-5 uses a tree with TWO roots: one correct
(path continues), one wrong (rejected, its child skipped via ancestor check).

---

## Rankings

| Rank | Model | Rationale |
|------|-------|-----------|
| **1** | **GLM-5** | Only correct logits extraction. Only test that actually exercises branching subtree invalidation. Path-based acceptance with bonus token. 60-config golden test. Clean class architecture. |
| **2** | **GLM-5.1** | Correct logits extraction. Simpler acceptance (all-matching). But tests are chain-only, missing the branching edge case coverage. |
| **3** | **Qwen3-6** | Most tests, best bonus coverage. But depth-based logits extraction is wrong for branching trees. Tests' break-on-reject hides the bug. |
| **4** | **Kimi K2.6** | Most concise. But positional logits extraction is wrong for branching trees. Tests' break-on-reject hides the bug. Fewest tests. |
| — | **MiniMax-M2.7** | No submission (only PROMPT.md, no .py file produced). |

---

## Key Takeaways

1. **The prompt's pseudo-code was a trap.** The `logits[len(generated_tokens):]` line
   is wrong for branching trees but looks natural. Only GLM-5 and GLM-5.1 recognized
   it needed correction to parent-indexed logits.

2. **Testing is the differentiator, not code.** All four implementations run and pass
   the basic tests. The gap is in whether the tests would catch branching-tree bugs.
   GLM-5's subtree invalidation test is the only one that exercises two branches
   simultaneously and verifies ancestor-based skipping.

3. **Kimi K2.6 and Qwen3-6 share the same logits bug but manifest it differently.**
   Kimi uses sequential indexing (`logits[P-1+i]`), Qwen uses depth-based indexing
   (`logits[P+depth-2]`). Both are correct for chains, wrong for branches.

4. **The "bonus token" distinction matters.** GLM-5 emits a bonus token at cycle end
   from the last position on the accepted path — this matches the DFlash spec.
   GLM-5.1 and Kimi rely on the generation loop's fallback. Qwen's approach is
   more nuanced (uses depth-based positions for all nodes).

5. **This is the first challenge where GLM-5 decisively beats Qwen3-6.** In all
   previous rounds (backwards, fuse, KV-cache, flash attention, beam search),
   Qwen3-6 either won or tied. In DFlash, GLM-5 is the only model that both
   understands the logits extraction AND tests it properly.

## Final DFlash Ranking

1. **GLM-5** — Grade: A — Only model with fully correct branching support + proper testing
2. **GLM-5.1** — Grade: B+ — Correct algorithm, insufficient testing
3. **Qwen3-6** — Grade: B- — Most code/thoroughness, but depth-based logits is wrong
4. **Kimi K2.6** — Grade: B- — Cleanest code, but positional logits is wrong
5. **MiniMax-M2.7** — No submission
