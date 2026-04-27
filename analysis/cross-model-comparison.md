# Comprehensive Cross-Model Comparison: All Challenges

This synthesizes results across all 8 challenges: Layer Norm Backward, Fused Softmax+Top-K,
KV-Cache, Flash Attention Forward, Beam Search, Flash Attention Backward, DFlash,
and Ternary Training.

Models that participated in ≥2 rounds: **GLM-5** (8 challenges, all), **Qwen3-6** (7),
**Claude Opus 4.7** (7, absent ternary), **Kimi K2.6** (3), **GLM-5.1** (3),
**MiniMax-M2.7** (4, absent from last 4 rounds).

---

## Per-Challenge Grade Matrix

| Challenge | Difficulty | GLM-5 | Qwen3-6 | **Opus 4.7** | Kimi K2.6 | GLM-5.1 | MiniMax |
|-----------|------------|-------|---------|-------------|-----------|---------|---------|
| Layer Norm Backward | Medium | B+ (2nd) | **A-** (1st) | **A (1st)** | — | — | B (3rd) |
| Fused Softmax+TopK | Medium | **A- (tie)** | **A (tie)** | **A (tie)** | — | — | B (4th) |
| KV-Cache | Medium | A- (2nd) | **A (1st)** | **B+** (3rd) | — | — | B- (4th) |
| Flash Attn Forward | Hard | A- (2nd) | **A (1st)** | **A-** (tie 2nd) | — | — | B (4th) |
| Beam Search | Hard | B+ (3rd) | **A (1st)** | **A (1st)** | — | — | B- (4th) |
| Flash Attn Backward | Extra Hard | A- (2nd) | A- (3rd) | **A-** (tie 2nd) | **A** (1st) | B+ (4th) | — |
| DFlash | Extra Hard | **A (1st)** | B- (4th=) | **A (1st)** | B- (4th=) | B+ (3rd) | — |
| Ternary Training | SOTA Research | **A-** (1st) | B+ (2nd) | **B+** (3rd) | C (5th) | C+ (4th) | — |

**Note:** The real Ternary Bonsai 8B HF model card confirms "ternary coverage: Embeddings, attention projections, MLP projections, LM head" — embeddings ARE ternary in the shipping model. Opus 4.7's decision to leave them non-ternary, while well-argued, is a deviation from both the spec and the actual shipped model. GLM-5 alone matches the real Bonsai's ternary coverage.

---

## Head-to-Head Breakdown

### GLM-5 vs Qwen3-6 (7 overlapping challenges)

| Challenge | Winner | Margin |
|-----------|--------|--------|
| Layer Norm Backward | **Qwen** | Decisive (edge cases, cross-verification, broader GPU fusion scope) |
| Fused Softmax+TopK | Split | Qwen wins on production (float4, K≤256), GLM wins on algorithm (true single-pass online softmax) |
| KV-Cache | **Qwen** | Decisive (8 modular files, 10 demos, GQA/MQA, GPU mapping, 6 real model specs) |
| Flash Attn Forward | **Qwen** | Narrow (5 tests vs 2, batched einsum, uneven tiles tested) |
| Beam Search | **Qwen** | Decisive (Beam class, 4 tests, length penalty×EOS interaction tested) |
| Flash Attn Backward | **GLM** | Narrow (D-optimization single-pass vs Qwen's two-pass, higher precision) |
| DFlash | **GLM** | Decisive (only model with correct branching logits + proper subtree invalidation test) |
| Ternary Training | **GLM** | Moderate (clean rerun PPL=594 in 250 steps; Qwen's honest PPL=319 is better generalization but original was inflated) |

**Record: Qwen3-6 leads 4-3-1**

The arc is consistent: Qwen3-6 dominates the early-round "engineering breadth" challenges
(backwards, KV-cache, beam search), but GLM-5 wins all three late-round challenges
(flash attn bwd, DFlash, ternary). This pattern is too strong to be coincidence:
- **GLM-5 is stronger on deeper algorithmic reasoning** (backward passes, tree attention, unconventional training)
- **Qwen3-6 is stronger on engineering breadth and production-minded implementation**
- As challenges get harder and more open-ended, GLM-5's correctness-first approach wins

### GLM-5 vs Kimi K2.6 (3 overlapping challenges)

| Challenge | Winner | Margin |
|-----------|--------|--------|
| Flash Attn Backward | **Kimi** | Narrow (cleaner code, 3.4e-09 precision vs 1.2e-07, but GLM's D-optimization is more efficient) |
| DFlash | **GLM** | Decisive (correct branching logits vs Kimi's chain-only positional extraction) |
| Ternary Training | **GLM** | Decisive (honest PPL=594 vs 5,501; Kimi's embeddings not ternary, catastrophic overfit) |

**Record: GLM-5 leads 2-1**

Kimi K2.6 has beaten GLM-5 on exactly one challenge: flash attn bwd (best precision, cleanest code).
But in both DFlash and Ternary, fundamental correctness issues (broken logits for branching, non-ternary
embeddings, overfitting) separate them substantially.

### GLM-5 vs GLM-5.1 (3 overlapping challenges)

| Challenge | Winner | Margin |
|-----------|--------|--------|
| Flash Attn Backward | **GLM-5** | Narrow (lower errors: 1.2e-07 vs 8.5e-06, GLM-5.1's `break` optimization is fragile) |
| DFlash | **GLM-5** | Decisive (proper branching subtree invalidation test; GLM-5.1 only tests chain) |
| Ternary Training | **GLM-5** | Decisive (honest PPL=594 vs 30,731; GLM-5.1 catastrophically overfit at 1500 steps) |

**Record: GLM-5 leads 3-0**

GLM-5.1 is consistently a regression from GLM-5 — same architectural instincts (parent-indexed
DFlash logits, ternary embeddings) but worse execution: higher numerical errors, insufficient
testing, catastrophic overfitting due to poor hyperparameter choices. GLM-5.1 writes code that
looks right but falls apart under stress (small data, many steps).

### Qwen3-6 vs Kimi K2.6 (3 overlapping challenges)

| Challenge | Winner | Margin |
|-----------|--------|--------|
| Flash Attn Backward | **Kimi** | Narrow (better precision, cleaner code; Qwen has lower memory and more tests) |
| DFlash | **Qwen** | Narrow (both B-, but Qwen has 7 tests vs 3, golden test, logits consistency bonus) |
| Ternary Training | **Qwen** | Decisive (honest PPL=319 vs 5,501; Kimi's embeddings not ternary, catastrophic overfit) |

**Record: Qwen3-6 leads 2-1**

Qwen3-6 and Kimi share the same DFlash logits bug (depth-based vs positional), keeping them
in B- range. Kimi's flash attn bwd precision is genuinely better. But Qwen3-6's ternary
implementation is fundamentally more correct — all layers ternary, PPL=319 vs 5,501.

### Qwen3-6 vs GLM-5.1 (2 overlapping challenges: DFlash, Ternary)

| Challenge | Winner | Margin |
|-----------|--------|--------|
| DFlash | **GLM-5.1** | Algorithmic correctness (parent-indexed logits) vs Qwen's broken depth-based approach |
| Ternary Training | **Qwen** | Decisive (honest PPL=319 vs 30,731; GLM-5.1 catastrophically overfit) |

**Record: Tied 1-1**

A fascinating split. GLM-5.1 gets DFlash right where Qwen3-6 gets it wrong — the parent-indexed
logits insight that only GLM lineage models caught. But in ternary training, the roles reverse:
Qwen3-6 generalizes reasonably (PPL=319) while GLM-5.1 completely collapses. Qwen3-6's
engineering discipline (train/val separation, moderate step counts) is the difference.

### Kimi K2.6 vs GLM-5.1 (3 overlapping challenges)

| Challenge | Winner | Margin |
|-----------|--------|--------|
| Flash Attn Backward | **Kimi** | Decisive (3.4e-09 vs 8.5e-06, cleaner code) |
| DFlash | **GLM-5.1** | Decisive (correct parent-indexed logits vs Kimi's broken positional extraction) |
| Ternary Training | **Kimi** | Narrow (both overfit catastrophically: PPL=5,501 vs 30,731; Kimi's is less bad) |

**Record: Kimi leads 2-1**

---

### Opus 4.7 Head-to-Head (7 overlapping challenges)

### Opus 4.7 vs GLM-5 (8 challenges — all)

| Challenge | Winner | Margin |
|-----------|--------|--------|
| Layer Norm Backward | **Opus** | Narrow (A vs B+; better docs, GPU fusion sketch, 5 edge cases, optimal 3-cache) |
| Fused Softmax+TopK | Tie | Both A: single-pass online softmax, template-based K, excellent design docs |
| KV-Cache | **GLM-5** | Moderate (both correct; GLM-5 has actual tests, Opus uses Python lists) |
| Flash Attn Forward | Tie | Both A-; Opus has memory test and causal row-0 check; GLM-5 uses MLX |
| Beam Search | **Opus** | Moderate (A vs B+; both correct, Opus has mock model EOS test, 2K expansion) |
| Flash Attn Backward | Tie | Both A-; Opus has better tests, GLM-5 has batched D-optimization |
| DFlash | Tie | Both A; only two models with correct parent-indexed logits |
| Ternary Training | **GLM-5** | Narrow (PPL=594 vs 643 at 250 steps; GLM has ternary embeddings matching real Bonsai's shipped coverage; Opus left embeddings non-ternary with a well-argued but factually-incorrect justification) |

**Record: Opus 4.7 leads 2-2-4**

GLM-5's embedding decision was correct per both spec and the real model. This is the
one place where Opus 4.7's "read the literature" approach backfired — BitNet b1.58
the paper may keep embeddings higher precision, but PrismML's shipped Ternary Bonsai
doesn't. Following the spec was right.

### Opus 4.7 vs Qwen3-6

| Challenge | Winner | Margin |
|-----------|--------|--------|
| Layer Norm Backward | **Opus** | Narrow (A vs A-; more edge cases, GPU fusion, optimal cache) |
| Fused Softmax+TopK | **Opus** | Narrow (single-pass online vs Qwen's 2-pass; both have template K) |
| KV-Cache | **Qwen** | Decisive (8 modular files + 10 demos vs Python lists + good analysis) |
| Flash Attn Forward | **Qwen** | Narrow (batched einsum vs unbatched per-(b,h) loops) |
| Beam Search | Tie | Both A; correct EOS retention, mock model tests |
| Flash Attn Backward | **Opus** | Narrow (better tests: dV ALL elements; Qwen has lower memory) |
| DFlash | **Opus** | Decisive (correct parent-indexed logits vs Qwen's broken depth-based approach) |

**Record: Opus 4.7 leads 4-1-2**

Opus beats Qwen3-6 on algorithmic depth (DFlash, fuse, backwards) while Qwen wins on engineering
breadth (KV-cache modularity, batched flash attention). This mirrors the GLM-5 vs Qwen3-6 dynamic:
algorithmic correctness wins the hard challenges; production engineering wins the broad ones.

### Opus 4.7 vs Kimi K2.6 (2 overlapping challenges)

| Challenge | Winner | Margin |
|-----------|--------|--------|
| Flash Attn Backward | **Kimi** | Narrow (3.4e-09 precision vs Opus's unbatched but well-tested approach) |
| DFlash | **Opus** | Decisive (correct parent-indexed logits vs Kimi's broken positional extraction) |

**Record: Tied 1-1**

### Opus 4.7 vs GLM-5.1 (2 overlapping challenges)

| Challenge | Winner | Margin |
|-----------|--------|--------|
| Flash Attn Backward | **Opus** | Moderate (better dV test coverage, no fragile `break` optimization) |
| DFlash | Tie | Both A/B+: parent-indexed logits correct; GLM-5.1 has insufficient tests |

**Record: Opus 4.7 leads 1-0-1**

---

## The Shape of Each Model

### GLM-5 — The Algorithmic Thinker

*Strengths:* Algorithmic elegance (single-pass online softmax, D-optimization in
backward, parent-indexed DFlash logits, `@mx.custom_function` STE with explicit VJP),
code clarity, correctness-first, concise implementations. **The only model that
participated in all 8 challenges** and the only model that never declined in grade.

*Weakness:* Limited scope (single-file, fewer tests, no GQA/MQA, K≤32 in fuse kernel).

*Signature pattern:* True single-pass fused softmax. Parent-indexed DFlash logits.
Gradient clipping at norm=1.0 in ternary. Values "why" over "how much."

*Best showing:* DFlash (only model with fully correct branching tree verification).
Ternary (only model robust across two completely different datasets).

### Claude Opus 4.7 — The Deep Generalist

*Strengths:* Algorithmic depth across every domain: correct parent-indexed DFlash
logits, single-pass online softmax in fuse, optimal 3-item cache in backwards,
correct EOS retention in beam search, proper causal skip in flash attention.
The only model besides GLM-5 to catch the DFlash logits trap. In ternary: PPL=643
at 250 steps (tied with GLM-5), best FINDINGS.md of any model (weight_decay=0.0
rationale, paragraph-based val split, BitNet embedding convention, warmup analysis).
Remarkably consistent — scores range from A to B+ across all challenges. Participated in all
8 challenges.

*Weakness:* Uses raw Python lists for KV-cache (toy-grade implementation despite
excellent ANALYSIS.md). Unbatched per-(b,h) loops in flash attention. Leaving
embeddings non-ternary is well-justified (BitNet b1.58 convention, gathers not
matmuls, confirmed against real GGUFs) but deviates from the prompt's spec.

*Signature pattern:* Desktop-quality CUDA kernel design with single-pass streaming.
Parent-indexed DFlash logits. Paragraph-based train/val split that no other model
thought to do. Weight_decay=0.0 with a paragraph explaining *why* (ternary threshold
crossing). Reads the research literature, not just the prompt.

*Best showing:* DFlash and Fuse (A grades). Ternary FINDINGS.md (best write-up).
Backwards (A grade, best numerical stability discussion).

### Qwen3-6 — The Full-Stack Engineer

*Strengths:* Modular code (3-8 files per task), comprehensive tests (always exceeds
minimum), edge cases handled, real model specs, GPU mapping, attention variants
(GQA/MQA), cross-verification of formulas.

*Weakness:* Breadth over depth. Falls behind on deep algorithmic reasoning (DFlash
depth-based logits bug, flash attn bwd two-pass). Data leakage in original ternary
run inflated results.

*Signature pattern:* Writes `Beam` class with `__slots__`, separates model.py from
algorithm.py, tests `N=300, tile_size=97` to catch uneven tile bugs.

*Best showing:* KV-Cache (8 files, 10 demos, 6 real model specs, GQA/MQA).

### Kimi K2.6 — The Precision Instrument

*Strengths:* Best numerical precision (3.4e-09 dV error), cleanest code.

*Weakness:* Only 3 challenges. Pattern-matches prompts rather than understanding
(DFlash positional logits, ternary embeddings not ternary).

*Best showing:* Flash Attention Backward (A grade, best precision).

### GLM-5.1 — The Weaker Sibling

*Strengths:* Correct parent-indexed DFlash logits. Best debugging docs (MLX `__dict__` trap).

*Weakness:* Regression from GLM-5 in every dimension. Catastrophic ternary overfitting (PPL=30K).

*Best showing:* DFlash (correct algorithm, insufficient tests). The `break` vs `continue` in flash attn's causal loop, the
`assert n % group_size == 0` in ternary, the 1500 steps on 48K tokens — all
examples of getting the big picture but missing the detail that makes it work.

---

## The Trajectory: Who Got Better (and Where) Over Time?

Over 5 rounds of increasingly difficult challenges:

```
        R1     R2     R3    DFlash  Ternary
GLM-5:  B+  →  B+  →  A-  →  A   →  A-    (rising, then sustained)
Opus:   A   →  A-  →  A-  →  A   →  B+    (strong early, slight late dip)
Qwen:   A-  →  A   →  A-  →  B-  →  B+    (early peak, late dip, partial recovery)
Kimi:   —   →  —   →  A   →  B-  →  C     (flash of brilliance, steep decline)
GLM5.1: —   →  —   →  B+  →  B+  →  C+    (flat then fell off)
MiniM:  B   →  B-  →  —   →  —   →  —     (exited)
```

**Opus 4.7 and GLM-5 are mirror images.** GLM-5 rises as challenges get harder;
Opus maintains a flat A-tier from the start. Both end at A- in ternary. Both are
the only models with A grades in DFlash. The difference is trajectory shape: GLM-5
grows into hard problems, Opus comes pre-loaded to handle them.

**GLM-5 is the only model that doesn't decline.** It rises through R3/DFlash and holds at A-
into ternary. This is the trajectory you want: consistent competence that improves or
maintains as challenges get harder and more open-ended.

**Qwen3-6 has the steepest decline.** From A/A- in early rounds to B- in DFlash, then
partial recovery to B+ in ternary after the data leakage was corrected. The pattern
suggests ceiling effects — Qwen3-6's breadth-first approach excels when the challenge
rewards systematic testing (early rounds), but struggles when the challenge requires
deep algorithmic insight (DFlash) or hyperparameter discipline (ternary overfitting).

**Kimi and GLM-5.1 have a "one-hit wonder" pattern.** Kimi's flash attn bwd (A) is genuinely
excellent but surrounded by B- and C performances. GLM-5.1's parent-indexed DFlash logits
are correct but everything else is regression from GLM-5. Neither is reliable across
diverse challenges.

---

## Aggregate Ranking

| Rank | Model | Avg Grade | R1 (3) | R2 (2) | R3 (1) | DFlash | Ternary | Strength | Weakness |
|------|-------|-----------|--------|--------|--------|--------|---------|----------|----------|
| **1** | **GLM-5** | **A-/B+** | B+ (2nd=) | B+ (2nd) | A- (2nd) | **A** (1st=) | **A-** (1st) | Algorithmic reasoning, never declines, ternary coverage matches real Bonsai, gradient clipping | Scope (fewer tests, single-file) |
| **2** | **Opus 4.7** | **A-/B+** | **A** (1st) | **A-** (1st=) | **A-** (1st) | **A** (1st=) | **B+** (3rd) | Algorithmic depth, best docs (FINDINGS.md, ANALYSIS.md), highest floor, paragraph-based val split | Non-ternary embeddings (deviates from spec + real Bonsai), Python lists in KV |
| **3** | **Qwen3-6** | **B+** | A- (2nd=) | A (1st) | A- (3rd) | B- (4th=) | B+ (2nd) | Engineering breadth, modularity, generalization | DFlash logits trap, data leakage, algorithmic depth ceiling |
| **4** | **Kimi K2.6** | **B** | — | — | **A** (1st) | B- (4th=) | C (4th) | Numeric precision | Only 3 challenges; DFlash/ternary bugs |
| **5** | **GLM-5.1** | **B-/C+** | — | — | B+ (4th) | B+ (3rd) | C+ (3rd) | Correct DFlash algorithm | Regression from GLM-5; catastrophic ternary overfitting |
| **6** | **MiniMax-M2.7** | **B/B-** | B (3rd) | B- (3rd) | — | — | — | Ambition | Bugs, no tests, exited after 4 rounds |

**GLM-5 takes sole #1.** The real Ternary Bonsai model card confirms that embeddings ARE
ternary in the shipped model — matching GLM-5's implementation and the prompt spec. Opus 4.7's
non-ternary embedding decision, while the most carefully argued deviation in the field, is both
a spec deviation and factually mismatched with what PrismML actually ships. The PPL difference
(GLM-5: 594 vs Opus: 643 at 250 steps) is small, but the correctness of the architectural
choice is now unambiguous.

---

## The Two Most Differentiating Moments

### 1. DFlash logits extraction (Algorithmic Insight Test)

The prompt's pseudo-code says `tree_logits = logits[len(generated_tokens):]` which
is wrong for branching trees. Only three models corrected this to parent-indexed
logits: **GLM-5**, **GLM-5.1**, and **Opus 4.7**. Qwen3-6 and Kimi K2.6 — widely
considered top-tier coding models — both took the pseudo-code at face value.

In this test:
- GLM-5: Understands ✓ (proper branching subtree invalidation test)
- Opus 4.7: Understands ✓ (chain-following acceptance, proper subtree invalidation)
- GLM-5.1: Understands ✓ (but chain-only tests)
- Qwen3-6: Pattern-matches ✗ (depth-based logits)
- Kimi K2.6: Pattern-matches ✗ (positional logits, chain-only)

### 2. Ternary clean-data rerun (Engineering Discipline Test)

When models were given identical `train_data.txt` (48K tokens):
- GLM-5: PPL=594 in 250 steps — moderate overfitting, still learning
- Qwen3-6: PPL=319 in 300 steps — best generalization, disclosed data leakage
- Kimi K2.6: PPL=5,501 with train loss 0.016 — memorized completely
- GLM-5.1: PPL=30,731 with train loss 0.18 — catastrophic overfitting

### 3. Single-pass vs. two-pass fuse (Algorithmic Taste Test)

Only GLM-5 and Opus 4.7 implemented true single-pass online softmax for fuse.
Qwen3-6, Kimi, and MiniMax all use two-pass. This distinction doesn't affect
correctness (all pass), but it reveals models that think about "can I collapse this
into one stream?" vs "what's the standard recipe?" — a proxy for algorithmic taste.

---

## If You Could Only Pick One Model

| Criterion | Pick | Why |
|-----------|------|-----|
| Build a production system | **Qwen3-6** | Modular, tested, handles edge cases, thinks about GPU limits and real model specs |
| Solve a hard algorithmic problem | **GLM-5 / Opus 4.7** | Both caught DFlash logits trap, both do single-pass online softmax |
| Write numerically perfect code | **Kimi K2.6** | Best precision, cleanest code structure (for challenges that play to its strengths) |
| Get a correct answer quickly | **GLM-5** | Concisest implementations, fewest lines, correct-first philosophy |
| Most reliable across diverse tasks | **Opus 4.7** | Narrowest grade range (B+ to A across 7 challenges), most consistent performer |
| Most improved / highest ceiling | **GLM-5** | Only model that IMPROVES as challenges get harder |
| Best documentation & transparency | **Opus 4.7 / GLM-5** | Both write excellent design docs with bandwidth analysis, GPU mapping, failure modes |
| Most complete participation | **GLM-5** | Only model in all 8 challenges (including ternary) |

### The Definitive Answer

**GLM-5 is the clear #1 across all 8 challenges.** It participated in every challenge,
never declined in grade, won the hardest challenges (DFlash, ternary), and — uniquely — is the
only model whose ternary implementation matches what PrismML actually ships (embeddings
included). It also has the key hyperparameter insight (gradient clipping at norm=1.0) that
no other model documented.

**Opus 4.7 is a strong #2** — highest floor of any model (B+ to A range), best documentation,
and the only other model to catch the DFlash logits trap and implement single-pass fuse. Its
one meaningful miss was leaving embeddings non-ternary with a justification that turned out to
be factually incorrect.

**Qwen3-6 is the best production engineer** but has a clear algorithmic ceiling. Great for
well-specified problems; verify for correctness on deep reasoning tasks.

**Kimi K2.6 is the best numerical programmer** but only for the narrow class of problems it's
good at. Beautiful code, wrong algorithms on harder challenges.
