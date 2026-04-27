#!/usr/bin/env bash
set -euo pipefail

# deploy_challenges.sh — scaffold a new model directory with all challenge prompts
# Usage: ./deploy_challenges.sh -n model_name

usage() {
    echo "Usage: $0 -n <model_name>"
    exit 1
}

MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name) MODEL="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -z "$MODEL" ]] && usage

BASE="$(cd "$(dirname "$0")" && pwd)"
DEST="$BASE/$MODEL"

if [[ -d "$DEST" ]]; then
    echo "ERROR: '$DEST' already exists. Remove it or pick a different name."
    exit 1
fi

# ── Challenge definitions ─────────────────────────────────────────
# Format: subfolder_name|difficulty_label

declare -a CHALLENGES=(
    "backwards|MEDIUM"
    "fuse|MEDIUM"
    "kv|MEDIUM"
    "beam_search|HARD"
    "flash_attention|HARD"
    "dflash_verify|EXTRA HARD"
    "flash_attention_bwd|EXTRA HARD"
    "ternary_training|OPEN-ENDED RESEARCH"
)

# ── Helpers ───────────────────────────────────────────────────────


write_train_data() {
    local folder="$1"
    if [[ "$folder" == "ternary_training" ]]; then
        cp "$BASE/train_data.txt" "$DEST/$folder/train_data.txt"
        echo "    [data] train_data.txt"
    fi
}
write_prompt() {
    local folder="$1"
    local path="$DEST/$folder/PROMPT.md"
    case "$folder" in
        backwards)
            cat > "$path" << 'EOF'
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
EOF
            ;;
        fuse)
            cat > "$path" << 'EOF'
Design and implement a high-performance fused softmax + top-k kernel in CUDA (or CUDA-like pseudocode).

Requirements:
- Input: logits [B, T, V]
- Output:
    - top-k indices per (B, T)
    - top-k probabilities (after softmax)

Constraints:
1. Do NOT materialize the full softmax matrix in global memory.
2. Must be numerically stable (log-sum-exp).
3. Minimize global memory reads/writes.
4. Use shared memory where appropriate.
5. Handle large V (e.g., 50k+) efficiently.

Deliver:
- Kernel pseudocode or CUDA code
- Memory access pattern explanation
- Warp-level optimization strategy
- Complexity analysis (bandwidth vs compute bound)
- Comparison to naive implementation
EOF
            ;;
        kv)
            cat > "$path" << 'EOF'
Implement an efficient KV-cache system for autoregressive transformer inference from scratch.

Requirements:
1. Support incremental decoding (one token at a time).
2. Avoid recomputing attention for past tokens.
3. Handle:
   - multi-head attention
   - batching with variable sequence lengths
4. Provide:
   - data structure layout (memory format)
   - update logic per step
   - attention computation using cached keys/values

Additionally:
- Analyze memory growth over long sequences.
- Propose at least two optimizations (e.g., paged attention, chunking, compression).
- Explain how this would map to GPU execution.

Do not use any frameworks.
EOF
            ;;
        beam_search)
            cat > "$path" << 'EOF'
Implement a correct batched beam search decoder for autoregressive
generation in pure NumPy.

Simulate a minimal language model:
- vocab_size = 1000
- d_model = 64
- Use random embeddings + 1 transformer block with random weights
  (correctness depends on beam search logic, not model quality)

Requirements:

1. MULTI-BATCH SUPPORT:
   - Accept prompt_token_ids: list[list[int]] — one prompt per batch item
   - beam_width K per batch item
   - Each batch item's beams are INDEPENDENT (no cross-contamination between
     different prompts)

2. PER-STEP BEAM EXPANSION:
   - For each UNFINISHED beam, compute logits for the next token
   - Take top-(2*K) candidates per beam (not just top-K, to preserve diversity)
   - Compute total logprob = accumulated_logprob + new_logprob
   - Pool all candidates across all beams, sort globally by total logprob
     (most negative = worst), take top K
   - These K become the active beams for the next step

3. LENGTH PENALTY (for ranking only, not for accumulated score):
   - adjusted_score = accumulated_logprob / (generated_length ^ alpha)
   - alpha is a hyperparameter (default 0.6)
   - The accumulated logprob is NEVER modified by length penalty — it stays
     as the raw sum of logprobs
   - Length penalty is used ONLY when comparing beams for ranking/selection
   - generated_length = number of NEW tokens generated (NOT including prompt
     tokens — the prompt does not count toward length penalty)

4. EOS HANDLING (the critical part — get this right):
   - When a beam produces token_id == eos_token:
     * Mark that beam as FINISHED
     * Freeze its accumulated_logprob and generated_length
     * The beam STAYS in the pool — it competes with unfinished beams
   - At each step, the top-K selection pool includes BOTH:
     (a) all FINISHED beams, and (b) all candidates from expanding UNFINISHED beams
   - If all K beams in a batch item are finished, that item stops expanding
   - If all batch items have K finished beams, stop early
   - IMPORTANT: Do NOT remove finished beams from the pool. They must compete
     against unfinished beams using their length-penalized scores. If you
     remove them, a short, high-confidence sequence that hit EOS early will
     be wrongly discarded in favor of a longer, lower-confidence sequence.

5. RETURN:
   - For each batch item: a list of K sequences (generated token IDs only,
     NOT including prompt tokens), sorted by length-penalized score
     descending (best/highest score first)
   - Each sequence's score = accumulated_logprob / (len(seq) ^ alpha)

6. EDGE CASES:
   - If max_new_tokens is reached before K beams finish, return the best K
     (finished + unfinished) by length-penalized score
   - A batch item may end with fewer than K finished beams (if max_new_tokens
     hit). Return whatever is available.
   - Log-space accumulation: keep everything in log space; avoid unnecessary
     exp/log conversions. Don't let very negative numbers cause underflow.

Deliver:
- A class or function `batched_beam_search(prompts, beam_width, max_new_tokens,
  alpha, eos_token_id)` that returns the K best sequences per batch item
- Test 1: Single batch item, K=1, short prompt, alpha=0
  → verify this behaves identically to greedy decoding (always pick argmax)
- Test 2: batch=2, beam_width=3, different prompt lengths [3, 5], alpha=0.6
  → verify per-batch independence: beams from prompt 0 never interact with
    beams from prompt 1
- Test 3: THE EOS RETENTION TEST. Monkey-patch or modify the model's forward
  pass so that at step 1, one beam produces EOS with total logprob=-3.0
  while another beam continues with logprob=-4.0. At step 2, the continuing
  beam has logprob=-5.0. Verify that the EOS beam with score=-3.0 is
  correctly returned as the winner (even though it stopped early). If you
  had removed EOS beams from the pool, the unfinished beam with score=-5.0
  would wrongly win. This test distinguishes correct from buggy
  implementations.
- Comments explaining why finished beams must NOT be removed from the pool

Use only NumPy. No PyTorch, JAX, TensorFlow, or autograd.
EOF
            ;;
        flash_attention)
            cat > "$path" << 'EOF'
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
EOF
            ;;
        dflash_verify)
            cat > "$path" << 'EOF'
Implement the TREE ATTENTION VERIFICATION and ACCEPTANCE/REJECTION
algorithm for DFlash-style speculative decoding, in pure NumPy.

BACKGROUND:
Speculative decoding uses a fast draft model to propose candidate tokens,
then the target model verifies them in parallel. Standard speculative
decoding uses a linear chain of candidates. DFlash uses a TREE of
candidates — each candidate token can have multiple children, forming
a tree of possible futures. The target model verifies all tree nodes
in one forward pass using a tree-structured attention mask.

SETUP:
You are given:
  - A minimal target model (you write it: 1 transformer layer, ~64 dim,
    1000 vocab, random weights). It's small but structurally correct.
  - A draft model mock that produces a FIXED tree of tokens per step.
    You don't need to implement a real draft model — just pass in the
    tree tokens and structure as test input.

REQUIREMENTS:

1. TREE DATA STRUCTURE:
   A tree step is defined by:
     - tree_tokens: list[int] of length N — token IDs at each tree node
     - tree_parents: list[int] of length N — parent index for each node
       (-1 for root nodes, which are children of the last prompt token)
     - tree_children: list[list[int]] — child indices for each node
  
   Nodes are indexed 0..N-1 in topological order (a parent always
   appears before its children). Root nodes are at depth 1 (their
   logical "parent" is the last prompt token).

2. TREE ATTENTION MASK CONSTRUCTION:
   Given P prompt tokens and N tree nodes, the full sequence for the
   verification pass is [prompt_0, ..., prompt_{P-1}, tree_0, ..., tree_{N-1}].
   Length = P + N.

   Build a boolean attention mask M of shape (P+N, P+N) where M[i, j] = True
   means position i CAN attend to position j:

   RULES:
     a) Prompt tokens attend causally to each other: for 0 <= i < P,
        0 <= j < P: M[i, j] = (j <= i)
     b) ALL tree nodes attend to ALL prompt tokens: for P <= i < P+N,
        0 <= j < P: M[i, j] = True
     c) Each tree node attends to ITSELF: M[i, i] = True for all i
     d) A tree node attends to its ANCESTORS in the tree (transitively):
        if node k is an ancestor of node i, then M[i, j] = True where
        j = P + k (the global position of ancestor node k)
        Find ancestors by following parent pointers to root.
     e) A tree node does NOT attend to siblings, cousins, or the
        descendants of other branches

   Masked-out positions get score = -inf before softmax.
   The mask is converted to additive form: mask_add[i, j] = 0 if allowed,
   -inf if disallowed.

3. VERIFICATION FORWARD PASS:
   - Concatenate prompt embeddings + tree node embeddings into a single
     tensor of shape (P+N, d_model)
   - Run ONE forward pass through the target model's transformer block
     with the tree attention mask applied
   - The model returns logits for each position in the concatenated sequence
   - We only care about logits at tree node positions (indices P..P+N-1)

4. ACCEPTANCE/REJECTION SAMPLING:
   For each tree node i in topological order (0..N-1):
   
   a) If ANY ancestor of node i was REJECTED in a previous step:
        → SKIP this node and mark it as REJECTED (subtree invalidation)
        → Continue to next node
   
   b) Get the target model's logits at position P+i
      Convert to log-probabilities via log_softmax
      The target's greedy prediction = argmax(log_probs)
   
   c) The draft model proposed token = tree_tokens[i]
   
   d) ACCEPTANCE CHECK (greedy mode, temperature=0):
        If tree_tokens[i] == target_greedy_prediction:
          → ACCEPT. Keep tree_tokens[i]. Continue to children.
        Else:
          → REJECT. Take target_greedy_prediction instead.
          → INVALIDATE entire subtree (all descendants of node i
            will be skipped in subsequent steps due to rule 4a)
          → STOP processing further tree nodes for this cycle
            (the rejected replacement token is the last accepted
            token of this verification step)

   CRITICAL: The subtree invalidation at step (a) is the most common bug.
   Rejecting node i means ALL its descendants are invalid, even if they
   would have matched the target's predictions. They were generated
   conditioned on node i being correct, which turned out false.

5. FULL GENERATION LOOP:
   ```
   generated_tokens = list(prompt)
   while len(generated_tokens) < max_tokens:
       # Draft model produces a tree (mocked: you pass it in)
       tree_tokens, tree_parents = draft_model(generated_tokens)
       
       # Build tree attention mask
       mask = build_tree_mask(len(generated_tokens), tree_parents)
       
       # Run target model on [generated_tokens | tree_tokens]
       logits = target_model(generated_tokens + tree_tokens, mask)
       
       # Extract logits at tree positions only
       tree_logits = logits[len(generated_tokens):]
       
       # Acceptance/rejection
       accepted = accept_reject(tree_tokens, tree_parents,
                                tree_logits, temperature=0)
       
       # Append accepted tokens
       for token in accepted:
           generated_tokens.append(token)
       
       # If nothing accepted, fall back to target's greedy prediction
       # at the last prompt position
       if not accepted:
           prompt_logits = target_model(generated_tokens, causal_mask)
           new_token = argmax(prompt_logits[-1])
           generated_tokens.append(new_token)
   ```

6. DELIVERABLES:
   - Function build_tree_mask(prompt_len, tree_parents) → mask array (P+N, P+N)
   - Function verify_and_accept(prompt_tokens, tree_tokens, tree_parents,
                                target_model, temperature) → (accepted_tokens, new_token)
   - A MinimalLM class (or equivalent) for the target model
   - Test 1 (BASIC): prompt=[10, 20, 30], tree with 3 root nodes (no depth-2),
     temperature=0. Compare generated sequence against autoregressive
     greedy decoding. Must match EXACTLY.
   - Test 2 (SUBTREE INVALIDATION): Construct a tree where a depth-1
     node is REJECTED but its depth-2 children WOULD have been accepted
     (if processed independently). Verify the depth-2 children are
     correctly SKIPPED and the output matches autoregressive.
   - Test 3 (MULTI-STEP): Run 3 consecutive verification cycles where
     accepted tokens from cycle N become the prompt for cycle N+1.
     Verify the full generated sequence matches autoregressive.

   THE GOLDEN TEST: for temperature=0, speculative decoding MUST produce
   EXACTLY the same output sequence as autoregressive greedy decoding of
   the same target model. Any deviation is a bug in the implementation.

Use only NumPy. No PyTorch, JAX, TensorFlow, or autograd.
EOF
            ;;
        flash_attention_bwd)
            cat > "$path" << 'EOF'
Implement the BACKWARD pass of tiled (Flash) attention using online softmax
recomputation, from scratch in NumPy.

You must also write (or include) a minimal forward pass. The forward pass MUST
store only these intermediates per (B, H) head for the backward pass:
  - O:    (N, D)  — attention output
  - L:    (N,)    — logsumexp per query row: L_i = m_i + log(l_i)
    where m_i is the final row max and l_i is the final row sum of exps.
  - Q, K, V: the original inputs (needed for recomputation).

The forward MUST NOT store the full (N, N) attention matrix or softmax matrix.
It MUST process Q and K/V in tiles of size T using the online softmax recurrence.

BACKWARD PASS REQUIREMENTS:

1. RECOMPUTATION:
   Given dO (upstream gradient, same shape as O), Q, K, V, O, and L, compute:
     dQ: (B, H, N, D) — gradient w.r.t. queries
     dK: (B, H, N, D) — gradient w.r.t. keys
     dV: (B, H, N, D) — gradient w.r.t. values

   The backward pass must NOT materialize the full (N, N) attention matrix
   either. It recomputes softmax probabilities P on-the-fly from the stored
   L and locally recomputed S = Q_tile @ K_tile^T * scale.

2. GRADIENT FORMULAS (for a single tile interaction):
   Let scale = 1/sqrt(D). For each (Q_tile, KV_tile) pair:
   
   a) Recompute local attention scores: S = Q_tile @ K_tile^T * scale
      Shape: S is (T_q, T_kv) where T_q and T_kv are tile lengths.
   b) Recompute local softmax:
        P = exp(S - L_query[:, None])
      L_query are the logsumexp values for the query rows in this tile,
      broadcast against the key dimension. Shape: P is (T_q, T_kv).
   c) Compute local dV contribution and ACCUMULATE:
        dV_tile += P^T @ dO_tile
   d) Compute local dP:
        dP = dO_tile @ V_tile^T     Shape: (T_q, T_kv)
   e) Compute local dS via the softmax gradient:
        rowsum_PdP = (P * dP).sum(axis=-1, keepdims=True)   # shape (T_q, 1)
        dS = P * (dP - rowsum_PdP)
      This is the dsoftmax formula. The rowsum is over the KEY axis (last axis).
      The subtraction broadcasts rowsum_PdP from (T_q, 1) to (T_q, T_kv).
      The elementwise multiply by P is the FINAL step.
   f) Compute local dQ contribution and ACCUMULATE:
        dQ_tile += dS @ K_tile
   g) Compute local dK contribution and ACCUMULATE:
        dK_tile += dS^T @ Q_tile

   IMPORTANT: dQ, dK, dV contributions must be ACCUMULATED (added) across all
   KV tiles within a Q tile, not overwritten.

3. TILING:
   The backward pass uses the same tiling pattern as forward:
   - Outer loop over Q tiles (query blocks)
   - Inner loop over KV tiles (key/value blocks)
   - For causal attention, skip (Q_tile, KV_tile) pairs that are entirely
     above the diagonal (all key positions > all query positions)
   - Within each Q tile, initialize dQ_tile, dK_tile, dV_tile accumulators
     and accumulate contributions from each KV tile

4. BATCHING:
   Handle (B, H, N, D) tensors. You may loop over (b, h) or use batched
   operations — either is acceptable.

5. CAUSAL MASKING IN BACKWARD:
   When causal=True, the backward pass must apply the same masking pattern
   as the forward pass. For each (Q_tile, KV_tile) pair:
   - If the entire block is above the diagonal, SKIP it (no contribution
     to any gradient)
   - If partially masked, apply the causal mask to S before computing P:
       S = S + causal_mask  (masked positions = -inf)
     Then exp(S - L) gives 0 for masked positions, which correctly
     zeros out their contribution to dV, dS, dQ, and dK.

6. NUMERICAL STABILITY:
   - L already incorporates the row max from forward, so exp(S - L[:, None])
     has arguments ≤ 0, which is stable (no overflow).
   - The dsoftmax formula computes (dP - rowsum(P*dP)). Both dP and rowsum
     can be large, but the subtraction is benign because the result is
     multiplied by P (which sums to 1 per row), keeping dS bounded.
   - Use float64 for intermediate reductions if possible.

Deliver:
- Function flash_attention_fwd(Q, K, V, tile_size, causal=True)
  → returns (O, cache) where cache = {'O': O, 'L': L, 'Q': Q, 'K': K, 'V': V}
  and L has shape (B, H, N)
- Function flash_attention_bwd(dO, cache, tile_size, causal=True)
  → returns (dQ, dK, dV) each of shape (B, H, N, D)
- Test 1 (gradient check): (B=1, H=1, N=64, D=32, T=16, causal=True)
  → Compare dV against central finite differences across ALL elements
  → Spot-check dQ and dK at 10 random positions
  → Assert relative error < 1e-5 for all
- Test 2 (vs naive backward): (B=2, H=4, N=256, D=64, T=64, causal=True)
  → Compare dQ, dK, dV against naive full-materialized backward
  → Assert max relative error < 1e-4
- Test 3 (memory): (B=1, H=1, N=4096, D=64, T=128, causal=True)
  → Use tracemalloc to verify peak memory is less than 20% of the
    memory required for a single (N, N) matrix

Use only NumPy. No PyTorch, JAX, TensorFlow, or autograd.
EOF
            ;;
        ternary_training)
            cat > "$path" << 'EOF'
You are attempting to replicate Ternary Bonsai (PrismML, April 2026) — a family of
language models natively trained with ternary weights {-1, 0, +1} that achieve
competitive benchmark scores at 1/9th the memory of full-precision models.

This is an OPEN RESEARCH PROBLEM. PrismML has not released their training code.
What follows is everything the public knows. Your job is to fill in the gaps
and produce a working ternary training procedure.

================================================================================
WHAT IS KNOWN
================================================================================

Architecture:
- Ternary Bonsai uses the EXACT Qwen3 architecture (confirmed by HF model card,
  config.json, and multiple community sources).
- Qwen3 features: Grouped Query Attention (2:1 ratio), SwiGLU MLP, RoPE
  positional embeddings, RMSNorm, no bias in linear layers.
- Qwen3-0.6B: 28 layers, hidden_size=1024, 16 query heads, 8 KV heads,
  intermediate_size=3072, vocab_size=151936, max_position_embeddings=32768.
- ALL linear layers are ternary: embeddings, Q/K/V/O projections, SwiGLU
  gate/up/down projections, LM head. No high-precision escape hatches.
- RMSNorm and other normalization layers remain in FP16/FP32 (few params).

Ternary weight format:
- Group-wise quantization: groups of 128 weights share one FP16 scale factor `s`.
- Each weight in a group is {-s, 0, +s}, stored as {-1, 0, +1} (2 bits each).
- The scale factor per group is computed as: s = mean(|W_group|).
  Some BitNet variants use max(|W_group|) or a learned scale — the community
  believes PrismML uses mean absolute value based on ablation studies.
- Q2_0 is the GGUF packing format: 2 bits per weight, 4 code points where
  q=0 → -s, q=1 → 0, q=2 → +s, q=3 → reserved/unused.

Training procedure (from BitNet b1.58 lineage + PrismML hints):
- Weights are stored in full precision (FP32/FP16) as LATENT weights.
- On the FORWARD pass: project latent weights to ternary using group-wise
  scales, then use the ternary weights for computation.
- On the BACKWARD pass: use the Straight-Through Estimator (STE).
  The gradient through the rounding operation is treated as identity.
  dL/dW_latent = dL/dW_ternary. The scale factor is treated as constant.
- Training is done FROM SCRATCH (not post-training quantization of an
  existing model). However, the architecture is identical to Qwen3.
- The initialization likely follows BitNet: weights initialized with
  a normal distribution scaled by (fan_in)^(-0.5), then the ternary
  projection is applied from step 0.
- Optimizer: likely AdamW with weight decay. BitNet uses a specific
  learning rate schedule with warmup.
- Training data: unknown, but PrismML claims the models are competitive
  with Qwen3-8B, suggesting similar-scale pretraining data.

Key references to consult (web search recommended):
1. "BitNet b1.58" paper (Microsoft Research, 2024) — the foundation
2. PrismML blog: https://prismml.com/news/ternary-bonsai
3. PrismML GitHub: https://github.com/PrismML-Eng/Bonsai-demo
4. PrismML whitepaper (PDF in Bonsai-demo repo): ternary-bonsai-8b-whitepaper.pdf
5. HF model card: https://huggingface.co/prism-ml/Ternary-Bonsai-8B-mlx-2bit
6. llama.cpp Q2_0 kernel implementation (for packing format reference)
7. Bankai: https://github.com/... (post-training ternary adaptation method,
   different approach but relevant)

================================================================================
YOUR TASK
================================================================================

Implement ternary training and apply it to produce a working ternary model.
You have TWO paths — choose the one you can complete successfully:

PATH A (Recommended — Real Scale):
1. Use MLX (Apple's ML framework, native on this Mac) to load the Qwen3-0.6B
   checkpoint. MLX is pre-installed. Import it as `import mlx.core as mx`
   and `import mlx.nn as nn`. MLX tensors are NumPy-compatible.
2. Implement the ternary linear layer as an MLX module that:
   - Stores latent weights in float32
   - Projects to ternary on forward pass using group_size=128
   - Uses STE for gradient propagation
   - Handles the scale factor computation: s = mean(|W|) per group
3. Convert the loaded Qwen3-0.6B model to use ternary linear layers.
   Keep RMSNorm in float16. Keep the attention mechanism unchanged (it
   operates on activations, not stored weights).
4. Fine-tune the ternary model on a small text dataset for at least 200 steps.
   Use cross-entropy loss. Show that loss decreases.
5. After training, verify:
   a) ALL weights in ternary linear layers project to {-1, 0, +1} (× scales)
   b) The model can generate coherent text (qualitative check)
   c) Perplexity on a held-out set is not astronomical (< 100)
6. Explain your training procedure, hyperparameters chosen, and any
   observations about what worked and what didn't.

PATH B (NumPy-only, smaller scale):
1. Using only NumPy, implement a Qwen3-style transformer with the SAME
   architecture features (GQA 2:1, SwiGLU, RMSNorm, RoPE) but at a smaller
   scale: 6-8 layers, d_model=512-768, at least 4 attention heads.
2. Implement the ternary linear layer with group_size=128 and STE.
3. Train from scratch on a text corpus (WikiText-2 or similar) for at
   least 1000 steps. Use batch_size >= 16.
4. Verify ternary projection and measure perplexity improvement.
5. Explain your procedure and hyperparameters.

================================================================================
EVALUATION CRITERIA
================================================================================

Your solution will be judged on:
1. CORRECTNESS: After training, projected weights MUST be in {-1, 0, +1}.
   This is non-negotiable. Check with: abs(round(W/s) - {-1,0,+1}) < 1e-5.

2. CONVERGENCE: Training loss must decrease. If loss stays flat or increases,
   your STE implementation or learning rate is wrong.

3. FUNCTIONALITY: The model must produce non-random text. Even if quality is
   low, it must demonstrate it learned SOMETHING from the data.

4. ENGINEERING JUDGMENT: Explain your choices. Why group_size=128 and not 256?
   Why mean(|W|) for scale and not max(|W|)? What learning rate worked? What
   broke and how did you fix it?

================================================================================
RESOURCES ON THIS MACHINE
================================================================================

- MLX is available: `import mlx.core as mx`, `import mlx.nn as nn`
- NumPy is available
- GPU: Apple M4 with unified memory (use MLX for GPU acceleration)
- Qwen3-0.6B weights may be downloaded via:
  `from mlx_lm import load; model, tokenizer = load("Qwen/Qwen3-0.6B")`
  or from HuggingFace: Qwen/Qwen3-0.6B
- WikiText-2 is available via `from datasets import load_dataset` or
  can be downloaded as raw text
- Web search is available if you need to check paper details or APIs

================================================================================
NOTE
================================================================================

This is a GENUINELY OPEN PROBLEM. PrismML has not released their training code.
The BitNet b1.58 paper describes the concept but not the exact recipe for
training a competitive 8B model. Your implementation may not match PrismML's
exactly — that's expected. The goal is to produce a working ternary training
procedure and learn what works. Document your findings.

================================================================================
TRAINING DATA
================================================================================

A train_data.txt file is provided in the ternary_training/ folder. You MUST use
this file as your training data for ALL training, testing, and evaluation.

Instructions:
1. Read train_data.txt from the current folder
2. Tokenize it with the same tokenizer your model uses
3. Train on those tokens
4. For evaluation and generation tests, use samples from this same data
5. Keep all other architectural choices the same — only change the data source

After training, report:
1. Final training loss
2. Validation perplexity (measured on a held-out portion of train_data.txt)
3. Ternary verification result (are all weights in {-1, 0, +1}?)
4. 3-5 text generation samples from different prompts
5. Any interesting observations from this run
EOF
            ;;
        *)
            echo "ERROR: Unknown challenge '$folder'"
            exit 1
            ;;
    esac
}

# ── Scaffold ──────────────────────────────────────────────────────

mkdir -p "$DEST"

TOTAL=0
for entry in "${CHALLENGES[@]}"; do
    IFS='|' read -r folder difficulty <<< "$entry"
    sub="$DEST/$folder"
    mkdir -p "$sub"
    write_prompt "$folder"
    write_train_data "$folder"
    TOTAL=$((TOTAL + 1))
    echo "  [+] $folder"
done

echo ""
echo "Scaffolded $TOTAL challenges into '$DEST/'"
echo ""

# ── Instructions ──────────────────────────────────────────────────

cat << 'INSTRUCTIONS'

══════════════════════════════════════════════════════════════════════
 DIFFICULTY GUIDE
══════════════════════════════════════════════════════════════════════

  MEDIUM
    backwards/          Layer Norm backward pass (manual gradients)
    fuse/               Fused softmax + top-K GPU kernel (CUDA)
    kv/                 KV-cache system design (variable-length batching)

  HARD
    beam_search/        Batched beam search with length penalty + EOS
    flash_attention/    Tiled Flash Attention forward pass (online softmax)

  EXTRA HARD
    dflash_verify/      DFlash tree attention verification (speculative decoding)
    flash_attention_bwd/ Flash Attention backward pass (tiled, recompute, dsoftmax)

  OPEN-ENDED RESEARCH (SOTA-grade)
    ternary_training/   Native ternary-weight LLM training from scratch
                        (BitNet b1.58 lineage, STE, group-wise quantization)

══════════════════════════════════════════════════════════════════════
 HOW TO USE
══════════════════════════════════════════════════════════════════════

  1. Start with MEDIUM challenges to baseline the model
  2. Each subfolder contains PROMPT.md — feed this to the model as-is
  3. The model should produce a .py (or .cu for fuse/) implementation
  4. Run it:  cd <model>/<challenge> && python3 <file>.py
  5. All challenges have objective pass/fail criteria in the prompt

  Suggested order (easiest first):
    backwards → fuse → kv → flash_attention → beam_search →
    dflash_verify → flash_attention_bwd → ternary_training

══════════════════════════════════════════════════════════════════════

INSTRUCTIONS

echo "Done."
