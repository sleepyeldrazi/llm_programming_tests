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
