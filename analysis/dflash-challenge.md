# DFlash: Tree Attention Verification for Speculative Decoding

## What this is

DFlash (z-lab, Feb 2025) is a speculative decoding technique where a tiny
block-diffusion draft model generates a TREE of candidate tokens in one
forward pass, and the target model verifies them all at once using a
tree-structured attention mask. This is fundamentally harder than standard
linear-chain speculative decoding because:

1. The attention mask isn't causal, isn't full — it's a DAG
2. The acceptance/rejection algorithm must handle subtree invalidation
3. The llama.cpp PR (#22105) and the Luce-Org fork both have bugs where
   subtrees of rejected nodes are incorrectly processed

The challenge: implement the verification pass and acceptance/rejection
correctly. The test is binary — output must match autoregressive greedy
decoding exactly.

## The prompt

```
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
        if node k is an ancestor of node i, then M[P+i, P+k] = True
        (Find ancestors by following parent pointers to root)
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
            will be skipped in subsequent steps)
          → The rejected token (target_greedy_prediction) becomes
            the LAST token of this verification cycle — no further
            tree nodes are processed for this step
   
   e) After each ACCEPTED token, the target model's hidden state at
      that position becomes the new context for the NEXT verification
      cycle. In our simplified version, accepted tokens are appended
      to the generated sequence and the process repeats.

   f) For non-greedy (temperature > 0) mode:
        Compute q = draft model's log-probability for tree_tokens[i]
        Compute p = target model's log-probability for tree_tokens[i]
        Sample r ~ Uniform(0, 1)
        If r < exp(p - q):  [equivalent to min(1, p(x_d)/q(x_d))]
          → ACCEPT
        Else:
          → REJECT: sample replacement from softmax(max(0, p - q))
            where p and q are probability vectors, not log-probs
          → INVALIDATE subtree

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
       accepted, new_token = accept_reject(tree_tokens, tree_parents,
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
   - Function build_tree_mask(prompt_len, tree_parents) → mask (P+N, P+N)
   - Function verify_and_accept(prompt_tokens, tree_tokens, tree_parents,
                                target_model, temperature) → (accepted_tokens, new_token)
   - A MinimalLM class (or equivalent) for the target model
   - Test 1 (BASIC): prompt=[10, 20, 30], tree with 3 root nodes,
     temperature=0. Compare generated sequence against autoregressive
     greedy decoding. Must match EXACTLY.
   - Test 2 (SUBTREE INVALIDATION): Construct a tree where a depth-1
     node is REJECTED but its depth-2 children WOULD have been accepted
     (if processed independently). Verify the depth-2 children are
     correctly SKIPPED and the output matches autoregressive.
   - Test 3 (MULTI-STEP): Run 3 consecutive verification cycles where
     accepted tokens from cycle N become the prompt for cycle N+1.
     Verify the full generated sequence matches autoregressive.

   The test oracle: for temperature=0, speculative decoding MUST produce
   EXACTLY the same output sequence as autoregressive greedy decoding of
   the same target model. Any deviation is a bug.

Use only NumPy. No PyTorch, JAX, TensorFlow, or autograd.
```
