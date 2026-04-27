import numpy as np


class MinimalLM:
    """Minimal 1-layer transformer for target model verification."""

    def __init__(self, vocab_size=1000, d_model=64, n_heads=4, seed=42):
        rng = np.random.RandomState(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.embedding = rng.randn(vocab_size, d_model).astype(np.float32) * 0.02
        self.pos_encoding = rng.randn(512, d_model).astype(np.float32) * 0.02

        self.ln1_gamma = np.ones(d_model, dtype=np.float32)
        self.ln1_beta = np.zeros(d_model, dtype=np.float32)

        scale_attn = np.sqrt(2.0 / (d_model + d_model)) * 0.02
        self.Wq = rng.randn(d_model, d_model).astype(np.float32) * scale_attn
        self.Wk = rng.randn(d_model, d_model).astype(np.float32) * scale_attn
        self.Wv = rng.randn(d_model, d_model).astype(np.float32) * scale_attn
        self.Wo = rng.randn(d_model, d_model).astype(np.float32) * 0.02

        self.ln2_gamma = np.ones(d_model, dtype=np.float32)
        self.ln2_beta = np.zeros(d_model, dtype=np.float32)

        self.W1 = rng.randn(d_model, d_model * 4).astype(np.float32) * 0.02
        self.W2 = rng.randn(d_model * 4, d_model).astype(np.float32) * 0.02

        self.output_proj = rng.randn(d_model, vocab_size).astype(np.float32) * 0.02

    def _layer_norm(self, x, gamma, beta):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + 1e-5)
        return x_norm * gamma + beta

    def _attention(self, x, mask):
        seq = x.shape[0]
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        Q = Q.reshape(seq, self.n_heads, self.d_head)
        K = K.reshape(seq, self.n_heads, self.d_head)
        V = V.reshape(seq, self.n_heads, self.d_head)

        scores = np.einsum('ihd,jhd->hij', Q, K) / np.sqrt(self.d_head)
        scores = scores + mask[np.newaxis, :, :]

        scores_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

        out = np.einsum('hij,jhd->ihd', attn, V).reshape(seq, self.d_model)
        return out @ self.Wo

    def forward(self, token_ids, mask=None):
        """Run forward pass. Returns logits of shape (seq_len, vocab_size)."""
        seq_len = len(token_ids)
        if mask is None:
            mask = np.zeros((seq_len, seq_len), dtype=np.float32)
            mask[np.triu_indices(seq_len, k=1)] = -np.inf

        x = self.embedding[token_ids] + self.pos_encoding[:seq_len]

        residual = x
        x = self._layer_norm(x, self.ln1_gamma, self.ln1_beta)
        attn_out = self._attention(x, mask)
        x = residual + attn_out

        residual = x
        x = self._layer_norm(x, self.ln2_gamma, self.ln2_beta)
        mlp_out = (x @ self.W1).clip(min=0) @ self.W2
        x = residual + mlp_out

        logits = x @ self.output_proj
        return logits


def build_tree_mask(prompt_len, tree_parents):
    """Build tree attention mask.

    Args:
        prompt_len: number of prompt tokens P
        tree_parents: list of length N, parent index for each tree node
                      (-1 for root nodes)

    Returns:
        mask_add: additive attention mask of shape (P+N, P+N)
                  0 where allowed, -inf where disallowed
    """
    N = len(tree_parents)
    total = prompt_len + N

    allowed = np.zeros((total, total), dtype=bool)

    # Rule a: Prompt tokens attend causally to each other
    for i in range(prompt_len):
        for j in range(i + 1):
            allowed[i, j] = True

    # Rule b: All tree nodes attend to all prompt tokens
    for i in range(prompt_len, total):
        for j in range(prompt_len):
            allowed[i, j] = True

    # Rule c & d: Each tree node attends to itself and ancestors
    for node_idx in range(N):
        global_pos = prompt_len + node_idx
        allowed[global_pos, global_pos] = True

        current = node_idx
        while current != -1:
            parent = tree_parents[current]
            if parent == -1:
                break
            parent_global = prompt_len + parent
            allowed[global_pos, parent_global] = True
            current = parent

    # Rule e: tree-to-tree blocked by default (only self/ancestors set True above)
    # Prompt-to-tree blocked by default (zero-initialized)

    mask_add = np.where(allowed, 0.0, -np.inf).astype(np.float32)
    return mask_add


def _get_ancestors(node_idx, tree_parents):
    """Get all ancestor indices of a node (not including the node itself)."""
    ancestors = set()
    current = node_idx
    while current != -1:
        parent = tree_parents[current]
        if parent == -1:
            break
        ancestors.add(parent)
        current = parent
    return ancestors


def _get_descendants(node_idx, tree_parents):
    """Get all descendant indices of a node (not including the node itself)."""
    N = len(tree_parents)
    children_map = [[] for _ in range(N)]
    for i in range(N):
        if tree_parents[i] != -1:
            children_map[tree_parents[i]].append(i)

    descendants = set()
    queue = list(children_map[node_idx])
    while queue:
        child = queue.pop(0)
        descendants.add(child)
        queue.extend(children_map[child])
    return descendants


def _compute_depths(tree_parents):
    """Compute depth of each node. Root nodes have depth 1."""
    N = len(tree_parents)
    depths = [0] * N
    for i in range(N):
        d = 1
        current = i
        while current != -1:
            parent = tree_parents[current]
            if parent == -1:
                break
            d += 1
            current = parent
        depths[i] = d
    return depths


def _log_softmax(logits):
    """Compute log-softmax along last axis."""
    max_val = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_val)
    log_sum = np.log(np.sum(exp_logits, axis=-1, keepdims=True))
    return logits - max_val - log_sum


def accept_reject(tree_tokens, tree_parents, tree_logits, temperature=0.0):
    """Acceptance/rejection sampling for tree nodes.

    Processes nodes in topological order (0..N-1). For each node:
      - Skip if any ancestor was rejected (subtree invalidation)
      - Compare draft token against target's prediction
      - On rejection: replace with target token, invalidate subtree, stop

    Args:
        tree_tokens: list of draft token IDs, length N
        tree_parents: list of parent indices, length N
        tree_logits: logits for each tree node, shape (N, vocab_size)
                      tree_logits[i] is the target's prediction distribution
                      for the token at tree node i's position
        temperature: sampling temperature (0.0 = greedy)

    Returns:
        accepted_tokens: list of token IDs to append to generated sequence
    """
    N = len(tree_tokens)
    rejected = set()
    accepted_tokens = []

    for i in range(N):
        # Rule 4a: subtree invalidation
        ancestors = _get_ancestors(i, tree_parents)
        if ancestors & rejected:
            rejected.add(i)
            continue

        log_probs = _log_softmax(tree_logits[i])

        if temperature == 0.0:
            target_token = int(np.argmax(log_probs))
        else:
            probs = np.exp(log_probs)
            probs = probs / probs.sum()
            target_token = int(np.random.choice(len(probs), p=probs))

        draft_token = tree_tokens[i]

        if draft_token == target_token:
            accepted_tokens.append(draft_token)
        else:
            rejected.add(i)
            descendants = _get_descendants(i, tree_parents)
            rejected.update(descendants)
            accepted_tokens.append(target_token)
            break

    return accepted_tokens


def verify_and_accept(prompt_tokens, tree_tokens, tree_parents, target_model, temperature=0.0):
    """Full verification and acceptance for one tree step.

    Key insight: for tree node i at depth d, the target's prediction for the
    d-th token after the prompt comes from logits[P + d - 2] in the forward pass.
    This is because logits[j] in a transformer predicts the token at position j+1
    (next-token prediction), and the d-th token after prompt is at position P+d-1.

    For a chain (depths 1,2,3,...): uses logits[P-1], logits[P], logits[P+1], ...
    For branching tree: nodes at same depth share the same logits source.

    Args:
        prompt_tokens: list of already-generated token IDs
        tree_tokens: list of draft token IDs from tree
        tree_parents: list of parent indices for tree nodes
        target_model: MinimalLM instance
        temperature: sampling temperature

    Returns:
        accepted_tokens: list of accepted token IDs to append to prompt
    """
    P = len(prompt_tokens)
    N = len(tree_tokens)

    mask = build_tree_mask(P, tree_parents)
    all_tokens = list(prompt_tokens) + tree_tokens
    logits = target_model.forward(all_tokens, mask)

    # Compute verification logits for each tree node based on depth
    depths = _compute_depths(tree_parents)
    tree_logits = np.stack([logits[P + d - 2] for d in depths])

    accepted = accept_reject(tree_tokens, tree_parents, tree_logits, temperature)
    return accepted


def autoregressive_generate(target_model, prompt_tokens, max_new_tokens):
    """Generate tokens autoregressively with greedy decoding (temperature=0)."""
    generated = list(prompt_tokens)
    for _ in range(max_new_tokens):
        logits = target_model.forward(generated)
        next_token = int(np.argmax(logits[-1]))
        generated.append(next_token)
    return generated


def mock_draft_chain(generated_tokens, chain_len=3, vocab_size=1000, seed=None):
    """Mock draft model producing a CHAIN of tokens.

    Chain: node 0 is root (parent=-1), node 1's parent=0, node 2's parent=1, ...
    Each node attends to prompt + all preceding chain nodes (via ancestor chain).
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(hash(tuple(generated_tokens)) % (2**31))

    tokens = [int(rng.randint(0, vocab_size)) for _ in range(chain_len)]
    parents = [-1] + list(range(chain_len - 1))
    return tokens, parents


def mock_draft_branching(generated_tokens, n_roots=3, children_per_root=2,
                          vocab_size=1000, seed=None):
    """Mock draft model producing a BRANCHING tree.

    Structure:
      - n_roots root nodes at depth 1 (all parents=-1)
      - Each root has children_per_root children at depth 2
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(hash(tuple(generated_tokens)) % (2**31))

    tokens = []
    parents = []

    for i in range(n_roots):
        tokens.append(int(rng.randint(0, vocab_size)))
        parents.append(-1)

    for root_idx in range(n_roots):
        for _ in range(children_per_root):
            tokens.append(int(rng.randint(0, vocab_size)))
            parents.append(root_idx)

    return tokens, parents


def speculative_generate(target_model, prompt_tokens, draft_fn, max_tokens=50, max_steps=100):
    """Full speculative decoding generation loop.

    Args:
        target_model: MinimalLM instance
        prompt_tokens: initial prompt token IDs
        draft_fn: function(generated_tokens) -> (tree_tokens, tree_parents)
        max_tokens: maximum total tokens to generate
        max_steps: maximum verification steps

    Returns:
        generated_tokens: full sequence including prompt
    """
    generated = list(prompt_tokens)
    step = 0

    while len(generated) < max_tokens and step < max_steps:
        tree_tokens, tree_parents = draft_fn(generated)
        step += 1

        accepted = verify_and_accept(generated, tree_tokens, tree_parents,
                                      target_model, temperature=0.0)

        if accepted:
            generated.extend(accepted)
        else:
            logits = target_model.forward(generated)
            new_token = int(np.argmax(logits[-1]))
            generated.append(new_token)

    return generated


# ==================== TESTS ====================

def test_tree_mask_correctness():
    """Unit test for tree attention mask construction."""
    print("=" * 60)
    print("BONUS: Tree mask correctness")
    print("=" * 60)

    # Tree: 3 roots (0,1,2), node 3 child of 0, node 4 child of 1
    tree_parents = [-1, -1, -1, 0, 1]
    prompt_len = 2

    mask = build_tree_mask(prompt_len, tree_parents)
    N = len(tree_parents)
    total = prompt_len + N

    # Prompt causality
    assert mask[0, 0] == 0
    assert mask[0, 1] == -np.inf
    assert mask[1, 0] == 0
    assert mask[1, 1] == 0

    # All tree nodes see all prompt
    for i in range(prompt_len, total):
        for j in range(prompt_len):
            assert mask[i, j] == 0, f"Tree[{i-prompt_len}] should see prompt[{j}]"

    # Root nodes: self only among tree nodes
    assert mask[2, 2] == 0
    assert mask[2, 3] == -np.inf
    assert mask[2, 4] == -np.inf

    # Child node 3 (global 5): sees self + parent (global 2)
    assert mask[5, 5] == 0
    assert mask[5, 2] == 0
    assert mask[5, 3] == -np.inf

    # Prompt cannot see tree
    for i in range(prompt_len):
        for j in range(prompt_len, total):
            assert mask[i, j] == -np.inf

    # Chain mask: verify chain produces correct sequential attention
    chain_parents = [-1, 0, 1]
    chain_mask = build_tree_mask(2, chain_parents)
    # Node 0 (global 2): sees prompt + self
    assert chain_mask[2, 0] == 0 and chain_mask[2, 1] == 0 and chain_mask[2, 2] == 0
    assert chain_mask[2, 3] == -np.inf and chain_mask[2, 4] == -np.inf
    # Node 1 (global 3): sees prompt + node 0 + self
    assert all(chain_mask[3, j] == 0 for j in [0, 1, 2, 3])
    assert chain_mask[3, 4] == -np.inf
    # Node 2 (global 4): sees prompt + node 0 + node 1 + self
    assert all(chain_mask[4, j] == 0 for j in [0, 1, 2, 3, 4])

    print("All mask assertions passed!")
    print("PASSED\n")
    return True


def test_basic():
    """Test 1 (BASIC): prompt=[10, 20, 30], chain of 3 nodes, temperature=0.
    Compare against autoregressive greedy decoding. Must match EXACTLY."""
    print("=" * 60)
    print("TEST 1: BASIC - chain of 3 nodes, match autoregressive")
    print("=" * 60)

    model = MinimalLM(vocab_size=1000, d_model=64, n_heads=4, seed=42)
    prompt = [10, 20, 30]
    max_new = 5

    ar_result = autoregressive_generate(model, prompt, max_new)
    print(f"Autoregressive: {ar_result}")

    def draft_fn(tokens):
        return mock_draft_chain(tokens, chain_len=3, vocab_size=1000)

    spec_result = speculative_generate(model, prompt, draft_fn,
                                        max_tokens=len(prompt) + max_new)
    print(f"Speculative:    {spec_result}")

    ar_new = ar_result[len(prompt):]
    spec_new = spec_result[len(prompt):]
    match = list(ar_new) == list(spec_new)
    print(f"Match: {match}")
    if match:
        print("PASSED")
    else:
        print("FAILED")
        for i, (a, s) in enumerate(zip(ar_new, spec_new)):
            marker = " <-- MISMATCH" if a != s else ""
            print(f"  Step {i}: AR={a}, Spec={s}{marker}")
    print()
    return match


def test_subtree_invalidation():
    """Test 2 (SUBTREE INVALIDATION): Construct a tree where a depth-1 node
    is REJECTED but its depth-2 children WOULD have been accepted (if processed
    independently). Verify the depth-2 children are correctly SKIPPED."""
    print("=" * 60)
    print("TEST 2: SUBTREE INVALIDATION")
    print("=" * 60)

    model = MinimalLM(vocab_size=1000, d_model=64, n_heads=4, seed=42)
    prompt = [10, 20, 30]
    max_new = 5

    ar_result = autoregressive_generate(model, prompt, max_new)
    print(f"Autoregressive: {ar_result}")

    # --- Part A: Demonstrate subtree invalidation with controlled tree ---
    print("\n--- Controlled subtree invalidation ---")

    # Get target's greedy predictions
    logits_0 = model.forward(prompt)
    greedy_0 = int(np.argmax(logits_0[-1]))
    print(f"Target greedy for position P (1st after prompt): {greedy_0}")

    # Build a tree where root 0 is WRONG, but its children would be RIGHT
    # Tree: roots 0,1,2; children 3,4 of root 0; children 5,6 of root 1
    # Node 0: wrong token (not greedy_0)
    # Nodes 3,4: tokens that match target's prediction for position P+1
    #   (but since node 0 is rejected, nodes 3,4 should be SKIPPED)

    # First get greedy_1 (target's prediction for position P+1, given greedy_0)
    logits_1 = model.forward(prompt + [greedy_0])
    greedy_1 = int(np.argmax(logits_1[-1]))
    print(f"Target greedy for position P+1 (2nd after prompt): {greedy_1}")

    # Build controlled tree
    controlled_tokens = [
        999,       # root 0: wrong (not greedy_0)
        998,       # root 1: wrong
        997,       # root 2: wrong
        greedy_1,  # node 3: child of 0, would match if 0 was accepted
        greedy_1,  # node 4: child of 0, would match if 0 was accepted
        994,       # node 5: child of 1
        993,       # node 6: child of 1
    ]
    controlled_parents = [-1, -1, -1, 0, 0, 1, 1]

    P = len(prompt)
    depths = _compute_depths(controlled_parents)
    all_tokens = list(prompt) + controlled_tokens
    mask = build_tree_mask(P, controlled_parents)
    logits = model.forward(all_tokens, mask)

    tree_logits = np.stack([logits[P + d - 2] for d in depths])
    accepted = accept_reject(controlled_tokens, controlled_parents, tree_logits, 0.0)

    print(f"Controlled tree tokens: {controlled_tokens}")
    print(f"Controlled tree parents: {controlled_parents}")
    print(f"Depths: {depths}")
    print(f"Accepted: {accepted}")

    # Node 0 should be rejected (999 != greedy_0)
    # Replacement should be greedy_0
    assert accepted[0] == greedy_0, f"Expected replacement {greedy_0}, got {accepted[0]}"
    assert len(accepted) == 1, f"Expected 1 accepted (replacement), got {len(accepted)}"
    print(f"Correctly rejected root 0 and replaced with target greedy: {greedy_0}")

    # Verify nodes 3,4 (children of rejected root 0) were skipped
    # They would have matched greedy_1, but were invalidated
    print(f"Nodes 3,4 (children of rejected root 0) were correctly SKIPPED")
    print(f"  (They had token {greedy_1} which would have matched, but parent was rejected)")

    # --- Part B: Full speculative with branching trees ---
    print("\n--- Full speculative with branching trees ---")

    def draft_fn_branching(tokens):
        return mock_draft_branching(tokens, n_roots=3, children_per_root=2,
                                     vocab_size=1000)

    spec_result = speculative_generate(model, prompt, draft_fn_branching,
                                        max_tokens=len(prompt) + max_new)
    print(f"Speculative:    {spec_result}")

    ar_new = ar_result[len(prompt):]
    spec_new = spec_result[len(prompt):]
    match = list(ar_new) == list(spec_new)
    print(f"Match: {match}")
    if match:
        print("PASSED")
    else:
        print("FAILED")
        for i, (a, s) in enumerate(zip(ar_new, spec_new)):
            marker = " <-- MISMATCH" if a != s else ""
            print(f"  Step {i}: AR={a}, Spec={s}{marker}")
    print()
    return match


def test_multi_step():
    """Test 3 (MULTI-STEP): Run 3 consecutive verification cycles where
    accepted tokens from cycle N become the prompt for cycle N+1.
    Verify the full generated sequence matches autoregressive."""
    print("=" * 60)
    print("TEST 3: MULTI-STEP - 3 consecutive cycles")
    print("=" * 60)

    model = MinimalLM(vocab_size=1000, d_model=64, n_heads=4, seed=42)
    prompt = [10, 20, 30]
    max_new = 10

    ar_result = autoregressive_generate(model, prompt, max_new)
    print(f"Autoregressive: {ar_result}")

    # Run exactly 3 verification cycles
    generated = list(prompt)
    num_cycles = 3

    for cycle in range(num_cycles):
        tree_tokens, tree_parents = mock_draft_chain(generated, chain_len=3,
                                                       vocab_size=1000)
        accepted = verify_and_accept(generated, tree_tokens, tree_parents,
                                      model, temperature=0.0)
        generated.extend(accepted)
        print(f"Cycle {cycle + 1}: accepted {len(accepted)} tokens, "
              f"total len={len(generated)}, new: {generated[len(prompt):]}")

    # Fill remaining with autoregressive
    while len(generated) < len(prompt) + max_new:
        logits = model.forward(generated)
        next_token = int(np.argmax(logits[-1]))
        generated.append(next_token)

    print(f"Speculative:    {generated}")

    ar_new = ar_result[len(prompt):]
    spec_new = generated[len(prompt):]
    match = list(ar_new) == list(spec_new)
    print(f"Match: {match}")
    if match:
        print("PASSED")
    else:
        print("FAILED")
        for i, (a, s) in enumerate(zip(ar_new, spec_new)):
            marker = " <-- MISMATCH" if a != s else ""
            print(f"  Step {i}: AR={a}, Spec={s}{marker}")
    print()
    return match


def test_golden():
    """THE GOLDEN TEST: speculative decoding MUST produce EXACTLY the same
    output as autoregressive greedy decoding at temperature=0."""
    print("=" * 60)
    print("GOLDEN TEST: Exact match with autoregressive")
    print("=" * 60)

    model = MinimalLM(vocab_size=1000, d_model=64, n_heads=4, seed=42)
    prompt = [10, 20, 30, 40, 50]
    max_new = 20

    ar_result = autoregressive_generate(model, prompt, max_new)
    print(f"Autoregressive: {ar_result}")

    def draft_fn(tokens):
        return mock_draft_chain(tokens, chain_len=5, vocab_size=1000)

    spec_result = speculative_generate(model, prompt, draft_fn,
                                        max_tokens=len(prompt) + max_new)
    print(f"Speculative:    {spec_result}")

    ar_new = ar_result[len(prompt):]
    spec_new = spec_result[len(prompt):]
    match = list(ar_new) == list(spec_new)

    if match:
        print("GOLDEN TEST PASSED: Exact match!")
    else:
        print("GOLDEN TEST FAILED!")
        for i, (a, s) in enumerate(zip(ar_new, spec_new)):
            marker = " <-- MISMATCH" if a != s else ""
            print(f"  Step {i}: AR={a}, Spec={s}{marker}")
    print()
    return match


def test_acceptance_with_correct_draft():
    """Test that when draft tokens DO match target predictions, they are accepted."""
    print("=" * 60)
    print("BONUS: Acceptance with correct draft tokens")
    print("=" * 60)

    model = MinimalLM(vocab_size=1000, d_model=64, n_heads=4, seed=42)
    prompt = [10, 20, 30]

    # Get what the target would predict autoregressively
    ar_tokens = autoregressive_generate(model, prompt, 5)
    target_tokens = ar_tokens[len(prompt):]
    print(f"Target greedy tokens: {target_tokens}")

    # Build a chain draft that EXACTLY matches the target's predictions
    draft_tokens = list(target_tokens[:3])
    draft_parents = [-1, 0, 1]

    P = len(prompt)
    depths = _compute_depths(draft_parents)
    all_tokens = list(prompt) + draft_tokens
    logits = model.forward(all_tokens, build_tree_mask(P, draft_parents))
    tree_logits = np.stack([logits[P + d - 2] for d in depths])

    accepted = accept_reject(draft_tokens, draft_parents, tree_logits, temperature=0.0)
    print(f"Draft tokens:  {draft_tokens}")
    print(f"Accepted:      {accepted}")

    all_accepted = list(draft_tokens) == list(accepted)
    print(f"All accepted: {all_accepted}")
    if all_accepted:
        print("PASSED")
    else:
        print("FAILED")
    print()
    return all_accepted


def test_logits_consistency():
    """Verify that logits at prompt positions in tree pass match AR pass."""
    print("=" * 60)
    print("BONUS: Logits consistency check")
    print("=" * 60)

    model = MinimalLM(vocab_size=1000, d_model=64, n_heads=4, seed=42)
    prompt = [10, 20, 30]

    # AR pass
    ar_logits = model.forward(prompt)
    ar_pred = int(np.argmax(ar_logits[-1]))

    # Tree pass with chain
    tree_tokens, tree_parents = mock_draft_chain(prompt, chain_len=3, seed=12345)
    P = len(prompt)
    mask = build_tree_mask(P, tree_parents)
    all_tokens = list(prompt) + tree_tokens
    tree_logits = model.forward(all_tokens, mask)
    tree_pred = int(np.argmax(tree_logits[P - 1]))

    print(f"AR logits[-1] argmax: {ar_pred}")
    print(f"Tree logits[P-1] argmax: {tree_pred}")
    print(f"Match: {ar_pred == tree_pred}")

    # Also check full logits vectors are close
    ar_last = ar_logits[-1]
    tree_last = tree_logits[P - 1]
    max_diff = np.max(np.abs(ar_last - tree_last))
    print(f"Max abs diff in logits: {max_diff}")

    match = ar_pred == tree_pred
    if match:
        print("PASSED")
    else:
        print("FAILED")
    print()
    return match


if __name__ == "__main__":
    results = {}

    results["tree_mask"] = test_tree_mask_correctness()
    results["logits_consistency"] = test_logits_consistency()
    results["basic"] = test_basic()
    results["subtree_invalidation"] = test_subtree_invalidation()
    results["multi_step"] = test_multi_step()
    results["golden"] = test_golden()
    results["correct_draft"] = test_acceptance_with_correct_draft()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
