"""DFlash-style tree-attention speculative decoding (NumPy only).

Implements:
  - MinimalLM: 1-block transformer target model with random weights.
  - build_tree_mask: constructs the (P+N, P+N) tree attention mask.
  - verify_and_accept: greedy verification + acceptance/rejection on a draft tree.
  - Tests asserting that for temperature=0 the speculative path produces
    EXACTLY the same sequence as autoregressive greedy decoding.
"""

import numpy as np


# ---------- numerics ----------

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return x - m - np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))


def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


# ---------- target model ----------

class MinimalLM:
    """Pre-norm decoder block with multi-head self-attention + ReLU FFN.

    No positional embeddings: order is enforced solely by the attention mask.
    This is what makes tree-decoded logits identical to autoregressive logits
    (a tree node's logits depend only on its ancestors, which is the same
    context an autoregressive run would have for the same token).
    """

    def __init__(self, vocab_size=1000, d_model=64, n_heads=4, d_ff=128, seed=42):
        assert d_model % n_heads == 0
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        scale = 1.0 / np.sqrt(d_model)

        self.tok_emb = rng.standard_normal((vocab_size, d_model)) * scale
        self.W_q = rng.standard_normal((d_model, d_model)) * scale
        self.W_k = rng.standard_normal((d_model, d_model)) * scale
        self.W_v = rng.standard_normal((d_model, d_model)) * scale
        self.W_o = rng.standard_normal((d_model, d_model)) * scale
        self.ln1_g = np.ones(d_model)
        self.ln1_b = np.zeros(d_model)
        self.ln2_g = np.ones(d_model)
        self.ln2_b = np.zeros(d_model)
        self.ln_f_g = np.ones(d_model)
        self.ln_f_b = np.zeros(d_model)
        self.W_ff1 = rng.standard_normal((d_model, d_ff)) * scale
        self.b_ff1 = np.zeros(d_ff)
        self.W_ff2 = rng.standard_normal((d_ff, d_model)) * scale
        self.b_ff2 = np.zeros(d_model)
        self.W_lm = rng.standard_normal((d_model, vocab_size)) * scale

    def forward(self, tokens, mask):
        tokens = np.asarray(tokens, dtype=int)
        T = len(tokens)
        assert mask.shape == (T, T), f"mask shape {mask.shape} != ({T},{T})"

        x = self.tok_emb[tokens]

        h = layer_norm(x, self.ln1_g, self.ln1_b)
        Q = (h @ self.W_q).reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)
        K = (h @ self.W_k).reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)
        V = (h @ self.W_v).reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_head)
        add_mask = np.where(mask, 0.0, -np.inf)
        scores = scores + add_mask[None, :, :]
        attn = softmax(scores, axis=-1)
        attn = np.nan_to_num(attn, nan=0.0)
        ctx = (attn @ V).transpose(1, 0, 2).reshape(T, self.d_model)
        x = x + ctx @ self.W_o

        h = layer_norm(x, self.ln2_g, self.ln2_b)
        h = np.maximum(0.0, h @ self.W_ff1 + self.b_ff1)
        x = x + h @ self.W_ff2 + self.b_ff2

        x = layer_norm(x, self.ln_f_g, self.ln_f_b)
        return x @ self.W_lm


# ---------- masks ----------

def causal_mask(T):
    return np.tril(np.ones((T, T), dtype=bool))


def build_tree_mask(prompt_len, tree_parents):
    P = prompt_len
    N = len(tree_parents)
    T = P + N
    mask = np.zeros((T, T), dtype=bool)

    for i in range(P):
        mask[i, : i + 1] = True

    if N > 0:
        mask[P:, :P] = True

    for i in range(N):
        mask[P + i, P + i] = True
        cur = tree_parents[i]
        while cur != -1:
            assert cur < i, "tree_parents must be in topological order"
            mask[P + i, P + cur] = True
            cur = tree_parents[cur]

    return mask


# ---------- decoding ----------

def autoregressive_greedy(model, prompt, num_tokens):
    tokens = list(prompt)
    for _ in range(num_tokens):
        logits = model.forward(np.array(tokens, dtype=int),
                               causal_mask(len(tokens)))
        tokens.append(int(np.argmax(logits[-1])))
    return tokens


def verify_and_accept(prompt_tokens, tree_tokens, tree_parents, target_model,
                      temperature=0):
    """Verify a draft tree against the target model and return accepted tokens.

    Returns (accepted_tokens, new_token):
      accepted_tokens: drafted tokens (from tree_tokens) accepted along the
        single chain that matches the target's greedy predictions.
      new_token: the next token to append — either the target's replacement
        (on rejection) or the bonus token from the deepest accepted position.

    The verification check at node i uses the logits at the PARENT'S position
    (P-1 for roots, P+parent_idx otherwise). Those logits are the target's
    greedy prediction for the slot tree_tokens[i] is competing for.
    """
    if temperature != 0:
        raise NotImplementedError("Only temperature=0 (greedy) supported")

    P = len(prompt_tokens)
    N = len(tree_tokens)

    if N == 0:
        logits = target_model.forward(np.array(prompt_tokens, dtype=int),
                                      causal_mask(P))
        return [], int(np.argmax(logits[-1]))

    full = np.array(list(prompt_tokens) + list(tree_tokens), dtype=int)
    mask = build_tree_mask(P, tree_parents)
    logits = target_model.forward(full, mask)

    accepted_chain = []
    rejected = set()
    new_token = None

    for i in range(N):
        # (4a) Subtree invalidation: skip if any ancestor was rejected.
        cur = tree_parents[i]
        anc_rejected = False
        while cur != -1:
            if cur in rejected:
                anc_rejected = True
                break
            cur = tree_parents[cur]
        if anc_rejected:
            rejected.add(i)
            continue

        parent_idx = tree_parents[i]
        if parent_idx == -1:
            on_active = (len(accepted_chain) == 0)
        else:
            on_active = (len(accepted_chain) > 0
                         and accepted_chain[-1] == parent_idx)
        if not on_active:
            rejected.add(i)
            continue

        parent_pos = (P - 1) if parent_idx == -1 else (P + parent_idx)
        log_probs = log_softmax(logits[parent_pos])
        target_token = int(np.argmax(log_probs))

        if target_token == tree_tokens[i]:
            accepted_chain.append(i)
        else:
            new_token = target_token
            rejected.add(i)
            break

    if new_token is None:
        last_pos = (P + accepted_chain[-1]) if accepted_chain else (P - 1)
        new_token = int(np.argmax(logits[last_pos]))

    accepted_tokens = [int(tree_tokens[i]) for i in accepted_chain]
    return accepted_tokens, new_token


# ---------- tests ----------

def _assert_eq(a, b, msg):
    if a != b:
        raise AssertionError(f"{msg}\n  got: {a}\n  exp: {b}")


def test_basic():
    model = MinimalLM(seed=42)
    prompt = [10, 20, 30]

    expected = autoregressive_greedy(model, prompt, 3)
    next_tok = expected[len(prompt)]

    # Three sibling roots; only one matches the target's greedy prediction.
    tree_tokens = [next_tok, (next_tok + 1) % 1000, (next_tok + 2) % 1000]
    tree_parents = [-1, -1, -1]

    accepted, new_token = verify_and_accept(prompt, tree_tokens, tree_parents,
                                            model)
    generated = list(prompt) + accepted + [new_token]

    expected_seq = autoregressive_greedy(model, prompt, len(generated) - len(prompt))
    _assert_eq(generated, expected_seq, "Test 1 (basic) mismatch")

    print("Test 1 (BASIC): PASS")
    print(f"  accepted={accepted}, new_token={new_token}, generated={generated}")


def test_subtree_invalidation():
    model = MinimalLM(seed=42)
    prompt = [10, 20, 30]
    expected = autoregressive_greedy(model, prompt, 3)
    next_tok = expected[3]

    # depth-1 root: a token that is NOT the target's greedy prediction → reject.
    wrong_root = (next_tok + 1) % 1000
    # depth-2 child: under autoregressive [prompt, wrong_root], some token would
    # be predicted; but since wrong_root is rejected, the child must be SKIPPED
    # regardless of its value. We pick something arbitrary.
    suspicious_child = (next_tok + 7) % 1000

    tree_tokens = [wrong_root, suspicious_child]
    tree_parents = [-1, 0]

    accepted, new_token = verify_and_accept(prompt, tree_tokens, tree_parents,
                                            model)

    _assert_eq(accepted, [], "Test 2: nothing should be accepted")
    _assert_eq(new_token, next_tok, "Test 2: replacement should be target's argmax")

    generated = list(prompt) + accepted + [new_token]
    _assert_eq(generated, expected[:4], "Test 2: output != autoregressive")

    print("Test 2 (SUBTREE INVALIDATION): PASS")
    print(f"  rejected wrong_root={wrong_root}, skipped child={suspicious_child}, "
          f"new_token={new_token}")


def test_multi_step():
    model = MinimalLM(seed=42)
    prompt = [10, 20, 30]
    cycles = 3
    expected = autoregressive_greedy(model, prompt, cycles * 2)

    generated = list(prompt)

    for cycle in range(cycles):
        idx = len(generated)
        # Mock draft: oracle the correct depth-1 root and an intentionally wrong
        # depth-2 child. The cycle should accept the root and reject the child,
        # contributing exactly 2 tokens (1 accepted draft + 1 replacement).
        ar = autoregressive_greedy(model, generated, 2)
        correct_root = ar[idx]
        wrong_child = (ar[idx + 1] + 7) % 1000

        tree_tokens = [correct_root, wrong_child]
        tree_parents = [-1, 0]

        accepted, new_token = verify_and_accept(generated, tree_tokens,
                                                tree_parents, model)
        _assert_eq(accepted, [correct_root],
                   f"Test 3 cycle {cycle}: expected accept of correct root")
        _assert_eq(new_token, ar[idx + 1],
                   f"Test 3 cycle {cycle}: expected replacement = autoregressive next")
        generated += accepted + [new_token]

    expected_seq = autoregressive_greedy(model, prompt, len(generated) - len(prompt))
    _assert_eq(generated, expected_seq, "Test 3 mismatch vs. autoregressive")

    print("Test 3 (MULTI-STEP): PASS")
    print(f"  generated={generated}")


def test_mask_shape_and_rules():
    # Spot-check the mask construction itself.
    P = 3
    # Tree: 0 (root), 1 (root), 2 (child of 0), 3 (child of 2), 4 (child of 1)
    parents = [-1, -1, 0, 2, 1]
    M = build_tree_mask(P, parents)
    T = P + len(parents)
    assert M.shape == (T, T)

    # Prompt causal block.
    for i in range(P):
        for j in range(T):
            if j < P:
                assert M[i, j] == (j <= i), (i, j)
            else:
                assert M[i, j] == False, (i, j)

    # Tree rows: must attend to all prompt + self + transitive ancestors only.
    def ancestors(k):
        s = {k}
        cur = parents[k]
        while cur != -1:
            s.add(cur)
            cur = parents[cur]
        return s

    for k in range(len(parents)):
        row = M[P + k]
        assert row[:P].all(), f"tree node {k} should attend to all prompt"
        anc = ancestors(k)
        for kk in range(len(parents)):
            assert row[P + kk] == (kk in anc), (k, kk)

    print("Test 0 (MASK): PASS")


if __name__ == "__main__":
    test_mask_shape_and_rules()
    test_basic()
    test_subtree_invalidation()
    test_multi_step()
    print("\nAll tests passed.")
