"""
DFlash-style Tree Attention Verification for Speculative Decoding.
Pure NumPy implementation.

Convention: logits[i] predicts the next token after position i.
To verify tree_tokens[i], we check the target's prediction at the
parent's position (or P-1 for root nodes).
"""
import numpy as np


# ── Utility functions ──────────────────────────────────────────────

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    lse = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return x - m - lse


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def sinusoidal_pe(max_len, d):
    pe = np.zeros((max_len, d))
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d, 2) * -(np.log(10000.0) / d))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


# ── Model components ───────────────────────────────────────────────

class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.g = np.ones(d)
        self.b = np.zeros(d)
        self.eps = eps

    def __call__(self, x):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return self.g * (x - mu) / np.sqrt(var + self.eps) + self.b


class Linear:
    def __init__(self, d_in, d_out, rng):
        self.w = rng.randn(d_in, d_out) * np.sqrt(2.0 / d_in)
        self.b = np.zeros(d_out)

    def __call__(self, x):
        return x @ self.w + self.b


class TransformerBlock:
    def __init__(self, d, nh, d_ff, rng):
        self.nh = nh
        self.dh = d // nh
        self.wq = Linear(d, d, rng)
        self.wk = Linear(d, d, rng)
        self.wv = Linear(d, d, rng)
        self.wo = Linear(d, d, rng)
        self.ff1 = Linear(d, d_ff, rng)
        self.ff2 = Linear(d_ff, d, rng)
        self.ln1 = LayerNorm(d)
        self.ln2 = LayerNorm(d)

    def __call__(self, x, mask_add=None):
        S = x.shape[0]
        nh, dh = self.nh, self.dh

        Q = self.wq(x).reshape(S, nh, dh).transpose(1, 0, 2)
        K = self.wk(x).reshape(S, nh, dh).transpose(1, 0, 2)
        V = self.wv(x).reshape(S, nh, dh).transpose(1, 0, 2)

        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)
        if mask_add is not None:
            scores = scores + mask_add[None]
        attn = softmax(scores, -1)
        out = (attn @ V).transpose(1, 0, 2).reshape(S, -1)
        out = self.wo(out)

        x = self.ln1(x + out)
        x = self.ln2(x + self.ff2(gelu(self.ff1(x))))
        return x


class MinimalLM:
    """Single-layer transformer language model in pure NumPy."""

    def __init__(self, vocab_size=1000, d=64, nh=4, d_ff=256, seed=42):
        rng = np.random.RandomState(seed)
        self.V = vocab_size
        self.emb = rng.randn(vocab_size, d) * 0.02
        self.pe = sinusoidal_pe(512, d)
        self.block = TransformerBlock(d, nh, d_ff, rng)
        self.ln_f = LayerNorm(d)
        self.head = Linear(d, vocab_size, rng)

    def forward(self, tokens, mask_add=None):
        x = self.emb[tokens] + self.pe[:len(tokens)]
        x = self.block(x, mask_add)
        x = self.ln_f(x)
        return self.head(x)

    def greedy_generate(self, prompt, n):
        toks = list(prompt)
        for _ in range(n):
            logits = self.forward(toks)
            toks.append(int(np.argmax(logits[-1])))
        return toks


# ── Mask builders ──────────────────────────────────────────────────

def build_causal_mask(L):
    """Standard causal (lower-triangular) additive attention mask."""
    return np.where(np.tril(np.ones((L, L))), 0.0, -np.inf)


def build_tree_mask(P, tree_parents):
    """
    Build tree attention mask for DFlash verification.

    Args:
        P: number of prompt tokens
        tree_parents: list of parent index per tree node (-1 for roots)
    Returns:
        additive mask of shape (P+N, P+N) with N = len(tree_parents).
        0.0 = attend, -inf = blocked.

    Rules (from spec):
      a) Prompt tokens attend causally to each other.
      b) All tree nodes attend to ALL prompt tokens.
      c) Every position attends to itself.
      d) Each tree node attends to its ancestors in the tree.
      e) No attendance to siblings, cousins, or other branches.
    """
    N = len(tree_parents)
    T = P + N
    m = np.zeros((T, T), dtype=bool)

    for i in range(P):
        m[i, : i + 1] = True

    m[P:, :P] = True
    np.fill_diagonal(m, True)

    for i in range(N):
        a = tree_parents[i]
        while a != -1:
            m[P + i, P + a] = True
            a = tree_parents[a]

    return np.where(m, 0.0, -np.inf)


# ── Verification / acceptance ─────────────────────────────────────

def _ancestors(i, tree_parents):
    out = []
    c = tree_parents[i]
    while c != -1:
        out.append(c)
        c = tree_parents[c]
    return out


def verify_and_accept(prompt_tokens, tree_tokens, tree_parents, model,
                      temperature=0):
    """
    Run one tree-verification cycle at the given temperature.

    Accepted-path algorithm
    ───────────────────────
    We follow ONE path through the tree (the one whose tokens match the
    target model's greedy predictions).  Processing order is topological.

    * A node whose parent is the current path-end is "on the path".
    * Accept on-path  → extend path, continue.
    * Reject on-path  → emit target prediction, STOP cycle.
    * Reject off-path → mark rejected (descendants skipped by rule 4a).
    * Accept off-path → mark accepted (no effect on output).
    * After all nodes: emit a bonus token from the last path position.

    Returns list of tokens to append to the generated sequence.
    """
    P = len(prompt_tokens)
    N = len(tree_tokens)
    full = list(prompt_tokens) + list(tree_tokens)
    mask = build_tree_mask(P, tree_parents)
    logits = model.forward(full, mask)

    accepted = []
    path_end = -1
    rejected = set()

    for i in range(N):
        if any(a in rejected for a in _ancestors(i, tree_parents)):
            rejected.add(i)
            continue

        parent = tree_parents[i]
        logit_pos = (P - 1) if parent == -1 else (P + parent)
        target_pred = int(np.argmax(logits[logit_pos]))
        on_path = parent == path_end

        if tree_tokens[i] == target_pred:
            if on_path:
                accepted.append(tree_tokens[i])
                path_end = i
        else:
            rejected.add(i)
            if on_path:
                accepted.append(target_pred)
                return accepted

    bonus_pos = (P - 1) if path_end == -1 else (P + path_end)
    accepted.append(int(np.argmax(logits[bonus_pos])))
    return accepted


def _verify_detailed(prompt_tokens, tree_tokens, tree_parents, model):
    """Like verify_and_accept but returns internals for testing."""
    P = len(prompt_tokens)
    N = len(tree_tokens)
    full = list(prompt_tokens) + list(tree_tokens)
    mask = build_tree_mask(P, tree_parents)
    logits = model.forward(full, mask)

    accepted = []
    path_end = -1
    rejected = set()
    skipped_by_ancestor = set()
    decisions = []

    for i in range(N):
        anc = _ancestors(i, tree_parents)
        if any(a in rejected for a in anc):
            rejected.add(i)
            skipped_by_ancestor.add(i)
            decisions.append(("skipped_ancestor", i, anc))
            continue

        parent = tree_parents[i]
        logit_pos = (P - 1) if parent == -1 else (P + parent)
        target_pred = int(np.argmax(logits[logit_pos]))
        on_path = parent == path_end

        if tree_tokens[i] == target_pred:
            if on_path:
                accepted.append(tree_tokens[i])
                path_end = i
                decisions.append(("accepted_path", i, target_pred))
            else:
                decisions.append(("accepted_branch", i, target_pred))
        else:
            rejected.add(i)
            if on_path:
                accepted.append(target_pred)
                decisions.append(("rejected_path", i, target_pred))
                return accepted, rejected, skipped_by_ancestor, decisions
            else:
                decisions.append(("rejected_branch", i, target_pred))

    bonus_pos = (P - 1) if path_end == -1 else (P + path_end)
    accepted.append(int(np.argmax(logits[bonus_pos])))
    return accepted, rejected, skipped_by_ancestor, decisions


def speculative_generate(model, prompt, max_new_tokens, draft_fn):
    """Full generation loop using tree speculative decoding."""
    tokens = list(prompt)
    gen = 0
    while gen < max_new_tokens:
        tt, tp = draft_fn(tokens)
        if not tt:
            logits = model.forward(tokens)
            tokens.append(int(np.argmax(logits[-1])))
            gen += 1
            continue
        acc = verify_and_accept(tokens, tt, tp, model)
        for t in acc:
            if gen >= max_new_tokens:
                break
            tokens.append(t)
            gen += 1
    return tokens


# ── Draft helpers ──────────────────────────────────────────────────

def _make_draft_fn(model, depth=2, n_wrong_branches=2):
    """Draft fn: correct main chain from target + wrong branches off node 0."""

    def draft_fn(current):
        chain = []
        tmp = list(current)
        for _ in range(depth):
            logits = model.forward(tmp)
            chain.append(int(np.argmax(logits[-1])))
            tmp.append(chain[-1])

        tt = [chain[0]]
        tp = [-1]
        for k in range(1, depth):
            tt.append(chain[k])
            tp.append(k - 1)
        for w in range(n_wrong_branches):
            tt.append((chain[0] + 5 + w * 7) % model.V)
            tp.append(0)
        return tt, tp

    return draft_fn


# ── Tests ──────────────────────────────────────────────────────────

def test_tree_mask_correctness():
    """Verify tree mask structure matches spec rules a–e."""
    print("=" * 60)
    print("TEST 0  TREE MASK CORRECTNESS")
    print("=" * 60)

    P = 3
    tree_parents = [-1, 0, 0, 1]
    mask = build_tree_mask(P, tree_parents)
    T = P + len(tree_parents)

    for i in range(P):
        for j in range(P):
            assert (mask[i, j] == 0.0) == (j <= i), \
                f"Rule a) causal broken at ({i},{j})"

    for i in range(P, T):
        for j in range(P):
            assert mask[i, j] == 0.0, \
                f"Rule b) tree node {i} can't attend prompt {j}"

    for i in range(T):
        assert mask[i, i] == 0.0, f"Rule c) self-attention broken at {i}"

    ancestors_of = {0: [], 1: [0], 2: [0], 3: [1, 0]}
    for i in range(len(tree_parents)):
        gi = P + i
        for j in range(len(tree_parents)):
            gj = P + j
            expect = (j in ancestors_of[i]) or (j == i)
            actual = mask[gi, gj] == 0.0
            assert actual == expect, (
                f"Rule d/e) node {i}->node {j}: expected={expect} got={actual}")

    print("  Rules a-e verified on 4-node tree.")
    print("  PASSED\n")


def test_basic():
    """Test 1 (BASIC): prompt=[10,20,30], 3 root nodes, no depth-2, temp=0.
    Must match autoregressive greedy EXACTLY."""
    print("=" * 60)
    print("TEST 1  BASIC — 3 root nodes, temperature=0")
    print("=" * 60)

    model = MinimalLM(seed=42)
    prompt = [10, 20, 30]
    ref = model.greedy_generate(prompt, 6)

    logits0 = model.forward(prompt)
    t0 = int(np.argmax(logits0[-1]))

    tree_tokens = [t0, (t0 + 5) % 1000, (t0 + 10) % 1000]
    tree_parents = [-1, -1, -1]

    acc = verify_and_accept(prompt, tree_tokens, tree_parents, model)
    print(f"  prompt         = {prompt}")
    print(f"  tree_tokens    = {tree_tokens}")
    print(f"  tree_parents   = {tree_parents}")
    print(f"  accepted       = {acc}")
    print(f"  autoregressive = {ref}")

    assert acc == ref[len(prompt): len(prompt) + len(acc)], \
        f"Single-cycle mismatch"

    def draft_flat(cur):
        lg = model.forward(cur)
        tk = int(np.argmax(lg[-1]))
        return [tk, (tk + 5) % 1000, (tk + 10) % 1000], [-1, -1, -1]

    spec = speculative_generate(model, prompt, 6, draft_flat)
    assert spec == ref, f"MISMATCH\n  spec={spec}\n  ref ={ref}"
    print(f"  speculative    = {spec}")
    print("  PASSED\n")


def test_subtree_invalidation():
    """Test 2 (SUBTREE INVALIDATION):
    A depth-1 node is REJECTED, and its depth-2 child WOULD have matched
    the target model's prediction, but is correctly SKIPPED by rule 4a.

    Tree layout:
        root0 (accepted) ── child0 (on main chain)
          └─ root1 (rejected) ── child1 (would match, but skipped)

    We verify:
      1. child1's token matches what the target would predict via root1.
      2. child1 is in the skipped_by_ancestor set.
      3. Output matches autoregressive greedy.
    """
    print("=" * 60)
    print("TEST 2  SUBTREE INVALIDATION")
    print("=" * 60)

    tested_configs = []

    for seed, prompt, wrong_offset in [
        (42, [10, 20, 30], 5),
        (99, [5, 15, 25], 7),
        (7, [100, 200, 300], 13),
        (314, [42], 9),
    ]:
        model = MinimalLM(seed=seed)
        P = len(prompt)

        logits0 = model.forward(prompt)
        t0 = int(np.argmax(logits0[-1]))
        wrong_root = (t0 + wrong_offset) % model.V

        logits_t0 = model.forward(prompt + [t0])
        t1 = int(np.argmax(logits_t0[-1]))

        dummy_tt = [t0, t1, wrong_root, 0]
        dummy_tp = [-1, 0, 0, 2]
        dummy_mask = build_tree_mask(P, dummy_tp)
        dummy_logits = model.forward(prompt + dummy_tt, dummy_mask)

        t1_given_wrong = int(np.argmax(dummy_logits[P + 2]))

        tree_tokens = [t0, t1, wrong_root, t1_given_wrong]
        tree_parents = [-1, 0, 0, 2]

        acc, rejected, skipped, decisions = _verify_detailed(
            prompt, tree_tokens, tree_parents, model)

        ref = model.greedy_generate(prompt, len(acc))

        assert acc == ref[P: P + len(acc)], (
            f"seed={seed} output mismatch: acc={acc} ref={ref[P:]}")

        assert 2 in rejected, f"seed={seed}: root1 (node 2) not rejected"
        assert 3 in skipped, (
            f"seed={seed}: child1 (node 3) not skipped by ancestor")

        assert tree_tokens[3] == t1_given_wrong, "construction error"
        parent_of_3 = tree_parents[3]
        logit_pos_3 = (P - 1) if parent_of_3 == -1 else (P + parent_of_3)
        would_match = tree_tokens[3] == int(np.argmax(dummy_logits[logit_pos_3]))

        print(f"  seed={seed:3d}  prompt={prompt}")
        print(f"    t0={t0}  wrong_root={wrong_root}  t1={t1}  "
              f"child_of_wrong={t1_given_wrong}")
        print(f"    node3 would match target: {would_match}")
        print(f"    node3 skipped by ancestor: {3 in skipped}")
        print(f"    output matches autoregressive: True")

        tested_configs.append(seed)

    print(f"\n  Tested {len(tested_configs)} configs: {tested_configs}")
    print("  PASSED\n")


def test_multi_step():
    """Test 3 (MULTI-STEP): 3+ consecutive verification cycles.
    Accepted tokens from cycle N become the prompt for cycle N+1."""
    print("=" * 60)
    print("TEST 3  MULTI-STEP (3+ verification cycles)")
    print("=" * 60)

    prompt = [10, 20, 30]
    n_tokens = 10

    for seed in [42, 7, 123, 999, 0]:
        model = MinimalLM(seed=seed)
        ref = model.greedy_generate(prompt, n_tokens)
        spec = speculative_generate(model, prompt, n_tokens,
                                    _make_draft_fn(model, depth=2))
        assert spec == ref, (
            f"seed={seed} MISMATCH\n  spec={spec}\n  ref ={ref}")
        print(f"  seed={seed:3d}  match=True  "
              f"tokens={ref[len(prompt):len(prompt)+6]}...")

    print("  PASSED\n")


def test_golden():
    """THE GOLDEN TEST: speculative == autoregressive for many configs.
    At temperature=0, tree speculative decoding MUST produce EXACTLY
    the same output sequence as autoregressive greedy decoding."""
    print("=" * 60)
    print("GOLDEN TEST")
    print("=" * 60)

    prompts = [[10, 20, 30], [1], [100, 200], list(range(5, 15))]
    seeds = [42, 7, 123, 0, 999]
    depths = [1, 2, 3]
    n_configs = 0
    fails = []

    for seed in seeds:
        model = MinimalLM(seed=seed)
        for prompt in prompts:
            for depth in depths:
                ref = model.greedy_generate(prompt, 12)
                draft_fn = _make_draft_fn(model, depth=depth,
                                          n_wrong_branches=depth)
                spec = speculative_generate(model, prompt, 12, draft_fn)
                n_configs += 1
                if spec != ref:
                    fails.append((seed, prompt[:3], depth))

    if fails:
        for s, p, d in fails:
            print(f"  FAIL  seed={s} prompt={p}.. depth={d}")
        assert False, f"{len(fails)}/{n_configs} configs FAILED"
    else:
        print(f"  {n_configs} configurations: ALL PASSED")

    print("  GOLDEN TEST PASSED\n")


if __name__ == "__main__":
    test_tree_mask_correctness()
    test_basic()
    test_subtree_invalidation()
    test_multi_step()
    test_golden()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
