import numpy as np


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_sum_exp


def layer_norm(x, weight, bias, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias


def linear(x, weight, bias):
    return x @ weight.T + bias


class MinimalLM:
    def __init__(self, d_model=64, n_heads=4, vocab_size=1000, seed=42):
        rng = np.random.RandomState(seed)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.vocab_size = vocab_size

        scale = 0.02

        self.tok_emb = rng.randn(vocab_size, d_model).astype(np.float32) * scale
        self.pos_emb = rng.randn(2048, d_model).astype(np.float32) * scale

        self.q_w = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.k_w = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.v_w = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.out_w = rng.randn(d_model, d_model).astype(np.float32) * scale

        self.attn_ln_w = np.ones(d_model, dtype=np.float32)
        self.attn_ln_b = np.zeros(d_model, dtype=np.float32)

        self.ff1_w = rng.randn(d_model * 4, d_model).astype(np.float32) * scale
        self.ff1_b = np.zeros(d_model * 4, dtype=np.float32)
        self.ff2_w = rng.randn(d_model, d_model * 4).astype(np.float32) * scale
        self.ff2_b = np.zeros(d_model, dtype=np.float32)

        self.ff_ln_w = np.ones(d_model, dtype=np.float32)
        self.ff_ln_b = np.zeros(d_model, dtype=np.float32)

        self.lm_head_w = rng.randn(vocab_size, d_model).astype(np.float32) * scale
        self.lm_head_b = np.zeros(vocab_size, dtype=np.float32)

    def forward(self, token_ids, mask_add):
        seq_len = len(token_ids)
        positions = np.arange(seq_len)

        x = self.tok_emb[token_ids] + self.pos_emb[positions]

        residual = x
        x_ln = layer_norm(x, self.attn_ln_w, self.attn_ln_b)

        Q = linear(x_ln, self.q_w, np.zeros(self.d_model, dtype=np.float32))
        K = linear(x_ln, self.k_w, np.zeros(self.d_model, dtype=np.float32))
        V = linear(x_ln, self.v_w, np.zeros(self.d_model, dtype=np.float32))

        Q = Q.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)

        scale_factor = 1.0 / np.sqrt(self.d_head)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) * scale_factor

        scores = scores + mask_add[np.newaxis, :, :]

        attn_weights = softmax(scores, axis=-1)
        attn_out = np.matmul(attn_weights, V)

        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        attn_out = linear(attn_out, self.out_w, np.zeros(self.d_model, dtype=np.float32))

        x = residual + attn_out

        residual = x
        x_ln = layer_norm(x, self.ff_ln_w, self.ff_ln_b)
        h = linear(x_ln, self.ff1_w, self.ff1_b)
        h = np.maximum(h, 0)
        h = linear(h, self.ff2_w, self.ff2_b)
        x = residual + h

        logits = linear(x, self.lm_head_w, self.lm_head_b)
        return logits


def build_tree_mask(prompt_len, tree_parents):
    n_tree = len(tree_parents)
    total = prompt_len + n_tree

    mask = np.zeros((total, total), dtype=bool)

    for i in range(prompt_len):
        for j in range(i + 1):
            mask[i, j] = True

    for i in range(prompt_len, total):
        for j in range(prompt_len):
            mask[i, j] = True

    for i in range(n_tree):
        global_i = prompt_len + i
        mask[global_i, global_i] = True
        parent = tree_parents[i]
        while parent != -1:
            global_parent = prompt_len + parent
            mask[global_i, global_parent] = True
            parent = tree_parents[parent]

    mask_add = np.where(mask, 0.0, -np.inf).astype(np.float32)
    return mask_add


def build_causal_mask(seq_len):
    mask = np.zeros((seq_len, seq_len), dtype=bool)
    for i in range(seq_len):
        for j in range(i + 1):
            mask[i, j] = True
    return np.where(mask, 0.0, -np.inf).astype(np.float32)


def get_ancestors(node_idx, tree_parents):
    ancestors = []
    parent = tree_parents[node_idx]
    while parent != -1:
        ancestors.append(parent)
        parent = tree_parents[parent]
    return ancestors


def verify_and_accept(prompt_tokens, tree_tokens, tree_parents, target_model, temperature=0):
    prompt_len = len(prompt_tokens)
    full_tokens = list(prompt_tokens) + list(tree_tokens)
    full_ids = np.array(full_tokens, dtype=np.int64)

    mask_add = build_tree_mask(prompt_len, tree_parents)
    logits = target_model.forward(full_ids, mask_add)

    n_tree = len(tree_tokens)
    accepted = []
    rejected_ancestors = set()

    for i in range(n_tree):
        ancestors = get_ancestors(i, tree_parents)
        ancestor_rejected = any(a in rejected_ancestors for a in ancestors)

        if ancestor_rejected:
            rejected_ancestors.add(i)
            continue

        if tree_parents[i] == -1:
            parent_logit_idx = prompt_len - 1
        else:
            parent_logit_idx = prompt_len + tree_parents[i]

        log_probs = log_softmax(logits[parent_logit_idx])
        target_greedy = int(np.argmax(log_probs))

        if temperature == 0:
            if tree_tokens[i] == target_greedy:
                accepted.append(tree_tokens[i])
            else:
                accepted.append(target_greedy)
                rejected_ancestors.add(i)
                break

    if not accepted:
        causal_mask = build_causal_mask(prompt_len)
        prompt_logits = target_model.forward(np.array(prompt_tokens, dtype=np.int64), causal_mask)
        new_token = int(np.argmax(prompt_logits[-1]))
        accepted = [new_token]

    return accepted


def autoregressive_greedy(model, prompt_tokens, max_tokens):
    tokens = list(prompt_tokens)
    while len(tokens) < max_tokens:
        mask = build_causal_mask(len(tokens))
        logits = model.forward(np.array(tokens, dtype=np.int64), mask)
        next_token = int(np.argmax(logits[-1]))
        tokens.append(next_token)
    return tokens


def speculative_generate(model, prompt_tokens, draft_fn, max_tokens, temperature=0):
    generated = list(prompt_tokens)

    while len(generated) < max_tokens:
        tree_tokens, tree_parents = draft_fn(generated)

        if len(tree_tokens) == 0:
            causal_mask = build_causal_mask(len(generated))
            logits = model.forward(np.array(generated, dtype=np.int64), causal_mask)
            next_token = int(np.argmax(logits[-1]))
            generated.append(next_token)
            continue

        accepted = verify_and_accept(
            generated, tree_tokens, tree_parents, model, temperature
        )

        generated.extend(accepted)

    return generated[:max_tokens]


def run_all_tests():
    print("=" * 60)
    print("TEST 1: BASIC - 3 root nodes (no depth-2)")
    print("=" * 60)

    np.random.seed(42)
    model = MinimalLM(d_model=64, n_heads=4, vocab_size=100, seed=42)

    prompt = [10, 20, 30]
    tree_tokens = [50, 60, 70]
    tree_parents = [-1, -1, -1]

    ar_result = autoregressive_greedy(model, prompt, max_tokens=6)
    print(f"Autoregressive tokens after prompt: {ar_result[3:]}")

    accepted = verify_and_accept(prompt, tree_tokens, tree_parents, model, temperature=0)
    spec_result = list(prompt) + accepted
    print(f"Accepted tokens: {accepted}")
    print(f"Speculative result: {spec_result}")

    assert spec_result == ar_result[:len(spec_result)], \
        f"MISMATCH: {spec_result} != {ar_result[:len(spec_result)]}"
    print("TEST 1 PASSED\n")

    print("=" * 60)
    print("TEST 2: SUBTREE INVALIDATION")
    print("=" * 60)

    np.random.seed(42)
    model2 = MinimalLM(d_model=64, n_heads=4, vocab_size=100, seed=42)

    prompt2 = [5, 15, 25]
    ar_result2 = autoregressive_greedy(model2, prompt2, max_tokens=10)
    print(f"Autoregressive result: {ar_result2}")

    causal_mask2 = build_causal_mask(len(prompt2))
    logits_p2 = model2.forward(np.array(prompt2, dtype=np.int64), causal_mask2)
    token_after_prompt = int(np.argmax(logits_p2[-1]))
    print(f"Target predicts after prompt: {token_after_prompt}")

    next_input = list(prompt2) + [token_after_prompt]
    causal_mask_next = build_causal_mask(len(next_input))
    logits_next = model2.forward(np.array(next_input, dtype=np.int64), causal_mask_next)
    token_after_accepted = int(np.argmax(logits_next[-1]))
    print(f"Target predicts after {token_after_prompt}: {token_after_accepted}")

    next_input2 = next_input + [token_after_accepted]
    causal_mask_next2 = build_causal_mask(len(next_input2))
    logits_next2 = model2.forward(np.array(next_input2, dtype=np.int64), causal_mask_next2)
    token_after_2 = int(np.argmax(logits_next2[-1]))
    print(f"Target predicts after {token_after_prompt}, {token_after_accepted}: {token_after_2}")

    wrong_token = token_after_accepted + 1
    if wrong_token >= 100:
        wrong_token = token_after_accepted - 1

    tree_tokens2 = [token_after_prompt, wrong_token, token_after_2]
    tree_parents2 = [-1, 0, 1]

    print(f"Tree tokens: {tree_tokens2}")
    print(f"Tree parents: {tree_parents2}")
    print(f"Node 0 (root): draft={tree_tokens2[0]}, should be accepted (matches target)")
    print(f"Node 1 (child of 0): draft={tree_tokens2[1]}, should be REJECTED (wrong token)")
    print(f"Node 2 (child of 1): draft={tree_tokens2[2]}, would match target but should be SKIPPED")

    accepted2 = verify_and_accept(prompt2, tree_tokens2, tree_parents2, model2, temperature=0)
    spec_result2 = list(prompt2) + accepted2
    print(f"Accepted tokens: {accepted2}")
    print(f"Speculative result: {spec_result2}")

    assert len(accepted2) == 2, \
        f"Expected 2 tokens (accepted root + rejection correction), got {len(accepted2)}"
    assert accepted2[0] == token_after_prompt, \
        f"First token should be {token_after_prompt}, got {accepted2[0]}"
    assert accepted2[1] == token_after_accepted, \
        f"Second token should be {token_after_accepted} (correction), got {accepted2[1]}"

    assert spec_result2 == ar_result2[:len(spec_result2)], \
        f"MISMATCH: {spec_result2} != {ar_result2[:len(spec_result2)]}"
    print("TEST 2 PASSED\n")

    print("=" * 60)
    print("TEST 3: MULTI-STEP - 3 consecutive verification cycles")
    print("=" * 60)

    np.random.seed(42)
    model3 = MinimalLM(d_model=64, n_heads=4, vocab_size=100, seed=42)

    prompt3 = [10, 20, 30]
    max_tokens = 12
    ar_result3 = autoregressive_greedy(model3, prompt3, max_tokens=max_tokens)
    print(f"Autoregressive result: {ar_result3}")

    def make_draft_fn(cycles):
        idx = [0]

        def draft_fn(generated):
            if idx[0] >= len(cycles):
                return [], []
            tt, tp = cycles[idx[0]]
            idx[0] += 1
            return tt, tp

        return draft_fn

    def autoregressive_draft(model, prompt_tokens, num_draft=3):
        tokens = list(prompt_tokens)
        draft_tokens = []
        draft_parents = []

        for i in range(num_draft):
            mask = build_causal_mask(len(tokens))
            logits = model.forward(np.array(tokens, dtype=np.int64), mask)
            next_tok = int(np.argmax(logits[-1]))
            draft_tokens.append(next_tok)
            if i == 0:
                draft_parents.append(-1)
            else:
                draft_parents.append(i - 1)
            tokens.append(next_tok)

        return draft_tokens, draft_parents

    generated3 = list(prompt3)
    cycle_drafts = []

    for step in range(5):
        if len(generated3) >= max_tokens:
            break
        tt, tp = autoregressive_draft(model3, generated3, num_draft=3)
        cycle_drafts.append((tt, tp))
        accepted3 = verify_and_accept(generated3, tt, tp, model3, temperature=0)
        generated3.extend(accepted3)

    spec_result3 = generated3[:max_tokens]
    print(f"Speculative result: {spec_result3}")

    assert spec_result3 == ar_result3, \
        f"MISMATCH:\n  speculative: {spec_result3}\n  autoregressive: {ar_result3}"
    print("TEST 3 PASSED\n")

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()