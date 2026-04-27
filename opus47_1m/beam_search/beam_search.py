"""Batched beam search decoder for autoregressive generation in pure NumPy."""

import numpy as np


def log_softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    shifted = x - m
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


class TinyLM:
    """Random-weight 1-block transformer. Correctness of decoding is the
    point — the model itself produces meaningless logits."""

    def __init__(self, vocab_size=1000, d_model=64, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        s = 1.0 / np.sqrt(d_model)
        self.embed = rng.standard_normal((vocab_size, d_model)) * s
        self.Wq = rng.standard_normal((d_model, d_model)) * s
        self.Wk = rng.standard_normal((d_model, d_model)) * s
        self.Wv = rng.standard_normal((d_model, d_model)) * s
        self.Wo = rng.standard_normal((d_model, d_model)) * s
        self.W1 = rng.standard_normal((d_model, 4 * d_model)) * s
        self.W2 = rng.standard_normal((4 * d_model, d_model)) * s
        self.lm_head = rng.standard_normal((d_model, vocab_size)) * s

    def forward(self, token_ids):
        # token_ids: (N, T) -> last-position logits (N, V)
        x = self.embed[token_ids]
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_model)
        T = scores.shape[-1]
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
        attn = np.exp(scores - scores.max(-1, keepdims=True))
        attn = attn / attn.sum(-1, keepdims=True)
        h = (attn @ V) @ self.Wo
        x = x + h
        h2 = np.maximum(0, x @ self.W1) @ self.W2
        x = x + h2
        return x[:, -1, :] @ self.lm_head


def batched_beam_search(model, prompts, beam_width, max_new_tokens,
                        alpha=0.6, eos_token_id=0):
    """Beam search over multiple prompts, returning K best generations each.

    Returns: list of length B; each element is a list of up to K dicts
        {tokens, score, logprob, finished} sorted by length-penalized
        score descending.

    Why finished beams are NOT removed from the pool:
        A beam that hits EOS early may have a high length-penalized score
        (its short length means a small denominator). If we drop it from
        the candidate pool the moment it finishes, an unfinished beam
        with worse cumulative logprob can win simply because we never let
        the finished beam compete. Keeping finished beams in the pool —
        and ranking by length-penalized score — lets early-stoppers
        legitimately defend their lead. (See test_eos_retention.)
    """
    K = beam_width
    B = len(prompts)

    # Per batch item: list of beam dicts with
    #   tokens   : full token list (prompt + generated)
    #   gen      : generated-only token list (prompt does NOT count)
    #   logprob  : raw accumulated logprob (never modified by length penalty)
    #   finished : True iff this beam has emitted EOS
    state = [[{
        "tokens": list(p),
        "gen": [],
        "logprob": 0.0,
        "finished": False,
    }] for p in prompts]

    def lp_score(b):
        L = len(b["gen"])
        if L == 0:
            # Only the initial beam has L=0; never compared against others.
            return b["logprob"]
        return b["logprob"] / (L ** alpha)

    for _ in range(max_new_tokens):
        # Stop early if every batch item already holds K finished beams.
        if all(len(beams) >= K and all(b["finished"] for b in beams)
               for beams in state):
            break

        # Gather every unfinished beam across all batch items.
        active = []  # (batch_idx, beam_idx)
        for bi, beams in enumerate(state):
            for ki, b in enumerate(beams):
                if not b["finished"]:
                    active.append((bi, ki))
        if not active:
            break

        # One forward call per active beam. Lengths can differ across
        # batches, so per-beam calls keep this simple and correct.
        active_logps = []
        for (bi, ki) in active:
            tokens = state[bi][ki]["tokens"]
            arr = np.array([tokens], dtype=np.int64)
            logits = model.forward(arr)[0]  # (V,)
            active_logps.append(log_softmax(logits))

        # For each batch item, build the candidate pool and pick top K.
        for bi in range(B):
            beams = state[bi]

            pool = []

            # Carry finished beams forward — they MUST stay eligible for
            # selection so they compete against new candidates by
            # length-penalized score. See module docstring on why.
            for b in beams:
                if b["finished"]:
                    pool.append(b)

            # Expand each unfinished beam with its top-2K next-token
            # candidates (2K, not K, preserves diversity).
            for active_idx, (abi, aki) in enumerate(active):
                if abi != bi:
                    continue
                b = beams[aki]
                lp = active_logps[active_idx]  # (V,)
                m = min(2 * K, lp.shape[0])
                # argpartition gives unsorted top-m; that's fine because we
                # re-sort the whole pool below.
                top_idx = np.argpartition(-lp, m - 1)[:m]
                for tok_idx in top_idx:
                    tok = int(tok_idx)
                    new_logprob = b["logprob"] + float(lp[tok])
                    pool.append({
                        "tokens": b["tokens"] + [tok],
                        "gen": b["gen"] + [tok],
                        "logprob": new_logprob,
                        "finished": (tok == eos_token_id),
                    })

            # Rank all pool entries (finished + new candidates) by the
            # length-penalized score and keep the top K.
            pool.sort(key=lp_score, reverse=True)
            state[bi] = pool[:K]

    # Final result: sort once more, return generated tokens only.
    results = []
    for beams in state:
        beams_sorted = sorted(beams, key=lp_score, reverse=True)
        results.append([
            {
                "tokens": b["gen"],
                "score": lp_score(b),
                "logprob": b["logprob"],
                "finished": b["finished"],
            }
            for b in beams_sorted
        ])
    return results


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def greedy_decode(model, prompt, max_new_tokens, eos_token_id):
    tokens = list(prompt)
    gen = []
    for _ in range(max_new_tokens):
        arr = np.array([tokens], dtype=np.int64)
        logits = model.forward(arr)[0]
        tok = int(np.argmax(logits))
        tokens.append(tok)
        gen.append(tok)
        if tok == eos_token_id:
            break
    return gen


def test_greedy_equivalence():
    """Test 1: K=1, alpha=0 must equal greedy decoding."""
    model = TinyLM(seed=42)
    prompt = [3, 14, 159]
    eos = 0
    max_new = 12

    greedy = greedy_decode(model, prompt, max_new, eos)
    beam_results = batched_beam_search(
        model, [prompt], beam_width=1, max_new_tokens=max_new,
        alpha=0.0, eos_token_id=eos,
    )
    beam_tokens = beam_results[0][0]["tokens"]

    assert beam_tokens == greedy, (
        f"Beam (K=1, alpha=0) diverged from greedy:\n"
        f"  greedy = {greedy}\n  beam   = {beam_tokens}"
    )
    print(f"Test 1 OK — greedy == beam(K=1, alpha=0): {greedy}")


def test_per_batch_independence():
    """Test 2: beams from one prompt must not affect another prompt's
    results. Run prompt-0 alone vs in a batch with prompt-1; the
    prompt-0 result must be identical."""
    model = TinyLM(seed=7)
    p0 = [11, 22, 33]
    p1 = [44, 55, 66, 77, 88]
    eos = 0
    K = 3
    max_new = 8

    solo = batched_beam_search(
        model, [p0], beam_width=K, max_new_tokens=max_new,
        alpha=0.6, eos_token_id=eos,
    )[0]
    together = batched_beam_search(
        model, [p0, p1], beam_width=K, max_new_tokens=max_new,
        alpha=0.6, eos_token_id=eos,
    )

    assert len(together) == 2
    assert len(together[0]) <= K and len(together[1]) <= K

    solo_seqs = [tuple(b["tokens"]) for b in solo]
    batch_seqs = [tuple(b["tokens"]) for b in together[0]]
    assert solo_seqs == batch_seqs, (
        f"Per-batch independence violated:\n"
        f"  solo      = {solo_seqs}\n  in-batch  = {batch_seqs}"
    )

    # Sanity: prompt-1's beams should be different from prompt-0's.
    other_seqs = [tuple(b["tokens"]) for b in together[1]]
    assert other_seqs != batch_seqs
    print(f"Test 2 OK — prompt-0 results identical solo vs batched (K={K}).")


class _EOSMockModel:
    """Hand-crafted forward pass for the EOS retention test.

    Step 1 (first call): produces logits whose softmax gives
        logp(eos)   = -3.0
        logp(tok 1) = -4.0
        logp(other) ≈ -6.977
    Step 2 (second call): logits whose softmax gives
        logp(tok 1) = -1.0   (the survivor extends with this)
        logp(other) ≈ -7.365 (eos included, so beam stays unfinished)
    """

    def __init__(self, eos_token=0, vocab_size=1000):
        self.eos = eos_token
        self.V = vocab_size
        self.calls = 0

    def forward(self, token_ids):
        N = token_ids.shape[0]
        logits = np.zeros((N, self.V))
        if self.calls == 0:
            # Distribute mass so e^logits sums to ~1, making logits == logp.
            # p(eos)=e^-3, p(1)=e^-4, rest split: (1 - e^-3 - e^-4)/(V-2)
            other_p = (1.0 - np.exp(-3.0) - np.exp(-4.0)) / (self.V - 2)
            other_lp = float(np.log(other_p))
            logits[:, :] = other_lp
            logits[:, self.eos] = -3.0
            logits[:, 1] = -4.0
        else:
            other_p = (1.0 - np.exp(-1.0)) / (self.V - 1)
            other_lp = float(np.log(other_p))
            logits[:, :] = other_lp
            logits[:, 1] = -1.0  # winning continuation
            # eos stays at other_lp ≈ -7.365 → not picked first
        self.calls += 1
        return logits


def test_eos_retention():
    """Test 3: the critical EOS-retention test.

    Step 1: beam A emits EOS → logprob -3, len 1, finished.
            beam B continues with token 1 → logprob -4, len 1.
    Step 2: beam B extends with token 1 again → logprob -5, len 2.

    Length-penalized scores (alpha=0.6):
        A: -3 / 1^0.6 = -3.000
        B: -5 / 2^0.6 ≈ -3.296
    A must win. A buggy implementation that drops finished beams from
    the pool would return B as the top result.
    """
    eos = 0
    K = 2
    max_new = 2
    model = _EOSMockModel(eos_token=eos, vocab_size=1000)

    results = batched_beam_search(
        model, [[42]], beam_width=K, max_new_tokens=max_new,
        alpha=0.6, eos_token_id=eos,
    )
    top = results[0][0]
    runner_up = results[0][1]

    assert top["tokens"] == [eos], (
        f"EOS beam was not the winner. Got tokens={top['tokens']}. "
        f"This indicates finished beams were wrongly dropped from the pool."
    )
    assert top["finished"] is True
    assert abs(top["logprob"] - (-3.0)) < 1e-6, top["logprob"]
    assert abs(top["score"] - (-3.0)) < 1e-6, top["score"]

    assert runner_up["tokens"] == [1, 1]
    assert runner_up["finished"] is False
    assert abs(runner_up["logprob"] - (-5.0)) < 1e-6, runner_up["logprob"]
    expected_runner_score = -5.0 / (2.0 ** 0.6)
    assert abs(runner_up["score"] - expected_runner_score) < 1e-6

    print(
        f"Test 3 OK — EOS beam (score={top['score']:.4f}) correctly beat "
        f"unfinished beam (score={runner_up['score']:.4f})."
    )


if __name__ == "__main__":
    test_greedy_equivalence()
    test_per_batch_independence()
    test_eos_retention()
    print("\nAll tests passed.")
