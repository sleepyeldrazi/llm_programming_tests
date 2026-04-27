import numpy as np


def log_softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


class MinimalTransformer:
    def __init__(self, vocab_size=1000, d_model=64, seed=42):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        s = 0.1
        self.embed = rng.standard_normal((vocab_size, d_model)) * s
        self.Wq = rng.standard_normal((d_model, d_model)) * s
        self.Wk = rng.standard_normal((d_model, d_model)) * s
        self.Wv = rng.standard_normal((d_model, d_model)) * s
        self.Wo = rng.standard_normal((d_model, d_model)) * s
        self.ln1_g = np.ones(d_model)
        self.ln1_b = np.zeros(d_model)
        self.ln2_g = np.ones(d_model)
        self.ln2_b = np.zeros(d_model)
        d_ff = d_model * 4
        self.W1 = rng.standard_normal((d_model, d_ff)) * s
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.standard_normal((d_ff, d_model)) * s
        self.b2 = np.zeros(d_model)
        self.Wout = rng.standard_normal((d_model, vocab_size)) * s
        self.bout = np.zeros(vocab_size)

    def _layer_norm(self, x, g, b):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return g * (x - mu) / np.sqrt(var + 1e-5) + b

    def _softmax(self, x, axis=-1):
        m = x.max(axis=axis, keepdims=True)
        e = np.exp(x - m)
        return e / e.sum(axis=axis, keepdims=True)

    def forward(self, token_ids):
        x = self.embed[token_ids]

        h = self._layer_norm(x, self.ln1_g, self.ln1_b)
        Q, K, V = h @ self.Wq, h @ self.Wk, h @ self.Wv
        seq_len = token_ids.shape[-1]
        scores = Q @ np.swapaxes(K, -2, -1) / np.sqrt(self.d_model)
        causal = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores = scores + causal
        x = x + self._softmax(scores, axis=-1) @ V @ self.Wo

        h = self._layer_norm(x, self.ln2_g, self.ln2_b)
        x = x + np.maximum(0, h @ self.W1 + self.b1) @ self.W2 + self.b2

        return x[..., -1, :] @ self.Wout + self.bout


def batched_beam_search(model, prompt_token_ids, beam_width, max_new_tokens,
                        alpha=0.6, eos_token_id=2):
    """
    Batched beam search decoder.

    Args:
        model: object with forward(token_ids: ndarray) -> logits ndarray
        prompt_token_ids: list[list[int]], one prompt per batch item
        beam_width: int K, beams per batch item
        max_new_tokens: int, max generation steps
        alpha: float, length penalty exponent (default 0.6)
        eos_token_id: int, end-of-sequence token

    Returns:
        list of list of (generated_tokens: list[int], score: float),
        one inner list per batch item, sorted by length-penalized score
        descending (best first).
    """

    def penalized_score(acc_lp, gen_len):
        if gen_len == 0:
            return acc_lp
        return acc_lp / (gen_len ** alpha)

    all_results = []

    for prompt in prompt_token_ids:
        prompt = list(prompt)
        K = beam_width
        beams = [([], 0.0, False)]

        for _ in range(max_new_tokens):
            finished = [b for b in beams if b[2]]
            unfinished = [b for b in beams if not b[2]]

            if not unfinished:
                break

            seqs = np.array(
                [prompt + b[0] for b in unfinished], dtype=np.int64
            )
            lp = log_softmax(model.forward(seqs), axis=-1)

            candidates = []
            for i, (toks, acc, _) in enumerate(unfinished):
                top2k = np.argsort(lp[i])[-(2 * K):]
                for tid in top2k:
                    tid = int(tid)
                    candidates.append((
                        toks + [tid],
                        acc + float(lp[i, tid]),
                        tid == eos_token_id,
                    ))

            # FINISHED BEAMS MUST REMAIN IN THE POOL.
            #
            # If we removed them, a short high-confidence sequence that
            # produced EOS early (e.g. logprob=-3.0 at length 1) would be
            # discarded. A longer, lower-confidence sequence (e.g. logprob=-5.0
            # at length 2) would then wrongly win. Keeping finished beams in
            # the pool ensures they compete on equal footing via their
            # length-penalized scores, so the best sequence is always selected
            # regardless of when it finished.
            pool = candidates + finished
            pool.sort(
                key=lambda b: penalized_score(b[1], len(b[0])),
                reverse=True,
            )
            beams = pool[:K]

        beams.sort(
            key=lambda b: penalized_score(b[1], len(b[0])),
            reverse=True,
        )
        all_results.append([
            (b[0], penalized_score(b[1], len(b[0]))) for b in beams
        ])

    return all_results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_logits(desired_logprobs, vocab_size):
    """Build logits whose log_softmax yields *exactly* the desired logprobs.

    Remaining probability mass is spread uniformly over all other tokens so
    that total probability sums to 1.
    """
    spec_prob = sum(np.exp(lp) for lp in desired_logprobs.values())
    remaining = 1.0 - spec_prob
    n_other = vocab_size - len(desired_logprobs)
    other_lp = (
        np.log(remaining / n_other)
        if n_other > 0 and remaining > 0
        else -100.0
    )
    logits = np.full(vocab_size, other_lp)
    for tid, lp in desired_logprobs.items():
        logits[tid] = lp
    return logits


class _MockModel:
    def __init__(self, logits_schedule, vocab_size):
        self.logits_schedule = logits_schedule
        self.vocab_size = vocab_size
        self._call = 0

    def forward(self, token_ids):
        out = self.logits_schedule[self._call]
        self._call += 1
        if out.ndim == 1:
            out = np.broadcast_to(
                out, (token_ids.shape[0], self.vocab_size)
            ).copy()
        return out


def test_greedy():
    """Test 1: K=1 with alpha=0 must behave identically to greedy decoding."""
    model = MinimalTransformer(seed=42)
    prompt = [[10, 20, 30]]

    result = batched_beam_search(
        model, prompt, beam_width=1, max_new_tokens=10, alpha=0.0
    )

    toks = list(prompt[0])
    greedy = []
    for _ in range(10):
        logits = model.forward(np.array([toks], dtype=np.int64))
        t = int(np.argmax(logits[0]))
        greedy.append(t)
        toks.append(t)
        if t == 2:
            break

    assert result[0][0][0] == greedy, (
        f"K=1 beam search differs from greedy:\n"
        f"  beam  = {result[0][0][0]}\n"
        f"  greedy= {greedy}"
    )
    print("Test 1 PASSED: K=1 beam search matches greedy decoding")


def test_batch_independence():
    """Test 2: beams from different batch items never interact."""
    model = MinimalTransformer(seed=42)
    prompts = [[1, 2, 3], [10, 20, 30, 40, 50]]

    batch = batched_beam_search(
        model, prompts, beam_width=3, max_new_tokens=10, alpha=0.6
    )
    solo0 = batched_beam_search(
        model, [prompts[0]], beam_width=3, max_new_tokens=10, alpha=0.6
    )
    solo1 = batched_beam_search(
        model, [prompts[1]], beam_width=3, max_new_tokens=10, alpha=0.6
    )

    for i in range(len(batch[0])):
        assert batch[0][i][0] == solo0[0][i][0], (
            f"Batch item 0 beam {i} tokens differ"
        )
        assert abs(batch[0][i][1] - solo0[0][i][1]) < 1e-10, (
            f"Batch item 0 beam {i} scores differ"
        )
    for i in range(len(batch[1])):
        assert batch[1][i][0] == solo1[0][i][0], (
            f"Batch item 1 beam {i} tokens differ"
        )
        assert abs(batch[1][i][1] - solo1[0][i][1]) < 1e-10, (
            f"Batch item 1 beam {i} scores differ"
        )

    print("Test 2 PASSED: per-batch independence verified")


def test_eos_retention():
    """Test 3: finished beams must stay in the pool and can win.

    Step 0: EOS beam gets total logprob -3.0; another beam continues at -4.0.
    Step 1: continuing beam reaches total -5.0.
    The EOS beam (score -3.0) must beat the continuing beam (score -5.0).
    A buggy implementation that removes finished beams would wrongly pick -5.0.
    """
    V = 1000
    EOS = 999
    OTHER = 42
    CONT = 99

    logits_step0 = _make_logits({EOS: -3.0, OTHER: -4.0}, V)
    logits_step1 = _make_logits({CONT: -1.0}, V)

    mock = _MockModel([logits_step0, logits_step1], V)

    result = batched_beam_search(
        mock,
        [[100, 200, 300]],
        beam_width=2,
        max_new_tokens=2,
        alpha=0.0,
        eos_token_id=EOS,
    )

    best_toks, best_score = result[0][0]
    assert EOS in best_toks, (
        f"Winner should contain EOS={EOS}, got {best_toks}"
    )
    assert abs(best_score - (-3.0)) < 0.01, (
        f"Winner score should be ≈ -3.0, got {best_score}"
    )

    _, second_score = result[0][1]
    assert second_score < best_score, (
        f"Second beam ({second_score}) should have worse score than "
        f"EOS beam ({best_score})"
    )

    print("Test 3 PASSED: EOS beam correctly retained and wins")


if __name__ == "__main__":
    test_greedy()
    test_batch_independence()
    test_eos_retention()
    print("\nAll tests passed!")
