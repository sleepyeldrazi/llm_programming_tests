import numpy as np
from model import MinimalLM
from beam_search import batched_beam_search, Beam


class MockModel:
    """Model that returns controlled log probs based on input sequence."""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self._callbacks = []

    def set_log_probs(self, token_seq, log_probs):
        """Set log probs to return when given a specific token sequence."""
        self._callbacks.append((tuple(token_seq), log_probs))

    def get_log_probs(self, token_ids):
        key = tuple(token_ids)
        for seq, log_probs in self._callbacks:
            if key == seq:
                return log_probs.copy()
        # Default: uniform (very negative) for all tokens
        default = np.full(self.vocab_size, -1e6, dtype=np.float64)
        return default


def test_greedy_equivalence():
    """Test 1: K=1, alpha=0 should behave identically to greedy decoding."""
    print("=" * 60)
    print("Test 1: Greedy equivalence (K=1, alpha=0)")
    print("=" * 60)

    model = MinimalLM(vocab_size=1000, d_model=64, seed=42)
    prompt = [10, 20, 30]
    eos_token_id = 0
    max_new = 5

    # Beam search with K=1, alpha=0
    beam_results = batched_beam_search(
        prompts=[prompt],
        beam_width=1,
        max_new_tokens=max_new,
        alpha=0.0,
        eos_token_id=eos_token_id,
        model=model,
    )
    beam_seq = beam_results[0][0][0]  # First (and only) batch item, first beam
    beam_score = beam_results[0][0][1]

    # Greedy decoding: always pick argmax at each step
    greedy_seq = []
    greedy_logprob = 0.0
    current = np.array(prompt, dtype=np.int64)
    for _ in range(max_new):
        log_probs = model.get_log_probs(current)
        next_token = int(np.argmax(log_probs))
        greedy_seq.append(next_token)
        greedy_logprob += float(log_probs[next_token])
        current = np.append(current, next_token)
        if next_token == eos_token_id:
            break

    print(f"  Beam search sequence: {beam_seq}")
    print(f"  Beam search score:    {beam_score:.6f}")
    print(f"  Greedy sequence:      {greedy_seq}")
    print(f"  Greedy logprob:       {greedy_logprob:.6f}")

    assert beam_seq == greedy_seq, (
        f"Beam search (K=1, alpha=0) should match greedy! "
        f"beam={beam_seq}, greedy={greedy_seq}"
    )
    assert abs(beam_score - greedy_logprob) < 1e-5, (
        f"Scores should match! beam={beam_score}, greedy={greedy_logprob}"
    )
    print("  PASSED: Beam search with K=1, alpha=0 matches greedy decoding.\n")


def test_batch_independence():
    """Test 2: Per-batch independence with different prompt lengths."""
    print("=" * 60)
    print("Test 2: Batch independence (batch=2, K=3, alpha=0.6)")
    print("=" * 60)

    model = MinimalLM(vocab_size=1000, d_model=64, seed=42)
    prompts = [
        [10, 20, 30],       # Prompt 0: length 3
        [50, 60, 70, 80, 90],  # Prompt 1: length 5
    ]
    beam_width = 3
    eos_token_id = 0
    max_new = 8
    alpha = 0.6

    results = batched_beam_search(
        prompts=prompts,
        beam_width=beam_width,
        max_new_tokens=max_new,
        alpha=alpha,
        eos_token_id=eos_token_id,
        model=model,
    )

    # Verify structure
    assert len(results) == 2, f"Expected 2 batch items, got {len(results)}"
    for i, batch_result in enumerate(results):
        assert len(batch_result) == beam_width, (
            f"Batch {i}: expected {beam_width} beams, got {len(batch_result)}"
        )
        # Verify sorted by score descending
        scores = [s for _, s in batch_result]
        for j in range(len(scores) - 1):
            assert scores[j] >= scores[j + 1], (
                f"Batch {i}: scores not sorted descending! "
                f"{scores[j]} < {scores[j+1]}"
            )
        print(f"  Batch {i}: {len(batch_result)} beams, "
              f"scores={[round(s, 4) for s in scores]}")

    # Verify independence: run each prompt separately and compare
    result0_alone = batched_beam_search(
        prompts=[prompts[0]],
        beam_width=beam_width,
        max_new_tokens=max_new,
        alpha=alpha,
        eos_token_id=eos_token_id,
        model=model,
    )
    result1_alone = batched_beam_search(
        prompts=[prompts[1]],
        beam_width=beam_width,
        max_new_tokens=max_new,
        alpha=alpha,
        eos_token_id=eos_token_id,
        model=model,
    )

    for i in range(beam_width):
        seq_batched, score_batched = results[0][i]
        seq_alone, score_alone = result0_alone[0][i]
        assert seq_batched == seq_alone, (
            f"Prompt 0, beam {i}: batched={seq_batched} != alone={seq_alone}"
        )
        assert abs(score_batched - score_alone) < 1e-6, (
            f"Prompt 0, beam {i}: score mismatch"
        )

    for i in range(beam_width):
        seq_batched, score_batched = results[1][i]
        seq_alone, score_alone = result1_alone[0][i]
        assert seq_batched == seq_alone, (
            f"Prompt 1, beam {i}: batched={seq_batched} != alone={seq_alone}"
        )
        assert abs(score_batched - score_alone) < 1e-6, (
            f"Prompt 1, beam {i}: score mismatch"
        )

    print("  PASSED: Per-batch independence verified. "
          "Beams from prompt 0 never interact with beams from prompt 1.\n")


def test_eos_retention():
    """Test 3: THE EOS RETENTION TEST.

    Monkey-patch the model so that:
    - Step 1: one beam produces EOS with total logprob=-3.0
              another beam continues with logprob=-4.0
    - Step 2: the continuing beam reaches logprob=-5.0

    With alpha=0, the EOS beam (score=-3.0) should win over
    the continuing beam (score=-5.0). If finished beams were
    removed from the pool, the continuing beam would wrongly win.

    This test distinguishes correct implementations from buggy ones
    that discard finished beams.
    """
    print("=" * 60)
    print("Test 3: EOS retention (finished beams must NOT be removed)")
    print("=" * 60)

    vocab_size = 100
    eos_token_id = 1
    continue_token = 2
    next_token = 3
    prompt = [10, 20]

    mock = MockModel(vocab_size=vocab_size)

    # Step 1: given prompt [10, 20], return controlled log probs
    step1_log_probs = np.full(vocab_size, -1e6, dtype=np.float64)
    step1_log_probs[eos_token_id] = -3.0     # EOS: total = -3.0
    step1_log_probs[continue_token] = -4.0    # Continue: total = -4.0
    mock.set_log_probs(prompt, step1_log_probs)

    # Step 2: given prompt + [continue_token], return controlled log probs
    step2_log_probs = np.full(vocab_size, -1e6, dtype=np.float64)
    step2_log_probs[next_token] = -1.0        # total = -4.0 + -1.0 = -5.0
    step2_log_probs[eos_token_id] = -10.0     # total = -4.0 + -10.0 = -14.0
    mock.set_log_probs(prompt + [continue_token], step2_log_probs)

    # Step 3: given prompt + [continue_token, next_token]
    step3_log_probs = np.full(vocab_size, -1e6, dtype=np.float64)
    step3_log_probs[eos_token_id] = -1.0       # total = -5.0 + -1.0 = -6.0
    mock.set_log_probs(prompt + [continue_token, next_token], step3_log_probs)

    beam_width = 2
    alpha = 0.0  # No length penalty for clarity

    results = batched_beam_search(
        prompts=[prompt],
        beam_width=beam_width,
        max_new_tokens=5,
        alpha=alpha,
        eos_token_id=eos_token_id,
        model=mock,
    )

    print(f"  Results (top {beam_width} beams):")
    for i, (seq, score) in enumerate(results[0]):
        status = "FINISHED" if eos_token_id in seq else "unfinished"
        print(f"    Beam {i}: seq={seq}, score={score:.4f} [{status}]")

    # The EOS beam (score=-3.0) must be the winner.
    best_seq, best_score = results[0][0]
    print(f"\n  Best beam: seq={best_seq}, score={best_score:.4f}")

    assert best_score == -3.0, (
        f"The EOS beam with score=-3.0 should win! Got score={best_score}. "
        f"This means finished beams were incorrectly removed from the pool."
    )
    assert eos_token_id in best_seq, (
        f"The winning beam should contain EOS! Got seq={best_seq}."
    )
    assert best_seq == [eos_token_id], (
        f"The EOS beam should be [{eos_token_id}]! Got seq={best_seq}."
    )

    # Verify the second beam is the continuing one (eventually hits EOS at -6.0)
    second_seq, second_score = results[0][1]
    print(f"  Second beam: seq={second_seq}, score={second_score:.4f}")
    assert second_score < best_score, (
        f"Second beam score ({second_score}) should be worse than best ({best_score})!"
    )
    # The continuing beam went: -4.0 (step1) + -1.0 (step2) + -1.0 (step3 EOS) = -6.0
    assert second_score == -6.0, (
        f"Second beam should have score=-6.0! Got {second_score}."
    )

    print("  PASSED: EOS beam correctly retained and ranked as winner.\n")
    print("  This confirms finished beams are NOT removed from the pool.")
    print("  If they were removed, the continuing beam (score=-5.0) would")
    print("  have wrongly won, because the EOS beam would have been discarded.\n")


def test_eos_retention_with_length_penalty():
    """Extended EOS test with alpha=0.6 to verify length penalty interaction.

    Scenario: two beams both hit EOS, but at different lengths.
    - Step 1 EOS: acc=-2.0, len=1, score=-2.0/(1^0.6) = -2.0
    - Step 2 EOS: acc=-1.0, len=2, score=-1.0/(2^0.6) = -1.0/1.516 = -0.660

    The longer beam wins due to length penalty, proving that:
    1) The step 1 EOS beam was retained in the pool (not discarded)
    2) Length penalty correctly favors the longer, higher-quality sequence
    """
    print("=" * 60)
    print("Test 3b: EOS retention with length penalty (alpha=0.6)")
    print("=" * 60)

    vocab_size = 100
    eos_token_id = 1
    continue_token = 2
    prompt = [10, 20]

    mock = MockModel(vocab_size=vocab_size)

    # Step 1: EOS with -2.0, continue with -0.5
    step1_log_probs = np.full(vocab_size, -1e6, dtype=np.float64)
    step1_log_probs[eos_token_id] = -2.0       # acc=-2.0, len=1, score=-2.0
    step1_log_probs[continue_token] = -0.5     # acc=-0.5, len=1
    mock.set_log_probs(prompt, step1_log_probs)

    # Step 2: continuing beam hits EOS with -0.5 → acc=-1.0, len=2
    step2_log_probs = np.full(vocab_size, -1e6, dtype=np.float64)
    step2_log_probs[eos_token_id] = -0.5        # acc=-0.5+(-0.5)=-1.0, len=2
    step2_log_probs[continue_token] = -1e5
    mock.set_log_probs(prompt + [continue_token], step2_log_probs)

    beam_width = 2
    alpha = 0.6

    results = batched_beam_search(
        prompts=[prompt],
        beam_width=beam_width,
        max_new_tokens=5,
        alpha=alpha,
        eos_token_id=eos_token_id,
        model=mock,
    )

    print(f"  Results (top {beam_width} beams):")
    for i, (seq, score) in enumerate(results[0]):
        status = "FINISHED" if seq and seq[-1] == eos_token_id else "unfinished"
        print(f"    Beam {i}: seq={seq}, score={score:.4f} [{status}]")

    # Verify both EOS beams are in results (step 1 EOS was retained, not discarded)
    assert len(results[0]) == 2, f"Expected 2 beams, got {len(results[0])}"
    all_finished = all(
        seq and seq[-1] == eos_token_id
        for seq, _ in results[0]
    )
    assert all_finished, "Both beams should be finished (hit EOS)."

    # Step 2 EOS beam should win: score = -1.0 / (2^0.6) ≈ -0.660
    # Step 1 EOS beam: score = -2.0 / (1^0.6) = -2.0
    best_seq, best_score = results[0][0]
    second_seq, second_score = results[0][1]

    expected_best_score = -1.0 / (2 ** alpha)
    expected_second_score = -2.0 / (1 ** alpha)

    print(f"\n  Best beam: seq={best_seq}, score={best_score:.4f} "
          f"(expected ~{expected_best_score:.4f})")
    print(f"  Second:   seq={second_seq}, score={second_score:.4f} "
          f"(expected ~{expected_second_score:.4f})")

    assert abs(best_score - expected_best_score) < 1e-4, (
        f"Best score {best_score} != expected {expected_best_score}"
    )
    assert abs(second_score - expected_second_score) < 1e-4, (
        f"Second score {second_score} != expected {expected_second_score}"
    )
    assert best_seq == [continue_token, eos_token_id], (
        f"Longer beam should win! Got {best_seq}"
    )
    assert second_seq == [eos_token_id], (
        f"Step 1 EOS beam should be second (retained, not discarded)! Got {second_seq}"
    )

    print("  PASSED: Length penalty correctly applied. "
          "Step 1 EOS beam retained and competed fairly.\n")


if __name__ == "__main__":
    test_greedy_equivalence()
    test_batch_independence()
    test_eos_retention()
    test_eos_retention_with_length_penalty()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
