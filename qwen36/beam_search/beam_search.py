import numpy as np
from model import MinimalLM


class Beam:
    """Represents a single beam in beam search."""

    __slots__ = ('sequence', 'accumulated_logprob', 'finished', 'generated_length')

    def __init__(self, sequence, accumulated_logprob, finished, generated_length):
        self.sequence = sequence
        self.accumulated_logprob = accumulated_logprob
        self.finished = finished
        self.generated_length = generated_length

    def length_penalized_score(self, alpha):
        """Compute length-penalized score for ranking.

        IMPORTANT: This is used ONLY for ranking/selection. The accumulated_logprob
        is NEVER modified by length penalty — it remains the raw sum of logprobs.
        """
        if self.generated_length == 0:
            return self.accumulated_logprob
        return self.accumulated_logprob / (self.generated_length ** alpha)


def batched_beam_search(
    prompts,
    beam_width,
    max_new_tokens,
    alpha,
    eos_token_id,
    model,
):
    """Batched beam search decoder for multiple independent prompts.

    Args:
        prompts: list[list[int]] — one prompt (list of token IDs) per batch item.
        beam_width: int K — number of beams per batch item.
        max_new_tokens: int — maximum number of new tokens to generate.
        alpha: float — length penalty exponent (0.0 = no penalty).
        eos_token_id: int — token ID that marks end of sequence.
        model: MinimalLM instance (or any object with get_log_probs(token_ids)).

    Returns:
        list[list[tuple[list[int], float]]] — for each batch item, a list of
        (sequence, score) tuples sorted by length-penalized score descending.
        Sequences contain generated token IDs only (NOT prompt tokens).

    Key design decision — why finished beams must NOT be removed from the pool:
    ===========================================================================
    When a beam hits EOS, it represents a complete, high-confidence candidate
    sequence. If we remove it from the pool, we lose the ability to compare it
    against longer, still-growing beams. A short sequence with accumulated
    logprob=-3.0 and length=2 has score=-3.0/(2^0.6) ≈ -2.10, which may be
    better than a longer sequence with logprob=-5.0 and length=3 scoring
    -5.0/(3^0.6) ≈ -3.31. By keeping finished beams in the pool, they compete
    fairly using length-penalized scores. Removing them would incorrectly favor
    longer, lower-confidence sequences simply because they haven't stopped yet.
    This is the canonical beam search EOS bug — removing finished beams causes
    the decoder to miss the best sequence when it terminates early.
    """
    results = []

    for batch_idx, prompt in enumerate(prompts):
        prompt_arr = np.array(prompt, dtype=np.int64)

        # Initialize with a single beam: no tokens generated yet
        beams = [Beam([], 0.0, False, 0)]

        # finished_beams tracks beams that have produced EOS.
        # They remain in the pool and compete with unfinished beams.
        # We do NOT discard them — they persist across steps.
        finished_beams = []

        for step in range(max_new_tokens):
            # Separate finished and unfinished beams
            unfinished = [b for b in beams if not b.finished]

            # If all beams are finished, stop expanding this batch item
            if not unfinished:
                break

            # Expand each unfinished beam
            all_candidates = []
            top_k_expand = min(2 * beam_width, model.vocab_size)

            for beam in unfinished:
                # Full context: prompt + generated tokens so far
                full_seq = np.concatenate([
                    prompt_arr,
                    np.array(beam.sequence, dtype=np.int64)
                ])
                log_probs = model.get_log_probs(full_seq)

                # Top-(2*K) candidates to preserve diversity
                top_indices = np.argpartition(log_probs, -top_k_expand)[-top_k_expand:]
                top_indices = top_indices[np.argsort(log_probs[top_indices])[::-1]]

                for token_id in top_indices:
                    token_id_int = int(token_id)
                    new_logprob = float(log_probs[token_id])
                    new_acc_logprob = beam.accumulated_logprob + new_logprob
                    new_length = beam.generated_length + 1
                    new_seq = beam.sequence + [token_id_int]

                    # If this token is EOS, the beam is finished
                    is_finished = (token_id_int == eos_token_id)

                    candidate = Beam(new_seq, new_acc_logprob, is_finished, new_length)
                    all_candidates.append(candidate)

            # Build the selection pool:
            # (a) All previously finished beams — they STAY and compete.
            #     This is the critical design choice. Removing finished beams
            #     would discard high-confidence short sequences that terminated
            #     early, causing the decoder to incorrectly prefer longer
            #     lower-confidence sequences.
            # (b) All new candidates from expanding unfinished beams.
            pool = finished_beams + all_candidates

            # Rank by length-penalized score (descending — higher = better).
            # Length penalty is applied ONLY for ranking, not to accumulated_logprob.
            # The raw accumulated_logprob stays intact for final scoring.
            pool.sort(key=lambda b: b.length_penalized_score(alpha), reverse=True)

            # Take top K beams
            beams = pool[:beam_width]

            # Separate finished from unfinished for next iteration.
            # Finished beams are carried forward in the pool but not expanded.
            finished_beams = [b for b in beams if b.finished]
            # Unfinished beams continue to be expanded next step.
            # We keep all beams (finished + unfinished) in `beams`.

        # Final result: sort all beams by length-penalized score descending
        beams.sort(key=lambda b: b.length_penalized_score(alpha), reverse=True)
        result = [
            (b.sequence, b.length_penalized_score(alpha))
            for b in beams
        ]
        results.append(result)

    return results


def beam_search_wrapper(
    prompts,
    beam_width=1,
    max_new_tokens=20,
    alpha=0.6,
    eos_token_id=0,
    model=None,
    seed=42,
):
    """Convenience wrapper that creates a model if not provided."""
    if model is None:
        model = MinimalLM(seed=seed)
    return batched_beam_search(
        prompts, beam_width, max_new_tokens, alpha, eos_token_id, model
    )
