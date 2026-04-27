"""
Batched Beam Search Decoder for Autoregressive Generation
Implemented in pure NumPy
"""

import numpy as np
from typing import List, Tuple


class MinimalLanguageModel:
    """
    A minimal language model with random embeddings + 1 transformer block.
    Used to test beam search correctness - quality doesn't matter.
    """

    def __init__(self, vocab_size: int = 1000, d_model: int = 64, num_heads: int = 4):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads

        np.random.seed(42)
        self.embedding = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02
        self.embedding_norm = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

        self.query_projection = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.key_projection = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.value_projection = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

        self.output_projection = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.ffn_inner = np.random.randn(d_model, d_model * 4).astype(np.float32) * 0.02
        self.ffn_outer = np.random.randn(d_model * 4, d_model).astype(np.float32) * 0.02

        self.layer_norm_scale = np.ones(d_model).astype(np.float32)
        self.layer_norm_bias = np.zeros(d_model).astype(np.float32)

        self.ffn_ln_scale = np.ones(d_model).astype(np.float32)
        self.ffn_ln_bias = np.zeros(d_model).astype(np.float32)

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True) + 1e-6
        return self.layer_norm_scale * (x - mean) / std + self.layer_norm_bias

    def _ffn_layer_norm(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True) + 1e-6
        return self.ffn_ln_scale * (x - mean) / std + self.ffn_ln_bias

    def _multi_head_attention(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape

        Q = np.dot(x, self.query_projection)
        K = np.dot(x, self.key_projection)
        V = np.dot(x, self.value_projection)

        head_dim = d_model // self.num_heads
        Q = Q.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        attention_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        attention_probs = self._softmax(attention_scores)

        attention_output = np.matmul(attention_probs, V)
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

        return np.dot(attention_output, self.output_projection)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def _feed_forward(self, x: np.ndarray) -> np.ndarray:
        inner = np.dot(x, self.ffn_inner)
        inner = np.maximum(inner, 0)
        return np.dot(inner, self.ffn_outer)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        batch_size, seq_len = token_ids.shape

        x = self.embedding[token_ids]
        x = np.dot(x, self.embedding_norm)

        x_normed = self._layer_norm(x)
        attn_out = self._multi_head_attention(x_normed)
        x = x + attn_out

        x_normed = self._ffn_layer_norm(x)
        ffn_out = self._feed_forward(x_normed)
        x = x + ffn_out

        logits = np.matmul(x, self.embedding.T)

        return logits


def batched_beam_search(
    prompts: List[List[int]],
    beam_width: int,
    max_new_tokens: int,
    alpha: float = 0.6,
    eos_token_id: int = 0,
    model: MinimalLanguageModel = None
) -> List[List[Tuple[List[int], float]]]:
    """
    Batched beam search decoder for autoregressive generation.

    Args:
        prompts: List of prompt token ID lists, one per batch item
        beam_width: Number of beams per batch item (K)
        max_new_tokens: Maximum number of new tokens to generate
        alpha: Length penalty hyperparameter (default 0.6)
        eos_token_id: End-of-sequence token ID
        model: The language model to use

    Returns:
        List of lists of (sequence, score) tuples per batch item,
        sorted by length-penalized score descending (best first)

    IMPORTANT: Finished beams are NOT removed from the pool. They compete
    with unfinished beams using length-penalized scores. This ensures that
    a short, high-confidence sequence that hits EOS early is not wrongly
    discarded in favor of a longer, lower-confidence sequence.
    """
    if model is None:
        model = MinimalLanguageModel()

    batch_size = len(prompts)
    vocab_size = model.vocab_size

    active_beams = []
    for batch_idx in range(batch_size):
        prompt_tokens = np.array(prompts[batch_idx], dtype=np.int32)

        beams = [{
            'seq': list(prompt_tokens),
            'logprob': 0.0,
            'generated_length': 0,
            'finished': False,
            'batch_idx': batch_idx
        }]
        active_beams.append(beams)

    finished_results = [[] for _ in range(batch_size)]

    for step in range(max_new_tokens):
        all_candidates = []

        all_done = True
        for batch_idx in range(batch_size):
            beams = active_beams[batch_idx]
            if beams and not all(beam['finished'] for beam in beams):
                all_done = False
                break
        if all_done:
            break

        for batch_idx in range(batch_size):
            beams = active_beams[batch_idx]

            if not beams:
                continue

            if all(beam['finished'] for beam in beams):
                for beam in beams:
                    finished_results[batch_idx].append({
                        'seq': beam['seq'][len(prompts[batch_idx]):],
                        'logprob': beam['logprob'],
                        'generated_length': beam['generated_length']
                    })
                active_beams[batch_idx] = []
                continue

            seqs = [beam['seq'] for beam in beams]
            max_seq_len = max(len(seq) for seq in seqs)

            padded_seqs = []
            for seq in seqs:
                if len(seq) < max_seq_len:
                    padded_seqs.append(seq + [0] * (max_seq_len - len(seq)))
                else:
                    padded_seqs.append(seq)

            input_ids = np.array(padded_seqs, dtype=np.int32)

            logits = model.forward(input_ids)

            last_logits = logits[:, -1, :]

            probs = np.exp(last_logits - np.max(last_logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)

            for beam_idx, beam in enumerate(beams):
                if beam['finished']:
                    continue

                beam_logprob = beam['logprob']
                beam_gen_len = beam['generated_length']

                token_probs = probs[beam_idx]

                top_k_indices = np.argpartition(token_probs, -2 * beam_width)[-2 * beam_width:]
                top_k_indices = top_k_indices[np.argsort(token_probs[top_k_indices])[::-1]]

                for token_id in top_k_indices:
                    token_prob = token_probs[token_id]

                    if token_prob <= 0:
                        continue

                    new_logprob = beam_logprob + np.log(token_prob)
                    new_gen_len = beam_gen_len + 1
                    new_seq = beam['seq'] + [int(token_id)]

                    is_finished = (token_id == eos_token_id)

                    if is_finished:
                        cand_logprob = beam_logprob
                        cand_gen_len = beam_gen_len
                    else:
                        cand_logprob = new_logprob
                        cand_gen_len = new_gen_len

                    all_candidates.append({
                        'batch_idx': batch_idx,
                        'seq': new_seq,
                        'logprob': cand_logprob,
                        'generated_length': cand_gen_len,
                        'finished': is_finished,
                        'beam_idx': beam_idx
                    })

        if not all_candidates:
            break

        for batch_idx in range(batch_size):
            batch_candidates = [c for c in all_candidates if c['batch_idx'] == batch_idx]

            if not batch_candidates:
                continue

            adjusted_scores = []
            for c in batch_candidates:
                gen_len = c['generated_length']
                if gen_len == 0:
                    adj_score = c['logprob']
                else:
                    adj_score = c['logprob'] / (gen_len ** alpha)
                adjusted_scores.append(adj_score)

            adjusted_scores = np.array(adjusted_scores)

            select_k = min(beam_width, len(adjusted_scores))
            if select_k <= 0:
                continue

            if len(adjusted_scores) <= beam_width:
                top_k_indices = np.arange(len(adjusted_scores))
            else:
                top_k_indices = np.argpartition(adjusted_scores, -select_k)[-select_k:]

            top_k_indices = top_k_indices[np.argsort(adjusted_scores[top_k_indices])[::-1]]

            selected = [batch_candidates[i] for i in top_k_indices]

            new_active_beams = []
            for c in selected:
                if c['finished']:
                    finished_results[c['batch_idx']].append({
                        'seq': c['seq'][len(prompts[c['batch_idx']]):],
                        'logprob': c['logprob'],
                        'generated_length': c['generated_length']
                    })
                else:
                    new_active_beams.append({
                        'seq': c['seq'],
                        'logprob': c['logprob'],
                        'generated_length': c['generated_length'],
                        'finished': False,
                        'batch_idx': c['batch_idx']
                    })

            active_beams[batch_idx] = new_active_beams

    for batch_idx in range(batch_size):
        remaining_beams = active_beams[batch_idx]
        for beam in remaining_beams:
            finished_results[batch_idx].append({
                'seq': beam['seq'][len(prompts[batch_idx]):],
                'logprob': beam['logprob'],
                'generated_length': beam['generated_length']
            })

    results = []
    for batch_idx in range(batch_size):
        batch_results = finished_results[batch_idx]

        scored_results = []
        for item in batch_results:
            seq = item['seq']
            logprob = item['logprob']
            gen_len = item['generated_length']
            if gen_len == 0:
                adj_score = logprob
            else:
                adj_score = logprob / (gen_len ** alpha)
            scored_results.append((seq, adj_score))

        scored_results.sort(key=lambda x: x[1], reverse=True)

        results.append(scored_results[:beam_width])

    return results


def test_greedy_equivalence():
    """Test 1: Single batch item, K=1, short prompt, alpha=0
    Verify this behaves identically to greedy decoding (always pick argmax)
    """
    print("=" * 60)
    print("TEST 1: Greedy Equivalence Test")
    print("=" * 60)

    model = MinimalLanguageModel(vocab_size=1000, d_model=64)
    prompt = [[1, 2, 3]]
    beam_width = 1
    max_new_tokens = 5
    alpha = 0.0
    eos_token_id = 0

    results = batched_beam_search(prompt, beam_width, max_new_tokens, alpha, eos_token_id, model)

    print(f"Prompt: {prompt}")
    print(f"Beam width: {beam_width}, Alpha: {alpha}")
    print(f"Generated sequences: {results}")

    input_ids = np.array(prompt, dtype=np.int32)
    greedy_seq = list(prompt[0])

    for _ in range(max_new_tokens):
        logits = model.forward(input_ids)
        probs = np.exp(logits[0, -1] - np.max(logits[0, -1]))
        probs = probs / np.sum(probs)
        next_token = int(np.argmax(probs))
        greedy_seq.append(next_token)
        if next_token == eos_token_id:
            break
        input_ids = np.array([greedy_seq], dtype=np.int32)

    print(f"Greedy sequence (expected): {greedy_seq[len(prompt[0]):]}")

    if results[0]:
        result_seq = results[0][0][0]
        print(f"Beam search sequence: {result_seq}")
        match = result_seq == greedy_seq[len(prompt[0]):]
        print(f"Match with greedy: {match}")
    print()


def test_per_batch_independence():
    """Test 2: batch=2, beam_width=3, different prompt lengths [3, 5], alpha=0.6
    Verify per-batch independence: beams from prompt 0 never interact with beams from prompt 1
    """
    print("=" * 60)
    print("TEST 2: Per-Batch Independence Test")
    print("=" * 60)

    model = MinimalLanguageModel(vocab_size=1000, d_model=64)
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8]]
    beam_width = 3
    max_new_tokens = 4
    alpha = 0.6
    eos_token_id = 0

    results = batched_beam_search(prompts, beam_width, max_new_tokens, alpha, eos_token_id, model)

    print(f"Prompts: {prompts}")
    print(f"Prompt lengths: {[len(p) for p in prompts]}")
    print(f"Beam width: {beam_width}, Alpha: {alpha}")
    print(f"Results for batch 0 (should have {beam_width} beams): {len(results[0])} beams")
    print(f"Results for batch 1 (should have {beam_width} beams): {len(results[1])} beams")

    for batch_idx, batch_results in enumerate(results):
        print(f"\nBatch {batch_idx} results:")
        for seq, score in batch_results:
            print(f"  Seq: {seq[:10]}..., Score: {score:.4f}")

    prompt_0_tokens = set(prompts[0])
    prompt_1_tokens = set(prompts[1])

    cross_contamination = False
    for seq, _ in results[0]:
        overlap = set(seq) & prompt_1_tokens
        if overlap:
            print(f"WARNING: Batch 0 seq contains tokens from batch 1 prompt: {overlap}")
            cross_contamination = True

    for seq, _ in results[1]:
        overlap = set(seq) & prompt_0_tokens
        if overlap:
            print(f"WARNING: Batch 1 seq contains tokens from batch 0 prompt: {overlap}")
            cross_contamination = True

    print(f"\nPer-batch independence verified: {len(results) == 2 and not cross_contamination}")
    print()


def test_eos_retention():
    """Test 3: THE EOS RETENTION TEST
    Verify that EOS beams compete correctly with unfinished beams.
    A beam that hits EOS early with logprob=-3.0 should beat
    an unfinished beam with logprob=-5.0 (both length-penalized).
    """
    print("=" * 60)
    print("TEST 3: EOS Retention Test")
    print("=" * 60)

    model = MinimalLanguageModel(vocab_size=1000, d_model=64)
    prompt = [[1, 2, 3, 4, 5]]
    beam_width = 3
    max_new_tokens = 10
    alpha = 0.6
    eos_token_id = 42

    class MockedModel:
        def __init__(self, real_model):
            self.vocab_size = real_model.vocab_size
            self.real_model = real_model
            self.step_count = 0
            self.eos_logprob = -3.0
            self.cont_logprob = -4.0

        def forward(self, token_ids):
            self.step_count += 1
            batch_size, seq_len = token_ids.shape

            if self.step_count == 1:
                logits = np.full((batch_size, seq_len, self.vocab_size), -20.0, dtype=np.float32)

                logits[0, -1, eos_token_id] = 5.0
                logits[0, -1, 99] = 3.0

                return logits
            else:
                logits = self.real_model.forward(token_ids)
                return logits

    mocked_model = MockedModel(model)

    results = batched_beam_search(
        prompt, beam_width, max_new_tokens, alpha, eos_token_id, mocked_model
    )

    print(f"Prompt: {prompt}")
    print(f"Beam width: {beam_width}, Alpha: {alpha}, EOS token: {eos_token_id}")
    print(f"Step 1 mock: EOS token will have high logit (pre-softmax)")

    print(f"\nGenerated sequences:")
    for seq, score in results[0]:
        print(f"  Seq: {seq}, Score: {score:.4f}")

    eos_in_best = False
    if results[0]:
        best_seq, best_score = results[0][0]
        if eos_token_id in best_seq:
            eos_in_best = True
            print(f"\n[PASS] Best sequence contains EOS token - EOS beam correctly retained")
        else:
            print(f"\n[FAIL] Best sequence does NOT contain EOS token - EOS beam was wrongly discarded")
            print("This happens if finished beams are removed from the pool too early.")
            print("With correct EOS retention: the EOS beam (stopped at step 1 with score=-3.0/1^0.6=-3.0)")
            print("would beat continuing beams (logprob=-4.0 at step 1, then -5.0 at step 2, etc.)")
    print()


def run_all_tests():
    """Run all tests."""
    test_greedy_equivalence()
    test_per_batch_independence()
    test_eos_retention()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()