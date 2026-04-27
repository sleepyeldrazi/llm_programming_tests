import numpy as np


class MinimalLM:
    """Minimal language model: random embeddings + 1 transformer block + LM head."""

    def __init__(self, vocab_size=1000, d_model=64, seed=42):
        self.vocab_size = vocab_size
        self.d_model = d_model
        rng = np.random.RandomState(seed)

        # Token embeddings
        self.embeddings = rng.randn(vocab_size, d_model).astype(np.float32)

        # Transformer block (single layer, no layer norm for simplicity)
        # Self-attention
        self.Wq = rng.randn(d_model, d_model).astype(np.float32) * 0.01
        self.Wk = rng.randn(d_model, d_model).astype(np.float32) * 0.01
        self.Wv = rng.randn(d_model, d_model).astype(np.float32) * 0.01
        self.Wo = rng.randn(d_model, d_model).astype(np.float32) * 0.01

        # FFN
        self.W1 = rng.randn(d_model, d_model * 4).astype(np.float32) * 0.01
        self.W2 = rng.randn(d_model * 4, d_model).astype(np.float32) * 0.01

        # LM head (projection back to vocab)
        self.lm_head = rng.randn(d_model, vocab_size).astype(np.float32) * 0.01

    def forward(self, token_ids):
        """
        Run forward pass on a sequence of token IDs.

        Args:
            token_ids: np.ndarray of shape (seq_len,) with integer token IDs.

        Returns:
            logits: np.ndarray of shape (vocab_size,) for the last token.
        """
        seq_len = len(token_ids)

        # Embed all tokens
        h = self.embeddings[token_ids]  # (seq_len, d_model)

        # Self-attention
        Q = h @ self.Wq  # (seq_len, d_model)
        K = h @ self.Wk
        V = h @ self.Wv

        # Scaled dot-product attention (causal mask)
        scores = Q @ K.T / np.sqrt(self.d_model)  # (seq_len, seq_len)
        mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)
        scores = scores - mask * 1e9
        # Softmax
        scores = scores - scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        attn_out = attn @ V  # (seq_len, d_model)
        attn_out = attn_out @ self.Wo

        # Residual + FFN
        h = h + attn_out
        ffn = h @ self.W1
        ffn = np.maximum(ffn, 0)  # ReLU
        ffn = ffn @ self.W2
        h = h + ffn

        # LM head on last token
        last_hidden = h[-1]  # (d_model,)
        logits = last_hidden @ self.lm_head  # (vocab_size,)
        return logits

    def get_log_probs(self, token_ids):
        """Get log probabilities for next token given a sequence."""
        logits = self.forward(token_ids)
        # Log-softmax in a numerically stable way
        max_logit = logits.max()
        log_probs = logits - max_logit - np.log(np.exp(logits - max_logit).sum())
        return log_probs
