"""
Ternary training for Qwen3-0.6B on train_data.txt.

Approach:
- Load Qwen3-0.6B via mlx_lm (latent weights start as pretrained FP weights).
- Replace every nn.Linear under the transformer blocks (Q/K/V/O, gate/up/down)
  with a TernaryLinear module.
  - The lm_head is tied to the input embedding for Qwen3-0.6B; we leave the
    embedding/lm_head non-ternary (BitNet b1.58 lineage keeps embeddings in
    higher precision; ternarizing a 151936x1024 lookup table also doesn't
    save FLOPs since it's a gather, not a matmul).
- Group-wise scale s = mean(|W|) over groups of 128 along the input dim.
- Forward uses W_t = clip(round(W / s), -1, 1) * s. STE makes the gradient
  of the projection an identity so dL/dW_latent = dL/dW_ternary.
- Train with AdamW for >=200 steps on train_data.txt, then verify, generate,
  and compute held-out perplexity.
"""

import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.generate import generate


import os

GROUP_SIZE = 128
SEQ_LEN = int(os.environ.get("SEQ_LEN", 256))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
NUM_STEPS = int(os.environ.get("NUM_STEPS", 250))
LR = float(os.environ.get("LR", 5e-4))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 30))
SEED = int(os.environ.get("SEED", 0))
DATA_PATH = Path(__file__).parent / "train_data.txt"


# -------------------------- TernaryLinear --------------------------

class TernaryLinear(nn.Module):
    """Linear layer with weight stored in float (latent) but projected to
    a group-wise ternary {-s, 0, +s} representation in the forward pass.

    Gradient flows through the projection via a Straight-Through Estimator:
        W_eff = W + stop_gradient(W_t - W)
    so dL/dW = dL/dW_eff at the W_eff = W_t point.
    """

    def __init__(self, in_features: int, out_features: int, group_size: int = GROUP_SIZE):
        super().__init__()
        if in_features % group_size != 0:
            raise ValueError(f"in_features={in_features} not divisible by group_size={group_size}")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        scale = in_features ** -0.5
        self.weight = mx.random.normal((out_features, in_features)) * scale

    @staticmethod
    def ternarize(W: mx.array, group_size: int) -> mx.array:
        out, in_ = W.shape
        Wg = W.reshape(out, in_ // group_size, group_size)
        # Per-group scale: mean(|W|).
        s = mx.mean(mx.abs(Wg), axis=-1, keepdims=True)
        s = mx.maximum(s, 1e-8)
        q = mx.clip(mx.round(Wg / s), -1, 1)
        Wt = q * s
        return Wt.reshape(out, in_)

    def __call__(self, x: mx.array) -> mx.array:
        Wt = self.ternarize(self.weight, self.group_size)
        # STE: identity gradient through the projection.
        W_eff = self.weight + mx.stop_gradient(Wt - self.weight)
        return x @ W_eff.T


def replace_linear_with_ternary(parent: nn.Module):
    """Recursively walk `parent` and swap nn.Linear children for TernaryLinear,
    transferring the pretrained weight into the latent slot.
    """
    children = parent.children()
    for name, child in children.items():
        if isinstance(child, nn.Linear) and not isinstance(child, TernaryLinear):
            in_f = child.weight.shape[1]
            out_f = child.weight.shape[0]
            tl = TernaryLinear(in_f, out_f, group_size=GROUP_SIZE)
            tl.weight = child.weight
            setattr(parent, name, tl)
        elif isinstance(child, nn.Module):
            replace_linear_with_ternary(child)
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, nn.Module):
                    replace_linear_with_ternary(item)


def count_ternary_layers(model):
    out = collect_ternary_modules(model)
    return len(out), sum(tl.weight.size for tl in out)


def collect_ternary_modules(model):
    """Walk the tree and return all TernaryLinear modules."""
    found = []

    def walk(mod):
        for _, child in mod.children().items():
            if isinstance(child, TernaryLinear):
                found.append(child)
            elif isinstance(child, nn.Module):
                walk(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, nn.Module):
                        walk(item)

    walk(model)
    return found


# -------------------------- Data --------------------------

def load_tokens(tokenizer, path: Path):
    text = path.read_text()
    # Split by blank lines into "paragraphs" so val/train are coherent units.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Hold out the last 10% as validation.
    n_val = max(1, len(paragraphs) // 10)
    train_paras = paragraphs[:-n_val]
    val_paras = paragraphs[-n_val:]

    train_ids = tokenizer.encode("\n\n".join(train_paras))
    val_ids = tokenizer.encode("\n\n".join(val_paras))
    return mx.array(train_ids, dtype=mx.int32), mx.array(val_ids, dtype=mx.int32)


def sample_batch(tokens: mx.array, seq_len: int, batch_size: int, rng):
    n = tokens.shape[0]
    max_start = n - seq_len - 1
    starts = rng.integers(0, max_start, size=(batch_size,))
    x = mx.stack([tokens[s : s + seq_len] for s in starts.tolist()])
    y = mx.stack([tokens[s + 1 : s + 1 + seq_len] for s in starts.tolist()])
    return x, y


# -------------------------- Loss --------------------------

def loss_fn(model, x, y):
    logits = model(x)
    logits = logits.astype(mx.float32)
    # Cross-entropy over vocab.
    log_probs = nn.log_softmax(logits, axis=-1)
    # Gather target log-probs.
    B, L, V = log_probs.shape
    flat = log_probs.reshape(B * L, V)
    flat_y = y.reshape(B * L)
    nll = -flat[mx.arange(B * L), flat_y]
    return nll.mean()


# -------------------------- LR schedule --------------------------

def lr_at(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    # Cosine decay to 10% of peak.
    progress = (step - WARMUP_STEPS) / max(1, NUM_STEPS - WARMUP_STEPS)
    progress = min(1.0, progress)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


# -------------------------- Verification --------------------------

def verify_ternary(model):
    """For every TernaryLinear, recompute the projection and check it is
    exactly {-s, 0, +s} per-group.
    """
    bad = 0
    total = 0
    max_err = 0.0
    for tl in collect_ternary_modules(model):
        W = tl.weight
        Wt = TernaryLinear.ternarize(W, tl.group_size)
        out, in_ = Wt.shape
        Wg = Wt.reshape(out, in_ // tl.group_size, tl.group_size)
        s = mx.mean(mx.abs(Wg), axis=-1, keepdims=True)  # recompute scale ON the projected tensor
        # The recomputed s should equal the s used in the projection for
        # values that were already ternary; for verification we compare
        # Wt / s_orig to integers in {-1, 0, +1}.
        s_orig = mx.maximum(mx.mean(mx.abs(W.reshape(out, in_ // tl.group_size, tl.group_size)), axis=-1, keepdims=True), 1e-8)
        q = Wg / s_orig
        # Distance to nearest integer in {-1, 0, +1}.
        nearest = mx.clip(mx.round(q), -1, 1)
        err = mx.max(mx.abs(q - nearest)).item()
        max_err = max(max_err, err)
        out_size = Wt.size
        bad_here = (mx.abs(q - nearest) > 1e-5).sum().item()
        bad += bad_here
        total += out_size
    return bad, total, max_err


# -------------------------- Main --------------------------

def main():
    mx.random.seed(SEED)
    import numpy as np
    rng = np.random.default_rng(SEED)

    print("[1/6] Loading Qwen3-0.6B via mlx_lm…", flush=True)
    t0 = time.time()
    model, tokenizer = load("Qwen/Qwen3-0.6B")
    print(f"      loaded in {time.time()-t0:.1f}s", flush=True)

    print("[2/6] Replacing nn.Linear with TernaryLinear (Q/K/V/O + gate/up/down)…", flush=True)
    # Walk into model.model.layers (transformer blocks) only.
    for layer in model.model.layers:
        replace_linear_with_ternary(layer)
    n_ternary, n_ternary_params = count_ternary_layers(model)
    print(f"      replaced {n_ternary} linear layers ({n_ternary_params/1e6:.1f}M ternary params)", flush=True)

    # Force the parameter tree to materialize on device with proper dtypes.
    # Latent weights are kept in float32 for stable optimizer math.
    def to_f32(p):
        if isinstance(p, mx.array) and p.dtype != mx.float32 and p.ndim >= 1:
            return p.astype(mx.float32)
        return p
    # Only cast the ternary latent weights to fp32; leave norms/embeddings alone.
    for tl in collect_ternary_modules(model):
        tl.weight = tl.weight.astype(mx.float32)

    # Quick smoke test: forward pass on 1 token.
    test_in = mx.array([[tokenizer.eos_token_id]])
    _ = model(test_in)
    mx.eval(_)
    print("      smoke forward pass ok", flush=True)

    print("[3/6] Tokenizing train_data.txt…", flush=True)
    train_tokens, val_tokens = load_tokens(tokenizer, DATA_PATH)
    print(f"      train tokens: {train_tokens.size}, val tokens: {val_tokens.size}", flush=True)

    print(f"[4/6] Training for {NUM_STEPS} steps (batch={BATCH_SIZE}, seq_len={SEQ_LEN}, lr_peak={LR})…", flush=True)

    optimizer = optim.AdamW(learning_rate=LR, weight_decay=0.0, betas=(0.9, 0.95))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    t0 = time.time()
    for step in range(NUM_STEPS):
        x, y = sample_batch(train_tokens, SEQ_LEN, BATCH_SIZE, rng)
        optimizer.learning_rate = lr_at(step)
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        l = loss.item()
        losses.append(l)
        if step % 10 == 0 or step == NUM_STEPS - 1:
            elapsed = time.time() - t0
            print(f"      step {step:4d}/{NUM_STEPS}  loss={l:.4f}  lr={optimizer.learning_rate.item():.2e}  ({elapsed:.0f}s)", flush=True)

    print(f"      first 5 loss avg: {sum(losses[:5])/5:.4f}", flush=True)
    print(f"      last 20 loss avg: {sum(losses[-20:])/20:.4f}", flush=True)

    print("[5/6] Verifying ternary projection…", flush=True)
    bad, total, max_err = verify_ternary(model)
    print(f"      bad weights: {bad}/{total}  max projection error: {max_err:.2e}", flush=True)
    ternary_ok = bad == 0 and max_err < 1e-4
    print(f"      TERNARY OK: {ternary_ok}", flush=True)

    print("[6/6] Validation perplexity + samples…", flush=True)
    # Compute val perplexity over non-overlapping windows.
    n_val = val_tokens.size
    win = SEQ_LEN
    n_windows = max(1, (n_val - 1) // win)
    nll_sum = 0.0
    tok_count = 0
    for i in range(n_windows):
        s = i * win
        x = val_tokens[s : s + win][None, :]
        y = val_tokens[s + 1 : s + 1 + win][None, :]
        if y.shape[1] < win:
            break
        l = loss_fn(model, x, y)
        mx.eval(l)
        nll_sum += l.item() * y.size
        tok_count += y.size
    val_nll = nll_sum / max(1, tok_count)
    val_ppl = math.exp(val_nll)
    print(f"      val nll: {val_nll:.4f}  val ppl: {val_ppl:.2f}", flush=True)

    # Generate samples. We need to make the model use the ternary forward path
    # during generation — it does, since model.__call__ calls TernaryLinear.
    samples = []
    prompts = [
        "Open source software has",
        "World War II was",
        "The development of antibiotics",
        "Sleep is essential for",
        "The scientific method is",
    ]
    for p in prompts:
        try:
            txt = generate(model, tokenizer, prompt=p, max_tokens=60, verbose=False)
        except TypeError:
            # Older/newer mlx_lm signatures.
            txt = generate(model, tokenizer, p, max_tokens=60)
        samples.append((p, txt))
        print(f"      [{p!r}] -> {txt!r}", flush=True)

    report = {
        "first_5_loss_avg": sum(losses[:5])/5,
        "last_20_loss_avg": sum(losses[-20:])/20,
        "final_loss": losses[-1],
        "ternary_ok": ternary_ok,
        "ternary_max_err": max_err,
        "val_nll": val_nll,
        "val_ppl": val_ppl,
        "num_steps": NUM_STEPS,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "lr_peak": LR,
        "group_size": GROUP_SIZE,
        "n_ternary_layers": n_ternary,
        "n_ternary_params": int(n_ternary_params),
        "samples": [{"prompt": p, "generation": t} for p, t in samples],
        "loss_curve": losses,
    }
    out_path = Path(__file__).parent / "report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"      wrote report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
