"""
Ternary Bonsai Training Script
===============================

Fine-tunes a Qwen3-0.6B model with ternary weights on WikiText-2.
Uses Straight-Through Estimator (STE) for gradient propagation through
the non-differentiable ternary quantization.

Usage:
    python train.py
    python train.py --steps 200 --lr 3e-4 --batch-size 2 --seq-len 512
"""

import argparse
import math
import time
import json
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW

from ternary_model import (
    Model,
    ModelArgs,
    TernaryLinear,
    TernaryEmbedding,
    project_ternary,
    copy_weights,
)


def cross_entropy(logits, targets):
    B, L, V = logits.shape
    logit_max = mx.max(logits, axis=-1, keepdims=True)
    shifted = logits - logit_max
    log_sum_exp = mx.log(mx.sum(mx.exp(shifted), axis=-1))
    targets_flat = targets.reshape(-1)
    idx = mx.arange(targets_flat.shape[0])
    target_logits = shifted.reshape(-1, V)[idx, targets_flat]
    return mx.mean(log_sum_exp.reshape(-1) - target_logits)


def _add_grads(g1, g2):
    if isinstance(g1, mx.array):
        return g1 + g2
    elif isinstance(g1, dict):
        return {k: _add_grads(g1[k], g2[k]) for k in g1}
    elif isinstance(g1, list):
        return [_add_grads(a, b) for a, b in zip(g1, g2)]
    return g1


def _scale_grads(g, scale):
    if isinstance(g, mx.array):
        return g * scale
    elif isinstance(g, dict):
        return {k: _scale_grads(v, scale) for k, v in g.items()}
    elif isinstance(g, list):
        return [_scale_grads(v, scale) for v in g]
    return g


def load_model_and_tokenizer(model_name="Qwen/Qwen3-0.6B"):
    print(f"Loading {model_name} ...")
    from mlx_lm import load

    orig_model, tokenizer = load(model_name)
    print(f"  Original model loaded. type={type(orig_model).__name__}")

    args_dict = {}
    for k in [
        "model_type", "hidden_size", "num_hidden_layers", "intermediate_size",
        "num_attention_heads", "rms_norm_eps", "vocab_size", "num_key_value_heads",
        "max_position_embeddings", "rope_theta", "head_dim", "tie_word_embeddings",
    ]:
        v = getattr(orig_model.args, k, None)
        if v is not None:
            args_dict[k] = v

    if hasattr(orig_model.args, "rope_scaling"):
        args_dict["rope_scaling"] = orig_model.args.rope_scaling

    args = ModelArgs.from_dict(args_dict)
    print(f"  Config: {args}")

    ternary_model = Model(args, group_size=128)
    print(f"  Ternary model created. Copying weights ...")
    copy_weights(orig_model, ternary_model)

    del orig_model
    mx.synchronize()
    print(f"  Done. Model ready for ternary training.")
    return ternary_model, tokenizer, args


def prepare_dataset(tokenizer, seq_len=512, split="train"):
    print(f"Loading train_data.txt ({split}) ...")
    with open("train_data.txt", "r") as f:
        full_text = f.read()

    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    split_idx = int(len(paragraphs) * 0.9)
    if split == "train":
        text = "\n\n".join(paragraphs[:split_idx])
    else:
        text = "\n\n".join(paragraphs[split_idx:])

    tokens = tokenizer.encode(text)
    print(f"  Tokenized {split}: {len(tokens)} tokens ({len(paragraphs[:split_idx] if split == 'train' else paragraphs[split_idx:])} paragraphs)")

    tokens = mx.array(tokens, dtype=mx.int32)
    n_seq = len(tokens) // (seq_len + 1)
    if n_seq == 0:
        n_seq = 1
    tokens = tokens[: n_seq * (seq_len + 1)]
    tokens = tokens.reshape(n_seq, seq_len + 1)

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    print(f"  Sequences: {n_seq}, seq_len={seq_len}")
    return inputs, targets


def train(
    model,
    tokenizer,
    args,
    train_inputs,
    train_targets,
    val_inputs,
    val_targets,
    num_steps=200,
    batch_size=2,
    lr=3e-4,
    warmup_steps=20,
    weight_decay=0.01,
    log_every=10,
    eval_every=50,
    grad_accum=1,
):
    optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay, betas=[0.9, 0.95])

    def loss_fn(model, inp, tgt):
        logits = model(inp)
        return cross_entropy(logits, tgt)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    n_train = train_inputs.shape[0]

    print(f"\n{'='*60}")
    print(f"Training: {num_steps} steps, batch_size={batch_size}, lr={lr}")
    print(f"{'='*60}\n")

    step = 0
    losses = []
    t0 = time.time()

    while step < num_steps:
        perm = mx.random.permutation(n_train)

        for i in range(0, n_train - batch_size + 1, batch_size):
            if step >= num_steps:
                break

            idx = perm[i : i + batch_size]
            inp = train_inputs[idx]
            tgt = train_targets[idx]

            if step < warmup_steps:
                current_lr = lr * (step + 1) / warmup_steps
            else:
                current_lr = lr
            optimizer.learning_rate = current_lr

            loss, grads = loss_and_grad(model, inp, tgt)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            loss_val = float(loss)
            losses.append(loss_val)

            if step % log_every == 0:
                elapsed = time.time() - t0
                ppl = math.exp(min(loss_val, 20))
                print(
                    f"  step {step:4d} | loss {loss_val:.4f} | "
                    f"ppl {ppl:.2f} | lr {current_lr:.2e} | "
                    f"{elapsed:.1f}s"
                )

            if eval_every > 0 and step > 0 and step % eval_every == 0:
                val_loss = evaluate(model, val_inputs, val_targets, batch_size=4)
                val_ppl = math.exp(min(val_loss, 20))
                print(f"  >>> EVAL step {step}: val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

            step += 1

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"  Final loss: {losses[-1]:.4f} (ppl={math.exp(min(losses[-1], 20)):.2f})")
    if len(losses) > 10:
        first_avg = sum(losses[:5]) / 5
        last_avg = sum(losses[-5:]) / 5
        print(f"  Loss improvement: {first_avg:.4f} -> {last_avg:.4f}")
    return losses


def evaluate(model, inputs, targets, batch_size=4):
    n = inputs.shape[0]
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, min(n, 20 * batch_size), batch_size):
        end = min(i + batch_size, n)
        inp = inputs[i:end]
        tgt = targets[i:end]
        logits = model(inp)
        loss = cross_entropy(logits, tgt)
        mx.eval(loss)
        total_loss += float(loss) * (end - i) * tgt.shape[1]
        total_tokens += (end - i) * tgt.shape[1]
    return total_loss / total_tokens


def verify_ternary_weights(model, group_size=128):
    print(f"\n{'='*60}")
    print("VERIFICATION: Checking ternary weight projection")
    print(f"{'='*60}")

    all_pass = True
    stats = {"-1": 0, "0": 0, "+1": 0, "total": 0}

    def _check(module, prefix=""):
        nonlocal all_pass
        for name, child in module.__dict__.items():
            full = f"{prefix}{name}" if prefix else name
            if isinstance(child, TernaryLinear):
                W = child.weight
                W_q, _ = project_ternary(W, group_size)

                vals = W_q.reshape(-1)
                is_valid = mx.all(
                    (vals == -1) | (vals == 0) | (vals == 1)
                )
                mx.eval(is_valid)

                n_neg = int(mx.sum(vals == -1))
                n_zero = int(mx.sum(vals == 0))
                n_pos = int(mx.sum(vals == 1))
                total = vals.size

                status = "PASS" if is_valid else "FAIL"
                if not is_valid:
                    all_pass = False

                print(
                    f"  [{status}] {full:50s}  "
                    f"-1:{n_neg/total:.1%}  0:{n_zero/total:.1%}  +1:{n_pos/total:.1%}"
                )

                stats["-1"] += n_neg
                stats["0"] += n_zero
                stats["+1"] += n_pos
                stats["total"] += total

            elif isinstance(child, TernaryEmbedding):
                W = child.weight
                W_q, _ = project_ternary(W, group_size)
                vals = W_q.reshape(-1)
                is_valid = mx.all(
                    (vals == -1) | (vals == 0) | (vals == 1)
                )
                mx.eval(is_valid)
                status = "PASS" if is_valid else "FAIL"
                if not is_valid:
                    all_pass = False
                n_neg = int(mx.sum(vals == -1))
                n_zero = int(mx.sum(vals == 0))
                n_pos = int(mx.sum(vals == 1))
                total = vals.size
                print(
                    f"  [{status}] {full:50s}  "
                    f"-1:{n_neg/total:.1%}  0:{n_zero/total:.1%}  +1:{n_pos/total:.1%}"
                )
                stats["-1"] += n_neg
                stats["0"] += n_zero
                stats["+1"] += n_pos
                stats["total"] += total

            elif isinstance(child, nn.Module):
                _check(child, f"{full}.")
            elif isinstance(child, list):
                for i, item in enumerate(child):
                    if isinstance(item, nn.Module):
                        _check(item, f"{full}.{i}.")

    _check(model)

    t = stats["total"]
    if t > 0:
        print(f"\n  Overall distribution: -1:{stats['-1']/t:.1%}  "
              f"0:{stats['0']/t:.1%}  +1:{stats['+1']/t:.1%}")

    print(f"\n  All weights ternary: {'YES' if all_pass else 'NO'}")
    return all_pass


def generate_text(model, tokenizer, prompt, max_tokens=80, seq_len=512):
    print(f"\n{'='*60}")
    print("TEXT GENERATION")
    print(f"{'='*60}")
    print(f"Prompt: {prompt!r}\n")

    tokens = tokenizer.encode(prompt)
    generated = list(tokens)

    for _ in range(max_tokens):
        context = generated[-seq_len:]
        input_ids = mx.array([context], dtype=mx.int32)
        logits = model(input_ids)
        next_logits = logits[0, -1, :]
        next_token = int(mx.argmax(next_logits))
        mx.eval(next_token)
        generated.append(next_token)

    text = tokenizer.decode(generated)
    print(f"Generated:\n{text}")
    return text


def measure_perplexity(model, inputs, targets, label="val", batch_size=4, max_batches=50):
    print(f"\n{'='*60}")
    print(f"PERPLEXITY MEASUREMENT ({label})")
    print(f"{'='*60}")

    n = min(inputs.shape[0], max_batches * batch_size)
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        inp = inputs[i:end]
        tgt = targets[i:end]
        logits = model(inp)
        loss = cross_entropy(logits, tgt)
        mx.eval(loss)
        total_loss += float(loss) * (end - i) * tgt.shape[1]
        total_tokens += (end - i) * tgt.shape[1]

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {ppl:.2f}")
    return ppl


def main():
    parser = argparse.ArgumentParser(description="Ternary Bonsai Training")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)

    model, tokenizer, model_args = load_model_and_tokenizer(args.model)

    train_inputs, train_targets = prepare_dataset(
        tokenizer, seq_len=args.seq_len, split="train"
    )
    val_inputs, val_targets = prepare_dataset(
        tokenizer, seq_len=args.seq_len, split="validation"
    )

    print("\n--- Pre-training verification ---")
    verify_ternary_weights(model)
    pre_ppl = measure_perplexity(
        model, val_inputs, val_targets, label="pre-training"
    )

    losses = train(
        model,
        tokenizer,
        model_args,
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        eval_every=args.eval_every,
        grad_accum=args.grad_accum,
    )

    print("\n--- Post-training verification ---")
    verify_ternary_weights(model)
    post_ppl = measure_perplexity(
        model, val_inputs, val_targets, label="post-training"
    )

    prompts = [
        "The capital of France is",
        "In mathematics, a prime number is",
        "The most important thing about",
    ]
    for p in prompts:
        generate_text(model, tokenizer, p, max_tokens=60, seq_len=args.seq_len)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Pre-training perplexity:  {pre_ppl:.2f}")
    print(f"  Post-training perplexity: {post_ppl:.2f}")
    print(f"  Loss trajectory: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  Training steps: {len(losses)}")


if __name__ == "__main__":
    main()
