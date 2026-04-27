"""
Train the ternary Qwen3 model on WikiText-2.

Training procedure:
- Loads Qwen3-0.6B, converts to ternary model
- Fine-tunes on WikiText-2 using cross-entropy loss
- Uses AdamW optimizer with linear warmup + cosine decay
- STE handles gradient flow through ternary projection
"""

import argparse
import math
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.tokenizer import Tokenizer

from .ternary_model import TernaryModel, ModelArgs
from .ternary_linear import ternary_projection, GROUP_SIZE
from .convert import load_qwen3_config, copy_weights


def load_wikitext2(tokenizer, seq_len=512, split="train"):
    """Load and tokenize WikiText-2 dataset."""
    from datasets import load_dataset
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    if split == "train":
        text = dataset["train"]["text"]
    elif split == "validation":
        text = dataset["validation"]["text"]
    else:
        text = dataset["test"]["text"]
    
    # Join all text
    full_text = "\n".join(text)
    
    # Tokenize
    tokens = tokenizer.encode(full_text)
    print(f"WikiText-2 {split}: {len(tokens)} tokens")
    
    return tokens


def create_batches(tokens, batch_size, seq_len):
    """Create batches of sequences for training."""
    # Total tokens per batch
    total_len = batch_size * seq_len
    n_batches = len(tokens) // total_len
    
    if n_batches == 0:
        raise ValueError(f"Not enough tokens ({len(tokens)}) for batch_size={batch_size}, seq_len={seq_len}")
    
    # Truncate to exact multiple
    tokens = tokens[:n_batches * total_len]
    
    # Reshape into batches
    tokens = np.array(tokens).reshape(n_batches, batch_size, seq_len)
    
    return tokens


def compute_loss(model, tokens, seq_len=512):
    """Compute cross-entropy loss on a batch of tokens."""
    input_ids = mx.array(tokens[:, :-1])
    targets = mx.array(tokens[:, 1:])
    
    # Forward pass
    logits = model(input_ids)
    
    # Cross-entropy loss
    loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
    
    return loss


def compute_perplexity(model, tokens, batch_size=4, seq_len=512):
    """Compute perplexity on a dataset."""
    total_loss = 0.0
    total_tokens = 0
    
    n_sequences = len(tokens) // seq_len
    
    for i in range(0, n_sequences - batch_size, batch_size):
        batch_tokens = []
        for j in range(batch_size):
            start = (i + j) * seq_len
            end = start + seq_len + 1
            if end > len(tokens):
                break
            batch_tokens.append(tokens[start:end])
        
        if len(batch_tokens) < batch_size:
            break
        
        batch = np.array(batch_tokens)
        loss = compute_loss(model, batch, seq_len)
        total_loss += float(loss) * batch_size
        total_tokens += batch_size * seq_len
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss


def generate_text(model, tokenizer, prompt, max_tokens=100, temp=0.8):
    """Generate text from the model for qualitative evaluation."""
    tokens = tokenizer.encode(prompt)
    
    for _ in range(max_tokens):
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        
        # Sample from last position
        last_logits = logits[:, -1, :] / temp
        
        # Apply softmax
        probs = mx.softmax(last_logits, axis=-1)
        
        # Sample
        next_token = mx.random.categorical(last_logits, axis=-1)
        tokens.append(int(next_token[0]))
    
    return tokenizer.decode(tokens)


class LRSchedule:
    """Learning rate schedule with linear warmup + cosine decay."""
    
    def __init__(self, base_lr, warmup_steps, total_steps, min_lr=1e-5):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
    
    def __call__(self, step):
        if step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay


def train(args):
    """Main training loop."""
    # Load original model for weight initialization
    print("Loading Qwen3-0.6B model...")
    src_model, tokenizer = load(args.model_name)
    
    # Create ternary model with same config
    print("Creating ternary model...")
    config = load_qwen3_config(src_model)
    model = TernaryModel(config)
    
    # Copy weights
    print("Copying weights to ternary latent weights...")
    copy_weights(src_model, src_model, config, model)
    
    # Free source model
    del src_model
    mx.clear_cache()
    
    # Load training data
    print("Loading WikiText-2 dataset...")
    train_tokens = load_wikitext2(tokenizer, seq_len=args.seq_len, split="train")
    val_tokens = load_wikitext2(tokenizer, seq_len=args.seq_len, split="validation")
    
    # Create batches
    train_batches = create_batches(train_tokens, args.batch_size, args.seq_len + 1)
    print(f"Training: {len(train_batches)} batches")
    
    # Set up optimizer
    lr_schedule = LRSchedule(
        base_lr=args.lr,
        warmup_steps=args.warmup,
        total_steps=args.steps,
        min_lr=args.min_lr,
    )
    
    # Collect trainable parameters
    def get_trainable_params(module):
        params = []
        for name in module:
            obj = module[name]
            if isinstance(obj, (nn.Module,)):
                params.extend(get_trainable_params(obj))
            elif hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
                params.append(obj)
        return params
    
    trainable_params = get_trainable_params(model)
    print(f"Total trainable parameters: {sum(p.size for p in trainable_params)}")
    
    # Optimizer
    optimizer = nn.optim.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Training state
    state = model.state
    
    step = 0
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {args.steps} steps...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Warmup steps: {args.warmup}")
    print(f"  Weight decay: {args.weight_decay}")
    print()
    
    start_time = time.time()
    
    while step < args.steps:
        # Shuffle batch order
        indices = np.random.permutation(len(train_batches))
        
        for batch_idx in indices:
            if step >= args.steps:
                break
            
            # Get batch
            batch = train_batches[batch_idx]
            
            # Update learning rate
            current_lr = lr_schedule(step)
            optimizer.learning_rate = current_lr
            
            # Forward + backward
            def loss_fn(model):
                return compute_loss(model, batch, args.seq_len)
            
            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            
            # Update
            optimizer.update(model, grads)
            
            # Evaluate
            mx.eval(loss, model.state)
            
            step += 1
            
            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = args.log_every * args.batch_size * args.seq_len / elapsed
                print(
                    f"Step {step:5d}/{args.steps} | "
                    f"Loss: {float(loss):.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Tokens/s: {tokens_per_sec:.0f}"
                )
                start_time = time.time()
            
            # Evaluation
            if step % args.eval_every == 0:
                print(f"\n--- Evaluating at step {step} ---")
                
                # Subsample validation tokens
                val_subset = val_tokens[:args.eval_size]
                val_batch = np.array([
                    val_subset[i:i + args.seq_len + 1]
                    for i in range(0, len(val_subset) - args.seq_len - 1, args.seq_len + 1)
                ][:args.batch_size])
                
                if len(val_batch) > 0:
                    val_loss = compute_loss(model, val_batch, args.seq_len)
                    mx.eval(val_loss)
                    val_ppl = math.exp(float(val_loss))
                    print(f"  Val loss: {float(val_loss):.4f} | Val perplexity: {val_ppl:.2f}")
                
                # Check ternary weights
                from .ternary_linear import verify_ternary_weights
                results = verify_ternary_weights(model)
                all_ternary = all(r['is_ternary'] for r in results.values())
                print(f"  All weights ternary: {all_ternary}")
                if not all_ternary:
                    for name, r in results.items():
                        if not r['is_ternary']:
                            print(f"    NOT TERNARY: {name} (max_round_error={r['max_round_error']:.6f})")
                
                # Generate text
                try:
                    prompt = "The history of the United States"
                    generated = generate_text(model, tokenizer, prompt, max_tokens=50)
                    print(f"  Generated: {generated[:200]}...")
                except Exception as e:
                    print(f"  Generation failed: {e}")
                
                print()
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    
    # Compute final training loss
    sample_batch = train_batches[0]
    final_loss = compute_loss(model, sample_batch, args.seq_len)
    mx.eval(final_loss)
    print(f"Final training loss: {float(final_loss):.4f}")
    print(f"Final training perplexity: {math.exp(float(final_loss)):.2f}")
    
    # Validate
    val_subset = val_tokens[:args.eval_size]
    val_losses = []
    for i in range(0, min(len(val_subset) - args.seq_len - 1, 2048), args.seq_len + 1):
        chunk = val_subset[i:i + args.seq_len + 1]
        if len(chunk) < args.seq_len + 1:
            continue
        batch = np.array([chunk])
        vl = compute_loss(model, batch, args.seq_len)
        mx.eval(vl)
        val_losses.append(float(vl))
    
    avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
    print(f"Average validation loss: {avg_val_loss:.4f}")
    print(f"Validation perplexity: {math.exp(avg_val_loss):.2f}")
    
    # Final ternary check
    results = verify_ternary_weights(model)
    all_ternary = all(r['is_ternary'] for r in results.values())
    print(f"\nAll weights ternary: {all_ternary}")
    for name, r in results.items():
        status = "OK" if r['is_ternary'] else "FAIL"
        print(f"  [{status}] {name}: shape={r['shape']}")
    
    # Generate text
    prompts = [
        "The capital of France is",
        "In the year 2024, artificial intelligence",
        "The most important thing about",
    ]
    print("\n--- Text Generation Samples ---")
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_tokens=80)
        print(f"Prompt: {prompt}")
        print(f"Output: {generated}")
        print()
    
    # Save model
    if args.save_path:
        print(f"Saving model to {args.save_path}...")
        import os
        os.makedirs(args.save_path, exist_ok=True)
        
        weights = {}
        def collect_weights(module, prefix=''):
            for name in module:
                obj = module[name]
                full_name = f'{prefix}{name}'
                if isinstance(obj, (TernaryLinear, TernaryEmbedding)):
                    weights[f'{full_name}.weight'] = obj.weight
                elif isinstance(obj, nn.RMSNorm):
                    weights[f'{full_name}.weight'] = obj.weight
        collect_weights(model, '')
        
        mx.save_safetensors(
            os.path.join(args.save_path, "weights.safetensors"),
            dict(zip(weights.keys(), weights.values()))
        )
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-size", type=int, default=10000)
    parser.add_argument("--save-path", default="./ternary_trained")
    args = parser.parse_args()
    
    train(args)