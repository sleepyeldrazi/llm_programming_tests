import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.models.qwen3 import Model
import numpy as np
from typing import Optional, Tuple, List
import time
import json

# ==============================================================================
# Ternary Linear Layer with Straight-Through Estimator (STE)
# ==============================================================================

class TernaryLinear(nn.Module):
    """
    Ternary linear layer: weights are projected to {-1, 0, +1} * scale
    during forward pass, with STE for backward pass.
    
    Group-wise quantization: groups of `group_size` weights share one FP32 scale factor.
    Scale factor: s = mean(|W_group|)
    """
    def __init__(self, in_features: int, out_features: int, group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        if in_features % group_size != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by group_size ({group_size})")
        
        self.num_groups = in_features // group_size
        
        # Latent weights in float32 (trainable)
        scale = (1.0 / in_features) ** 0.5
        self.weight = mx.random.normal((out_features, in_features), scale=scale)
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128):
        """Initialize from an existing Linear layer."""
        in_features = linear.weight.shape[1]
        out_features = linear.weight.shape[0]
        layer = cls(in_features, out_features, group_size)
        # Reinitialize weights randomly for training from scratch
        # rather than copying pretrained weights
        scale = (1.0 / in_features) ** 0.5
        layer.weight = mx.random.normal((out_features, in_features), scale=scale)
        return layer
    
    def _quantize(self, weight):
        """
        Project latent weights to ternary using group-wise scales.
        """
        # Reshape to (out_features, num_groups, group_size)
        w_reshaped = weight.reshape(self.out_features, self.num_groups, self.group_size)
        
        # Compute scale per group: s = mean(|W|)
        scales = mx.mean(mx.abs(w_reshaped), axis=-1, keepdims=True)
        
        # Quantize to {-1, 0, +1}
        epsilon = 1e-8
        w_norm = w_reshaped / (scales + epsilon)
        w_quant = mx.clip(mx.round(w_norm), -1, 1)
        
        # Dequantize back
        w_ternary = w_quant * scales
        
        return w_ternary.reshape(self.out_features, self.in_features), scales
    
    def __call__(self, x):
        """Forward pass with STE."""
        w_ternary, _ = self._quantize(mx.stop_gradient(self.weight))
        
        # STE: forward uses ternary, backward uses latent
        w_effective = w_ternary + (self.weight - mx.stop_gradient(self.weight))
        
        return x @ w_effective.T
    
    def get_ternary_weights(self):
        """Get the actual ternary-projected weights."""
        w_ternary, scales = self._quantize(self.weight)
        return w_ternary, scales
    
    def verify_ternary(self, tol=1e-3):
        """Verify that weights project cleanly to {-1, 0, +1} * scale."""
        w_ternary, scales = self.get_ternary_weights()
        w_reshaped = w_ternary.reshape(self.out_features, self.num_groups, self.group_size)
        
        w_norm = w_reshaped / (scales + 1e-8)
        w_rounded = mx.round(w_norm)
        
        is_valid_value = mx.all(
            (mx.abs(w_rounded - (-1.0)) < 1e-3) | 
            (mx.abs(w_rounded - 0.0) < 1e-3) | 
            (mx.abs(w_rounded - 1.0) < 1e-3)
        )
        
        is_ternary = mx.all(mx.abs(w_norm - w_rounded) < tol)
        
        return is_ternary.item() and is_valid_value.item()


# ==============================================================================
# Model Conversion Utilities
# ==============================================================================

def convert_qwen3_to_ternary(model: Model, group_size: int = 128) -> Model:
    """
    Convert all linear layers in a Qwen3 model to ternary.
    Keeps RMSNorm and embeddings in float.
    """
    print("Converting model to ternary...")
    
    # Skip embedding - it's an Embedding layer, not Linear
    if hasattr(model.model, 'embed_tokens'):
        print(f"  Skipping embedding (not Linear): {model.model.embed_tokens.weight.shape}")
    
    # Convert each transformer block
    for i, layer in enumerate(model.model.layers):
        print(f"\n  Layer {i}:")
        
        # Attention projections
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(attn, proj_name):
                    proj = getattr(attn, proj_name)
                    if isinstance(proj, nn.Linear):
                        setattr(attn, proj_name, TernaryLinear.from_linear(proj, group_size))
                        print(f"    {proj_name}: {proj.weight.shape}")
        
        # MLP projections
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(mlp, proj_name):
                    proj = getattr(mlp, proj_name)
                    if isinstance(proj, nn.Linear):
                        setattr(mlp, proj_name, TernaryLinear.from_linear(proj, group_size))
                        print(f"    {proj_name}: {proj.weight.shape}")
    
    # Skip LM head if tied or not Linear
    if hasattr(model, 'lm_head'):
        lm = model.lm_head
        if isinstance(lm, nn.Linear):
            in_features = lm.weight.shape[1]
            if in_features % group_size == 0:
                model.lm_head = TernaryLinear.from_linear(lm, group_size)
                print(f"  Converting lm_head: {lm.weight.shape}")
            else:
                print(f"  Skipping lm_head (not divisible): {lm.weight.shape}")
        else:
            print(f"  Skipping lm_head (not Linear): {type(lm)}")
    
    print("\nConversion complete!")
    return model


def count_ternary_layers(model):
    """Count the number of TernaryLinear layers in the model."""
    count = 0
    def count_module(module):
        nonlocal count
        if isinstance(module, TernaryLinear):
            count += 1
        if hasattr(module, 'items'):
            for _, child in module.items():
                count_module(child)
        elif isinstance(module, list):
            for child in module:
                count_module(child)
    count_module(model)
    return count


# ==============================================================================
# Verification
# ==============================================================================

def verify_model_ternary(model: Model) -> Tuple[bool, List[str]]:
    """Verify all TernaryLinear layers produce clean ternary weights."""
    all_pass = True
    failed_layers = []
    
    def check_module(module, name=""):
        nonlocal all_pass
        if isinstance(module, TernaryLinear):
            is_ternary = module.verify_ternary()
            if not is_ternary:
                all_pass = False
                failed_layers.append(name)
                print(f"  FAIL: {name}")
            else:
                print(f"  PASS: {name}")
        
        if hasattr(module, 'items'):
            for child_name, child in module.items():
                check_module(child, f"{name}.{child_name}" if name else child_name)
        elif isinstance(module, list):
            for i, child in enumerate(module):
                check_module(child, f"{name}[{i}]" if name else f"[{i}]")
    
    check_module(model)
    return all_pass, failed_layers


# ==============================================================================
# Dataset Utilities
# ==============================================================================

def load_wikitext_data(tokenizer, split="train", max_samples=1000, seq_length=256):
    """Load WikiText-2 dataset and tokenize."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    except Exception as e:
        print(f"Could not load dataset: {e}")
        print("Using fallback sample text...")
        return create_fallback_data(tokenizer, seq_length)
    
    # Tokenize
    all_tokens = []
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        text = example["text"].strip()
        if len(text) < 50:  # Skip very short lines
            continue
        tokens = tokenizer.encode(text)
        if len(tokens) > 10:
            all_tokens.append(tokens)
    
    print(f"Loaded {len(all_tokens)} sequences from WikiText-2 {split}")
    return all_tokens


def create_fallback_data(tokenizer, seq_length=256, num_samples=500):
    """Create simple fallback training data."""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. ",
        "In machine learning, neural networks are powerful models. ",
        "The Earth orbits around the Sun in an elliptical path. ",
        "Python is a popular programming language for data science. ",
        "The history of artificial intelligence dates back to the 1950s. ",
        "Deep learning models can process images, text, and speech. ",
        "The capital of France is Paris, known for the Eiffel Tower. ",
        "Water boils at 100 degrees Celsius at standard pressure. ",
        "The human brain contains approximately 86 billion neurons. ",
        "Quantum computing uses quantum bits to perform calculations. ",
    ]
    
    all_tokens = []
    for i in range(num_samples):
        text = " ".join(sample_texts[i % len(sample_texts)] * 20)
        tokens = tokenizer.encode(text)[:seq_length]
        if len(tokens) > 10:
            all_tokens.append(tokens)
    
    print(f"Created {len(all_tokens)} fallback sequences")
    return all_tokens


def create_batches(token_sequences, batch_size=4, seq_length=256):
    """Create batches of token sequences."""
    batches = []
    current_batch = []
    
    for tokens in token_sequences:
        if len(tokens) < 2:
            continue
        # Truncate or pad to seq_length
        if len(tokens) > seq_length:
            tokens = tokens[:seq_length]
        else:
            tokens = tokens + [0] * (seq_length - len(tokens))
        current_batch.append(tokens)
        
        if len(current_batch) == batch_size:
            batches.append(mx.array(current_batch))
            current_batch = []
    
    if current_batch:
        # Pad last batch
        while len(current_batch) < batch_size:
            current_batch.append([0] * seq_length)
        batches.append(mx.array(current_batch))
    
    return batches


# ==============================================================================
# Training Utilities
# ==============================================================================

def loss_fn(model, inputs, targets):
    """Compute cross-entropy loss for next-token prediction."""
    logits = model(inputs)
    # logits shape: (batch, seq_len, vocab_size)
    
    # Flatten
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    
    # Cross entropy
    probs = mx.softmax(logits_flat, axis=-1)
    log_probs = mx.log(probs + 1e-10)
    
    # Use advanced indexing instead of mx.take
    # log_probs has shape (batch*seq, vocab)
    # targets_flat has shape (batch*seq,)
    # We want log_probs[i, targets_flat[i]] for each i
    batch_seq_len = logits_flat.shape[0]
    indices = mx.arange(batch_seq_len)
    target_log_probs = log_probs[indices, targets_flat]
    nll = -target_log_probs
    
    # Mask padding
    mask = targets_flat >= 0
    nll = nll * mask
    
    return mx.sum(nll) / mx.sum(mask)


def step(model, inputs, targets, optimizer):
    """Single training step."""
    loss_and_grad = mx.value_and_grad(loss_fn)
    loss, grads = loss_and_grad(model, inputs, targets)
    
    # Update parameters
    optimizer.update(model, grads)
    
    return loss


def compute_perplexity(model, tokens_batch):
    """Compute perplexity on a batch of token sequences."""
    total_loss = 0.0
    total_tokens = 0
    
    for tokens in tokens_batch:
        if len(tokens) < 2:
            continue
        inputs = mx.array(tokens[:-1])
        targets = mx.array(tokens[1:])
        
        logits = model(inputs[None, :])
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        
        probs = mx.softmax(logits_flat, axis=-1)
        log_probs = mx.log(probs + 1e-10)
        
        # Use advanced indexing
        seq_len = logits_flat.shape[0]
        indices = mx.arange(seq_len)
        target_log_probs = log_probs[indices, targets_flat]
        nll = -target_log_probs
        
        total_loss += mx.sum(nll).item()
        total_tokens += len(targets_flat)
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity


def generate_text(model, tokenizer, prompt, max_tokens=30, temperature=1.0, top_k=None):
    """Generate text from prompt using greedy or top-k sampling."""
    tokens = mx.array(tokenizer.encode(prompt))
    
    for _ in range(max_tokens):
        logits = model(tokens[None, :])
        next_token_logits = logits[0, -1, :] / temperature
        
        if top_k is not None and top_k > 0:
            # Top-k filtering
            top_k_values, top_k_indices = mx.topk(next_token_logits, top_k)
            mask = mx.zeros_like(next_token_logits)
            mask = mask.at[top_k_indices].set(1.0)
            filtered_logits = next_token_logits * mask + (1 - mask) * (-1e10)
            probs = mx.softmax(filtered_logits)
            next_token = mx.argmax(probs)
        else:
            # Greedy
            next_token = mx.argmax(next_token_logits)
        
        tokens = mx.concatenate([tokens, next_token[None]])
    
    return tokenizer.decode(tokens.tolist())


# ==============================================================================
# Main Training Script
# ==============================================================================

def main():
    print("=" * 80)
    print("Ternary Bonsai Training - Qwen3-0.6B")
    print("=" * 80)
    
    # Hyperparameters
    GROUP_SIZE = 128
    SEQ_LENGTH = 128
    BATCH_SIZE = 2  # Small batch for M4 Mac
    NUM_STEPS = 500
    LEARNING_RATE = 5e-5
    WARMUP_STEPS = 50
    EVAL_EVERY = 50
    GRAD_CLIP = 1.0
    
    print(f"\nHyperparameters:")
    print(f"  Group size: {GROUP_SIZE}")
    print(f"  Sequence length: {SEQ_LENGTH}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Training steps: {NUM_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Warmup steps: {WARMUP_STEPS}")
    print(f"  Grad clip: {GRAD_CLIP}")
    
    # Load model
    print("\n[1/6] Loading Qwen3-0.6B...")
    model, tokenizer = load("Qwen/Qwen3-0.6B")
    print(f"Model loaded successfully")
    
    # Convert to ternary
    print("\n[2/6] Converting to ternary...")
    model = convert_qwen3_to_ternary(model, group_size=GROUP_SIZE)
    print(f"Converted {count_ternary_layers(model)} linear layers to ternary")
    
    # Verify
    print("\n[3/6] Verifying ternary projection...")
    all_pass, failed = verify_model_ternary(model)
    if all_pass:
        print("All layers pass ternary verification!")
    else:
        print(f"Failed layers: {failed}")
        return
    
    # Load dataset
    print("\n[4/6] Loading dataset...")
    train_data = load_wikitext_data(tokenizer, split="train", max_samples=2000, seq_length=SEQ_LENGTH)
    val_data = load_wikitext_data(tokenizer, split="validation", max_samples=200, seq_length=SEQ_LENGTH)
    
    train_batches = create_batches(train_data, batch_size=BATCH_SIZE, seq_length=SEQ_LENGTH)
    print(f"Created {len(train_batches)} training batches")
    
    # Test generation before training
    print("\n[5/6] Testing generation (pre-training)...")
    prompt = "The quick brown fox"
    generated = generate_text(model, tokenizer, prompt, max_tokens=20)
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{generated}'")
    
    # Initialize optimizer
    print("\n[6/6] Starting training...")
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE)
    
    # Training loop
    losses = []
    start_time = time.time()
    
    def get_lr(step_num):
        """Learning rate schedule with warmup and cosine decay."""
        if step_num < WARMUP_STEPS:
            return LEARNING_RATE * (step_num + 1) / WARMUP_STEPS
        else:
            progress = (step_num - WARMUP_STEPS) / (NUM_STEPS - WARMUP_STEPS)
            return LEARNING_RATE * 0.5 * (1 + np.cos(np.pi * progress))
    
    for step_num in range(NUM_STEPS):
        # Update learning rate
        current_lr = get_lr(step_num)
        optimizer.learning_rate = current_lr
        
        # Get batch
        batch_idx = step_num % len(train_batches)
        batch = train_batches[batch_idx]
        
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        # Training step with gradient clipping
        loss_and_grad = mx.value_and_grad(loss_fn)
        loss, grads = loss_and_grad(model, inputs, targets)
        
        # Gradient clipping
        if GRAD_CLIP > 0:
            def clip_grads(g):
                if isinstance(g, dict):
                    return {k: clip_grads(v) for k, v in g.items()}
                elif isinstance(g, list):
                    return [clip_grads(v) for v in g]
                else:
                    return mx.clip(g, -GRAD_CLIP, GRAD_CLIP)
            grads = clip_grads(grads)
        
        optimizer.update(model, grads)
        mx.eval(loss)
        
        losses.append(loss.item())
        
        # Logging
        if (step_num + 1) % 10 == 0:
            avg_loss = np.mean(losses[-10:])
            print(f"Step {step_num + 1}/{NUM_STEPS} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Time: {time.time() - start_time:.1f}s")
        
        # Evaluation
        if (step_num + 1) % EVAL_EVERY == 0:
            print(f"\n--- Evaluation at step {step_num + 1} ---")
            
            # Generate sample
            prompt = "Artificial intelligence is"
            generated = generate_text(model, tokenizer, prompt, max_tokens=30, temperature=0.8)
            print(f"Prompt: '{prompt}'")
            print(f"Generated: '{generated}'")
            
            # Compute perplexity on small validation set
            if val_data:
                ppl = compute_perplexity(model, val_data[:20])
                print(f"Perplexity: {ppl:.2f}")
            
            # Verify ternary
            all_pass, _ = verify_model_ternary(model)
            print(f"Ternary verification: {'PASS' if all_pass else 'FAIL'}")
            print("-" * 40 + "\n")
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    # Loss curve
    print(f"\nInitial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss decrease: {losses[0] - losses[-1]:.4f}")
    
    # Generate multiple samples
    prompts = [
        "The capital of France is",
        "Machine learning is a type of",
        "In 1492, Christopher Columbus",
    ]
    
    print("\n--- Generation Samples ---")
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_tokens=30, temperature=0.8)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        print()
    
    # Perplexity
    if val_data:
        ppl = compute_perplexity(model, val_data[:50])
        print(f"Final perplexity: {ppl:.2f}")
    
    # Verify ternary one final time
    print("\n--- Ternary Verification ---")
    all_pass, failed = verify_model_ternary(model)
    print(f"All layers ternary: {all_pass}")
    if failed:
        print(f"Failed: {failed}")
    
    # Save results
    results = {
        "hyperparameters": {
            "group_size": GROUP_SIZE,
            "seq_length": SEQ_LENGTH,
            "batch_size": BATCH_SIZE,
            "num_steps": NUM_STEPS,
            "learning_rate": LEARNING_RATE,
        },
        "training": {
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "loss_curve": [float(l) for l in losses],
        },
        "verification": {
            "all_ternary": all_pass,
            "failed_layers": failed,
        },
        "perplexity": float(ppl) if val_data else None,
    }
    
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to training_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
