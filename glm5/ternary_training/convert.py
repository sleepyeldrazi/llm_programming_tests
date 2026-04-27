"""
Convert a pre-trained Qwen3-0.6B model to use ternary layers.

This script:
1. Loads the Qwen3-0.6B model
2. Creates a matching TernaryModel
3. Copies weights from the original model into the ternary model's latent weights
4. Saves the ternary model
"""

import argparse
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from .ternary_model import TernaryModel, ModelArgs
from .ternary_linear import TernaryLinear, TernaryEmbedding


def load_qwen3_config(model) -> ModelArgs:
    """Extract ModelArgs from a loaded Qwen3 model."""
    args = model.args
    return ModelArgs(
        model_type=args.model_type,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_attention_heads,
        rms_norm_eps=args.rms_norm_eps,
        vocab_size=args.vocab_size,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_position_embeddings,
        rope_theta=args.rope_theta,
        head_dim=args.head_dim,
        tie_word_embeddings=args.tie_word_embeddings,
        rope_scaling=args.rope_scaling,
    )


def copy_weights(src_model, dst_model):
    """Copy weights from source Qwen3 model to destination TernaryModel.
    
    Linear weights -> TernaryLinear.weight (latent weights)
    Embedding weights -> TernaryEmbedding.weight (latent weights)
    RMSNorm weights -> kept as-is (float16)
    """
    # Get source weights as dict
    src_weights = {}
    def collect_weights(module, prefix=''):
        for name in module:
            obj = module[name]
            full_name = f'{prefix}{name}'
            if isinstance(obj, nn.Linear):
                src_weights[f'{full_name}.weight'] = obj.weight
                if obj.bias is not None:
                    src_weights[f'{full_name}.bias'] = obj.bias
            elif isinstance(obj, nn.Embedding):
                src_weights[f'{full_name}.weight'] = obj.weight
            elif isinstance(obj, nn.RMSNorm):
                src_weights[f'{full_name}.weight'] = obj.weight
            elif isinstance(obj, nn.Module):
                collect_weights(obj, f'{full_name}.')
    
    collect_weights(src_model, '')
    
    # Set weights in destination model
    def set_weights(module, prefix=''):
        for name in module:
            obj = module[name]
            full_name = f'{prefix}{name}'
            if isinstance(obj, TernaryLinear):
                wkey = f'{full_name}.weight'
                if wkey in src_weights:
                    obj.weight = src_weights[wkey].astype(mx.float32)
                if obj.bias is not None:
                    bkey = f'{full_name}.bias'
                    if bkey in src_weights:
                        obj.bias = src_weights[bkey].astype(mx.float32)
            elif isinstance(obj, TernaryEmbedding):
                wkey = f'{full_name}.weight'
                if wkey in src_weights:
                    obj.weight = src_weights[wkey].astype(mx.float32)
            elif isinstance(obj, nn.RMSNorm):
                wkey = f'{full_name}.weight'
                if wkey in src_weights:
                    obj.weight = src_weights[wkey]
            # Skip RoPE and other non-parametric modules
    
    set_weights(dst_model, '')
    
    return dst_model


def convert_model(model_name="Qwen/Qwen3-0.6B", output_path=None):
    """Load Qwen3 and convert to ternary model."""
    print(f"Loading {model_name}...")
    src_model, tokenizer = load(model_name)
    
    print("Creating ternary model...")
    config = load_qwen3_config(src_model)
    dst_model = TernaryModel(config)
    
    print("Copying weights to ternary model...")
    copy_weights(src_model, dst_model)
    
    if output_path:
        print(f"Saving ternary model to {output_path}...")
        # Save weights
        weights = {}
        def collect_weights(module, prefix=''):
            for name in module:
                obj = module[name]
                full_name = f'{prefix}{name}'
                if isinstance(obj, (TernaryLinear, TernaryEmbedding)):
                    weights[f'{full_name}.weight'] = obj.weight
                elif isinstance(obj, nn.RMSNorm):
                    weights[f'{full_name}.weight'] = obj.weight
        
        collect_weights(dst_model, '')
        mx.save_safetensors(output_path + "/weights.safetensors", 
                            dict(zip(weights.keys(), weights.values())))
    
    print("Conversion complete!")
    return dst_model, tokenizer, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", default="./ternary_model")
    args = parser.parse_args()
    convert_model(args.model, args.output)