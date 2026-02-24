#!/usr/bin/env python3
"""Print config, layer structure, and parameter counts for a Hugging Face model."""

import argparse

from transformers import AutoConfig

from src.model import get_model_and_tokenizer


CONFIG_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "vocab_size",
    "max_position_embeddings",
    "intermediate_size",
)


def get_transformer_layers(model):
    """Return list of transformer layer modules if available (e.g. DistilBERT)."""
    try:
        base = getattr(model, "distilbert", None) or getattr(model, "bert", None) or getattr(model, "transformer", None)
        if base is not None:
            transformer = getattr(base, "transformer", base)
            layers = getattr(transformer, "layer", None)
            if layers is not None:
                return list(layers)
    except AttributeError:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Display model config, layers, and parameter counts")
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert-base-uncased",
        help="Model name or path (e.g. runs/distilbert)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every named module (full tree)",
    )
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model)
    model, _ = get_model_and_tokenizer(args.model)

    # Config
    print("Config")
    print("-" * 40)
    for key in CONFIG_KEYS:
        if hasattr(config, key):
            print(f"  {key}: {getattr(config, key)}")
    print()

    # Layers
    print("Transformer layers")
    print("-" * 40)
    layers = get_transformer_layers(model)
    if layers is not None:
        for i, layer in enumerate(layers):
            n = sum(p.numel() for p in layer.parameters())
            print(f"  layer[{i}]: {type(layer).__name__}  params={n:,}")
    else:
        print("  (no standard transformer.layer found)")
    if args.verbose:
        print("\nAll named modules (verbose)")
        print("-" * 40)
        for name, module in model.named_modules():
            if name:
                print(f"  {name}: {type(module).__name__}")
    print()

    # Parameters
    print("Parameters")
    print("-" * 40)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total: {total:,}")
    print(f"  trainable: {trainable:,}")


if __name__ == "__main__":
    main()
