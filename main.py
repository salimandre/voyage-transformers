#!/usr/bin/env python3
"""Entrypoint: finetune DistilBERT on vp_corpus for source_type classification."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from src.data_processing import load_and_process
from src.model import train


def main():
    parser = argparse.ArgumentParser(description="Finetune DistilBERT on vp_corpus (source_type classification)")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/vp_corpus_en_sample.json",
        help="Path to vp_corpus JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/distilbert-vp",
        help="Directory for checkpoints and final model",
    )
    parser.add_argument("--max_length", type=int, default=256, help="Max token length")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_dataset, eval_dataset, label2id, id2label, num_labels = load_and_process(
        args.data_path,
        tokenizer,
        max_length=args.max_length,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(
        train_dataset,
        eval_dataset,
        label2id=label2id,
        id2label=id2label,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
