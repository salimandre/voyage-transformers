#!/usr/bin/env python3
"""Print statistics for the travel corpus JSON."""

import argparse
from collections import Counter

from datasets import concatenate_datasets
from transformers import AutoTokenizer

from src.data_processing import load_and_process


def main():
    parser = argparse.ArgumentParser(description="Corpus statistics")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/vp_corpus_en_sample.json",
        help="Path to corpus JSON",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="distilbert-base-uncased",
        help="Tokenizer name (must match load_and_process)",
    )
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length used in tokenization")
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Validation ratio for split (0 = no split, use all for stats)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # train_test_split requires test_size in (0, 1); use a tiny value when 0 so we keep all data and concatenate
    val_ratio = args.val_ratio if args.val_ratio > 0 else 1e-6

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    train_dataset, eval_dataset, label2id, id2label, num_labels = load_and_process(
        args.data_path,
        tokenizer,
        max_length=args.max_length,
        val_ratio=val_ratio,
        seed=args.seed,
    )
    full = concatenate_datasets([train_dataset, eval_dataset])
    n = len(full)
    print(f"Total records: {n}")

    # source_type distribution (label may be tensor from set_format; map back with id2label)
    label_ids = full["label"]
    def _label_id(x):
        return int(x.item() if hasattr(x, "item") else x)
    source_counts = Counter(id2label[_label_id(i)] for i in label_ids)
    print(f"\nUnique source_type: {len(source_counts)}")
    print("source_type distribution:")
    for label, count in source_counts.most_common():
        pct = 100 * count / n
        print(f"  {label}: {count} ({pct:.1f}%)")

    # token length (real tokens per row from attention_mask)
    lengths_tokens = []
    for m in full["attention_mask"]:
        s = sum(m)
        lengths_tokens.append(int(s) if not hasattr(s, "item") else s.item())
    total_tokens = sum(lengths_tokens)
    print(f"\nText length (tokens, {args.tokenizer}): total={total_tokens}, min={min(lengths_tokens)}, max={max(lengths_tokens)}, mean={total_tokens/n:.0f}")


if __name__ == "__main__":
    main()
