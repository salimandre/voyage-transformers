#!/usr/bin/env python3
"""Print statistics for the VP corpus JSON."""

import argparse
from collections import Counter

from src.data_processing import load_corpus


def main():
    parser = argparse.ArgumentParser(description="Corpus statistics")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/vp_corpus_en_sample.json",
        help="Path to vp_corpus JSON",
    )
    args = parser.parse_args()

    records = load_corpus(args.data_path)
    n = len(records)
    print(f"Total records: {n}")

    # source_type distribution
    source_counts = Counter(r["source_type"] for r in records)
    print(f"\nUnique source_type: {len(source_counts)}")
    print("source_type distribution:")
    for label, count in source_counts.most_common():
        pct = 100 * count / n
        print(f"  {label}: {count} ({pct:.1f}%)")

    # text length (chars and words)
    lengths_chars = [len(r["sale_text_en"]) for r in records]
    lengths_words = [len(r["sale_text_en"].split()) for r in records]
    print(f"\nText length (characters): min={min(lengths_chars)}, max={max(lengths_chars)}, mean={sum(lengths_chars)/n:.0f}")
    print(f"Text length (words):      min={min(lengths_words)}, max={max(lengths_words)}, mean={sum(lengths_words)/n:.0f}")


if __name__ == "__main__":
    main()
