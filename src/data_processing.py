"""Load and preprocess travel corpus JSON for sequence classification."""

import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


def load_corpus(data_path: str) -> list[dict]:
    """Load JSON corpus from path. Expects list of dicts with sale_text_en and source_type."""
    path = Path(data_path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_label_mapping(records: list[dict]) -> tuple[dict[str, int], dict[int, str]]:
    """Build label2id and id2label from unique source_type values (sorted for stability)."""
    labels = sorted({r["source_type"] for r in records})
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def load_and_process(
    data_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 256,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Load JSON, build datasets, and tokenize for sequence classification.

    Returns:
        train_dataset, eval_dataset, label2id, id2label, num_labels
    """
    records = load_corpus(data_path)
    label2id, id2label = build_label_mapping(records)

    rows = [
        {"text": r["sale_text_en"].strip(), "label": label2id[r["source_type"]]}
        for r in records
    ]
    full = Dataset.from_list(rows)
    split = full.train_test_split(test_size=val_ratio, seed=seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )
    eval_dataset = eval_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    num_labels = len(label2id)
    return train_dataset, eval_dataset, label2id, id2label, num_labels
