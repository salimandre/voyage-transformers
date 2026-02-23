"""Load and preprocess travel corpus JSON for language model finetuning."""

import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


def load_corpus(data_path: str) -> list[dict]:
    """Load JSON corpus from path. Expects list of dicts with sale_text_en (sale_uid and source_type are ids)."""
    path = Path(data_path)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_and_process(
    data_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 256,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Load JSON corpus (text only), tokenize for language model finetuning.

    Returns:
        train_dataset, eval_dataset
    """
    records = load_corpus(data_path)
    rows = [{"text": r["sale_text_en"].strip()} for r in records]
    full = Dataset.from_list(rows)
    split = full.train_test_split(test_size=val_ratio, seed=seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
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
    return train_dataset, eval_dataset
