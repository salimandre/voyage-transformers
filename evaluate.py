#!/usr/bin/env python3
"""Evaluate a finetuned masked LM: load checkpoint and report eval loss and perplexity.

Uses the same data split and tokenization as training (same --val_ratio and --seed)
so results are comparable."""

import argparse
import math
import sys
from pathlib import Path

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.data_processing import load_and_process


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a finetuned model on the corpus (eval loss and perplexity)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert-base-uncased",
        help="Path to saved checkpoint or Hugging Face model id (default: vanilla pretrained)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/vp_corpus_en_sample.json",
        help="Path to corpus JSON",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max token length (must match training)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (must match training for comparable eval)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split (must match training)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Per-device eval batch size",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if model_path.is_dir() and not (model_path / "config.json").exists():
        print(
            f"Model path '{args.model}' does not contain config.json. Run training first (python main.py) or set --model to a valid checkpoint or hub id (e.g. distilbert-base-uncased).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load from local checkpoint or from Hugging Face hub (e.g. vanilla pretrained)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    _, eval_dataset = load_and_process(
        args.data_path,
        tokenizer,
        max_length=args.max_length,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    output_dir = args.model if (model_path.is_dir() and (model_path / "config.json").exists()) else "eval_out"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss", float("nan"))
    perplexity = math.exp(eval_loss)

    print(f"Eval loss: {eval_loss:.4f}")
    print(f"Eval perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    main()
