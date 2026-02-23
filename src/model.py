"""DistilBERT sequence classification model and training."""

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL_NAME = "distilbert-base-uncased"


def get_model_and_tokenizer(model_name: str, num_labels: int, id2label: dict[int, str]):
    """Load tokenizer and sequence classification model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
    )
    return model, tokenizer


def train(
    train_dataset,
    eval_dataset,
    label2id: dict,
    id2label: dict,
    output_dir: str,
    model_name: str = DEFAULT_MODEL_NAME,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    **training_kwargs,
):
    """
    Train DistilBERT for sequence classification with Hugging Face Trainer.

    Saves checkpoints and final model to output_dir.
    """
    num_labels = len(label2id)
    model, tokenizer = get_model_and_tokenizer(model_name, num_labels, id2label)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        **training_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer
