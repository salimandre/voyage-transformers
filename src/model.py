"""DistilBERT masked language model finetuning."""

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL_NAME = "distilbert-base-uncased"


def get_model_and_tokenizer(model_name: str):
    """Load tokenizer and masked LM model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return model, tokenizer


def train(
    train_dataset,
    eval_dataset,
    tokenizer,
    output_dir: str,
    model_name: str = DEFAULT_MODEL_NAME,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    **training_kwargs,
):
    """
    Finetune DistilBERT as a masked language model on the corpus.

    Saves checkpoints and final model to output_dir.
    """
    model, _ = get_model_and_tokenizer(model_name)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

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
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer
