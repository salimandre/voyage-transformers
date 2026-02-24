"""DistilBERT masked language model finetuning."""

from __future__ import annotations

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


def _freeze_first_layers_train_last(
    model,
    num_trainable_layers: int | None,
):
    """Freeze all parameters, then unfreeze last N transformer layers and MLM head (DistilBERT)."""
    if num_trainable_layers is None:
        return
    num_hidden_layers = model.config.num_hidden_layers
    n = min(num_trainable_layers, num_hidden_layers)
    for p in model.parameters():
        p.requires_grad = False
    transformer_layers = model.distilbert.transformer.layer
    for i in range(num_hidden_layers - n, num_hidden_layers):
        for p in transformer_layers[i].parameters():
            p.requires_grad = True
    for mod_name in ("vocab_transform", "vocab_layer_norm", "vocab_projector"):
        m = getattr(model, mod_name, None)
        if m is not None:
            for p in m.parameters():
                p.requires_grad = True


def train(
    train_dataset,
    eval_dataset,
    tokenizer,
    output_dir: str,
    model_name: str = DEFAULT_MODEL_NAME,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    num_trainable_layers: int | None = None,
    **training_kwargs,
):
    """
    Finetune DistilBERT as a masked language model on the corpus.

    Saves checkpoints and final model to output_dir.
    """
    model, _ = get_model_and_tokenizer(model_name)
    _freeze_first_layers_train_last(model, num_trainable_layers)
    if num_trainable_layers is not None:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_trainable:,}")
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
