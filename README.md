# voyage-transformers

Finetune BERT-style models (e.g. DistilBERT) on the travel corpus as a language model (masked LM). Corpus fields `sale_uid` and `source_type` are used only as identifiers; training uses the text (`sale_text_en`).

## Setup

```bash
pip install -r requirements.txt
```

## Run

**Model info** (config, layers, parameter counts):

```bash
python model_info.py
python model_info.py --model distilbert-base-uncased --verbose
```

**Corpus statistics** (optional):

```bash
python stats.py --data_path data/vp_corpus_en_sample.json
```

**Finetune DistilBERT** on the sample corpus (default: `data/vp_corpus_en_sample.json`):

```bash
python main.py
```

Options:

```bash
python main.py --data_path data/vp_corpus_en_sample.json --output_dir runs/distilbert --epochs 3 --batch_size 16
```

To train only the last 2 transformer layers (freeze the rest): `--num_trainable_layers 2`.

Outputs (checkpoints and final model) are written to `--output_dir`.
