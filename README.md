# SmolLM2 Clinical LoRA Experiment

A small experimentation project for fine-tuning `HuggingFaceTB/SmolLM2-135M` on clinical notes with LoRA.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Edit settings in `configs/lora_config.json`, then train:

```bash
python src/train_lora.py
```

The base model and dataset are downloaded automatically from Hugging Face. Checkpoints are saved to the configured `output_dir`.
