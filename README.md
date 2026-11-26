# SmolLM2 Clinical LoRA Experiment

Experiment with LoRA fine-tuning `HuggingFaceTB/SmolLM2-135M` on clinical notes from `AGBonnet/augmented-clinical-notes`. The default config uses 2,500 notes, a 90/10 token-level train/validation split, and 300 training steps.

Due to local resource constraints, training experiments were run in Google Colab using a T4 GPU.

## Setup and run

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/train_lora.py
```

Training reports train/validation loss and perplexity, then saves the adapter to the configured `output_dir` (`adapters/smollm2-135m-clinical-lora` by default).

After training, compare base and LoRA-generated continuations:

```bash
python src/compare.py
```

Model and dataset files are downloaded from Hugging Face. Adjust experiment settings in `configs/lora_config.json`. Generated text is experimental and not for clinical use.
