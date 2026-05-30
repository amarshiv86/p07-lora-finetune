# P07 · LoRA Fine-tuned LLM — Runbook Q&A

Fine-tunes **Phi-3-mini-4k-instruct** on runbook Q&A pairs using **QLoRA + PEFT**.
Part of the [Staff SRE · AI Engineer Portfolio](https://github.com/amarshiv86).

## Where things live

| What | Where |
|------|-------|
| Training + eval code | This repo (`src/`) |
| LoRA adapter weights | [HF Model Hub](https://huggingface.co/amarshiv86/p07-sre-lora-phi3) |
| Gradio demo | [HF Space](https://huggingface.co/spaces/amarshiv86/p07-sre-lora-demo) |
| Training data + eval results | [HF Dataset](https://huggingface.co/datasets/amarshiv86/p07-sre-lora-dataset) |

## Results — Before vs After

| Metric | Base Phi-3-mini | Fine-tuned | Improvement |
|--------|----------------|------------|-------------|
| ROUGE-1 | 0.2841 | 0.4712 | **+65.8%** |
| ROUGE-2 | 0.0923 | 0.2341 | **+153.6%** |
| ROUGE-L | 0.1876 | 0.3854 | **+105.4%** |

## SRE additions
- Training data covers real SRE topics: incident response, Kubernetes, SLOs, post-mortems
- ROUGE-score eval tracks before/after improvement quantitatively
- GPU cost per training run logged in eval results
- Adapter published to HF Hub with full model card

## How to train

**Requirements: GPU with 8GB+ VRAM (Colab/Kaggle free tier works)**

```bash
git clone https://github.com/amarshiv86/p07-lora-finetune
cd p07-lora-finetune
pip install -r requirements.txt
cp .env.example .env  # add your HF_TOKEN

# Train
python src/train.py

# Evaluate (before/after comparison)
python src/evaluate.py

# Push adapter to HF Hub
HF_TOKEN=hf_xxx python src/push_to_hub.py
```

## Run tests (no GPU needed)

```bash
pytest tests/ -v
```

## Project structure

```
p07-lora-finetune/
├── src/
│   ├── train.py          # QLoRA fine-tuning with PEFT
│   ├── evaluate.py       # Before/after ROUGE comparison
│   └── push_to_hub.py    # Push adapter + model card to HF Hub
├── tests/
│   └── test_train.py     # Unit tests (no GPU required)
├── hf_space/             # → deployed to HF Space
│   ├── app.py            # Gradio demo (local inference, no external API)
│   ├── README.md         # HF Space config (YAML front matter)
│   └── requirements.txt
├── data/                 # → deployed to HF Dataset
│   ├── raw/
│   │   └── sre_runbook_qa.jsonl    # 15 SRE Q&A training pairs
│   └── processed/
│       └── eval_results.json       # Before/after ROUGE scores
├── .github/workflows/
│   ├── ci.yml                  # Tests + lint on push
│   ├── deploy-hf-space.yml     # Auto-deploy hf_space/ → HF Space
│   └── deploy-hf-dataset.yml   # Auto-deploy data/ → HF Dataset
├── Dockerfile            # Local GPU training (not used by HF)
├── requirements.txt
├── .env.example
└── README.md
```

## Stack

`Phi-3-mini-4k-instruct` · `QLoRA` · `PEFT` · `HuggingFace Hub` · `Gradio` · `ROUGE` · `GitHub Actions`
