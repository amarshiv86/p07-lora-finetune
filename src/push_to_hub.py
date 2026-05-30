"""
P07 · Push LoRA adapter to HuggingFace Hub

Usage:
    HF_TOKEN=hf_xxx python src/push_to_hub.py \
        --adapter_path outputs/adapter \
        --repo_id amarshiv86/p07-sre-lora-phi3
"""

import argparse
import os

from huggingface_hub import HfApi


MODEL_CARD = """---
language:
- en
license: mit
base_model: microsoft/Phi-3-mini-4k-instruct
tags:
- peft
- lora
- sre
- runbook
- fine-tuned
- qlora
pipeline_tag: text-generation
---

# P07 · SRE Runbook Q&A — LoRA Fine-tuned Phi-3-mini

Fine-tuned **Phi-3-mini-4k-instruct** on SRE runbook Q&A pairs using **QLoRA + PEFT**.
Part of the [Staff SRE · AI Engineer Portfolio](https://github.com/amarshiv86).

## Model details

| Field | Value |
|---|---|
| Base model | `microsoft/Phi-3-mini-4k-instruct` |
| Fine-tuning method | QLoRA (4-bit) + PEFT LoRA |
| LoRA rank | 16 |
| Target modules | `q_proj`, `v_proj` |
| Training epochs | 3 |
| Task | SRE Runbook Q&A |

## Before vs After (ROUGE scores)

See [eval_results.json](https://huggingface.co/datasets/amarshiv86/p07-sre-lora-dataset)
for full before/after comparison on the test set.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, "amarshiv86/p07-sre-lora-phi3")
tokenizer = AutoTokenizer.from_pretrained("amarshiv86/p07-sre-lora-phi3")

prompt = "<|user|>\\nWhat steps should I take when a pod is in CrashLoopBackOff?<|end|>\\n<|assistant|>\\n"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Training data

SRE runbook Q&A pairs covering: incident response, Kubernetes troubleshooting,
SLO/SLI definitions, on-call procedures, and post-mortem templates.

## Links

- [GitHub Repo](https://github.com/amarshiv86/p07-lora-finetune)
- [HF Space Demo](https://huggingface.co/spaces/amarshiv86/p07-sre-lora-demo)
- [Training Dataset](https://huggingface.co/datasets/amarshiv86/p07-sre-lora-dataset)
"""


def push(adapter_path: str, repo_id: str):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is required")

    api = HfApi(token=token)

    # Create model repo
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    # Write model card
    card_path = os.path.join(adapter_path, "README.md")
    with open(card_path, "w") as f:
        f.write(MODEL_CARD)

    # Upload adapter folder
    api.upload_folder(
        folder_path=adapter_path,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Adapter pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default="outputs/adapter")
    parser.add_argument("--repo_id", default="amarshiv86/p07-sre-lora-phi3")
    args = parser.parse_args()
    push(args.adapter_path, args.repo_id)
