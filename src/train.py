"""
P07 · LoRA Fine-tuning Script
Fine-tunes Phi-3-mini-4k-instruct on SRE runbook Q&A using QLoRA + PEFT.

Run this on a GPU machine (Colab/Kaggle free tier works):
    python src/train.py

Outputs:
    ./outputs/adapter/   <- LoRA adapter weights (push to HF Hub)
    ./outputs/logs/      <- training loss logs
"""

import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DATA_PATH = "data/raw/sre_runbook_qa.jsonl"
OUTPUT_DIR = "outputs/adapter"
LOG_DIR = "outputs/logs"

LORA_CONFIG = dict(
    r=16,                        # LoRA rank
    lora_alpha=32,               # scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

TRAINING_ARGS = dict(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
)

MAX_SEQ_LENGTH = 512


def format_prompt(example: dict) -> str:
    """Format a Q&A pair into Phi-3 instruction format."""
    return (
        f"<|user|>\n{example['question']}<|end|>\n"
        f"<|assistant|>\n{example['answer']}<|end|>"
    )


def load_dataset_from_jsonl(path: str) -> Dataset:
    """Load JSONL training data."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def tokenize(example: dict, tokenizer) -> dict:
    """Tokenize a single example."""
    text = format_prompt(example)
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def load_model_and_tokenizer():
    """Load base model with 4-bit quantization + LoRA."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train():
    print(f"Loading dataset from {DATA_PATH}...")
    dataset = load_dataset_from_jsonl(DATA_PATH)
    print(f"Dataset size: {len(dataset)} examples")

    print(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = load_model_and_tokenizer()

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        lambda x: tokenize(x, tokenizer),
        remove_columns=dataset.column_names,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOG_DIR,
        **TRAINING_ARGS,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete!")


if __name__ == "__main__":
    train()
