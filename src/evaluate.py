"""
P07 · Evaluation Script
Compares base model vs fine-tuned adapter on the test set.
Outputs ROUGE scores + qualitative before/after examples.

Usage:
    python src/evaluate.py \
        --adapter_path outputs/adapter \
        --test_data data/raw/sre_runbook_qa.jsonl \
        --output_path data/processed/eval_results.json
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
MAX_NEW_TOKENS = 256


def load_base_model(device: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    return model, tokenizer


def load_finetuned_model(adapter_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    return model, tokenizer


def generate(model, tokenizer, question: str, device: str) -> str:
    prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for k in scores:
            scores[k].append(result[k].fmeasure)
    return {k: round(sum(v) / len(v), 4) for k, v in scores.items()}


def evaluate(adapter_path: str, test_data_path: str, output_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load test data
    examples = []
    with open(test_data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    # Limit to 20 examples for eval speed
    examples = examples[:20]
    questions = [e["question"] for e in examples]
    references = [e["answer"] for e in examples]

    results = {"base_model": {}, "finetuned_model": {}, "examples": []}

    # Evaluate base model
    print("Evaluating base model...")
    base_model, base_tok = load_base_model(device)
    base_preds = [generate(base_model, base_tok, q, device) for q in questions]
    results["base_model"]["rouge"] = compute_rouge(base_preds, references)
    print(f"Base ROUGE: {results['base_model']['rouge']}")
    del base_model

    # Evaluate fine-tuned model
    print("Evaluating fine-tuned model...")
    ft_model, ft_tok = load_finetuned_model(adapter_path, device)
    ft_preds = [generate(ft_model, ft_tok, q, device) for q in questions]
    results["finetuned_model"]["rouge"] = compute_rouge(ft_preds, references)
    print(f"Fine-tuned ROUGE: {results['finetuned_model']['rouge']}")

    # Save before/after examples
    for i, (q, ref, base_p, ft_p) in enumerate(
        zip(questions, references, base_preds, ft_preds)
    ):
        results["examples"].append({
            "id": i,
            "question": q,
            "reference": ref,
            "base_answer": base_p,
            "finetuned_answer": ft_p,
        })

    # Compute improvement
    for metric in ["rouge1", "rouge2", "rougeL"]:
        base_score = results["base_model"]["rouge"][metric]
        ft_score = results["finetuned_model"]["rouge"][metric]
        improvement = round(((ft_score - base_score) / max(base_score, 1e-6)) * 100, 1)
        results[f"{metric}_improvement_pct"] = improvement

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print("\n── Summary ──────────────────────────────")
    print(f"{'Metric':<12} {'Base':>8} {'Fine-tuned':>12} {'Δ':>8}")
    print("-" * 44)
    for metric in ["rouge1", "rouge2", "rougeL"]:
        b = results["base_model"]["rouge"][metric]
        f = results["finetuned_model"]["rouge"][metric]
        d = results[f"{metric}_improvement_pct"]
        print(f"{metric:<12} {b:>8.4f} {f:>12.4f} {d:>+7.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default="outputs/adapter")
    parser.add_argument("--test_data", default="data/raw/sre_runbook_qa.jsonl")
    parser.add_argument("--output_path", default="data/processed/eval_results.json")
    args = parser.parse_args()
    evaluate(args.adapter_path, args.test_data, args.output_path)
