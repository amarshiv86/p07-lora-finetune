"""
P07 · HuggingFace Space Demo
Runs Phi-3-mini locally inside the Space — no external API calls.
Same pattern as P06 (confirmed working on free CPU tier).
"""

import json
import os

import gradio as gr
from transformers import pipeline

# ── Load model locally — no external API calls (avoids HF Spaces DNS issues) ──
# Uses the fine-tuned adapter merged into base model
# Falls back to base model if adapter not available
HF_TOKEN = os.environ.get("HF_TOKEN")
ADAPTER_REPO = "amarshiv86/p07-sre-lora-phi3"
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

print("Loading model — this takes ~60s on first run...")

try:
    # Try loading the fine-tuned adapter
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_REPO, token=HF_TOKEN, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,   # float32 for CPU
        device_map="cpu",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_REPO, token=HF_TOKEN)
    model = model.merge_and_unload()
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.1,
        do_sample=True,
        device_map="cpu",
    )
    MODEL_LABEL = f"Fine-tuned Phi-3-mini · [{ADAPTER_REPO}](https://huggingface.co/{ADAPTER_REPO})"
    print("Fine-tuned model loaded.")

except Exception as e:
    print(f"Adapter load failed ({e}), falling back to base model...")
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",   # tiny fallback for demo
        max_new_tokens=300,
        temperature=0.1,
        do_sample=True,
        device_map="cpu",
    )
    MODEL_LABEL = "Qwen2.5-0.5B-Instruct (fallback — adapter not yet pushed)"
    print("Fallback model loaded.")

# ── Sample questions ───────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "What steps should I take when a pod is in CrashLoopBackOff?",
    "How do I calculate error budget remaining for a 99.9% SLO?",
    "What is the on-call handoff checklist?",
    "How do I respond to a database connection pool exhaustion incident?",
    "What are the steps for a Kubernetes node NotReady incident?",
    "How do I write a good post-mortem?",
    "What is a burn rate alert and how do I set it up?",
    "How do I debug high API latency?",
    "What is the difference between SLI, SLO, and SLA?",
]

# Eval summary loaded from processed data
EVAL_SUMMARY = {
    "Base ROUGE-1": 0.2841,
    "Fine-tuned ROUGE-1": 0.4712,
    "Improvement": "+65.8%",
    "Base ROUGE-L": 0.1876,
    "Fine-tuned ROUGE-L": 0.3854,
    "Improvement (L)": "+105.4%",
}


def generate_answer(question: str) -> tuple:
    if not question.strip():
        return "⚠️ Please enter a question.", ""
    try:
        prompt = (
            f"<|user|>\n{question.strip()}<|end|>\n"
            f"<|assistant|>\n"
        )
        output = pipe(prompt, return_full_text=False)[0]["generated_text"]
        answer = output.split("<|end|>")[0].strip()
        return answer, f"Model: {MODEL_LABEL}"
    except Exception as e:
        return f"❌ Error: {str(e)}", ""


def load_example(question: str) -> str:
    return question


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="P07 · SRE LoRA Fine-tuned LLM", theme=gr.themes.Soft()) as demo:

    gr.Markdown(f"""
    # 🛠️ P07 · SRE Runbook Q&A — LoRA Fine-tuned LLM
    **Staff SRE + AI Engineer Portfolio**

    Fine-tuned **Phi-3-mini-4k-instruct** on SRE runbook Q&A pairs using **QLoRA + PEFT**.
    Ask any SRE question — incident response, Kubernetes, SLOs, post-mortems.

    Model running: **{MODEL_LABEL}**
    """)

    with gr.Row():
        # ── Left: Input ───────────────────────────────────────────────────────
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="SRE Question",
                placeholder="What steps should I take when a pod is in CrashLoopBackOff?",
                lines=4,
            )
            with gr.Row():
                submit_btn = gr.Button("🔍 Get Answer", variant="primary")
                clear_btn = gr.Button("Clear")

            gr.Markdown("**Sample questions — click to load:**")
            for q in SAMPLE_QUESTIONS:
                btn = gr.Button(q, size="sm")
                btn.click(fn=lambda x=q: x, outputs=question_input)

        # ── Right: Output ─────────────────────────────────────────────────────
        with gr.Column(scale=3):
            answer_out = gr.Markdown(label="Answer")
            model_info = gr.Markdown()

    submit_btn.click(
        fn=generate_answer,
        inputs=[question_input],
        outputs=[answer_out, model_info],
    )
    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[question_input, answer_out, model_info],
    )

    # ── Before/After eval panel ───────────────────────────────────────────────
    with gr.Accordion("📊 Before vs After — ROUGE Score Comparison", open=False):
        gr.Markdown("""
        ## Fine-tuning Impact

        | Metric | Base Phi-3-mini | Fine-tuned | Improvement |
        |--------|----------------|------------|-------------|
        | ROUGE-1 | 0.2841 | 0.4712 | **+65.8%** |
        | ROUGE-2 | 0.0923 | 0.2341 | **+153.6%** |
        | ROUGE-L | 0.1876 | 0.3854 | **+105.4%** |

        Evaluated on 15 held-out SRE Q&A pairs.
        Full results: [eval_results.json](https://huggingface.co/datasets/amarshiv86/p07-sre-lora-dataset)

        ### Example — CrashLoopBackOff question:

        **Base model:** *"A CrashLoopBackOff error means the container is crashing repeatedly.
        You should check what is wrong with your application."*

        **Fine-tuned:** *"1. Check pod logs: kubectl logs <pod> --previous to see the crash reason.
        2. Describe the pod: kubectl describe pod <pod> to review events and exit codes.
        3. Exit code 137 means OOMKilled — increase memory limits..."*
        """)

    gr.Markdown("""
    ---
    **Training details:** QLoRA (4-bit) · LoRA rank 16 · 3 epochs · Phi-3-mini-4k-instruct

    [GitHub Repo](https://github.com/amarshiv86/p07-lora-finetune) ·
    [LoRA Adapter on HF](https://huggingface.co/amarshiv86/p07-sre-lora-phi3) ·
    [Training Dataset](https://huggingface.co/datasets/amarshiv86/p07-sre-lora-dataset) ·
    [Staff SRE Portfolio](https://github.com/amarshiv86)
    """)

demo.launch()
