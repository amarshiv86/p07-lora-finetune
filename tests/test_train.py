import json
import os
import tempfile

import pytest


class TestDataLoading:
    def test_jsonl_loads_correctly(self, tmp_path):
        """Training data should load as list of dicts with question/answer keys."""
        data_path = tmp_path / "test.jsonl"
        records = [
            {"question": "What is SLO?", "answer": "Service Level Objective."},
            {"question": "What is SLI?", "answer": "Service Level Indicator."},
        ]
        with open(data_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        # Import here to avoid GPU requirements at test collection
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from src.train import load_dataset_from_jsonl

        dataset = load_dataset_from_jsonl(str(data_path))
        assert len(dataset) == 2
        assert "question" in dataset.features
        assert "answer" in dataset.features

    def test_jsonl_skips_empty_lines(self, tmp_path):
        """Empty lines in JSONL should be skipped gracefully."""
        data_path = tmp_path / "test.jsonl"
        with open(data_path, "w") as f:
            f.write('{"question": "Q1", "answer": "A1"}\n')
            f.write("\n")
            f.write('{"question": "Q2", "answer": "A2"}\n')

        from src.train import load_dataset_from_jsonl
        dataset = load_dataset_from_jsonl(str(data_path))
        assert len(dataset) == 2

    def test_real_training_data_valid(self):
        """Real training data file should load and have required keys."""
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "raw", "sre_runbook_qa.jsonl"
        )
        from src.train import load_dataset_from_jsonl
        dataset = load_dataset_from_jsonl(data_path)
        assert len(dataset) > 10, "Should have at least 10 training examples"
        for example in dataset:
            assert "question" in example, "Each example must have 'question'"
            assert "answer" in example, "Each example must have 'answer'"
            assert len(example["question"]) > 10, "Questions should not be empty"
            assert len(example["answer"]) > 20, "Answers should not be empty"


class TestPromptFormat:
    def test_format_prompt_contains_question_and_answer(self):
        from src.train import format_prompt
        example = {"question": "What is SLO?", "answer": "Service Level Objective."}
        prompt = format_prompt(example)
        assert "What is SLO?" in prompt
        assert "Service Level Objective." in prompt

    def test_format_prompt_phi3_format(self):
        """Prompt should use Phi-3 instruction format."""
        from src.train import format_prompt
        example = {"question": "Q", "answer": "A"}
        prompt = format_prompt(example)
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt
        assert "<|end|>" in prompt


class TestEvalResults:
    def test_processed_eval_results_valid(self):
        """Processed eval results file should have required structure."""
        results_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "processed", "eval_results.json"
        )
        with open(results_path) as f:
            results = json.load(f)

        assert "base_model" in results
        assert "finetuned_model" in results
        assert "rouge" in results["base_model"]
        assert "rouge" in results["finetuned_model"]

        for metric in ["rouge1", "rouge2", "rougeL"]:
            assert metric in results["base_model"]["rouge"]
            assert metric in results["finetuned_model"]["rouge"]
            base = results["base_model"]["rouge"][metric]
            ft = results["finetuned_model"]["rouge"][metric]
            assert ft > base, f"Fine-tuned should outperform base on {metric}"

    def test_eval_results_has_examples(self):
        """Eval results should contain before/after examples."""
        results_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "processed", "eval_results.json"
        )
        with open(results_path) as f:
            results = json.load(f)
        assert "examples" in results
        assert len(results["examples"]) > 0
        for ex in results["examples"]:
            assert "question" in ex
            assert "base_answer" in ex
            assert "finetuned_answer" in ex
