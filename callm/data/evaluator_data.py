"""
DataModule for the correctness evaluator.

Reads LLM outputs from CSV and creates tokenized semantic equivalence prompts
for batched evaluation.
"""

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import Dataset
from jinja2 import Template
import csv
import os
import glob
import torch

from callm.utils import get_tokenizer_for_model


SEMANTIC_EQUIVALENCE_PROMPT = Template(
    """Is the following answer to my question Q semantically equivalent to any of the following answers?

Question: {{ question }}
Answer: {{ pred_answer }}
Answers: {{ gold_answers }}

Please answer with a single word, either "Yes." or "No.", and explain your reasoning."""
)

DEFAULT_MAX_LENGTH = 512


class EvaluatorDataModule(LightningDataModule):
    """DataModule for batched correctness evaluation."""

    def __init__(
        self,
        llm_outputs_path: str = None,
        model_name: str = "google/flan-t5-base",
        batch_size: int = 1,
        num_workers: int = 8,
        max_length: int = None,
        resume_from: str = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.llm_outputs_path = llm_outputs_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = None
        self.max_length = max_length
        self.resume_from = resume_from
        self.skip_indices = set()

    def setup(self, stage: str = None):
        if self.tokenizer is None:
            self.tokenizer = get_tokenizer_for_model(self.model_name)

            # Determine max_length from model config
            # Try common config attributes for max sequence length
            model_max_length = getattr(self.tokenizer, "model_max_length", None)
            if (
                model_max_length and model_max_length < 1e9
            ):  # Check it's not a huge default
                self.max_length = model_max_length
            else:
                if self.max_length is None:
                    self.max_length = DEFAULT_MAX_LENGTH
        # Read LLM outputs from CSV
        questions = []
        gold_answers_list = []
        pred_answers = []
        confidences = []
        raw_outputs = []
        original_indices = []

        self.skip_indices = self._get_skip_indices()

        with open(self.llm_outputs_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i in self.skip_indices:
                    continue
                questions.append(row["question"])
                gold_answers_list.append(
                    [g.strip().lower() for g in row["gold_answers"].split("|")]
                )
                pred_answers.append(row["pred_answer"])
                confidences.append(row["confidence"])
                raw_outputs.append(row.get("raw_output", ""))
                original_indices.append(i)

        print(f"\nLoaded {len(questions)} rows from CSV file: {self.llm_outputs_path}")

        # Create evaluation prompts
        prompts = []
        exact_matches = []
        for question, gold_answers, pred_answer in zip(
            questions, gold_answers_list, pred_answers
        ):
            # Check exact match first (short-circuit)
            if pred_answer.lower() in gold_answers:
                exact_matches.append(True)
                prompts.append("")  # Won't be used
            else:
                exact_matches.append(False)
                prompt = SEMANTIC_EQUIVALENCE_PROMPT.render(
                    question=question,
                    pred_answer=pred_answer,
                    gold_answers=gold_answers,
                )
                prompts.append(prompt)

        # Tokenize non-exact-match prompts
        # First pass: determine actual max length needed
        actual_max_length = 0
        for prompt, exact_match in zip(prompts, exact_matches):
            if not exact_match:
                tokens = self.tokenizer(prompt, return_tensors="pt")
                actual_max_length = max(actual_max_length, tokens["input_ids"].size(1))

        # Use the minimum of model's max and actual max to avoid unnecessary padding
        effective_max_length = (
            min(self.max_length, actual_max_length)
            if actual_max_length > 0
            else self.max_length
        )

        # Second pass: tokenize with the effective max length
        data = []
        for prompt, exact_match in zip(prompts, exact_matches):
            if exact_match:
                data.append(None)
            else:
                tokens = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=effective_max_length,
                    padding="max_length",
                    truncation=True,
                )
                data.append(tokens)

        self.dataset = Dataset.from_dict(
            {
                "data": data,
                "exact_match": exact_matches,
                "question": questions,
                "gold_answers": gold_answers_list,
                "pred_answer": pred_answers,
                "confidence": confidences,
                "raw_output": raw_outputs,
                "index": original_indices,
            }
        ).with_format("torch")

    def _get_skip_indices(self):
        skip_indices = set()
        if self.resume_from:
            if not os.path.exists(self.resume_from):
                raise ValueError(f"Resume path {self.resume_from} does not exist.")
            else:
                # Look for files matching pattern
                pattern = os.path.join(self.resume_from, "temp_eval_results_rank0_*.pt")
                found_files = sorted(
                    glob.glob(pattern),
                    key=lambda x: int(x.split("_")[-1].split(".")[0]),
                )

                if found_files:
                    print(f"Found {len(found_files)} temp files to resume from.")
                    # Load all files to get indices
                    for fpath in found_files:
                        try:
                            # Load on CPU to avoid CUDA errors if not available or OOM
                            chunk = torch.load(fpath, map_location="cpu")
                            for item in chunk:
                                idx = item.get("index")
                                if idx is not None:
                                    skip_indices.add(idx)
                        except Exception as e:
                            print(f"Error loading {fpath}: {e}")

                    print(f"Found {len(skip_indices)} indices to skip.")
                else:
                    print(f"No temp files found in {self.resume_from}")
        return skip_indices

    def val_dataloader(self):
        def collate_fn(batch):
            import torch

            # Separate exact matches from those needing evaluation
            needs_eval = [item for item in batch if not item["exact_match"]]

            if len(needs_eval) == 0:
                # All exact matches
                return {
                    "input_ids": None,
                    "attention_mask": None,
                    "exact_match": [item["exact_match"] for item in batch],
                    "question": [item["question"] for item in batch],
                    "gold_answers": [item["gold_answers"] for item in batch],
                    "pred_answer": [item["pred_answer"] for item in batch],
                    "confidence": [item["confidence"] for item in batch],
                    "raw_output": [item["raw_output"] for item in batch],
                    "index": [item["index"] for item in batch],
                }

            # Stack tokenized inputs for non-exact-matches
            input_ids = torch.stack(
                [item["data"]["input_ids"].squeeze(0) for item in needs_eval]
            )
            attention_mask = torch.stack(
                [item["data"]["attention_mask"].squeeze(0) for item in needs_eval]
            )

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "exact_match": [item["exact_match"] for item in batch],
                "question": [item["question"] for item in batch],
                "gold_answers": [item["gold_answers"] for item in batch],
                "pred_answer": [item["pred_answer"] for item in batch],
                "confidence": [item["confidence"] for item in batch],
                "raw_output": [item["raw_output"] for item in batch],
                "index": [item["index"] for item in batch],
            }

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
