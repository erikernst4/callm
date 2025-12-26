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

from callm.utils import get_tokenizer_for_model


SEMANTIC_EQUIVALENCE_PROMPT = Template(
    """Is the following answer to my question Q semantically equivalent to any of the following answers?

Question: {{ question }}
Answer: {{ pred_answer }}
Answers: {{ gold_answers }}

Please answer with a single word, either "Yes." or "No.", and explain your reasoning."""
)


class EvaluatorDataModule(LightningDataModule):
    """DataModule for batched correctness evaluation."""

    def __init__(
        self,
        llm_outputs_path: str,
        model_name: str = "google/flan-t5-base",
        batch_size: int = 8,
    ):
        super().__init__()
        self.llm_outputs_path = llm_outputs_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = None

    def setup(self, stage: str = None):
        if self.tokenizer is None:
            self.tokenizer = get_tokenizer_for_model(self.model_name)

        # Read LLM outputs from CSV
        questions = []
        gold_answers_list = []
        pred_answers = []
        confidences = []
        raw_outputs = []

        with open(self.llm_outputs_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append(row["question"])
                gold_answers_list.append(row["gold_answers"])
                pred_answers.append(row["pred_answer"])
                confidences.append(row["confidence"])
                raw_outputs.append(row.get("raw_output", ""))

        # Create evaluation prompts
        prompts = []
        exact_matches = []
        for question, gold_answers, pred_answer in zip(
            questions, gold_answers_list, pred_answers
        ):
            # Check exact match first (short-circuit)
            if pred_answer is not None and pred_answer in gold_answers:
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
        # Find max length for padding
        max_length = 0
        tokenized_prompts = []
        for i, prompt in enumerate(prompts):
            if exact_matches[i]:
                tokenized_prompts.append(None)
            else:
                tokens = self.tokenizer(prompt, return_tensors="pt")
                tokenized_prompts.append(tokens)
                if tokens["input_ids"].size(1) > max_length:
                    max_length = tokens["input_ids"].size(1)

        # Re-tokenize with padding
        data = []
        for i, prompt in enumerate(prompts):
            if exact_matches[i]:
                data.append(None)
            else:
                tokens = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=max_length,
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
                "index": list(range(len(questions))),
            }
        ).with_format("torch")

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
            self.dataset, batch_size=self.batch_size, collate_fn=collate_fn
        )
