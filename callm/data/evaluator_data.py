"""
DataModule for the correctness evaluator.

Reads LLM outputs from CSV and creates tokenized semantic equivalence prompts
for batched evaluation.
"""

from torch.utils.data import DataLoader
from datasets import Dataset
from jinja2 import Template

from callm.data.answers_data import AnswersDataModule
from callm.utils import subsample_dataset

SEMANTIC_EQUIVALENCE_PROMPT = Template(
    """Is the following answer to my question Q semantically equivalent to any of the following answers?

Question: {{ question }}
Answer: {{ pred_answer }}
Answers: {{ gold_answers }}

Please answer with a single word, either "Yes." or "No.", and explain your reasoning."""
)


class EvaluatorDataModule(AnswersDataModule):
    """DataModule for batched correctness evaluation."""

    def setup(self, stage: str = None):
        self._setup_tokenizer()

        # Load LLM outputs from CSV using base class method
        rows = self.load_llm_outputs_from_csv()

        questions = [row["question"] for row in rows]
        gold_answers_list = [row["gold_answers"] for row in rows]
        pred_answers = [row["pred_answer"] for row in rows]
        confidences = [row["confidence"] for row in rows]
        raw_outputs = [row["raw_output"] for row in rows]
        original_indices = [row["index"] for row in rows]

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

        self.dataset = subsample_dataset(self.dataset, self.max_samples, self.seed)

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
