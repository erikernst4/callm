"""
DataModule for IS_TRUE validation on MMLU answers without tokenization.

Designed for GCP models that handle raw strings.
"""

from torch.utils.data import DataLoader
from datasets import Dataset

from callm.data.answers_data import AnswersDataModule
from callm.prompts.mmlu import GCP_CHAT_MMLU_IS_TRUE_PROMPT
from callm.utils import subsample_dataset


class UntokenizedMMLUIsTrueDataModule(AnswersDataModule):
    """DataModule for Is True/False validation of MMLU LLM answers without tokenization."""

    def setup(self, stage: str = None):
        # Load LLM outputs from CSV using base class method
        rows = self.load_llm_outputs_from_csv()

        questions = [row["question"] for row in rows]
        gold_answers_list = [row["gold_answers"] for row in rows]
        pred_answers = [row["pred_answer"] for row in rows]
        confidences = [row["confidence"] for row in rows]
        raw_outputs = [row["raw_output"] for row in rows]
        original_indices = [row["index"] for row in rows]
        choices_list = [row.get("choices", None) for row in rows]

        # Create IS_TRUE prompts
        prompts = []
        for question, pred_answer, choices in zip(
            questions, pred_answers, choices_list
        ):
            prompt = GCP_CHAT_MMLU_IS_TRUE_PROMPT(
                question=question, answer=pred_answer, choices=choices
            )
            prompts.append(prompt)

        self.dataset = Dataset.from_dict(
            {
                "input": prompts,
                "question": questions,
                "label": gold_answers_list,
                "pred_answer": pred_answers,
                "confidence": confidences,
                "raw_output": raw_outputs,
                "index": original_indices,
            }
        ).with_format("torch")

        self.dataset = subsample_dataset(self.dataset, self.max_samples, self.seed)

    def _setup_tokenizer(self):
        """Override to skip tokenizer setup."""
        pass

    def val_dataloader(self):
        def collate_fn(batch):
            return {
                "input": [item["input"] for item in batch],
                "question": [item["question"] for item in batch],
                "label": [item["label"] for item in batch],
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
