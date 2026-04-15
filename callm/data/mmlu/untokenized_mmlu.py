"""
Untokenized DataModule for MMLU multiple-choice question answering.

Designed for GCP models that handle raw strings directly.
"""

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from callm.utils import subsample_dataset
from callm.prompts import (
    Prompt,
    GCP_CHAT_MMLU_LABEL_PROB_PROMPT,
    format_choices,
    answer_index_to_letter,
)


class UntokenizedMMLUDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        prompt: Prompt = GCP_CHAT_MMLU_LABEL_PROB_PROMPT,
        max_samples: int = None,
        seed: int = 42,
        num_workers: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.prompt = prompt
        self.max_samples = max_samples
        self.seed = seed
        self.num_workers = num_workers

    def prepare_data(self):
        load_dataset("cais/mmlu", "all")

    def setup(self, stage: str):
        dataset = load_dataset("cais/mmlu", "all")["test"]

        # Limit samples if requested
        dataset = subsample_dataset(dataset, self.max_samples, self.seed)

        questions = dataset["question"]
        choices_list = dataset["choices"]
        answer_indices = dataset["answer"]

        # Convert answer indices to letters and format choices
        answers = [answer_index_to_letter(idx) for idx in answer_indices]
        formatted_choices = [format_choices(choices) for choices in choices_list]

        input_texts = [
            self.prompt(question=question, choices=choices)
            for question, choices in zip(questions, formatted_choices)
        ]

        self.mmlu_val = Dataset.from_dict(
            {
                "input": input_texts,
                "question": questions,
                "label": answers,
                "choices": formatted_choices,
            }
        ).with_format("torch")

    def train_dataloader(self):
        return None

    @staticmethod
    def collate_fn(batch):
        """Collate function that prepares batch for forward pass."""
        res = {
            "input": [item["input"] for item in batch],
            "question": [item["question"] for item in batch],
            "label": [item["label"] for item in batch],
        }
        if "choices" in batch[0]:
            res["choices"] = [item["choices"] for item in batch]
        return res

    def val_dataloader(self):
        return DataLoader(
            self.mmlu_val,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return None
