from lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from callm.utils import get_tokenizer_for_model
from callm.prompts import VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT, Prompt


class TriviaQADataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        model_name: str = "google/flan-t5-small",
        prompt: Prompt = VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT,
        max_samples: int = None,
        seed: int = 42,
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.model_name = model_name
        self.tokenizer = None
        self.prompt = prompt
        self.max_samples = max_samples
        self.seed = seed
        self.num_workers = num_workers

    def prepare_data(self):
        load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")

    def setup(self, stage: str):
        # Initialize tokenizer if not already set (e.g., from model)
        if self.tokenizer is None:
            self.tokenizer = get_tokenizer_for_model(self.model_name)

        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")["validation"]

        # Limit samples if requested
        if self.max_samples is not None:
            if self.seed is not None:
                dataset = dataset.shuffle(seed=self.seed)
            dataset = dataset.select(range(min(len(dataset), self.max_samples)))

        questions = dataset["question"]
        answers = []
        for value in dataset["answer"]:
            answers.append(value["aliases"] + value["normalized_aliases"])

        input_texts = [self.prompt(question=question) for question in questions]

        # Calculate max lengths for padding/truncation
        max_token_seq_length = 0
        for input_text in input_texts:
            x = self.tokenizer(input_text, return_tensors="pt")
            if x["input_ids"].size(1) > max_token_seq_length:
                max_token_seq_length = x["input_ids"].size(1)

        # Tokenize
        data = [
            self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=max_token_seq_length,
                padding="max_length",
                truncation=True,
            )
            for input_text in input_texts
        ]

        self.triviaQA_val = Dataset.from_dict(
            {"data": data, "question": questions, "label": answers}
        ).with_format("torch")

    def train_dataloader(self):
        return None

    @staticmethod
    def collate_fn(batch):
        """Collate function that prepares tensors for forward pass."""
        import torch

        # Stack input_ids and attention_mask into batch tensors
        input_ids = torch.stack(
            [item["data"]["input_ids"].squeeze(0) for item in batch]
        )
        attention_mask = torch.stack(
            [item["data"]["attention_mask"].squeeze(0) for item in batch]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "question": [item["question"] for item in batch],
            "label": [item["label"] for item in batch],
        }

    def val_dataloader(self):
        return DataLoader(
            self.triviaQA_val,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return None
