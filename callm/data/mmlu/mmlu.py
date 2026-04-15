"""
DataModule for MMLU multiple-choice question answering.

Loads the cais/mmlu dataset from HuggingFace and formats questions with
A/B/C/D choices for confidence-augmented generation.
"""

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from callm.utils import get_tokenizer_for_model, subsample_dataset
from callm.prompts.triviaqa import ChatPrompt, Prompt
from callm.prompts.mmlu import (
    CHAT_MMLU_LABEL_PROB_PROMPT,
    format_choices,
    answer_index_to_letter,
)


class MMLUDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        model_name: str = "google/flan-t5-small",
        prompt: Prompt = CHAT_MMLU_LABEL_PROB_PROMPT,
        max_samples: int = None,
        seed: int = 42,
        num_workers: int = 0,
        disable_thinking: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.model_name = model_name
        self.tokenizer = None
        self.prompt = prompt
        self.max_samples = max_samples
        self.seed = seed
        self.num_workers = num_workers
        self.disable_thinking = disable_thinking

    def prepare_data(self):
        load_dataset("cais/mmlu", "all")

    def setup(self, stage: str):
        # Initialize tokenizer if not already set
        if self.tokenizer is None:
            self.tokenizer = get_tokenizer_for_model(self.model_name)

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

        # Determine whether to use continue_final_message (assistant prefill)
        # or add_generation_prompt based on whether the prompt has an assistant template
        has_assistant_prefill = (
            isinstance(self.prompt, ChatPrompt)
            and self.prompt.assistant_template is not None
        )

        chat_template_kwargs = {
            "tokenize": True,
            "return_dict": True,
        }
        if self.disable_thinking:
            chat_template_kwargs["enable_thinking"] = False
        if has_assistant_prefill:
            chat_template_kwargs["continue_final_message"] = True
        else:
            chat_template_kwargs["add_generation_prompt"] = True

        # Calculate max lengths for padding/truncation
        max_token_seq_length = 0
        for input_text in input_texts:
            if isinstance(self.prompt, ChatPrompt):
                tokenized = self.tokenizer.apply_chat_template(
                    input_text,
                    return_tensors="pt",
                    **chat_template_kwargs,
                )
                length = tokenized["input_ids"].size(1)
            else:
                x = self.tokenizer(input_text, return_tensors="pt")
                length = x["input_ids"].size(1)

            if length > max_token_seq_length:
                max_token_seq_length = length

        # Tokenize
        data = []
        for input_text in input_texts:
            if isinstance(self.prompt, ChatPrompt):
                tokens = self.tokenizer.apply_chat_template(
                    input_text,
                    return_tensors="pt",
                    max_length=max_token_seq_length,
                    padding="max_length",
                    truncation=True,
                    **chat_template_kwargs,
                )
            else:
                tokens = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=max_token_seq_length,
                    padding="max_length",
                    truncation=True,
                )
            data.append(tokens)

        self.mmlu_val = Dataset.from_dict(
            {
                "data": data,
                "question": questions,
                "label": answers,
                "choices": formatted_choices,
            }
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

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
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
