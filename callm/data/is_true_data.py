"""
DataModule for IS_TRUE_PROB_PROMPT validation.

Reads LLM outputs from CSV and creates prompts asking if the proposed answer
is True or False for the given question.
"""

from torch.utils.data import DataLoader
from datasets import Dataset
import torch

from callm.data.answers_data import AnswersDataModule
from callm.prompts import CHAT_IS_TRUE_PROB_PROMPT, ChatPrompt
from callm.utils import subsample_dataset


class IsTrueDataModule(AnswersDataModule):
    """DataModule for Is True/False validation of LLM answers."""

    def __init__(
        self,
        prompt: ChatPrompt = CHAT_IS_TRUE_PROB_PROMPT,
        disable_thinking: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt = prompt
        self.disable_thinking = disable_thinking

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

        # Create IS_TRUE prompts
        prompts = []
        for question, pred_answer in zip(questions, pred_answers):
            prompt = self.prompt(question=question, answer=pred_answer)
            prompts.append(prompt)

        # Calculate max length for padding/truncation
        actual_max_length = 0
        for prompt in prompts:
            if isinstance(self.prompt, ChatPrompt):
                tokenized = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    return_dict=True,
                    enable_thinking=not self.disable_thinking,
                )
                length = tokenized["input_ids"].size(1)
            else:
                tokens = self.tokenizer(prompt, return_tensors="pt")
                length = tokens["input_ids"].size(1)
            actual_max_length = max(actual_max_length, length)

        # Use the minimum of model's max and actual max to avoid unnecessary padding
        effective_max_length = (
            min(self.max_length, actual_max_length)
            if actual_max_length > 0
            else self.max_length
        )

        # Tokenize prompts
        data = []
        for prompt in prompts:
            if isinstance(self.prompt, ChatPrompt):
                tokens = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=True,
                    return_tensors="pt",
                    max_length=effective_max_length,
                    padding="max_length",
                    truncation=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    enable_thinking=not self.disable_thinking,
                )
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
            # Stack tokenized inputs
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
