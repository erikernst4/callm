"""
DataModule for IS_TRUE validation on MMLU answers.

Reads LLM outputs from CSV and creates prompts asking if the proposed answer
is True or False for the given MMLU question (including choices).
"""

from torch.utils.data import DataLoader
from datasets import Dataset
import torch

from callm.data.answers_data import AnswersDataModule
from callm.prompts.mmlu import CHAT_MMLU_IS_TRUE_PROMPT
from callm.prompts.triviaqa import ChatPrompt
from callm.utils import subsample_dataset


class MMLUIsTrueDataModule(AnswersDataModule):
    """DataModule for Is True/False validation of MMLU LLM answers."""

    def __init__(
        self,
        prompt: ChatPrompt = CHAT_MMLU_IS_TRUE_PROMPT,
        disable_thinking: bool = False,
        choices_in_csv: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt = prompt
        self.disable_thinking = disable_thinking
        self.choices_in_csv = choices_in_csv

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

        # For MMLU IsTrue, we need the choices to include in the prompt.
        # If choices are stored in the CSV, use them. Otherwise, the
        # question text should already include the formatted choices.
        choices_list = [row.get("choices", "") for row in rows]

        # Create IS_TRUE prompts
        prompts = []
        for question, pred_answer, choices in zip(
            questions, pred_answers, choices_list
        ):
            prompt = self.prompt(question=question, answer=pred_answer, choices=choices)
            prompts.append(prompt)

        # Determine whether to use continue_final_message (assistant prefill)
        # or add_generation_prompt
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

        # Calculate max length for padding/truncation
        actual_max_length = 0
        for prompt in prompts:
            if isinstance(self.prompt, ChatPrompt):
                tokenized = self.tokenizer.apply_chat_template(
                    prompt,
                    return_tensors="pt",
                    **chat_template_kwargs,
                )
                length = tokenized["input_ids"].size(1)
            else:
                tokens = self.tokenizer(prompt, return_tensors="pt")
                length = tokens["input_ids"].size(1)
            actual_max_length = max(actual_max_length, length)

        # Use the minimum of model's max and actual max
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
                    return_tensors="pt",
                    max_length=effective_max_length,
                    padding="max_length",
                    truncation=True,
                    **chat_template_kwargs,
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
