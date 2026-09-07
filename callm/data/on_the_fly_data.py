"""
On-the-fly activations data module for probe training at scale.

Instead of pre-saving activations to disk (which is infeasible for large N),
this data module pre-computes only correctness labels and answer texts, then
at training time provides tokenized [prompt + answer] batches so the LLM can
run a single non-generative forward pass to extract activations on the fly.
"""

import csv
import random
import torch
import importlib
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from callm.utils import get_tokenizer_for_model
from callm.prompts import ChatPrompt


class OnTheFlyDataset(Dataset):
    """Simple PyTorch Dataset for on-the-fly activation samples."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class OnTheFlyActivationsDataModule(LightningDataModule):
    """
    LightningDataModule that loads pre-computed correctness labels and answer
    texts, tokenizes [prompt + answer] at setup time, and provides batches
    with (input_ids, attention_mask, prompt_lengths, correct) for
    ConfidenceAugmentedLLM to run a single non-generative forward pass.

    Pre-computation step produces two CSVs:
    - ``llm_outputs.csv``: has ``question``, ``raw_output`` (full answer text)
    - ``evaluation_results.csv``: has ``Correct`` (Yes/No)

    Both CSVs must have the same number of rows in the same sample order.
    """

    def __init__(
        self,
        llm_outputs_csv: str,
        evaluation_csv: str,
        model_name: str = "Qwen/Qwen3.5-9B",
        prompt: str = "callm.prompts.TRANSPARENT_PROMPT_ZERO_SHOT",
        batch_size: int = 4,
        num_workers: int = 0,
        train_split: float = 0.8,
        seed: int = 42,
        max_samples: int | None = None,
        disable_thinking: bool = True,
    ):
        super().__init__()
        self.llm_outputs_csv = llm_outputs_csv
        self.evaluation_csv = evaluation_csv
        self.model_name = model_name
        self.prompt_path = prompt
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.seed = seed
        self.max_samples = max_samples
        self.disable_thinking = disable_thinking

        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        # CSVs are already on disk from pre-computation step
        pass

    def setup(self, stage: str = None):
        import os

        if not os.path.exists(self.llm_outputs_csv):
            raise FileNotFoundError(
                f"LLM outputs CSV not found at {self.llm_outputs_csv}"
            )
        if not os.path.exists(self.evaluation_csv):
            raise FileNotFoundError(
                f"Evaluation CSV not found at {self.evaluation_csv}"
            )

        # 1. Load tokenizer and prompt
        if self.tokenizer is None:
            self.tokenizer = get_tokenizer_for_model(self.model_name)

        if isinstance(self.prompt_path, str):
            module_name, obj_name = self.prompt_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            self.prompt_obj = getattr(module, obj_name)
        else:
            self.prompt_obj = self.prompt_path

        # 2. Load raw outputs from llm_outputs.csv
        raw_outputs = []
        questions_from_llm = []
        with open(self.llm_outputs_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            field_map = {col.lower(): col for col in reader.fieldnames or []}

            raw_col = field_map.get("raw_output") or field_map.get("raw output")
            q_col = field_map.get("question")

            if raw_col is None:
                raise ValueError(
                    f"Could not find 'raw_output' column in {self.llm_outputs_csv}. "
                    f"Available columns: {list(reader.fieldnames or [])}"
                )

            for row in reader:
                raw_output = row.get(raw_col, "")
                # Unescape newlines that were escaped during CSV save
                raw_output = raw_output.replace("\\n", "\n")
                raw_outputs.append(raw_output)
                if q_col:
                    questions_from_llm.append(row.get(q_col, ""))

        # 3. Load correctness labels from evaluation_results.csv
        correctness_labels = []
        questions_from_eval = []
        with open(self.evaluation_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            field_map = {col.lower(): col for col in reader.fieldnames or []}

            correct_col = field_map.get("correct")
            q_col = field_map.get("question")

            if correct_col is None:
                raise ValueError(
                    f"Could not find 'Correct' column in {self.evaluation_csv}. "
                    f"Available columns: {list(reader.fieldnames or [])}"
                )

            for row in reader:
                is_correct = row[correct_col].strip().lower() == "yes"
                correctness_labels.append(is_correct)
                if q_col:
                    questions_from_eval.append(row.get(q_col, ""))

        # 4. Validate matching
        if len(raw_outputs) != len(correctness_labels):
            raise ValueError(
                f"Row count mismatch: llm_outputs.csv has {len(raw_outputs)} rows, "
                f"evaluation_results.csv has {len(correctness_labels)} rows."
            )

        num_samples = len(raw_outputs)
        if self.max_samples is not None and self.max_samples < num_samples:
            num_samples = self.max_samples

        print(f"Loaded {num_samples} samples from pre-computed CSVs")

        # 5. Tokenize [prompt + answer] for each sample
        # Determine chat template kwargs
        has_assistant_prefill = (
            isinstance(self.prompt_obj, ChatPrompt)
            and self.prompt_obj.assistant_template is not None
        )

        chat_template_kwargs = {"tokenize": True}
        if self.disable_thinking:
            chat_template_kwargs["enable_thinking"] = False
        if has_assistant_prefill:
            chat_template_kwargs["continue_final_message"] = True
        else:
            chat_template_kwargs["add_generation_prompt"] = True

        samples = []
        for i in range(num_samples):
            question = (
                questions_from_llm[i] if questions_from_llm else questions_from_eval[i]
            )
            raw_output = raw_outputs[i]

            # Tokenize prompt
            prompt_text = self.prompt_obj(question=question)

            if isinstance(self.prompt_obj, ChatPrompt):
                prompt_tokens = self.tokenizer.apply_chat_template(
                    prompt_text, **chat_template_kwargs
                )
                # Handle various return types
                if hasattr(prompt_tokens, "input_ids"):
                    prompt_ids = prompt_tokens["input_ids"]
                elif isinstance(prompt_tokens, dict) and "input_ids" in prompt_tokens:
                    prompt_ids = prompt_tokens["input_ids"]
                else:
                    prompt_ids = prompt_tokens

                if torch.is_tensor(prompt_ids):
                    prompt_ids = prompt_ids.squeeze().tolist()
                    if isinstance(prompt_ids, int):
                        prompt_ids = [prompt_ids]
                elif not isinstance(prompt_ids, list):
                    prompt_ids = list(prompt_ids)
            else:
                prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
                if torch.is_tensor(prompt_ids):
                    prompt_ids = prompt_ids.squeeze().tolist()
                    if isinstance(prompt_ids, int):
                        prompt_ids = [prompt_ids]
                elif not isinstance(prompt_ids, list):
                    prompt_ids = list(prompt_ids)

            # Tokenize answer (no special tokens — continuation of prompt)
            answer_ids = self.tokenizer.encode(raw_output, add_special_tokens=False)
            if torch.is_tensor(answer_ids):
                answer_ids = answer_ids.squeeze().tolist()
                if isinstance(answer_ids, int):
                    answer_ids = [answer_ids]
            elif not isinstance(answer_ids, list):
                answer_ids = list(answer_ids)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "answer_ids": answer_ids,
                    "prompt_length": len(prompt_ids),
                    "correct": correctness_labels[i],
                }
            )

        # 6. Split into train and validation
        rng = random.Random(self.seed)
        rng.shuffle(samples)

        split_idx = int(len(samples) * self.train_split)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        self.train_dataset = OnTheFlyDataset(train_samples)
        self.val_dataset = OnTheFlyDataset(val_samples)

        print(
            f"Train split: {len(self.train_dataset)} samples, "
            f"Val split: {len(self.val_dataset)} samples."
        )

    @staticmethod
    def collate_fn(batch):
        """
        Collate function that left-pads [prompt + answer] sequences to the
        max length in the batch, creates attention masks, and stacks labels.

        Returns a dict with:
        - input_ids: (batch_size, max_seq_len)
        - attention_mask: (batch_size, max_seq_len) — 0 for padding, 1 for real tokens
        - prompt_lengths: (batch_size,) — number of prompt tokens per sample
        - correct: (batch_size,) — correctness labels
        """
        # Concatenate prompt + answer for each sample
        full_ids_list = []
        prompt_lengths = []
        correct_labels = []

        for item in batch:
            full_ids = item["prompt_ids"] + item["answer_ids"]
            full_ids_list.append(full_ids)
            prompt_lengths.append(item["prompt_length"])
            correct_labels.append(item["correct"])

        # Find max length in batch
        max_len = max(len(ids) for ids in full_ids_list)

        # Left-pad (padding on the left for causal LMs)
        padded_input_ids = []
        attention_masks = []
        for ids in full_ids_list:
            pad_len = max_len - len(ids)
            # Use 0 as pad token id (will be masked anyway)
            padded_ids = [0] * pad_len + ids
            mask = [0] * pad_len + [1] * len(ids)
            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "correct": torch.tensor(correct_labels, dtype=torch.long),
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
