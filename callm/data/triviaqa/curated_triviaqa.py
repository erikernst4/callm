from lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from callm.utils import get_tokenizer_for_model, subsample_dataset
from callm.prompts import TRANSPARENT_PROMPT_ZERO_SHOT, ChatPrompt, Prompt


class CuratedTriviaQADataModule(LightningDataModule):
    """
    Data module for the curated 455-question TriviaQA curated evaluation subset
    from HuggingFace: ErikErnst/triviaqa-extended-subset.
    """

    requires_semantic_equivalence = False  # Evaluated directly via exact match

    def __init__(
        self,
        batch_size: int = 1,
        model_name: str = "Qwen/Qwen3.5-9B",
        prompt: Prompt = TRANSPARENT_PROMPT_ZERO_SHOT,
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
        load_dataset("ErikErnst/triviaqa-extended-subset")

    def setup(self, stage: str = None):
        if self.tokenizer is None:
            self.tokenizer = get_tokenizer_for_model(self.model_name)

        dataset = load_dataset("ErikErnst/triviaqa-extended-subset")["train"]
        dataset = subsample_dataset(dataset, self.max_samples, self.seed)

        questions = dataset["question"]
        answers = []
        for value in dataset["answer"]:
            aliases = value.get("aliases", []) or []
            norm_aliases = value.get("normalized_aliases", []) or []
            val = value.get("value", "")
            possible_answers = list(
                set([a.lower() for a in aliases + norm_aliases + [val] if a])
            )
            answers.append(possible_answers)

        input_texts = [self.prompt(question=question) for question in questions]

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

        self.triviaQA_val = Dataset.from_dict(
            {"data": data, "question": questions, "label": answers}
        ).with_format("torch")

    def train_dataloader(self):
        return None

    @staticmethod
    def collate_fn(batch):
        import torch

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
            "gold_answers": [item["label"] for item in batch],
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
