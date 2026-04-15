from lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from callm.utils import get_tokenizer_for_model, subsample_dataset
from callm.prompts import CHAT_LABEL_PROB_PROMPT_ZERO_SHOT, ChatPrompt, Prompt


class TriviaQADataModule(LightningDataModule):
    requires_semantic_equivalence = True

    def __init__(
        self,
        batch_size: int = 32,
        model_name: str = "google/flan-t5-small",
        prompt: Prompt = CHAT_LABEL_PROB_PROMPT_ZERO_SHOT,
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
        load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")

    def setup(self, stage: str):
        # Initialize tokenizer if not already set (e.g., from model)
        if self.tokenizer is None:
            self.tokenizer = get_tokenizer_for_model(self.model_name)

        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")["validation"]

        # Limit samples if requested
        dataset = subsample_dataset(dataset, self.max_samples, self.seed)

        questions = dataset["question"]
        answers = []
        for value in dataset["answer"]:
            possible_answers = [
                answer.lower()
                for answer in value["aliases"] + value["normalized_aliases"]
            ]
            possible_answers = list(set(possible_answers))  # Remove duplicates
            answers.append(possible_answers)

        input_texts = [self.prompt(question=question) for question in questions]

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
