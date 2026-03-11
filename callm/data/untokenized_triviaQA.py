from lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from callm.utils import subsample_dataset
from callm.prompts import VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT, Prompt


class UntokenizedTriviaQADataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        prompt: Prompt = VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT,
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
        load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")

    def setup(self, stage: str):
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

        self.triviaQA_val = Dataset.from_dict(
            {"input": input_texts, "question": questions, "label": answers}
        ).with_format("torch")

    def train_dataloader(self):
        return None

    @staticmethod
    def collate_fn(batch):
        """Collate function that prepares batch for forward pass."""
        return {
            "input": [item["input"] for item in batch],
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
