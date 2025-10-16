from lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from callm.utils import get_tokenizer_for_model
from callm.prompts import VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT, generate_prompt
from jinja2 import Template

class TriviaQADataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        model_name: str = "flan-t5-small",
        prompt: Template = VERBALIZED_ONE_SENTENCE_TOP_1_PROMPT,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.tokenizer = get_tokenizer_for_model(model_name)
        self.prompt = prompt

    def prepare_data(self):
        self.dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")

    def setup(self, stage: str):
        questions = self.dataset["validation"]["question"]
        answers = []
        for value in self.dataset["validation"]["answer"]:
            answers.append(value["aliases"] + value["normalized_aliases"])

        input_texts = [generate_prompt(self.prompt, question=question) for question in questions]

        # Calculate max lengths for padding/truncation
        max_token_seq_length = 0
        for input in input_texts:
            x = self.tokenizer(input, return_tensors='pt')
            if x["input_ids"].size(1) > max_token_seq_length:
                max_token_seq_length = x["input_ids"].size(1)

        # Tokenize
        data = [
            self.tokenizer(input, return_tensors='pt', max_length=max_token_seq_length, padding="max_length", truncation=True)
            for input in input_texts
        ]

        self.triviaQA_val = Dataset.from_dict({"data": data, "label": answers}).with_format("torch")
    
    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return DataLoader(self.triviaQA_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return None
