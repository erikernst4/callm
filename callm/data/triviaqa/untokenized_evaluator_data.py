from torch.utils.data import DataLoader
from datasets import Dataset

from callm.data.answers_data import AnswersDataModule
from callm.utils import subsample_dataset, check_exact_match
from callm.data.triviaqa.evaluator_data import SEMANTIC_EQUIVALENCE_PROMPT


class UntokenizedEvaluatorDataModule(AnswersDataModule):
    """DataModule for batched correctness evaluation using external APIs (e.g. GCP).

    Skips tokenization entirely and supplies the raw evaluator string as 'input'.
    """

    def setup(self, stage: str = None):
        # We purposely don't initialize internal self.tokenizer from the base class `self._setup_tokenizer()`

        # Load LLM outputs from CSV using base class method
        rows = self.load_llm_outputs_from_csv()

        questions = [row["question"] for row in rows]
        gold_answers_list = [row["gold_answers"] for row in rows]
        pred_answers = [row["pred_answer"] for row in rows]
        confidences = [row["confidence"] for row in rows]
        raw_outputs = [row["raw_output"] for row in rows]
        original_indices = [row["index"] for row in rows]

        # Create evaluation prompts
        prompts = []
        exact_matches = []
        for question, gold_answers, pred_answer in zip(
            questions, gold_answers_list, pred_answers
        ):
            # Check exact match first
            # The pred_answer might not be lowercased yet, while gold answers usually are.
            if check_exact_match(pred_answer, gold_answers):
                exact_matches.append(True)
                prompts.append("")  # Won't be used
            else:
                exact_matches.append(False)
                prompt = SEMANTIC_EQUIVALENCE_PROMPT.render(
                    question=question,
                    pred_answer=pred_answer,
                    gold_answers=gold_answers,
                )
                prompts.append(prompt)

        self.dataset = Dataset.from_dict(
            {
                "input": prompts,
                "exact_match": exact_matches,
                "question": questions,
                "gold_answers": gold_answers_list,
                "pred_answer": pred_answers,
                "confidence": confidences,
                "raw_output": raw_outputs,
                "index": original_indices,
            }
        ).with_format("torch")

        self.dataset = subsample_dataset(self.dataset, self.max_samples, self.seed)

    def _setup_tokenizer(self):
        pass

    @staticmethod
    def collate_fn(batch):
        return {
            "input": [item["input"] for item in batch],
            "exact_match": [item["exact_match"] for item in batch],
            "question": [item["question"] for item in batch],
            "gold_answers": [item["gold_answers"] for item in batch],
            "pred_answer": [item["pred_answer"] for item in batch],
            "confidence": [item["confidence"] for item in batch],
            "raw_output": [item["raw_output"] for item in batch],
            "index": [item["index"] for item in batch],
        }

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
