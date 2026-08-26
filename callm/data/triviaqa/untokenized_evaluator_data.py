from torch.utils.data import DataLoader
from datasets import Dataset
from typing import Optional, Union, List
import csv
import os

from callm.data.answers_data import AnswersDataModule
from callm.utils import subsample_dataset, check_exact_match
from callm.prompts import Prompt, SEMANTIC_EQUIVALENCE_PROMPT_FOR_FREE_ANSWERS


def load_previous_eval_cache(previous_eval_csv: Union[str, List[str]]):
    """
    Load previously evaluated correctness labels from one or more CSV files.
    Returns:
        exact_lookup: dict mapping (normalized_question, normalized_pred_answer) -> {correct: bool, evaluator_response: str}
        prefix_lookup: list of tuples for prefix matching if questions/answers were truncated in older CSVs.
    """
    if isinstance(previous_eval_csv, str):
        csv_paths = [p.strip() for p in previous_eval_csv.split(",") if p.strip()]
    elif isinstance(previous_eval_csv, (list, tuple)):
        csv_paths = list(previous_eval_csv)
    else:
        return {}, []

    exact_lookup = {}
    prefix_lookup = []

    for path in csv_paths:
        if not path or not os.path.exists(path):
            print(f"Warning: previous_eval_csv path does not exist: {path}")
            continue

        print(f"Loading cached evaluation results from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            field_map = {col.lower(): col for col in reader.fieldnames or []}
            q_col = field_map.get("question")
            pred_col = field_map.get("predicted answer") or field_map.get("pred_answer")
            corr_col = field_map.get("correct")
            resp_col = field_map.get("evaluator response") or field_map.get(
                "evaluator_response"
            )

            if not q_col or not corr_col:
                print(
                    f"Warning: CSV {path} missing 'Question' or 'Correct' column. Skipping."
                )
                continue

            for row in reader:
                q = row.get(q_col, "").strip()
                a = row.get(pred_col, "").strip() if pred_col else ""
                corr_val = row.get(corr_col, "").strip().lower() == "yes"
                resp_val = row.get(resp_col, "").strip() if resp_col else ""

                q_clean = q.lower()
                a_clean = a.lower()
                val = {"correct": corr_val, "evaluator_response": resp_val}

                exact_lookup[(q_clean, a_clean)] = val

                q_is_pref = q_clean.endswith("...")
                a_is_pref = a_clean.endswith("...")
                if q_is_pref or a_is_pref:
                    prefix_lookup.append(
                        (
                            q_clean[:-3] if q_is_pref else q_clean,
                            q_is_pref,
                            a_clean[:-3] if a_is_pref else a_clean,
                            a_is_pref,
                            val,
                        )
                    )

    print(
        f"Loaded {len(exact_lookup)} cached sample evaluations for incremental labeling."
    )
    return exact_lookup, prefix_lookup


def lookup_cached_evaluation(
    question: str, pred_answer: str, exact_lookup: dict, prefix_lookup: list
):
    """Check if (question, pred_answer) has a cached evaluation result."""
    q_clean = question.strip().lower()
    a_clean = pred_answer.strip().lower()

    if (q_clean, a_clean) in exact_lookup:
        return exact_lookup[(q_clean, a_clean)]

    for q_pref, q_is_pref, a_pref, a_is_pref, val in prefix_lookup:
        q_match = q_clean.startswith(q_pref) if q_is_pref else q_clean == q_pref
        a_match = a_clean.startswith(a_pref) if a_is_pref else a_clean == a_pref
        if q_match and a_match:
            return val

    return None


class UntokenizedEvaluatorDataModule(AnswersDataModule):
    """DataModule for batched correctness evaluation using external APIs (e.g. GCP).

    Skips tokenization entirely and supplies the raw evaluator string as 'input'.
    Supports incremental evaluation via `previous_eval_csv` to skip API calls for
    previously evaluated samples as well as exact string matches.
    """

    def __init__(
        self,
        prompt: Optional[Prompt] = None,
        previous_eval_csv: Optional[Union[str, List[str]]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt = (
            prompt
            if prompt is not None
            else SEMANTIC_EQUIVALENCE_PROMPT_FOR_FREE_ANSWERS
        )
        self.previous_eval_csv = previous_eval_csv

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

        # Load cached previous evaluations if available
        exact_lookup, prefix_lookup = {}, []
        if self.previous_eval_csv:
            exact_lookup, prefix_lookup = load_previous_eval_cache(
                self.previous_eval_csv
            )

        # Create evaluation prompts & track exact matches / cached labels
        prompts = []
        exact_matches = []
        cached_corrects = []
        cached_evaluator_responses = []

        exact_match_count = 0
        cached_match_count = 0
        needs_api_call_count = 0

        for question, gold_answers, pred_answer in zip(
            questions, gold_answers_list, pred_answers
        ):
            # 1. Check exact match first (no API call needed)
            if check_exact_match(pred_answer, gold_answers):
                exact_matches.append(True)
                cached_corrects.append(True)
                cached_evaluator_responses.append("")
                prompts.append("")  # Won't be sent to API
                exact_match_count += 1
            else:
                # 2. Check if already evaluated in previous evaluation CSV
                cached_val = lookup_cached_evaluation(
                    question, pred_answer, exact_lookup, prefix_lookup
                )
                if cached_val is not None:
                    exact_matches.append(False)
                    cached_corrects.append(cached_val["correct"])
                    cached_evaluator_responses.append(cached_val["evaluator_response"])
                    prompts.append("")  # No prompt needed, already labeled!
                    cached_match_count += 1
                else:
                    # 3. Needs external API call
                    exact_matches.append(False)
                    cached_corrects.append(None)
                    cached_evaluator_responses.append(None)
                    prompt = self.prompt(
                        question=question,
                        pred_answer=pred_answer,
                        gold_answers=gold_answers,
                    )
                    prompts.append(prompt)
                    needs_api_call_count += 1

        print(
            f"\n--- Incremental Evaluation Summary ---\n"
            f"Total samples            : {len(questions)}\n"
            f"Exact matches (free)     : {exact_match_count}\n"
            f"Cached from previous CSV : {cached_match_count}\n"
            f"API calls needed         : {needs_api_call_count}\n"
            f"--------------------------------------\n"
        )

        self.dataset = Dataset.from_dict(
            {
                "input": prompts,
                "exact_match": exact_matches,
                "cached_correct": cached_corrects,
                "cached_evaluator_response": cached_evaluator_responses,
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
            "cached_correct": [item["cached_correct"] for item in batch],
            "cached_evaluator_response": [
                item["cached_evaluator_response"] for item in batch
            ],
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
