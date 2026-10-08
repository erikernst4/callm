"""
Correctness evaluators as LightningModules for batched evaluation.

A judge decides whether each predicted answer is semantically equivalent to
the gold answers. Exact matches are correct without asking the judge.
"""

import torch
import os

from callm.utils import (
    generation_pad_token_id,
    get_tokenizer_for_model,
    initialize_model,
)
from callm.models.base import BaseLightningModule
from callm.models.outputs import (
    log_answer_metrics,
    read_evaluation_results,
    write_evaluation_results_csv,
)


class BaseEvaluatorModule(BaseLightningModule):
    """
    Batched correctness evaluation with a judge model.

    Each sample is correct if it is an exact match, else if it has a label
    from a previous evaluation (``cached_correct``), else if the judge's
    response starts with "yes". Results go to ``evaluation_results.csv`` in
    the log directory, and calibration metrics are logged.

    Subclasses ask the judge in ``_ask_judge(batch)``, returning the
    backend-specific fields stored for each sample, and read its response back
    from a stored sample in ``_judge_response(result)``.
    """

    flush_prefix = "temp_eval_results"
    # Truncation of text fields in evaluation_results.csv (None: no truncation)
    text_limit: int | None = None
    results_description = "Evaluation results"

    def __init__(
        self,
        model_name: str,
        flush_outputs_every_n_steps: int = -1,
        save_outputs: bool = False,
        resume_from: str = None,
        max_new_tokens: int = 100,
    ):
        super().__init__(
            flush_outputs_every_n_steps=flush_outputs_every_n_steps,
            save_outputs=save_outputs,
        )
        self.model_name = model_name
        self.resume_from = resume_from
        self.max_new_tokens = max_new_tokens

    def _ask_judge(self, batch) -> list[dict]:
        """Per-sample fields with the judge's (raw) answer for ``batch``."""
        raise NotImplementedError

    def _judge_response(self, result: dict) -> str | None:
        """The judge's response text for a stored sample, if it was asked."""
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """Ask the judge about a batch, skipping exact matches."""
        exact_matches = batch["exact_match"]
        indices = batch["index"]
        cached_corrects = batch.get("cached_correct")

        for i, judge_fields in enumerate(self._ask_judge(batch)):
            self.outputs.append(
                {
                    "index": indices[i].item()
                    if hasattr(indices[i], "item")
                    else indices[i],
                    "question": batch["question"][i],
                    "gold_answers": batch["gold_answers"][i],
                    "pred_answer": batch["pred_answer"][i],
                    "confidence": batch["confidence"][i],
                    "raw_output": batch["raw_output"][i],
                    "exact_match": exact_matches[i],
                    "cached_correct": cached_corrects[i]
                    if cached_corrects is not None
                    else None,
                    **judge_fields,
                }
            )

        self._maybe_flush()

        return {"batch_size": len(exact_matches)}

    def on_validation_start(self):
        """Prepare evaluation by loading previous results if resuming."""
        if self.resume_from:
            self._adopt_resume_files(self._find_resume_files(self.resume_from))

    def process_outputs(self, outputs: list[dict]):
        """
        Compute correctness and calibration metrics and save results.
        """
        # Sort by original index
        outputs.sort(key=lambda x: x["index"])

        for result in outputs:
            # Already judged (e.g. reloaded results)
            if "correct" in result:
                continue

            if result.get("exact_match"):
                result["correct"] = True
                result["evaluator_response"] = ""
            elif result.get("cached_correct") is not None:
                result["correct"] = bool(result["cached_correct"])
                result["evaluator_response"] = (
                    result.get("evaluator_response_raw") or ""
                )
            else:
                response = self._judge_response(result) or ""
                result["correct"] = response.lower().strip().startswith("yes")
                result["evaluator_response"] = response

        self.calculate_metrics()

        write_evaluation_results_csv(
            outputs,
            os.path.join(self.log_dir, "evaluation_results.csv"),
            text_limit=self.text_limit,
            description=self.results_description,
        )

    def load_evaluation_results_from_csv(self, csv_path: str):
        """
        Load evaluation results from a CSV file.

        The CSV is expected to have columns: 'Confidence' and 'Correct'.
        'Correct' should be 'Yes' or 'No'.
        """
        self.outputs = read_evaluation_results(csv_path)

    def calculate_metrics(self):
        log_answer_metrics(self, self.outputs, save_csv=False)


class EvaluatorModule(BaseEvaluatorModule):
    """
    Correctness evaluation with a local HF judge model.

    Uses an LLM to check if predicted answers are semantically equivalent
    to ground truth answers via batched generation. Generated token ids are
    stored per sample and decoded at the end of the epoch.
    """

    text_limit = 200

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        hf_token: str = None,
        flush_outputs_every_n_steps: int = -1,
        save_outputs: bool = False,
        resume_from: str = None,
        max_new_tokens: int = 100,
        *args,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            flush_outputs_every_n_steps=flush_outputs_every_n_steps,
            save_outputs=save_outputs,
            resume_from=resume_from,
            max_new_tokens=max_new_tokens,
        )

        # Load evaluator model
        self.model, self.is_seq2seq = initialize_model(model_name, hf_token)
        self.tokenizer = get_tokenizer_for_model(model_name)

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id
        self.pad_token_id = generation_pad_token_id(
            self.model, self.tokenizer, self.is_seq2seq
        )

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Generate responses for batched semantic equivalence prompts.
        """
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }
        if self.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = self.pad_token_id

        return self.model.generate(**generation_kwargs)

    def _ask_judge(self, batch) -> list[dict]:
        """
        Generate for the non-exact samples, which are the only rows of
        ``input_ids`` (see ``EvaluatorDataModule``).
        """
        exact_matches = batch["exact_match"]
        generated_ids_list = [None] * len(exact_matches)

        if batch["input_ids"] is not None:
            input_ids = batch["input_ids"]
            input_length = input_ids.shape[1] if not self.is_seq2seq else 0

            with torch.no_grad():
                output_ids = self.forward(input_ids, batch["attention_mask"])

            non_exact = [i for i, is_exact in enumerate(exact_matches) if not is_exact]
            for row, i in enumerate(non_exact):
                generated_ids_list[i] = output_ids[row][input_length:]

        return [{"output_ids": ids} for ids in generated_ids_list]

    def _judge_response(self, result: dict) -> str | None:
        if result.get("output_ids") is None:
            return None
        return self.tokenizer.decode(result["output_ids"], skip_special_tokens=True)
