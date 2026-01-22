"""
Correctness evaluator as a LightningModule for batched evaluation.

Performs batched inference to determine if predicted answers are
semantically equivalent to ground truth answers.
"""

from lightning.pytorch import LightningModule
import torch
import numpy as np
import os
import csv
import shutil
import glob

from callm.utils import initialize_model, get_tokenizer_for_model
from callm.metrics import (
    expected_calibration_error,
    brier_score,
    cross_entropy,
    auc_score,
)


class EvaluatorModule(LightningModule):
    """
    LightningModule for batched correctness evaluation.

    Uses an LLM to check if predicted answers are semantically equivalent
    to ground truth answers via batched inference.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        hf_token: str = None,
        flush_outputs_every_n_steps: int = -1,
        save_outputs: bool = False,
        resume_from: str = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model_name = model_name
        self.initial_flushed_files = []

        # Load evaluator model
        self.model, self.is_seq2seq = initialize_model(model_name, hf_token)
        self.tokenizer = get_tokenizer_for_model(model_name)

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.flush_outputs_every_n_steps = flush_outputs_every_n_steps
        self.save_outputs = save_outputs
        self.resume_from = resume_from

        # Storage for evaluation results
        self.evaluation_results = []
        self.flushed_output_files = []

    def forward(self, input_ids, attention_mask):
        """
        Generate responses for batched semantic equivalence prompts.
        """
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 100,
            "do_sample": False,
        }

        if not self.is_seq2seq:
            if (
                hasattr(self.model.config, "pad_token_id")
                and self.model.config.pad_token_id is not None
            ):
                generation_kwargs["pad_token_id"] = self.model.config.pad_token_id
            elif (
                hasattr(self.tokenizer, "pad_token_id")
                and self.tokenizer.pad_token_id is not None
            ):
                generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        return self.model.generate(**generation_kwargs)

    def validation_step(self, batch, batch_idx):
        """
        Evaluate correctness for a batch.
        """
        exact_matches = batch["exact_match"]
        questions = batch["question"]
        gold_answers = batch["gold_answers"]
        pred_answers = batch["pred_answer"]
        confidences = batch["confidence"]
        raw_outputs = batch["raw_output"]
        indices = batch["index"]

        # Handle non-exact matches with batched generation
        generated_ids_list = [None] * len(exact_matches)

        if batch["input_ids"] is not None:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            input_length = input_ids.shape[1] if not self.is_seq2seq else 0

            with torch.no_grad():
                output_ids = self.forward(input_ids, attention_mask)

            # Store generated IDs
            eval_idx = 0
            for i, is_exact in enumerate(exact_matches):
                if not is_exact:
                    if self.is_seq2seq:
                        generated_ids_list[i] = output_ids[eval_idx]
                    else:
                        generated_ids_list[i] = output_ids[eval_idx][input_length:]
                    eval_idx += 1

        # Store results
        for i in range(len(exact_matches)):
            self.evaluation_results.append(
                {
                    "index": indices[i].item()
                    if hasattr(indices[i], "item")
                    else indices[i],
                    "question": questions[i],
                    "gold_answers": gold_answers[i],
                    "pred_answer": pred_answers[i],
                    "confidence": confidences[i],
                    "raw_output": raw_outputs[i],
                    "exact_match": exact_matches[i],
                    "output_ids": generated_ids_list[i],
                }
            )

        # Periodically flush results to disk to save memory
        if (
            self.flush_outputs_every_n_steps > 0
            and len(self.evaluation_results) >= self.flush_outputs_every_n_steps
        ):
            self._flush_evaluation_results()

        return {"batch_size": len(exact_matches)}

    def _flush_evaluation_results(self):
        """Helper to save current evaluation results to a temporary file."""
        if not self.evaluation_results:
            return

        # Use trainer log_dir or current directory
        log_dir = self.trainer.log_dir or os.getcwd()
        os.makedirs(log_dir, exist_ok=True)

        batch_idx = len(self.flushed_output_files)
        filename = os.path.join(
            log_dir, f"temp_eval_results_rank{self.global_rank}_{batch_idx}.pt"
        )

        torch.save(self.evaluation_results, filename)
        self.flushed_output_files.append(filename)
        self.evaluation_results = []  # Clear memory

    def on_validation_start(self):
        """Prepare evaluation by loading previous results if resuming."""
        if self.resume_from:
            if not os.path.exists(self.resume_from):
                raise ValueError(f"Resume path {self.resume_from} does not exist.")
            else:
                # Look for files matching pattern
                pattern = os.path.join(self.resume_from, "temp_eval_results_rank0_*.pt")
                found_files = sorted(
                    glob.glob(pattern),
                    key=lambda x: int(x.split("_")[-1].split(".")[0]),
                )

                if found_files:
                    print(f"Found {len(found_files)} temp files to resume from.")
                    self.initial_flushed_files = found_files

        for i, src_path in enumerate(self.initial_flushed_files):
            # We enforce the naming convention: temp_eval_results_rank{rank}_{i}.pt
            # Start index from 0
            filename = os.path.join(
                self.trainer.log_dir, f"temp_eval_results_rank{self.global_rank}_{i}.pt"
            )
            # Only copy if source is different from destination
            if os.path.abspath(src_path) != os.path.abspath(filename):
                shutil.copy(src_path, filename)

            self.flushed_output_files.append(filename)

    def on_validation_epoch_end(self):
        """
        Compute calibration metrics and save results.
        """
        # Flush any remaining results only if we have already flushed some,
        # or if periodic flushing is enabled.
        if self.evaluation_results and (
            self.flushed_output_files or self.flush_outputs_every_n_steps > 0
        ):
            self._flush_evaluation_results()

        # If we have flushed files, reload them all
        if self.flushed_output_files:
            all_results = []
            for filepath in self.flushed_output_files:
                try:
                    chunk = torch.load(filepath)
                    all_results.extend(chunk)
                except Exception as e:
                    print(f"Error loading flushed file {filepath}: {e}")
                finally:
                    # Clean up file
                    if not self.save_outputs and os.path.exists(filepath):
                        os.remove(filepath)

            self.flushed_output_files = []  # Reset list
            self.evaluation_results = all_results  # Restore full list for processing

        if len(self.evaluation_results) == 0:
            return

        # Sort by original index
        self.evaluation_results.sort(key=lambda x: x["index"])

        # Decode and compute correctness
        for result in self.evaluation_results:
            if "correct" in result:
                continue

            if result.get("exact_match"):
                result["correct"] = True
            else:
                if result.get("output_ids") is not None:
                    response = self.tokenizer.decode(
                        result["output_ids"], skip_special_tokens=True
                    )
                    response_lower = response.lower().strip()
                    result["correct"] = response_lower.startswith("yes")
                    result["evaluator_response"] = response
                else:
                    result["correct"] = False
                    result["evaluator_response"] = ""

        self.calculate_metrics()

        # Save final results
        log_dir = self.trainer.log_dir or os.getcwd()
        results_file = os.path.join(log_dir, "evaluation_results.csv")

        def short_output(txt: str, limit: int = 200) -> str:
            if not txt:
                return ""
            truncated = txt[:limit] + ("..." if len(txt) > limit else "")
            return truncated.replace("\n", "\\n")

        try:
            with open(results_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Question",
                        "Gold Answers",
                        "Predicted Answer",
                        "Confidence",
                        "Correct",
                        "Evaluator Response",
                        "Raw Output",
                    ]
                )
                for result in self.evaluation_results:
                    conf_str = (
                        f"{float(result['confidence']):.6f}"
                        if result["confidence"] not in ["nan", ""]
                        else "nan"
                    )
                    evaluator_resp = result.get("evaluator_response", "")
                    writer.writerow(
                        [
                            short_output(result["question"]),
                            result["gold_answers"],
                            short_output(result["pred_answer"]),
                            conf_str,
                            "Yes" if result["correct"] else "No",
                            short_output(evaluator_resp),
                            short_output(result["raw_output"]),
                        ]
                    )
            print(f"\nEvaluation results saved to {results_file}")
        except Exception as e:
            print(f"Failed to save evaluation results: {e}")

        # Clear for next epoch
        self.evaluation_results = []

    def configure_optimizers(self):
        return None

    def load_evaluation_results_from_csv(self, csv_path: str):
        """
        Load evaluation results from a CSV file.

        The CSV is expected to have columns: 'Confidence' and 'Correct'.
        'Correct' should be 'Yes' or 'No'.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.evaluation_results = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    confidence = float(row["Confidence"])
                except (ValueError, KeyError):
                    confidence = float("nan")

                is_correct = row.get("Correct", "").strip().lower() == "yes"

                self.evaluation_results.append(
                    {
                        "confidence": confidence,
                        "correct": is_correct,
                        # Other fields are not strictly necessary for calculate_metrics
                        # but we can keep them for completeness if needed.
                        "question": row.get("Question", ""),
                        "gold_answers": row.get("Gold Answers", ""),
                        "pred_answer": row.get("Predicted Answer", ""),
                    }
                )
        print(f"Loaded {len(self.evaluation_results)} results from {csv_path}")

    def calculate_metrics(self):
        all_confidences = []
        all_correctness = []
        for result in self.evaluation_results:
            try:
                conf = float(result["confidence"])
            except (ValueError, TypeError):
                conf = float("nan")
            all_confidences.append(conf)
            all_correctness.append(result["correct"])

        all_confidences = np.array(all_confidences)
        all_correctness = np.array(all_correctness)

        # Filter invalid confidences
        valid_indices = ~np.isnan(all_confidences)
        n_invalid = len(all_confidences) - np.sum(valid_indices)

        if n_invalid > 0:
            print(
                f"\nWarning: {n_invalid} samples have invalid confidence (NaN). "
                "Ignoring them for metrics."
            )

        confidences = all_confidences[valid_indices]
        correctness = all_correctness[valid_indices]

        # Calculate metrics
        ece = expected_calibration_error(confidences, correctness, n_bins=10)
        bs = brier_score(confidences, correctness)
        ce = cross_entropy(confidences, correctness)
        auc = auc_score(confidences, correctness)
        accuracy = float(np.mean(correctness)) if len(correctness) > 0 else 0.0

        # Log metrics
        self.log("val_ece", ece, prog_bar=True, sync_dist=True)
        self.log("val_brier_score", bs, prog_bar=True, sync_dist=True)
        self.log("val_cross_entropy", ce, prog_bar=True, sync_dist=True)
        self.log("val_auc", auc, prog_bar=True, sync_dist=True)
        self.log("val_accuracy", accuracy, prog_bar=True, sync_dist=True)
