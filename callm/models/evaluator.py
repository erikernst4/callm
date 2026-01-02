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
    ):
        super().__init__()

        self.model_name = model_name

        # Load evaluator model
        self.model, self.is_seq2seq = initialize_model(model_name, hf_token)
        self.tokenizer = get_tokenizer_for_model(model_name)

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Storage for evaluation results
        self.evaluation_results = []

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

        # Initialize correctness list
        correctness = [None] * len(exact_matches)

        # Handle exact matches
        for i, is_exact in enumerate(exact_matches):
            if is_exact:
                correctness[i] = True

        # Handle non-exact matches with batched generation
        if batch["input_ids"] is not None:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            input_length = input_ids.shape[1] if not self.is_seq2seq else 0

            with torch.no_grad():
                output_ids = self.forward(input_ids, attention_mask)

            # Decode and parse responses
            eval_idx = 0
            for i, is_exact in enumerate(exact_matches):
                if not is_exact:
                    if self.is_seq2seq:
                        response = self.tokenizer.decode(
                            output_ids[eval_idx], skip_special_tokens=True
                        )
                    else:
                        generated_tokens = output_ids[eval_idx][input_length:]
                        response = self.tokenizer.decode(
                            generated_tokens, skip_special_tokens=True
                        )

                    # Parse Yes/No
                    response_lower = response.lower().strip()
                    correctness[i] = response_lower.startswith("yes")
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
                    "correct": correctness[i],
                }
            )

        return {"batch_size": len(exact_matches)}

    def on_validation_epoch_end(self):
        """
        Compute calibration metrics and save results.
        """
        if len(self.evaluation_results) == 0:
            return

        # Sort by original index
        self.evaluation_results.sort(key=lambda x: x["index"])

        # Extract arrays
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
        self.log("val_ece", ece, prog_bar=True)
        self.log("val_brier_score", bs, prog_bar=True)
        self.log("val_cross_entropy", ce, prog_bar=True)
        self.log("val_auc", auc, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)

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
                        "Raw Output",
                    ]
                )
                for result in self.evaluation_results:
                    conf_str = (
                        f"{float(result['confidence']):.6f}"
                        if result["confidence"] not in ["nan", ""]
                        else "nan"
                    )
                    writer.writerow(
                        [
                            short_output(result["question"]),
                            result["gold_answers"],
                            short_output(result["pred_answer"]),
                            conf_str,
                            "Yes" if result["correct"] else "No",
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
