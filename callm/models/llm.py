from lightning.pytorch import LightningModule
import torch
import numpy as np
import os
import csv
from callm.extractors import VerbalizedConfidenceExtractor
from callm.evaluator import CorrectnessEvaluator
from callm.metrics import (
    expected_calibration_error,
    brier_score,
    cross_entropy,
    auc_score,
)
from callm.utils import initialize_model, get_tokenizer_for_model


class LLM(LightningModule):
    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        evaluator_model_name: str = "google/flan-t5-base",
        hf_token: str = None,
        train: bool = False,
    ):
        super().__init__()

        self.model_name = model_name

        # Load main model
        self.model, self.is_seq2seq = initialize_model(model_name, hf_token)
        self.tokenizer = get_tokenizer_for_model(model_name)

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        if not train:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        # Initialize extractor and evaluator
        self.extractor = VerbalizedConfidenceExtractor()
        self.evaluator = CorrectnessEvaluator(
            model_name=evaluator_model_name,
        )

        # Storage for validation predictions
        self.validation_outputs = []

    def forward(self, input_ids, attention_mask):
        """
        Generate text output from the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Generated token IDs
        """
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 100,
            "do_sample": False,
        }

        # For causal models, ensure pad_token_id is set to avoid issues
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

    def training_step(self, batch, batch_idx):
        # Not implemented for this task
        pass

    def validation_step(self, batch, batch_idx):
        """
        Validation step: generate output IDs only.
        Decoding and evaluation are deferred to on_validation_epoch_end for efficiency.
        """
        # Batch has pre-stacked tensors and questions from collate_fn
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        questions = batch["question"]
        gold_answers = batch["label"]

        with torch.no_grad():
            output_ids = self.forward(input_ids, attention_mask)

        # Store output IDs and metadata for later processing at epoch end
        # Keep tensors on GPU to avoid per-batch CPU transfer overhead
        input_length = input_ids.shape[1] if not self.is_seq2seq else 0

        for i, (question, gold_answer_list) in enumerate(zip(questions, gold_answers)):
            self.validation_outputs.append(
                {
                    "output_ids": output_ids[i],
                    "input_length": input_length,
                    "question": question,
                    "gold_answers": gold_answer_list,
                }
            )

        return {"batch_size": len(questions)}

    def on_validation_epoch_end(self):
        """
        Decode outputs, evaluate correctness, and calculate calibration metrics.
        """
        if len(self.validation_outputs) == 0:
            return

        # Step 1: Decode all output IDs and extract answers/confidence
        for out in self.validation_outputs:
            if self.is_seq2seq:
                raw_output = self.tokenizer.decode(
                    out["output_ids"], skip_special_tokens=True
                )
            else:
                # For causal models, skip input tokens
                generated_tokens = out["output_ids"][out["input_length"] :]
                raw_output = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
            out["raw_output"] = raw_output

            # Extract answer and confidence
            pred_answer, confidence = self.extractor.extract(raw_output)
            out["pred_answer"] = pred_answer
            out["confidence"] = confidence

        # Step 2: Evaluate correctness for all outputs
        for out in self.validation_outputs:
            out["correct"] = self.evaluator.evaluate(
                out["question"], out["pred_answer"], out["gold_answers"][0]
            )

        all_confidences = np.array(
            [out["confidence"] for out in self.validation_outputs]
        )
        all_correctness = np.array([out["correct"] for out in self.validation_outputs])

        # Filter out invalid confidences (NaN)
        valid_indices = ~np.isnan(all_confidences)
        n_invalid = len(all_confidences) - np.sum(valid_indices)

        # Helper to make raw output readable
        def short_output(txt: str, limit: int = 200) -> str:
            """Keep only first line, truncate if too long."""
            if not txt:
                return ""
            truncated_text = txt[:limit] + ("..." if len(txt) > limit else "")
            return truncated_text.replace("\n", "\\n")

        # Use log_dir if available, else current directory
        log_dir = self.trainer.log_dir or os.getcwd()
        all_outputs_file = os.path.join(log_dir, "all_outputs.csv")
        failure_file = os.path.join(log_dir, "failures.csv")

        # Write all outputs to CSV
        try:
            with open(all_outputs_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Question",
                        "Gold Answer",
                        "Predicted Answer",
                        "Confidence",
                        "Correct",
                        "Raw Output",
                    ]
                )

                for out in self.validation_outputs:
                    confidence_str = (
                        f"{out['confidence']:.6f}"
                        if not np.isnan(out["confidence"])
                        else "nan"
                    )
                    writer.writerow(
                        [
                            short_output(out["question"]),
                            out["gold_answers"][0] if out["gold_answers"] else "N/A",
                            short_output(out["pred_answer"]),
                            confidence_str,
                            "Yes" if out["correct"] else "No",
                            short_output(out["raw_output"]),
                        ]
                    )
        except Exception as e:
            print(f"Failed to log all outputs: {e}")

        # Write failures to separate CSV (only if there are failures)
        if n_invalid > 0:
            print(
                f"\nWarning: {n_invalid} samples have invalid confidence (NaN). Ignoring them for metrics."
            )

            try:
                with open(failure_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "Question",
                            "Gold Answer",
                            "Predicted Answer",
                            "Confidence",
                            "Raw Output",
                            "Reason",
                        ]
                    )

                    # Find invalid indices
                    invalid_indices = np.where(~valid_indices)[0]
                    for idx in invalid_indices:
                        out = self.validation_outputs[idx]
                        writer.writerow(
                            [
                                out["question"],
                                out["gold_answers"][0]
                                if out["gold_answers"]
                                else "N/A",
                                short_output(
                                    out["pred_answer"]
                                ),  # Truncate predicted answer too
                                out["confidence"],
                                short_output(out["raw_output"]),
                                "Extractor could not parse confidence",
                            ]
                        )

                print(f"Invalid samples logged to {failure_file}")
            except Exception as e:
                print(f"Failed to log failures: {e}")

        confidences = all_confidences[valid_indices]
        correctness = all_correctness[valid_indices]

        # Calculate metrics
        ece = expected_calibration_error(confidences, correctness, n_bins=10)
        bs = brier_score(confidences, correctness)
        ce = cross_entropy(confidences, correctness)
        auc = auc_score(confidences, correctness)

        # Calculate accuracy
        accuracy = float(np.mean(correctness)) if len(correctness) > 0 else 0.0

        # Log metrics
        self.log("val_ece", ece, prog_bar=True)
        self.log("val_brier_score", bs, prog_bar=True)
        self.log("val_cross_entropy", ce, prog_bar=True)
        self.log("val_auc", auc, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)

        # Clear outputs for next epoch
        self.validation_outputs = []

    def configure_optimizers(self):
        # Not training, return None
        return None
