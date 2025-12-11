from lightning.pytorch import LightningModule
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from callm.config import CACHE_PATH, HF_TOKEN
from callm.extractors import VerbalizedConfidenceExtractor
from callm.evaluator import CorrectnessEvaluator
from callm.metrics import (
    expected_calibration_error,
    brier_score,
    cross_entropy,
    auc_score,
)


class LLM(LightningModule):
    def __init__(
        self,
        model_name: str = "flan-t5-small",
        evaluator_model_name: str = "google/flan-t5-base",
        hf_token: str = None,
        train: bool = False,
    ):
        super().__init__()

        # Use provided token or fall back to config
        self.hf_token = hf_token or HF_TOKEN
        self.model_name = model_name

        # Load main model
        if model_name in [
            "flan-t5-small",
            "flan-t5-base",
            "flan-t5-large",
            "flan-t5-xl",
            "flan-t5-xxl",
        ]:
            model_load_name = f"google/{model_name}"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_load_name, cache_dir=CACHE_PATH
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_load_name)
            self.is_seq2seq = True
        elif model_name in ["Llama-2-7b-chat-hf"]:
            model_load_name = f"meta-llama/{model_name}"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_load_name, cache_dir=CACHE_PATH, use_auth_token=self.hf_token
            )
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_load_name, padding_side="left", use_auth_token=self.hf_token
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.is_seq2seq = False
        elif model_name.startswith("Qwen/"):
            # Qwen models (e.g., Qwen/Qwen3-0.6B)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir=CACHE_PATH, trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left", trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.model.config.eos_token_id
            self.is_seq2seq = False
        else:
            raise NotImplementedError(f"Model {model_name} not supported")

        if not train:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        # Initialize extractor and evaluator
        self.extractor = VerbalizedConfidenceExtractor()
        self.evaluator = CorrectnessEvaluator(
            model_name=evaluator_model_name,
            device=self.device.type if hasattr(self.device, "type") else "cpu",
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
        Validation step: generate answer, extract confidence, evaluate correctness.
        """
        # Batch has pre-stacked tensors and questions from collate_fn
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        questions = batch["question"]
        gold_answers = batch["label"]

        # Generate outputs
        with torch.no_grad():
            output_ids = self.forward(input_ids, attention_mask)

        # Decode outputs
        # For causal models, we need to skip the input tokens (only decode the generated part)
        if self.is_seq2seq:
            # Seq2seq models: decode the full output
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        else:
            # Causal models: skip input tokens, only decode generated tokens
            # For left-padded sequences, input_ids length is the same for all items in batch
            # model.generate() returns [input_tokens..., generated_tokens...]
            # So we need to skip the original input_ids length
            input_length = input_ids.shape[1]
            outputs = []
            for i in range(len(output_ids)):
                # Get the generated tokens (everything after the original input length)
                generated_tokens = output_ids[i][input_length:]
                # Decode only the generated part
                output_text = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                outputs.append(output_text)

        # Process each sample in the batch
        for i, (output_text, question, gold_answer_list) in enumerate(
            zip(outputs, questions, gold_answers)
        ):
            # Extract answer and confidence
            pred_answer, confidence = self.extractor.extract(output_text)

            # Evaluate correctness
            is_correct = self.evaluator.evaluate(
                question, pred_answer, gold_answer_list
            )

            # Store for metrics calculation
            self.validation_outputs.append(
                {
                    "question": question,
                    "pred_answer": pred_answer,
                    "gold_answers": gold_answer_list,
                    "confidence": confidence,
                    "correct": is_correct,
                    "raw_output": output_text,
                }
            )

        return {"batch_size": len(questions)}

    def on_validation_epoch_end(self):
        """
        Calculate and log calibration metrics at the end of validation.
        """
        if len(self.validation_outputs) == 0:
            return

        # Extract confidences and correctness
        import numpy as np

        all_confidences = np.array(
            [out["confidence"] for out in self.validation_outputs]
        )
        all_correctness = np.array([out["correct"] for out in self.validation_outputs])

        # Filter out invalid confidences (NaN)
        valid_indices = ~np.isnan(all_confidences)
        n_invalid = len(all_confidences) - np.sum(valid_indices)

        # Log all outputs and failures to CSV files
        import csv
        import os

        # Helper to make raw output readable
        def short_output(txt: str, limit: int = 200) -> str:
            """Keep only first line, truncate if too long."""
            if not txt:
                return ""
            first_line = txt.splitlines()[0] if txt else ""
            return first_line[:limit] + ("..." if len(first_line) > limit else "")

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
                            out["question"],
                            out["gold_answers"][0] if out["gold_answers"] else "N/A",
                            short_output(out["pred_answer"]),
                            confidence_str,
                            "Yes" if out["correct"] else "No",
                            short_output(out["raw_output"]),
                        ]
                    )

            print(f"All outputs logged to {all_outputs_file}")
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

        # Log some example predictions for debugging
        print("\n" + "=" * 80)
        print("Sample Predictions:")
        print("=" * 80)
        for i, out in enumerate(self.validation_outputs[:5]):  # Show first 5
            print(f"\nExample {i+1}:")
            print(f"  Question: {out['question']}")
            print(f"  Gold: {out['gold_answers'][0] if out['gold_answers'] else 'N/A'}")
            print(f"  Predicted: {out['pred_answer']}")
            print(f"  Confidence: {out['confidence']:.3f}")
            print(f"  Correct: {out['correct']}")
        print("=" * 80 + "\n")

        # Clear outputs for next epoch
        self.validation_outputs = []

    def configure_optimizers(self):
        # Not training, return None
        return None
