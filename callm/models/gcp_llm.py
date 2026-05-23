import os
import csv
import torch
import numpy as np
from callm.extractors import BaseExtractor
from callm.models.base import BaseLightningModule
from ecuas import (
    ExpectedCalibrationError,
    ConfidenceBrierScore,
    ConfidenceCrossEntropy,
    ConfidenceAUCScore,
    CCAS,
)

from google import genai
from google.genai import types


class GCPLLM(BaseLightningModule):
    def __init__(
        self,
        extractor: BaseExtractor,
        model_name: str = "gemini-3-flash-preview",
        location: str = "global",
        return_logits: bool = False,
        flush_outputs_every_n_steps: int = -1,
        save_outputs: bool = False,
        max_new_tokens: int = 100,
    ):
        super().__init__(
            flush_outputs_every_n_steps=flush_outputs_every_n_steps,
            save_outputs=save_outputs,
        )

        self.model_name = model_name

        # Security: Allow implicit credentials (like GOOGLE_APPLICATION_CREDENTIALS)
        # or require explicitly setting project logic securely, without hardcoding service account JSON paths inside the code.
        kwargs = {"vertexai": True, "location": location}

        self.client = genai.Client(**kwargs)

        self.extractor: BaseExtractor = extractor
        self.return_logits = return_logits
        self.max_new_tokens = max_new_tokens

    def _build_contents(self, prompt_text):
        """Convert prompt input to GCP Content objects.

        Args:
            prompt_text: Either a plain string (from Prompt), a list of
                message dicts in GCP format (from ChatPrompt with gcp=True),
                or a list of message dicts in standard format (from ChatPrompt
                with gcp=False).

        Returns:
            Tuple of (contents list, system_instruction string or None).
        """
        system_instruction = None

        if isinstance(prompt_text, list):
            contents = []
            for msg in prompt_text:
                role = msg["role"]
                # Extract text from either format
                if "parts" in msg:
                    text = msg["parts"][0]["text"]
                else:
                    text = msg["content"]

                if role == "system":
                    system_instruction = text
                else:
                    # Map "assistant" role to "model" for GCP
                    gcp_role = "model" if role == "assistant" else role
                    contents.append(
                        types.Content(
                            role=gcp_role,
                            parts=[types.Part.from_text(text=text)],
                        )
                    )
        else:
            # Plain string prompt
            contents = [
                types.Content(
                    role="user", parts=[types.Part.from_text(text=prompt_text)]
                )
            ]

        return contents, system_instruction

    def validation_step(self, batch, batch_idx):
        """
        Validation step: make API calls to GCP for each text input in the batch.
        """
        inputs = batch["input"]
        questions = batch["question"]
        gold_answers = (
            batch["gold_answers"]
            if "gold_answers" in batch
            else (
                batch["label"]
                if "label" in batch
                else [None for _ in range(len(questions))]
            )
        )
        choices = batch.get("choices", [None for _ in range(len(questions))])

        for i, (prompt_text, question, gold_answer_list, choice) in enumerate(
            zip(inputs, questions, gold_answers, choices)
        ):
            contents, system_instruction = self._build_contents(prompt_text)

            # Use `return_logits` appropriately
            # In google.genai API, you request logprobs with specific configs
            # And then fetch them from the response candidates
            config_kwargs = {
                "max_output_tokens": self.max_new_tokens,
                "temperature": 0.0,
            }
            if self.return_logits:
                config_kwargs["response_logprobs"] = True
                config_kwargs["logprobs"] = 1  # fetch logprob only for the chosen token
            if system_instruction:
                config_kwargs["system_instruction"] = system_instruction

            config = types.GenerateContentConfig(**config_kwargs)

            out = {
                "question": question,
                "gold_answers": gold_answer_list,
                # For compatibility with standard huggingface LLM extraction
                "output_ids": None,  # GCP abstract away tokens for standard interactions
            }
            if choice is not None:
                out["choices"] = choice
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name, contents=contents, config=config
                )
                raw_text = resp.text or ""

                # Fetch logits if configured
                logits_list = None
                if (
                    self.return_logits
                    and hasattr(resp, "candidates")
                    and resp.candidates
                ):
                    candidate = resp.candidates[0]
                    if (
                        hasattr(candidate, "logprobs_result")
                        and candidate.logprobs_result
                    ):
                        res = candidate.logprobs_result
                        # Depending on the chosen API version behavior,
                        # chosen_candidates gives the log prob sequence of chosen tokens
                        if hasattr(res, "chosen_candidates") and res.chosen_candidates:
                            logits_list = [
                                token.log_probability for token in res.chosen_candidates
                            ]

                if logits_list is not None:
                    out["logits"] = torch.tensor(logits_list, dtype=torch.float32)
                else:
                    if self.return_logits:
                        # Fallback if no logits were returned by the API
                        print(f"Warning: No logits returned for question '{question}'")
                    out["logits"] = None

                out["raw_output"] = raw_text

            except Exception as e:
                print(f"Error calling GCP inference for question: {question} / {e}")
                out["raw_output"] = ""
                out["logits"] = None

            self.outputs.append(out)

        # Periodically flush outputs to disk to save memory
        if (
            self.flush_outputs_every_n_steps > 0
            and len(self.outputs) >= self.flush_outputs_every_n_steps
        ):
            self._flush_outputs(prefix="temp_val_outputs")

        return {"batch_size": len(questions)}

    def on_validation_epoch_end(self):
        """
        Decode outputs, extract answers/confidence, and save to CSV.
        If the datamodule does not require semantic equivalence checking,
        compute correctness and calibration metrics directly.
        """
        if self.outputs and (
            self.flushed_output_files or self.flush_outputs_every_n_steps > 0
        ):
            self._flush_outputs(prefix="temp_val_outputs")

        self._reload_flushed_outputs()

        if len(self.outputs) == 0:
            return

        # Extract answer and confidence using the same BaseExtractor logic
        for out in self.outputs:
            raw_output = out.get("raw_output", "")

            # BaseExtractor can take logits when return_logits is True
            # if the specific extractor knows how to use them
            pred_answer, confidence = self.extractor(
                raw_output, out.get("logits"), out.get("output_ids")
            )
            out["pred_answer"] = pred_answer
            out["confidence"] = confidence

        log_dir = self.trainer.log_dir or os.getcwd()
        outputs_file = os.path.join(log_dir, "llm_outputs.csv")

        def short_output(txt: str, limit: int = 200) -> str:
            if not txt:
                return ""
            truncated_text = txt[:limit] + ("..." if len(txt) > limit else "")
            return truncated_text.replace("\n", "\\n")

        try:
            with open(outputs_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                has_choices = any("choices" in out for out in self.outputs)
                headers = [
                    "question",
                    "gold_answers",
                ]
                if has_choices:
                    headers.append("choices")
                headers.extend(
                    [
                        "pred_answer",
                        "confidence",
                        "raw_output",
                    ]
                )
                writer.writerow(headers)

                for out in self.outputs:
                    gold_str = (
                        "|".join(out["gold_answers"]) if out["gold_answers"] else ""
                    )
                    conf_str = (
                        f"{out['confidence']:.6f}"
                        if out["confidence"] is not None
                        and not (
                            isinstance(out["confidence"], float)
                            and out["confidence"] != out["confidence"]
                        )
                        else "nan"
                    )

                    row = [out["question"], gold_str]
                    if has_choices:
                        row.append(out.get("choices", ""))
                    row.extend(
                        [
                            out["pred_answer"],
                            conf_str,
                            short_output(out["raw_output"]),
                        ]
                    )
                    writer.writerow(row)
            print(f"\nGCP LLM outputs saved to {outputs_file}")
        except Exception as e:
            print(f"Failed to save GCP LLM outputs: {e}")

        # If the datamodule does not require semantic equivalence,
        # compute correctness and metrics directly
        datamodule = self.trainer.datamodule
        if not getattr(datamodule, "requires_semantic_equivalence", False):
            self._calculate_metrics_direct()

        # Clear outputs for next epoch
        self.outputs = []

    def _calculate_metrics_direct(self):
        """
        Compute correctness by exact string match and calculate
        calibration metrics. Used when semantic equivalence checking
        is not required (e.g. MMLU multiple-choice).
        """
        # Determine correctness by exact match
        for out in self.outputs:
            pred = (out["pred_answer"] or "").strip().upper()
            gold = out["gold_answers"]
            if isinstance(gold, list):
                out["correct"] = any(pred == g.strip().upper() for g in gold)
            else:
                out["correct"] = pred == (gold or "").strip().upper()

        all_confidences = []
        all_correctness = []
        for out in self.outputs:
            try:
                conf = float(out["confidence"])
            except (ValueError, TypeError):
                conf = float("nan")
            all_confidences.append(conf)
            all_correctness.append(out["correct"])

        all_confidences = np.array(all_confidences)
        all_correctness = np.array(all_correctness)

        # Log accuracy on all samples
        accuracy = float(all_correctness.mean()) if len(all_correctness) > 0 else 0.0
        self.log("val_accuracy", accuracy, prog_bar=True, sync_dist=True)

        metrics_results = {"val_accuracy": accuracy}

        n_invalid = np.isnan(all_confidences).sum()
        if n_invalid > 0:
            print(
                f"Warning: {n_invalid} samples have invalid confidence (NaN). "
                "Using fallback confidence of 0.5 for these samples."
            )

        confidences = torch.tensor(all_confidences, dtype=torch.float32)
        correctness = torch.tensor(all_correctness, dtype=torch.float32)

        # NaN confidences are handled internally by the metrics (fallback to 0.5)
        metrics = {
            "val_ece": ExpectedCalibrationError(n_bins=10),
            "val_brier_score": ConfidenceBrierScore(),
            "val_cross_entropy": ConfidenceCrossEntropy(),
            "val_auc": ConfidenceAUCScore(),
            "val_ccas": CCAS(),
        }

        for name, metric in metrics.items():
            metric.update(confidences, correctness)
            value = float(metric.compute())
            self.log(name, value, prog_bar=True, sync_dist=True)
            metrics_results[name] = value

        # Save metrics to CSV in the log directory
        log_dir = self.trainer.log_dir or os.getcwd()
        metrics_file = os.path.join(log_dir, "metrics.csv")
        try:
            with open(metrics_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(list(metrics_results.keys()))
                writer.writerow(list(metrics_results.values()))
            print(f"Metrics saved to {metrics_file}")
        except Exception as e:
            print(f"Failed to save metrics: {e}")
