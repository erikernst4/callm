import os
from callm.models.base import BaseLightningModule
from callm.models.outputs import (
    log_answer_metrics,
    read_evaluation_results,
    write_evaluation_results_csv,
)

from google import genai
from google.genai import types


class GCPEvaluatorModule(BaseLightningModule):
    """
    LightningModule for batched correctness evaluation using Google Cloud LLMs.
    Supports incremental evaluation by honoring cached labels from previous evaluation CSVs
    and exact matches, preventing redundant external API calls.
    """

    flush_prefix = "temp_eval_results"

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        location: str = "global",
        flush_outputs_every_n_steps: int = -1,
        save_outputs: bool = False,
        resume_from: str = None,
        max_new_tokens: int = 100,
        *args,
        **kwargs,
    ):
        super().__init__(
            flush_outputs_every_n_steps=flush_outputs_every_n_steps,
            save_outputs=save_outputs,
        )

        self.model_name = model_name

        kwargs = {"vertexai": True, "location": location}

        self.client = genai.Client(**kwargs)

        self.resume_from = resume_from
        self.max_new_tokens = max_new_tokens

    def validation_step(self, batch, batch_idx):
        """
        Evaluate correctness for a batch using GCP.
        Skips API calls for exact matches and samples already labeled in previous runs.
        """
        exact_matches = batch["exact_match"]
        questions = batch["question"]
        gold_answers = batch["gold_answers"]
        pred_answers = batch["pred_answer"]
        confidences = batch["confidence"]
        raw_outputs = batch["raw_output"]
        indices = batch["index"]

        prompts = batch.get("input")
        cached_corrects = batch.get("cached_correct")
        cached_evaluator_responses = batch.get("cached_evaluator_response")

        generated_responses = [None] * len(exact_matches)

        for i, is_exact in enumerate(exact_matches):
            is_cached = cached_corrects is not None and cached_corrects[i] is not None

            # Only call GCP if NOT an exact match and NOT already cached
            if not is_exact and not is_cached and prompts is not None and prompts[i]:
                contents = [
                    types.Content(
                        role="user", parts=[types.Part.from_text(text=prompts[i])]
                    )
                ]

                config = types.GenerateContentConfig(
                    max_output_tokens=self.max_new_tokens,
                    temperature=0.0,
                )

                try:
                    resp = self.client.models.generate_content(
                        model=self.model_name, contents=contents, config=config
                    )
                    generated_responses[i] = resp.text
                except Exception as e:
                    print(
                        f"Error calling GCP evaluator inference for question: {questions[i]} / {e}"
                    )
                    generated_responses[i] = ""
            elif is_cached:
                generated_responses[i] = (
                    cached_evaluator_responses[i]
                    if cached_evaluator_responses
                    and cached_evaluator_responses[i] is not None
                    else ""
                )

        # Store results
        for i in range(len(exact_matches)):
            self.outputs.append(
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
                    "cached_correct": cached_corrects[i]
                    if cached_corrects is not None
                    else None,
                    "evaluator_response_raw": generated_responses[i],
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
        Compute calibration metrics and save results.
        """
        outputs.sort(key=lambda x: x["index"])

        for result in outputs:
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
                response = result.get("evaluator_response_raw", "")
                response_lower = response.lower().strip() if response else ""
                result["correct"] = response_lower.startswith("yes")
                result["evaluator_response"] = response

        self.calculate_metrics()

        write_evaluation_results_csv(
            outputs,
            os.path.join(self.log_dir, "evaluation_results.csv"),
            text_limit=None,
            description="GCP Evaluation results",
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
