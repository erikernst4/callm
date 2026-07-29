"""
Exact answer extractor module using GCP Gemini API.

Analogous to EvaluatorModule/GCPEvaluatorModule, this module extracts the
short exact answer entity from model-generated long answers, using either
substring matching (for correct answers) or GCP LLM inference (for incorrect ones).
"""

import os
import csv

from callm.models.base import BaseLightningModule

from google import genai
from google.genai import types


class ExactAnswerExtractorModule(BaseLightningModule):
    """
    LightningModule for batched exact answer extraction using Google Cloud LLMs.

    For correct predictions, performs substring search of gold answers in raw_output.
    For incorrect predictions, queries Gemini to extract the short answer entity.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.5-flash-lite",
        location: str = "global",
        flush_outputs_every_n_steps: int = -1,
        save_outputs: bool = False,
        max_new_tokens: int = 100,
        max_retries: int = 3,
        *args,
        **kwargs,
    ):
        super().__init__(
            flush_outputs_every_n_steps=flush_outputs_every_n_steps,
            save_outputs=save_outputs,
        )

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries

        client_kwargs = {"vertexai": True, "location": location}
        self.client = genai.Client(**client_kwargs)

    def _find_verbatim_substring(self, query: str, text: str):
        """
        Finds the exact verbatim substring in text that matches query (case-insensitively).
        Returns (verbatim_substring, True) if found, else (None, False).
        """
        import re

        if not query or not text or query.strip().upper() == "NO ANSWER":
            return None, False

        # Clean leading/trailing markdown/quotes/periods from query
        clean_query = (
            query.strip().split("\n")[0].split("(")[0].strip().strip(".*_\"'`")
        )
        if not clean_query:
            return None, False

        # 1. Direct case-insensitive search
        idx = text.lower().find(clean_query.lower())
        if idx != -1:
            return text[idx : idx + len(clean_query)], True

        # 2. Match span allowing markdown symbols inside query (e.g. query: "Parkinson's Law", text: "**Parkinson's Law**")
        try:
            escaped_query = re.escape(clean_query)
            pattern = re.sub(r"\\ ", r"[\\*\\_\\s]+", escaped_query)
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0), True
        except Exception:
            pass

        return None, False

    def _find_gold_answer_in_output(self, gold_answers, raw_output):
        """For correct answers, find the earliest verbatim gold answer substring in raw_output."""
        if not raw_output or not isinstance(raw_output, str):
            return None

        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]

        best_span = None
        best_index = len(raw_output)

        for ans in gold_answers:
            if not ans or not isinstance(ans, str):
                continue
            span, found = self._find_verbatim_substring(ans, raw_output)
            if found:
                idx = raw_output.lower().find(span.lower())
                if idx != -1 and idx < best_index:
                    best_span = span
                    best_index = idx

        return best_span

    def _extract_via_gemini(self, prompt_text, raw_output):
        """Query Gemini to extract the short answer, and verify it is a verbatim substring of raw_output."""
        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])
        ]

        config = types.GenerateContentConfig(
            max_output_tokens=self.max_new_tokens,
            temperature=0.0,
        )

        gemini_output = ""
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name, contents=contents, config=config
                )
                gemini_output = (resp.text or "").strip()

                if gemini_output.upper() == "NO ANSWER":
                    return "NO ANSWER", True, gemini_output

                span, found = self._find_verbatim_substring(gemini_output, raw_output)
                if found:
                    return span, True, gemini_output

            except Exception as e:
                print(f"Gemini extraction attempt {attempt + 1} failed: {e}")
                continue

        return "NO ANSWER", False, gemini_output

    def validation_step(self, batch, batch_idx):
        """Extract exact answers for a batch."""
        questions = batch["question"]
        gold_answers_list = batch["gold_answers"]
        pred_answers = batch["pred_answer"]
        corrects = batch["correct"]
        raw_outputs = batch["raw_output"]
        confidences = batch["confidence"]
        indices = batch["index"]
        needs_extraction = batch["needs_extraction"]
        prompts = batch["input"]

        for i in range(len(questions)):
            result = {
                "index": indices[i].item()
                if hasattr(indices[i], "item")
                else indices[i],
                "question": questions[i],
                "gold_answers": gold_answers_list[i],
                "pred_answer": pred_answers[i],
                "correct": corrects[i],
                "raw_output": raw_outputs[i],
                "confidence": confidences[i],
                "gemini_output": "",
            }

            if not needs_extraction[i]:
                # Correct answer: find gold answer substring
                gold_ans = gold_answers_list[i]
                exact = self._find_gold_answer_in_output(gold_ans, raw_outputs[i])
                if exact:
                    result["exact_answer"] = exact
                    result["valid_exact_answer"] = True
                else:
                    span, found = self._find_verbatim_substring(
                        pred_answers[i], raw_outputs[i]
                    )
                    if found:
                        result["exact_answer"] = span
                        result["valid_exact_answer"] = True
                    else:
                        result["exact_answer"] = "NO ANSWER"
                        result["valid_exact_answer"] = False
            else:
                # Incorrect answer: use Gemini extraction
                exact_answer, valid, gemini_out = self._extract_via_gemini(
                    prompts[i], raw_outputs[i]
                )
                result["exact_answer"] = exact_answer
                result["valid_exact_answer"] = valid
                result["gemini_output"] = gemini_out

            self.outputs.append(result)

        if (
            self.flush_outputs_every_n_steps > 0
            and len(self.outputs) >= self.flush_outputs_every_n_steps
        ):
            self._flush_outputs(prefix="temp_exact_answer_results")

        return {"batch_size": len(questions)}

    def on_validation_epoch_end(self):
        """Save extraction results to CSV."""
        if self.outputs and (
            self.flushed_output_files or self.flush_outputs_every_n_steps > 0
        ):
            self._flush_outputs(prefix="temp_exact_answer_results")

        self._reload_flushed_outputs()

        if len(self.outputs) == 0:
            return

        self.outputs.sort(key=lambda x: x["index"])

        total = len(self.outputs)
        valid_count = sum(1 for r in self.outputs if r.get("valid_exact_answer"))
        no_answer_count = sum(
            1 for r in self.outputs if r.get("exact_answer", "").upper() == "NO ANSWER"
        )
        print(
            f"\nExtraction stats: {valid_count}/{total} valid, {no_answer_count} NO ANSWER"
        )

        log_dir = self.trainer.log_dir or os.getcwd()
        results_file = os.path.join(log_dir, "evaluation_results_exact.csv")

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
                        "Gemini Output",
                        "Exact Answer",
                        "Valid Exact Answer",
                        "Raw Output",
                    ]
                )
                for result in self.outputs:
                    conf_str = (
                        f"{float(result['confidence']):.6f}"
                        if result["confidence"] not in ["nan", "", None]
                        else "nan"
                    )

                    gold_str = result["gold_answers"]
                    if isinstance(gold_str, list):
                        gold_str = str(gold_str)

                    writer.writerow(
                        [
                            (result["question"] or "").replace("\n", "\\n"),
                            gold_str,
                            (result["pred_answer"] or "").replace("\n", "\\n"),
                            conf_str,
                            "Yes" if result["correct"] else "No",
                            (result.get("gemini_output", "") or "").replace(
                                "\n", "\\n"
                            ),
                            result.get("exact_answer", ""),
                            "Yes" if result.get("valid_exact_answer") else "No",
                            (result["raw_output"] or "").replace("\n", "\\n"),
                        ]
                    )

            print(f"\nExact answer results saved to {results_file}")
        except Exception as e:
            print(f"Failed to save exact answer results: {e}")

        self.outputs = []
