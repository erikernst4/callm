from callm.models.evaluator import BaseEvaluatorModule

from google import genai
from google.genai import types


class GCPEvaluatorModule(BaseEvaluatorModule):
    """
    Correctness evaluation with a Google Cloud judge model.
    Supports incremental evaluation by honoring cached labels from previous evaluation CSVs
    and exact matches, preventing redundant external API calls.
    """

    results_description = "GCP Evaluation results"

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
            model_name=model_name,
            flush_outputs_every_n_steps=flush_outputs_every_n_steps,
            save_outputs=save_outputs,
            resume_from=resume_from,
            max_new_tokens=max_new_tokens,
        )
        self.client = genai.Client(vertexai=True, location=location)

    def _ask_judge(self, batch) -> list[dict]:
        """
        Call GCP for samples that are neither exact matches nor already labelled
        in a previous run; labelled samples keep their previous response.
        """
        prompts = batch.get("input")
        cached_corrects = batch.get("cached_correct")
        cached_evaluator_responses = batch.get("cached_evaluator_response")

        generated_responses = [None] * len(batch["exact_match"])
        for i, is_exact in enumerate(batch["exact_match"]):
            is_cached = cached_corrects is not None and cached_corrects[i] is not None

            if not is_exact and not is_cached and prompts is not None and prompts[i]:
                generated_responses[i] = self._generate(
                    prompts[i], batch["question"][i]
                )
            elif is_cached:
                generated_responses[i] = (
                    cached_evaluator_responses[i]
                    if cached_evaluator_responses
                    and cached_evaluator_responses[i] is not None
                    else ""
                )

        return [{"evaluator_response_raw": r} for r in generated_responses]

    def _generate(self, prompt: str, question: str) -> str:
        """Judge response for one prompt, or ``""`` if the API call fails."""
        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        ]
        config = types.GenerateContentConfig(
            max_output_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        try:
            resp = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )
            return resp.text
        except Exception as e:
            print(
                f"Error calling GCP evaluator inference for question: {question} / {e}"
            )
            return ""

    def _judge_response(self, result: dict) -> str | None:
        return result.get("evaluator_response_raw")
