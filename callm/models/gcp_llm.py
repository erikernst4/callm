import os
import torch
from callm.extractors import BaseExtractor
from callm.models.base import BaseLightningModule
from callm.models.outputs import letter_match, log_answer_metrics, write_llm_outputs_csv

from google import genai
from google.genai import types


class GCPLLM(BaseLightningModule):
    flush_prefix = "temp_val_outputs"

    def __init__(
        self,
        extractor: BaseExtractor,
        model_name: str = "gemini-3-flash-preview",
        location: str = "global",
        return_logits: bool = False,
        flush_outputs_every_n_steps: int = -1,
        save_outputs: bool = False,
        max_new_tokens: int = 100,
        top_logprobs: int = 0,
    ):
        super().__init__(
            flush_outputs_every_n_steps=flush_outputs_every_n_steps,
            save_outputs=save_outputs,
        )

        self.model_name = model_name
        # >0: also store the top-k {token: logprob} of each step (API max 20)
        self.top_logprobs = top_logprobs

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
                # top-k candidates per step; 1 keeps only the chosen token
                config_kwargs["logprobs"] = max(1, self.top_logprobs)
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
                top_list = None
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
                        if self.top_logprobs > 0 and getattr(
                            res, "top_candidates", None
                        ):
                            top_list = [
                                {c.token: c.log_probability for c in step.candidates}
                                for step in res.top_candidates
                            ]

                if self.top_logprobs > 0:
                    out["top_logprobs"] = top_list

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

        self._maybe_flush()

        return {"batch_size": len(questions)}

    def process_outputs(self, outputs: list[dict]):
        """
        Extract answers/confidence, and save to CSV.
        If the datamodule does not require semantic equivalence checking,
        compute correctness by exact match and calibration metrics directly.
        """
        for out in outputs:
            # BaseExtractor can take logits when return_logits is True
            # if the specific extractor knows how to use them
            out["pred_answer"], out["confidence"] = self.extractor(
                out.get("raw_output", ""),
                out.get("logits"),
                out.get("output_ids"),
                top_logprobs=out.get("top_logprobs"),
            )

        write_llm_outputs_csv(
            outputs,
            os.path.join(self.log_dir, "llm_outputs.csv"),
            raw_output_max_length=200,
        )

        datamodule = self.trainer.datamodule
        if not getattr(datamodule, "requires_semantic_equivalence", False):
            for out in outputs:
                out["correct"] = letter_match(out["pred_answer"], out["gold_answers"])
            log_answer_metrics(self, outputs)
