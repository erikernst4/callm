from __future__ import annotations

import torch
import os
from callm.extractors import BaseExtractor
from callm.utils import initialize_model, get_tokenizer_for_model
from callm.models.base import BaseLightningModule
from callm.models.outputs import (
    get_gold_answers,
    letter_match,
    log_answer_metrics,
    write_llm_outputs_csv,
)


class LLM(BaseLightningModule):
    flush_prefix = "temp_val_outputs"

    def __init__(
        self,
        extractor: BaseExtractor,
        model_name: str = "google/flan-t5-small",
        hf_token: str = None,
        train: bool = False,
        return_logits: bool = False,
        flush_outputs_every_n_steps: int = -1,
        save_outputs: bool = False,
        max_new_tokens: int = 100,
        raw_output_max_length: int | None = None,
    ):
        super().__init__(
            flush_outputs_every_n_steps=flush_outputs_every_n_steps,
            save_outputs=save_outputs,
        )

        self.model_name = model_name

        # Load main model
        self.model, self.is_seq2seq = initialize_model(model_name, hf_token)
        self.tokenizer = get_tokenizer_for_model(model_name)

        if hasattr(self.model.config, "text_config"):
            self.model.config.pad_token_id = self.model.config.text_config.pad_token_id
        elif self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # Causal models get an explicit pad_token_id for generate()
        self.pad_token_id = None
        if not self.is_seq2seq:
            self.pad_token_id = getattr(self.model.config, "pad_token_id", None)
            if self.pad_token_id is None:
                self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)

        if not train:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.extractor: BaseExtractor = extractor
        self.return_logits = return_logits
        self.max_new_tokens = max_new_tokens
        self.raw_output_max_length = raw_output_max_length

    def forward(
        self,
        input_ids,
        attention_mask,
        generate: bool = True,
        return_hidden_states: bool = False,
    ):
        """
        Forward pass of LLM to generate text and/or return hidden states.
        """
        if not generate:
            return self._hidden_states(input_ids, attention_mask)

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "output_scores": self.return_logits,
            "return_dict_in_generate": self.return_logits,
        }
        if self.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = self.pad_token_id

        generated_ids = self.model.generate(**generation_kwargs)

        if return_hidden_states:
            return generated_ids, self._hidden_states(input_ids, attention_mask)
        return generated_ids

    def _hidden_states(self, input_ids, attention_mask):
        """Hidden states of all layers for a non-generative forward pass."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states

    @staticmethod
    def _generated_token_log_probs(
        scores: tuple[torch.Tensor, ...], generated_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        (batch, steps) log-probability of the generated token at each decoding
        step, given per-step ``scores`` of shape (batch, vocab). Only these are
        kept instead of the full logits to save memory and disk I/O.
        """
        return torch.cat(
            [
                torch.log_softmax(step_scores, dim=-1).gather(
                    -1, generated_tokens[:, step, None]
                )
                for step, step_scores in enumerate(scores)
            ],
            dim=1,
        )

    def strip_prompt(
        self, output_ids: torch.Tensor, input_ids: torch.Tensor
    ) -> list[torch.Tensor]:
        """Per-sample generated tokens, dropping the echoed prompt for causal models."""
        input_length = 0 if self.is_seq2seq else input_ids.shape[1]
        return [ids[input_length:] for ids in output_ids]

    def training_step(self, batch, batch_idx):
        # Not implemented for this task
        pass

    def validation_step(self, batch, batch_idx):
        """
        Validation step: generate output IDs only.
        Decoding and extraction are deferred to process_outputs.
        """
        # Batch has pre-stacked tensors and questions from collate_fn
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        questions = batch["question"]
        gold_answers = get_gold_answers(batch)
        choices = batch.get("choices", [None for _ in range(len(questions))])

        with torch.no_grad():
            generation_output = self.forward(input_ids, attention_mask)

        if self.return_logits:
            output_sequences = generation_output.sequences
            output_scores = generation_output.scores  # Tuple of (batch, vocab) per step
        else:
            output_sequences = generation_output
            output_scores = None

        generated = self.strip_prompt(output_sequences, input_ids)
        log_probs = None
        if self.return_logits and output_scores:
            log_probs = self._generated_token_log_probs(
                output_scores, torch.stack(generated)
            )

        for i, (question, gold_answer_list, choice) in enumerate(
            zip(questions, gold_answers, choices)
        ):
            out = {
                "output_ids": generated[i],
                "question": question,
                "gold_answers": gold_answer_list,
            }
            if choice is not None:
                out["choices"] = choice
            if log_probs is not None:
                # clone so flushing does not serialize the whole batch storage
                out["logits"] = log_probs[i].clone()  # [generated_seq_len]
            self.outputs.append(out)

        self._maybe_flush()
        return {"batch_size": len(questions)}

    def process_outputs(self, outputs: list[dict]):
        """
        Decode outputs, extract answers/confidence, and save to CSV.
        If the datamodule does not require semantic equivalence checking,
        compute correctness by exact match and calibration metrics directly.
        """
        for out in outputs:
            out["raw_output"] = self.tokenizer.decode(
                out["output_ids"], skip_special_tokens=True
            )
            out["pred_answer"], out["confidence"] = self.extractor(
                out["raw_output"], out.get("logits"), out["output_ids"]
            )

        write_llm_outputs_csv(
            outputs,
            os.path.join(self.log_dir, "llm_outputs.csv"),
            self.raw_output_max_length,
        )

        datamodule = self.trainer.datamodule
        if not getattr(datamodule, "requires_semantic_equivalence", False):
            for out in outputs:
                out["correct"] = letter_match(out["pred_answer"], out["gold_answers"])
            log_answer_metrics(self, outputs)
