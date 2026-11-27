from __future__ import annotations

import torch
import os
from callm.extractors import BaseExtractor
from callm.utils import (
    generation_pad_token_id,
    get_tokenizer_for_model,
    initialize_model,
)
from callm.models.base import BaseLightningModule
from callm.models.outputs import (
    get_gold_answers,
    letter_match,
    log_answer_metrics,
    write_llm_outputs_csv,
)


def _eos_token_ids(model, tokenizer) -> list[int]:
    """
    Token ids that end a generated sequence: the model's generation config
    (what ``generate()`` stops on), else the tokenizer's EOS; empty if neither
    is available.
    """
    sources = (getattr(model, "generation_config", None), tokenizer)
    for eos in (getattr(source, "eos_token_id", None) for source in sources):
        if isinstance(eos, int):
            return [eos]
        if (
            isinstance(eos, (list, tuple))
            and eos
            and all(isinstance(e, int) for e in eos)
        ):
            return list(eos)
    return []


class LLM(BaseLightningModule):
    flush_prefix = "temp_val_outputs"

    def __init__(
        self,
        extractor: BaseExtractor,
        model_name: str = "Qwen/Qwen3.5-9B",
        hf_token: str = None,
        train: bool = False,
        return_logits: bool = False,
        flush_outputs_every_n_steps: int = -1,
        save_outputs: bool = False,
        max_new_tokens: int = 100,
        raw_output_max_length: int | None = None,
        top_logprobs: int = 0,
    ):
        super().__init__(
            flush_outputs_every_n_steps=flush_outputs_every_n_steps,
            save_outputs=save_outputs,
        )

        self.model_name = model_name

        # Load main model
        self.model = initialize_model(model_name, hf_token)
        self.tokenizer = get_tokenizer_for_model(model_name)

        if hasattr(self.model.config, "text_config"):
            self.model.config.pad_token_id = self.model.config.text_config.pad_token_id
        elif self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # Explicit pad_token_id for generate()
        self.pad_token_id = generation_pad_token_id(self.model, self.tokenizer)
        self.eos_token_ids = _eos_token_ids(self.model, self.tokenizer)

        if not train:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.extractor: BaseExtractor = extractor
        self.return_logits = return_logits
        self.max_new_tokens = max_new_tokens
        self.raw_output_max_length = raw_output_max_length
        # >0: also store the top-k {token: logprob} of each generation step
        self.top_logprobs = top_logprobs

    def forward(
        self,
        input_ids,
        attention_mask,
        generate: bool = True,
    ):
        """
        Generate answers (``generate=True``) or return the hidden states of
        all layers for the given tokens (``generate=False``).
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

        return self.model.generate(**generation_kwargs)

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

    def _top_k_log_probs(
        self, scores: tuple[torch.Tensor, ...]
    ) -> list[list[dict[str, float]]]:
        """
        Per sample, the top-``self.top_logprobs`` {token: logprob} of each
        decoding step, given per-step ``scores`` of shape (batch, vocab).
        """
        per_sample = [[] for _ in range(scores[0].shape[0])] if scores else []
        for step_scores in scores:
            values, ids = torch.topk(
                torch.log_softmax(step_scores, dim=-1), self.top_logprobs, dim=-1
            )
            for sample, (sample_values, sample_ids) in enumerate(zip(values, ids)):
                per_sample[sample].append(
                    {
                        self.tokenizer.decode([int(t)]): float(v)
                        for v, t in zip(sample_values, sample_ids)
                    }
                )
        return per_sample

    def strip_prompt(
        self, output_ids: torch.Tensor, input_ids: torch.Tensor
    ) -> list[torch.Tensor]:
        """Per-sample generated tokens, dropping the echoed prompt."""
        return [ids[input_ids.shape[1] :] for ids in output_ids]

    def full_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Token ids and attention mask of [prompt + answer], for the forward pass
        that extracts activations of a generated answer.

        ``generated_ids`` already is [prompt | answer]; the mask keeps the
        prompt's (possibly left-padded) mask and marks each answer's tokens up
        to and including its first EOS. In a batch, ``generate()`` keeps
        filling sequences that finished early, so tokens after the first EOS
        are masked out; this matches batch size 1, where nothing follows the
        EOS.
        """
        answer_ids = generated_ids[:, input_ids.shape[1] :]
        answer_mask = torch.ones_like(answer_ids, dtype=attention_mask.dtype)
        if self.eos_token_ids:
            is_eos = torch.isin(
                answer_ids, torch.tensor(self.eos_token_ids, device=answer_ids.device)
            ).long()
            # Positions after the first EOS (the EOS itself stays)
            answer_mask[(is_eos.cumsum(dim=1) - is_eos) > 0] = 0
        return generated_ids, torch.cat([attention_mask, answer_mask], dim=1)

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
        top_logprobs = None
        if self.return_logits and output_scores:
            log_probs = self._generated_token_log_probs(
                output_scores, torch.stack(generated)
            )
            if self.top_logprobs > 0:
                top_logprobs = self._top_k_log_probs(output_scores)

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
            if top_logprobs is not None:
                out["top_logprobs"] = top_logprobs[i]
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
                out["raw_output"],
                out.get("logits"),
                out["output_ids"],
                top_logprobs=out.get("top_logprobs"),
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
