import torch
import os
import csv
from callm.extractors import BaseExtractor
from callm.utils import initialize_model, get_tokenizer_for_model
from callm.models.base import BaseLightningModule


class LLM(BaseLightningModule):
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

        if not train:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.extractor: BaseExtractor = extractor
        self.return_logits = return_logits
        self.max_new_tokens = max_new_tokens

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
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "output_scores": self.return_logits,
            "return_dict_in_generate": self.return_logits,
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
        Decoding and extraction are deferred to on_validation_epoch_end.
        """
        # Batch has pre-stacked tensors and questions from collate_fn
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        questions = batch["question"]
        gold_answers = (
            batch["label"]
            if "label" in batch
            else [None for _ in range(len(questions))]
        )

        with torch.no_grad():
            generation_output = self.forward(input_ids, attention_mask)

        if self.return_logits:
            output_sequences = generation_output.sequences
            output_scores = generation_output.scores  # Tuple of (batch, vocab) per step
        else:
            output_sequences = generation_output
            output_scores = None

        input_length = input_ids.shape[1] if not self.is_seq2seq else 0

        for i, (question, gold_answer_list) in enumerate(zip(questions, gold_answers)):
            if self.is_seq2seq:
                generated_tokens = output_sequences[i]
            else:
                generated_tokens = output_sequences[i][input_length:]

            out = {
                "output_ids": generated_tokens,
                "question": question,
                "gold_answers": gold_answer_list,
            }
            if self.return_logits and output_scores is not None:
                # OPTIMIZATION: Store only the log probability of the generated token
                # instead of the full logits tensor (size [seq_len, vocab_size]).
                # This drastically reduces memory usage and disk I/O.

                # generated_tokens has shape [seq_len] (single sequence)
                # We need to verify alignment.
                # output_scores has length = number of generated tokens.

                scores_list = []

                for step_idx, step_score_tensor in enumerate(output_scores):
                    # step_score_tensor: [batch_size, vocab_size]
                    # Get score for this sample (index i)
                    token_logits = step_score_tensor[i]  # [vocab_size]

                    log_probs = torch.log_softmax(token_logits, dim=-1)

                    # Store only the log prob of the chosen token
                    # equivalent to max log prob since we use greedy decoding
                    token_id = generated_tokens[step_idx]
                    scores_list.append(log_probs[token_id])

                if scores_list:
                    out["logits"] = torch.stack(scores_list)  # [generated_seq_len]

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
        Evaluation is handled separately by EvaluatorModule.
        """
        # Flush any remaining outputs
        if self.outputs and (
            self.flushed_output_files or self.flush_outputs_every_n_steps > 0
        ):
            self._flush_outputs(prefix="temp_val_outputs")

        self._reload_flushed_outputs()

        if len(self.outputs) == 0:
            return

        # Decode all output IDs and extract answers/confidence
        for out in self.outputs:
            raw_output = self.tokenizer.decode(
                out["output_ids"], skip_special_tokens=True
            )
            out["raw_output"] = raw_output

            # Extract answer and confidence
            pred_answer, confidence = self.extractor(
                raw_output, out.get("logits"), out["output_ids"]
            )
            out["pred_answer"] = pred_answer
            out["confidence"] = confidence

        # Save outputs to CSV for evaluator
        log_dir = self.trainer.log_dir or os.getcwd()
        outputs_file = os.path.join(log_dir, "llm_outputs.csv")

        def short_output(txt: str, limit: int = 200) -> str:
            """Keep only first line, truncate if too long."""
            if not txt:
                return ""
            truncated_text = txt[:limit] + ("..." if len(txt) > limit else "")
            return truncated_text.replace("\n", "\\n")

        try:
            with open(outputs_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "question",
                        "gold_answers",
                        "pred_answer",
                        "confidence",
                        "raw_output",
                    ]
                )

                for out in self.outputs:
                    # Format gold_answers as a string representation
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
                    writer.writerow(
                        [
                            out["question"],
                            gold_str,
                            out["pred_answer"],
                            conf_str,
                            short_output(out["raw_output"]),
                        ]
                    )
            print(f"\nLLM outputs saved to {outputs_file}")
        except Exception as e:
            print(f"Failed to save LLM outputs: {e}")

        # Clear outputs for next epoch
        self.outputs = []
