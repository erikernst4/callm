from lightning.pytorch import LightningModule
import torch
import os
import csv
from callm.extractors import VerbalizedConfidenceExtractor
from callm.utils import initialize_model, get_tokenizer_for_model


class LLM(LightningModule):
    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
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

        # Initialize extractor
        self.extractor = VerbalizedConfidenceExtractor()

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
        Decoding and extraction are deferred to on_validation_epoch_end.
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
        Decode outputs, extract answers/confidence, and save to CSV.
        Evaluation is handled separately by EvaluatorModule.
        """
        if len(self.validation_outputs) == 0:
            return

        # Decode all output IDs and extract answers/confidence
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

                for out in self.validation_outputs:
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
        self.validation_outputs = []

    def configure_optimizers(self):
        # Not training, return None
        return None
