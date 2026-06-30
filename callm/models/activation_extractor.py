from __future__ import annotations

import os
import csv
import json
import torch
from typing import TYPE_CHECKING
from callm.models.base import BaseLightningModule

if TYPE_CHECKING:
    from callm.models.llm import LLM


class ActivationExtractor(BaseLightningModule):
    """
    LightningModule that wraps a frozen LLM to run batch validation,
    extract activations from configured layers (stripping padding tokens),
    and save them as sharded .pt files alongside the llm_outputs.csv.
    """

    def __init__(
        self,
        llm: LLM,
        output_dir: str,
        layer_indices: list[int] | None = None,
        save_dtype: str = "bfloat16",
        flush_outputs_every_n_steps: int = 100,
    ):
        super().__init__(
            flush_outputs_every_n_steps=flush_outputs_every_n_steps,
            save_outputs=True,
        )
        self.llm = llm
        self.output_dir = output_dir
        self.layer_indices = layer_indices
        self.save_dtype = save_dtype
        self.tokenizer = llm.tokenizer
        self.sample_counter = 0

        # Ensure LLM is in evaluation mode
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
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

        # 1. Run LLM with generate=True to get generated_ids
        with torch.no_grad():
            generated_ids = self.llm(input_ids, attention_mask, generate=True)

        # 2. Run LLM with generate=False to extract activations
        with torch.no_grad():
            hidden_states = self.llm(input_ids, attention_mask, generate=False)

        if not isinstance(hidden_states, (list, tuple)):
            hidden_states = (hidden_states,)

        batch_size = input_ids.shape[0]
        input_length = input_ids.shape[1] if not self.llm.is_seq2seq else 0

        for i in range(batch_size):
            # Decode generated output
            gen_tokens = generated_ids[i]
            if not self.llm.is_seq2seq:
                gen_tokens = gen_tokens[input_length:]

            raw_output = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            pred_answer, _ = self.llm.extractor(raw_output, None, gen_tokens)

            # Extract specified layers
            sample_activations = {}
            valid_mask = attention_mask[i] == 1
            num_layers = len(hidden_states)
            target_layers = self.layer_indices
            if target_layers is None:
                target_layers = list(range(num_layers))

            for layer_idx in target_layers:
                pos_idx = layer_idx % num_layers
                layer_tensor = hidden_states[pos_idx][i]  # (seq_len, hidden_dim)
                valid_tensor = layer_tensor[valid_mask]  # (valid_seq_len, hidden_dim)

                if self.save_dtype == "bfloat16":
                    valid_tensor = valid_tensor.to(torch.bfloat16)
                elif self.save_dtype == "float16":
                    valid_tensor = valid_tensor.to(torch.float16)
                elif self.save_dtype == "float32":
                    valid_tensor = valid_tensor.to(torch.float32)

                sample_activations[pos_idx] = valid_tensor.cpu()

            sample_idx = self.sample_counter
            self.sample_counter += 1

            self.outputs.append(
                {
                    "sample_idx": sample_idx,
                    "question": questions[i],
                    "gold_answers": gold_answers[i],
                    "pred_answer": pred_answer,
                    "raw_output": raw_output,
                    "activations": sample_activations,
                    "seq_len": int(valid_mask.sum().item()),
                }
            )

        # Periodically flush outputs to disk
        if (
            self.flush_outputs_every_n_steps > 0
            and len(self.outputs) >= self.flush_outputs_every_n_steps
        ):
            self._flush_outputs(prefix="activations_chunk")

        return {"batch_size": batch_size}

    def on_validation_epoch_end(self):
        if self.outputs:
            self._flush_outputs(prefix="activations_chunk")

        self._reload_flushed_outputs()

        if not self.outputs:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        num_samples = len(self.outputs)
        shard_size = 1000

        # Save llm_outputs.csv
        llm_outputs_file = os.path.join(self.output_dir, "llm_outputs.csv")
        try:
            with open(llm_outputs_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["question", "gold_answers", "pred_answer", "raw_output"]
                )
                for out in self.outputs:
                    gold_str = (
                        "|".join(out["gold_answers"]) if out["gold_answers"] else ""
                    )
                    writer.writerow(
                        [
                            out["question"],
                            gold_str,
                            out["pred_answer"],
                            out["raw_output"].replace("\n", "\\n"),
                        ]
                    )
            print(f"\nLLM outputs saved to {llm_outputs_file}")
        except Exception as e:
            print(f"Failed to save LLM outputs: {e}")

        # Save sharded activations
        for shard_idx, i in enumerate(range(0, num_samples, shard_size)):
            shard_samples = self.outputs[i : i + shard_size]
            shard_data = []
            for sample in shard_samples:
                shard_data.append(
                    {
                        "activations": sample["activations"],
                        "seq_len": sample["seq_len"],
                        "sample_idx": sample["sample_idx"],
                    }
                )
            shard_file = os.path.join(
                self.output_dir, f"activations_{shard_idx:04d}.pt"
            )
            torch.save(shard_data, shard_file)
            print(f"Saved activation shard to {shard_file}")

        # Save metadata.json
        first_sample = self.outputs[0]
        first_layer_idx = list(first_sample["activations"].keys())[0]
        hidden_size = first_sample["activations"][first_layer_idx].shape[-1]
        model_name = self.llm.model_name

        metadata = {
            "model_name": model_name,
            "layer_indices": self.layer_indices
            if self.layer_indices is not None
            else "all",
            "hidden_size": hidden_size,
            "num_samples": num_samples,
            "dtype": self.save_dtype,
        }

        metadata_file = os.path.join(self.output_dir, "metadata.json")
        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            print(f"Saved metadata to {metadata_file}")
        except Exception as e:
            print(f"Failed to save metadata: {e}")

        # Clear memory and reset counter
        self.outputs = []
        self.sample_counter = 0
