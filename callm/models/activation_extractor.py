from __future__ import annotations

import os
import re
import csv
import json
import shutil
import torch
from typing import TYPE_CHECKING
from callm.models.base import BaseLightningModule
from callm.utils import find_chunk_files

if TYPE_CHECKING:
    from callm.models.llm import LLM


class ActivationExtractor(BaseLightningModule):
    """
    LightningModule that wraps a frozen LLM to run batch validation,
    extract activations from configured layers (stripping padding tokens),
    and save them as sharded .pt files alongside the llm_outputs.csv.
    """

    flush_prefix = "activations_chunk"

    def __init__(
        self,
        llm: LLM,
        output_dir: str,
        layer_indices: list[int] | None = None,
        save_dtype: str = "bfloat16",
        flush_outputs_every_n_steps: int = 100,
        resume_from: str | None = None,
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
        self.resume_from = resume_from
        self.initial_flushed_files = []
        self.sample_counter = 0

        # Ensure LLM is in evaluation mode
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False

    def on_validation_start(self):
        """Prepare activation extraction by loading previous chunks if resuming."""
        if self.resume_from:
            found_files = find_chunk_files(self.resume_from, prefix="activations_chunk")
            valid_files = []
            resumed_samples = 0
            for fpath in found_files:
                try:
                    try:
                        chunk = torch.load(
                            fpath, map_location="cpu", weights_only=False, mmap=True
                        )
                    except TypeError:
                        chunk = torch.load(
                            fpath, map_location="cpu", weights_only=False
                        )
                    if isinstance(chunk, list) and len(chunk) > 0:
                        valid_files.append(fpath)
                        resumed_samples += len(chunk)
                    del chunk
                except Exception as e:
                    print(f"Skipping corrupted or unreadable chunk {fpath}: {e}")

            if valid_files:
                print(
                    f"Found {len(valid_files)} valid temp chunk files ({resumed_samples} samples) to resume from."
                )
                self.initial_flushed_files = valid_files
                self.sample_counter = resumed_samples

        self._adopt_resume_files(
            self.initial_flushed_files,
            log_dir=self.trainer.log_dir or self.output_dir,
            link=True,
        )

    def _extract_gold_answers(self, batch):
        """Extract gold answers from batch. Override for dataset-specific logic."""
        if "gold_answers" in batch:
            return batch["gold_answers"]
        elif "label" in batch:
            return batch["label"]
        return [None] * len(batch["question"])

    def _serialize_gold_answers(self, gold_answers) -> str:
        """Serialize gold answers for CSV output.

        Handles both list (TriviaQA) and string (MMLU) formats.
        """
        if gold_answers is None:
            return ""
        if isinstance(gold_answers, list):
            return "|".join(str(a) for a in gold_answers)
        return str(gold_answers)

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        questions = batch["question"]
        gold_answers = self._extract_gold_answers(batch)

        # 1. Run LLM with generate=True to get generated_ids
        with torch.no_grad():
            generated_output = self.llm(input_ids, attention_mask, generate=True)

        # When return_logits=True, model.generate() returns a GenerateOutput
        # dict-like object with a .sequences attribute instead of a plain tensor.
        if hasattr(generated_output, "sequences"):
            generated_ids = generated_output.sequences
        else:
            generated_ids = generated_output

        batch_size = input_ids.shape[0]
        input_length = input_ids.shape[1] if not self.llm.is_seq2seq else 0

        # 2. Run forward pass on full sequence (prompt + answer) to extract activations
        full_ids, full_attention_mask = self.llm.full_sequence(
            input_ids, attention_mask, generated_ids
        )
        with torch.no_grad():
            hidden_states = self.llm(full_ids, full_attention_mask, generate=False)

        if not isinstance(hidden_states, (list, tuple)):
            hidden_states = (hidden_states,)

        for i in range(batch_size):
            # Decode generated output
            gen_tokens = generated_ids[i]
            if not self.llm.is_seq2seq:
                gen_tokens = gen_tokens[input_length:]

            raw_output = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            pred_answer, _ = self.llm.extractor(raw_output, None, gen_tokens)

            # Extract specified layers
            sample_activations = {}
            # valid_mask covers the full sequence (prompt + answer), minus left-padding
            valid_mask = full_attention_mask[i] == 1
            prompt_length = int((attention_mask[i] == 1).sum().item())
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

            if "sample_idx" in batch:
                s_idx = batch["sample_idx"][i]
                sample_idx = s_idx.item() if hasattr(s_idx, "item") else int(s_idx)
            else:
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
                    "prompt_length": prompt_length,
                }
            )

        # Periodically flush outputs to disk
        self._maybe_flush()

        return {"batch_size": batch_size}

    def on_validation_epoch_end(self):
        log_dir = self.trainer.log_dir or self.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        llm_outputs_file = os.path.join(log_dir, "llm_outputs.csv")
        shard_size = 1000

        # Case 1: Chunks were periodically flushed to disk
        if self.flushed_output_files:
            self._flush_outputs()

            chunk_files = list(self.flushed_output_files)
            if not chunk_files:
                return

            def get_chunk_num(p: str) -> int:
                match = re.search(
                    r"activations_chunk_rank\d+_(\d+)\.pt", os.path.basename(p)
                )
                if match:
                    return int(match.group(1))
                match = re.search(r"(\d+)\.pt$", os.path.basename(p))
                return int(match.group(1)) if match else 0

            chunk_files = sorted(chunk_files, key=get_chunk_num)

            total_samples = 0
            first_sample = None

            # Stream write llm_outputs.csv. Errors propagate so that chunks are
            # kept on disk and the run can be resumed.
            with open(llm_outputs_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["question", "gold_answers", "pred_answer", "raw_output"]
                )
                for c_file in chunk_files:
                    chunk = torch.load(c_file, map_location="cpu", weights_only=False)
                    if first_sample is None and chunk:
                        first_sample = chunk[0]
                    for out in chunk:
                        # Labels are later joined to activations by CSV row
                        # position, so row order must match sample_idx.
                        if out["sample_idx"] != total_samples:
                            raise ValueError(
                                f"Chunk {c_file} has sample_idx {out['sample_idx']} "
                                f"at row {total_samples}; llm_outputs.csv rows would "
                                "be misaligned with activations. Chunks were kept "
                                "on disk."
                            )
                        total_samples += 1
                        gold_str = self._serialize_gold_answers(out["gold_answers"])
                        writer.writerow(
                            [
                                out["question"],
                                gold_str,
                                out["pred_answer"],
                                out["raw_output"].replace("\n", "\\n"),
                            ]
                        )
                    del chunk
            print(f"\nLLM outputs saved to {llm_outputs_file}")
            # Also ensure llm_outputs.csv is available in output_dir
            output_csv = os.path.join(self.output_dir, "llm_outputs.csv")
            if os.path.abspath(llm_outputs_file) != os.path.abspath(output_csv):
                try:
                    shutil.copy(llm_outputs_file, output_csv)
                except Exception:
                    pass

            if total_samples == 0 or first_sample is None:
                self.flushed_output_files = []
                self.outputs = []
                self.sample_counter = 0
                return

            first_layer_idx = list(first_sample["activations"].keys())[0]
            hidden_size = first_sample["activations"][first_layer_idx].shape[-1]

            # Stream consolidate activations into shards. Chunks are only deleted
            # once all shards and metadata.json are written.
            shard_buffer = []
            current_shard_idx = 0

            for c_file in chunk_files:
                chunk = torch.load(c_file, map_location="cpu", weights_only=False)

                for sample in chunk:
                    shard_buffer.append(
                        {
                            "activations": sample["activations"],
                            "seq_len": sample["seq_len"],
                            "prompt_length": sample["prompt_length"],
                            "sample_idx": sample["sample_idx"],
                        }
                    )

                    if len(shard_buffer) >= shard_size:
                        shard_file = os.path.join(
                            self.output_dir, f"activations_{current_shard_idx:04d}.pt"
                        )
                        torch.save(shard_buffer, shard_file)
                        print(f"Saved activation shard to {shard_file}")
                        current_shard_idx += 1
                        shard_buffer = []

                del chunk

            if shard_buffer:
                shard_file = os.path.join(
                    self.output_dir, f"activations_{current_shard_idx:04d}.pt"
                )
                torch.save(shard_buffer, shard_file)
                print(f"Saved activation shard to {shard_file}")
                shard_buffer = []

            num_samples = total_samples
            chunks_to_delete = chunk_files

        # Case 2: No chunks flushed (all in self.outputs in-memory)
        else:
            if not self.outputs:
                return

            num_samples = len(self.outputs)
            chunks_to_delete = []

            try:
                with open(llm_outputs_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        ["question", "gold_answers", "pred_answer", "raw_output"]
                    )
                    for out in self.outputs:
                        gold_str = self._serialize_gold_answers(out["gold_answers"])
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

            for shard_idx, i in enumerate(range(0, num_samples, shard_size)):
                shard_samples = self.outputs[i : i + shard_size]
                shard_data = []
                for sample in shard_samples:
                    shard_data.append(
                        {
                            "activations": sample["activations"],
                            "seq_len": sample["seq_len"],
                            "prompt_length": sample["prompt_length"],
                            "sample_idx": sample["sample_idx"],
                        }
                    )
                shard_file = os.path.join(
                    self.output_dir, f"activations_{shard_idx:04d}.pt"
                )
                torch.save(shard_data, shard_file)
                print(f"Saved activation shard to {shard_file}")

            first_sample = self.outputs[0]
            first_layer_idx = list(first_sample["activations"].keys())[0]
            hidden_size = first_sample["activations"][first_layer_idx].shape[-1]

        # Save metadata.json
        metadata = {
            "model_name": self.llm.model_name,
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
            print(f"Failed to save metadata: {e}. Keeping activation chunks.")
            chunks_to_delete = []

        for c_file in chunks_to_delete:
            if os.path.exists(c_file):
                os.remove(c_file)

        # Clear memory and reset counter
        self.outputs = []
        self.flushed_output_files = []
        self.sample_counter = 0
