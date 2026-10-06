from __future__ import annotations

import torch
import torch.nn as nn
import json
import os
import csv
import numpy as np
from lightning.pytorch import LightningModule
from typing import Callable
from callm.extractors import BaseExtractor
from callm.utils import initialize_model, get_tokenizer_for_model, check_exact_match
from callm.models.base import BaseLightningModule
from callm.models.probe import Probe, SplitProbe
from callm.models.layer_pooling import (
    LayerWeightedAverage,
    SingleLayerSelector,
    GlobalZNormWeightedAverage,
    LayerwiseZNormWeightedAverage,
)
from ecuas import (
    ExpectedCalibrationError,
    ConfidenceBrierScore,
    ConfidenceCrossEntropy,
    ConfidenceAUCScore,
    ConfidenceECUAS,
    CCAS,
)


def _get_gold_answers(batch: dict) -> list:
    """Gold answers from ``gold_answers`` or ``label``, or ``None`` per question."""
    if "gold_answers" in batch:
        return batch["gold_answers"]
    if "label" in batch:
        return batch["label"]
    return [None for _ in range(len(batch["question"]))]


def _write_csv(path: str, header: list, rows: list[list], description: str):
    """Write ``header`` + ``rows`` to ``path``, reporting success or failure."""
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"{description} saved to {path}")
    except Exception as e:
        print(f"Failed to save {description}: {e}")


def _log_calibration_metrics(
    module: LightningModule, confidences: np.ndarray, correctness: np.ndarray
) -> dict[str, float]:
    """
    Log accuracy and calibration metrics for ``module`` and save them to
    ``metrics.csv`` in the trainer's log directory.
    """
    # Log accuracy on all samples
    accuracy = float(correctness.mean()) if len(correctness) > 0 else 0.0
    module.log("val_accuracy", accuracy, prog_bar=True, sync_dist=True)

    metrics_results = {"val_accuracy": accuracy}

    confidences_t = torch.tensor(confidences, dtype=torch.float32)
    correctness_t = torch.tensor(correctness, dtype=torch.float32)

    # NaN confidences are handled internally by the metrics (fallback to 0.5)
    metrics = {
        "val_ece": ExpectedCalibrationError(n_bins=10),
        "val_brier_score": ConfidenceBrierScore(),
        "val_cross_entropy": ConfidenceCrossEntropy(),
        "val_auc": ConfidenceAUCScore(),
        "val_ccas": CCAS(),
    }

    for name, metric in metrics.items():
        metric.update(confidences_t, correctness_t)
        value = float(metric.compute())
        module.log(name, value, prog_bar=True, sync_dist=True)
        metrics_results[name] = value

    log_dir = module.trainer.log_dir or os.getcwd()
    _write_csv(
        os.path.join(log_dir, "metrics.csv"),
        list(metrics_results.keys()),
        [list(metrics_results.values())],
        "Metrics",
    )
    return metrics_results


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
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            return outputs.hidden_states

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

        generated_ids = self.model.generate(**generation_kwargs)

        if return_hidden_states:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            return generated_ids, outputs.hidden_states

        return generated_ids

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
        Decoding and extraction are deferred to on_validation_epoch_end.
        """
        # Batch has pre-stacked tensors and questions from collate_fn
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        questions = batch["question"]
        gold_answers = _get_gold_answers(batch)
        choices = batch.get("choices", [None for _ in range(len(questions))])

        with torch.no_grad():
            generation_output = self.forward(input_ids, attention_mask)

        if self.return_logits:
            output_sequences = generation_output.sequences
            output_scores = generation_output.scores  # Tuple of (batch, vocab) per step
        else:
            output_sequences = generation_output
            output_scores = None

        for i, (question, gold_answer_list, choice, generated_tokens) in enumerate(
            zip(
                questions,
                gold_answers,
                choices,
                self.strip_prompt(output_sequences, input_ids),
            )
        ):
            out = {
                "output_ids": generated_tokens,
                "question": question,
                "gold_answers": gold_answer_list,
            }
            if choice is not None:
                out["choices"] = choice
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
        If the datamodule does not require semantic equivalence checking,
        compute correctness and calibration metrics directly.
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

        def short_output(
            txt: str, limit: int | None = self.raw_output_max_length
        ) -> str:
            """Escape newlines and optionally truncate long outputs."""
            if not txt:
                return ""
            if limit is not None:
                txt = txt[:limit] + ("..." if len(txt) > limit else "")
            return txt.replace("\n", "\\n")

        has_choices = any("choices" in out for out in self.outputs)
        headers = ["question", "gold_answers"]
        if has_choices:
            headers.append("choices")
        headers.extend(["pred_answer", "confidence", "raw_output"])

        rows = []
        for out in self.outputs:
            # Format gold_answers as a string representation
            gold_str = "|".join(out["gold_answers"]) if out["gold_answers"] else ""
            conf_str = (
                f"{out['confidence']:.6f}"
                if out["confidence"] is not None
                and not (
                    isinstance(out["confidence"], float)
                    and out["confidence"] != out["confidence"]
                )
                else "nan"
            )

            row = [out["question"], gold_str]
            if has_choices:
                row.append(out.get("choices", ""))
            row.extend(
                [
                    out["pred_answer"],
                    conf_str,
                    short_output(out["raw_output"]),
                ]
            )
            rows.append(row)
        _write_csv(outputs_file, headers, rows, "LLM outputs")

        # If the datamodule does not require semantic equivalence,
        # compute correctness and metrics directly
        datamodule = self.trainer.datamodule
        if not getattr(datamodule, "requires_semantic_equivalence", False):
            self._calculate_metrics_direct()

        # Clear outputs for next epoch
        self.outputs = []

    def _calculate_metrics_direct(self):
        """
        Compute correctness by exact string match and calculate
        calibration metrics. Used when semantic equivalence checking
        is not required (e.g. MMLU multiple-choice).
        """
        # Determine correctness by exact match
        for out in self.outputs:
            pred = (out["pred_answer"] or "").strip().upper()
            gold = out["gold_answers"]
            if isinstance(gold, list):
                out["correct"] = any(pred == g.strip().upper() for g in gold)
            else:
                out["correct"] = pred == (gold or "").strip().upper()

        all_confidences = []
        all_correctness = []
        for out in self.outputs:
            try:
                conf = float(out["confidence"])
            except (ValueError, TypeError):
                conf = float("nan")
            all_confidences.append(conf)
            all_correctness.append(out["correct"])

        all_confidences = np.array(all_confidences)
        all_correctness = np.array(all_correctness)

        n_invalid = np.isnan(all_confidences).sum()
        if n_invalid > 0:
            print(
                f"Warning: {n_invalid} samples have invalid confidence (NaN). "
                "Using fallback confidence of 0.5 for these samples."
            )

        _log_calibration_metrics(self, all_confidences, all_correctness)


class ConfidenceAugmentedLLM(LightningModule):
    """
    LightningModule that wraps a Probe and (optionally) a LLM module.

    Three modes of operation:
    - **Live mode** (``llm`` provided): Runs the LLM forward pass to extract
      activations on the fly and feeds them to the probe.
    - **Cached mode** (``llm=None``): Expects pre-computed activations in the
      batch (key ``"activations"``), skipping the LLM entirely. Requires
      ``model_dim`` to be specified explicitly.
    - **Live evaluation mode** (``llm`` provided + ``probe_checkpoint``): Loads
      a trained probe from a checkpoint and runs end-to-end evaluation:
      generate response → extract activations → probe → confidence.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model_dim: int | None = None,
        probe: nn.Module | None = None,
        layer_idx: int | list[int] | str = -1,
        num_layers: int | None = None,
        layer_pooling: nn.Module | None = None,
        seq_pooling: nn.Module | None = None,
        lr: float = 1e-3,
        lr_scheduler: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ]
        | None = None,
        probe_checkpoint: str | None = None,
        pool_strategy: str = "mean",
        activations_to_use: str = "answer",
    ):
        super().__init__()
        self.llm = llm
        self.layer_idx = layer_idx
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.pool_strategy = pool_strategy
        self.activations_to_use = activations_to_use

        # Determine model_dim: explicit arg > inferred from LLM config
        if model_dim is not None:
            self.model_dim = model_dim
        elif self.llm is not None:
            if hasattr(self.llm.model.config, "hidden_size"):
                self.model_dim = self.llm.model.config.hidden_size
            elif hasattr(self.llm.model.config, "d_model"):
                self.model_dim = self.llm.model.config.d_model
            else:
                raise ValueError(
                    "Could not determine model dimension from LLM config. "
                    "Pass model_dim explicitly."
                )
        else:
            raise ValueError("Either llm or model_dim must be provided.")

        self.num_layers = num_layers
        if layer_pooling is None:
            if isinstance(layer_idx, list):
                self.num_layers = len(layer_idx)
                layer_pooling = LayerWeightedAverage(num_layers=self.num_layers)
            elif isinstance(layer_idx, str) and layer_idx.lower() == "all":
                if self.num_layers is None and self.llm is not None:
                    if hasattr(self.llm.model.config, "num_hidden_layers"):
                        self.num_layers = self.llm.model.config.num_hidden_layers
                    elif hasattr(self.llm.model.config, "n_layer"):
                        self.num_layers = self.llm.model.config.n_layer
                if self.num_layers is None:
                    raise ValueError(
                        "When layer_idx is 'all' and running in cached mode (no llm provided), "
                        "you must specify 'num_layers' explicitly."
                    )
                layer_pooling = LayerWeightedAverage(num_layers=self.num_layers)
        else:
            if (
                hasattr(layer_pooling, "num_layers")
                and layer_pooling.num_layers is not None
            ):
                self.num_layers = layer_pooling.num_layers

        if probe is None:
            self.probe = Probe(
                self.model_dim,
                layer_pooling=layer_pooling,
                seq_pooling=seq_pooling,
            )
        else:
            self.probe = probe

        # Load probe weights from a trained checkpoint if provided
        if probe_checkpoint is not None:
            self._load_probe_checkpoint(probe_checkpoint)

        # BCELoss because self.probe outputs confidence probabilities in [0, 1]
        self.loss_fn = nn.BCELoss()

        self.val_confidences = []
        self.val_correctness = []
        # Storage for live evaluation outputs
        self._eval_outputs = []

        # Layer weight snapshot tracking for visualization
        self._weight_snapshots = []
        self._weight_snapshot_interval = 50  # log every N training steps

    def _load_probe_checkpoint(self, probe_checkpoint: str):
        """
        Load probe weights from a trained ConfidenceAugmentedLLM checkpoint.

        Extracts only the ``probe.*`` keys from the checkpoint state_dict
        and loads them into ``self.probe``. The probe is then frozen for
        evaluation.

        This is the scalable pattern: any checkpoint trained with
        ConfidenceAugmentedLLM can be loaded by specifying
        ``probe_checkpoint`` together with matching hyperparameters
        (``layer_idx``, ``pool_strategy``, ``activations_to_use``,
        ``model_dim``).
        """
        ckpt = torch.load(probe_checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # Extract only probe.* keys and strip the prefix
        probe_state = {
            k.replace("probe.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("probe.")
        }
        if not probe_state:
            raise ValueError(
                f"No 'probe.*' keys found in checkpoint {probe_checkpoint}. "
                f"Available keys: {list(state_dict.keys())[:20]}"
            )

        self.probe.load_state_dict(probe_state)
        self.probe.eval()
        for p in self.probe.parameters():
            p.requires_grad = False
        print(f"Loaded probe weights from {probe_checkpoint}")

    def on_fit_start(self):
        """
        Inject normalization statistics from the datamodule into the
        layer_pooling module, if applicable.

        Called automatically by Lightning before training begins.
        """
        layer_pooling = getattr(self.probe, "layer_pooling", None)
        if layer_pooling is None or not hasattr(
            layer_pooling, "set_normalization_stats"
        ):
            return

        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return

        layer_stats = getattr(datamodule, "layer_stats", None)
        if layer_stats is None:
            print(
                "Warning: layer_pooling has set_normalization_stats() but "
                "datamodule has no layer_stats. Normalization will use default "
                "(zero-mean, unit-std) values."
            )
            return

        if isinstance(layer_pooling, GlobalZNormWeightedAverage):
            layer_pooling.set_normalization_stats(
                layer_stats["global_mean"].to(self.device),
                layer_stats["global_std"].to(self.device),
            )
            print("Injected global Z-normalization statistics into layer_pooling")
        elif isinstance(layer_pooling, LayerwiseZNormWeightedAverage):
            layer_pooling.set_normalization_stats(
                layer_stats["layer_means"].to(self.device),
                layer_stats["layer_stds"].to(self.device),
            )
            print("Injected layer-wise Z-normalization statistics into layer_pooling")

    def on_save_checkpoint(self, checkpoint: dict):
        """
        Exclude frozen LLM weights from probe checkpoints to save disk space and I/O.
        Only probe parameters are trained and need to be saved (~tens of KB instead of 18+ GB).
        """
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith("llm.")
        }

    def _select_layer_activations(self, hidden_states):
        """
        Select the appropriate layer(s) from hidden_states based on
        ``self.layer_idx`` and the probe's layer_pooling module.

        Returns:
            activations tensor ready for per-sample slicing/pooling.
        """
        num_all_layers = len(hidden_states)
        layer_pooling = getattr(self.probe, "layer_pooling", None)

        if isinstance(layer_pooling, SingleLayerSelector):
            return torch.stack(hidden_states, dim=1)
        elif layer_pooling is not None:
            if isinstance(self.layer_idx, list):
                selected = [
                    hidden_states[layer_i % num_all_layers]
                    for layer_i in self.layer_idx
                ]
                return torch.stack(selected, dim=1)
            else:
                return torch.stack(hidden_states, dim=1)
        else:
            if isinstance(self.layer_idx, int):
                return hidden_states[self.layer_idx % num_all_layers]
            else:
                return torch.stack(hidden_states, dim=1)

    def _slice_samples(
        self,
        activations: torch.Tensor,
        prompt_lengths: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Strip padding from each sample and slice it to the region selected by
        ``self.activations_to_use`` (answer/prompt/both).

        Args:
            activations: (batch, seq_len, dim) tensor, after layer pooling
            prompt_lengths: (batch,) tensor of prompt token counts
            attention_mask: (batch, seq_len) attention mask for the full sequence

        Returns:
            list of (sample_len, dim) tensors; empty slices are replaced by a
            single zero vector.
        """
        sliced_list = []
        for i in range(activations.shape[0]):
            valid_mask = attention_mask[i] == 1
            sample_act = activations[i][valid_mask]  # strip padding
            prompt_len = int(prompt_lengths[i].item())

            if self.activations_to_use == "answer":
                sample_act = sample_act[prompt_len:]
            elif self.activations_to_use == "prompt":
                sample_act = sample_act[:prompt_len]
            # "both" keeps all valid tokens

            if sample_act.shape[0] == 0:
                sample_act = torch.zeros(
                    1,
                    sample_act.shape[-1],
                    device=sample_act.device,
                    dtype=sample_act.dtype,
                )
            sliced_list.append(sample_act)
        return sliced_list

    def _extract_and_pool_activations(
        self,
        hidden_states,
        prompt_lengths: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract, slice, and pool activations for probe evaluation.

        Handles per-sample slicing (answer/prompt/both) based on
        ``self.activations_to_use`` and pooling (mean/last) based on
        ``self.pool_strategy``.

        Args:
            hidden_states: tuple of layer tensors from the LLM forward pass
            prompt_lengths: (batch,) tensor of prompt token counts
            attention_mask: (batch, seq_len) attention mask for the full sequence

        Returns:
            pooled activations tensor of shape (batch, model_dim)
        """
        activations = self.probe.pool_layers(
            self._select_layer_activations(hidden_states)
        )

        # Now activations is (batch, seq, dim) or (batch, dim)
        if activations.ndim == 2:
            # Already pooled (no sequence dimension), return directly
            return activations

        sliced_list = self._slice_samples(activations, prompt_lengths, attention_mask)
        if self.pool_strategy == "mean":
            pooled_list = [sample_act.mean(dim=0) for sample_act in sliced_list]
        elif self.pool_strategy == "last":
            pooled_list = [sample_act[-1] for sample_act in sliced_list]
        else:
            raise ValueError(f"Unknown pool_strategy: {self.pool_strategy}")

        return torch.stack(pooled_list)  # (batch, dim)

    def _run_llm_hidden_states(self, input_ids, attention_mask):
        """Non-generative LLM forward pass returning a tuple of hidden states."""
        with torch.no_grad():
            hidden_states = self.llm(
                input_ids, attention_mask, generate=False, return_hidden_states=True
            )
        return self._as_tuple(hidden_states)

    @staticmethod
    def _as_tuple(hidden_states):
        if not isinstance(hidden_states, (list, tuple)):
            return (hidden_states,)
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generate: bool = True,
    ):
        """
        Run forward pass of LLM to generate response, extract activations,
        and run Probe to get confidence.

        Requires ``self.llm`` to be set (live mode).

        When ``generate=True``, the method:
        1. Generates the response via ``self.llm.generate()``
        2. Re-runs a forward pass on ``[prompt + generated answer]``
           to extract hidden states for the full sequence
        3. Slices activations to the configured region
           (``self.activations_to_use``)
        4. Pools and feeds through the probe

        Returns:
            If generate is True: (generated_ids, confidence)
            If generate is False: confidence
        """
        if self.llm is None:
            raise RuntimeError(
                "forward() requires an LLM but this instance was created "
                "without one (cached-activations mode). Use training_step/"
                "validation_step with pre-computed activations instead."
            )

        if not generate:
            # Non-generative forward: get hidden states for the input only
            hidden_states = self.llm(
                input_ids, attention_mask, generate=False, return_hidden_states=True
            )
            activations = self._select_layer_activations(self._as_tuple(hidden_states))
            return self.probe(activations)

        # --- Generate response, then extract activations ---
        with torch.no_grad():
            llm_output = self.llm(
                input_ids, attention_mask, generate=True, return_hidden_states=True
            )

        if isinstance(llm_output, tuple) and len(llm_output) == 2:
            generated_output, hidden_states = llm_output
        else:
            generated_output = llm_output
            hidden_states = None

        # Handle GenerateOutput (dict-like with .sequences) vs plain tensor
        if hasattr(generated_output, "sequences"):
            generated_ids = generated_output.sequences
        else:
            generated_ids = generated_output

        if hidden_states is None:
            # Re-run forward on [prompt + answer] to extract hidden states
            batch_size = input_ids.shape[0]

            if not self.llm.is_seq2seq:
                full_ids = generated_ids  # already [prompt | answer]
                answer_len = full_ids.shape[1] - input_ids.shape[1]
                answer_mask = torch.ones(
                    batch_size,
                    answer_len,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                full_attention_mask = torch.cat([attention_mask, answer_mask], dim=1)
            else:
                full_ids = input_ids
                full_attention_mask = attention_mask

            hidden_states = self._run_llm_hidden_states(full_ids, full_attention_mask)

        activations = self._select_layer_activations(self._as_tuple(hidden_states))
        confidence = self.probe(activations)

        return generated_ids, confidence

    def _get_confidence(self, batch):
        """
        Get confidence from batch, supporting cached, on-the-fly, and live modes.

        - Cached mode: batch contains ``"activations"`` key with pre-computed
          activations (already pooled or raw).
        - On-the-fly mode: batch contains ``"prompt_lengths"`` — runs a single
          non-generative LLM forward pass on pre-tokenized [prompt + answer]
          inputs, slices activations, and feeds to probe.
        - Live mode: batch contains ``"input_ids"`` and ``"attention_mask"``;
          the LLM is run to extract activations (prompt-only).
        """
        if "activations" in batch:
            return self.probe(
                batch["activations"],
                seq_mask=batch.get("seq_mask"),
            )
        elif "prompt_lengths" in batch:
            return self._get_confidence_on_the_fly(batch)
        else:
            return self(batch["input_ids"], batch["attention_mask"], generate=False)

    def _get_confidence_on_the_fly(self, batch):
        """
        Get confidence using on-the-fly LLM forward pass on pre-tokenized
        [prompt + answer] sequences.

        Runs a single non-generative forward pass, extracts hidden states,
        slices to the configured region (prompt/answer/both), pools, and
        feeds through the probe.

        Args:
            batch: dict with keys ``input_ids``, ``attention_mask``,
                   ``prompt_lengths``, ``correct``
        """
        attention_mask = batch["attention_mask"]
        prompt_lengths = batch["prompt_lengths"]

        # Single non-generative forward pass — no generation needed
        hidden_states = self._run_llm_hidden_states(batch["input_ids"], attention_mask)

        # For pool_strategy in ("mean", "last"): pool here, then classify
        if self.pool_strategy in ("mean", "last"):
            pooled = self._extract_and_pool_activations(
                hidden_states, prompt_lengths, attention_mask
            )
            return self.probe.classify(pooled)

        # pool_strategy == "none": preserve sequence for attentive/seq pooling
        activations = self.probe.pool_layers(
            self._select_layer_activations(hidden_states)
        )

        # activations is now (batch, seq, dim) or (batch, dim)
        if activations.ndim == 2:
            # Already pooled (no sequence dimension)
            return self.probe.classify(activations)

        # Pad variable-length sliced sequences and create seq_mask
        sliced_list = self._slice_samples(activations, prompt_lengths, attention_mask)
        padded = torch.nn.utils.rnn.pad_sequence(
            sliced_list,
            batch_first=True,
            padding_value=0.0,
        )
        seq_lengths = torch.tensor(
            [s.shape[0] for s in sliced_list], device=padded.device
        )
        seq_mask = (
            torch.arange(padded.shape[1], device=padded.device) < seq_lengths[:, None]
        )

        # seq_pooling (or masked mean fallback) + linear + sigmoid
        return self.probe.classify(self.probe.pool_sequence(padded, seq_mask))

    def training_step(self, batch, batch_idx):
        confidence = self._get_confidence(batch)
        loss = self.loss_fn(confidence, batch["correct"].float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if self._trainer is not None and len(self.trainer.optimizers) > 0:
            self.log(
                "lr",
                self.trainer.optimizers[0].param_groups[0]["lr"],
                on_step=True,
                prog_bar=True,
            )

        # Snapshot layer weights for visualization
        if self.global_step % self._weight_snapshot_interval == 0:
            layer_pooling = getattr(self.probe, "layer_pooling", None)
            if layer_pooling is not None and hasattr(layer_pooling, "get_weights"):
                weights = layer_pooling.get_weights().detach().cpu().tolist()
                self._weight_snapshots.append(
                    {"step": self.global_step, "weights": weights}
                )

        return loss

    def on_fit_end(self):
        """Save layer weight evolution to a JSON file for later visualization."""
        if not self._weight_snapshots:
            return

        log_dir = self.trainer.log_dir or os.getcwd()
        weights_file = os.path.join(log_dir, "layer_weights_history.json")
        try:
            # Include layer_idx mapping so we know which layers the weights correspond to
            layer_indices = (
                self.layer_idx
                if isinstance(self.layer_idx, list)
                else list(range(self.num_layers or 0))
            )
            data = {
                "layer_indices": layer_indices,
                "snapshots": self._weight_snapshots,
            }
            with open(weights_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Layer weight history saved to {weights_file}")
        except Exception as e:
            print(f"Failed to save layer weight history: {e}")

    def on_validation_epoch_start(self):
        self.val_confidences = []
        self.val_correctness = []
        self._eval_outputs = []

    def validation_step(self, batch, batch_idx):
        # --- Live evaluation path (question answering evaluation with text output) ---
        if "question" in batch:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            questions = batch["question"]
            gold_answers = _get_gold_answers(batch)

            with torch.no_grad():
                generated_ids, confidence = self.forward(
                    input_ids, attention_mask, generate=True
                )

            for i, gen_tokens in enumerate(
                self.llm.strip_prompt(generated_ids, input_ids)
            ):
                raw_output = self.llm.tokenizer.decode(
                    gen_tokens, skip_special_tokens=True
                )
                pred_answer = self.llm.extractor.extract_answer(raw_output)

                self._eval_outputs.append(
                    {
                        "question": questions[i],
                        "gold_answers": gold_answers[i],
                        "pred_answer": pred_answer,
                        "confidence": confidence[i].item(),
                        "raw_output": raw_output,
                    }
                )

            return {"batch_size": len(questions)}

        # --- Standard training/validation step (cached activations or live LLM) ---
        confidence = self._get_confidence(batch)
        correct = batch["correct"]
        loss = self.loss_fn(confidence, correct.float())
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.val_confidences.append(confidence.detach().cpu())
        self.val_correctness.append(correct.detach().cpu())
        return {"val_loss": loss, "confidence": confidence, "correct": correct}

    def on_validation_epoch_end(self):
        # --- Live evaluation path ---
        if self._eval_outputs:
            self._compute_eval_metrics()
            return

        # --- Cached-mode training path ---
        if not self.val_confidences or not self.val_correctness:
            return

        confidences = torch.cat(self.val_confidences, dim=0).float()
        correctness = torch.cat(self.val_correctness, dim=0).float()

        metrics = {
            "val_ece": ExpectedCalibrationError(n_bins=10),
            "val_cross_entropy": ConfidenceCrossEntropy(),
            "val_auc": ConfidenceAUCScore(),
            "val_ecuas_0": ConfidenceECUAS(n=0),
            "val_ecuas_1": ConfidenceECUAS(n=1),
            "val_ecuas_128": ConfidenceECUAS(n=128),
        }
        for name, metric in metrics.items():
            metric.update(confidences, correctness)
            self.log(name, metric.compute(), prog_bar=True, sync_dist=True)

        self.val_confidences.clear()
        self.val_correctness.clear()

    def _compute_eval_metrics(self):
        """
        Compute exact-match correctness and calibration metrics for live
        evaluation outputs.  Saves ``llm_outputs.csv`` and ``metrics.csv``
        to the trainer's log directory.
        """
        # Determine correctness by exact match
        for out in self._eval_outputs:
            pred = out["pred_answer"] or ""
            gold = out["gold_answers"]
            if isinstance(gold, list):
                out["correct"] = check_exact_match(pred, gold)
            else:
                out["correct"] = check_exact_match(pred, [gold] if gold else [])

        all_confidences = np.array(
            [float(out["confidence"]) for out in self._eval_outputs]
        )
        all_correctness = np.array([out["correct"] for out in self._eval_outputs])
        metrics_results = _log_calibration_metrics(
            self, all_confidences, all_correctness
        )

        # Save llm_outputs.csv
        log_dir = self.trainer.log_dir or os.getcwd()
        rows = []
        for out in self._eval_outputs:
            gold_str = (
                "|".join(out["gold_answers"])
                if isinstance(out["gold_answers"], list)
                else str(out["gold_answers"] or "")
            )
            rows.append(
                [
                    out["question"],
                    gold_str,
                    out["pred_answer"],
                    f"{out['confidence']:.6f}",
                    "Yes" if out["correct"] else "No",
                    out["raw_output"],
                ]
            )
        _write_csv(
            os.path.join(log_dir, "llm_outputs.csv"),
            [
                "question",
                "gold_answers",
                "pred_answer",
                "confidence",
                "correct",
                "raw_output",
            ],
            rows,
            "LLM outputs",
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"Evaluation Results ({len(self._eval_outputs)} samples)")
        print(f"{'='*60}")
        for name, value in metrics_results.items():
            print(f"  {name}: {value:.4f}")
        print(f"{'='*60}")

        self._eval_outputs.clear()

    def configure_optimizers(self):
        # Only self.probe parameters are optimized
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr)
        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(
                optimizer, max_steps=self.trainer.estimated_stepping_batches
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        return optimizer


class SplitConfidenceAugmentedLLM(ConfidenceAugmentedLLM):
    """
    Cached-mode ConfidenceAugmentedLLM that pools the prompt and answer
    activations separately and merges them in a ``SplitProbe``.

    Expects batches from ``ActivationsDataModule(activations_to_use="split")``
    with keys ``prompt_activations`` and ``answer_activations`` (plus optional
    ``prompt_seq_mask`` / ``answer_seq_mask``). ``layer_pooling`` and
    ``seq_pooling`` are copied, so each part gets its own weights.
    """

    def __init__(
        self,
        model_dim: int,
        merge: str = "concat",
        layer_idx: int | list[int] | str = -1,
        num_layers: int | None = None,
        layer_pooling: nn.Module | None = None,
        seq_pooling: nn.Module | None = None,
        lr: float = 1e-3,
        lr_scheduler: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ]
        | None = None,
    ):
        # The parent resolves the default layer pooling from layer_idx and
        # builds a single-part Probe; reuse its pooling modules for both parts.
        super().__init__(
            llm=None,
            model_dim=model_dim,
            layer_idx=layer_idx,
            num_layers=num_layers,
            layer_pooling=layer_pooling,
            seq_pooling=seq_pooling,
            lr=lr,
            lr_scheduler=lr_scheduler,
        )
        self.merge = merge
        self.probe = SplitProbe(
            self.model_dim,
            merge=merge,
            layer_pooling=self.probe.layer_pooling,
            seq_pooling=self.probe.seq_pooling,
        )

    def _get_confidence(self, batch):
        if "prompt_activations" not in batch or "answer_activations" not in batch:
            raise ValueError(
                "SplitConfidenceAugmentedLLM expects 'prompt_activations' and "
                "'answer_activations' in the batch; use "
                "ActivationsDataModule(activations_to_use='split')."
            )
        return self.probe(
            batch["prompt_activations"],
            batch["answer_activations"],
            prompt_seq_mask=batch.get("prompt_seq_mask"),
            answer_seq_mask=batch.get("answer_seq_mask"),
        )
