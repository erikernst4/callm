from __future__ import annotations

import os
import csv
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from callm.models.llm import LLM


class LayerWeightedAverage(nn.Module):
    """
    Learns a weight for each layer and computes the weighted average of
    activations across the layer dimension.
    """

    def __init__(self, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: shape (#batch, num_layers, #seq, #model_dim) or (#batch, num_layers, #model_dim)
        Returns:
            weighted_activations: shape (#batch, #seq, #model_dim) or (#batch, #model_dim)
        """
        weights = torch.softmax(self.layer_weights, dim=0).to(dtype=activations.dtype)

        if activations.ndim == 4:
            weights = weights.view(1, self.num_layers, 1, 1)
        elif activations.ndim == 3:
            weights = weights.view(1, self.num_layers, 1)
        else:
            raise ValueError(
                f"LayerWeightedAverage expected 3D or 4D input, got {activations.ndim}D"
            )

        return (activations * weights).sum(dim=1)

    def get_weights(self) -> torch.Tensor:
        """Returns the normalized (softmax) layer weights."""
        return torch.softmax(self.layer_weights, dim=0)


class SingleLayerSelector(nn.Module):
    """
    Selects activations from a single specified layer index.
    """

    def __init__(self, layer_idx: int, num_layers: int | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = num_layers

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: shape (#batch, num_layers, #seq, #model_dim) or (#batch, num_layers, #model_dim)
        Returns:
            layer_activations: shape (#batch, #seq, #model_dim) or (#batch, #model_dim)
        """
        if activations.ndim not in (3, 4):
            raise ValueError(
                f"SingleLayerSelector expected 3D or 4D input, got {activations.ndim}D"
            )

        n_layers = activations.shape[1]
        resolved_idx = self.layer_idx % n_layers
        return activations[:, resolved_idx]


class Probe(nn.Module):
    """
    A probe that gets single layer activations, pools across the sequence
    dimension (via a configurable ``seq_pooling`` module), and applies a
    linear layer + sigmoid to return the confidence directly.
    """

    def __init__(
        self,
        model_dim: int,
        layer_pooling: nn.Module | None = None,
        seq_pooling: nn.Module | None = None,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.layer_pooling = layer_pooling
        self.seq_pooling = (
            seq_pooling  # None → fallback to plain mean (backward compat)
        )
        self.linear = nn.Linear(model_dim, 1)

    def forward(
        self,
        activations: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            activations: shape (#batch, #seq, #model_dim) or (#batch, #model_dim) if pre-pooled
                         or (#batch, num_layers, #seq, #model_dim) / (#batch, num_layers, #model_dim)
                         if layer_pooling is specified.
            seq_mask: optional (batch, seq_len) bool mask — True = valid position.
                      Used by attentive pooling modules to ignore padding.
        Returns:
            confidence: shape (#batch)
        """
        if self.layer_pooling is not None:
            activations = self.layer_pooling(activations)

        if activations.ndim == 3:
            if self.seq_pooling is not None:
                pooled = self.seq_pooling(activations, mask=seq_mask)
            else:
                # Default mean pooling (backward compatible, no mask support)
                pooled = activations.mean(dim=1)  # (#batch, #model_dim)
        elif activations.ndim == 2:
            # Pre-pooled input: already (#batch, #model_dim)
            pooled = activations
        else:
            raise ValueError(
                f"Expected 2D or 3D activations after layer pooling, got {activations.ndim}D"
            )

        pooled = pooled.to(dtype=self.linear.weight.dtype)
        logits = self.linear(pooled)  # (#batch, 1)
        return torch.sigmoid(logits.squeeze(-1))  # (#batch)

    def get_layer_weights(self) -> torch.Tensor | None:
        """Returns the normalized (softmax) layer weights if layer_pooling is a LayerWeightedAverage."""
        if isinstance(self.layer_pooling, LayerWeightedAverage):
            return self.layer_pooling.get_weights()
        return None


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
        activations = self._select_layer_activations(hidden_states)

        # If activations have a layer dimension (multi-layer), we let the
        # probe's layer_pooling handle it first
        layer_pooling = getattr(self.probe, "layer_pooling", None)
        if layer_pooling is not None and activations.ndim >= 3:
            # activations: (batch, num_layers, seq, dim) or (batch, num_layers, dim)
            activations = layer_pooling(activations)

        # Now activations is (batch, seq, dim) or (batch, dim)
        if activations.ndim == 2:
            # Already pooled (no sequence dimension), return directly
            return activations

        # Per-sample slicing + pooling over the sequence dimension
        pooled_list = []
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
                pooled_list.append(
                    torch.zeros(
                        sample_act.shape[-1],
                        device=sample_act.device,
                        dtype=sample_act.dtype,
                    )
                )
            elif self.pool_strategy == "mean":
                pooled_list.append(sample_act.mean(dim=0))
            elif self.pool_strategy == "last":
                pooled_list.append(sample_act[-1])
            else:
                raise ValueError(f"Unknown pool_strategy: {self.pool_strategy}")

        return torch.stack(pooled_list)  # (batch, dim)

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
            if not isinstance(hidden_states, (list, tuple)):
                hidden_states = (hidden_states,)

            activations = self._select_layer_activations(hidden_states)
            confidence = self.probe(activations)
            return confidence

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

            with torch.no_grad():
                hidden_states = self.llm(
                    full_ids,
                    full_attention_mask,
                    generate=False,
                    return_hidden_states=True,
                )

        if not isinstance(hidden_states, (list, tuple)):
            hidden_states = (hidden_states,)

        activations = self._select_layer_activations(hidden_states)
        confidence = self.probe(activations)

        return generated_ids, confidence

    def _get_confidence(self, batch):
        """
        Get confidence from batch, supporting both cached and live modes.

        - Cached mode: batch contains ``"activations"`` key with pre-computed
          activations (already pooled or raw).
        - Live mode: batch contains ``"input_ids"`` and ``"attention_mask"``;
          the LLM is run to extract activations.
        """
        if "activations" in batch:
            return self.probe(
                batch["activations"],
                seq_mask=batch.get("seq_mask"),
            )
        else:
            return self(batch["input_ids"], batch["attention_mask"], generate=False)

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
        return loss

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
            gold_answers = (
                batch["gold_answers"]
                if "gold_answers" in batch
                else (
                    batch["label"]
                    if "label" in batch
                    else [None for _ in range(len(questions))]
                )
            )

            with torch.no_grad():
                generated_ids, confidence = self.forward(
                    input_ids, attention_mask, generate=True
                )

            input_length = input_ids.shape[1] if not self.llm.is_seq2seq else 0

            for i in range(len(questions)):
                if self.llm.is_seq2seq:
                    gen_tokens = generated_ids[i]
                else:
                    gen_tokens = generated_ids[i][input_length:]

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

        from ecuas import (
            ExpectedCalibrationError,
            ConfidenceCrossEntropy,
            ConfidenceAUCScore,
            ConfidenceECUAS,
        )

        ece_metric = ExpectedCalibrationError(n_bins=10)
        ce_metric = ConfidenceCrossEntropy()
        auc_metric = ConfidenceAUCScore()
        ecuas_0_metric = ConfidenceECUAS(n=0)
        ecuas_1_metric = ConfidenceECUAS(n=1)
        ecuas_128_metric = ConfidenceECUAS(n=128)

        ece_metric.update(confidences, correctness)
        ce_metric.update(confidences, correctness)
        auc_metric.update(confidences, correctness)
        ecuas_0_metric.update(confidences, correctness)
        ecuas_1_metric.update(confidences, correctness)
        ecuas_128_metric.update(confidences, correctness)

        self.log("val_ece", ece_metric.compute(), prog_bar=True, sync_dist=True)
        self.log(
            "val_cross_entropy", ce_metric.compute(), prog_bar=True, sync_dist=True
        )
        self.log("val_auc", auc_metric.compute(), prog_bar=True, sync_dist=True)
        self.log("val_ecuas_0", ecuas_0_metric.compute(), prog_bar=True, sync_dist=True)
        self.log("val_ecuas_1", ecuas_1_metric.compute(), prog_bar=True, sync_dist=True)
        self.log(
            "val_ecuas_128", ecuas_128_metric.compute(), prog_bar=True, sync_dist=True
        )

        self.val_confidences.clear()
        self.val_correctness.clear()

    def _compute_eval_metrics(self):
        """
        Compute exact-match correctness and calibration metrics for live
        evaluation outputs.  Saves ``llm_outputs.csv`` and ``metrics.csv``
        to the trainer's log directory.
        """
        from callm.utils import check_exact_match
        from ecuas import (
            ExpectedCalibrationError,
            ConfidenceBrierScore,
            ConfidenceCrossEntropy,
            ConfidenceAUCScore,
            CCAS,
        )

        # Determine correctness by exact match
        for out in self._eval_outputs:
            pred = out["pred_answer"] or ""
            gold = out["gold_answers"]
            if isinstance(gold, list):
                out["correct"] = check_exact_match(pred, gold)
            else:
                out["correct"] = check_exact_match(pred, [gold] if gold else [])

        all_confidences = []
        all_correctness = []
        for out in self._eval_outputs:
            all_confidences.append(float(out["confidence"]))
            all_correctness.append(out["correct"])

        all_confidences = np.array(all_confidences)
        all_correctness = np.array(all_correctness)

        # Log accuracy
        accuracy = float(all_correctness.mean()) if len(all_correctness) > 0 else 0.0
        self.log("val_accuracy", accuracy, prog_bar=True, sync_dist=True)

        metrics_results = {"val_accuracy": accuracy}

        confidences_t = torch.tensor(all_confidences, dtype=torch.float32)
        correctness_t = torch.tensor(all_correctness, dtype=torch.float32)

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
            self.log(name, value, prog_bar=True, sync_dist=True)
            metrics_results[name] = value

        # Save llm_outputs.csv
        log_dir = self.trainer.log_dir or os.getcwd()
        outputs_file = os.path.join(log_dir, "llm_outputs.csv")

        try:
            with open(outputs_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "question",
                        "gold_answers",
                        "pred_answer",
                        "confidence",
                        "correct",
                        "raw_output",
                    ]
                )
                for out in self._eval_outputs:
                    gold_str = (
                        "|".join(out["gold_answers"])
                        if isinstance(out["gold_answers"], list)
                        else str(out["gold_answers"] or "")
                    )
                    writer.writerow(
                        [
                            out["question"],
                            gold_str,
                            out["pred_answer"],
                            f"{out['confidence']:.6f}",
                            "Yes" if out["correct"] else "No",
                            out["raw_output"],
                        ]
                    )
            print(f"\nLLM outputs saved to {outputs_file}")
        except Exception as e:
            print(f"Failed to save LLM outputs: {e}")

        # Save metrics.csv
        metrics_file = os.path.join(log_dir, "metrics.csv")
        try:
            with open(metrics_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(list(metrics_results.keys()))
                writer.writerow(list(metrics_results.values()))
            print(f"Metrics saved to {metrics_file}")
        except Exception as e:
            print(f"Failed to save metrics: {e}")

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
            scheduler = self.lr_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        return optimizer
