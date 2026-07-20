from __future__ import annotations

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from typing import TYPE_CHECKING

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
        weights = torch.softmax(self.layer_weights, dim=0)

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
    A simple probe that gets single layer activations, mean pools across sequence,
    and applies a linear layer + sigmoid to return the confidence directly.
    """

    def __init__(self, model_dim: int, layer_pooling: nn.Module | None = None):
        super().__init__()
        self.model_dim = model_dim
        self.layer_pooling = layer_pooling
        self.linear = nn.Linear(model_dim, 1)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: shape (#batch, #seq, #model_dim) or (#batch, #model_dim) if pre-pooled
                         or (#batch, num_layers, #seq, #model_dim) / (#batch, num_layers, #model_dim)
                         if layer_pooling is specified.
        Returns:
            confidence: shape (#batch)
        """
        if self.layer_pooling is not None:
            activations = self.layer_pooling(activations)

        if activations.ndim == 3:
            # Raw sequence input: mean pool over sequence dimension
            pooled = activations.mean(dim=1)  # (#batch, #model_dim)
        elif activations.ndim == 2:
            # Pre-pooled input: already (#batch, #model_dim)
            pooled = activations
        else:
            raise ValueError(
                f"Expected 2D or 3D activations after layer pooling, got {activations.ndim}D"
            )

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

    Two modes of operation:
    - **Live mode** (``llm`` provided): Runs the LLM forward pass to extract
      activations on the fly and feeds them to the probe.
    - **Cached mode** (``llm=None``): Expects pre-computed activations in the
      batch (key ``"activations"``), skipping the LLM entirely. Requires
      ``model_dim`` to be specified explicitly.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model_dim: int | None = None,
        probe: nn.Module | None = None,
        layer_idx: int | list[int] | str = -1,
        num_layers: int | None = None,
        layer_pooling: nn.Module | None = None,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.llm = llm
        self.layer_idx = layer_idx
        self.lr = lr

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
            self.probe = Probe(self.model_dim, layer_pooling=layer_pooling)
        else:
            self.probe = probe

        # BCELoss because self.probe outputs confidence probabilities in [0, 1]
        self.loss_fn = nn.BCELoss()

        self.val_confidences = []
        self.val_correctness = []

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

        llm_output = self.llm(
            input_ids,
            attention_mask,
            generate=generate,
            return_hidden_states=True,
        )

        if generate:
            generated_ids, hidden_states = llm_output
        else:
            hidden_states = llm_output

        num_all_layers = len(hidden_states)
        layer_pooling = getattr(self.probe, "layer_pooling", None)

        if isinstance(layer_pooling, SingleLayerSelector):
            activations = torch.stack(hidden_states, dim=1)
        elif layer_pooling is not None:
            if isinstance(self.layer_idx, list):
                selected = [
                    hidden_states[layer_i % num_all_layers]
                    for layer_i in self.layer_idx
                ]
                activations = torch.stack(selected, dim=1)
            elif isinstance(self.layer_idx, str) and self.layer_idx.lower() == "all":
                activations = torch.stack(hidden_states, dim=1)
            else:
                activations = torch.stack(hidden_states, dim=1)
        else:
            if isinstance(self.layer_idx, int):
                activations = hidden_states[self.layer_idx % num_all_layers]
            else:
                activations = torch.stack(hidden_states, dim=1)

        confidence = self.probe(activations)

        if generate:
            return generated_ids, confidence
        return confidence

    def _get_confidence(self, batch):
        """
        Get confidence from batch, supporting both cached and live modes.

        - Cached mode: batch contains ``"activations"`` key with pre-computed
          activations (already pooled or raw).
        - Live mode: batch contains ``"input_ids"`` and ``"attention_mask"``;
          the LLM is run to extract activations.
        """
        if "activations" in batch:
            return self.probe(batch["activations"])
        else:
            return self(batch["input_ids"], batch["attention_mask"], generate=False)

    def training_step(self, batch, batch_idx):
        confidence = self._get_confidence(batch)
        loss = self.loss_fn(confidence, batch["correct"].float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_confidences = []
        self.val_correctness = []

    def validation_step(self, batch, batch_idx):
        confidence = self._get_confidence(batch)
        correct = batch["correct"]
        loss = self.loss_fn(confidence, correct.float())
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        self.val_confidences.append(confidence.detach().cpu())
        self.val_correctness.append(correct.detach().cpu())
        return {"val_loss": loss, "confidence": confidence, "correct": correct}

    def on_validation_epoch_end(self):
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

    def configure_optimizers(self):
        # Only self.probe parameters are optimized
        return torch.optim.AdamW(self.probe.parameters(), lr=self.lr)
