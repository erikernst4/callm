from __future__ import annotations

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from callm.models.llm import LLM


class Probe(nn.Module):
    """
    A simple probe that gets single layer activations, mean pools across sequence,
    and applies a linear layer + sigmoid to return the confidence directly.
    """

    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        self.linear = nn.Linear(model_dim, 1)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: shape (#batch, #seq, #model_dim) or (#batch, #model_dim) if pre-pooled
        Returns:
            confidence: shape (#batch)
        """
        if activations.ndim == 3:
            # Raw sequence input: mean pool over sequence dimension
            pooled = activations.mean(dim=1)  # (#batch, #model_dim)
        elif activations.ndim == 2:
            # Pre-pooled input: already (#batch, #model_dim)
            pooled = activations
        else:
            raise ValueError(f"Expected 2D or 3D activations, got {activations.ndim}D")

        logits = self.linear(pooled)  # (#batch, 1)
        return torch.sigmoid(logits.squeeze(-1))  # (#batch)


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
        layer_idx: int = -1,
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

        if probe is None:
            self.probe = Probe(self.model_dim)
        else:
            self.probe = probe

        # BCELoss because self.probe outputs confidence probabilities in [0, 1]
        self.loss_fn = nn.BCELoss()

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

        activations = hidden_states[self.layer_idx]
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
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        confidence = self._get_confidence(batch)
        correct = batch["correct"]
        loss = self.loss_fn(confidence, correct.float())
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return {"val_loss": loss, "confidence": confidence, "correct": correct}

    def configure_optimizers(self):
        # Only self.probe parameters are optimized
        return torch.optim.AdamW(self.probe.parameters(), lr=self.lr)
