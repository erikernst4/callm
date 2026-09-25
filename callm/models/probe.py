from __future__ import annotations

import torch
import torch.nn as nn

from .layer_pooling import LayerWeightedAverage


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
        activations = self.pool_layers(activations)
        if activations.ndim == 3:
            # Without seq_pooling, fall back to plain (unmasked) mean for backward compat
            if self.seq_pooling is None:
                seq_mask = None
            activations = self.pool_sequence(activations, seq_mask)
        elif activations.ndim != 2:
            raise ValueError(
                f"Expected 2D or 3D activations after layer pooling, got {activations.ndim}D"
            )
        return self.classify(activations)

    def pool_layers(self, activations: torch.Tensor) -> torch.Tensor:
        """Cast to the probe dtype and apply ``layer_pooling`` if present."""
        # Ensure activations match probe parameters dtype (e.g. if loaded as bfloat16/float16)
        activations = activations.to(dtype=self.linear.weight.dtype)
        if self.layer_pooling is not None:
            activations = self.layer_pooling(activations)
        return activations

    def pool_sequence(
        self,
        activations: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool (#batch, #seq, #model_dim) activations to (#batch, #model_dim)."""
        if self.seq_pooling is not None:
            return self.seq_pooling(activations, mask=seq_mask)
        if seq_mask is None:
            # Default mean pooling (backward compatible)
            return activations.mean(dim=1)
        # Masked mean pooling
        mask = seq_mask.unsqueeze(-1).to(dtype=activations.dtype)
        lengths = mask.sum(dim=1).clamp(min=1)
        return (activations * mask).sum(dim=1) / lengths

    def classify(self, pooled: torch.Tensor) -> torch.Tensor:
        """Map pooled (#batch, #model_dim) activations to confidences (#batch)."""
        pooled = pooled.to(dtype=self.linear.weight.dtype)
        logits = self.linear(pooled)  # (#batch, 1)
        return torch.sigmoid(logits.squeeze(-1))  # (#batch)

    def get_layer_weights(self) -> torch.Tensor | None:
        """Returns the normalized (softmax) layer weights if layer_pooling is a LayerWeightedAverage."""
        if isinstance(self.layer_pooling, LayerWeightedAverage):
            return self.layer_pooling.get_weights()
        return None
