from __future__ import annotations

import copy

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


class SplitProbe(nn.Module):
    """
    A probe that pools the prompt and answer activations separately, each with
    its own layer pooling and sequence pooling, and merges the two pooled
    representations before the final linear layer + sigmoid.

    ``merge`` selects how the two parts are combined:
    - ``"concat"``: linear layer over the concatenation [h_prompt; h_answer].
    - ``"mean"``: linear layer over the mean (h_prompt + h_answer) / 2.
    - ``"mean_logits"``: one linear layer per part, averaging their logits.
    """

    PARTS = ("prompt", "answer")
    MERGES = ("concat", "mean", "mean_logits")

    def __init__(
        self,
        model_dim: int,
        merge: str = "concat",
        layer_pooling: nn.Module | None = None,
        seq_pooling: nn.Module | None = None,
    ):
        super().__init__()
        if merge not in self.MERGES:
            raise ValueError(f"merge must be one of {self.MERGES}. Got: {merge}")
        self.model_dim = model_dim
        self.merge = merge
        # Each part gets its own copy, so prompt and answer pooling are trained independently
        self.layer_pooling = self._per_part(layer_pooling)
        self.seq_pooling = self._per_part(seq_pooling)
        if merge == "concat":
            self.linear = nn.Linear(2 * model_dim, 1)
        elif merge == "mean":
            self.linear = nn.Linear(model_dim, 1)
        else:
            self.linear = nn.ModuleDict(
                {part: nn.Linear(model_dim, 1) for part in self.PARTS}
            )

    @classmethod
    def _per_part(cls, module: nn.Module | None) -> nn.ModuleDict | None:
        if module is None:
            return None
        return nn.ModuleDict({part: copy.deepcopy(module) for part in cls.PARTS})

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        prompt_activations: torch.Tensor,
        answer_activations: torch.Tensor,
        prompt_seq_mask: torch.Tensor | None = None,
        answer_seq_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            prompt_activations, answer_activations: same shapes accepted by
                ``Probe.forward`` ((#batch, [num_layers,] [#seq,] #model_dim)).
            prompt_seq_mask, answer_seq_mask: optional (batch, seq_len) bool
                masks — True = valid position.
        Returns:
            confidence: shape (#batch)
        """
        h_prompt = self.pool("prompt", prompt_activations, prompt_seq_mask)
        h_answer = self.pool("answer", answer_activations, answer_seq_mask)
        if self.merge == "concat":
            logits = self.linear(torch.cat([h_prompt, h_answer], dim=-1))
        elif self.merge == "mean":
            logits = self.linear((h_prompt + h_answer) / 2)
        else:
            logits = (
                self.linear["prompt"](h_prompt) + self.linear["answer"](h_answer)
            ) / 2
        return torch.sigmoid(logits.squeeze(-1))

    def pool(
        self,
        part: str,
        activations: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the layer and sequence pooling of ``part``, returning (#batch, #model_dim)."""
        activations = activations.to(dtype=self.dtype)
        if self.layer_pooling is not None:
            activations = self.layer_pooling[part](activations)
        if activations.ndim == 3:
            if self.seq_pooling is not None:
                return self.seq_pooling[part](activations, mask=seq_mask)
            if seq_mask is None:
                return activations.mean(dim=1)
            mask = seq_mask.unsqueeze(-1).to(dtype=activations.dtype)
            return (activations * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        if activations.ndim != 2:
            raise ValueError(
                f"Expected 2D or 3D activations after layer pooling, got {activations.ndim}D"
            )
        return activations
