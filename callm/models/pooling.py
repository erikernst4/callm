"""
Sequence pooling modules for probe training.

All modules share a consistent interface:
    __init__(model_dim: int, ...)
    forward(x: Tensor, mask: Tensor | None = None) -> Tensor

Input:  x of shape (batch, seq_len, model_dim)
        mask of shape (batch, seq_len) — True = valid, False = padding
Output: (batch, model_dim)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    """
    Mean pooling over the sequence dimension.

    When a mask is provided, computes the masked mean (only valid tokens).
    Without a mask, computes the standard mean.
    """

    def __init__(self, model_dim: int = 0):
        # model_dim accepted for interface compatibility but not used.
        super().__init__()

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, model_dim)
            mask: (batch, seq_len) — True for valid positions
        Returns:
            (batch, model_dim)
        """
        if mask is not None:
            x_masked = x * mask.unsqueeze(-1).to(dtype=x.dtype)
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=x.dtype)
            return x_masked.sum(dim=1) / lengths  # (batch, model_dim)
        return x.mean(dim=1)


class AttentivePooling(nn.Module):
    """
    MLP-based attentive pooling.

    Learns a small MLP (project → tanh → score) that produces per-token
    attention scores, softmaxes them, and computes the weighted sum.
    The query is implicitly derived from the sequence itself through the MLP.
    """

    def __init__(self, model_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(1, model_dim // 4)
        self.attention = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, model_dim)
            mask: (batch, seq_len) — True for valid positions
        Returns:
            (batch, model_dim)
        """
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        weights = weights.nan_to_num(0.0)  # safety for all-masked edge case
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # (batch, model_dim)


class QueryAttentivePooling(nn.Module):
    """
    Query-based attentive pooling with a fixed learnable query.

    A single learnable query vector (like a CLS token) computes dot-product
    attention scores against all sequence positions.  Cheaper than the MLP
    version and often just as effective.
    """

    def __init__(self, model_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(model_dim))
        nn.init.normal_(self.query, std=0.02)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, model_dim)
            mask: (batch, seq_len) — True for valid positions
        Returns:
            (batch, model_dim)
        """
        # Dot-product attention: score = x · query
        scores = torch.matmul(x, self.query) / math.sqrt(x.size(-1))  # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        weights = weights.nan_to_num(0.0)
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # (batch, model_dim)


class MultiHeadAttentivePooling(nn.Module):
    """
    Multi-head attentive pooling using PyTorch's ``nn.MultiheadAttention``.

    The query is derived from the sequence (masked mean), and keys/values
    are the sequence itself.  Different heads attend to different aspects.
    Output is concat + projected (standard MHA behaviour).
    """

    def __init__(self, model_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, model_dim)
            mask: (batch, seq_len) — True for valid positions
        Returns:
            (batch, model_dim)
        """
        # Query: masked mean of the sequence
        if mask is not None:
            x_masked = x * mask.unsqueeze(-1).to(dtype=x.dtype)
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=x.dtype)
            query = (x_masked.sum(dim=1) / lengths).unsqueeze(
                1
            )  # (batch, 1, model_dim)
        else:
            query = x.mean(dim=1, keepdim=True)  # (batch, 1, model_dim)

        # key_padding_mask: True = ignore (PyTorch convention, opposite of our mask)
        key_padding_mask = ~mask if mask is not None else None
        out, _ = self.mha(query, x, x, key_padding_mask=key_padding_mask)
        return out.squeeze(1)  # (batch, model_dim)


class QueryMultiHeadAttentivePooling(nn.Module):
    """
    Learnable query + ``nn.MultiheadAttention``.

    A single learnable query parameter (shared across all inputs) attends
    over the sequence via multi-head attention.  Keys and values are the
    sequence itself.  This is similar to a CLS-token approach but with
    multi-head expressiveness.
    """

    def __init__(self, model_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, model_dim))
        nn.init.normal_(self.query, std=0.02)
        self.mha = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, model_dim)
            mask: (batch, seq_len) — True for valid positions
        Returns:
            (batch, model_dim)
        """
        batch_size = x.shape[0]
        query = self.query.expand(batch_size, -1, -1)  # (batch, 1, model_dim)

        key_padding_mask = ~mask if mask is not None else None
        out, _ = self.mha(query, x, x, key_padding_mask=key_padding_mask)
        return out.squeeze(1)  # (batch, model_dim)


class TransformerPooling(nn.Module):
    """
    Full Transformer encoder block followed by (masked) mean pooling.

    A single ``nn.TransformerEncoderLayer`` processes the sequence to capture
    inter-token relationships, then the output is mean-pooled (with mask
    support) to produce a fixed-size representation.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 4,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_feedforward = dim_feedforward or model_dim
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, model_dim)
            mask: (batch, seq_len) — True for valid positions
        Returns:
            (batch, model_dim)
        """
        # src_key_padding_mask: True = ignore (PyTorch convention)
        src_key_padding_mask = ~mask if mask is not None else None
        out = self.transformer_layer(x, src_key_padding_mask=src_key_padding_mask)

        # Masked mean pooling over the transformer output
        if mask is not None:
            out_masked = out * mask.unsqueeze(-1).to(dtype=out.dtype)
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=out.dtype)
            return out_masked.sum(dim=1) / lengths
        return out.mean(dim=1)
