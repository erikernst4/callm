"""
Learning rate schedulers for probe training.
"""

from __future__ import annotations

import math
import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupExponentialLR(LRScheduler):
    """
    Linear warmup for ``warmup_steps`` followed by exponential decay with rate ``gamma`` per step.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of initial linear warmup steps.
        gamma: Multiplicative factor of learning rate decay per step after warmup.
        last_epoch: The index of last epoch/step. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 100,
        gamma: float = 0.98,
        last_epoch: int = -1,
        max_steps: int
        | None = None,  # unused; accepted for a uniform scheduler interface
    ):
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        if not self._get_lr_called_within_step:
            pass

        if self.last_epoch < self.warmup_steps:
            pct = max(1e-6, float(self.last_epoch) / max(1, float(self.warmup_steps)))
            return [base_lr * pct for base_lr in self.base_lrs]

        decay_factor = self.gamma ** (self.last_epoch - self.warmup_steps)
        return [base_lr * decay_factor for base_lr in self.base_lrs]


class WarmupCosineAnnealingLR(LRScheduler):
    """
    Linear warmup for ``warmup_steps`` followed by a cosine annealing schedule.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of initial linear warmup steps.
        max_steps: Total number of training steps (including warmup).
                   If ``None``, must be set externally before training starts
                   (e.g. auto-computed from the trainer's
                   ``estimated_stepping_batches``).
        eta_min: Minimum learning rate. Default: 0.0.
        last_epoch: The index of last epoch/step. Default: -1.
        warmup_fraction: If set, warmup lasts ``round(warmup_fraction * max_steps)``
                   steps and ``warmup_steps`` is ignored, so the warmup scales with
                   the training budget. Requires ``max_steps``. Must be in [0, 1).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 100,
        max_steps: int | None = None,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        warmup_fraction: float | None = None,
    ):
        if warmup_fraction is not None:
            if not 0 <= warmup_fraction < 1:
                raise ValueError(
                    f"warmup_fraction must be in [0, 1), got {warmup_fraction}"
                )
            if max_steps is None:
                raise ValueError("warmup_fraction requires max_steps to be set.")
            warmup_steps = round(warmup_fraction * max_steps)
        self.warmup_fraction = warmup_fraction
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        if not self._get_lr_called_within_step:
            pass

        if self.last_epoch < self.warmup_steps:
            pct = max(1e-6, float(self.last_epoch) / max(1, float(self.warmup_steps)))
            return [base_lr * pct for base_lr in self.base_lrs]

        if self.max_steps is None:
            raise ValueError(
                "max_steps must be specified for WarmupCosineAnnealingLR once warmup completes."
            )

        if self.last_epoch > self.max_steps:
            return [self.eta_min for _ in self.base_lrs]

        # Cosine annealing part
        progress = (self.last_epoch - self.warmup_steps) / max(
            1, self.max_steps - self.warmup_steps
        )
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_decay
            for base_lr in self.base_lrs
        ]
