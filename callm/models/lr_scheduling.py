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
        eta_min: Minimum learning rate. Default: 0.0.
        last_epoch: The index of last epoch/step. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 100,
        max_steps: int = 1000,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
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
