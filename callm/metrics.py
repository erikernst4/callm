"""
Calibration metrics for evaluating confidence predictions.

Uses torchmetrics built-ins where available (ECE, AUROC, Brier/MSE) and
provides custom implementations for Cross Entropy and Confidence Cost.
"""

import torch
from torchmetrics import Metric, MeanSquaredError
from torchmetrics.classification import BinaryCalibrationError, BinaryAUROC
import torch.nn.functional as F


# ──────────────────────────────────────────────────────
# Built-in wrappers — thin adapters so every metric
# has the same (confidences, correctness) signature.
# ──────────────────────────────────────────────────────


class ExpectedCalibrationError(BinaryCalibrationError):
    """ECE via torchmetrics BinaryCalibrationError (L1 norm)."""

    def __init__(self, n_bins: int = 10, **kwargs):
        super().__init__(n_bins=n_bins, norm="l1", **kwargs)

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        super().update(confidences.float(), correctness.long())


class AUCScore(BinaryAUROC):
    """AUROC via torchmetrics BinaryAUROC."""

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        super().update(confidences.float(), correctness.long())

    def compute(self) -> torch.Tensor:
        try:
            return super().compute()
        except (ValueError, IndexError):
            # Single class or empty — undefined
            return torch.tensor(float("nan"))


class BrierScore(MeanSquaredError):
    """Brier Score = MSE between confidence and correctness."""

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        super().update(confidences.float(), correctness.float())


# ──────────────────────────────────────────────────────
# Custom metrics
# ──────────────────────────────────────────────────────


class CrossEntropy(Metric):
    """
    Binary Cross Entropy between confidence and correctness.

    CE = -mean[ y·log(p) + (1-y)·log(1-p) ]
    """

    full_state_update = False

    def __init__(self, epsilon: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.add_state("sum_ce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        conf = confidences.float().clamp(self.epsilon, 1 - self.epsilon)
        corr = correctness.float()
        ce = F.binary_cross_entropy(conf, corr, reduction="sum")
        self.sum_ce += ce
        self.count += confidences.numel()

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        return self.sum_ce / self.count


class ConfidenceCost(Metric):
    """
    Confidence Cost metric.

    cost = log(2 - q_m) · (2 - 𝟙) − log(1 - q_m) · (1 - 𝟙)

    where q_m is the confidence and 𝟙 is 1 when correct, 0 otherwise.
    """

    full_state_update = False

    def __init__(self, epsilon: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.add_state("sum_cost", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        q = confidences.float().clamp(self.epsilon, 1 - self.epsilon)
        indicator = correctness.float()
        cost = torch.log(2 - q) * (2 - indicator) - torch.log(1 - q) * (1 - indicator)
        self.sum_cost += cost.sum()
        self.count += confidences.numel()

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        return self.sum_cost / self.count
