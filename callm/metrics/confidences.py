"""
Calibration metrics for evaluating confidence predictions.

Uses torchmetrics built-ins where available (ECE, AUROC, Brier/MSE) and
provides custom implementations for Cross Entropy, CCAS, and Gamma-CCAS.
"""

import torch
from torchmetrics import Metric, MeanSquaredError
from torchmetrics.classification import BinaryCalibrationError, BinaryAUROC
import torch.nn.functional as F
from torch_uncertainty.metrics.classification import AURC as _AURC, FPRx as _FPRx


# ──────────────────────────────────────────────────────
# Built-in wrappers — thin adapters so every metric
# has the same (confidences, correctness) signature.
# ──────────────────────────────────────────────────────


class ExpectedCalibrationError(BinaryCalibrationError):
    """ECE via torchmetrics BinaryCalibrationError (L1 norm)."""

    def __init__(self, n_bins: int = 10, **kwargs):
        super().__init__(n_bins=n_bins, norm="l1", **kwargs)

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        if torch.isnan(confidences).any() or torch.isnan(correctness).any():
            raise ValueError("NaN values found in input tensors.")
        super().update(confidences.float(), correctness.long())

    @classmethod
    def create_shortcut_function(cls):
        def shortcut_function(
            confidences: torch.Tensor, correctness: torch.Tensor, nbins: int = 10
        ) -> torch.Tensor:
            metric = cls(n_bins=nbins)
            metric.update(confidences, correctness)
            return metric.compute().item()

        return shortcut_function


class ConfidenceErrorRate(Metric):
    """
    Error Rate for confidence predictions.

    Error Rate = mean[ y != argmax(p) ]
    """

    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("num_errors", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        if torch.isnan(confidences).any() or torch.isnan(correctness).any():
            raise ValueError("NaN values found in input tensors.")
        if confidences.ndim != 1 or correctness.ndim != 1:
            raise ValueError("Confidences must be 1D and correctness must be 1D.")
        self.num_errors += (correctness == 0).sum()
        self.count += confidences.size(0)

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            raise ValueError("No samples to compute error rate.")
        error_rate = self.num_errors / self.count
        return error_rate

    @classmethod
    def create_shortcut_function(cls):
        def shortcut_function(
            confidences: torch.Tensor, correctness: torch.Tensor
        ) -> torch.Tensor:
            metric = cls()
            metric.update(confidences, correctness)
            return metric.compute().item()

        return shortcut_function


class ConfidenceAUCScore(BinaryAUROC):
    """AUROC via torchmetrics BinaryAUROC."""

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        if torch.isnan(confidences).any() or torch.isnan(correctness).any():
            raise ValueError("NaN values found in input tensors.")
        super().update(confidences.float(), correctness.long())

    def compute(self) -> torch.Tensor:
        try:
            return super().compute()
        except (ValueError, IndexError):
            # Single class or empty — undefined
            raise ValueError("No samples to compute AUC score.")

    @classmethod
    def create_shortcut_function(cls):
        def shortcut_function(
            confidences: torch.Tensor, correctness: torch.Tensor
        ) -> torch.Tensor:
            metric = cls()
            metric.update(confidences, correctness)
            return metric.compute().item()

        return shortcut_function


class ConfidenceBrierScore(MeanSquaredError):
    """Brier Score = MSE between confidence and correctness."""

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        if torch.isnan(confidences).any() or torch.isnan(correctness).any():
            raise ValueError("NaN values found in input tensors.")
        super().update(confidences.float(), correctness.float())

    @classmethod
    def create_shortcut_function(cls):
        def shortcut_function(
            confidences: torch.Tensor, correctness: torch.Tensor
        ) -> torch.Tensor:
            metric = cls()
            metric.update(confidences, correctness)
            return metric.compute().item()

        return shortcut_function


# ──────────────────────────────────────────────────────
# Custom metrics
# ──────────────────────────────────────────────────────


class ConfidenceCrossEntropy(Metric):
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
        if torch.isnan(confidences).any() or torch.isnan(correctness).any():
            raise ValueError("NaN values found in input tensors.")
        conf = confidences.float().clamp(self.epsilon, 1 - self.epsilon)
        corr = correctness.float()
        ce = F.binary_cross_entropy(conf, corr, reduction="sum")
        self.sum_ce += ce
        self.count += confidences.numel()

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            raise ValueError("No samples to compute cross entropy.")
        return self.sum_ce / self.count

    @classmethod
    def create_shortcut_function(cls):
        def shortcut_function(
            confidences: torch.Tensor, correctness: torch.Tensor
        ) -> torch.Tensor:
            metric = cls()
            metric.update(confidences, correctness)
            return metric.compute().item()

        return shortcut_function


class ConfidenceNCCAS(Metric):
    """
    nCCAS (Confidence Cost Abstention Game) metric.
    """

    full_state_update = False

    def __init__(self, n: int = 0, epsilon: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.epsilon = epsilon
        if n == 0:
            self.cost_fun = (
                lambda q, correct_indicator: 1
                - q
                - (1 - correct_indicator) * torch.log(1 - q)
            )
        elif n > 0:
            self.cost_fun = lambda q, correct_indicator: (1 - q) ** (n + 1) + (
                n + 1
            ) / n * (1 - (1 - q) ** n) * (1 - correct_indicator)
        else:
            raise ValueError("n must be non-negative.")
        self.add_state("sum_cost", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        if torch.isnan(confidences).any() or torch.isnan(correctness).any():
            raise ValueError("NaN values found in input tensors.")
        q = confidences.float().clamp(self.epsilon, 1 - self.epsilon)
        indicator = correctness.float()
        cost = self.cost_fun(q, indicator)
        self.sum_cost += cost.sum()
        self.count += confidences.numel()

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            raise ValueError("No samples to compute nCCAS.")
        return self.sum_cost / self.count

    @classmethod
    def create_shortcut_function(cls):
        def shortcut_function(
            confidences: torch.Tensor, correctness: torch.Tensor, n: int = 0
        ) -> torch.Tensor:
            metric = cls(n=n)
            metric.update(confidences, correctness)
            return metric.compute().item()

        return shortcut_function


class ConfidenceGammaCCAS(Metric):
    """
    Gamma-nCCAS (Confidence Cost Abstention Game) metric.

    Evaluates the expected cost of a selective prediction system that can
    abstain based on confidence scores. At a given gamma, the cost is:

        C_γ(y_k, d_j)  = I(k ≠ j)   if d_j ≠ d_r  (answer)
                       = γ          if d_j = d_r  (abstain)

    The decision rule abstains when s(q) < γ, where s = 1 - confidence.

    Parameters
    ----------
    gamma : float
        The operating point gamma ∈ (0, 1).
    epsilon : float
        Small value to avoid division by zero.
    """

    full_state_update = False

    def __init__(
        self,
        gamma: float = 0.5,
        epsilon: float = 1e-7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.add_state("all_confidences", default=[], dist_reduce_fx="cat")
        self.add_state("all_correctness", default=[], dist_reduce_fx="cat")

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        if torch.isnan(confidences).any() or torch.isnan(correctness).any():
            raise ValueError("NaN values found in input tensors.")
        self.all_confidences.append(confidences.float())
        self.all_correctness.append(correctness.float())

    def compute(self) -> torch.Tensor:
        if not self.all_confidences:
            raise ValueError("No samples to compute Gamma-CCAS.")

        confidences = torch.cat(self.all_confidences)
        correctness = torch.cat(self.all_correctness)

        # Score: estimated error probability
        s = 1.0 - confidences

        # Decision: abstain if s < γ, answer otherwise
        abstain_mask = s < self.gamma
        answer_mask = ~abstain_mask

        # Cost when answering: C̃  where C̃ = 1 - correctness (0-1 loss)
        base_cost = 1.0 - correctness  # 0 if correct, 1 if incorrect
        answer_costs = base_cost[answer_mask]

        # Cost when abstaining: γ per sample
        n_abstain = abstain_mask.sum()
        abstain_costs = self.gamma * n_abstain.float()

        # Expected cost = mean over all samples
        total_cost = answer_costs.sum() + abstain_costs

        return total_cost / len(confidences)

    @classmethod
    def create_shortcut_function(cls):
        def shortcut_function(
            confidences: torch.Tensor, correctness: torch.Tensor, gamma: float = 0.5
        ) -> torch.Tensor:
            metric = cls(gamma=gamma)
            metric.update(confidences, correctness)
            return metric.compute().item()

        return shortcut_function


class CCAS(ConfidenceNCCAS):
    def __init__(self, *args, **kwargs):
        super().__init__(n=0, *args, **kwargs)


class ConfidenceAURC(_AURC):
    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        # Concat (conf, 1-conf)
        probs = torch.cat([1 - confidences.view(-1, 1), confidences.view(-1, 1)], dim=1)
        return super().update(probs, correctness.long())

    @classmethod
    def create_shortcut_function(cls):
        def shortcut_function(
            confidences: torch.Tensor, correctness: torch.Tensor
        ) -> torch.Tensor:
            metric = cls()
            metric.update(confidences, correctness)
            return metric.compute().item()

        return shortcut_function


class FPR95(_FPRx):
    def __init__(self, **kwargs):
        super().__init__(recall_level=0.95, pos_label=1, **kwargs)

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        return super().update(confidences, correctness.long())

    @classmethod
    def create_shortcut_function(cls):
        def shortcut_function(
            confidences: torch.Tensor, correctness: torch.Tensor
        ) -> torch.Tensor:
            metric = cls()
            metric.update(confidences, correctness)
            return metric.compute().item()

        return shortcut_function
