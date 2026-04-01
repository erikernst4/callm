"""
Calibration metrics for evaluating confidence predictions.

Uses torchmetrics built-ins where available (ECE, AUROC, Brier/MSE) and
provides custom implementations for Cross Entropy, CCAG, and Gamma-CCAG.
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
        if torch.isnan(confidences).any() or torch.isnan(correctness).any():
            raise ValueError("NaN values found in input tensors.")
        super().update(confidences.float(), correctness.long())


class AUCScore(BinaryAUROC):
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
            return torch.tensor(float("nan"))


class BrierScore(MeanSquaredError):
    """Brier Score = MSE between confidence and correctness."""

    def update(self, confidences: torch.Tensor, correctness: torch.Tensor) -> None:
        if torch.isnan(confidences).any() or torch.isnan(correctness).any():
            raise ValueError("NaN values found in input tensors.")
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
        if torch.isnan(confidences).any() or torch.isnan(correctness).any():
            raise ValueError("NaN values found in input tensors.")
        conf = confidences.float().clamp(self.epsilon, 1 - self.epsilon)
        corr = correctness.float()
        ce = F.binary_cross_entropy(conf, corr, reduction="sum")
        self.sum_ce += ce
        self.count += confidences.numel()

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        return self.sum_ce / self.count


class CCAG(Metric):
    """
    CCAG (Confidence Cost Abstention Game) metric.

    cost = log(2 - q_m) · (2 - 𝟙) − log(1 - q_m) · (1 - 𝟙)

    where q_m is the confidence and 𝟙 is 1 when correct, 0 otherwise.
    """

    full_state_update = False

    def __init__(self, cost_func=None, epsilon: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.cost_fun = (
            cost_func if cost_func is not None else self._default_integrated_cost
        )
        self.add_state("sum_cost", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def _default_integrated_cost(self, q: float, correct_indicator: float) -> float:
        return torch.log(2 - q) * (2 - correct_indicator) - torch.log(1 - q) * (
            1 - correct_indicator
        )

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
            return torch.tensor(float("nan"))
        return self.sum_cost / self.count


class GammaCCAG(Metric):
    """
    Gamma-CCAG (Confidence Cost Abstention Game) metric.

    Evaluates the expected cost of a selective prediction system that can
    abstain based on confidence scores. At a given gamma, the cost is:

        C_γ(y_k, d_j) = a(γ) · C̃(y_k, d_j)   if d_j ≠ d_r  (answer)
                       = b(γ)                    if d_j = d_r  (abstain)

    The decision rule abstains when s(q) > b(γ)/a(γ), where s = 1 - confidence.
    The base cost C̃ is 0-1 loss (1 if incorrect, 0 if correct).

    Parameters
    ----------
    a_func : callable
        Function gamma -> a(gamma). Scales the base cost when answering.
    b_func : callable
        Function gamma -> b(gamma). Fixed cost when abstaining.
    gamma : float
        The operating point gamma ∈ (0, 1).
    epsilon : float
        Small value to avoid division by zero.
    """

    full_state_update = False

    def __init__(
        self,
        a_func=None,
        b_func=None,
        gamma: float = 0.5,
        epsilon: float = 1e-7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.a_func = a_func if a_func is not None else (lambda g: 1.0)
        self.b_func = b_func if b_func is not None else (lambda g: g)
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
            return torch.tensor(float("nan"))

        confidences = torch.cat(self.all_confidences)
        correctness = torch.cat(self.all_correctness)

        if len(confidences) == 0:
            return torch.tensor(float("nan"))

        a_val = self.a_func(self.gamma)
        b_val = self.b_func(self.gamma)
        threshold = b_val / a_val

        # Score: estimated error probability
        s = 1.0 - confidences

        # Decision: abstain if s > threshold, answer otherwise
        abstain_mask = s > threshold
        answer_mask = ~abstain_mask

        # Cost when answering: a(γ) * C̃  where C̃ = 1 - correctness (0-1 loss)
        base_cost = 1.0 - correctness  # 0 if correct, 1 if incorrect
        answer_costs = a_val * base_cost[answer_mask]

        # Cost when abstaining: b(γ)
        n_abstain = abstain_mask.sum()
        abstain_costs = b_val * n_abstain.float()

        # Expected cost = mean over all samples
        total_cost = answer_costs.sum() + abstain_costs
        return total_cost / len(confidences)
