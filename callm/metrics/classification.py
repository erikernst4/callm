import torch
from torchmetrics import Metric
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, MulticlassCalibrationError
from torch_uncertainty.metrics.classification import AURC as _AURC, FPRx as _FPRx

def log1mexp(x):
    """
    Numerically stable implementation of log(1 - exp(x)) for x < 0.
    """
    # Use a cutoff of -log(2)
    cutoff = -0.6931471805599453  # math.log(2)
    
    return torch.where(
        x > cutoff,
        torch.log(-torch.expm1(x)),  # Stable for small |x|
        torch.log1p(-torch.exp(x))   # Stable for large |x|
    )

class ClassificationErrorRate(Metric):
    """
    Error Rate for multi-class classification.

    Error Rate = mean[ y != argmax(p) ]
    """

    full_state_update = False

    def __init__(self, normalize: bool = True, reduction: str = "mean", **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize
        self.reduction = reduction
        self.add_state("all_logits", default=[], dist_reduce_fx="cat")
        self.add_state("all_labels", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if torch.isnan(logits).any() or torch.isnan(labels).any():
            raise ValueError("NaN values found in input tensors.")
        if logits.ndim != 2 or labels.ndim != 1:
            raise ValueError("Logits must be 2D and labels must be 1D.")
        self.all_logits.append(logits.float())
        self.all_labels.append(labels.long())

    def compute(self) -> torch.Tensor:
        if not self.all_logits:
            raise ValueError("No data to compute metric.")
        logits = torch.cat(self.all_logits)
        labels = torch.cat(self.all_labels)
        preds = torch.argmax(logits, dim=1)
        er = self._reduce((preds != labels).float())
        if self.normalize:
            prior = torch.bincount(
                labels.long(), minlength=logits.size(1)
            ).float() / labels.size(0)
            prior_pred = prior.argmax()
            prior_er = self._reduce((labels != prior_pred).float())
            er = er / prior_er
        return er

    def _reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return tensor.mean()
        elif self.reduction == "sum":
            return tensor.sum()
        elif self.reduction == "none":
            return tensor
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

    @classmethod
    def create_shortcut_function(cls, normalize: bool = True):
        def shortcut_function(
            logits: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            metric = cls(normalize=normalize)
            metric.update(logits, labels)
            return metric.compute().item()

        return shortcut_function


class ClassificationCrossEntropy(Metric):
    """
    Cross Entropy between predicted probabilities and true labels for multi-class classification.

    CE = -mean[ y·log(p) ]

    if normalize=True, then, this term is divided by the CE of the prior distribution.

    """

    full_state_update = False

    def __init__(
        self, normalize=True, epsilon: float = 1e-7, reduction: str = "mean", **kwargs
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.normalize = normalize
        self.reduction = reduction
        self.add_state("all_logits", default=[], dist_reduce_fx="cat")
        self.add_state("all_labels", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if torch.isnan(logits).any() or torch.isnan(labels).any():
            raise ValueError("NaN values found in input tensors.")
        if logits.ndim != 2 or labels.ndim != 1:
            raise ValueError("Logits must be 2D and labels must be 1D.")
        self.all_logits.append(logits.float())
        self.all_labels.append(labels.long())

    def compute(self) -> torch.Tensor:
        if not self.all_logits:
            raise ValueError("No data to compute metric.")
        logits = torch.cat(self.all_logits)
        labels = torch.cat(self.all_labels)
        # xx = F.cross_entropy(torch.log_softmax(logits, dim=1), labels.long(), reduction="none")
        # logq_e, indices = torch.max(torch.log_softmax(logits, dim=1), dim=1)
        # print(logq_e[~(indices == labels).bool()])
        # import pdb; pdb.set_trace()
        ce = F.cross_entropy(logits, labels.long(), reduction=self.reduction)
        if self.normalize:
            priors = torch.bincount(
                labels.long(), minlength=logits.size(1)
            ).float() / labels.size(0)
            priors = priors.unsqueeze(0).expand(logits.size(0), -1)
            prior_ce = F.cross_entropy(
                torch.log(priors), labels.long(), reduction=self.reduction
            )
            ce = ce / prior_ce
        return ce

    @classmethod
    def create_shortcut_function(cls, normalize: bool = True):
        def shortcut_function(
            logits: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            metric = cls(normalize=normalize)
            metric.update(logits, labels)
            return metric.compute().item()

        return shortcut_function


class ClassificationBrierScore(Metric):
    """
    Brier Score for multi-class classification.

    Brier Score = mean[ sum_k (p_k - y_k)^2 ]
    """

    full_state_update = False

    def __init__(self, normalize: bool = True, reduction: str = "mean", **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize
        self.reduction = reduction
        self.add_state("all_logits", default=[], dist_reduce_fx="cat")
        self.add_state("all_labels", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if torch.isnan(logits).any() or torch.isnan(labels).any():
            raise ValueError("NaN values found in input tensors.")
        if logits.ndim != 2 or labels.ndim != 1:
            raise ValueError("Logits must be 2D and labels must be 1D.")
        self.all_logits.append(logits.float())
        self.all_labels.append(labels.long())

    def compute(self) -> torch.Tensor:
        if not self.all_logits:
            raise ValueError("No data to compute metric.")
        logits = torch.cat(self.all_logits)
        labels = torch.cat(self.all_labels)
        probs = F.softmax(logits, dim=1)
        one_hot_labels = F.one_hot(labels, num_classes=logits.size(1)).float()
        brier = self._reduce(((probs - one_hot_labels) ** 2).mean(dim=1))
        if self.normalize:
            priors = torch.bincount(
                labels, minlength=logits.size(1)
            ).float() / labels.size(0)
            priors = priors.unsqueeze(0).expand(logits.size(0), -1)
            prior_brier = self._reduce(((priors - one_hot_labels) ** 2).mean(dim=1))
            brier = brier / prior_brier
        return brier

    def _reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return tensor.mean()
        elif self.reduction == "sum":
            return tensor.sum()
        elif self.reduction == "none":
            return tensor
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

    @classmethod
    def create_shortcut_function(cls, normalize: bool = True):
        def shortcut_function(
            logits: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            metric = cls(normalize=normalize)
            metric.update(logits, labels)
            return metric.compute().item()

        return shortcut_function


class ClassificationAUC(Metric):
    """Multiclass AUC metric."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("all_logits", default=[], dist_reduce_fx="cat")
        self.add_state("all_labels", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if torch.isnan(logits).any() or torch.isnan(labels).any():
            raise ValueError("NaN values found in input tensors.")
        if logits.ndim != 2 or labels.ndim != 1:
            raise ValueError("Logits must be 2D and labels must be 1D.")
        self.all_logits.append(logits.float())
        self.all_labels.append(labels.long())

    def compute(self) -> torch.Tensor:
        if not self.all_logits:
            raise ValueError("No data to compute metric.")
        probs = torch.softmax(torch.cat(self.all_logits), dim=1)
        confidences, indices = probs.max(dim=1)
        labels = torch.cat(self.all_labels)
        correctness = (indices == labels).long()
        auroc = BinaryAUROC()
        auroc.update(confidences, correctness)
        return auroc.compute()

    @classmethod
    def create_shortcut_function(cls):
        def shortcut_function(
            logits: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            metric = cls()
            metric.update(logits, labels)
            return metric.compute().item()

        return shortcut_function


class ClassificationECE(Metric):
    """Multiclass Expected Calibration Error (ECE) metric."""

    def __init__(self, n_bins: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.add_state("all_logits", default=[], dist_reduce_fx="cat")
        self.add_state("all_labels", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if torch.isnan(logits).any() or torch.isnan(labels).any():
            raise ValueError("NaN values found in input tensors.")
        if logits.ndim != 2 or labels.ndim != 1:
            raise ValueError("Logits must be 2D and labels must be 1D.")
        self.all_logits.append(logits.float())
        self.all_labels.append(labels.long())

    def compute(self) -> torch.Tensor:
        if not self.all_logits:
            raise ValueError("No data to compute metric.")
        probs = torch.softmax(torch.cat(self.all_logits), dim=1)
        labels = torch.cat(self.all_labels)
        ece = MulticlassCalibrationError(
            num_classes=probs.size(1), n_bins=self.n_bins, norm="l1"
        )
        ece.update(probs, labels.long())
        return ece.compute()

    @classmethod
    def create_shortcut_function(cls):
        def shortcut_function(
            logits: torch.Tensor, labels: torch.Tensor, nbins: int = 10
        ) -> torch.Tensor:
            metric = cls(n_bins=nbins)
            metric.update(logits, labels)
            return metric.compute().item()

        return shortcut_function


class ClassificationECUAS(Metric):
    """
    n-ECUAS (Expected Cost for Uncertainty-Augmented Systems) metric.
    """

    full_state_update = False

    def __init__(
        self,
        n: int = 0,
        normalize: bool = True,
        reduction: str = "mean",
        epsilon: float = 1e-7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n = n
        self.normalize = normalize
        self.reduction = reduction
        self.epsilon = epsilon
        
        self.add_state("all_logits", default=[], dist_reduce_fx="cat")
        self.add_state("all_labels", default=[], dist_reduce_fx="cat")

    def _cost_fn(self, logq_e: torch.Tensor, correct_indicator: torch.Tensor, K: int) -> torch.Tensor:
        u_M = torch.tensor(1 - 1 / K)
        alpha=(self.n + 1) / (u_M ** (self.n + 1))
        u_c = 1 - torch.exp(logq_e)
        log_u_c = log1mexp(logq_e - self.epsilon)
        if self.n == 0:
            return torch.where(
                condition=correct_indicator.bool(),
                input=alpha * u_c,
                other=alpha * u_c + alpha * (torch.log(u_M) - log_u_c)
            )
        elif self.n > 0:
            return torch.where(
                condition=correct_indicator.bool(),
                input=(u_c / u_M) ** (self.n + 1),
                other=(u_c / u_M) ** (self.n + 1) + (self.n + 1) / self.n * (1/u_M - (u_c/u_M) ** self.n / u_M)
            )
        else:
            raise ValueError("n must be non-negative.")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if torch.isnan(logits).any() or torch.isnan(labels).any():
            raise ValueError("NaN values found in input tensors.")
        if logits.ndim != 2 or labels.ndim != 1:
            raise ValueError("Logits must be 2D and labels must be 1D.")
        self.all_logits.append(logits.float())
        self.all_labels.append(labels.long())

    def compute(self) -> torch.Tensor:
        if not self.all_logits:
            raise ValueError("No data to compute metric.")
        logprobs = torch.log_softmax(torch.cat(self.all_logits), dim=1)
        labels = torch.cat(self.all_labels)
        logq_e, indices = torch.max(logprobs, dim=1)
        indicator = (indices == labels).float()
        K = logprobs.size(1)
        cost = self._reduce(self._cost_fn(logq_e, indicator, K))
        if self.normalize:
            priors = torch.bincount(
                labels.long(), minlength=K
            ) / labels.size(0)
            logqe_prior, indices = torch.max(torch.log(priors + self.epsilon), dim=0)
            logqe_prior = logqe_prior.expand(labels.size(0))
            prior_correct_indicator = (indices == labels).float()
            prior_cost = self._reduce(
                self._cost_fn(logqe_prior, prior_correct_indicator, K)
            )
            cost = cost / prior_cost
        return cost

    def _reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return tensor.mean()
        elif self.reduction == "sum":
            return tensor.sum()
        elif self.reduction == "none":
            return tensor
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

    @classmethod
    def create_shortcut_function(cls, normalize: bool = True):
        def shortcut_function(
            logits: torch.Tensor, labels: torch.Tensor, n: int = 0
        ) -> torch.Tensor:
            metric = cls(n=n, normalize=normalize)
            metric.update(logits, labels)
            return metric.compute().item()

        return shortcut_function

class ClassificationLogLog(Metric):
    """
    n-ECUAS (Expected Cost for Uncertainty-Augmented Systems) metric.
    """

    full_state_update = False

    def __init__(
        self,
        normalize: bool = True,
        reduction: str = "mean",
        epsilon: float = 1e-7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.normalize = normalize
        self.reduction = reduction
        self.epsilon = epsilon
        self.add_state("all_logits", default=[], dist_reduce_fx="cat")
        self.add_state("all_labels", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if torch.isnan(logits).any() or torch.isnan(labels).any():
            raise ValueError("NaN values found in input tensors.")
        if logits.ndim != 2 or labels.ndim != 1:
            raise ValueError("Logits must be 2D and labels must be 1D.")
        self.all_logits.append(logits.float())
        self.all_labels.append(labels.long())

    def compute(self) -> torch.Tensor:
        if not self.all_logits:
            raise ValueError("No data to compute metric.")
        logprobs = torch.log_softmax(torch.cat(self.all_logits), dim=1)
        labels = torch.cat(self.all_labels)
        N, K = logprobs.size()
        entropy = -torch.sum(torch.exp(logprobs) * logprobs, dim=1)
        cost = entropy - logprobs[torch.arange(N), labels.long()] * torch.log(torch.log(torch.tensor(K)) / entropy)
        cost = self._reduce(cost)
        if self.normalize:
            priors = torch.bincount(
                labels.long(), minlength=K
            ) / N
            prior_entropy = -torch.sum(priors * torch.log(priors))
            prior_cost = prior_entropy - torch.log(priors[labels.long()]) * torch.log(torch.log(torch.tensor(K)) / prior_entropy)
            prior_cost = self._reduce(prior_cost)
            cost = cost / prior_cost
        return cost

    def _reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return tensor.mean()
        elif self.reduction == "sum":
            return tensor.sum()
        elif self.reduction == "none":
            return tensor
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

    @classmethod
    def create_shortcut_function(cls, normalize: bool = True):
        def shortcut_function(
            logits: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            metric = cls(normalize=normalize)
            metric.update(logits, labels)
            return metric.compute().item()

        return shortcut_function


class ClassificationGammaECUAS(Metric):
    """
    Gamma-ECUAS (Expected Cost for Uncertainty-Augmented Systems) metric.

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
        normalize: bool = True,
        reduction: str = "mean",
        epsilon: float = 1e-7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.normalize = normalize
        self.reduction = reduction
        self.epsilon = epsilon
        self.add_state("all_logits", default=[], dist_reduce_fx="cat")
        self.add_state("all_labels", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if torch.isnan(logits).any() or torch.isnan(labels).any():
            raise ValueError("NaN values found in input tensors.")
        if logits.ndim != 2 or labels.ndim != 1:
            raise ValueError("Logits must be 2D and labels must be 1D.")
        self.all_logits.append(logits.float())
        self.all_labels.append(labels.float())

    def compute(self) -> torch.Tensor:
        if not self.all_logits:
            raise ValueError("No data to compute metric.")

        logits = torch.cat(self.all_logits)
        confidences, indices = torch.softmax(logits, dim=1).max(dim=1)
        labels = torch.cat(self.all_labels)
        correctness = (indices == labels).float()

        cost = self._compute_cost(confidences, correctness)
        if self.normalize:
            prior = torch.bincount(
                labels.long(), minlength=logits.size(1)
            ).float() / labels.size(0)
            prior_max, prior_argmax = prior.max(dim=0)
            prior_confidences = torch.ones(labels.size(0)) * prior_max
            prior_correctness = (prior_argmax == labels).float()
            prior_cost = self._compute_cost(prior_confidences, prior_correctness)
            cost = cost / prior_cost

        return cost

    def _compute_cost(
        self, confidences: torch.Tensor, correctness: torch.Tensor
    ) -> torch.Tensor:
        # Score: estimated error probability
        s = 1.0 - confidences

        # Decision: abstain if s > threshold, answer otherwise
        abstain_mask = s < self.gamma
        answer_mask = ~abstain_mask

        # Create vector of cost
        total_cost = torch.zeros_like(confidences)

        # Cost when abstaining: γ
        total_cost[abstain_mask] = self.gamma

        # Cost when answering: C̃  where C̃ = 1 - correctness (0-1 loss)
        total_cost[answer_mask] = (
            1.0 - correctness[answer_mask]
        )  # 0 if correct, 1 if incorrect

        return self._reduce(total_cost)

    def _reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return tensor.mean()
        elif self.reduction == "sum":
            return tensor.sum()
        elif self.reduction == "none":
            return tensor
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

    @classmethod
    def create_shortcut_function(cls, normalize: bool = True):
        def shortcut_function(
            logits: torch.Tensor, labels: torch.Tensor, gamma: float = 0.5
        ) -> torch.Tensor:
            metric = cls(gamma=gamma, normalize=normalize)
            metric.update(logits, labels)
            return metric.compute().item()

        return shortcut_function


class ClassificationAURC(_AURC):
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        probs = torch.softmax(logits, dim=1)
        return super().update(probs, labels)

    @classmethod
    def create_shortcut_function(cls):
        def shortcut_function(
            logits: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            metric = cls()
            metric.update(logits, labels)
            return metric.compute().item()

        return shortcut_function


class ClassificationFPR95(_FPRx):
    def __init__(self, **kwargs):
        super().__init__(recall_level=0.95, pos_label=1, **kwargs)

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        confidences = None
        pos_labels = None
        return super().update(confidences, pos_labels)

    @classmethod
    def create_shortcut_function(cls):
        def shortcut_function(
            logits: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            metric = cls()
            metric.update(logits, labels)
            return metric.compute().item()

        return shortcut_function
