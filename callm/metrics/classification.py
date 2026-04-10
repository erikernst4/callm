


import torch
from torchmetrics import Metric
import torch.nn.functional as F
from torchmetrics.classification import AUROC, MulticlassCalibrationError

class ClassificationErrorRate(Metric):
    """
    Error Rate for multi-class classification.

    Error Rate = mean[ y != argmax(p) ]
    """

    full_state_update = False

    def __init__(self, normalize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize
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
        er = (preds != labels).float().mean()
        if self.normalize:
            prior = torch.bincount(labels.long(), minlength=logits.size(1)).float() / labels.size(0)
            prior_pred = prior.argmax()
            prior_er = (labels != prior_pred).float().mean()
            er = er / prior_er
        return er


class ClassificationCrossEntropy(Metric):
    """
    Cross Entropy between predicted probabilities and true labels for multi-class classification.

    CE = -mean[ y·log(p) ]

    if normalize=True, then, this term is divided by the CE of the prior distribution.

    """

    full_state_update = False

    def __init__(self, normalize=True, epsilon: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.normalize = normalize
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
        ce = F.cross_entropy(logits, labels.long(), reduction="mean")
        if self.normalize:
            priors = torch.bincount(labels.long(), minlength=logits.size(1)).float() / labels.size(0)
            priors = priors.unsqueeze(0).expand(logits.size(0), -1)
            prior_ce = F.cross_entropy(torch.log(priors), labels.long(), reduction="mean")
            ce = ce / prior_ce
        return ce


class ClassificationBrierScore(Metric):
    """
    Brier Score for multi-class classification.

    Brier Score = mean[ sum_k (p_k - y_k)^2 ]
    """

    full_state_update = False

    def __init__(self, normalize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize
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
        brier = ((probs - one_hot_labels) ** 2).sum() / logits.numel()
        if self.normalize:
            priors = torch.bincount(labels, minlength=logits.size(1)).float() / labels.size(0)
            priors = priors.unsqueeze(0).expand(logits.size(0), -1)
            prior_brier = ((priors - one_hot_labels) ** 2).sum() / logits.numel()
            brier = brier / prior_brier
        return brier
    

class ClassificationAUROC(Metric):
    """Multiclass AUROC metric."""

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
        labels = torch.cat(self.all_labels)
        auroc = AUROC(task="multiclass", num_classes=probs.size(1))
        auroc.update(probs, labels.long())
        return auroc.compute()

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
        ece = MulticlassCalibrationError(num_classes=probs.size(1), n_bins=self.n_bins, norm="l1")
        ece.update(probs, labels.long())
        return ece.compute()
        

class ClassificationCnCAG(Metric):
    """
    CnCAG (Confidence Cost Abstention Game) metric.
    """

    full_state_update = False

    def __init__(self, n: int = 0, normalize: bool = True, epsilon: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.normalize = normalize
        self.epsilon = epsilon
        if n == 0:
            self.cost_fun = lambda logq_e, correct_indicator: 1 - torch.exp(logq_e) - (1 - correct_indicator) * logq_e
        elif n > 0:
            self.cost_fun = lambda logq_e, correct_indicator: (1 - torch.exp(logq_e))**(n+1) + (n+1)/n * (1 - (1 - torch.exp(logq_e))**n) * (1 - correct_indicator)
        else:
            raise ValueError("n must be non-negative.")
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
        logq_e, indices = torch.max(logprobs, dim=1)
        indicator = (indices == labels).float()
        cost = self.cost_fun(logq_e, indicator).sum() / labels.size(0)
        if self.normalize:
            priors = torch.bincount(labels.long(), minlength=logprobs.size(1)) / labels.size(0)
            logqe_prior, indices = torch.max(torch.log(priors + self.epsilon), dim=0)
            logqe_prior = logqe_prior.expand(labels.size(0))
            prior_correct_indicator = (indices == labels).float()
            prior_cost = self.cost_fun(logqe_prior, prior_correct_indicator).sum() / labels.size(0)
            cost = cost / prior_cost
        return cost


class ClassificationGammaCCAG(Metric):
    """
    Gamma-CnCAG (Confidence Cost Abstention Game) metric.

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
        epsilon: float = 1e-7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.normalize = normalize
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
            prior = torch.bincount(labels.long(), minlength=logits.size(1)).float() / labels.size(0)
            prior_max, prior_argmax = prior.max(dim=0)
            prior_confidences = torch.ones(labels.size(0)) * prior_max
            prior_correctness = (prior_argmax == labels).float()
            prior_cost = self._compute_cost(prior_confidences, prior_correctness)
            cost = cost / prior_cost
    
        return cost
            
    def _compute_cost(self, confidences: torch.Tensor, correctness: torch.Tensor) -> torch.Tensor:

        # Score: estimated error probability
        s = 1.0 - confidences

        # Decision: abstain if s > threshold, answer otherwise
        abstain_mask = s < self.gamma
        answer_mask = ~abstain_mask

        # Cost when answering: C̃  where C̃ = 1 - correctness (0-1 loss)
        base_cost = 1.0 - correctness  # 0 if correct, 1 if incorrect
        answer_costs = base_cost[answer_mask]

        # Cost when abstaining: γ
        n_abstain = abstain_mask.sum()
        abstain_costs = self.gamma * n_abstain.float()

        # Expected cost = mean over all samples
        total_cost = answer_costs.sum() + abstain_costs
        
        return total_cost / len(confidences)
