


import torch
from torchmetrics import Metric
import torch.nn.functional as F
from torchmetrics.classification import AUROC



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
        self.add_state("sum_ce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        if self.normalize:
            self.add_state("labels_bincounts", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if torch.isnan(logits).any() or torch.isnan(labels).any():
            raise ValueError("NaN values found in input tensors.")
        if logits.ndim() != 2 or labels.ndim() != 1:
            raise ValueError("Logits must be 2D and labels must be 1D.")
        ce = F.cross_entropy(logits, labels, reduction="sum")
        self.sum_ce += ce
        if self.normalize:
            bincounts = torch.bincount(labels, minlength=logits.size(1)).float()
            self.labels_bincounts += bincounts
        self.count += logits.size(0)

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        ce = self.sum_ce / self.count
        if self.normalize:
            prior = self.labels_bincounts / self.count
            prior_ce = -torch.sum(prior * torch.log(prior + self.epsilon))
            ce = ce / prior_ce
        return ce


class ClassificationBrierScore(Metric):
    """
    Brier Score for multi-class classification.

    Brier Score = mean[ sum_k (p_k - y_k)^2 ]
    """

    full_state_update = False

    def __init__(self, num_classes: int, normalize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.normalize = normalize
        self.add_state("sum_brier", default=torch.tensor(0.0), dist_reduce_fx="sum")
        if self.normalize:
            self.add_state("labels", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if torch.isnan(logits).any() or torch.isnan(labels).any():
            raise ValueError("NaN values found in input tensors.")
        if logits.ndim() != 2 or labels.ndim() != 1:
            raise ValueError("Logits must be 2D and labels must be 1D.")
        probs = F.softmax(logits, dim=1)
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()
        brier = ((probs - one_hot_labels) ** 2).sum()
        self.sum_brier += brier
        if self.normalize:
            self.labels = torch.cat([self.labels, labels.float()])
        self.count += logits.numel()

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        brier = self.sum_brier  / self.count
        if self.normalize:
            labels_bincounts = torch.bincount(self.labels.long(), minlength=self.num_classes).float()
            prior = labels_bincounts / self.count
            one_hot_labels = F.one_hot(self.labels, num_classes=self.num_classes).float()
            prior_brier = ((prior - one_hot_labels) ** 2).sum() / self.count
            brier = brier / prior_brier
        return brier
    

class ClassificationAUROC(AUROC):
    """Multiclass AUROC metric."""

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(task="multiclass", num_classes=num_classes, **kwargs)

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if torch.isnan(logits).any() or torch.isnan(labels).any():
            raise ValueError("NaN values found in input tensors.")
        preds = torch.softmax(logits, dim=1)
        super().update(preds, labels.long())


class ClassificationErrorRate(Metric):
    """
    Error Rate for multi-class classification.

    Error Rate = mean[ y != argmax(p) ]
    """

    full_state_update = False

    def __init__(self, num_classes: int, normalize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.normalize = normalize
        self.add_state("num_errors", default=torch.tensor(0), dist_reduce_fx="sum")
        if self.normalize:
            self.add_state("labels", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if torch.isnan(logits).any() or torch.isnan(labels).any():
            raise ValueError("NaN values found in input tensors.")
        if logits.ndim() != 2 or labels.ndim() != 1:
            raise ValueError("Logits must be 2D and labels must be 1D.")
        preds = torch.argmax(logits, dim=1)
        errors = (preds != labels).sum()
        self.num_errors += errors
        if self.normalize:
            self.labels = torch.cat([self.labels, labels.float()])
        self.count += logits.size(0)

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        error_rate = self.num_errors / self.count
        if self.normalize:
            prior_pred = torch.bincount(self.labels.long(), minlength=self.num_classes).argmax()
            prior_error_rate = (self.labels != prior_pred).float().mean()
            error_rate = error_rate / prior_error_rate
        return error_rate
    


class ClassificationCnCAG(Metric):
    """
    CnCAG (Confidence Cost Abstention Game) metric.

    $$
    \\begin{align}
    C_n^*(y_k,\\mathbf{q}) = 
        \\begin{cases}
            (1-q_e)^{n+1} + \\frac{(n+1)}{n}(1-(1-q_e)^n) I(k \\neq\ e) & \\text{if } n \in \mathbb{N} \\
            1-q_e - I(k \\neq e) \log(1-q_e) & \\text{if } n = 0
        \end{cases}
    \end{align}
    $$

    where q_e = max(q) and I(k \\neq e) is the indicator function.
    """

    full_state_update = False

    def __init__(self, n: int = 0, normalize: bool = True, epsilon: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.normalize = normalize
        self.epsilon = epsilon
        if n == 0:
            self.cost_fun = lambda q_e, correct_indicator: 1 - q_e - (1 - correct_indicator) * torch.log(1 - q_e)
        elif n > 0:
            self.cost_fun = lambda q_e, correct_indicator: (1 - q_e)**(n+1) + (n+1)/n * (1 - (1 - q_e)**n) * (1 - correct_indicator)
        else:
            raise ValueError("n must be non-negative.")
        self.add_state("sum_cost", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("labels", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if torch.isnan(logits).any() or torch.isnan(labels).any():
            raise ValueError("NaN values found in input tensors.")
        q_e = torch.max(logits, dim=1).values.float().clamp(self.epsilon, 1 - self.epsilon)
        indicator = (torch.argmax(logits, dim=1) == labels).float()
        cost = self.cost_fun(q_e, indicator)
        self.sum_cost += cost.sum()
        self.labels = torch.cat([self.labels, labels.float()])
        self.count += logits.size(0)

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        cost = self.sum_cost / self.count
        if self.normalize:
            values, indices = torch.max(torch.bincount(self.labels.long(), minlength=self.num_classes) / self.count)
            prior_correct_indicator = (indices == self.labels).float()
            prior_cost = self.cost_fun(values, prior_correct_indicator)
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
        self.all_logits.append(logits.float())
        self.all_labels.append(labels.float())

    def compute(self) -> torch.Tensor:
        if not self.all_logits:
            return torch.tensor(float("nan"))

        logits = torch.stack(self.all_logits)
        confidences, indices = torch.softmax(logits, dim=1).max(dim=1)
        labels = torch.stack(self.all_labels)
        correctness = (indices == labels).float()

        if len(logits) == 0:
            return torch.tensor(float("nan"))
        
        cost = self._compute_cost(confidences, correctness)
        if self.normalize:
            bincounts = torch.bincount(labels.long(), minlength=logits.size(1)).float()
            prior = bincounts / labels.numel()
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
