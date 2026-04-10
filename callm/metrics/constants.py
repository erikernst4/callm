from collections import OrderedDict

from .confidences import (
    ExpectedCalibrationError,
    ConfidenceAUCScore,
    ConfidenceBrierScore,
    ConfidenceCnCAG,
    ConfidenceCrossEntropy,
    ConfidenceErrorRate,
    ConfidenceGammaCCAG,
)
from .classification import (
    ClassificationErrorRate,
    ClassificationAUROC,
    ClassificationBrierScore,
    ClassificationCrossEntropy,
    ClassificationECE,
    ClassificationCnCAG,
    ClassificationGammaCCAG,
)

CONFIDENCE_METRICS = OrderedDict([
    ("ece", {
        "full_name": "Expected Calibration Error",
        "function": ExpectedCalibrationError,
        "higher_is_better": False,
        "display": "ECE",
    }),
    ("brier", {
        "full_name": "Confidence Brier Score",
        "function": ConfidenceBrierScore,
        "higher_is_better": False,
        "display": "BS$^*$",
    }),
    ("cross_entropy", {
        "full_name": "Confidence Cross-Entropy",
        "function": ConfidenceCrossEntropy,
        "higher_is_better": False,
        "display": "CE$^*$",
    }),
    ("auc", {
        "full_name": "Confidence AUC Score",
        "function": ConfidenceAUCScore,
        "higher_is_better": True,
        "display": "AUC",
    }),
    ("error_rate", {
        "full_name": "Confidence Error Rate",
        "function": ConfidenceErrorRate,
        "higher_is_better": False,
        "display": "ER$^*$",
    }),
    ("cncag", {
        "full_name": "Confidence CnCAG",
        "function": ConfidenceCnCAG,
        "higher_is_better": False,
        "display": "CnCAG(n={n})$^*$",
    }),
    ("gamma_ccag", {
        "full_name": "Confidence GammaCCAG",
        "function": ConfidenceGammaCCAG,
        "higher_is_better": False,
        "display": "γ-CCAG(γ={gamma})$^*$",
    }),
])

CLASSIFICATION_METRICS = OrderedDict([
    ("error_rate", {
        "full_name": "Classification Error Rate",
        "function": ClassificationErrorRate,
        "higher_is_better": False,
        "display": "ER",
    }),
    ("auroc", {
        "full_name": "Classification AUROC",
        "function": ClassificationAUROC,
        "higher_is_better": True,
        "display": "AUC",
    }),
    ("brier", {
        "full_name": "Classification Brier Score",
        "function": ClassificationBrierScore,
        "higher_is_better": False,
        "display": "BS",
    }),
    ("cross_entropy", {
        "full_name": "Classification Cross-Entropy",
        "function": ClassificationCrossEntropy,
        "higher_is_better": False,
        "display": "CE",
    }),
    ("ece", {
        "full_name": "Classification ECE",
        "function": ClassificationECE,
        "higher_is_better": False,
        "display": "ECE",
    }),
    ("cncag", {
        "full_name": "Classification CnCAG",
        "function": ClassificationCnCAG,
        "higher_is_better": False,
        "display": "CnCAG(n={n})",
    }),
    ("gamma_ccag", {
        "full_name": "Classification GammaCCAG",
        "function": ClassificationGammaCCAG,
        "higher_is_better": False,
        "display": "γ-CCAG(γ={gamma})",
    }),
])