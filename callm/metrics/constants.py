from collections import OrderedDict

from .confidences import (
    ExpectedCalibrationError,
    ConfidenceAUCScore,
    ConfidenceBrierScore,
    ConfidencenCCAS,
    ConfidenceCrossEntropy,
    ConfidenceErrorRate,
    ConfidencegammaCCAS,
)
from .classification import (
    ClassificationErrorRate,
    ClassificationAUC,
    ClassificationBrierScore,
    ClassificationCrossEntropy,
    ClassificationECE,
    ClassificationnCCAS,
    ClassificationGammaCCAS,
)

METRICS = OrderedDict([
    ## CONFIDENCE METRICS
    ("conf_auc", {
        "full_name": "Confidence AUC Score",
        "function": ConfidenceAUCScore.create_shortcut_function(),
        "higher_is_better": True,
        "display": "AUC",
    }),
    ("conf_ece", {
        "full_name": "Expected Calibration Error",
        "function": ExpectedCalibrationError.create_shortcut_function(),
        "higher_is_better": False,
        "display": None,
    }),
    ("conf_brier", {
        "full_name": "Confidence Brier Score",
        "function": ConfidenceBrierScore.create_shortcut_function(),
        "higher_is_better": False,
        "display": "BS$^*$",
    }),
    ("conf_cross_entropy", {
        "full_name": "Confidence Cross-Entropy",
        "function": ConfidenceCrossEntropy.create_shortcut_function(),
        "higher_is_better": False,
        "display": "CE$^*$",
    }),
    ("conf_error_rate", {
        "full_name": "Confidence Error Rate",
        "function": ConfidenceErrorRate.create_shortcut_function(),
        "higher_is_better": False,
        "display": "ER$^*$",
    }),
    ("conf_n-ccas", {
        "full_name": "Confidence n-CCAS",
        "function": ConfidencenCCAS.create_shortcut_function(),
        "higher_is_better": False,
        "display": None,
    }),
    ("conf_gamma-ccas", {
        "full_name": "Confidence γ-CCAS",
        "function": ConfidencegammaCCAS.create_shortcut_function(),
        "higher_is_better": False,
        "display": None,
    }),
    ## CLASSIFICATION METRICS
    ("cls_error_rate", {
        "full_name": "Classification Error Rate",
        "function": ClassificationErrorRate.create_shortcut_function(normalize=False),
        "higher_is_better": False,
        "display": "ER",
    }),
    ("cls_ner", {
        "full_name": "Classification Normalized Error Rate",
        "function": ClassificationErrorRate.create_shortcut_function(normalize=True),
        "higher_is_better": False,
        "display": "NER",
    }),
    ("cls_auc", {
        "full_name": "Classification AUC",
        "function": ClassificationAUC.create_shortcut_function(),
        "higher_is_better": True,
        "display": "AUC",
    }),
    ("cls_brier", {
        "full_name": "Classification Brier Score",
        "function": ClassificationBrierScore.create_shortcut_function(normalize=False),
        "higher_is_better": False,
        "display": "BS",
    }),
    ("cls_nbs", {
        "full_name": "Classification Normalized Brier Score",
        "function": ClassificationBrierScore.create_shortcut_function(normalize=True),
        "higher_is_better": False,
        "display": "NBS",
    }),
    ("cls_cross_entropy", {
        "full_name": "Classification Cross-Entropy",
        "function": ClassificationCrossEntropy.create_shortcut_function(normalize=False),
        "higher_is_better": False,
        "display": "CE",
    }),
    ("cls_nce", {
        "full_name": "Classification Normalized Cross-Entropy",
        "function": ClassificationCrossEntropy.create_shortcut_function(normalize=True),
        "higher_is_better": False,
        "display": "NCE",
    }),
    ("cls_ece", {
        "full_name": "Classification ECE",
        "function": ClassificationECE.create_shortcut_function(),
        "higher_is_better": False,
        "display": None,
    }),
    ("cls_n-ccas", {
        "full_name": "Classification n-CCAS",
        "function": ClassificationnCCAS.create_shortcut_function(normalize=False),
        "higher_is_better": False,
        "display": None,
    }),
    ("cls_norm_n-ccas", {
        "full_name": "Classification Normalized n-CCAS",
        "function": ClassificationnCCAS.create_shortcut_function(normalize=True),
        "higher_is_better": False,
        "display": None,
    }),
    ("cls_gamma-ccas", {
        "full_name": "Classification γ-CCAS",
        "function": ClassificationGammaCCAS.create_shortcut_function(normalize=False),
        "higher_is_better": False,
        "display": None,
    }),
    ("cls_norm_gamma-ccas", {
        "full_name": "Classification Normalized γ-CCAS",
        "function": ClassificationGammaCCAS.create_shortcut_function(normalize=True),
        "higher_is_better": False,
        "display": None,
    }),
])