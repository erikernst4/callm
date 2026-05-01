from collections import OrderedDict
from functools import partial

from .confidences import (
    FPR95,
    ConfidenceAURC,
    ExpectedCalibrationError,
    ConfidenceAUCScore,
    ConfidenceBrierScore,
    ConfidenceECUAS,
    ConfidenceCrossEntropy,
    ConfidenceErrorRate,
    ConfidenceGammaCCAS,
)
from .classification import (
    ClassificationAURC,
    ClassificationErrorRate,
    ClassificationAUC,
    ClassificationBrierScore,
    ClassificationCrossEntropy,
    ClassificationECE,
    ClassificationFPR95,
    ClassificationECUAS,
    ClassificationLogLog,
    ClassificationGammaCCAS,
)

METRICS = OrderedDict(
    [
        ## CONFIDENCE METRICS
        (
            "conf_auc",
            {
                "full_name": "Confidence AUC Score",
                "function": ConfidenceAUCScore.create_shortcut_function(),
                "cls": ConfidenceAUCScore,
                "higher_is_better": True,
                "display": "AUC",
            },
        ),
        (
            "conf_ece",
            {
                "full_name": "Expected Calibration Error",
                "function": ExpectedCalibrationError.create_shortcut_function(),
                "cls": ExpectedCalibrationError,
                "higher_is_better": False,
                "display": None,
            },
        ),
        (
            "conf_brier",
            {
                "full_name": "Confidence Brier Score",
                "function": ConfidenceBrierScore.create_shortcut_function(),
                "cls": ConfidenceBrierScore,
                "higher_is_better": False,
                "display": "BS$^*$",
            },
        ),
        (
            "conf_cross_entropy",
            {
                "full_name": "Confidence Cross-Entropy",
                "function": ConfidenceCrossEntropy.create_shortcut_function(),
                "cls": ConfidenceCrossEntropy,
                "higher_is_better": False,
                "display": "CE$^*$",
            },
        ),
        (
            "conf_error_rate",
            {
                "full_name": "Confidence Error Rate",
                "function": ConfidenceErrorRate.create_shortcut_function(),
                "cls": ConfidenceErrorRate,
                "higher_is_better": False,
                "display": "ER$^*$",
            },
        ),
        (
            "conf_n-ccas",
            {
                "full_name": "Confidence n-CCAS",
                "function": ConfidenceECUAS.create_shortcut_function(),
                "cls": ConfidenceECUAS,
                "higher_is_better": False,
                "display": None,
            },
        ),
        (
            "conf_gamma-ccas",
            {
                "full_name": "Confidence γ-CCAS",
                "function": ConfidenceGammaCCAS.create_shortcut_function(),
                "cls": ConfidenceGammaCCAS,
                "higher_is_better": False,
                "display": None,
            },
        ),
        (
            "conf_fpr95",
            {
                "full_name": "Confidence FPR95",
                "function": FPR95.create_shortcut_function(),
                "cls": FPR95,
                "higher_is_better": False,
                "display": "FPR95$^*$",
            },
        ),
        (
            "conf_aurc",
            {
                "full_name": "Confidence AURC",
                "function": ConfidenceAURC.create_shortcut_function(),
                "cls": ConfidenceAURC,
                "higher_is_better": False,
                "display": "AURC$^*$",
            },
        ),
        ## CLASSIFICATION METRICS
        (
            "cls_error_rate",
            {
                "full_name": "Classification Error Rate",
                "function": ClassificationErrorRate.create_shortcut_function(
                    normalize=False
                ),
                "cls": partial(ClassificationErrorRate, normalize=False),
                "higher_is_better": False,
                "display": "ER",
            },
        ),
        (
            "cls_ner",
            {
                "full_name": "Classification Normalized Error Rate",
                "function": ClassificationErrorRate.create_shortcut_function(
                    normalize=True
                ),
                "cls": partial(ClassificationErrorRate, normalize=True),
                "higher_is_better": False,
                "display": "NER",
            },
        ),
        (
            "cls_auc",
            {
                "full_name": "Classification AUC",
                "function": ClassificationAUC.create_shortcut_function(),
                "cls": ClassificationAUC,
                "higher_is_better": True,
                "display": "AUC",
            },
        ),
        (
            "cls_brier",
            {
                "full_name": "Classification Brier Score",
                "function": ClassificationBrierScore.create_shortcut_function(
                    normalize=False
                ),
                "cls": partial(ClassificationBrierScore, normalize=False),
                "higher_is_better": False,
                "display": "BS",
            },
        ),
        (
            "cls_nbs",
            {
                "full_name": "Classification Normalized Brier Score",
                "function": ClassificationBrierScore.create_shortcut_function(
                    normalize=True
                ),
                "cls": partial(ClassificationBrierScore, normalize=True),
                "higher_is_better": False,
                "display": "NBS",
            },
        ),
        (
            "cls_cross_entropy",
            {
                "full_name": "Classification Cross-Entropy",
                "function": ClassificationCrossEntropy.create_shortcut_function(
                    normalize=False
                ),
                "cls": partial(ClassificationCrossEntropy, normalize=False),
                "higher_is_better": False,
                "display": "CE",
            },
        ),
        (
            "cls_nce",
            {
                "full_name": "Classification Normalized Cross-Entropy",
                "function": ClassificationCrossEntropy.create_shortcut_function(
                    normalize=True
                ),
                "cls": partial(ClassificationCrossEntropy, normalize=True),
                "higher_is_better": False,
                "display": "NCE",
            },
        ),
        (
            "cls_ece",
            {
                "full_name": "Classification ECE",
                "function": ClassificationECE.create_shortcut_function(),
                "cls": ClassificationECE,
                "higher_is_better": False,
                "display": None,
            },
        ),
        (
            "cls_n-ccas",
            {
                "full_name": "Classification n-CCAS",
                "function": ClassificationECUAS.create_shortcut_function(
                    normalize=False
                ),
                "cls": partial(ClassificationECUAS, normalize=False),
                "higher_is_better": False,
                "display": None,
            },
        ),
        (
            "cls_norm_n-ccas",
            {
                "full_name": "Classification Normalized n-CCAS",
                "function": ClassificationECUAS.create_shortcut_function(
                    normalize=True
                ),
                "cls": partial(ClassificationECUAS, normalize=True),
                "higher_is_better": False,
                "display": None,
            },
        ),
        (
            "cls_loglog",
            {
                "full_name": "Classification LogLog",
                "function": ClassificationLogLog.create_shortcut_function(
                    normalize=False
                ),
                "cls": partial(ClassificationLogLog, normalize=False),
                "higher_is_better": False,
                "display": None,
            },
        ),
        (
            "cls_norm_loglog",
            {
                "full_name": "Classification Normalized LogLog",
                "function": ClassificationLogLog.create_shortcut_function(
                    normalize=True
                ),
                "cls": partial(ClassificationLogLog, normalize=True),
                "higher_is_better": False,
                "display": None,
            },
        ),
        (
            "cls_gamma-ccas",
            {
                "full_name": "Classification γ-CCAS",
                "function": ClassificationGammaCCAS.create_shortcut_function(
                    normalize=False
                ),
                "cls": partial(ClassificationGammaCCAS, normalize=False),
                "higher_is_better": False,
                "display": None,
            },
        ),
        (
            "cls_norm_gamma-ccas",
            {
                "full_name": "Classification Normalized γ-CCAS",
                "function": ClassificationGammaCCAS.create_shortcut_function(
                    normalize=True
                ),
                "cls": partial(ClassificationGammaCCAS, normalize=True),
                "higher_is_better": False,
                "display": None,
            },
        ),
        (
            "cls_aurc",
            {
                "full_name": "Classification AURC",
                "function": ClassificationAURC.create_shortcut_function(),
                "cls": ClassificationAURC,
                "higher_is_better": False,
                "display": "AURC",
            },
        ),
        (
            "cls_fpr95",
            {
                "full_name": "Classification FPR95",
                "function": ClassificationFPR95.create_shortcut_function(),
                "cls": ClassificationFPR95,
                "higher_is_better": False,
                "display": "FPR95$^*$",
            },
        ),
    ]
)
