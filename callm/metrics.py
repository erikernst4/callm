"""
Calibration metrics for evaluating confidence predictions.

Implements ECE, AUC, Brier Score, and Cross Entropy metrics.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import List, Union


def expected_calibration_error(
    confidences: Union[List[float], np.ndarray],
    correctness: Union[List[bool], np.ndarray],
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and actual accuracy
    across bins.
    
    Args:
        confidences: Array of confidence scores (0.0 to 1.0)
        correctness: Array of boolean correctness values
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value (lower is better, 0 is perfect calibration), or np.nan if undefined
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=float)
    
    if len(confidences) == 0:
        return np.nan
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            bin_confidences = confidences[bin_mask]
            bin_correctness = correctness[bin_mask]
            
            avg_confidence = np.mean(bin_confidences)
            avg_accuracy = np.mean(bin_correctness)
            bin_weight = np.sum(bin_mask) / len(confidences)
            
            ece += bin_weight * np.abs(avg_confidence - avg_accuracy)
    
    return float(ece)


def brier_score(
    confidences: Union[List[float], np.ndarray],
    correctness: Union[List[bool], np.ndarray]
) -> float:
    """
    Calculate Brier Score (BS).
    
    BS measures the mean squared difference between predicted probabilities
    and actual outcomes.
    
    Args:
        confidences: Array of confidence scores (0.0 to 1.0)
        correctness: Array of boolean correctness values
        
    Returns:
        Brier Score (lower is better, 0 is perfect), or np.nan if undefined
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=float)
    
    if len(confidences) == 0:
        return np.nan
    
    return float(np.mean((confidences - correctness) ** 2))


def cross_entropy(
    confidences: Union[List[float], np.ndarray],
    correctness: Union[List[bool], np.ndarray],
    epsilon: float = 1e-10
) -> float:
    """
    Calculate Cross Entropy (CE).
    
    CE measures the negative log-likelihood of the predictions.
    
    Args:
        confidences: Array of confidence scores (0.0 to 1.0)
        correctness: Array of boolean correctness values
        epsilon: Small value to avoid log(0)
        
    Returns:
        Cross Entropy (lower is better), or np.nan if undefined
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=float)
    
    if len(confidences) == 0:
        return np.nan
    
    # Clip confidences to avoid log(0)
    confidences = np.clip(confidences, epsilon, 1 - epsilon)
    
    # Calculate cross entropy
    ce = -np.mean(
        correctness * np.log(confidences) + 
        (1 - correctness) * np.log(1 - confidences)
    )
    
    return float(ce)


def auc_score(
    confidences: Union[List[float], np.ndarray],
    correctness: Union[List[bool], np.ndarray]
) -> float:
    """
    Calculate Area Under ROC Curve (AUC).
    
    AUC measures the model's ability to rank correct predictions higher
    than incorrect ones.
    
    Args:
        confidences: Array of confidence scores (0.0 to 1.0)
        correctness: Array of boolean correctness values
        
    Returns:
        AUC score (higher is better, 0.5 is random, 1.0 is perfect), or np.nan if undefined
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=int)
    
    if len(confidences) == 0 or len(np.unique(correctness)) < 2:
        # Cannot compute AUC with empty data or only one class
        return np.nan
    
    return float(roc_auc_score(correctness, confidences))
