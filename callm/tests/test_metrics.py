"""Unit tests for calibration metrics."""

import numpy as np
from callm.metrics import (
    expected_calibration_error,
    brier_score,
    cross_entropy,
    auc_score,
)


class TestExpectedCalibrationError:
    """Tests for ECE metric."""

    def test_perfect_calibration(self):
        """Test ECE with perfectly calibrated predictions."""
        # All predictions with 0.9 confidence should have 90% accuracy
        confidences = [0.9] * 100
        correctness = [True] * 90 + [False] * 10
        ece = expected_calibration_error(confidences, correctness, n_bins=10)
        assert ece < 0.05  # Should be very small

    def test_worst_calibration(self):
        """Test ECE with worst case calibration."""
        # High confidence but all wrong
        confidences = [0.9] * 100
        correctness = [False] * 100
        ece = expected_calibration_error(confidences, correctness, n_bins=10)
        assert ece > 0.8  # Should be high

    def test_empty_input(self):
        """Test ECE with empty arrays."""
        ece = expected_calibration_error([], [])
        assert np.isnan(ece)

    def test_single_bin(self):
        """Test ECE with single bin."""
        confidences = [0.5, 0.5, 0.5, 0.5]
        correctness = [True, True, False, False]
        ece = expected_calibration_error(confidences, correctness, n_bins=1)
        assert ece == 0.0  # Perfect calibration in single bin


class TestBrierScore:
    """Tests for Brier Score metric."""

    def test_perfect_predictions(self):
        """Test BS with perfect predictions."""
        confidences = [1.0, 1.0, 0.0, 0.0]
        correctness = [True, True, False, False]
        bs = brier_score(confidences, correctness)
        assert bs == 0.0

    def test_worst_predictions(self):
        """Test BS with worst predictions."""
        confidences = [0.0, 0.0, 1.0, 1.0]
        correctness = [True, True, False, False]
        bs = brier_score(confidences, correctness)
        assert bs == 1.0

    def test_uniform_predictions(self):
        """Test BS with uniform 0.5 predictions."""
        confidences = [0.5] * 10
        correctness = [True] * 5 + [False] * 5
        bs = brier_score(confidences, correctness)
        assert bs == 0.25  # (0.5-1)^2 = 0.25 and (0.5-0)^2 = 0.25

    def test_empty_input(self):
        """Test BS with empty arrays."""
        bs = brier_score([], [])
        assert np.isnan(bs)


class TestCrossEntropy:
    """Tests for Cross Entropy metric."""

    def test_perfect_predictions(self):
        """Test CE with perfect predictions."""
        confidences = [0.999, 0.999, 0.001, 0.001]
        correctness = [True, True, False, False]
        ce = cross_entropy(confidences, correctness)
        assert ce < 0.01  # Should be very small

    def test_uniform_predictions(self):
        """Test CE with uniform predictions."""
        confidences = [0.5] * 10
        correctness = [True] * 5 + [False] * 5
        ce = cross_entropy(confidences, correctness)
        # CE for p=0.5: -log(0.5) ≈ 0.693
        assert abs(ce - 0.693) < 0.01

    def test_empty_input(self):
        """Test CE with empty arrays."""
        ce = cross_entropy([], [])
        assert np.isnan(ce)

    def test_clipping(self):
        """Test that extreme values are clipped."""
        # This should not cause log(0) errors
        confidences = [1.0, 0.0, 1.0, 0.0]
        correctness = [True, False, True, False]
        ce = cross_entropy(confidences, correctness)
        assert ce >= 0  # Should be finite and non-negative


class TestAUCScore:
    """Tests for AUC metric."""

    def test_perfect_ranking(self):
        """Test AUC with perfect ranking."""
        confidences = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        correctness = [True, True, True, False, False, False]
        auc = auc_score(confidences, correctness)
        assert auc == 1.0

    def test_worst_ranking(self):
        """Test AUC with worst ranking."""
        confidences = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        correctness = [True, True, True, False, False, False]
        auc = auc_score(confidences, correctness)
        assert auc == 0.0

    def test_random_ranking(self):
        """Test AUC with random-like ranking."""
        confidences = [0.5] * 10
        correctness = [True] * 5 + [False] * 5
        auc = auc_score(confidences, correctness)
        assert abs(auc - 0.5) < 0.1  # Should be close to 0.5

    def test_empty_input(self):
        """Test AUC with empty arrays."""
        auc = auc_score([], [])
        assert np.isnan(auc)

    def test_single_class(self):
        """Test AUC with only one class."""
        confidences = [0.8, 0.7, 0.6]
        correctness = [True, True, True]
        auc = auc_score(confidences, correctness)
        assert np.isnan(auc)  # Undefined with single class


class TestIntegration:
    """Integration tests for all metrics together."""

    def test_metrics_consistency(self):
        """Test that all metrics can be computed on same data."""
        np.random.seed(42)
        confidences = np.random.random(100)
        correctness = np.random.random(100) > 0.5

        # All should run without errors
        ece = expected_calibration_error(confidences, correctness)
        bs = brier_score(confidences, correctness)
        ce = cross_entropy(confidences, correctness)
        auc = auc_score(confidences, correctness)

        # All should be in valid ranges
        assert 0 <= ece <= 1
        assert 0 <= bs <= 1
        assert ce >= 0
        assert 0 <= auc <= 1
