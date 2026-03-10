"""Unit tests for calibration metrics."""

import torch
import numpy as np
from callm.metrics import (
    ExpectedCalibrationError,
    BrierScore,
    CrossEntropy,
    AUCScore,
    ConfidenceCost,
)


class TestExpectedCalibrationError:
    """Tests for ECE metric."""

    def test_perfect_calibration(self):
        """Test ECE with perfectly calibrated predictions."""
        # All predictions with 0.9 confidence should have 90% accuracy
        confidences = torch.tensor([0.9] * 100)
        correctness = torch.tensor([True] * 90 + [False] * 10)
        metric = ExpectedCalibrationError(n_bins=10)
        metric.update(confidences, correctness)
        ece = metric.compute().item()
        assert ece < 0.05  # Should be very small

    def test_worst_calibration(self):
        """Test ECE with worst case calibration."""
        # High confidence but all wrong
        confidences = torch.tensor([0.9] * 100)
        correctness = torch.tensor([False] * 100)
        metric = ExpectedCalibrationError(n_bins=10)
        metric.update(confidences, correctness)
        ece = metric.compute().item()
        assert ece > 0.8  # Should be high

    def test_empty_input(self):
        """Test ECE with empty arrays."""
        metric = ExpectedCalibrationError()
        # Update with empty tensors
        metric.update(torch.tensor([]), torch.tensor([]))
        ece = metric.compute().item()
        assert np.isnan(ece)

    def test_single_bin(self):
        """Test ECE with single bin."""
        confidences = torch.tensor([0.5, 0.5, 0.5, 0.5])
        correctness = torch.tensor([True, True, False, False])
        metric = ExpectedCalibrationError(n_bins=1)
        metric.update(confidences, correctness)
        ece = metric.compute().item()
        assert ece == 0.0  # Perfect calibration in single bin

    def test_incremental_update(self):
        """Test that multiple updates accumulate correctly."""
        # Single-shot
        confidences = torch.tensor([0.9] * 100)
        correctness = torch.tensor([True] * 90 + [False] * 10)
        metric1 = ExpectedCalibrationError(n_bins=10)
        metric1.update(confidences, correctness)
        ece1 = metric1.compute().item()

        # Incremental (two batches)
        metric2 = ExpectedCalibrationError(n_bins=10)
        metric2.update(confidences[:50], correctness[:50])
        metric2.update(confidences[50:], correctness[50:])
        ece2 = metric2.compute().item()

        assert abs(ece1 - ece2) < 1e-6


class TestBrierScore:
    """Tests for Brier Score metric."""

    def test_perfect_predictions(self):
        """Test BS with perfect predictions."""
        confidences = torch.tensor([1.0, 1.0, 0.0, 0.0])
        correctness = torch.tensor([True, True, False, False])
        metric = BrierScore()
        metric.update(confidences, correctness)
        bs = metric.compute().item()
        assert bs == 0.0

    def test_worst_predictions(self):
        """Test BS with worst predictions."""
        confidences = torch.tensor([0.0, 0.0, 1.0, 1.0])
        correctness = torch.tensor([True, True, False, False])
        metric = BrierScore()
        metric.update(confidences, correctness)
        bs = metric.compute().item()
        assert bs == 1.0

    def test_uniform_predictions(self):
        """Test BS with uniform 0.5 predictions."""
        confidences = torch.tensor([0.5] * 10)
        correctness = torch.tensor([True] * 5 + [False] * 5)
        metric = BrierScore()
        metric.update(confidences, correctness)
        bs = metric.compute().item()
        assert bs == 0.25  # (0.5-1)^2 = 0.25 and (0.5-0)^2 = 0.25

    def test_empty_input(self):
        """Test BS with empty arrays."""
        metric = BrierScore()
        bs = metric.compute().item()
        assert np.isnan(bs)

    def test_incremental_update(self):
        """Test that multiple updates accumulate correctly."""
        confidences = torch.tensor([0.8, 0.6, 0.3, 0.9])
        correctness = torch.tensor([True, True, False, True])

        metric1 = BrierScore()
        metric1.update(confidences, correctness)
        bs1 = metric1.compute().item()

        metric2 = BrierScore()
        metric2.update(confidences[:2], correctness[:2])
        metric2.update(confidences[2:], correctness[2:])
        bs2 = metric2.compute().item()

        assert abs(bs1 - bs2) < 1e-6


class TestCrossEntropy:
    """Tests for Cross Entropy metric."""

    def test_perfect_predictions(self):
        """Test CE with perfect predictions."""
        confidences = torch.tensor([0.999, 0.999, 0.001, 0.001])
        correctness = torch.tensor([True, True, False, False])
        metric = CrossEntropy()
        metric.update(confidences, correctness)
        ce = metric.compute().item()
        assert ce < 0.01  # Should be very small

    def test_uniform_predictions(self):
        """Test CE with uniform predictions."""
        confidences = torch.tensor([0.5] * 10)
        correctness = torch.tensor([True] * 5 + [False] * 5)
        metric = CrossEntropy()
        metric.update(confidences, correctness)
        ce = metric.compute().item()
        # CE for p=0.5: -log(0.5) ≈ 0.693
        assert abs(ce - 0.693) < 0.01

    def test_empty_input(self):
        """Test CE with empty arrays."""
        metric = CrossEntropy()
        ce = metric.compute().item()
        assert np.isnan(ce)

    def test_clipping(self):
        """Test that extreme values are clipped."""
        # This should not cause log(0) errors
        confidences = torch.tensor([1.0, 0.0, 1.0, 0.0])
        correctness = torch.tensor([True, False, True, False])
        metric = CrossEntropy()
        metric.update(confidences, correctness)
        ce = metric.compute().item()
        assert ce >= 0  # Should be finite and non-negative


class TestAUCScore:
    """Tests for AUC metric."""

    def test_perfect_ranking(self):
        """Test AUC with perfect ranking."""
        confidences = torch.tensor([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        correctness = torch.tensor([True, True, True, False, False, False])
        metric = AUCScore()
        metric.update(confidences, correctness)
        auc = metric.compute().item()
        assert auc == 1.0

    def test_worst_ranking(self):
        """Test AUC with worst ranking."""
        confidences = torch.tensor([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        correctness = torch.tensor([True, True, True, False, False, False])
        metric = AUCScore()
        metric.update(confidences, correctness)
        auc = metric.compute().item()
        assert auc == 0.0

    def test_random_ranking(self):
        """Test AUC with random-like ranking."""
        confidences = torch.tensor([0.5] * 10)
        correctness = torch.tensor([True] * 5 + [False] * 5)
        metric = AUCScore()
        metric.update(confidences, correctness)
        auc = metric.compute().item()
        assert abs(auc - 0.5) < 0.1  # Should be close to 0.5

    def test_empty_input(self):
        """Test AUC with empty arrays."""
        metric = AUCScore()
        metric.update(torch.tensor([]), torch.tensor([]))
        auc = metric.compute().item()
        assert np.isnan(auc)

    def test_single_class(self):
        """Test AUC with only one class. BinaryAUROC returns 0.0."""
        confidences = torch.tensor([0.8, 0.7, 0.6])
        correctness = torch.tensor([True, True, True])
        metric = AUCScore()
        metric.update(confidences, correctness)
        auc = metric.compute().item()
        assert auc == 0.0


class TestConfidenceCost:
    """Tests for Confidence Cost metric."""

    def test_correct_high_confidence(self):
        """When correct with high confidence, cost should be relatively low."""
        confidences = torch.tensor([0.9])
        correctness = torch.tensor([True])
        metric = ConfidenceCost()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # indicator=1: log(2-0.9)*(2-1) - log(1-0.9)*(1-1)
        # = log(1.1)*1 - log(0.1)*0 = log(1.1) ≈ 0.0953
        assert abs(cost - np.log(1.1)) < 1e-4

    def test_correct_low_confidence(self):
        """When correct with low confidence, cost should still be moderate."""
        confidences = torch.tensor([0.2])
        correctness = torch.tensor([True])
        metric = ConfidenceCost()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # indicator=1: log(2-0.2)*1 - log(1-0.2)*0 = log(1.8) ≈ 0.5878
        assert abs(cost - np.log(1.8)) < 1e-4

    def test_incorrect_high_confidence(self):
        """When incorrect with high confidence, cost should be high (penalized)."""
        confidences = torch.tensor([0.9])
        correctness = torch.tensor([False])
        metric = ConfidenceCost()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # indicator=0: log(2-0.9)*2 - log(1-0.9)*1
        # = log(1.1)*2 - log(0.1)*1 = 2*0.0953 - (-2.3026) ≈ 2.4932
        expected = 2 * np.log(1.1) - np.log(0.1)
        assert abs(cost - expected) < 1e-3

    def test_incorrect_low_confidence(self):
        """When incorrect with low confidence, cost should be lower."""
        confidences = torch.tensor([0.2])
        correctness = torch.tensor([False])
        metric = ConfidenceCost()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # indicator=0: log(2-0.2)*2 - log(1-0.2)*1
        # = log(1.8)*2 - log(0.8) = 2*0.5878 - (-0.2231) ≈ 1.3988
        expected = 2 * np.log(1.8) - np.log(0.8)
        assert abs(cost - expected) < 1e-3

    def test_mixed_predictions(self):
        """Test with a mix of correct and incorrect predictions."""
        confidences = torch.tensor([0.9, 0.2])
        correctness = torch.tensor([True, False])
        metric = ConfidenceCost()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # Mean of correct_high (≈0.0953) and incorrect_low (≈1.3988)
        cost_correct = np.log(1.1)
        cost_incorrect = 2 * np.log(1.8) - np.log(0.8)
        expected = (cost_correct + cost_incorrect) / 2
        assert abs(cost - expected) < 1e-3

    def test_empty_input(self):
        """Test with no data."""
        metric = ConfidenceCost()
        cost = metric.compute().item()
        assert np.isnan(cost)

    def test_incremental_update(self):
        """Test that multiple updates accumulate correctly."""
        confidences = torch.tensor([0.9, 0.2, 0.5, 0.7])
        correctness = torch.tensor([True, False, True, False])

        metric1 = ConfidenceCost()
        metric1.update(confidences, correctness)
        cost1 = metric1.compute().item()

        metric2 = ConfidenceCost()
        metric2.update(confidences[:2], correctness[:2])
        metric2.update(confidences[2:], correctness[2:])
        cost2 = metric2.compute().item()

        assert abs(cost1 - cost2) < 1e-6

    def test_edge_confidence_near_one(self):
        """Test that confidence near 1.0 doesn't cause errors."""
        confidences = torch.tensor([0.9999])
        correctness = torch.tensor([False])
        metric = ConfidenceCost()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        assert np.isfinite(cost)

    def test_edge_confidence_near_zero(self):
        """Test that confidence near 0.0 doesn't cause errors."""
        confidences = torch.tensor([0.0001])
        correctness = torch.tensor([True])
        metric = ConfidenceCost()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        assert np.isfinite(cost)


class TestIntegration:
    """Integration tests for all metrics together."""

    def test_metrics_consistency(self):
        """Test that all metrics can be computed on same data."""
        torch.manual_seed(42)
        confidences = torch.rand(100)
        correctness = torch.rand(100) > 0.5

        # All should run without errors
        metrics = {
            "ece": ExpectedCalibrationError(),
            "brier": BrierScore(),
            "ce": CrossEntropy(),
            "auc": AUCScore(),
            "cc": ConfidenceCost(),
        }

        for metric in metrics.values():
            metric.update(confidences, correctness)

        ece = metrics["ece"].compute().item()
        bs = metrics["brier"].compute().item()
        ce = metrics["ce"].compute().item()
        auc = metrics["auc"].compute().item()
        cc = metrics["cc"].compute().item()

        # All should be in valid ranges
        assert 0 <= ece <= 1
        assert 0 <= bs <= 1
        assert ce >= 0
        assert 0 <= auc <= 1
        assert np.isfinite(cc)

    def test_reset_works(self):
        """Test that metric reset clears state."""
        metric = BrierScore()
        confidences = torch.tensor([0.8, 0.2])
        correctness = torch.tensor([True, False])
        metric.update(confidences, correctness)
        _ = metric.compute()
        metric.reset()
        # After reset, compute should return nan (no data)
        assert np.isnan(metric.compute().item())
