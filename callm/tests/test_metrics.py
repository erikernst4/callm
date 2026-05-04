"""Unit tests for calibration metrics."""

import torch
import numpy as np
from callm.metrics import (
    ExpectedCalibrationError,
    ConfidenceBrierScore,
    ConfidenceCrossEntropy,
    ConfidenceAUCScore,
    CCAS,
    ConfidenceGammaECUAS,
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

    def test_nan_confidence_uses_fallback(self):
        """Test that NaN confidences are replaced with 0.5 fallback."""
        metric = ExpectedCalibrationError()
        confidences = torch.tensor([float("nan"), 0.5])
        correctness = torch.tensor([True, False])
        metric.update(confidences, correctness)
        ece = metric.compute().item()
        assert np.isfinite(ece)

    def test_nan_correctness_raises_error(self):
        """Test that NaN correctness still raises ValueError."""
        metric = ExpectedCalibrationError()
        confidences = torch.tensor([0.5, 0.5])
        correctness = torch.tensor([float("nan"), 0.0])
        import pytest

        with pytest.raises(ValueError, match="NaN values found in correctness tensor."):
            metric.update(confidences, correctness)

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


class TestConfidenceBrierScore:
    """Tests for Brier Score metric."""

    def test_perfect_predictions(self):
        """Test BS with perfect predictions."""
        confidences = torch.tensor([1.0, 1.0, 0.0, 0.0])
        correctness = torch.tensor([True, True, False, False])
        metric = ConfidenceBrierScore()
        metric.update(confidences, correctness)
        bs = metric.compute().item()
        assert bs == 0.0

    def test_worst_predictions(self):
        """Test BS with worst predictions."""
        confidences = torch.tensor([0.0, 0.0, 1.0, 1.0])
        correctness = torch.tensor([True, True, False, False])
        metric = ConfidenceBrierScore()
        metric.update(confidences, correctness)
        bs = metric.compute().item()
        assert bs == 1.0

    def test_uniform_predictions(self):
        """Test BS with uniform 0.5 predictions."""
        confidences = torch.tensor([0.5] * 10)
        correctness = torch.tensor([True] * 5 + [False] * 5)
        metric = ConfidenceBrierScore()
        metric.update(confidences, correctness)
        bs = metric.compute().item()
        assert bs == 0.25  # (0.5-1)^2 = 0.25 and (0.5-0)^2 = 0.25

    def test_empty_input(self):
        """Test BS with empty arrays."""
        metric = ConfidenceBrierScore()
        bs = metric.compute().item()
        assert np.isnan(bs)

    def test_nan_confidence_uses_fallback(self):
        """Test that NaN confidences are replaced with 0.5 fallback."""
        metric = ConfidenceBrierScore()
        confidences = torch.tensor([float("nan"), 0.5])
        correctness = torch.tensor([True, False])
        metric.update(confidences, correctness)
        bs = metric.compute().item()
        assert np.isfinite(bs)

    def test_nan_correctness_raises_error(self):
        """Test that NaN correctness still raises ValueError."""
        metric = ConfidenceBrierScore()
        confidences = torch.tensor([0.5, 0.5])
        correctness = torch.tensor([float("nan"), 0.0])
        import pytest

        with pytest.raises(ValueError, match="NaN values found in correctness tensor."):
            metric.update(confidences, correctness)

    def test_incremental_update(self):
        """Test that multiple updates accumulate correctly."""
        confidences = torch.tensor([0.8, 0.6, 0.3, 0.9])
        correctness = torch.tensor([True, True, False, True])

        metric1 = ConfidenceBrierScore()
        metric1.update(confidences, correctness)
        bs1 = metric1.compute().item()

        metric2 = ConfidenceBrierScore()
        metric2.update(confidences[:2], correctness[:2])
        metric2.update(confidences[2:], correctness[2:])
        bs2 = metric2.compute().item()

        assert abs(bs1 - bs2) < 1e-6


class TestConfidenceCrossEntropy:
    """Tests for Cross Entropy metric."""

    def test_perfect_predictions(self):
        """Test CE with perfect predictions."""
        confidences = torch.tensor([0.999, 0.999, 0.001, 0.001])
        correctness = torch.tensor([True, True, False, False])
        metric = ConfidenceCrossEntropy()
        metric.update(confidences, correctness)
        ce = metric.compute().item()
        assert ce < 0.01  # Should be very small

    def test_uniform_predictions(self):
        """Test CE with uniform predictions."""
        confidences = torch.tensor([0.5] * 10)
        correctness = torch.tensor([True] * 5 + [False] * 5)
        metric = ConfidenceCrossEntropy()
        metric.update(confidences, correctness)
        ce = metric.compute().item()
        # CE for p=0.5: -log(0.5) ≈ 0.693
        assert abs(ce - 0.693) < 0.01

    def test_empty_input(self):
        """Test CE with empty arrays raises ValueError."""
        import pytest

        metric = ConfidenceCrossEntropy()
        with pytest.raises(ValueError, match="No samples to compute cross entropy."):
            metric.compute()

    def test_nan_confidence_uses_fallback(self):
        """Test that NaN confidences are replaced with 0.5 fallback."""
        metric = ConfidenceCrossEntropy()
        confidences = torch.tensor([float("nan"), 0.5])
        correctness = torch.tensor([True, False])
        metric.update(confidences, correctness)
        ce = metric.compute().item()
        assert np.isfinite(ce)

    def test_nan_correctness_raises_error(self):
        """Test that NaN correctness still raises ValueError."""
        metric = ConfidenceCrossEntropy()
        confidences = torch.tensor([0.5, 0.5])
        correctness = torch.tensor([float("nan"), 0.0])
        import pytest

        with pytest.raises(ValueError, match="NaN values found in correctness tensor."):
            metric.update(confidences, correctness)

    def test_clipping(self):
        """Test that extreme values are clipped."""
        # This should not cause log(0) errors
        confidences = torch.tensor([1.0, 0.0, 1.0, 0.0])
        correctness = torch.tensor([True, False, True, False])
        metric = ConfidenceCrossEntropy()
        metric.update(confidences, correctness)
        ce = metric.compute().item()
        assert ce >= 0  # Should be finite and non-negative


class TestConfidenceAUCScore:
    """Tests for AUC metric."""

    def test_perfect_ranking(self):
        """Test AUC with perfect ranking."""
        confidences = torch.tensor([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        correctness = torch.tensor([True, True, True, False, False, False])
        metric = ConfidenceAUCScore()
        metric.update(confidences, correctness)
        auc = metric.compute().item()
        assert auc == 1.0

    def test_worst_ranking(self):
        """Test AUC with worst ranking."""
        confidences = torch.tensor([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        correctness = torch.tensor([True, True, True, False, False, False])
        metric = ConfidenceAUCScore()
        metric.update(confidences, correctness)
        auc = metric.compute().item()
        assert auc == 0.0

    def test_random_ranking(self):
        """Test AUC with random-like ranking."""
        confidences = torch.tensor([0.5] * 10)
        correctness = torch.tensor([True] * 5 + [False] * 5)
        metric = ConfidenceAUCScore()
        metric.update(confidences, correctness)
        auc = metric.compute().item()
        assert abs(auc - 0.5) < 0.1  # Should be close to 0.5

    def test_empty_input(self):
        """Test AUC with empty arrays raises ValueError."""
        import pytest

        metric = ConfidenceAUCScore()
        metric.update(torch.tensor([]), torch.tensor([]))
        with pytest.raises(ValueError, match="No samples to compute AUC score."):
            metric.compute()

    def test_nan_confidence_uses_fallback(self):
        """Test that NaN confidences are replaced with 0.5 fallback."""
        metric = ConfidenceAUCScore()
        confidences = torch.tensor([float("nan"), 0.5])
        correctness = torch.tensor([True, False])
        metric.update(confidences, correctness)
        auc = metric.compute().item()
        assert np.isfinite(auc)

    def test_nan_correctness_raises_error(self):
        """Test that NaN correctness still raises ValueError."""
        metric = ConfidenceAUCScore()
        confidences = torch.tensor([0.5, 0.5])
        correctness = torch.tensor([float("nan"), 0.0])
        import pytest

        with pytest.raises(ValueError, match="NaN values found in correctness tensor."):
            metric.update(confidences, correctness)

    def test_single_class(self):
        """Test AUC with only one class. BinaryAUROC returns 0.0."""
        confidences = torch.tensor([0.8, 0.7, 0.6])
        correctness = torch.tensor([True, True, True])
        metric = ConfidenceAUCScore()
        metric.update(confidences, correctness)
        auc = metric.compute().item()
        assert auc == 0.0


class TestCCAS:
    """Tests for CCAS metric."""

    def test_correct_high_confidence(self):
        """When correct with high confidence, cost should be low."""
        confidences = torch.tensor([0.9])
        correctness = torch.tensor([True])
        metric = CCAS()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # cost = 1 - q - (1 - indicator) * log(1 - q)
        # indicator=1, q=0.9: 1 - 0.9 - 0 = 0.1
        expected = 1 - 0.9
        assert abs(cost - expected) < 1e-4

    def test_correct_low_confidence(self):
        """When correct with low confidence, cost should be moderate."""
        confidences = torch.tensor([0.2])
        correctness = torch.tensor([True])
        metric = CCAS()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # indicator=1, q=0.2: 1 - 0.2 - 0 = 0.8
        expected = 1 - 0.2
        assert abs(cost - expected) < 1e-4

    def test_incorrect_high_confidence(self):
        """When incorrect with high confidence, cost should be high (heavy penalty)."""
        confidences = torch.tensor([0.9])
        correctness = torch.tensor([False])
        metric = CCAS()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # indicator=0, q=0.9: 1 - 0.9 - 1*log(0.1) ≈ 2.4026
        expected = 1 - 0.9 - np.log(1 - 0.9)
        assert abs(cost - expected) < 1e-3

    def test_incorrect_low_confidence(self):
        """When incorrect with low confidence, cost should be moderate-high."""
        confidences = torch.tensor([0.2])
        correctness = torch.tensor([False])
        metric = CCAS()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # indicator=0, q=0.2: 1 - 0.2 - 1*log(0.8) ≈ 1.0231
        expected = 1 - 0.2 - np.log(1 - 0.2)
        assert abs(cost - expected) < 1e-3

    def test_mixed_predictions(self):
        """Test with a mix of correct and incorrect predictions."""
        confidences = torch.tensor([0.9, 0.2])
        correctness = torch.tensor([True, False])
        metric = CCAS()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # cost_correct_high = 1 - 0.9 = 0.1
        # cost_incorrect_low = 1 - 0.2 - np.log(0.8) ≈ 1.0231
        cost_correct = 1 - 0.9
        cost_incorrect = 1 - 0.2 - np.log(1 - 0.2)
        expected = (cost_correct + cost_incorrect) / 2
        assert abs(cost - expected) < 1e-3

    def test_empty_input(self):
        """Test with no data raises ValueError."""
        import pytest

        metric = CCAS()
        with pytest.raises(ValueError, match="No samples to compute ECUAS."):
            metric.compute()

    def test_nan_confidence_uses_fallback(self):
        """Test that NaN confidences are replaced with 0.5 fallback."""
        metric = CCAS()
        confidences = torch.tensor([float("nan"), 0.5])
        correctness = torch.tensor([True, False])
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        assert np.isfinite(cost)

    def test_nan_correctness_raises_error(self):
        """Test that NaN correctness still raises ValueError."""
        metric = CCAS()
        confidences = torch.tensor([0.5, 0.5])
        correctness = torch.tensor([float("nan"), 0.0])
        import pytest

        with pytest.raises(ValueError, match="NaN values found in correctness tensor."):
            metric.update(confidences, correctness)

    def test_incremental_update(self):
        """Test that multiple updates accumulate correctly."""
        confidences = torch.tensor([0.9, 0.2, 0.5, 0.7])
        correctness = torch.tensor([True, False, True, False])

        metric1 = CCAS()
        metric1.update(confidences, correctness)
        cost1 = metric1.compute().item()

        metric2 = CCAS()
        metric2.update(confidences[:2], correctness[:2])
        metric2.update(confidences[2:], correctness[2:])
        cost2 = metric2.compute().item()

        assert abs(cost1 - cost2) < 1e-6

    def test_edge_confidence_near_one(self):
        """Test that confidence near 1.0 doesn't cause errors."""
        confidences = torch.tensor([0.9999])
        correctness = torch.tensor([False])
        metric = CCAS()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        assert np.isfinite(cost)

    def test_edge_confidence_near_zero(self):
        """Test that confidence near 0.0 doesn't cause errors."""
        confidences = torch.tensor([0.0001])
        correctness = torch.tensor([True])
        metric = CCAS()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        assert np.isfinite(cost)


class TestGammaECUAS:
    """Tests for Gamma-ECUAS metric."""

    def test_high_confidence_abstains(self):
        """High confidence (s < gamma) triggers abstain, cost = gamma."""
        confidences = torch.tensor([0.9])
        correctness = torch.tensor([True])
        metric = ConfidenceGammaECUAS(gamma=0.5)
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # s = 0.1 < 0.5 → abstain → cost = 0.5
        assert abs(cost - 0.5) < 1e-6

    def test_low_confidence_answers_correctly(self):
        """Low confidence (s >= gamma) triggers answer, correct → cost = 0."""
        confidences = torch.tensor([0.3])
        correctness = torch.tensor([True])
        metric = ConfidenceGammaECUAS(gamma=0.5)
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # s = 0.7 >= 0.5 → answer, correct → cost = 0
        assert cost == 0.0

    def test_low_confidence_answers_incorrectly(self):
        """Low confidence (s >= gamma) triggers answer, incorrect → cost = 1."""
        confidences = torch.tensor([0.3])
        correctness = torch.tensor([False])
        metric = ConfidenceGammaECUAS(gamma=0.5)
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # s = 0.7 >= 0.5 → answer, incorrect → cost = 1
        assert abs(cost - 1.0) < 1e-6

    def test_mixed_predictions(self):
        """Mix of high-confidence (abstain) and low-confidence (answer) samples."""
        confidences = torch.tensor([0.9, 0.3, 0.8])
        correctness = torch.tensor([True, False, False])
        metric = ConfidenceGammaECUAS(gamma=0.5)
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # s = [0.1, 0.7, 0.2]
        # s[0]=0.1 < 0.5 → abstain → cost = 0.5
        # s[1]=0.7 >= 0.5 → answer, incorrect → cost = 1
        # s[2]=0.2 < 0.5 → abstain → cost = 0.5
        # mean = (0.5 + 1 + 0.5) / 3
        expected = 2.0 / 3
        assert abs(cost - expected) < 1e-5

    def test_all_high_confidence_abstain(self):
        """All high-confidence → all abstain → cost = gamma."""
        confidences = torch.tensor([0.99, 0.95, 0.98])
        correctness = torch.tensor([True, False, True])
        metric = ConfidenceGammaECUAS(gamma=0.5)
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # s = [0.01, 0.05, 0.02], all < 0.5 → all abstain → cost = 0.5
        assert abs(cost - 0.5) < 1e-6

    def test_all_low_confidence_answer(self):
        """All low-confidence → all answer → cost depends on correctness."""
        confidences = torch.tensor([0.1, 0.2, 0.3])
        correctness = torch.tensor([True, True, False])
        metric = ConfidenceGammaECUAS(gamma=0.5)
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # s = [0.9, 0.8, 0.7], all >= 0.5 → all answer
        # costs = [0, 0, 1], mean = 1/3
        expected = 1.0 / 3
        assert abs(cost - expected) < 1e-5

    def test_different_gammas_produce_different_results(self):
        """Different gammas shift the abstain/answer boundary."""
        confidences = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])
        correctness = torch.tensor([True, False, True, False, True])

        costs = []
        for gamma in [0.15, 0.45, 0.85]:
            metric = ConfidenceGammaECUAS(gamma=gamma)
            metric.update(confidences, correctness)
            costs.append(metric.compute().item())

        # Different gammas cross different confidence boundaries
        assert not all(abs(c - costs[0]) < 1e-6 for c in costs[1:])

    def test_nan_confidence_uses_fallback(self):
        """Test that NaN confidences are replaced with 0.5 fallback."""
        metric = ConfidenceGammaECUAS(gamma=0.5)
        confidences = torch.tensor([float("nan"), 0.5])
        correctness = torch.tensor([True, False])
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        assert np.isfinite(cost)

    def test_nan_correctness_raises_error(self):
        """Test that NaN correctness still raises ValueError."""
        metric = ConfidenceGammaECUAS(gamma=0.5)
        confidences = torch.tensor([0.5, 0.5])
        correctness = torch.tensor([float("nan"), 0.0])
        import pytest

        with pytest.raises(ValueError, match="NaN values found in correctness tensor."):
            metric.update(confidences, correctness)

    def test_empty_input(self):
        """Test with no data raises ValueError."""
        import pytest

        metric = ConfidenceGammaECUAS(gamma=0.5)
        with pytest.raises(ValueError, match="No samples to compute Gamma-ECUAS."):
            metric.compute()

    def test_incremental_update(self):
        """Multiple updates accumulate correctly."""
        confidences = torch.tensor([0.9, 0.5, 0.3, 0.8])
        correctness = torch.tensor([True, False, True, False])

        m1 = ConfidenceGammaECUAS(gamma=0.5)
        m1.update(confidences, correctness)
        c1 = m1.compute().item()

        m2 = ConfidenceGammaECUAS(gamma=0.5)
        m2.update(confidences[:2], correctness[:2])
        m2.update(confidences[2:], correctness[2:])
        c2 = m2.compute().item()

        assert abs(c1 - c2) < 1e-6

    def test_cost_non_negative(self):
        """Cost should always be non-negative."""
        torch.manual_seed(42)
        for gamma in [0.05, 0.1, 0.5, 0.9, 0.95]:
            metric = ConfidenceGammaECUAS(gamma=gamma)
            metric.update(torch.rand(50), (torch.rand(50) > 0.5).float())
            cost = metric.compute().item()
            assert cost >= 0, f"Negative cost at gamma={gamma}"


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
            "brier": ConfidenceBrierScore(),
            "ce": ConfidenceCrossEntropy(),
            "auc": ConfidenceAUCScore(),
            "cc": CCAS(),
            "gamma_ecuas": ConfidenceGammaECUAS(gamma=0.5),
        }

        for metric in metrics.values():
            metric.update(confidences, correctness)

        ece = metrics["ece"].compute().item()
        bs = metrics["brier"].compute().item()
        ce = metrics["ce"].compute().item()
        auc = metrics["auc"].compute().item()
        cc = metrics["cc"].compute().item()
        gc = metrics["gamma_ecuas"].compute().item()

        # All should be in valid ranges
        assert 0 <= ece <= 1
        assert 0 <= bs <= 1
        assert ce >= 0
        assert 0 <= auc <= 1
        assert np.isfinite(cc)
        assert gc >= 0

    def test_reset_works(self):
        """Test that metric reset clears state."""
        metric = ConfidenceBrierScore()
        confidences = torch.tensor([0.8, 0.2])
        correctness = torch.tensor([True, False])
        metric.update(confidences, correctness)
        _ = metric.compute()
        metric.reset()
        # After reset, compute should return nan (no data)
        assert np.isnan(metric.compute().item())
