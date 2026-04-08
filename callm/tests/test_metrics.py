"""Unit tests for calibration metrics."""

import torch
import numpy as np
from callm.metrics import (
    ExpectedCalibrationError,
    BrierScore,
    CrossEntropy,
    AUCScore,
    CCAG,
    GammaCCAG,
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

    def test_nan_input_raises_error(self):
        """Test that passing NaN values raises ValueError."""
        metric = ExpectedCalibrationError()
        confidences = torch.tensor([float("nan"), 0.5])
        correctness = torch.tensor([True, False])
        import pytest

        with pytest.raises(ValueError, match="NaN values found in input tensors."):
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

    def test_nan_input_raises_error(self):
        """Test that passing NaN values raises ValueError."""
        metric = BrierScore()
        confidences = torch.tensor([float("nan"), 0.5])
        correctness = torch.tensor([True, False])
        import pytest

        with pytest.raises(ValueError, match="NaN values found in input tensors."):
            metric.update(confidences, correctness)

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

    def test_nan_input_raises_error(self):
        """Test that passing NaN values raises ValueError."""
        metric = CrossEntropy()
        confidences = torch.tensor([float("nan"), 0.5])
        correctness = torch.tensor([True, False])
        import pytest

        with pytest.raises(ValueError, match="NaN values found in input tensors."):
            metric.update(confidences, correctness)

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

    def test_nan_input_raises_error(self):
        """Test that passing NaN values raises ValueError."""
        metric = AUCScore()
        confidences = torch.tensor([float("nan"), 0.5])
        correctness = torch.tensor([True, False])
        import pytest

        with pytest.raises(ValueError, match="NaN values found in input tensors."):
            metric.update(confidences, correctness)

    def test_single_class(self):
        """Test AUC with only one class. BinaryAUROC returns 0.0."""
        confidences = torch.tensor([0.8, 0.7, 0.6])
        correctness = torch.tensor([True, True, True])
        metric = AUCScore()
        metric.update(confidences, correctness)
        auc = metric.compute().item()
        assert auc == 0.0


class TestCCAG:
    """Tests for CCAG metric."""

    def test_correct_high_confidence(self):
        """When correct with high confidence, cost should be relatively high (penalizes uncertainty)."""
        confidences = torch.tensor([0.9])
        correctness = torch.tensor([True])
        metric = CCAG()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # Default cost_func = _integrated_cost_case1_w_1_gamma:
        # cost = 1 - q - indicator * log(1 - q)
        # indicator=1, q=0.9: 1 - 0.9 - 1*log(0.1) = 0.1 + 2.3026 ≈ 2.4026
        expected = 1 - 0.9 - np.log(1 - 0.9)
        assert abs(cost - expected) < 1e-4

    def test_correct_low_confidence(self):
        """When correct with low confidence, cost should still be moderate."""
        confidences = torch.tensor([0.2])
        correctness = torch.tensor([True])
        metric = CCAG()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # indicator=1, q=0.2: 1 - 0.2 - 1*log(0.8) = 0.8 + 0.2231 ≈ 1.0231
        expected = 1 - 0.2 - np.log(1 - 0.2)
        assert abs(cost - expected) < 1e-4

    def test_incorrect_high_confidence(self):
        """When incorrect with high confidence, cost should be low (close to abstain cost)."""
        confidences = torch.tensor([0.9])
        correctness = torch.tensor([False])
        metric = CCAG()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # indicator=0, q=0.9: 1 - 0.9 - 0 = 0.1
        expected = 1 - 0.9
        assert abs(cost - expected) < 1e-3

    def test_incorrect_low_confidence(self):
        """When incorrect with low confidence, cost should be higher."""
        confidences = torch.tensor([0.2])
        correctness = torch.tensor([False])
        metric = CCAG()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # indicator=0, q=0.2: 1 - 0.2 - 0 = 0.8
        expected = 1 - 0.2
        assert abs(cost - expected) < 1e-3

    def test_mixed_predictions(self):
        """Test with a mix of correct and incorrect predictions."""
        confidences = torch.tensor([0.9, 0.2])
        correctness = torch.tensor([True, False])
        metric = CCAG()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # cost = 1 - q - indicator * log(1 - q)
        cost_correct = 1 - 0.9 - np.log(1 - 0.9)  # ≈ 2.4026
        cost_incorrect = 1 - 0.2  # = 0.8
        expected = (cost_correct + cost_incorrect) / 2
        assert abs(cost - expected) < 1e-3

    def test_empty_input(self):
        """Test with no data."""
        metric = CCAG()
        cost = metric.compute().item()
        assert np.isnan(cost)

    def test_nan_input_raises_error(self):
        """Test that passing NaN values raises ValueError."""
        metric = CCAG()
        confidences = torch.tensor([float("nan"), 0.5])
        correctness = torch.tensor([True, False])
        import pytest

        with pytest.raises(ValueError, match="NaN values found in input tensors."):
            metric.update(confidences, correctness)

    def test_incremental_update(self):
        """Test that multiple updates accumulate correctly."""
        confidences = torch.tensor([0.9, 0.2, 0.5, 0.7])
        correctness = torch.tensor([True, False, True, False])

        metric1 = CCAG()
        metric1.update(confidences, correctness)
        cost1 = metric1.compute().item()

        metric2 = CCAG()
        metric2.update(confidences[:2], correctness[:2])
        metric2.update(confidences[2:], correctness[2:])
        cost2 = metric2.compute().item()

        assert abs(cost1 - cost2) < 1e-6

    def test_edge_confidence_near_one(self):
        """Test that confidence near 1.0 doesn't cause errors."""
        confidences = torch.tensor([0.9999])
        correctness = torch.tensor([False])
        metric = CCAG()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        assert np.isfinite(cost)

    def test_edge_confidence_near_zero(self):
        """Test that confidence near 0.0 doesn't cause errors."""
        confidences = torch.tensor([0.0001])
        correctness = torch.tensor([True])
        metric = CCAG()
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        assert np.isfinite(cost)


class TestGammaCCAG:
    """Tests for Gamma-CCAG metric."""

    # Parametrization 1: a=1, b=gamma
    P1_A = staticmethod(lambda g: 1.0)
    P1_B = staticmethod(lambda g: g)

    # Parametrization 2: a=1/(1-gamma), b=1/gamma
    P2_A = staticmethod(lambda g: 1.0 / (1.0 - g))
    P2_B = staticmethod(lambda g: 1.0 / g)

    def test_all_correct_high_confidence_p1(self):
        """All correct with high confidence → all answers, cost = 0."""
        confidences = torch.tensor([0.99, 0.95, 0.98])
        correctness = torch.tensor([True, True, True])
        metric = GammaCCAG(a_func=self.P1_A, b_func=self.P1_B, gamma=0.5)
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # s = [0.01, 0.05, 0.02], threshold = 0.5
        # All s <= 0.5, so all answer. Cost = 1 * 0 = 0 for each
        assert cost == 0.0

    def test_all_incorrect_high_confidence_p1(self):
        """All incorrect with high confidence → all answers, cost = a(γ)."""
        confidences = torch.tensor([0.99, 0.95, 0.98])
        correctness = torch.tensor([False, False, False])
        metric = GammaCCAG(a_func=self.P1_A, b_func=self.P1_B, gamma=0.5)
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # All answer (s <= 0.5), cost = 1*1 = 1 each, mean = 1.0
        assert abs(cost - 1.0) < 1e-6

    def test_all_abstain_low_confidence_p1(self):
        """Low confidence, low gamma → all abstain, cost = b(γ) = gamma."""
        confidences = torch.tensor([0.01, 0.02, 0.03])
        correctness = torch.tensor([True, False, True])
        metric = GammaCCAG(a_func=self.P1_A, b_func=self.P1_B, gamma=0.05)
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # s = [0.99, 0.98, 0.97], threshold = 0.05
        # All s > 0.05, so all abstain. Cost = 0.05 each
        assert abs(cost - 0.05) < 1e-6

    def test_manual_mixed_p1(self):
        """Manual calculation with mixed predictions, P1, gamma=0.5."""
        confidences = torch.tensor([0.8, 0.3, 0.9])
        correctness = torch.tensor([True, False, True])
        metric = GammaCCAG(a_func=self.P1_A, b_func=self.P1_B, gamma=0.5)
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # s = [0.2, 0.7, 0.1], threshold = 0.5
        # Sample 0: s=0.2 <= 0.5 → answer, correct → cost = 1*0 = 0
        # Sample 1: s=0.7 > 0.5 → abstain → cost = 0.5
        # Sample 2: s=0.1 <= 0.5 → answer, correct → cost = 1*0 = 0
        # Mean = (0 + 0.5 + 0) / 3 = 0.1667
        expected = 0.5 / 3
        assert abs(cost - expected) < 1e-5

    def test_manual_mixed_p2(self):
        """Manual calculation with P2, gamma=0.5."""
        confidences = torch.tensor([0.8, 0.3, 0.9])
        correctness = torch.tensor([True, False, True])
        # P2: a=1/(1-0.5)=2, b=1/0.5=2, threshold=1.0
        metric = GammaCCAG(a_func=self.P2_A, b_func=self.P2_B, gamma=0.5)
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # s = [0.2, 0.7, 0.1], threshold = 2/2 = 1.0
        # All s <= 1.0, so all answer
        # Sample 0: correct → cost = 2*0 = 0
        # Sample 1: incorrect → cost = 2*1 = 2
        # Sample 2: correct → cost = 2*0 = 0
        # Mean = (0 + 2 + 0) / 3 = 0.6667
        expected = 2.0 / 3
        assert abs(cost - expected) < 1e-5

    def test_different_gammas_produce_different_results(self):
        """Different gamma values should produce different costs."""
        # Use data with confidence values that cross different thresholds
        confidences = torch.tensor([0.95, 0.85, 0.55, 0.35, 0.15, 0.05])
        correctness = torch.tensor([True, False, True, False, True, False])

        costs = []
        for gamma in [0.1, 0.5, 0.9]:
            metric = GammaCCAG(a_func=self.P1_A, b_func=self.P1_B, gamma=gamma)
            metric.update(confidences, correctness)
            costs.append(metric.compute().item())

        # All costs should be different since thresholds cross different
        # confidence boundaries
        assert costs[0] != costs[1]
        assert costs[1] != costs[2]

    def test_p1_vs_p2_different(self):
        """P1 and P2 should generally produce different results."""
        confidences = torch.tensor([0.9, 0.5, 0.3, 0.8, 0.1])
        correctness = torch.tensor([True, False, True, True, False])

        m1 = GammaCCAG(a_func=self.P1_A, b_func=self.P1_B, gamma=0.2)
        m1.update(confidences, correctness)
        c1 = m1.compute().item()

        m2 = GammaCCAG(a_func=self.P2_A, b_func=self.P2_B, gamma=0.2)
        m2.update(confidences, correctness)
        c2 = m2.compute().item()

        assert c1 != c2

    def test_nan_input_raises_error(self):
        """Test that NaN inputs raise ValueError."""
        metric = GammaCCAG(gamma=0.5)
        confidences = torch.tensor([float("nan"), 0.5])
        correctness = torch.tensor([True, False])
        import pytest

        with pytest.raises(ValueError, match="NaN values found in input tensors."):
            metric.update(confidences, correctness)

    def test_empty_input(self):
        """Test with no data returns NaN."""
        metric = GammaCCAG(gamma=0.5)
        cost = metric.compute().item()
        assert np.isnan(cost)

    def test_incremental_update(self):
        """Test that multiple updates accumulate correctly."""
        confidences = torch.tensor([0.9, 0.5, 0.3, 0.8])
        correctness = torch.tensor([True, False, True, False])

        # Single update
        m1 = GammaCCAG(a_func=self.P1_A, b_func=self.P1_B, gamma=0.5)
        m1.update(confidences, correctness)
        c1 = m1.compute().item()

        # Incremental updates
        m2 = GammaCCAG(a_func=self.P1_A, b_func=self.P1_B, gamma=0.5)
        m2.update(confidences[:2], correctness[:2])
        m2.update(confidences[2:], correctness[2:])
        c2 = m2.compute().item()

        assert abs(c1 - c2) < 1e-6

    def test_extreme_gamma_p1_low(self):
        """At very low gamma (P1), threshold is very low → most abstain."""
        confidences = torch.tensor([0.9, 0.5, 0.3])
        correctness = torch.tensor([True, False, True])
        metric = GammaCCAG(a_func=self.P1_A, b_func=self.P1_B, gamma=0.05)
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # threshold = 0.05, s = [0.1, 0.5, 0.7]
        # All s > 0.05 → all abstain → cost = 0.05
        assert abs(cost - 0.05) < 1e-6

    def test_extreme_gamma_p1_high(self):
        """At very high gamma (P1), threshold is high → most answer."""
        confidences = torch.tensor([0.9, 0.5, 0.3])
        correctness = torch.tensor([True, False, True])
        metric = GammaCCAG(a_func=self.P1_A, b_func=self.P1_B, gamma=0.95)
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # threshold = 0.95
        # s = [0.1, 0.5, 0.7] → all s <= 0.95 → all answer
        # costs = [0, 1, 0] → mean = 1/3
        expected = 1.0 / 3
        assert abs(cost - expected) < 1e-5

    def test_extreme_gamma_p2_low(self):
        """At very low gamma (P2), b is huge → all abstain at high cost."""
        confidences = torch.tensor([0.9, 0.5, 0.3])
        correctness = torch.tensor([True, False, True])
        metric = GammaCCAG(a_func=self.P2_A, b_func=self.P2_B, gamma=0.05)
        metric.update(confidences, correctness)
        cost = metric.compute().item()
        # a = 1/0.95 ≈ 1.053, b = 1/0.05 = 20, threshold = 20/1.053 ≈ 19
        # s = [0.1, 0.5, 0.7] → all s <= 19 → all ANSWER
        # costs = [1.053*0, 1.053*1, 1.053*0] → mean = 1.053/3
        expected = (1.0 / 0.95) / 3
        assert abs(cost - expected) < 1e-4

    def test_cost_non_negative(self):
        """Cost should always be non-negative."""
        torch.manual_seed(42)
        for gamma in [0.05, 0.1, 0.5, 0.9, 0.95]:
            for a_func, b_func in [(self.P1_A, self.P1_B), (self.P2_A, self.P2_B)]:
                metric = GammaCCAG(a_func=a_func, b_func=b_func, gamma=gamma)
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
            "brier": BrierScore(),
            "ce": CrossEntropy(),
            "auc": AUCScore(),
            "cc": CCAG(),
            "gamma_ccag_p1": GammaCCAG(
                a_func=lambda g: 1.0, b_func=lambda g: g, gamma=0.5
            ),
            "gamma_ccag_p2": GammaCCAG(
                a_func=lambda g: 1.0 / (1.0 - g),
                b_func=lambda g: 1.0 / g,
                gamma=0.5,
            ),
        }

        for metric in metrics.values():
            metric.update(confidences, correctness)

        ece = metrics["ece"].compute().item()
        bs = metrics["brier"].compute().item()
        ce = metrics["ce"].compute().item()
        auc = metrics["auc"].compute().item()
        cc = metrics["cc"].compute().item()
        gc1 = metrics["gamma_ccag_p1"].compute().item()
        gc2 = metrics["gamma_ccag_p2"].compute().item()

        # All should be in valid ranges
        assert 0 <= ece <= 1
        assert 0 <= bs <= 1
        assert ce >= 0
        assert 0 <= auc <= 1
        assert np.isfinite(cc)
        assert gc1 >= 0
        assert gc2 >= 0

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
