import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
from callm.models.probe import Probe, ConfidenceAugmentedLLM
from callm.models.llm import LLM


class TestProbe:
    """Unit tests for Probe."""

    def test_probe_shapes(self):
        batch_size = 4
        seq_len = 10
        model_dim = 16

        probe = Probe(model_dim=model_dim)
        activations = torch.randn(batch_size, seq_len, model_dim)
        confidence = probe(activations)

        # Output shape should be (#batch)
        assert confidence.shape == (batch_size,)
        # Sigmoid outputs should be in [0, 1]
        assert torch.all(confidence >= 0.0)
        assert torch.all(confidence <= 1.0)

    def test_probe_pooling(self):
        batch_size = 1
        seq_len = 3
        model_dim = 4

        probe = Probe(model_dim=model_dim)
        # Set weight and bias to easily check output
        with torch.no_grad():
            probe.linear.weight.fill_(1.0)
            probe.linear.bias.fill_(0.0)

        # Activations with known values
        # activations shape: (1, 3, 4)
        activations = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                     [2.0, 3.0, 4.0, 5.0],
                                     [3.0, 4.0, 5.0, 6.0]]])
        # Mean across dim 1 (sequence len 3):
        # [ (1+2+3)/3, (2+3+4)/3, (3+4+5)/3, (4+5+6)/3 ] = [2.0, 3.0, 4.0, 5.0]
        # Sum of [2.0, 3.0, 4.0, 5.0] = 14.0
        # Sigmoid(14.0) should be the output
        expected = torch.sigmoid(torch.tensor([14.0]))
        confidence = probe(activations)
        assert torch.allclose(confidence, expected)


class TestConfidenceAugmentedLLM:
    """Unit tests for ConfidenceAugmentedLLM."""

    @pytest.fixture
    def mock_llm(self):
        mock_instance = Mock()
        mock_instance.model = Mock()
        mock_instance.model.config = Mock()
        mock_instance.model.config.hidden_size = 8
        mock_instance.parameters.return_value = iter([])
        return mock_instance

    def test_initialization(self, mock_llm):
        # Test default initialization
        model = ConfidenceAugmentedLLM(llm=mock_llm, layer_idx=-2, lr=1e-2)
        assert model.llm == mock_llm
        assert model.layer_idx == -2
        assert model.lr == 1e-2
        assert model.model_dim == 8
        assert isinstance(model.probe, Probe)
        assert model.probe.model_dim == 8
        assert isinstance(model.loss_fn, nn.BCELoss)

    def test_forward_with_generate_true(self, mock_llm):
        model = ConfidenceAugmentedLLM(llm=mock_llm, layer_idx=-1)

        # Mock LLM forward output: (generated_ids, hidden_states)
        generated_ids = torch.tensor([[1, 2, 3]])
        # Mock hidden states tuple: 2 layers, shape (1, 5, 8)
        mock_hidden_states = (
            torch.randn(1, 5, 8),
            torch.randn(1, 5, 8)
        )
        mock_llm.return_value = (generated_ids, mock_hidden_states)

        input_ids = torch.tensor([[4, 5]])
        attention_mask = torch.tensor([[1, 1]])

        gen_ids, conf = model(input_ids, attention_mask, generate=True)

        assert torch.equal(gen_ids, generated_ids)
        assert conf.shape == (1,)
        # LLM should be called with generate=True and return_hidden_states=True
        mock_llm.assert_called_once_with(
            input_ids,
            attention_mask,
            generate=True,
            return_hidden_states=True
        )

    def test_forward_with_generate_false(self, mock_llm):
        model = ConfidenceAugmentedLLM(llm=mock_llm, layer_idx=0)

        # Mock LLM forward output: hidden_states tuple
        mock_hidden_states = (
            torch.randn(2, 4, 8),
        )
        mock_llm.return_value = mock_hidden_states

        input_ids = torch.tensor([[1, 2]])
        attention_mask = torch.tensor([[1, 1]])

        conf = model(input_ids, attention_mask, generate=False)

        assert conf.shape == (2,)
        # LLM should be called with generate=False and return_hidden_states=True
        mock_llm.assert_called_once_with(
            input_ids,
            attention_mask,
            generate=False,
            return_hidden_states=True
        )

    def test_training_step(self, mock_llm):
        model = ConfidenceAugmentedLLM(llm=mock_llm, layer_idx=0)

        # Mock LLM forward returning hidden states
        mock_hidden_states = (
            torch.randn(2, 4, 8),
        )
        mock_llm.return_value = mock_hidden_states

        batch = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
            "correct": torch.tensor([1, 0])
        }

        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0 # Scalar loss

    def test_validation_step(self, mock_llm):
        model = ConfidenceAugmentedLLM(llm=mock_llm, layer_idx=0)

        mock_hidden_states = (
            torch.randn(2, 4, 8),
        )
        mock_llm.return_value = mock_hidden_states

        batch = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
            "correct": torch.tensor([1, 0])
        }

        # Mock trainer logging behavior
        model.log = Mock()

        outputs = model.validation_step(batch, 0)
        assert "val_loss" in outputs
        assert "confidence" in outputs
        assert "correct" in outputs
        assert outputs["confidence"].shape == (2,)

    def test_configure_optimizers(self, mock_llm):
        model = ConfidenceAugmentedLLM(llm=mock_llm, layer_idx=0, lr=1e-3)
        optimizer = model.configure_optimizers()

        assert isinstance(optimizer, torch.optim.AdamW)
        # Check that optimizer only tracks probe parameters
        param_groups = optimizer.param_groups
        assert len(param_groups) == 1
        params = param_groups[0]["params"]
        # Probe has weights and biases for the linear layer
        probe_params = list(model.probe.parameters())
        assert len(params) == len(probe_params)
        for p1, p2 in zip(params, probe_params):
            assert p1 is p2
