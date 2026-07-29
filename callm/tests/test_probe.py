import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, mock_open
from callm.models.probe import (
    Probe,
    ConfidenceAugmentedLLM,
    LayerWeightedAverage,
    SingleLayerSelector,
)


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
        model_dim = 4

        probe = Probe(model_dim=model_dim)
        # Set weight and bias to easily check output
        with torch.no_grad():
            probe.linear.weight.fill_(1.0)
            probe.linear.bias.fill_(0.0)

        # Activations with known values
        # activations shape: (1, 3, 4)
        activations = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]]]
        )
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
        mock_hidden_states = (torch.randn(1, 5, 8), torch.randn(1, 5, 8))
        mock_llm.return_value = (generated_ids, mock_hidden_states)

        input_ids = torch.tensor([[4, 5]])
        attention_mask = torch.tensor([[1, 1]])

        gen_ids, conf = model(input_ids, attention_mask, generate=True)

        assert torch.equal(gen_ids, generated_ids)
        assert conf.shape == (1,)
        # LLM should be called with generate=True and return_hidden_states=True
        mock_llm.assert_called_once_with(
            input_ids, attention_mask, generate=True, return_hidden_states=True
        )

    def test_forward_with_generate_false(self, mock_llm):
        model = ConfidenceAugmentedLLM(llm=mock_llm, layer_idx=0)

        # Mock LLM forward output: hidden_states tuple
        mock_hidden_states = (torch.randn(2, 4, 8),)
        mock_llm.return_value = mock_hidden_states

        input_ids = torch.tensor([[1, 2]])
        attention_mask = torch.tensor([[1, 1]])

        conf = model(input_ids, attention_mask, generate=False)

        assert conf.shape == (2,)
        # LLM should be called with generate=False and return_hidden_states=True
        mock_llm.assert_called_once_with(
            input_ids, attention_mask, generate=False, return_hidden_states=True
        )

    def test_training_step(self, mock_llm):
        model = ConfidenceAugmentedLLM(llm=mock_llm, layer_idx=0)

        # Mock LLM forward returning hidden states
        mock_hidden_states = (torch.randn(2, 4, 8),)
        mock_llm.return_value = mock_hidden_states

        batch = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
            "correct": torch.tensor([1, 0]),
        }

        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss

    def test_validation_step(self, mock_llm):
        model = ConfidenceAugmentedLLM(llm=mock_llm, layer_idx=0)

        mock_hidden_states = (torch.randn(2, 4, 8),)
        mock_llm.return_value = mock_hidden_states

        batch = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
            "correct": torch.tensor([1, 0]),
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


class TestLayerPooling:
    """Unit tests for LayerWeightedAverage and SingleLayerSelector."""

    def test_layer_weighted_average_4d(self):
        batch_size = 4
        num_layers = 5
        seq_len = 10
        model_dim = 16

        lwa = LayerWeightedAverage(num_layers=num_layers)
        activations = torch.randn(batch_size, num_layers, seq_len, model_dim)
        output = lwa(activations)

        # Output shape should be (batch_size, seq_len, model_dim)
        assert output.shape == (batch_size, seq_len, model_dim)
        # Weights should sum to 1.0 (softmax)
        weights = lwa.get_weights()
        assert weights.shape == (num_layers,)
        assert torch.allclose(weights.sum(), torch.tensor(1.0))

    def test_layer_weighted_average_3d(self):
        batch_size = 4
        num_layers = 5
        model_dim = 16

        lwa = LayerWeightedAverage(num_layers=num_layers)
        activations = torch.randn(batch_size, num_layers, model_dim)
        output = lwa(activations)

        # Output shape should be (batch_size, model_dim)
        assert output.shape == (batch_size, model_dim)

    def test_single_layer_selector(self):
        batch_size = 2
        num_layers = 3
        model_dim = 8

        selector = SingleLayerSelector(layer_idx=1, num_layers=num_layers)
        activations = torch.randn(batch_size, num_layers, model_dim)
        output = selector(activations)

        # Output shape should be (batch_size, model_dim)
        assert output.shape == (batch_size, model_dim)
        # Output should be equal to the second layer (index 1)
        assert torch.equal(output, activations[:, 1])

        # Test negative modulo selection
        selector_neg = SingleLayerSelector(layer_idx=-1, num_layers=num_layers)
        output_neg = selector_neg(activations)
        assert torch.equal(output_neg, activations[:, 2])


class TestProbeWithPooling:
    """Unit tests for Probe with integrated layer pooling modules."""

    def test_probe_with_weighted_average(self):
        batch_size = 3
        num_layers = 4
        model_dim = 8

        lwa = LayerWeightedAverage(num_layers=num_layers)
        probe = Probe(model_dim=model_dim, layer_pooling=lwa)
        activations = torch.randn(batch_size, num_layers, model_dim)
        confidence = probe(activations)

        assert confidence.shape == (batch_size,)
        assert torch.all(confidence >= 0.0)
        assert torch.all(confidence <= 1.0)
        assert probe.get_layer_weights() is not None
        assert probe.get_layer_weights().shape == (num_layers,)


class TestConfidenceAugmentedLLMMultiLayer:
    """Unit tests for multi-layer configurations of ConfidenceAugmentedLLM."""

    @pytest.fixture
    def mock_llm(self):
        mock_instance = Mock()
        mock_instance.model = Mock()
        mock_instance.model.config = Mock()
        mock_instance.model.config.hidden_size = 8
        mock_instance.parameters.return_value = iter([])
        return mock_instance

    def test_multi_layer_initialization_with_list(self, mock_llm):
        model = ConfidenceAugmentedLLM(llm=mock_llm, layer_idx=[0, 2], lr=1e-2)
        assert model.num_layers == 2
        assert isinstance(model.probe.layer_pooling, LayerWeightedAverage)

    def test_multi_layer_initialization_with_all_live(self, mock_llm):
        mock_llm.model.config.num_hidden_layers = 12
        model = ConfidenceAugmentedLLM(llm=mock_llm, layer_idx="all", lr=1e-2)
        assert model.num_layers == 12
        assert isinstance(model.probe.layer_pooling, LayerWeightedAverage)

    def test_multi_layer_forward_with_list(self, mock_llm):
        model = ConfidenceAugmentedLLM(llm=mock_llm, layer_idx=[0, 1])

        # Mock hidden states tuple: 2 layers, shape (1, 5, 8)
        mock_hidden_states = (torch.randn(1, 5, 8), torch.randn(1, 5, 8))
        mock_llm.return_value = mock_hidden_states

        input_ids = torch.tensor([[4, 5]])
        attention_mask = torch.tensor([[1, 1]])

        conf = model(input_ids, attention_mask, generate=False)
        assert conf.shape == (1,)


class TestActivationsDataModuleMultiLayer:
    """Unit tests for ActivationsDataModule multi-layer collation."""

    def test_collate_fn_multi_layer_pooled(self):
        from callm.data.activations_data import ActivationsDataModule

        batch = [
            {
                "activations": torch.randn(4, 16),
                "correct": True,
                "is_multi_layer": True,
            },
            {
                "activations": torch.randn(4, 16),
                "correct": False,
                "is_multi_layer": True,
            },
        ]

        collated = ActivationsDataModule.collate_fn(batch)
        assert "activations" in collated
        assert "correct" in collated
        # Expected shape: (batch_size, num_layers, model_dim) -> (2, 4, 16)
        assert collated["activations"].shape == (2, 4, 16)
        assert torch.equal(collated["correct"], torch.tensor([1, 0]))

    def test_collate_fn_multi_layer_sequence(self):
        from callm.data.activations_data import ActivationsDataModule

        batch = [
            {
                "activations": torch.randn(4, 5, 16),
                "correct": True,
                "is_multi_layer": True,
            },
            {
                "activations": torch.randn(4, 8, 16),
                "correct": False,
                "is_multi_layer": True,
            },
        ]

        collated = ActivationsDataModule.collate_fn(batch)
        assert "activations" in collated
        assert "correct" in collated
        # Expected shape: (batch_size, num_layers, max_seq_len, model_dim) -> (2, 4, 8, 16)
        assert collated["activations"].shape == (2, 4, 8, 16)
        # Verify padding on the first sample (length 5, padded to 8)
        assert torch.all(collated["activations"][0, :, 5:, :] == 0.0)


class TestActivationsDataModuleSlicing:
    """Unit tests for activations_to_use slicing in ActivationsDataModule."""

    def test_invalid_activations_to_use(self):
        from callm.data.activations_data import ActivationsDataModule

        with pytest.raises(ValueError, match="activations_to_use must be one of"):
            ActivationsDataModule(
                activations_dir="fake",
                evaluation_csv="fake.csv",
                activations_to_use="invalid_option",
            )

    @patch("os.path.exists")
    @patch("glob.glob")
    @patch("torch.load")
    def test_slicing_strategies(self, mock_torch_load, mock_glob, mock_exists):
        from callm.data.activations_data import ActivationsDataModule

        mock_exists.return_value = True
        mock_glob.return_value = ["fake_shard_0.pt"]

        # Mock evaluation CSV contents
        csv_data = "question,gold_answers,correct\nQ1,A1,yes\n"

        # Mock loaded activations (a single shard with 1 sample, prompt_length=3, total seq_len=5)
        # activations dictionary has 1 layer (0) with a tensor of shape (5, 8)
        mock_sample_tensor = torch.arange(40, dtype=torch.float32).view(5, 8)
        mock_torch_load.return_value = [
            {
                "sample_idx": 0,
                "seq_len": 5,
                "prompt_length": 3,
                "activations": {0: mock_sample_tensor},
            }
        ]

        # Strategy: prompt -> should slice :3
        with patch("builtins.open", mock_open(read_data=csv_data)):
            dm_prompt = ActivationsDataModule(
                activations_dir="fake",
                evaluation_csv="fake.csv",
                pool_strategy="none",
                activations_to_use="prompt",
                train_split=1.0,
            )
            dm_prompt.setup()
        assert dm_prompt.train_dataset[0]["activations"].shape == (3, 8)
        assert torch.equal(
            dm_prompt.train_dataset[0]["activations"], mock_sample_tensor[:3]
        )

        # Strategy: answer -> should slice 3:
        with patch("builtins.open", mock_open(read_data=csv_data)):
            dm_answer = ActivationsDataModule(
                activations_dir="fake",
                evaluation_csv="fake.csv",
                pool_strategy="none",
                activations_to_use="answer",
                train_split=1.0,
            )
            dm_answer.setup()
        assert dm_answer.train_dataset[0]["activations"].shape == (2, 8)
        assert torch.equal(
            dm_answer.train_dataset[0]["activations"], mock_sample_tensor[3:]
        )

        # Strategy: both -> should keep full sequence
        with patch("builtins.open", mock_open(read_data=csv_data)):
            dm_both = ActivationsDataModule(
                activations_dir="fake",
                evaluation_csv="fake.csv",
                pool_strategy="none",
                activations_to_use="both",
                train_split=1.0,
            )
            dm_both.setup()
        assert dm_both.train_dataset[0]["activations"].shape == (5, 8)
        assert torch.equal(dm_both.train_dataset[0]["activations"], mock_sample_tensor)


class TestGetIndicesOfExactAnswer:
    """Unit tests for get_indices_of_exact_answer."""

    def test_exact_answer_token_shrinking(self):
        from callm.utils import get_indices_of_exact_answer

        class DummyTokenizer:
            def decode(self, ids):
                table = {
                    1: "What ",
                    2: "is ",
                    3: "it?",
                    4: " The ",
                    5: "capital ",
                    6: "is ",
                    7: "Paris",
                    8: ".",
                }
                return "".join(table.get(i, "") for i in ids)

        tokenizer = DummyTokenizer()
        input_output_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        prompt_len = 3  # IDs [1, 2, 3] are prompt

        indices = get_indices_of_exact_answer(
            tokenizer, input_output_ids, "Paris", prompt_len
        )
        assert indices == [6]  # 0-indexed: index 6 corresponds to ID 7 ("Paris")

    def test_missing_exact_answer_returns_none(self):
        from callm.utils import get_indices_of_exact_answer

        class DummyTokenizer:
            def decode(self, ids):
                return "The capital is Paris."

        tokenizer = DummyTokenizer()
        indices = get_indices_of_exact_answer(
            tokenizer, [1, 2, 3, 4], "London", prompt_len=2
        )
        assert indices is None


class TestSelectedTokensActivationsDataModule:
    """Unit tests for SelectedTokensActivationsDataModule retention and window slicing."""

    @patch("os.path.exists")
    @patch("glob.glob")
    @patch("torch.load")
    @patch("callm.utils.get_tokenizer_for_model")
    def test_100_percent_retention_and_window_strategies(
        self, mock_get_tokenizer, mock_torch_load, mock_glob, mock_exists
    ):
        from callm.data.activations_data import SelectedTokensActivationsDataModule

        mock_exists.return_value = True
        mock_glob.return_value = ["fake_shard_0.pt"]

        class DummyTokenizer:
            def decode(self, ids):
                return "My Fair Lady"

            def encode(self, text, add_special_tokens=True):
                return [1, 2]

            def apply_chat_template(self, conversation, **kwargs):
                return [1, 2, 3]

        mock_get_tokenizer.return_value = DummyTokenizer()

        csv_data = (
            "question,gold_answers,correct,exact_answer,valid_exact_answer,raw_output\n"
            "Q1,['My Fair Lady'],yes,My Fair Lady,Yes,My Fair Lady\n"
            "Q2,['London'],no,NO ANSWER,No,Some long incorrect answer\n"
        )

        mock_sample_tensor_1 = torch.randn(6, 8)
        mock_sample_tensor_2 = torch.randn(6, 8)

        mock_torch_load.return_value = [
            {
                "sample_idx": 0,
                "seq_len": 6,
                "prompt_length": 3,
                "activations": {0: mock_sample_tensor_1},
            },
            {
                "sample_idx": 1,
                "seq_len": 6,
                "prompt_length": 3,
                "activations": {0: mock_sample_tensor_2},
            },
        ]

        with patch("builtins.open", mock_open(read_data=csv_data)):
            dm = SelectedTokensActivationsDataModule(
                activations_dir="fake",
                evaluation_csv="fake.csv",
                pool_strategy="none",
                activations_to_use="exact_answer_first_token",
                train_split=1.0,
                model_name="dummy",
            )
            dm.setup()

        # Check 100% sample retention: both samples must be retained
        assert len(dm.train_dataset) == 2
