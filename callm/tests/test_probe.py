import pytest
import csv
import functools
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, mock_open
from callm.models.probe import Probe
from callm.models.confidence_llm import ConfidenceAugmentedLLM
from callm.models.layer_pooling import LayerWeightedAverage, SingleLayerSelector


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

    def test_probe_masked_mean_ignores_padding(self):
        """Without seq_pooling, padded positions must not enter the mean."""
        torch.manual_seed(0)
        probe = Probe(model_dim=4)
        lengths = [5, 2]
        samples = [torch.randn(n, 4) for n in lengths]
        padded = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True)
        seq_mask = torch.arange(padded.shape[1]) < torch.tensor(lengths)[:, None]

        batched = probe(padded, seq_mask=seq_mask)
        unbatched = torch.cat([probe(s.unsqueeze(0)) for s in samples])
        assert torch.allclose(batched, unbatched)


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

    @pytest.mark.parametrize("pool_strategy", ["mean", "last", "none"])
    def test_multi_layer_on_the_fly(self, mock_llm, pool_strategy):
        model = ConfidenceAugmentedLLM(
            llm=mock_llm, layer_idx=[0, 1], pool_strategy=pool_strategy
        )
        mock_llm.return_value = (torch.randn(2, 5, 8), torch.randn(2, 5, 8))
        batch = {
            "input_ids": torch.zeros(2, 5, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]),
            "prompt_lengths": torch.tensor([2, 3]),
        }

        conf = model._get_confidence(batch)
        assert conf.shape == (2,)

    def test_multi_layer_on_the_fly_mean_matches_none(self, mock_llm):
        """Mean pooling via pool_strategy='mean' and via the probe's fallback agree."""
        mock_llm.return_value = (torch.randn(2, 5, 8), torch.randn(2, 5, 8))
        batch = {
            "input_ids": torch.zeros(2, 5, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]),
            "prompt_lengths": torch.tensor([2, 3]),
        }
        mean_model = ConfidenceAugmentedLLM(
            llm=mock_llm, layer_idx=[0, 1], pool_strategy="mean"
        )
        none_model = ConfidenceAugmentedLLM(
            llm=mock_llm, layer_idx=[0, 1], pool_strategy="none"
        )
        none_model.probe.load_state_dict(mean_model.probe.state_dict())

        assert torch.allclose(
            mean_model._get_confidence(batch), none_model._get_confidence(batch)
        )


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


class TestActivationsDataModuleSeparateVal:
    """Unit tests for training/validating on separate activation directories."""

    @staticmethod
    def _write_split(root, name, labels):
        split_dir = root / name
        split_dir.mkdir()
        torch.save(
            [
                {
                    "sample_idx": i,
                    "seq_len": 1,
                    "prompt_length": 1,
                    "activations": {0: torch.full((1, 4), float(i))},
                }
                for i in range(len(labels))
            ],
            split_dir / "activations_0000.pt",
        )
        csv_path = root / f"{name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "Correct"])
            writer.writerows([[f"Q{i}", label] for i, label in enumerate(labels)])
        return str(split_dir), str(csv_path)

    def test_uses_all_train_and_all_val_samples(self, tmp_path):
        from callm.data.activations_data import ActivationsDataModule

        train_dir, train_csv = self._write_split(tmp_path, "train", ["Yes"] * 5)
        val_dir, val_csv = self._write_split(tmp_path, "val", ["No", "No", "Yes"])

        dm = ActivationsDataModule(
            activations_dir=train_dir,
            evaluation_csv=train_csv,
            val_activations_dir=val_dir,
            val_evaluation_csv=val_csv,
            layer_idx=0,
            activations_to_use="both",
            num_workers=0,
        )
        dm.setup()

        assert len(dm.train_dataset) == 5
        assert all(s["correct"] for s in dm.train_dataset.samples)
        assert [s["correct"] for s in dm.val_dataset.samples] == [False, False, True]
        assert torch.equal(dm.val_dataset[2]["activations"], torch.full((4,), 2.0))

    def test_max_samples_subsamples_train_only(self, tmp_path):
        from callm.data.activations_data import ActivationsDataModule

        train_dir, train_csv = self._write_split(tmp_path, "train", ["Yes"] * 5)
        val_dir, val_csv = self._write_split(tmp_path, "val", ["No"] * 3)

        dm = ActivationsDataModule(
            activations_dir=train_dir,
            evaluation_csv=train_csv,
            val_activations_dir=val_dir,
            val_evaluation_csv=val_csv,
            layer_idx=0,
            max_samples=2,
            num_workers=0,
        )
        dm.setup()

        assert len(dm.train_dataset) == 2
        assert len(dm.val_dataset) == 3

    def test_requires_both_val_args(self):
        from callm.data.activations_data import ActivationsDataModule

        with pytest.raises(ValueError, match="must be set together"):
            ActivationsDataModule(
                activations_dir="fake",
                evaluation_csv="fake.csv",
                val_activations_dir="fake_val",
            )


class TestActivationsDataModuleLazy:
    """Unit tests for lazy (memory-mapped) loading of activation shards."""

    @staticmethod
    def _write_split(root, name, num_samples, num_shards=2):
        split_dir = root / name
        split_dir.mkdir()
        gen = torch.Generator().manual_seed(0)
        samples = []
        for i in range(num_samples):
            seq_len = 3 + i % 4
            samples.append(
                {
                    "sample_idx": i,
                    "seq_len": seq_len,
                    "prompt_length": seq_len - 2,
                    "activations": {
                        layer: torch.randn(seq_len, 4, generator=gen).bfloat16()
                        for layer in (12, 14)
                    },
                }
            )
        per_shard = -(-num_samples // num_shards)
        for s in range(num_shards):
            torch.save(
                samples[s * per_shard : (s + 1) * per_shard],
                split_dir / f"activations_{s:04d}.pt",
            )
        csv_path = root / f"{name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "Correct"])
            writer.writerows(
                [[f"Q{i}", "Yes" if i % 3 else "No"] for i in range(num_samples)]
            )
        return str(split_dir), str(csv_path)

    @staticmethod
    def _items(dataset):
        return [dataset[i] for i in range(len(dataset))]

    @pytest.mark.parametrize(
        "pool_strategy,activations_to_use,layer_idx",
        [("none", "both", 14), ("mean", "answer", 12), ("none", "prompt", [12, 14])],
    )
    def test_lazy_matches_eager(
        self, tmp_path, pool_strategy, activations_to_use, layer_idx
    ):
        from callm.data.activations_data import (
            ActivationsDataModule,
            LazyActivationsDataset,
        )

        train_dir, train_csv = self._write_split(tmp_path, "train", 10)
        val_dir, val_csv = self._write_split(tmp_path, "val", 5)
        kwargs = dict(
            activations_dir=train_dir,
            evaluation_csv=train_csv,
            val_activations_dir=val_dir,
            val_evaluation_csv=val_csv,
            layer_idx=layer_idx,
            pool_strategy=pool_strategy,
            activations_to_use=activations_to_use,
            dtype="bfloat16",
            max_samples=7,
            num_workers=0,
        )
        eager = ActivationsDataModule(**kwargs)
        lazy = ActivationsDataModule(**kwargs, lazy=True)
        eager.setup()
        lazy.setup()

        assert isinstance(lazy.train_dataset, LazyActivationsDataset)
        assert lazy.layer_stats is None
        for split in ("train_dataset", "val_dataset"):
            expected = self._items(getattr(eager, split))
            actual = self._items(getattr(lazy, split))
            assert len(actual) == len(expected)
            for a, e in zip(actual, expected):
                assert a["correct"] == e["correct"]
                assert a["is_multi_layer"] == e["is_multi_layer"]
                assert torch.equal(a["activations"], e["activations"])

    def test_lazy_ratio_split_and_collate(self, tmp_path):
        from callm.data.activations_data import ActivationsDataModule

        train_dir, train_csv = self._write_split(tmp_path, "train", 10)
        kwargs = dict(
            activations_dir=train_dir,
            evaluation_csv=train_csv,
            layer_idx=12,
            pool_strategy="none",
            activations_to_use="both",
            train_split=0.6,
            num_workers=0,
        )
        eager = ActivationsDataModule(**kwargs)
        lazy = ActivationsDataModule(**kwargs, lazy=True)
        eager.setup()
        lazy.setup()

        assert len(lazy.train_dataset) == 6 and len(lazy.val_dataset) == 4
        lazy_batch = next(iter(lazy.val_dataloader()))
        eager_batch = next(iter(eager.val_dataloader()))
        assert torch.equal(lazy_batch["activations"], eager_batch["activations"])
        assert torch.equal(lazy_batch["seq_mask"], eager_batch["seq_mask"])
        assert torch.equal(lazy_batch["correct"], eager_batch["correct"])


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


class TestConfidenceAugmentedLLMLiveEvaluation:
    """Tests for the live evaluation path (generate → probe → metrics)."""

    def test_validation_writes_outputs_and_metrics(self, tmp_path):
        from callm.models.llm import LLM

        mock_llm = Mock()
        mock_llm.is_seq2seq = False
        mock_llm.strip_prompt = functools.partial(LLM.strip_prompt, mock_llm)
        mock_llm.tokenizer.decode.side_effect = lambda ids, **_: " ".join(
            str(int(t)) for t in ids
        )
        mock_llm.extractor.extract_answer.side_effect = ["Paris", "Rome"]

        prompt = torch.tensor([[1, 2], [3, 4]])
        generated = torch.tensor([[1, 2, 7], [3, 4, 9]])
        mock_llm.return_value = (generated, (torch.randn(2, 2, 8),))

        model = ConfidenceAugmentedLLM(llm=mock_llm, model_dim=8, layer_idx=0)
        model._trainer = Mock(log_dir=str(tmp_path))
        model.log = Mock()

        model.on_validation_epoch_start()
        model.validation_step(
            {
                "input_ids": prompt,
                "attention_mask": torch.ones_like(prompt),
                "question": ["q1", "q2"],
                "gold_answers": [["Paris"], ["Madrid"]],
            },
            0,
        )
        # Only the generated tokens (prompt stripped) are decoded
        decoded = [c.args[0].tolist() for c in mock_llm.tokenizer.decode.call_args_list]
        assert decoded == [[7], [9]]

        model.on_validation_epoch_end()

        logged = {c.args[0]: c.args[1] for c in model.log.call_args_list}
        assert logged["val_accuracy"] == 0.5
        assert {"val_ece", "val_brier_score", "val_auc", "val_ecuas_0"} <= logged.keys()

        with open(tmp_path / "llm_outputs.csv") as f:
            rows = list(csv.reader(f))
        assert rows[0] == [
            "question",
            "gold_answers",
            "pred_answer",
            "confidence",
            "correct",
            "raw_output",
        ]
        assert [r[4] for r in rows[1:]] == ["Yes", "No"]
        with open(tmp_path / "metrics.csv") as f:
            assert next(csv.reader(f))[0] == "val_accuracy"
