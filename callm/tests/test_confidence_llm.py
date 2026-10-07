"""
Characterization tests for ``ConfidenceAugmentedLLM`` hooks and helpers that
are not exercised by ``test_probe.py``.
"""

import json
from unittest.mock import Mock

import pytest
import torch
from ecuas import (
    ConfidenceAUCScore,
    ConfidenceCrossEntropy,
    ConfidenceECUAS,
    ExpectedCalibrationError,
)

from callm.models.layer_pooling import (
    GlobalZNormWeightedAverage,
    LayerWeightedAverage,
    LayerwiseZNormWeightedAverage,
    SingleLayerSelector,
)
from callm.models.confidence_llm import ConfidenceAugmentedLLM
from callm.models.probe import Probe

DIM = 4


def _hidden_states(batch=2, seq=5, layers=3):
    torch.manual_seed(0)
    return tuple(torch.randn(batch, seq, DIM) for _ in range(layers))


def _mock_llm(**config):
    llm = Mock()
    llm.model.config = Mock(spec=list(config), **config)
    llm.parameters.return_value = iter([])
    return llm


class TestInit:
    def test_model_dim_resolution(self):
        assert ConfidenceAugmentedLLM(model_dim=3).model_dim == 3
        assert ConfidenceAugmentedLLM(llm=_mock_llm(hidden_size=5)).model_dim == 5
        assert ConfidenceAugmentedLLM(llm=_mock_llm(d_model=6)).model_dim == 6
        # Explicit model_dim wins over the LLM config
        assert (
            ConfidenceAugmentedLLM(llm=_mock_llm(hidden_size=5), model_dim=2).model_dim
            == 2
        )
        with pytest.raises(ValueError, match="Could not determine model dimension"):
            ConfidenceAugmentedLLM(llm=_mock_llm())
        with pytest.raises(ValueError, match="Either llm or model_dim"):
            ConfidenceAugmentedLLM()

    def test_default_layer_pooling(self):
        model = ConfidenceAugmentedLLM(model_dim=DIM)
        assert model.probe.layer_pooling is None
        assert model.num_layers is None

        model = ConfidenceAugmentedLLM(model_dim=DIM, layer_idx=[1, 2])
        assert isinstance(model.probe.layer_pooling, LayerWeightedAverage)
        assert model.num_layers == 2

        model = ConfidenceAugmentedLLM(model_dim=DIM, layer_idx="ALL", num_layers=7)
        assert model.probe.layer_pooling.num_layers == 7

        model = ConfidenceAugmentedLLM(
            llm=_mock_llm(hidden_size=DIM, n_layer=9), layer_idx="all"
        )
        assert model.num_layers == 9

        with pytest.raises(ValueError, match="num_layers"):
            ConfidenceAugmentedLLM(model_dim=DIM, layer_idx="all")

    def test_explicit_layer_pooling_sets_num_layers(self):
        pooling = LayerWeightedAverage(num_layers=5)
        model = ConfidenceAugmentedLLM(model_dim=DIM, layer_pooling=pooling)
        assert model.probe.layer_pooling is pooling
        assert model.num_layers == 5

        model = ConfidenceAugmentedLLM(
            model_dim=DIM, layer_pooling=SingleLayerSelector(1), num_layers=3
        )
        assert model.num_layers == 3

    def test_explicit_probe(self):
        probe = Probe(DIM)
        model = ConfidenceAugmentedLLM(model_dim=DIM, probe=probe)
        assert model.probe is probe


class TestSelectLayerActivations:
    @pytest.mark.parametrize(
        "kwargs,expected_layers",
        [
            ({"layer_idx": 1}, 1),
            ({"layer_idx": -1}, 2),
            ({"layer_idx": [0, -1]}, [0, 2]),
            ({"layer_idx": "all", "num_layers": 3}, [0, 1, 2]),
            ({"layer_idx": 1, "layer_pooling": SingleLayerSelector(1)}, [0, 1, 2]),
            (
                {"layer_idx": [0, 2], "layer_pooling": SingleLayerSelector(0)},
                [0, 1, 2],
            ),
            (
                {"layer_idx": 0, "layer_pooling": LayerWeightedAverage(num_layers=3)},
                [0, 1, 2],
            ),
            ({"layer_idx": "all", "num_layers": 3, "probe": Probe(DIM)}, [0, 1, 2]),
        ],
    )
    def test_selection(self, kwargs, expected_layers):
        hidden_states = _hidden_states()
        model = ConfidenceAugmentedLLM(model_dim=DIM, **kwargs)
        selected = model._select_layer_activations(hidden_states)
        if isinstance(expected_layers, int):
            assert selected is hidden_states[expected_layers]
        else:
            expected = torch.stack([hidden_states[i] for i in expected_layers], dim=1)
            assert torch.equal(selected, expected)


class TestOnTheFlyPooling:
    BATCH = {
        "input_ids": torch.zeros(2, 5, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]),
        "prompt_lengths": torch.tensor([2, 4]),
    }

    def _model(self, **kwargs):
        llm = _mock_llm(hidden_size=DIM)
        llm.return_value = _hidden_states()
        return ConfidenceAugmentedLLM(llm=llm, layer_idx=1, **kwargs)

    @pytest.mark.parametrize(
        "activations_to_use,slices",
        [
            ("answer", [slice(2, 5), slice(5, 5)]),
            ("prompt", [slice(0, 2), slice(1, 5)]),
            ("both", [slice(0, 5), slice(1, 5)]),
        ],
    )
    @pytest.mark.parametrize("pool_strategy", ["mean", "last", "none"])
    def test_matches_manual_pooling(self, pool_strategy, activations_to_use, slices):
        model = self._model(
            pool_strategy=pool_strategy, activations_to_use=activations_to_use
        )
        layer = _hidden_states()[1]
        pooled = []
        for i, s in enumerate(slices):
            sample = layer[i, s]
            if sample.shape[0] == 0:
                sample = torch.zeros(1, DIM)
            pooled.append(sample[-1] if pool_strategy == "last" else sample.mean(0))
        expected = model.probe.classify(torch.stack(pooled))

        assert torch.allclose(model._get_confidence(self.BATCH), expected, atol=1e-6)

    def test_unknown_pool_strategy(self):
        with pytest.raises(ValueError, match="pool_strategy"):
            self._model(pool_strategy="max")._extract_and_pool_activations(
                _hidden_states(),
                self.BATCH["prompt_lengths"],
                self.BATCH["attention_mask"],
            )

    @pytest.mark.parametrize("pool_strategy", ["mean", "none"])
    def test_prepooled_hidden_states_skip_slicing(self, pool_strategy):
        llm = _mock_llm(hidden_size=DIM)
        llm.return_value = tuple(torch.randn(2, DIM) for _ in range(3))
        model = ConfidenceAugmentedLLM(
            llm=llm, layer_idx=[0, 1], pool_strategy=pool_strategy
        )
        expected = model.probe.classify(
            model.probe.layer_pooling(torch.stack(llm.return_value[:2], dim=1))
        )
        assert torch.allclose(model._get_confidence(self.BATCH), expected)


class TestForward:
    def test_requires_llm(self):
        model = ConfidenceAugmentedLLM(model_dim=DIM)
        with pytest.raises(RuntimeError, match="requires an LLM"):
            model(torch.zeros(1, 1), torch.ones(1, 1))

    def test_generate_without_hidden_states_reruns_on_full_sequence(self):
        llm = _mock_llm(hidden_size=DIM)
        llm.is_seq2seq = False
        prompt = torch.tensor([[1, 2], [3, 4]])
        mask = torch.tensor([[0, 1], [1, 1]])
        sequences = torch.tensor([[1, 2, 7, 8], [3, 4, 9, 0]])
        hidden = _hidden_states(seq=4)
        llm.side_effect = [Mock(sequences=sequences), hidden]
        model = ConfidenceAugmentedLLM(llm=llm, layer_idx=0)

        generated, confidence = model(prompt, mask)

        assert generated is sequences
        rerun = llm.call_args_list[1]
        assert torch.equal(rerun.args[0], sequences)
        assert torch.equal(rerun.args[1], torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]))
        assert rerun.kwargs == {"generate": False, "return_hidden_states": True}
        assert torch.allclose(confidence, model.probe(hidden[0]))

    def test_generate_seq2seq_reruns_on_prompt(self):
        llm = _mock_llm(hidden_size=DIM)
        llm.is_seq2seq = True
        prompt, mask = torch.tensor([[1, 2]]), torch.tensor([[1, 1]])
        llm.side_effect = [torch.tensor([[0, 5]]), _hidden_states(batch=1, seq=2)]
        model = ConfidenceAugmentedLLM(llm=llm, layer_idx=0)

        model(prompt, mask)

        assert llm.call_args_list[1].args[0] is prompt
        assert llm.call_args_list[1].args[1] is mask


class TestTrainingHooks:
    def _attach(self, model, tmp_path, global_step=0, datamodule=None):
        model._trainer = Mock(
            log_dir=str(tmp_path),
            global_step=global_step,
            optimizers=[],
            datamodule=datamodule,
        )
        model.log = Mock()

    def test_weight_snapshots_written_on_fit_end(self, tmp_path):
        model = ConfidenceAugmentedLLM(model_dim=DIM, layer_idx=[3, 5])
        batch = {"activations": torch.randn(2, 2, DIM), "correct": torch.tensor([1, 0])}
        for step in [0, 1, 50, 100]:
            self._attach(model, tmp_path, global_step=step)
            model.training_step(batch, step)

        model.on_fit_end()

        data = json.loads((tmp_path / "layer_weights_history.json").read_text())
        weights = model.probe.layer_pooling.get_weights().tolist()
        assert data == {
            "layer_indices": [3, 5],
            "snapshots": [{"step": s, "weights": weights} for s in [0, 50, 100]],
        }

    def test_weight_snapshots_layer_indices_for_all(self, tmp_path):
        model = ConfidenceAugmentedLLM(model_dim=DIM, layer_idx="all", num_layers=3)
        self._attach(model, tmp_path)
        model.training_step(
            {"activations": torch.randn(2, 3, DIM), "correct": torch.tensor([1, 0])}, 0
        )
        model.on_fit_end()
        data = json.loads((tmp_path / "layer_weights_history.json").read_text())
        assert data["layer_indices"] == [0, 1, 2]

    def test_no_snapshots_without_layer_weights(self, tmp_path):
        model = ConfidenceAugmentedLLM(model_dim=DIM)
        self._attach(model, tmp_path)
        model.training_step(
            {"activations": torch.randn(2, DIM), "correct": torch.tensor([1, 0])}, 0
        )
        model.on_fit_end()
        assert not (tmp_path / "layer_weights_history.json").exists()

    def test_training_step_logs_lr(self, tmp_path):
        model = ConfidenceAugmentedLLM(model_dim=DIM, lr=0.25)
        self._attach(model, tmp_path)
        model._trainer.optimizers = [model.configure_optimizers()]
        model.training_step(
            {"activations": torch.randn(2, DIM), "correct": torch.tensor([1, 0])}, 0
        )
        logged = {c.args[0]: c.args[1] for c in model.log.call_args_list}
        assert logged["lr"] == 0.25

    @pytest.mark.parametrize(
        "pooling_cls,stats,buffers",
        [
            (
                GlobalZNormWeightedAverage,
                {
                    "global_mean": torch.full((DIM,), 2.0),
                    "global_std": torch.full((DIM,), 3.0),
                },
                ("global_mean", "global_std"),
            ),
            (
                LayerwiseZNormWeightedAverage,
                {
                    "layer_means": torch.full((2, DIM), 2.0),
                    "layer_stds": torch.full((2, DIM), 3.0),
                },
                ("layer_means", "layer_stds"),
            ),
        ],
    )
    def test_znorm_stats_injected(self, tmp_path, pooling_cls, stats, buffers):
        pooling = pooling_cls(num_layers=2, model_dim=DIM)
        model = ConfidenceAugmentedLLM(model_dim=DIM, layer_pooling=pooling)
        self._attach(model, tmp_path, datamodule=Mock(layer_stats=stats))

        model.on_fit_start()

        for buffer, key in zip(buffers, stats):
            assert torch.equal(getattr(pooling, buffer), stats[key])
        assert pooling._stats_set

    def test_znorm_without_stats_or_datamodule(self, tmp_path):
        pooling = GlobalZNormWeightedAverage(num_layers=2, model_dim=DIM)
        model = ConfidenceAugmentedLLM(model_dim=DIM, layer_pooling=pooling)
        self._attach(model, tmp_path, datamodule=Mock(layer_stats=None))
        model.on_fit_start()
        self._attach(model, tmp_path, datamodule=None)
        model.on_fit_start()
        assert not pooling._stats_set

    def test_on_fit_start_ignores_plain_pooling(self, tmp_path):
        model = ConfidenceAugmentedLLM(model_dim=DIM, layer_idx=[0, 1])
        self._attach(model, tmp_path, datamodule=Mock(layer_stats={}))
        model.on_fit_start()  # no set_normalization_stats → no-op


class TestCachedValidationMetrics:
    def test_logs_probe_metrics_and_resets(self, tmp_path):
        torch.manual_seed(0)
        model = ConfidenceAugmentedLLM(model_dim=DIM)
        model.log = Mock()
        model.on_validation_epoch_start()
        correct = torch.tensor([1, 0, 1, 1])
        activations = torch.randn(4, DIM)
        model.validation_step(
            {"activations": activations[:2], "correct": correct[:2]}, 0
        )
        model.validation_step(
            {"activations": activations[2:], "correct": correct[2:]}, 1
        )
        model.log.reset_mock()

        model.on_validation_epoch_end()

        confidences = model.probe(activations).detach()
        expected = {}
        for name, metric in {
            "val_ece": ExpectedCalibrationError(n_bins=10),
            "val_cross_entropy": ConfidenceCrossEntropy(),
            "val_auc": ConfidenceAUCScore(),
            "val_ecuas_0": ConfidenceECUAS(n=0),
            "val_ecuas_1": ConfidenceECUAS(n=1),
            "val_ecuas_128": ConfidenceECUAS(n=128),
        }.items():
            metric.update(confidences, correct.float())
            expected[name] = float(metric.compute())
        logged = {c.args[0]: float(c.args[1]) for c in model.log.call_args_list}
        assert logged == pytest.approx(expected)
        assert model.val_confidences == [] and model.val_correctness == []

    def test_no_outputs_logs_nothing(self):
        model = ConfidenceAugmentedLLM(model_dim=DIM)
        model.log = Mock()
        model.on_validation_epoch_end()
        model.log.assert_not_called()


class TestCheckpoints:
    def test_save_drops_llm_weights(self):
        model = ConfidenceAugmentedLLM(model_dim=DIM)
        checkpoint = {"state_dict": {"llm.w": 1, "probe.linear.weight": 2}}
        model.on_save_checkpoint(checkpoint)
        assert checkpoint == {"state_dict": {"probe.linear.weight": 2}}

    @pytest.mark.parametrize("wrapped", [True, False])
    def test_load_probe_checkpoint(self, tmp_path, wrapped):
        trained = ConfidenceAugmentedLLM(model_dim=DIM, layer_idx=[0, 1])
        state = {f"probe.{k}": v for k, v in trained.probe.state_dict().items()}
        state["llm.ignored"] = torch.zeros(1)
        path = tmp_path / "probe.ckpt"
        torch.save({"state_dict": state} if wrapped else state, path)

        model = ConfidenceAugmentedLLM(
            model_dim=DIM, layer_idx=[0, 1], probe_checkpoint=str(path)
        )

        for k, v in trained.probe.state_dict().items():
            assert torch.equal(model.probe.state_dict()[k], v)
        assert not model.probe.training
        assert not any(p.requires_grad for p in model.probe.parameters())

    def test_load_probe_checkpoint_without_probe_keys(self, tmp_path):
        path = tmp_path / "bad.ckpt"
        torch.save({"state_dict": {"llm.w": torch.zeros(1)}}, path)
        with pytest.raises(ValueError, match="No 'probe.\\*' keys"):
            ConfidenceAugmentedLLM(model_dim=DIM, probe_checkpoint=str(path))


class TestConfigureOptimizers:
    def test_with_scheduler(self):
        scheduler = Mock(return_value="scheduler")
        model = ConfidenceAugmentedLLM(model_dim=DIM, lr_scheduler=scheduler)
        model._trainer = Mock(estimated_stepping_batches=123)

        config = model.configure_optimizers()

        assert config["lr_scheduler"] == {"scheduler": "scheduler", "interval": "step"}
        scheduler.assert_called_once_with(config["optimizer"], max_steps=123)
