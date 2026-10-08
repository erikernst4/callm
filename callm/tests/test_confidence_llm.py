"""
Characterization tests for the ``ConfidenceAugmentedLLM`` classes' hooks and helpers that
are not exercised by ``test_probe.py``.
"""

import csv
import functools
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

from callm.callbacks import LayerWeightsHistoryCallback
from callm.models.layer_pooling import (
    GlobalZNormWeightedAverage,
    LayerWeightedAverage,
    LayerwiseZNormWeightedAverage,
    SingleLayerSelector,
)
from callm.models.confidence_llm import (
    CachedActivationsConfidenceAugmentedLLM,
    ConfidenceAugmentedLLM,
    LiveEvalConfidenceAugmentedLLM,
)
from callm.models.llm import LLM
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
        assert CachedActivationsConfidenceAugmentedLLM(model_dim=3).model_dim == 3
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
            ConfidenceAugmentedLLM(llm=None)

    def test_default_layer_pooling(self):
        model = CachedActivationsConfidenceAugmentedLLM(model_dim=DIM)
        assert model.probe.layer_pooling is None
        assert model.num_layers is None

        model = CachedActivationsConfidenceAugmentedLLM(model_dim=DIM, layer_idx=[1, 2])
        assert isinstance(model.probe.layer_pooling, LayerWeightedAverage)
        assert model.num_layers == 2

        model = CachedActivationsConfidenceAugmentedLLM(
            model_dim=DIM, layer_idx="ALL", num_layers=7
        )
        assert model.probe.layer_pooling.num_layers == 7

        model = ConfidenceAugmentedLLM(
            llm=_mock_llm(hidden_size=DIM, n_layer=9), layer_idx="all"
        )
        assert model.num_layers == 9

        with pytest.raises(ValueError, match="num_layers"):
            CachedActivationsConfidenceAugmentedLLM(model_dim=DIM, layer_idx="all")

    def test_explicit_layer_pooling_sets_num_layers(self):
        pooling = LayerWeightedAverage(num_layers=5)
        model = CachedActivationsConfidenceAugmentedLLM(
            model_dim=DIM, layer_pooling=pooling
        )
        assert model.probe.layer_pooling is pooling
        assert model.num_layers == 5

        model = CachedActivationsConfidenceAugmentedLLM(
            model_dim=DIM, layer_pooling=SingleLayerSelector(1), num_layers=3
        )
        assert model.num_layers == 3

    def test_explicit_probe(self):
        probe = Probe(DIM)
        model = CachedActivationsConfidenceAugmentedLLM(model_dim=DIM, probe=probe)
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
        model = CachedActivationsConfidenceAugmentedLLM(model_dim=DIM, **kwargs)
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
            self._model(pool_strategy="max")

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


class TestInvalidBatch:
    def test_token_model_rejects_activation_batches(self):
        model = ConfidenceAugmentedLLM(llm=_mock_llm(hidden_size=DIM))
        with pytest.raises(
            ValueError, match="prompt_lengths.*CachedActivationsConfidenceAugmentedLLM"
        ):
            model.training_step(
                {"activations": torch.randn(1, DIM), "correct": torch.tensor([1])}, 0
            )

    def test_cached_model_rejects_token_batches(self):
        model = CachedActivationsConfidenceAugmentedLLM(model_dim=DIM)
        with pytest.raises(ValueError, match="'activations'.*ConfidenceAugmentedLLM"):
            model.training_step(
                {
                    "input_ids": torch.zeros(1, 2, dtype=torch.long),
                    "attention_mask": torch.ones(1, 2),
                    "prompt_lengths": torch.tensor([1]),
                    "correct": torch.tensor([1]),
                },
                0,
            )


class TestLiveEval:
    @staticmethod
    def _llm(is_seq2seq=False):
        llm = _mock_llm(hidden_size=DIM)
        llm.is_seq2seq = is_seq2seq
        llm.strip_prompt = functools.partial(LLM.strip_prompt, llm)
        llm.full_sequence = functools.partial(LLM.full_sequence, llm)
        llm.eos_token_ids = []
        return llm

    @staticmethod
    def _checkpoint(tmp_path, **kwargs):
        trained = CachedActivationsConfidenceAugmentedLLM(model_dim=DIM, **kwargs)
        path = tmp_path / "probe.ckpt"
        torch.save(
            {
                "state_dict": {
                    f"probe.{k}": v for k, v in trained.probe.state_dict().items()
                }
            },
            path,
        )
        return str(path)

    @pytest.mark.parametrize(
        "activations_to_use,slices",
        [
            ("answer", [slice(2, 4), slice(2, 4)]),
            ("prompt", [slice(1, 2), slice(0, 2)]),
            ("both", [slice(1, 4), slice(0, 4)]),
        ],
    )
    @pytest.mark.parametrize("pool_strategy", ["mean", "last", "none"])
    def test_scores_prompt_plus_answer_like_training(
        self, tmp_path, activations_to_use, slices, pool_strategy
    ):
        llm = self._llm()
        prompt = torch.tensor([[0, 2], [3, 4]])
        mask = torch.tensor([[0, 1], [1, 1]])  # first prompt is left-padded
        sequences = torch.tensor([[0, 2, 7, 8], [3, 4, 9, 6]])
        hidden = _hidden_states(seq=4)
        llm.side_effect = [Mock(sequences=sequences), hidden]
        model = LiveEvalConfidenceAugmentedLLM(
            llm=llm,
            probe_checkpoint=self._checkpoint(tmp_path, layer_idx=1),
            layer_idx=1,
            activations_to_use=activations_to_use,
            pool_strategy=pool_strategy,
        )

        generated, confidence = model._generate_and_score(prompt, mask)

        assert generated is sequences
        # Hidden states come from a second pass over [prompt + answer]
        rerun = llm.call_args_list[1]
        assert torch.equal(rerun.args[0], sequences)
        assert torch.equal(rerun.args[1], torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]))
        assert rerun.kwargs == {"generate": False}
        pooled = [
            hidden[1][i, s][-1] if pool_strategy == "last" else hidden[1][i, s].mean(0)
            for i, s in enumerate(slices)
        ]
        expected = model.probe.classify(torch.stack(pooled))
        assert torch.allclose(confidence, expected, atol=1e-6)

    def test_seq2seq_scores_encoder_input(self, tmp_path):
        llm = self._llm(is_seq2seq=True)
        prompt, mask = torch.tensor([[1, 2]]), torch.tensor([[1, 1]])
        llm.side_effect = [torch.tensor([[0, 5]]), _hidden_states(batch=1, seq=2)]
        model = LiveEvalConfidenceAugmentedLLM(
            llm=llm, probe_checkpoint=self._checkpoint(tmp_path), layer_idx=0
        )

        _, confidence = model._generate_and_score(prompt, mask)

        assert llm.call_args_list[1].args[0] is prompt
        assert llm.call_args_list[1].args[1] is mask
        # No answer tokens in the encoder input: the answer slice is a zero vector
        assert torch.allclose(confidence, model.probe.classify(torch.zeros(1, DIM)))

    def test_probe_is_loaded_and_frozen(self, tmp_path):
        model = LiveEvalConfidenceAugmentedLLM(
            llm=self._llm(), probe_checkpoint=self._checkpoint(tmp_path), layer_idx=0
        )
        assert not any(p.requires_grad for p in model.probe.parameters())

    def test_validation_writes_outputs_and_metrics(self, tmp_path):
        llm = self._llm()
        llm.tokenizer.decode.side_effect = lambda ids, **_: " ".join(
            str(int(t)) for t in ids
        )
        llm.extractor.extract_answer.side_effect = ["Paris", "Rome"]
        prompt = torch.tensor([[1, 2], [3, 4]])
        llm.side_effect = [
            torch.tensor([[1, 2, 7], [3, 4, 9]]),
            _hidden_states(batch=2, seq=3),
        ]
        model = LiveEvalConfidenceAugmentedLLM(
            llm=llm, probe_checkpoint=self._checkpoint(tmp_path), layer_idx=0
        )
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
        decoded = [c.args[0].tolist() for c in llm.tokenizer.decode.call_args_list]
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
        assert [r[:3] + r[4:] for r in rows[1:]] == [
            ["q1", "Paris", "Paris", "Yes", "7"],
            ["q2", "Madrid", "Rome", "No", "9"],
        ]
        with open(tmp_path / "metrics.csv") as f:
            assert next(csv.reader(f))[0] == "val_accuracy"
        assert model.outputs == []


class TestTrainingHooks:
    def _attach(self, model, tmp_path, global_step=0, datamodule=None):
        model._trainer = Mock(
            log_dir=str(tmp_path),
            global_step=global_step,
            optimizers=[],
            datamodule=datamodule,
        )
        model.log = Mock()

    def _run_weight_history(self, model, tmp_path, activations, steps=(0,)):
        callback = LayerWeightsHistoryCallback()
        for step in steps:
            self._attach(model, tmp_path, global_step=step)
            callback.on_train_batch_start(model.trainer, model, None, step)
            model.training_step(
                {"activations": activations, "correct": torch.tensor([1, 0])}, step
            )
        callback.on_fit_end(model.trainer, model)
        path = tmp_path / "layer_weights_history.json"
        return json.loads(path.read_text()) if path.exists() else None

    def test_weight_snapshots_written_on_fit_end(self, tmp_path):
        model = CachedActivationsConfidenceAugmentedLLM(model_dim=DIM, layer_idx=[3, 5])
        data = self._run_weight_history(
            model, tmp_path, torch.randn(2, 2, DIM), steps=[0, 1, 50, 100]
        )
        weights = model.probe.layer_pooling.get_weights().tolist()
        assert data == {
            "layer_indices": [3, 5],
            "snapshots": [{"step": s, "weights": weights} for s in [0, 50, 100]],
        }

    def test_weight_snapshots_layer_indices_for_all(self, tmp_path):
        model = CachedActivationsConfidenceAugmentedLLM(
            model_dim=DIM, layer_idx="all", num_layers=3
        )
        data = self._run_weight_history(model, tmp_path, torch.randn(2, 3, DIM))
        assert data["layer_indices"] == [0, 1, 2]

    def test_no_snapshots_without_layer_weights(self, tmp_path):
        model = CachedActivationsConfidenceAugmentedLLM(model_dim=DIM)
        assert self._run_weight_history(model, tmp_path, torch.randn(2, DIM)) is None

    def test_training_step_logs_lr(self, tmp_path):
        model = CachedActivationsConfidenceAugmentedLLM(model_dim=DIM, lr=0.25)
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
        model = CachedActivationsConfidenceAugmentedLLM(
            model_dim=DIM, layer_pooling=pooling
        )
        self._attach(model, tmp_path, datamodule=Mock(layer_stats=stats))

        model.on_fit_start()

        for buffer, key in zip(buffers, stats):
            assert torch.equal(getattr(pooling, buffer), stats[key])
        assert pooling._stats_set

    def test_znorm_without_stats_or_datamodule(self, tmp_path):
        pooling = GlobalZNormWeightedAverage(num_layers=2, model_dim=DIM)
        model = CachedActivationsConfidenceAugmentedLLM(
            model_dim=DIM, layer_pooling=pooling
        )
        self._attach(model, tmp_path, datamodule=Mock(layer_stats=None))
        model.on_fit_start()
        self._attach(model, tmp_path, datamodule=None)
        model.on_fit_start()
        assert not pooling._stats_set

    def test_on_fit_start_ignores_plain_pooling(self, tmp_path):
        model = CachedActivationsConfidenceAugmentedLLM(model_dim=DIM, layer_idx=[0, 1])
        self._attach(model, tmp_path, datamodule=Mock(layer_stats={}))
        model.on_fit_start()  # no set_normalization_stats → no-op


class TestCachedValidationMetrics:
    def test_logs_probe_metrics_and_resets(self, tmp_path):
        torch.manual_seed(0)
        model = CachedActivationsConfidenceAugmentedLLM(model_dim=DIM)
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
        model = CachedActivationsConfidenceAugmentedLLM(model_dim=DIM)
        model.log = Mock()
        model.on_validation_epoch_end()
        model.log.assert_not_called()


class TestCheckpoints:
    def test_save_drops_llm_weights(self):
        model = CachedActivationsConfidenceAugmentedLLM(model_dim=DIM)
        checkpoint = {"state_dict": {"llm.w": 1, "probe.linear.weight": 2}}
        model.on_save_checkpoint(checkpoint)
        assert checkpoint == {"state_dict": {"probe.linear.weight": 2}}

    @pytest.mark.parametrize("wrapped", [True, False])
    def test_load_probe_checkpoint(self, tmp_path, wrapped):
        trained = CachedActivationsConfidenceAugmentedLLM(
            model_dim=DIM, layer_idx=[0, 1]
        )
        state = {f"probe.{k}": v for k, v in trained.probe.state_dict().items()}
        state["llm.ignored"] = torch.zeros(1)
        path = tmp_path / "probe.ckpt"
        torch.save({"state_dict": state} if wrapped else state, path)

        model = CachedActivationsConfidenceAugmentedLLM(model_dim=DIM, layer_idx=[0, 1])
        model._load_probe_checkpoint(str(path))

        for k, v in trained.probe.state_dict().items():
            assert torch.equal(model.probe.state_dict()[k], v)
        assert not model.probe.training
        assert not any(p.requires_grad for p in model.probe.parameters())

    def test_load_probe_checkpoint_without_probe_keys(self, tmp_path):
        path = tmp_path / "bad.ckpt"
        torch.save({"state_dict": {"llm.w": torch.zeros(1)}}, path)
        with pytest.raises(ValueError, match="No 'probe.\\*' keys"):
            CachedActivationsConfidenceAugmentedLLM(
                model_dim=DIM
            )._load_probe_checkpoint(str(path))


class TestConfigureOptimizers:
    def test_with_scheduler(self):
        scheduler = Mock(return_value="scheduler")
        model = CachedActivationsConfidenceAugmentedLLM(
            model_dim=DIM, lr_scheduler=scheduler
        )
        model._trainer = Mock(estimated_stepping_batches=123)

        config = model.configure_optimizers()

        assert config["lr_scheduler"] == {"scheduler": "scheduler", "interval": "step"}
        scheduler.assert_called_once_with(config["optimizer"], max_steps=123)
