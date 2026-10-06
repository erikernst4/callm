import csv

import pytest
import torch

from callm.data.activations_data import ActivationsDataModule, LazyActivationsDataset
from callm.models.llm import SplitConfidenceAugmentedLLM
from callm.models.probe import SplitProbe
from callm.models.sequence_pooling import AttentivePooling


class TestSplitProbe:
    """Unit tests for SplitProbe."""

    @pytest.mark.parametrize("merge", ["concat", "mean", "mean_logits"])
    def test_pooled_shapes(self, merge):
        probe = SplitProbe(model_dim=8, merge=merge)
        confidence = probe(torch.randn(4, 8), torch.randn(4, 8))
        assert confidence.shape == (4,)
        assert torch.all((confidence >= 0) & (confidence <= 1))

    @pytest.mark.parametrize("merge", ["concat", "mean", "mean_logits"])
    def test_sequence_shapes_with_attentive_pooling(self, merge):
        probe = SplitProbe(
            model_dim=8, merge=merge, seq_pooling=AttentivePooling(model_dim=8)
        )
        prompt_mask = torch.ones(3, 6, dtype=torch.bool)
        answer_mask = torch.ones(3, 2, dtype=torch.bool)
        answer_mask[1, 1:] = False
        confidence = probe(
            torch.randn(3, 6, 8),
            torch.randn(3, 2, 8),
            prompt_seq_mask=prompt_mask,
            answer_seq_mask=answer_mask,
        )
        assert confidence.shape == (3,)
        assert not torch.isnan(confidence).any()

    def test_parts_have_separate_pooling_weights(self):
        probe = SplitProbe(model_dim=8, seq_pooling=AttentivePooling(model_dim=8))
        prompt_params = list(probe.seq_pooling["prompt"].parameters())
        answer_params = list(probe.seq_pooling["answer"].parameters())
        assert prompt_params and len(prompt_params) == len(answer_params)
        for p, a in zip(prompt_params, answer_params):
            assert p is not a and p.data_ptr() != a.data_ptr()

    def test_merges(self):
        h_p, h_a = torch.randn(2, 4), torch.randn(2, 4)
        concat = SplitProbe(model_dim=4, merge="concat")
        expected = torch.sigmoid(concat.linear(torch.cat([h_p, h_a], -1))).squeeze(-1)
        assert torch.allclose(concat(h_p, h_a), expected)

        mean = SplitProbe(model_dim=4, merge="mean")
        expected = torch.sigmoid(mean.linear((h_p + h_a) / 2)).squeeze(-1)
        assert torch.allclose(mean(h_p, h_a), expected)

        mean_logits = SplitProbe(model_dim=4, merge="mean_logits")
        logits = (
            mean_logits.linear["prompt"](h_p) + mean_logits.linear["answer"](h_a)
        ) / 2
        assert torch.allclose(mean_logits(h_p, h_a), torch.sigmoid(logits).squeeze(-1))

    def test_masked_mean_fallback_ignores_padding(self):
        probe = SplitProbe(model_dim=4, merge="mean_logits")
        lengths = [3, 1]
        samples = [torch.randn(n, 4) for n in lengths]
        padded = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True)
        mask = torch.arange(3) < torch.tensor(lengths)[:, None]
        pooled = probe.pool("answer", padded, mask)
        assert torch.allclose(pooled, torch.stack([s.mean(0) for s in samples]))

    def test_invalid_merge(self):
        with pytest.raises(ValueError, match="merge must be one of"):
            SplitProbe(model_dim=4, merge="sum")


class TestSplitConfidenceAugmentedLLM:
    """Unit tests for SplitConfidenceAugmentedLLM."""

    @pytest.mark.parametrize("attentive", [False, True])
    def test_training_and_validation_step(self, attentive):
        seq_pooling = AttentivePooling(model_dim=8) if attentive else None
        model = SplitConfidenceAugmentedLLM(
            model_dim=8, merge="concat", layer_idx=12, seq_pooling=seq_pooling
        )
        assert isinstance(model.probe, SplitProbe)
        if attentive:
            batch = {
                "prompt_activations": torch.randn(4, 6, 8),
                "answer_activations": torch.randn(4, 2, 8),
                "prompt_seq_mask": torch.ones(4, 6, dtype=torch.bool),
                "answer_seq_mask": torch.ones(4, 2, dtype=torch.bool),
            }
        else:
            batch = {
                "prompt_activations": torch.randn(4, 8),
                "answer_activations": torch.randn(4, 8),
            }
        batch["correct"] = torch.tensor([1, 0, 1, 0])

        loss = model.training_step(batch, 0)
        assert loss.ndim == 0 and loss.requires_grad
        out = model.validation_step(batch, 0)
        assert out["confidence"].shape == (4,)

    def test_layer_weighted_average_per_part(self):
        model = SplitConfidenceAugmentedLLM(model_dim=8, layer_idx=[12, 14])
        assert set(model.probe.layer_pooling.keys()) == {"prompt", "answer"}
        confidence = model.probe(torch.randn(2, 2, 8), torch.randn(2, 2, 8))
        assert confidence.shape == (2,)

    def test_requires_split_batch(self):
        model = SplitConfidenceAugmentedLLM(model_dim=8)
        batch = {"activations": torch.randn(2, 8), "correct": torch.tensor([1, 0])}
        with pytest.raises(ValueError, match="prompt_activations"):
            model.training_step(batch, 0)


def _write_split(root, name, samples):
    split_dir = root / name
    split_dir.mkdir()
    torch.save(samples, split_dir / "activations_0000.pt")
    csv_path = root / f"{name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Correct"])
        writer.writerows(
            [[f"Q{i}", "Yes" if i % 2 else "No"] for i in range(len(samples))]
        )
    return str(split_dir), str(csv_path)


def _random_samples(num_samples):
    gen = torch.Generator().manual_seed(0)
    samples = []
    for i in range(num_samples):
        seq_len = 3 + i % 4
        samples.append(
            {
                "sample_idx": i,
                "seq_len": seq_len,
                "prompt_length": seq_len - 1 - i % 2,
                "activations": {
                    layer: torch.randn(seq_len, 4, generator=gen) for layer in (12, 14)
                },
            }
        )
    return samples


class TestActivationsDataModuleSplit:
    """Unit tests for activations_to_use='split' in ActivationsDataModule."""

    @pytest.fixture
    def sample(self):
        return {
            "sample_idx": 0,
            "seq_len": 5,
            "prompt_length": 3,
            "activations": {0: torch.arange(40, dtype=torch.float32).view(5, 8)},
        }

    @staticmethod
    def _dm(pool_strategy, layer_idx=0):
        return ActivationsDataModule(
            activations_dir="fake",
            evaluation_csv="fake.csv",
            layer_idx=layer_idx,
            pool_strategy=pool_strategy,
            activations_to_use="split",
        )

    def test_split_slicing(self, sample):
        act = sample["activations"][0]
        out, is_multi_layer = self._dm("none")._process_sample(sample)
        assert not is_multi_layer
        assert torch.equal(out["prompt"], act[:3])
        assert torch.equal(out["answer"], act[3:])

        out, _ = self._dm("mean")._process_sample(sample)
        assert torch.equal(out["prompt"], act[:3].mean(0))
        assert torch.equal(out["answer"], act[3:].mean(0))

    def test_split_multi_layer(self, sample):
        sample["activations"][1] = sample["activations"][0] + 100
        out, is_multi_layer = self._dm("none", layer_idx=[0, 1])._process_sample(sample)
        assert is_multi_layer
        assert out["prompt"].shape == (2, 3, 8)
        assert out["answer"].shape == (2, 2, 8)

    def test_empty_answer_keeps_one_zero_token(self, sample):
        sample["prompt_length"] = 5
        out, _ = self._dm("none")._process_sample(sample)
        assert torch.equal(out["answer"], torch.zeros(1, 8))

    def test_collate_split(self):
        batch = [
            {
                "activations": {
                    "prompt": torch.randn(4, 8),
                    "answer": torch.randn(1, 8),
                },
                "correct": True,
                "is_multi_layer": False,
            },
            {
                "activations": {
                    "prompt": torch.randn(2, 8),
                    "answer": torch.randn(3, 8),
                },
                "correct": False,
                "is_multi_layer": False,
            },
        ]
        out = ActivationsDataModule.collate_fn(batch)
        assert set(out) == {
            "correct",
            "prompt_activations",
            "prompt_seq_mask",
            "answer_activations",
            "answer_seq_mask",
        }
        assert out["prompt_activations"].shape == (2, 4, 8)
        assert out["answer_activations"].shape == (2, 3, 8)
        assert out["prompt_seq_mask"].sum(1).tolist() == [4, 2]
        assert out["answer_seq_mask"].sum(1).tolist() == [1, 3]

    def test_collate_split_pooled(self):
        batch = [
            {
                "activations": {"prompt": torch.randn(8), "answer": torch.randn(8)},
                "correct": True,
                "is_multi_layer": False,
            }
        ] * 3
        out = ActivationsDataModule.collate_fn(batch)
        assert set(out) == {"correct", "prompt_activations", "answer_activations"}
        assert out["prompt_activations"].shape == (3, 8)

    @pytest.mark.parametrize(
        "pool_strategy,layer_idx", [("none", 14), ("mean", [12, 14])]
    )
    def test_lazy_matches_eager(self, tmp_path, pool_strategy, layer_idx):
        train_dir, train_csv = _write_split(tmp_path, "train", _random_samples(8))
        val_dir, val_csv = _write_split(tmp_path, "val", _random_samples(4))
        kwargs = dict(
            activations_dir=train_dir,
            evaluation_csv=train_csv,
            val_activations_dir=val_dir,
            val_evaluation_csv=val_csv,
            layer_idx=layer_idx,
            pool_strategy=pool_strategy,
            activations_to_use="split",
            num_workers=0,
        )
        eager = ActivationsDataModule(**kwargs)
        lazy = ActivationsDataModule(**kwargs, lazy=True)
        eager.setup()
        lazy.setup()

        assert isinstance(lazy.train_dataset, LazyActivationsDataset)
        assert eager.layer_stats is None
        for split in ("train_dataset", "val_dataset"):
            eager_ds, lazy_ds = getattr(eager, split), getattr(lazy, split)
            assert len(eager_ds) == len(lazy_ds)
            for i in range(len(eager_ds)):
                for part in ("prompt", "answer"):
                    assert torch.equal(
                        eager_ds[i]["activations"][part],
                        lazy_ds[i]["activations"][part],
                    )

        batch = next(iter(eager.train_dataloader()))
        model = SplitConfidenceAugmentedLLM(
            model_dim=4,
            layer_idx=layer_idx,
            seq_pooling=AttentivePooling(model_dim=4)
            if pool_strategy == "none"
            else None,
        )
        assert model.training_step(batch, 0).isfinite()
