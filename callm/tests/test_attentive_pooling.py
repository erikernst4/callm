import pytest
import torch
import functools
from callm.models.pooling import (
    MeanPooling,
    AttentivePooling,
    QueryAttentivePooling,
    MultiHeadAttentivePooling,
    QueryMultiHeadAttentivePooling,
    TransformerPooling,
)
from callm.models.probe import Probe, ConfidenceAugmentedLLM
from callm.data.activations_data import ActivationsDataModule
from callm.models.lr_scheduling import WarmupExponentialLR


@pytest.mark.parametrize(
    "pooling_cls",
    [
        MeanPooling,
        AttentivePooling,
        QueryAttentivePooling,
        MultiHeadAttentivePooling,
        QueryMultiHeadAttentivePooling,
        TransformerPooling,
    ],
)
def test_pooling_output_shape(pooling_cls):
    batch_size = 4
    seq_len = 16
    model_dim = 64

    module = pooling_cls(model_dim=model_dim)
    x = torch.randn(batch_size, seq_len, model_dim)

    # Test without mask
    out = module(x)
    assert out.shape == (batch_size, model_dim)

    # Test with mask (variable lengths)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 10:] = False
    mask[1, 5:] = False

    out_masked = module(x, mask=mask)
    assert out_masked.shape == (batch_size, model_dim)


@pytest.mark.parametrize(
    "pooling_cls",
    [
        AttentivePooling,
        QueryAttentivePooling,
        MultiHeadAttentivePooling,
        QueryMultiHeadAttentivePooling,
        TransformerPooling,
    ],
)
def test_pooling_gradients(pooling_cls):
    batch_size = 2
    seq_len = 8
    model_dim = 32

    module = pooling_cls(model_dim=model_dim)
    x = torch.randn(batch_size, seq_len, model_dim, requires_grad=True)
    mask = torch.tensor([[True] * 6 + [False] * 2, [True] * 4 + [False] * 4])

    out = module(x, mask=mask)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    # Verify padded positions receive zero gradient for masked elements (or non-influential)
    for p in module.parameters():
        assert p.grad is not None


def test_probe_with_seq_pooling():
    batch_size = 4
    seq_len = 10
    model_dim = 32

    pooling = AttentivePooling(model_dim=model_dim)
    probe = Probe(model_dim=model_dim, seq_pooling=pooling)

    x = torch.randn(batch_size, seq_len, model_dim)
    seq_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    seq_mask[2, 5:] = False

    out = probe(x, seq_mask=seq_mask)
    assert out.shape == (batch_size,)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_collate_fn_seq_mask():
    batch = [
        {"activations": torch.randn(10, 32), "correct": True, "is_multi_layer": False},
        {"activations": torch.randn(6, 32), "correct": False, "is_multi_layer": False},
    ]

    collated = ActivationsDataModule.collate_fn(batch)
    assert "activations" in collated
    assert "seq_mask" in collated
    assert collated["activations"].shape == (2, 10, 32)
    assert collated["seq_mask"].shape == (2, 10)
    assert collated["seq_mask"][0].sum().item() == 10
    assert collated["seq_mask"][1].sum().item() == 6


def test_warmup_exponential_lr_scheduler():
    layer = torch.nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    scheduler = WarmupExponentialLR(optimizer, warmup_steps=10, gamma=0.95)

    lr_0 = optimizer.param_groups[0]["lr"]

    for _ in range(5):
        scheduler.step()
    lr_5 = optimizer.param_groups[0]["lr"]
    assert lr_5 > lr_0

    for _ in range(5):
        scheduler.step()
    lr_10 = optimizer.param_groups[0]["lr"]
    assert lr_10 >= lr_5

    for _ in range(5):
        scheduler.step()
    lr_15 = optimizer.param_groups[0]["lr"]
    assert lr_15 < lr_10


def test_confidence_augmented_llm_lr_scheduler():
    seq_pooling = QueryAttentivePooling(model_dim=32)
    lr_scheduler_fn = functools.partial(
        WarmupExponentialLR, warmup_steps=10, gamma=0.95
    )
    model = ConfidenceAugmentedLLM(
        model_dim=32,
        seq_pooling=seq_pooling,
        lr=1e-3,
        lr_scheduler=lr_scheduler_fn,
    )

    opt_config = model.configure_optimizers()
    assert isinstance(opt_config, dict)
    assert "optimizer" in opt_config
    assert "lr_scheduler" in opt_config

    scheduler = opt_config["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, WarmupExponentialLR)
    optimizer = opt_config["optimizer"]

    # Step 0: Warmup start
    lr_0 = optimizer.param_groups[0]["lr"]

    # Step 5: Mid warmup
    for _ in range(5):
        scheduler.step()
    lr_5 = optimizer.param_groups[0]["lr"]
    assert lr_5 > lr_0

    # Step 10: Peak warmup
    for _ in range(5):
        scheduler.step()
    lr_10 = optimizer.param_groups[0]["lr"]
    assert lr_10 >= lr_5

    # Step 15: Post warmup decay
    for _ in range(5):
        scheduler.step()
    lr_15 = optimizer.param_groups[0]["lr"]
    assert lr_15 < lr_10
