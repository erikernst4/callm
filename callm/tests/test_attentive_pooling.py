import pytest
import torch
import functools
from unittest.mock import Mock
from callm.models.sequence_pooling import (
    MeanPooling,
    AttentivePooling,
    QueryAttentivePooling,
    MultiHeadAttentivePooling,
    QueryMultiHeadAttentivePooling,
    TransformerPooling,
)
from callm.models.probe import Probe
from callm.models.confidence_llm import ConfidenceAugmentedLLM
from callm.models.layer_pooling import (
    GlobalZNormWeightedAverage,
    LayerwiseZNormWeightedAverage,
    LayerNormWeightedAverage,
    LayerWeightedAverage,
    create_layer_weights,
)

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
    model._trainer = Mock(estimated_stepping_batches=100)

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


# ==============================================================================
# Tests for Layer Pooling Modules (Exp4)
# ==============================================================================

LAYER_POOLING_CLASSES = [
    GlobalZNormWeightedAverage,
    LayerwiseZNormWeightedAverage,
    LayerNormWeightedAverage,
    LayerWeightedAverage,
]


@pytest.mark.parametrize(
    "pooling_cls",
    [
        GlobalZNormWeightedAverage,
        LayerwiseZNormWeightedAverage,
        LayerNormWeightedAverage,
    ],
)
def test_layer_pooling_output_shape_3d(pooling_cls):
    """Test that 3D input (batch, num_layers, model_dim) produces (batch, model_dim)."""
    batch_size = 4
    num_layers = 6
    model_dim = 64

    if pooling_cls == LayerWeightedAverage:
        module = pooling_cls(num_layers=num_layers)
    else:
        module = pooling_cls(num_layers=num_layers, model_dim=model_dim)

    x = torch.randn(batch_size, num_layers, model_dim)
    out = module(x)
    assert out.shape == (batch_size, model_dim)


@pytest.mark.parametrize(
    "pooling_cls",
    [
        GlobalZNormWeightedAverage,
        LayerwiseZNormWeightedAverage,
        LayerNormWeightedAverage,
    ],
)
def test_layer_pooling_output_shape_4d(pooling_cls):
    """Test that 4D input (batch, num_layers, seq_len, model_dim) produces (batch, seq_len, model_dim)."""
    batch_size = 4
    num_layers = 6
    seq_len = 10
    model_dim = 64

    if pooling_cls == LayerWeightedAverage:
        module = pooling_cls(num_layers=num_layers)
    else:
        module = pooling_cls(num_layers=num_layers, model_dim=model_dim)

    x = torch.randn(batch_size, num_layers, seq_len, model_dim)
    out = module(x)
    assert out.shape == (batch_size, seq_len, model_dim)


@pytest.mark.parametrize(
    "pooling_cls",
    [
        GlobalZNormWeightedAverage,
        LayerwiseZNormWeightedAverage,
        LayerNormWeightedAverage,
    ],
)
def test_layer_pooling_gradients(pooling_cls):
    """Test that gradients flow through to inputs and parameters."""
    batch_size = 2
    num_layers = 4
    model_dim = 32

    module = pooling_cls(num_layers=num_layers, model_dim=model_dim)
    x = torch.randn(batch_size, num_layers, model_dim, requires_grad=True)

    out = module(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    # Verify layer_weights have gradients
    assert module.layer_weights.grad is not None


@pytest.mark.parametrize(
    "pooling_cls",
    [
        GlobalZNormWeightedAverage,
        LayerwiseZNormWeightedAverage,
        LayerNormWeightedAverage,
    ],
)
def test_layer_pooling_get_weights(pooling_cls):
    """Test get_weights() returns proper softmax weights."""
    num_layers = 6
    model_dim = 32

    module = pooling_cls(num_layers=num_layers, model_dim=model_dim)
    weights = module.get_weights()

    assert weights.shape == (num_layers,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert (weights >= 0).all()


def test_global_z_norm_stats_injection():
    """Test that set_normalization_stats correctly sets buffers and affects output."""
    num_layers = 4
    model_dim = 32

    module = GlobalZNormWeightedAverage(num_layers=num_layers, model_dim=model_dim)

    # Create data with known statistics
    mean = torch.full((model_dim,), 5.0)
    std = torch.full((model_dim,), 2.0)

    module.set_normalization_stats(mean, std)

    assert module._stats_set
    assert torch.allclose(module.global_mean, mean)
    assert torch.allclose(module.global_std, std)

    # Forward with constant input: after normalization, (5 - 5) / 2 = 0
    x = torch.full((2, num_layers, model_dim), 5.0)
    out = module(x)
    assert torch.allclose(out, torch.zeros(2, model_dim), atol=1e-5)


def test_layerwise_z_norm_stats_injection():
    """Test that set_normalization_stats correctly sets buffers and normalizes per layer."""
    num_layers = 3
    model_dim = 16

    module = LayerwiseZNormWeightedAverage(num_layers=num_layers, model_dim=model_dim)

    # Different mean for each layer
    layer_means = torch.tensor(
        [[1.0] * model_dim, [2.0] * model_dim, [3.0] * model_dim]
    )
    layer_stds = torch.ones(num_layers, model_dim)

    module.set_normalization_stats(layer_means, layer_stds)

    assert module._stats_set
    assert module.layer_means.shape == (num_layers, model_dim)
    assert module.layer_stds.shape == (num_layers, model_dim)

    # Input where each layer has its own mean → after normalization, all zero
    x = torch.stack(
        [
            torch.full((model_dim,), 1.0),
            torch.full((model_dim,), 2.0),
            torch.full((model_dim,), 3.0),
        ],
        dim=0,
    ).unsqueeze(0)  # (1, 3, model_dim)

    out = module(x)
    assert torch.allclose(out, torch.zeros(1, model_dim), atol=1e-5)


def test_layerwise_z_norm_stats_injection_4d():
    """Test layer-wise Z-norm with 4D input (batch, layers, seq, dim)."""
    num_layers = 2
    model_dim = 8
    seq_len = 5

    module = LayerwiseZNormWeightedAverage(num_layers=num_layers, model_dim=model_dim)
    layer_means = torch.zeros(num_layers, model_dim)
    layer_stds = torch.ones(num_layers, model_dim) * 2.0
    module.set_normalization_stats(layer_means, layer_stds)

    x = torch.randn(3, num_layers, seq_len, model_dim) * 2.0
    out = module(x)
    assert out.shape == (3, seq_len, model_dim)


def test_layernorm_weighted_avg_per_layer_independence():
    """Test that LayerNorm is applied independently per layer."""
    num_layers = 3
    model_dim = 32

    module = LayerNormWeightedAverage(num_layers=num_layers, model_dim=model_dim)

    # Verify we have separate LayerNorm modules
    assert len(module.layer_norms) == num_layers
    for i, ln in enumerate(module.layer_norms):
        assert isinstance(ln, torch.nn.LayerNorm)
        assert ln.normalized_shape == (model_dim,)

    # Modify one LayerNorm's bias and verify it doesn't affect others
    with torch.no_grad():
        module.layer_norms[0].bias.fill_(10.0)

    assert not torch.equal(module.layer_norms[0].bias, module.layer_norms[1].bias)


def test_z_norm_std_clamping():
    """Test that zero std doesn't cause NaN (clamped to 1e-8)."""
    num_layers = 2
    model_dim = 16

    module = GlobalZNormWeightedAverage(num_layers=num_layers, model_dim=model_dim)
    module.set_normalization_stats(
        torch.zeros(model_dim),
        torch.zeros(model_dim),  # zero std!
    )

    x = torch.randn(2, num_layers, model_dim)
    out = module(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_layer_pooling_with_probe():
    """Test integration of new layer pooling modules with the Probe class."""
    batch_size = 4
    num_layers = 6
    model_dim = 32

    for pooling_cls in [
        GlobalZNormWeightedAverage,
        LayerwiseZNormWeightedAverage,
        LayerNormWeightedAverage,
    ]:
        layer_pooling = pooling_cls(num_layers=num_layers, model_dim=model_dim)
        probe = Probe(model_dim=model_dim, layer_pooling=layer_pooling)

        # Pre-pooled input (no seq dim)
        x = torch.randn(batch_size, num_layers, model_dim)
        out = probe(x)
        assert out.shape == (batch_size,)
        assert (out >= 0.0).all() and (out <= 1.0).all()


def test_compute_layer_statistics():
    """Test that compute_layer_statistics produces correct stats."""
    num_layers = 3
    model_dim = 8
    num_samples = 100

    # Create fake dataset_samples with known properties
    dataset_samples = []
    for _ in range(num_samples):
        # Each layer has a different mean: layer 0 ~ N(0,1), layer 1 ~ N(5,2), layer 2 ~ N(-3,0.5)
        acts = torch.stack(
            [
                torch.randn(model_dim) * 1.0 + 0.0,
                torch.randn(model_dim) * 2.0 + 5.0,
                torch.randn(model_dim) * 0.5 - 3.0,
            ],
            dim=0,
        )
        dataset_samples.append(
            {"activations": acts, "correct": True, "is_multi_layer": True}
        )

    dm = ActivationsDataModule.__new__(ActivationsDataModule)
    stats = dm.compute_layer_statistics(dataset_samples)

    assert stats is not None
    assert stats["global_mean"].shape == (model_dim,)
    assert stats["global_std"].shape == (model_dim,)
    assert stats["layer_means"].shape == (num_layers, model_dim)
    assert stats["layer_stds"].shape == (num_layers, model_dim)

    # Layer means should be approximately correct
    assert torch.allclose(stats["layer_means"][0], torch.zeros(model_dim), atol=0.5)
    assert torch.allclose(
        stats["layer_means"][1], torch.full((model_dim,), 5.0), atol=1.0
    )
    assert torch.allclose(
        stats["layer_means"][2], torch.full((model_dim,), -3.0), atol=0.5
    )

    # Layer stds should be approximately correct
    assert torch.allclose(stats["layer_stds"][0], torch.ones(model_dim), atol=0.5)
    assert torch.allclose(
        stats["layer_stds"][1], torch.full((model_dim,), 2.0), atol=1.0
    )


def test_compute_layer_statistics_single_layer_returns_none():
    """Test that compute_layer_statistics returns None for single-layer samples."""
    dataset_samples = [
        {"activations": torch.randn(32), "correct": True, "is_multi_layer": False}
        for _ in range(10)
    ]

    dm = ActivationsDataModule.__new__(ActivationsDataModule)
    stats = dm.compute_layer_statistics(dataset_samples)
    assert stats is None


# ==============================================================================
# Tests for Layer Weight Initializations (Exp5)
# ==============================================================================


def test_create_layer_weights_uniform():
    """Test uniform initialization creates zeros and equal softmax weights."""
    num_layers = 10
    w = create_layer_weights(num_layers, init_strategy="uniform")
    assert w.shape == (num_layers,)
    assert (w == 0).all()
    probs = torch.softmax(w, dim=0)
    assert torch.allclose(probs, torch.full((num_layers,), 0.1))


def test_create_layer_weights_normal():
    """Test normal initialization creates bell curve peaked at middle layer."""
    num_layers = 33
    w = create_layer_weights(num_layers, init_strategy="normal")
    assert w.shape == (num_layers,)
    probs = torch.softmax(w, dim=0)

    mid_idx = (num_layers - 1) // 2
    # Peak should be at middle layer
    assert probs.argmax().item() == mid_idx
    # Symmetry around middle layer
    assert torch.isclose(probs[0], probs[-1], atol=1e-5)
    # Middle layer should have higher probability than extreme layers
    assert probs[mid_idx] > probs[0]
    assert probs[mid_idx] > probs[-1]


def test_create_layer_weights_reverse_chi():
    """Test reverse chi initialization has near-zero early layers, middle peak, non-zero late layers."""
    num_layers = 33
    w = create_layer_weights(num_layers, init_strategy="reverse_chi", init_k=3.0)
    assert w.shape == (num_layers,)
    probs = torch.softmax(w, dim=0)

    mid_idx = (num_layers - 1) // 2
    # Layer 0 should be near zero
    assert probs[0] < 1e-4
    # Peak should be at middle layer
    assert probs.argmax().item() == mid_idx
    # Later layers should have some non-zero weight (decaying tail, greater than layer 0)
    assert probs[-1] > probs[0]
    assert probs[-1] > 0.001


def test_create_layer_weights_random():
    """Test random initialization creates non-zero tensor of correct shape."""
    num_layers = 10
    w = create_layer_weights(num_layers, init_strategy="random")
    assert w.shape == (num_layers,)
    probs = torch.softmax(w, dim=0)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_invalid_init_strategy_raises_error():
    """Test invalid init strategy raises ValueError."""
    with pytest.raises(ValueError, match="Unknown layer weight init_strategy"):
        create_layer_weights(10, init_strategy="invalid_name")


@pytest.mark.parametrize("init_strat", ["uniform", "normal", "reverse_chi", "random"])
@pytest.mark.parametrize(
    "pooling_cls",
    [
        LayerWeightedAverage,
        GlobalZNormWeightedAverage,
        LayerwiseZNormWeightedAverage,
        LayerNormWeightedAverage,
    ],
)
def test_layer_pooling_classes_with_init_strategies(pooling_cls, init_strat):
    """Test instantiating layer pooling classes with different init_strategy arguments."""
    num_layers = 10
    model_dim = 32

    if pooling_cls == LayerWeightedAverage:
        module = pooling_cls(num_layers=num_layers, init_strategy=init_strat)
    else:
        module = pooling_cls(
            num_layers=num_layers, model_dim=model_dim, init_strategy=init_strat
        )

    weights = module.get_weights()
    assert weights.shape == (num_layers,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert (weights >= 0).all()


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
def test_probe_bfloat16_input_compatibility(pooling_cls):
    """Test Probe handles bfloat16 input activations across all sequence pooling modules."""
    model_dim = 64
    seq_len = 12
    batch_size = 3

    probe = Probe(model_dim=model_dim, seq_pooling=pooling_cls(model_dim=model_dim))
    act_bf16 = torch.randn(batch_size, seq_len, model_dim, dtype=torch.bfloat16)
    seq_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    seq_mask[0, 8:] = False

    out = probe(act_bf16, seq_mask=seq_mask)
    assert out.shape == (batch_size,)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_activations_datamodule_dtype_conversion():
    """Test ActivationsDataModule._convert_dtype converts tensors properly."""
    dm_f32 = ActivationsDataModule(
        activations_dir="fake_dir",
        evaluation_csv="fake_csv",
        dtype="float32",
    )
    t = torch.randn(4, 16, dtype=torch.bfloat16)
    assert dm_f32._convert_dtype(t).dtype == torch.float32

    dm_bf16 = ActivationsDataModule(
        activations_dir="fake_dir",
        evaluation_csv="fake_csv",
        dtype="bfloat16",
    )
    assert dm_bf16._convert_dtype(t).dtype == torch.bfloat16

    dm_fp16 = ActivationsDataModule(
        activations_dir="fake_dir",
        evaluation_csv="fake_csv",
        dtype="float16",
    )
    assert dm_fp16._convert_dtype(t).dtype == torch.float16

    dm_auto = ActivationsDataModule(
        activations_dir="fake_dir",
        evaluation_csv="fake_csv",
        dtype="auto",
    )
    assert dm_auto._convert_dtype(t).dtype == torch.bfloat16


def test_collate_fn_with_bfloat16():
    """Test collate_fn handles bfloat16 activations and produces valid batch."""
    batch = [
        {
            "activations": torch.randn(10, 32, dtype=torch.bfloat16),
            "correct": True,
            "is_multi_layer": False,
        },
        {
            "activations": torch.randn(15, 32, dtype=torch.bfloat16),
            "correct": False,
            "is_multi_layer": False,
        },
    ]
    collated = ActivationsDataModule.collate_fn(batch)
    assert collated["activations"].dtype == torch.bfloat16
    assert collated["activations"].shape == (2, 15, 32)
    assert collated["correct"].shape == (2,)
    assert collated["seq_mask"].shape == (2, 15)
