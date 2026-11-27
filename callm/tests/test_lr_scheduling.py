import math
import torch
import torch.nn as nn
from callm.models.lr_scheduling import WarmupExponentialLR, WarmupCosineAnnealingLR


def test_warmup_exponential_lr():
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = WarmupExponentialLR(optimizer, warmup_steps=10, gamma=0.5)

    # In PyTorch, scheduler initializes at last_epoch=0 (after internal step)
    # So the LR is for step 0. 0 / 10 is 0, bounded by 1e-6.
    expected_lr_0 = 0.1 * 1e-6
    assert math.isclose(scheduler.get_last_lr()[0], expected_lr_0, rel_tol=1e-5)

    # Step through warmup (steps 1 to 10)
    for i in range(1, 11):
        optimizer.step()
        scheduler.step()
        expected_lr = 0.1 * (i / 10)
        assert math.isclose(scheduler.get_last_lr()[0], expected_lr, rel_tol=1e-5)

    # Step through decay (steps 11 to 15)
    for i in range(1, 6):
        optimizer.step()
        scheduler.step()
        expected_lr = 0.1 * (0.5**i)
        assert math.isclose(scheduler.get_last_lr()[0], expected_lr, rel_tol=1e-5)


def test_warmup_cosine_annealing_lr():
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = WarmupCosineAnnealingLR(
        optimizer, warmup_steps=10, max_steps=20, eta_min=0.01
    )

    expected_lr_0 = 0.1 * 1e-6
    assert math.isclose(scheduler.get_last_lr()[0], expected_lr_0, rel_tol=1e-5)

    # Step through warmup
    for i in range(1, 11):
        optimizer.step()
        scheduler.step()
        expected_lr = 0.1 * (i / 10)
        assert math.isclose(scheduler.get_last_lr()[0], expected_lr, rel_tol=1e-5)

    # Step through cosine decay
    for i in range(1, 11):
        optimizer.step()
        scheduler.step()
        progress = i / 10.0
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        expected_lr = 0.01 + (0.1 - 0.01) * cosine_decay
        assert math.isclose(scheduler.get_last_lr()[0], expected_lr, rel_tol=1e-5)

    # Beyond max_steps, it should stay at eta_min
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        assert math.isclose(scheduler.get_last_lr()[0], 0.01, rel_tol=1e-5)


def test_warmup_cosine_annealing_lr_raises_when_max_steps_none():
    import pytest

    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = WarmupCosineAnnealingLR(
        optimizer, warmup_steps=5, max_steps=None, eta_min=0.01
    )

    # Step through warmup without error (last_epoch goes from 0 to 4)
    for _ in range(4):
        optimizer.step()
        scheduler.step()

    # Step at warmup_steps (last_epoch=5) triggers cosine decay and should raise ValueError
    with pytest.raises(ValueError, match="max_steps must be specified"):
        optimizer.step()
        scheduler.step()


def test_schedulers_accept_max_steps():
    """configure_optimizers passes max_steps to every scheduler."""
    from callm.models.lr_scheduling import WarmupExponentialLR, WarmupCosineAnnealingLR

    for cls in (WarmupExponentialLR, WarmupCosineAnnealingLR):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
        cls(optimizer, warmup_steps=5, max_steps=100)


def test_warmup_cosine_annealing_lr_warmup_fraction():
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = WarmupCosineAnnealingLR(
        optimizer, max_steps=50, eta_min=0.01, warmup_fraction=0.1
    )
    assert scheduler.warmup_steps == 5

    # Warmup over the first 5 steps
    for i in range(1, 6):
        optimizer.step()
        scheduler.step()
        assert math.isclose(scheduler.get_last_lr()[0], 0.1 * i / 5, rel_tol=1e-5)

    # Cosine decay from step 5 to max_steps=50
    for step in range(6, 51):
        optimizer.step()
        scheduler.step()
        progress = (step - 5) / 45
        expected = 0.01 + (0.1 - 0.01) * 0.5 * (1 + math.cos(math.pi * progress))
        assert math.isclose(scheduler.get_last_lr()[0], expected, rel_tol=1e-5)
    assert math.isclose(scheduler.get_last_lr()[0], 0.01, rel_tol=1e-5)


def test_warmup_fraction_overrides_warmup_steps():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    scheduler = WarmupCosineAnnealingLR(
        optimizer, warmup_steps=100, max_steps=6930, warmup_fraction=0.05
    )
    assert (
        scheduler.warmup_steps == 346
    )  # round(0.05 * 6930) = round(346.5), banker's rounding


def test_warmup_fraction_requires_max_steps():
    import pytest

    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    with pytest.raises(ValueError, match="requires max_steps"):
        WarmupCosineAnnealingLR(optimizer, max_steps=None, warmup_fraction=0.05)


def test_warmup_fraction_out_of_range():
    import pytest

    for fraction in (1.0, -0.1):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
        with pytest.raises(ValueError, match="warmup_fraction must be in"):
            WarmupCosineAnnealingLR(optimizer, max_steps=100, warmup_fraction=fraction)
