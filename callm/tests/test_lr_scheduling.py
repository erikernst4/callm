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
