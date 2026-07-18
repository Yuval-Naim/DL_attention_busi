"""Stage 3 loss tests (Soft Dice). Focal-Tversky tests are deferred to Stage 8."""

import pytest
import torch

from busi.losses import SoftDiceLoss2D, get_loss


def _target(b=2, h=8, w=8):
    t = torch.zeros(b, h, w, dtype=torch.long)
    t[:, 2:6, 2:6] = 1                       # a lesion square
    return t


def _logits_for(target, correct=True, scale=10.0):
    """Confident logits that argmax to `target` (correct) or its opposite."""
    oh = torch.nn.functional.one_hot(target, 2).permute(0, 3, 1, 2).float()
    if not correct:
        oh = 1.0 - oh
    return oh * scale


def test_dice_perfect():
    t = _target()
    loss = SoftDiceLoss2D()( _logits_for(t, correct=True), t)
    assert loss.item() < 0.05


def test_dice_worst():
    t = _target()
    loss = SoftDiceLoss2D()(_logits_for(t, correct=False), t)
    assert loss.item() > 0.9


def test_dice_differentiable():
    t = _target()
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    loss = SoftDiceLoss2D()(logits, t)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_dice_device_agnostic():
    # Must run without any hardcoded .cuda() (the bug we fixed).
    t = _target()
    logits = _logits_for(t)
    assert SoftDiceLoss2D()(logits, t).item() < 0.05     # CPU
    if torch.backends.mps.is_available():
        loss = SoftDiceLoss2D()(logits.to("mps"), t.to("mps"))
        assert loss.item() < 0.05


def test_loss_range():
    t = _target()
    loss = SoftDiceLoss2D()(torch.randn(2, 2, 8, 8), t).item()
    assert 0.0 <= loss <= 1.0


def test_get_loss():
    assert isinstance(get_loss("dice"), SoftDiceLoss2D)
    with pytest.raises(NotImplementedError):
        get_loss("focal_tversky")
