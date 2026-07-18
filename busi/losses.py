"""Losses (Stage 3 — not yet implemented). See DESIGN.md sec. 4, PLAN.md Stage 3.

Primary: SoftDiceLoss2D = the authors' Soft Dice, made device-agnostic
(original One_Hot hardcodes .cuda()). Stretch: focal_tversky_loss.
Selected via Config.loss_name.
"""

from __future__ import annotations


class SoftDiceLoss2D:
    """Soft Dice loss over the 2-channel softmax output (device-agnostic)."""

    def __init__(self, n_classes=2):
        raise NotImplementedError("Stage 3 / T3.1")


def focal_tversky_loss(logits, target_long, alpha=0.7, beta=0.3, gamma=0.75):
    """Focal-Tversky loss (stretch, T8.4) — full impl deferred to the stretch milestone."""
    raise NotImplementedError("Stage 8 / T8.4")
