"""Loss functions. See DESIGN.md sec. 4 and PLAN.md Stage 3.

Why Soft Dice (the primary loss)?
    Lesion segmentation is heavily class-imbalanced at the pixel level — the
    lesion is a small fraction of the image, and normal scans have *no*
    foreground at all (Option B). A pixel-wise cross-entropy is dominated by the
    easy background majority, so it under-segments small lesions. The Dice loss
    optimises overlap (F1) directly and is far less sensitive to that imbalance,
    which is exactly why the original Attention U-Net paper trains with it.

We reuse the authors' Soft-Dice *formulation* (Oktay et al., MIT) but make it
device-agnostic: their `One_Hot` hardcodes `.cuda()`, which breaks on MPS/CPU.
Here the one-hot encoding is built with `F.one_hot` on the target's own device.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss2D(nn.Module):
    """Soft Dice loss over a 2-channel-softmax output (background, lesion).

    loss = 1 - mean_over(batch, classes) [ 2*|P∩G| / (|P| + |G|) ]

    where P is the softmax probability map (soft, hence differentiable) and G is
    the one-hot ground truth. The `smooth` term (a) avoids division by zero when
    a class is absent from a patch — common here, since normal scans have an
    empty lesion channel — and (b) makes an empty-prediction/empty-target pair
    score ~1 instead of NaN.
    """

    def __init__(self, n_classes: int = 2, smooth: float = 0.01):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: (B, C, H, W) raw scores; target: (B, H, W) int64 in [0, C)."""
        b = logits.size(0)
        probs = F.softmax(logits, dim=1).view(b, self.n_classes, -1)
        # One-hot on the target's device (device-agnostic; replaces authors' .cuda()).
        target_1h = (F.one_hot(target.long(), self.n_classes)   # (B, H, W, C)
                     .permute(0, 3, 1, 2)                        # (B, C, H, W)
                     .contiguous().view(b, self.n_classes, -1).float())

        inter = torch.sum(probs * target_1h, dim=2) + self.smooth
        union = torch.sum(probs, dim=2) + torch.sum(target_1h, dim=2) + self.smooth
        dice = 2.0 * inter / union                     # (B, C)
        return 1.0 - dice.mean()


def focal_tversky_loss(logits, target_long, alpha=0.7, beta=0.3, gamma=0.75):
    """Focal-Tversky loss (STRETCH — full implementation lands in Stage 8 / T8.4).

    Rationale (for when we add it): the Tversky index generalises Dice by
    weighting false negatives (`alpha`) vs false positives (`beta`) separately;
    with `alpha > beta` it penalises *missed* lesion pixels harder — useful for
    small lesions. The focal exponent `gamma` further focuses training on hard,
    low-overlap cases. Deferred so the Core milestone stays lean.
    """
    raise NotImplementedError("Focal-Tversky is a stretch goal (PLAN.md T8.4)")


def get_loss(name: str, n_classes: int = 2):
    """Build a loss by name (driven by Config.loss_name)."""
    if name == "dice":
        return SoftDiceLoss2D(n_classes=n_classes)
    if name == "focal_tversky":
        raise NotImplementedError("Focal-Tversky is a stretch goal (PLAN.md T8.4)")
    raise ValueError(f"unknown loss name: {name!r}")
