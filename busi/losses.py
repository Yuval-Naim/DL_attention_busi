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


def focal_tversky_loss(logits, target_long, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
    """Focal-Tversky loss on the lesion class (STRETCH).

    The Tversky index generalises Dice by weighting false negatives (`alpha`) and
    false positives (`beta`) separately; with `alpha > beta` it penalises *missed*
    lesion pixels harder — helpful for small lesions. The focal exponent `gamma`
    concentrates training on hard, low-overlap cases:

        TI  = TP / (TP + alpha*FN + beta*FP)     (per image, lesion class)
        FTL = mean_over_batch (1 - TI) ** gamma

    Computed on the softmax lesion probability (soft, differentiable).
    """
    probs = F.softmax(logits, dim=1)
    p1 = probs[:, 1]                              # (B, H, W) lesion probability
    g1 = (target_long == 1).float()
    tp = (p1 * g1).sum(dim=(1, 2))
    fn = (g1 * (1.0 - p1)).sum(dim=(1, 2))
    fp = (p1 * (1.0 - g1)).sum(dim=(1, 2))
    ti = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return ((1.0 - ti) ** gamma).mean()


class FocalTverskyLoss(nn.Module):
    """nn.Module wrapper around `focal_tversky_loss` (holds alpha/beta/gamma)."""

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def forward(self, logits, target):
        return focal_tversky_loss(logits, target, self.alpha, self.beta, self.gamma)


def get_loss(name: str, n_classes: int = 2, ft_alpha=0.7, ft_beta=0.3, ft_gamma=0.75):
    """Build a loss by name (driven by Config.loss_name + Config.ft_* params)."""
    if name == "dice":
        return SoftDiceLoss2D(n_classes=n_classes)
    if name == "focal_tversky":
        return FocalTverskyLoss(alpha=ft_alpha, beta=ft_beta, gamma=ft_gamma)
    raise ValueError(f"unknown loss name: {name!r}")
