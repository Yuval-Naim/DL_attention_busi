"""Metrics (Stage 3 — not yet implemented). See DESIGN.md sec. 5, PLAN.md Stage 3.

Lesion Dice/IoU (benign+malignant), specificity (normals), attention-focus
ratio (interpretability). Empty-mask convention: empty GT + empty pred -> 1.0;
empty GT + any predicted lesion -> 0.0.
"""

from __future__ import annotations


def dice_score(pred_mask, gt_mask, eps=1e-6):
    """Binary Dice (F1) between two (H, W) {0,1} masks."""
    raise NotImplementedError("Stage 3 / T3.3")


def iou_score(pred_mask, gt_mask, eps=1e-6):
    """Binary IoU (Jaccard) between two (H, W) {0,1} masks."""
    raise NotImplementedError("Stage 3 / T3.3")


def specificity_on_normals(pred_masks, area_thresh=0.005):
    """Fraction of normal-image predictions whose lesion area <= area_thresh."""
    raise NotImplementedError("Stage 3 / T3.4")


def attention_focus_ratio(att_map, gt_mask):
    """Attention mass inside the GT lesion / total attention mass, in [0, 1]."""
    raise NotImplementedError("Stage 3 / T3.5")
