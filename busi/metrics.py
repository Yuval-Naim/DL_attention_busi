"""Evaluation metrics. See DESIGN.md sec. 5 and PLAN.md Stage 3.

Three families, each answering a different question:

- **Dice / IoU** — overlap between the predicted lesion mask and ground truth.
  Reported on lesion images only (benign + malignant). Dice (F1) is the field-
  standard segmentation score; IoU (Jaccard) is a stricter sibling.

- **Specificity on normals** — the whole point of Option B (including healthy
  scans): does the model *stay silent* on tissue with no lesion? We count a
  normal prediction as correct if its predicted lesion area is <= a small
  threshold (a few stray pixels are tolerated). Reported separately so it never
  inflates the lesion Dice.

- **Attention-focus ratio** — an interpretability metric: what fraction of the
  attention mass lands inside the true lesion. High = the gate "looks where it
  should", which is the paper's central claim ("learning where to look").

Empty-mask convention (needed because normal GT masks are all-zero):
    empty GT + empty prediction -> 1.0   (correctly predicted nothing)
    empty GT + any prediction   -> 0.0   (a false positive)

All functions accept numpy arrays or torch tensors of shape (H, W).
"""

from __future__ import annotations

import numpy as np


def _to_np(x) -> np.ndarray:
    """Accept a torch tensor or numpy array; return a detached numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def dice_score(pred_mask, gt_mask, eps: float = 1e-6) -> float:
    """Binary Dice (F1) between two {0,1} masks. Both-empty -> 1.0 (see convention).

    `eps` is unused in the division (the both-empty case is guarded explicitly,
    so the denominator is never zero); it is kept only for signature stability.
    """
    p = (_to_np(pred_mask) > 0).astype(np.float64)
    g = (_to_np(gt_mask) > 0).astype(np.float64)
    p_sum, g_sum = p.sum(), g.sum()
    if p_sum == 0 and g_sum == 0:
        return 1.0                     # correctly predicted "nothing"
    inter = float((p * g).sum())
    return 2.0 * inter / (p_sum + g_sum)


def iou_score(pred_mask, gt_mask, eps: float = 1e-6) -> float:
    """Binary IoU (Jaccard) between two {0,1} masks. Both-empty -> 1.0.

    `eps` kept for signature stability; the zero-union case is guarded explicitly.
    """
    p = (_to_np(pred_mask) > 0).astype(np.float64)
    g = (_to_np(gt_mask) > 0).astype(np.float64)
    inter = float((p * g).sum())
    union = float(p.sum() + g.sum() - inter)
    if union == 0:
        return 1.0                     # both empty
    return inter / union


def specificity_on_normals(pred_masks, area_thresh: float = 0.005) -> float:
    """Fraction of NORMAL-image predictions that are (near-)empty.

    `pred_masks` is an iterable of predicted {0,1} masks for normal images. A
    prediction counts as correct (no false positive) if the predicted lesion
    area fraction <= `area_thresh` (default 0.5% of pixels — tolerates a few
    stray pixels). Returns NaN if there are no normal images.
    """
    preds = list(pred_masks)
    if not preds:
        return float("nan")
    correct = 0
    for m in preds:
        p = (_to_np(m) > 0)
        if p.mean() <= area_thresh:
            correct += 1
    return correct / len(preds)


def attention_focus_ratio(att_map, gt_mask) -> float:
    """Attention mass inside the GT lesion / total attention mass, in [0, 1].

    `att_map` is a (H, W) map in [0, 1]. Returns 0.0 if there is no attention
    mass at all (degenerate). Higher means the attention concentrates on the
    lesion — the interpretability signal for "learning where to look".
    """
    a = _to_np(att_map).astype(np.float64)
    g = (_to_np(gt_mask) > 0).astype(np.float64)
    total = float(a.sum())
    if total == 0:
        return 0.0
    return float((a * g).sum()) / total
