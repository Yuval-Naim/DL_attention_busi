"""Training & evaluation. See DESIGN.md sec. 6 and PLAN.md Stage 4.

Design choices & justification:
- **Device-agnostic** (`cuda -> mps -> cpu`) so the exact same code runs on the
  local Mac (MPS) and on Colab (CUDA). No hardcoded devices anywhere.
- **Reproducibility**: `set_seed` seeds python/numpy/torch so a run can be
  repeated exactly (the whole project reports mean +/- std over 3 seeds).
- **Model selection on validation lesion-Dice** (not loss): Dice is the metric
  we ultimately report, and selecting on it avoids rewarding a model that lowers
  loss without improving overlap. `ReduceLROnPlateau(mode='max')` and early
  stopping both watch this same signal.
- **Honest metric split** (Option B): `evaluate` computes lesion Dice/IoU on
  benign+malignant only, and specificity on normal images separately, so empty
  normal masks never inflate the segmentation score.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from busi.losses import get_loss
from busi.metrics import dice_score, iou_score, specificity_on_normals
from busi.data import CLASS_TO_IDX

_LESION_LABELS = (CLASS_TO_IDX["benign"], CLASS_TO_IDX["malignant"])
_NORMAL_LABEL = CLASS_TO_IDX["normal"]


def set_seed(seed: int = 42) -> None:
    """Seed python / numpy / torch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> str:
    """Resolve the compute device. 'auto' -> cuda, else mps, else cpu."""
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    """Run one training epoch; return the mean training loss.

    Batches are `(image, mask, label)`; the label is unused during training
    (it only matters at evaluation, to split lesion vs normal metrics).
    """
    model.train()
    total, n = 0.0, 0
    for images, masks, _labels in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * images.size(0)
        n += images.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Evaluate the model, splitting metrics by class.

    Returns lesion Dice/IoU (benign+malignant), per-class Dice, specificity and
    false-positive rate (normals). Metrics with no supporting samples are NaN.
    """
    model.eval()
    dices, ious = [], []
    per_class = {"benign": [], "malignant": []}
    normal_preds = []

    for images, masks, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()      # (B, H, W) predicted class
        labels = [int(x) for x in labels]
        for b, lab in enumerate(labels):
            pred_mask = (preds[b] == 1)
            gt_mask = (masks[b] == 1)
            if lab in _LESION_LABELS:
                d = dice_score(pred_mask, gt_mask)
                dices.append(d)
                ious.append(iou_score(pred_mask, gt_mask))
                per_class["benign" if lab == CLASS_TO_IDX["benign"] else "malignant"].append(d)
            elif lab == _NORMAL_LABEL:
                normal_preds.append(pred_mask)

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    spec = specificity_on_normals(normal_preds) if normal_preds else float("nan")
    return {
        "lesion_dice": _mean(dices),
        "lesion_iou": _mean(ious),
        "dice_benign": _mean(per_class["benign"]),
        "dice_malignant": _mean(per_class["malignant"]),
        "specificity": spec,
        "fp_rate": (1.0 - spec) if spec == spec else float("nan"),  # nan-safe
    }


def fit(model, train_loader, val_loader, cfg) -> dict:
    """Full training loop: Adam + ReduceLROnPlateau, early stopping on val Dice,
    best-checkpointing. Returns history, best metric, and the checkpoint path.
    """
    device = get_device(cfg.device)
    model.to(device)
    loss_fn = get_loss(cfg.loss_name, cfg.n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=cfg.lr_patience)

    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    best_path = os.path.join(cfg.checkpoints_dir, f"{cfg.experiment_name}.pt")

    best, since_improved, history = -float("inf"), 0, []
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val = evaluate(model, val_loader, device)
        metric = val["lesion_dice"]
        metric = metric if metric == metric else -float("inf")   # NaN-safe
        scheduler.step(metric)
        history.append({"epoch": epoch, "train_loss": train_loss, **val})

        if metric > best or not os.path.exists(best_path):
            best, since_improved = metric, 0
            torch.save(model.state_dict(), best_path)
        else:
            since_improved += 1
            if since_improved >= cfg.early_stop_patience:
                break

    return {"history": history, "best_metric": best, "best_path": best_path}
