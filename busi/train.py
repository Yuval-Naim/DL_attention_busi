"""Training & evaluation (Stage 4 — not yet implemented). See DESIGN.md sec. 6, PLAN.md Stage 4.

Device-agnostic (cuda -> mps -> cpu), seeded, early stopping, best-checkpointing,
ReduceLROnPlateau on val Dice. evaluate() splits metrics by class: lesion
Dice/IoU on benign+malignant, specificity on normals.
"""

from __future__ import annotations


def set_seed(seed=42):
    """Seed python / numpy / torch for reproducibility."""
    raise NotImplementedError("Stage 4 / T4.1")


def get_device(preference="auto"):
    """Resolve the compute device: 'auto' -> cuda, else mps, else cpu."""
    raise NotImplementedError("Stage 4 / T4.1")


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Run one training epoch; return mean train loss."""
    raise NotImplementedError("Stage 4 / T4.2")


def evaluate(model, loader, device):
    """Return dict: lesion_dice, lesion_iou, dice_benign, dice_malignant, specificity, fp_rate."""
    raise NotImplementedError("Stage 4 / T4.3")


def fit(model, train_loader, val_loader, cfg):
    """Full training loop with early stopping + best-checkpointing; return history + best path."""
    raise NotImplementedError("Stage 4 / T4.4")
