"""Data pipeline for BUSI (Stage 1 — not yet implemented).

Reads BUSI PNG image/mask pairs, merges masks, builds a reproducible stratified
split, and serves tensors for training. See DESIGN.md sec. 2 and PLAN.md Stage 1.

Locked decisions: include normal images (empty masks); resize masks with
nearest-neighbor; 2-channel-softmax target (mask in {0,1} long); albumentations
for augmentation with image+mask sync.
"""

from __future__ import annotations

# NOTE: implementation lands in Stage 1 (T1.2-T1.6). Signatures are fixed here
# so tests and callers can be written against them.


def list_busi_samples(root, classes=("benign", "malignant", "normal")):
    """List BUSI samples.

    Returns a list of dicts: {'image': path, 'masks': [paths] (empty for
    normal), 'label': 'benign'|'malignant'|'normal'}.
    """
    raise NotImplementedError("Stage 1 / T1.2")


def merge_masks(mask_paths, size):
    """OR-merge mask files into one binary (H, W) uint8 mask in {0, 1}.

    Empty `mask_paths` (normal images) -> all-zero mask. Resize uses
    nearest-neighbor to keep the mask binary.
    """
    raise NotImplementedError("Stage 1 / T1.3")


def build_transforms(img_size=256, train=True):
    """Return an albumentations pipeline (image bilinear + /255, mask nearest).

    Train pipeline adds flip/rotate/scale/brightness applied identically to
    image and mask.
    """
    raise NotImplementedError("Stage 1 / T1.4")


def make_split(samples, ratios=(0.7, 0.15, 0.15), seed=42):
    """Stratified (3-class) split -> {'train': [...], 'val': [...], 'test': [...]}."""
    raise NotImplementedError("Stage 1 / T1.5")


def save_split(split, path):
    """Persist a split to JSON (filename lists)."""
    raise NotImplementedError("Stage 1 / T1.5")


def load_split(path):
    """Load a split from JSON."""
    raise NotImplementedError("Stage 1 / T1.5")


class BUSIDataset:
    """torch Dataset yielding (image (1,H,W) float32 [0,1], mask (H,W) int64 {0,1})."""

    def __init__(self, samples, transform=None, img_size=256):
        raise NotImplementedError("Stage 1 / T1.6")
