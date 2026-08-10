"""Data pipeline for BUSI.

Locked decisions:
- Include normal images (Option B). Real BUSI normal images ship with an
  all-black mask; some lesion images have multiple masks. We handle both:
  merge all mask files (logical OR); black or absent -> all-zero mask.
- Resize image bilinear, mask nearest-neighbor (keep mask binary).
- 2-channel-softmax target: mask is a Long tensor in {0, 1}.
- albumentations for augmentation, applied identically to image + mask.

Design notes (refinements over the first DESIGN draft):
- `list_busi_samples` returns paths **relative to `root`** so the persisted
  split is portable across machines (local <-> Colab/Drive).
- `BUSIDataset(samples, root, ...)` joins `root` to those relative paths.
- `__getitem__` returns `(image, mask, label_idx)`; the label lets `evaluate`
  compute lesion metrics on benign+malignant and specificity on normal.
"""

from __future__ import annotations

import json
import os
from glob import glob

# Disable albumentations' online version check (avoids a network call + SSL warning).
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Class label <-> integer index (kept separate from the 2 segmentation classes).
CLASS_TO_IDX = {"benign": 0, "malignant": 1, "normal": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


# ---------------------------------------------------------------------------
# Listing & mask merging
# ---------------------------------------------------------------------------
def list_busi_samples(root, classes=("benign", "malignant", "normal")):
    """List BUSI samples under `root`.

    Expects `root/<class>/<class> (n).png` images and `<class> (n)_mask*.png`
    masks. Returns a sorted list of dicts with paths **relative to `root`**:
        {'image': rel_path, 'masks': [rel_paths], 'label': class_name}
    (`masks` may be empty; normal images typically have one all-black mask.)
    """
    samples = []
    for cls in classes:
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        pngs = sorted(os.path.basename(p) for p in glob(os.path.join(cls_dir, "*.png")))
        images = [f for f in pngs if "_mask" not in f]
        # Group masks by the image stem that precedes "_mask".
        masks_by_stem = {}
        for f in pngs:
            if "_mask" in f:
                stem = f.split("_mask")[0]
                masks_by_stem.setdefault(stem, []).append(f)
        for img in images:
            stem = img[:-4]  # strip ".png"
            mask_files = sorted(masks_by_stem.get(stem, []))
            samples.append({
                "image": os.path.join(cls, img),
                "masks": [os.path.join(cls, m) for m in mask_files],
                "label": cls,
            })
    return samples


def merge_masks(mask_paths, size):
    """OR-merge mask files into one binary (H, W) uint8 mask in {0, 1}.

    `size` is (H, W). Each mask is read grayscale, resized nearest-neighbor to
    `size`, thresholded (>0 -> 1) and OR-ed. An empty `mask_paths` (or masks
    that fail to load) yields an all-zero mask.
    """
    h, w = int(size[0]), int(size[1])
    out = np.zeros((h, w), dtype=np.uint8)
    for p in mask_paths:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out |= (m > 0).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Augmentation / transforms
# ---------------------------------------------------------------------------
def build_transforms(img_size=256, train=True, hflip=True, rotate_deg=15.0,
                     scale=(0.9, 1.1), brightness=0.10):
    """Return an albumentations pipeline (image bilinear, mask nearest).

    The validation/test pipeline is just a resize. The train pipeline adds flip
    / affine (scale+rotate) / brightness-contrast, applied identically to image
    and mask (masks always interpolate nearest to stay binary).
    """
    import albumentations as A

    resize = A.Resize(img_size, img_size,
                      interpolation=cv2.INTER_LINEAR,
                      mask_interpolation=cv2.INTER_NEAREST)
    if not train:
        return A.Compose([resize])

    aug = [resize]
    if hflip:
        aug.append(A.HorizontalFlip(p=0.5))
    aug.append(A.Affine(scale=scale, rotate=(-rotate_deg, rotate_deg),
                        translate_percent=(0.0, 0.05), p=0.7,
                        interpolation=cv2.INTER_LINEAR,
                        mask_interpolation=cv2.INTER_NEAREST))
    aug.append(A.RandomBrightnessContrast(brightness_limit=brightness,
                                          contrast_limit=brightness, p=0.5))
    return A.Compose(aug)


def transforms_from_config(cfg, train):
    """Build train/val transforms straight from a Config object."""
    return build_transforms(img_size=cfg.img_size, train=train,
                            hflip=cfg.aug_hflip, rotate_deg=cfg.aug_rotate_deg,
                            scale=tuple(cfg.aug_scale), brightness=cfg.aug_brightness)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def make_split(samples, ratios=(0.7, 0.15, 0.15), seed=42):
    """Stratified (by class) split -> {'train': [...], 'val': [...], 'test': [...]}."""
    from sklearn.model_selection import train_test_split

    labels = [s["label"] for s in samples]
    idx = list(range(len(samples)))
    train_idx, temp_idx = train_test_split(
        idx, train_size=ratios[0], random_state=seed, stratify=labels)
    temp_labels = [labels[i] for i in temp_idx]
    val_frac = ratios[1] / (ratios[1] + ratios[2])
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_frac, random_state=seed, stratify=temp_labels)
    return {
        "train": [samples[i] for i in train_idx],
        "val": [samples[i] for i in val_idx],
        "test": [samples[i] for i in test_idx],
    }


def save_split(split, path):
    """Persist a split (relative-path samples) to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path):
    """Load a split from JSON."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BUSIDataset(Dataset):
    """torch Dataset over relative-path samples rooted at `root`.

    Yields `(image, mask, label_idx)`:
        image: FloatTensor (1, H, W) in [0, 1]
        mask:  LongTensor  (H, W) in {0, 1}
        label_idx: int (see CLASS_TO_IDX)
    """

    def __init__(self, samples, root, transform=None, img_size=256):
        super().__init__()
        self.samples = samples
        self.root = root
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img_path = os.path.join(self.root, s["image"])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(img_path)
        mask_paths = [os.path.join(self.root, m) for m in s["masks"]]
        mask = merge_masks(mask_paths, size=image.shape[:2])  # (H, W) {0,1}

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        image_t = torch.from_numpy(np.ascontiguousarray(image)).float().div(255.0).unsqueeze(0)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).long()
        label_idx = CLASS_TO_IDX[s["label"]]
        return image_t, mask_t, label_idx
