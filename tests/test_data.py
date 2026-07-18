"""Stage 1 data-pipeline tests. Uses synthetic BUSI-like fixtures so it runs
without the real dataset. The real-count test auto-skips if data is absent."""

import os

import cv2
import numpy as np
import pytest

from busi import data
from busi.config import Config

SYNTH_COUNTS = {"benign": 10, "malignant": 8, "normal": 6}


@pytest.fixture
def fake_busi(tmp_path):
    """Build a tiny BUSI-like tree: images + masks with realistic naming."""
    root = tmp_path / "BUSI"
    rng = np.random.default_rng(0)
    for cls, n in SYNTH_COUNTS.items():
        d = root / cls
        d.mkdir(parents=True)
        for i in range(1, n + 1):
            img = rng.integers(0, 255, (64, 64), dtype=np.uint8)
            cv2.imwrite(str(d / f"{cls} ({i}).png"), img)
            mask = np.zeros((64, 64), np.uint8)
            if cls != "normal":
                mask[10:30, 10:30] = 255            # a lesion square
            cv2.imwrite(str(d / f"{cls} ({i})_mask.png"), mask)  # normal mask stays black
    # Give benign (1) a second, disjoint mask to exercise OR-merge.
    extra = np.zeros((64, 64), np.uint8)
    extra[40:50, 40:50] = 255
    cv2.imwrite(str(root / "benign" / "benign (1)_mask_1.png"), extra)
    return str(root)


# --- listing ---------------------------------------------------------------
def test_list_counts_synthetic(fake_busi):
    samples = data.list_busi_samples(fake_busi)
    by = {c: [s for s in samples if s["label"] == c] for c in SYNTH_COUNTS}
    assert {c: len(v) for c, v in by.items()} == SYNTH_COUNTS
    assert len(samples) == sum(SYNTH_COUNTS.values())
    # every benign/malignant image has >= 1 mask; normal has exactly its (black) mask
    for s in samples:
        assert len(s["masks"]) >= 1


def test_list_multiple_masks(fake_busi):
    samples = data.list_busi_samples(fake_busi)
    b1 = next(s for s in samples if s["image"].endswith("benign (1).png"))
    assert len(b1["masks"]) == 2  # _mask.png + _mask_1.png


def test_paths_are_relative(fake_busi):
    samples = data.list_busi_samples(fake_busi)
    assert all(not os.path.isabs(s["image"]) for s in samples)
    assert samples[0]["image"].startswith("benign" + os.sep)


# --- merge_masks -----------------------------------------------------------
def test_merge_or(tmp_path):
    a = np.zeros((32, 32), np.uint8); a[0:10, 0:10] = 255
    b = np.zeros((32, 32), np.uint8); b[20:30, 20:30] = 255
    pa, pb = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    cv2.imwrite(pa, a); cv2.imwrite(pb, b)
    merged = data.merge_masks([pa, pb], (32, 32))
    assert merged[5, 5] == 1 and merged[25, 25] == 1
    assert set(np.unique(merged)).issubset({0, 1})


def test_merge_empty_is_zeros():
    merged = data.merge_masks([], (16, 24))
    assert merged.shape == (16, 24)
    assert merged.max() == 0


def test_merge_binary_after_resize(tmp_path):
    m = np.zeros((64, 64), np.uint8); m[10:40, 10:40] = 255
    p = str(tmp_path / "m.png"); cv2.imwrite(p, m)
    merged = data.merge_masks([p], (256, 256))  # upsampled
    assert merged.shape == (256, 256)
    assert set(np.unique(merged)).issubset({0, 1})  # nearest -> still binary


def test_normal_empty_mask(fake_busi):
    samples = data.list_busi_samples(fake_busi)
    normal = next(s for s in samples if s["label"] == "normal")
    paths = [os.path.join(fake_busi, m) for m in normal["masks"]]
    assert data.merge_masks(paths, (64, 64)).max() == 0


# --- splitting -------------------------------------------------------------
def test_split_no_leakage(fake_busi):
    split = data.make_split(data.list_busi_samples(fake_busi), seed=42)
    imgs = {k: {s["image"] for s in split[k]} for k in ("train", "val", "test")}
    assert imgs["train"] & imgs["val"] == set()
    assert imgs["train"] & imgs["test"] == set()
    assert imgs["val"] & imgs["test"] == set()
    total = sum(len(split[k]) for k in split)
    assert total == sum(SYNTH_COUNTS.values())


def test_split_stratified(fake_busi):
    split = data.make_split(data.list_busi_samples(fake_busi), seed=42)
    for k in ("train", "val", "test"):
        labels = {s["label"] for s in split[k]}
        assert labels == set(SYNTH_COUNTS)  # every class present in every split


def test_split_deterministic(fake_busi):
    samples = data.list_busi_samples(fake_busi)
    s1 = data.make_split(samples, seed=42)
    s2 = data.make_split(samples, seed=42)
    for k in s1:
        assert [x["image"] for x in s1[k]] == [x["image"] for x in s2[k]]


def test_split_roundtrip(tmp_path, fake_busi):
    split = data.make_split(data.list_busi_samples(fake_busi), seed=42)
    p = str(tmp_path / "split.json")
    data.save_split(split, p)
    loaded = data.load_split(p)
    assert [s["image"] for s in loaded["train"]] == [s["image"] for s in split["train"]]


# --- dataset ---------------------------------------------------------------
def test_dataset_item_shapes(fake_busi):
    samples = data.list_busi_samples(fake_busi)
    tf = data.build_transforms(img_size=256, train=False)
    ds = data.BUSIDataset(samples, root=fake_busi, transform=tf, img_size=256)
    img, mask, label = ds[0]
    assert tuple(img.shape) == (1, 256, 256) and img.dtype.is_floating_point
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    assert tuple(mask.shape) == (256, 256) and str(mask.dtype) == "torch.int64"
    assert set(mask.unique().tolist()).issubset({0, 1})
    assert isinstance(label, int) and label in (0, 1, 2)
    assert len(ds) == sum(SYNTH_COUNTS.values())


def test_transform_mask_binary(fake_busi):
    samples = data.list_busi_samples(fake_busi)
    tf = data.build_transforms(img_size=256, train=True)   # with augmentation
    ds = data.BUSIDataset(samples, root=fake_busi, transform=tf, img_size=256)
    _, mask, _ = ds[0]
    assert set(mask.unique().tolist()).issubset({0, 1})


def test_geometry_synced():
    import albumentations as A
    img = np.zeros((64, 64), np.uint8); img[:20, :20] = 255      # top-left block
    mask = np.zeros((64, 64), np.uint8); mask[:20, :20] = 1
    out = A.Compose([A.HorizontalFlip(p=1.0)])(image=img, mask=mask)
    oi, om = out["image"], out["mask"]
    # after a horizontal flip both move to the top-right, together
    assert oi[:20, -20:].mean() > oi[:20, :20].mean()
    assert om[:20, -20:].sum() > om[:20, :20].sum()
    assert (om[:20, -20:] > 0).all()


# --- real dataset (skips if absent) ----------------------------------------
def test_list_counts_real():
    root = Config().data_root
    if not os.path.isdir(root):
        pytest.skip(f"real BUSI not present at {root!r}")
    samples = data.list_busi_samples(root)
    counts = {c: sum(s["label"] == c for s in samples) for c in ("benign", "malignant", "normal")}
    assert counts == {"benign": 437, "malignant": 210, "normal": 133}
