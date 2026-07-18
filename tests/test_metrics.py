"""Stage 3 metric tests, including known-value checks (Dice=0.667, IoU=0.5)."""

import numpy as np

from busi import metrics as MT


def _square(h=10, w=10, sl=None):
    m = np.zeros((h, w), np.uint8)
    if sl is not None:
        m[sl] = 1
    return m


# --- Dice ------------------------------------------------------------------
def test_dice_identical():
    m = _square(sl=(slice(0, 5), slice(0, 5)))
    assert MT.dice_score(m, m) == 1.0


def test_dice_disjoint():
    a = _square(sl=(slice(0, 3), slice(0, 3)))
    b = _square(sl=(slice(7, 10), slice(7, 10)))
    assert MT.dice_score(a, b) == 0.0


def test_dice_known():
    pred = np.ones((10, 10), np.uint8)                  # all foreground (100)
    gt = _square(sl=(slice(0, 5), slice(0, 10)))        # half (50)
    assert abs(MT.dice_score(pred, gt) - (2 * 50) / (100 + 50)) < 1e-3   # 0.667


# --- IoU -------------------------------------------------------------------
def test_iou_identical():
    m = _square(sl=(slice(0, 5), slice(0, 5)))
    assert MT.iou_score(m, m) == 1.0


def test_iou_disjoint():
    a = _square(sl=(slice(0, 3), slice(0, 3)))
    b = _square(sl=(slice(7, 10), slice(7, 10)))
    assert MT.iou_score(a, b) == 0.0


def test_iou_known():
    pred = np.ones((10, 10), np.uint8)                  # 100
    gt = _square(sl=(slice(0, 5), slice(0, 10)))        # 50; union 100, inter 50
    assert abs(MT.iou_score(pred, gt) - 0.5) < 1e-3


# --- empty-mask conventions ------------------------------------------------
def test_empty_both():
    z = _square()
    assert MT.dice_score(z, z) == 1.0 and MT.iou_score(z, z) == 1.0


def test_empty_gt_nonempty_pred():
    pred = _square(sl=(slice(0, 3), slice(0, 3)))
    gt = _square()
    assert MT.dice_score(pred, gt) == 0.0 and MT.iou_score(pred, gt) == 0.0


# --- specificity -----------------------------------------------------------
def test_specificity_all_clean():
    preds = [_square() for _ in range(5)]                # all empty
    assert MT.specificity_on_normals(preds) == 1.0


def test_specificity_all_hallucinate():
    preds = [np.ones((10, 10), np.uint8) for _ in range(5)]  # all full lesion
    assert MT.specificity_on_normals(preds) == 0.0


# --- attention-focus ratio -------------------------------------------------
def test_focus_all_inside():
    gt = _square(sl=(slice(0, 5), slice(0, 5)))
    att = np.zeros((10, 10)); att[0:5, 0:5] = 0.9       # mass only inside gt
    assert abs(MT.attention_focus_ratio(att, gt) - 1.0) < 1e-6


def test_focus_all_outside():
    gt = _square(sl=(slice(0, 5), slice(0, 5)))
    att = np.zeros((10, 10)); att[7:10, 7:10] = 0.9     # mass only outside gt
    assert MT.attention_focus_ratio(att, gt) == 0.0


def test_focus_uniform():
    gt = _square(sl=(slice(0, 2), slice(0, 10)))         # 20/100 = 0.2 of image
    att = np.ones((10, 10))                              # uniform attention
    assert abs(MT.attention_focus_ratio(att, gt) - 0.2) < 1e-6
