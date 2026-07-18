"""Stage 5 visualization tests (headless: force the Agg backend)."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

from busi import viz


def test_overlay_mask_shape():
    img = np.random.rand(32, 32)
    mask = np.zeros((32, 32), np.uint8); mask[8:20, 8:20] = 1
    out = viz.overlay_mask(img, mask)
    assert out.shape == (32, 32, 3)
    assert 0.0 <= out.min() and out.max() <= 1.0
    # tinted region differs from the untouched background
    assert not np.allclose(out[10, 10], out[0, 0])


def test_overlay_mask_accepts_tensors():
    img = torch.rand(1, 16, 16)                 # (1,H,W) like a dataset sample
    mask = torch.zeros(16, 16, dtype=torch.long); mask[2:6, 2:6] = 1
    out = viz.overlay_mask(img, mask)
    assert out.shape == (16, 16, 3)


def test_overlay_attention_shape_and_resize():
    img = np.random.rand(32, 32)
    att = np.random.rand(8, 8)                   # coarse map -> must upsample
    out = viz.overlay_attention(img, att)
    assert out.shape == (32, 32, 3)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_plot_prediction_smoke():
    img = np.random.rand(32, 32)
    gt = np.zeros((32, 32), np.uint8); gt[4:12, 4:12] = 1
    pred = np.zeros((32, 32), np.uint8); pred[5:13, 5:13] = 1
    fig = viz.plot_prediction(img, gt, pred)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_prediction_with_attention():
    img = np.random.rand(32, 32)
    gt = np.zeros((32, 32), np.uint8); gt[4:12, 4:12] = 1
    pred = gt.copy()
    att_maps = {"att3": np.random.rand(8, 8), "att4": np.random.rand(4, 4)}
    fig = viz.plot_prediction(img, gt, pred, att_maps=att_maps)
    assert len(fig.axes) == 5          # image + gt + pred + 2 attention panels
    import matplotlib.pyplot as plt
    plt.close(fig)
