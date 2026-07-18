"""Visualization for the report. See DESIGN.md sec. 7 and PLAN.md Stage 5.

Two overlay helpers return plain RGB arrays (no plotting backend needed, so
they're trivial to test and to embed anywhere), plus one figure builder:

- `overlay_mask`      — tint the lesion region on the grayscale scan. Used for
                         GT-vs-prediction qualitative panels.
- `overlay_attention` — heatmap the attention map on the scan. This is the
                         interpretability visual — "where did the gate look?".
                         Attention maps live at a coarser skip resolution, so we
                         upsample them to the image size before overlaying.
- `plot_prediction`   — assemble a one-row comparison figure (image | GT | pred
                         | attention...) for the report.

Inputs may be numpy arrays or torch tensors; images are normalized to [0, 1].
"""

from __future__ import annotations

import cv2
import matplotlib
import numpy as np


def _to_hw(x) -> np.ndarray:
    """Return a (H, W) float image in [0, 1] from tensor/array of shape (1,H,W)/(H,W)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {x.shape}")
    if x.max() > 1.0:            # assume 0..255
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _to_mask(x) -> np.ndarray:
    """Return a (H, W) boolean mask."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.squeeze(np.asarray(x)) > 0


def overlay_mask(image, mask, color=(1.0, 0.0, 0.0), alpha=0.4) -> np.ndarray:
    """Tint `mask` over grayscale `image`; return RGB (H, W, 3) in [0, 1]."""
    img = _to_hw(image)
    m = _to_mask(mask)
    rgb = np.stack([img, img, img], axis=-1)
    color = np.asarray(color, dtype=np.float64)
    rgb[m] = (1.0 - alpha) * rgb[m] + alpha * color
    return np.clip(rgb, 0.0, 1.0)


def overlay_attention(image, att_map, cmap="jet", alpha=0.5) -> np.ndarray:
    """Overlay an attention heatmap on `image`; return RGB (H, W, 3) in [0, 1].

    The attention map is min-max normalized and upsampled (bilinear) to the image
    size before blending.
    """
    img = _to_hw(image)
    att = np.asarray(att_map.detach().cpu().numpy() if hasattr(att_map, "detach") else att_map,
                     dtype=np.float64)
    att = np.squeeze(att)
    if att.shape != img.shape:
        att = cv2.resize(att, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    rng = att.max() - att.min()
    att = (att - att.min()) / rng if rng > 0 else np.zeros_like(att)
    heat = matplotlib.colormaps[cmap](att)[..., :3]        # (H, W, 3)
    rgb = np.stack([img, img, img], axis=-1)
    out = (1.0 - alpha) * rgb + alpha * heat
    return np.clip(out, 0.0, 1.0)


def plot_prediction(image, gt, pred, att_maps=None):
    """Build a one-row comparison figure and return the matplotlib Figure.

    Panels: original | GT overlay (green) | prediction overlay (red) | one
    attention overlay per entry in `att_maps` (a dict like {'att3': map, ...}).
    """
    import matplotlib.pyplot as plt

    panels = [("image", None), ("ground truth", gt), ("prediction", pred)]
    att_items = list((att_maps or {}).items())
    ncols = len(panels) + len(att_items)

    fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3))
    if ncols == 1:
        axes = [axes]

    img = _to_hw(image)
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("image")
    axes[1].imshow(overlay_mask(image, gt, color=(0, 1, 0))); axes[1].set_title("ground truth")
    axes[2].imshow(overlay_mask(image, pred, color=(1, 0, 0))); axes[2].set_title("prediction")
    for ax, (name, amap) in zip(axes[3:], att_items):
        ax.imshow(overlay_attention(image, amap)); ax.set_title(name)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig
