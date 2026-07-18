"""Visualization (Stage 5 — not yet implemented). See DESIGN.md sec. 7, PLAN.md Stage 5.

Mask overlays, attention-map overlays, and comparison figures for the report.
"""

from __future__ import annotations


def overlay_mask(image, mask, color=(1, 0, 0), alpha=0.4):
    """Overlay a binary mask on a grayscale image; return an RGB (H, W, 3) array."""
    raise NotImplementedError("Stage 5 / T5.1")


def overlay_attention(image, att_map, cmap="jet", alpha=0.5):
    """Overlay an attention heatmap on an image; return an RGB (H, W, 3) array."""
    raise NotImplementedError("Stage 5 / T5.1")


def plot_prediction(image, gt, pred, att_maps=None):
    """Build a comparison figure (image | GT | pred | attention); return a Figure."""
    raise NotImplementedError("Stage 5 / T5.1")
