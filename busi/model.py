"""Models (Stage 2 — not yet implemented).

A single backbone (2D U-Net with deep supervision) with a *swappable*
skip-attention module, so all variants are compared fairly. See DESIGN.md
sec. 3 and PLAN.md Stage 2.

Skip-module contract: every attention module exposes
    forward(x, g=None) -> (x_filtered, spatial_map)   # spatial_map in [0,1]
so variants are drop-in interchangeable. The additive gate uses `g`; CBAM and
scSE are self-attention and ignore it. Keeping a spatial map from every variant
lets the interpretability analysis work across all of them.

Reused from the authors (imported in Stage 2):
    GridAttentionBlock2D, unetConv2, unetUp, init_weights, unet_2D
New here:
    UnetGridGatingSignal2D, UnetDsv2D, CBAM2D (desired), scSE2D (stretch),
    AttentionUNet2D, get_model
"""

from __future__ import annotations


def get_model(name, in_channels=1, n_classes=2, feature_scale=4):
    """Factory. `name` in {'unet','attention_unet','cbam_unet','scse_unet'}.

    All attention variants share one backbone; only the skip-attention module
    differs. Driven by Config.model_name.
    """
    raise NotImplementedError("Stage 2 / T2.5")
