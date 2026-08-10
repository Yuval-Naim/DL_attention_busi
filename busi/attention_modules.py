"""Alternative attention modules for the comparison study.

These plug into the same backbone as the paper's additive gate via the shared
skip-module contract:

    forward(x, g=None) -> (x_filtered, spatial_map)   # spatial_map in [0, 1]

Unlike the additive gate, CBAM and scSE are *self-attention* — they recalibrate
the skip feature `x` using `x` itself and ignore the decoder gating signal `g`.
Each still exposes a spatial attention map, so the interpretability analysis
("where does it look?") works across all variants.

- CBAM (Woo et al., 2018): channel attention then spatial attention.  [Stage 7]
- scSE (Roy et al., 2018, from medical segmentation): parallel spatial + channel
  squeeze-and-excitation.                                              [Stage 8]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks_other import init_weights


class CBAM2D(nn.Module):
    """Convolutional Block Attention Module (2D).

    Channel attention (shared MLP over avg+max pooled descriptors) reweights
    *which* feature channels matter; spatial attention (a conv over channel-pooled
    maps) reweights *where* to attend. The returned spatial map is the CBAM
    spatial-attention map, used for interpretability.
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        mid = max(channels // reduction, 1)   # guard tiny channel counts
        # Channel attention: shared MLP applied to avg- and max-pooled descriptors.
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True))
        # Spatial attention: conv over [avg, max] across the channel axis.
        pad = spatial_kernel // 2
        self.spatial_conv = nn.Conv2d(2, 1, spatial_kernel, padding=pad, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming")

    def forward(self, x, g=None):
        # --- channel attention ---
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        ca = torch.sigmoid(avg + mx)                     # (B, C, 1, 1)
        x = x * ca
        # --- spatial attention ---
        avg_c = x.mean(dim=1, keepdim=True)
        max_c = x.max(dim=1, keepdim=True).values
        sa = torch.sigmoid(self.spatial_conv(torch.cat([avg_c, max_c], dim=1)))  # (B,1,H,W)
        return x * sa, sa


class scSE2D(nn.Module):
    """Concurrent Spatial and Channel Squeeze-&-Excitation (Roy et al., 2018).

    Runs two lightweight recalibrations in parallel and adds them:
    - **cSE** (channel SE): global-pool -> bottleneck MLP -> per-channel gate.
    - **sSE** (spatial SE): 1x1 conv -> per-pixel gate.
    Designed for medical segmentation and cheaper than CBAM. The returned spatial
    map is the sSE gate (used for interpretability).
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1))
        self.sse = nn.Conv2d(channels, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming")

    def forward(self, x, g=None):
        cse = torch.sigmoid(self.cse(x))     # (B, C, 1, 1) channel gate
        sse = torch.sigmoid(self.sse(x))     # (B, 1, H, W) spatial gate
        return x * cse + x * sse, sse
