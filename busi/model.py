"""Models. See DESIGN.md sec. 3 and PLAN.md Stage 2.

A single 2D U-Net backbone (with deep supervision) whose *skip-attention module*
is swappable, so all variants are compared fairly:
    unet            -> plain U-Net baseline (authors' unet_2D, verbatim)
    attention_unet  -> paper's additive attention gate  (this stage)
    cbam_unet       -> CBAM self-attention              (Stage 7)
    scse_unet       -> scSE self-attention              (Stage 8)

Skip-module contract: `forward(x, g=None) -> (x_filtered, spatial_map)` with
`spatial_map` in [0, 1] at x's spatial size. The additive gate uses `g`; CBAM /
scSE are self-attention and ignore it. Keeping a spatial map from every variant
lets the interpretability analysis work across all of them.

Reused from the authors (MIT): GridAttentionBlock2D, unetConv2, unetUp,
init_weights, unet_2D. New here: UnetGridGatingSignal2D, UnetDsv2D,
AdditiveAttentionGate, AttentionUNet2D, get_model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.grid_attention_layer import GridAttentionBlock2D
from models.networks.utils import unetConv2, unetUp
from models.networks_other import init_weights
from models.networks.unet_2D import unet_2D


# ---------------------------------------------------------------------------
# New 2D building blocks (2D copies of the authors' 3D-only versions)
# ---------------------------------------------------------------------------
class UnetGridGatingSignal2D(nn.Module):
    """2D copy of UnetGridGatingSignal3: 1x1 Conv -> BN -> ReLU (the gating signal)."""

    def __init__(self, in_size, out_size, kernel_size=1, is_batchnorm=True):
        super().__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size, 1, 0),
                nn.BatchNorm2d(out_size), nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size, 1, 0), nn.ReLU(inplace=True))
        for m in self.children():
            init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        return self.conv1(inputs)


class UnetDsv2D(nn.Module):
    """2D copy of UnetDsv3: 1x1 Conv to n_classes -> bilinear upsample to full res."""

    def __init__(self, in_size, out_size, scale_factor):
        super().__init__()
        self.dsv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False))

    def forward(self, x):
        return self.dsv(x)


# ---------------------------------------------------------------------------
# Skip-attention modules (all share the (x, g) -> (x_filtered, spatial_map) contract)
# ---------------------------------------------------------------------------
class AdditiveAttentionGate(nn.Module):
    """The paper's attention gate: GridAttentionBlock2D + a 1x1 conv/BN/ReLU combine.

    Mirrors the authors' MultiAttentionBlock (2D). Uses the coarse gating signal
    `g` to filter the skip feature `x`.
    """

    def __init__(self, in_size, gate_size, inter_size=None, sub_sample_factor=(2, 2)):
        super().__init__()
        self.gate = GridAttentionBlock2D(
            in_channels=in_size, gating_channels=gate_size,
            inter_channels=inter_size or in_size, mode="concatenation",
            sub_sample_factor=sub_sample_factor)
        self.combine = nn.Sequential(
            nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_size), nn.ReLU(inplace=True))
        for m in self.children():
            if m.__class__.__name__.find("GridAttentionBlock2D") != -1:
                continue
            init_weights(m, init_type="kaiming")

    def forward(self, x, g=None):
        gated, att = self.gate(x, g)
        return self.combine(gated), att


def _make_attention(kind, in_size, gate_size):
    """Factory for the swappable skip-attention module."""
    if kind == "gate":
        return AdditiveAttentionGate(in_size, gate_size)
    if kind == "cbam":
        from busi.attention_modules import CBAM2D  # Stage 7
        return CBAM2D(in_size)
    if kind == "scse":
        from busi.attention_modules import scSE2D  # Stage 8
        return scSE2D(in_size)
    raise ValueError(f"unknown attention kind: {kind!r}")


# ---------------------------------------------------------------------------
# The attention U-Net (backbone shared by all attention variants)
# ---------------------------------------------------------------------------
class AttentionUNet2D(nn.Module):
    """2D U-Net with deep supervision and a swappable skip-attention module.

    Faithful 2D port of the authors' `unet_CT_single_att_dsv_3D`.

    Design justifications:
    - **Attention on skips 2/3/4, not 1.** The gating signal comes from a coarser
      decoder level, so it is meaningful for the deeper skips; the full-resolution
      level-1 skip is left ungated (as in the authors' single-attention net) — a
      gate there is the most expensive and adds little, since coarse gating can't
      localise at full resolution.
    - **`is_deconv=True` (ConvTranspose upsampling).** The authors' `unetUp`
      concatenates the skip with the upsampled feature and feeds a conv sized for
      *halved* channels; only ConvTranspose (which halves channels on upsample)
      makes those channel counts line up. Plain bilinear upsampling would mismatch.
    - **Deep supervision.** Auxiliary 1x1-conv heads at each decoder level are
      upsampled to full resolution and fused. This injects gradient at multiple
      scales (helping small lesions) and is part of the paper's contribution, so
      we keep it for fidelity; it is toggleable via `deep_supervision`.
    """

    def __init__(self, in_channels=1, n_classes=2, feature_scale=4,
                 attention="gate", is_batchnorm=True, is_deconv=True,
                 deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        f = [int(x / feature_scale) for x in (64, 128, 256, 512, 1024)]

        # Encoder
        self.conv1 = unetConv2(in_channels, f[0], is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = unetConv2(f[0], f[1], is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = unetConv2(f[1], f[2], is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(2)
        self.conv4 = unetConv2(f[2], f[3], is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(2)

        # Bottleneck + gating signal
        self.center = unetConv2(f[3], f[4], is_batchnorm)
        self.gating = UnetGridGatingSignal2D(f[4], f[4], kernel_size=1, is_batchnorm=is_batchnorm)

        # Swappable attention on skips 2/3/4
        self.attentionblock2 = _make_attention(attention, f[1], f[2])
        self.attentionblock3 = _make_attention(attention, f[2], f[3])
        self.attentionblock4 = _make_attention(attention, f[3], f[4])

        # Decoder
        self.up_concat4 = unetUp(f[4], f[3], is_deconv)
        self.up_concat3 = unetUp(f[3], f[2], is_deconv)
        self.up_concat2 = unetUp(f[2], f[1], is_deconv)
        self.up_concat1 = unetUp(f[1], f[0], is_deconv)

        # Deep supervision heads
        if deep_supervision:
            self.dsv4 = UnetDsv2D(f[3], n_classes, scale_factor=8)
            self.dsv3 = UnetDsv2D(f[2], n_classes, scale_factor=4)
            self.dsv2 = UnetDsv2D(f[1], n_classes, scale_factor=2)
            self.dsv1 = nn.Conv2d(f[0], n_classes, kernel_size=1)
            self.final = nn.Conv2d(n_classes * 4, n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(f[0], n_classes, kernel_size=1)

        # Init new conv/BN layers (attention blocks self-initialise).
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                init_weights(m, init_type="kaiming")

    def forward(self, inputs, return_attention=False):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(self.maxpool1(conv1))
        conv3 = self.conv3(self.maxpool2(conv2))
        conv4 = self.conv4(self.maxpool3(conv3))
        center = self.center(self.maxpool4(conv4))
        gating = self.gating(center)

        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        if self.deep_supervision:
            logits = self.final(torch.cat(
                [self.dsv1(up1), self.dsv2(up2), self.dsv3(up3), self.dsv4(up4)], dim=1))
        else:
            logits = self.final(up1)

        if return_attention:
            return logits, {"att2": att2, "att3": att3, "att4": att4}
        return logits

    @staticmethod
    def apply_argmax_softmax(logits):
        return F.softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_model(name, in_channels=1, n_classes=2, feature_scale=4, deep_supervision=True):
    """Build a model by name (driven by Config.model_name).

    All attention variants share one backbone; only the skip module differs.
    """
    if name == "unet":
        # Authors' plain U-Net baseline (is_deconv=True for consistent channels).
        return unet_2D(feature_scale=feature_scale, n_classes=n_classes,
                       is_deconv=True, in_channels=in_channels, is_batchnorm=True)
    attention_kind = {
        "attention_unet": "gate",
        "cbam_unet": "cbam",
        "scse_unet": "scse",
    }.get(name)
    if attention_kind is None:
        raise ValueError(f"unknown model name: {name!r}")
    return AttentionUNet2D(in_channels=in_channels, n_classes=n_classes,
                           feature_scale=feature_scale, attention=attention_kind,
                           deep_supervision=deep_supervision)
