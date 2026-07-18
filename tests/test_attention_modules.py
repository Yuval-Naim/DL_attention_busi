"""Stage 7 tests: CBAM2D module + cbam_unet variant (Stage 8 scSE tests later)."""

import torch

from busi.attention_modules import CBAM2D, scSE2D
from busi import model as M


def test_cbam_contract():
    x = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        out, sa = CBAM2D(32)(x)
    assert tuple(out.shape) == (2, 32, 16, 16)          # recalibrated, same shape
    assert tuple(sa.shape) == (2, 1, 16, 16)            # spatial map
    assert float(sa.min()) >= 0.0 and float(sa.max()) <= 1.0


def test_cbam_small_channels():
    # reduction guard: channels // reduction must not be 0
    out, sa = CBAM2D(8, reduction=16)(torch.randn(2, 8, 8, 8))
    assert tuple(out.shape) == (2, 8, 8, 8) and tuple(sa.shape) == (2, 1, 8, 8)


def test_cbam_ignores_gating_signal():
    # self-attention: passing a gating signal must not matter
    x = torch.randn(2, 16, 8, 8)
    m = CBAM2D(16)
    torch.manual_seed(0)
    a, _ = m(x, g=torch.randn(2, 99, 4, 4))
    b, _ = m(x, g=None)
    assert torch.allclose(a, b)


def test_cbam_unet_builds_and_forwards():
    net = M.get_model("cbam_unet")
    out = net(torch.randn(2, 1, 64, 64))
    assert tuple(out.shape) == (2, 2, 64, 64)


def test_cbam_unet_attention_maps():
    net = M.get_model("cbam_unet")
    with torch.no_grad():
        _, maps = net(torch.randn(2, 1, 64, 64), return_attention=True)
    assert set(maps) == {"att2", "att3", "att4"}
    for k, m in maps.items():
        assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0, k


def test_cbam_unet_backward():
    net = M.get_model("cbam_unet")
    net(torch.randn(2, 1, 64, 64)).sum().backward()
    missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"params without grad: {missing[:3]}"


# --- scSE (stretch) --------------------------------------------------------
def test_scse_contract():
    with torch.no_grad():
        out, sa = scSE2D(32)(torch.randn(2, 32, 16, 16))
    assert tuple(out.shape) == (2, 32, 16, 16)
    assert tuple(sa.shape) == (2, 1, 16, 16)
    assert float(sa.min()) >= 0.0 and float(sa.max()) <= 1.0


def test_scse_unet_builds_and_maps():
    net = M.get_model("scse_unet")
    with torch.no_grad():
        logits, maps = net(torch.randn(2, 1, 64, 64), return_attention=True)
    assert tuple(logits.shape) == (2, 2, 64, 64)
    assert set(maps) == {"att2", "att3", "att4"}
    for k, m in maps.items():
        assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0, k


def test_scse_unet_backward():
    net = M.get_model("scse_unet")
    net(torch.randn(2, 1, 64, 64)).sum().backward()
    missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"params without grad: {missing[:3]}"
