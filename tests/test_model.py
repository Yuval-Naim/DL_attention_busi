"""Stage 2 model tests. Run on CPU with small (64x64) inputs for speed."""

import torch

from busi import model as M


def _in(n=2, c=1, s=64):
    return torch.randn(n, c, s, s)


def test_forward_shape():
    for name in ("unet", "attention_unet"):
        out = M.get_model(name)(_in())
        assert tuple(out.shape) == (2, 2, 64, 64), name


def test_softmax_valid():
    logits = M.get_model("attention_unet")(_in())
    probs = M.AttentionUNet2D.apply_argmax_softmax(logits)
    assert torch.allclose(probs.sum(1), torch.ones(2, 64, 64), atol=1e-5)


def test_attention_range():
    net = M.get_model("attention_unet")
    with torch.no_grad():
        _, maps = net(_in(), return_attention=True)
    assert set(maps) == {"att2", "att3", "att4"}
    for k, m in maps.items():
        assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0, k
    # spatial sizes match the skip levels (H/2, H/4, H/8 for input 64)
    assert tuple(maps["att2"].shape) == (2, 1, 32, 32)
    assert tuple(maps["att3"].shape) == (2, 1, 16, 16)
    assert tuple(maps["att4"].shape) == (2, 1, 8, 8)


def test_params_more_than_baseline():
    n_unet = sum(p.numel() for p in M.get_model("unet").parameters())
    n_att = sum(p.numel() for p in M.get_model("attention_unet").parameters())
    assert n_att > n_unet > 0


def test_gating_signal_shape():
    g = M.UnetGridGatingSignal2D(32, 32)(torch.randn(2, 32, 16, 16))
    assert tuple(g.shape) == (2, 32, 16, 16)


def test_dsv_upsample():
    out = M.UnetDsv2D(64, 2, scale_factor=4)(torch.randn(2, 64, 32, 32))
    assert tuple(out.shape) == (2, 2, 128, 128)


def test_deterministic_init():
    torch.manual_seed(0); a = M.get_model("attention_unet")
    torch.manual_seed(0); b = M.get_model("attention_unet")
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_backward():
    for name in ("unet", "attention_unet"):
        net = M.get_model(name)
        out = net(_in())
        out.sum().backward()
        missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f"{name}: params without grad: {missing[:3]}"


def test_variants_build():
    for name in ("unet", "attention_unet"):
        out = M.get_model(name)(_in())
        assert tuple(out.shape) == (2, 2, 64, 64)
