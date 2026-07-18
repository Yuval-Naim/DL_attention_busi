"""Stage 4 training/eval tests. The key one is test_overfit_one_batch (CP4):
if the model can't memorise a handful of images, something is wrong (mask
alignment, channels, loss) — so it must reach Dice > 0.95 before real training."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from busi import train as T
from busi import model as M
from busi.losses import SoftDiceLoss2D
from busi.config import Config


def _tiny_dataset(n=4, s=32, with_normal=False):
    """Synthetic (image, mask, label): a bright square = the lesion, varied
    location per image so the net must learn image->mask (not a constant)."""
    imgs, masks, labels = [], [], []
    for i in range(n):
        m = torch.zeros(s, s, dtype=torch.long)
        if i % 2 == 0:
            m[4:16, 4:16] = 1
        else:
            m[16:28, 16:28] = 1
        img = m.float() * 0.9 + 0.05 * torch.randn(s, s)   # image correlated with mask
        imgs.append(img.unsqueeze(0))
        masks.append(m)
        labels.append(i % 2)                                # benign / malignant
    if with_normal:
        imgs.append(torch.zeros(1, s, s))
        masks.append(torch.zeros(s, s, dtype=torch.long))
        labels.append(2)                                    # normal, empty mask
    return TensorDataset(torch.stack(imgs), torch.stack(masks), torch.tensor(labels))


def test_set_seed_reproducible():
    T.set_seed(0); a = M.get_model("attention_unet").state_dict()
    T.set_seed(0); b = M.get_model("attention_unet").state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_get_device():
    assert T.get_device("cpu") == "cpu"
    assert T.get_device("auto") in ("cuda", "mps", "cpu")


def test_train_step_updates():
    ds = _tiny_dataset()
    loader = DataLoader(ds, batch_size=4)
    net = M.get_model("attention_unet")
    before = [p.detach().clone() for p in net.parameters()]
    loss = T.train_one_epoch(net, loader, SoftDiceLoss2D(), torch.optim.Adam(net.parameters()), "cpu")
    after = list(net.parameters())
    assert torch.isfinite(torch.tensor(loss))
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_evaluate_keys():
    ds = _tiny_dataset(with_normal=True)
    loader = DataLoader(ds, batch_size=5)
    out = T.evaluate(M.get_model("attention_unet"), loader, "cpu")
    assert set(out) == {"lesion_dice", "lesion_iou", "dice_benign",
                        "dice_malignant", "specificity", "fp_rate"}
    for k in ("lesion_dice", "lesion_iou", "specificity"):
        v = out[k]
        assert (v != v) or (0.0 <= v <= 1.0)   # NaN or in [0,1]


def test_overfit_one_batch():
    """CP4: the model must overfit a tiny fixed set to Dice > 0.95.

    Uses lr=1e-3 (the project default). Note: lr=1e-2 causes the soft-Dice
    cold-start collapse (model gets stuck predicting all-background), so the
    default 1e-3 is the validated choice for real training too.
    """
    T.set_seed(0)
    ds = _tiny_dataset(n=4, s=32)
    loader = DataLoader(ds, batch_size=4)
    net = M.get_model("attention_unet")
    loss_fn = SoftDiceLoss2D()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(300):
        T.train_one_epoch(net, loader, loss_fn, opt, "cpu")
    dice = T.evaluate(net, loader, "cpu")["lesion_dice"]
    assert dice > 0.95, f"overfit failed: Dice={dice:.3f}"


def test_smoke_one_epoch(tmp_path):
    cfg = Config(model_name="unet", epochs=1, device="cpu",
                 checkpoints_dir=str(tmp_path))
    train_loader = DataLoader(_tiny_dataset(), batch_size=2)
    val_loader = DataLoader(_tiny_dataset(), batch_size=2)
    out = T.fit(M.get_model("unet"), train_loader, val_loader, cfg)
    assert out["best_path"].endswith(".pt")
    import os
    assert os.path.exists(out["best_path"])
    assert len(out["history"]) == 1
