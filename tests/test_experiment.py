"""Stage 6 orchestration tests. Trains tiny models on a synthetic BUSI tree so
the whole pipeline (split -> loaders -> fit -> evaluate -> aggregate -> report)
is verified end-to-end without the real dataset."""

import os

import cv2
import numpy as np
import pytest

from busi import experiment as E
from busi.config import Config

COUNTS = {"benign": 10, "malignant": 8, "normal": 6}


@pytest.fixture
def busi_cfg(tmp_path):
    """Build a synthetic BUSI tree and a tiny, fast Config pointing at it."""
    root = tmp_path / "BUSI"
    rng = np.random.default_rng(0)
    for cls, n in COUNTS.items():
        d = root / cls; d.mkdir(parents=True)
        for i in range(1, n + 1):
            m = np.zeros((64, 64), np.uint8)
            if cls != "normal":
                m[16:40, 16:40] = 255
            img = (m.astype(np.float32) * 0.9 + rng.integers(0, 30, (64, 64))).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(d / f"{cls} ({i}).png"), img)
            cv2.imwrite(str(d / f"{cls} ({i})_mask.png"), m)
    return Config(
        data_root=str(root), split_path=str(tmp_path / "split.json"),
        checkpoints_dir=str(tmp_path / "ckpt"), results_dir=str(tmp_path / "res"),
        img_size=64, batch_size=4, num_workers=0, epochs=1, feature_scale=8, device="cpu",
    )


# --- pure functions --------------------------------------------------------
def test_aggregate():
    dicts = [{"lesion_dice": 0.4, "specificity": float("nan")},
             {"lesion_dice": 0.6, "specificity": 1.0}]
    agg = E.aggregate(dicts)
    assert abs(agg["lesion_dice"]["mean"] - 0.5) < 1e-9
    assert abs(agg["lesion_dice"]["std"] - 0.1) < 1e-9
    assert agg["specificity"]["mean"] == 1.0        # NaN dropped
    assert len(agg["lesion_dice"]["values"]) == 2


def test_make_results_table():
    fake = [{"model": "unet", "n_params": 2_000_000,
             "aggregate": {k: {"mean": 0.5, "std": 0.01, "values": [0.5]}
                           for k in ("lesion_dice", "lesion_iou", "dice_benign",
                                     "dice_malignant", "specificity")}}]
    table = E.make_results_table(fake)
    assert "| Model |" in table and "unet" in table and "2.00M" in table


def test_save_load_results_roundtrip(tmp_path):
    r = {"model": "unet", "n_params": 1, "aggregate": {}, "runs": []}
    E.save_results(r, "unet", str(tmp_path))
    assert E.load_results("unet", str(tmp_path))["model"] == "unet"


# --- data plumbing ---------------------------------------------------------
def test_build_loaders(busi_cfg):
    train, val, test = E.build_loaders(busi_cfg)
    imgs, masks, labels = next(iter(train))
    assert imgs.shape[1:] == (1, 64, 64)
    assert masks.shape[1:] == (64, 64)
    assert os.path.exists(busi_cfg.split_path)      # split persisted


def test_split_is_reused(busi_cfg):
    s1 = E.get_or_make_split(busi_cfg)
    s2 = E.get_or_make_split(busi_cfg)               # second call loads from disk
    assert [x["image"] for x in s1["train"]] == [x["image"] for x in s2["train"]]


# --- end-to-end runs -------------------------------------------------------
def test_run_experiment_end_to_end(busi_cfg):
    cfg = E._cfg_for(busi_cfg, "unet", seed=42)
    out = E.run_experiment(cfg)
    assert set(out) == {"config", "best_val_metric", "test", "history"}
    assert "lesion_dice" in out["test"]
    assert os.path.exists(os.path.join(cfg.checkpoints_dir, f"{cfg.experiment_name}.pt"))


def test_run_seeds(busi_cfg):
    result = E.run_seeds("unet", cfg=busi_cfg, seeds=[42, 1])
    assert result["model"] == "unet" and result["n_params"] > 0
    assert len(result["aggregate"]["lesion_dice"]["values"]) == 2
    assert os.path.exists(os.path.join(busi_cfg.results_dir, "unet.json"))


def test_run_experiment_skips_if_done(busi_cfg):
    cfg = E._cfg_for(busi_cfg, "unet", seed=42)
    r1 = E.run_experiment(cfg)
    r2 = E.run_experiment(cfg)                        # second call must SKIP (result exists)
    assert r2["test"]["lesion_dice"] == r1["test"]["lesion_dice"]


def test_fit_resumes_from_checkpoint(busi_cfg):
    from busi import train as T, model as M
    d2 = {**E._cfg_for(busi_cfg, "unet", 42).to_dict(), "epochs": 2}
    cfg2 = Config(**d2)
    tr, va, te = E.build_loaders(cfg2)
    out1 = T.fit(M.get_model("unet", feature_scale=cfg2.feature_scale), tr, va, cfg2)
    assert len(out1["history"]) == 2                  # ran 2 epochs, wrote <exp>_last.pt

    cfg4 = Config(**{**d2, "epochs": 4})              # same exp name + ckpt dir -> resumes
    out2 = T.fit(M.get_model("unet", feature_scale=cfg4.feature_scale), tr, va, cfg4)
    assert len(out2["history"]) == 4                  # 2 resumed + 2 new (not 4 fresh)


def test_collect_results_partial(busi_cfg):
    E.run_seeds("unet", cfg=busi_cfg, seeds=[42])
    got = E.collect_results(["unet", "attention_unet"], busi_cfg.results_dir)
    assert len(got) == 1 and got[0]["model"] == "unet"   # attention_unet not run -> skipped


def test_save_prediction_figures(busi_cfg):
    cfg = E._cfg_for(busi_cfg, "attention_unet", seed=42)
    out = E.run_experiment(cfg)                       # produces a checkpoint
    ckpt = os.path.join(cfg.checkpoints_dir, f"{cfg.experiment_name}.pt")
    figs = E.save_prediction_figures(cfg, ckpt, str(cfg.checkpoints_dir) + "_figs", n=3)
    assert len(figs) == 3 and all(os.path.exists(p) for p in figs)
