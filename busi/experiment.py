"""Experiment orchestration. See DESIGN.md sec. 8 and PLAN.md Stage 6.

This is the glue that turns the building blocks (data / model / train / metrics /
viz) into reproducible, multi-seed experiments and report artifacts. The notebook
only calls these functions, so all logic is tested here.

Key reproducibility choice: the train/val/test **split is created once with a
fixed seed and persisted** (`SPLIT_SEED`, independent of the per-run `cfg.seed`),
so every model variant and every seed is evaluated on the *exact same* split —
a fair comparison. Only model initialisation and data shuffling vary with the
run seed.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from busi.config import Config
from busi.data import (BUSIDataset, list_busi_samples, load_split, make_split,
                       save_split, transforms_from_config)
from busi.model import get_model
from busi import train as T

SPLIT_SEED = 42  # fixed data-split seed (independent of the per-run cfg.seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_or_make_split(cfg: Config) -> dict:
    """Load the persisted split, or create it (once) from cfg.data_root and save it."""
    if os.path.exists(cfg.split_path):
        return load_split(cfg.split_path)
    samples = list_busi_samples(cfg.data_root, classes=tuple(cfg.classes))
    if not samples:
        raise FileNotFoundError(
            f"No BUSI samples under {cfg.data_root!r}. Download the dataset first "
            f"(see README) or set Config.data_root.")
    split = make_split(samples, ratios=tuple(cfg.split_ratios), seed=SPLIT_SEED)
    save_split(split, cfg.split_path)
    return split


def build_loaders(cfg: Config):
    """Return (train, val, test) DataLoaders for the persisted split."""
    split = get_or_make_split(cfg)
    tf_train = transforms_from_config(cfg, train=True)
    tf_eval = transforms_from_config(cfg, train=False)

    def _loader(key, tf, shuffle):
        ds = BUSIDataset(split[key], root=cfg.data_root, transform=tf, img_size=cfg.img_size)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers)

    return (_loader("train", tf_train, True),
            _loader("val", tf_eval, False),
            _loader("test", tf_eval, False))


# ---------------------------------------------------------------------------
# Single run / multi-seed
# ---------------------------------------------------------------------------
def run_experiment(cfg: Config) -> dict:
    """Train one model (one seed), select the best on val Dice, evaluate on test.

    Returns {config, best_val_metric, test (metrics dict), history}.
    """
    T.set_seed(cfg.seed)
    train_loader, val_loader, test_loader = build_loaders(cfg)
    model = get_model(cfg.model_name, in_channels=cfg.in_channels,
                      n_classes=cfg.n_classes, feature_scale=cfg.feature_scale,
                      deep_supervision=cfg.deep_supervision)
    fit_out = T.fit(model, train_loader, val_loader, cfg)

    device = T.get_device(cfg.device)
    model.load_state_dict(torch.load(fit_out["best_path"], map_location=device))
    test_metrics = T.evaluate(model, test_loader, device)
    return {
        "config": cfg.to_dict(),
        "best_val_metric": fit_out["best_metric"],
        "test": test_metrics,
        "history": fit_out["history"],
    }


def _cfg_for(base: Config, model_name: str, seed: int) -> Config:
    """Clone `base`, overriding model/seed and re-deriving experiment_name."""
    d = base.to_dict()
    d.update(model_name=model_name, seed=seed, experiment_name="")
    return Config(**d)


def aggregate(metric_dicts: list[dict]) -> dict:
    """Aggregate a list of metric dicts into {metric: {mean, std, values}} (NaN-safe)."""
    out = {}
    for k in metric_dicts[0]:
        vals = [m[k] for m in metric_dicts]
        arr = np.array([v for v in vals if v == v], dtype=float)  # drop NaN
        out[k] = {
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "std": float(arr.std()) if arr.size else float("nan"),
            "values": vals,
        }
    return out


def run_seeds(model_name: str, cfg: Config | None = None, seeds=None) -> dict:
    """Run `model_name` over multiple seeds; aggregate test metrics; save to results/."""
    cfg = cfg or Config()
    seeds = list(seeds if seeds is not None else cfg.seeds)
    runs = [run_experiment(_cfg_for(cfg, model_name, s)) for s in seeds]
    n_params = sum(p.numel() for p in get_model(
        model_name, in_channels=cfg.in_channels, n_classes=cfg.n_classes,
        feature_scale=cfg.feature_scale, deep_supervision=cfg.deep_supervision).parameters())
    result = {
        "model": model_name,
        "seeds": seeds,
        "n_params": n_params,
        "aggregate": aggregate([r["test"] for r in runs]),
        "runs": runs,
    }
    save_results(result, model_name, cfg.results_dir)
    return result


# ---------------------------------------------------------------------------
# Persistence & reporting
# ---------------------------------------------------------------------------
def save_results(result: dict, model_name: str, results_dir: str) -> str:
    """Write a run_seeds result dict to `results_dir/<model_name>.json`; return the path."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{model_name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def load_results(model_name: str, results_dir: str) -> dict:
    """Load a previously saved run_seeds result dict from `results_dir/<model_name>.json`."""
    with open(os.path.join(results_dir, f"{model_name}.json")) as f:
        return json.load(f)


def make_results_table(results_list: list[dict]) -> str:
    """Build a markdown results table (mean±std over seeds) from run_seeds outputs."""
    cols = [("lesion_dice", "Dice"), ("lesion_iou", "IoU"),
            ("dice_benign", "Dice(ben)"), ("dice_malignant", "Dice(mal)"),
            ("specificity", "Specificity")]
    header = "| Model | Params | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 2)
    lines = [header, sep]
    for r in results_list:
        agg = r["aggregate"]
        cells = [f"{agg[k]['mean']:.3f}±{agg[k]['std']:.3f}" for k, _ in cols]
        lines.append(f"| {r['model']} | {r['n_params']/1e6:.2f}M | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_prediction_figures(cfg: Config, ckpt_path: str, out_dir: str, n: int = 4) -> list:
    """Save up to `n` qualitative figures (image|GT|pred|attention) from the test set."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from busi import viz

    device = T.get_device(cfg.device)
    _, _, test_loader = build_loaders(cfg)
    model = get_model(cfg.model_name, in_channels=cfg.in_channels, n_classes=cfg.n_classes,
                      feature_scale=cfg.feature_scale, deep_supervision=cfg.deep_supervision)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    is_attention = cfg.model_name != "unet"
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    with torch.no_grad():
        for images, masks, _labels in test_loader:
            for b in range(images.size(0)):
                x = images[b:b + 1].to(device)
                if is_attention:
                    logits, atts = model(x, return_attention=True)
                    amaps = {k: v[0] for k, v in atts.items()}
                else:
                    logits, amaps = model(x), None
                pred = logits.argmax(1)[0].cpu()
                fig = viz.plot_prediction(images[b], masks[b], pred, att_maps=amaps)
                p = os.path.join(out_dir, f"{cfg.model_name}_{len(saved)}.png")
                fig.savefig(p, dpi=80, bbox_inches="tight")
                plt.close(fig)
                saved.append(p)
                if len(saved) >= n:
                    return saved
    return saved
