"""Central configuration — the single control panel for the whole project.

Everything you'd change between runs lives here. Two ways to use it:

1) Edit the defaults in `Config` below, then run.
2) Override per run without editing the file, e.g. in the notebook:
       from busi.config import Config
       cfg = Config(model_name="cbam_unet", seed=1, epochs=50)

The most-frequently-changed knob is `model_name` — switch between the plain
U-Net, the paper's Attention U-Net, CBAM, or scSE by changing this one string.

Nothing here imports torch, so `import busi.config` works before the deep-
learning stack is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json

# --- Allowed choices (validated in Config.__post_init__) --------------------
MODEL_CHOICES = ("unet", "attention_unet", "cbam_unet", "scse_unet")
LOSS_CHOICES = ("dice_ce", "dice", "focal_tversky")
DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


@dataclass
class Config:
    """All project settings. Grouped by how often you touch them."""

    # ========================================================================
    # WHAT TO RUN  (change these most often)
    # ========================================================================
    model_name: str = "attention_unet"   # one of MODEL_CHOICES
    loss_name: str = "dice_ce"            # one of LOSS_CHOICES (dice_ce: pure dice collapses on BUSI)
    seed: int = 42                        # single-run seed

    # ========================================================================
    # DATA
    # ========================================================================
    data_root: str = "data/BUSI"          # folder holding benign/ malignant/ normal/
    classes: tuple = ("benign", "malignant", "normal")  # include normal (Option B)
    img_size: int = 256                   # images + masks resized to img_size x img_size
    split_ratios: tuple = (0.7, 0.15, 0.15)   # train / val / test
    split_path: str = "splits/split.json"     # persisted split (tracked in git)

    # ========================================================================
    # MODEL
    # ========================================================================
    in_channels: int = 1                  # grayscale ultrasound
    n_classes: int = 2                    # background + lesion (softmax head)
    feature_scale: int = 4                # filters = [64,128,256,512,1024] // feature_scale
    deep_supervision: bool = True         # keep the paper's deep-supervision heads

    # ========================================================================
    # TRAINING
    # ========================================================================
    epochs: int = 100
    batch_size: int = 8                   # 8-16 on Colab T4; 4-8 on local MPS
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lr_patience: int = 8                  # ReduceLROnPlateau patience (on val Dice)
    early_stop_patience: int = 15         # stop if val Dice hasn't improved
    ckpt_every: int = 1                   # save a resume checkpoint every N epochs (to Drive) — crash safety
    num_workers: int = 0                  # 0 avoids macOS 'spawn'/notebook multiprocessing issues; dataset is small so no throughput cost

    # ========================================================================
    # AUGMENTATION  (train split only; applied identically to image + mask)
    # ========================================================================
    aug_hflip: bool = True
    aug_rotate_deg: float = 15.0
    aug_scale: tuple = (0.9, 1.1)
    aug_brightness: float = 0.10          # brightness/contrast jitter magnitude

    # ========================================================================
    # LOSS — Focal-Tversky params (only used when loss_name == "focal_tversky")
    # ========================================================================
    ft_alpha: float = 0.7                 # weight on false negatives
    ft_beta: float = 0.3                  # weight on false positives
    ft_gamma: float = 0.75                # focal exponent

    # ========================================================================
    # METRICS
    # ========================================================================
    specificity_area_thresh: float = 0.005  # normal pred counts as "clean" if lesion area <= this fraction

    # ========================================================================
    # EXPERIMENT / RUNTIME
    # ========================================================================
    seeds: tuple = (42, 1, 7)             # used by the multi-seed experiment runner
    device: str = "auto"                  # auto -> cuda, else mps, else cpu
    checkpoints_dir: str = "checkpoints"  # best-model weights (git-ignored)
    results_dir: str = "results"          # metrics JSON (tracked)
    experiment_name: str = ""             # auto-derived from model/loss/seed if left empty

    # ------------------------------------------------------------------------
    def __post_init__(self):
        if self.model_name not in MODEL_CHOICES:
            raise ValueError(f"model_name must be one of {MODEL_CHOICES}, got {self.model_name!r}")
        if self.loss_name not in LOSS_CHOICES:
            raise ValueError(f"loss_name must be one of {LOSS_CHOICES}, got {self.loss_name!r}")
        if self.device not in DEVICE_CHOICES:
            raise ValueError(f"device must be one of {DEVICE_CHOICES}, got {self.device!r}")
        if not self.experiment_name:
            self.experiment_name = f"{self.model_name}_{self.loss_name}_seed{self.seed}"

    # ------------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a plain dict (for logging / saving alongside results)."""
        return asdict(self)

    def save(self, path: str) -> None:
        """Persist the config as JSON so any run is fully reproducible."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        """Rebuild a Config from a saved JSON file."""
        with open(path) as f:
            data = json.load(f)
        # tuples are stored as lists in JSON; dataclass fields accept either
        return cls(**data)


# Convenience default instance for quick imports / notebooks.
DEFAULT = Config()
