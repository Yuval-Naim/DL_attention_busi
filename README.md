# Attention U-Net → BUSI (Breast Ultrasound Lesion Segmentation)

Course project (Medical Images Processing with Deep Learning, 336033). We extend
**Attention U-Net** (Oktay et al., 2018) to a new domain — 2D breast-ultrasound
lesion segmentation on the **BUSI** dataset — and compare attention mechanisms
(additive gate vs CBAM vs scSE).

This repo is derived from the authors' MIT-licensed code
(https://github.com/ozan-oktay/Attention-Gated-Networks, kept as the `upstream`
git remote). Our code lives under `busi/`.

> **Status:** setup stage. See `../PLAN.md` for the full task plan and progress.

## Everything is config-driven

Switch model, loss, and hyperparameters in **`busi/config.py`** — no code edits.
The most important knob is `model_name`:

```python
from busi.config import Config
cfg = Config(model_name="attention_unet")  # or "unet", "cbam_unet", "scse_unet"
```

## Repo layout

```
busi/        # our code: config, data, model, losses, metrics, train, viz
tests/       # unit / validation tests (pytest)
splits/      # persisted train/val/test split (tracked)
results/     # metrics JSON (tracked)
data/        # BUSI dataset (git-ignored — download separately)
checkpoints/ # model weights (git-ignored)
models/ dataio/ ...   # authors' original code (reused: attention gate, conv blocks, Dice loss)
```

## Setup (local, Apple Silicon / MPS)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1
pytest -q            # run the test suite
```

If Python 3.13 causes wheel issues, use a 3.11/3.12 virtualenv.

## Data access (BUSI)

Download from Kaggle: `aryashah2k/breast-ultrasound-images-dataset`, and place
the `benign/ malignant/ normal/` folders under `data/BUSI/` (or set
`Config.data_root`). Dataset files are **not** committed.

## Running experiments

The Colab notebook `project.ipynb` orchestrates: clone → data → runs → figures.
See `../PLAN.md` (Stage 6) for the milestone experiment matrix.
