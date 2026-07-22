# Attention U-Net → BUSI (Breast Ultrasound Lesion Segmentation)

Course project (Medical Images Processing with Deep Learning, 336033). We extend
**Attention U-Net** (Oktay et al., 2018) to a new domain — 2D breast-ultrasound
lesion segmentation on the **BUSI** dataset — and compare attention mechanisms
(additive gate vs CBAM vs scSE).

This repo is derived from the authors' MIT-licensed code
(https://github.com/ozan-oktay/Attention-Gated-Networks, kept as the `upstream`
git remote). Our code lives under `busi/`.

> **Status:** code complete & tested (crash-safe, resumable); running the PoC
> experiments on Colab, then the report. See `PLAN.md` for the full plan/progress
> and `HANDOFF.md` for setup + how to run.

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
data/        # BUSI dataset (git-ignored — downloaded at runtime, see below)
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

## Hebrew guides (open in a browser)

Plain-language overviews for the team, in `docs/` — just open the `.html` files:
- `docs/overview_he.html` — plan, roadmap, status & first results
- `docs/walkthrough_he.html` — guided tour of all the code, tests, results
- `docs/changes_he.html` — exactly what we changed vs the original repo

## Data (BUSI)

The dataset is **not** committed or submitted — it is **downloaded at runtime**.
The notebook (`project.ipynb`, section 3) downloads BUSI from Kaggle
(`aryashah2k/breast-ultrasound-images-dataset`) into `data/BUSI/` (780 images),
reading a Kaggle token from a Colab secret / env vars / `~/.kaggle/kaggle.json`
(never hardcoded).

Dataset: Al-Dhabyani, Gomaa, Khaled & Fahmy, *"Dataset of breast ultrasound
images"*, Data in Brief 28 (2020), **CC BY 4.0**. Keep this attribution and cite
it in the report.

## Running experiments

The Colab notebook `project.ipynb` orchestrates everything and is **crash-safe &
resumable**: bootstrap → mount Drive → data check → **KNOBS** (edit
`EPOCHS`/`PATIENCE`/`SEEDS`/`MODELS`) → **one cell per model** → results + figures.
Outputs are saved to Drive every epoch; if Colab disconnects, re-run the cells —
finished seeds skip and the in-progress one resumes. PoC defaults: 50 epochs,
patience 8, 3 seeds (≈ 1–1.5 h on a T4). See `HANDOFF.md` §4 and `PLAN.md`.
