# Attention U-Net → BUSI (Breast Ultrasound Lesion Segmentation)

Course project (Medical Images Processing with Deep Learning, 336033). We extend
**Attention U-Net** (Oktay et al., 2018) to a new domain — 2D breast-ultrasound
lesion segmentation on the **BUSI** dataset — and compare attention mechanisms
(additive gate vs CBAM vs scSE).

This code is derived from the authors' MIT-licensed implementation
(https://github.com/ozan-oktay/Attention-Gated-Networks); their `LICENSE` is
retained.

## Ours vs. the authors' code

**Ours:** everything in `busi/`, `tests/`, and `project.ipynb`.

**From the authors** we import exactly four modules — our link to the paper:

- `models/layers/grid_attention_layer.py` → `GridAttentionBlock2D`, the paper's attention gate
- `models/networks/utils.py` → `unetConv2`, `unetUp` (conv / upsample blocks)
- `models/networks_other.py` → `init_weights`
- `models/networks/unet_2D.py` → plain 2D U-Net, used verbatim as our ablation baseline

Their remaining files (3D/CT networks, SonoNet classification, `dataio/`,
`utils/`, `configs/`, the root-level `train_*.py` / `visualise_*.py`) are not
used by our extension; they are included unchanged so the lineage is verifiable.

> **How to run it:** see **section 1 of `project.ipynb`** — it runs on Colab
> straight from this submitted folder (or its `.zip`), with nothing cloned or
> downloaded from GitHub. The training loop is crash-safe and resumable.

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
patience 8, 3 seeds (≈ 1–1.5 h per model on a T4).
