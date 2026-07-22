# HANDOFF — set up this project in your own repo

This is the code for our course project: **Attention U-Net applied to BUSI
breast-ultrasound lesion segmentation** (an extension of Oktay et al., 2018).
It's a working, tested codebase (79 automated tests pass). You'll put it in
**your own GitHub repo** and continue from here.

**Read these first** (in the repo): `WALKTHROUGH.md` (plain-language tour of all
the code, tests, and results), then `DESIGN.md` (decisions) and `PLAN.md`
(step-by-step plan + progress checklist).

---

## What you received

The full project **with its git history** (so our commit-by-commit work is
preserved). It does **not** include: the dataset (`data/`), trained weights
(`checkpoints/`), or the Python virtualenv (`.venv/`) — you'll recreate those.

---

## 1. Put it in your own GitHub repo

Unzip, then open a terminal in the `attention-busi/` folder:

```bash
# point this code at YOUR new (empty) GitHub repo instead of the original
git remote remove origin
git remote add origin https://github.com/<YOUR-USER>/<YOUR-REPO>.git
git push -u origin busi-extension      # pushes all our history to your repo
```

(The `upstream` remote points at the paper authors' original repo — keep it; it
documents that our work is a fair, MIT-licensed extension of their code.)

> Prefer the git-native way? We also included `attention-busi.bundle`:
> `git clone attention-busi.bundle attention-busi` gives you the same repo+history.

## 2. Get the dataset (BUSI)

Download from Kaggle: `aryashah2k/breast-ultrasound-images-dataset`, and arrange
it so the three class folders sit at:

```
attention-busi/data/BUSI/benign/
attention-busi/data/BUSI/malignant/
attention-busi/data/BUSI/normal/
```

(Expected: 437 / 210 / 133 = 780 images. `tests/test_data.py::test_list_counts_real`
verifies this automatically once the data is present.)

## 3. Set up Python and run the tests

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1        # (Apple Silicon; harmless elsewhere)
python -m pytest -q                          # expect: all tests pass
```

If Python 3.13 gives you install trouble, use a 3.11 or 3.12 virtualenv.

## 4. Run experiments

- **Fast local smoke (~2 min):** in `project.ipynb` set `QUICK=True`, or:
  ```python
  from busi.config import Config
  from busi import experiment as E
  E.run_seeds("attention_unet", cfg=Config(epochs=3), seeds=[42])
  ```
- **Full runs (GPU):** open `project.ipynb` on Google Colab, run the bootstrap
  cell (clone your repo, mount Drive with the data), then the experiment cells.
  `MODELS` already lists all four variants (`unet`, `attention_unet`, `cbam_unet`,
  `scse_unet`). Everything is controlled from `busi/config.py`.

## 5. What's left to do

See `PLAN.md` for the tracked plan. In short: run the full experiment matrix on
Colab, then write the 6-page report and package the submission (`id1_id2.zip`).
`WALKTHROUGH.md` section "how to critique this work" lists the open questions and
honest limitations to keep in mind.

## Lineage & license

This repo derives from https://github.com/ozan-oktay/Attention-Gated-Networks
(MIT). We reuse their attention gate and a few building blocks (imported in
`busi/model.py`) and wrote everything under `busi/` ourselves. Keep the `LICENSE`
file and cite both the paper and that repo in the report.
