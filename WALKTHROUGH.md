# WALKTHROUGH — understand the whole project

> **Who this is for:** our team. Read this to understand *everything* we built — the
> idea, every source file, the tests, the results so far, and what's left — well
> enough to check our work critically and continue it. No prior deep-learning
> experience assumed; terms are explained as they appear.
>
> **The three docs:** `DESIGN.md` = *what & why* (decisions). `PLAN.md` = *the
> step-by-step build plan with a progress checklist*. **This file** = *a friendly
> tour of the actual code and results.*

---

## 1. The 30-second picture

We take a published idea — **Attention U-Net** (a neural network that segments
medical images and "learns where to look") — and apply it to a **new domain**:
finding **breast-tumor lesions in ultrasound images** (the **BUSI** dataset).
On top of that we (a) compare the paper's attention against two other attention
mechanisms (CBAM, scSE), (b) check whether the model correctly stays silent on
*healthy* scans, and (c) visualize *where* each model looks.

**Segmentation** = for every pixel, decide "is this lesion, or background?" The
output is a **mask** (a black-and-white image: white = lesion, black = background).
We compare the mask the model predicts against the mask a doctor drew.

---

## 2. Words you'll see (mini-glossary)

| Term | Plain meaning |
|---|---|
| **Segmentation** | Labeling each pixel (here: lesion vs background). |
| **Mask** | A 0/1 image marking the lesion region. |
| **U-Net** | A popular network shape for image segmentation (encoder that shrinks the image, decoder that rebuilds it, with "skip connections" linking them). |
| **Skip connection** | A shortcut passing fine detail from the encoder straight to the decoder. |
| **Attention gate** | A small add-on that filters a skip connection to emphasize the important region and suppress background — the paper's core idea. |
| **CBAM / scSE** | Two *other* popular attention add-ons we compare against the paper's gate. |
| **Dice / IoU** | Scores (0–1) for how much the predicted mask overlaps the true mask. Higher = better. |
| **Loss** | A number measuring how wrong the model is *right now*; training tries to shrink it. |
| **Epoch** | One full pass over all training images. |
| **Seed** | A fixed number that makes "random" choices repeatable, so a run can be reproduced. |
| **Specificity** | On *healthy* images (no lesion), how often the model correctly predicts "nothing". |

---

## 3. How the code is organized

The project is a **fork** (a copy) of the paper authors' code (`models/`, `dataio/`,
… — kept for lineage and because we *reuse* their attention gate). **All of our new
code lives in `busi/`**, and our tests in `tests/`.

```
busi/                     OUR CODE
  config.py               the control panel — all settings in one place
  data.py                 load BUSI images/masks, split, augment, feed the model
  model.py                the U-Net variants (baseline + attention) + factory
  attention_modules.py    CBAM and scSE attention blocks (the comparison)
  losses.py               Dice, Dice+CE (default), Focal-Tversky
  metrics.py              Dice / IoU / specificity / attention-focus scoring
  train.py                the training loop, evaluation, seeding, device pick
  experiment.py           orchestration: run all models x seeds, tables, figures
  viz.py                  draw overlays and comparison figures
tests/                    one test file per module (79 tests total)
project.ipynb             the Colab notebook that runs everything
splits/split.json         the fixed train/val/test split (committed = reproducible)
results/                  metrics saved here as JSON
data/BUSI/                the dataset (NOT in git — download separately)
checkpoints/              trained model weights (NOT in git)
models/ dataio/ ...       the authors' original code (we import a few pieces)
DESIGN.md PLAN.md         decisions / build plan
```

**Everything is driven by `busi/config.py`.** To run a different model or change a
setting, you edit one file (or override in the notebook). You almost never touch
the other files to *run* experiments.

---

## 4. The data flow (end to end)

```
BUSI PNGs  ->  list_busi_samples  ->  make_split (train/val/test)
           ->  BUSIDataset (reads image+mask, resizes, augments)
           ->  DataLoader (batches)  ->  model  ->  loss  ->  optimizer updates
           ->  evaluate (Dice/IoU/specificity)  ->  results JSON + figures
```

Read that top to bottom — each arrow is a function you can open and read.

---

## 5. Module-by-module (the important part)

### `config.py` — the control panel
One `Config` dataclass holds **every** setting: which model, which loss, image
size, learning rate, epochs, seeds, file paths, augmentation strengths, etc. Each
field has a comment. It validates your choices (e.g. rejects an unknown model
name) and can save/load itself as JSON so any run is reproducible. **Start here to
see all the knobs.**

Most-changed knob: `model_name` ∈ {`unet`, `attention_unet`, `cbam_unet`,
`scse_unet`}. Loss: `loss_name` ∈ {`dice_ce` (default), `dice`, `focal_tversky`}.

### `data.py` — getting images into the model
- `list_busi_samples(root)` — finds every image and pairs it with its mask
  file(s). Paths are stored **relative** so the split works on any machine.
- `merge_masks(...)` — some lesions have several mask files; we combine them with
  logical OR into one 0/1 mask. Healthy images have an all-black (or missing) mask
  → an all-zero mask. **Masks are resized with "nearest-neighbor" so they stay
  crisp 0/1** (never blurry gray) — a subtle but important correctness point.
- `make_split(...)` — splits the 780 images into train/val/test (70/15/15),
  **stratified** (each split keeps the same benign/malignant/normal proportions),
  with a fixed seed → identical every time. Saved to `splits/split.json`.
- `build_transforms(...)` — resizing + (for training) flips, small rotations,
  brightness jitter — applied **identically to the image and its mask** so they
  stay aligned. Uses the `albumentations` library.
- `BUSIDataset` — the object the model reads from. Each item is
  `(image[1×H×W float 0..1], mask[H×W of 0/1], label)` where label says
  benign/malignant/normal (needed to score healthy images separately).

### `model.py` — the networks
- `AttentionUNet2D` — our 2D version of the paper's network: a U-Net with a
  **swappable attention block** on the skip connections plus "deep supervision"
  (extra prediction heads at several scales, fused at the end). It can return the
  attention maps so we can *see* where it looks.
- `get_model(name)` — the factory. `unet` = the authors' plain U-Net (our
  baseline, no attention). `attention_unet` = the paper's additive gate.
  `cbam_unet` / `scse_unet` = the same backbone with CBAM / scSE instead.
- **All variants share one backbone; only the attention block changes** — that's
  what makes the comparison fair. See the docstring's "Design justifications" for
  three non-obvious choices (why attention on some skips only, why ConvTranspose
  upsampling, why deep supervision).

### `attention_modules.py` — the two alternative attentions
- `CBAM2D` — channel attention (which feature channels matter) + spatial
  attention (where to look).
- `scSE2D` — a lighter medical-imaging attention (parallel channel + spatial
  squeeze-excitation).
- Both follow the same **contract** as the paper's gate: `forward(x, g=None) ->
  (filtered_x, spatial_map)`. The `spatial_map` (0–1) is what we visualize.

### `losses.py` — how "wrongness" is measured
- `SoftDiceLoss2D` — the paper's Dice loss, made device-agnostic (their version
  was hardcoded to NVIDIA GPUs).
- `DiceCELoss` — **our default**: Dice **+** Cross-Entropy. *We found pure Dice
  fails to train here* (see §7); adding Cross-Entropy fixes it. Dice handles the
  lesion-is-tiny imbalance; CE gives a strong signal early so the model doesn't
  collapse to "predict nothing".
- `focal_tversky_loss` — a stretch alternative that penalizes missed lesion pixels
  harder (good for small lesions); part of the loss comparison.

### `metrics.py` — how we score results
- `dice_score`, `iou_score` — overlap scores, with sensible rules for empty masks
  (both empty → perfect 1.0; predicting a lesion on a healthy image → 0.0).
- `specificity_on_normals` — on healthy images, fraction where the model correctly
  predicts (near-)nothing. This is *the* number for "does it avoid false alarms".
- `attention_focus_ratio` — of all the attention the model paid, how much landed
  inside the true lesion (0–1). This quantifies "is it looking in the right place".

### `train.py` — the training engine
- `set_seed`, `get_device` (auto-picks GPU→Apple-GPU→CPU).
- `train_one_epoch` — one pass: predict, measure loss, update weights.
- `evaluate` — runs the model on a set and returns lesion Dice/IoU **and**
  specificity, **split by image type** so healthy images never inflate the lesion
  score.
- `fit` — the full loop: trains, watches validation Dice, lowers the learning rate
  when it plateaus, **stops early** if it stops improving, saves the best model,
  and **saves a resume checkpoint every epoch** (so a crash mid-training can be
  continued — see "Crash-safety" below). Prints progress every epoch.

### `experiment.py` — running the whole study
- `build_loaders` — makes the train/val/test DataLoaders from the saved split.
- `run_experiment` — one model, one seed: train → pick best → score on test.
  **Crash-safe:** if this seed's result JSON already exists it is *skipped*;
  otherwise training *resumes* from any partial checkpoint left by a crash.
- `run_seeds` — repeats over the seeds and reports **mean ± standard deviation**
  (so a result isn't just luck). Saves `results/<model>.json`. A failing seed is
  logged and skipped rather than killing the batch.
- `collect_results` — loads whatever per-model results exist (so the table works
  even from a partially-finished run).
- `make_results_table` — the markdown comparison table for the report.
- `save_prediction_figures` — the image | ground-truth | prediction | attention
  figures for the report.

**Crash-safety (why a Colab disconnect can't hurt you):** `fit` saves the best
weights *and* a full resume checkpoint (model+optimizer+epoch+history) to
`checkpoints_dir` every epoch; point that (and `results_dir`) at Google Drive.
On a disconnect, just re-run — finished seeds skip, the in-progress one resumes
from its last epoch. Progress is also printed live per epoch.

### `viz.py` — pictures
- `overlay_mask`, `overlay_attention`, `plot_prediction` — draw masks and attention
  heatmaps over the scan and assemble the comparison figure.

---

## 6. The tests (why you can trust the code)

We wrote a test for essentially every function — **79 tests, all passing**, and
they run **without needing the dataset** (they build tiny fake images). Run them:

```bash
cd attention-busi
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m pytest -q
```

| Test file | Checks that… |
|---|---|
| `test_setup.py` | the package + config import and validate correctly |
| `test_data.py` | image/mask pairing, mask-merge, masks stay 0/1 after resize, split has no leakage & is reproducible, dataset returns right shapes |
| `test_model.py` | every network outputs the right shape, attention maps are 0–1, gradients flow, init is reproducible |
| `test_attention_modules.py` | CBAM & scSE obey the contract, build into the U-Net, produce valid maps, gradients flow |
| `test_losses.py` | Dice≈0 when perfect / ≈1 when wrong, gradients finite, Dice+CE & Focal-Tversky behave |
| `test_metrics.py` | known-value checks (Dice=0.667, IoU=0.5), empty-mask rules, specificity extremes |
| `test_train.py` | **overfit-one-batch reaches Dice>0.95** (the key sanity check), seeding reproducible, one-epoch training runs |
| `test_experiment.py` | the whole run→aggregate→table→figures pipeline works end-to-end on fake data |

**The most important test is `test_overfit_one_batch`:** if the model can't memorize
a handful of images, something is fundamentally broken. It passing is strong
evidence the plumbing is correct.

---

## 7. Results so far (real data)

We downloaded BUSI (verified exactly **437 benign / 210 malignant / 133 normal =
780**) and ran short training on the laptop (Apple MPS, ~14 s/epoch).

**Confirmatory run — `attention_unet`, Dice+CE, 40 epochs, seed 42:**

| Metric | Value |
|---|---|
| Lesion Dice (test) | **0.781** |
| Lesion IoU (test) | 0.698 |
| Specificity (healthy scans) | **0.950** |
| Dice benign / malignant | 0.811 / 0.718 |

These are realistic BUSI numbers (published attention-U-Net results are ~0.7–0.8
Dice). Specificity 0.95 = it correctly stays silent on ~95% of healthy scans.

### Two findings worth putting in the report ("challenges")

1. **Learning rate 1e-2 collapses; 1e-3 works.** With Adam at 1e-2 the model gets
   stuck predicting all-background (the soft-Dice cold-start problem). We use
   **1e-3** (the config default), validated by the overfit test.
2. **Pure Dice loss fails to train on BUSI; Dice+CE fixes it.** With pure Dice,
   `attention_unet` was stuck at val Dice 0 for 16 epochs. Adding Cross-Entropy
   (→ `DiceCELoss`, now the default) reached ~0.62 val Dice in 12 epochs. We keep
   pure Dice selectable so the *collapse itself* is a reportable result.

> These are **our real results** from short runs; the **final report numbers** come
> from the full runs on Colab (see §8/§9).

---

## 8. How to run it yourself

**On Colab (the real runs — GPU):** open `project.ipynb` and run top-to-bottom:
bootstrap (clone public repo + install) → mount Drive + set paths → data check →
**KNOBS** cell (edit `EPOCHS`/`PATIENCE`/`SEEDS`/`MODELS`/`QUICK`; PoC defaults are
50 / 8 / 3-seeds) → **one cell per model** → results table + figures. Outputs go to
Drive. **If it disconnects, re-run — finished seeds skip, the in-progress one
resumes.** PoC run ≈ ~1–1.5 h on a T4.

**Locally (fast smoke, ~2 min):** set `QUICK=True` in the notebook, or:
```python
from busi.config import Config
from busi import experiment as E
E.run_seeds("attention_unet", cfg=Config(epochs=3), seeds=[42])
```

---

## 9. What's left (next stages)

- **Run the full experiments on Colab** (4 models × 3 seeds; optionally the loss
  study: `dice` vs `dice_ce` vs `focal_tversky`). → fills the results table and
  attention figures. *(PLAN.md Stages 6–8 "runs".)*
- **Write the 6-page report** (Stage 9): intro, the extension, methodology,
  results table + figures, the two findings above, limitations, future work.
- **Package for submission**: pinned `requirements.txt`, README data instructions,
  a clean top-to-bottom Colab re-run, then `id1_id2.zip`.

---

## 10. How to critique this work (be tough on it)

Good questions to challenge us with:

- **Is the comparison fair?** All variants share the backbone, split, seeds, and
  loss — check that in `model.py`/`experiment.py`. Only the attention block should
  differ.
- **Is the split honest?** No image should appear in two splits, and healthy-image
  scores must be reported *separately* (specificity), never mixed into lesion Dice.
  See `test_data.py::test_split_no_leakage` and `train.evaluate`.
- **Are the findings real or artifacts?** The lr and pure-Dice collapses — do they
  reproduce? (They should; the scripts are simple to rerun.)
- **Known limitations to state honestly:** small dataset (780 images, proof-of-
  concept scale); BUSI has some documented duplicate/annotation issues; we train
  from scratch (no pretraining); attention maps are suggestive, not proof of
  clinical reasoning; results depend on the single fixed split (we mitigate with
  3 seeds, but not with cross-validation).
- **Where could bugs hide?** Mask interpolation (must be nearest), the empty-mask
  metric conventions, and channel bookkeeping in the decoder. Each has a test —
  read them and try to break them.

If something here doesn't match the code, trust the code and tell us — that's
exactly the kind of review we want.
