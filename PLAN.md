# PLAN — Implementation Plan (Attention U-Net → BUSI)

> **Purpose:** the start-to-finish execution plan (tasks, order, done-criteria, tests, checkpoints).
> **Companion doc:** [`DESIGN.md`](./DESIGN.md) is the source of truth for *decisions* (architecture, data, metrics). This file is the source of truth for *what to do, in what order, and how to know it's done.*
> **Status:** NOT STARTED. No code written yet.

---

## How to use this document (resume protocol)

1. Read the **Progress Tracker** (below) to see what is done. A task is done only when its **Definition of Done (DoD)** holds and its tests pass.
2. Read `DESIGN.md` §0 (locked decisions) before writing any code — do not re-litigate locked choices.
3. Work **stage by stage, top to bottom**. Do not start a stage until the previous stage's **Checkpoint** passes (checkpoints are hard gates).
4. After finishing a task: tick its box here, note the commit hash, and (if a decision changed) update `DESIGN.md` and the memory file.
5. **State lives in:** `DESIGN.md` (decisions) · `busi/config.py` (hyperparameters/paths) · `splits/split.json` (the persisted split) · `results/*.json` (metrics) · `checkpoints/*.pt` (weights) · this file (progress). All code lives in the repo clone at `attention-busi/`.
6. Milestones are independently shippable: **Core** alone is a complete submission. Only proceed to Desired/Stretch if Core's checkpoint passed and time remains.

---

## Progress Tracker

Legend: `[ ]` todo · `[~]` in progress · `[x]` done (DoD met + tests green).

**Stage 0 — Setup**
- [x] T0.1 Fork + clone + branch — *private duplicate* `YuvalDziesietnik/attention-unet-busi` (PRIVATE), cloned to `attention-busi/`, branch `busi-extension`, `upstream` = authors' repo
- [x] T0.2 Package skeleton (`busi/`, `tests/`, `config.py`, `requirements.txt`, `.gitignore`, README stub) — config imports clean; split moved to `splits/split.json`
- [x] T0.3 Local env (venv + MPS) & verify author-code imports — `.venv` (py3.13), torch 2.13.0 + MPS, all reused author modules import; `pytest.ini` scopes tests to `tests/`
- [x] T0.4 Colab bootstrap cell — `project.ipynb` (mount Drive, clone private repo via token, pip install, sanity cell)
- [x] **CP0** setup checkpoint — 4 tests green; `import busi` + author imports OK; device=mps. Committed `ef91e38` (local, not pushed)

**Stage 1 — Data** *(Core)*
- [~] T1.1 Obtain BUSI + configurable path — path configurable (`Config.data_root`); **real download still pending** (dev uses synthetic fixtures)
- [x] T1.2 `list_busi_samples` (+ normal, relative paths)
- [x] T1.3 `merge_masks` (OR, empty→zeros, nearest resize, binary)
- [x] T1.4 `build_transforms` (albumentations; mask nearest) + `transforms_from_config`
- [x] T1.5 `make_split` (stratified 3-class) / `save_split` / `load_split`
- [x] T1.6 `BUSIDataset` (returns image, mask, label_idx)
- [x] T1.7 `tests/test_data.py` — 18 pass, 1 skip (real-count, auto-skips)
- [~] **CP1** data checkpoint — all automated tests green; **visual eyeball of real overlays deferred** until dataset present (needs Stage 5 viz)

**Stage 2 — Model** *(Core)*
- [x] T2.1 Import + modernize reused author blocks — F.upsample→interpolate, F.sigmoid→torch.sigmoid, init.*→init.*_ (in grid_attention_layer.py, networks_other.py)
- [x] T2.2 `UnetGridGatingSignal2D`, `UnetDsv2D`
- [x] T2.3 Skip-module wrapper contract — `AdditiveAttentionGate`, `forward(x,g)->(x_filtered, spatial_map)`
- [x] T2.4 `AttentionUNet2D` (+ plain `unet` baseline via authors' unet_2D, is_deconv=True)
- [x] T2.5 `get_model` factory — `unet`, `attention_unet` (cbam/scse wired, land Stage 7/8 via `busi/attention_modules.py`)
- [x] T2.6 `tests/test_model.py` — 9 tests
- [x] **CP2** model checkpoint — 27 pass/1 skip; attention maps in [0,1] at skip resolutions; attention overhead ~10% params (2.47M→2.71M)

**Stage 3 — Losses & Metrics** *(Core)*
- [x] T3.1 `SoftDiceLoss2D` (device-agnostic, F.one_hot) + `get_loss` factory
- [x] T3.2 `focal_tversky_loss` — documented stub (full impl Stage 8 / T8.4)
- [x] T3.3 `dice_score`, `iou_score` (empty-mask conventions; exact, no eps)
- [x] T3.4 `specificity_on_normals`
- [x] T3.5 `attention_focus_ratio`
- [x] T3.6 `tests/test_losses.py`, `tests/test_metrics.py`
- [x] **CP3** losses/metrics checkpoint — 46 pass/1 skip; known-value checks (Dice=0.667, IoU=0.5) green. Full docstring coverage across `busi/`.

**Stage 4 — Training & Eval** *(Core)*
- [x] T4.1 `set_seed`, `get_device` (cuda→mps→cpu)
- [x] T4.2 `train_one_epoch`
- [x] T4.3 `evaluate` (splits lesion Dice/IoU vs specificity by class label)
- [x] T4.4 `fit` (Adam + ReduceLROnPlateau on val Dice, early stop, best-ckpt, history)
- [x] T4.5 `tests/test_train.py` — 6 tests (incl. overfit-one-batch)
- [x] **CP4** training checkpoint *(critical gate)* — overfit→Dice>0.95, smoke 1-epoch fit OK. **Finding: lr=1e-2 collapses (soft-Dice cold start); lr=1e-3 default validated.** Colab 1-epoch smoke deferred to Stage 6.

**Stage 5 — Visualization** *(Core)*
- [x] T5.1 `overlay_mask`, `overlay_attention` (upsamples coarse maps), `plot_prediction`
- [x] T5.2 `tests/test_viz.py` — 5 tests
- [x] **CP5** viz checkpoint — 57 pass/1 skip; sample figure rendered & eyeballed (overlays + attention align)

**Stage 6 — Orchestration + Core experiments** → **MILESTONE: CORE**
- [x] T6.1 `project.ipynb` bootstrap (clone/mount/pip + data verify + experiment cells)
- [x] T6.2 Experiment runner — `busi/experiment.py`: build_loaders, run_experiment, run_seeds (aggregate mean/std), tested end-to-end on synthetic BUSI
- [ ] T6.3 Run C1 (baseline) ×3 seeds — **needs real dataset**
- [ ] T6.4 Run C2 (attention) ×3 seeds — **needs real dataset**
- [~] T6.5 Results table + figures — builder code done (`make_results_table`, `save_prediction_figures`); awaits real runs
- [ ] **CP6** ✅ **CORE MILESTONE COMPLETE (submittable)** — awaits real dataset + runs

**Stage 7 — CBAM** → **MILESTONE: DESIRED**
- [x] T7.1 `CBAM2D` (`busi/attention_modules.py`) + `cbam_unet` registered; gating made conditional (self-attention variants skip it)
- [x] T7.2 CBAM tests — 6 (contract, small-channels, ignores-g, builds, attention maps, backward); full suite 71 pass/1 skip
- [ ] T7.3 Run D1 ×3 seeds + extend table/figures — **needs real dataset** (already in notebook `MODELS`)
- [ ] **CP7** ✅ **DESIRED MILESTONE COMPLETE** — awaits real runs

**Stage 8 — scSE (+ Focal-Tversky)** → **MILESTONE: STRETCH**
- [x] T8.1 `scSE2D` (`busi/attention_modules.py`) + `scse_unet` registered
- [x] T8.2 scSE tests — contract/build/maps/backward
- [ ] T8.3 Run S1 ×3 seeds + extend table/figures — **needs real dataset** (in notebook `MODELS`)
- [x] T8.4 `focal_tversky_loss` implemented (+ `FocalTverskyLoss`, wired via `get_loss`/`fit`) + tests; **run** S2 needs real dataset
- [ ] **CP8** ✅ **STRETCH MILESTONE COMPLETE** — code done; runs await real dataset

**Stage 9 — Report & Submission**
- [ ] T9.1 6-page report (required structure)
- [ ] T9.2 README + pinned `requirements.txt` + data-access instructions
- [ ] T9.3 Clean-room Colab reproduction from scratch
- [ ] T9.4 Package `id1_id2.zip`
- [ ] **CP9** ✅ **SUBMISSION READY**

---

## Requirements

### Functional (FR)
| ID | Requirement |
|---|---|
| FR1 | Load BUSI incl. `normal` (empty masks); keep per-sample class label |
| FR2 | Merge multiple masks (OR); normal → all-zero mask; binary output |
| FR3 | Stratified (3-class) 70/15/15 split, persisted + reproducible (seed 42) |
| FR4 | Augmentation applied identically to image+mask (albumentations); mask stays binary |
| FR5 | 2D Attention U-Net (additive gate + deep supervision) + plain U-Net baseline |
| FR6 | Skip-attention module is swappable: additive gate / CBAM / scSE, shared backbone |
| FR7 | Soft Dice loss (device-agnostic); Focal-Tversky as stretch |
| FR8 | Metrics: lesion Dice + IoU (benign+malignant), specificity (normals), attention-focus ratio |
| FR9 | Training loop: seeded, device-agnostic, early stopping, best-checkpointing, LR schedule |
| FR10 | Multi-seed experiment runner (3 seeds) with mean±std aggregation |
| FR11 | Visualization: mask overlays, attention-map overlays, comparison figures |
| FR12 | Single Colab notebook orchestrates clone→data→runs→figures |
| FR13 | Report (≤6 pp) + reproducible submission package |

### Non-functional (NFR)
| ID | Requirement |
|---|---|
| NFR1 | Device-agnostic: `cuda → mps → cpu`; same code local + Colab |
| NFR2 | Reproducibility: fixed seeds, persisted split, pinned deps |
| NFR3 | Runs on free Colab T4 within session/memory limits; checkpoints to Drive |
| NFR4 | Docstrings on all functions; MIT attribution on reused author code |
| NFR5 | Every public function has ≥1 unit/validation test |
| NFR6 | All logic in `busi/*.py`; notebook only orchestrates |

---

## Stage detail (tasks, files, DoD, tests)

> Global DoD for every code task: docstring written, function importable, its tests green, no hardcoded absolute paths (use `busi/config.py`), no `.cuda()` (use `device`).

### Stage 0 — Setup & environment
- **T0.1 Fork + clone + branch.** Fork `ozan-oktay/Attention-Gated-Networks` to the team account; clone; create a working branch. *DoD:* fork exists; repo cloned; MIT `LICENSE` retained.
- **T0.2 Package skeleton.** Create `busi/{__init__,config,data,model,losses,metrics,train,viz}.py` (stubs), `tests/`, `requirements.txt`, `.gitignore` (ignore `data/`, `checkpoints/`, `results/`, `__pycache__`), README stub, `data/ checkpoints/ results/` dirs. `config.py` holds: paths, `IMG_SIZE=256`, `SEEDS=[42,1,7]`, split ratios, Adam/LR/epoch/patience/batch settings. *DoD:* `python -c "import busi"` works; `pytest -q` collects 0 tests without error.
- **T0.3 Local env.** Create venv (fall back to Python 3.11/3.12 if 3.13 wheels break); `pip install torch torchvision albumentations opencv-python scikit-learn matplotlib pytest`; `export PYTORCH_ENABLE_MPS_FALLBACK=1`. Verify reused author modules import (`GridAttentionBlock2D`, `unetConv2`, `unetUp`, `init_weights`, `unet_2D`, `SoftDiceLoss`). *DoD:* imports succeed; `get_device()` returns `mps` locally.
- **T0.4 Colab bootstrap cell.** A notebook cell: clone fork, `pip install` deps, mount Drive, set data/checkpoint paths. *DoD:* documented in `project.ipynb` (can stay untested until Stage 6).
- **CP0 (gate):** `pytest` runs clean; `import busi` + author-module imports work locally; device detection OK.

### Stage 1 — Data pipeline (`busi/data.py`) — *Core*
- **T1.1 Obtain BUSI.** Download from Kaggle (`aryashah2k/breast-ultrasound-images-dataset`); place at `config.DATA_ROOT` (local) / Drive (Colab). Document exact steps in README. *DoD:* folders `benign/ malignant/ normal/` present at configurable path.
- **T1.2 `list_busi_samples(root, classes=('benign','malignant','normal'))`.** Returns samples with image path, mask paths (empty list for normal), label. *DoD:* signature per DESIGN §2.2.
- **T1.3 `merge_masks(mask_paths, size)`.** OR-merge + binarize; empty list → all-zeros. Mask resized nearest-neighbor.
- **T1.4 `build_transforms(img_size=256, train=True)`.** albumentations; image bilinear + `/255`, mask nearest; train adds flip/rotate/scale/brightness; returns tensors (image `(1,H,W)` float, mask `(H,W)` long).
- **T1.5 `make_split` / `save_split` / `load_split`.** Stratified 3-class, seed 42; persist to `splits/split.json` (tracked in git; `data/` itself is ignored); commit it.
- **T1.6 `BUSIDataset`.** `__getitem__ → (image (1,H,W) float32 [0,1], mask (H,W) int64 {0,1})`.
- **T1.7 `tests/test_data.py`.** All tests in DESIGN §2.3: counts (437/210/133, total 780), `test_normal_empty_mask`, merge OR/empty/binary, binary-after-resize, split no-leakage / stratified / deterministic / roundtrip, item shapes, transform mask-binary, geometry-synced.
- **CP1 (gate):** all data tests green; **manual sanity:** overlay 3–4 random image+mask pairs and eyeball alignment; `splits/split.json` committed.

### Stage 2 — Model (`busi/model.py`) — *Core*
- **T2.1 Import + modernize reused blocks.** Import `GridAttentionBlock2D`, `unetConv2`, `unetUp`, `init_weights`, `unet_2D`. Fix deprecations in the code paths you touch: `F.upsample → F.interpolate`, drop `Variable`, remove hardcoded `.cuda()`. *DoD:* a forward pass through `GridAttentionBlock2D` on random 2D tensors works on CPU/MPS.
- **T2.2 `UnetGridGatingSignal2D`, `UnetDsv2D`.** 2D copies of the 3D-only originals (DESIGN §3.2).
- **T2.3 Skip-module wrapper contract.** All skip-attention modules expose `forward(x, g=None) -> (x_filtered, spatial_map∈[0,1])`. Wrap `GridAttentionBlock2D` to this contract (uses `g`).
- **T2.4 `AttentionUNet2D`.** Encoder (`unetConv2`+`MaxPool2d`) → bottleneck + gating → skip-attention at levels 2/3/4 → decoder (`unetUp`) → deep supervision (`UnetDsv2D`) → fused `final`. `forward(x, return_attention=False)`. Baseline `unet` = author's `unet_2D` (verbatim, 1-channel in / 2-class out).
- **T2.5 `get_model(name, ...)`.** Names: `unet`, `attention_unet` (this stage). Variants share one backbone; only the skip module differs.
- **T2.6 `tests/test_model.py`.** DESIGN §3.4 subset for {unet, attention_unet}: forward shape, softmax valid, attention range, params-vs-baseline, gating/dsv shapes, deterministic init, backward, `test_variants_build` (unet + attention).
- **CP2 (gate):** model tests green; attention maps in `[0,1]`; param overhead of attention vs baseline < ~10%.

### Stage 3 — Losses & Metrics (`busi/losses.py`, `busi/metrics.py`) — *Core*
- **T3.1 `SoftDiceLoss2D`.** Author's Dice, device-agnostic (`One_Hot` uses input device, not `.cuda()`); on `final`.
- **T3.2 `focal_tversky_loss`.** Signature + stub now; full impl deferred to Stretch (T8.4).
- **T3.3 `dice_score`, `iou_score`.** Binary, with empty-mask conventions (both-empty → 1.0; empty-GT + pred → 0.0).
- **T3.4 `specificity_on_normals(pred_masks, area_thresh=0.005)`.**
- **T3.5 `attention_focus_ratio(att_map, gt_mask)`.**
- **T3.6 Tests.** DESIGN §4.3 + §5.3: dice perfect/worst/differentiable/device-agnostic/range; dice/iou identical/disjoint/known(0.667); focus inside/outside/uniform; empty-both, empty-gt-nonempty-pred; specificity all-clean / all-hallucinate.
- **CP3 (gate):** all loss + metric tests green (known-value checks included).

### Stage 4 — Training & Eval (`busi/train.py`) — *Core*
- **T4.1 `set_seed(seed=42)`, `get_device()`.**
- **T4.2 `train_one_epoch(model, loader, loss_fn, optimizer, device) -> float`.**
- **T4.3 `evaluate(model, loader, device) -> dict`.** Keys: `lesion_dice, lesion_iou, dice_benign, dice_malignant, specificity, fp_rate`. Lesion metrics on benign+malignant only; specificity on normals.
- **T4.4 `fit(model, train_loader, val_loader, cfg) -> dict`.** Adam+wd, `ReduceLROnPlateau` on val Dice, early stopping (patience 15), best-checkpoint to `checkpoints/`, returns history + best path.
- **T4.5 `tests/test_train.py`.** seed-reproducible, train-step-updates, evaluate-keys, **`test_overfit_one_batch`** (attention_unet, 2–4 imgs, Dice>0.95), `test_smoke_one_epoch` (20-img subset).
- **CP4 (critical gate):** `test_overfit_one_batch` passes on MPS locally, **and** a 1-epoch smoke run completes once on Colab. If overfit fails, STOP and debug (mask alignment / channels / loss) before any real training.

### Stage 5 — Visualization (`busi/viz.py`) — *Core*
- **T5.1** `overlay_mask`, `overlay_attention`, `plot_prediction`.
- **T5.2 `tests/test_viz.py`.** overlay shape, overlay smoke, plot_prediction smoke (with/without attention).
- **CP5 (gate):** viz tests green; produce one sample figure (image | GT | pred | attention) and eyeball it.

### Stage 6 — Orchestration + Core experiments (`project.ipynb`) — **MILESTONE: CORE**
- **T6.1** Notebook bootstrap (clone/mount/download/config) — finalize T0.4.
- **T6.2 Experiment runner.** `run_experiment(model_name, loss_name, seed) -> metrics dict`; loop over `SEEDS`; aggregate mean±std; save to `results/<model>.json`.
- **T6.3** Run **C1** (`unet`, Dice) ×3 seeds.
- **T6.4** Run **C2** (`attention_unet`, Dice) ×3 seeds.
- **T6.5** Build results table (lesion Dice/IoU, specificity, #params, per-class, mean±std) + figures (seg overlays; attention maps for benign/malignant/normal; focus-ratio).
- **CP6 ✅ CORE COMPLETE:** table with C1 vs C2 (mean±std), attention-map figures, specificity reported, checkpoints + `results/*.json` saved. **This is a complete, submittable project.** Commit + tag.

### Stage 7 — CBAM — **MILESTONE: DESIRED**
- **T7.1 `CBAM2D(channels, reduction=16)`** obeying the skip-module contract; register `cbam_unet` in `get_model`.
- **T7.2** Tests: `test_cbam_contract` + extend `test_variants_build` / `test_attention_range` / `test_backward` to include `cbam_unet`.
- **T7.3** Run **D1** (`cbam_unet`) ×3 seeds; add row to table + attention-map figures.
- **CP7 ✅ DESIRED COMPLETE:** CBAM tests green; D1 in table/figures. Commit + tag.

### Stage 8 — scSE (+ Focal-Tversky) — **MILESTONE: STRETCH**
- **T8.1 `scSE2D(channels, reduction=8)`** (contract); register `scse_unet`.
- **T8.2** Tests: `test_scse_contract` + extend variant tests.
- **T8.3** Run **S1** (`scse_unet`) ×3 seeds; extend table/figures.
- **T8.4 (optional)** Finish `focal_tversky_loss` + its tests; run **S2** (best variant + Focal-Tversky); add loss-sensitivity row.
- **CP8 ✅ STRETCH COMPLETE:** scSE (and optionally Focal-Tversky) tests green + in results. Commit + tag.

### Stage 9 — Report & Submission
- **T9.1 Report (≤6 pp)** per instructions: Title, Introduction (paper summary + motivation), Proposed Extension (gap/hypothesis/alternatives), Methodology (model changes, dataset/preprocessing, training details, tools), Results & Analysis (table, figures, comparison, failure cases), Conclusion (findings, limitations incl. specificity finding, future work).
- **T9.2** README (Kaggle link + exact download steps, how to run tests, how to reproduce each milestone) + pinned `requirements.txt`.
- **T9.3 Clean-room reproduction:** run `project.ipynb` top-to-bottom on a fresh Colab runtime from the fork; confirm figures/tables regenerate.
- **T9.4** Package `id1_id2.zip` (code + PDF report; dataset excluded, instructions included).
- **CP9 ✅ SUBMISSION READY.**

---

## Dependencies & order
- Strict order: **0 → 1 → 2 → 3 → 4 → 5 → 6**. Stage 4 depends on 1–3; Stage 6 depends on 1–5.
- **7 and 8 depend only on 2/4/6** (add a module + a run); either can be skipped without breaking Core.
- `focal_tversky_loss` (T3.2 stub → T8.4 full) is the only intentionally split task.

## Risks & gotchas (carry forward)
- **Mask interpolation:** must be nearest-neighbor everywhere, or masks go non-binary (caught by `test_transform_mask_binary`).
- **Old author code:** `F.upsample`, `Variable`, hardcoded `.cuda()` — modernize on touch (T2.1, T3.1).
- **MPS op gaps:** keep `PYTORCH_ENABLE_MPS_FALLBACK=1`; if an op errors, run that piece on CPU.
- **Python 3.13 wheels:** fall back to 3.11/3.12 venv if install friction.
- **Colab ephemerality:** always checkpoint to Drive; the notebook must be re-runnable after a disconnect.
- **Metric honesty:** never average lesion Dice over normal (empty-GT) images — report specificity separately (enforced by `evaluate` splitting by label).
- **Fair comparison:** all variants share backbone + split + seeds + budget; only the skip module changes.

## Definition of "milestone done"
A milestone is done when: its stage checkpoint passed, results are in `results/*.json`, figures are regenerated by the notebook, the branch is committed and tagged, and this tracker + memory file are updated.

---

## Addendum — crash-safety & PoC run settings (post Stage 8)

- Training is resumable: `fit` writes `<exp>_last.pt` (full state) every epoch and
  resumes from it; `run_experiment` skips already-finished seeds. Outputs go to
  Google Drive (`checkpoints_dir`/`results_dir`) so a disconnect loses nothing.
- Notebook restructured: one atomic cell per model + a KNOBS block. Live per-epoch
  logging added throughout.
- PoC run settings: `EPOCHS=50`, `PATIENCE=8`, `SEEDS=[42,1,7]` (≈1–1.5 h on a T4).
