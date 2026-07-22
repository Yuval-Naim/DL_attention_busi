# DESIGN — Attention U-Net → BUSI (Breast Ultrasound Lesion Segmentation)

**Course:** Medical Images Processing with Deep Learning – 336033 (Technion).
**Paper (extended):** Attention U-Net: Learning Where to Look for the Pancreas (Oktay et al., 2018, arXiv:1804.03999).
**Original code (MIT):** https://github.com/ozan-oktay/Attention-Gated-Networks
**Extension type:** apply to a new domain (breast ultrasound) + architecture comparison (attention modules) + interpretability.
**Deadline:** 2026-08-20. Submission: single `.zip` (`id1_id2.zip`) with code + 6-page PDF report; implementation runs on Google Colab.

---

## 0. Locked decisions (the single source of truth)

| Decision | Value | Rationale |
|---|---|---|
| Task | Binary lesion segmentation (2D) | BUSI is 2D PNG; lesion vs background |
| Dataset | BUSI, **all 780** = benign (437) + malignant (210) + **normal (133, empty mask)** | Option B: include healthy scans so the model learns to stay quiet on clean tissue (specificity) |
| Image size | **256 × 256**, grayscale (1 channel) | Fits Colab memory; standard for BUSI seg |
| Output head | **2-channel softmax** (background, lesion) | Lets us reuse the authors' `SoftDiceLoss` / `One_Hot` / `apply_argmax_softmax` verbatim |
| Loss | **Dice + CE** (default); compared vs pure Dice & Focal-Tversky | **Empirical:** pure soft-Dice collapses to all-background on BUSI (val Dice stuck at 0); adding CE breaks the cold-start. Dice handles imbalance, CE gives the per-pixel gradient. |
| Augmentation | **albumentations** (image+mask synced) | Clean geometric sync between image and mask |
| Split | **Stratified 70/15/15**, **seed 42**, persisted to file | Reproducible; balanced benign/malignant across splits |
| Model | **2D Attention U-Net** (AG on skips 2/3/4 + deep supervision) | Faithful 2D port of the authors' `unet_CT_single_att_dsv_3D` |
| Attention variants (comparison) | Additive gate (paper) · **CBAM** (desired) · **scSE** (stretch), each replacing the skip-attention module | Which attention mechanism best segments breast lesions? Backbone identical otherwise |
| Baseline (ablation) | Authors' `unet_2D` (plain U-Net), **verbatim** | Isolates the attention contribution |
| Optimizer | **Adam**, lr `1e-3`, weight_decay `1e-5` | Standard, robust |
| LR schedule | `ReduceLROnPlateau` on val Dice (patience 8) | Simple, effective |
| Batch size | 8–16 (Colab T4) / 4–8 (local MPS) | Fits memory |
| Epochs | 50–100, **early stopping** on val Dice (patience 15) | Proof-of-concept, avoids overfitting |
| Metrics | **Lesion Dice + IoU** (benign+malignant) · **Specificity** (normals) · **attention-focus ratio** | Honest split: seg quality vs false-positive suppression vs interpretability |
| Rigor | **3 seeds** {42, 1, 7}; report **mean ± std** | Distinguishes real gains from noise; credibility |
| Device | auto: `cuda → mps → cpu` | Same code local + Colab |
| Code base | **Fork of the authors' repo**; new code under `busi/` | Provable lineage to the paper (MIT license) |

**Non-goals:** no 3D, no full-repo port (Visdom/configs/NIfTI loaders skipped), no full-scale training, no large hyperparameter search, **no transfer learning / pretrained encoder** (no compatible weights), **no synthetic speckle-noise study** (US is already noisy → artificial).

### Milestones (build order — collect wins, don't jump ahead)
1. **Core** — *a complete, submittable project on its own:* data pipeline (Option B) → 2D Attention U-Net + plain U-Net baseline → training → lesion Dice/IoU + specificity + attention maps, over 3 seeds. This alone is the full ablation + interpretability + specificity story.
2. **Desired** — add the **CBAM** variant to the comparison.
3. **Stretch** — add the **scSE** variant; and/or a **Focal-Tversky** loss run on the best variant.

Each milestone is independently reportable — if a later one is cut for time, the project still stands.

---

## 1. Repository & code layout

Fork `ozan-oktay/Attention-Gated-Networks`, add:

```
busi/
  __init__.py
  config.py      # all hyperparameters + paths (one place)
  data.py        # sample listing, mask merge, split, BUSIDataset, transforms
  model.py       # AttentionUNet2D + CBAM2D + scSE2D + UnetGridGatingSignal2D + UnetDsv2D + get_model
  losses.py      # SoftDiceLoss2D (adapted) + focal_tversky_loss (stretch)
  metrics.py     # dice_score, iou_score, specificity_on_normals, attention_focus_ratio
  train.py       # set_seed, get_device, train_one_epoch, evaluate, fit
  viz.py         # overlay_mask, overlay_attention, plot_prediction
tests/
  test_data.py test_model.py test_losses.py test_metrics.py test_train.py test_viz.py
project.ipynb    # orchestration for Colab (clone → data → run milestones → figures)
```

**Reused verbatim / imported from the authors (the link to the paper):**
- `models/layers/grid_attention_layer.py` → `GridAttentionBlock2D` (**the attention gate itself**).
- `models/networks/utils.py` → `unetConv2`, `unetUp`.
- `models/networks_other.py` → `init_weights`.
- `models/layers/loss.py` → `SoftDiceLoss`, `One_Hot` (adapted: device-agnostic, see §5).
- `models/networks/unet_2D.py` → plain U-Net baseline for the ablation.

**Written new (the extension — the authors never did 2D breast US):**
- `AttentionUNet2D`, `UnetGridGatingSignal2D`, `UnetDsv2D` (2D versions of their 3D-only `UnetGridGatingSignal3` / `UnetDsv3`).
- `CBAM2D` (desired) and `scSE2D` (stretch) — alternative attention modules that drop into the skip-attention slot for the comparison study.
- Entire `busi/` data + training + eval + viz stack.

**Attribution:** keep the MIT LICENSE; add a header in every file that copies authors' code citing the repo + paper.

---

## 2. Data pipeline (`busi/data.py`)

### 2.1 Design decisions
- BUSI folder layout: `<root>/<class>/<class> (n).png` and `<class> (n)_mask.png` (+ occasional `<class> (n)_mask_1.png`).
- **Merge** all mask files of an image with logical OR, then binarize `>0 → 1`. **Normal images ship with an all-black mask (or none) → all-zero mask either way.**
- Paths in `Sample` are stored **relative to `root`** so the persisted split is portable (local ↔ Colab/Drive); `BUSIDataset(samples, root, ...)` joins `root`.
- **Resize** image bilinear, **mask nearest-neighbor** (keeps mask binary — critical).
- Normalize image `/255.0 → [0,1]`.
- Tensors: image `(1, H, W)` float32; mask `(H, W)` int64 (long) with values `{0,1}` (softmax target).
- Keep the **class label** (`benign`/`malignant`/`normal`) per sample — needed for per-class Dice and for the specificity split.
- Split: **stratified across all 3 classes**, ratios 70/15/15, seed 42, saved as JSON of filename lists.
- Augmentation (train only): HFlip, rotate ±15°, mild scale/shift, brightness/gamma jitter; geometry applied identically to image+mask by albumentations.

### 2.2 Functions & signatures
```python
list_busi_samples(root: str, classes=('benign','malignant','normal')) -> list[Sample]
    # Sample = {'image': path, 'masks': [paths] (empty list for normal), 'label': 'benign'|'malignant'|'normal'}
merge_masks(mask_paths: list[str], size: tuple[int,int]) -> np.ndarray  # (H,W) uint8 {0,1}
make_split(samples, ratios=(0.7,0.15,0.15), seed=42) -> dict[str, list[Sample]]
save_split(split, path) -> None ; load_split(path) -> dict
build_transforms(img_size=256, train=True) -> albumentations.Compose
class BUSIDataset(Dataset):
    __init__(samples, root, transform=None, img_size=256)   # joins root to relative sample paths
    __getitem__(i) -> (image: FloatTensor[1,H,W], mask: LongTensor[H,W], label_idx: int)
    __len__() -> int
    # label_idx (CLASS_TO_IDX benign=0/malignant=1/normal=2) lets evaluate() split
    # lesion metrics (benign+malignant) from specificity (normal).
```

### 2.3 Unit / validation tests (`tests/test_data.py`)
- `test_list_counts`: `len(benign)==437`, `len(malignant)==210`, `len(normal)==133`, total 780; benign+malignant have ≥1 mask file, normal has 0.
- `test_normal_empty_mask`: a normal sample → merged mask is all-zeros; dataset returns a mask of all `0`.
- `test_merge_or`: two masks with disjoint blobs → union has both; single mask → identical; empty list → all-zeros; output values ⊆ {0,1}; shape == size.
- `test_merge_binary_after_resize`: after `merge_masks` at 256, `set(unique) ⊆ {0,1}` (no 0.5 gray from interpolation).
- `test_split_no_leakage`: intersection of image filenames across train/val/test is empty.
- `test_split_stratified`: per-split class ratio within ±3% of global ratio.
- `test_split_deterministic`: `make_split(...,seed=42)` twice → identical filename lists.
- `test_split_roundtrip`: `load_split(save_split(s))==s`.
- `test_dataset_item_shapes`: image `(1,256,256)` float32 in `[0,1]`; mask `(256,256)` int64 in `{0,1}`.
- `test_transform_mask_binary`: after `build_transforms(train=True)`, mask still binary (nearest interp).
- `test_geometry_synced`: with a flip-only transform (p=1), flipping image and mask stays aligned (overlay of pred==gt region preserved) — checked by applying a known transform to a synthetic image where lesion is a corner square and asserting the square moved consistently in both.

---

## 3. Model (`busi/model.py`)

### 3.1 Architecture (faithful 2D port of `unet_CT_single_att_dsv_3D`)
- `filters = [64,128,256,512,1024] // feature_scale`, default `feature_scale=4 → [16,32,64,128,256]`.
- Encoder: 4× (`unetConv2` → `MaxPool2d(2)`).
- Bottleneck: `unetConv2` → `UnetGridGatingSignal2D` (gating signal `g`).
- **Skip-attention module (swappable)** on skips at levels 2/3/4 — default `GridAttentionBlock2D` (paper's additive gate); alternatives `CBAM2D`/`scSE2D` for the comparison. Level-1 skip passes through directly. **All variants share the identical backbone / decoder / DSV** — only the skip module changes.
- Decoder: 4× `unetUp`.
- Deep supervision: `UnetDsv2D` at up4 (×8), up3 (×4), up2 (×2) + 1×1 conv at up1; concat 4 → final `Conv2d(4*n_classes, n_classes, 1)`.
- Output: logits `(B, 2, H, W)`; `apply_argmax_softmax` = softmax over dim=1.
- `forward(x, return_attention=False)` → logits, and (optionally) dict of attention maps `{att2, att3, att4}` each in `[0,1]`, upsampled to their skip resolution.

### 3.2 New 2D blocks
- `UnetGridGatingSignal2D(in, out, ks=1, bn=True)`: Conv2d→BN→ReLU. (2D copy of `UnetGridGatingSignal3`.)
- `UnetDsv2D(in, n_classes, scale_factor)`: Conv2d(1×1) → `Upsample(scale_factor, mode='bilinear')`. (2D copy of `UnetDsv3`.)
- **`CBAM2D(channels, reduction=16)`** *(desired)*: channel attention (avg+max pool → shared MLP → sigmoid) then spatial attention (conv over channel-pooled maps → sigmoid). ~40 lines.
- **`scSE2D(channels, reduction=8)`** *(stretch)*: parallel spatial-SE (1×1 conv → sigmoid) + channel-SE, combined. ~20 lines; from the medical-segmentation literature (Roy et al.).
- **Skip-module contract:** every skip-attention module exposes `forward(x, g=None) -> (x_filtered, spatial_map)` where `spatial_map ∈ [0,1]` (same H×W as `x`), so all variants are drop-in interchangeable. The additive gate uses `g`; CBAM/scSE are self-attention and ignore `g`. Keeping a spatial map from every variant means the **interpretability analysis (§5/§7) works across all of them**.

### 3.3 Factory
```python
get_model(name: str, in_channels=1, n_classes=2, feature_scale=4) -> nn.Module
    # name in {'unet' (baseline),
    #          'attention_unet' (paper's additive gate),
    #          'cbam_unet' (desired),
    #          'scse_unet' (stretch)}
    # the three attention variants share one backbone; only the skip-attention module differs
```

### 3.4 Unit tests (`tests/test_model.py`)
- `test_forward_shape`: input `(2,1,256,256)` → output `(2,2,256,256)`.
- `test_softmax_valid`: `softmax(logits).sum(dim=1)≈1` for all pixels.
- `test_attention_range`: for **every** attention variant, each returned spatial map has values in `[0,1]` and shapes match skip levels (e.g. att4 at H/8).
- `test_params_more_than_baseline`: `#params(attention_unet) > #params(unet)` (attention adds params) but overhead small (< ~10%).
- `test_gating_signal_shape`: `UnetGridGatingSignal2D(32,32)` on `(2,32,16,16)` → `(2,32,16,16)`.
- `test_dsv_upsample`: `UnetDsv2D(64,2,scale_factor=4)` on `(2,64,32,32)` → `(2,2,128,128)`.
- `test_cbam_contract`: `CBAM2D(32)` on `(2,32,16,16)` → `(x_filtered (2,32,16,16), spatial_map (2,1,16,16) ∈ [0,1])`.
- `test_scse_contract`: `scSE2D(32)` obeys the same `(x_filtered, spatial_map)` contract and shapes.
- `test_variants_build`: `get_model` builds all of {`unet`,`attention_unet`,`cbam_unet`,`scse_unet`}; each forward on `(2,1,256,256)` → `(2,2,256,256)`.
- `test_deterministic_init`: same seed → identical initial weights (allclose).
- `test_backward`: `loss.backward()` populates grads on all trainable params (no None), for every variant.

---

## 4. Losses (`busi/losses.py`)

### 4.1 Decisions
- **Default: `DiceCELoss` = SoftDiceLoss2D + cross-entropy.** *Empirical finding:* pure soft-Dice **fails to train from scratch on BUSI** — the model collapses to all-background and val Dice stays 0 (Dice's gradient vanishes when the foreground prediction is ~0). Adding CE supplies a strong per-pixel gradient that breaks the collapse. Measured: Dice+CE ≈0.62 val Dice @12 epochs vs pure Dice 0.00. (Great material for the report's "challenges".)
- `SoftDiceLoss2D` — authors' `SoftDiceLoss` made **device-agnostic** (original `One_Hot` hardcodes `.cuda()`); kept selectable so the *collapse itself* is a reportable result in the loss study.
- Loss computed on the fused `final` output (matches authors).
- Stretch: `focal_tversky_loss(alpha=0.7, beta=0.3, gamma=0.75)` for small/imbalanced lesions (loss-study comparison).

### 4.2 Signatures
```python
class SoftDiceLoss2D(nn.Module):   # __init__(n_classes=2); forward(logits, target_long) -> scalar
def focal_tversky_loss(logits, target_long, alpha=0.7, beta=0.3, gamma=0.75) -> scalar
```

### 4.3 Unit tests (`tests/test_losses.py`)
- `test_dice_perfect`: logits that argmax exactly to target → loss ≈ 0 (< 0.05).
- `test_dice_worst`: logits opposite to target → loss ≈ 1 (> 0.9).
- `test_dice_differentiable`: `loss.requires_grad` and `backward()` gives finite grads.
- `test_dice_device_agnostic`: runs on cpu (and mps/cuda if available) without `.cuda()` error.
- `test_focal_tversky_reduces`: with `alpha=beta=0.5, gamma=1` value equals plain Tversky/Dice-like baseline (sanity).
- `test_loss_range`: `0 ≤ loss ≤ 1` on random inputs.

---

## 5. Metrics (`busi/metrics.py`)

### 5.1 Decisions
- Binary Dice (F1) and IoU (Jaccard) from predicted lesion mask (argmax==1) vs GT.
- **Empty-mask convention (needed for normal images):** empty GT + empty pred → Dice = IoU = 1.0; empty GT + any predicted lesion → 0.0.
- **Reporting is split (honest, not inflated):** *lesion Dice/IoU* computed **only on benign+malignant**; *specificity* computed **on normals** = fraction of normal images whose predicted lesion area ≤ τ (e.g. τ = 0.5% of pixels). Also report the false-positive rate (1 − specificity).
- `attention_focus_ratio` = (attention mass inside GT lesion) / (total attention mass) ∈ [0,1]; interpretability metric (lesion images only).

### 5.2 Signatures
```python
dice_score(pred_mask, gt_mask, eps=1e-6) -> float          # inputs (H,W) in {0,1}
iou_score(pred_mask, gt_mask, eps=1e-6) -> float
specificity_on_normals(pred_masks, area_thresh=0.005) -> float  # frac of normal preds with lesion area ≤ thresh
attention_focus_ratio(att_map, gt_mask) -> float           # att (H,W) in [0,1], gt (H,W) {0,1}
```

### 5.3 Unit tests (`tests/test_metrics.py`)
- `test_dice_identical`: identical masks → 1.0.
- `test_dice_disjoint`: non-overlapping → 0.0.
- `test_dice_known`: pred=full image, gt=half → Dice = 2*0.5/(1+0.5)=0.667 (±1e-3).
- `test_iou_identical`/`test_iou_disjoint`/`test_iou_known` (half-overlap known value).
- `test_focus_all_inside`: att nonzero only inside gt → 1.0.
- `test_focus_all_outside`: att nonzero only outside gt → 0.0.
- `test_focus_uniform`: uniform att → equals lesion area fraction (±1e-3).
- `test_empty_both`: pred and gt empty → dice=iou=1.0 (convention).
- `test_empty_gt_nonempty_pred`: empty gt + predicted lesion → dice=iou=0.0 (convention).
- `test_specificity_all_clean`: all normal predictions empty → specificity 1.0.
- `test_specificity_all_hallucinate`: all normal predictions above threshold → specificity 0.0.

---

## 6. Training (`busi/train.py`)

### 6.1 Decisions
- `set_seed(seed)`: seeds python/numpy/torch (+ `torch.use_deterministic_algorithms` best-effort).
- `get_device()`: `cuda → mps → cpu`.
- Adam lr 1e-3, wd 1e-5; `ReduceLROnPlateau(mode='max', patience=8)` on val Dice.
- Early stopping (patience 15) on val Dice; checkpoint best weights.
- Checkpoints to Google Drive path on Colab.

### 6.2 Signatures
```python
set_seed(seed=42) -> None ; get_device() -> str
train_one_epoch(model, loader, loss_fn, optimizer, device) -> float          # mean train loss
evaluate(model, loader, device) -> dict  # {'lesion_dice','lesion_iou','dice_benign','dice_malignant','specificity','fp_rate'}
fit(model, train_loader, val_loader, cfg) -> dict  # history + best checkpoint path
```

### 6.3 Unit / integration tests (`tests/test_train.py`)
- `test_set_seed_reproducible`: two models built after `set_seed(42)` → identical initial weights.
- `test_train_step_updates`: after `train_one_epoch` on a tiny loader, at least one parameter changed and loss is finite.
- `test_evaluate_keys`: `evaluate` returns the expected metric keys and values in `[0,1]`.
- **`test_overfit_one_batch`** (the key sanity test): train `attention_unet` on 2–4 fixed images for ~200 steps → train Dice > 0.95. Catches mask misalignment, wrong channels, broken loss.
- `test_smoke_one_epoch`: full `fit` for 1 epoch on a 20-image subset runs end-to-end and returns a checkpoint path.

---

## 7. Visualization (`busi/viz.py`)

### 7.1 Signatures
```python
overlay_mask(image, mask, color=(1,0,0), alpha=0.4) -> np.ndarray        # RGB (H,W,3)
overlay_attention(image, att_map, cmap='jet', alpha=0.5) -> np.ndarray
plot_prediction(image, gt, pred, att_maps=None) -> matplotlib.Figure
```

### 7.2 Tests (`tests/test_viz.py`)
- `test_overlay_shape`: output `(H,W,3)`, dtype float in `[0,1]` or uint8.
- `test_overlay_smoke`: runs without exception on a random image+mask.
- `test_plot_prediction_smoke`: returns a Figure; no exception with and without attention maps.

---

## 8. Experiment matrix (runs → report artifacts)

Structured by milestone (§0). Every run uses the **same split + same training budget**, and is repeated over **3 seeds {42, 1, 7}** → report **mean ± std**.

**Core milestone**
| ID | Model | Loss | Produces |
|---|---|---|---|
| C1 | Plain U-Net (baseline) | Dice | Table rows (lesion Dice/IoU, specificity, #params, time); seg overlays |
| C2 | Attention U-Net (paper's additive gate) | Dice | Table rows; **attention-map figures**; focus-ratio |

**Desired milestone**
| ID | Model | Loss | Produces |
|---|---|---|---|
| D1 | U-Net + **CBAM** | Dice | Table rows; attention-map figures; focus-ratio |

**Stretch milestone**
| ID | Model | Loss | Produces |
|---|---|---|---|
| S1 | U-Net + **scSE** | Dice | Table rows; attention-map figures; focus-ratio |
| S2 | Best attention variant | **Focal-Tversky** | Extra table row (loss sensitivity) |

**Report figures:** (1) results table — lesion Dice, IoU, specificity, #params, per-class, mean±std across variants; (2) qualitative GT-vs-pred overlays per model; (3) **attention-map overlays comparing where each mechanism looks** (benign, malignant, + a normal image to show suppression); (4) failure-case discussion.

**Hypotheses:** (H1) attention improves lesion Dice over plain U-Net at small parameter overhead; (H2) the attention mechanisms differ — we identify which best fits breast US and *where each looks*; (H3) including normals lets the model suppress false positives on healthy tissue (high specificity).

---

## 9. Running: local vs Colab

**Local (M4 Pro, MPS)** — fast dev/debug loop:
- venv + `pip install torch torchvision albumentations opencv-python scikit-learn matplotlib pytest`.
- `export PYTORCH_ENABLE_MPS_FALLBACK=1` (some ops fall back to CPU).
- If Python 3.13 causes wheel issues, use a 3.11/3.12 venv.
- Run: all unit tests, overfit-one-batch, short runs, viz development.

**Colab (T4, CUDA)** — the required + final environment:
- `!git clone <fork>`; mount Drive for BUSI data + checkpoints (sessions are ephemeral, ~12h limit).
- Run: the milestone experiments (Core → Desired → Stretch), produce all report figures/tables.

**Portability rule:** all logic in `busi/*.py`, device auto-selected; the notebook only orchestrates. Same files run both places.

### Test ladder (cheapest → most convincing)
1. Shape test (§3.4). 2. Overfit-one-batch (§6.3) — best single bug-catcher. 3. Data integrity (§2.3). 4. Determinism (split + init). 5. Smoke 1-epoch run on both devices.

---

## 10. Reproducibility checklist
- Fixed seed 42 everywhere; split persisted to JSON and committed.
- All hyperparameters in `busi/config.py`.
- `requirements.txt` with pinned versions.
- README: exact Kaggle dataset link + download steps; how to run tests; how to reproduce the milestone runs.
- Report cites paper + repo; MIT license retained.

---

## 11. Addendum — crash-safety (added after the first Colab run)

Training is **resumable** and **Drive-persistent** so a Colab disconnect loses
nothing: `fit` saves the best weights + a full resume checkpoint (model+optimizer+
scheduler+epoch+history) every `ckpt_every` epochs to `checkpoints_dir`; point
`checkpoints_dir`/`results_dir` at Google Drive. `run_experiment` skips a seed whose
result JSON already exists and resumes an interrupted one from its last epoch.
Per-epoch logs print live. Notebook: one cell per model + a KNOBS block
(`EPOCHS`/`PATIENCE`/`SEEDS`/`MODELS`/`QUICK`). PoC defaults: 50 epochs, patience 8.
