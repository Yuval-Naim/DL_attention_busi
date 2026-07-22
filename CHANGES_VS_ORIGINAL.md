# What we changed vs the original Attention U-Net repo

This project is a **fork** of the paper authors' code —
[`ozan-oktay/Attention-Gated-Networks`](https://github.com/ozan-oktay/Attention-Gated-Networks)
(MIT license). This file explains, in plain terms, **exactly what is "ours" (the
extension) vs "theirs"**, so anyone can see the boundary at a glance.

**The short version:** we kept the authors' **attention gate untouched** (only
fixed two files so they run on today's PyTorch), and built an entirely **new 2D
pipeline around it** — under `busi/` — to apply and extend it to breast-ultrasound
(BUSI). Everything under `busi/`, `tests/`, and the docs/notebook is new.

> Want to see the raw diff yourself? From the repo:
> `git diff eee4881 HEAD --stat` (eee4881 = the authors' code we started from).

---

## 1. What we ADDED — this is our work (the extension)

Nothing here existed in the original repo:

| Added | What it is |
|---|---|
| `busi/config.py` | The control panel — all settings in one place |
| `busi/data.py` | BUSI image/mask loading, split, augmentation, Dataset |
| `busi/model.py` | **2D** Attention U-Net + plain-U-Net baseline + factory |
| `busi/attention_modules.py` | **CBAM** and **scSE** attention blocks (the comparison) |
| `busi/losses.py` | Device-agnostic Dice, **Dice+CE** (default), Focal-Tversky |
| `busi/metrics.py` | Dice / IoU / **specificity** / attention-focus scoring |
| `busi/train.py` | Training loop, evaluation, seeding, device selection |
| `busi/experiment.py` | Multi-seed orchestration, results tables, figures |
| `busi/viz.py` | Mask / attention overlays and comparison figures |
| `tests/` (8 files) | 79 automated tests |
| `project.ipynb` | The Colab/VS Code notebook that runs everything |
| `config`, `requirements.txt`, `pytest.ini`, `splits/split.json` | Setup + reproducible split |
| `README / HANDOFF / WALKTHROUGH / DESIGN / PLAN / CLAUDE.md` | Documentation |

## 2. What we MODIFIED in the authors' files — tiny, mechanical only

We changed **only two of their code files**, and only to run on modern PyTorch —
**no change to their method or math**:

| File | What we changed | Why |
|---|---|---|
| `models/layers/grid_attention_layer.py` | `F.upsample → F.interpolate(align_corners=False)`; `F.sigmoid → torch.sigmoid` | The old calls are removed/deprecated in current PyTorch. Behavior identical. |
| `models/networks_other.py` | `init.kaiming_normal → kaiming_normal_` (and `normal_`, `constant_`, `xavier_normal_`, `orthogonal_`) | Deprecated in-place-init aliases. Same behavior. |

Plus project housekeeping: `.gitignore` and `README.md`.

## 3. What we REUSE **unchanged** (imported, not copied/edited)

These are the authors' pieces our code imports directly — the real link to the
paper:

- **`GridAttentionBlock2D`** (`grid_attention_layer.py`) — the paper's **attention
  gate itself**. This is the heart of the method; we use it verbatim.
- `unetConv2`, `unetUp` (`models/networks/utils.py`) — encoder/decoder conv blocks.
- `init_weights` (`models/networks_other.py`) — weight initialisation.
- `SoftDiceLoss`, `One_Hot` (`models/layers/loss.py`) — we re-implement a
  device-agnostic version in `busi/losses.py`; the original stays as-is.
- `unet_2D` (`models/networks/unet_2D.py`) — the plain U-Net we use as the
  ablation baseline.

Everything else in `models/`, `dataio/`, `utils/` (their 3D networks, SonoNet,
NIfTI loaders, Visdom visualiser, training scripts) is **left untouched and
unused** — it's the original framework we didn't need.

## 4. In one sentence

> We reused the authors' attention-gate implementation intact and wrote a new 2D
> pipeline around it to take Attention U-Net from **3D pancreas CT** to **2D breast
> ultrasound (BUSI)** — adding a baseline + CBAM + scSE comparison, BUSI data
> handling, Dice+CE training, specificity + interpretability metrics, and
> multi-seed experiments. The original repo did none of this.
