# CLAUDE.md

Course project: **Attention U-Net applied to BUSI breast-ultrasound lesion
segmentation** (extension of Oktay et al., 2018). Working, tested codebase.

## Read these (they're the source of truth — don't duplicate them)
- **`HANDOFF.md`** — first-time setup: push to your own repo, get the dataset, venv, run tests, run experiments.
- **`WALKTHROUGH.md`** — plain-language tour of all the code, the tests, the results so far, and how to critique the work.
- **`DESIGN.md`** — every locked design decision and why.
- **`PLAN.md`** — the step-by-step build plan with a progress checklist (what's done, what's next).
- **`CHANGES_VS_ORIGINAL.md`** — exactly what we changed vs the original forked repo (what's "ours" vs "theirs").

## Essentials
- All our code is in `busi/`; tests in `tests/`. Everything is driven by `busi/config.py` (switch model/loss/params there).
- Run the tests (offline, no dataset needed):
  `PYTORCH_ENABLE_MPS_FALLBACK=1 python -m pytest -q` (expect all passing).
- Status: code complete through PLAN.md Stage 8; remaining = run the full experiments (Colab GPU) then write the report. See `PLAN.md`.
