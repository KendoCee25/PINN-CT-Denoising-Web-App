# PINN vs. U-Net for Low-Dose CT Image Denoising

An interactive web application that lets a user select a phantom CT image, choose
a simulated dose level, and compare — side by side, in real time — a standard
**U-Net** denoiser against an architecturally identical **Physics-Informed Neural
Network (PINN)**, with live PSNR / SSIM scores.

The working web application is the primary deliverable; the model comparison is
its core interactive content. See `docs/` for the full proposal.

---

## Architecture (three tiers)

1. **Offline training pipeline** (`training/`) — Shepp-Logan phantom generation →
   differentiable Radon forward projection → Poisson noise at 3 dose levels → FBP
   reconstruction → train baseline U-Net (MSE) and PINN (MSE + λ·sinogram-consistency).
   Runs on a **GPU** (Colab/Kaggle — see below). Outputs two `.pt` checkpoints.
2. **FastAPI backend** (`api/`) — loads both checkpoints once at startup, exposes
   `POST /denoise`, computes metrics server-side, returns base64 PNGs + metrics.
3. **React frontend** (`frontend/`) — phantom selector, dose slider, side-by-side
   results panel with metric badges (winner highlighted).

The API contract between backend and frontend is **frozen** in
[`docs/api_contract.md`](docs/api_contract.md).

## Hardware note

Development machine is **CPU-only** (no NVIDIA GPU). Per the proposal's
contingency plan, model **training happens on Google Colab / Kaggle** (free GPU);
trained checkpoints are downloaded to `models/` and run locally for inference
(phantom-scale images are small enough that CPU inference is fast).

## Repository layout

```
PINN/
├── api/            FastAPI inference backend
│   └── schemas.py  Frozen request/response models
├── training/       Offline pipeline: phantoms, Radon, noise, U-Net, train loop
│   ├── pinn.py     Differentiable sinogram-consistency loss (the PINN)
│   └── ablate.py   λ sweep with a λ=0 control arm
├── frontend/       React SPA (scaffolded in Week 7)
├── models/         Trained checkpoints (.pt) — git-ignored
├── data/           Generated phantoms & sinograms — git-ignored
├── notebooks/      Colab/Kaggle training notebooks, experiments
├── tests/          pytest suite (backend endpoint tests, pipeline sanity checks)
└── docs/           Proposal, API contract, figures
```

## Getting started (backend/dev)

```bash
python -m venv .venv
source .venv/Scripts/activate          # Windows Git Bash;  .venv\Scripts\activate on cmd/PS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Run the backend (once implemented, Week 6):

```bash
uvicorn api.main:app --reload
```

## Project plan

8-week plan in `docs/` (proposal §6). Current status: **Week 3 — PINN
sinogram-consistency loss & λ ablation.**

Evaluation note: a denoiser is judged against the *untouched noisy input*, not on
its absolute PSNR. `notebooks/train_baseline.ipynb` §4b prints that comparison per
dose level; the pass condition is a positive delta at every dose.

## Training the models

Both models share `training/train.py` and an identical U-Net backbone — only the
loss differs, which is what isolates the effect of the physics term.

```bash
# Dataset (stores measured sinograms, which the PINN loss needs)
python -m training.make_dataset --n-phantoms 500 --size 128 --device cuda

# Baseline: MSE only
python -m training.train --data-dir data/dataset --epochs 40 --out models/baseline.pt

# PINN: MSE + λ·sinogram-consistency
python -m training.train --data-dir data/dataset --epochs 40 --lam 0.1 \
    --loss-angles 60 --out models/pinn.pt

# λ ablation (proposal §6, Week 3) — includes a λ=0 control arm
python -m training.ablate --data-dir data/dataset --epochs 15 --device cuda
```

The physics term is **normalised** by the target sinogram's mean square, making it
dimensionless. Without that, an MSE over raw line integrals (peaking near 45) sits
four to five orders of magnitude above an MSE over [0,1] pixels, and every λ in the
proposal's range would be physics-dominated. Normalising makes λ a genuine relative
weight. `--loss-angles 60` subsamples the projection for speed, the proposal's
stated mitigation for a slow forward transform.
