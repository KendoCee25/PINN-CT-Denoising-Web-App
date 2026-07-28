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
├── training/       Offline pipeline: phantoms, Radon, noise, U-Net, PINN, train loop
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

8-week plan in `docs/` (proposal §6). Current status: **Week 2 — synthetic data
pipeline & baseline U-Net training.**

Evaluation note: a denoiser is judged against the *untouched noisy input*, not on
its absolute PSNR. `notebooks/train_baseline.ipynb` §4b prints that comparison per
dose level; the pass condition is a positive delta at every dose.
