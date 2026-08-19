#FastAPI inference backend — Week 6.

"""Serves the frozen contract in docs/api_contract.md / api/schemas.py. Both model
checkpoints and the held-out phantom set are loaded once at startup; /denoise
runs a single forward pass through each model and computes metrics server-side
so the frontend never needs ML dependencies (proposal §3.2).
"""

from __future__ import annotations

import base64
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api.schemas import (
    DenoiseRequest,
    DenoiseResponse,
    DoseLevel,
    HealthResponse,
    Images,
    Metric,
    Metrics,
    PhantomInfo,
    PhantomListResponse,
    RealCaseInfo,
    RealCaseListResponse,
    RealDenoiseRequest,
    RealDenoiseResponse,
    Winner,
)
from training.metrics import psnr, ssim
from training.unet import UNet

MODELS_DIR = Path("models")
DATA_PATH = Path("data/dataset/test.npz")
REAL_DATA_PATH = Path("data/real_ldct/real_ldct.npz")
REAL_MANIFEST_PATH = Path("data/real_ldct/manifest.json")
DOSE_CODE = {"low": 0, "medium": 1, "high": 2}
# Degenerate (near pure-air) slices where full-dose and low-dose are pixel-
# identical after windowing — same threshold as training/eval_real_ldct.py, so
# the live app and the offline generalisation numbers in the dissertation
# refer to exactly the same cases.
REAL_DEGENERATE_STD = 1e-4

# Populated once at startup by load_state(); never touched per-request.
state: dict = {"unet": None, "pinn": None, "device": "cpu", "clean": None,
               "noisy": None, "dose": None, "pid": None, "phantom_row": {},
               "real_clean": None, "real_noisy": None, "real_case_row": {},
               "real_patient_id": None, "real_slice_idx": None, "real_patients": []}


def _load_model(path: Path, device: str) -> UNet | None:
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = UNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _encode_png(arr: np.ndarray) -> str:
    """(H, W) float array in [0, 1] -> base64 PNG data URI (contract's image format)."""
    img = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img, mode="L").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def load_state() -> None:
    """(Re)populate `state` from disk. Split out from the lifespan hook so tests
    can call it directly against a temp models/data dir without booting the app."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["device"] = device
    state["unet"] = _load_model(MODELS_DIR / "baseline.pt", device)
    state["pinn"] = _load_model(MODELS_DIR / "pinn.pt", device)

    if DATA_PATH.exists():
        data = np.load(DATA_PATH)
        state["clean"] = data["clean"]
        state["noisy"] = data["noisy"]
        state["dose"] = data["dose"]
        state["pid"] = data["pid"]
        # One representative row per distinct phantom id (clean image is the
        # same across dose variants) for the /phantoms listing and lookups.
        row_for_pid: dict[int, int] = {}
        for idx, pid in enumerate(state["pid"]):
            row_for_pid.setdefault(int(pid), idx)
        state["phantom_row"] = row_for_pid
    else:
        state["clean"] = None
        state["phantom_row"] = {}

    if REAL_DATA_PATH.exists():
        real = np.load(REAL_DATA_PATH)
        real_clean, real_noisy = real["clean"], real["noisy"]
        real_patient_id, real_slice_idx = real["patient_id"], real["slice_idx"]

        # Same filter as training/eval_real_ldct.py, so the cases offered here
        # match the pairs behind the numbers reported in the dissertation.
        keep = real_clean.std(axis=(1, 2)) > REAL_DEGENERATE_STD
        real_clean, real_noisy = real_clean[keep], real_noisy[keep]
        real_patient_id, real_slice_idx = real_patient_id[keep], real_slice_idx[keep]

        patients = []
        if REAL_MANIFEST_PATH.exists():
            patients = json.loads(REAL_MANIFEST_PATH.read_text()).get("patients", [])

        state["real_clean"] = real_clean
        state["real_noisy"] = real_noisy
        state["real_patient_id"] = real_patient_id
        state["real_slice_idx"] = real_slice_idx
        state["real_patients"] = patients
        state["real_case_row"] = {f"real_{row:03d}": row for row in range(len(real_clean))}
    else:
        state["real_clean"] = None
        state["real_case_row"] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_state()
    yield


app = FastAPI(title="PINN vs U-Net CT Denoising", version="1.0", lifespan=lifespan)

# The React dev server (Week 7) runs on a different origin/port than the API,
# so the browser needs this to allow the fetch calls. In production the
# deployed frontend's origin is supplied via FRONTEND_ORIGIN (Render env var).
origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
if extra := os.getenv("FRONTEND_ORIGIN"):
    origins.append(extra)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        unet_loaded=state["unet"] is not None,
        pinn_loaded=state["pinn"] is not None,
        device=state["device"],
    )


@app.get("/phantoms", response_model=PhantomListResponse)
def phantoms() -> PhantomListResponse:
    items = [
        PhantomInfo(
            id=f"phantom_{pid:03d}",
            label=f"Phantom {pid:03d}",
            thumbnail=_encode_png(state["clean"][row]),
        )
        for pid, row in sorted(state["phantom_row"].items())
    ]
    return PhantomListResponse(phantoms=items)


def _find_row(phantom_id: str, dose_level: DoseLevel) -> int:
    prefix = "phantom_"
    if not phantom_id.startswith(prefix) or not phantom_id[len(prefix):].isdigit():
        raise HTTPException(status_code=404, detail=f"phantom_id {phantom_id!r} not found")
    pid = int(phantom_id[len(prefix):])
    if pid not in state["phantom_row"]:
        raise HTTPException(status_code=404, detail=f"phantom_id {phantom_id!r} not found")

    code = DOSE_CODE[dose_level.value]
    matches = np.where((state["pid"] == pid) & (state["dose"] == code))[0]
    if len(matches) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"no {dose_level.value}-dose variant stored for {phantom_id!r}",
        )
    return int(matches[0])


@app.post("/denoise", response_model=DenoiseResponse)
def denoise(req: DenoiseRequest) -> DenoiseResponse:
    if state["unet"] is None or state["pinn"] is None:
        raise HTTPException(status_code=503, detail="model checkpoints not loaded")
    if state["clean"] is None:
        raise HTTPException(status_code=503, detail="phantom dataset not loaded")

    row = _find_row(req.phantom_id, req.dose_level)
    device = state["device"]
    clean = state["clean"][row]
    noisy = state["noisy"][row]

    x = torch.from_numpy(noisy)[None, None].to(device)
    with torch.no_grad():
        unet_out = state["unet"](x).clamp(0.0, 1.0).cpu().numpy()[0, 0]
        pinn_out = state["pinn"](x).clamp(0.0, 1.0).cpu().numpy()[0, 0]

    metrics = Metrics(
        noisy=Metric(psnr=psnr(noisy, clean), ssim=ssim(noisy, clean)),
        unet=Metric(psnr=psnr(unet_out, clean), ssim=ssim(unet_out, clean)),
        pinn=Metric(psnr=psnr(pinn_out, clean), ssim=ssim(pinn_out, clean)),
    )
    winner = Winner(
        psnr="pinn" if metrics.pinn.psnr > metrics.unet.psnr else "unet",
        ssim="pinn" if metrics.pinn.ssim > metrics.unet.ssim else "unet",
    )
    return DenoiseResponse(
        phantom_id=req.phantom_id,
        dose_level=req.dose_level,
        images=Images(
            clean=_encode_png(clean),
            noisy=_encode_png(noisy),
            unet=_encode_png(unet_out),
            pinn=_encode_png(pinn_out),
        ),
        metrics=metrics,
        winner=winner,
    )


@app.get("/real_cases", response_model=RealCaseListResponse)
def real_cases() -> RealCaseListResponse:
    patients = state["real_patients"]
    items = []
    for case_id, row in sorted(state["real_case_row"].items(), key=lambda kv: kv[1]):
        pid = int(state["real_patient_id"][row])
        sidx = int(state["real_slice_idx"][row])
        name = patients[pid] if pid < len(patients) else f"patient{pid}"
        items.append(RealCaseInfo(
            id=case_id,
            label=f"{name} · slice {sidx}",
            thumbnail=_encode_png(state["real_clean"][row]),
        ))
    return RealCaseListResponse(cases=items)


@app.post("/real_denoise", response_model=RealDenoiseResponse)
def real_denoise(req: RealDenoiseRequest) -> RealDenoiseResponse:
    if state["unet"] is None or state["pinn"] is None:
        raise HTTPException(status_code=503, detail="model checkpoints not loaded")
    if state["real_clean"] is None:
        raise HTTPException(status_code=503, detail="real LDCT dataset not loaded")

    row = state["real_case_row"].get(req.case_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"case_id {req.case_id!r} not found")

    device = state["device"]
    clean = state["real_clean"][row]
    noisy = state["real_noisy"][row]

    x = torch.from_numpy(noisy)[None, None].to(device)
    with torch.no_grad():
        unet_out = state["unet"](x).clamp(0.0, 1.0).cpu().numpy()[0, 0]
        pinn_out = state["pinn"](x).clamp(0.0, 1.0).cpu().numpy()[0, 0]

    metrics = Metrics(
        noisy=Metric(psnr=psnr(noisy, clean), ssim=ssim(noisy, clean)),
        unet=Metric(psnr=psnr(unet_out, clean), ssim=ssim(unet_out, clean)),
        pinn=Metric(psnr=psnr(pinn_out, clean), ssim=ssim(pinn_out, clean)),
    )
    winner = Winner(
        psnr="pinn" if metrics.pinn.psnr > metrics.unet.psnr else "unet",
        ssim="pinn" if metrics.pinn.ssim > metrics.unet.ssim else "unet",
    )
    return RealDenoiseResponse(
        case_id=req.case_id,
        images=Images(
            clean=_encode_png(clean),
            noisy=_encode_png(noisy),
            unet=_encode_png(unet_out),
            pinn=_encode_png(pinn_out),
        ),
        metrics=metrics,
        winner=winner,
    )
