"""FastAPI inference backend — Week 6.

Serves the frozen contract in docs/api_contract.md / api/schemas.py. Both model
checkpoints and the held-out phantom set are loaded once at startup; /denoise
runs a single forward pass through each model and computes metrics server-side
so the frontend never needs ML dependencies (proposal §3.2).
"""

from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
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
    Winner,
)
from training.metrics import psnr, ssim
from training.unet import UNet

MODELS_DIR = Path("models")
DATA_PATH = Path("data/dataset/test.npz")
DOSE_CODE = {"low": 0, "medium": 1, "high": 2}

# Populated once at startup by load_state(); never touched per-request.
state: dict = {"unet": None, "pinn": None, "device": "cpu", "clean": None,
               "noisy": None, "dose": None, "pid": None, "phantom_row": {}}


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_state()
    yield


app = FastAPI(title="PINN vs U-Net CT Denoising", version="1.0", lifespan=lifespan)


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
