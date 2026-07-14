"""Pydantic schemas — the FROZEN API contract in code.

These mirror docs/api_contract.md. Treat both as a single source of truth: if one
changes, the other must change with it. Frozen in Week 1 to allow the frontend to
be built against a mock backend in parallel with the real backend.
"""

from enum import Enum

from pydantic import BaseModel, Field


class DoseLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


# --- GET /health 
class HealthResponse(BaseModel):
    status: str = "ok"
    unet_loaded: bool
    pinn_loaded: bool
    device: str


# --- GET /phantoms 
class PhantomInfo(BaseModel):
    id: str
    label: str
    thumbnail: str = Field(..., description="Base64 PNG data URI of clean ground truth")


class PhantomListResponse(BaseModel):
    phantoms: list[PhantomInfo]


# --- POST /denoise 
class DenoiseRequest(BaseModel):
    phantom_id: str
    dose_level: DoseLevel


class Images(BaseModel):
    clean: str
    noisy: str
    unet: str
    pinn: str


class Metric(BaseModel):
    psnr: float
    ssim: float


class Metrics(BaseModel):
    noisy: Metric
    unet: Metric
    pinn: Metric


class Winner(BaseModel):
    psnr: str  # "unet" | "pinn"
    ssim: str  # "unet" | "pinn"


class DenoiseResponse(BaseModel):
    phantom_id: str
    dose_level: DoseLevel
    images: Images
    metrics: Metrics
    winner: Winner
