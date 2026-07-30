"""Endpoint tests for the FastAPI backend (Week 6).

Runs against the real checkpoints/dataset already on disk via TestClient, which
drives the app's lifespan (startup) exactly as uvicorn would. If models/data
aren't present, the loaded-state tests are skipped rather than failed, since
that reflects missing local artifacts, not a backend bug.
"""

from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import DATA_PATH, MODELS_DIR, app

pytestmark = pytest.mark.skipif(
    not ((MODELS_DIR / "baseline.pt").exists() and (MODELS_DIR / "pinn.pt").exists()
         and DATA_PATH.exists()),
    reason="requires trained checkpoints and the generated test dataset",
)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_models_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["unet_loaded"] is True
    assert body["pinn_loaded"] is True
    assert body["device"] in ("cpu", "cuda")


def test_phantoms_list_nonempty_with_valid_thumbnails(client):
    r = client.get("/phantoms")
    assert r.status_code == 200
    items = r.json()["phantoms"]
    assert len(items) > 0
    first = items[0]
    assert first["id"].startswith("phantom_")
    assert first["thumbnail"].startswith("data:image/png;base64,")
    # Thumbnail must decode to a real PNG.
    png_bytes = base64.b64decode(first["thumbnail"].split(",", 1)[1])
    img = Image.open(io.BytesIO(png_bytes))
    assert img.format == "PNG"
    assert img.mode == "L"  # grayscale, matching the training pipeline


def test_denoise_returns_images_and_metrics_in_range(client):
    phantom_id = client.get("/phantoms").json()["phantoms"][0]["id"]
    r = client.post("/denoise", json={"phantom_id": phantom_id, "dose_level": "low"})
    assert r.status_code == 200
    body = r.json()

    assert body["phantom_id"] == phantom_id
    assert body["dose_level"] == "low"

    for key in ("clean", "noisy", "unet", "pinn"):
        assert body["images"][key].startswith("data:image/png;base64,")

    for key in ("noisy", "unet", "pinn"):
        m = body["metrics"][key]
        assert m["psnr"] > 0
        assert 0.0 <= m["ssim"] <= 1.0

    assert body["winner"]["psnr"] in ("unet", "pinn")
    assert body["winner"]["ssim"] in ("unet", "pinn")


def test_denoise_output_image_shape_matches_dataset(client):
    phantom_id = client.get("/phantoms").json()["phantoms"][0]["id"]
    r = client.post("/denoise", json={"phantom_id": phantom_id, "dose_level": "medium"})
    body = r.json()

    clean_bytes = base64.b64decode(body["images"]["clean"].split(",", 1)[1])
    unet_bytes = base64.b64decode(body["images"]["unet"].split(",", 1)[1])
    clean_img = Image.open(io.BytesIO(clean_bytes))
    unet_img = Image.open(io.BytesIO(unet_bytes))
    assert clean_img.size == unet_img.size


def test_denoise_unknown_phantom_is_404(client):
    r = client.post("/denoise", json={"phantom_id": "phantom_999999", "dose_level": "low"})
    assert r.status_code == 404


def test_denoise_invalid_dose_level_is_422(client):
    phantom_id = client.get("/phantoms").json()["phantoms"][0]["id"]
    r = client.post("/denoise", json={"phantom_id": phantom_id, "dose_level": "ultra"})
    assert r.status_code == 422


def test_denoise_missing_field_is_422(client):
    r = client.post("/denoise", json={"dose_level": "low"})
    assert r.status_code == 422
