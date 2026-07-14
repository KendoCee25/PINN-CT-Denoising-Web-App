# API Contract — PINN CT Denoising Web App

**Status: FROZEN (Week 1).** Changes here require a version bump and must be
agreed before backend/frontend integration in Weeks 6–7. Freezing this contract
is what allows the React frontend and FastAPI backend to be built in parallel
(and the frontend to be developed against a mock backend).

Base URL (local dev): `http://localhost:8000`

---

## `GET /health`

Liveness/readiness check. Confirms both model checkpoints are loaded.

**200 Response**
```json
{
  "status": "ok",
  "unet_loaded": true,
  "pinn_loaded": true,
  "device": "cpu"
}
```

---

## `GET /phantoms`

Returns the held-out phantom set available for comparison. Populated from the
pre-generated test dataset at backend startup.

**200 Response**
```json
{
  "phantoms": [
    { "id": "phantom_000", "label": "Phantom 000", "thumbnail": "data:image/png;base64,..." },
    { "id": "phantom_001", "label": "Phantom 001", "thumbnail": "data:image/png;base64,..." }
  ]
}
```

| Field       | Type   | Notes                                              |
|-------------|--------|----------------------------------------------------|
| `id`        | string | Stable identifier passed back to `/denoise`.       |
| `label`     | string | Human-readable name for the dropdown.              |
| `thumbnail` | string | Base64 PNG data URI of the clean ground truth.     |

---

## `POST /denoise`

The single core inference endpoint. Runs the noisy input through **both** models
in one forward pass and returns images + metrics.

**Request body**
```json
{
  "phantom_id": "phantom_000",
  "dose_level": "low"
}
```

| Field        | Type   | Constraints                          |
|--------------|--------|--------------------------------------|
| `phantom_id` | string | Must match an `id` from `/phantoms`. |
| `dose_level` | string | One of `"low"`, `"medium"`, `"high"`. |

**200 Response**
```json
{
  "phantom_id": "phantom_000",
  "dose_level": "low",
  "images": {
    "clean":  "data:image/png;base64,...",
    "noisy":  "data:image/png;base64,...",
    "unet":   "data:image/png;base64,...",
    "pinn":   "data:image/png;base64,..."
  },
  "metrics": {
    "noisy": { "psnr": 22.14, "ssim": 0.612 },
    "unet":  { "psnr": 31.08, "ssim": 0.884 },
    "pinn":  { "psnr": 33.47, "ssim": 0.951 }
  },
  "winner": { "psnr": "pinn", "ssim": "pinn" }
}
```

| Field              | Type   | Notes                                                       |
|--------------------|--------|-------------------------------------------------------------|
| `images.*`         | string | Base64 PNG data URIs. `clean` = ground truth reference.     |
| `metrics.<m>.psnr` | number | Peak signal-to-noise ratio (dB) vs. clean ground truth.     |
| `metrics.<m>.ssim` | number | Structural similarity in `[0, 1]` vs. clean ground truth.   |
| `winner.psnr`      | string | `"unet"` or `"pinn"` — higher PSNR. Drives UI highlight.     |
| `winner.ssim`      | string | `"unet"` or `"pinn"` — higher SSIM. Drives UI highlight.     |

**Error responses**

| Status | Meaning                                             |
|--------|-----------------------------------------------------|
| 404    | `phantom_id` not found in the held-out set.         |
| 422    | Validation error (bad `dose_level`, missing field). |
| 503    | Model checkpoints not loaded.                       |

---

## Notes for implementers

- Metrics are computed **server-side** against the clean ground truth so the
  frontend never needs ML dependencies.
- Both checkpoints are loaded **once at startup**, never per request.
- Images are phantom-scale (e.g. 256×256 grayscale), so a single JSON response
  per interaction keeps latency low and avoids extra round-trips.
