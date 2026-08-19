# API Contract — PINN CT Denoising Web App

**Status: FROZEN (Week 1), v1.1.** Changes here require a version bump and must
be agreed before backend/frontend integration in Weeks 6–7. Freezing this
contract is what allows the React frontend and FastAPI backend to be built in
parallel (and the frontend to be developed against a mock backend).

**v1.1** adds `/real_cases` and `/real_denoise` (additive, no breaking change to
the v1.0 endpoints below): a live-app path for the real TCIA LDCT generalisation
check, previously only run offline via `training/eval_real_ldct.py`.

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

## `GET /real_cases` (v1.1)

Returns the held-out real TCIA low-dose/full-dose case set available for
comparison — the live-app counterpart of `/phantoms`, but sourced from
`data/real_ldct/real_ldct.npz` (real clinical acquisitions) instead of the
synthetic phantom pipeline. Populated at backend startup, with the same
degenerate-slice filter used by `training/eval_real_ldct.py` already applied,
so the cases offered here match the pairs behind the dissertation's reported
numbers.

**200 Response**
```json
{
  "cases": [
    { "id": "real_000", "label": "C016 · slice 40", "thumbnail": "data:image/png;base64,..." }
  ]
}
```

| Field       | Type   | Notes                                                    |
|-------------|--------|-----------------------------------------------------------|
| `id`        | string | Stable identifier passed back to `/real_denoise`.        |
| `label`     | string | Patient ID and slice index, for the dropdown.            |
| `thumbnail` | string | Base64 PNG data URI of the real full-dose (clean) slice. |

---

## `POST /real_denoise` (v1.1)

Real-data counterpart of `/denoise`. There is no `dose_level` — a real case is
a single fixed low-dose/full-dose pair, not a simulated multi-level sweep — and
no `phantom_id` echo, since the case is uniquely identified by `case_id`.

**Request body**
```json
{ "case_id": "real_000" }
```

**200 Response** — same `images` / `metrics` / `winner` shape as `/denoise`:
```json
{
  "case_id": "real_000",
  "images": {
    "clean":  "data:image/png;base64,...",
    "noisy":  "data:image/png;base64,...",
    "unet":   "data:image/png;base64,...",
    "pinn":   "data:image/png;base64,..."
  },
  "metrics": {
    "noisy": { "psnr": 23.93, "ssim": 0.854 },
    "unet":  { "psnr": 23.51, "ssim": 0.830 },
    "pinn":  { "psnr": 23.49, "ssim": 0.829 }
  },
  "winner": { "psnr": "unet", "ssim": "unet" }
}
```

**Error responses**

| Status | Meaning                                       |
|--------|------------------------------------------------|
| 404    | `case_id` not found in the real case set.      |
| 422    | Validation error (missing field).              |
| 503    | Model checkpoints or real dataset not loaded.  |

Both trained checkpoints are evaluated **as-is**, with no fine-tuning on real
data — this endpoint is a generalisation check, not a second training path
(Chapter 3 of the dissertation). Because the source download provides
reconstructed images only (no raw projection data), there is no sinogram to
show alongside these results.

---

## Notes for implementers

- Metrics are computed **server-side** against the clean ground truth so the
  frontend never needs ML dependencies.
- Both checkpoints are loaded **once at startup**, never per request.
- Images are phantom-scale (e.g. 256×256 grayscale), so a single JSON response
  per interaction keeps latency low and avoids extra round-trips.
