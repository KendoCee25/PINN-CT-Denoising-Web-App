// Thin client for the frozen contract in docs/api_contract.md. Every field
// name here mirrors api/schemas.py exactly, so the response can be used as-is.

const BASE_URL = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

async function handle(res) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `request failed with status ${res.status}`)
  }
  return res.json()
}

export function getPhantoms() {
  return fetch(`${BASE_URL}/phantoms`).then(handle)
}

export function denoise(phantomId, doseLevel) {
  return fetch(`${BASE_URL}/denoise`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ phantom_id: phantomId, dose_level: doseLevel }),
  }).then(handle)
}

// Real TCIA low-dose/full-dose cases (v1.1) — same shape as the synthetic
// phantom endpoints above, minus dose_level (a real case has one fixed pair).
export function getRealCases() {
  return fetch(`${BASE_URL}/real_cases`).then(handle)
}

export function denoiseReal(caseId) {
  return fetch(`${BASE_URL}/real_denoise`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ case_id: caseId }),
  }).then(handle)
}
