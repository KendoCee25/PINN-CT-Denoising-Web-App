import { useEffect, useState } from 'react'
import './App.css'
import { getPhantoms, denoise, getRealCases, denoiseReal } from './api'
import PhantomSelector from './components/PhantomSelector'
import DoseSlider, { DOSE_LEVELS } from './components/DoseSlider'
import RealCaseSelector from './components/RealCaseSelector'
import ResultsPanel from './components/ResultsPanel'
import MetricsTable from './components/MetricsTable'
import SliceCompare from './components/SliceCompare'

const MODES = [
  { key: 'synthetic', label: 'Synthetic Phantoms' },
  { key: 'real', label: 'Real Clinical (TCIA)' },
]

const PIPELINE_STEPS = [
  'Generate Shepp-Logan phantoms',
  'Differentiable Radon forward projection',
  'Poisson noise at 3 dose levels',
  'Filtered back-projection reconstruction',
  'Train U-Net and PINN on the noisy/clean pairs',
]

const WHY_IT_MATTERS = [
  {
    num: '01',
    title: 'Does the physics term help?',
    body: 'Both models run on the same phantom slice in one pass, so the sinogram-consistency term in the PINN loss shows up as a visible difference in the reconstruction — not just a table row.',
  },
  {
    num: '02',
    title: 'Where does it help most?',
    body: 'Sweep dose level from low to high in the Compare panel and watch PSNR/SSIM shift — the winner is not fixed. Noise regime and phantom structure both change which model comes out ahead.',
  },
  {
    num: '03',
    title: 'Does it hold on real data?',
    body: 'Both checkpoints are trained entirely on this synthetic pipeline. Switch the Comparison setup below to "Real Clinical (TCIA)" to run them live against real low-dose/full-dose acquisitions they never trained on.',
  },
]

function scrollToId(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

export default function App() {
  const [mode, setMode] = useState('synthetic')
  const [phantoms, setPhantoms] = useState([])
  const [phantomId, setPhantomId] = useState(null)
  const [dose, setDose] = useState(DOSE_LEVELS[0])
  const [realCases, setRealCases] = useState([])
  const [realCaseId, setRealCaseId] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const currentPhantom = phantoms.find((p) => p.id === phantomId)
  const currentRealCase = realCases.find((c) => c.id === realCaseId)
  const inputLabel = mode === 'synthetic' ? `${dose} dose` : 'real low-dose (TCIA)'

  // Load the held-out phantom set and the real TCIA case set once on mount.
  useEffect(() => {
    getPhantoms()
      .then((data) => {
        setPhantoms(data.phantoms)
        if (data.phantoms.length > 0) setPhantomId(data.phantoms[0].id)
      })
      .catch((err) => setError(err.message))
    getRealCases()
      .then((data) => {
        setRealCases(data.cases)
        if (data.cases.length > 0) setRealCaseId(data.cases[0].id)
      })
      .catch((err) => setError(err.message))
  }, [])

  // Re-run inference whenever the user changes mode, phantom, dose, or real case.
  useEffect(() => {
    if (mode === 'synthetic' && !phantomId) return
    if (mode === 'real' && !realCaseId) return
    let cancelled = false
    setLoading(true)
    setError(null)
    const request = mode === 'synthetic' ? denoise(phantomId, dose) : denoiseReal(realCaseId)
    request
      .then((data) => {
        if (!cancelled) setResult(data)
      })
      .catch((err) => {
        if (!cancelled) setError(err.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [mode, phantomId, dose, realCaseId])

  function pickPhantom(id) {
    setMode('synthetic')
    setPhantomId(id)
    scrollToId('compare')
  }

  return (
    <div className="app">
      <nav className="nav">
        <a href="#top" className="logo" onClick={(e) => { e.preventDefault(); scrollToId('top') }}>
          <span className="logo-dot" aria-hidden="true" />
          PINN&middot;CT
        </a>
        <div className="nav-links">
          <a href="#compare" onClick={(e) => { e.preventDefault(); scrollToId('compare') }}>Compare</a>
          <a href="#method" onClick={(e) => { e.preventDefault(); scrollToId('method') }}>Method</a>
          <a href="#metrics" onClick={(e) => { e.preventDefault(); scrollToId('metrics') }}>Metrics</a>
          <a href="#dataset" onClick={(e) => { e.preventDefault(); scrollToId('dataset') }}>Dataset</a>
        </div>
      </nav>

      <header className="hero" id="top">
        <div className="eyebrow">Low-dose CT &middot; Denoising comparison</div>
        <h1>PINN vs U-Net — Low-Dose CT Denoising</h1>
        <p className="subtitle">
          Compare a physics-informed denoiser against a standard U-Net, side by side, in real time.
        </p>
      </header>

      <section id="compare" className="anchor-section">
        <div className="console" aria-label="Comparison controls">
          <div className="console-header">
            <span className="mono-label">Comparison setup</span>
            <span className="live-badge">
              <span className="live-dot" aria-hidden="true" />
              Live local inference
            </span>
          </div>
          <div className="console-body">
            <div className="toggle-section" style={{ marginBottom: 20 }}>
              <span className="mono-label">Data source</span>
              <div className="switch">
                {MODES.map((m) => (
                  <button
                    key={m.key}
                    type="button"
                    className={m.key === mode ? 'active' : undefined}
                    onClick={() => setMode(m.key)}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="controls-body">
              {mode === 'synthetic' ? (
                <>
                  <PhantomSelector phantoms={phantoms} value={phantomId} onChange={setPhantomId} />
                  <DoseSlider value={dose} onChange={setDose} />
                </>
              ) : (
                <RealCaseSelector cases={realCases} value={realCaseId} onChange={setRealCaseId} />
              )}
            </div>
          </div>
        </div>

        {error && (
          <p className="error" role="alert">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="9" />
              <line x1="12" y1="8" x2="12" y2="13" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            {error}
          </p>
        )}

        <SliceCompare
          result={loading ? null : result}
          inputLabel={inputLabel}
          smooth={mode === 'real'}
        />

        <div className="console" aria-label="Results">
          <div className="console-header">
            <span className="mono-label">
              Slice output {mode === 'synthetic'
                ? (currentPhantom ? `· ${currentPhantom.label}` : '')
                : (currentRealCase ? `· ${currentRealCase.label}` : '')}
            </span>
            {loading ? (
              <span className="live-badge live-badge-loading">
                <span className="spinner" aria-hidden="true" />
                Running inference…
              </span>
            ) : (
              <span className="mono-label">
                {mode === 'synthetic' ? `Dose: ${dose}` : 'Source: TCIA (real)'}
              </span>
            )}
          </div>
          <div className="console-body">
            <ResultsPanel result={loading ? null : result} smooth={mode === 'real'} />
          </div>
        </div>
      </section>

      <section id="method" className="section anchor-section">
        <div className="section-head">
          <h2 className="section-title">How the comparison is built</h2>
          <p className="section-note mono-label">Same architecture, different loss function</p>
        </div>
        <ol className="method-steps">
          {PIPELINE_STEPS.map((step, i) => (
            <li key={step}>
              <span className="mono-label step-num">{String(i + 1).padStart(2, '0')}</span>
              {step}
            </li>
          ))}
        </ol>
        <div className="cards">
          {WHY_IT_MATTERS.map((card) => (
            <div className="card" key={card.num}>
              <div className="card-num mono-label">{card.num}</div>
              <h3>{card.title}</h3>
              <p>{card.body}</p>
            </div>
          ))}
        </div>
      </section>

      <section id="metrics" className="section anchor-section">
        <div className="section-head">
          <h2 className="section-title">What PSNR and SSIM mean here</h2>
          <p className="section-note mono-label">Computed server-side vs. the clean phantom</p>
        </div>
        <div className="metrics-explainer">
          <div className="metrics-term">
            <h3>PSNR (dB)</h3>
            <p>Peak signal-to-noise ratio against the clean ground truth. Higher is better; a few dB of gain is a visibly cleaner slice.</p>
          </div>
          <div className="metrics-term">
            <h3>SSIM (0–1)</h3>
            <p>Structural similarity — how well edges and texture are preserved, not just pixel error. Higher is better.</p>
          </div>
        </div>
        <MetricsTable result={loading ? null : result} />
      </section>

      <section id="dataset" className="section anchor-section">
        <div className="section-head">
          <h2 className="section-title">Held-out phantom set</h2>
          <p className="section-note mono-label">
            {phantoms.length || '—'} phantoms &middot; click one to compare
          </p>
        </div>
        <div className="phantom-grid">
          {phantoms.map((p) => (
            <button
              key={p.id}
              type="button"
              className={p.id === phantomId ? 'phantom-card phantom-card-active' : 'phantom-card'}
              onClick={() => pickPhantom(p.id)}
            >
              <img src={p.thumbnail} alt="" />
              <span>{p.label}</span>
            </button>
          ))}
        </div>
        <p className="dataset-note">
          Both checkpoints are trained entirely on this synthetic Shepp-Logan set. Switch
          "Data source" to Real Clinical (TCIA) above to run the same two checkpoints — no
          fine-tuning — against real low-dose/full-dose acquisitions from the TCIA
          LDCT-and-Projection-data collection as a live generalisation check.
        </p>
      </section>

      <footer className="footer">
        <span className="mono-label">PINN vs U-Net &middot; MSc Dissertation</span>
        <span className="mono-label">FastAPI + PyTorch + React/Vite</span>
      </footer>
    </div>
  )
}
