import { useEffect, useRef, useState } from 'react'

const MODELS = [
  { key: 'unet', label: 'U-Net' },
  { key: 'pinn', label: 'PINN' },
]

function fmtDelta(value, baseline, decimals) {
  const d = value - baseline
  const sign = d >= 0 ? '+' : ''
  return `${sign}${d.toFixed(decimals)} vs. input`
}

export default function SliceCompare({ result, inputLabel, smooth }) {
  const [model, setModel] = useState('pinn')
  const [split, setSplit] = useState(50)
  const stageRef = useRef(null)
  const draggingRef = useRef(false)

  // Sweep the divider once per new result so the difference is obvious, then
  // hand control back to the user for manual dragging.
  useEffect(() => {
    if (!result) return
    let raf
    const start = performance.now()
    const duration = 1300
    const from = 14
    const to = 86

    function tick(now) {
      const t = Math.min(1, (now - start) / duration)
      const eased = 0.5 - Math.cos(t * Math.PI) / 2
      setSplit(from + (to - from) * eased)
      if (t < 1) {
        raf = requestAnimationFrame(tick)
      } else {
        setSplit(50)
      }
    }

    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [result])

  function updateFromPointer(clientX) {
    const stage = stageRef.current
    if (!stage) return
    const rect = stage.getBoundingClientRect()
    const pct = ((clientX - rect.left) / rect.width) * 100
    setSplit(Math.min(97, Math.max(3, pct)))
  }

  function handlePointerDown(e) {
    draggingRef.current = true
    updateFromPointer(e.clientX)
  }

  function handlePointerMove(e) {
    if (!draggingRef.current) return
    updateFromPointer(e.clientX)
  }

  function stopDrag() {
    draggingRef.current = false
  }

  if (!result) {
    return (
      <div className="console slice-compare">
        <div className="console-header">
          <span className="mono-label">Slice compare</span>
        </div>
        <div className="console-body">
          <p className="slice-placeholder mono-label">Loading a comparison…</p>
        </div>
      </div>
    )
  }

  const { images, metrics, winner } = result
  const modelInfo = MODELS.find((m) => m.key === model)
  const rightMetric = metrics[model]
  const bothWin = winner?.psnr === model && winner?.ssim === model
  const noWin = winner?.psnr !== model && winner?.ssim !== model

  return (
    <div className="console slice-compare">
      <div className="console-header">
        <span className="mono-label">Slice compare</span>
        <span className="live-badge">
          <span className="live-dot" aria-hidden="true" />
          Live comparison
        </span>
      </div>
      <div className="console-body">
        <div
          className={smooth ? 'slice-stage slice-stage-smooth' : 'slice-stage'}
          ref={stageRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={stopDrag}
          onPointerLeave={stopDrag}
        >
          <img className="slice-img slice-img-base" src={images.noisy} alt="Noisy input" draggable={false} />
          <div className="slice-img-clip" style={{ clipPath: `inset(0 0 0 ${split}%)` }}>
            <img className="slice-img" src={images[model]} alt={`${modelInfo.label} reconstruction`} draggable={false} />
          </div>
          <div className="slice-handle" style={{ left: `${split}%` }}>
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="m9 6-6 6 6 6M15 6l6 6-6 6" />
            </svg>
          </div>
          <span className="slice-tag slice-tag-left">{inputLabel} &mdash; noisy input</span>
          <span className="slice-tag slice-tag-right">{modelInfo.label} reconstruction</span>
        </div>

        <div className="slice-console">
          <div className="slice-metric">
            <div className="mono-label">PSNR</div>
            <div className="slice-metric-value">
              {rightMetric.psnr.toFixed(1)} <small>dB</small>
            </div>
            <div className="slice-metric-delta">{fmtDelta(rightMetric.psnr, metrics.noisy.psnr, 1)}</div>
          </div>
          <div className="slice-metric">
            <div className="mono-label">SSIM</div>
            <div className="slice-metric-value">{rightMetric.ssim.toFixed(3)}</div>
            <div className="slice-metric-delta">{fmtDelta(rightMetric.ssim, metrics.noisy.ssim, 3)}</div>
          </div>
          <div className="slice-metric">
            <div className="mono-label">Winner</div>
            <div className={bothWin ? 'slice-metric-value slice-metric-winner' : 'slice-metric-value'}>
              {bothWin ? modelInfo.label : noWin ? 'other model' : 'split'}
            </div>
            <div className="slice-metric-delta">by PSNR &amp; SSIM</div>
          </div>
        </div>

        <div className="toggle-section">
          <span className="mono-label">Model</span>
          <div className="switch">
            {MODELS.map((m) => (
              <button
                key={m.key}
                type="button"
                className={m.key === model ? 'active' : undefined}
                onClick={() => setModel(m.key)}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
