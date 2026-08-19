const DOSE_LEVELS = ['low', 'medium', 'high']

const DOSE_HINTS = {
  low: 'Heaviest noise',
  medium: 'Moderate noise',
  high: 'Lightest noise',
}

export default function DoseSlider({ value, onChange }) {
  const index = DOSE_LEVELS.indexOf(value)
  const fill = `${(index / (DOSE_LEVELS.length - 1)) * 100}%`

  return (
    <label className="control">
      <span className="control-label">
        <span className="control-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 17v-3a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v3" />
            <path d="M6 17v2M12 17v2M18 17v2" />
            <path d="M6 12V7a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v5" />
          </svg>
        </span>
        Dose level
        <span className="control-value">{value}</span>
      </span>
      <input
        type="range"
        min={0}
        max={2}
        step={1}
        value={index}
        style={{ '--fill': fill }}
        onChange={(e) => onChange(DOSE_LEVELS[Number(e.target.value)])}
        aria-label="Dose level"
      />
      <div className="slider-ticks">
        {DOSE_LEVELS.map((d) => (
          <span key={d} className={d === value ? 'tick active' : 'tick'}>
            <span className="tick-label">{d}</span>
            <span className="tick-hint">{DOSE_HINTS[d]}</span>
          </span>
        ))}
      </div>
    </label>
  )
}

export { DOSE_LEVELS }
