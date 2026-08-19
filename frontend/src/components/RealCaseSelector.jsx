export default function RealCaseSelector({ cases, value, onChange }) {
  const current = cases.find((c) => c.id === value)

  return (
    <label className="control">
      <span className="control-label">
        <span className="control-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M4 14a8 8 0 0 1 8-8V4l4 4-4 4V8a6 6 0 0 0-6 6H4Z" />
            <path d="M20 10a8 8 0 0 1-8 8v2l-4-4 4-4v2a6 6 0 0 0 6-6h2Z" />
          </svg>
        </span>
        Real case (TCIA)
      </span>
      <div className="phantom-row">
        {current?.thumbnail && (
          <img
            className="phantom-thumb phantom-thumb-smooth"
            src={current.thumbnail}
            alt=""
            aria-hidden="true"
          />
        )}
        <select
          value={value ?? ''}
          onChange={(e) => onChange(e.target.value)}
          disabled={cases.length === 0}
        >
          {cases.map((c) => (
            <option key={c.id} value={c.id}>
              {c.label}
            </option>
          ))}
        </select>
      </div>
    </label>
  )
}
