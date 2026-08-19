export default function PhantomSelector({ phantoms, value, onChange }) {
  const current = phantoms.find((p) => p.id === value)

  return (
    <label className="control">
      <span className="control-label">
        <span className="control-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="18" height="18" rx="3" />
            <circle cx="9" cy="9" r="2" />
            <path d="m21 15-5-5L5 21" />
          </svg>
        </span>
        Phantom
      </span>
      <div className="phantom-row">
        {current?.thumbnail && (
          <img className="phantom-thumb" src={current.thumbnail} alt="" aria-hidden="true" />
        )}
        <select
          value={value ?? ''}
          onChange={(e) => onChange(e.target.value)}
          disabled={phantoms.length === 0}
        >
          {phantoms.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
      </div>
    </label>
  )
}
