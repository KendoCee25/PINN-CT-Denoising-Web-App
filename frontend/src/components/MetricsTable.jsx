function delta(value, baseline, decimals) {
  const d = value - baseline
  const sign = d >= 0 ? '+' : ''
  return `${sign}${d.toFixed(decimals)}`
}

export default function MetricsTable({ result }) {
  if (!result) {
    return (
      <p className="metrics-placeholder">
        Run a comparison above — this table fills in with live PSNR / SSIM numbers for the phantom and
        dose level you picked.
      </p>
    )
  }

  const { metrics, winner } = result
  const rows = [
    { key: 'noisy', label: 'Noisy input', metric: metrics.noisy, baseline: true },
    { key: 'unet', label: 'U-Net', metric: metrics.unet },
    { key: 'pinn', label: 'PINN', metric: metrics.pinn },
  ]

  return (
    <table className="metrics-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>PSNR</th>
          <th>SSIM</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const psnrWin = !row.baseline && winner?.psnr === row.key
          const ssimWin = !row.baseline && winner?.ssim === row.key
          return (
            <tr key={row.key} className={psnrWin || ssimWin ? 'metrics-row-winner' : undefined}>
              <td>{row.label}</td>
              <td>
                {row.metric.psnr.toFixed(2)} <small>dB</small>
                {psnrWin ? ' ★' : ''}
                {!row.baseline && (
                  <span className="metrics-delta">
                    {delta(row.metric.psnr, metrics.noisy.psnr, 1)} dB vs. input
                  </span>
                )}
              </td>
              <td>
                {row.metric.ssim.toFixed(3)}
                {ssimWin ? ' ★' : ''}
                {!row.baseline && (
                  <span className="metrics-delta">
                    {delta(row.metric.ssim, metrics.noisy.ssim, 3)} vs. input
                  </span>
                )}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
