function TrophyIcon() {
  return (
    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2.5">
      <path d="M8 21h8M12 17v4M7 4h10v4a5 5 0 0 1-10 0V4Z" />
      <path d="M7 5H4a3 3 0 0 0 3 5M17 5h3a3 3 0 0 1-3 5" />
    </svg>
  )
}

function Badges({ metric, model, winner }) {
  const psnrWins = winner && winner.psnr === model
  const ssimWins = winner && winner.ssim === model
  return (
    <div className="badge-row">
      <span className={psnrWins ? 'badge badge-winner' : 'badge'}>
        PSNR {metric.psnr.toFixed(2)} dB{psnrWins ? ' ★' : ''}
      </span>
      <span className={ssimWins ? 'badge badge-winner' : 'badge'}>
        SSIM {metric.ssim.toFixed(3)}{ssimWins ? ' ★' : ''}
      </span>
    </div>
  )
}

function Panel({ title, src, tag, highlight, children }) {
  return (
    <figure className={highlight ? 'panel panel-winner' : 'panel'}>
      <div className="panel-media">
        <img src={src} alt={title} />
        {tag && <span className="panel-tag">{tag}</span>}
        {highlight && (
          <span className="panel-crown" aria-hidden="true">
            <TrophyIcon />
          </span>
        )}
      </div>
      <figcaption>
        <div className="panel-title">{title}</div>
        {children}
      </figcaption>
    </figure>
  )
}

export default function ResultsPanel({ result, smooth }) {
  if (!result) return null
  const { images, metrics, winner } = result
  const pinnHighlight = winner && (winner.psnr === 'pinn' || winner.ssim === 'pinn')
  const unetHighlight = winner && (winner.psnr === 'unet' || winner.ssim === 'unet')

  return (
    <div className={smooth ? 'results-panel results-panel-smooth' : 'results-panel'}>
      <Panel title="Noisy input" src={images.noisy} tag="Input">
        <div className="badge-row">
          <span className="badge">PSNR {metrics.noisy.psnr.toFixed(2)} dB</span>
          <span className="badge">SSIM {metrics.noisy.ssim.toFixed(3)}</span>
        </div>
      </Panel>
      <Panel title="U-Net output" src={images.unet} tag="Model" highlight={unetHighlight}>
        <Badges metric={metrics.unet} model="unet" winner={winner} />
      </Panel>
      <Panel title="PINN output" src={images.pinn} tag="Model" highlight={pinnHighlight}>
        <Badges metric={metrics.pinn} model="pinn" winner={winner} />
      </Panel>
      <Panel title="Clean (ground truth)" src={images.clean} tag="Reference">
        <div className="badge-row">
          <span className="badge badge-reference">reference</span>
        </div>
      </Panel>
    </div>
  )
}
