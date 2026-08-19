import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const PHANTOMS = {
  phantoms: [
    { id: 'phantom_002', label: 'Phantom 002', thumbnail: 'data:image/png;base64,AAA' },
    { id: 'phantom_005', label: 'Phantom 005', thumbnail: 'data:image/png;base64,BBB' },
  ],
}

const REAL_CASES = {
  cases: [
    { id: 'real_000', label: 'C016 · slice 40', thumbnail: 'data:image/png;base64,RRR' },
  ],
}

function denoiseResponse(phantomId, doseLevel) {
  return {
    phantom_id: phantomId,
    dose_level: doseLevel,
    images: {
      clean: 'data:image/png;base64,clean',
      noisy: 'data:image/png;base64,noisy',
      unet: 'data:image/png;base64,unet',
      pinn: 'data:image/png;base64,pinn',
    },
    metrics: {
      noisy: { psnr: 27.9, ssim: 0.48 },
      unet: { psnr: 38.3, ssim: 0.964 },
      pinn: { psnr: 38.2, ssim: 0.96 },
    },
    winner: { psnr: 'unet', ssim: 'unet' },
  }
}

function realDenoiseResponse(caseId) {
  return {
    case_id: caseId,
    images: {
      clean: 'data:image/png;base64,rclean',
      noisy: 'data:image/png;base64,rnoisy',
      unet: 'data:image/png;base64,runet',
      pinn: 'data:image/png;base64,rpinn',
    },
    metrics: {
      noisy: { psnr: 23.9, ssim: 0.85 },
      unet: { psnr: 23.5, ssim: 0.83 },
      pinn: { psnr: 23.4, ssim: 0.82 },
    },
    winner: { psnr: 'unet', ssim: 'unet' },
  }
}

// Full user journey: phantom list loads, first phantom auto-fetches a result,
// changing the dose slider re-fetches and re-renders the panel with the new
// winner highlighted (Week 7 deliverable: "integration test passing").
describe('App — full user journey', () => {
  beforeEach(() => {
    global.fetch = vi.fn((url) => {
      if (url.toString().endsWith('/phantoms')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(PHANTOMS) })
      }
      if (url.toString().endsWith('/real_cases')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(REAL_CASES) })
      }
      if (url.toString().endsWith('/real_denoise')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(realDenoiseResponse('real_000')),
        })
      }
      if (url.toString().endsWith('/denoise')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(denoiseResponse('phantom_002', 'low')),
        })
      }
      return Promise.reject(new Error(`unexpected fetch: ${url}`))
    })
  })

  it('loads phantoms, runs an initial comparison, and shows the winning badges', async () => {
    render(<App />)

    // Phantom dropdown populates from GET /phantoms.
    await waitFor(() => {
      expect(screen.getByRole('combobox')).toHaveValue('phantom_002')
    })
    expect(screen.getByRole('option', { name: 'Phantom 005' })).toBeInTheDocument()

    // POST /denoise fires automatically for the first phantom + default dose.
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/denoise'),
        expect.objectContaining({
          body: JSON.stringify({ phantom_id: 'phantom_002', dose_level: 'low' }),
        }),
      )
    })

    // Results render: four panels, U-Net's winning PSNR/SSIM starred.
    await screen.findByText('U-Net output')
    expect(screen.getByText('PINN output')).toBeInTheDocument()
    expect(screen.getByText('Clean (ground truth)')).toBeInTheDocument()

    const unetPanel = screen.getByText('U-Net output').closest('figure')
    expect(within(unetPanel).getByText(/PSNR 38.30 dB ★/)).toBeInTheDocument()
    expect(within(unetPanel).getByText(/SSIM 0.964 ★/)).toBeInTheDocument()

    const pinnPanel = screen.getByText('PINN output').closest('figure')
    expect(within(pinnPanel).queryByText(/★/)).not.toBeInTheDocument()
  })

  it('re-fetches with the new dose level when the slider changes', async () => {
    render(<App />)

    await screen.findByText('U-Net output')
    global.fetch.mockClear()

    const slider = screen.getByLabelText('Dose level')
    fireEvent.change(slider, { target: { value: '1' } }) // low -> medium

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/denoise'),
        expect.objectContaining({
          body: JSON.stringify({ phantom_id: 'phantom_002', dose_level: 'medium' }),
        }),
      )
    })
  })

  it('switches to real TCIA data and runs /real_denoise for the selected case', async () => {
    render(<App />)

    await screen.findByText('U-Net output')
    global.fetch.mockClear()

    fireEvent.click(screen.getByRole('button', { name: 'Real Clinical (TCIA)' }))

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/real_denoise'),
        expect.objectContaining({
          body: JSON.stringify({ case_id: 'real_000' }),
        }),
      )
    })

    await waitFor(() => {
      const unetPanel = screen.getByText('U-Net output').closest('figure')
      expect(within(unetPanel).getByText(/PSNR 23.50 dB ★/)).toBeInTheDocument()
    })
  })
})
