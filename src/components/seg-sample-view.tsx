'use client'

import { useEffect, useRef } from 'react'

interface SegSampleViewProps {
  imagePath: string
  maskPath: string
  /**
   * Number of classes in the dataset. When provided, the mask URL is fetched
   * with `colorize=1&classes=N` so that grayscale label-map masks (produced by
   * labelme-to-paddleseg) get pseudo-colored server-side using the same VOC
   * palette we render swatches with. Multi-channel (already pseudo-color)
   * masks are passed through by the endpoint, so this is safe to always set.
   */
  numClasses?: number
  /** Class-0 (background) color to render transparent. */
  backgroundColor?: [number, number, number]
  /** Overlay opacity for non-background pixels (0..1). */
  opacity?: number
  /** Whether to hide the background class (class 0). */
  hideBackground?: boolean
  /** Longest-side cap for the internal canvas resolution. */
  maxSize?: number
  className?: string
}

const imageUrl = (p: string) => `/api/datasets/image?path=${encodeURIComponent(p)}`
const maskUrl = (p: string, numClasses?: number) =>
  numClasses && numClasses > 0
    ? `${imageUrl(p)}&colorize=1&classes=${numClasses}`
    : imageUrl(p)

/**
 * Renders a segmentation sample: the original image with its pseudo-color mask
 * composited on top. The mask is a paletted PNG, so the browser resolves the
 * palette to RGB when drawing it to a canvas; we then make background pixels
 * transparent and blend the remaining classes at the requested opacity.
 */
export function SegSampleView({
  imagePath,
  maskPath,
  numClasses,
  backgroundColor = [128, 0, 0],
  opacity = 0.5,
  hideBackground = true,
  maxSize = 512,
  className,
}: SegSampleViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    let cancelled = false
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    const mask = new Image()
    let imgReady = false
    let maskReady = false

    const render = () => {
      if (cancelled || !imgReady) return
      const w0 = img.naturalWidth
      const h0 = img.naturalHeight
      if (!w0 || !h0) return
      const scale = Math.min(1, maxSize / Math.max(w0, h0))
      const w = Math.max(1, Math.round(w0 * scale))
      const h = Math.max(1, Math.round(h0 * scale))
      canvas.width = w
      canvas.height = h
      ctx.clearRect(0, 0, w, h)
      ctx.drawImage(img, 0, 0, w, h)

      if (maskReady && mask.naturalWidth && mask.naturalHeight) {
        const tmp = document.createElement('canvas')
        tmp.width = w
        tmp.height = h
        const tctx = tmp.getContext('2d')
        if (tctx) {
          // Nearest-neighbour keeps the palette colors exact (no anti-alias blending).
          tctx.imageSmoothingEnabled = false
          tctx.drawImage(mask, 0, 0, w, h)
          try {
            const id = tctx.getImageData(0, 0, w, h)
            const d = id.data
            const [br, bg, bb] = backgroundColor
            const alpha = Math.round(Math.min(1, Math.max(0, opacity)) * 255)
            for (let i = 0; i < d.length; i += 4) {
              const isBg = d[i] === br && d[i + 1] === bg && d[i + 2] === bb
              d[i + 3] = hideBackground && isBg ? 0 : alpha
            }
            tctx.putImageData(id, 0, 0)
            ctx.drawImage(tmp, 0, 0)
          } catch {
            // getImageData throws on tainted canvases; skip the overlay if so.
          }
        }
      }
    }

    img.onload = () => { imgReady = true; render() }
    img.onerror = () => { /* leave canvas blank */ }
    mask.onload = () => { maskReady = true; render() }
    mask.onerror = () => { maskReady = false; render() }
    img.src = imageUrl(imagePath)
    mask.src = maskUrl(maskPath, numClasses)

    return () => { cancelled = true }
  }, [imagePath, maskPath, numClasses, backgroundColor, opacity, hideBackground, maxSize])

  return <canvas ref={canvasRef} className={className} />
}
