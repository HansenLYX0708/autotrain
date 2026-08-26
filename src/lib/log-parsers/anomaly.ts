/**
 * TorchAnomaly (anomalib adapter) training log parser.
 *
 * The adapter in `torchtrain/torchtrain/ad/` prints PaddleSeg-shaped TRAIN lines
 * on purpose, so that half of this parser is the same regex work the Seg parser
 * does. Only the EVAL shape is new, because the metrics are different:
 *
 * 1. TRAIN (single line, identical format to PaddleSeg):
 *    [2026/08/19 10:21:03] INFO: [TRAIN] epoch: 3, iter: 600/8000, loss: 0.1837, \
 *      lr: 0.000100, batch_cost: 0.0921, reader_cost: 0.0113, \
 *      ips: 86.8000 samples/sec, max_mem_reserved: 2048 MB, \
 *      max_mem_allocated: 1902 MB | ETA 00:11:22
 *
 * 2. EVAL (1-2 lines):
 *    [ts] INFO: [EVAL] #Images: 40 image_auroc: 0.9812 image_f1: 0.9231 \
 *      pixel_auroc: 0.9633 pixel_f1: 0.5412 threshold: 12.3456
 *    [ts] INFO: [EVAL] The model with the best validation image_auroc (0.9812) \
 *      was saved at iter 3000.
 *
 * Two deliberate differences from `segmentation.ts`:
 *
 * - **Metric names are not hard-coded.** Every `key: value` pair on the EVAL
 *   line is read generically; the five that have DB columns are mapped onto
 *   them and the rest survive in `extraMetrics`. anomalib lets an `Evaluator`
 *   register arbitrary metrics, so a fixed list would silently drop data the
 *   moment someone adds AUPRO.
 * - **The closing "best" line names its own metric** (`image_auroc` here, but
 *   `pixel_auroc` for a localisation-first job), so `bestMetricName` is parsed
 *   rather than assumed.
 *
 * Memory-bank models (PatchCore, PaDiM) have no loss: the adapter reports
 * `loss: 0.0000` while `iter` tracks memory-bank fill, so progress still moves.
 */

import type { ParsedTrainLog } from './types'

/** Metric names that have dedicated `TrainingLog` columns. */
const COLUMN_METRICS = new Set(['image_auroc', 'image_f1', 'pixel_auroc', 'pixel_f1', 'threshold'])

interface OpenEval {
  numImages: number | null
  metrics: Record<string, number>
  bestIter: number | null
  bestMetric: number | null
  bestMetricName: string | null
  rawLines: string[]
  epochAtOpen: number
  iterAtOpen: number
  totalIterAtOpen: number
}

export interface AnomalyParserState {
  lastEpoch: number
  lastIter: number
  lastTotalIter: number
  open: OpenEval | null
}

export function createAnomalyState(): AnomalyParserState {
  return { lastEpoch: 0, lastIter: 0, lastTotalIter: 0, open: null }
}

function num(line: string, re: RegExp): number | null {
  const m = line.match(re)
  if (!m) return null
  const value = parseFloat(m[1])
  return Number.isFinite(value) ? value : null
}

/**
 * Collect every `key: value` pair whose value is numeric.
 *
 * Keys are lower-cased so `Image_AUROC` and `image_auroc` land in the same slot.
 * The leading `#` is captured rather than excluded so that `#Images: 40` — the
 * sample count, not a metric — is skipped instead of being reported as a metric
 * called `images`. (A lookbehind would read better but is avoided: this module is
 * bundled for the browser too.)
 */
function readMetricPairs(line: string): Record<string, number> {
  const out: Record<string, number> = {}
  const re = /(#?)([A-Za-z][A-Za-z0-9_]*)\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)/g
  let match: RegExpExecArray | null
  while ((match = re.exec(line)) !== null) {
    if (match[1] === '#') continue
    const value = parseFloat(match[3])
    if (Number.isFinite(value)) out[match[2].toLowerCase()] = value
  }
  return out
}

function finalizeOpenEval(open: OpenEval): ParsedTrainLog {
  const extra: Record<string, number> = {}
  for (const [key, value] of Object.entries(open.metrics)) {
    if (!COLUMN_METRICS.has(key)) extra[key] = value
  }
  return {
    kind: 'eval',
    epoch: open.epochAtOpen,
    iteration: open.iterAtOpen,
    totalIter: open.totalIterAtOpen,
    loss: null,
    learningRate: null,
    eta: null,
    batchCost: null,
    dataCost: null,
    readerCost: null,
    ips: null,
    memReserved: null,
    memAllocated: null,
    imageAuroc: open.metrics.image_auroc ?? null,
    imageF1: open.metrics.image_f1 ?? null,
    pixelAuroc: open.metrics.pixel_auroc ?? null,
    pixelF1: open.metrics.pixel_f1 ?? null,
    threshold: open.metrics.threshold ?? null,
    extraMetrics: Object.keys(extra).length > 0 ? extra : undefined,
    bestIter: open.bestIter,
    bestMetric: open.bestMetric,
    bestMetricName: open.bestMetricName,
    rawLog: open.rawLines.join('\n'),
  }
}

/**
 * Feed one line into the anomaly parser.
 *
 * Emits at most two records, following the same contract as `parseSegLine`: an
 * in-flight EVAL block is flushed when a TRAIN line, a new EVAL block, or an
 * unrelated line interrupts it, so nothing is lost if the trainer dies between
 * the metrics line and the "best" line.
 */
export function parseAnomalyLine(line: string, state: AnomalyParserState): ParsedTrainLog[] {
  const trimmed = line.trimEnd()
  if (!trimmed.trim()) return []

  const results: ParsedTrainLog[] = []

  // --- 1. TRAIN line ----------------------------------------------------
  if (/\[TRAIN\]/i.test(trimmed)) {
    if (state.open) {
      results.push(finalizeOpenEval(state.open))
      state.open = null
    }

    const epochMatch = trimmed.match(/epoch:\s*(\d+)/i)
    const epoch = epochMatch ? parseInt(epochMatch[1], 10) : 0
    const iterMatch = trimmed.match(/iter:\s*(\d+)\/(\d+)/i)
    const iteration = iterMatch ? parseInt(iterMatch[1], 10) : 0
    const totalIter = iterMatch ? parseInt(iterMatch[2], 10) : 0
    const etaMatch = trimmed.match(/ETA\s+(\d+:\d{2}:\d{2})/i)
    const memReservedMatch = trimmed.match(/max_mem_reserved:\s*(\d+)/i)
    const memAllocatedMatch = trimmed.match(/max_mem_allocated:\s*(\d+)/i)

    if (epoch) state.lastEpoch = epoch
    if (iteration) state.lastIter = iteration
    if (totalIter) state.lastTotalIter = totalIter

    results.push({
      kind: 'train',
      epoch,
      iteration,
      totalIter,
      loss: num(trimmed, /loss:\s*([\d.e+-]+)/i),
      learningRate: num(trimmed, /lr:\s*([\d.e+-]+)/i),
      eta: etaMatch ? etaMatch[1] : null,
      batchCost: num(trimmed, /batch_cost:\s*([\d.]+)/i),
      dataCost: null,
      readerCost: num(trimmed, /reader_cost:\s*([\d.]+)/i),
      ips: num(trimmed, /ips:\s*([\d.]+)/i),
      memReserved: memReservedMatch ? parseInt(memReservedMatch[1], 10) : null,
      memAllocated: memAllocatedMatch ? parseInt(memAllocatedMatch[1], 10) : null,
      rawLog: trimmed,
    })
    return results
  }

  const isEval = /\[EVAL\]/i.test(trimmed)

  // --- 2. EVAL closing line: "... best validation <metric> (X) ... iter N" ---
  // Checked before the metrics line because it also contains a `(...)` number
  // that `readMetricPairs` must not be handed.
  if (isEval && /best\s+validation/i.test(trimmed)) {
    const bestMatch = trimmed.match(/best\s+validation\s+([A-Za-z][A-Za-z0-9_]*)\s*\(\s*([\d.]+)\s*\)/i)
    const iterMatch = trimmed.match(/iter\s+(\d+)/i)
    // The line may arrive with no metrics line before it (e.g. a resumed run),
    // so synthesise an empty block rather than dropping the best-model update.
    const open: OpenEval =
      state.open ??
      {
        numImages: null,
        metrics: {},
        bestIter: null,
        bestMetric: null,
        bestMetricName: null,
        rawLines: [],
        epochAtOpen: state.lastEpoch,
        iterAtOpen: state.lastIter,
        totalIterAtOpen: state.lastTotalIter,
      }
    if (bestMatch) {
      open.bestMetricName = bestMatch[1].toLowerCase()
      open.bestMetric = parseFloat(bestMatch[2])
    }
    if (iterMatch) open.bestIter = parseInt(iterMatch[1], 10)
    open.rawLines.push(trimmed)
    results.push(finalizeOpenEval(open))
    state.open = null
    return results
  }

  // --- 3. EVAL metrics line --------------------------------------------
  if (isEval && /#Images:/i.test(trimmed)) {
    if (state.open) results.push(finalizeOpenEval(state.open))
    state.open = {
      numImages: num(trimmed, /#Images:\s*(\d+)/i),
      metrics: readMetricPairs(trimmed),
      bestIter: null,
      bestMetric: null,
      bestMetricName: null,
      rawLines: [trimmed],
      epochAtOpen: state.lastEpoch,
      iterAtOpen: state.lastIter,
      totalIterAtOpen: state.lastTotalIter,
    }
    return results
  }

  // --- 4. Anything else -------------------------------------------------
  if (state.open) {
    results.push(finalizeOpenEval(state.open))
    state.open = null
  }
  return results
}
