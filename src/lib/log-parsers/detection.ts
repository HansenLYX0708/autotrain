/**
 * Detection training log parser (PaddleDetection and TorchDet).
 *
 * Handles three line shapes.
 *
 * 1. TRAIN lines (one per `log_iter`, single line):
 *      [03/03 10:20:46] ppdet.engine.callbacks INFO: Epoch: [8] [60/79] \
 *        learning_rate: 0.000996 loss: 4.193813 loss_cls: 1.671748 \
 *        loss_iou: ... eta: 0:02:24 batch_cost: 0.34 data_cost: 0.01 \
 *        ips: 5.05 images/s max_mem_reserved: 3988 max_mem_allocated: 3542
 *
 * 2. COCO eval blocks (13 consecutive lines, emitted per eval interval):
 *      Evaluate annotation type *bbox*
 *       Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.212
 *       Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.423
 *       ... 10 more rows ...
 *    The block is accumulated by an explicit state machine and emitted as one
 *    `kind: 'eval'` record carrying mAP / mAP50, which is what gives detection
 *    jobs an evaluation chart at all (previously only the loss curve existed).
 *
 * 3. Best-checkpoint lines, which set the job's `bestMetric` / `bestIter`:
 *      TorchDet:         Best mAP(0.50:0.95) = 0.2116 at epoch 1.
 *      PaddleDetection:  Best test bbox ap is 0.4567.
 *
 * `torchtrain` reproduces shapes 1 and 2 byte-for-byte (see
 * `torchtrain/torchtrain/logger.py`) so both frameworks share this parser.
 *
 * The TRAIN regexes are lifted verbatim from the original monolithic parser in
 * `route.ts` so behaviour is byte-identical for existing PaddleDetection users.
 */

import type { ParsedTrainLog } from './types'

function extractFloat(line: string, key: string): number | null {
  // Word-boundary + optional whitespace after colon, mirrors the original.
  const m = line.match(new RegExp(`${key}:\\s*([\\d.e-]+)`, 'i'))
  return m ? parseFloat(m[1]) : null
}

interface OpenCocoEval {
  mAP: number | null
  mAP50: number | null
  rawLines: string[]
  /** How many `Average Precision/Recall` rows have been consumed so far. */
  rows: number
  /** Snapshot of the last TRAIN progress, so the eval row shares an x-axis. */
  epochAtOpen: number
  iterAtOpen: number
  totalIterAtOpen: number
}

export interface DetParserState {
  lastEpoch: number
  lastIter: number
  lastTotalIter: number
  open: OpenCocoEval | null
}

export function createDetState(): DetParserState {
  return { lastEpoch: 0, lastIter: 0, lastTotalIter: 0, open: null }
}

/** `Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.212` */
const AP_ALL_100 = /Average Precision\s+\(AP\)\s*@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*(-?[\d.]+)/i
const AP50_ALL_100 = /Average Precision\s+\(AP\)\s*@\[\s*IoU=0\.50\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\]\s*=\s*(-?[\d.]+)/i
const COCO_ROW = /Average (?:Precision|Recall)\s+\((?:AP|AR)\)\s*@\[/i
const COCO_BLOCK_START = /Evaluate annotation type\s+\*(\w+)\*/i
/** Total rows pycocotools' `summarize()` prints per annotation type. */
const COCO_ROWS_PER_BLOCK = 12

function finalizeOpenEval(open: OpenCocoEval): ParsedTrainLog {
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
    // A COCO value of -1 means "no ground truth in this bucket"; storing it
    // would draw a -1 point on the chart, so treat it as absent.
    mAP: open.mAP !== null && open.mAP >= 0 ? open.mAP : null,
    mAP50: open.mAP50 !== null && open.mAP50 >= 0 ? open.mAP50 : null,
    rawLog: open.rawLines.join('\n'),
  }
}

/**
 * Parse a single complete line. May emit 0, 1 or 2 records (the latter when a
 * TRAIN line arrives while a COCO block is still open).
 */
export function parseDetectionLines(line: string, state: DetParserState): ParsedTrainLog[] {
  if (!line || !line.trim()) return []
  const results: ParsedTrainLog[] = []

  // --- Best-checkpoint line -------------------------------------------
  const bestTorch = line.match(/Best\s+mAP\(0\.50:0\.95\)\s*=\s*([\d.]+)\s+at\s+epoch\s+(\d+)/i)
  const bestPaddle = line.match(/Best\s+test\s+bbox\s+ap\s+is\s+([\d.]+)/i)
  if (bestTorch || bestPaddle) {
    if (state.open) {
      results.push(finalizeOpenEval(state.open))
      state.open = null
    }
    results.push({
      kind: 'eval',
      epoch: state.lastEpoch,
      iteration: state.lastIter,
      totalIter: state.lastTotalIter,
      loss: null,
      learningRate: null,
      eta: null,
      batchCost: null,
      dataCost: null,
      readerCost: null,
      ips: null,
      memReserved: null,
      memAllocated: null,
      bestMetric: parseFloat((bestTorch ?? bestPaddle)![1]),
      bestIter: bestTorch ? parseInt(bestTorch[2], 10) : state.lastEpoch,
      rawLog: line.trim(),
    })
    return results
  }

  // --- COCO block: opening marker -------------------------------------
  const blockStart = line.match(COCO_BLOCK_START)
  if (blockStart) {
    if (state.open) results.push(finalizeOpenEval(state.open))
    // Instance-segmentation models print a `*segm*` block after `*bbox*`. Only
    // the bbox block feeds the chart; opening a second one would overwrite the
    // detection metric with the mask metric.
    state.open =
      blockStart[1].toLowerCase() === 'bbox'
        ? {
            mAP: null,
            mAP50: null,
            rawLines: [line.trim()],
            rows: 0,
            epochAtOpen: state.lastEpoch,
            iterAtOpen: state.lastIter,
            totalIterAtOpen: state.lastTotalIter,
          }
        : null
    return results
  }

  // --- COCO block: metric rows ----------------------------------------
  if (state.open && COCO_ROW.test(line)) {
    const ap = line.match(AP_ALL_100)
    if (ap) state.open.mAP = parseFloat(ap[1])
    const ap50 = line.match(AP50_ALL_100)
    if (ap50) state.open.mAP50 = parseFloat(ap50[1])
    state.open.rawLines.push(line.trim())
    state.open.rows += 1
    if (state.open.rows >= COCO_ROWS_PER_BLOCK) {
      results.push(finalizeOpenEval(state.open))
      state.open = null
    }
    return results
  }

  // --- TRAIN line ------------------------------------------------------
  const parsed = parseDetectionLine(line)
  if (parsed) {
    // A TRAIN line means any open block is done; flush it first.
    if (state.open) {
      results.push(finalizeOpenEval(state.open))
      state.open = null
    }
    if (parsed.epoch) state.lastEpoch = parsed.epoch
    if (parsed.iteration) state.lastIter = parsed.iteration
    if (parsed.totalIter) state.lastTotalIter = parsed.totalIter
    results.push(parsed)
    return results
  }

  // A non-matching line while a partial block is open: keep the block open.
  // pycocotools interleaves progress chatter ("DONE (t=0.01s)") between rows,
  // and closing on the first unrelated line would lose the metrics.
  return results
}

/**
 * Stateless single-line TRAIN parser.
 *
 * Kept exported because callers that only care about progress (and tests) use it
 * directly; `parseDetectionLines` wraps it with the eval-block state machine.
 */
export function parseDetectionLine(line: string): ParsedTrainLog | null {
  if (!line || !line.trim()) return null

  // Iteration markers: Detection prints `Epoch: [E] [i/total]`.
  const iterPatterns = line.matchAll(/\[(\d+)\/(\d+)\]/g)
  const iterMatches = Array.from(iterPatterns)
  if (iterMatches.length === 0) return null

  const epochMatch = line.match(/Epoch:\s*\[(\d+)\]/i)
  const epoch = epochMatch ? parseInt(epochMatch[1], 10) : 0

  // Detection places [iter/total] AFTER [epoch]; take the last [x/y] as
  // iteration to be robust against extra brackets in prefixes.
  const lastMatch = iterMatches[iterMatches.length - 1]
  const iteration = parseInt(lastMatch[1], 10)
  const totalIter = parseInt(lastMatch[2], 10)

  // Learning rate is emitted as `learning_rate` (Detection) or `lr` (Seg but
  // we also accept it here for eval-mode logs coming through detection code
  // paths — matches the pre-refactor behaviour).
  const lr =
    extractFloat(line, 'learning_rate') ??
    extractFloat(line, 'lr')

  // `loss:` but not `loss_cls:` / `loss_iou:` etc.
  const lossMatch = line.match(/(?:^|\s)loss:\s*([\d.]+)(?!\w)/i)
  const loss = lossMatch ? parseFloat(lossMatch[1]) : null

  const lossCls = extractFloat(line, 'loss_cls')
  const lossIou = extractFloat(line, 'loss_iou')
  const lossDfl = extractFloat(line, 'loss_dfl')
  const lossL1 = extractFloat(line, 'loss_l1')

  // Detection ETA supports both plain "0:02:24" and "1 day, 20:03:57".
  const etaMatch = line.match(/eta:\s*([^\s,]+(?:,\s*\d{1,2}:\d{2}:\d{2})?)/i)
  const eta = etaMatch ? etaMatch[1].trim() : null

  const batchCost = extractFloat(line, 'batch_cost')
  const dataCost = extractFloat(line, 'data_cost')

  const ipsMatch = line.match(/ips:\s*([\d.]+)/i)
  const ips = ipsMatch ? parseFloat(ipsMatch[1]) : null

  const memReservedMatch = line.match(/max_mem_reserved:\s*(\d+)/i)
  const memAllocatedMatch = line.match(/max_mem_allocated:\s*(\d+)/i)
  const memReserved = memReservedMatch ? parseInt(memReservedMatch[1], 10) : null
  const memAllocated = memAllocatedMatch ? parseInt(memAllocatedMatch[1], 10) : null

  // No usable training metrics? Not a training line.
  if (!epochMatch && loss === null && lr === null) return null

  return {
    kind: 'train',
    epoch,
    iteration,
    totalIter,
    loss,
    learningRate: lr,
    eta,
    batchCost,
    dataCost,
    readerCost: null,
    ips,
    memReserved,
    memAllocated,
    lossCls,
    lossIou,
    lossDfl,
    lossL1,
    rawLog: line.trim(),
  }
}
