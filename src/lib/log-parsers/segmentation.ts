/**
 * PaddleSeg training log parser.
 *
 * Handles two very different line shapes:
 *
 * 1. TRAIN lines (one per emit interval, single line):
 *    [2026/07/20 11:08:59] INFO: [TRAIN] epoch: 2278, iter: 20500/160000, \
 *      loss: 0.5440, lr: 0.004445, batch_cost: 0.1866, reader_cost: 0.06624, \
 *      ips: 21.4352 samples/sec, max_mem_reserved: 3988 MB, \
 *      max_mem_allocated: 3542 MB | ETA 07:13:51
 *
 * 2. EVAL blocks (5 consecutive lines, emitted per eval interval):
 *    [ts] INFO: [EVAL] #Images: 10 mIoU: 0.5696 Acc: 0.9900 Kappa: ... Dice: ...
 *    [ts] INFO: [EVAL] Class IoU:
 *    [0.9952 0.3447 0.6496 0.2887]
 *    [ts] INFO: [EVAL] Class Precision:
 *    [0.9976 0.4916 0.764  0.4805]
 *    [ts] INFO: [EVAL] Class Recall:
 *    [0.9976 0.5358 0.8128 0.4198]
 *    [ts] INFO: [EVAL] The model with the best validation mIoU (0.6675) was saved at iter 5000.
 *
 * The block is aggregated across lines by an explicit state machine
 * (`SegParserState`) and one combined `ParsedTrainLog` (`kind: 'eval'`) is
 * emitted when the closing "best mIoU" line arrives OR when a non-EVAL line
 * interrupts the block (safety fallback).
 */

import type { ParsedTrainLog } from './types'

interface OpenEval {
  mIoU: number | null
  acc: number | null
  kappa: number | null
  dice: number | null
  classIoU?: number[]
  classPrecision?: number[]
  classRecall?: number[]
  bestIter?: number | null
  bestMetric?: number | null
  /** Which `Class <kind>:` header we most recently saw; the *next* `[..]`
   *  array line is consumed into that slot. */
  expectClassArray: 'iou' | 'precision' | 'recall' | null
  /**
   * Partially-read float array, for the case where numpy wrapped it.
   *
   * PaddleSeg prints these arrays with `str(np.round(class_iou, 4))`, and
   * numpy breaks its output at `linewidth` (75 by default). From roughly ten
   * classes on, a single logical array therefore arrives as two or more
   * physical lines:
   *
   *     [0.9952 0.3447 0.6496 0.2887 0.1234 0.5678 0.9012 0.3456 0.789  0.1234
   *      0.4567 0.6789]
   *
   * Lines are concatenated here until the closing `]` shows up.
   */
  pendingArray: string | null
  rawLines: string[]
  /** Snapshot of the current train iter/epoch when EVAL opened, so the eval
   *  row can be plotted on the same x-axis. Populated by the dispatcher
   *  before it hands the line to this parser. */
  epochAtOpen: number
  iterAtOpen: number
  totalIterAtOpen: number
}

export interface SegParserState {
  /** Most recent per-iter progress, tracked so EVAL rows inherit an x-axis. */
  lastEpoch: number
  lastIter: number
  lastTotalIter: number
  open: OpenEval | null
}

export function createSegState(): SegParserState {
  return { lastEpoch: 0, lastIter: 0, lastTotalIter: 0, open: null }
}

function segFloat(line: string, re: RegExp): number | null {
  const m = line.match(re)
  return m ? parseFloat(m[1]) : null
}

/** Parse a complete `[a b c d]` whitespace-separated float array. Returns
 *  `null` if the text is not a closed float array. */
function parseFloatArray(text: string): number[] | null {
  const stripped = text.trim()
  if (!stripped.startsWith('[') || !stripped.endsWith(']')) return null
  const inner = stripped.slice(1, -1).trim()
  if (!inner) return null
  const nums = inner.split(/\s+/).map((tok) => parseFloat(tok))
  if (nums.some((n) => Number.isNaN(n))) return null
  return nums
}

/**
 * Could `line` be the start (or continuation) of a numpy float array that has
 * not been closed yet? Used to decide whether to keep buffering instead of
 * treating the line as "not part of this EVAL block".
 *
 * Deliberately strict: only digits, dots, signs, exponents, whitespace and a
 * single leading `[` are allowed, so a genuine log line never gets swallowed.
 */
function isOpenArrayFragment(text: string): boolean {
  const stripped = text.trim()
  if (!stripped || stripped.endsWith(']')) return false
  const body = stripped.startsWith('[') ? stripped.slice(1) : stripped
  return body.length > 0 && /^[\d.eE+\-\s]+$/.test(body)
}

function finalizeOpenEval(open: OpenEval): ParsedTrainLog {
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
    mIoU: open.mIoU,
    acc: open.acc,
    kappa: open.kappa,
    dice: open.dice,
    classIoU: open.classIoU,
    classPrecision: open.classPrecision,
    classRecall: open.classRecall,
    bestIter: open.bestIter ?? null,
    bestMetric: open.bestMetric ?? null,
    rawLog: open.rawLines.join('\n'),
  }
}

/**
 * Feed a single line into the PaddleSeg parser. May emit:
 *   - 0 records: line consumed into an in-progress EVAL block
 *   - 1 record: TRAIN line, or EVAL block just closed
 *   - 2 records: rare — non-TRAIN/EVAL line arrives while an EVAL block is
 *     open, we flush the block (record 1) and then attempt to parse the
 *     interrupting line as TRAIN (record 2 if it matched).
 */
export function parseSegLine(
  line: string,
  state: SegParserState,
): ParsedTrainLog[] {
  const trimmed = line.trimEnd()
  if (!trimmed.trim()) return []

  const results: ParsedTrainLog[] = []

  // --- 1. TRAIN line ---------------------------------------------------
  // Note: even if an EVAL block was mid-flight, a TRAIN line means the
  // block is done — flush it first.
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
    const loss = segFloat(trimmed, /loss:\s*([\d.]+)/i)
    // Seg's lr can be scientific (e.g., `lr: 1.234e-05`).
    const lr = segFloat(trimmed, /lr:\s*([\d.e+-]+)/i)
    const batchCost = segFloat(trimmed, /batch_cost:\s*([\d.]+)/i)
    const readerCost = segFloat(trimmed, /reader_cost:\s*([\d.]+)/i)
    const ipsMatch = trimmed.match(/ips:\s*([\d.]+)/i)
    const ips = ipsMatch ? parseFloat(ipsMatch[1]) : null
    const etaMatch = trimmed.match(/ETA\s+(\d+:\d{2}:\d{2})/i)
    const memReservedMatch = trimmed.match(/max_mem_reserved:\s*(\d+)/i)
    const memAllocatedMatch = trimmed.match(/max_mem_allocated:\s*(\d+)/i)

    // Update x-axis snapshot for any subsequent EVAL block.
    if (epoch) state.lastEpoch = epoch
    if (iteration) state.lastIter = iteration
    if (totalIter) state.lastTotalIter = totalIter

    results.push({
      kind: 'train',
      epoch,
      iteration,
      totalIter,
      loss,
      learningRate: lr,
      eta: etaMatch ? etaMatch[1] : null,
      batchCost,
      dataCost: null,
      readerCost,
      ips,
      memReserved: memReservedMatch ? parseInt(memReservedMatch[1], 10) : null,
      memAllocated: memAllocatedMatch ? parseInt(memAllocatedMatch[1], 10) : null,
      rawLog: trimmed,
    })
    return results
  }

  // --- 2. EVAL block: opening line ------------------------------------
  //  [EVAL] #Images: N mIoU: X Acc: Y Kappa: Z Dice: W
  if (/\[EVAL\]/i.test(trimmed) && /mIoU:/i.test(trimmed) && /#Images:/i.test(trimmed)) {
    // If a previous block never closed cleanly (shouldn't happen but be
    // defensive), flush it before starting a new one.
    if (state.open) {
      results.push(finalizeOpenEval(state.open))
    }
    state.open = {
      mIoU: segFloat(trimmed, /mIoU:\s*([\d.]+)/i),
      acc: segFloat(trimmed, /Acc:\s*([\d.]+)/i),
      kappa: segFloat(trimmed, /Kappa:\s*([\d.]+)/i),
      dice: segFloat(trimmed, /Dice:\s*([\d.]+)/i),
      expectClassArray: null,
      pendingArray: null,
      rawLines: [trimmed],
      epochAtOpen: state.lastEpoch,
      iterAtOpen: state.lastIter,
      totalIterAtOpen: state.lastTotalIter,
    }
    return results
  }

  // --- 3. EVAL block: `Class <IoU|Precision|Recall>:` header ----------
  if (state.open && /\[EVAL\]\s*Class\s+(IoU|Precision|Recall):/i.test(trimmed)) {
    const m = trimmed.match(/Class\s+(IoU|Precision|Recall):/i)!
    const which = m[1].toLowerCase() as 'iou' | 'precision' | 'recall'
    state.open.expectClassArray = which
    // A new header always starts a fresh array, even if the previous one was
    // left unterminated by a truncated log.
    state.open.pendingArray = null
    state.open.rawLines.push(trimmed)
    return results
  }

  // --- 4. EVAL block: `[a b c d]` array payload -----------------------
  // The array may be spread over several physical lines, because numpy wraps
  // at `linewidth=75` (so ~10 classes and up always wrap). Accumulate until
  // the closing `]` arrives; only then assign it to its slot.
  if (state.open && state.open.expectClassArray) {
    const candidate =
      state.open.pendingArray === null ? trimmed : `${state.open.pendingArray} ${trimmed.trim()}`

    const arr = parseFloatArray(candidate)
    if (arr) {
      const slot = state.open.expectClassArray
      if (slot === 'iou') state.open.classIoU = arr
      else if (slot === 'precision') state.open.classPrecision = arr
      else if (slot === 'recall') state.open.classRecall = arr
      state.open.expectClassArray = null
      state.open.pendingArray = null
      state.open.rawLines.push(trimmed)
      return results
    }

    if (isOpenArrayFragment(candidate)) {
      // Still waiting for the closing bracket — keep buffering. Without this
      // the wrapped array fell through to case 6, which closed the block and
      // silently discarded every per-class metric *and* the trailing
      // "best validation mIoU" line (so bestIter/bestMetric stayed null).
      state.open.pendingArray = candidate
      state.open.rawLines.push(trimmed)
      return results
    }

    // Not an array at all — fall through and let the closing-line / TRAIN
    // logic decide. Clear the expectation so a bogus line doesn't hijack the
    // next `[..]`.
    state.open.expectClassArray = null
    state.open.pendingArray = null
  }

  // --- 5. EVAL block: closing "best mIoU" line ------------------------
  if (state.open && /\[EVAL\][^]*best validation mIoU/i.test(trimmed)) {
    const bestMatch = trimmed.match(/mIoU\s*\(\s*([\d.]+)\s*\)/i)
    const iterMatch = trimmed.match(/iter\s+(\d+)/i)
    if (bestMatch) state.open.bestMetric = parseFloat(bestMatch[1])
    if (iterMatch) state.open.bestIter = parseInt(iterMatch[1], 10)
    state.open.rawLines.push(trimmed)
    results.push(finalizeOpenEval(state.open))
    state.open = null
    return results
  }

  // --- 6. Line does not belong to a Seg train/eval shape --------------
  // If a block is open, close it defensively so we don't lose the head.
  // Note: PaddleSeg prints an "Evaluating..." progress bar line
  // (`10/10 [====]...`) *between* the "Start evaluating" line and the
  // mIoU summary line. That progress bar is not part of the eval block
  // we care about, so we simply ignore non-matching lines when no block
  // is open.
  if (state.open) {
    results.push(finalizeOpenEval(state.open))
    state.open = null
  }
  return results
}
