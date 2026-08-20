/**
 * Framework-aware training log parser dispatcher.
 *
 * The training runner in `src/app/api/training-jobs/[id]/route.ts` spawns
 * `paddle.distributed.launch` and pipes stdout into Node. Node emits `data`
 * events in arbitrary chunks — a chunk may contain a partial line, one line,
 * or many lines. This dispatcher owns two responsibilities:
 *
 *   1. Line buffering: split incoming chunks on `\n`, hold the trailing
 *      unterminated fragment until the next chunk. This eliminates the
 *      previous chunk-boundary regex bug that could drop or duplicate rows.
 *   2. Framework dispatch: for each complete line, call the appropriate
 *      per-framework parser. PaddleSeg's parser is stateful (multi-line EVAL
 *      accumulator); Detection's is stateless. Both are pure functions of
 *      the current line + optional per-job state.
 *
 * Usage:
 *   const state = createParserState(job.project.framework)
 *   // ... on each stdout chunk ...
 *   const logs = feed(state, chunk)
 *   for (const log of logs) await writeToDb(job.id, log)
 *   // ... when the process exits, flush any half-line the parser was holding:
 *   const tail = flush(state)
 *   for (const log of tail) await writeToDb(job.id, log)
 *   disposeParserState(state)
 */

import { parseDetectionLines, createDetState, type DetParserState } from './detection'
import { parseSegLine, createSegState, type SegParserState } from './segmentation'
import { parseClassificationLine } from './classification'
import type { ParsedTrainLog } from './types'
import { isClassification, isSegmentation, normalizeFramework, type Framework } from '@/lib/frameworks'

export type { ParsedTrainLog } from './types'

export interface JobParserState {
  framework: Framework
  /** Trailing bytes from the previous chunk that had no `\n` yet. */
  buffer: string
  /** Segmentation needs multi-line EVAL accumulation; unused for other tasks. */
  segState?: SegParserState
  /** Detection needs multi-line COCO-block accumulation. */
  detState?: DetParserState
}

export function createParserState(framework: string | null | undefined): JobParserState {
  const fw = normalizeFramework(framework)
  const state: JobParserState = { framework: fw, buffer: '' }
  if (isSegmentation(fw)) state.segState = createSegState()
  else if (!isClassification(fw)) state.detState = createDetState()
  return state
}

/**
 * Parse a single complete line via the appropriate framework parser.
 * Exported so consumers that already own their own line boundaries (tests,
 * bulk import from a log file) can bypass the buffering step.
 */
export function parseLine(line: string, state: JobParserState): ParsedTrainLog[] {
  // Dispatch on the *task kind*, not the framework name: `torchtrain` emits
  // byte-identical log lines to PaddleSeg (segmentation) and PaddleDetection
  // (detection) precisely so these parsers, and the monitoring charts they feed,
  // are shared rather than duplicated. See `torchtrain/torchtrain/logger.py`.
  if (isSegmentation(state.framework)) {
    return parseSegLine(line, state.segState!)
  }
  if (isClassification(state.framework)) {
    const parsed = parseClassificationLine(line)
    return parsed ? [parsed] : []
  }
  return parseDetectionLines(line, state.detState ?? (state.detState = createDetState()))
}

/**
 * Feed a raw stdout chunk. Returns all `ParsedTrainLog` records that became
 * complete as a result of consuming this chunk (0..N).
 */
export function feed(state: JobParserState, chunk: string): ParsedTrainLog[] {
  if (!chunk) return []
  state.buffer += chunk
  const nlIdx = state.buffer.lastIndexOf('\n')
  if (nlIdx === -1) return [] // no complete line yet

  const complete = state.buffer.slice(0, nlIdx)
  state.buffer = state.buffer.slice(nlIdx + 1)

  const out: ParsedTrainLog[] = []
  for (const line of complete.split(/\r?\n/)) {
    if (!line) continue
    const rows = parseLine(line, state)
    if (rows.length > 0) out.push(...rows)
  }
  return out
}

/**
 * On process exit, flush any lingering line fragment and any open EVAL block.
 * Safe to call multiple times; idempotent.
 */
export function flush(state: JobParserState): ParsedTrainLog[] {
  const out: ParsedTrainLog[] = []

  const tail = state.buffer.trim()
  state.buffer = ''
  if (tail) {
    const rows = parseLine(tail, state)
    if (rows.length > 0) out.push(...rows)
  }

  // Force-close a half-parsed eval block by feeding a sentinel line that
  // matches no pattern — both parsers finalize the open block on that miss.
  // Without this, the last evaluation of a run (which is also usually the best
  // one) would be dropped whenever the process exits right after printing it.
  if (state.segState?.open) {
    const rows = parseSegLine('__eof_sentinel__', state.segState)
    if (rows.length > 0) out.push(...rows)
  }
  if (state.detState?.open) {
    const rows = parseDetectionLines('Best test bbox ap is 0.0.', state.detState)
    // Drop the synthetic best-metric row the sentinel produces; only the
    // flushed COCO block is real data.
    if (rows.length > 0) out.push(...rows.filter((r) => r.mAP != null || r.mAP50 != null))
  }
  return out
}

/** Free per-job state at job end. Keeps the app from leaking Map entries. */
export function disposeParserState(_state: JobParserState): void {
  // Nothing to close explicitly; the reference is dropped by the caller.
  // Kept as a symmetric API for future-proofing (e.g., open file handles).
}
