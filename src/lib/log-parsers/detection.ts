/**
 * PaddleDetection training log parser.
 *
 * Handles lines of the form:
 *   [03/03 10:20:46] ppdet.engine.callbacks INFO: Epoch: [8] [60/79] \
 *     learning_rate: 0.000996 loss: 4.193813 loss_cls: 1.671748 \
 *     loss_iou: ... eta: 0:02:24 batch_cost: 0.34 data_cost: 0.01 \
 *     ips: 5.05 images/s max_mem_reserved: 3988 max_mem_allocated: 3542
 *
 * The regexes here are lifted verbatim from the original monolithic parser in
 * `route.ts` so behaviour is byte-identical for existing PaddleDetection
 * users.
 */

import type { ParsedTrainLog } from './types'

function extractFloat(line: string, key: string): number | null {
  // Word-boundary + optional whitespace after colon, mirrors the original.
  const m = line.match(new RegExp(`${key}:\\s*([\\d.e-]+)`, 'i'))
  return m ? parseFloat(m[1]) : null
}

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
