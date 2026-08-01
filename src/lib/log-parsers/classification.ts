/**
 * PaddleClas training log parser.
 *
 * Stub: PaddleClas support isn't wired end-to-end yet in this project. When
 * it is, extend this to parse lines like:
 *   [ts] INFO: [Train][Epoch 3/40][Iter: 40/74] loss: 3.10, lr: 0.001, ...
 *   [ts] INFO: [Eval][Epoch 3][top1: 0.234, top5: 0.678]
 *
 * The stub returns null for every line so the dispatcher falls back to
 * detection heuristics if the project mis-labels a Detection job as
 * PaddleClas. That preserves the pre-refactor behaviour.
 */

import type { ParsedTrainLog } from './types'

export function parseClassificationLine(_line: string): ParsedTrainLog | null {
  return null
}
