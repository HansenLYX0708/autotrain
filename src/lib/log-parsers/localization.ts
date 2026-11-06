import type { ParsedTrainLog } from './types'
import { NonFiniteCollector, parseNumericToken } from './numbers'

const PREFIX = 'AUTOTRAIN_LOCALIZATION '

/** The adapter emits one JSON record per line; arbitrary stdout remains console-only. */
export function parseLocalizationLine(line: string): ParsedTrainLog[] {
  if (!line.startsWith(PREFIX)) return []
  let data: Record<string, unknown>
  try { data = JSON.parse(line.slice(PREFIX.length)) } catch { return [] }
  if (!data || !['train', 'eval'].includes(String(data.kind))) return []
  const collector = new NonFiniteCollector()
  const numeric = (key: string): number | null => typeof data[key] === 'number' ? collector.add(parseNumericToken(String(data[key])), key) : null
  const row: ParsedTrainLog = {
    kind: data.kind as 'train' | 'eval', epoch: numeric('epoch') ?? 0,
    iteration: numeric('iteration') ?? 0, totalIter: numeric('totalIter') ?? 0,
    loss: numeric('loss'), learningRate: numeric('learningRate'),
    lossL1: numeric('lossL1'), lossCls: numeric('lossCls'), lossDfl: numeric('lossDfl'),
    mAP: numeric('mAP'), mAP50: numeric('mAP50'), bestMetric: numeric('bestMetric'), bestIter: numeric('bestIter'),
    bestMetricName: typeof data.bestMetricName === 'string' ? data.bestMetricName : null,
    memAllocated: numeric('memAllocated'), memReserved: numeric('memReserved'),
    eta: null, batchCost: null, dataCost: null, readerCost: null, ips: null, rawLog: line,
  }
  if (data.extraMetrics && typeof data.extraMetrics === 'object' && !Array.isArray(data.extraMetrics)) {
    row.extraMetrics = {}
    for (const [key, value] of Object.entries(data.extraMetrics)) {
      if (typeof value !== 'number') continue
      const metric = collector.add(parseNumericToken(String(value)), `extraMetrics.${key}`)
      if (metric !== null) row.extraMetrics[key] = metric
    }
  }
  if (Array.isArray(data.nonFinite)) data.nonFinite.filter((name): name is string => typeof name === 'string').forEach(name => collector.mark(name))
  row.nonFinite = collector.names()
  return [row]
}
