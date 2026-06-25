/**
 * PaddleDetection Log Parser
 * Parses training log lines and extracts structured metrics
 */

export interface ParsedLog {
  timestamp: string
  epoch: number
  iteration: number
  totalIter: number
  learningRate: number | null
  loss: number | null
  lossCls: number | null
  lossIou: number | null
  lossDfl: number | null
  lossL1: number | null
  eta: string | null
  batchCost: number | null
  dataCost: number | null
  ips: number | null
  memReserved: number | null
  memAllocated: number | null
  // PaddleSeg-specific (optional)
  readerCost?: number | null
  mIoU?: number | null
  acc?: number | null
  kappa?: number | null
  dice?: number | null
  rawLog: string
}

/**
 * Parse a PaddleDetection training log line
 * Example: Epoch: [8] [60/79] learning_rate: 0.000996 loss: 4.193813 ...
 * Also supports: [03/03 10:20:46] ppdet.engine.callbacks INFO: Epoch: [0] [100/827] ...
 */
export function parsePaddleDetectionLog(line: string): ParsedLog | null {
  // Skip empty lines
  if (!line.trim()) {
    return null
  }

  // Match PaddleDetection log format
  const timestampMatch = line.match(/^\[(\d{2}\/\d{2} \d{2}:\d{2}:\d{2})\]/)
  const timestamp = timestampMatch ? timestampMatch[1] : ''

  // Match epoch: Epoch: [0] or Epoch: [8]
  const epochMatch = line.match(/Epoch:\s*\[(\d+)\]/)
  const epoch = epochMatch ? parseInt(epochMatch[1], 10) : 0

  // Match iteration: [iter/total] - must find the one AFTER the epoch bracket
  // The pattern is: Epoch: [X] [iter/total]
  // So we look for the second bracket pair with a slash
  let iteration = 0
  let totalIter = 0
  
  // Find all [x/y] patterns in the line
  const iterPatterns = line.matchAll(/\[(\d+)\/(\d+)\]/g)
  const iterMatches = Array.from(iterPatterns)
  
  // If there's an epoch match, the iteration is the next [x/y] pattern after it
  // Otherwise, take the first [x/y] pattern
  if (iterMatches.length > 0) {
    // Always use the last [x/y] pattern as iteration (after epoch bracket)
    const lastMatch = iterMatches[iterMatches.length - 1]
    iteration = parseInt(lastMatch[1], 10)
    totalIter = parseInt(lastMatch[2], 10)
  }

  // Skip lines that don't have training progress info
  if (iterMatches.length === 0) {
    return null
  }

  // Extract metrics using helper function
  const learningRate = extractFloat(line, 'learning_rate')
  const loss = extractFloat(line, 'loss')
  const lossCls = extractFloat(line, 'loss_cls')
  const lossIou = extractFloat(line, 'loss_iou')
  const lossDfl = extractFloat(line, 'loss_dfl')
  const lossL1 = extractFloat(line, 'loss_l1')

  // Extract ETA (format: "eta: 0:02:24" or "eta: 1 day, 20:03:57")
  const etaMatch = line.match(/eta:\s*([^\s,]+(?:,\s*\d{1,2}:\d{2}:\d{2})?)/)
  const eta = etaMatch ? etaMatch[1].trim() : null

  // Extract costs
  const batchCost = extractFloat(line, 'batch_cost')
  const dataCost = extractFloat(line, 'data_cost')

  // Extract IPS (images per second) - handle "ips: 5.0540 images/s"
  const ipsMatch = line.match(/ips:\s*([\d.]+)/)
  const ips = ipsMatch ? parseFloat(ipsMatch[1]) : null

  // Extract memory info (MB)
  const memReservedMatch = line.match(/max_mem_reserved:\s*(\d+)/)
  const memAllocatedMatch = line.match(/max_mem_allocated:\s*(\d+)/)
  const memReserved = memReservedMatch ? parseInt(memReservedMatch[1], 10) : null
  const memAllocated = memAllocatedMatch ? parseInt(memAllocatedMatch[1], 10) : null

  return {
    timestamp,
    epoch,
    iteration,
    totalIter,
    learningRate,
    loss,
    lossCls,
    lossIou,
    lossDfl,
    lossL1,
    eta,
    batchCost,
    dataCost,
    ips,
    memReserved,
    memAllocated,
    rawLog: line.trim()
  }
}

/**
 * Parse a PaddleSeg training/eval log line.
 * Train: ... [TRAIN] epoch: 1, iter: 10/1000, loss: 0.52, lr: 0.0099, batch_cost: 0.34, reader_cost: 0.01, ips: 11.5 samples/sec | ETA 00:05:23
 * Eval:  ... [EVAL] #Images: 76 mIoU: 0.8923 Acc: 0.9856 Kappa: 0.8123 Dice: 0.9234
 */
export function parsePaddleSegLog(line: string): ParsedLog | null {
  if (!line.trim()) return null

  const f = (re: RegExp): number | null => {
    const m = line.match(re)
    return m ? parseFloat(m[1]) : null
  }

  const base: ParsedLog = {
    timestamp: '',
    epoch: 0,
    iteration: 0,
    totalIter: 0,
    learningRate: null,
    loss: null,
    lossCls: null,
    lossIou: null,
    lossDfl: null,
    lossL1: null,
    eta: null,
    batchCost: null,
    dataCost: null,
    ips: null,
    memReserved: null,
    memAllocated: null,
    rawLog: line.trim(),
  }

  // [EVAL] metrics line
  if (/\[EVAL\]/i.test(line) && /mIoU/i.test(line)) {
    return {
      ...base,
      mIoU: f(/mIoU:\s*([\d.]+)/i),
      acc: f(/Acc:\s*([\d.]+)/i),
      kappa: f(/Kappa:\s*([\d.]+)/i),
      dice: f(/Dice:\s*([\d.]+)/i),
    }
  }

  // [TRAIN] progress line
  if (/\[TRAIN\]/i.test(line)) {
    const epochMatch = line.match(/epoch:\s*(\d+)/i)
    const iterMatch = line.match(/iter:\s*(\d+)\/(\d+)/i)
    const etaMatch = line.match(/ETA\s*(\d+:\d{2}:\d{2})/i)
    return {
      ...base,
      epoch: epochMatch ? parseInt(epochMatch[1], 10) : 0,
      iteration: iterMatch ? parseInt(iterMatch[1], 10) : 0,
      totalIter: iterMatch ? parseInt(iterMatch[2], 10) : 0,
      learningRate: f(/lr:\s*([\d.e-]+)/i),
      loss: f(/loss:\s*([\d.]+)/i),
      batchCost: f(/batch_cost:\s*([\d.]+)/i),
      readerCost: f(/reader_cost:\s*([\d.]+)/i),
      ips: f(/ips:\s*([\d.]+)/i),
      eta: etaMatch ? etaMatch[1] : null,
    }
  }

  return null
}

/**
 * Extract a float value from a log line
 */
function extractFloat(line: string, key: string): number | null {
  const regex = new RegExp(`${key}:\\s*([\\d.]+)`)
  const match = line.match(regex)
  return match ? parseFloat(match[1]) : null
}

/**
 * Format a number to fixed decimal places
 */
export function formatNumber(num: number | null, decimals: number = 6): string {
  if (num === null) return 'N/A'
  return num.toFixed(decimals)
}

/**
 * Format memory in MB to human readable format
 */
export function formatMemory(mb: number | null): string {
  if (mb === null) return 'N/A'
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(2)} GB`
  }
  return `${mb} MB`
}

/**
 * Calculate average from an array of numbers
 */
export function calculateAverage(values: (number | null)[]): number | null {
  const validValues = values.filter((v): v is number => v !== null)
  if (validValues.length === 0) return null
  return validValues.reduce((sum, v) => sum + v, 0) / validValues.length
}

/**
 * Get min/max from an array of numbers
 */
export function getMinMax(values: (number | null)[]): { min: number | null; max: number | null } {
  const validValues = values.filter((v): v is number => v !== null)
  if (validValues.length === 0) return { min: null, max: null }
  return {
    min: Math.min(...validValues),
    max: Math.max(...validValues)
  }
}
