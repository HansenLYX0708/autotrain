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
  rawLog: string
}

/**
 * Parse a PaddleDetection training log line
 * Example: [03/03 10:20:46] ppdet.engine.callbacks INFO: Epoch: [0] [100/827] learning_rate: 0.000024 loss: 4.534513 ...
 */
export function parsePaddleDetectionLog(line: string): ParsedLog | null {
  // Skip empty lines
  if (!line.trim()) {
    return null
  }

  // Match PaddleDetection log format
  const timestampMatch = line.match(/^\[(\d{2}\/\d{2} \d{2}:\d{2}:\d{2})\]/)
  const timestamp = timestampMatch ? timestampMatch[1] : ''

  // Match epoch: Epoch: [0]
  const epochMatch = line.match(/Epoch:\s*\[(\d+)\]/)
  const epoch = epochMatch ? parseInt(epochMatch[1], 10) : 0

  // Match iteration: [100/827]
  const iterMatch = line.match(/\[(\d+)\/(\d+)\]/)
  const iteration = iterMatch ? parseInt(iterMatch[1], 10) : 0
  const totalIter = iterMatch ? parseInt(iterMatch[2], 10) : 0

  // Skip lines that don't have training progress info
  if (!iterMatch) {
    return null
  }

  // Extract metrics using helper function
  const learningRate = extractFloat(line, 'learning_rate')
  const loss = extractFloat(line, 'loss')
  const lossCls = extractFloat(line, 'loss_cls')
  const lossIou = extractFloat(line, 'loss_iou')
  const lossDfl = extractFloat(line, 'loss_dfl')
  const lossL1 = extractFloat(line, 'loss_l1')

  // Extract ETA (format: "1 day, 20:03:57" or "20:03:57")
  const etaMatch = line.match(/eta:\s*([^,]+(?:,\s*\d{1,2}:\d{2}:\d{2})?)/)
  const eta = etaMatch ? etaMatch[1].trim() : null

  // Extract costs
  const batchCost = extractFloat(line, 'batch_cost')
  const dataCost = extractFloat(line, 'data_cost')

  // Extract IPS (images per second)
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
