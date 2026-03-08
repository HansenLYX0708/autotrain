/**
 * Training Execution Module
 * Handles training job execution with mock log generation for demo purposes
 */

import { EventEmitter } from 'events'
import { v4 as uuidv4 } from 'uuid'
import { parsePaddleDetectionLog, type ParsedLog } from './log-parser.js'

export interface TrainingJobConfig {
  jobId: string
  epochs: number
  batchSize: number
  baseLr: number
  totalIterations: number
  warmupIterations: number
  outputDir: string
  command?: string
}

export interface TrainingProgress {
  jobId: string
  epoch: number
  iteration: number
  totalIter: number
  loss: number
  learningRate: number
  eta: string
  timestamp: Date
}

export type TrainingStatus = 'pending' | 'running' | 'completed' | 'failed' | 'stopped'

export interface TrainingState {
  jobId: string
  status: TrainingStatus
  config: TrainingJobConfig
  currentEpoch: number
  currentIteration: number
  startTime: Date | null
  endTime: Date | null
  error: string | null
}

/**
 * Generate a realistic PaddleDetection log line
 */
export function generateMockLog(
  epoch: number,
  iteration: number,
  totalIter: number,
  baseLr: number,
  warmupIterations: number,
  epochLoss: number
): string {
  const now = new Date()
  const dateStr = `${String(now.getMonth() + 1).padStart(2, '0')}/${String(now.getDate()).padStart(2, '0')}`
  const timeStr = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`

  // Calculate learning rate with warmup
  let lr: number
  if (iteration < warmupIterations) {
    // Linear warmup
    lr = baseLr * (iteration / warmupIterations)
  } else {
    // Cosine decay (simplified)
    const progress = (iteration - warmupIterations) / (totalIter - warmupIterations)
    lr = baseLr * 0.5 * (1 + Math.cos(Math.PI * progress))
  }

  // Generate loss values with some randomness and decay
  const progress = iteration / totalIter
  const baseLoss = epochLoss * (1 - progress * 0.6) // Loss decreases over time
  const noise = (Math.random() - 0.5) * 0.2
  
  const loss = baseLoss + noise
  const lossCls = baseLoss * 0.5 + (Math.random() - 0.5) * 0.3
  const lossIou = baseLoss * 0.1 + (Math.random() - 0.5) * 0.05
  const lossDfl = baseLoss * 0.3 + (Math.random() - 0.5) * 0.2
  const lossL1 = baseLoss * 0.15 + (Math.random() - 0.5) * 0.1

  // Calculate ETA (simplified)
  const remainingIters = totalIter - iteration
  const avgIterTime = 2.5 // seconds per iteration
  const remainingSeconds = remainingIters * avgIterTime
  const eta = formatETA(remainingSeconds)

  // Random performance metrics
  const batchCost = 2.0 + Math.random() * 1.0
  const dataCost = 1.5 + Math.random() * 0.8
  const ips = 3.5 + Math.random() * 2.0
  const memReserved = 13000 + Math.floor(Math.random() * 2000)
  const memAllocated = 10000 + Math.floor(Math.random() * 3000)

  return `[${dateStr} ${timeStr}] ppdet.engine.callbacks INFO: Epoch: [${epoch}] [${iteration}/${totalIter}] learning_rate: ${lr.toFixed(6)} loss: ${loss.toFixed(6)} loss_cls: ${lossCls.toFixed(6)} loss_iou: ${lossIou.toFixed(6)} loss_dfl: ${lossDfl.toFixed(6)} loss_l1: ${lossL1.toFixed(6)} eta: ${eta} batch_cost: ${batchCost.toFixed(4)} data_cost: ${dataCost.toFixed(4)} ips: ${ips.toFixed(4)} images/s, max_mem_reserved: ${memReserved} MB, max_mem_allocated: ${memAllocated} MB`
}

/**
 * Format seconds to human readable ETA
 */
function formatETA(seconds: number): string {
  if (seconds < 0) return '00:00:00'

  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)

  const timeStr = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`

  if (days > 0) {
    return `${days} day${days > 1 ? 's' : ''}, ${timeStr}`
  }
  return timeStr
}

/**
 * Training execution class
 */
export class Trainer extends EventEmitter {
  private jobs: Map<string, TrainingState> = new Map()
  private intervals: Map<string, NodeJS.Timeout> = new Map()

  /**
   * Start a new training job
   */
  startTraining(config: TrainingJobConfig): TrainingState {
    // Check if job is already running
    if (this.jobs.has(config.jobId)) {
      const existingJob = this.jobs.get(config.jobId)!
      if (existingJob.status === 'running') {
        return existingJob
      }
    }

    const state: TrainingState = {
      jobId: config.jobId,
      status: 'running',
      config,
      currentEpoch: 0,
      currentIteration: 0,
      startTime: new Date(),
      endTime: null,
      error: null
    }

    this.jobs.set(config.jobId, state)
    this.runTrainingLoop(state)

    return state
  }

  /**
   * Run the training loop (mock implementation)
   */
  private runTrainingLoop(state: TrainingState): void {
    const { config } = state
    const iterationsPerEpoch = Math.floor(config.totalIterations / config.epochs)
    let iteration = 0
    let epoch = 0
    let iterInEpoch = 0

    // Base loss for this training run (randomized)
    const baseLoss = 3.0 + Math.random() * 2.0

    // Log interval (every N iterations)
    const logInterval = 50

    // Simulate training with interval
    const interval = setInterval(() => {
      // Check if job was stopped
      const currentState = this.jobs.get(config.jobId)
      if (!currentState || currentState.status === 'stopped' || currentState.status === 'failed') {
        clearInterval(interval)
        this.intervals.delete(config.jobId)
        return
      }

      iteration++
      iterInEpoch++

      // Check epoch completion
      if (iterInEpoch >= iterationsPerEpoch) {
        epoch++
        iterInEpoch = 0
        state.currentEpoch = epoch

        // Emit epoch completion
        this.emit('epoch:complete', {
          jobId: config.jobId,
          epoch,
          timestamp: new Date()
        })
      }

      state.currentIteration = iteration

      // Generate and emit log at intervals
      if (iteration % logInterval === 0 || iteration === 1) {
        const logLine = generateMockLog(
          epoch,
          iteration,
          config.totalIterations,
          config.baseLr,
          config.warmupIterations,
          baseLoss
        )

        const parsedLog = parsePaddleDetectionLog(logLine)

        // Emit raw log
        this.emit('log', {
          jobId: config.jobId,
          log: logLine,
          parsed: parsedLog
        })

        // Emit progress update
        if (parsedLog) {
          this.emit('progress', {
            jobId: config.jobId,
            epoch: parsedLog.epoch,
            iteration: parsedLog.iteration,
            totalIter: parsedLog.totalIter,
            loss: parsedLog.loss,
            learningRate: parsedLog.learningRate,
            eta: parsedLog.eta,
            timestamp: new Date()
          })
        }
      }

      // Check training completion
      if (iteration >= config.totalIterations) {
        clearInterval(interval)
        this.intervals.delete(config.jobId)
        
        state.status = 'completed'
        state.endTime = new Date()
        state.currentEpoch = config.epochs

        // Emit completion
        this.emit('complete', {
          jobId: config.jobId,
          status: 'completed',
          startTime: state.startTime,
          endTime: state.endTime,
          totalIterations: iteration
        })
      }
    }, 100) // Run every 100ms for demo purposes

    this.intervals.set(config.jobId, interval)
  }

  /**
   * Stop a running training job
   */
  stopTraining(jobId: string): TrainingState | null {
    const state = this.jobs.get(jobId)
    if (!state) return null

    // Clear the interval
    const interval = this.intervals.get(jobId)
    if (interval) {
      clearInterval(interval)
      this.intervals.delete(jobId)
    }

    state.status = 'stopped'
    state.endTime = new Date()

    // Emit stop event
    this.emit('stopped', {
      jobId,
      status: 'stopped',
      endTime: state.endTime
    })

    return state
  }

  /**
   * Get the status of a training job
   */
  getJobStatus(jobId: string): TrainingState | null {
    return this.jobs.get(jobId) || null
  }

  /**
   * Check if a job is currently running
   */
  isJobRunning(jobId: string): boolean {
    const state = this.jobs.get(jobId)
    return state?.status === 'running'
  }

  /**
   * Clean up resources for a job
   */
  cleanupJob(jobId: string): void {
    this.stopTraining(jobId)
    this.jobs.delete(jobId)
  }

  /**
   * Clean up all jobs
   */
  cleanup(): void {
    for (const [jobId] of this.intervals) {
      this.stopTraining(jobId)
    }
    this.jobs.clear()
  }
}

// Export singleton instance
export const trainer = new Trainer()
