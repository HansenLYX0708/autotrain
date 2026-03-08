/**
 * Training Service - Main Entry Point
 * Socket.io server for real-time training log streaming
 * 
 * Port: 3003
 * 
 * Events to handle:
 * - training:start - Start a new training job
 * - training:stop - Stop a running job
 * - training:status - Get job status
 * 
 * Events to emit:
 * - training:log - Stream log lines
 * - training:progress - Progress updates
 * - training:complete - Training finished
 * - training:error - Training failed
 */

import { createServer } from 'http'
import { Server, Socket } from 'socket.io'
import { trainer, type TrainingJobConfig, type TrainingProgress } from './trainer.js'
import { parsePaddleDetectionLog, type ParsedLog } from './log-parser.js'

// Prisma client import from main project
// The mini-service shares the same database with the main Next.js app
// @ts-ignore - Importing from main project's generated Prisma client
import { PrismaClient } from '/home/z/my-project/node_modules/.prisma/client'

const prisma = new PrismaClient()

// HTTP Server setup
const httpServer = createServer()

// Socket.io server setup
const io = new Server(httpServer, {
  path: '/',
  cors: {
    origin: ['http://localhost:3000', 'http://127.0.0.1:3000'],
    methods: ['GET', 'POST'],
    credentials: true
  },
  pingTimeout: 60000,
  pingInterval: 25000
})

// Log interface for database
interface LogEntry {
  jobId: string
  epoch: number
  iteration: number
  totalIter: number
  loss: number | null
  lossCls: number | null
  lossIou: number | null
  lossDfl: number | null
  lossL1: number | null
  learningRate: number | null
  eta: string | null
  batchCost: number | null
  dataCost: number | null
  ips: number | null
  memReserved: number | null
  memAllocated: number | null
  rawLog: string
}

// Track socket subscriptions to jobs
const socketJobs = new Map<string, Set<string>>()

/**
 * Subscribe a socket to job updates
 */
function subscribeToJob(socket: Socket, jobId: string): void {
  if (!socketJobs.has(socket.id)) {
    socketJobs.set(socket.id, new Set())
  }
  socketJobs.get(socket.id)!.add(jobId)
  socket.join(`job:${jobId}`)
  console.log(`[Socket ${socket.id}] Subscribed to job: ${jobId}`)
}

/**
 * Unsubscribe a socket from job updates
 */
function unsubscribeFromJob(socket: Socket, jobId: string): void {
  const jobs = socketJobs.get(socket.id)
  if (jobs) {
    jobs.delete(jobId)
    socket.leave(`job:${jobId}`)
    console.log(`[Socket ${socket.id}] Unsubscribed from job: ${jobId}`)
  }
}

/**
 * Update job status in database
 */
async function updateJobStatus(
  jobId: string, 
  status: string, 
  progress?: { epoch?: number; loss?: number; lr?: number }
): Promise<void> {
  try {
    const updateData: Record<string, unknown> = { status }
    
    if (status === 'running') {
      updateData.startedAt = new Date()
    } else if (status === 'completed' || status === 'failed' || status === 'stopped') {
      updateData.completedAt = new Date()
    }
    
    if (progress) {
      if (progress.epoch !== undefined) updateData.currentEpoch = progress.epoch
      if (progress.loss !== undefined) updateData.currentLoss = progress.loss
      if (progress.lr !== undefined) updateData.currentLr = progress.lr
    }
    
    await prisma.trainingJob.update({
      where: { id: jobId },
      data: updateData
    })
    console.log(`[DB] Updated job ${jobId} status to: ${status}`)
  } catch (error) {
    console.error(`[DB] Failed to update job ${jobId} status:`, error)
  }
}

/**
 * Insert log entry into database
 */
async function insertLogEntry(log: LogEntry): Promise<void> {
  try {
    await prisma.trainingLog.create({
      data: {
        jobId: log.jobId,
        epoch: log.epoch,
        iteration: log.iteration,
        totalIter: log.totalIter,
        loss: log.loss,
        lossCls: log.lossCls,
        lossIou: log.lossIou,
        lossDfl: log.lossDfl,
        lossL1: log.lossL1,
        learningRate: log.learningRate,
        eta: log.eta,
        batchCost: log.batchCost,
        dataCost: log.dataCost,
        ips: log.ips,
        memReserved: log.memReserved,
        memAllocated: log.memAllocated,
        rawLog: log.rawLog
      }
    })
  } catch (error) {
    console.error(`[DB] Failed to insert log entry:`, error)
  }
}

// Setup trainer event handlers
trainer.on('log', async (data: { jobId: string; log: string; parsed: ParsedLog | null }) => {
  const { jobId, log, parsed } = data
  
  // Emit log to all clients subscribed to this job
  io.to(`job:${jobId}`).emit('training:log', {
    jobId,
    log,
    parsed,
    timestamp: new Date().toISOString()
  })
  
  // Insert log into database
  if (parsed) {
    await insertLogEntry({
      jobId,
      epoch: parsed.epoch,
      iteration: parsed.iteration,
      totalIter: parsed.totalIter,
      loss: parsed.loss,
      lossCls: parsed.lossCls,
      lossIou: parsed.lossIou,
      lossDfl: parsed.lossDfl,
      lossL1: parsed.lossL1,
      learningRate: parsed.learningRate,
      eta: parsed.eta,
      batchCost: parsed.batchCost,
      dataCost: parsed.dataCost,
      ips: parsed.ips,
      memReserved: parsed.memReserved,
      memAllocated: parsed.memAllocated,
      rawLog: log
    })
  }
})

trainer.on('progress', async (data: TrainingProgress) => {
  const { jobId, epoch, iteration, totalIter, loss, learningRate, eta } = data
  
  // Emit progress to all clients subscribed to this job
  io.to(`job:${jobId}`).emit('training:progress', {
    jobId,
    epoch,
    iteration,
    totalIter,
    loss,
    learningRate,
    eta,
    timestamp: new Date().toISOString()
  })
  
  // Update job progress in database
  await updateJobStatus(jobId, 'running', { epoch, loss: loss ?? undefined, lr: learningRate ?? undefined })
})

trainer.on('epoch:complete', async (data: { jobId: string; epoch: number; timestamp: Date }) => {
  const { jobId, epoch, timestamp } = data
  
  io.to(`job:${jobId}`).emit('training:epoch', {
    jobId,
    epoch,
    timestamp: timestamp.toISOString()
  })
  
  console.log(`[Training] Job ${jobId} completed epoch ${epoch}`)
})

trainer.on('complete', async (data: { jobId: string; status: string; startTime: Date | null; endTime: Date | null; totalIterations: number }) => {
  const { jobId, status, startTime, endTime, totalIterations } = data
  
  // Update job status
  await updateJobStatus(jobId, 'completed')
  
  // Emit completion to all clients
  io.to(`job:${jobId}`).emit('training:complete', {
    jobId,
    status,
    startTime: startTime?.toISOString(),
    endTime: endTime?.toISOString(),
    totalIterations,
    timestamp: new Date().toISOString()
  })
  
  console.log(`[Training] Job ${jobId} completed with ${totalIterations} iterations`)
})

trainer.on('stopped', async (data: { jobId: string; status: string; endTime: Date | null }) => {
  const { jobId, status, endTime } = data
  
  // Update job status
  await updateJobStatus(jobId, 'stopped')
  
  // Emit stop event to all clients
  io.to(`job:${jobId}`).emit('training:stopped', {
    jobId,
    status,
    endTime: endTime?.toISOString(),
    timestamp: new Date().toISOString()
  })
  
  console.log(`[Training] Job ${jobId} stopped`)
})

// Socket.io connection handler
io.on('connection', (socket: Socket) => {
  console.log(`[Socket] Client connected: ${socket.id}`)
  
  // Initialize socket jobs set
  socketJobs.set(socket.id, new Set())

  // Handle training:start event
  socket.on('training:start', async (data: { jobId: string; config?: Partial<TrainingJobConfig> }) => {
    const { jobId, config = {} } = data
    console.log(`[Socket] training:start received for job: ${jobId}`)
    
    try {
      // Check if job exists in database
      const job = await prisma.trainingJob.findUnique({
        where: { id: jobId },
        include: {
          project: true,
          dataset: true,
          model: true,
          config: true
        }
      })
      
      if (!job) {
        socket.emit('training:error', {
          jobId,
          error: 'Job not found in database',
          timestamp: new Date().toISOString()
        })
        return
      }
      
      // Check if job is already running
      if (trainer.isJobRunning(jobId)) {
        socket.emit('training:error', {
          jobId,
          error: 'Job is already running',
          timestamp: new Date().toISOString()
        })
        return
      }
      
      // Subscribe client to job updates
      subscribeToJob(socket, jobId)
      
      // Build training config from job data
      const trainingConfig: TrainingJobConfig = {
        jobId,
        epochs: job.config?.epoch ?? 100,
        batchSize: job.config?.batchSize ?? 8,
        baseLr: job.config?.baseLr ?? 0.001,
        totalIterations: config.totalIterations ?? 1000, // Demo default
        warmupIterations: Math.floor((config.totalIterations ?? 1000) * 0.05),
        outputDir: job.outputDir ?? `./outputs/${jobId}`,
        command: job.command ?? undefined
      }
      
      // Update job status to running
      await updateJobStatus(jobId, 'running')
      
      // Start training
      const state = trainer.startTraining(trainingConfig)
      
      // Emit acknowledgment
      socket.emit('training:started', {
        jobId,
        status: state.status,
        config: trainingConfig,
        timestamp: new Date().toISOString()
      })
      
    } catch (error) {
      console.error(`[Socket] Error starting training for job ${jobId}:`, error)
      socket.emit('training:error', {
        jobId,
        error: error instanceof Error ? error.message : 'Failed to start training',
        timestamp: new Date().toISOString()
      })
    }
  })

  // Handle training:stop event
  socket.on('training:stop', async (data: { jobId: string }) => {
    const { jobId } = data
    console.log(`[Socket] training:stop received for job: ${jobId}`)
    
    try {
      const state = trainer.stopTraining(jobId)
      
      if (!state) {
        socket.emit('training:error', {
          jobId,
          error: 'Job not found or not running',
          timestamp: new Date().toISOString()
        })
        return
      }
      
      // Already handled by trainer event handler
      console.log(`[Socket] Job ${jobId} stopped successfully`)
      
    } catch (error) {
      console.error(`[Socket] Error stopping training for job ${jobId}:`, error)
      socket.emit('training:error', {
        jobId,
        error: error instanceof Error ? error.message : 'Failed to stop training',
        timestamp: new Date().toISOString()
      })
    }
  })

  // Handle training:status event
  socket.on('training:status', async (data: { jobId: string }) => {
    const { jobId } = data
    console.log(`[Socket] training:status received for job: ${jobId}`)
    
    try {
      // Get status from trainer
      const state = trainer.getJobStatus(jobId)
      
      if (state) {
        socket.emit('training:status', {
          jobId,
          status: state.status,
          currentEpoch: state.currentEpoch,
          currentIteration: state.currentIteration,
          startTime: state.startTime?.toISOString(),
          endTime: state.endTime?.toISOString(),
          error: state.error,
          timestamp: new Date().toISOString()
        })
      } else {
        // Get status from database
        const job = await prisma.trainingJob.findUnique({
          where: { id: jobId },
          select: {
            status: true,
            currentEpoch: true,
            currentLoss: true,
            currentLr: true,
            startedAt: true,
            completedAt: true
          }
        })
        
        if (job) {
          socket.emit('training:status', {
            jobId,
            status: job.status,
            currentEpoch: job.currentEpoch,
            currentLoss: job.currentLoss,
            currentLr: job.currentLr,
            startTime: job.startedAt?.toISOString(),
            endTime: job.completedAt?.toISOString(),
            timestamp: new Date().toISOString()
          })
        } else {
          socket.emit('training:error', {
            jobId,
            error: 'Job not found',
            timestamp: new Date().toISOString()
          })
        }
      }
    } catch (error) {
      console.error(`[Socket] Error getting status for job ${jobId}:`, error)
      socket.emit('training:error', {
        jobId,
        error: error instanceof Error ? error.message : 'Failed to get job status',
        timestamp: new Date().toISOString()
      })
    }
  })

  // Handle training:subscribe event
  socket.on('training:subscribe', (data: { jobId: string }) => {
    const { jobId } = data
    subscribeToJob(socket, jobId)
    socket.emit('training:subscribed', { jobId })
  })

  // Handle training:unsubscribe event
  socket.on('training:unsubscribe', (data: { jobId: string }) => {
    const { jobId } = data
    unsubscribeFromJob(socket, jobId)
    socket.emit('training:unsubscribed', { jobId })
  })

  // Handle disconnect
  socket.on('disconnect', () => {
    console.log(`[Socket] Client disconnected: ${socket.id}`)
    
    // Clean up socket subscriptions
    socketJobs.delete(socket.id)
  })

  // Handle errors
  socket.on('error', (error) => {
    console.error(`[Socket] Error on ${socket.id}:`, error)
  })
})

// Start server
const PORT = 3003
httpServer.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════════════════════════╗
║           Training Service - Socket.io Server              ║
╠════════════════════════════════════════════════════════════╣
║  Port: ${PORT}                                                 ║
║  Status: Running                                            ║
║  CORS: localhost:3000                                       ║
╚════════════════════════════════════════════════════════════╝

Available Events:
  Client -> Server:
    - training:start    : Start a new training job
    - training:stop     : Stop a running job
    - training:status   : Get job status
    - training:subscribe: Subscribe to job updates
    - training:unsubscribe: Unsubscribe from job updates

  Server -> Client:
    - training:started  : Job started successfully
    - training:log      : Stream log lines
    - training:progress : Progress updates
    - training:epoch    : Epoch completed
    - training:complete : Training finished
    - training:stopped  : Training stopped
    - training:error    : Training failed
`)
})

// Graceful shutdown
async function gracefulShutdown(signal: string) {
  console.log(`\n[Server] Received ${signal}, shutting down...`)
  
  // Cleanup all training jobs
  trainer.cleanup()
  
  // Close all socket connections
  io.close(() => {
    console.log('[Server] Socket.io server closed')
  })
  
  // Close HTTP server
  httpServer.close(() => {
    console.log('[Server] HTTP server closed')
  })
  
  // Disconnect Prisma
  await prisma.$disconnect()
  console.log('[Server] Database disconnected')
  
  process.exit(0)
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'))
process.on('SIGINT', () => gracefulShutdown('SIGINT'))

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  console.error('[Server] Uncaught exception:', error)
})

process.on('unhandledRejection', (reason) => {
  console.error('[Server] Unhandled rejection:', reason)
})
