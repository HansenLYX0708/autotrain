# Training Service

A standalone Bun service that handles training jobs with real-time log streaming via Socket.io.

## Directory Structure

```
training-service/
├── index.ts          # Main entry point with socket.io server
├── package.json      # Dependencies
├── trainer.ts        # Training execution logic
├── log-parser.ts     # PaddleDetection log parser
└── outputs/          # Output directory for training results
```

## Running the Service

```bash
# Development mode (with hot reload)
bun run dev

# Production mode
bun run start
```

## Configuration

- **Port**: 3003
- **CORS**: Configured for `localhost:3000`
- **Database**: Uses the same SQLite database as the main Next.js app

## Socket.io Events

### Client -> Server Events

| Event | Description | Payload |
|-------|-------------|---------|
| `training:start` | Start a new training job | `{ jobId: string, config?: Partial<TrainingJobConfig> }` |
| `training:stop` | Stop a running job | `{ jobId: string }` |
| `training:status` | Get job status | `{ jobId: string }` |
| `training:subscribe` | Subscribe to job updates | `{ jobId: string }` |
| `training:unsubscribe` | Unsubscribe from job updates | `{ jobId: string }` |

### Server -> Client Events

| Event | Description | Payload |
|-------|-------------|---------|
| `training:started` | Job started successfully | `{ jobId, status, config, timestamp }` |
| `training:log` | Stream log lines | `{ jobId, log, parsed, timestamp }` |
| `training:progress` | Progress updates | `{ jobId, epoch, iteration, totalIter, loss, learningRate, eta, timestamp }` |
| `training:epoch` | Epoch completed | `{ jobId, epoch, timestamp }` |
| `training:complete` | Training finished | `{ jobId, status, startTime, endTime, totalIterations, timestamp }` |
| `training:stopped` | Training stopped | `{ jobId, status, endTime, timestamp }` |
| `training:error` | Training failed | `{ jobId, error, timestamp }` |
| `training:subscribed` | Subscription confirmed | `{ jobId }` |
| `training:unsubscribed` | Unsubscription confirmed | `{ jobId }` |

## Example Usage

```typescript
import { io } from 'socket.io-client'

const socket = io('http://localhost:3003')

// Start training
socket.emit('training:start', {
  jobId: 'my-training-job-id',
  config: {
    totalIterations: 1000
  }
})

// Listen for logs
socket.on('training:log', (data) => {
  console.log('Log:', data.log)
  console.log('Parsed:', data.parsed)
})

// Listen for progress
socket.on('training:progress', (data) => {
  console.log(`Epoch ${data.epoch}, Iteration ${data.iteration}/${data.totalIter}`)
  console.log(`Loss: ${data.loss}, LR: ${data.learningRate}`)
})

// Listen for completion
socket.on('training:complete', (data) => {
  console.log('Training completed:', data)
})
```

## Log Format

The service generates mock PaddleDetection training logs in the following format:

```
[03/03 10:20:46] ppdet.engine.callbacks INFO: Epoch: [0] [100/827] learning_rate: 0.000024 loss: 4.534513 loss_cls: 2.415748 loss_iou: 0.530099 loss_dfl: 1.626410 loss_l1: 1.266741 eta: 1 day, 20:03:57 batch_cost: 2.3920 data_cost: 2.1271 ips: 4.1807 images/s, max_mem_reserved: 14321 MB, max_mem_allocated: 12036 MB
```

## Database Integration

The service updates the following database tables:

- `TrainingJob`: Updates status, progress, and timing fields
- `TrainingLog`: Inserts parsed log entries with metrics

## Demo Mode

For demo purposes, the training is simulated with mock logs since we don't have actual PaddleDetection installed. The simulation:

1. Generates realistic PaddleDetection log format
2. Simulates loss decay over iterations
3. Implements learning rate warmup and cosine decay
4. Reports memory and performance metrics

To connect to real training, modify `trainer.ts` to spawn actual training processes.
