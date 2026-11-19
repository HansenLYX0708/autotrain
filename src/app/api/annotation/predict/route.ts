import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs/promises'
import path from 'path'
import { spawn } from 'child_process'
import { randomUUID } from 'crypto'
import sharp from 'sharp'
import { parse, stringify } from 'yaml'
import { requireAdmin } from '@/lib/auth'
import { db } from '@/lib/db'
import { getWorkDir, isLocalization, resolvePythonPath } from '@/lib/frameworks'
import { bestWeightsPath } from '@/lib/job-commands'
import { isInside, toSafeSlug } from '@/lib/safe-path'
import { paddleDetectionWeights } from '@/lib/paddledet-weights'

export const runtime = 'nodejs'
export const maxDuration = 600

type CheckpointJob = { name: string; outputDir: string | null; project: { framework: string; user: { username: string } | null } }
type CheckpointSystem = { userDatabasePath?: string | null } | null

/**
 * Candidate save_dirs, in order: the recorded outputDir (absolute, or legacy
 * PaddleDetection paths relative to the PD repo), then the platform's standard
 * job directory {userDatabasePath}/{owner}/jobs/{jobName} — legacy PD records
 * still say `output/<project>/<job>` although their weights live there now.
 */
function jobOutputDirs(job: CheckpointJob, system: CheckpointSystem, pdWorkDir: string | null | undefined): string[] {
  const dirs: string[] = []
  const { outputDir } = job
  if (outputDir && path.isAbsolute(outputDir)) dirs.push(outputDir)
  else if (outputDir && job.project.framework === 'PaddleDetection' && pdWorkDir) dirs.push(path.join(pdWorkDir, outputDir))
  if (system?.userDatabasePath && job.project.user?.username) dirs.push(path.join(system.userDatabasePath, job.project.user.username, 'jobs', job.name))
  return [...new Set(dirs)]
}

async function jobCheckpoint(job: CheckpointJob, system: CheckpointSystem, pdWorkDir: string | null | undefined): Promise<string | null> {
  const { framework } = job.project
  for (const dir of jobOutputDirs(job, system, pdWorkDir)) {
    if (framework === 'PaddleDetection') {
      const weights = await paddleDetectionWeights(dir)
      if (weights) return weights
    } else if ((await fs.stat(bestWeightsPath(framework, dir)).catch(() => null))?.isFile()) return bestWeightsPath(framework, dir)
  }
  return null
}

export async function GET(request: NextRequest) {
  const auth = await requireAdmin(request)
  if (auth instanceof NextResponse) return auth
  const system = await db.systemConfig.findFirst()
  const pdWorkDir = getWorkDir('PaddleDetection', system)
  const jobs = await db.trainingJob.findMany({
    where: { status: { in: ['completed', 'stopped'] }, project: { framework: { in: ['TorchOBB', 'TorchPose', 'PaddleDetection'] } } },
    include: { project: { select: { framework: true, name: true, user: { select: { username: true } } } } },
    orderBy: { createdAt: 'desc' }, take: 100,
  })
  const data: { id: string; name: string; framework: string; projectName: string; checkpoint: string; bestMetric: number | null }[] = []
  for (const job of jobs) {
    const checkpoint = await jobCheckpoint(job, system, pdWorkDir)
    if (!checkpoint) continue
    data.push({ id: job.id, name: job.name, framework: job.project.framework, projectName: job.project.name, checkpoint: path.basename(checkpoint), bestMetric: job.bestMetric })
  }
  return NextResponse.json({ success: true, data })
}

export async function POST(request: NextRequest) {
  const auth = await requireAdmin(request)
  if (auth instanceof NextResponse) return auth
  let temporary: string | undefined
  const predictionRoot = path.join(process.cwd(), 'temp', 'localization-predict')
  try {
    const form = await request.formData()
    const jobId = String(form.get('jobId') || '')
    const confidence = Number(form.get('confidence') ?? 0.25)
    if (!Number.isFinite(confidence) || confidence < 0.01 || confidence > 0.99) throw new Error('Confidence must be between 0.01 and 0.99.')
    const files = form.getAll('images')
    const names: unknown = JSON.parse(String(form.get('names') || '[]'))
    if (!files.length || files.length > 16 || !Array.isArray(names) || names.length !== files.length || names.some(name => typeof name !== 'string')) throw new Error('Supply 1–16 images and corresponding names.')
    if (files.some(file => !(file instanceof File)) || files.reduce((sum, file) => sum + (file as File).size, 0) > 128 * 1024 * 1024) throw new Error('Prediction batch exceeds 128 MB.')
    const job = await db.trainingJob.findUnique({ where: { id: jobId }, include: { project: { include: { user: { select: { username: true } } } } } })
    const framework = job?.project.framework ?? ''
    const isPd = framework === 'PaddleDetection'
    if (!job || !(isPd || isLocalization(framework)) || !['completed', 'stopped'].includes(job.status) || !job.outputDir || !job.yamlConfig) throw new Error('Choose a completed PaddleDetection or localization training job.')
    const system = await db.systemConfig.findFirstOrThrow()
    const pdWorkDir = getWorkDir('PaddleDetection', system)
    const checkpoint = await jobCheckpoint(job, system, pdWorkDir)
    if (!checkpoint) throw new Error(`No usable checkpoint was found for this job (looked in: ${jobOutputDirs(job, system, pdWorkDir).join('; ') || 'no resolvable output directory'}).`)
    if (isPd && !pdWorkDir) throw new Error('Configure the PaddleDetection repo path.')
    if (!isPd && !system.torchPath) throw new Error('Configure the PyTorch trainer path.')
    const params = JSON.parse(job.trainingParams || '{}')
    const gpu = String(params.gpuIds || '0').split(',')[0]
    const { pythonPath } = resolvePythonPath(framework, gpu, system)
    if (!pythonPath) throw new Error(`Configure the ${framework} Python environment.`)
    const base = path.join(predictionRoot, toSafeSlug(auth.userId, 'user'))
    await fs.mkdir(base, { recursive: true })
    temporary = path.join(base, randomUUID())
    const input = path.join(temporary, 'images'), output = path.join(temporary, 'predictions')
    await fs.mkdir(input, { recursive: true })
    const mapping = new Map<string, string>()
    for (const [index, item] of files.entries()) {
      const file = item as File
      const extension = path.extname(file.name).toLowerCase()
      if (!['.jpg', '.jpeg', '.png', '.bmp', '.webp'].includes(extension)) throw new Error('Unsupported prediction image format.')
      const name = `${String(index).padStart(4, '0')}${extension}`
      const bytes = Buffer.from(await file.arrayBuffer())
      // BMP is decoded by Pillow in the inference runtime; Sharp does not support it.
      if (extension !== '.bmp') {
        const metadata = await sharp(bytes, { limitInputPixels: 100000000 }).metadata()
        if (!metadata.width || !metadata.height) throw new Error('Could not decode image.')
      }
      await fs.writeFile(path.join(input, name), bytes)
      mapping.set(name, names[index])
    }
    if (isPd) {
      // PaddleDetection configs carry custom YAML tags (!COCODataSet etc.) —
      // never round-trip job.yamlConfig through YAML.parse. Reuse the stored
      // config file when it still exists; otherwise dump the raw text.
      const userConfigsPath = (system as { userConfigsPath?: string | null }).userConfigsPath
      const candidates = [
        job.configPath && path.isAbsolute(job.configPath) ? job.configPath : null,
        userConfigsPath && job.configPath ? path.join(userConfigsPath, job.configPath) : null,
        pdWorkDir && job.configPath ? path.join(pdWorkDir, job.configPath) : null,
      ].filter((candidate): candidate is string => !!candidate)
      let configPath: string | null = null
      for (const candidate of candidates) {
        if ((await fs.stat(candidate).catch(() => null))?.isFile()) { configPath = candidate; break }
      }
      if (!configPath) {
        configPath = path.join(temporary, 'job-config.yaml')
        await fs.writeFile(configPath, job.yamlConfig, 'utf8')
      }
      const jsonOutput = path.join(temporary, 'predictions.json')
      await new Promise<void>((resolve, reject) => {
        const child = spawn(pythonPath, [path.join(process.cwd(), 'tools', 'paddledet_annotate.py'),
          '--ppdet_root', pdWorkDir!, '--config', configPath!, '--weights', checkpoint,
          '--image_dir', input, '--output', jsonOutput, '--confidence', String(confidence)], {
          cwd: pdWorkDir!, windowsHide: true,
          env: { ...process.env, CUDA_VISIBLE_DEVICES: gpu, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
        })
        let log = ''
        const capture = (chunk: Buffer) => { log = (log + chunk.toString('utf8')).slice(-8000) }
        child.stdout.on('data', capture); child.stderr.on('data', capture)
        const timer = setTimeout(() => child.kill(), 600000)
        child.on('error', error => { clearTimeout(timer); reject(error) })
        child.on('close', code => {
          clearTimeout(timer)
          if (code === 0) resolve()
          else reject(new Error(log || 'Prediction failed.'))
        })
      })
      const result = JSON.parse(await fs.readFile(jsonOutput, 'utf8'))
      return NextResponse.json({ success: true, data: {
        jobId, task: result.task, labels: result.labels, keypointNames: result.keypointNames ?? [],
        images: result.images.map((image: { source: string }) => ({ ...image, source: mapping.get(path.basename(image.source)) })),
      } })
    }
    const config = parse(job.yamlConfig)
    config.localization.inference = { ...config.localization.inference, confidence }
    const configPath = path.join(temporary, 'config.yaml')
    await fs.writeFile(configPath, stringify(config), 'utf8')
    await new Promise<void>((resolve, reject) => {
      const child = spawn(pythonPath, [path.join(system.torchPath!, 'tools/localization.py'), 'infer',
        '--config', configPath, '--model_path', checkpoint, '--image_path', input, '--save_dir', output], {
        cwd: system.torchPath!, windowsHide: true,
        env: { ...process.env, CUDA_VISIBLE_DEVICES: gpu, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
      })
      let log = ''
      const capture = (chunk: Buffer) => { log = (log + chunk.toString('utf8')).slice(-8000) }
      child.stdout.on('data', capture); child.stderr.on('data', capture)
      const timer = setTimeout(() => child.kill(), 600000)
      child.on('error', error => { clearTimeout(timer); reject(error) })
      child.on('close', code => {
        clearTimeout(timer)
        if (code === 0) resolve()
        else reject(new Error(log || 'Prediction failed.'))
      })
    })
    const predictions = JSON.parse(await fs.readFile(path.join(output, 'predictions.json'), 'utf8'))
    const schema = JSON.parse(await fs.readFile(path.join(output, 'schema.json'), 'utf8'))
    return NextResponse.json({ success: true, data: {
      jobId, task: schema.task, labels: schema.labels, keypointNames: schema.keypointNames,
      images: predictions.images.map((image: { source: string }) => ({ ...image, source: mapping.get(path.basename(image.source)) })),
    } })
  } catch (error) {
    return NextResponse.json({ success: false, error: error instanceof Error ? error.message : 'Prediction failed' }, { status: 400 })
  } finally {
    if (temporary && isInside(predictionRoot, temporary)) await fs.rm(temporary, { recursive: true, force: true }).catch(() => {})
  }
}
