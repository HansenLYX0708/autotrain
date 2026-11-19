import { NextRequest, NextResponse } from 'next/server'
import path from 'path'
import fs from 'fs/promises'
import { existsSync } from 'fs'
import { randomUUID } from 'crypto'
import { spawn } from 'child_process'
import { stringify } from 'yaml'
import { db } from '@/lib/db'
import { requireAuth } from '@/lib/auth'
import { logActivity } from '@/lib/activity-log'
import { frameworkMeta, resolvePythonPath } from '@/lib/frameworks'
import { readLocalizationDataset } from '@/lib/localization-dataset'
import { projectSchema } from '@/lib/localization-annotations'
import { isInside, toSafeSlug } from '@/lib/safe-path'
import { folderSize } from '@/lib/dataset-merge'

export const runtime = 'nodejs'
export const maxDuration = 600

function runRotateAugment(python: string, script: string, args: string[], cwd: string): Promise<void> {
  return new Promise((resolve, reject) => {
    // AUTOTRAIN_CPU_AFFINITY: pin to E-cores; this host's P-cores crash the box search (0xC0000005).
    const child = spawn(python, [script, 'rotate-augment', ...args], { cwd, windowsHide: true, env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8', AUTOTRAIN_CPU_AFFINITY: 'efficiency' } })
    let output = ''
    const capture = (chunk: Buffer) => { output = (output + chunk.toString('utf8')).slice(-12000) }
    child.stdout.on('data', capture)
    child.stderr.on('data', capture)
    const timer = setTimeout(() => { child.kill(); reject(new Error('rotate-augment timed out.')) }, 540000)
    child.on('error', error => { clearTimeout(timer); reject(error) })
    child.on('close', code => {
      clearTimeout(timer)
      if (code === 0) resolve()
      else reject(new Error(output || `rotate-augment failed (${code})`))
    })
  })
}

/** POST /api/datasets/rotate-augment — {action:'preview'|'create', datasetId, name?, degrees, copies, seed} */
export async function POST(request: NextRequest) {
  try {
    const auth = await requireAuth(request)
    if (auth instanceof NextResponse) return auth
    const body = await request.json()
    const action = body.action === 'create' ? 'create' : 'preview'
    const datasetId = String(body.datasetId ?? '')
    const degrees = Number(body.degrees)
    const copies = Math.floor(Number(body.copies))
    const seed = Math.floor(Number(body.seed ?? 42))
    if (!datasetId || !Number.isFinite(degrees) || degrees <= 0 || degrees > 45 || !(copies >= 1 && copies <= 20) || !Number.isFinite(seed) || seed < 0) {
      return NextResponse.json({ success: false, error: 'Invalid parameters: 0 < degrees ≤ 45, 1 ≤ copies ≤ 20, seed ≥ 0.' }, { status: 400 })
    }
    const dataset = await db.dataset.findUnique({ where: { id: datasetId }, include: { project: { select: { id: true, framework: true } } } })
    if (!dataset) return NextResponse.json({ success: false, error: 'Dataset not found.' }, { status: 404 })
    if (auth.role !== 'admin' && dataset.userId !== auth.userId) {
      return NextResponse.json({ success: false, error: 'You can only augment your own datasets.' }, { status: 403 })
    }
    const meta = frameworkMeta(dataset.project.framework)
    const format = dataset.format || meta.datasetFormat
    if (meta.localizationTask !== 'obb' || format !== 'YOLOOBB' || !dataset.datasetDir) {
      return NextResponse.json({ success: false, error: 'Only managed YOLOOBB datasets can be rotated.' }, { status: 400 })
    }
    const root = path.dirname(dataset.datasetDir) // managed root: project.json + manifest.json + obb/
    if (!existsSync(path.join(root, 'project.json'))) {
      return NextResponse.json({ success: false, error: 'Managed dataset files (project.json) not found.' }, { status: 400 })
    }
    const project = projectSchema.parse(JSON.parse(await fs.readFile(path.join(root, 'project.json'), 'utf-8')))
    const trainImages = project.images.filter(i => i.split === 'train').length
    const trainInstances = project.images.filter(i => i.split === 'train').reduce((n, i) => n + i.objects.length, 0)
    const planned = {
      dataset: { id: dataset.id, name: dataset.name },
      trainImages, trainInstances, copies,
      newImages: trainImages * copies,
      totalImages: project.images.length + trainImages * copies,
      note: 'Only the train split is augmented; copies inherit the source group and split. Enabling online inscribed rotation on top of an offline-rotated dataset rotates twice — keep inscribed_rotate_degrees: 0 for jobs on this dataset.',
    }
    if (action === 'preview') return NextResponse.json({ success: true, data: planned })

    const name = String(body.name ?? '').trim()
    if (!name || name.length > 100 || /[\\/\x00-\x1f]/.test(name) || name.includes('..')) {
      return NextResponse.json({ success: false, error: 'Invalid name: required, 1–100 chars, no path separators or control characters.' }, { status: 400 })
    }
    const owner = await db.user.findUnique({ where: { id: dataset.userId! }, select: { id: true, username: true, maxStorageQuota: true } })
    const config = await db.systemConfig.findFirst()
    if (!owner || !config?.userDatabasePath || !config.torchPath) {
      return NextResponse.json({ success: false, error: 'User storage or the PyTorch trainer path is not configured.' }, { status: 400 })
    }
    const { pythonPath: python } = resolvePythonPath(dataset.project.framework, '0', config as any)
    if (!python) return NextResponse.json({ success: false, error: `Configure the Python environment for ${dataset.project.framework} in Settings.` }, { status: 400 })

    const userRoot = path.join(config.userDatabasePath, owner.username)
    const sourceSize = await folderSize(path.join(root, 'obb'))
    const estimated = Math.ceil(sourceSize * (1 + copies * (trainImages / Math.max(project.images.length, 1))))
    if (await folderSize(userRoot) + estimated > Number(owner.maxStorageQuota)) {
      return NextResponse.json({ success: false, error: 'Storage quota exceeded by this augmentation.' }, { status: 403 })
    }
    const slug = `${toSafeSlug(name, 'localization')}-${randomUUID().slice(0, 8)}`
    const container = path.join(userRoot, 'localization')
    const target = path.join(container, slug)
    if (!isInside(path.resolve(config.userDatabasePath), userRoot) || !isInside(userRoot, target)) {
      return NextResponse.json({ success: false, error: 'Invalid managed dataset path.' }, { status: 400 })
    }
    await fs.mkdir(container, { recursive: true })
    const stage = await fs.mkdtemp(path.join(container, '.augment-'))
    const stageTarget = path.join(stage, slug)
    let done = false
    try {
      await runRotateAugment(python, path.join(config.torchPath, 'tools/localization.py'), [
        '--source', root, '--save_dir', stageTarget, '--task', 'obb',
        '--degrees', String(degrees), '--copies', String(copies), '--seed', String(seed),
      ], config.torchPath)
      if (existsSync(target)) throw new Error('Target dataset directory was created concurrently.')
      await fs.rename(stageTarget, target)
      // prepare bakes the staged absolute path into data.yaml; point it at the final dir.
      const dataYaml = path.join(target, 'obb', 'data.yaml')
      const parsed = JSON.parse(await fs.readFile(dataYaml, 'utf-8'))
      parsed.path = path.join(target, 'obb').replace(/\\/g, '/')
      await fs.writeFile(dataYaml, JSON.stringify(parsed, null, 2), 'utf-8')
      done = true
    } finally {
      await fs.rm(stage, { recursive: true, force: true })
      if (!done) await fs.rm(target, { recursive: true, force: true }).catch(() => {})
    }
    const base = path.join(target, 'obb')
    const { columns } = readLocalizationDataset(base)
    const datasetYaml = stringify({
      num_classes: project.labels.length,
      localization: { task: 'obb', data: path.join(base, 'data.yaml').replace(/\\/g, '/') },
    })
    const created = await db.dataset.create({
      data: {
        name, description: `Inscribed rotation ±${degrees}°, ${copies} copies per train image, seed ${seed}. See augment-report.json.`,
        projectId: dataset.projectId, userId: owner.id, format: 'YOLOOBB',
        datasetDir: base, trainImagePath: 'images/train', trainAnnoPath: 'labels/train',
        evalImagePath: 'images/val', evalAnnoPath: 'labels/val',
        ...columns, yamlConfig: datasetYaml,
      },
    }).catch(async error => {
      await fs.rm(target, { recursive: true, force: true }).catch(() => {})
      throw error
    })
    const report = JSON.parse(await fs.readFile(path.join(target, 'augment-report.json'), 'utf-8'))
    await logActivity(auth.userId, {
      action: 'import_dataset', entityType: 'dataset', entityId: created.id, entityName: name,
      details: { augmentedFrom: datasetId, degrees, copies, seed, totals: report.totals },
    })
    return NextResponse.json({ success: true, data: { dataset: created, report } })
  } catch (error) {
    console.error('rotate-augment failed:', error)
    return NextResponse.json({ success: false, error: error instanceof Error ? error.message : 'Failed to augment dataset.' }, { status: 500 })
  }
}
