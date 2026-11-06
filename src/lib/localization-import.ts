import { createHash, randomUUID } from 'crypto'
import { spawn } from 'child_process'
import fs from 'fs/promises'
import path from 'path'
import { stringify } from 'yaml'
import { z } from 'zod'
import { db } from './db'
import { projectSchema, projectIssues, type AnnotationProject } from './localization-annotations'
import { frameworkMeta, resolvePythonPath } from './frameworks'
import { toSafeSlug, isInside } from './safe-path'
import { defaultTrainingParams, generateTrainingYaml, parseTrainingParams, trainingParamsToColumns } from './training-yaml'
import { defaultModelParams, generateModelYaml, parseModelParams, modelParamsToColumns } from './model-yaml'
import { readLocalizationDataset } from './localization-dataset'

export const localizationImportSchema = z.object({
  annotationProjectId: z.string().uuid(),
  imagesRoot: z.string().min(1),
  name: z.string().trim().min(1).max(100),
  task: z.enum(['obb', 'pose']),
  splitMode: z.enum(['existing', 'filename-index']),
  valRatio: z.number().min(0.05).max(0.4).default(0.2),
  testRatio: z.number().min(0).max(0.4).default(0.2),
  seed: z.number().int().min(0).max(2147483647).default(42),
  preview: z.boolean().default(true),
  sourceRevision: z.string().optional(),
})
export type LocalizationImportOptions = z.infer<typeof localizationImportSchema>

/** Exclude images with no objects. Missing points remain missing; never invent visibility. */
export function selectLocalizationData(source: AnnotationProject, options: Pick<LocalizationImportOptions, 'task' | 'splitMode' | 'valRatio' | 'testRatio' | 'seed'>) {
  const project = projectSchema.parse(source)
  const excluded = project.images.filter(image => !image.objects.length).map(image => image.name)
  project.images = project.images.filter(image => image.objects.length)
  if (!project.images.length) throw new Error('No annotated images were found.')
  const used = new Set(project.images.flatMap(image => image.objects.map(object => object.label)))
  project.labels = project.labels.filter(label => used.has(label))
  if (options.splitMode === 'filename-index') {
    for (const image of project.images) {
      const match = path.basename(image.name).match(/-(\d+)\.[^.]+$/)
      if (!match) throw new Error(`Cannot infer a capture index from ${image.name}; assign groups in Annotation instead.`)
      image.group = `capture-${Number(match[1])}`
    }
    const hash = (group: string) => createHash('sha256').update(`${options.seed}:${group}`).digest('hex')
    const groups = [...new Set(project.images.map(image => image.group))].sort((a, b) => hash(a).localeCompare(hash(b)))
    const nVal = Math.max(1, Math.round(groups.length * options.valRatio))
    const nTest = options.testRatio > 0 ? Math.max(1, Math.round(groups.length * options.testRatio)) : 0
    if (groups.length <= nVal + nTest) throw new Error('Not enough capture groups for non-empty train and validation/test splits.')
    const val = new Set(groups.slice(0, nVal))
    const test = new Set(groups.slice(nVal, nVal + nTest))
    for (const image of project.images) image.split = val.has(image.group) ? 'val' : test.has(image.group) ? 'test' : 'train'
  }
  const errors = projectIssues(project, options.task === 'pose')
  const splitCounts = Object.fromEntries(['train', 'val', 'test'].map(split => [split, {
    images: project.images.filter(image => image.split === split).length,
    instances: project.images.filter(image => image.split === split).reduce((total, image) => total + image.objects.length, 0),
    groups: [...new Set(project.images.filter(image => image.split === split).map(image => image.group))],
  }]))
  return { project, report: {
    sourceImages: source.images.length, includedImages: project.images.length, excludedImages: excluded,
    instances: project.images.reduce((total, image) => total + image.objects.length, 0),
    labels: project.labels, keypointNames: options.task === 'pose' ? project.keypointNames : [],
    splitMode: options.splitMode, seed: options.seed, splits: splitCounts,
    groupAssumption: options.splitMode === 'filename-index'
      ? 'Files with the same trailing numeric index are treated as one capture. Verify this against acquisition records.'
      : 'Uses the groups and splits explicitly saved by the annotator.',
    errors,
  } }
}

function prepareWithPython(python: string, script: string, args: string[], cwd: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(python, [script, 'prepare', ...args], { cwd, windowsHide: true, env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' } })
    let output = ''
    const capture = (chunk: Buffer) => { output = (output + chunk.toString('utf8')).slice(-10000) }
    child.stdout.on('data', capture)
    child.stderr.on('data', capture)
    const timer = setTimeout(() => { child.kill(); reject(new Error('Dataset preparation timed out.')) }, 600000)
    child.on('error', error => { clearTimeout(timer); reject(error) })
    child.on('close', code => {
      clearTimeout(timer)
      if (code === 0) resolve()
      else reject(new Error(output || `Dataset preparation failed (${code})`))
    })
  })
}

/** Called only after admin authentication. Original annotations/images are read-only. */
export async function importLocalizationDataset(userId: string, options: LocalizationImportOptions) {
  if (!path.isAbsolute(options.imagesRoot)) throw new Error('Images root must be an absolute directory.')
  const sourcePath = path.join(process.cwd(), 'annotations', toSafeSlug(userId, 'user'), 'projects', `${options.annotationProjectId}.json`)
  const raw = await fs.readFile(sourcePath, 'utf8')
  const sourceRevision = createHash('sha256').update(raw).digest('hex')
  const selected = selectLocalizationData(projectSchema.parse(JSON.parse(raw)), options)
  const report = { ...selected.report, sourceRevision, sourcePath, imagesRoot: options.imagesRoot }
  if (options.preview) return { preview: true, report }
  if (!options.sourceRevision || options.sourceRevision !== sourceRevision) throw new Error('Annotations changed or were not previewed. Preview this import again.')
  if (report.errors.length) throw new Error(`Annotation validation failed:\n${report.errors.slice(0, 20).join('\n')}`)
  const user = await db.user.findUniqueOrThrow({ where: { id: userId } })
  const system = await db.systemConfig.findFirstOrThrow()
  if (!system.userDatabasePath || !system.userConfigsPath || !system.torchPath) throw new Error('Configure dataset storage, config storage and the PyTorch trainer path in Settings.')
  const framework = options.task === 'obb' ? 'TorchOBB' as const : 'TorchPose' as const
  const { pythonPath: python } = resolvePythonPath(framework, '0', system)
  if (!python) throw new Error(`Configure the Python environment for ${framework} in Settings.`)
  const slug = `${toSafeSlug(options.name, 'localization')}-${randomUUID().slice(0, 8)}`
  const userRoot = path.resolve(system.userDatabasePath, user.username)
  const container = path.join(userRoot, 'localization')
  const target = path.join(container, slug)
  if (!isInside(path.resolve(system.userDatabasePath), userRoot) || !isInside(userRoot, target)) throw new Error('Invalid managed dataset path.')
  await fs.mkdir(container, { recursive: true })
  const temporary = await fs.mkdtemp(path.join(container, '.import-'))
  const snapshot = path.join(temporary, 'project.json')
  await fs.writeFile(snapshot, JSON.stringify(selected.project), 'utf8')
  let imported = false
  try {
    await prepareWithPython(python, path.join(system.torchPath, 'tools/localization.py'), [
      '--project', snapshot, '--images', options.imagesRoot, '--save_dir', target, '--task', options.task,
    ], system.torchPath)
    await fs.writeFile(path.join(target, 'import-report.json'), JSON.stringify(report, null, 2), 'utf8')
    const base = path.join(target, options.task)
    const datasetYaml = stringify({ num_classes: selected.project.labels.length, localization: {
      task: options.task, data: path.join(base, 'data.yaml').replace(/\\/g, '/'),
    } })
    const modelParams = { ...defaultModelParams(framework), numClasses: selected.project.labels.length }
    const modelYaml = generateModelYaml(framework, modelParams)
    const trainingParams = defaultTrainingParams(framework)
    const trainingYaml = generateTrainingYaml(framework, trainingParams)
    const { columns } = readLocalizationDataset(base)
    const records = await db.$transaction(async tx => {
      const project = await tx.project.create({ data: {
        name: slug, description: `Imported localization annotations: ${options.name}. ${report.groupAssumption}`,
        framework, task: frameworkMeta(framework).defaultTask, userId,
      } })
      const dataset = await tx.dataset.create({ data: {
        name: slug, description: `Excluded ${report.excludedImages.length} unannotated images. See import-report.json.`,
        projectId: project.id, userId, format: frameworkMeta(framework).datasetFormat,
        datasetDir: base, trainImagePath: 'images/train', trainAnnoPath: 'labels/train',
        evalImagePath: 'images/val', evalAnnoPath: 'labels/val',
        ...columns, yamlConfig: datasetYaml,
      } })
      const model = await tx.model.create({ data: {
        ...modelParamsToColumns({ ...modelParams, ...parseModelParams(framework, modelYaml) }),
        projectId: project.id, userId, name: `${slug}-model`, yamlConfig: modelYaml,
      } })
      const trainingConfig = await tx.trainingConfig.create({ data: {
        ...trainingParamsToColumns(framework, { ...trainingParams, ...parseTrainingParams(framework, trainingYaml) }),
        projectId: project.id, userId, name: `${slug}-training`, yamlConfig: trainingYaml,
      } })
      return { project, dataset, model, trainingConfig }
    })
    imported = true
    return { preview: false, report, ...records }
  } finally {
    await fs.rm(temporary, { recursive: true, force: true })
    if (!imported && isInside(container, target)) await fs.rm(target, { recursive: true, force: true })
  }
}
