import { createHash, randomUUID } from 'crypto'
import { spawn } from 'child_process'
import fs from 'fs/promises'
import path from 'path'
import { stringify } from 'yaml'
import { z } from 'zod'
import { db } from './db'
import { aabbFromObject, projectSchema, projectIssues, type AnnotationProject } from './localization-annotations'
import { frameworkMeta, resolvePythonPath } from './frameworks'
import { toSafeSlug, isInside } from './safe-path'
import { generateDatasetYaml } from './dataset-yaml'
import { projectsDir, readAnnotationSet } from './annotation-store'
import { defaultTrainingParams, generateTrainingYaml, parseTrainingParams, trainingParamsToColumns } from './training-yaml'
import { defaultModelParams, generateModelYaml, parseModelParams, modelParamsToColumns } from './model-yaml'
import { readLocalizationDataset } from './localization-dataset'

export const localizationImportSchema = z.object({
  annotationProjectId: z.string().uuid(),
  imagesRoot: z.string().min(1),
  name: z.string().trim().min(1).max(100),
  task: z.enum(['obb', 'pose', 'detect']),
  /** Only meaningful for task 'detect': which detection trainer consumes the COCO output. */
  detectFramework: z.enum(['TorchDet', 'PaddleDetection']).default('TorchDet'),
  splitMode: z.enum(['existing', 'filename-index', 'random']),
  valRatio: z.number().min(0.05).max(0.4).default(0.2),
  testRatio: z.number().min(0).max(0.4).default(0.2),
  seed: z.number().int().min(0).max(2147483647).default(42),
  preview: z.boolean().default(true),
  sourceRevision: z.string().optional(),
  createConfigs: z.enum(['all', 'dataset']).default('all'),
})
export type LocalizationImportOptions = z.infer<typeof localizationImportSchema>

/** Import only reviewed images with objects; unreviewed labels (e.g. model pre-labels) are excluded. Missing points remain missing; never invent visibility. */
export function selectLocalizationData(source: AnnotationProject, options: Pick<LocalizationImportOptions, 'task' | 'splitMode' | 'valRatio' | 'testRatio' | 'seed'>) {
  const project = projectSchema.parse(source)
  const emptyImages = project.images.filter(image => !image.objects.length).map(image => image.name)
  const unreviewedImages = project.images.filter(image => image.objects.length && !image.reviewed).map(image => image.name)
  project.images = project.images.filter(image => image.objects.length && image.reviewed)
  if (!project.images.length) throw new Error('No reviewed annotated images were found.')
  const used = new Set(project.images.flatMap(image => image.objects.map(object => object.label)))
  project.labels = project.labels.filter(label => used.has(label))
  if (options.splitMode === 'filename-index') {
    for (const image of project.images) {
      const match = path.basename(image.name).match(/-(\d+)\.[^.]+$/)
      if (!match) throw new Error(`Cannot infer a capture index from ${image.name}; assign groups in Annotation instead.`)
      image.group = `capture-${Number(match[1])}`
    }
  } else if (options.splitMode === 'random') {
    // Images sharing a saved capture group stay together (the trainer rejects a group crossing splits);
    // ungrouped images are independent units.
    for (const image of project.images) if (!image.group) image.group = `image:${image.name}`
  }
  if (options.splitMode !== 'existing') {
    const hash = (group: string) => createHash('sha256').update(`${options.seed}:${group}`).digest('hex')
    const groups = [...new Set(project.images.map(image => image.group))].sort((a, b) => hash(a).localeCompare(hash(b)))
    const nVal = Math.max(1, Math.round(groups.length * options.valRatio))
    const nTest = options.testRatio > 0 ? Math.max(1, Math.round(groups.length * options.testRatio)) : 0
    if (groups.length <= nVal + nTest) throw new Error(`Not enough ${options.splitMode === 'random' ? 'images/groups' : 'capture groups'} (${groups.length}) for non-empty train and validation/test splits.`)
    const val = new Set(groups.slice(0, nVal))
    const test = new Set(groups.slice(nVal, nVal + nTest))
    for (const image of project.images) image.split = val.has(image.group) ? 'val' : test.has(image.group) ? 'test' : 'train'
  }
  const errors = projectIssues(project, options.task === 'pose')
  // Detection export flattens every non-axis-aligned OBB to the bounding box of
  // its corners; surface the count so the import dialog can warn about it.
  const rotatedConverted = options.task === 'detect'
    ? project.images.reduce((total, image) => total + image.objects.filter(object => object.kind === 'obb' && object.angle !== 0).length, 0)
    : 0
  const splitCounts = Object.fromEntries(['train', 'val', 'test'].map(split => [split, {
    images: project.images.filter(image => image.split === split).length,
    instances: project.images.filter(image => image.split === split).reduce((total, image) => total + image.objects.length, 0),
    groups: [...new Set(project.images.filter(image => image.split === split).map(image => image.group))],
    // Unreviewed images the annotator had placed in this split; excluded, shown so the counts reconcile with Annotation.
    excludedUnreviewed: options.splitMode === 'existing' ? source.images.filter(image => image.split === split && image.objects.length && !image.reviewed).length : 0,
  }]))
  return { project, report: {
    sourceImages: source.images.length, includedImages: project.images.length,
    excludedImages: [...emptyImages, ...unreviewedImages], emptyImages, unreviewedImages,
    instances: project.images.reduce((total, image) => total + image.objects.length, 0),
    rotatedConverted,
    labels: project.labels, keypointNames: options.task === 'pose' ? project.keypointNames : [],
    splitMode: options.splitMode, seed: options.seed, splits: splitCounts,
    groupAssumption: {
      'filename-index': 'Files with the same trailing numeric index are treated as one capture. Verify this against acquisition records.',
      random: 'Seeded random split of reviewed images. Images sharing a saved capture group stay in one split; ungrouped images are split independently, so near-duplicate shots can land in both train and val/test and inflate evaluation metrics.',
      existing: 'Uses the groups and splits explicitly saved by the annotator.',
    }[options.splitMode],
    errors,
  } }
}

export function prepareWithPython(python: string, script: string, args: string[], cwd: string): Promise<void> {
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
  const sourcePath = path.join(projectsDir(userId), `${options.annotationProjectId}.json`)
  const stored = await readAnnotationSet(projectsDir(userId), options.annotationProjectId)
  if (stored.meta.status === 'archived') {
    throw new Error('Annotation set is archived; reactivate it on the Annotation page first.')
  }
  const raw = stored.raw
  const sourceRevision = createHash('sha256').update(raw).digest('hex')
  const selected = selectLocalizationData(stored.project, options)
  const report = { ...selected.report, sourceRevision, sourcePath, imagesRoot: options.imagesRoot }
  if (options.preview) return { preview: true, report }
  if (!options.sourceRevision || options.sourceRevision !== sourceRevision) throw new Error('Annotations changed or were not previewed. Preview this import again.')
  if (report.errors.length) throw new Error(`Annotation validation failed:\n${report.errors.slice(0, 20).join('\n')}`)
  const user = await db.user.findUniqueOrThrow({ where: { id: userId } })
  const system = await db.systemConfig.findFirstOrThrow()
  if (!system.userDatabasePath || !system.userConfigsPath) throw new Error('Configure dataset storage and config storage in Settings.')
  if (options.task === 'detect') {
    // Pure TypeScript path: no trainer repo or Python env is needed to emit COCO.
    return importDetectionDataset(user, system, selected, report, options)
  }
  if (!system.torchPath) throw new Error('Configure the PyTorch trainer path in Settings.')
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
    const datasetOnly = options.createConfigs === 'dataset'
    const modelParams = datasetOnly ? null : { ...defaultModelParams(framework), numClasses: selected.project.labels.length }
    const modelYaml = modelParams ? generateModelYaml(framework, modelParams) : null
    const trainingParams = datasetOnly ? null : defaultTrainingParams(framework)
    const trainingYaml = trainingParams ? generateTrainingYaml(framework, trainingParams) : null
    const { columns } = readLocalizationDataset(base)
    const records = await db.$transaction(async tx => {
      const project = await tx.project.create({ data: {
        name: slug, description: `Imported localization annotations: ${options.name}. ${report.groupAssumption}`,
        framework, task: frameworkMeta(framework).defaultTask, userId,
      } })
      const dataset = await tx.dataset.create({ data: {
        name: slug, description: `Excluded ${report.emptyImages.length} unannotated and ${report.unreviewedImages.length} unreviewed images. See import-report.json.`,
        projectId: project.id, userId, format: frameworkMeta(framework).datasetFormat,
        datasetDir: base, trainImagePath: 'images/train', trainAnnoPath: 'labels/train',
        evalImagePath: 'images/val', evalAnnoPath: 'labels/val',
        ...columns, yamlConfig: datasetYaml,
      } })
      if (datasetOnly) return { project, dataset, model: null, trainingConfig: null }
      const model = await tx.model.create({ data: {
        ...modelParamsToColumns({ ...modelParams!, ...parseModelParams(framework, modelYaml!) }),
        projectId: project.id, userId, name: `${slug}-model`, yamlConfig: modelYaml!,
      } })
      const trainingConfig = await tx.trainingConfig.create({ data: {
        ...trainingParamsToColumns(framework, { ...trainingParams!, ...parseTrainingParams(framework, trainingYaml!) }),
        projectId: project.id, userId, name: `${slug}-training`, yamlConfig: trainingYaml!,
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

/** Report shape passed to importDetectionDataset: selectLocalizationData output plus the source bookkeeping. */
type ImportReport = ReturnType<typeof selectLocalizationData>['report'] & {
  sourceRevision: string; sourcePath: string; imagesRoot: string
}

type CocoImage = { id: number; file_name: string; width: number; height: number }
type CocoAnnotation = { id: number; image_id: number; category_id: number; bbox: number[]; area: number; iscrowd: number; segmentation: number[][] }

/**
 * task 'detect': emit a plain COCO detection dataset from the same annotation
 * projects, entirely in TypeScript — no torchtrain checkout or Python env is
 * involved. Rectangles pass through; rotated boxes become the axis-aligned
 * bounding box of their (clipped) corners; keypoints are ignored.
 */
async function importDetectionDataset(
  user: { id: string; username: string },
  system: { userDatabasePath: string | null },
  selected: ReturnType<typeof selectLocalizationData>,
  report: ImportReport,
  options: LocalizationImportOptions,
) {
  const framework = options.detectFramework
  const slug = `${toSafeSlug(options.name, 'localization')}-${randomUUID().slice(0, 8)}`
  const userRoot = path.resolve(system.userDatabasePath!, user.username)
  const container = path.join(userRoot, 'COCO')
  const target = path.join(container, slug)
  if (!isInside(path.resolve(system.userDatabasePath!), userRoot) || !isInside(userRoot, target)) throw new Error('Invalid managed dataset path.')

  const categories = selected.project.labels.map((name, index) => ({ id: index + 1, name, supercategory: 'object' }))
  const categoryId = new Map(selected.project.labels.map((name, index) => [name, index + 1]))
  const imagesRoot = path.resolve(options.imagesRoot)
  const splits = (['train', 'val', 'test'] as const).map(split => ({
    split,
    images: selected.project.images.filter(image => image.split === split),
  }))
  if (!splits.find(s => s.split === 'train')!.images.length || !splits.find(s => s.split === 'val')!.images.length) {
    throw new Error('Detection export needs at least one train and one val image.')
  }

  await fs.mkdir(container, { recursive: true })
  const temporary = await fs.mkdtemp(path.join(container, '.import-'))
  let imported = false
  let moved = false
  try {
    const stats: Record<string, { images: number; annotations: number; classStats: { id: number; name: string; count: number; imageCount: number }[] }> = {}
    let dropped = 0
    for (const { split, images } of splits) {
      if (!images.length) continue
      const cocoImages: CocoImage[] = []
      const cocoAnnotations: CocoAnnotation[] = []
      for (const [index, image] of images.entries()) {
        // image.name is a forward-slash relative path; keep the subfolder layout.
        const source = path.join(imagesRoot, ...image.name.split('/'))
        if (!isInside(imagesRoot, source)) throw new Error(`Image path escapes the images root: ${image.name}`)
        const destination = path.join(temporary, 'data', split, ...image.name.split('/'))
        await fs.mkdir(path.dirname(destination), { recursive: true })
        await fs.copyFile(source, destination)
        cocoImages.push({ id: index + 1, file_name: image.name, width: image.width, height: image.height })
        for (const object of image.objects) {
          const aabb = aabbFromObject(object, image)
          if (!aabb) { dropped++; continue }
          const { x, y, width, height } = aabb
          cocoAnnotations.push({
            id: cocoAnnotations.length + 1, image_id: index + 1, category_id: categoryId.get(object.label)!,
            bbox: [x, y, width, height], area: width * height, iscrowd: 0,
            segmentation: [[x, y, x + width, y, x + width, y + height, x, y + height]],
          })
        }
      }
      await fs.mkdir(path.join(temporary, 'data', 'annotations'), { recursive: true })
      const document = {
        info: {
          description: `COCO Format Dataset: ${slug}`, version: '1.0', year: new Date().getFullYear(),
          contributor: 'Auto Training Platform', date_created: new Date().toISOString(),
        },
        images: cocoImages, annotations: cocoAnnotations, categories,
      }
      if (split !== 'test' || cocoImages.length) {
        await fs.writeFile(path.join(temporary, 'data', 'annotations', `instance_${split}.json`), JSON.stringify(document, null, 2), 'utf8')
      }
      stats[split] = {
        images: cocoImages.length,
        annotations: cocoAnnotations.length,
        classStats: categories.map(category => ({
          ...category,
          count: cocoAnnotations.filter(a => a.category_id === category.id).length,
          imageCount: cocoImages.filter(image => cocoAnnotations.some(a => a.image_id === image.id && a.category_id === category.id)).length,
        })),
      }
    }
    await fs.writeFile(path.join(temporary, 'project.json'), JSON.stringify(selected.project), 'utf8')
    await fs.writeFile(path.join(temporary, 'import-report.json'), JSON.stringify({ ...report, droppedAnnotations: dropped }, null, 2), 'utf8')
    await fs.rename(temporary, target)
    moved = true

    const datasetYaml = generateDatasetYaml({
      name: slug, format: 'COCO', numClasses: categories.length, datasetDir: target,
      trainImagePath: 'data/train', trainAnnoPath: 'data/annotations/instance_train.json',
      evalImagePath: 'data/val', evalAnnoPath: 'data/annotations/instance_val.json',
      task: frameworkMeta(framework).defaultTask, framework,
    })
    const datasetOnly = options.createConfigs === 'dataset'
    const modelParams = datasetOnly ? null : { ...defaultModelParams(framework), numClasses: categories.length }
    const modelYaml = modelParams ? generateModelYaml(framework, modelParams) : null
    const trainingParams = datasetOnly ? null : defaultTrainingParams(framework)
    const trainingYaml = trainingParams ? generateTrainingYaml(framework, trainingParams) : null
    const records = await db.$transaction(async tx => {
      const project = await tx.project.create({ data: {
        name: slug, description: `Imported localization annotations (COCO detection): ${options.name}. ${report.groupAssumption}`,
        framework, task: frameworkMeta(framework).defaultTask, userId: user.id,
      } })
      const dataset = await tx.dataset.create({ data: {
        name: slug, description: `Excluded ${report.emptyImages.length} unannotated and ${report.unreviewedImages.length} unreviewed images. See import-report.json.`,
        projectId: project.id, userId: user.id, format: 'COCO',
        datasetDir: target, trainImagePath: 'data/train', trainAnnoPath: 'data/annotations/instance_train.json',
        evalImagePath: 'data/val', evalAnnoPath: 'data/annotations/instance_val.json',
        numClasses: categories.length,
        numAnnotations: Object.values(stats).reduce((sum, s) => sum + s.annotations, 0),
        numTrainImages: stats.train?.images ?? 0, numEvalImages: stats.val?.images ?? 0,
        classStats: JSON.stringify({ train: stats.train?.classStats ?? [], eval: stats.val?.classStats ?? [] }),
        yamlConfig: datasetYaml,
      } })
      if (datasetOnly) return { project, dataset, model: null, trainingConfig: null }
      const model = await tx.model.create({ data: {
        ...modelParamsToColumns({ ...modelParams!, ...parseModelParams(framework, modelYaml!) }),
        projectId: project.id, userId: user.id, name: `${slug}-model`, yamlConfig: modelYaml!,
      } })
      const trainingConfig = await tx.trainingConfig.create({ data: {
        ...trainingParamsToColumns(framework, { ...trainingParams!, ...parseTrainingParams(framework, trainingYaml!) }),
        projectId: project.id, userId: user.id, name: `${slug}-training`, yamlConfig: trainingYaml!,
      } })
      return { project, dataset, model, trainingConfig }
    })
    imported = true
    return { preview: false, report, ...records }
  } finally {
    // Once the temp dir has been renamed into place, a DB failure must remove
    // `target`, not the (now non-existent) `temporary`.
    if (!imported && moved && isInside(container, target)) await fs.rm(target, { recursive: true, force: true })
    if (!imported && !moved) await fs.rm(temporary, { recursive: true, force: true })
  }
}
