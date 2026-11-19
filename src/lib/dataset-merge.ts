/**
 * Generic same-type dataset merge. Produces a NEW managed dataset; sources are
 * never modified and each source keeps its own train/val/test split.
 *
 * Effective format = Dataset.format, or frameworkMeta(project.framework)
 * .datasetFormat when the column is empty (some older localization rows have '').
 * Every copied file is prefixed `<safe source name>__`; paired files (Seg
 * image/label, anomaly abnormal/mask, COCO file_name, localization
 * manifest/project names) are renamed consistently.
 */
import { createHash, randomUUID } from 'crypto'
import { spawn } from 'child_process'
import fs from 'fs/promises'
import { existsSync } from 'fs'
import path from 'path'
import { stringify } from 'yaml'
import { frameworkMeta } from './frameworks'
import { generateDatasetYaml, ANOMALY_IMAGE_EXTENSIONS } from './dataset-yaml'
import { readLocalizationDataset } from './localization-dataset'
import { projectSchema, type AnnotationProject } from './localization-annotations'
import { toSafeSlug } from './safe-path'

export type MergeRow = {
  id: string; name: string; userId: string; projectId: string; format: string;
  datasetDir: string | null; trainImagePath: string | null; trainAnnoPath: string | null;
  evalImagePath: string | null; evalAnnoPath: string | null;
  normalDir: string | null; normalTestDir: string | null; abnormalDir: string | null; maskDir: string | null;
  numClasses: number; yamlConfig: string | null;
  project: { id: string; framework: string; task: string | null };
}

export function effectiveFormat(dataset: MergeRow): string {
  return dataset.format || frameworkMeta(dataset.project.framework).datasetFormat
}

export function datasetRoot(userRoot: string, dataset: MergeRow): string {
  const dir = dataset.datasetDir || ''
  return path.isAbsolute(dir) ? dir : path.join(userRoot, dir)
}

async function fileHash(file: string) {
  const hash = createHash('sha256')
  hash.update(await fs.readFile(file))
  return hash.digest('hex')
}

async function folderSize(directory: string): Promise<number> {
  if (!existsSync(directory)) return 0
  let total = 0
  for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
    const full = path.join(directory, entry.name)
    total += entry.isDirectory() ? await folderSize(full) : (await fs.stat(full)).size
  }
  return total
}
export { folderSize }

/** size-prefilter then SHA256, across sources only (same-source duplicates are legitimate). */
async function findDuplicates(entries: { source: number; file: string }[]) {
  const bySize = new Map<number, { source: number; file: string }[]>()
  for (const entry of entries) {
    const size = (await fs.stat(entry.file)).size
    const list = bySize.get(size) ?? []
    list.push(entry)
    bySize.set(size, list)
  }
  const hashes = new Map<string, { source: number; file: string }>()
  const duplicates: { first: string; second: string }[] = []
  for (const group of bySize.values()) {
    if (group.length < 2) continue
    for (const entry of group) {
      const hash = await fileHash(entry.file)
      const previous = hashes.get(hash)
      if (previous && previous.source !== entry.source) {
        duplicates.push({ first: previous.file, second: entry.file })
      } else if (!previous) {
        hashes.set(hash, entry)
      }
    }
  }
  return duplicates
}

interface CocoImage { id: number; file_name: string; width: number; height: number; [key: string]: unknown }
interface CocoAnnotation { id: number; image_id: number; category_id: number; [key: string]: unknown }
interface CocoDoc { images: CocoImage[]; annotations: CocoAnnotation[]; categories: { id: number; name: string; [key: string]: unknown }[]; [key: string]: unknown }

async function readJson<T>(file: string): Promise<T> {
  return JSON.parse(await fs.readFile(file, 'utf-8'))
}

function cocoClassStats(doc: CocoDoc) {
  const counts = new Map<number, number>()
  const imageIds = new Map<number, Set<number>>()
  for (const ann of doc.annotations) {
    counts.set(ann.category_id, (counts.get(ann.category_id) || 0) + 1)
    if (!imageIds.has(ann.category_id)) imageIds.set(ann.category_id, new Set())
    imageIds.get(ann.category_id)!.add(ann.image_id)
  }
  return doc.categories.map(category => ({
    id: category.id, name: category.name,
    count: counts.get(category.id) || 0, imageCount: imageIds.get(category.id)?.size || 0,
  }))
}

function spawnPython(python: string, script: string, args: string[], cwd: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(python, [script, ...args], { cwd, windowsHide: true, env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' } })
    let output = ''
    const capture = (chunk: Buffer) => { output = (output + chunk.toString('utf8')).slice(-10000) }
    child.stdout.on('data', capture)
    child.stderr.on('data', capture)
    const timer = setTimeout(() => { child.kill(); reject(new Error('Python step timed out.')) }, 600000)
    child.on('error', error => { clearTimeout(timer); reject(error) })
    child.on('close', code => {
      clearTimeout(timer)
      if (code === 0) resolve()
      else reject(new Error(output || `Python step failed (${code})`))
    })
  })
}

export interface SourceSummary {
  id: string; name: string; prefix: string; root: string;
  classes: string[]; splits: Record<string, number>; fileCount: number; bytes: number;
}
export interface MergePlan {
  format: string; framework: string; canMerge: boolean; blockers: string[]; warnings: string[];
  sources: SourceSummary[];
  classes?: { union: string[]; mismatched: boolean };
  duplicates: { first: string; second: string }[];
  needsAllowClassMismatch?: boolean;
  mergedName?: string; targetDir?: string;
  newDataset?: {
    datasetDir: string; trainImagePath: string | null; trainAnnoPath: string | null;
    evalImagePath: string | null; evalAnnoPath: string | null;
    normalDir: string | null; normalTestDir: string | null; abnormalDir: string | null; maskDir: string | null;
    numClasses: number; numAnnotations: number; numTrainImages: number; numEvalImages: number;
    classStats: string; yamlConfig: string;
  };
}

function prefixFor(name: string) {
  return toSafeSlug(name, 'dataset')
}

function unionClasses(perSource: string[][]): { union: string[]; mismatched: boolean } {
  const union: string[] = []
  for (const names of perSource) for (const name of names) if (!union.includes(name)) union.push(name)
  const mismatched = perSource.some(names => new Set(names).size !== union.length || names.some(n => !union.includes(n)))
  return { union, mismatched }
}

/* ---------------- per-format planners ---------------- */

async function planCoco(userRoot: string, rows: MergeRow[]): Promise<{ summary: SourceSummary[]; classes: { union: string[]; mismatched: boolean }; docs: { doc: CocoDoc; imageDir: string; split: 'train' | 'val' }[][]; files: { source: number; file: string }[] }> {
  const docs: { doc: CocoDoc; imageDir: string; split: 'train' | 'val' }[][] = []
  const summary: SourceSummary[] = []
  const files: { source: number; file: string }[] = []
  for (const [i, row] of rows.entries()) {
    const root = datasetRoot(userRoot, row)
    const splits: { doc: CocoDoc; imageDir: string; split: 'train' | 'val' }[] = []
    const splitCounts: Record<string, number> = {}
    let bytes = 0, fileCount = 0
    for (const [split, annoRel, imageRel] of [['train', row.trainAnnoPath || 'data/annotations/instance_train.json', row.trainImagePath || 'data/train'], ['val', row.evalAnnoPath || 'data/annotations/instance_val.json', row.evalImagePath || 'data/val']] as const) {
      const annoFile = path.isAbsolute(annoRel) ? annoRel : path.join(root, annoRel)
      const imageDir = path.isAbsolute(imageRel) ? imageRel : path.join(root, imageRel)
      const doc = await readJson<CocoDoc>(annoFile)
      if (!Array.isArray(doc.images) || !Array.isArray(doc.annotations) || !Array.isArray(doc.categories)) {
        throw new Error(`Invalid COCO annotation file: ${annoFile}`)
      }
      splits.push({ doc, imageDir, split })
      splitCounts[split] = doc.images.length
      for (const image of doc.images) {
        const file = path.join(imageDir, image.file_name)
        files.push({ source: i, file })
        fileCount++
        bytes += (await fs.stat(file)).size
      }
    }
    // class order = first listed doc order (train then val), ids renumbered later
    const ordered = [...new Set(splits.flatMap(s => s.doc.categories.sort((x, y) => x.id - y.id).map(c => c.name.trim())))]
    docs.push(splits)
    summary.push({ id: row.id, name: row.name, prefix: prefixFor(row.name), root, classes: ordered, splits: splitCounts, fileCount, bytes })
  }
  const classes = unionClasses(summary.map(s => s.classes))
  return { summary, classes, docs, files }
}

interface ReportData { hashes: Record<string, string>[]; renameMap: { source: string; from: string; to: string; split: string }[]; sourceSplits: Record<string, Record<string, number>> }
async function writeMergeReport(targetDir: string, plan: MergePlan, data: ReportData) {
  await fs.writeFile(path.join(targetDir, 'merge-report.json'), JSON.stringify({
    mergedName: plan.mergedName, format: plan.format,
    sources: plan.sources.map((s, i) => ({ id: s.id, name: s.name, annotationSha256: data.hashes[i] ?? {}, splits: s.splits })),
    classes: plan.classes ?? null, renameMap: data.renameMap,
    mergedSplits: data.sourceSplits.merged, warnings: plan.warnings,
  }, null, 2), 'utf-8')
}

async function mergeCoco(plan: MergePlan, docs: { doc: CocoDoc; imageDir: string; split: 'train' | 'val' }[][], targetDir: string, report: ReportData) {
  const categoryIds = new Map(plan.classes!.union.map((name, index) => [name, index + 1]))
  for (const split of ['train', 'val'] as const) {
    const merged: CocoDoc = { info: {}, licenses: [], images: [], annotations: [], categories: [] }
    for (const name of plan.classes!.union) merged.categories.push({ id: categoryIds.get(name)!, name, supercategory: 'object' })
    let imageId = 1, annotationId = 1
    const imageDir = path.join(targetDir, 'data', split)
    await fs.mkdir(imageDir, { recursive: true })
    for (const [i, splits] of docs.entries()) {
      const entry = splits.find(s => s.split === split)
      if (!entry) continue
      const idMap = new Map<number, number>()
      for (const image of entry.doc.images) {
        // file_name may carry subdirectories; flatten so copies stay a flat dir.
        const fileName = `${plan.sources[i].prefix}__${image.file_name.replace(/[\\/]+/g, '__')}`
        report.renameMap.push({ source: plan.sources[i].name, from: `${split}/${image.file_name}`, to: `${split}/${fileName}`, split })
        report.sourceSplits[plan.sources[i].name] = report.sourceSplits[plan.sources[i].name] ?? {}
        report.sourceSplits[plan.sources[i].name][split] = (report.sourceSplits[plan.sources[i].name][split] || 0) + 1
        await fs.copyFile(path.join(entry.imageDir, image.file_name), path.join(imageDir, fileName))
        idMap.set(image.id, imageId)
        merged.images.push({ ...image, id: imageId, file_name: fileName })
        imageId++
      }
      const catById = new Map(entry.doc.categories.map(c => [c.id, c.name.trim()]))
      for (const ann of entry.doc.annotations) {
        if (!idMap.has(ann.image_id)) continue
        const className = catById.get(ann.category_id)
        if (!className || !categoryIds.has(className)) continue
        merged.annotations.push({ ...ann, id: annotationId++, image_id: idMap.get(ann.image_id)!, category_id: categoryIds.get(className)! })
      }
      report.sourceSplits.merged[split] = merged.images.length
    }
    await fs.mkdir(path.join(targetDir, 'data', 'annotations'), { recursive: true })
    await fs.writeFile(path.join(targetDir, 'data', 'annotations', `instance_${split}.json`), JSON.stringify(merged, null, 2), 'utf-8')
    if (split === 'train') plan.newDataset!.numAnnotations = merged.annotations.length
    if (split === 'train') plan.newDataset!.classStats = JSON.stringify({ train: cocoClassStats(merged) })
    else {
      plan.newDataset!.numEvalImages = merged.images.length
      const stats = JSON.parse(plan.newDataset!.classStats)
      stats.eval = cocoClassStats(merged)
      plan.newDataset!.classStats = JSON.stringify(stats)
    }
    if (split === 'train') plan.newDataset!.numTrainImages = merged.images.length
  }
}

async function planSeg(userRoot: string, rows: MergeRow[]): Promise<{ summary: SourceSummary[]; classNames: string[]; identical: boolean; lists: { lines: { split: string; image: string; label: string }[]; classNames: string[] }[]; files: { source: number; file: string }[] }> {
  const summary: SourceSummary[] = []
  const lists: { lines: { split: string; image: string; label: string }[]; classNames: string[] }[] = []
  const files: { source: number; file: string }[] = []
  let firstNames: string[] | null = null
  let identical = true
  for (const [i, row] of rows.entries()) {
    const root = datasetRoot(userRoot, row)
    const classFile = path.join(root, 'class_names.txt')
    const classNames = existsSync(classFile)
      ? (await fs.readFile(classFile, 'utf-8')).split(/\r?\n/).map(l => l.trim()).filter(Boolean)
      : []
    if (i === 0) firstNames = classNames
    else if (classNames.join('\n') !== firstNames!.join('\n')) identical = false
    const lines: { split: string; image: string; label: string }[] = []
    const splitCounts: Record<string, number> = {}
    let bytes = 0, fileCount = 0
    for (const [split, listRel] of [['train', row.trainAnnoPath || 'train.txt'], ['val', row.evalAnnoPath || 'val.txt'], ['test', 'test.txt']] as const) {
      const listFile = path.isAbsolute(listRel) ? listRel : path.join(root, listRel)
      if (!existsSync(listFile)) continue
      for (const raw of (await fs.readFile(listFile, 'utf-8')).split(/\r?\n/)) {
        const trimmed = raw.trim()
        if (!trimmed) continue
        const [image, label] = trimmed.split(/\s+/)
        lines.push({ split, image, label })
        splitCounts[split] = (splitCounts[split] || 0) + 1
        for (const rel of [image, label]) {
          const file = path.join(root, rel)
          // Duplicate detection covers IMAGE content only: identical label/mask
          // files (e.g. empty masks) across batches must not block the merge.
          if (rel === image) files.push({ source: i, file })
          fileCount++
          bytes += (await fs.stat(file)).size
        }
      }
    }
    lists.push({ lines, classNames })
    summary.push({ id: row.id, name: row.name, prefix: prefixFor(row.name), root, classes: classNames, splits: splitCounts, fileCount, bytes })
  }
  return { summary, classNames: firstNames ?? [], identical, lists, files }
}

async function mergeSeg(plan: MergePlan, lists: { lines: { split: string; image: string; label: string }[] }[], targetDir: string, report: ReportData) {
  const imageDir = path.join(targetDir, 'JPEGImages')
  const annoDir = path.join(targetDir, 'Annotations')
  await fs.mkdir(imageDir, { recursive: true })
  await fs.mkdir(annoDir, { recursive: true })
  const out: Record<string, string[]> = { train: [], val: [], test: [] }
  for (const [i, source] of lists.entries()) {
    for (const line of source.lines) {
      const image = `${plan.sources[i].prefix}__${path.basename(line.image)}`
      const label = `${plan.sources[i].prefix}__${path.basename(line.label)}`
      await fs.copyFile(path.join(plan.sources[i].root, line.image), path.join(imageDir, image))
      await fs.copyFile(path.join(plan.sources[i].root, line.label), path.join(annoDir, label))
      out[line.split] = out[line.split] ?? []
      out[line.split].push(`JPEGImages/${image} Annotations/${label}`)
      report.renameMap.push({ source: plan.sources[i].name, from: `${line.split}/${line.image}`, to: `${line.split}/JPEGImages/${image}`, split: line.split })
    }
  }
  for (const [split, lines] of Object.entries(out)) {
    if (lines.length) await fs.writeFile(path.join(targetDir, `${split}.txt`), lines.join('\n') + '\n', 'utf-8')
  }
  await fs.writeFile(path.join(targetDir, 'class_names.txt'), plan.classes!.union.join('\n') + '\n', 'utf-8')
  for (const [split, lines] of Object.entries(out)) report.sourceSplits.merged[split] = lines.length
  plan.newDataset!.numTrainImages = out.train.length
  plan.newDataset!.numEvalImages = out.val.length
  plan.newDataset!.numAnnotations = out.train.length + out.val.length + out.test.length
}

async function planLocalization(userRoot: string, rows: MergeRow[]): Promise<{
  summary: SourceSummary[]; classes: { union: string[]; mismatched: boolean };
  projects: AnnotationProject[]; manifests: Map<string, { file: string; split: string }>[];
  files: { source: number; file: string }[]; keypointNames: string[][]; task: string;
}> {
  const task = frameworkMeta(rows[0].project.framework).localizationTask as string
  const summary: SourceSummary[] = []
  const projects: AnnotationProject[] = []
  const manifests: Map<string, { file: string; split: string }>[] = []
  const files: { source: number; file: string }[] = []
  const keypointNames: string[][] = []
  for (const [i, row] of rows.entries()) {
    const root = path.dirname(datasetRoot(userRoot, row))
    const project = projectSchema.parse(await readJson(path.join(root, 'project.json')))
    const manifest = new Map((await readJson<{ source: string; file: string; split: string }[]>(path.join(root, 'manifest.json'))).map(m => [m.source, m]))
    projects.push(project)
    manifests.push(manifest)
    keypointNames.push(project.keypointNames)
    const splitCounts: Record<string, number> = {}
    let bytes = 0, fileCount = 0
    for (const image of project.images) {
      const entry = manifest.get(image.name)
      if (!entry) throw new Error(`manifest.json has no entry for ${image.name} in ${row.name}`)
      const file = path.join(root, task, 'images', entry.split, entry.file)
      files.push({ source: i, file })
      fileCount++
      bytes += (await fs.stat(file)).size
      splitCounts[image.split] = (splitCounts[image.split] || 0) + 1
    }
    summary.push({ id: row.id, name: row.name, prefix: prefixFor(row.name), root, classes: project.labels, splits: splitCounts, fileCount, bytes })
  }
  return { summary, classes: unionClasses(summary.map(s => s.classes)), projects, manifests, files, keypointNames, task }
}

async function mergeLocalization(
  plan: MergePlan,
  prepared: { projects: AnnotationProject[]; manifests: Map<string, { file: string; split: string }>[]; task: string },
  targetDir: string,
  python: string, torchPath: string,
  finalDir: string | undefined,
  reportOut: ReportData,
) {
  const merged: AnnotationProject = {
    format: 'autotrain-localization', version: 1, id: randomUUID(),
    name: plan.mergedName ?? 'merged', labels: plan.classes!.union,
    keypointNames: prepared.projects[0].keypointNames, images: [],
  }
  const stage = await fs.mkdtemp(path.join(path.dirname(targetDir), '.merge-images-'))
  const renameMap: { source: string; from: string; to: string; split: string; group: string }[] = []
  try {
    for (const [i, project] of prepared.projects.entries()) {
      const manifest = prepared.manifests[i]
      for (const image of project.images) {
        const name = `${plan.sources[i].prefix}__${image.name}`
        const group = `${plan.sources[i].prefix}__${image.group}`
        const entry = manifest.get(image.name)!
        const dest = path.join(stage, ...name.split('/'))
        await fs.mkdir(path.dirname(dest), { recursive: true })
        await fs.copyFile(path.join(plan.sources[i].root, prepared.task, 'images', entry.split, entry.file), dest)
        merged.images.push({ ...image, name, group })
        renameMap.push({ source: plan.sources[i].name, from: image.name, to: name, split: image.split, group })
      }
    }
    const snapshot = path.join(stage, 'project.json')
    await fs.writeFile(snapshot, JSON.stringify(merged), 'utf-8')
    await spawnPython(python, path.join(torchPath, 'tools/localization.py'), [
      'prepare', '--project', snapshot, '--images', stage, '--save_dir', targetDir, '--task', prepared.task,
    ], torchPath)
  } finally {
    await fs.rm(stage, { recursive: true, force: true })
  }
  reportOut.renameMap = renameMap
  reportOut.sourceSplits.merged = Object.fromEntries(['train', 'val', 'test'].map(s => [s, merged.images.filter(i => i.split === s).length]))
  await fs.writeFile(path.join(targetDir, 'merge-report.json'), JSON.stringify({
    mergedName: plan.mergedName, format: plan.format,
    sources: plan.sources.map((s, i) => ({ id: s.id, name: s.name, annotationSha256: reportOut.hashes[i] ?? {}, splits: s.splits })),
    classes: plan.classes, renameMap: reportOut.renameMap, mergedSplits: reportOut.sourceSplits.merged, warnings: plan.warnings,
  }, null, 2), 'utf-8')
  const base = path.join(finalDir ?? targetDir, prepared.task)
  const { columns } = readLocalizationDataset(path.join(targetDir, prepared.task))
  plan.newDataset!.numClasses = columns.numClasses
  plan.newDataset!.numAnnotations = columns.numAnnotations
  plan.newDataset!.numTrainImages = columns.numTrainImages
  plan.newDataset!.numEvalImages = columns.numEvalImages
  plan.newDataset!.classStats = columns.classStats
  plan.newDataset!.datasetDir = base
  plan.newDataset!.trainImagePath = 'images/train'
  plan.newDataset!.trainAnnoPath = 'labels/train'
  plan.newDataset!.evalImagePath = 'images/val'
  plan.newDataset!.evalAnnoPath = 'labels/val'
  plan.newDataset!.yamlConfig = stringify({
    num_classes: merged.labels.length,
    localization: { task: prepared.task, data: path.join(base, 'data.yaml').replace(/\\/g, '/') },
  })
}

async function planAnomaly(userRoot: string, rows: MergeRow[]): Promise<{ summary: SourceSummary[]; files: { source: number; file: string }[]; dirs: { source: number; dir: string; kind: string }[] }> {
  const summary: SourceSummary[] = []
  const files: { source: number; file: string }[] = []
  const dirs: { source: number; dir: string; kind: string }[] = []
  for (const [i, row] of rows.entries()) {
    const root = datasetRoot(userRoot, row)
    const splitCounts: Record<string, number> = {}
    let bytes = 0, fileCount = 0
    for (const [kind, rel] of [['normal', row.normalDir || 'train_good'], ['normalTest', row.normalTestDir], ['abnormal', row.abnormalDir], ['mask', row.maskDir]] as const) {
      if (!rel) continue
      const dir = path.isAbsolute(rel) ? rel : path.join(root, rel)
      if (!existsSync(dir)) continue
      dirs.push({ source: i, dir, kind })
      for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
        if (!entry.isFile() || !ANOMALY_IMAGE_EXTENSIONS.includes(path.extname(entry.name).toLowerCase())) continue
        if (kind !== 'mask') files.push({ source: i, file: path.join(dir, entry.name) })
        fileCount++
        bytes += (await fs.stat(path.join(dir, entry.name))).size
        splitCounts[kind] = (splitCounts[kind] || 0) + 1
      }
    }
    summary.push({ id: row.id, name: row.name, prefix: prefixFor(row.name), root, classes: [], splits: splitCounts, fileCount, bytes })
  }
  return { summary, files, dirs }
}

async function mergeAnomaly(plan: MergePlan, dirs: { source: number; dir: string; kind: string }[], rows: MergeRow[], targetDir: string, report: ReportData) {
  const first = rows[0]
  const names = { normal: first.normalDir || 'train_good', normalTest: first.normalTestDir, abnormal: first.abnormalDir, mask: first.maskDir } as Record<string, string | null>
  const counts: Record<string, number> = {}
  const abnormalStems = new Set<string>()
  const masks: { kindDir: string; source: number; file: string }[] = []
  for (const { source, dir, kind } of dirs) {
    const targetName = names[kind]
    if (!targetName) continue
    for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
      if (!entry.isFile() || !ANOMALY_IMAGE_EXTENSIONS.includes(path.extname(entry.name).toLowerCase())) continue
      const renamed = `${plan.sources[source].prefix}__${entry.name}`
      if (kind === 'mask') { masks.push({ kindDir: targetName || '', source, file: path.join(dir, entry.name) }); continue }
      const dest = path.join(targetDir, targetName)
      await fs.mkdir(dest, { recursive: true })
      await fs.copyFile(path.join(dir, entry.name), path.join(dest, renamed))
      report.renameMap.push({ source: plan.sources[source].name, from: `${kind}/${entry.name}`, to: `${kind}/${renamed}`, split: kind })
      report.sourceSplits[plan.sources[source].name][kind] = (report.sourceSplits[plan.sources[source].name][kind] || 0) + 1
      counts[kind] = (counts[kind] || 0) + 1
      if (kind === 'abnormal') abnormalStems.add(`${plan.sources[source].prefix}__${path.basename(entry.name, path.extname(entry.name))}`)
    }
  }
  for (const { source, file, kindDir } of masks) {
    const stem = `${plan.sources[source].prefix}__${path.basename(file, path.extname(file))}`
    if (!abnormalStems.has(stem)) {
      plan.warnings.push(`Mask ${file} has no defect image with the same name; skipped.`)
      continue
    }
    if (!kindDir) {
      plan.warnings.push(`Mask ${file} skipped: the merged dataset has no mask directory.`)
      continue
    }
    const dest = path.join(targetDir, kindDir)
    await fs.mkdir(dest, { recursive: true })
    const maskName = `${stem}${path.extname(file)}`
    await fs.copyFile(file, path.join(dest, maskName))
    report.renameMap.push({ source: plan.sources[source].name, from: `mask/${path.basename(file)}`, to: `mask/${maskName}`, split: 'mask' })
    counts.mask = (counts.mask || 0) + 1
  }
  report.sourceSplits.merged = { ...counts } as Record<string, number>
  plan.newDataset!.numClasses = 2
  plan.newDataset!.numTrainImages = counts.normal || 0
  plan.newDataset!.numEvalImages = (counts.normalTest || 0) + (counts.abnormal || 0)
  plan.newDataset!.numAnnotations = counts.mask || 0
  plan.newDataset!.normalDir = names.normal
  plan.newDataset!.normalTestDir = names.normalTest ?? null
  plan.newDataset!.abnormalDir = names.abnormal ?? null
  plan.newDataset!.maskDir = names.mask ?? null
  plan.newDataset!.classStats = JSON.stringify({
    train: [{ id: 0, name: 'normal', count: counts.normal || 0, imageCount: counts.normal || 0 }],
    eval: [{ id: 1, name: 'anomalous', count: counts.abnormal || 0, imageCount: counts.abnormal || 0 }],
  })
}

/* ---------------- top-level ---------------- */

const FORMAT_CONTAINER: Record<string, string> = {
  COCO: 'COCO', PaddleSeg: 'PaddleSeg', AnomalyFolder: 'AnomalyFolder',
  YOLOOBB: 'localization', YOLOPose: 'localization',
}

export async function planDatasetMerge(opts: {
  userRoot: string; rows: MergeRow[]; mergedName: string; nameTaken?: boolean;
}): Promise<MergePlan> {
  const { userRoot, rows, mergedName } = opts
  const format = effectiveFormat(rows[0])
  const framework = rows[0].project.framework
  const plan: MergePlan = {
    format, framework, canMerge: false, blockers: [], warnings: [], sources: [], duplicates: [],
    mergedName,
  }
  const container = FORMAT_CONTAINER[format]
  if (!container) throw new Error(`Cannot merge datasets of format ${format}`)
  const slug = toSafeSlug(mergedName, 'dataset')
  // managed localization dirs get a uuid suffix like imports
  const managedRoot = (format === 'YOLOOBB' || format === 'YOLOPose')
    ? path.join(userRoot, container, `${slug}-${randomUUID().slice(0, 8)}`)
    : path.join(userRoot, container, slug)
  plan.targetDir = managedRoot
  if (opts.nameTaken) plan.blockers.push('Name is taken: a dataset with this name already exists; choose another name.')
  if ((format === 'COCO' || format === 'PaddleSeg' || format === 'AnomalyFolder') && existsSync(managedRoot)) {
    plan.blockers.push('Name is taken: the target directory already exists; choose another name.')
  }
  const newDataset: MergePlan['newDataset'] = {
    datasetDir: managedRoot,
    trainImagePath: null, trainAnnoPath: null, evalImagePath: null, evalAnnoPath: null,
    normalDir: null, normalTestDir: null, abnormalDir: null, maskDir: null,
    numClasses: 0, numAnnotations: 0, numTrainImages: 0, numEvalImages: 0, classStats: '{}',
    yamlConfig: '',
  }
  if (format === 'COCO') {
    const { summary, classes, docs, files } = await planCoco(userRoot, rows)
    plan.sources = summary
    plan.classes = classes
    plan.duplicates = await findDuplicates(files)
    Object.assign(newDataset, {
      datasetDir: path.join(container, slug).replace(/\\/g, '/'),
      trainImagePath: 'data/train', trainAnnoPath: 'data/annotations/instance_train.json',
      evalImagePath: 'data/val', evalAnnoPath: 'data/annotations/instance_val.json',
      numClasses: classes.union.length,
    })
    plan.newDataset = newDataset
    ;(plan as any)._docs = docs
  } else if (format === 'PaddleSeg') {
    const { summary, classNames, identical, lists, files } = await planSeg(userRoot, rows)
    plan.sources = summary
    plan.classes = { union: classNames, mismatched: false }
    if (!identical) plan.blockers.push('PaddleSeg class_names.txt must be identical across sources: pixel values encode the class index.')
    plan.duplicates = await findDuplicates(files)
    Object.assign(newDataset, {
      datasetDir: path.join(managedRoot, 'data'),
      numClasses: classNames.length, trainAnnoPath: 'train.txt', evalAnnoPath: 'val.txt',
    })
    plan.newDataset = newDataset
    ;(plan as any)._lists = lists
  } else if (format === 'YOLOOBB' || format === 'YOLOPose') {
    const prepared = await planLocalization(userRoot, rows)
    plan.sources = prepared.summary
    plan.classes = prepared.classes
    if (format === 'YOLOPose') {
      const first = prepared.keypointNames[0]
      if (prepared.keypointNames.some(names => names.join('\n') !== first.join('\n'))) {
        plan.blockers.push('Keypoint names must be identical in order across pose sources (flip_idx / kpt_shape consistency).')
      }
    }
    plan.duplicates = await findDuplicates(prepared.files)
    newDataset.datasetDir = plan.targetDir!
    plan.newDataset = newDataset
    ;(plan as any)._prepared = prepared
  } else if (format === 'AnomalyFolder') {
    const { summary, files, dirs } = await planAnomaly(userRoot, rows)
    plan.sources = summary
    plan.duplicates = await findDuplicates(files)
    newDataset.datasetDir = plan.targetDir!
    plan.newDataset = newDataset
    ;(plan as any)._dirs = dirs
  }
  plan.needsAllowClassMismatch = Boolean(plan.classes?.mismatched)
  if (plan.needsAllowClassMismatch) {
    plan.warnings.push('Class sets differ between sources; a batch lacking a class may contain unlabeled instances of it. Merge requires allowClassMismatch.')
  }
  if (plan.duplicates.length) plan.blockers.push(`${plan.duplicates.length} identical file(s) appear in multiple sources; remove or rename the overlap first.`)
  plan.canMerge = plan.blockers.length === 0 && !plan.needsAllowClassMismatch
  return plan
}

/** Write merged files. Caller guarantees quota and target-dir nonexistence. */
export async function executeDatasetMerge(opts: {
  plan: MergePlan; rows: MergeRow[]; userRoot: string; python?: string; torchPath?: string;
}): Promise<void> {
  const { plan, rows } = opts
  const targetDir = plan.targetDir!
  const parent = path.dirname(targetDir)
  await fs.mkdir(parent, { recursive: true })
  const stage = await fs.mkdtemp(path.join(parent, '.merge-'))
  const stageTarget = path.join(stage, path.basename(targetDir))
  const report: ReportData = { hashes: [], renameMap: [], sourceSplits: { merged: {} } }
  // sha256 of each source's annotation files
  for (const row of rows) {
    const root = datasetRoot(opts.userRoot, row)
    const files: Record<string, string> = {}
    try {
      if (plan.format === 'YOLOOBB' || plan.format === 'YOLOPose') {
        files['project.json'] = await fileHash(path.join(path.dirname(root), 'project.json'))
      } else if (plan.format === 'COCO') {
        for (const [key, rel] of [['train', row.trainAnnoPath || 'data/annotations/instance_train.json'], ['val', row.evalAnnoPath || 'data/annotations/instance_val.json']] as const) {
          files[key] = await fileHash(path.isAbsolute(rel) ? rel : path.join(root, rel))
        }
      } else if (plan.format === 'PaddleSeg') {
        for (const [key, rel] of [['train', row.trainAnnoPath || 'train.txt'], ['val', row.evalAnnoPath || 'val.txt'], ['test', 'test.txt'], ['class_names', 'class_names.txt']] as const) {
          const f = path.isAbsolute(rel) ? rel : path.join(root, rel)
          if (existsSync(f)) files[key] = await fileHash(f)
        }
      }
    } catch { /* missing side files are non-fatal for the report */ }
    report.hashes.push(files)
  }
  for (const s of plan.sources) report.sourceSplits[s.name] = {}
  try {
    if (plan.format === 'COCO') {
      await mergeCoco(plan, (plan as any)._docs, stageTarget, report)
      await writeMergeReport(stageTarget, plan, report)
    } else if (plan.format === 'PaddleSeg') {
      await mergeSeg(plan, (plan as any)._lists, path.join(stageTarget, 'data'), report)
      await writeMergeReport(stageTarget, plan, report)
    } else if (plan.format === 'YOLOOBB' || plan.format === 'YOLOPose') {
      await mergeLocalization(plan, (plan as any)._prepared, stageTarget, opts.python!, opts.torchPath!, targetDir, report)
    } else if (plan.format === 'AnomalyFolder') {
      await mergeAnomaly(plan, (plan as any)._dirs, rows, stageTarget, report)
      await writeMergeReport(stageTarget, plan, report)
    }
    // finalize row fields that need the generated yaml (datasetDir stays the real target)
    if (plan.format !== 'YOLOOBB' && plan.format !== 'YOLOPose') {
      plan.newDataset!.yamlConfig = generateDatasetYaml({
        name: plan.mergedName!, format: plan.format, numClasses: plan.newDataset!.numClasses,
        datasetDir: plan.newDataset!.datasetDir,
        trainImagePath: plan.newDataset!.trainImagePath, trainAnnoPath: plan.newDataset!.trainAnnoPath,
        evalImagePath: plan.newDataset!.evalImagePath, evalAnnoPath: plan.newDataset!.evalAnnoPath,
        normalDir: plan.newDataset!.normalDir, normalTestDir: plan.newDataset!.normalTestDir,
        abnormalDir: plan.newDataset!.abnormalDir, maskDir: plan.newDataset!.maskDir,
        task: rows[0].project.task ?? undefined, framework: plan.framework,
      })
    }
    if (existsSync(targetDir)) throw new Error('Target dataset directory was created concurrently.')
    await fs.rename(stageTarget, targetDir)
    if (plan.format === 'YOLOOBB' || plan.format === 'YOLOPose') {
      // prepare bakes the staged absolute path into data.yaml; point it at the final dir.
      const prepared = (plan as any)._prepared
      const dataYaml = path.join(targetDir, prepared.task, 'data.yaml')
      const parsed = JSON.parse(await fs.readFile(dataYaml, 'utf-8'))
      parsed.path = path.join(targetDir, prepared.task).replace(/\\/g, '/')
      await fs.writeFile(dataYaml, JSON.stringify(parsed, null, 2), 'utf-8')
    }
  } catch (error) {
    await fs.rm(stage, { recursive: true, force: true })
    throw error
  }
  await fs.rm(stage, { recursive: true, force: true })
}

/** Side files written at dataset creation: classes json + configs yml (existing files only is NOT applied for merge — create them like import does). */
export async function writeMergedSideFiles(opts: {
  config: { userDatabasePath: string | null; userConfigsPath: string | null };
  username: string; plan: MergePlan;
}) {
  const { config, username, plan } = opts
  const nd = plan.newDataset!
  if (config.userDatabasePath && plan.classes?.union.length) {
    const datasetsDir = path.join(config.userDatabasePath, username, 'datasets')
    await fs.mkdir(datasetsDir, { recursive: true })
    const stats = (() => { try { return JSON.parse(nd.classStats) } catch { return {} } })()
    await fs.writeFile(path.join(datasetsDir, `${toSafeSlug(plan.mergedName ?? 'merged', 'dataset')}_classes.json`), JSON.stringify({
      datasetName: plan.mergedName, numClasses: nd.numClasses,
      classNames: plan.classes.union,
      classDetails: stats.train ?? [],
      createdAt: new Date().toISOString(),
    }, null, 2), 'utf-8')
  }
  if (config.userConfigsPath && nd.yamlConfig) {
    const slug = toSafeSlug(plan.mergedName ?? 'merged', 'dataset').toLowerCase()
    const dir = path.join(config.userConfigsPath, username, 'datasets')
    await fs.mkdir(dir, { recursive: true })
    await fs.writeFile(path.join(dir, `${slug}.yml`), nd.yamlConfig, 'utf-8')
  }
}
