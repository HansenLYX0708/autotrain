/**
 * Lib-level checks for src/lib/dataset-merge.ts on synthetic datasets.
 * Run: npx tsx scripts/check-dataset-merge.ts
 */
import assert from 'node:assert/strict'
import fs from 'fs/promises'
import path from 'path'
import crypto from 'crypto'
import { existsSync } from 'fs'
import { planDatasetMerge, executeDatasetMerge, type MergeRow } from '../src/lib/dataset-merge'

const ROOT = path.resolve('temp/merge-check')
const USER_ROOT = path.join(ROOT, 'user')

function row(partial: Partial<MergeRow>): MergeRow {
  return {
    id: crypto.randomUUID(), name: 'ds', userId: 'u', projectId: 'p', format: 'COCO',
    datasetDir: null, trainImagePath: null, trainAnnoPath: null, evalImagePath: null, evalAnnoPath: null,
    normalDir: null, normalTestDir: null, abnormalDir: null, maskDir: null,
    numClasses: 1, yamlConfig: null, project: { id: 'p', framework: 'PaddleDetection', task: 'detection' },
    ...partial,
  }
}

async function coco(root: string, cats: string[], nTrain: number, nVal: number, content = 'x') {
  const dir = path.join(root, 'data')
  for (const [split, n] of [['train', nTrain], ['val', nVal]] as const) {
    await fs.mkdir(path.join(dir, split), { recursive: true })
    await fs.mkdir(path.join(dir, 'annotations'), { recursive: true })
    const doc = {
      images: [] as any[], annotations: [] as any[],
      categories: cats.map((name, i) => ({ id: i + 1, name })),
    }
    for (let i = 0; i < n; i++) {
      const name = `im${i}.jpg`
      await fs.writeFile(path.join(dir, split, name), content + root + split + i)
      doc.images.push({ id: i + 1, file_name: name, width: 100, height: 80 })
      doc.annotations.push({ id: i + 1, image_id: i + 1, category_id: 1, bbox: [1, 2, 3, 4], area: 12, iscrowd: 0, segmentation: [[1, 2, 3, 4, 5, 6]] })
    }
    await fs.writeFile(path.join(dir, 'annotations', `instance_${split}.json`), JSON.stringify(doc))
  }
}

async function seg(root: string, names: string[], n: number) {
  await fs.mkdir(path.join(root, 'JPEGImages'), { recursive: true })
  await fs.mkdir(path.join(root, 'Annotations'), { recursive: true })
  await fs.writeFile(path.join(root, 'class_names.txt'), names.join('\n') + '\n')
  const lines: string[] = []
  for (let i = 0; i < n; i++) {
    await fs.writeFile(path.join(root, 'JPEGImages', `s${i}.png`), `img${root}${i}`)
    await fs.writeFile(path.join(root, 'Annotations', `s${i}.png`), `lbl${root}${i}`)
    lines.push(`JPEGImages/s${i}.png Annotations/s${i}.png`)
  }
  await fs.writeFile(path.join(root, 'train.txt'), lines.join('\n') + '\n')
  await fs.writeFile(path.join(root, 'val.txt'), lines[0] + '\n')
}

async function localization(root: string, task: 'obb' | 'pose', labels: string[], keypoints: string[], n: number) {
  const project = {
    format: 'autotrain-localization', version: 1, id: crypto.randomUUID(), name: 'p',
    labels, keypointNames: keypoints,
    images: [] as any[],
  }
  const manifest: any[] = []
  for (let i = 0; i < n; i++) {
    const name = `im${i}.jpg`
    const split = i === 0 ? 'val' : 'train'
    project.images.push({
      name, width: 64, height: 64, reviewed: true, split, group: `g${i}`,
      objects: [{ id: `o${i}`, label: labels[0], kind: 'obb', cx: 30, cy: 30, width: 20, height: 10, angle: 15, keypoints: {} }],
    })
    manifest.push({ source: name, file: `00000${i}_deadbeef.jpg`, split, group: `g${i}`, sha256: `${i}` })
    await fs.mkdir(path.join(root, task, 'images', split), { recursive: true })
    // valid PIL-readable image (PPM magic inside a .jpg name is fine for prepare)
    const digest = crypto.createHash('sha256').update(`${root}:${i}`).digest()
    const pixels = Buffer.alloc(64 * 64 * 3)
    for (let j = 0; j < pixels.length; j++) pixels[j] = digest[j % digest.length]
    const ppm = Buffer.concat([Buffer.from(`P6\n64 64\n255\n`), pixels])
    await fs.writeFile(path.join(root, task, 'images', split, `00000${i}_deadbeef.jpg`), ppm)
    await fs.mkdir(path.join(root, task, 'labels', split), { recursive: true })
    await fs.writeFile(path.join(root, task, 'labels', split, `00000${i}_deadbeef.txt`), '')
  }
  await fs.writeFile(path.join(root, 'project.json'), JSON.stringify(project))
  await fs.writeFile(path.join(root, 'manifest.json'), JSON.stringify(manifest))
  await fs.writeFile(path.join(root, task, 'schema.json'), JSON.stringify({ task, labels, keypointNames: keypoints }))
}

async function anomaly(root: string, withMask = true) {
  for (const [dir, n] of [['train_good', 3], ['normal_test', 1], ['defects', 2]] as const) {
    await fs.mkdir(path.join(root, dir), { recursive: true })
    for (let i = 0; i < n; i++) await fs.writeFile(path.join(root, dir, `a${i}.png`), `a${root}${dir}${i}`)
  }
  if (withMask) {
    await fs.mkdir(path.join(root, 'masks'), { recursive: true })
    for (let i = 0; i < 2; i++) await fs.writeFile(path.join(root, 'masks', `a${i}.png`), `m${root}${i}`)
  }
}

const obbFramework = { id: 'p', framework: 'TorchOBB', task: 'detection' }
const poseFramework = { id: 'p', framework: 'TorchPose', task: 'detection' }

async function main() {
  await fs.rm(ROOT, { recursive: true, force: true })
  const python = 'D:/pythonEnvs/torchEnv/Scripts/python.exe'
  const torchPath = path.resolve('torchtrain')

  // --- COCO: same classes, merged files
  const r1 = row({ name: 'src-a', datasetDir: 'COCO/src-a', project: { ...obbFramework } })
  const r2 = row({ name: 'src-b', datasetDir: 'COCO/src-b', project: { ...obbFramework } })
  await coco(path.join(USER_ROOT, 'COCO/src-a'), ['cat', 'dog'], 2, 1)
  await coco(path.join(USER_ROOT, 'COCO/src-b'), ['cat', 'dog'], 1, 2)
  let plan = await planDatasetMerge({ userRoot: USER_ROOT, rows: [r1, r2], mergedName: 'coco-merged' })
  assert.equal(plan.canMerge, true)
  assert.deepEqual(plan.classes!.union, ['cat', 'dog'])
  await executeDatasetMerge({ plan, rows: [r1, r2], userRoot: USER_ROOT })
  const mergedTrain = JSON.parse(await fs.readFile(path.join(plan.targetDir!, 'data/annotations/instance_train.json'), 'utf-8'))
  assert.equal(mergedTrain.images.length, 3)
  assert.equal(mergedTrain.annotations.length, 3)
  assert.equal(mergedTrain.categories.length, 2)
  assert.ok(mergedTrain.images.every((i: any) => i.file_name.includes('__')))
  assert.ok(mergedTrain.images.every((i: any) => i.file_name.startsWith('src-') && i.file_name.includes('__im')))
  // merge-report.json written for COCO
  const cocoReport = JSON.parse(await fs.readFile(path.join(plan.targetDir!, 'merge-report.json'), 'utf-8'))
  assert.equal(cocoReport.sources.length, 2)
  assert.ok(cocoReport.sources[0].annotationSha256.train)

  // COCO file_name with a subdirectory must flatten into a single dir
  const sub = JSON.parse(await fs.readFile(path.join(USER_ROOT, 'COCO/src-b/data/annotations/instance_train.json'), 'utf-8'))
  sub.images[0].file_name = 'nested/dir/' + sub.images[0].file_name
  await fs.mkdir(path.join(USER_ROOT, 'COCO/src-b/data/train/nested/dir'), { recursive: true })
  await fs.rename(path.join(USER_ROOT, 'COCO/src-b/data/train/im0.jpg'), path.join(USER_ROOT, 'COCO/src-b/data/train', sub.images[0].file_name))
  await fs.writeFile(path.join(USER_ROOT, 'COCO/src-b/data/annotations/instance_train.json'), JSON.stringify(sub))
  plan = await planDatasetMerge({ userRoot: USER_ROOT, rows: [r1, r2], mergedName: 'coco-sub' })
  assert.equal(plan.canMerge, true)
  await executeDatasetMerge({ plan, rows: [r1, r2], userRoot: USER_ROOT })
  assert.ok(existsSync(path.join(plan.targetDir!, 'data/train/src-b__nested__dir__im0.jpg')), 'subdir file_name flattened')

  // --- COCO class mismatch requires allowClassMismatch
  const r3 = row({ name: 'src-c', datasetDir: 'COCO/src-c', project: { ...obbFramework } })
  await coco(path.join(USER_ROOT, 'COCO/src-c'), ['cat', 'fish'], 1, 1)
  plan = await planDatasetMerge({ userRoot: USER_ROOT, rows: [r1, r3], mergedName: 'mismatch' })
  assert.equal(plan.needsAllowClassMismatch, true)
  assert.equal(plan.canMerge, false)
  assert.deepEqual(plan.classes!.union, ['cat', 'dog', 'fish'])

  // --- duplicate content across sources blocks
  await fs.copyFile(path.join(USER_ROOT, 'COCO/src-a/data/train/im0.jpg'), path.join(USER_ROOT, 'COCO/src-c/data/train/im0.jpg'))
  plan = await planDatasetMerge({ userRoot: USER_ROOT, rows: [r1, r3], mergedName: 'dup' })
  assert.equal(plan.duplicates.length >= 1, true)
  assert.equal(plan.canMerge, false)

  // --- PaddleSeg merge + class_names mismatch rejection
  const s1 = row({ name: 'seg-a', format: 'PaddleSeg', datasetDir: path.join(USER_ROOT, 'PaddleSeg/seg-a/data'), project: { id: 'p', framework: 'PaddleSeg', task: 'segmentation' } })
  const s2 = row({ name: 'seg-b', format: 'PaddleSeg', datasetDir: path.join(USER_ROOT, 'PaddleSeg/seg-b/data'), project: { id: 'p', framework: 'PaddleSeg', task: 'segmentation' } })
  const s3 = row({ name: 'seg-c', format: 'PaddleSeg', datasetDir: path.join(USER_ROOT, 'PaddleSeg/seg-c/data'), project: { id: 'p', framework: 'PaddleSeg', task: 'segmentation' } })
  await seg(path.join(USER_ROOT, 'PaddleSeg/seg-a/data'), ['bg', 'c1'], 2)
  await seg(path.join(USER_ROOT, 'PaddleSeg/seg-b/data'), ['bg', 'c1'], 1)
  await seg(path.join(USER_ROOT, 'PaddleSeg/seg-c/data'), ['bg', 'other'], 1)
  plan = await planDatasetMerge({ userRoot: USER_ROOT, rows: [s1, s2], mergedName: 'seg-merged' })
  assert.equal(plan.canMerge, true)
  await executeDatasetMerge({ plan, rows: [s1, s2], userRoot: USER_ROOT })
  const trainTxt = await fs.readFile(path.join(plan.targetDir!, 'data/train.txt'), 'utf-8')
  assert.equal(trainTxt.trim().split('\n').length, 3)
  assert.ok(trainTxt.includes('seg-a__s0.png'))
  // identical label PNGs across sources must NOT block (dup detection = images only)
  await seg(path.join(USER_ROOT, 'PaddleSeg/seg-d/data'), ['bg', 'c1'], 2)
  const s4 = row({ name: 'seg-d', format: 'PaddleSeg', datasetDir: path.join(USER_ROOT, 'PaddleSeg/seg-d/data'), project: { id: 'p', framework: 'PaddleSeg', task: 'segmentation' } })
  // overwrite seg-d labels with seg-a's exact bytes; keep images different
  for (const f of await fs.readdir(path.join(USER_ROOT, 'PaddleSeg/seg-d/data/Annotations'))) {
    await fs.copyFile(path.join(USER_ROOT, 'PaddleSeg/seg-a/data/Annotations', f), path.join(USER_ROOT, 'PaddleSeg/seg-d/data/Annotations', f))
  }
  plan = await planDatasetMerge({ userRoot: USER_ROOT, rows: [s1, s4], mergedName: 'seg-samelabels' })
  assert.equal(plan.canMerge, true, 'identical label PNGs must not block')

  plan = await planDatasetMerge({ userRoot: USER_ROOT, rows: [s1, s3], mergedName: 'seg-mismatch' })
  assert.equal(plan.canMerge, false)
  assert.ok(plan.blockers[0].includes('class_names'))

  // --- localization (YOLOOBB): label union, group prefixing, prepare output
  const l1 = row({ name: 'loc-a', format: 'YOLOOBB', datasetDir: path.join(USER_ROOT, 'localization/loc-a/obb'), project: { ...obbFramework } })
  const l2 = row({ name: 'loc-b', format: 'YOLOOBB', datasetDir: path.join(USER_ROOT, 'localization/loc-b/obb'), project: { ...obbFramework } })
  await localization(path.join(USER_ROOT, 'localization/loc-a'), 'obb', ['slider_abs'], [], 3)
  await localization(path.join(USER_ROOT, 'localization/loc-b'), 'obb', ['slider_abs'], [], 2)
  plan = await planDatasetMerge({ userRoot: USER_ROOT, rows: [l1, l2], mergedName: 'loc-merged' })
  assert.equal(plan.canMerge, true)
  await executeDatasetMerge({ plan, rows: [l1, l2], userRoot: USER_ROOT, python, torchPath })
  const mp = JSON.parse(await fs.readFile(path.join(plan.targetDir!, 'project.json'), 'utf-8'))
  assert.equal(mp.images.length, 5)
  assert.ok(mp.images.every((i: any) => i.name.includes('__') && i.group.includes('__')))
  assert.ok((await fs.readFile(path.join(plan.targetDir!, 'obb/data.yaml'), 'utf-8')).includes('slider_abs'))
  const report = JSON.parse(await fs.readFile(path.join(plan.targetDir!, 'merge-report.json'), 'utf-8'))
  assert.equal(report.renameMap.length, 5)

  // --- pose: keypointNames mismatch rejected
  const p1 = row({ name: 'pose-a', format: 'YOLOPose', datasetDir: path.join(USER_ROOT, 'localization/pose-a/pose'), project: { ...poseFramework } })
  const p2 = row({ name: 'pose-b', format: 'YOLOPose', datasetDir: path.join(USER_ROOT, 'localization/pose-b/pose'), project: { ...poseFramework } })
  await localization(path.join(USER_ROOT, 'localization/pose-a'), 'pose', ['p1'], ['k1', 'k2'], 2)
  await localization(path.join(USER_ROOT, 'localization/pose-b'), 'pose', ['p1'], ['k2', 'k1'], 2)
  plan = await planDatasetMerge({ userRoot: USER_ROOT, rows: [p1, p2], mergedName: 'pose-mismatch' })
  assert.equal(plan.canMerge, false)
  assert.ok(plan.blockers[0].toLowerCase().includes('keypoint'))

  // --- anomaly: mask pairing + counts
  const a1 = row({ name: 'an-a', format: 'AnomalyFolder', datasetDir: path.join(USER_ROOT, 'AnomalyFolder/an-a'), normalDir: 'train_good', normalTestDir: 'normal_test', abnormalDir: 'defects', maskDir: 'masks', project: { id: 'p', framework: 'TorchAnomaly', task: 'anomaly' } })
  const a2 = row({ name: 'an-b', format: 'AnomalyFolder', datasetDir: path.join(USER_ROOT, 'AnomalyFolder/an-b'), normalDir: 'train_good', normalTestDir: 'normal_test', abnormalDir: 'defects', maskDir: 'masks', project: { id: 'p', framework: 'TorchAnomaly', task: 'anomaly' } })
  await anomaly(path.join(USER_ROOT, 'AnomalyFolder/an-a'))
  await anomaly(path.join(USER_ROOT, 'AnomalyFolder/an-b'))
  plan = await planDatasetMerge({ userRoot: USER_ROOT, rows: [a1, a2], mergedName: 'an-merged' })
  assert.equal(plan.canMerge, true)
  await executeDatasetMerge({ plan, rows: [a1, a2], userRoot: USER_ROOT })
  assert.equal((await fs.readdir(path.join(plan.targetDir!, 'masks'))).length, 4)
  assert.equal((await fs.readdir(path.join(plan.targetDir!, 'defects'))).length, 4)
  assert.equal(plan.newDataset!.numTrainImages, 6)

  // --- '' format falls back to frameworkMeta datasetFormat
  const le = row({ name: 'loc-e', format: '', datasetDir: path.join(USER_ROOT, 'localization/loc-a/obb'), project: { ...obbFramework } })
  plan = await planDatasetMerge({ userRoot: USER_ROOT, rows: [le, l2], mergedName: 'loc-fallback' })
  assert.equal(plan.format, 'YOLOOBB')
  assert.equal(plan.canMerge, true)

  console.log('PASS: dataset merge for COCO, PaddleSeg, YOLOOBB, YOLOPose, AnomalyFolder')
}

main().catch(error => { console.error(error); process.exit(1) })
