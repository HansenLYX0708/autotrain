import { NextRequest, NextResponse } from 'next/server'
import path from 'path'
import { db } from '@/lib/db'
import { requireAuth } from '@/lib/auth'
import { logActivity } from '@/lib/activity-log'
import { frameworkMeta, resolvePythonPath } from '@/lib/frameworks'
import { effectiveFormat, executeDatasetMerge, folderSize, planDatasetMerge, writeMergedSideFiles, type MergeRow } from '@/lib/dataset-merge'

export const runtime = 'nodejs'
export const maxDuration = 1800

/** POST /api/datasets/merge — {action: 'preview'|'merge', datasetIds: string[] (>=2), name, projectId?, allowClassMismatch?} */
export async function POST(request: NextRequest) {
  try {
    const auth = await requireAuth(request)
    if (auth instanceof NextResponse) return auth
    const body = await request.json()
    const action = body.action === 'merge' ? 'merge' : 'preview'
    const datasetIds: string[] = Array.isArray(body.datasetIds) ? body.datasetIds.map(String) : []
    if (datasetIds.length < 2) {
      return NextResponse.json({ success: false, error: 'Select at least two datasets to merge.' }, { status: 400 })
    }
    const name = String(body.name ?? '').trim()
    if (!name || name.length > 100 || /[\\/\x00-\x1f]/.test(name) || name.includes('..')) {
      return NextResponse.json({ success: false, error: 'Invalid name: required, 1–100 chars, no path separators or control characters.' }, { status: 400 })
    }

    const rows = await db.dataset.findMany({
      where: { id: { in: datasetIds } },
      include: { project: { select: { id: true, framework: true, task: true } } },
    }) as unknown as MergeRow[]
    if (rows.length !== datasetIds.length) {
      return NextResponse.json({ success: false, error: 'One or more datasets were not found.' }, { status: 404 })
    }
    // Keep caller order: the first source defines class order and defaults.
    rows.sort((a, b) => datasetIds.indexOf(a.id) - datasetIds.indexOf(b.id))
    if (auth.role !== 'admin' && rows.some(row => row.userId !== auth.userId)) {
      return NextResponse.json({ success: false, error: 'You can only merge your own datasets.' }, { status: 403 })
    }
    if (new Set(rows.map(row => row.userId)).size !== 1) {
      return NextResponse.json({ success: false, error: 'All sources must belong to the same user.' }, { status: 400 })
    }
    const framework = rows[0].project.framework
    if (rows.some(row => row.project.framework !== framework)) {
      return NextResponse.json({ success: false, error: 'Only datasets from the same framework can be merged.' }, { status: 400 })
    }
    const format = effectiveFormat(rows[0])
    if (rows.some(row => effectiveFormat(row) !== format)) {
      return NextResponse.json({ success: false, error: 'Only datasets of the same format can be merged.' }, { status: 400 })
    }

    const projectId = body.projectId ? String(body.projectId) : rows[0].projectId
    const project = await db.project.findUnique({ where: { id: projectId }, select: { id: true, framework: true, userId: true } })
    if (!project || project.framework !== framework) {
      return NextResponse.json({ success: false, error: 'Target project must exist and use the same framework.' }, { status: 400 })
    }
    if (auth.role !== 'admin' && project.userId !== auth.userId) {
      return NextResponse.json({ success: false, error: 'Target project does not belong to you.' }, { status: 403 })
    }

    const owner = await db.user.findUnique({ where: { id: rows[0].userId }, select: { id: true, username: true, maxStorageQuota: true } })
    const config = await db.systemConfig.findFirst()
    if (!owner || !config?.userDatabasePath) {
      return NextResponse.json({ success: false, error: 'User storage is not configured.' }, { status: 400 })
    }
    const userRoot = path.join(config.userDatabasePath, owner.username)

    const nameTaken = !!(await db.dataset.findFirst({ where: { name, userId: rows[0].userId }, select: { id: true } }))
    if (nameTaken && action === 'merge') {
      return NextResponse.json({ success: false, error: 'A dataset with this name already exists.' }, { status: 400 })
    }
    const plan = await planDatasetMerge({ userRoot, rows, mergedName: name || rows[0].name, nameTaken })
    if (action === 'preview') {
      const { _docs, _lists, _prepared, _dirs, ...preview } = plan as unknown as Record<string, unknown>
      return NextResponse.json({ success: true, data: preview })
    }
    if (plan.blockers.length) {
      return NextResponse.json({ success: false, error: plan.blockers[0], data: { ...plan, _docs: undefined, _lists: undefined, _prepared: undefined, _dirs: undefined } }, { status: 400 })
    }
    if (plan.needsAllowClassMismatch && body.allowClassMismatch !== true) {
      return NextResponse.json({ success: false, error: 'Class sets differ between sources; confirm with allowClassMismatch.' }, { status: 400 })
    }

    const additional = plan.sources.reduce((sum, s) => sum + s.bytes, 0)
    const used = await folderSize(userRoot)
    if (used + additional > Number(owner.maxStorageQuota)) {
      return NextResponse.json({ success: false, error: 'Storage quota exceeded by this merge.' }, { status: 403 })
    }

    let python: string | undefined, torchPath: string | undefined
    if (format === 'YOLOOBB' || format === 'YOLOPose') {
      if (!config.torchPath) return NextResponse.json({ success: false, error: 'Configure the PyTorch trainer path in Settings.' }, { status: 400 })
      const resolved = resolvePythonPath(framework, '0', config)
      if (!resolved.pythonPath) return NextResponse.json({ success: false, error: `Configure the Python environment for ${framework} in Settings.` }, { status: 400 })
      python = resolved.pythonPath
      torchPath = config.torchPath
    }

    const created = await executeDatasetMerge({ plan, rows, userRoot, python, torchPath }).then(async () => {
      const nd = plan.newDataset!
      return db.dataset.create({
        data: {
          name, description: `Merged from: ${plan.sources.map(s => s.name).join(', ')}`,
          projectId: project.id, userId: owner.id, format,
          datasetDir: nd.datasetDir,
          trainImagePath: nd.trainImagePath, trainAnnoPath: nd.trainAnnoPath,
          evalImagePath: nd.evalImagePath, evalAnnoPath: nd.evalAnnoPath,
          normalDir: nd.normalDir, normalTestDir: nd.normalTestDir,
          abnormalDir: nd.abnormalDir, maskDir: nd.maskDir,
          numClasses: nd.numClasses, numAnnotations: nd.numAnnotations,
          numTrainImages: nd.numTrainImages, numEvalImages: nd.numEvalImages,
          classStats: nd.classStats, yamlConfig: nd.yamlConfig,
        },
      })
    }).catch(async error => {
      // Row creation happens only after files are complete; a failed row leaves
      // files behind, so clean the staged target up.
      await import('fs/promises').then(fsm => fsm.rm(plan.targetDir!, { recursive: true, force: true })).catch(() => {})
      throw error
    })
    await writeMergedSideFiles({ config, username: owner.username, plan }).catch(() => {})
    await logActivity(auth.userId, {
      action: 'import_dataset', entityType: 'dataset', entityId: created.id, entityName: name,
      details: { mergedFrom: datasetIds, format, framework: frameworkMeta(framework).datasetFormat },
    })
    return NextResponse.json({ success: true, data: { dataset: created, sources: plan.sources, classes: plan.classes, warnings: plan.warnings } })
  } catch (error) {
    console.error('Dataset merge failed:', error)
    return NextResponse.json({ success: false, error: error instanceof Error ? error.message : 'Failed to merge datasets.' }, { status: 500 })
  }
}
