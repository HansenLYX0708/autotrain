/** Browser/server shared annotation contract. Coordinates are original-image pixels. */
import { z } from 'zod'

const finite = z.number().finite()
const name = z.string().trim().min(1).max(100)
const names = z.array(name).max(200).refine(v => new Set(v).size === v.length, 'Names must be unique')
const relativePath = z.string().min(1).max(500).refine(
  v => !/[\\:\x00-\x1f]/.test(v) && v.split('/').every(s => s && s !== '.' && s !== '..'),
  'Image name must be a relative path with forward slashes',
)
export const pointSchema = z.object({ x: finite, y: finite })
export const objectSchema = z.object({
  id: name,
  label: name,
  kind: z.enum(['rectangle', 'obb']),
  cx: finite, cy: finite,
  width: finite.positive(), height: finite.positive(),
  // Clockwise degrees in image coordinates (x right, y down).
  angle: finite.min(-180).max(180),
  keypoints: z.record(name, z.object({ x: finite, y: finite, visibility: z.union([z.literal(0), z.literal(1), z.literal(2)]) })),
})
export const imageSchema = z.object({
  name: relativePath,
  width: z.number().int().positive(), height: z.number().int().positive(),
  objects: z.array(objectSchema).max(10000),
  reviewed: z.boolean(),
  split: z.enum(['unassigned', 'train', 'val', 'test']),
  group: z.string().trim().max(200),
})
export const projectSchema = z.object({
  format: z.literal('autotrain-localization'),
  version: z.literal(1),
  id: z.string().uuid(),
  name,
  labels: names.min(1),
  keypointNames: names,
  images: z.array(imageSchema).max(50000),
}).superRefine((project, ctx) => {
  const seen = new Set<string>()
  project.images.forEach((image, i) => {
    const key = image.name.toLowerCase()
    if (seen.has(key)) ctx.addIssue({ code: 'custom', path: ['images', i], message: `Duplicate image: ${image.name}` })
    seen.add(key)
    const ids = new Set<string>()
    for (const object of image.objects) {
      if (ids.has(object.id)) ctx.addIssue({ code: 'custom', message: `Duplicate instance: ${image.name}/${object.id}` })
      ids.add(object.id)
      if (!project.labels.includes(object.label)) ctx.addIssue({ code: 'custom', message: `Unknown class: ${object.label}` })
      if (object.kind === 'rectangle' && object.angle !== 0) ctx.addIssue({ code: 'custom', message: 'Axis-aligned rectangle must have angle 0' })
      for (const keypoint of Object.keys(object.keypoints)) {
        if (!project.keypointNames.includes(keypoint)) ctx.addIssue({ code: 'custom', message: `Unknown keypoint: ${keypoint}` })
      }
    }
  })
})

export type Point = z.infer<typeof pointSchema>
export type AnnotationObject = z.infer<typeof objectSchema>
export type AnnotationImage = z.infer<typeof imageSchema>
export type AnnotationProject = z.infer<typeof projectSchema>
export type Split = AnnotationImage['split']

export function newProject(): AnnotationProject {
  return {
    format: 'autotrain-localization', version: 1, id: crypto.randomUUID(),
    name: 'localization', labels: ['slider_abs', 'tapring'],
    keypointNames: ['datum_a', 'datum_b', 'datum_c'],
    images: [],
  }
}

export function boxCorners(box: Pick<AnnotationObject, 'cx' | 'cy' | 'width' | 'height' | 'angle'>): Point[] {
  const radians = box.angle * Math.PI / 180
  const c = Math.cos(radians), s = Math.sin(radians)
  return [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([x, y]) => ({
    x: box.cx + x * box.width / 2 * c - y * box.height / 2 * s,
    y: box.cy + x * box.width / 2 * s + y * box.height / 2 * c,
  }))
}

/** Three clicks: directed first edge, then a point on the opposite edge. */
export function boxFromThreePoints(a: Point, b: Point, p: Point) {
  const dx = b.x - a.x, dy = b.y - a.y
  const width = Math.hypot(dx, dy)
  if (width < 1) throw new Error('第一条边太短，请重新绘制')
  const nx = -dy / width, ny = dx / width
  const signedHeight = (p.x - a.x) * nx + (p.y - a.y) * ny
  if (Math.abs(signedHeight) < 1) throw new Error('矩形高度必须至少为 1 像素')
  return {
    cx: (a.x + b.x) / 2 + nx * signedHeight / 2,
    cy: (a.y + b.y) / 2 + ny * signedHeight / 2,
    width, height: Math.abs(signedHeight), angle: Math.atan2(dy, dx) * 180 / Math.PI,
  }
}

export function containsPoint(box: AnnotationObject, p: Point): boolean {
  const angle = -box.angle * Math.PI / 180
  const dx = p.x - box.cx, dy = p.y - box.cy
  return Math.abs(dx * Math.cos(angle) - dy * Math.sin(angle)) <= box.width / 2 &&
    Math.abs(dx * Math.sin(angle) + dy * Math.cos(angle)) <= box.height / 2
}

export function insideImage(p: Point, image: Pick<AnnotationImage, 'width' | 'height'>): boolean {
  return p.x >= -1e-6 && p.y >= -1e-6 && p.x <= image.width + 1e-6 && p.y <= image.height + 1e-6
}

export function imageIssues(image: AnnotationImage, keypointNames: string[], pose = true): string[] {
  const issues: string[] = []
  if (!image.reviewed) issues.push('尚未确认完成（空图也需要确认）')
  if (image.split === 'unassigned') issues.push('尚未分配 train / val / test')
  if (!image.group) issues.push('尚未填写采集批次')
  image.objects.forEach((object, index) => {
    const prefix = `实例 ${index + 1} (${object.label})`
    if (object.width < 1 || object.height < 1) issues.push(`${prefix}：框小于 1 像素`)
    if (!boxCorners(object).every(p => insideImage(p, image))) issues.push(`${prefix}：框超出图片`)
    if (pose && !keypointNames.some(n => (object.keypoints[n]?.visibility ?? 0) > 0)) issues.push(`${prefix}：没有可监督的关键点`)
    for (const key of keypointNames) {
      const point = object.keypoints[key]
      if (!point) {
        if (pose) issues.push(`${prefix}：关键点 ${key} 尚未处理`)
      } else if (point.visibility > 0 && !insideImage(point, image)) {
        issues.push(`${prefix}：关键点 ${key} 超出图片`)
      }
    }
  })
  return issues
}

export function projectIssues(project: AnnotationProject, pose = true): string[] {
  const issues = project.images.flatMap(image => imageIssues(image, project.keypointNames, pose).map(issue => `${image.name}：${issue}`))
  if (!project.images.length) issues.push('尚未载入图片')
  if (pose && !project.keypointNames.length) issues.push('关键点模板为空')
  for (const split of ['train', 'val'] as const) {
    if (!project.images.some(image => image.split === split)) issues.push(`缺少 ${split} 图片`)
    for (const label of project.labels) {
      if (!project.images.some(image => image.split === split && image.objects.some(o => o.label === label))) issues.push(`${split} 缺少类别 ${label} 的实例`)
    }
  }
  const groups = new Map<string, Split>()
  for (const image of project.images) {
    if (!image.group || image.split === 'unassigned') continue
    const split = groups.get(image.group)
    if (split && split !== image.split) issues.push(`采集批次 ${image.group} 跨越 ${split} / ${image.split}，存在数据泄漏`)
    groups.set(image.group, image.split)
  }
  return issues
}

/** Ordinary COCO/semantic-mask converters cannot preserve this task's semantics. */
export function localizationConversionError(data: unknown): string | null {
  if (!data || typeof data !== 'object' || !('shapes' in data) || !Array.isArray(data.shapes)) return null
  if (data.shapes.some(shape => shape?.shape_type === 'point' || shape?.flags?.autotrain_obb === true)) {
    return 'OBB/keypoint annotations cannot be converted to ordinary detection/segmentation without losing pose information. Export the annotation project and use tools/localization/prepare.py instead.'
  }
  return null
}

type LabelmeShape = {
  label: string; points: number[][]; group_id: number | string | null
  shape_type: string; flags: Record<string, unknown>
}

export function toLabelme(image: AnnotationImage, keypointNames: string[]) {
  const shapes: LabelmeShape[] = []
  image.objects.forEach((object, index) => {
    const group = index + 1
    const corners = boxCorners(object)
    shapes.push({
      label: object.label, group_id: group,
      shape_type: object.kind === 'rectangle' ? 'rectangle' : 'polygon',
      points: (object.kind === 'rectangle' ? [corners[0], corners[2]] : corners).map(p => [p.x, p.y]),
      flags: { autotrain_obb: object.kind === 'obb' },
    })
    for (const key of keypointNames) {
      const point = object.keypoints[key]
      if (!point) continue
      shapes.push({
        label: key, group_id: group, shape_type: 'point',
        points: [[point.visibility === 0 ? 0 : point.x, point.visibility === 0 ? 0 : point.y]],
        flags: { visibility: point.visibility },
      })
    }
  })
  return {
    version: '5.5.0', flags: { autotrain_reviewed: image.reviewed },
    shapes, imagePath: image.name.split('/').pop(), imageData: null,
    imageHeight: image.height, imageWidth: image.width,
    autotrain: { keypointNames, split: image.split, group: image.group },
  }
}

const labelmeSchema = z.object({
  imageWidth: z.number().int().positive(), imageHeight: z.number().int().positive(),
  flags: z.record(z.string(), z.unknown()).optional(),
  shapes: z.array(z.object({
    label: name, points: z.array(z.tuple([finite, finite])),
    shape_type: z.string(), group_id: z.union([z.number(), z.string(), z.null()]).optional(),
    flags: z.record(z.string(), z.unknown()).optional(),
  })),
  autotrain: z.object({ keypointNames: names, split: imageSchema.shape.split, group: imageSchema.shape.group }).optional(),
})

/** Reject unsupported shapes instead of silently flattening or losing their geometry. */
export function fromLabelme(raw: unknown, imageName: string): { image: AnnotationImage; keypointNames: string[] } {
  const data = labelmeSchema.parse(raw)
  const objects: AnnotationObject[] = []
  const groups = new Map<string, AnnotationObject>()
  for (const [index, shape] of data.shapes.entries()) {
    if (shape.shape_type === 'point') continue
    const points = shape.points.map(([x, y]) => ({ x, y }))
    let geometry: ReturnType<typeof boxFromThreePoints>
    let kind: AnnotationObject['kind']
    if (shape.shape_type === 'rectangle' && [2, 4].includes(points.length)) {
      const xs = points.map(p => p.x), ys = points.map(p => p.y)
      const x1 = Math.min(...xs), y1 = Math.min(...ys), x2 = Math.max(...xs), y2 = Math.max(...ys)
      geometry = { cx: (x1 + x2) / 2, cy: (y1 + y2) / 2, width: x2 - x1, height: y2 - y1, angle: 0 }
      kind = 'rectangle'
    } else if (shape.shape_type === 'polygon' && points.length === 4) {
      geometry = boxFromThreePoints(points[0], points[1], points[2])
      const expected = boxCorners(geometry)
      const tolerance = Math.max(0.01, Math.max(geometry.width, geometry.height) * 1e-5)
      if (points.some(p => !expected.some(q => Math.hypot(q.x - p.x, q.y - p.y) < tolerance))) {
        throw new Error(`${shape.label} 是非矩形四边形；请保留原始标注，另行标注 OBB，不能直接当作旋转矩形`)
      }
      kind = 'obb'
    } else {
      throw new Error(`不支持 ${shape.shape_type} / ${points.length} 点形状：${shape.label}`)
    }
    const object = objectSchema.parse({ id: `instance-${index + 1}`, label: shape.label, kind, ...geometry, keypoints: {} })
    if (shape.group_id != null) {
      const key = String(shape.group_id)
      if (groups.has(key)) throw new Error(`group_id ${key} 对应多个框`)
      groups.set(key, object)
    }
    objects.push(object)
  }
  const keypointNames = [...(data.autotrain?.keypointNames ?? [])]
  for (const shape of data.shapes) {
    if (shape.shape_type !== 'point') continue
    const object = shape.group_id == null ? undefined : groups.get(String(shape.group_id))
    if (!object || shape.points.length !== 1) throw new Error(`关键点 ${shape.label} 必须通过 group_id 对应一个框`)
    if (object.keypoints[shape.label]) throw new Error(`实例内重复关键点：${shape.label}`)
    const visibility = shape.flags?.visibility ?? 2
    if (visibility !== 0 && visibility !== 1 && visibility !== 2) throw new Error(`关键点 ${shape.label} 的 visibility 无效`)
    object.keypoints[shape.label] = { x: shape.points[0][0], y: shape.points[0][1], visibility }
    if (!keypointNames.includes(shape.label)) keypointNames.push(shape.label)
  }
  return {
    image: imageSchema.parse({
      name: imageName, width: data.imageWidth, height: data.imageHeight, objects,
      reviewed: data.flags?.autotrain_reviewed === true,
      split: data.autotrain?.split ?? 'unassigned', group: data.autotrain?.group ?? '',
    }),
    keypointNames,
  }
}
