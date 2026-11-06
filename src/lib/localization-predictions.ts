import { z } from 'zod'
import { imageSchema, type AnnotationImage, type AnnotationObject } from './localization-annotations'

const finite = z.number().finite()
const predictedInstance = z.object({
  label: z.string(), confidence: finite.min(0).max(1),
  center_px: z.tuple([finite, finite]).optional(),
  long_side_px: finite.positive().optional(), short_side_px: finite.positive().optional(),
  angle_deg_mod180: finite.optional(),
  bbox_xyxy_px: z.tuple([finite, finite, finite, finite]).optional(),
  keypoints: z.record(z.string(), z.object({ x: finite, y: finite, confidence: finite.nullable() })).optional(),
})
export const predictionsSchema = z.object({
  jobId: z.string(), task: z.enum(['obb', 'pose']), labels: z.array(z.string()),
  keypointNames: z.array(z.string()),
  images: z.array(z.object({
    source: z.string(), width: finite.int().positive(), height: finite.int().positive(),
    instances: z.array(predictedInstance),
  })),
})

export function predictionToImage(image: AnnotationImage, result: z.infer<typeof predictionsSchema>, prediction: z.infer<typeof predictionsSchema>['images'][number]): AnnotationImage {
  if (image.name !== prediction.source || image.width !== prediction.width || image.height !== prediction.height) throw new Error(`预测图片或原始尺寸不匹配：${image.name}`)
  const objects: AnnotationObject[] = prediction.instances.map(instance => {
    let box: Pick<AnnotationObject, 'kind' | 'cx' | 'cy' | 'width' | 'height' | 'angle'>
    if (result.task === 'obb') {
      if (!instance.center_px || !instance.long_side_px || !instance.short_side_px || instance.angle_deg_mod180 === undefined) throw new Error('模型未返回旋转框几何')
      box = { kind: 'obb', cx: instance.center_px[0], cy: instance.center_px[1], width: instance.long_side_px, height: instance.short_side_px, angle: instance.angle_deg_mod180 }
    } else {
      if (!instance.bbox_xyxy_px) throw new Error('模型未返回关键点实例框')
      const [x1, y1, x2, y2] = instance.bbox_xyxy_px
      box = { kind: 'rectangle', cx: (x1 + x2) / 2, cy: (y1 + y2) / 2, width: x2 - x1, height: y2 - y1, angle: 0 }
    }
    return {
      ...box, id: crypto.randomUUID(), label: instance.label,
      prediction: { jobId: result.jobId, confidence: instance.confidence },
      keypoints: Object.fromEntries(Object.entries(instance.keypoints || {})
        .filter(([, point]) => point.confidence !== null && point.confidence >= 0.5)
        .map(([name, point]) => [name, { x: point.x, y: point.y, visibility: 2 as const }])),
    }
  })
  return imageSchema.parse({
    ...image, objects, reviewed: false,
    prediction: { jobId: result.jobId, task: result.task, createdAt: new Date().toISOString() },
  })
}
