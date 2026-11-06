import fs from 'fs'
import path from 'path'
import { boxCorners, projectSchema } from './localization-annotations'

/** Managed YOLO datasets retain their immutable annotation snapshot and file manifest. */
export function readLocalizationDataset(root: string) {
  const project = projectSchema.parse(JSON.parse(fs.readFileSync(path.join(root, '..', 'project.json'), 'utf8')))
  const manifest: { source: string; file: string; split: string }[] = JSON.parse(fs.readFileSync(path.join(root, '..', 'manifest.json'), 'utf8'))
  const files = new Map(manifest.map(entry => [entry.source, entry]))
  const categories = project.labels.map((name, id) => ({ id, name }))
  const stats = (split: string) => {
    const images = project.images.filter(image => image.split === split)
    return categories.map(category => ({
      ...category, count: images.reduce((sum, image) => sum + image.objects.filter(object => object.label === category.name).length, 0),
      imageCount: images.filter(image => image.objects.some(object => object.label === category.name)).length,
    }))
  }
  const samples = project.images.map((image, index) => {
    const file = files.get(image.name)
    if (!file || path.basename(file.file) !== file.file || !['train', 'val', 'test'].includes(file.split)) throw new Error(`Invalid dataset manifest for ${image.name}`)
    return {
      id: index, fileName: `${image.split}/${image.name}`, width: image.width, height: image.height,
      imagePath: path.join(root, 'images', file.split, file.file),
      annotations: image.objects.map((object, i) => {
        const polygon = boxCorners(object)
        const xs = polygon.map(point => point.x), ys = polygon.map(point => point.y)
        return {
          id: i, categoryId: project.labels.indexOf(object.label), categoryName: object.label,
          bbox: [Math.min(...xs), Math.min(...ys), Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)],
          polygon, keypoints: object.keypoints, area: object.width * object.height,
        }
      }),
    }
  })
  return {
    project, categories, samples,
    columns: {
      numClasses: categories.length, numAnnotations: project.images.reduce((sum, image) => sum + image.objects.length, 0),
      numTrainImages: project.images.filter(image => image.split === 'train').length,
      numEvalImages: project.images.filter(image => image.split === 'val').length,
      classStats: JSON.stringify({ train: stats('train'), eval: stats('val'), test: stats('test') }),
    },
  }
}
