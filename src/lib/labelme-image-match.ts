/** Pure matching rules shared by filesystem converters and the browser annotator. */
export type ImageResolution = { path: string; via: 'imagePath' | 'jsonName' } | { error: string }
export interface ImageNameIndex {
  byName: Map<string, string[]>
  byStem: Map<string, string[]>
}

const segments = (value: string) => value.split(/[\\/]/).filter(v => v && v !== '.' && v !== '..').map(v => v.toLowerCase())
const extension = (value: string) => (value.match(/\.[^./\\]+$/)?.[0] ?? '').toLowerCase()

export function resolveIndexedLabelmeImage(
  index: ImageNameIndex, jsonFileName: string, imagePath: unknown, relativeName: (file: string) => string = file => file,
): ImageResolution {
  const raw = imagePath == null ? '' : String(imagePath)
  const wanted = segments(raw)
  const name = wanted.at(-1) ?? ''
  const jsonStem = jsonFileName.replace(/\.json$/i, '')
  const listed = (paths: string[]) => paths.slice(0, 3).map(relativeName).join(', ') + (paths.length > 3 ? ', …' : '')
  if (name) {
    const candidates = index.byName.get(name) ?? []
    if (candidates.length === 1) return { path: candidates[0], via: 'imagePath' }
    if (candidates.length > 1) {
      const matches = candidates.filter(candidate => {
        const rel = segments(relativeName(candidate))
        const take = Math.min(wanted.length, rel.length)
        return rel.slice(-take).join('/') === wanted.slice(-take).join('/')
      })
      if (matches.length === 1) return { path: matches[0], via: 'imagePath' }
      return { error: `ambiguous image "${name}": ${candidates.length} files under imgs (${listed(candidates)})` }
    }
  }
  if (jsonStem) {
    const candidates = index.byStem.get(jsonStem.toLowerCase()) ?? []
    if (candidates.length === 1) return { path: candidates[0], via: 'jsonName' }
    if (candidates.length > 1) {
      const matches = candidates.filter(candidate => extension(candidate) === extension(name))
      if (matches.length === 1) return { path: matches[0], via: 'jsonName' }
      return { error: `ambiguous image "${jsonStem}.*": ${candidates.length} files under imgs (${listed(candidates)})` }
    }
  }
  return { error: `image not found: imagePath "${raw}"; looked for ${name ? `"${name}" and ` : ''}"${jsonStem}${extension(name) || '.*'}" under imgs (including subfolders)` }
}

/** Browser uploads retain each JSON's directory, which can resolve a remaining tie. */
export function resolveUploadedLabelmeImage(index: ImageNameIndex, jsonRelativePath: string, imagePath: unknown): ImageResolution {
  const jsonName = jsonRelativePath.split('/').pop()!
  const resolved = resolveIndexedLabelmeImage(index, jsonName, imagePath)
  if (!('error' in resolved)) return resolved
  const stem = jsonName.replace(/\.json$/i, '').toLowerCase()
  const relativeStem = jsonRelativePath.replace(/\.json$/i, '').toLowerCase()
  const siblings = (index.byStem.get(stem) ?? []).filter(file => file.replace(/\.[^./]+$/, '').toLowerCase() === relativeStem)
  return siblings.length === 1 ? { path: siblings[0], via: 'jsonName' } : resolved
}
