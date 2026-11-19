import fs from 'fs/promises'
import path from 'path'

/**
 * Pick the usable PaddleDetection checkpoint inside a save_dir:
 * best_model.pdparams (eval enabled) → model_final.pdparams (finished run) →
 * the highest numbered N.pdparams (stopped run). Null when none exists.
 */
export async function paddleDetectionWeights(saveDir: string): Promise<string | null> {
  for (const name of ['best_model.pdparams', 'model_final.pdparams']) {
    if ((await fs.stat(path.join(saveDir, name)).catch(() => null))?.isFile()) return path.join(saveDir, name)
  }
  const numbered = (await fs.readdir(saveDir).catch(() => [] as string[]))
    .map(name => /^(\d+)\.pdparams$/.exec(name))
    .filter((m): m is RegExpExecArray => !!m)
    .map(m => Number(m[1]))
    .sort((a, b) => b - a)
  return numbered.length ? path.join(saveDir, `${numbered[0]}.pdparams`) : null
}
