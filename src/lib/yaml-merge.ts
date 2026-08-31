/**
 * AST-level deep merge for Paddle YAML configs.
 *
 * A training job's final config is assembled from three independently authored
 * documents: dataset -> training -> model. The original implementation simply
 * concatenated the three texts, which has two problems:
 *
 *   1. Duplicate top-level keys. PyYAML tolerates them (last occurrence wins),
 *      but the document is invalid YAML — `yaml.parse()` on the JS side throws
 *      `Map keys must be unique`, so nothing in this app can re-read a job
 *      config it just wrote.
 *   2. All-or-nothing overrides. Because the collision is resolved at the top
 *      level, a training config cannot refine one nested field (e.g. PaddleSeg's
 *      `train_dataset.transforms[0].target_size`) without restating the whole
 *      `train_dataset` block that the dataset config owns.
 *
 * Merging on the parsed document tree fixes both. We deliberately operate on
 * `yaml`'s AST rather than plain JS objects because PaddleDetection configs use
 * custom tags (`!COCODataSet`, `!CosineDecay`, ...). `YAML.parse()` silently
 * drops those tags; `YAML.parseDocument()` preserves them across a round trip.
 *
 * Merge semantics (later source wins):
 *   - map + map      -> recursive merge
 *   - anything else  -> replace wholesale (sequences are replaced, not
 *                       concatenated: an overriding transform pipeline should
 *                       fully define itself)
 *   - a tag on the overriding node replaces the tag on the base node
 */

import { Document, isMap, isSeq, parseDocument, type Node, type YAMLMap } from 'yaml';

export function setTopLevelSaveDir(yaml: string, saveDir: string): string {
  if (/^save_dir:\s*.*$/m.test(yaml)) {
    return yaml.replace(/^save_dir:\s*.*$/gm, `save_dir: ${saveDir}`);
  }
  return `${yaml.trimEnd()}\nsave_dir: ${saveDir}\n`;
}

export function setPaddleClasOutputDir(yaml: string, outputDir: string): string {
  const doc = parseDocument(yaml, { logLevel: 'silent' });
  if (doc.errors.length > 0 || !isMap(doc.contents)) return yaml;
  doc.setIn(['Global', 'output_dir'], outputDir);
  return doc.toString({ lineWidth: 0 });
}

export interface YamlSource {
  /** Human-readable label used in the emitted header comment and in errors. */
  label: string;
  /** Raw YAML text. Empty/whitespace-only sources are skipped. */
  content: string | null | undefined;
}

export interface MergeResult {
  /** The merged YAML text, ready to be written to disk. */
  yaml: string;
  /**
   * True when every source parsed cleanly and a real deep merge happened.
   * False means we fell back to plain text concatenation (legacy behaviour) —
   * the job will still run under PyYAML, but nested overrides were not applied.
   */
  merged: boolean;
  /** Parse/merge problems, safe to surface to the user or the server log. */
  warnings: string[];
}

/** Recursively merge `override` into `base`, mutating `base`. */
function mergeNodes(doc: Document, base: YAMLMap, override: YAMLMap): void {
  for (const item of override.items) {
    const key = item.key;
    // `YAMLMap.get` with keepScalar returns the node; we need the existing
    // *value node* so we can recurse into it when both sides are maps.
    const existing = base.get(key as never, true) as Node | undefined;
    const incoming = item.value as Node | undefined;

    if (isMap(existing) && isMap(incoming)) {
      mergeNodes(doc, existing, incoming);
      // A tag on the incoming node is an explicit intent to retype the block
      // (e.g. switching `!COCODataSet` -> `!VOCDataSet`), so it wins.
      if (incoming.tag) existing.tag = incoming.tag;
      continue;
    }

    // Scalars, sequences, and map/non-map mismatches: replace outright.
    base.set(key, incoming);
  }
}

function nonEmpty(sources: YamlSource[]): YamlSource[] {
  return sources.filter((s) => typeof s.content === 'string' && s.content.trim().length > 0);
}

/** Legacy behaviour: header-commented text concatenation. */
function concatenate(sources: YamlSource[]): string {
  return sources.map((s) => `# ${s.label}\n${s.content!.trim()}`).join('\n\n') + '\n';
}

/**
 * Deep-merge Paddle YAML documents in order (later sources override earlier).
 *
 * Never throws: if any source fails to parse we fall back to concatenation and
 * report the reason in `warnings`, so a malformed hand-written config degrades
 * to the old behaviour instead of blocking job creation.
 */
export function mergeYamlConfigs(sources: YamlSource[]): MergeResult {
  const present = nonEmpty(sources);
  const warnings: string[] = [];

  if (present.length === 0) return { yaml: '', merged: true, warnings };

  const docs: Document[] = [];
  for (const source of present) {
    const doc = parseDocument(source.content!, { logLevel: 'silent' });
    if (doc.errors.length > 0) {
      warnings.push(`${source.label}: ${doc.errors[0].message}`);
      return { yaml: concatenate(present), merged: false, warnings };
    }
    if (!isMap(doc.contents)) {
      warnings.push(`${source.label}: top level is not a mapping, cannot deep-merge`);
      return { yaml: concatenate(present), merged: false, warnings };
    }
    docs.push(doc);
  }

  const target = docs[0];
  for (let i = 1; i < docs.length; i++) {
    mergeNodes(target, target.contents as YAMLMap, docs[i].contents as YAMLMap);
  }

  const header = `# Merged by AutoTrain from: ${present.map((s) => s.label).join(' -> ')}\n`;
  // `lineWidth: 0` disables line folding — Paddle configs contain long path and
  // target_size lists that must not be wrapped.
  return { yaml: header + target.toString({ lineWidth: 0 }), merged: true, warnings };
}

/**
 * Read a single top-level scalar from a YAML document without throwing.
 * Returns undefined for malformed input or a missing key.
 */
export function readTopLevel<T = unknown>(yamlText: string | null | undefined, key: string): T | undefined {
  if (!yamlText || !yamlText.trim()) return undefined;
  try {
    const doc = parseDocument(yamlText, { logLevel: 'silent' });
    if (doc.errors.length > 0 || !isMap(doc.contents)) return undefined;
    const value = doc.contents.get(key);
    return value as T | undefined;
  } catch {
    return undefined;
  }
}

/** True when the node is a sequence — re-exported so callers avoid a direct `yaml` import. */
export { isSeq };
