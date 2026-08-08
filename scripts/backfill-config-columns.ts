/**
 * Backfill the display columns on TrainingConfig / Model / TrainingJob from
 * each record's authoritative `yamlConfig`.
 *
 * Why this is needed: the original import code parsed YAML with a
 * PaddleDetection-only reader (and a hand-rolled regex scanner for models), so
 * rows created before the framework-aware parsers landed hold wrong values.
 * Observed on a real database:
 *   - a PaddleSeg training config stored "100 epochs / batch 8 / lr 0.001 /
 *     CosineDecay / 640x640" for a YAML that actually said "160000 iters /
 *     batch 4 / lr 0.005 / PolynomialDecay / 512x512"
 *   - a PaddleSeg model stored "YOLOv3 / CSPResNet / PPYOLOEHead / 1 class"
 *     for a PPLiteSeg / STDC2 / 2-class model
 *   - PaddleDetection configs stored scheduler="LinearWarmup" because the old
 *     parser let the trailing warmup entry overwrite the decay policy
 *   - a PaddleSeg job stored totalEpochs=100 while currentEpoch had reached
 *     4313, so its progress bar read 4313%
 *
 * `yamlConfig` is never modified — this only recomputes the derived columns.
 *
 * Usage:
 *   npx tsx scripts/backfill-config-columns.ts                  # dry run
 *   npx tsx scripts/backfill-config-columns.ts --apply          # write changes
 *   npx tsx scripts/backfill-config-columns.ts --db "file:D:/path/custom.db"
 */

import { PrismaClient } from '@prisma/client';
import {
  asConfigFramework,
  defaultTrainingParams,
  parseTrainingParams,
  trainingParamsToColumns,
  totalStepsFor,
  type ConfigFramework,
  type TrainingParams,
} from '../src/lib/training-yaml';
import { defaultModelParams, modelParamsToColumns, parseModelParams } from '../src/lib/model-yaml';
import { mergeYamlConfigs } from '../src/lib/yaml-merge';

/**
 * Parse YAML into training params, but refuse to guess.
 *
 * `parseTrainingParams` returns `{}` for a document it cannot read, and layering
 * that over the defaults yields a full, plausible-looking parameter set that
 * corresponds to nothing. Writing that to the database would be worse than
 * leaving the stale values alone — during development this script initially
 * proposed changing a job's length to 20000 (the PaddleSeg default) for a job
 * whose YAML plainly said `iters: 160000`, purely because the stored document
 * had duplicate keys and failed to parse.
 *
 * Returns null when nothing could be recovered, so callers can skip.
 */
function resolveTrainingParams(
  framework: ConfigFramework,
  yamlText: string | null,
): { params: TrainingParams; stated: Set<string> } | null {
  if (!yamlText || !yamlText.trim()) return null;
  const parsed = parseTrainingParams(framework, yamlText);
  if (Object.keys(parsed).length === 0) return null;
  return {
    params: { ...defaultTrainingParams(framework), ...parsed },
    stated: new Set(Object.keys(parsed)),
  };
}

/**
 * Columns whose value is only trustworthy if the YAML actually stated the
 * underlying parameter.
 *
 * A config's YAML frequently omits things — a PaddleSeg schedule that uses
 * scale-jitter + crop has no `Resize.target_size`, and `save_interval` is often
 * passed on the command line instead. For those, `defaultTrainingParams` fills
 * in a plausible number that came from nowhere. Overwriting a stale value with
 * an invented one is not an improvement, so each column is gated on the
 * parameters it is derived from.
 */
const CONFIG_COLUMN_SOURCES: Record<string, string[]> = {
  epoch: ['epochs', 'iters'],
  maxEpochs: ['maxEpochs', 'iters'],
  iters: ['iters'],
  saveInterval: ['saveInterval'],
  snapshotEpoch: ['snapshotEpoch', 'saveInterval'],
  batchSize: ['trainBatchSize'],
  baseLr: ['baseLr'],
  momentum: ['momentum'],
  weightDecay: ['weightDecay'],
  scheduler: ['scheduler'],
  warmupEpochs: ['warmupEpochs', 'warmupIters'],
  workerNum: ['workerNum'],
  evalHeight: ['imageHeight'],
  evalWidth: ['imageWidth'],
  useGpu: ['useGpu'],
  logIter: ['logIter'],
  saveDir: ['saveDir'],
  outputDir: ['outputDir'],
  weights: ['weights'],
  pretrainWeights: ['pretrainWeights'],
};

/** Model columns are a 1:1 projection of the same-named parameter. */
const MODEL_COLUMN_SOURCES: Record<string, string[]> = Object.fromEntries(
  [
    'architecture', 'backbone', 'neck', 'head', 'numClasses',
    'normType', 'useEma', 'emaDecay', 'depthMult', 'widthMult', 'pretrainWeights',
  ].map((k) => [k, [k]]),
);

/** Drop columns the YAML said nothing about. */
function keepStatedOnly(
  columns: Record<string, unknown>,
  stated: Set<string>,
  sources: Record<string, string[]>,
): { kept: Record<string, unknown>; omitted: string[] } {
  const kept: Record<string, unknown> = {};
  const omitted: string[] = [];
  for (const [column, value] of Object.entries(columns)) {
    const from = sources[column];
    if (!from || from.some((param) => stated.has(param))) kept[column] = value;
    else omitted.push(column);
  }
  return { kept, omitted };
}

const args = process.argv.slice(2);
const APPLY = args.includes('--apply');
const dbIndex = args.indexOf('--db');
const dbUrl = dbIndex >= 0 ? args[dbIndex + 1] : process.env.DATABASE_URL;

if (!dbUrl) {
  console.error('No database URL. Pass --db "file:ABSOLUTE/PATH/custom.db" or set DATABASE_URL.');
  process.exit(1);
}

const prisma = new PrismaClient({ datasources: { db: { url: dbUrl } } });

type Diff = { field: string; from: unknown; to: unknown };

/** Compare only the fields we intend to write, ignoring float noise. */
function diffColumns(current: Record<string, unknown>, next: Record<string, unknown>): Diff[] {
  const diffs: Diff[] = [];
  for (const [field, to] of Object.entries(next)) {
    const from = current[field];
    if (typeof from === 'number' && typeof to === 'number') {
      if (Math.abs(from - to) > 1e-9) diffs.push({ field, from, to });
      continue;
    }
    // Normalise null vs '' vs undefined so we do not report cosmetic churn.
    const norm = (v: unknown) => (v === null || v === undefined || v === '' ? null : v);
    if (norm(from) !== norm(to)) diffs.push({ field, from, to });
  }
  return diffs;
}

function report(label: string, diffs: Diff[]): void {
  if (diffs.length === 0) {
    console.log(`  ${label}: already consistent`);
    return;
  }
  console.log(`  ${label}: ${diffs.length} field(s) to fix`);
  for (const d of diffs) {
    console.log(`      ${d.field.padEnd(16)} ${String(d.from).padEnd(18)} ->  ${d.to}`);
  }
}

let totalChanges = 0;

async function main() {
  console.log(`Database : ${dbUrl}`);
  console.log(`Mode     : ${APPLY ? 'APPLY (writing changes)' : 'DRY RUN (no writes)'}\n`);

  // --- TrainingConfig -------------------------------------------------------
  console.log('=== TrainingConfig ===');
  const configs = await prisma.trainingConfig.findMany({
    include: { project: { select: { framework: true } } },
  });
  for (const config of configs) {
    console.log(`\n"${config.name}" [${config.project?.framework ?? 'PaddleDetection'}]`);
    if (!config.yamlConfig) {
      console.log('  skipped: no yamlConfig to derive from');
      continue;
    }
    const framework = asConfigFramework(config.project?.framework);
    const resolved = resolveTrainingParams(framework, config.yamlConfig);
    if (!resolved) {
      console.log('  SKIPPED: yamlConfig could not be parsed; refusing to write defaults');
      continue;
    }
    const all = trainingParamsToColumns(framework, resolved.params);
    const { kept, omitted } = keepStatedOnly(all, resolved.stated, CONFIG_COLUMN_SOURCES);
    if (omitted.length > 0) {
      console.log(`  not stated in YAML, left untouched: ${omitted.join(', ')}`);
    }
    const diffs = diffColumns(config as unknown as Record<string, unknown>, kept);
    report('columns', diffs);
    totalChanges += diffs.length;
    if (APPLY && diffs.length > 0) {
      await prisma.trainingConfig.update({
        where: { id: config.id },
        data: Object.fromEntries(diffs.map((d) => [d.field, d.to])),
      });
      console.log('      written');
    }
  }

  // --- Model ---------------------------------------------------------------
  console.log('\n\n=== Model ===');
  const models = await prisma.model.findMany({
    include: { project: { select: { framework: true } } },
  });
  for (const model of models) {
    console.log(`\n"${model.name}" [${model.project?.framework ?? 'PaddleDetection'}]`);
    if (!model.yamlConfig) {
      console.log('  skipped: no yamlConfig to derive from');
      continue;
    }
    const framework = asConfigFramework(model.project?.framework);
    const parsedModel = parseModelParams(framework, model.yamlConfig);
    if (Object.keys(parsedModel).length === 0) {
      console.log('  SKIPPED: yamlConfig could not be parsed; refusing to write defaults');
      continue;
    }
    const resolved = { ...defaultModelParams(framework), ...parsedModel };
    const all = modelParamsToColumns(resolved);

    // Same rule as training configs: only write columns the YAML states.
    // A PaddleSeg model config typically has no `num_classes` (PaddleSeg takes
    // it from the dataset), so the default 2 would otherwise be written as if
    // it were fact.
    const stated = new Set(Object.keys(parsedModel));
    // `neck` / `head` do not exist as concepts in PaddleSeg or PaddleClas, so
    // clearing the leftover detection values is a genuine correction.
    if (framework !== 'PaddleDetection') {
      stated.add('neck');
      stated.add('head');
    }
    const { kept, omitted } = keepStatedOnly(all, stated, MODEL_COLUMN_SOURCES);
    if (omitted.length > 0) {
      console.log(`  not stated in YAML, left untouched: ${omitted.join(', ')}`);
    }
    const diffs = diffColumns(model as unknown as Record<string, unknown>, kept);
    report('columns', diffs);
    totalChanges += diffs.length;
    if (APPLY && diffs.length > 0) {
      await prisma.model.update({
        where: { id: model.id },
        data: Object.fromEntries(diffs.map((d) => [d.field, d.to])),
      });
      console.log('      written');
    }
  }

  // --- TrainingJob.totalEpochs --------------------------------------------
  // Only this one column: everything else on a job is a record of what actually
  // ran and must not be rewritten.
  console.log('\n\n=== TrainingJob.totalEpochs ===');
  const jobs = await prisma.trainingJob.findMany({
    include: {
      project: { select: { framework: true } },
      dataset: { select: { yamlConfig: true } },
      model: { select: { yamlConfig: true } },
      config: { select: { yamlConfig: true } },
    },
  });
  for (const job of jobs) {
    const framework = asConfigFramework(job.project?.framework);
    console.log(`\n"${job.name}" [${framework}] status=${job.status} currentEpoch=${job.currentEpoch}`);

    // Prefer the job's own stored config, since that is what actually ran.
    // Jobs created before the deep merge landed hold a text concatenation with
    // duplicate top-level keys, which is not parseable — for those, re-derive
    // from the linked dataset/config/model exactly the way job creation now
    // does. If neither route yields a parse, skip: an unchanged wrong number is
    // better than a confidently wrong one.
    let resolved = resolveTrainingParams(framework, job.yamlConfig);
    let source = 'stored job yaml';

    if (!resolved) {
      const remerged = mergeYamlConfigs([
        { label: 'Dataset Configuration', content: job.dataset?.yamlConfig },
        { label: 'Training Configuration', content: job.config?.yamlConfig },
        { label: 'Model Configuration', content: job.model?.yamlConfig },
      ]);
      if (remerged.merged) {
        resolved = resolveTrainingParams(framework, remerged.yaml);
        source = 're-merged dataset+config+model (stored job yaml is unparseable)';
      }
    }

    if (!resolved) {
      console.log('  SKIPPED: could not determine the training length from any source');
      continue;
    }

    // The training length must be stated, not defaulted: writing a made-up
    // total is how a stopped job ended up claiming 4313/100.
    const lengthParam = framework === 'PaddleSeg' ? 'iters' : 'epochs';
    if (!resolved.stated.has(lengthParam)) {
      console.log(`  SKIPPED: YAML does not state \`${lengthParam}\`; refusing to write a default`);
      continue;
    }

    const totalSteps = totalStepsFor(framework, resolved.params);
    if (job.totalEpochs === totalSteps) {
      console.log(`  totalEpochs: already consistent (via ${source})`);
      continue;
    }
    console.log(`  totalEpochs: ${job.totalEpochs} ->  ${totalSteps}   [via ${source}]`);
    totalChanges++;
    if (APPLY) {
      await prisma.trainingJob.update({ where: { id: job.id }, data: { totalEpochs: totalSteps } });
      console.log('      written');
    }
  }

  console.log('\n' + '='.repeat(60));
  if (totalChanges === 0) {
    console.log('Nothing to do — all derived columns already match their YAML.');
  } else if (APPLY) {
    console.log(`Applied ${totalChanges} field update(s).`);
  } else {
    console.log(`${totalChanges} field(s) would change. Re-run with --apply to write them.`);
  }
}

main()
  .catch((error) => {
    console.error('\nBackfill failed:', error);
    process.exitCode = 1;
  })
  .finally(() => prisma.$disconnect());
