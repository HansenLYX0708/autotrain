/**
 * Framework-aware training-config model: parameters -> YAML and YAML -> parameters.
 *
 * This is the single source of truth for what a "training config" means in this
 * app. It is imported by BOTH the browser (the create/edit dialog renders the
 * form from `TRAINING_FIELD_SUPPORT` and previews `generateTrainingYaml`) and
 * the server (import/update routes call `parseTrainingParams` to fill the
 * display columns). Keeping one implementation is what guarantees the YAML the
 * user previews is byte-identical to the YAML that gets trained on.
 *
 * Design notes:
 *   - Must stay dependency-light and filesystem-free so it can be bundled into
 *     a client component.
 *   - `yamlConfig` on the DB row is the source of truth. The scalar columns
 *     (epoch, baseLr, ...) are a *display cache* derived by `parseTrainingParams`
 *     and are never used to reconstruct the config.
 *   - No YAML anchors are emitted. Anchors (`&eval_size` / `*eval_size`) survive
 *     a deep merge only by luck, and the merged job config is assembled from
 *     three separate documents (see `@/lib/yaml-merge`). Literal values are
 *     merge-safe.
 */

import { parseDocument, isMap, isSeq, type Document } from 'yaml';

export type ConfigFramework =
  | 'PaddleDetection'
  | 'PaddleClas'
  | 'PaddleSeg'
  | 'TorchDet'
  | 'TorchSeg'
  | 'TorchAnomaly';

const CONFIG_FRAMEWORKS: ConfigFramework[] = [
  'PaddleDetection',
  'PaddleClas',
  'PaddleSeg',
  'TorchDet',
  'TorchSeg',
  'TorchAnomaly',
];

export function asConfigFramework(value: string | null | undefined): ConfigFramework {
  return CONFIG_FRAMEWORKS.includes(value as ConfigFramework)
    ? (value as ConfigFramework)
    : 'PaddleDetection';
}

/**
 * The torch frameworks reuse the Paddle *schemas* on purpose: `TorchSeg`
 * consumes PaddleSeg-shaped YAML and `TorchDet` consumes PaddleDetection-shaped
 * YAML (see `torchtrain/torchtrain/config.py`). Keeping one dialect per task
 * means the generators, parsers and the deep merge in `@/lib/yaml-merge` are
 * shared instead of forked, and a project can be migrated between runtimes by
 * changing `project.framework` alone.
 */
type ConfigSchema = 'detection' | 'classification' | 'segmentation' | 'anomaly';

const CONFIG_SCHEMA: Record<ConfigFramework, ConfigSchema> = {
  PaddleDetection: 'detection',
  PaddleClas: 'classification',
  PaddleSeg: 'segmentation',
  TorchDet: 'detection',
  TorchSeg: 'segmentation',
  // TorchAnomaly is the exception to the "reuse a Paddle schema" rule: there is
  // no Paddle anomaly-detection framework to mirror, so it emits anomalib's own
  // `trainer:` / `data:` / `model:` shape. The three-config split still lines up
  // one-to-one with those three blocks.
  TorchAnomaly: 'anomaly',
};

export function configSchemaOf(framework: ConfigFramework): ConfigSchema {
  return CONFIG_SCHEMA[framework];
}

export function isTorchConfigFramework(framework: ConfigFramework): boolean {
  return framework === 'TorchDet' || framework === 'TorchSeg' || framework === 'TorchAnomaly';
}

/** True when the framework measures training length in iterations, not epochs. */
export function countsIterations(framework: ConfigFramework): boolean {
  const schema = CONFIG_SCHEMA[framework];
  return schema === 'segmentation' || schema === 'anomaly';
}

// ---------------------------------------------------------------------------
// Parameter model
// ---------------------------------------------------------------------------

export interface TrainingParams {
  // Schedule ---------------------------------------------------------------
  /** Detection / classification train length. */
  epochs: number;
  /** PaddleSeg trains by iteration, not epoch. */
  iters: number;
  /** Detection/Clas checkpoint cadence, in epochs. */
  snapshotEpoch: number;
  /** PaddleSeg checkpoint cadence, in iterations. */
  saveInterval: number;

  // Optimizer --------------------------------------------------------------
  optimizerType: string;
  baseLr: number;
  momentum: number;
  weightDecay: number;
  /** Detection only: L1/L2 regularizer applied via OptimizerBuilder. */
  regularizerType: string;
  /** Detection only: gradient clipping. `null` omits the key entirely. */
  clipGradByNorm: number | null;

  // LR schedule ------------------------------------------------------------
  scheduler: string;
  /** Detection: `max_epochs` of the decay scheduler (often < `epochs`). */
  maxEpochs: number;
  /** Detection/Clas linear warmup length in epochs. 0 disables warmup. */
  warmupEpochs: number;
  /** PaddleSeg warmup length in iterations. 0 disables warmup. */
  warmupIters: number;
  warmupStartLr: number;
  /** PolynomialDecay exponent (PaddleSeg). */
  power: number;
  /** Final LR for PolynomialDecay (PaddleSeg). */
  endLr: number;
  /** Decay factor for Exp/Piecewise/Step schedulers. */
  gamma: number;
  /** PiecewiseDecay boundaries, expressed in epochs (Det/Clas). */
  milestones: number[];

  // Data pipeline ----------------------------------------------------------
  trainBatchSize: number;
  evalBatchSize: number;
  workerNum: number;
  imageWidth: number;
  imageHeight: number;
  /** Detection: emit BatchRandomResize with `multiScaleSizes`. */
  multiScaleTrain: boolean;
  multiScaleSizes: number[];
  /** Detection augmentation toggles (TrainReader.sample_transforms). */
  augRandomDistort: boolean;
  augRandomExpand: boolean;
  augRandomCrop: boolean;
  augRandomFlip: boolean;
  /** Normalisation. `none` keeps raw 0-255 pixels (PP-YOLOE style). */
  normalizeType: 'none' | 'mean_std';
  normMean: number[];
  normStd: number[];

  /**
   * PaddleSeg: when true the training config restates `train_dataset.transforms`
   * / `val_dataset.transforms`, which the deep merge folds into the dataset
   * config. When false the dataset config's own transforms are used untouched.
   */
  segOverrideTransforms: boolean;
  segAugFlipHorizontal: boolean;
  segAugFlipVertical: boolean;
  segAugDistort: boolean;
  segAugScaleAspect: boolean;
  segAugBlur: boolean;

  // Anomaly detection (TorchAnomaly / anomalib) ----------------------------
  /**
   * Input tiling. Keeps the effective resolution while feeding the model small
   * crops, which is the only way small defects survive on a large image.
   * Supported by PaDiM / PatchCore / ReverseDistillation / STFPM **only** —
   * `ANOMALY_PRESETS[...].supportsTiling` gates the control.
   */
  adTileEnabled: boolean;
  adTileSize: number;
  adTileStride: number;
  /**
   * Validation cadence in steps. Not emitted as Lightning's
   * `val_check_interval`, because an int larger than the number of training
   * batches is a hard error unless `check_val_every_n_epoch` is None — and for
   * a one-epoch model (PatchCore/PaDiM) disabling epoch-end validation would
   * skip validation entirely. The adapter converts this into a safe pair of
   * Trainer arguments once it knows the batch count and the model.
   */
  adValInterval: number;
  /** Metric the best-checkpoint callback monitors, e.g. `image_AUROC`. */
  adBestMetric: string;

  // Runtime ----------------------------------------------------------------
  useGpu: boolean;
  useAmp: boolean;
  useEma: boolean;
  emaDecay: number;
  logIter: number;
  saveDir: string;
  outputDir: string;
  weights: string;
  pretrainWeights: string;
}

export const DEFAULT_TRAINING_PARAMS: TrainingParams = {
  epochs: 80,
  iters: 20000,
  snapshotEpoch: 5,
  saveInterval: 1000,

  optimizerType: 'Momentum',
  baseLr: 0.001,
  momentum: 0.9,
  weightDecay: 0.0005,
  regularizerType: 'L2',
  clipGradByNorm: 35,

  scheduler: 'CosineDecay',
  maxEpochs: 80,
  warmupEpochs: 5,
  warmupIters: 0,
  warmupStartLr: 0,
  power: 0.9,
  endLr: 0,
  gamma: 0.1,
  milestones: [60, 72],

  trainBatchSize: 8,
  evalBatchSize: 2,
  workerNum: 4,
  imageWidth: 640,
  imageHeight: 640,
  multiScaleTrain: true,
  multiScaleSizes: [320, 384, 448, 512, 576, 640, 704, 768],
  augRandomDistort: true,
  augRandomExpand: true,
  augRandomCrop: true,
  augRandomFlip: true,
  normalizeType: 'none',
  normMean: [0, 0, 0],
  normStd: [1, 1, 1],

  segOverrideTransforms: false,
  segAugFlipHorizontal: true,
  segAugFlipVertical: false,
  segAugDistort: false,
  segAugScaleAspect: false,
  segAugBlur: false,

  adTileEnabled: false,
  adTileSize: 512,
  adTileStride: 256,
  adValInterval: 500,
  adBestMetric: 'image_AUROC',

  useGpu: true,
  useAmp: false,
  useEma: false,
  emaDecay: 0.9998,
  logIter: 20,
  saveDir: '',
  outputDir: '',
  weights: '',
  pretrainWeights: '',
};

const SEG_DEFAULTS: Partial<TrainingParams> = {
  optimizerType: 'SGD',
  scheduler: 'PolynomialDecay',
  baseLr: 0.01,
  weightDecay: 0.0005,
  trainBatchSize: 4,
  imageWidth: 512,
  imageHeight: 512,
  multiScaleTrain: false,
  normalizeType: 'mean_std',
  normMean: [0.5, 0.5, 0.5],
  normStd: [0.5, 0.5, 0.5],
  clipGradByNorm: null,
};

/** Per-framework defaults that differ from the shared baseline. */
const FRAMEWORK_DEFAULT_OVERRIDES: Record<ConfigFramework, Partial<TrainingParams>> = {
  PaddleDetection: {},
  PaddleClas: {
    optimizerType: 'Momentum',
    scheduler: 'Cosine',
    baseLr: 0.01,
    weightDecay: 0.00005,
    trainBatchSize: 32,
    evalBatchSize: 32,
    imageWidth: 224,
    imageHeight: 224,
    multiScaleTrain: false,
    normalizeType: 'mean_std',
    normMean: [0.485, 0.456, 0.406],
    normStd: [0.229, 0.224, 0.225],
    clipGradByNorm: null,
  },
  PaddleSeg: SEG_DEFAULTS,
  TorchSeg: {
    ...SEG_DEFAULTS,
    // torchvision segmentation backbones are ImageNet-normalised, and UNet (the
    // sensible default for the small microscopy datasets this platform is used
    // with) trains fine either way.
    normMean: [0.485, 0.456, 0.406],
    normStd: [0.229, 0.224, 0.225],
    segOverrideTransforms: true,
  },
  TorchAnomaly: {
    // `iters` is anomalib's `trainer.max_steps` and is the knob that matters for
    // the student-teacher models (EfficientAD's reference recipe is 70k steps).
    // `epochs` is only a ceiling: the memory-bank models publish
    // `trainer_arguments = {max_epochs: 1}` and anomalib's argument cache
    // overrides whatever we pass, so an oversized value here is harmless.
    iters: 8000,
    epochs: 200,
    trainBatchSize: 8,
    evalBatchSize: 4,
    workerNum: 4,
    useGpu: true,
    useAmp: false,
    logIter: 20,
  },
  TorchDet: {
    // torchvision detectors resize internally to a [min_size, max_size] range,
    // which the multi-scale list maps onto directly.
    baseLr: 0.005,
    optimizerType: 'Momentum',
    weightDecay: 0.0001,
    trainBatchSize: 2,
    evalBatchSize: 1,
    multiScaleTrain: true,
    multiScaleSizes: [640, 800, 1024, 1333],
    // Normalisation happens inside the model's GeneralizedRCNNTransform, so
    // these values are informational only (and hidden from the form below).
    normalizeType: 'none',
    epochs: 24,
    maxEpochs: 24,
    warmupEpochs: 1,
    warmupStartLr: 0.001,
    snapshotEpoch: 1,
  },
};

export function defaultTrainingParams(framework: ConfigFramework): TrainingParams {
  return { ...DEFAULT_TRAINING_PARAMS, ...FRAMEWORK_DEFAULT_OVERRIDES[framework] };
}

// ---------------------------------------------------------------------------
// Which knobs each framework actually honours
// ---------------------------------------------------------------------------

/**
 * Drives which controls the form renders. Showing a PaddleSeg user an
 * "Epochs" slider that silently does nothing is exactly the kind of thing that
 * made the previous advanced panel untrustworthy.
 */
export type TrainingFieldKey = keyof TrainingParams;

const DETECTION_FIELDS: TrainingFieldKey[] = [
  'epochs', 'snapshotEpoch',
  'optimizerType', 'baseLr', 'momentum', 'weightDecay', 'regularizerType', 'clipGradByNorm',
  'scheduler', 'maxEpochs', 'warmupEpochs', 'warmupStartLr', 'gamma', 'milestones',
  'trainBatchSize', 'evalBatchSize', 'workerNum', 'imageWidth', 'imageHeight',
  'multiScaleTrain', 'multiScaleSizes',
  'augRandomDistort', 'augRandomExpand', 'augRandomCrop', 'augRandomFlip',
  'normalizeType', 'normMean', 'normStd',
  'useGpu', 'useAmp', 'logIter', 'saveDir', 'outputDir', 'weights', 'pretrainWeights',
];

const CLAS_FIELDS: TrainingFieldKey[] = [
  'epochs', 'snapshotEpoch',
  'optimizerType', 'baseLr', 'momentum', 'weightDecay',
  'scheduler', 'warmupEpochs', 'warmupStartLr', 'gamma', 'milestones',
  'trainBatchSize', 'evalBatchSize', 'workerNum', 'imageWidth', 'imageHeight',
  'normalizeType', 'normMean', 'normStd',
  'useGpu', 'useAmp', 'logIter', 'outputDir', 'pretrainWeights',
];

const SEG_FIELDS: TrainingFieldKey[] = [
  'iters', 'saveInterval',
  'optimizerType', 'baseLr', 'momentum', 'weightDecay',
  'scheduler', 'warmupIters', 'warmupStartLr', 'power', 'endLr', 'gamma',
  'trainBatchSize', 'workerNum', 'imageWidth', 'imageHeight',
  'segOverrideTransforms', 'segAugFlipHorizontal', 'segAugFlipVertical',
  'segAugDistort', 'segAugScaleAspect', 'segAugBlur',
  'normalizeType', 'normMean', 'normStd',
  'useGpu', 'logIter', 'saveDir',
];

// TorchSeg honours everything PaddleSeg does, plus mixed precision (torchtrain's
// `tools/train.py --amp`), which PaddleSeg's CLI spells differently and the form
// therefore never offered.
const TORCH_SEG_FIELDS: TrainingFieldKey[] = [...SEG_FIELDS, 'useAmp'];

// TorchDet differs from PaddleDetection in two honest ways:
//   - `regularizer.type` is ignored (weight decay is always L2 in torch
//     optimizers), so offering an L1 choice would be a lie.
//   - normalisation is performed by torchvision's own transform, so
//     `normalizeType`/`normMean`/`normStd` cannot change anything.
// Both are therefore omitted rather than shown as no-op controls.
const TORCH_DET_FIELDS: TrainingFieldKey[] = DETECTION_FIELDS.filter(
  (field) => !['regularizerType', 'normalizeType', 'normMean', 'normStd', 'saveDir'].includes(field),
);

// TorchAnomaly deliberately offers no optimizer / scheduler / normalisation
// controls:
//   - In anomalib the learning rate is a *constructor argument of the model*
//     (`EfficientAd(lr=...)`), and the model config is merged last, so an lr set
//     in the training config would be silently overridden. It therefore lives in
//     the model config — the same reasoning that puts PaddleSeg's `loss:` there.
//   - The memory-bank models (PatchCore, PaDiM) do no gradient descent at all,
//     so an optimizer dropdown would be a no-op for half the algorithm list.
//   - Input size and normalisation are part of the model's `PreProcessor`, which
//     the model config owns.
const ANOMALY_FIELDS: TrainingFieldKey[] = [
  'iters', 'epochs', 'trainBatchSize', 'evalBatchSize', 'workerNum',
  'useGpu', 'useAmp', 'logIter', 'saveDir',
  'adTileEnabled', 'adTileSize', 'adTileStride', 'adValInterval', 'adBestMetric',
];

export const TRAINING_FIELD_SUPPORT: Record<ConfigFramework, Set<TrainingFieldKey>> = {
  PaddleDetection: new Set(DETECTION_FIELDS),
  PaddleClas: new Set(CLAS_FIELDS),
  PaddleSeg: new Set(SEG_FIELDS),
  TorchSeg: new Set(TORCH_SEG_FIELDS),
  TorchDet: new Set(TORCH_DET_FIELDS),
  TorchAnomaly: new Set(ANOMALY_FIELDS),
};

/** Metrics the anomaly best-checkpoint callback can monitor. */
export const ANOMALY_BEST_METRICS = ['image_AUROC', 'image_F1Score', 'pixel_AUROC', 'pixel_F1Score'];

export function supportsField(framework: ConfigFramework, field: TrainingFieldKey): boolean {
  return TRAINING_FIELD_SUPPORT[framework].has(field);
}

/** Selectable optimizer / scheduler values, per framework. */
export const OPTIMIZER_OPTIONS: Record<ConfigFramework, string[]> = {
  PaddleDetection: ['Momentum', 'SGD', 'Adam', 'AdamW', 'RMSProp'],
  PaddleClas: ['Momentum', 'SGD', 'Adam', 'AdamW'],
  PaddleSeg: ['SGD', 'Momentum', 'Adam', 'AdamW'],
  // Mirrors `build_optimizer` in `torchtrain/torchtrain/utils.py`.
  TorchSeg: ['SGD', 'Momentum', 'Adam', 'AdamW', 'RMSProp'],
  TorchDet: ['Momentum', 'SGD', 'Adam', 'AdamW', 'RMSProp'],
  // Each anomalib model owns its optimizer (`configure_optimizers`), and the
  // memory-bank models have none. Nothing to choose from.
  TorchAnomaly: [],
};

export const SCHEDULER_OPTIONS: Record<ConfigFramework, string[]> = {
  PaddleDetection: ['CosineDecay', 'PiecewiseDecay', 'ExpDecay', 'ConstLR'],
  PaddleClas: ['Cosine', 'Piecewise', 'Linear', 'MultiStepDecay', 'Constant'],
  PaddleSeg: ['PolynomialDecay', 'CosineAnnealingDecay', 'PiecewiseDecay', 'StepDecay', 'ExponentialDecay'],
  // Mirrors `LrScheduler.lr_at` in `torchtrain/torchtrain/utils.py`.
  TorchSeg: ['PolynomialDecay', 'CosineAnnealingDecay', 'PiecewiseDecay', 'StepDecay', 'ExponentialDecay'],
  TorchDet: ['CosineDecay', 'PiecewiseDecay', 'ExpDecay', 'ConstLR'],
  TorchAnomaly: [],
};

/** Schedulers whose YAML carries `milestones` / boundary values. */
const PIECEWISE_SCHEDULERS = new Set(['PiecewiseDecay', 'Piecewise', 'MultiStepDecay', 'StepDecay']);

// ---------------------------------------------------------------------------
// YAML generation
// ---------------------------------------------------------------------------

const num = (n: number) => (Number.isFinite(n) ? String(n) : '0');
const list = (xs: number[]) => `[${xs.join(', ')}]`;

function detectionNormalize(p: TrainingParams): string {
  return p.normalizeType === 'mean_std'
    ? `    - NormalizeImage: {mean: ${list(p.normMean)}, std: ${list(p.normStd)}, is_scale: true}`
    : `    - NormalizeImage: {mean: ${list(p.normMean)}, std: ${list(p.normStd)}, norm_type: none}`;
}

function generateDetectionYaml(
  p: TrainingParams,
  name: string,
  framework: ConfigFramework = 'PaddleDetection',
): string {
  const isTorch = framework === 'TorchDet';
  const sampleTransforms = ['    - Decode: {}'];
  if (p.augRandomDistort) sampleTransforms.push('    - RandomDistort: {}');
  if (p.augRandomExpand) sampleTransforms.push('    - RandomExpand: {fill_value: [123.675, 116.28, 103.53]}');
  if (p.augRandomCrop) sampleTransforms.push('    - RandomCrop: {}');
  if (p.augRandomFlip) sampleTransforms.push('    - RandomFlip: {}');

  const batchTransforms: string[] = [];
  if (p.multiScaleTrain && p.multiScaleSizes.length > 0) {
    batchTransforms.push(
      `    - BatchRandomResize: {target_size: ${list(p.multiScaleSizes)}, random_size: True, random_interp: True, keep_ratio: False}`,
    );
  } else {
    batchTransforms.push(
      `    - BatchRandomResize: {target_size: [[${num(p.imageHeight)}, ${num(p.imageWidth)}]], random_size: False, keep_ratio: False}`,
    );
  }
  batchTransforms.push(detectionNormalize(p));
  batchTransforms.push('    - Permute: {}');
  batchTransforms.push('    - PadGT: {}');

  // LearningRate.schedulers is an ordered list: the decay policy first, then an
  // optional LinearWarmup. Warmup is omitted entirely at 0 epochs rather than
  // emitted with `epochs: 0`, which PaddleDetection rejects.
  const schedulers: string[] = [];
  if (PIECEWISE_SCHEDULERS.has(p.scheduler)) {
    schedulers.push(`  - !PiecewiseDecay`);
    schedulers.push(`    gamma: ${num(p.gamma)}`);
    schedulers.push(`    milestones: ${list(p.milestones)}`);
  } else if (p.scheduler === 'ExpDecay') {
    schedulers.push(`  - !ExpDecay`);
    schedulers.push(`    gamma: ${num(p.gamma)}`);
  } else if (p.scheduler === 'ConstLR') {
    schedulers.push(`  - !ConstLR`);
  } else {
    schedulers.push(`  - !CosineDecay`);
    schedulers.push(`    max_epochs: ${num(p.maxEpochs)}`);
  }
  if (p.warmupEpochs > 0) {
    schedulers.push(`  - !LinearWarmup`);
    schedulers.push(`    start_factor: ${num(p.warmupStartLr)}`);
    schedulers.push(`    epochs: ${num(p.warmupEpochs)}`);
  }

  const optimizerBlock = [
    'OptimizerBuilder:',
    ...(p.clipGradByNorm !== null ? [`  clip_grad_by_norm: ${num(p.clipGradByNorm)}`] : []),
    '  optimizer:',
    `    type: ${p.optimizerType}`,
    ...(p.optimizerType === 'Momentum' || p.optimizerType === 'SGD'
      ? [`    momentum: ${num(p.momentum)}`]
      : []),
    '  regularizer:',
    // torch optimizers only implement L2 weight decay, so TorchDet always emits
    // L2 rather than echoing a choice it would ignore.
    `    type: ${isTorch ? 'L2' : p.regularizerType}`,
    `    factor: ${num(p.weightDecay)}`,
  ].join('\n');

  const evalSize = `[${num(p.imageHeight)}, ${num(p.imageWidth)}]`;

  // TorchDet consumes this same schema (see torchtrain/torchtrain/det/), with
  // two documented differences worth stating in the file the user will read.
  const header = isTorch
    ? `# ${name}
# Training configuration generated by AutoTrain (TorchDet / PyTorch)
#
# This is the PaddleDetection schema; torchtrain reads it directly. Two notes:
#   - NormalizeImage / Permute / PadGT describe work torchvision performs inside
#     the model (GeneralizedRCNNTransform), so they are accepted and ignored.
#   - Resize / BatchRandomResize target sizes become the model's
#     min_size / max_size, which is how torchvision expresses scale jitter.`
    : `# ${name}
# Training configuration generated by AutoTrain (PaddleDetection)`;

  return `${header}

epoch: ${num(p.epochs)}

LearningRate:
  base_lr: ${num(p.baseLr)}
  schedulers:
${schedulers.join('\n')}

${optimizerBlock}

# Reader settings
worker_num: ${num(p.workerNum)}

TrainReader:
  sample_transforms:
${sampleTransforms.join('\n')}
  batch_transforms:
${batchTransforms.join('\n')}
  batch_size: ${num(p.trainBatchSize)}
  shuffle: true
  drop_last: true
  use_shared_memory: true
  collate_batch: true

EvalReader:
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: ${evalSize}, keep_ratio: False, interp: 2}
${detectionNormalize(p)}
    - Permute: {}
  batch_size: ${num(p.evalBatchSize)}

TestReader:
  inputs_def:
    image_shape: [3, ${num(p.imageHeight)}, ${num(p.imageWidth)}]
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: ${evalSize}, keep_ratio: False, interp: 2}
${detectionNormalize(p)}
    - Permute: {}
  batch_size: 1

# Runtime settings
use_gpu: ${p.useGpu}
use_amp: ${p.useAmp}
log_iter: ${num(p.logIter)}
snapshot_epoch: ${num(p.snapshotEpoch)}
print_flops: false
print_params: false
${p.saveDir ? `save_dir: ${p.saveDir}\n` : ''}
# Export settings
export:
  post_process: True
  nms: True
  benchmark: False
  fuse_conv_bn: False
${p.outputDir ? `output_dir: ${p.outputDir}\n` : ''}${p.weights ? `weights: ${p.weights}\n` : ''}${p.pretrainWeights ? `pretrain_weights: ${p.pretrainWeights}\n` : ''}`;
}

function generateClasYaml(p: TrainingParams, name: string): string {
  const lrBlock = [
    '  lr:',
    `    name: ${p.scheduler}`,
    `    learning_rate: ${num(p.baseLr)}`,
    ...(PIECEWISE_SCHEDULERS.has(p.scheduler)
      ? [`    decay_epochs: ${list(p.milestones)}`, `    values: [${p.milestones.map((_, i) => p.baseLr * Math.pow(p.gamma, i + 1)).map(num).join(', ')}]`]
      : []),
    ...(p.warmupEpochs > 0
      ? [`    warmup_epoch: ${num(p.warmupEpochs)}`, `    warmup_start_lr: ${num(p.warmupStartLr)}`]
      : []),
  ].join('\n');

  const transformOps = (train: boolean) =>
    [
      train
        ? `        - RandCropImage:\n            size: ${num(p.imageWidth)}`
        : `        - ResizeImage:\n            resize_short: ${num(Math.round(p.imageWidth * 1.14))}\n        - CropImage:\n            size: ${num(p.imageWidth)}`,
      ...(train ? ['        - RandFlipImage:\n            flip_code: 1'] : []),
      `        - NormalizeImage:\n            scale: 1.0/255.0\n            mean: ${list(p.normMean)}\n            std: ${list(p.normStd)}\n            order: ''`,
    ].join('\n');

  return `# ${name}
# Training configuration generated by AutoTrain (PaddleClas)

Global:
  epochs: ${num(p.epochs)}
  use_gpu: ${p.useGpu}
  use_amp: ${p.useAmp}
  save_interval: ${num(p.snapshotEpoch)}
  print_batch_step: ${num(p.logIter)}
  eval_during_train: True
  eval_interval: 1
  output_dir: ${p.outputDir || './output'}
${p.pretrainWeights ? `  pretrained_model: ${p.pretrainWeights}\n` : ''}
Optimizer:
  name: ${p.optimizerType}
${p.optimizerType === 'Momentum' || p.optimizerType === 'SGD' ? `  momentum: ${num(p.momentum)}\n` : ''}${lrBlock}
  regularizer:
    name: 'L2'
    coeff: ${num(p.weightDecay)}

DataLoader:
  Train:
    dataset:
      transform_ops:
${transformOps(true)}
    sampler:
      name: DistributedBatchSampler
      batch_size: ${num(p.trainBatchSize)}
      drop_last: False
      shuffle: True
    loader:
      num_workers: ${num(p.workerNum)}
      use_shared_memory: True

  Eval:
    dataset:
      transform_ops:
${transformOps(false)}
    sampler:
      name: DistributedBatchSampler
      batch_size: ${num(p.evalBatchSize)}
      drop_last: False
      shuffle: False
    loader:
      num_workers: ${num(p.workerNum)}
      use_shared_memory: True
`;
}

function generateSegYaml(
  p: TrainingParams,
  name: string,
  framework: ConfigFramework = 'PaddleSeg',
): string {
  const isTorch = framework === 'TorchSeg';
  const lrLines: string[] = [`  type: ${p.scheduler}`, `  learning_rate: ${num(p.baseLr)}`];
  if (p.scheduler === 'PolynomialDecay') {
    lrLines.push(`  power: ${num(p.power)}`, `  end_lr: ${num(p.endLr)}`);
  } else if (p.scheduler === 'CosineAnnealingDecay') {
    lrLines.push(`  T_max: ${num(p.iters)}`, `  eta_min: ${num(p.endLr)}`);
  } else if (PIECEWISE_SCHEDULERS.has(p.scheduler)) {
    lrLines.push(`  gamma: ${num(p.gamma)}`);
  } else if (p.scheduler === 'ExponentialDecay') {
    lrLines.push(`  gamma: ${num(p.gamma)}`);
  }
  if (p.warmupIters > 0) {
    lrLines.push(`  warmup_iters: ${num(p.warmupIters)}`, `  warmup_start_lr: ${num(p.warmupStartLr)}`);
  }

  const optimizerLines = [`  type: ${p.optimizerType}`];
  if (p.optimizerType === 'SGD' || p.optimizerType === 'Momentum') {
    optimizerLines.push(`  momentum: ${num(p.momentum)}`);
  }
  optimizerLines.push(`  weight_decay: ${num(p.weightDecay)}`);

  // Optional transform override. Deep-merged into the dataset config, so we
  // only restate `transforms` — dataset_root / train_path / num_classes / mode
  // stay owned by the dataset config.
  let transformsBlock = '';
  if (p.segOverrideTransforms) {
    const train = [`    - type: Resize\n      target_size: [${num(p.imageWidth)}, ${num(p.imageHeight)}]`];
    if (p.segAugScaleAspect) {
      train.push('    - type: ResizeStepScaling\n      min_scale_factor: 0.75\n      max_scale_factor: 1.25\n      scale_step_size: 0.05');
    }
    if (p.segAugFlipHorizontal) train.push('    - type: RandomHorizontalFlip');
    if (p.segAugFlipVertical) train.push('    - type: RandomVerticalFlip');
    if (p.segAugDistort) train.push('    - type: RandomDistort\n      brightness_range: 0.4\n      contrast_range: 0.4\n      saturation_range: 0.4');
    if (p.segAugBlur) train.push('    - type: RandomBlur\n      prob: 0.1');
    train.push(`    - type: Normalize\n      mean: ${list(p.normMean)}\n      std: ${list(p.normStd)}`);

    transformsBlock = `
# Transform overrides. Deep-merged onto the dataset config, which keeps
# ownership of dataset_root / train_path / num_classes / mode.
train_dataset:
  transforms:
${train.join('\n')}

val_dataset:
  transforms:
    - type: Resize
      target_size: [${num(p.imageWidth)}, ${num(p.imageHeight)}]
    - type: Normalize
      mean: ${list(p.normMean)}
      std: ${list(p.normStd)}
`;
  }

  // `loss:` is intentionally absent — both PaddleSeg and TorchSeg enforce
  // len(loss.types) == len(model.logits), so the loss belongs to the model
  // config. See `@/lib/model-yaml`.
  const header = isTorch
    ? `# ${name}
# Training configuration generated by AutoTrain (TorchSeg / PyTorch)
#
# This is the PaddleSeg schema; torchtrain reads it directly. \`loss:\` lives in
# the model config because len(loss.types) must equal the number of logits the
# architecture emits. Do not add it here.`
    : `# ${name}
# Training configuration generated by AutoTrain (PaddleSeg)
# Note: \`loss:\` lives in the model config because it must match the model's
# logits count. Do not add it here.`;

  // PaddleSeg ignores these three YAML keys (it only accepts them as CLI flags);
  // torchtrain reads either, and the platform passes them on the CLI for both.
  const runtimeComment = isTorch
    ? '# Runtime (torchtrain reads these from the YAML or the CLI; the CLI wins)'
    : '# Runtime (PaddleSeg reads these from the CLI; kept here for reference)';

  return `${header}

batch_size: ${num(p.trainBatchSize)}
iters: ${num(p.iters)}

optimizer:
${optimizerLines.join('\n')}

lr_scheduler:
${lrLines.join('\n')}
${transformsBlock}
${runtimeComment}
use_gpu: ${p.useGpu}
${isTorch ? `use_amp: ${p.useAmp}\n` : ''}num_workers: ${num(p.workerNum)}
save_interval: ${num(p.saveInterval)}
log_iters: ${num(p.logIter)}
${p.saveDir ? `save_dir: ${p.saveDir}\n` : ''}`;
}

/**
 * anomalib training config: the `trainer:` block plus the two loader knobs that
 * live on the datamodule, plus the platform's own `autotrain:` block.
 *
 * What is deliberately *not* here:
 *
 * - `val_check_interval`. Lightning rejects an int larger than the number of
 *   training batches unless `check_val_every_n_epoch` is None, and setting it to
 *   None would disable epoch-end validation — which is the *only* validation a
 *   one-epoch memory-bank model ever gets. The intent is expressed as
 *   `autotrain.val_interval` and converted by the adapter, which knows both the
 *   batch count and the model.
 * - The learning rate / optimizer, which anomalib models take as constructor
 *   arguments; see the comment on `ANOMALY_FIELDS`.
 */
function generateAnomalyYaml(p: TrainingParams, name: string): string {
  const tiling = p.adTileEnabled
    ? `  callbacks:
    - class_path: anomalib.callbacks.tiler_configuration.TilerConfigurationCallback
      init_args:
        enable: true
        tile_size: [${num(p.adTileSize)}, ${num(p.adTileSize)}]
        stride: ${num(p.adTileStride)}
`
    : '';

  return `# ${name}
# Training configuration generated by AutoTrain (TorchAnomaly / anomalib)
#
# Unlike the other frameworks this is anomalib's own schema, not a Paddle one:
#   trainer:   Lightning Trainer arguments
#   data:      refinements to the Folder datamodule the dataset config declares
#   autotrain: read by torchtrain/torchtrain/ad/ and stripped before anomalib
#              ever sees the config
#
# The learning rate is NOT set here. In anomalib \`lr\` is a constructor argument
# of the model, and the model config is deep-merged last, so a value here would
# be silently overridden. It belongs in the model config — the same reason
# PaddleSeg's \`loss:\` lives there.

trainer:
  # max_steps is the real training length. max_epochs is only a ceiling: the
  # memory-bank models (PatchCore, PaDiM) declare max_epochs=1 themselves and
  # anomalib's argument cache overrides whatever we pass.
  max_steps: ${num(p.iters)}
  max_epochs: ${num(p.epochs)}
  accelerator: ${p.useGpu ? 'gpu' : 'cpu'}
  devices: 1
  precision: ${p.useAmp ? '16-mixed' : '32-true'}
  # The runner parses stdout to build TrainingLog rows; a rich progress bar
  # would flood it with carriage returns and no parsable lines.
  enable_progress_bar: false
  num_sanity_val_steps: 0
${tiling}
data:
  init_args:
    train_batch_size: ${num(p.trainBatchSize)}
    eval_batch_size: ${num(p.evalBatchSize)}
    num_workers: ${num(p.workerNum)}

autotrain:
  log_iter: ${num(p.logIter)}
  val_interval: ${num(p.adValInterval)}
  best_metric: ${p.adBestMetric || 'image_AUROC'}
${p.saveDir ? `  save_dir: ${p.saveDir}\n` : ''}`;
}

export function generateTrainingYaml(
  framework: ConfigFramework,
  params: TrainingParams,
  configName = 'Training Config',
): string {
  switch (CONFIG_SCHEMA[framework]) {
    case 'classification':
      return generateClasYaml(params, configName);
    case 'segmentation':
      return generateSegYaml(params, configName, framework);
    case 'anomaly':
      return generateAnomalyYaml(params, configName);
    default:
      return generateDetectionYaml(params, configName, framework);
  }
}

// ---------------------------------------------------------------------------
// YAML -> parameters
// ---------------------------------------------------------------------------

function toNum(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function toBool(value: unknown): boolean | undefined {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    if (/^(true|yes|on)$/i.test(value)) return true;
    if (/^(false|no|off)$/i.test(value)) return false;
  }
  return undefined;
}

function toNumList(value: unknown): number[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const nums = value.map(toNum).filter((n): n is number => n !== undefined);
  return nums.length === value.length ? nums : undefined;
}

/** Assign only when the parsed value is usable, so defaults survive gaps. */
function put<K extends keyof TrainingParams>(
  target: Partial<TrainingParams>,
  key: K,
  value: TrainingParams[K] | undefined,
): void {
  if (value !== undefined) target[key] = value;
}

/**
 * Parse a YAML document into training parameters.
 *
 * Tolerant by design: unknown structure yields an empty object rather than an
 * exception, and every field is optional so callers can layer the result over
 * `defaultTrainingParams(framework)`.
 *
 * The previous implementation only understood PaddleDetection keys, so every
 * imported PaddleSeg config silently persisted as "100 epochs / lr 0.001 /
 * batch 8" and the job progress bar was computed against the wrong total.
 */
export function parseTrainingParams(
  framework: ConfigFramework,
  yamlText: string | null | undefined,
): Partial<TrainingParams> {
  if (!yamlText || !yamlText.trim()) return {};

  let doc: Record<string, any>;
  let parsed: Document;
  try {
    parsed = parseDocument(yamlText, { logLevel: 'silent' });
    if (parsed.errors.length > 0 || !isMap(parsed.contents)) return {};
    doc = parsed.toJS({ maxAliasCount: -1 }) as Record<string, any>;
  } catch {
    return {};
  }
  if (!doc || typeof doc !== 'object') return {};

  const out: Partial<TrainingParams> = {};

  switch (CONFIG_SCHEMA[framework]) {
    case 'segmentation':
      parseSeg(doc, out);
      break;
    case 'classification':
      parseClas(doc, out);
      break;
    case 'anomaly':
      parseAnomaly(doc, out);
      break;
    default:
      parseDetection(doc, out, schedulerTags(parsed));
  }

  // Runtime keys shared by Detection and Seg top-level configs.
  put(out, 'useGpu', toBool(doc.use_gpu));
  put(out, 'useAmp', toBool(doc.use_amp));
  if (typeof doc.save_dir === 'string') put(out, 'saveDir', doc.save_dir);
  if (typeof doc.output_dir === 'string') put(out, 'outputDir', doc.output_dir);
  if (typeof doc.weights === 'string') put(out, 'weights', doc.weights);
  if (typeof doc.pretrain_weights === 'string') put(out, 'pretrainWeights', doc.pretrain_weights);

  return out;
}

/**
 * Read the YAML tag of each entry in `LearningRate.schedulers`.
 *
 * PaddleDetection identifies schedulers by tag (`- !CosineDecay`), and
 * `Document.toJS()` discards tags — so the plain-JS view of a scheduler list is
 * just its keys. Pulling the tags off the AST is the only way to recover which
 * policy was configured. Returns '' for entries that use the `name:` style
 * instead (handled by the caller).
 */
function schedulerTags(parsed: Document): string[] {
  const node = parsed.getIn(['LearningRate', 'schedulers'], true);
  if (!isSeq(node)) return [];
  return node.items.map((item) =>
    isMap(item) && typeof item.tag === 'string' ? item.tag.replace(/^!/, '') : '',
  );
}

function parseDetection(
  doc: Record<string, any>,
  out: Partial<TrainingParams>,
  tags: string[] = [],
): void {
  put(out, 'epochs', toNum(doc.epoch));
  put(out, 'snapshotEpoch', toNum(doc.snapshot_epoch));
  put(out, 'logIter', toNum(doc.log_iter));
  put(out, 'workerNum', toNum(doc.worker_num));

  put(out, 'baseLr', toNum(doc.LearningRate?.base_lr));
  const schedulers = Array.isArray(doc.LearningRate?.schedulers) ? doc.LearningRate.schedulers : [];
  schedulers.forEach((s: any, i: number) => {
    if (!s || typeof s !== 'object') return;
    // Both `- !CosineDecay` (tag) and `- name: CosineDecay` are in the wild.
    const name = typeof s.name === 'string' ? s.name : tags[i] || undefined;
    const isWarmup = name === 'LinearWarmup' || s.start_factor !== undefined;
    if (isWarmup) {
      put(out, 'warmupEpochs', toNum(s.epochs));
      put(out, 'warmupStartLr', toNum(s.start_factor));
      return;
    }
    if (name) put(out, 'scheduler', name);
    put(out, 'maxEpochs', toNum(s.max_epochs));
    put(out, 'gamma', toNum(s.gamma));
    put(out, 'milestones', toNumList(s.milestones));
    if (!name && s.milestones !== undefined) put(out, 'scheduler', 'PiecewiseDecay');
  });

  const optimizer = doc.OptimizerBuilder?.optimizer;
  if (optimizer && typeof optimizer === 'object') {
    if (typeof optimizer.type === 'string') put(out, 'optimizerType', optimizer.type);
    put(out, 'momentum', toNum(optimizer.momentum));
  }
  const regularizer = doc.OptimizerBuilder?.regularizer;
  if (regularizer && typeof regularizer === 'object') {
    if (typeof regularizer.type === 'string') put(out, 'regularizerType', regularizer.type);
    put(out, 'weightDecay', toNum(regularizer.factor));
  }
  const clip = toNum(doc.OptimizerBuilder?.clip_grad_by_norm);
  out.clipGradByNorm = clip === undefined ? null : clip;

  put(out, 'trainBatchSize', toNum(doc.TrainReader?.batch_size));
  put(out, 'evalBatchSize', toNum(doc.EvalReader?.batch_size));

  // Image size: prefer the explicit eval resize, fall back to the legacy
  // `eval_height` / `eval_width` anchors older configs still use.
  const evalResize = (Array.isArray(doc.EvalReader?.sample_transforms) ? doc.EvalReader.sample_transforms : [])
    .map((t: any) => t?.Resize?.target_size)
    .find((t: any) => Array.isArray(t) && t.length === 2);
  const target = toNumList(evalResize);
  if (target) {
    put(out, 'imageHeight', target[0]);
    put(out, 'imageWidth', target[1]);
  } else {
    put(out, 'imageHeight', toNum(doc.eval_height));
    put(out, 'imageWidth', toNum(doc.eval_width));
  }

  const sampleTransforms = Array.isArray(doc.TrainReader?.sample_transforms)
    ? doc.TrainReader.sample_transforms
    : undefined;
  if (sampleTransforms) {
    const has = (op: string) => sampleTransforms.some((t: any) => t && typeof t === 'object' && op in t);
    out.augRandomDistort = has('RandomDistort');
    out.augRandomExpand = has('RandomExpand');
    out.augRandomCrop = has('RandomCrop');
    out.augRandomFlip = has('RandomFlip');
  }

  const batchTransforms = Array.isArray(doc.TrainReader?.batch_transforms)
    ? doc.TrainReader.batch_transforms
    : [];
  const batchResize = batchTransforms.find((t: any) => t && typeof t === 'object' && 'BatchRandomResize' in t);
  if (batchResize) {
    const sizes = batchResize.BatchRandomResize?.target_size;
    const flat = toNumList(sizes);
    out.multiScaleTrain = batchResize.BatchRandomResize?.random_size !== false && !!flat && flat.length > 1;
    if (flat && flat.length > 1) put(out, 'multiScaleSizes', flat);
  }
  const normalize = batchTransforms.find((t: any) => t && typeof t === 'object' && 'NormalizeImage' in t)
    ?.NormalizeImage;
  if (normalize) {
    out.normalizeType = normalize.norm_type === 'none' ? 'none' : 'mean_std';
    put(out, 'normMean', toNumList(normalize.mean));
    put(out, 'normStd', toNumList(normalize.std));
  }
}

function parseClas(doc: Record<string, any>, out: Partial<TrainingParams>): void {
  const g = doc.Global ?? {};
  put(out, 'epochs', toNum(g.epochs));
  put(out, 'snapshotEpoch', toNum(g.save_interval));
  put(out, 'logIter', toNum(g.print_batch_step));
  put(out, 'useGpu', toBool(g.use_gpu));
  put(out, 'useAmp', toBool(g.use_amp));
  if (typeof g.output_dir === 'string') put(out, 'outputDir', g.output_dir);
  if (typeof g.pretrained_model === 'string') put(out, 'pretrainWeights', g.pretrained_model);

  const opt = doc.Optimizer ?? {};
  if (typeof opt.name === 'string') put(out, 'optimizerType', opt.name);
  put(out, 'momentum', toNum(opt.momentum));
  put(out, 'weightDecay', toNum(opt.regularizer?.coeff));
  if (typeof opt.lr?.name === 'string') put(out, 'scheduler', opt.lr.name);
  put(out, 'baseLr', toNum(opt.lr?.learning_rate));
  put(out, 'warmupEpochs', toNum(opt.lr?.warmup_epoch));
  put(out, 'warmupStartLr', toNum(opt.lr?.warmup_start_lr));
  put(out, 'milestones', toNumList(opt.lr?.decay_epochs));

  const train = doc.DataLoader?.Train ?? {};
  const evalLoader = doc.DataLoader?.Eval ?? {};
  put(out, 'trainBatchSize', toNum(train.sampler?.batch_size));
  put(out, 'evalBatchSize', toNum(evalLoader.sampler?.batch_size));
  put(out, 'workerNum', toNum(train.loader?.num_workers));

  const ops = Array.isArray(train.dataset?.transform_ops) ? train.dataset.transform_ops : [];
  const crop = ops.find((t: any) => t && ('RandCropImage' in t || 'ResizeImage' in t));
  const size = toNum(crop?.RandCropImage?.size ?? crop?.ResizeImage?.size);
  if (size !== undefined) {
    put(out, 'imageWidth', size);
    put(out, 'imageHeight', size);
  }
  const norm = ops.find((t: any) => t && 'NormalizeImage' in t)?.NormalizeImage;
  if (norm) {
    out.normalizeType = 'mean_std';
    put(out, 'normMean', toNumList(norm.mean));
    put(out, 'normStd', toNumList(norm.std));
  }
}

function parseSeg(doc: Record<string, any>, out: Partial<TrainingParams>): void {
  put(out, 'iters', toNum(doc.iters));
  put(out, 'trainBatchSize', toNum(doc.batch_size));
  put(out, 'saveInterval', toNum(doc.save_interval));
  put(out, 'logIter', toNum(doc.log_iters));
  put(out, 'workerNum', toNum(doc.num_workers));

  const opt = doc.optimizer ?? {};
  if (typeof opt.type === 'string') put(out, 'optimizerType', opt.type);
  put(out, 'momentum', toNum(opt.momentum));
  put(out, 'weightDecay', toNum(opt.weight_decay));

  const lr = doc.lr_scheduler ?? {};
  if (typeof lr.type === 'string') put(out, 'scheduler', lr.type);
  put(out, 'baseLr', toNum(lr.learning_rate));
  put(out, 'power', toNum(lr.power));
  put(out, 'endLr', toNum(lr.end_lr ?? lr.eta_min));
  put(out, 'gamma', toNum(lr.gamma));
  put(out, 'warmupIters', toNum(lr.warmup_iters));
  put(out, 'warmupStartLr', toNum(lr.warmup_start_lr));

  const transforms = Array.isArray(doc.train_dataset?.transforms) ? doc.train_dataset.transforms : undefined;
  if (transforms) {
    out.segOverrideTransforms = true;
    const has = (type: string) => transforms.some((t: any) => t?.type === type);
    out.segAugFlipHorizontal = has('RandomHorizontalFlip');
    out.segAugFlipVertical = has('RandomVerticalFlip');
    out.segAugDistort = has('RandomDistort');
    out.segAugScaleAspect = has('ResizeStepScaling');
    out.segAugBlur = has('RandomBlur');

    // Training resolution. A plain `Resize` states it directly, but the common
    // PaddleSeg recipe is scale-jitter + `RandomPaddingCrop`, where the crop
    // size *is* the network input size. Reading only `Resize` reported the
    // default 512 for configs that actually train at 1024.
    const resize = transforms.find((t: any) => t?.type === 'Resize');
    const crop = transforms.find(
      (t: any) => t?.type === 'RandomPaddingCrop' || t?.type === 'RandomCrop',
    );
    const size = toNumList(resize?.target_size) ?? toNumList(crop?.crop_size);
    if (size && size.length === 2) {
      put(out, 'imageWidth', size[0]);
      put(out, 'imageHeight', size[1]);
    }
    const norm = transforms.find((t: any) => t?.type === 'Normalize');
    if (norm) {
      // PaddleSeg's Normalize always applies mean/std; there is no `none` mode.
      out.normalizeType = 'mean_std';
      put(out, 'normMean', toNumList(norm.mean));
      put(out, 'normStd', toNumList(norm.std));
    }
  }
}

/**
 * anomalib config -> parameters.
 *
 * Reads both the trainer block and the tiling callback, so a hand-written or
 * imported anomalib config shows the right values in the edit dialog instead of
 * silently reverting to defaults on the next save.
 */
function parseAnomaly(doc: Record<string, any>, out: Partial<TrainingParams>): void {
  const trainer = doc.trainer ?? {};
  // `-1` is Lightning's "unlimited" for max_steps; treat it as "not stated" so
  // the default survives rather than persisting a nonsensical length.
  const maxSteps = toNum(trainer.max_steps);
  if (maxSteps !== undefined && maxSteps > 0) put(out, 'iters', maxSteps);
  const maxEpochs = toNum(trainer.max_epochs);
  if (maxEpochs !== undefined && maxEpochs > 0) put(out, 'epochs', maxEpochs);
  if (typeof trainer.accelerator === 'string') out.useGpu = trainer.accelerator !== 'cpu';
  if (typeof trainer.precision === 'string') out.useAmp = /^(16|bf16)/.test(trainer.precision);

  const dataArgs = doc.data?.init_args ?? {};
  put(out, 'trainBatchSize', toNum(dataArgs.train_batch_size));
  put(out, 'evalBatchSize', toNum(dataArgs.eval_batch_size));
  put(out, 'workerNum', toNum(dataArgs.num_workers));

  const platform = doc.autotrain ?? {};
  put(out, 'logIter', toNum(platform.log_iter));
  put(out, 'adValInterval', toNum(platform.val_interval));
  if (typeof platform.best_metric === 'string') put(out, 'adBestMetric', platform.best_metric);
  if (typeof platform.save_dir === 'string') put(out, 'saveDir', platform.save_dir);

  // `trainer.callbacks` may be a single mapping or a list of them.
  const callbacks = Array.isArray(trainer.callbacks)
    ? trainer.callbacks
    : trainer.callbacks
      ? [trainer.callbacks]
      : [];
  const tiler = callbacks.find(
    (c: any) => typeof c?.class_path === 'string' && c.class_path.includes('TilerConfiguration'),
  );
  if (tiler) {
    const args = tiler.init_args ?? {};
    out.adTileEnabled = toBool(args.enable) ?? true;
    const size = toNumList(args.tile_size) ?? (toNum(args.tile_size) !== undefined ? [toNum(args.tile_size)!] : undefined);
    if (size && size.length > 0) put(out, 'adTileSize', size[0]);
    const stride = toNumList(args.stride) ?? (toNum(args.stride) !== undefined ? [toNum(args.stride)!] : undefined);
    if (stride && stride.length > 0) put(out, 'adTileStride', stride[0]);
  }
}

// ---------------------------------------------------------------------------
// Persistence helpers
// ---------------------------------------------------------------------------

/**
 * Project parameters onto the flat `TrainingConfig` columns.
 *
 * These columns exist only so the list view can show "N epochs / lr X / batch Y"
 * without parsing YAML on every render. `yamlConfig` remains authoritative.
 *
 * Unit convention: the epoch-named columns hold values in **the framework's
 * native training unit**. PaddleSeg counts iterations, so for a Seg config
 * `epoch`/`maxEpochs` carry `iters`, `warmupEpochs` carries `warmup_iters`, and
 * `snapshotEpoch` carries `save_interval`. The alternative — leaving them at
 * their epoch-shaped defaults — put numbers in the database that corresponded to
 * nothing at all (a 160k-iteration Seg config used to read "100 epochs").
 * Consumers that need the unit should branch on `project.framework`; the
 * dedicated `iters` / `saveInterval` columns are Seg-only and null elsewhere.
 */
export function trainingParamsToColumns(framework: ConfigFramework, params: TrainingParams) {
  const isSeg = countsIterations(framework);
  const nativeLength = Math.max(1, Math.round(isSeg ? params.iters : params.epochs));
  // The "checkpoint/eval cadence" column. Anomaly runs have no `save_interval`:
  // checkpointing is driven by the monitored validation metric, so the closest
  // honest value is how often validation happens.
  const cadence = Math.max(
    1,
    Math.round(
      CONFIG_SCHEMA[framework] === 'anomaly'
        ? params.adValInterval
        : isSeg
          ? params.saveInterval
          : params.snapshotEpoch,
    ),
  );
  return {
    epoch: nativeLength,
    batchSize: Math.max(1, Math.round(params.trainBatchSize)),
    baseLr: params.baseLr,
    momentum: params.momentum,
    weightDecay: params.weightDecay,
    scheduler: params.scheduler,
    warmupEpochs: Math.max(0, Math.round(isSeg ? params.warmupIters : params.warmupEpochs)),
    maxEpochs: isSeg ? nativeLength : Math.max(1, Math.round(params.maxEpochs)),
    iters: isSeg ? Math.max(1, Math.round(params.iters)) : null,
    saveInterval: isSeg ? cadence : null,
    workerNum: Math.max(0, Math.round(params.workerNum)),
    evalHeight: Math.max(1, Math.round(params.imageHeight)),
    evalWidth: Math.max(1, Math.round(params.imageWidth)),
    useGpu: params.useGpu,
    logIter: Math.max(1, Math.round(params.logIter)),
    snapshotEpoch: cadence,
    saveDir: params.saveDir || null,
    outputDir: params.outputDir || null,
    weights: params.weights || null,
    pretrainWeights: params.pretrainWeights || null,
  };
}

/**
 * Total training length used for the job progress bar.
 * PaddleSeg reports iterations, everything else reports epochs — conflating the
 * two is why Seg jobs used to show progress against a hardcoded 100.
 */
export function totalStepsFor(framework: ConfigFramework, params: TrainingParams): number {
  return countsIterations(framework)
    ? Math.max(1, Math.round(params.iters))
    : Math.max(1, Math.round(params.epochs));
}
