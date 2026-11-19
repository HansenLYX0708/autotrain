import { stringify } from 'yaml'
import { frameworkMeta } from './frameworks'
import type { TrainingParams } from './training-yaml'
import type { ModelParams } from './model-yaml'

export function localizationArchitectures(framework: string): string[] {
  const task = frameworkMeta(framework).localizationTask
  return ['n', 's', 'm'].map(size => `yolo11${size}-${task}`)
}

export function localizationTrainingYaml(framework: string, p: TrainingParams): string {
  return stringify({
    use_gpu: p.useGpu,
    localization: { train: {
      seed: 42, deterministic: true,
      translate: 0.05, scale: 0.2,
      fliplr: 0, flipud: 0, mosaic: 0, mixup: 0, perspective: 0,
      ...p.localizationTrainExtras,
      degrees: p.localizationDegrees,
      ...(frameworkMeta(framework).localizationTask === 'obb' ? {
        fliplr: p.localizationFlipLR, flipud: p.localizationFlipUD,
        inscribed_rotate_degrees: p.localizationInscribedDegrees,
        inscribed_rotate_prob: p.localizationInscribedProb,
      } : {}),
      // Inscribed rotation replaces Ultralytics geometric augmentation: those
      // transforms crop at the canvas edge and refit cut boxes with minAreaRect,
      // corrupting the exact angles inscribed rotation preserves.
      ...(p.localizationInscribedDegrees > 0 ? {
        degrees: 0, translate: 0, scale: 0, shear: 0, perspective: 0, mosaic: 0,
      } : {}),
      epochs: p.epochs, batch: p.trainBatchSize, imgsz: p.imageWidth,
      nbs: p.localizationNominalBatchSize,
      ...(frameworkMeta(framework).localizationTask === 'obb' ? {
        angle_weight: p.obbAngleWeight, selection_metric: p.obbSelectionMetric,
        selection_min_recall: p.obbSelectionMinRecall, selection_max_false_positives: p.obbSelectionMaxFalsePositives,
      } : {}),
      workers: p.workerNum, optimizer: p.optimizerType === 'Momentum' ? 'SGD' : p.optimizerType,
      lr0: p.baseLr, momentum: p.momentum, weight_decay: p.weightDecay,
      warmup_epochs: p.warmupEpochs, cos_lr: p.scheduler === 'CosineDecay',
      save_period: p.snapshotEpoch, amp: p.useAmp,
      patience: frameworkMeta(framework).localizationTask === 'obb' && p.obbSelectionMetric === 'angle' ? 0 : p.localizationTrainExtras?.patience ?? 30,
      log_interval: p.logIter,
    } },
  })
}

export function parseLocalizationTraining(doc: Record<string, any>, framework: string): Partial<TrainingParams> {
  const train = doc.localization?.train ?? {}
  const out: Partial<TrainingParams> = {}
  const fields: [keyof TrainingParams, string][] = [
    ['epochs', 'epochs'], ['trainBatchSize', 'batch'], ['evalBatchSize', 'batch'],
    ['localizationNominalBatchSize', 'nbs'], ['localizationDegrees', 'degrees'],
    ...(frameworkMeta(framework).localizationTask === 'obb'
      ? [['localizationFlipLR', 'fliplr'], ['localizationFlipUD', 'flipud'],
         ['localizationInscribedDegrees', 'inscribed_rotate_degrees'], ['localizationInscribedProb', 'inscribed_rotate_prob'],
        ] as [keyof TrainingParams, string][]
      : []),
    ['obbAngleWeight', 'angle_weight'],
    ['obbSelectionMinRecall', 'selection_min_recall'], ['obbSelectionMaxFalsePositives', 'selection_max_false_positives'],
    ['imageWidth', 'imgsz'], ['imageHeight', 'imgsz'], ['workerNum', 'workers'],
    ['baseLr', 'lr0'], ['momentum', 'momentum'], ['weightDecay', 'weight_decay'],
    ['warmupEpochs', 'warmup_epochs'], ['snapshotEpoch', 'save_period'], ['logIter', 'log_interval'],
  ]
  for (const [key, source] of fields) {
    if (typeof train[source] === 'number' && Number.isFinite(train[source])) Object.assign(out, { [key]: train[source] })
  }
  const controlled = new Set([...fields.map(([, source]) => source), 'optimizer', 'selection_metric', 'cos_lr', 'amp'])
  out.localizationTrainExtras = Object.fromEntries(Object.entries(train).filter(([key]) => !controlled.has(key)))
  if (typeof train.optimizer === 'string') out.optimizerType = train.optimizer
  if (train.selection_metric === 'map' || train.selection_metric === 'angle') out.obbSelectionMetric = train.selection_metric
  if (typeof train.cos_lr === 'boolean') out.scheduler = train.cos_lr ? 'CosineDecay' : 'Linear'
  if (typeof train.amp === 'boolean') out.useAmp = train.amp
  if (typeof doc.use_gpu === 'boolean') out.useGpu = doc.use_gpu
  return out
}

export function localizationModelYaml(framework: string, p: ModelParams): string {
  return stringify({
    num_classes: p.numClasses,
    localization: {
      task: frameworkMeta(framework).localizationTask,
      model: { architecture: p.architecture, weights: p.pretrainWeights || null },
    },
  })
}

export function parseLocalizationModel(doc: Record<string, any>): Partial<ModelParams> {
  const model = doc.localization?.model ?? {}
  return {
    ...(typeof model.architecture === 'string' ? { architecture: model.architecture } : {}),
    ...(typeof model.weights === 'string' ? { pretrainWeights: model.weights } : {}),
    ...(Number.isInteger(doc.num_classes) && doc.num_classes > 0 ? { numClasses: doc.num_classes } : {}),
    backbone: 'Ultralytics', neck: '', head: '',
  }
}
