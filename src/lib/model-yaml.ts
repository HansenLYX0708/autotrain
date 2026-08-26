/**
 * Framework-aware model-config model: parameters -> YAML and YAML -> parameters.
 *
 * Counterpart to `@/lib/training-yaml`, shared by the browser (model create/edit
 * dialog) and the server (model import/update routes).
 *
 * Two correctness rules this module exists to enforce:
 *
 *  1. **PaddleDetection component wiring is architecture-specific.** The key
 *     that carries the detection head differs per meta-architecture
 *     (`yolo_head` for YOLOv3/PP-YOLOE, `head` for RetinaNet/CenterNet,
 *     `bbox_head` for FasterRCNN, `detr_head` for DETR). Emitting `yolo_head`
 *     for everything — as the previous generator did — produces a config that
 *     PaddleDetection cannot build. Likewise, backbone hyper-parameters are
 *     backbone-specific: writing CSPResNet's `layers`/`channels`/`use_alpha`
 *     under a `MobileNetV3:` block is a hard failure.
 *
 *  2. **PaddleSeg requires len(loss.types) == number of logits the model emits.**
 *     Each architecture emits a different count (main head + N auxiliary heads).
 *     A mismatch raises `RuntimeError: The length of logits_list should equal to
 *     the types of loss config`. The loss block therefore belongs to the *model*
 *     config, and `SEG_ARCHITECTURES[].logits` is what keeps it in sync.
 */

import { parseDocument, isMap } from 'yaml';
import type { ConfigFramework } from './training-yaml';

export type { ConfigFramework };

// ---------------------------------------------------------------------------
// Parameter model
// ---------------------------------------------------------------------------

export interface ModelParams {
  architecture: string;
  backbone: string;
  neck: string;
  head: string;
  numClasses: number;

  // PaddleDetection ---------------------------------------------------------
  normType: string;
  useEma: boolean;
  emaDecay: number;
  depthMult: number;
  widthMult: number;

  // PaddleSeg ---------------------------------------------------------------
  /** One entry per logit the architecture emits. */
  segLossTypes: string[];
  /** Same length as `segLossTypes`. */
  segLossCoef: number[];
  segAlignCorners: boolean;

  // Anomaly detection (TorchAnomaly / anomalib) -----------------------------
  /**
   * Network input size. Lives in the model config rather than the training
   * config because in anomalib the resize is part of the model's
   * `PreProcessor`, and the recommended size differs per algorithm.
   */
  adImageWidth: number;
  adImageHeight: number;
  /** Centre crop applied after the resize; 0 disables it (PatchCore: 256->224). */
  adCenterCrop: number;
  /** Feature layers to extract from the backbone. */
  adLayers: string[];
  /** Only meaningful for models that accept `lr` / `weight_decay` (EfficientAd). */
  adLr: number;
  adWeightDecay: number;
  /** PatchCore: coreset subsampling ratio and kNN count. */
  adCoresetRatio: number;
  adNumNeighbors: number;
  /** EfficientAd: `small` or `medium`. */
  adModelSize: string;

  /** Backbone/whole-model pretrained weights. Empty string omits the key. */
  pretrainWeights: string;
}

// ---------------------------------------------------------------------------
// PaddleDetection architecture presets
// ---------------------------------------------------------------------------

export interface DetectionPreset {
  label: string;
  /** Value of the top-level `architecture:` key. */
  architecture: string;
  backbones: string[];
  necks: string[];
  heads: string[];
  /** The key under the architecture block that carries the head. */
  headKey: 'yolo_head' | 'head' | 'bbox_head' | 'detr_head';
  /** Extra keys inside the architecture block, e.g. `post_process: ~`. */
  archExtras?: string[];
  /** Whether depth_mult / width_mult are meaningful for this family. */
  usesMultipliers: boolean;
}

export const DETECTION_PRESETS: Record<string, DetectionPreset> = {
  'PP-YOLOE': {
    label: 'PP-YOLOE (YOLOv3 meta-arch)',
    architecture: 'YOLOv3',
    backbones: ['CSPResNet'],
    necks: ['CustomCSPPAN'],
    heads: ['PPYOLOEHead'],
    headKey: 'yolo_head',
    archExtras: ['post_process: ~'],
    usesMultipliers: true,
  },
  'YOLOv3': {
    label: 'YOLOv3 (classic)',
    architecture: 'YOLOv3',
    backbones: ['DarkNet', 'MobileNetV1', 'MobileNetV3', 'ResNet'],
    necks: ['YOLOv3FPN', 'PPYOLOFPN'],
    heads: ['YOLOv3Head'],
    headKey: 'yolo_head',
    usesMultipliers: false,
  },
  'PicoDet': {
    label: 'PicoDet',
    architecture: 'PicoDet',
    backbones: ['LCNet', 'ESNet'],
    necks: ['LCPAN', 'CSPPAN'],
    heads: ['PicoHeadV2'],
    headKey: 'head',
    usesMultipliers: false,
  },
  'RT-DETR': {
    label: 'RT-DETR (DETR meta-arch)',
    architecture: 'DETR',
    backbones: ['ResNet', 'HGNetv2'],
    necks: ['HybridEncoder'],
    heads: ['DINOHead'],
    headKey: 'detr_head',
    archExtras: ['transformer: RTDETRTransformer', 'post_process: DETRPostProcess'],
    usesMultipliers: false,
  },
  'FasterRCNN': {
    label: 'Faster R-CNN',
    architecture: 'FasterRCNN',
    backbones: ['ResNet', 'ResNeXt'],
    necks: ['FPN'],
    heads: ['BBoxHead'],
    headKey: 'bbox_head',
    archExtras: ['rpn_head: RPNHead', 'bbox_post_process: BBoxPostProcess'],
    usesMultipliers: false,
  },
  'RetinaNet': {
    label: 'RetinaNet',
    architecture: 'RetinaNet',
    backbones: ['ResNet'],
    necks: ['FPN'],
    heads: ['RetinaHead'],
    headKey: 'head',
    usesMultipliers: false,
  },
  'CenterNet': {
    label: 'CenterNet',
    architecture: 'CenterNet',
    backbones: ['DLA', 'ResNet'],
    necks: ['CenterNetDLAFPN'],
    heads: ['CenterNetHead'],
    headKey: 'head',
    usesMultipliers: false,
  },
};

export const DETECTION_PRESET_KEYS = Object.keys(DETECTION_PRESETS);

/**
 * Find the preset whose wiring matches a stored model row. Falls back to
 * PP-YOLOE, which is what the app has always defaulted to.
 */
export function detectionPresetFor(architecture: string, head: string): string {
  const exact = DETECTION_PRESET_KEYS.find(
    (k) => DETECTION_PRESETS[k].architecture === architecture && DETECTION_PRESETS[k].heads.includes(head),
  );
  if (exact) return exact;
  const byArch = DETECTION_PRESET_KEYS.find((k) => DETECTION_PRESETS[k].architecture === architecture);
  return byArch ?? 'PP-YOLOE';
}

/**
 * Backbone hyper-parameter blocks. Only emitted when the chosen backbone has a
 * known block — an unknown backbone gets no block at all, which lets
 * PaddleDetection fall back to its own registered defaults instead of being fed
 * parameters that belong to a different network.
 */
const BACKBONE_BLOCKS: Record<string, (p: ModelParams) => string[]> = {
  CSPResNet: () => [
    'layers: [3, 6, 6, 3]',
    'channels: [64, 128, 256, 512, 1024]',
    'return_idx: [1, 2, 3]',
    'use_large_stem: True',
    'use_alpha: True',
  ],
  DarkNet: () => ['depth: 53', 'return_idx: [2, 3, 4]', 'freeze_at: -1', 'freeze_norm: false'],
  ResNet: () => ['depth: 50', 'variant: d', 'norm_type: bn', 'freeze_at: 0', 'return_idx: [1, 2, 3]', 'num_stages: 4'],
  MobileNetV3: () => ['scale: 1.0', 'model_name: large', 'with_extra_blocks: false', 'extra_block_filters: []', 'feature_maps: [7, 13, 16]'],
  LCNet: () => ['scale: 1.5', 'feature_maps: [3, 4, 5]'],
  ESNet: () => ['scale: 1.0', 'feature_maps: [4, 11, 14]', 'act: hard_swish'],
  DLA: () => ['depth: 34'],
};

const NECK_BLOCKS: Record<string, (p: ModelParams) => string[]> = {
  CustomCSPPAN: () => [
    'out_channels: [768, 384, 192]',
    'stage_num: 1',
    'block_num: 3',
    "act: 'swish'",
    'spp: true',
  ],
  YOLOv3FPN: () => ['norm_type: bn'],
  FPN: () => ['out_channel: 256'],
  HybridEncoder: () => ['hidden_dim: 256', 'use_encoder_idx: [2]', 'num_encoder_layers: 1', "expansion: 1.0"],
  LCPAN: () => ['out_channels: 128'],
  CSPPAN: () => ['out_channels: 128', 'use_depthwise: True', 'num_csp_blocks: 1'],
  CenterNetDLAFPN: () => ['first_level: 2', 'last_level: 5', 'down_ratio: 4'],
};

const HEAD_BLOCKS: Record<string, (p: ModelParams) => string[]> = {
  PPYOLOEHead: (p) => [
    `num_classes: ${p.numClasses}`,
    'fpn_strides: [32, 16, 8]',
    'grid_cell_scale: 5.0',
    'grid_cell_offset: 0.5',
    'static_assigner_epoch: 30',
    'use_varifocal_loss: True',
    'loss_weight: {class: 1.0, iou: 2.5, dfl: 0.5}',
  ],
  YOLOv3Head: (p) => [
    `num_classes: ${p.numClasses}`,
    'anchors: [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]',
    'anchor_masks: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]',
    'loss: YOLOv3Loss',
  ],
  PicoHeadV2: (p) => [
    `num_classes: ${p.numClasses}`,
    'fpn_stride: [8, 16, 32, 64]',
    'feat_in_chan: 128',
    'grid_cell_scale: 5.0',
  ],
  DINOHead: (p) => [`num_classes: ${p.numClasses}`, 'loss: DINOLoss'],
  BBoxHead: (p) => [`num_classes: ${p.numClasses}`, 'head: TwoFCHead', 'roi_extractor: RoIAlign', 'bbox_assigner: BBoxAssigner'],
  RetinaHead: (p) => [`num_classes: ${p.numClasses}`, 'conv_feat: RetinaFeat', 'anchor_generator: RetinaAnchorGenerator'],
  CenterNetHead: (p) => [`num_classes: ${p.numClasses}`, 'head_planes: 256', 'regress_ltrb: True'],
};

// ---------------------------------------------------------------------------
// PaddleSeg architectures
// ---------------------------------------------------------------------------

export interface SegArchitecture {
  value: string;
  label: string;
  /**
   * Number of logits the architecture emits during training (main head plus
   * auxiliary heads). `loss.types` must have exactly this many entries.
   */
  logits: number;
  /** Default per-logit loss weights; length always equals `logits`. */
  defaultCoef: number[];
  /** Backbone is a separate config block for these architectures only. */
  needsBackbone: boolean;
  backbones: string[];
}

export const SEG_ARCHITECTURES: SegArchitecture[] = [
  { value: 'PPLiteSeg', label: 'PP-LiteSeg', logits: 3, defaultCoef: [1, 1, 1], needsBackbone: true, backbones: ['STDC1', 'STDC2'] },
  { value: 'BiSeNetV2', label: 'BiSeNetV2', logits: 5, defaultCoef: [1, 1, 1, 1, 1], needsBackbone: false, backbones: [] },
  { value: 'OCRNet', label: 'OCRNet', logits: 2, defaultCoef: [1, 0.4], needsBackbone: true, backbones: ['HRNet_W18', 'HRNet_W48'] },
  { value: 'UNet', label: 'UNet', logits: 1, defaultCoef: [1], needsBackbone: false, backbones: [] },
  { value: 'DeepLabV3P', label: 'DeepLabV3+', logits: 1, defaultCoef: [1], needsBackbone: true, backbones: ['ResNet50_vd', 'ResNet101_vd'] },
  { value: 'SegFormer_B0', label: 'SegFormer-B0', logits: 1, defaultCoef: [1], needsBackbone: false, backbones: [] },
  { value: 'FCN', label: 'FCN', logits: 1, defaultCoef: [1], needsBackbone: true, backbones: ['HRNet_W18', 'HRNet_W48'] },
];

/**
 * TorchSeg architectures. Must stay in sync with `_TV_BUILDERS` / `UNet` in
 * `torchtrain/torchtrain/seg/models.py`.
 *
 * `logits` is what makes the shared PaddleSeg loss rule work for torch too: the
 * torchvision builders attach an auxiliary FCN head when `aux_loss=True`, and
 * torchtrain enables it exactly when the loss config declares two entries. UNet
 * and LRASPP emit a single logit and take a single loss.
 */
export const TORCH_SEG_ARCHITECTURES: SegArchitecture[] = [
  {
    value: 'UNet',
    label: 'UNet (from scratch)',
    logits: 1,
    defaultCoef: [1],
    needsBackbone: false,
    backbones: [],
  },
  {
    value: 'DeepLabV3P',
    label: 'DeepLabV3+ (torchvision, aux head)',
    logits: 2,
    defaultCoef: [1, 0.4],
    needsBackbone: true,
    backbones: ['ResNet50', 'ResNet101', 'MobileNetV3-Large'],
  },
  {
    value: 'FCN',
    label: 'FCN (torchvision, aux head)',
    logits: 2,
    defaultCoef: [1, 0.4],
    needsBackbone: true,
    backbones: ['ResNet50', 'ResNet101'],
  },
  {
    value: 'LRASPP',
    label: 'LR-ASPP (torchvision, lightweight)',
    logits: 1,
    defaultCoef: [1],
    needsBackbone: true,
    backbones: ['MobileNetV3-Large'],
  },
];

/** Segmentation architecture table for a framework. */
export function segArchitecturesFor(framework: ConfigFramework): SegArchitecture[] {
  return framework === 'TorchSeg' ? TORCH_SEG_ARCHITECTURES : SEG_ARCHITECTURES;
}

/**
 * TorchDet architectures. Must stay in sync with `DET_ARCHITECTURES` in
 * `torchtrain/torchtrain/det/models.py`.
 *
 * Unlike PaddleDetection there is no separate neck/head to wire: torchvision
 * builders are named `<arch>_<backbone>` and construct the whole detector, so
 * the backbone label *is* the variant selector.
 */
export interface TorchDetPreset {
  label: string;
  architecture: string;
  backbones: string[];
  /** Whether COCO-pretrained weights can have their head replaced (transfer). */
  supportsCocoTransfer: boolean;
}

export const TORCH_DET_PRESETS: Record<string, TorchDetPreset> = {
  FasterRCNN: {
    label: 'Faster R-CNN (two-stage, best accuracy)',
    architecture: 'FasterRCNN',
    backbones: ['ResNet50-FPN', 'ResNet50-FPN-v2', 'MobileNetV3-Large-FPN', 'MobileNetV3-Large-320-FPN'],
    supportsCocoTransfer: true,
  },
  RetinaNet: {
    label: 'RetinaNet (one-stage, focal loss)',
    architecture: 'RetinaNet',
    backbones: ['ResNet50-FPN', 'ResNet50-FPN-v2'],
    supportsCocoTransfer: true,
  },
  FCOS: {
    label: 'FCOS (anchor-free)',
    architecture: 'FCOS',
    backbones: ['ResNet50-FPN'],
    supportsCocoTransfer: true,
  },
  SSD: {
    label: 'SSD (fastest, lower accuracy)',
    architecture: 'SSD',
    backbones: ['VGG16', 'MobileNetV3-Large'],
    // Head replacement is not implemented for SSD upstream, so a COCO request
    // degrades to ImageNet backbone weights (with a warning at train time).
    supportsCocoTransfer: false,
  },
};

export const TORCH_DET_PRESET_KEYS = Object.keys(TORCH_DET_PRESETS);

/** Where TorchDet initial weights come from; see `build_model` in det/models.py. */
export const TORCH_PRETRAIN_OPTIONS = [
  { value: 'COCO', label: 'COCO-pretrained (recommended)' },
  { value: 'ImageNet', label: 'ImageNet backbone only' },
  { value: '', label: 'Random initialisation' },
];

// ---------------------------------------------------------------------------
// TorchAnomaly (anomalib) presets
// ---------------------------------------------------------------------------

/**
 * One entry per anomalib algorithm the platform exposes.
 *
 * Every flag here was read off anomalib's source rather than its docs, because
 * the docs are behind in places (the tiling tutorial still shows
 * `Engine(image_metrics=...)`, which no longer exists). The class names are the
 * exported symbols of `anomalib.models` — note `Supersimplenet`, which the
 * module docstring spells `SuperSimpleNet` but does not export under that name.
 */
export interface AnomalyPreset {
  label: string;
  /** Import path used verbatim as `model.class_path`. */
  classPath: string;
  /** Selectable backbones; empty when the architecture is fixed. */
  backbones: string[];
  /** Default feature layers; empty when the model takes no `layers` argument. */
  layers: string[];
  /**
   * Whether `TilerConfigurationCallback` works. Only PaDiM, PatchCore,
   * ReverseDistillation and STFPM implement a `tiler` attribute; the callback
   * raises `ValueError: Model does not support tiling` for anything else.
   */
  supportsTiling: boolean;
  /** Whether training runs backprop, i.e. whether a loss curve exists. */
  hasLoss: boolean;
  /** Whether `lr` / `weight_decay` are constructor arguments. */
  hasLr: boolean;
  /** Recommended input size `[w, h]`. */
  imageSize: [number, number];
  /** Hard requirement: `data.train_batch_size` must be 1. */
  requiresBatchSizeOne?: boolean;
  /** Downloads extra assets on first train (offline machines need a warm cache). */
  downloadsAssets?: string;
  notes: string;
}

export const ANOMALY_PRESETS: Record<string, AnomalyPreset> = {
  Patchcore: {
    label: 'PatchCore (memory bank, best few-shot baseline)',
    classPath: 'anomalib.models.Patchcore',
    backbones: ['wide_resnet50_2', 'resnet18', 'resnet50'],
    layers: ['layer2', 'layer3'],
    supportsTiling: true,
    hasLoss: false,
    hasLr: false,
    imageSize: [256, 256],
    notes:
      'No gradient descent: one pass fills a coreset memory bank, so training is ' +
      'a single epoch and the loss curve stays flat at 0. Works with a few dozen ' +
      'normal images. Memory grows with resolution x tiles x images — turn down ' +
      'coreset_sampling_ratio if it runs out of memory.',
  },
  Padim: {
    label: 'PaDiM (per-position Gaussian, fastest to train)',
    classPath: 'anomalib.models.Padim',
    backbones: ['resnet18', 'wide_resnet50_2'],
    layers: ['layer1', 'layer2', 'layer3'],
    supportsTiling: true,
    hasLoss: false,
    hasLr: false,
    imageSize: [256, 256],
    notes:
      'Models each patch *position* separately, so it only works when the part is ' +
      'rigidly aligned in the frame. If the slider shifts or rotates between ' +
      'images, prefer PatchCore.',
  },
  Stfpm: {
    label: 'STFPM (student-teacher, real loss curve)',
    classPath: 'anomalib.models.Stfpm',
    backbones: ['resnet18', 'wide_resnet50_2'],
    layers: ['layer1', 'layer2', 'layer3'],
    supportsTiling: true,
    hasLoss: true,
    hasLr: false,
    imageSize: [256, 256],
    notes:
      'Trains a student to match a frozen teacher. Optimizer is fixed inside ' +
      'anomalib (SGD, lr 0.4), which is why no learning-rate control is offered.',
  },
  EfficientAd: {
    label: 'EfficientAD (best accuracy/latency, no tiling)',
    classPath: 'anomalib.models.EfficientAd',
    backbones: [],
    layers: [],
    // EfficientAd's torch model has no `tiler`, so enabling tiling is a hard
    // error rather than a silent no-op.
    supportsTiling: false,
    hasLoss: true,
    hasLr: true,
    imageSize: [256, 256],
    requiresBatchSizeOne: true,
    downloadsAssets:
      'pretrained PDN teacher weights (~40 MB, from GitHub releases) and the ' +
      'ImageNette dataset (~1.5 GB, from AWS S3) on the first run',
    notes:
      'Two hard constraints enforced by anomalib at train start: train_batch_size ' +
      'must be 1, and the pre-processing transform must NOT normalise (the ' +
      'adapter uses the model\'s own configure_pre_processor, which honours this). ' +
      'Does not support tiling, so very small defects need a smaller field of view ' +
      'instead.',
  },
  Supersimplenet: {
    label: 'SuperSimpleNet (fast, semi-supervised capable)',
    classPath: 'anomalib.models.Supersimplenet',
    // The upstream default; it must be a torchvision-V1 (`.tv`) weight name.
    backbones: ['wide_resnet50_2.tv_in1k', 'resnet18.tv_in1k'],
    layers: ['layer2', 'layer3'],
    supportsTiling: false,
    hasLoss: true,
    hasLr: false,
    imageSize: [256, 256],
    notes:
      'Discriminative model with synthetic feature-level anomalies. It has a ' +
      '`supervised` flag for training on labelled defects, but anomalib currently ' +
      'wires only the unsupervised path — keep the defect images in the validation ' +
      'split.',
  },
};

export const ANOMALY_PRESET_KEYS = Object.keys(ANOMALY_PRESETS);

export function anomalyPreset(architecture: string): AnomalyPreset {
  return ANOMALY_PRESETS[architecture] ?? ANOMALY_PRESETS.Patchcore;
}

export const ANOMALY_MODEL_SIZES = ['small', 'medium'];

export const SEG_LOSS_TYPES = [
  'CrossEntropyLoss',
  'DiceLoss',
  'LovaszSoftmaxLoss',
  'OhemCrossEntropyLoss',
  'BCELoss',
  'FocalLoss',
  'MixedLoss',
];

export function segArchitecture(
  value: string,
  framework: ConfigFramework = 'PaddleSeg',
): SegArchitecture | undefined {
  return segArchitecturesFor(framework).find((a) => a.value === value);
}

/** Expected logits count for an architecture; 1 when unknown. */
export function segLogitsFor(architecture: string, framework: ConfigFramework = 'PaddleSeg'): number {
  return segArchitecture(architecture, framework)?.logits ?? 1;
}

/**
 * Resize a loss configuration to match the architecture's logits count,
 * preserving whatever the user already chose. This is what makes switching
 * architecture in the form safe.
 */
export function reconcileSegLoss(
  architecture: string,
  types: string[],
  coef: number[],
  framework: ConfigFramework = 'PaddleSeg',
): { segLossTypes: string[]; segLossCoef: number[] } {
  const arch = segArchitecture(architecture, framework);
  const n = arch?.logits ?? 1;
  const fallbackType = types[0] || 'CrossEntropyLoss';
  const nextTypes = Array.from({ length: n }, (_, i) => types[i] ?? fallbackType);
  const nextCoef = Array.from({ length: n }, (_, i) => coef[i] ?? arch?.defaultCoef[i] ?? 1);
  return { segLossTypes: nextTypes, segLossCoef: nextCoef };
}

export interface ModelValidationIssue {
  level: 'error' | 'warning';
  message: string;
}

/** Structural checks surfaced in the dialog before the user can save. */
export function validateModelParams(
  framework: ConfigFramework,
  params: ModelParams,
): ModelValidationIssue[] {
  const issues: ModelValidationIssue[] = [];

  // Anomaly detection has no classes at all — checked before the shared
  // num_classes rule so the dialog does not demand a value it never uses.
  if (framework === 'TorchAnomaly') {
    const preset = ANOMALY_PRESETS[params.architecture];
    if (!preset) {
      issues.push({
        level: 'error',
        message: `${params.architecture} is not available in TorchAnomaly. Choose one of: ${ANOMALY_PRESET_KEYS.join(', ')}.`,
      });
      return issues;
    }
    if (params.adImageWidth < 32 || params.adImageHeight < 32) {
      issues.push({ level: 'error', message: 'Input size must be at least 32x32.' });
    }
    if (params.adCenterCrop > 0 &&
        (params.adCenterCrop > params.adImageWidth || params.adCenterCrop > params.adImageHeight)) {
      issues.push({
        level: 'error',
        message: `Centre crop (${params.adCenterCrop}) cannot exceed the input size (${params.adImageWidth}x${params.adImageHeight}); anomalib raises on this.`,
      });
    }
    if (preset.backbones.length > 0 && params.backbone && !preset.backbones.includes(params.backbone)) {
      issues.push({
        level: 'warning',
        message: `${params.backbone} is not a listed backbone for ${preset.label}; it will fall back to ${preset.backbones[0]}.`,
      });
    }
    if (preset.requiresBatchSizeOne) {
      issues.push({
        level: 'warning',
        message: `${params.architecture} requires train_batch_size = 1; anomalib raises "train_batch_size for EfficientAd should be 1" at train start otherwise. Set it in the training config.`,
      });
    }
    if (!preset.supportsTiling) {
      issues.push({
        level: 'warning',
        message: `${params.architecture} does not support input tiling. Leave tiling off in the training config, or pick PatchCore / PaDiM / STFPM for small defects on large images.`,
      });
    }
    if (preset.downloadsAssets) {
      issues.push({
        level: 'warning',
        message: `First run downloads ${preset.downloadsAssets}. Pre-warm the cache on machines without internet access.`,
      });
    }
    return issues;
  }

  if (!Number.isFinite(params.numClasses) || params.numClasses < 1) {
    issues.push({ level: 'error', message: 'num_classes must be at least 1.' });
  }

  if (framework === 'PaddleSeg' || framework === 'TorchSeg') {
    const expected = segLogitsFor(params.architecture, framework);
    if (params.segLossTypes.length !== expected) {
      issues.push({
        level: 'error',
        message: `${params.architecture} emits ${expected} logits during training, but ${params.segLossTypes.length} loss type(s) are configured. ${framework} requires them to match exactly.`,
      });
    }
    if (params.segLossCoef.length !== params.segLossTypes.length) {
      issues.push({ level: 'error', message: 'loss.coef must have the same length as loss.types.' });
    }
    if (params.numClasses < 2) {
      issues.push({
        level: 'warning',
        message: 'Segmentation num_classes normally includes the background class, so it is usually >= 2.',
      });
    }
    const arch = segArchitecture(params.architecture, framework);
    if (arch?.needsBackbone && params.backbone && !arch.backbones.includes(params.backbone)) {
      issues.push({
        level: 'warning',
        message: `${params.backbone} is not a standard backbone for ${params.architecture}; it will fall back to ${arch.backbones[0]}.`,
      });
    }
    return issues;
  }

  if (framework === 'TorchDet') {
    const preset = TORCH_DET_PRESETS[params.architecture];
    if (!preset) {
      issues.push({
        level: 'error',
        message: `${params.architecture} is not available in TorchDet. Choose one of: ${TORCH_DET_PRESET_KEYS.join(', ')}.`,
      });
      return issues;
    }
    if (params.backbone && !preset.backbones.includes(params.backbone)) {
      issues.push({
        level: 'warning',
        message: `${params.backbone} is not a valid backbone for ${preset.architecture}; it will fall back to ${preset.backbones[0]}.`,
      });
    }
    if (!preset.supportsCocoTransfer && /^coco$/i.test(params.pretrainWeights)) {
      issues.push({
        level: 'warning',
        message: `${preset.architecture} does not support COCO head replacement; training will fall back to ImageNet backbone weights.`,
      });
    }
    return issues;
  }

  if (framework === 'PaddleDetection') {
    const presetKey = detectionPresetFor(params.architecture, params.head);
    const preset = DETECTION_PRESETS[presetKey];
    if (!preset.backbones.includes(params.backbone)) {
      issues.push({
        level: 'warning',
        message: `${params.backbone} is not a standard backbone for ${preset.label}; PaddleDetection may fail to build the model.`,
      });
    }
    if (!preset.necks.includes(params.neck)) {
      issues.push({
        level: 'warning',
        message: `${params.neck} is not a standard neck for ${preset.label}.`,
      });
    }
  }

  return issues;
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

/**
 * Baseline every framework's defaults are layered onto.
 *
 * Spelling out all fields in each branch meant a new parameter had to be added
 * in five places or TypeScript rejected the object — and a forgotten branch is a
 * silent `undefined` in a config generator.
 */
const BASE_MODEL_PARAMS: ModelParams = {
  architecture: '',
  backbone: '',
  neck: '',
  head: '',
  numClasses: 1,
  normType: 'bn',
  useEma: false,
  emaDecay: 0.9998,
  depthMult: 1,
  widthMult: 1,
  segLossTypes: [],
  segLossCoef: [],
  segAlignCorners: false,
  adImageWidth: 256,
  adImageHeight: 256,
  adCenterCrop: 0,
  adLayers: [],
  adLr: 0.0001,
  adWeightDecay: 0.00001,
  adCoresetRatio: 0.1,
  adNumNeighbors: 9,
  adModelSize: 'small',
  pretrainWeights: '',
};

export function defaultModelParams(framework: ConfigFramework): ModelParams {
  if (framework === 'TorchSeg') {
    // UNet is the default rather than a pretrained backbone network: this
    // platform's segmentation datasets are typically a few dozen single-channel
    // microscopy images, where an ImageNet ResNet overfits quickly.
    const arch = TORCH_SEG_ARCHITECTURES[0];
    return {
      ...BASE_MODEL_PARAMS,
      architecture: arch.value,
      backbone: arch.backbones[0] ?? '',
      numClasses: 2,
      segLossTypes: Array.from({ length: arch.logits }, () => 'CrossEntropyLoss'),
      segLossCoef: [...arch.defaultCoef],
    };
  }

  if (framework === 'TorchDet') {
    const preset = TORCH_DET_PRESETS.FasterRCNN;
    return {
      ...BASE_MODEL_PARAMS,
      architecture: preset.architecture,
      backbone: preset.backbones[0],
      // COCO transfer is the single biggest accuracy win on small datasets.
      pretrainWeights: 'COCO',
    };
  }

  if (framework === 'TorchAnomaly') {
    // PatchCore first: it needs no gradient descent, tolerates a few dozen
    // normal images, and supports tiling — the safest way to find out whether a
    // dataset is workable at all before spending time on a trained model.
    const preset = ANOMALY_PRESETS.Patchcore;
    return {
      ...BASE_MODEL_PARAMS,
      architecture: 'Patchcore',
      backbone: preset.backbones[0] ?? '',
      adLayers: [...preset.layers],
      adImageWidth: preset.imageSize[0],
      adImageHeight: preset.imageSize[1],
    };
  }

  if (framework === 'PaddleSeg') {
    const arch = SEG_ARCHITECTURES[0];
    return {
      ...BASE_MODEL_PARAMS,
      architecture: arch.value,
      backbone: arch.backbones[1] ?? arch.backbones[0] ?? '',
      numClasses: 2,
      segLossTypes: Array.from({ length: arch.logits }, () => 'CrossEntropyLoss'),
      segLossCoef: [...arch.defaultCoef],
    };
  }

  if (framework === 'PaddleClas') {
    return { ...BASE_MODEL_PARAMS, architecture: 'ResNet50', numClasses: 2 };
  }

  const preset = DETECTION_PRESETS['PP-YOLOE'];
  return {
    ...BASE_MODEL_PARAMS,
    architecture: preset.architecture,
    backbone: preset.backbones[0],
    neck: preset.necks[0],
    head: preset.heads[0],
    normType: 'sync_bn',
    useEma: true,
    depthMult: 0.33,
    widthMult: 0.5,
  };
}

export const CLAS_ARCHITECTURES = [
  'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101',
  'MobileNetV3_small_x1_0', 'MobileNetV3_large_x1_0',
  'PPLCNet_x1_0', 'PPHGNet_small', 'SwinTransformer_tiny_patch4_window7_224',
];

// ---------------------------------------------------------------------------
// YAML generation
// ---------------------------------------------------------------------------

function block(name: string, lines: string[]): string {
  if (lines.length === 0) return '';
  return `${name}:\n${lines.map((l) => `  ${l}`).join('\n')}\n`;
}

function generateDetectionModelYaml(p: ModelParams, name: string): string {
  const presetKey = detectionPresetFor(p.architecture, p.head);
  const preset = DETECTION_PRESETS[presetKey];

  const archLines = [
    `backbone: ${p.backbone}`,
    ...(p.neck ? [`neck: ${p.neck}`] : []),
    `${preset.headKey}: ${p.head}`,
    ...(preset.archExtras ?? []),
  ];

  // Only emit component blocks we have accurate parameters for. An unknown
  // component gets no block, so PaddleDetection uses its registered defaults
  // rather than parameters copied from an unrelated network.
  const backboneBlock = BACKBONE_BLOCKS[p.backbone] ? block(p.backbone, BACKBONE_BLOCKS[p.backbone](p)) : '';
  const neckBlock = p.neck && NECK_BLOCKS[p.neck] ? block(p.neck, NECK_BLOCKS[p.neck](p)) : '';
  const headBlock = HEAD_BLOCKS[p.head] ? block(p.head, HEAD_BLOCKS[p.head](p)) : block(p.head, [`num_classes: ${p.numClasses}`]);

  const multipliers = preset.usesMultipliers
    ? `depth_mult: ${p.depthMult}\nwidth_mult: ${p.widthMult}\n`
    : '';

  const emaLines = p.useEma
    ? `use_ema: true\nema_decay: ${p.emaDecay}\nema_black_list: ['proj_conv.weight']\n`
    : 'use_ema: false\n';

  return `# ${name}
# Model configuration generated by AutoTrain (PaddleDetection / ${preset.label})

architecture: ${preset.architecture}
norm_type: ${p.normType}
${emaLines}custom_black_list: ['reduce_mean']
${p.pretrainWeights ? `pretrain_weights: ${p.pretrainWeights}\n` : ''}
${block(preset.architecture, archLines)}
${backboneBlock}${neckBlock ? `\n${neckBlock}` : ''}
${headBlock}${multipliers ? `\n${multipliers}` : ''}`;
}

function generateSegModelYaml(
  p: ModelParams,
  name: string,
  framework: ConfigFramework = 'PaddleSeg',
): string {
  const arch = segArchitecture(p.architecture, framework);
  const needsBackbone = arch?.needsBackbone ?? false;
  const isTorch = framework === 'TorchSeg';

  const modelLines = [`type: ${p.architecture}`, `num_classes: ${p.numClasses}`];
  if (p.segAlignCorners) modelLines.push('align_corners: True');
  if (needsBackbone && p.backbone) {
    // torchtrain reads `pretrained: imagenet` as "download ImageNet backbone
    // weights"; PaddleSeg expects a URL or `Null`. Default accordingly so
    // neither framework silently trains a pretrained backbone from scratch.
    const pretrained = p.pretrainWeights || (isTorch ? 'imagenet' : 'Null');
    modelLines.push('backbone:', `  type: ${p.backbone}`, `  pretrained: ${pretrained}`);
  } else if (p.pretrainWeights) {
    modelLines.push(`pretrained: ${p.pretrainWeights}`);
  }

  const lossLines = [
    'types:',
    ...p.segLossTypes.map((t) => `  - type: ${t}`),
    `coef: [${p.segLossCoef.join(', ')}]`,
  ];

  const header = isTorch
    ? `# ${name}
# Model configuration generated by AutoTrain (TorchSeg / PyTorch)
# ${p.architecture} emits ${arch?.logits ?? '?'} logit(s) during training, so
# loss.types must have exactly that many entries. torchtrain attaches the
# torchvision auxiliary head exactly when two entries are configured.`
    : `# ${name}
# Model configuration generated by AutoTrain (PaddleSeg)
# ${p.architecture} emits ${arch?.logits ?? '?'} logit(s) during training, so
# loss.types must have exactly that many entries.`;

  return `${header}

${block('model', modelLines)}
${block('loss', lossLines)}`;
}

/**
 * TorchDet model config.
 *
 * Kept separate from `generateDetectionModelYaml` rather than parameterised:
 * torchvision builders construct the whole detector from `<arch>_<backbone>`, so
 * there is no neck/head to wire and none of PaddleDetection's component blocks
 * (`CSPResNet:`, `PPYOLOEHead:`, `norm_type`, `depth_mult`) mean anything. Emitting
 * them would produce a config that reads as if it configured something.
 */
function generateTorchDetModelYaml(p: ModelParams, name: string): string {
  const preset = TORCH_DET_PRESETS[p.architecture] ?? TORCH_DET_PRESETS.FasterRCNN;
  const backbone = preset.backbones.includes(p.backbone) ? p.backbone : preset.backbones[0];
  const pretrain = p.pretrainWeights || '';

  return `# ${name}
# Model configuration generated by AutoTrain (TorchDet / PyTorch)
#
# ${preset.label}
# torchvision builds the full detector from architecture + backbone, so there is
# no separate neck/head block. \`num_classes\` is set by the dataset config and
# counts foreground classes only (PaddleDetection's convention); torchtrain adds
# the background class itself.
#
# pretrain_weights:
#   COCO      COCO-pretrained detector with its classifier replaced${preset.supportsCocoTransfer ? '' : ' (unsupported for SSD; falls back to ImageNet)'}
#   ImageNet  ImageNet backbone only
#   <path>.pt a checkpoint produced by this platform
#   (empty)   random initialisation

architecture: ${preset.architecture}

${preset.architecture}:
  backbone: ${backbone}
${pretrain ? `\npretrain_weights: ${pretrain}\n` : ''}`;
}

function generateClasModelYaml(p: ModelParams, name: string): string {
  return `# ${name}
# Model configuration generated by AutoTrain (PaddleClas)

Arch:
  name: ${p.architecture}
  class_num: ${p.numClasses}
${p.pretrainWeights ? `  pretrained: ${p.pretrainWeights}\n` : ''}
Loss:
  Train:
    - CELoss:
        weight: 1.0
  Eval:
    - CELoss:
        weight: 1.0
`;
}

/**
 * anomalib model config: the `model:` block plus the platform's `autotrain:`
 * additions.
 *
 * `image_size` is deliberately *not* expressed as anomalib's own
 * `model.init_args.pre_processor`. Doing that would mean nesting a
 * `torchvision.transforms.v2.Compose` through jsonargparse's `class_path`
 * plumbing, which is verbose, easy to get subtly wrong, and would let a user
 * accidentally add a `Normalize` that makes EfficientAd refuse to train. Instead
 * the adapter calls the model class's own `configure_pre_processor(image_size=)`,
 * which is guaranteed to produce the transform that algorithm expects.
 */
function generateAnomalyModelYaml(p: ModelParams, name: string): string {
  const preset = anomalyPreset(p.architecture);
  const backbone = preset.backbones.length > 0
    ? (preset.backbones.includes(p.backbone) ? p.backbone : preset.backbones[0])
    : '';
  const layers = p.adLayers.length > 0 ? p.adLayers : preset.layers;

  const initArgs: string[] = [];
  if (backbone) initArgs.push(`    backbone: ${backbone}`);
  if (preset.layers.length > 0 && layers.length > 0) {
    initArgs.push(`    layers: [${layers.join(', ')}]`);
  }
  if (p.architecture === 'Patchcore') {
    initArgs.push(
      '    pre_trained: true',
      `    coreset_sampling_ratio: ${p.adCoresetRatio}`,
      `    num_neighbors: ${Math.max(1, Math.round(p.adNumNeighbors))}`,
    );
  }
  if (p.architecture === 'EfficientAd') {
    initArgs.push(
      `    model_size: ${ANOMALY_MODEL_SIZES.includes(p.adModelSize) ? p.adModelSize : 'small'}`,
      `    teacher_out_channels: 384`,
    );
  }
  if (preset.hasLr) {
    initArgs.push(`    lr: ${p.adLr}`, `    weight_decay: ${p.adWeightDecay}`);
  }

  const notes = preset.notes
    .split('\n')
    .map((line) => `# ${line}`)
    .join('\n');

  return `# ${name}
# Model configuration generated by AutoTrain (TorchAnomaly / anomalib)
#
# ${preset.label}
${notes}
#
# Trains on normal images only. \`image_size\` lives here rather than in the
# training config because anomalib applies the resize inside the model's
# PreProcessor, and each algorithm has its own recommended size.
${preset.supportsTiling ? '' : '#\n# NOTE: this model does not support input tiling; leave tiling off.\n'}
model:
  class_path: ${preset.classPath}
${initArgs.length > 0 ? `  init_args:\n${initArgs.join('\n')}\n` : ''}
autotrain:
  image_size: [${Math.max(32, Math.round(p.adImageWidth))}, ${Math.max(32, Math.round(p.adImageHeight))}]
${p.adCenterCrop > 0 ? `  center_crop_size: ${Math.round(p.adCenterCrop)}\n` : ''}`;
}

export function generateModelYaml(
  framework: ConfigFramework,
  params: ModelParams,
  modelName = 'Model Config',
): string {
  switch (framework) {
    case 'PaddleSeg':
    case 'TorchSeg':
      return generateSegModelYaml(params, modelName, framework);
    case 'PaddleClas':
      return generateClasModelYaml(params, modelName);
    case 'TorchDet':
      return generateTorchDetModelYaml(params, modelName);
    case 'TorchAnomaly':
      return generateAnomalyModelYaml(params, modelName);
    default:
      return generateDetectionModelYaml(params, modelName);
  }
}

// ---------------------------------------------------------------------------
// YAML -> parameters
// ---------------------------------------------------------------------------

function toNum(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

/**
 * Parse a model YAML into parameters.
 *
 * Replaces a hand-rolled line-by-line regex scanner that only recognised
 * PaddleDetection's flat keys — every imported PaddleSeg model was recorded as
 * "YOLOv3 / CSPResNet" regardless of its real architecture.
 */
export function parseModelParams(
  framework: ConfigFramework,
  yamlText: string | null | undefined,
): Partial<ModelParams> {
  if (!yamlText || !yamlText.trim()) return {};

  let doc: Record<string, any>;
  try {
    const parsed = parseDocument(yamlText, { logLevel: 'silent' });
    if (parsed.errors.length > 0 || !isMap(parsed.contents)) return {};
    doc = parsed.toJS({ maxAliasCount: -1 }) as Record<string, any>;
  } catch {
    return {};
  }
  if (!doc || typeof doc !== 'object') return {};

  const out: Partial<ModelParams> = {};

  // --- TorchAnomaly (anomalib) ---------------------------------------------
  // Must be tested before PaddleSeg: both schemas use a top-level `model:` key,
  // and only anomalib's carries `class_path`.
  if (doc.model && typeof doc.model === 'object' && typeof doc.model.class_path === 'string') {
    const args = doc.model.init_args ?? {};
    // `anomalib.models.Patchcore` -> `Patchcore`
    const className = doc.model.class_path.split('.').pop() ?? '';
    if (className) out.architecture = className;
    if (typeof args.backbone === 'string') out.backbone = args.backbone;
    if (Array.isArray(args.layers)) {
      out.adLayers = args.layers.filter((l: unknown): l is string => typeof l === 'string');
    }
    const coreset = toNum(args.coreset_sampling_ratio);
    if (coreset !== undefined) out.adCoresetRatio = coreset;
    const neighbors = toNum(args.num_neighbors);
    if (neighbors !== undefined) out.adNumNeighbors = neighbors;
    if (typeof args.model_size === 'string') out.adModelSize = args.model_size;
    const lr = toNum(args.lr);
    if (lr !== undefined) out.adLr = lr;
    const wd = toNum(args.weight_decay);
    if (wd !== undefined) out.adWeightDecay = wd;

    const platform = doc.autotrain ?? {};
    if (Array.isArray(platform.image_size)) {
      const size = platform.image_size.map(toNum).filter((n: number | undefined): n is number => n !== undefined);
      if (size.length >= 2) {
        out.adImageWidth = size[0];
        out.adImageHeight = size[1];
      } else if (size.length === 1) {
        out.adImageWidth = size[0];
        out.adImageHeight = size[0];
      }
    } else {
      const square = toNum(platform.image_size);
      if (square !== undefined) {
        out.adImageWidth = square;
        out.adImageHeight = square;
      }
    }
    const crop = toNum(platform.center_crop_size);
    if (crop !== undefined) out.adCenterCrop = crop;
    return out;
  }

  // --- PaddleSeg -----------------------------------------------------------
  if (doc.model && typeof doc.model === 'object') {
    const m = doc.model;
    if (typeof m.type === 'string') out.architecture = m.type;
    const nc = toNum(m.num_classes);
    if (nc !== undefined) out.numClasses = nc;
    if (typeof m.align_corners === 'boolean') out.segAlignCorners = m.align_corners;
    if (m.backbone && typeof m.backbone === 'object') {
      if (typeof m.backbone.type === 'string') out.backbone = m.backbone.type;
      if (typeof m.backbone.pretrained === 'string' && m.backbone.pretrained !== 'Null') {
        out.pretrainWeights = m.backbone.pretrained;
      }
    } else if (typeof m.pretrained === 'string' && m.pretrained !== 'Null') {
      out.pretrainWeights = m.pretrained;
    }

    if (doc.loss && typeof doc.loss === 'object') {
      const types = Array.isArray(doc.loss.types) ? doc.loss.types : [];
      const parsedTypes = types
        .map((t: any) => (typeof t?.type === 'string' ? t.type : undefined))
        .filter((t: unknown): t is string => typeof t === 'string');
      if (parsedTypes.length > 0) out.segLossTypes = parsedTypes;
      const coef = Array.isArray(doc.loss.coef)
        ? doc.loss.coef.map(toNum).filter((n: number | undefined): n is number => n !== undefined)
        : [];
      if (coef.length > 0) out.segLossCoef = coef;
    }
    return out;
  }

  // --- PaddleClas ----------------------------------------------------------
  if (doc.Arch && typeof doc.Arch === 'object') {
    if (typeof doc.Arch.name === 'string') out.architecture = doc.Arch.name;
    const nc = toNum(doc.Arch.class_num);
    if (nc !== undefined) out.numClasses = nc;
    if (typeof doc.Arch.pretrained === 'string') out.pretrainWeights = doc.Arch.pretrained;
    return out;
  }

  // --- PaddleDetection -----------------------------------------------------
  if (typeof doc.architecture === 'string') out.architecture = doc.architecture;
  if (typeof doc.norm_type === 'string') out.normType = doc.norm_type;
  if (typeof doc.use_ema === 'boolean') out.useEma = doc.use_ema;
  const emaDecay = toNum(doc.ema_decay);
  if (emaDecay !== undefined) out.emaDecay = emaDecay;
  const depthMult = toNum(doc.depth_mult);
  if (depthMult !== undefined) out.depthMult = depthMult;
  const widthMult = toNum(doc.width_mult);
  if (widthMult !== undefined) out.widthMult = widthMult;
  if (typeof doc.pretrain_weights === 'string') out.pretrainWeights = doc.pretrain_weights;

  const archBlock = out.architecture ? doc[out.architecture] : undefined;
  if (archBlock && typeof archBlock === 'object') {
    if (typeof archBlock.backbone === 'string') out.backbone = archBlock.backbone;
    if (typeof archBlock.neck === 'string') out.neck = archBlock.neck;
    // The head lives under a different key per meta-architecture.
    for (const key of ['yolo_head', 'head', 'bbox_head', 'detr_head'] as const) {
      if (typeof archBlock[key] === 'string') {
        out.head = archBlock[key];
        break;
      }
    }
  }

  // num_classes lives on the head block, not at the top level, in most
  // PaddleDetection model configs.
  if (out.head && doc[out.head] && typeof doc[out.head] === 'object') {
    const nc = toNum(doc[out.head].num_classes);
    if (nc !== undefined) out.numClasses = nc;
  }
  if (out.numClasses === undefined) {
    const nc = toNum(doc.num_classes);
    if (nc !== undefined) out.numClasses = nc;
  }

  void framework;
  return out;
}

/** Project parameters onto the flat `Model` columns used by the list view. */
export function modelParamsToColumns(params: ModelParams) {
  return {
    architecture: params.architecture,
    backbone: params.backbone || '',
    neck: params.neck || '',
    head: params.head || '',
    numClasses: Math.max(1, Math.round(params.numClasses)),
    normType: params.normType,
    useEma: params.useEma,
    emaDecay: params.emaDecay,
    depthMult: params.depthMult,
    widthMult: params.widthMult,
    pretrainWeights: params.pretrainWeights || null,
  };
}
