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

export const SEG_LOSS_TYPES = [
  'CrossEntropyLoss',
  'DiceLoss',
  'LovaszSoftmaxLoss',
  'OhemCrossEntropyLoss',
  'BCELoss',
  'FocalLoss',
  'MixedLoss',
];

export function segArchitecture(value: string): SegArchitecture | undefined {
  return SEG_ARCHITECTURES.find((a) => a.value === value);
}

/** Expected logits count for an architecture; 1 when unknown. */
export function segLogitsFor(architecture: string): number {
  return segArchitecture(architecture)?.logits ?? 1;
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
): { segLossTypes: string[]; segLossCoef: number[] } {
  const arch = segArchitecture(architecture);
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

  if (!Number.isFinite(params.numClasses) || params.numClasses < 1) {
    issues.push({ level: 'error', message: 'num_classes must be at least 1.' });
  }

  if (framework === 'PaddleSeg') {
    const expected = segLogitsFor(params.architecture);
    if (params.segLossTypes.length !== expected) {
      issues.push({
        level: 'error',
        message: `${params.architecture} emits ${expected} logits during training, but ${params.segLossTypes.length} loss type(s) are configured. PaddleSeg requires them to match exactly.`,
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

export function defaultModelParams(framework: ConfigFramework): ModelParams {
  if (framework === 'PaddleSeg') {
    const arch = SEG_ARCHITECTURES[0];
    return {
      architecture: arch.value,
      backbone: arch.backbones[1] ?? arch.backbones[0] ?? '',
      neck: '',
      head: '',
      numClasses: 2,
      normType: 'bn',
      useEma: false,
      emaDecay: 0.9998,
      depthMult: 1,
      widthMult: 1,
      segLossTypes: Array.from({ length: arch.logits }, () => 'CrossEntropyLoss'),
      segLossCoef: [...arch.defaultCoef],
      segAlignCorners: false,
      pretrainWeights: '',
    };
  }

  if (framework === 'PaddleClas') {
    return {
      architecture: 'ResNet50',
      backbone: '',
      neck: '',
      head: '',
      numClasses: 2,
      normType: 'bn',
      useEma: false,
      emaDecay: 0.9998,
      depthMult: 1,
      widthMult: 1,
      segLossTypes: [],
      segLossCoef: [],
      segAlignCorners: false,
      pretrainWeights: '',
    };
  }

  const preset = DETECTION_PRESETS['PP-YOLOE'];
  return {
    architecture: preset.architecture,
    backbone: preset.backbones[0],
    neck: preset.necks[0],
    head: preset.heads[0],
    numClasses: 1,
    normType: 'sync_bn',
    useEma: true,
    emaDecay: 0.9998,
    depthMult: 0.33,
    widthMult: 0.5,
    segLossTypes: [],
    segLossCoef: [],
    segAlignCorners: false,
    pretrainWeights: '',
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

function generateSegModelYaml(p: ModelParams, name: string): string {
  const arch = segArchitecture(p.architecture);
  const needsBackbone = arch?.needsBackbone ?? false;

  const modelLines = [`type: ${p.architecture}`, `num_classes: ${p.numClasses}`];
  if (p.segAlignCorners) modelLines.push('align_corners: True');
  if (needsBackbone && p.backbone) {
    modelLines.push('backbone:', `  type: ${p.backbone}`, `  pretrained: ${p.pretrainWeights || 'Null'}`);
  } else if (p.pretrainWeights) {
    modelLines.push(`pretrained: ${p.pretrainWeights}`);
  }

  const lossLines = [
    'types:',
    ...p.segLossTypes.map((t) => `  - type: ${t}`),
    `coef: [${p.segLossCoef.join(', ')}]`,
  ];

  return `# ${name}
# Model configuration generated by AutoTrain (PaddleSeg)
# ${p.architecture} emits ${arch?.logits ?? '?'} logit(s) during training, so
# loss.types must have exactly that many entries.

${block('model', modelLines)}
${block('loss', lossLines)}`;
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

export function generateModelYaml(
  framework: ConfigFramework,
  params: ModelParams,
  modelName = 'Model Config',
): string {
  switch (framework) {
    case 'PaddleSeg':
      return generateSegModelYaml(params, modelName);
    case 'PaddleClas':
      return generateClasModelYaml(params, modelName);
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
