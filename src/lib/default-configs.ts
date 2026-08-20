import * as fs from 'fs';
import * as path from 'path';

/**
 * Central definition of "default config folder" naming and built-in starter
 * configs served on the Load Model / Load Training Config pages.
 *
 * On disk we keep each framework's defaults in a separate folder under
 * `{userConfigsPath}` so users can browse them independently:
 *   - `default/`      — PaddleDetection / PaddleClas (object detection)
 *   - `defaultSeg/`   — PaddleSeg (segmentation)
 *   - `defaultTorchSeg/` — TorchSeg (PyTorch segmentation)
 *   - `defaultTorchDet/` — TorchDet (PyTorch detection)
 * Each folder mirrors the same substructure: `{models, training, datasets, jobs}`.
 *
 * The torch frameworks get their own folders even though they consume the same
 * *schema* as their Paddle counterparts, because the architectures differ (a
 * `PPLiteSeg` starter is useless to TorchSeg, and vice versa).
 *
 * The GET endpoints for models and training configs pick the folder for the
 * project's framework and, on first use, seed it with the built-in starter
 * files below so the dropdown is never empty for a fresh install.
 */

export interface BuiltinConfig {
  /** File name without extension. */
  name: string;
  /** Full YAML content. */
  content: string;
}

/** Return the top-level default-configs folder name for a framework. */
export function getDefaultFolderName(framework: string): string {
  if (framework === 'PaddleSeg') return 'defaultSeg';
  if (framework === 'TorchSeg') return 'defaultTorchSeg';
  if (framework === 'TorchDet') return 'defaultTorchDet';
  return 'default';
}

/** Absolute path to `{userConfigsPath}/{defaultFolder}/{subdir}` (e.g. `models`). */
export function getDefaultConfigDir(
  userConfigsPath: string,
  framework: string,
  subdir: 'models' | 'training' | 'datasets' | 'jobs',
): string {
  return path.join(userConfigsPath, getDefaultFolderName(framework), subdir);
}

// ---------------------------------------------------------------------------
// Built-in starter configs. Keep tiny, editable, and framework-appropriate.
// ---------------------------------------------------------------------------

// Each PaddleSeg architecture emits a different number of logits during
// training (main head + N auxiliary heads). PaddleSeg strictly enforces
//   len(loss.types) == len(model.logits)
// or it throws `RuntimeError: The length of logits_list should equal to the
// types of loss config`. So every model YAML ships with a matching `loss:`
// block. When merged with a training YAML the model YAML wins on duplicate
// keys (concatenation order = dataset → training → model), which is what we
// want since loss belongs to the model architecture, not the schedule.
const PADDLESEG_MODEL_CONFIGS: BuiltinConfig[] = [
  {
    name: 'pp_liteseg_stdc2',
    content: `# PP-LiteSeg with STDC2 backbone
# Real-time semantic segmentation, good speed/accuracy trade-off.
# Emits 3 logits (main + 2 auxiliary heads) during training.
model:
  type: PPLiteSeg
  num_classes: 2
  backbone:
    type: STDC2
    pretrained: Null

loss:
  types:
    - type: CrossEntropyLoss
    - type: CrossEntropyLoss
    - type: CrossEntropyLoss
  coef: [1, 1, 1]
`,
  },
  {
    name: 'pp_liteseg_stdc1',
    content: `# PP-LiteSeg with STDC1 backbone
# Lighter variant of PP-LiteSeg for faster inference on low-power devices.
# Emits 3 logits (main + 2 auxiliary heads) during training.
model:
  type: PPLiteSeg
  num_classes: 2
  backbone:
    type: STDC1
    pretrained: Null

loss:
  types:
    - type: CrossEntropyLoss
    - type: CrossEntropyLoss
    - type: CrossEntropyLoss
  coef: [1, 1, 1]
`,
  },
  {
    name: 'unet',
    content: `# UNet
# Classic encoder-decoder segmentation network. Works well on small datasets.
# Emits 1 logit (main head only).
model:
  type: UNet
  num_classes: 2

loss:
  types:
    - type: CrossEntropyLoss
  coef: [1]
`,
  },
  {
    name: 'deeplabv3p_resnet50',
    content: `# DeepLabV3+ with ResNet50-vd backbone
# Strong general-purpose segmentation model, higher accuracy at more compute.
# Emits 1 logit (main head only; no auxiliary head enabled).
model:
  type: DeepLabV3P
  num_classes: 2
  backbone:
    type: ResNet50_vd
    pretrained: Null

loss:
  types:
    - type: CrossEntropyLoss
  coef: [1]
`,
  },
  {
    name: 'ocrnet_hrnet_w18',
    content: `# OCRNet with HRNet-W18 backbone
# Object-Contextual Representations, competitive on scene segmentation.
# Emits 2 logits (main head + auxiliary head).
model:
  type: OCRNet
  num_classes: 2
  backbone:
    type: HRNet_W18
    pretrained: Null

loss:
  types:
    - type: CrossEntropyLoss
    - type: CrossEntropyLoss
  coef: [1, 0.4]
`,
  },
  {
    name: 'bisenetv2',
    content: `# BiSeNetV2
# Bilateral Segmentation Network v2, fast inference for real-time use.
# Emits 5 logits (main + 4 auxiliary segmentation heads) during training.
model:
  type: BiSeNetV2
  num_classes: 2

loss:
  types:
    - type: CrossEntropyLoss
    - type: CrossEntropyLoss
    - type: CrossEntropyLoss
    - type: CrossEntropyLoss
    - type: CrossEntropyLoss
  coef: [1, 1, 1, 1, 1]
`,
  },
];

// Training YAMLs contain only the schedule (iters/optimizer/lr) — the `loss:`
// block lives in the model YAML because it must match the model's logits
// count. Keeping the two separate lets any training preset combine with any
// model preset without producing a mismatched loss.
const PADDLESEG_TRAINING_CONFIGS: BuiltinConfig[] = [
  {
    name: 'seg_config_80k_512',
    content: `# PaddleSeg training preset - 80k iters @ 512x512
# Baseline schedule suitable for datasets with a few hundred to a few
# thousand training images. Adjust batch_size / iters to fit your GPU.
# Note: \`loss:\` is intentionally not defined here — it comes from the model
# YAML (each PaddleSeg architecture emits a different number of logits).

batch_size: 4
iters: 80000

optimizer:
  type: SGD
  momentum: 0.9
  weight_decay: 5.0e-4

lr_scheduler:
  type: PolynomialDecay
  learning_rate: 0.01
  power: 0.9
  end_lr: 0

# Runtime
use_gpu: true
save_interval: 2000
log_iters: 20
save_dir: output/seg_config_80k_512
`,
  },
  {
    name: 'seg_config_20k_512_quick',
    content: `# PaddleSeg training preset - 20k iters @ 512x512 (quick sanity run)
# Use this to verify the pipeline end-to-end before committing to a full run.
# Note: \`loss:\` is intentionally not defined here — it comes from the model
# YAML (each PaddleSeg architecture emits a different number of logits).

batch_size: 4
iters: 20000

optimizer:
  type: SGD
  momentum: 0.9
  weight_decay: 5.0e-4

lr_scheduler:
  type: PolynomialDecay
  learning_rate: 0.01
  power: 0.9
  end_lr: 0

use_gpu: true
save_interval: 1000
log_iters: 20
save_dir: output/seg_config_20k_512_quick
`,
  },
];

// ---------------------------------------------------------------------------
// TorchSeg (PyTorch semantic segmentation)
// ---------------------------------------------------------------------------

// Same PaddleSeg schema, different architectures. The logits rule is identical:
// `len(loss.types)` must equal the number of logits the architecture emits, and
// for the torchvision models that count is what decides whether the auxiliary
// head is attached. See `torchtrain/torchtrain/seg/models.py`.
const TORCHSEG_MODEL_CONFIGS: BuiltinConfig[] = [
  {
    name: 'unet',
    content: `# UNet (PyTorch, trained from scratch)
# The best default for small datasets (tens of images): no ImageNet backbone to
# overfit, and it handles single-channel microscopy imagery well.
# Emits 1 logit (main head only).
model:
  type: UNet
  num_classes: 2

loss:
  types:
    - type: CrossEntropyLoss
  coef: [1]
`,
  },
  {
    name: 'unet_dice_ce',
    content: `# UNet with a Dice + CrossEntropy mix
# Use when the foreground occupies only a small fraction of each image: plain
# cross-entropy converges to "predict background everywhere", while the Dice
# term keeps gradient on the rare classes.
# Emits 1 logit, so exactly one (mixed) loss entry.
model:
  type: UNet
  num_classes: 2

loss:
  types:
    - type: MixedLoss
      losses:
        - type: CrossEntropyLoss
        - type: DiceLoss
      coef: [1, 1]
  coef: [1]
`,
  },
  {
    name: 'deeplabv3p_resnet50',
    content: `# DeepLabV3+ with an ImageNet ResNet50 backbone (torchvision)
# Higher ceiling than UNet once you have a few hundred labelled images.
# torchvision attaches an auxiliary FCN head, so it emits 2 logits and the loss
# config must have exactly 2 entries.
model:
  type: DeepLabV3P
  num_classes: 2
  backbone:
    type: ResNet50
    pretrained: imagenet

loss:
  types:
    - type: CrossEntropyLoss
    - type: CrossEntropyLoss
  coef: [1, 0.4]
`,
  },
  {
    name: 'lraspp_mobilenetv3',
    content: `# LR-ASPP with MobileNetV3-Large (torchvision)
# Lightweight and fast; good when inference latency matters more than accuracy.
# Emits 1 logit (LR-ASPP has no auxiliary head).
model:
  type: LRASPP
  num_classes: 2
  backbone:
    type: MobileNetV3-Large
    pretrained: imagenet

loss:
  types:
    - type: CrossEntropyLoss
  coef: [1]
`,
  },
];

const TORCHSEG_TRAINING_CONFIGS: BuiltinConfig[] = [
  {
    name: 'torchseg_2k_512_quick',
    content: `# TorchSeg preset - 2k iters @ 512x512 (quick sanity run)
# Verifies the whole pipeline end-to-end in a couple of minutes on one GPU.
# Note: \`loss:\` is intentionally not defined here — it comes from the model
# YAML, because len(loss.types) must match the architecture's logits count.

batch_size: 4
iters: 2000

optimizer:
  type: Momentum
  momentum: 0.9
  weight_decay: 5.0e-4

lr_scheduler:
  type: PolynomialDecay
  learning_rate: 0.02
  power: 0.9
  end_lr: 0
  warmup_iters: 100
  warmup_start_lr: 1.0e-4

# Runtime (torchtrain reads these from the YAML or the CLI; the CLI wins)
use_gpu: true
use_amp: true
num_workers: 2
save_interval: 200
log_iters: 20
`,
  },
  {
    name: 'torchseg_20k_512',
    content: `# TorchSeg preset - 20k iters @ 512x512
# Baseline schedule for a few dozen to a few hundred training images.
# Note: \`loss:\` lives in the model YAML (see the quick preset above).

batch_size: 4
iters: 20000

optimizer:
  type: Momentum
  momentum: 0.9
  weight_decay: 5.0e-4

lr_scheduler:
  type: PolynomialDecay
  learning_rate: 0.01
  power: 0.9
  end_lr: 0
  warmup_iters: 500
  warmup_start_lr: 1.0e-5

use_gpu: true
use_amp: true
num_workers: 4
save_interval: 1000
log_iters: 20
`,
  },
];

// ---------------------------------------------------------------------------
// TorchDet (PyTorch object detection)
// ---------------------------------------------------------------------------

// PaddleDetection schema, torchvision architectures. There is no neck/head to
// wire: torchvision builds the whole detector from architecture + backbone.
const TORCHDET_MODEL_CONFIGS: BuiltinConfig[] = [
  {
    name: 'faster_rcnn_resnet50_fpn',
    content: `# Faster R-CNN with ResNet50-FPN (torchvision)
# Two-stage detector; the strongest default for the small, high-resolution
# datasets this platform is used with.
# pretrain_weights: COCO loads the COCO-pretrained detector and replaces its
# classifier to match num_classes (the dataset config supplies num_classes).

architecture: FasterRCNN

FasterRCNN:
  backbone: ResNet50-FPN

pretrain_weights: COCO
`,
  },
  {
    name: 'faster_rcnn_resnet50_fpn_v2',
    content: `# Faster R-CNN with the improved ResNet50-FPN v2 recipe (torchvision)
# Better accuracy than v1 at slightly higher cost.

architecture: FasterRCNN

FasterRCNN:
  backbone: ResNet50-FPN-v2

pretrain_weights: COCO
`,
  },
  {
    name: 'retinanet_resnet50_fpn',
    content: `# RetinaNet with ResNet50-FPN (torchvision)
# One-stage detector with focal loss; handles heavy class imbalance well.

architecture: RetinaNet

RetinaNet:
  backbone: ResNet50-FPN

pretrain_weights: COCO
`,
  },
  {
    name: 'fcos_resnet50_fpn',
    content: `# FCOS with ResNet50-FPN (torchvision)
# Anchor-free one-stage detector; no anchor tuning needed, which helps when
# object aspect ratios vary a lot.

architecture: FCOS

FCOS:
  backbone: ResNet50-FPN

pretrain_weights: COCO
`,
  },
  {
    name: 'ssdlite_mobilenetv3',
    content: `# SSDLite with MobileNetV3-Large (torchvision)
# Fastest option, lowest accuracy. Note that head replacement is not supported
# for SSD, so COCO weights degrade to ImageNet backbone weights.

architecture: SSD

SSD:
  backbone: MobileNetV3-Large

pretrain_weights: ImageNet
`,
  },
];

const TORCHDET_TRAINING_CONFIGS: BuiltinConfig[] = [
  {
    name: 'torchdet_12e_quick',
    content: `# TorchDet preset - 12 epochs (quick baseline)
# The classic COCO "1x" schedule, shortened. Batch size 2 fits a 24 GB card at
# 1333px; raise it (and the LR proportionally) on smaller images.
#
# NormalizeImage / Permute / PadGT below describe work torchvision performs
# inside the model (GeneralizedRCNNTransform); torchtrain accepts and ignores
# them. Resize / BatchRandomResize target sizes become min_size / max_size.

epoch: 12

LearningRate:
  base_lr: 0.005
  schedulers:
  - !CosineDecay
    max_epochs: 12
  - !LinearWarmup
    start_factor: 0.001
    epochs: 1

OptimizerBuilder:
  clip_grad_by_norm: 35
  optimizer:
    type: Momentum
    momentum: 0.9
  regularizer:
    type: L2
    factor: 0.0001

worker_num: 2

TrainReader:
  sample_transforms:
    - Decode: {}
    - RandomDistort: {}
    - RandomFlip: {}
  batch_transforms:
    - BatchRandomResize: {target_size: [640, 800, 1024, 1333], random_size: True, keep_ratio: False}
    - NormalizeImage: {mean: [0, 0, 0], std: [1, 1, 1], norm_type: none}
    - Permute: {}
    - PadGT: {}
  batch_size: 2
  shuffle: true
  drop_last: true

EvalReader:
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: [1333, 1333], keep_ratio: False, interp: 2}
    - NormalizeImage: {mean: [0, 0, 0], std: [1, 1, 1], norm_type: none}
    - Permute: {}
  batch_size: 1

use_gpu: true
use_amp: false
log_iter: 20
snapshot_epoch: 1
`,
  },
  {
    name: 'torchdet_36e',
    content: `# TorchDet preset - 36 epochs (COCO "3x" style)
# Use once the 12-epoch baseline shows the pipeline works and you want the
# extra accuracy. Same reader configuration as the quick preset.

epoch: 36

LearningRate:
  base_lr: 0.01
  schedulers:
  - !CosineDecay
    max_epochs: 36
  - !LinearWarmup
    start_factor: 0.001
    epochs: 1

OptimizerBuilder:
  clip_grad_by_norm: 35
  optimizer:
    type: Momentum
    momentum: 0.9
  regularizer:
    type: L2
    factor: 0.0001

worker_num: 4

TrainReader:
  sample_transforms:
    - Decode: {}
    - RandomDistort: {}
    - RandomExpand: {ratio: 1.5}
    - RandomFlip: {}
  batch_transforms:
    - BatchRandomResize: {target_size: [640, 800, 1024, 1333], random_size: True, keep_ratio: False}
    - NormalizeImage: {mean: [0, 0, 0], std: [1, 1, 1], norm_type: none}
    - Permute: {}
    - PadGT: {}
  batch_size: 2
  shuffle: true
  drop_last: true

EvalReader:
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: [1333, 1333], keep_ratio: False, interp: 2}
    - NormalizeImage: {mean: [0, 0, 0], std: [1, 1, 1], norm_type: none}
    - Permute: {}
  batch_size: 1

use_gpu: true
use_amp: true
log_iter: 20
snapshot_epoch: 3
`,
  },
];

const BUILTIN_MODELS: Record<string, BuiltinConfig[]> = {
  PaddleSeg: PADDLESEG_MODEL_CONFIGS,
  TorchSeg: TORCHSEG_MODEL_CONFIGS,
  TorchDet: TORCHDET_MODEL_CONFIGS,
};

const BUILTIN_TRAINING: Record<string, BuiltinConfig[]> = {
  PaddleSeg: PADDLESEG_TRAINING_CONFIGS,
  TorchSeg: TORCHSEG_TRAINING_CONFIGS,
  TorchDet: TORCHDET_TRAINING_CONFIGS,
};

/** Return built-in model starters for the given framework (empty if none). */
export function getBuiltinModelConfigs(framework: string): BuiltinConfig[] {
  return BUILTIN_MODELS[framework] ?? [];
}

/** Return built-in training starters for the given framework (empty if none). */
export function getBuiltinTrainingConfigs(framework: string): BuiltinConfig[] {
  return BUILTIN_TRAINING[framework] ?? [];
}

/**
 * Ensure the on-disk default folder for `framework`/`subdir` exists and is
 * seeded with the built-in starter configs. Existing files are never
 * overwritten so user edits are preserved.
 *
 * Returns the resolved absolute directory (so callers can immediately read it).
 */
export function ensureDefaultConfigs(
  userConfigsPath: string,
  framework: string,
  subdir: 'models' | 'training',
): string {
  const dir = getDefaultConfigDir(userConfigsPath, framework, subdir);
  try {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    const builtins =
      subdir === 'models'
        ? getBuiltinModelConfigs(framework)
        : getBuiltinTrainingConfigs(framework);
    for (const cfg of builtins) {
      const filePath = path.join(dir, `${cfg.name}.yml`);
      if (!fs.existsSync(filePath)) {
        fs.writeFileSync(filePath, cfg.content, 'utf-8');
      }
    }
  } catch (err) {
    // Seeding is best-effort; surface via console but don't block the request.
    console.warn('[default-configs] Failed to seed', dir, err);
  }
  return dir;
}
