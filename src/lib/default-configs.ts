import * as fs from 'fs';
import * as path from 'path';

/**
 * Central definition of "default config folder" naming and built-in starter
 * configs served on the Load Model / Load Training Config pages.
 *
 * On disk we keep each framework's defaults in a separate folder under
 * `{userConfigsPath}` so users can browse them independently:
 *   - `default/`     — PaddleDetection (object detection)
 *   - `defaultSeg/`  — PaddleSeg (segmentation)
 * Each folder mirrors the same substructure: `{models, training, datasets, jobs}`.
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

/** Return built-in model starters for the given framework (empty if none). */
export function getBuiltinModelConfigs(framework: string): BuiltinConfig[] {
  if (framework === 'PaddleSeg') return PADDLESEG_MODEL_CONFIGS;
  return [];
}

/** Return built-in training starters for the given framework (empty if none). */
export function getBuiltinTrainingConfigs(framework: string): BuiltinConfig[] {
  if (framework === 'PaddleSeg') return PADDLESEG_TRAINING_CONFIGS;
  return [];
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
