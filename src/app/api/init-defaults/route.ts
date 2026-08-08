import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { requireAdmin, requireAuth } from "@/lib/auth";

// Default model YAML configurations
const DEFAULT_MODEL_YAML = `# PP-YOLOE+ Model Configuration
# Architecture: PP-YOLOE+
# For object detection tasks

architecture: YOLOv3
backbone: CSPResNet
neck: CustomCSPPAN
head: PPYOLOEHead

# Model dimensions
depth_mult: 0.33
width_mult: 0.50

# Normalization
norm_type: sync_bn

# EMA (Exponential Moving Average)
use_ema: true
ema_decay: 0.9998

# Number of classes (adjust based on your dataset)
num_classes: 1

# Pretrained weights (optional)
# pretrain_weights: https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams
`;

// Default training config YAML
const DEFAULT_TRAINING_YAML = `# Training Configuration
# Default settings for PaddleDetection training

# Training epochs
epoch: 100

# Batch size (adjust based on GPU memory)
batch_size: 8

# Learning rate
base_lr: 0.001

# Optimizer settings
momentum: 0.9
weight_decay: 0.0005

# Learning rate scheduler
scheduler: CosineDecay
warmup_epochs: 5
max_epochs: 100

# Data loader settings
worker_num: 4

# Evaluation settings
eval_height: 640
eval_width: 640

# Runtime settings
use_gpu: true
log_iter: 20
snapshot_epoch: 1

# Output directory
# output_dir: output/
`;

// Default dataset YAML
const DEFAULT_DATASET_YAML = `# Dataset Configuration
# COCO format dataset

metric: COCO
num_classes: 1

TrainDataset:
  name: COCODataSet
  image_dir: images/train
  anno_path: annotations/train.json
  dataset_dir: dataset/default
  data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  name: COCODataSet
  image_dir: images/val
  anno_path: annotations/val.json
  dataset_dir: dataset/default
  allow_empty: true

TestDataset:
  name: ImageFolder
  anno_path: annotations/val.json
  dataset_dir: dataset/default
`;

/**
 * POST /api/init-defaults
 * Seed a sample project/dataset/model/training-config for a fresh install.
 *
 * Admin-only. This used to be unauthenticated, which let anyone write rows into
 * the database. Worse, the rows were created with `userId: null`, and
 * `buildUserFilter` deliberately treats ownerless rows as visible to *every*
 * user — so an anonymous request could inject a project that shows up in all
 * accounts. Everything created here is now owned by the calling admin.
 */
export async function POST(request: NextRequest) {
  try {
    const auth = await requireAdmin(request);
    if (auth instanceof NextResponse) return auth;
    const { userId } = auth;

    const results = {
      project: null as unknown,
      dataset: null as unknown,
      model: null as unknown,
      trainingConfig: null as unknown,
      systemConfig: null as unknown,
    };

    // 1. Ensure a SystemConfig row exists.
    //    Framework paths are intentionally left empty: they are machine-specific
    //    and must be set in Settings. The previous code defaulted
    //    paddleDetectionPath to a hardcoded Linux path from the original
    //    scaffold ("/home/z/PaddleDetection"), which is wrong everywhere.
    let systemConfig = await db.systemConfig.findFirst();
    if (!systemConfig) {
      systemConfig = await db.systemConfig.create({
        data: {
          condaEnv: "",
          condaPath: "",
          paddleDetectionPath: "",
          paddleClasPath: "",
          paddleSegPath: "",
          defaultFramework: "PaddleDetection",
        },
      });
    }
    results.systemConfig = systemConfig;

    // 2. Create default project if not exists
    let project = await db.project.findFirst({
      where: { name: "Default Project", userId },
    });
    if (!project) {
      project = await db.project.create({
        data: {
          name: "Default Project",
          description: "Sample project for PaddleDetection",
          framework: "PaddleDetection",
          task: "detection",
          status: "active",
          userId,
        },
      });
    }
    results.project = project;

    // 3. Create default dataset if not exists
    let dataset = await db.dataset.findFirst({
      where: { name: "Default Dataset", projectId: project.id },
    });
    if (!dataset) {
      dataset = await db.dataset.create({
        data: {
          name: "Default Dataset",
          description: "Sample COCO-format dataset",
          projectId: project.id,
          userId,
          format: "COCO",
          numClasses: 1,
          numAnnotations: 0,
          numTrainImages: 0,
          numEvalImages: 0,
          datasetDir: "dataset/default",
          yamlConfig: DEFAULT_DATASET_YAML,
        },
      });
    }
    results.dataset = dataset;

    // 4. Create default model if not exists
    let model = await db.model.findFirst({
      where: { name: "PP-YOLOE+ Small", projectId: project.id },
    });
    if (!model) {
      model = await db.model.create({
        data: {
          name: "PP-YOLOE+ Small",
          description: "PP-YOLOE+ small model for object detection (sample config)",
          projectId: project.id,
          userId,
          architecture: "YOLOv3",
          backbone: "CSPResNet",
          neck: "CustomCSPPAN",
          head: "PPYOLOEHead",
          numClasses: 1,
          normType: "sync_bn",
          useEma: true,
          emaDecay: 0.9998,
          depthMult: 0.33,
          widthMult: 0.5,
          yamlConfig: DEFAULT_MODEL_YAML,
        },
      });
    }
    results.model = model;

    // 5. Create default training config if not exists.
    //    `projectId` is required by the schema; omitting it (as the previous
    //    code did) makes this call throw, so POST could never succeed.
    let trainingConfig = await db.trainingConfig.findFirst({
      where: { name: "Default Training Config", projectId: project.id },
    });
    if (!trainingConfig) {
      trainingConfig = await db.trainingConfig.create({
        data: {
          name: "Default Training Config",
          projectId: project.id,
          userId,
          epoch: 100,
          batchSize: 8,
          baseLr: 0.001,
          momentum: 0.9,
          weightDecay: 0.0005,
          scheduler: "CosineDecay",
          warmupEpochs: 5,
          maxEpochs: 100,
          workerNum: 4,
          evalHeight: 640,
          evalWidth: 640,
          useGpu: true,
          logIter: 20,
          snapshotEpoch: 1,
          outputDir: "output/default",
          yamlConfig: DEFAULT_TRAINING_YAML,
        },
      });
    }
    results.trainingConfig = trainingConfig;

    return NextResponse.json({
      success: true,
      message: "Default sample data initialized successfully",
      data: results,
    });
  } catch (error) {
    console.error("Error initializing default data:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to initialize default data",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}

/**
 * GET /api/init-defaults
 * Check if default data exists
 */
export async function GET(request: NextRequest) {
  try {
    const auth = await requireAuth(request);
    if (auth instanceof NextResponse) return auth;
    const { userId } = auth;

    const [project, dataset, model, trainingConfig, systemConfig] = await Promise.all([
      db.project.findFirst({ where: { name: "Default Project", userId } }),
      db.dataset.findFirst({ where: { name: "Default Dataset", userId } }),
      db.model.findFirst({ where: { name: "PP-YOLOE+ Small", userId } }),
      db.trainingConfig.findFirst({ where: { name: "Default Training Config", userId } }),
      db.systemConfig.findFirst(),
    ]);

    return NextResponse.json({
      success: true,
      data: {
        hasProject: !!project,
        hasDataset: !!dataset,
        hasModel: !!model,
        hasTrainingConfig: !!trainingConfig,
        hasPaddleDetectionPath: !!(systemConfig?.paddleDetectionPath),
        paddleDetectionPath: systemConfig?.paddleDetectionPath || null,
      },
    });
  } catch (error) {
    console.error("Error checking default data:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to check default data",
      },
      { status: 500 }
    );
  }
}
