import { NextResponse } from "next/server";
import { db } from "@/lib/db";

// Default PaddleDetection path
const DEFAULT_PADDLE_DETECTION_PATH = "/home/z/PaddleDetection";

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
 * Initialize default test data including project, dataset, model, and training config
 */
export async function POST() {
  try {
    const results = {
      project: null as unknown,
      dataset: null as unknown,
      model: null as unknown,
      trainingConfig: null as unknown,
      systemConfig: null as unknown,
    };

    // 1. Create or update system config with default PaddleDetection path
    let systemConfig = await db.systemConfig.findFirst();
    if (!systemConfig) {
      systemConfig = await db.systemConfig.create({
        data: {
          pythonPath: "python",
          condaEnv: "",
          condaPath: "",
          paddleDetectionPath: DEFAULT_PADDLE_DETECTION_PATH,
          paddleClasPath: "",
          defaultGpu: 0,
          defaultFramework: "PaddleDetection",
        },
      });
    } else if (!systemConfig.paddleDetectionPath) {
      systemConfig = await db.systemConfig.update({
        where: { id: systemConfig.id },
        data: {
          paddleDetectionPath: DEFAULT_PADDLE_DETECTION_PATH,
        },
      });
    }
    results.systemConfig = systemConfig;

    // 2. Create default project if not exists
    let project = await db.project.findFirst({
      where: { name: "Default Project" },
    });
    if (!project) {
      project = await db.project.create({
        data: {
          name: "Default Project",
          description: "Default test project for PaddleDetection",
          framework: "PaddleDetection",
          status: "active",
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
          description: "Default test dataset with COCO format",
          projectId: project.id,
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
          description: "PP-YOLOE+ small model for object detection (default test config)",
          projectId: project.id,
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

    // 5. Create default training config if not exists
    let trainingConfig = await db.trainingConfig.findFirst({
      where: { name: "Default Training Config" },
    });
    if (!trainingConfig) {
      trainingConfig = await db.trainingConfig.create({
        data: {
          name: "Default Training Config",
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
      message: "Default test data initialized successfully",
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
export async function GET() {
  try {
    const [project, dataset, model, trainingConfig, systemConfig] = await Promise.all([
      db.project.findFirst({ where: { name: "Default Project" } }),
      db.dataset.findFirst({ where: { name: "Default Dataset" } }),
      db.model.findFirst({ where: { name: "PP-YOLOE+ Small" } }),
      db.trainingConfig.findFirst({ where: { name: "Default Training Config" } }),
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
