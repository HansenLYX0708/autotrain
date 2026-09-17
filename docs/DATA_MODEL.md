# 数据模型参考

数据源：[`prisma/schema.prisma`](../prisma/schema.prisma)，SQLite。

阅读前请先记住两条全局约定，否则很容易读错字段：

1. **`yamlConfig` 才是事实源。** `TrainingConfig` / `Model` / `Dataset` 上的扁平标量列是**展示缓存**，由 `src/lib/training-yaml.ts` / `src/lib/model-yaml.ts` 从 YAML 推导而来，绝不用于反向重建配置。
2. **以 epoch 命名的列存的是「该框架的原生训练单位」。** 对 PaddleSeg / TorchSeg / TorchAnomaly，它们存的是迭代数或 step 数。判断单位请调用 `countsIterations(framework)` / `tracksIterations(framework)`。

> Prisma migrations 停在 `20260403`，而 schema 远比它新。本项目使用 `prisma db push`，因此 `migrate deploy` **无法**复现当前结构。改 schema 后记得 `npm run db:generate`，否则代码要靠 `as any` 才能访问新列。

---

## 实体关系

```
User ─┬─< Project ─┬─< Dataset ──────┐
      │            ├─< Model ────────┤
      │            ├─< TrainingConfig┤
      │            ├─< TrainingJob >─┘   (job 引用 dataset/model/config)
      │            ├─< ValidationJob
      │            └─< DeployedModel
      ├─< ActivityLog
      └─< (以上各表的 userId，均为可空 + onDelete: SetNull)

TrainingJob ─┬─< TrainingLog        (onDelete: Cascade)
             └─< DeployedModel      (onDelete: SetNull)

SystemConfig  (单行全局配置)
GpuMetric     (无外键；当前无写入者)
```

删除语义：

- `Project` 删除 → 其下 `Dataset` / `Model` / `TrainingConfig` / `TrainingJob` / `ValidationJob` **级联删除**；`DeployedModel` 仅置空 `projectId`。
- `TrainingJob` 删除 → `TrainingLog` 级联删除。
- `User` 删除 → 业务数据**不删**，只把 `userId` 置为 null。注意 `buildUserFilter` 让 `userId = null` 的行对**所有用户可见**，因此删除用户会把其数据变成公共数据。`ActivityLog` 是唯一级联删除的例外。

---

## User

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | String @id cuid | |
| `username` | String @unique | `default` 为保留名，注册时被拒 |
| `password` | String | **无盐 SHA-256** 十六进制 |
| `role` | String | `admin` \| `user`，默认 `user`。**第一个注册的用户自动为 `admin`** |
| `status` | String | `active` \| `disabled`。非 `active` 时会话立即失效 |
| `maxStorageQuota` | BigInt | 存储配额（字节），默认 `214748364800`（200 GB） |
| `createdAt` / `updatedAt` | DateTime | |

配额用量不是存出来的，而是每次请求递归统计 `{userDatabasePath}/{username}` 目录大小得到。JSON 序列化时 `maxStorageQuota` 会转成字符串（BigInt 不能直接序列化）。

---

## SystemConfig

全局单行配置，由 `GET/PUT /api/settings` 读写（**admin only**）。不存在时 `GET` 会创建一条默认记录。

| 字段 | 说明 |
|---|---|
| `condaEnv` | conda 环境名。留空时会尝试从解释器路径自动识别，但只有真的找到 `conda` 可执行文件才启用 |
| `condaPath` | `conda` 可执行文件路径 |
| `pythonEnvsBasePath` | 多 Python 环境的根目录（仅供 UI 浏览用） |
| `gpuPythonMappings` | JSON 字符串，`{"0": "C:/envs/gpu0/python.exe"}`。**历史遗留**的按 GPU 映射，通常指向 Paddle 环境 |
| `frameworkPythonMappings` | JSON 字符串，`{"TorchSeg": "D:/pythonEnvs/torchEnv/Scripts/python.exe"}`。键可为框架名或 `"<framework>:<gpuId>"`。**优先级高于** `gpuPythonMappings` |
| `userConfigsPath` | 用户 YAML 配置根目录，布局 `{root}/{username}/{datasets,models,training,jobs}/` |
| `userDatabasePath` | 用户数据与训练产物根目录，布局 `{root}/{username}/jobs/{jobSlug}/` |
| `paddleDetectionPath` | PaddleDetection 仓库根 |
| `paddleClasPath` | PaddleClas 仓库根 |
| `paddleSegPath` | PaddleSeg 仓库根 |
| `torchPath` | 内置 `torchtrain/` 仓库根。`TorchDet`/`TorchSeg`/`TorchAnomaly` **共用**此路径，但不共用 Python 环境 |
| `defaultFramework` | 新建项目的默认框架，默认 `PaddleDetection` |

哪个框架读哪一列由 `FRAMEWORK_META[...].pathField` 决定，代码里应通过 `getWorkDir(framework, systemConfig)` 获取，不要硬编码。

---

## Project

| 字段 | 说明 |
|---|---|
| `name` / `description` | `name` 会被 slug 化后用作输出目录的路径段 |
| `framework` | 六个框架之一，默认 `PaddleDetection`。**这是整个任务行为的总开关** |
| `task` | `detection` \| `instance_segmentation` \| `classification` \| `semantic_segmentation` \| `anomaly_detection`。由 `resolveTask(framework, requestedTask)` 确定：检测框架额外支持 `instance_segmentation`，其余框架只有一个合法 task，切换框架时过期的 task 不会被保留 |
| `status` | `active` \| `archived`。Dashboard 只统计 `active` |
| `userId` | 可空外键 |

---

## Dataset

### 公共字段

| 字段 | 说明 |
|---|---|
| `format` | `COCO` \| `PaddleSeg` \| `AnomalyFolder`，默认 `COCO`。应与框架的 `datasetFormat` 匹配 |
| `datasetDir` | 数据集根目录。相对路径按 `{userDatabasePath}/{username}` 解析 |
| `yamlConfig` | 数据集配置 YAML，参与任务合并的**第一段** |

### COCO / PaddleSeg 路径

| 字段 | COCO 含义 | PaddleSeg 含义 |
|---|---|---|
| `trainImagePath` | 训练图片目录 | `JPEGImages` 等图像根 |
| `trainAnnoPath` | 训练标注 JSON | `train.txt` 列表文件 |
| `evalImagePath` | 验证图片目录 | 图像根 |
| `evalAnnoPath` | 验证标注 JSON | `val.txt` 列表文件 |

### AnomalyFolder 目录（无监督异常检测）

均为**相对 `datasetDir`** 的目录名，映射到 anomalib 的 `Folder` datamodule。

| 字段 | 必填 | 说明 |
|---|---|---|
| `normalDir` | ✅ | 正常样本。**唯一被训练的目录** |
| `normalTestDir` | | 测试用正常样本 |
| `abnormalDir` | | 缺陷样本，进入验证/测试划分 |
| `maskDir` | | 缺陷掩码，有它才会有 pixel 级指标 |

anomalib 会取 test 划分的一半作验证，并**在其上拟合异常分数阈值**，所以训练运行报出的指标是乐观的。详见 [ANOMALY_DETECTION_DESIGN.md](ANOMALY_DETECTION_DESIGN.md)。

### 统计（由 `/api/datasets/parse` 写入）

`numClasses`、`numAnnotations`、`numTrainImages`、`numEvalImages`、`classStats`（JSON 字符串，逐类别统计）。

---

## Model

模型**配置**记录，不是权重文件。

| 字段 | 说明 |
|---|---|
| `architecture` / `backbone` / `neck` / `head` | 展示缓存。合法组合编码在 `src/lib/model-yaml.ts` 的 `DETECTION_PRESETS` / `SEG_ARCHITECTURES` / `TORCH_SEG_ARCHITECTURES` / `TORCH_DET_PRESETS` / `ANOMALY_PRESETS` / `CLAS_ARCHITECTURES` |
| `numClasses` | 注意语义差异：PaddleDetection 与 TorchDet 配置**只算前景类**，torchvision 侧的 `+1`（含背景）由 `torchtrain/det/dataset.py` 完成 |
| `normType` | 如 `sync_bn` |
| `useEma` / `emaDecay` | 指数滑动平均 |
| `depthMult` / `widthMult` | 缩放系数（YOLO 系） |
| `pretrainWeights` | 预训练权重路径或 URL |
| `yamlConfig` | 模型配置 YAML，参与任务合并的**第三段（优先级最高）** |

分割任务的 `loss:` **属于模型配置**：`len(loss.types)` 必须等于架构输出的 logits 数（主头 + 辅助头），数量记在 `SEG_ARCHITECTURES[].logits`，否则 PaddleSeg 会抛 `RuntimeError: The length of logits_list should equal to the types of loss config`。

同理，`TorchAnomaly` 的学习率、输入尺寸与归一化都在模型配置里——anomalib 里这些是模型构造参数，而模型配置最后合并，写在训练配置里会被静默覆盖。

---

## TrainingConfig

| 字段 | 默认 | epoch 类框架 | iter 类框架（Seg / Anomaly） |
|---|---|---|---|
| `epoch` | 100 | epochs | **iters / max_steps** |
| `maxEpochs` | 100 | 衰减调度器的 `max_epochs` | **iters** |
| `warmupEpochs` | 5 | warmup epochs | **warmup_iters** |
| `snapshotEpoch` | 1 | 保存间隔（epoch） | **save_interval / 验证间隔** |
| `iters` | null | — | 迭代数（分割专用列） |
| `saveInterval` | null | — | 保存间隔（分割专用列） |

其余字段：`batchSize`、`baseLr`、`momentum`、`weightDecay`、`scheduler`、`workerNum`、`evalHeight`、`evalWidth`、`useGpu`、`logIter`、`saveDir`、`outputDir`、`weights`、`pretrainWeights`。

`yamlConfig` 是事实源，参与任务合并的**第二段**。表单实际可编辑的字段由 `TRAINING_FIELD_SUPPORT[framework]` 决定，完整参数模型是 `TrainingParams`（见 `src/lib/training-yaml.ts`），比这些列丰富得多——列只保存了最常用于列表展示的那几个。

---

## TrainingJob

| 字段 | 说明 |
|---|---|
| `projectId` / `datasetId` / `modelId` | 必填外键，级联删除 |
| `configId` | 可空；训练配置被删时置 null |
| `name` | 原始名称。落到文件系统时一律用 `toSafeSlug(name)` 的结果（slug 存在 `trainingParams.jobSlug`） |
| `status` | `pending` \| `running` \| `completed` \| `failed` \| `stopped` |
| `command` | 服务端生成的训练命令。**API 绝不接受请求体里的该字段**——它会被 `shell: true` 执行 |
| `evalCommand` / `inferCommand` | 预生成的评估/推理命令，指向 `bestWeightsPath()` |
| `configPath` | 合并后配置的路径。**可能是相对 `userConfigsPath` 的路径**，接口返回时会额外附带 `absoluteConfigPath` |
| `errorMessage` | 失败原因（含最后 10 行 stderr）。成功或正常停止时被清空 |
| `currentEpoch` / `totalEpochs` | 进度。**iter 类框架两者都存迭代数**；`totalEpochs` 由 `totalStepsFor()` 从合并后的 YAML 算出 |
| `currentLoss` / `currentLr` | 最近一条训练行的快照 |
| `bestIter` / `bestMetric` | 来自 "best mIoU (X) was saved at iter N" 一类的日志行。非分割框架为 null |
| `outputDir` | 命令实际使用的**绝对** `save_dir`，`/api/checkpoints` 依赖它定位检查点 |
| `weightsPath` / `vdlLogDir` | 权重路径 / VisualDL 日志目录（仅 `useVdl` 时） |
| `trainingParams` | JSON 字符串：`gpuIds`、`useAmp`、`useVdl`、`jobSlug`、`framework`、`totalSteps`、`epochs` 或 `iters`、`batchSize`、`baseLr` |
| `yamlConfig` | 本次任务实际使用的合并后 YAML（快照） |
| `startedAt` / `completedAt` | |

---

## TrainingLog

一条训练日志行或一次评估结果。由 `src/lib/log-parsers` 解析、`writeParsedLog()` 写入。字段为空应理解为「该框架不适用」，而非「数据丢失」。

**进度**：`epoch`、`iteration`、`totalIter`

**通用**：`loss`、`learningRate`、`eta`、`batchCost`、`dataCost`、`readerCost`（PaddleSeg 的 `reader_cost`）、`ips`、`memReserved`（MB）、`memAllocated`（MB）

**检测 loss 分解**：`lossCls`、`lossIou`、`lossDfl`、`lossL1`

**检测指标**：`mAP`（AP@[0.50:0.95]）、`mAP50`（AP@0.50），解析自 COCO 统计块

**分割指标**：`mIoU`、`acc`、`kappa`、`dice`

**异常检测指标**：`imageAuroc`、`imageF1`（图像级：这个件是不是坏的）、`pixelAuroc`、`pixelF1`（像素级：坏在哪里）、`threshold`（anomalib 自适应阈值选出的分数切点）

**`classMetrics`**：JSON 字符串，承载两类内容——

- 分割逐类别数组：`{"iou":[...],"precision":[...],"recall":[...]}`。用 JSON 而不是建列，是为了让 `num_classes` 变化不影响表结构。
- 没有专属列的异常检测指标：`{"metrics": {...}}`。anomalib 的 `Evaluator` 可以挂任意 `AnomalibMetric`，硬编码字段集会静默丢数据。

**`rawLog`**：原始日志行，截断到 2000 字符。

---

## ValidationJob

| 字段 | 说明 |
|---|---|
| `projectId` | 必填，级联删除 |
| `trainingJobId` | 可空的溯源链接（**注意：schema 里没有真正的外键关系**，只是存 id） |
| `type` | `eval` \| `infer` |
| `configPath` / `weightsPath` | 评估/推理使用的配置与权重 |
| `datasetPath` | 评估用数据集路径 |
| `inferInputPath` / `inferOutputPath` | 推理输入（单图或目录）与输出目录 |
| `status` | `pending` \| `running` \| `completed` \| `failed` |
| `command` | 生成的命令（实际执行时按 argv 数组重组） |
| `resultJson` | 指标 JSON（mAP / mIoU / AUROC 等） |
| `resultPath` | 结果文件或目录 |
| `outputLog` | 捕获的 stdout/stderr。**本表没有 `errorMessage` 列，失败原因也写在这里** |

---

## DeployedModel

把训练产物提升为可按产品切换的「生产模型」。

| 字段 | 说明 |
|---|---|
| `name` | 如 `line13-Aug-v3` |
| `product` | 产品标签，**切换的主键** |
| `framework` / `architecture` | 后者仅作展示 |
| `projectId` / `trainingJobId` | 可空溯源，删除时置 null |
| `configPath` / `weightsPath` | 绝对路径，推理/评估直接使用 |
| `exportedDir` | 导出目录（如 `export_model/<name>`），可空 |
| `metrics` | JSON：`{ mAP, mIoU, bestIter, ... }` |
| `notes` | 备注 |
| `isActive` | 同一 `product` **同时只应有一条为 true**，由应用层事务保证（`POST /api/deployed-models/[id]/activate`） |

索引：`@@index([product, framework])`、`@@index([userId, product])`、`@@unique([userId, product, name])`。

---

## ActivityLog

| 字段 | 说明 |
|---|---|
| `userId` | **级联删除**（唯一如此的表） |
| `action` | `create_project`、`import_dataset`、`import_model`、`import_config`、`create_job`、`start_training`、`stop_training`、`delete_job` … |
| `entityType` / `entityId` / `entityName` | 关联实体，`entityName` 供展示用 |
| `details` | 附加信息 JSON |

索引：`@@index([userId, createdAt])`。写入统一走 `src/lib/activity-log.ts` 的 `logActivity()`。

---

## GpuMetric

`gpuId`、`utilization`、`memoryUsed`（MB）、`memoryTotal`（MB）、`temperature`（℃）、`timestamp`。

用于 GPU 历史曲线，但**当前没有任何代码定时写入这张表**，Dashboard 读到的通常是空集；实时数值来自 `GET /api/system/gpu`（现场调用 `nvidia-smi`）。
