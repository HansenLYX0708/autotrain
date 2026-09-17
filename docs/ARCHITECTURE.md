# 架构与实现原理

本文描述 Hawkeye+ AutoTrain 平台**当前代码的实际行为**。术语、字段名、文件路径均与源码一致。

- 功能操作说明见 [FEATURES.md](FEATURES.md)
- 接口清单见 [API_REFERENCE.md](API_REFERENCE.md)
- 表结构见 [DATA_MODEL.md](DATA_MODEL.md)
- 开发时的坑与验证流程见 [../AGENTS.md](../AGENTS.md)

---

## 1. 总体结构

```
┌──────────────────────────── 浏览器 ────────────────────────────┐
│ src/app/page.tsx  (单路由 SPA)                                 │
│   侧边栏在 src/components/pages/*.tsx 之间切换                  │
│   fetch('/api/...') + 定时轮询 (Monitoring 5s / Dashboard 30s)  │
└───────────────────────────────┬───────────────────────────────┘
                                │ HTTP + Cookie(auth-token)
┌───────────────────────────────▼───────────────────────────────┐
│ Next.js 16 服务端 (App Router, Node runtime)                   │
│                                                                │
│  src/app/api/**/route.ts  ── 鉴权 ── Prisma ── 文件系统         │
│         │                                                      │
│         ├─ src/lib/frameworks.ts    框架注册表（唯一事实源）      │
│         ├─ src/lib/training-yaml.ts 训练配置 ⇄ YAML             │
│         ├─ src/lib/model-yaml.ts    模型配置 ⇄ YAML             │
│         ├─ src/lib/yaml-merge.ts    三段配置 AST 深度合并        │
│         ├─ src/lib/job-commands.ts  命令字符串（UI 预览同源）     │
│         └─ src/lib/log-parsers/*    stdout → ParsedTrainLog     │
│                                                                │
│  child_process.spawn(shell:true) ──► python tools/train.py      │
└───────────────────────────────┬───────────────────────────────┘
                                │
                   ┌────────────┴────────────┐
                   ▼                         ▼
          SQLite (Prisma)            用户数据目录 / 框架仓库
```

几个关键的架构事实：

- **单进程、单实例**。训练子进程句柄（`runningProcesses`）和登录会话（`sessions`）都只存在于 Node 进程内存里，没有外部队列或缓存。服务重启会丢失会话，并且**正在跑的训练进程会变成孤儿**——它继续训练，但平台再也停不掉它。
- **无 WebSocket**。实时性靠前端轮询实现。`src/middleware.ts` 只是把 `/socket.io/*` 请求拦下来返回 200，避免开发日志被 404 刷屏；`mini-services/training-service/` 是一个**未接入的 Socket.io 原型（生成合成日志）**，`examples/websocket/` 同理。真实训练在 `src/app/api/training-jobs/[id]/route.ts`。
- **前端有一处死代码**：`src/components/app-sidebar.tsx` 用 `next/link` 指向 `/projects`、`/datasets` 等路径，而这些路由并不存在（应用只有 `/`）。实际侧边栏在 `page.tsx` 内联实现。

---

## 2. 框架抽象层

`src/lib/frameworks.ts` 是**唯一事实源**。新增一个框架，原则上只需要在这里加一条 `FRAMEWORK_META`，再补上它的 YAML 生成器。

`FrameworkMeta` 声明的内容：

| 字段 | 作用 |
|---|---|
| `family` | `paddle` / `torch`，决定启动器与 Python 环境 |
| `taskKind` | `detection` / `classification` / `segmentation` / `anomaly`，决定日志解析器与指标 |
| `defaultTask` | `project.task` 的默认值 |
| `pythonModule` / `installHint` | 启动前预检要 import 的模块，及失败时的提示 |
| `pathField` | 该框架仓库路径存在 `SystemConfig` 的哪一列 |
| `requiredFiles` | 仓库目录下必须存在的标志文件 |
| `scripts` | `train` / `eval` / `infer` / `export` 脚本相对路径 |
| `cliStyle` | `dash-c`（`-c cfg -o weights=w`）或 `config-flags`（`--config cfg --model_path w`） |
| `saveDirOnCli` | `save_dir` 是 CLI 参数还是 YAML 键 |
| `checkpointLayout` / `weightFile` | `flat`（`<save_dir>/model_final.pdparams`）或 `nested`（`<save_dir>/best_model/<weightFile>`） |
| `datasetFormat` | `COCO` / `PaddleSeg` / `AnomalyFolder` |
| `stepUnit` | `epoch` / `iter`，决定进度条单位 |

### 必须用谓词，不要比较字符串

```ts
isSegmentation(f) / isDetection(f) / isClassification(f) / isAnomaly(f)
isTorch(f) / isPaddle(f)
tracksIterations(f)          // stepUnit === 'iter'
frameworkMeta(f).datasetFormat
```

**`isSegmentation()` 不等于「按迭代计数」。** 历史上二者等价所以被混用，直到 `TorchAnomaly` 出现——它按 step 计数但不是分割任务。凡是涉及进度、`currentEpoch`/`totalEpochs`、以 epoch 命名的配置列，正确的谓词是 `tracksIterations()`；凡是涉及数据集布局，用 `frameworkMeta(f).datasetFormat`。

### Python 解释器解析

两个运行时不能共用环境（Paddle 环境没有 `torch`，反之亦然），所以 `resolvePythonPath(framework, gpuId, systemConfig)` 按优先级查找：

1. `frameworkPythonMappings["<framework>:<gpuId>"]`
2. `frameworkPythonMappings["<framework>"]`
3. `gpuPythonMappings["<gpuId>"]`（历史遗留的按 GPU 映射，通常指向 Paddle 环境）

查不到返回 `null`，调用方必须报出可执行的错误，**不能退化成裸 `python`**。API 路由里读 `gpuPythonMappings` 的行为是被禁止的——曾经有三处各自解析、其中两处解析错了存储结构（存的是 `{"0": "path"}`，却按 `{"0": {pythonPath}}` 读，于是永远回退到裸 `python`）。

### 启动器差异

```ts
launchPrefix(framework, gpuIds)
// paddle: `python -m paddle.distributed.launch --gpus 0,1`
// torch : `python`  （GPU 通过 CUDA_VISIBLE_DEVICES 传入，由 runner 设置）
```

把 torch 任务包进 Paddle 启动器会立刻失败，因为 torch 环境里没有 `paddle`。

---

## 3. 配置体系：三段式与「YAML 才是事实源」

### 3.1 三份配置的分工

| 配置 | 负责回答 | 数据库表 |
|---|---|---|
| 数据集配置 | 「用什么数据」 | `Dataset.yamlConfig` |
| 训练配置 | 「怎么练」 | `TrainingConfig.yamlConfig` |
| 模型配置 | 「练什么模型」 | `Model.yamlConfig` |

对 `TorchAnomaly`（anomalib）来说，这三段恰好一对一映射到 anomalib 自己的 `data:` / `trainer:` / `model:` 三个块——这也是选择 anomalib 作后端的原因之一。

### 3.2 YAML 是事实源，标量列只是展示缓存

`TrainingConfig` 和 `Model` 上那些扁平标量列（`epoch`、`baseLr`、`architecture`…）**只用于列表展示与筛选**。真正参与合并、真正被训练的是 `yamlConfig`。

列的生成方向永远是 **YAML → 列**，由共享库完成：

- `src/lib/training-yaml.ts`：`generateTrainingYaml`、`parseTrainingParams`、`trainingParamsToColumns`、`totalStepsFor`、`countsIterations`、`TRAINING_FIELD_SUPPORT`
- `src/lib/model-yaml.ts`：`generateModelYaml`、`parseModelParams`、`modelParamsToColumns`、`validateModelParams`、架构表（`SEG_ARCHITECTURES`、`TORCH_SEG_ARCHITECTURES`、`DETECTION_PRESETS`、`TORCH_DET_PRESETS`、`ANOMALY_PRESETS`、`CLAS_ARCHITECTURES`）

这两个模块**不依赖 `fs`**，因此浏览器可以直接 import。创建/编辑对话框用 `TRAINING_FIELD_SUPPORT` 渲染表单、用 `generateTrainingYaml` 实时预览——这保证「用户看到的 YAML」就是「保存下来的 YAML」。

若列与 YAML 发生漂移，用脚本重新推导（先备份数据库）：

```bash
npx tsx scripts/backfill-config-columns.ts --db "file:D:/.../custom.db"          # 干跑
npx tsx scripts/backfill-config-columns.ts --db "file:D:/.../custom.db" --apply
```

该脚本只写 YAML 里确实声明了的列，不会用解析器默认值覆盖有效值，也会跳过 YAML 解析失败的记录。

### 3.3 默认模板目录

`src/lib/default-configs.ts` 负责「新装环境下拉列表不为空」。默认模板按**框架分目录**存放在 `{userConfigsPath}/` 下，各目录内部结构相同（`models` / `training` / `datasets` / `jobs`）：

| 框架 | `getDefaultFolderName()` |
|---|---|
| PaddleSeg | `defaultSeg/` |
| TorchSeg | `defaultTorchSeg/` |
| TorchDet | `defaultTorchDet/` |
| PaddleDetection / PaddleClas / TorchAnomaly | `default/` |

torch 框架单独建目录，是因为它们虽然复用 Paddle 的配置 *schema*，但可用架构不同——给 TorchSeg 一个 `PPLiteSeg` starter 毫无意义。`/api/models/import` 与 `/api/training-configs/import` 首次访问时会用内置 starter 播种对应目录。

注意 `TorchAnomaly` 目前没有独立目录，会落到 `default/`（该目录的 starter 是检测向的），因此异常检测项目通常要自己新建配置而不是导入模板。

### 3.4 字段支持矩阵

`TRAINING_FIELD_SUPPORT: Record<ConfigFramework, Set<TrainingFieldKey>>` 决定表单渲染哪些控件。原则是**绝不显示无效控件**：

- `TorchDet` 去掉了 `regularizerType`（torch 优化器的 weight decay 恒为 L2）和 `normalizeType`/`normMean`/`normStd`（归一化在 torchvision 的 `GeneralizedRCNNTransform` 内部完成）。
- `TorchSeg` = PaddleSeg 字段 + `useAmp`（`torchtrain` 的 `tools/train.py --amp`）。
- `TorchAnomaly` 不提供优化器/调度器/归一化控件：anomalib 的学习率是**模型构造参数**（`EfficientAd(lr=...)`），而模型配置最后合并会覆盖训练配置里的同名值，所以它归属模型配置；PatchCore/PaDiM 根本没有梯度下降。它额外提供 `adTileEnabled`/`adTileSize`/`adTileStride`/`adValInterval`/`adBestMetric`。

`OPTIMIZER_OPTIONS` / `SCHEDULER_OPTIONS` 同样按框架给定可选值，torch 侧的取值镜像 `torchtrain/torchtrain/utils.py` 里 `build_optimizer` 与 `LrScheduler.lr_at` 的实现。

### 3.5 epoch 命名列的单位约定

以 epoch 命名的列里存的是**该框架的原生训练单位**：

| 框架 | `epoch` / `maxEpochs` | `warmupEpochs` | `snapshotEpoch` |
|---|---|---|---|
| PaddleDetection / PaddleClas / TorchDet | epochs | warmup epochs | 保存间隔（epoch） |
| PaddleSeg / TorchSeg | **iters** | **warmup_iters** | **save_interval** |
| TorchAnomaly | **trainer.max_steps** | — | 验证间隔 |

`iters` / `saveInterval` 两列是分割专用，其他框架为 null。判断单位时调用 `countsIterations(framework)`。

同样的约定适用于 `TrainingJob.currentEpoch` / `totalEpochs`：对按迭代计数的框架，**两者都存迭代数**。

---

## 4. 任务配置合并

`POST /api/training-jobs` 用 `mergeYamlConfigs`（`src/lib/yaml-merge.ts`）按 **dataset → training → model** 顺序深度合并，后者覆盖前者。

### 为什么必须在 AST 上做

1. **PaddleDetection 配置使用自定义标签**（`!COCODataSet`、`!CosineDecay`）。`YAML.parse()` 会静默丢弃标签，只有 `YAML.parseDocument()` 能原样往返。因此合并在 `yaml` 的 AST 节点上进行。
2. **文本拼接会产生重复顶层键**。PyYAML 容忍（后者生效），但文档本身是非法 YAML——JS 侧 `yaml.parse()` 会抛 `Map keys must be unique`，导致平台读不回自己刚写出的配置。
3. **拼接使覆盖变成全有全无**。冲突在顶层解决，训练配置就无法只细化 `train_dataset.transforms` 而不重写整个 `train_dataset` 块。

合并语义：

- map + map → 递归合并
- 其他情况 → 整体替换（**序列是替换而非拼接**：覆盖方的 transform 流水线应当自我完备）
- 覆盖方节点上的标签替换基准方的标签（例如 `!COCODataSet` → `!VOCDataSet`）

任何一份源解析失败时，`mergeYamlConfigs` **不抛异常**，而是退回旧的文本拼接行为并在 `warnings` 里说明原因——一份手写坏了的配置会退化，而不是阻塞任务创建。

生成的 YAML 头部带 `# Merged by AutoTrain from: ...` 注释，并用 `lineWidth: 0` 关闭折行（Paddle 配置里的长路径和 `target_size` 列表不能被折断）。

### 合并后的加工

- 分类框架走 `setPaddleClasOutputDir`（写 `Global.output_dir`），其余走 `setTopLevelSaveDir`（写顶层 `save_dir`），值都是本次任务的绝对输出目录。
- 写入 `{userConfigsPath}/{username}/jobs/{jobSlug}.yml`；未配置 `userConfigsPath` 时退回 `{框架仓库}/configs/autotrain/jobs/`。
- 从合并后的 YAML 反解出 `TrainingParams`，用 `totalStepsFor` 得到 `totalEpochs`——**不是**从配置表的标量列取值，这样即便 YAML 被手工改过，进度条依然正确。

---

## 5. 命令生成

`src/lib/job-commands.ts` 是平台执行的**所有命令字符串**的唯一来源，它被 API 路由和 UI 预览同时 import，所以「预览的命令」必然等于「执行的命令」。（历史上有三份拷贝，预览里显示的 flag 跟 runner 实际传的并不一致。）

该模块**故意不依赖 `node:path`**，因此客户端组件可以 import；内部拼路径统一用正斜杠，Python 在 Windows 上也接受。

```ts
buildTrainCommand({ framework, configPath, saveDir, outputDir, gpuIds, useAmp, useVdl, doEval })
buildEvalCommand({ framework, configPath, weightsPath, python })
buildInferCommand({ ...eval, inputPath, outputPath, inputIsFile })
buildExportCommand({ ...eval, outputDir, format })   // 无 export 入口时返回 null
bestWeightsPath(framework, saveDir)
```

方言差异举例：

```bash
# config-flags（PaddleSeg / torchtrain）
python tools/train.py --config "<cfg>" --do_eval --save_dir "<dir>" [--amp] [--use_vdl]

# dash-c（PaddleDetection / PaddleClas）
python -m paddle.distributed.launch --gpus 0 tools/train.py -c "<cfg>" [--amp] [--use_vdl=true --vdl_log_dir=<out>/vdl]
```

`bestWeightsPath` 按 `checkpointLayout` 给出「最佳权重」路径：`nested` → `<save_dir>/best_model/<weightFile>`；`flat` → `<save_dir>/model_final<ext>`。

---

## 6. 训练任务生命周期

全部逻辑在 `src/app/api/training-jobs/route.ts`（创建）与 `src/app/api/training-jobs/[id]/route.ts`（启停、日志入库）。

### 6.1 创建（`POST /api/training-jobs`）

1. 鉴权，并校验 `projectId`/`datasetId`/`modelId`/`configId` 归属调用者（admin 不限）。
2. COCO 数据集：调用 `repairCocoAnnotationAreas` 修掉标注里非正的 `area`（会让训练直接崩）。仅在标注文件确实位于该用户数据目录内时执行。
3. **存储配额预检**：递归统计 `{userDatabasePath}/{username}` 大小，若 `已用 + 5 GB > maxStorageQuota` 则返回 403。
4. `toSafeSlug(body.name)` 得到 `jobSlug`；`sanitizeGpuIds(body.gpuIds)` 把 GPU 列表限制为逗号分隔的 0–63 整数（该值会进入 `shell: true` 的命令串，是注入面）。
5. 三段配置深度合并、改写 `save_dir`、落盘。
6. 生成 `command` / `evalCommand` / `inferCommand`，算出 `totalEpochs`，落库，状态 `pending`。

`outputDir` 存的是命令实际使用的**绝对** `save_dir`，这样 `/api/checkpoints` 能直接定位 `{save_dir}/best_model/...`，不必再去反解命令串。

### 6.2 启动（`PUT /api/training-jobs/[id]` with `{status:"running"}`）

按顺序做的检查，任一失败就把任务置为 `failed` 并写入**可执行的**错误信息：

1. `command` 存在（**绝不接受请求体里的 `command`**——它会被 `shell: true` 执行，等于任意命令执行漏洞）
2. `SystemConfig` 存在
3. `resolvePythonPath` 能解析出解释器
4. `getWorkDir(framework, systemConfig)` 能解析出框架仓库目录
5. **模块预检**：在目标解释器里跑 `importlib.util.find_spec('<pythonModule>')`（只定位不导入，快且无副作用）。失败时提示装什么包、torch 框架还会额外提示去 Settings 配 per-framework 解释器——而不是让任务深入 `tools/train.py` 之后抛 `ModuleNotFoundError`

然后 `startTrainingProcess()`：

- 环境变量：`PYTHONUNBUFFERED=1`；`PYTHONIOENCODING=utf-8`（stdout 是管道时 Python 会退回 locale 编码，中文 Windows 上是 GBK，而我们按 UTF-8 解码，结果 `rawLog` 和 `errorMessage` 全是乱码）；torch 框架设 `CUDA_VISIBLE_DEVICES=<gpuIds>`，Paddle 则删掉该变量（物理 ID 通过 `--gpus` 传）。
- conda：显式配置了 `condaEnv` 就用 `conda run -n <env> --no-capture-output`；否则尝试从解释器路径正则识别 conda 环境，但**只有真的找到 `conda` 可执行文件才启用**，否则退回直接执行解释器。
- `spawn(fullCommand, [], { cwd: workDir, shell: true, detached: process.platform !== 'win32' })`。POSIX 上 `detached` 让子进程自成进程组，以便整组终止。

### 6.3 stdout → 数据库

```
Buffer chunk
  └─ StringDecoder('utf8')      # 保住跨 chunk 切断的多字节字符（否则变 U+FFFD）
      └─ feed(parserState, str) # 按 \n 切行，末尾未完成的片段留到下一 chunk
          └─ parseLine()        # 按 taskKind 分派到对应解析器
              └─ writeParsedLog()
```

`writeParsedLog` 做两件事：

1. 滚动 `TrainingJob` 进度快照，**只写解析器确实产出的列**，这样一条没有 loss/lr 的 EVAL 行不会抹掉上一次的进度。`tracksIterations(framework)` 为真时 `currentEpoch` 存 `log.iteration`——分割日志行同时带 epoch 和 iter（`epoch: 2278, iter: 20500/160000`），存 epoch 会让进度条显示 2278/160000（1%）而不是 20500/160000（13%）。
2. 插入 `TrainingLog` 行。逐类别数组折叠成 `classMetrics` JSON（避免 `num_classes` 泄进表结构）；同一列也承载没有专属列的异常检测指标（`ParsedTrainLog.extraMetrics`），这样 anomalib 新增的指标不会被静默丢弃。`rawLog` 截断到 2000 字符。

写库失败被**吞掉**并只打日志：丢一行日志远不如因为数据库抖动搞死一次训练严重。

### 6.4 停止与结束

- `PUT {status:"stopped"}`：先把 job id 记入 `stoppingJobs`，**再**杀进程。否则 `close` 处理器会把「用户主动停止」当成「崩溃」——`taskkill` 让子进程非零退出，旧代码无条件写 `failed` + "exit code 1"，于是每次手动停止都显示成崩溃。
- `killProcessTree(pid)`：Windows 用 `taskkill /PID <pid> /T /F`；POSIX 对**进程组**发 SIGTERM，5 秒后 SIGKILL。只杀启动器是不够的——`paddle.distributed.launch` 会为每张卡 fork worker，它们会活下来并继续占着显存。
- 若服务重启后再请求停止，平台没有子进程句柄：此时仍会把行标记为 `stopped`，但在 `errorMessage` 里明确说明「本实例没有该进程的句柄，请用 nvidia-smi 确认是否有孤儿进程」。
- `close` 时先 `flush(parserState)` 排空残留半行与未闭合的 EVAL 块（否则一次运行**最后一次、通常也是最好的一次**评估会丢失），再按 `stoppingJobs` / exit code 决定 `stopped` / `completed` / `failed`。失败时把最后 10 行 stderr 写进 `errorMessage`。成功或正常停止会把 `errorMessage` 清空——否则一个曾失败过的任务会永远挂着旧错误。

---

## 7. 日志解析

`src/lib/log-parsers/` 按**任务类型**（不是框架名）分派：

| 解析器 | 适用 | 状态机 |
|---|---|---|
| `detection.ts` | PaddleDetection、TorchDet | 有状态：累积多行 COCO 统计块 |
| `segmentation.ts` | PaddleSeg、TorchSeg | 有状态：累积多行 EVAL 块与逐类别数组 |
| `anomaly.ts` | TorchAnomaly | 有状态：累积 2 行 EVAL 块 |
| `classification.ts` | PaddleClas | 无状态 |

之所以能共享，是因为 **`torchtrain/torchtrain/logger.py` 逐字节复刻了 Paddle 的 stdout 格式**。改任一侧的日志格式，必须同时改另一侧。校验脚本：`scripts/check-log-parsers.ts`。

所有解析器产出统一的 `ParsedTrainLog`（`kind: 'train' | 'eval'`），框架特有字段是可选的——消费方应把「字段缺失」理解为「该框架不适用」而不是「数据丢了」。

`flush()` 在进程退出时会给有状态解析器喂一个**哨兵行**（匹配不上任何模式），迫使它们闭合尚未提交的 EVAL 块。

---

## 8. 评估、推理与导出

`src/app/api/validation-jobs/route.ts`。与训练路径的关键差异：**用 argv 数组 spawn，而不是 shell 字符串**，因此带空格的路径无需引号即可工作。

- 命令预览用 `job-commands.ts`（与 UI 同源），真正执行时按 `FRAMEWORK_META.cliStyle` 重新组装 argv。
- `PYTHONPATH` 设为框架仓库目录。
- **PaddleSeg + 推理 + TIFF 输入**会先透明转存：PaddleSeg 的 `get_image_list()` 只认 JPEG/JPG/BMP/PNG，遇到 `.tif` 会当成「列表文件」去 `open(path,'r')`，在 Windows 上直接 `UnicodeDecodeError`。平台会用用户环境里的 Pillow 把 TIFF 转成 PNG 放进 `_input_staging` 目录，再把该目录喂给 `predict.py`。TorchSeg 不需要（`torchtrain` 的 predictor 直接读 TIFF）。
- 结果图片扫描会**递归一层**：分割的 `predict.py` 写进 `added_prediction/` 和 `pseudo_color_prediction/` 子目录，平铺扫描会一张都找不到（曾经就是这样，页面显示「成功但 0 张图」）。`_input_staging` 目录会被跳过。
- 异常检测评估行按 `key: value` 通配解析（anomalib 的 `Evaluator` 可以挂任意指标），推理结果额外读 `scores.json`；该文件缺失或损坏时返回 null，不会把一次成功的推理判成失败。
- `ValidationJob` **没有** `errorMessage` 列，失败原因写进 `outputLog`。

`POST /api/checkpoints/export` 负责导出（TensorRT / ONNX / TorchScript 方向），已改为 argv 数组调用；`GET /api/checkpoints/export` 下载产物。

---

## 9. 鉴权与多租户

### 会话

`POST /api/auth`（`action: login | register | logout | changePassword`）。登录成功后生成 32 字节随机 token，存入**进程内 Map**（`src/lib/session-store.ts`，挂在 `globalThis` 上以便扛住 HMR 和路由分包），并下发 `httpOnly` + `sameSite=strict` 的 `auth-token` cookie，有效期 7 天。

**第一个注册的用户自动是 `admin`**；用户名 `default` 被保留。

### 权限判定

`src/lib/auth.ts` 提供：

| 函数 | 用途 |
|---|---|
| `getCurrentUser(req)` | 读 cookie → 查会话 → 复核用户仍存在且 `status === 'active'` |
| `requireAuth(req)` | 未登录返回 401 `NextResponse` |
| `requireAdmin(req)` | 非 admin 返回 403 |
| `buildUserFilter(userId, role)` | 列表接口的 `where`：admin 不限；普通用户只见 `userId = 自己` **或 `userId = null`** |
| `requireOwnedScope(req, id)` | **每个 `[id]` 路由都必须用**，返回带归属约束的 `where` |
| `notFoundOrDenied(entity)` | 统一的 404「不存在或无权访问」 |

`[id]` 路由若按裸 `{ id }` 查询再操作结果就是 IDOR（id 可枚举，行可能属于别人）。同理，PATCH 类接口必须白名单字段——曾经 `db.dataset.update({ data: body })` 允许调用方改写 `userId`（把行送给/抢走别人）甚至 `id`。

### 无主数据的语义

`buildUserFilter` 有意让 `userId = null` 的行对**所有**用户可见（兼容用户系统之前的历史数据）。这也是 `POST /api/init-defaults` 改成 admin-only 的原因：它原先无鉴权且创建无主行，等于任何匿名请求都能往所有人的账号里注入一个项目。现在它创建的所有内容都归属调用的 admin。

### 已知缺口（当前代码事实）

- 密码是**无盐 SHA-256**；`changePassword` 不会吊销已有会话。
- 会话在内存，重启即失效；多实例部署需要换 Redis 或数据库表。
- 以下接口**目前没有鉴权**：`/api/system/environment-check`、`/api/system/gpu`、`/api/system/gpu-usage`、`/api/images`、`/api/datasets/image`、`/api/annotation/save`。登录页会轮询 `system/*`，因此加鉴权得先改 UI。（`/api/checkpoints` 已经补上鉴权。）

---

## 10. 路径、存储与配额

三个可配置的根目录（`SystemConfig`）：

| 列 | 含义 | 典型布局 |
|---|---|---|
| `userConfigsPath` | 用户 **YAML 配置** 根目录 | `{root}/{username}/{datasets,models,training,jobs}/` |
| `userDatabasePath` | 用户 **数据与产物** 根目录 | `{root}/{username}/jobs/{jobSlug}/`、数据集目录等 |
| `paddleDetectionPath` / `paddleClasPath` / `paddleSegPath` / `torchPath` | 各框架仓库根 | 子进程的 `cwd` |

用户注册时 `src/lib/user-folders.ts` 会创建上述子目录；配置项缺失则跳过并打告警。

### 路径安全

`src/lib/safe-path.ts`：

- `toSafeSlug(input, fallback)`：任何进入路径段的用户字符串都要过这一层——去分隔符、去盘符、去前导点、限长 100、小写。否则一个名为 `..` 的任务会把输出目录解析到父级，而「删除任务文件」接口就会递归删掉整个用户数据目录。
- `isInside(base, target)`：销毁性操作前的包含性复核。**不要用 `resolved.startsWith(base)`**——`/data/user1-backup` 是以 `/data/user1` 开头的。该函数比较 `base + sep`，并把完全相等视为在内部。
- `resolveWithin(base, ...segments)`：越界返回 null，让调用方 fail closed。

顺带一个坑：`DELETE /api/training-jobs/[id]/files` 必须按 **slug** 而不是原始 `job.name` 定位目录，否则名字里带空格或大写字母的任务永远匹配不上自己的文件夹，删除会静默无效。

### 配额

`User.maxStorageQuota`（BigInt，默认 200 GB）。用量由**递归统计目录大小**实时算出，没有缓存，也没有后台任务：

- `GET /api/users/storage`（本人）、`POST /api/users/storage`（admin 查任意用户）
- `POST /api/training-jobs` 预留 5 GB 做准入检查
- 数据集上传路径同样检查

---

## 11. 系统状态与环境检查

- `GET /api/system/gpu`：跑 `nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits`（5 秒超时）。无 NVIDIA 卡时退化为「CPU 作为计算设备」，并附 `os` 模块算出的 CPU/内存占用。异常时返回 **HTTP 200 + 默认数据**，避免打挂 UI。
- `GET /api/system/gpu-usage`：从**正在运行的训练任务**角度汇总显存占用。
- `GET /api/system/environment-check`：校验各框架仓库的 `requiredFiles` 是否存在，并对每个配置过的解释器起一个短命 Python 进程探测模块可用性与版本。版本区间比 PaddlePaddle 的要求更宽（torch 支持 3.8–3.12），Paddle 的兼容性由模块探测结果体现。**它只报告状态，不安装任何东西。**
- `page.tsx` 顶栏状态灯每 30 秒刷新一次，把上述三者聚合成 `ready` / `partial` / `occupied` / `busy` / `error`。

`GpuMetric` 表存在于 schema 中用于历史曲线，但**当前没有代码定时写入**；Dashboard 读它时通常为空，实时数值来自 `/api/system/gpu`。

---

## 12. 内置 PyTorch 训练器（torchtrain）

`torchtrain/` 不是对外部仓库的包装，而是一个自包含的包，**故意做成 Paddle 仓库的形状**（`torchtrain/` 包 + `tools/train.py`），好让平台用完全一致的方式对待它。详见 [../torchtrain/README.md](../torchtrain/README.md)。

核心决策：**复用 Paddle 的配置模式与日志格式**。`TorchSeg` 吃 PaddleSeg 形状的 YAML，`TorchDet` 吃 PaddleDetection 形状的 YAML，`logger.py` 逐字节复刻 Paddle stdout。于是 `log-parsers`、`yaml-merge`、配置生成器和监控图表得以共享而非分叉。

容易踩的点：

- `num_classes` 语义不同：PaddleDetection（以及 TorchDet 配置）只算前景类，torchvision 需要含背景。`torchtrain/det/dataset.py` 负责 `+1` 与标签偏移。
- 检测的归一化是空操作（torchvision 内部做），所以 `NormalizeImage`/`Permute`/`PadGT` 被接受但忽略。
- 检测指标不硬依赖 pycocotools：`torchtrain/det/metrics.py` 自带 NumPy 版 COCOeval，实测 12 项统计与 pycocotools 完全一致；可导入时优先用 pycocotools。
- 检查点是**嵌套**布局（`<save_dir>/best_model/model.pt`）。
- 架构表必须两处同步：`src/lib/model-yaml.ts` 的 `TORCH_SEG_ARCHITECTURES`/`TORCH_DET_PRESETS`（UI 与校验）和 `torchtrain/torchtrain/{seg,det}/models.py`（真正构建的东西）。

---

## 13. 各框架的特殊约束

### PaddleSeg / TorchSeg

- 按 **iteration** 训练。`TrainingJob.totalEpochs` 存迭代数，用 `totalStepsFor(framework, params)` 求得。
- `len(loss.types)` 必须等于架构输出的 logits 数（主头 + 辅助头），否则报 `RuntimeError: The length of logits_list should equal to the types of loss config`。数量记录在 `SEG_ARCHITECTURES[].logits`。**`loss:` 属于模型配置**，绝不放训练配置。
- `save_dir` / `save_interval` / `log_iters` 是 CLI 参数，不是 YAML 键。

### PaddleDetection / TorchDet

承载检测头的键随 meta-architecture 变化：`yolo_head`（YOLOv3/PP-YOLOE）、`head`（RetinaNet/CenterNet/PicoDet）、`bbox_head`（FasterRCNN）、`detr_head`（DETR/RT-DETR）。骨干网络的超参也是骨干专属——把 CSPResNet 的 `layers`/`use_alpha` 写到 `MobileNetV3:` 块下是硬失败。合法组合编码在 `DETECTION_PRESETS` 里。

### TorchAnomaly（无监督异常检测）

完整设计见 [ANOMALY_DETECTION_DESIGN.md](ANOMALY_DETECTION_DESIGN.md)。要点：

- **唯一不复用 Paddle 模式的框架**。合并后的任务配置是 anomalib 自己的 `model:`/`data:`/`trainer:`（jsonargparse 的 `class_path` + `init_args`），外加一个属于平台的 `autotrain:` 块，后者在交给 anomalib 之前被剥离。
- anomalib 默认的 `Evaluator` 只注册 test 指标，`fit()` 期间验证什么都不记录（没有 AUROC 曲线，`ModelCheckpoint(monitor=...)` 也无从监控）。`torchtrain/torchtrain/ad/evaluator.py` 就是为了修这个。
- `Engine.fit()` 写进自己的版本化目录（`<root>/<Model>/<dataset>/<category>/vN`），适配层把最佳检查点复制到 `<save_dir>/best_model/model.ckpt`——注意是 `.ckpt`，只有 Lightning 检查点能被 anomalib 重新加载。
- PatchCore/PaDiM 的 `trainer.global_step` 恒为 0（没有优化器步），进度改数 batch；它们的 loss 按设计恒为 0，**平坦的 loss 曲线不代表训练卡住**。
- `val_check_interval` 不能从 YAML 设置（大于 batch 数时是 Lightning 硬错误，除非 `check_val_every_n_epoch=None`，而那会让单 epoch 模型完全跳过验证）。训练配置写 `autotrain.val_interval`，由 `ad/config.py::resolve_val_args` 在知道 batch 数后换算。
- 分块（tiling）只对 PaDiM / PatchCore / ReverseDistillation / STFPM 有效，EfficientAD 没有 `tiler`；UI 会禁用该组合，适配层也会拒绝并给出替代建议。
- EfficientAD 有两条硬约束：`train_batch_size` 必须为 1，预处理不得做归一化。它首次运行会下载教师模型与 ImageNette（约 1.5 GB），离线机器需预热缓存。
- 数据集是 `AnomalyFolder`：`normalDir`（必填，**唯一被训练的目录**）、`normalTestDir`、`abnormalDir`、`maskDir`。anomalib 取 test 划分的一半做验证并在其上拟合分数阈值，所以**训练运行报出的指标是乐观的**——留一批独立缺陷图做无偏验证。
- anomalib 需要**独立 venv**（`torchtrain/requirements-ad.txt`，钉在 `anomalib==2.6.*`），注册在 `frameworkPythonMappings.TorchAnomaly` 下。

---

## 14. 数据集处理管线

数据集的**文件**在磁盘上、**元数据**在 `Dataset` 行里。三种格式由框架的 `datasetFormat` 决定：`COCO`、`PaddleSeg`、`AnomalyFolder`。

主要能力（接口细节见 [API_REFERENCE.md](API_REFERENCE.md#数据集)）：

- **分片上传**：`upload/init` → 多次 `upload/chunk` → `upload/complete` 合并，绕开大文件的请求体限制，并在开始前做配额检查。
- **解析统计**：`/api/datasets/parse` 读 COCO 标注或 PaddleSeg 列表文件，写回 `numClasses`/`numAnnotations`/`numTrainImages`/`numEvalImages`/`classStats`。
- **Labelme → COCO**：`/api/datasets/labelme-to-coco`。
- **Labelme → PaddleSeg**：`/api/datasets/labelme-to-paddleseg`，产出 PaddleSeg 官方 `labelme2seg` / Pascal VOC 风格布局：

  ```
  PaddleSeg/{name}/data/
    JPEGImages/       原图，文件名保留
    Annotations/      8-bit 调色板 PNG（palette index = class id）
    train.txt         每行 "JPEGImages/xxx Annotations/xxx.png"
    val.txt
    class_names.txt   第一行恒为 "_background_"
  ```

  类别顺序确定：0 = `_background_`，1..N 按文件名排序扫描时首次出现的顺序。掩码用**索引色 PNG**、调色板取 `src/lib/seg-colors.ts` 的 VOC 伪彩色：PaddleSeg/PIL 读到的是 `mode='P'`（二维索引即 class id，可直接训练），而任何图片查看器都能显示亮色便于目视校验——纯灰度 0..N-1（N < 5）在肉眼看来和全黑没区别。多边形按 even-odd 扫描线填充，**后面的多边形覆盖前面的**（形状顺序即 z-order）。
  文件名中 `[A-Za-z0-9._-]` 之外的字符会被换成 `_`（PaddleSeg 的列表文件用空白符分隔图像与掩码）；历史上带空格的数据集由 `src/lib/paddleseg-list.ts` 兜底。
- **合并 Labelme 到既有 COCO**：`/api/datasets/merge-labelme`（`src/lib/labelme-coco-merge.ts`），支持指定 `trainRatio`/`valRatio`。
- **样本预览**：`/api/datasets/samples` 按类别抽样，`/api/datasets/image` 支持 `colorize` 参数把分割掩码按同一调色板着色返回。

---

## 15. 发布模型（DeployedModel）

把一个训练出的检查点（可带导出目录）提升为一等公民的「生产模型」，按 `product` 标签快速切换。同一 `product` 可以有多条记录，**同时只应有一条 `isActive = true`**，这一点在应用层用事务保证（`POST /api/deployed-models/[id]/activate`）。

UI 在 `src/components/deployed-models-panel.tsx`，嵌在 Validation 页里：既能把某条已发布模型「加载」回评估/推理表单，也能把当次验证所用的配置与权重注册为新的发布模型。

---

## 16. 观察性与调试

- 服务端对每个任务打带前缀的日志：`[Job <id>] ...`、`[Validation <id>] ...`，包含最终执行的命令、工作目录、解释器、conda 判定结果。排查「为什么没跑起来」先看这些。
- `TrainingLog.rawLog` 保留原始日志行（截断 2000 字符），解析异常时可回溯。
- `ActivityLog` 记录用户动作（`create_project`、`start_training`、`stop_training`、`delete_job`…），Dashboard 展示最近活动。
- 校验脚本：`scripts/check-log-parsers.ts`（日志格式回归）、`scripts/check-anomaly-config-chain.ts`（异常检测配置链路）、`scripts/backfill-config-columns.ts`（列与 YAML 对齐）。

### 改动完成前的验证流程

1. `npx tsc --noEmit` —— **注意基线不为零**：`examples/websocket/*`、`mini-services/training-service/*`、`src/app/api/dashboard/stats/route.ts`、`.next/**/validator.ts` 存在既有错误。比较数量，不要假设应为 0。
2. `npx eslint <改动的文件>`
3. `npx next build`
4. `npx next dev -p 3010` 并实际打接口

---

## 17. 已知的架构性限制

| 限制 | 影响 | 位置 |
|---|---|---|
| 会话在内存 | 重启需重新登录；不支持多实例 | `src/lib/session-store.ts` |
| 训练进程句柄在内存 | 重启后正在跑的任务变孤儿，无法停止 | `training-jobs/[id]/route.ts` |
| 无盐 SHA-256 密码 | 抗彩虹表能力弱 | `api/auth/route.ts` |
| 无实时推送 | 依赖 5s/30s 轮询 | `monitoring.tsx`、`page.tsx` |
| `GpuMetric` 无写入者 | 无 GPU 历史曲线 | schema 已就绪 |
| Prisma migrations 落后于 schema | `migrate deploy` 无法复现当前结构，项目用 `db push` | `prisma/migrations/` 停在 `20260403` |
| 响应包络不统一 | `{data}` / `{success,data}` / 裸对象混用 | 新代码请统一用 `{success, data \| error}` |
| TIFF 转存用 shell 字符串 | 路径注入面 | `validation-jobs/route.ts` |
