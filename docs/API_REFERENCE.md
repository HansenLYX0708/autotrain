# REST 接口参考

共 46 个路由文件，全部位于 `src/app/api/**/route.ts`，均运行在 Node runtime（会访问文件系统与子进程）。

## 通用约定

**鉴权**：Cookie `auth-token`（`httpOnly`、`sameSite=strict`、7 天）。下表「鉴权」列含义：

| 标记 | 含义 | 实现 |
|---|---|---|
| — | **无鉴权**（当前事实，非设计意图） | |
| 登录 | 需要有效会话 | `requireAuth` / `getCurrentUser` |
| 归属 | 需要登录且该行属于调用者（admin 不限） | `requireOwnedScope` |
| admin | 仅管理员 | `requireAdmin` |

**数据可见性**：列表接口对普通用户过滤 `userId = 自己 OR userId = null`（`buildUserFilter`），admin 不过滤。无主行（`userId = null`）对所有人可见，这是用户系统上线前历史数据的兼容行为。

**响应包络不统一**（历史原因）：存在 `{ success, data }`、`{ data, pagination }`、裸对象三种。下表在「返回」列注明。新增接口请统一用 `{ success, data | error }`。

**错误体**：`{ error: string, message?: string }` 或 `{ success: false, error: string }`。

---

## 目录

- [认证与用户](#认证与用户)
- [项目](#项目)
- [数据集](#数据集)
- [模型配置](#模型配置)
- [训练配置](#训练配置)
- [训练任务](#训练任务)
- [检查点与导出](#检查点与导出)
- [验证（评估/推理）](#验证评估推理)
- [发布模型](#发布模型)
- [系统与监控](#系统与监控)
- [其他](#其他)

---

## 认证与用户

### `POST /api/auth`

单入口，用 `body.action` 分派。**无需鉴权**（`logout`/`changePassword` 自行校验 cookie）。

| action | body | 行为 |
|---|---|---|
| `login` | `username`, `password` | 校验后生成 token 写入内存会话 + 下发 cookie。账号 `status !== 'active'` 返回 403 |
| `register` | `username`, `password` | 密码 ≥ 6 位；`default` 为保留名；**第一个注册的用户自动为 admin**；同时创建用户配置目录与数据目录 |
| `logout` | — | 删除会话并清 cookie |
| `changePassword` | `oldPassword`, `newPassword` | 新密码 ≥ 6 位。**不会吊销其它会话** |

### `GET /api/auth`

当前用户信息。返回 `{ success, data: { id, username, role, status, createdAt } }`；未登录 401。

### `GET /api/users`

**admin**。全部用户列表（`maxStorageQuota` 序列化为字符串）。返回 `{ success, data }`。

### `POST /api/users`

**admin**。body：`username`, `password`, `role`（默认 `user`）, `maxStorageQuota`。密码 ≥ 6 位，`default` 保留。

### `PUT /api/users?id=<userId>`

**admin**。body：`username`, `role`, `status`, `password`, `maxStorageQuota`（均可选）。

### `DELETE /api/users?id=<userId>`

**admin**。拒绝删除**最后一个 admin**。用户的业务数据不会被删，只是 `userId` 置 null（随后对所有用户可见）。

### `GET /api/users/storage`

**登录**。当前用户的存储信息。递归统计 `{userDatabasePath}/{username}` 的实际大小。

返回 `{ success, data: { userId, username, maxStorageQuota, usedStorage, availableStorage, userDatabasePath } }`。

### `POST /api/users/storage`

**admin**。body：`userId` 或 `username`，查询任意用户的存储信息。（注意方法是 POST，注释里的 `?userId=` 写法已过时。）

---

## 项目

### `GET /api/projects`

**登录**。项目列表，含各关联实体计数。

### `POST /api/projects`

**登录**。创建项目。`task` 由 `resolveTask(framework, task)` 归一化。

### `GET /api/projects/[id]`

**归属**。单个项目及其关联数据。

### `PUT /api/projects/[id]`

**归属**。更新名称/描述/框架/任务/状态。切换 `framework` 会重新推导 `task`。

### `DELETE /api/projects/[id]`

**归属**。**级联删除**其下全部数据集、模型、训练配置、训练任务、验证任务。

---

## 数据集

### `GET /api/datasets?projectId=<id>`

**登录**。数据集列表（含项目信息）。

### `POST /api/datasets`

**登录**。创建数据集记录。body 字段：

- 公共：`name`, `description`, `projectId`, `format`, `datasetDir`, `username`
- COCO / PaddleSeg：`trainImagePath`, `trainAnnoPath`, `evalImagePath`, `evalAnnoPath`
- AnomalyFolder：`normalDir`（必填）, `normalTestDir`, `abnormalDir`, `maskDir`
- 统计：`numClasses`, `numAnnotations`, `numTrainImages`, `numEvalImages`

AnomalyFolder 的名称会被 slug 化（anomalib 用 datamodule `name` 拼结果路径，必须是安全路径段）；发现 `.tif` 文件时会给出告警（anomalib 经 PIL 读取，16-bit TIFF 不保证可用）。

### `GET | PUT | PATCH | DELETE /api/datasets/[id]`

**归属**。`PATCH` 只接受白名单字段——历史上它把整个 body 透传给 `db.dataset.update`，调用方可以改写 `userId`（把数据送给/抢走别人）甚至 `id`。

### `GET /api/datasets/available`

**登录**。扫描 `{userDatabasePath}/{username}` 下磁盘上真实存在的数据集目录（COCO 与 labelme 两种布局），用于「导入已有数据」。

### `DELETE /api/datasets/available?name=<datasetName>`

**登录**。从文件系统删除该数据集目录。找不到时返回 404 并回显搜索过的路径。

### `POST /api/datasets/upload`

**登录**。multipart 一次性上传：`files[]`, `format`（`COCO` \| `labelme`）, `datasetName`。名称只允许字母数字下划线连字符；同名已存在则报错。

### `POST /api/datasets/upload/init`

**登录**。分片上传初始化。body：`datasetName`, `format`, `totalFiles`, `totalSize`, `files`, `confirmedOverwrite`。**此处做存储配额检查**，不足返回 403。

### `POST /api/datasets/upload/chunk`

**登录**。multipart：`uploadId`, `relativePath`, `chunkIndex`, `totalChunks`, `chunk`（Blob）。

### `POST /api/datasets/upload/complete`

**登录**。body：`uploadId`, `targetDir`, `files`, `format`, `datasetName`, `overwrite`。合并分片；`targetDir` 越界返回 403。

### `POST /api/datasets/parse`

**登录**。body：`datasetId`, `annotationPath`, `imageDir`。解析标注并**写回**数据集统计（`numClasses` / `numAnnotations` / `numTrainImages` / `numEvalImages` / `classStats`）。

### `GET /api/datasets/parse?path=<file>&datasetDir=<dir>`

**登录**。只解析不落库，用于上传后的预览。

### `GET /api/datasets/samples?datasetId=<id>&categoryId=<id>&limit=20`

**登录**。按类别抽样返回样本图片与标注，供数据集预览使用。COCO 读标注 JSON，PaddleSeg 读列表文件。

### `GET /api/datasets/image?path=<abs>&colorize=1&classes=<n>`

**无鉴权**。提供数据集图片/掩码文件。

- 路径含 `..` 返回 403（仅此一道防线）
- TIFF/TIF 一律转码为 PNG（浏览器不显示 TIFF）
- `colorize=1` 且源图解码为**单通道**时，按 `src/lib/seg-colors.ts` 的 VOC 调色板把 class id 映射成伪彩色 PNG；多通道输入原样返回，以便已有的 PaddleSeg 伪彩色掩码继续可用

### `POST /api/datasets/labelme-to-coco`

**登录**。把 labelme 标注转换为 COCO 格式。

### `POST /api/datasets/labelme-to-paddleseg`

**登录**。把 labelme 多边形标注转为 PaddleSeg 布局（`{userDb}/{user}/PaddleSeg/{name}/data/`）：`JPEGImages/`、`Annotations/`（8-bit 调色板 PNG）、`train.txt`、`val.txt`、`class_names.txt`。类别顺序确定（0 恒为 `_background_`），多边形按 even-odd 扫描线填充且后者覆盖前者。详见 [ARCHITECTURE.md §14](ARCHITECTURE.md#14-数据集处理管线)。

### `POST /api/datasets/merge-labelme`

**登录**。把一批 labelme 标注合并进既有 COCO 数据集。body：`action`, `cocoDataset`, `labelmeDataset`, `trainRatio`, `valRatio`。

---

## 模型配置

### `GET /api/models?projectId=<id>`

**登录**。模型配置列表。

### `POST /api/models`

**登录**。创建模型配置。

### `GET | PUT | DELETE /api/models/[id]`

**登录 + 归属校验**。`PUT` 接受 `name`, `description`, `yamlConfig`；保存时会**把 YAML 同步写回**用户配置目录下的同名文件（best-effort），避免文件与数据库行漂移。文件名规则与导入接口一致，因此编辑是原地覆盖。

### `GET /api/models/import?projectId=<id>`

**登录**。列出可导入的模型配置：**框架专属的默认模板** + 该用户配置目录下的 YAML 文件。

默认模板存放在 `{userConfigsPath}/{默认文件夹}/models/`，文件夹名由 `getDefaultFolderName(framework)`（`src/lib/default-configs.ts`）决定：

| 框架 | 文件夹 |
|---|---|
| PaddleSeg | `defaultSeg/` |
| TorchSeg | `defaultTorchSeg/` |
| TorchDet | `defaultTorchDet/` |
| 其余（PaddleDetection / PaddleClas / **TorchAnomaly**） | `default/` |

首次访问时会用内置的 starter 配置**播种**该目录，保证新装环境下拉列表不为空。torch 框架单独建目录是因为——虽然它们复用 Paddle 的配置 *schema*，但架构不同（`PPLiteSeg` 的模板对 TorchSeg 毫无用处）。`userConfigsPath` 未配置时退回 `{框架仓库}/configs/autotrain/models/default/`。

### `POST /api/models/import`

**登录**。body：`projectId`, `name`, `description`, `yamlContent`, `isDefault`, `configPath`。导入并落库，同时用 `parseModelParams` 推导展示列。

---

## 训练配置

### `GET /api/training-configs?projectId=<id>&page=1&limit=<n>`

**登录**。分页列表。

### `POST /api/training-configs`

**登录**。创建。body 可带全部标量列（`epoch`, `batchSize`, `baseLr`, `momentum`, `weightDecay`, `scheduler`, `warmupEpochs`, `maxEpochs`, `workerNum`, `evalHeight`, `evalWidth`, `useGpu`, `logIter`, `saveDir`, `snapshotEpoch`, `outputDir`, `weights`, `pretrainWeights`）与 `yamlConfig`。**`yamlConfig` 是事实源**，标量列只是展示缓存。

### `GET | PUT | DELETE /api/training-configs/[id]`

**登录 + 归属校验**。`PUT` 接受 `name`, `yamlConfig`，并同步写回磁盘文件。

### `GET /api/training-configs/import?projectId=<id>`

**登录**。列出可导入的训练配置。与模型导入同一套机制，只是子目录是 `training/`：`{userConfigsPath}/{默认文件夹}/training/` + 用户自己的配置目录。

### `POST /api/training-configs/import`

**登录**。body：`projectId`, `name`, `description`, `yamlContent`, `isDefault`, `configPath`, `trainingParams`。

---

## 训练任务

### `GET /api/training-jobs`

**登录**。查询参数：`page`（默认 1）、`limit`（**默认 5**）、`status`、`projectId`、`datasetId`、`modelId`、`configId`。

返回 `{ data, pagination: { page, limit, total, totalPages } }`。每条记录额外附带：

- `trainingParams`：已解析成对象（不是字符串）
- `absoluteConfigPath`：把相对 `configPath` 按 `userConfigsPath` 解析后的绝对路径

### `POST /api/training-jobs`

**登录**。创建任务（状态 `pending`，**不会自动开始**）。

body：`projectId`, `datasetId`, `modelId`, `configId`（可选）, `name`, `gpuIds`, `useAmp`, `useVdl`。

服务端依次完成：归属校验 → COCO `area` 修复 → **存储配额预检（预留 5 GB，不足 403）** → 名称 slug 化 → `gpuIds` 校验（逗号分隔的 0–63 整数，最多 16 个）→ 三段 YAML 深度合并 → 改写 `save_dir` → 落盘到 `{userConfigsPath}/{username}/jobs/{slug}.yml` → 生成 `command`/`evalCommand`/`inferCommand` → 由合并后 YAML 算出 `totalEpochs`。

常见 400：三份配置都没有 YAML 内容；框架路径未配置；`gpuIds` 非法；缺 `name`。

### `GET /api/training-jobs/[id]`

**归属**。含 project/dataset/model/config、**最近 100 条**日志、日志总数，以及 `pythonPath`（展示用，解析不到时回退裸 `python`）和 `absoluteConfigPath`。

### `PUT /api/training-jobs/[id]`

**归属**。**这是启动/停止训练的入口。**

可更新字段：`name`, `status`, `errorMessage`, `currentEpoch`, `totalEpochs`, `currentLoss`, `currentLr`, `outputDir`, `weightsPath`, `vdlLogDir`, `startedAt`, `completedAt`。

> **`command` 被刻意拒绝**：它会被 `spawn(..., { shell: true })` 执行，接受调用方传值等于任意命令执行。

- `status: "running"`（且当前不是 running）→ 清空上次失败信息、记录活动日志，然后依次校验 command / SystemConfig / Python 解释器 / 框架目录 / **框架模块预检**，全部通过才 `spawn` 子进程。任一步失败直接把任务写成 `failed` 并附可执行的错误提示。
- `status: "stopped"`（且当前是 running）→ 先记入 `stoppingJobs` 再整树杀进程（Windows `taskkill /T /F`；POSIX 对进程组 SIGTERM → 5s 后 SIGKILL）。若本实例没有进程句柄（服务重启过），仍标记 `stopped`，但在 `errorMessage` 中提示用 `nvidia-smi` 检查孤儿进程。

`status` 只接受 `pending` / `running` / `completed` / `failed` / `stopped`。

### `DELETE /api/training-jobs/[id]`

**归属**。先整树杀掉进程，再删除任务（`TrainingLog` 级联删除）。

### `GET /api/training-jobs/[id]/logs`

**归属**。查询参数：`page`（默认 1）、`limit`（默认 100，**上限 1000**）、`epoch`、`minLoss`、`maxLoss`、`sort`（`asc` \| 默认 `desc`）。

### `POST /api/training-jobs/[id]/logs`

**归属**。手工插入日志行（同时会更新父任务的进度快照）。body 为 `TrainingLog` 的字段子集。返回 201。

### `DELETE /api/training-jobs/[id]/logs`

**归属**。清空该任务的全部日志。

### `DELETE /api/training-jobs/[id]/files`

**归属**。删除该任务的检查点与导出模型文件。目录按 **slug** 定位（用原始 `job.name` 会导致带空格/大写的任务永远匹配不上自己的文件夹，删除静默失效），删除前用 `isInside` 复核包含性。

---

## 检查点与导出

### `GET /api/checkpoints`

**登录**（`?jobId=` 额外走**归属**校验）。

| 参数 | 说明 |
|---|---|
| `jobId` | 从任务解析 `save_dir`：先看命令里的 `--save_dir`（`saveDirOnCli` 框架），再看 YAML 的 `save_dir`，再看 `outputDir` |
| `dir` | 显式目录。**仅允许指向调用者自己的存储目录内**（admin 不限）；用户存储未配置时返回 400 |
| `checkExported` | `true` 时附带导出目录的存在性信息 |

按 `FRAMEWORK_META.checkpointLayout` 列举检查点（`flat` 扫 `save_dir` 下的权重文件，`nested` 找 `best_model/<weightFile>`），返回文件名、大小、修改时间。

### `POST /api/checkpoints/export`

**登录**。body：`jobId`, `checkpointPath`, `checkpointName`, `configPath`（可选）, `outputDir`（可选）, `format`。用 argv 数组调用框架的 export 脚本。框架没有 export 入口、框架路径未配置、或该框架没有可用 Python 环境时返回 400。

### `GET /api/checkpoints/export?path=<p>&folder=true`

**登录**。下载导出产物；`folder=true` 时打包为 zip。越界返回 403。

---

## 验证（评估/推理）

### `GET /api/validation-jobs?projectId=&status=&type=`

**登录**。验证任务列表（含项目信息）。

### `POST /api/validation-jobs`

**登录**。创建并可选立即执行。

body：`projectId`（必填）, `name`（必填）, `type`（`eval` \| `infer`，默认 `eval`）, `trainingJobId`, `configPath`, `weightsPath`, `datasetPath`, `inferInputPath`, `inferOutputPath`, `customCommand`, `runImmediately`。

- 不传 `customCommand` 时由 `buildEvalCommand` / `buildInferCommand` 生成（与 UI 预览同源）。
- `runImmediately: true` 时用 **argv 数组** spawn（带空格的路径无需引号），`cwd` 为框架仓库，`PYTHONPATH` 指向该目录。
- **PaddleSeg + infer + TIFF 输入**会先透明转成 PNG 放进 `_input_staging`，再喂给 `predict.py`（PaddleSeg 的 `get_image_list()` 只认 JPEG/JPG/BMP/PNG）。TorchSeg 不需要。
- 结果图片扫描**递归一层**（分割结果写在 `added_prediction/`、`pseudo_color_prediction/` 子目录），跳过 `_input_staging`。
- 异常检测的 EVAL 行按 `key: value` 通配解析；推理额外读 `scores.json`，缺失或损坏时忽略而不是判失败。

失败原因写进 `outputLog`（本表**没有** `errorMessage` 列）。

### `GET | PUT | DELETE /api/validation-jobs/[id]`

**归属**。`PUT` 可更新 `name`, `type`, `status`, `configPath`, `weightsPath`, `datasetPath`, `inferInputPath`, `inferOutputPath`, `resultJson`, `resultPath`, `outputLog`, `startedAt`, `completedAt`。

---

## 发布模型

### `GET /api/deployed-models?product=&framework=&projectId=&activeOnly=true`

**登录**。返回 `{ success, data, products }`（`products` 是可切换的产品标签集合）。

### `POST /api/deployed-models`

**登录**。注册一条发布模型。必填 `name`, `product`, `framework`, `configPath`, `weightsPath`；可选 `architecture`, `projectId`, `trainingJobId`, `exportedDir`, `metrics`, `notes`。

`(userId, product, name)` 唯一，冲突返回 **409**。成功返回 201 `{ success, data, warnings }`。

### `GET | PATCH | DELETE /api/deployed-models/[id]`

**登录**。`PATCH` 可更新 `metrics` 等元数据。

### `POST /api/deployed-models/[id]/activate`

**登录**。在**事务**内把同一 `(userId, product)` 下的其他记录置为非激活，再激活本条——「同一产品同时只有一个激活模型」由此保证。

### `DELETE /api/deployed-models/[id]/activate`

**登录**。清除激活标记（已经是非激活时为空操作）。

### `GET /api/deployed-models/active?product=X&framework=Y`

**登录**。查询当前激活的模型，供外部推理服务拉取。

---

## 系统与监控

### `GET /api/settings`

**admin**。系统配置；不存在时创建默认记录。`torchPath` 在内置 `torchtrain/` 目录确实存在于服务旁时给出可用默认值。

### `PUT /api/settings`

**admin**。body：`condaEnv`, `condaPath`, `pythonEnvsBasePath`, `gpuPythonMappings`, `frameworkPythonMappings`, `userConfigsPath`, `userDatabasePath`, `paddleDetectionPath`, `paddleClasPath`, `paddleSegPath`, `torchPath`, `defaultFramework`。

### `GET /api/system/gpu`

**无鉴权**（登录页会轮询）。执行 `nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits`（5s 超时）。

返回 `{ success, data: { gpus[], cpu: { usage, memoryUsed, memoryTotal }, hasNvidia, platform } }`。无 NVIDIA 卡时 `gpus` 退化为一条「CPU (No GPU detected)」；**发生异常也返回 HTTP 200 + 默认数据**，以免打挂 UI。

### `GET /api/system/gpu-usage`

**无鉴权**。从正在运行的训练任务视角汇总 GPU 占用。

### `GET /api/system/environment-check`

**无鉴权**。对每个框架校验 `requiredFiles` 是否存在；对每个配置过的解释器起一个短命 Python 进程探测模块可用性与版本（接受 3.7–3.12）。

返回 `{ success, data: { validGpus, frameworks: {...}, paddleDetection: { isValid }, ... } }`。**只报告，不安装。**

### `GET /api/dashboard/stats`

**登录**。返回 `{ success, data: { totalProjects, totalDatasets, totalModels, runningTrainingJobs, completedTrainingJobs, pendingTrainingJobs, failedTrainingJobs, gpuMetrics[] } }`。

项目只统计 `status = 'active'`。`gpuMetrics` 来自 `GpuMetric` 表，而**当前没有任何代码写这张表**，因此通常为空。

### `GET /api/activity-logs`

**登录**。当前用户最近 **10** 条活动。

### `POST /api/activity-logs`

**登录**。body：`action`, `entityType`, `entityId`, `entityName`, `details`。服务端内部一般直接调用 `logActivity()` 而不走 HTTP。

### `POST /api/init-defaults`

**admin**。为新装环境播种一套示例项目/数据集/模型/训练配置（YAML 内容是该路由文件里的内联常量，与 `src/lib/default-configs.ts` 的 starter 模板是两套东西），全部**归属调用的 admin**。

> 该接口原先无鉴权且创建无主行（`userId: null`），而无主行对所有用户可见——等于匿名请求可以往每个账号里注入一个项目。现已收紧。

### `GET /api/init-defaults`

**登录**。检查默认数据是否已存在。

---

## 其他

### `GET /api/images`

**无鉴权**。

- `?path=<abs>`：直接返回图片字节（`Cache-Control: public, max-age=3600`）
- `?action=list&dir=<abs>`：列出目录内图片（`.jpg/.jpeg/.png/.bmp/.gif/.tiff/.webp`），按修改时间倒序，返回 `{ success, data: { dir, images[], imageCount } }`；目录不存在时返回**成功**加空数组

### `POST /api/annotation/save`

**无鉴权**。body：`fileName`, `annotation`。保存标注 JSON 文件。

### `GET /api/annotation/save`

**无鉴权**。列出已保存的标注。

### `GET /api`

健康检查，返回 `{ message: "Hello, world!" }`。

---

## 编写新接口时的强制约定

摘自 [../AGENTS.md](../AGENTS.md#api-routes)：

1. 每个 `[id]` 路由**必须**调用 `requireOwnedScope(request, id)` 并用 `findFirst({ where: scope.where })` 查询。按裸 `{ id }` 查再操作就是 IDOR。
2. **绝不**从请求体接受 `command`。命令由服务端生成并以 `shell: true` 执行。
3. 命令字符串**必须**由 `src/lib/job-commands.ts` 构造，不许手拼——它同时被 UI 预览 import，这是两边不漂移的唯一保证。
4. 解释器**必须**用 `resolvePythonPath(framework, gpuId, systemConfig)` 解析，不要直接读 `gpuPythonMappings`。
5. 任何进入路径段的用户字符串必须过 `toSafeSlug`；任何拼好的路径在销毁性操作前必须过 `isInside`。
6. PATCH 类接口必须白名单字段，绝不把 body 整体透传给 `update({ data: body })`。
