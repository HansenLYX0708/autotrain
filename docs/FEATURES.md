# 功能模块说明

按界面组织，说明每个模块**能做什么、怎么操作、背后发生了什么**。实现原理见 [ARCHITECTURE.md](ARCHITECTURE.md)，接口细节见 [API_REFERENCE.md](API_REFERENCE.md)。

应用是单页结构：左侧边栏切换模块，页面组件全部在 `src/components/pages/`。

| 侧边栏 | 组件 | 可见性 |
|---|---|---|
| Dashboard | `dashboard.tsx` | 全部用户 |
| Projects | `projects.tsx` | 全部用户 |
| Datasets | `datasets.tsx` | 全部用户 |
| Models | `models.tsx` | 全部用户 |
| Configurations | `training.tsx` | 全部用户 |
| Jobs | `jobs.tsx` | 全部用户 |
| Monitoring | `monitoring.tsx` | 全部用户 |
| Validation | `validation.tsx` | 全部用户 |
| Annotation | `annotation.tsx` | **仅 admin** |
| User Management | `users.tsx` | **仅 admin** |
| Settings（侧栏底部） | `settings.tsx` | **仅 admin** |
| Manual（顶栏/页脚按钮） | `user-manual.tsx` | 全部用户 |

顶栏还有：用户手册入口、系统状态灯、主题切换（明/暗/跟随系统）、用户菜单（改密码、登出）。

---

## 0. 标准工作流

```
① 建项目（选框架）
        ↓
② 备数据集 ──► 上传 / 导入 / 格式转换 / 解析统计 / 抽样预览
        ↓
③ 配模型     ──┐
④ 配训练参数 ──┼──► ⑤ 建训练任务（三份配置深度合并成一份任务 YAML）
   （数据集配置）─┘        ↓
                    ⑥ 启动 → Monitoring 看曲线
                          ↓
                    ⑦ Validation 评估 / 推理 / 导出 / 发布为生产模型
```

**框架在第 ① 步就定下来了**，后续每一步的表单字段、可选架构、数据集格式、命令行参数都随之变化。切换项目框架会重算 `task`，并使原本匹配旧框架的配置不再适用。

---

## 1. Dashboard

只读总览，**每 30 秒自动刷新**。

- **统计卡片**：活跃项目数、数据集数、模型数，以及训练任务按 running / completed / pending / failed 的计数（`GET /api/dashboard/stats`，普通用户只统计自己的）。
- **GPU / 计算设备**：现场调用 `nvidia-smi` 得到利用率、显存、温度；无 NVIDIA 卡时显示 "No GPU detected" 并退化为 CPU 指标。另外叠加「正在运行的训练任务占用了多少显存」（`/api/system/gpu-usage`）。
- **存储配额**：已用 / 配额 / 剩余。用量是**实时递归统计** `{userDatabasePath}/{username}` 目录大小得到的，数据集很大时这个请求会偏慢。
- **环境体检**（Version compatibility check）：各框架仓库标志文件是否齐、各 Python 环境能否 import 对应模块、Python 版本是否在 3.7–3.12。**只报告，不安装**。
- **Recent Activity**：当前用户最近 10 条操作（建项目、导数据集、启停训练…）。

> GPU **历史曲线**目前没有数据：`GpuMetric` 表虽在 schema 中，但没有任何代码定时写入。

---

## 2. Projects

项目是一切数据的归属容器，也是**框架开关**。

- 新建项目：名称、描述、**框架**（六选一）、任务类型。任务类型由框架推导——检测框架额外允许 `instance_segmentation`，其余框架只有唯一合法值。
- 列表展示每个项目下的数据集 / 模型 / 配置 / 任务数量。
- 归档：把 `status` 置为 `archived`，Dashboard 不再统计。
- **删除是级联的**：会连带删除该项目下所有数据集、模型、训练配置、训练任务与验证任务（已发布模型只解除关联）。

---

## 3. Datasets

最复杂的一个模块（`datasets.tsx` 约 126 KB），承担「把现场采集的图和标注变成框架能吃的数据」这件事。

### 3.1 三种数据集格式

| 格式 | 适用框架 | 磁盘布局 |
|---|---|---|
| `COCO` | PaddleDetection、PaddleClas、TorchDet | 图片目录 + COCO JSON 标注 |
| `PaddleSeg` | PaddleSeg、TorchSeg | `JPEGImages/` + `Annotations/` + `train.txt`/`val.txt` + `class_names.txt` |
| `AnomalyFolder` | TorchAnomaly | 四个目录：`normalDir`（必填）、`normalTestDir`、`abnormalDir`、`maskDir` |

### 3.2 上传

- **分片上传**（`upload/init` → `upload/chunk` × N → `upload/complete`）绕开大文件请求体限制。开始前做**配额检查**，不足直接拒绝。
- 数据集名称只允许字母、数字、下划线、连字符。
- 同名已存在时弹 **Confirm Overwrite** 确认框。
- **Manage Uploaded**：列出磁盘上真实存在的数据集目录（包括没有数据库记录的遗留目录），可直接删除。

### 3.3 格式转换

- **Convert Labelme**：
  - → **COCO**：多边形/矩形标注转 COCO JSON。
  - → **PaddleSeg**：产出官方 `labelme2seg` / Pascal VOC 风格布局。掩码是 **8-bit 索引色 PNG**——调色板索引即 class id，所以 PaddleSeg/PIL 读到的是可直接训练的二维标签图，而任何图片查看器都能看到亮色伪彩（纯灰度 0..N-1 在肉眼看来和全黑没区别）。类别顺序完全确定：0 恒为 `_background_`，1..N 按文件名排序扫描时首次出现的顺序。多边形按 even-odd 扫描线填充，**后面的形状覆盖前面的**（形状顺序即 z-order）。
- **Merge Labelme**：把一批新的 labelme 标注合并进既有 COCO 数据集，可指定 `trainRatio` / `valRatio`。合并前有 **Verify** 步骤，先给出影响预览再决定是否执行：
  - 训练集 / 验证集图片数的「合并前 → 合并后」及增量
  - 源数据里有效样本数（相对 JSON 总数）、按比例被跳过的数量
  - **类别名对照**：Matched / Only in Labelme / Only in COCO。类别名不一致是这类合并最常见的失败原因，`canMerge` 为假时会直接拦住。

### 3.4 解析与统计

触发解析后写回数据集记录：类别数、标注数、训练/验证图片数、逐类别统计。界面展示：

- Train Set / Validation Set 概览
- **Train Set Distribution / Val Set Distribution / Class Distribution** 柱状图
- Labelme categories 与 COCO categories 的对照（转换前后核对类别有没有丢）

### 3.5 样本预览

按类别抽样展示原图与标注。分割数据集用专门的 `SegSampleView` 组件叠加显示掩码——掩码经 `/api/datasets/image?colorize=1&classes=N` 按同一 VOC 调色板着色，**与训练读到的类别 id 完全一致**，所以肉眼看到的颜色错了就是标注错了。TIFF 会被自动转码成 PNG 以便浏览器显示。

---

## 4. Models（模型配置）

这里管理的是**模型配置**（`yamlConfig`），不是权重文件。

对话框分三个页签：

| 页签 | 内容 |
|---|---|
| **Basic** | 名称、描述、所属项目、类别数、预训练权重 |
| **Architecture** | 架构 / backbone / neck / head。可选组合由框架决定 |
| **Advanced** | 归一化类型、EMA、深度/宽度系数，以及分割任务的 `loss:` 等 |

- 右侧是 **YAML 实时预览**（`ConfigYamlPane`），可直接编辑；语法错误会即时提示。表单与预览共用 `src/lib/model-yaml.ts`，因此**看到的 YAML 就是保存的 YAML**。
- **Import Model**：从内置默认模板或用户配置目录里的 YAML 文件导入。
- 保存时会把 YAML **同步写回**用户配置目录的同名文件，避免文件与数据库行漂移。

### 架构选择的硬约束

- **检测**：承载检测头的键随 meta-architecture 变化（`yolo_head` / `head` / `bbox_head` / `detr_head`），骨干超参也是骨干专属。合法组合编码在 `DETECTION_PRESETS`，界面只给出合法项。
- **分割**：`len(loss.types)` 必须等于架构输出的 logits 数（主头 + 辅助头），否则训练直接 `RuntimeError`。数量记在 `SEG_ARCHITECTURES[].logits`，界面会自动对齐。**`loss:` 属于模型配置**，不要写到训练配置里。
- **异常检测**：`ANOMALY_PRESETS` 决定每个算法支持什么。分块（tiling）只有 PaDiM / PatchCore / ReverseDistillation / STFPM 支持，选 EfficientAD 时该开关会被禁用；EfficientAD 还强制 `train_batch_size = 1` 且预处理不得归一化。异常检测的**学习率、输入尺寸、归一化都在模型配置里**（anomalib 里它们是模型构造参数，写在训练配置会被合并覆盖）。
- **分类**：`CLAS_ARCHITECTURES`。

---

## 5. Configurations（训练配置）

对话框分四个页签：

| 页签 | 内容 |
|---|---|
| **Schedule** | 训练长度（epochs 或 iters）、保存间隔、LR 调度器、warmup、里程碑、power/endLr/gamma |
| **Optimizer** | 优化器类型、基础学习率、momentum、weight decay、正则化类型、梯度裁剪 |
| **Data** | batch size、worker 数、输入尺寸、多尺度训练、各类数据增强开关、归一化 |
| **Runtime** | 是否用 GPU、AMP、日志间隔、保存/输出目录、权重与预训练权重 |

**表单只渲染当前框架真正生效的字段**，由 `TRAINING_FIELD_SUPPORT` 驱动。举例：

- PaddleSeg / TorchSeg 看到的是 **iters / save_interval / warmup_iters**，而不是 epochs。
- TorchDet 没有「正则化类型」（torch 优化器的 weight decay 恒为 L2）和归一化字段（torchvision 内部做归一化）。
- TorchSeg 比 PaddleSeg 多一个 AMP 开关。
- TorchAnomaly 没有优化器和调度器可选，取而代之是分块参数、验证间隔（`autotrain.val_interval`）和「最佳检查点监控哪个指标」（`image_AUROC` / `image_F1Score` / `pixel_AUROC` / `pixel_F1Score`）。

给 PaddleSeg 用户看一个「Epochs」滑块但它什么都不做——这类事情是这个界面刻意避免的。

同样有右侧 YAML 实时预览、**Import Config**、保存时同步写回磁盘文件。

---

## 6. Jobs（训练任务）

### 创建任务

选择 **项目 + 数据集 + 模型配置 + 训练配置**，填名称，选 **GPU**（可多选，如 `0,1`），勾选 AMP / VisualDL。界面会显示**将要执行的命令预览**——该预览与 runner 用的是同一个 `job-commands.ts`，所以不会出现「预览与实际不一致」。

提交后服务端：修复 COCO 标注里非正的 `area` → **检查存储配额（预留 5 GB）** → 深度合并三份 YAML → 写任务配置文件 → 生成训练/评估/推理命令 → 从合并后的 YAML 算出总步数。任务初始状态是 `pending`，**不会自动开始**。

### 启动 / 停止 / 删除

- **启动**前会逐项预检：命令存在、系统配置存在、能解析出 Python 解释器、框架仓库目录存在、**该解释器里确实能 import 框架模块**。任一项失败，任务立刻变 `failed` 并给出可执行的错误说明（例如「TorchSeg 任务请在 Settings 里为该框架单独配一个装了 PyTorch 的环境，按 GPU 的映射通常指向 PaddlePaddle 环境」），而不是让任务深入 Python 之后抛 `ModuleNotFoundError`。
- **停止**会杀掉整棵进程树——只杀启动器不够，`paddle.distributed.launch` 为每张卡 fork 的 worker 会活下来继续占显存。手动停止的任务状态是 `stopped`，**不是** `failed`。
- 如果服务在任务运行期间重启过，平台就失去了进程句柄：此时「停止」仍会把记录标记为 `stopped`，但会提示你用 `nvidia-smi` 确认是否有孤儿进程需要手工清理。
- **删除**任务会连带删除其日志；另有单独的「删除文件」操作清理检查点与导出产物。

### 列表

按状态 / 项目 / 数据集 / 模型 / 配置筛选，分页（**默认每页 5 条**）。展示进度、当前 loss、学习率、耗时，以及失败任务的错误信息。

---

## 7. Monitoring（训练监控）

选一个训练任务，**每 5 秒轮询**刷新（无 WebSocket）。

**顶部指标卡**：Epoch Progress（进度条）、Current Loss、Learning Rate、ETA。

> 对按迭代计数的框架（PaddleSeg / TorchSeg / TorchAnomaly），进度条的分子分母**都是迭代/步数**，不是 epoch。

**图表**（随框架显示不同组合）：

| 图表 | 适用 | 数据来源 |
|---|---|---|
| Loss Over Time | 全部 | 每条训练日志行的 `loss` |
| Learning Rate Schedule | 全部 | `learningRate` |
| Detection Metrics | 检测 | `mAP`（AP@[0.50:0.95]）、`mAP50` |
| Segmentation Metrics | 分割 | `mIoU`、`acc`、`kappa`、`dice` |
| Anomaly Detection Metrics | 异常检测 | `imageAuroc`、`imageF1`、`pixelAuroc`、`pixelF1`、`threshold` |
| Performance Metrics | 全部 | `batchCost`、`dataCost`、`readerCost`、`ips`、显存占用 |
| Raw Log Output | 全部 | `rawLog` 原始日志行 |

### 看图时要知道的事

- **PatchCore / PaDiM 的 loss 恒为 0**（它们不做梯度下降，是内存库方法）。**平坦的 loss 曲线不代表训练卡住。**
- 异常检测的指标来自 anomalib 在验证划分上的自适应阈值，而该划分正是拟合阈值用的，因此**训练期指标偏乐观**。要无偏结论请单独跑一次 Validation。
- 评估点比训练点稀疏很多（按 `save_interval` / `snapshotEpoch` 触发），指标折线天然是阶梯状。
- 一次运行的**最后一次评估**（通常也是最好的一次）依赖进程退出时的日志 flush，这条路径已专门处理，不会丢。

---

## 8. Validation（评估 / 推理 / 导出 / 发布）

三个页签：**Eval**、**Infer**、**History**。

### 8.1 选择模型来源

- 选一个训练任务 → 平台自动列出它的检查点（`/api/checkpoints`）。按框架的检查点布局查找：`nested`（PaddleSeg/TorchSeg/TorchDet）找 `<save_dir>/best_model/<权重文件>`，`flat`（PaddleDetection/PaddleClas）扫 `save_dir` 下的权重文件。
- 或从 **Deployed Models 面板**把一条已发布模型「加载」进表单。

### 8.2 Eval

用任务的配置 + 选定权重跑框架的 eval 脚本，结果指标写进 `resultJson` 并展示在 **Evaluation Results**。

### 8.3 Infer

输入可以是单张图或整个目录，指定输出目录。**Inference Results** 以缩略图网格展示结果，点开看大图（Detection Result 面板）。

几个已经处理好的坑：

- **PaddleSeg 不认 TIFF**：其 `get_image_list()` 只接受 JPEG/JPG/BMP/PNG，遇到 `.tif` 会当成列表文件去按文本打开，在 Windows 上直接 `UnicodeDecodeError`。平台会先把 TIFF 透明转成 PNG 放进 `_input_staging` 目录再喂给 `predict.py`。**TorchSeg 不需要**——它直接读 TIFF。
- **分割结果不在输出根目录**：`predict.py` 写进 `added_prediction/` 和 `pseudo_color_prediction/` 子目录。结果扫描会递归一层，并跳过 `_input_staging`（那是输入的 PNG 副本，不是结果）。
- **异常检测**额外读输出目录下的 `scores.json` 展示每张图的异常分数；该文件缺失不会让任务判失败（热力图仍然有用）。

失败原因显示在 `outputLog`（验证任务没有独立的错误字段）。

### 8.4 导出

对有 export 入口的框架，可把检查点导出为推理模型（TensorRT / ONNX / TorchScript 方向，取决于框架），并打包下载（目录会压成 zip）。

### 8.5 发布模型（Deployed Models）

把「配置 + 权重（+ 导出目录）」注册为一条按 **product 标签** 管理的生产模型：

- 一个 product 可以有多条记录（多个版本），**同时只有一条处于激活状态**，切换在事务里完成。
- 记录可带 `metrics`（mAP / mIoU / bestIter…）和备注，便于追溯。
- 外部推理服务通过 `GET /api/deployed-models/active?product=X&framework=Y` 拉取当前该用的模型。

`(用户, product, 名称)` 三元组唯一，重复注册返回 409。

---

## 9. Annotation（仅 admin）

内置的轻量**矩形框标注**工具：

- 选择一个图片文件夹，逐张浏览（显示 `当前/总数`）。
- 在图上拉框，给框指定标签；**Add New Label** 可即时新增标签名。
- **Selected Box** 面板编辑或删除当前选中的框。
- **View** 控制显示方式。
- **Save All** 把标注写成 JSON 文件（`POST /api/annotation/save`）。

需要更强的分割标注能力时，用独立的 Python 工具：

- [`tools/napari_seg/`](../tools/napari_seg/readme.md) —— 基于 napari 的分割标注流程
- [`tools/tem_analysis/`](../tools/tem_analysis/readme.md) —— TEM 图像的几何测量与报告

---

## 10. User Management（仅 admin）

- 新建用户：用户名、密码（≥ 6 位）、角色、**存储配额**（默认 200 GB）。
- 修改用户名 / 角色 / 状态（`active` / `disabled`）/ 密码 / 配额。
- 禁用（`disabled`）会让该用户的会话立即失效。
- 删除用户：**拒绝删除最后一个 admin**。注意删除用户**不会**删除他的业务数据，只是把 `userId` 置空——而无主数据对**所有用户可见**，需要的话请先手工清理。
- `default` 是保留用户名。

新用户创建时会自动建好目录结构：`{userConfigsPath}/{username}/{datasets,models,training,jobs}/` 和 `{userDatabasePath}/{username}/`。若这两个路径没在 Settings 里配置，创建会被跳过（只打告警），后续训练会因为找不到目录而失败。

---

## 11. Settings（仅 admin）

平台的所有环境依赖都在这一页，配错了后面每一步都会失败。

| 分组 | 字段 | 说明 |
|---|---|---|
| 框架仓库路径 | `paddleDetectionPath`、`paddleClasPath`、`paddleSegPath`、`torchPath` | 子进程的工作目录。`torchPath` 指向内置 `torchtrain/`，三个 torch 框架**共用** |
| Python 环境 | `frameworkPythonMappings` | JSON，如 `{"TorchSeg": "D:/pythonEnvs/torchEnv/Scripts/python.exe"}`。键也可写 `"TorchSeg:0"` 指定某张卡 |
| Python 环境 | `gpuPythonMappings` | JSON，如 `{"0": "C:/envs/gpu0/python.exe"}`。历史遗留，优先级**低于**上者 |
| Python 环境 | `pythonEnvsBasePath` | 多环境根目录，仅供浏览 |
| conda | `condaEnv`、`condaPath` | 填了就用 `conda run -n <env>` 启动；留空时会尝试从解释器路径识别，但只有真找到 `conda` 可执行文件才启用 |
| 用户目录 | `userConfigsPath` | 用户 YAML 配置根 |
| 用户目录 | `userDatabasePath` | 用户数据与训练产物根 |
| 默认值 | `defaultFramework` | 新建项目的默认框架 |

**System Status** 面板实时展示环境体检结果：每个框架仓库的标志文件是否齐全、每个解释器能否 import 对应模块、Python 版本是否受支持。

### 为什么必须配 per-framework 解释器

Paddle 环境里没有 `torch`，torch 环境里没有 `paddle`，而 anomalib 还会钉住 Lightning 和 jsonargparse——所以 **TorchAnomaly 需要自己独立的 venv**（`torchtrain/requirements-ad.txt`，`anomalib==2.6.*`）。只配了按 GPU 的映射时，torch 任务会拿到 PaddlePaddle 的解释器；平台会在启动预检阶段拦住并明确告诉你去哪配。

---

## 12. 用户手册（应用内）

顶栏 **Manual** 按钮或页脚链接打开 `user-manual.tsx`，是一份内嵌在应用里的操作手册，与 [USER_MANUAL.md](USER_MANUAL.md) 内容同源。

---

## 13. 常见故障速查

| 现象 | 原因与处理 |
|---|---|
| 启动训练立刻 failed，提示没有 Python 环境 | Settings → Framework Python environments 里为该框架加一条映射 |
| `ModuleNotFoundError: No module named 'paddleseg' / 'torch'` | 解释器指向了另一个运行时的环境。按框架分别配置 |
| `train.py: error: unrecognized arguments: --do_eval --save_dir` | 项目框架与实际仓库路径不匹配（例如 PaddleSeg 任务跑到了 PaddleDetection 的 `train.py`） |
| `RuntimeError: The length of logits_list should equal to the types of loss config` | 分割模型的 `loss.types` 数量与架构 logits 数不符。检查模型配置，不要把 `loss:` 写进训练配置 |
| 进度条一直 0% 或显示成 2278/160000 | 分割/异常框架的进度单位是迭代数。若数值离谱，检查任务的 `totalEpochs` 是否由合并后 YAML 正确推导 |
| 分割推理「成功但 0 张结果图」 | 结果在 `added_prediction/`、`pseudo_color_prediction/` 子目录，确认扫描到了子目录 |
| PaddleSeg 推理报 `UnicodeDecodeError` | TIFF 输入。平台会自动转存 PNG；若失败，手动转好再传 |
| 异常检测 loss 曲线全是 0 | PatchCore / PaDiM 是内存库方法，没有梯度下降。这是正常的 |
| 停止训练后任务显示 failed | 旧版本行为。当前版本手动停止会写 `stopped` |
| 停不掉训练，显存一直被占 | 服务在任务运行期间重启过，进程句柄已丢失。用 `nvidia-smi` 找到 PID 手工结束 |
| 登录后一片空、项目全没了 | `DATABASE_URL` 指错，SQLite 新建了空库。看服务启动日志里的醒目告警 |
| 重启服务后需要重新登录 | 会话在内存里，重启即失效。这是已知限制 |
| 创建任务报「存储空间不足」 | 配额检查会预留 5 GB。联系 admin 扩容或清理数据 |
| 上传大数据集失败 | 用分片上传路径；确认 `userDatabasePath` 已配置且磁盘有空间 |
| EfficientAD 首次运行卡在下载 | 它会拉教师模型和 ImageNette（约 1.5 GB）。离线机器需预热缓存 |
