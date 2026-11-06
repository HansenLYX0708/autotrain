# Slider ABS / tapring：旋转框与结构关键点定位

目标是从不同背景、照明和视角的图片中定位工件，同时输出孔中心、缺口、基准点等结构位置。旋转框与关键点是两种不同的监督信号：OBB 描述工件占据的范围，语义关键点描述具体结构。实际结构点名称、允许误差、相机参数仍需通过样图和生产要求确定。

## 1. 实施计划与当前状态

| 阶段 | 交付与验收 | 状态 |
|---|---|---|
| 数据规范 | 按采集批次组织；固定类别与关键点顺序；区分可见/遮挡/无法确定；每实例独立关联 | 工具与约定已实现，业务点位待样图确定 |
| 标注工具 | 水平框、旋转框、结构点、可见性、编辑、撤销/重做、保存恢复、Labelme 兼容 | 已实现 |
| 数据准备 | 输出 OBB/Pose 训练目录；检查几何、点位、划分、同批次泄漏和同文件内容泄漏 | 已接入 Datasets 导入、预览和数据库 |
| 基线训练 | 标准 OBB 和 Pose 模型，训练/验证/推理/导出入口 | 已接入真实 Training Jobs；GPU 集成试跑完成 |
| 生产验收 | 关键点像素误差、方向误差、标定后的毫米误差、拒识策略、现场独立测试集 | 计划，需要相机/工件/精度信息 |
| 网页任务调度 | 框架注册、三段 YAML、数据导入、任务生命周期、专用指标、Validation 叠加显示、模型导出 | TorchOBB / TorchPose 已接入 |
| 模型辅助标注 | 使用已训练检查点预测、鼠标调框、逐图确认 | 已实现并用真实预测完成浏览器验收 |

项目框架使用 `TorchOBB` 或 `TorchPose`。标注草稿仍保存为用户隔离的 JSON；导入时创建既有 Prisma `Project`、`Dataset`、`Model`、`TrainingConfig` 记录，无需增加数据库表。

系统入口及预标注步骤见第 10 节。停止后的“重新开始”会重新执行保存的配置；本适配器尚未提供优化器状态续训。

## 2. 先确定“定位”输出

- 单纯平面定位：工件中心 `(x,y)`、平面方向、关键孔位 `(x_i,y_i)`。
- 有倾斜/透视：关键点仍标在原图的真实投影位置；OBB 只是包围区域，**不是透视四边形，也不是完整三维姿态**。
- 需要平面坐标转换：先做镜头畸变校正；至少 4 个具有已知平面坐标且布局不退化的对应点可求单应变换。3 个不共线点只足以确定仿射变换，不能确定一般透视变换。
- 需要三维姿态：相机内参与已知三维点配合 PnP，并单独评估平面/近共线布局的歧义。倾斜变化较大时，不应把一个固定的像素/毫米比例用于全图。
- OBB 的方向具有 180° 等价性，近正方形的长边方向也不稳定。tapring 若是没有缺口的完全对称圆环，仅凭外观无法得到唯一朝向。应使用能打破对称性的真实结构点或改变成像条件。

建议先提供每类 5–10 张代表图：正视、倾斜、强反光、弱对比、局部遮挡、多个工件。另提供期望定位点、允许误差（px 或 mm）、视野大小、图像分辨率和节拍。点位模板中的 `datum_a/b/c` 只是占位符，不能直接充当生产定义。

## 3. 数据准备和标注规范

1. **保留原始分辨率。** 不要标完后再缩放、旋转或裁剪原图。PNG/JPEG/BMP/WebP 可直接使用；16-bit TIFF 需要固定、可复现的显示映射后转 PNG，保留原件与映射参数。EXIF 方向必须在标注前规范化，训练准备工具会拒绝非正常 EXIF 方向。
2. **覆盖场景。** 覆盖不同批次、相机、曝光、光源、角度、背景、工件尺度和缺陷。包括没有工件、只有相似干扰物的负样本；不要只收集标准正视图。
3. **按批次分组。** 图片放入如 `lot01/*.png`、`lot02/*.png` 的子目录。一个工件的连续帧、多视角照片及派生图片应归到同一组。建议约 70/15/15 的组级 train/val/test，比例不是硬要求；每个类别在 train/val 都必须有实例。测试集在模型选择完成前保持独立。
4. **先试标。** 建议先标 50–100 张覆盖典型场景的图片作为流程试验，不代表这些数量足够满足生产精度。抽 10% 由另一人复核，统计同一点的标注分歧。若人工分歧已经超过目标误差，应先改进成像与点位定义。
5. **固定语义。** 例如 `hole_near_notch`、`hole_far_from_notch`、`notch_tip`、`tail_reference`，仅在工件确实有这些特征时使用。不能将关键点命名为“图像左孔/右孔”，因为旋转后会交换。孔中心、边缘交点等要有可复现的判断方法。
6. **一个物体一个实例。** 同一图中的多个 slider / tapring 分别画框，再给各自的实例标点。框尽量紧贴可定义的整体外形，不能把两件工件合成一框。
7. **遮挡规范。** `2` 是可见；`1` 是被遮挡但依据可靠结构能确定位置；`0` 是无法确定或不适用，不参与坐标监督。未处理与 `0` 不同，工具不会自动把漏标当作不可见。Pose 导出要求每个实例至少有一个可监督点。
8. **结构不同就分项目。** 同一项目所有类别共享固定的关键点名称与顺序。slider 与 tapring 若结构不同，推荐分别训练 Pose；OBB 可根据需要训练一个多类别模型。不要把不同含义的点强塞入相同索引。

## 4. 在 Annotation 页面操作

管理员登录后打开 **Annotation**：

1. 新建项目，设置项目名称、类别和关键点模板（每行一个名称，顺序固定）。移除项目中不使用的默认类别。
2. **关联图片文件夹**，选择图片的共同根目录。文件以根目录相对路径标识，因此不同批次可以包含同名文件。
3. 选择新框类别。**旋转框(3点)**：点击一条边的起点、终点，再点击对边任意位置；三点确定真正的矩形。**水平框(2点)**：点击对角两点。
4. 在实例列表选择工件，点击关键点名称，在图上放点。放置后会跳到下一未处理点；点击已标点名称可以重标。在可见性下拉框明确处理遮挡/不可确定点。
5. **选择/拖动**模式可以移动框和关联点，或直接拖动单个点。框的中心、宽高、角度可以输入数值；数值修改只改框，不会移动图上的结构点。角度以图像坐标 `x` 向右、`y` 向下、顺时针为正，单位度。
6. 缩放后通过滚动条或 `Alt + 拖动` / 鼠标中键平移；点击“适应”恢复视野。`Esc` 取消正在绘制的框，`Delete` 删除选中实例及其点；`Ctrl+Z` 撤销、`Ctrl+Shift+Z` / `Ctrl+Y` 重做。文本输入期间不会触发画布删除和切图快捷键。
7. 填写采集批次、分配 train/val/test；可将划分应用到同批次。确认本图完成；无目标图片必须点击“确认本图没有目标”。
8. **保存项目 / Ctrl+S** 将元数据原子写入服务器当前管理员自己的目录。**导出项目** 下载可迁移的 JSON；两者均不包含原图。服务器关闭或浏览器重开后，先打开项目，再关联相同的图片根目录。
9. 点击 **检查 OBB 数据 / 检查 Pose 数据**，修复问题后导出项目。允许保存未完成草稿，训练准备则严格拒绝未确认的数据。

撤销/重做保留最近 50 次操作。修改标注后该图自动恢复为待确认；修改模板会使所有图片重新待确认。已使用的类别或点名不能直接删除/改名，避免静默丢标签；应先明确迁移已有标注。

同目录同名 Labelme JSON 可以导入；跨目录或改名图片复用共享文件匹配规则（`imagePath`、子目录消歧、JSON 文件名回退），有歧义会报错。标准两点矩形、历史四点矩形、矩形多边形均可读取；任意透视四边形/分割多边形不会被悄悄压成 OBB。点必须通过 `group_id` 对应唯一物体。已有项目重新关联图片时保留项目中的标注，并校验图片尺寸；同尺寸换图无法仅凭尺寸发现，请保持原图不可变。

“导出当前图 Labelme”将 OBB 保存为四点 `polygon`（`flags.autotrain_obb=true`）、水平框保存为两点 `rectangle`、关键点保存为 `point`，由 `group_id` 关联。可见性在 `flags.visibility`。普通 Labelme 编辑器可能丢弃扩展字段，**完整项目 JSON 才是模板、划分和确认状态的备份**。

## 5. 生成两种训练数据

在仓库根目录运行，下面的 `python` 应是安装了 Pillow 的解释器：

```powershell
python tools/localization/prepare.py --project D:/data/localization-project.json --images D:/data/raw --output D:/data/localization-v1 --task both
```

`--task obb` 允许先训练定位框，不强制完成关键点；`--task pose` 只生成关键点任务；默认 `both`。输出目录必须不存在，其父目录必须已经存在。工具不会覆盖以前的数据版本。

```text
localization-v1/
  project.json                 原始标注、类别/点位顺序、采集批次
  manifest.json                原图路径 → 导出文件名、划分、SHA-256
  obb/
    data.yaml                  类别配置，绝对数据根路径
    schema.json
    images/{train,val,test}/
    labels/{train,val,test}/   class x1 y1 x2 y2 x3 y3 x4 y4
  pose/
    data.yaml                  kpt_shape: [K,3]
    schema.json                固定关键点名称
    images/{train,val,test}/
    labels/{train,val,test}/   class cx cy w h x1 y1 v1 ... xK yK vK
```

训练标签坐标按图像宽高归一化；项目与推理结果使用原图像素坐标。Pose 的框使用 OBB 的轴对齐外包框，**Pose 不会因而预测旋转框**。`v=0` 导出为 `0 0 0`。负样本导出空 `.txt`。

工具检查图像尺寸、EXIF、有限数值、越界框/点、模板完整性、train/val 类别覆盖、同批次跨集合，以及完全相同文件内容跨集合。重编码/裁剪/近重复照片不一定具有相同哈希，因此仍必须正确分组。转换先校验再写临时目录，成功后整体改名；输出包含图片副本，`both` 会占用约两份原图空间。移动输出目录后应更新两个 `data.yaml` 的 `path`。

普通 Labelme→COCO、→PaddleSeg 和 COCO 合并会拒绝带关键点或显式 OBB 标记的标注，避免丢失姿态信息。通用多边形 COCO 数据本身仍可用于原有分割流程。

## 6. 训练、验证与推理

推荐独立 Python 3.10/3.11 环境，先根据显卡/CUDA 安装相容 PyTorch，再安装：

```powershell
python -m pip install -r tools/localization/requirements.txt
```

Ultralytics 使用 AGPL-3.0 / 企业许可；生产集成前按部署方式确认适用许可。依赖和预训练权重需事先放到离线机器；不要与现有 Paddle / anomalib 环境混装。记录最终使用的 `pip freeze` 和 GPU 驱动版本。

采用稳定的标准任务作为第一版，例如 YOLO11n/s OBB 和 YOLO11n/s Pose。权重从官方来源准备到本地，只加载可信权重。两个模型分别训练，不是一个共享头的联合 OBB+Pose 网络：

```powershell
# 先做 1–3 个 epoch 的流程验证，确认标签和预测覆盖正确，再进行完整训练。
python tools/localization/run.py train --task obb --model D:/weights/yolo11n-obb.pt --data D:/data/localization-v1/obb/data.yaml --output D:/runs/slider-obb-v1 --epochs 100 --imgsz 1024 --batch 8 --device 0
python tools/localization/run.py train --task pose --model D:/weights/yolo11n-pose.pt --data D:/data/localization-v1/pose/data.yaml --output D:/runs/slider-pose-v1 --epochs 100 --imgsz 1024 --batch 8 --device 0

python tools/localization/run.py val --task pose --model D:/runs/slider-pose-v1/weights/best.pt --data D:/data/localization-v1/pose/data.yaml --split test --output D:/runs/slider-pose-test-v1 --device 0
python tools/localization/run.py val --task obb --model D:/runs/slider-obb-v1/weights/best.pt --data D:/data/localization-v1/obb/data.yaml --split test --output D:/runs/slider-obb-test-v1 --device 0

python tools/localization/run.py predict --task pose --model D:/runs/slider-pose-v1/weights/best.pt --schema D:/runs/slider-pose-v1/schema.json --source D:/incoming --output D:/runs/pose-predictions-v1 --device 0
python tools/localization/run.py predict --task obb --model D:/runs/slider-obb-v1/weights/best.pt --source D:/incoming --output D:/runs/obb-predictions-v1 --device 0

python tools/localization/run.py export --task pose --model D:/runs/slider-pose-v1/weights/best.pt --schema D:/runs/slider-pose-v1/schema.json --format onnx --output D:/runs/slider-pose-onnx-v1 --device cpu
```

上面的数据、权重和运行父目录必须存在；每次输出目录使用新名称。`--device cpu` 可用于小规模验证。显存不足先降低 batch；若孔位缩小后只有几像素，优先提高输入分辨率或使用高分辨率 ROI，不要只增加 epoch。ROI 方案需要另行生成裁剪数据并记录坐标逆变换，当前脚本使用全图。

默认固定种子，Windows `workers=0`，关闭镜像、mosaic、mixup；允许平面旋转和轻量尺度/位移。镜像映射 `flip_idx` 未知时不猜左右对称点。AMP 默认关闭以避开运行时校验模型的额外下载。根据现场真实变化调整增强，不能依靠强透视增强代替实际倾斜样本。未开启框架绘图报表；训练 CSV、检查点仍由 Ultralytics 保存。ONNX 导出可能需要额外的 onnx/onnxruntime 等依赖，离线环境应预先准备。

`run.json` 记录参数与 Ultralytics 版本；训练结果另存 `schema.json`，部署时与权重一起版本化。预测会输出叠加图和 `predictions.json`：

- OBB：类别、置信度、四角、中心、长短边、`angle_deg_mod180`。角度按长边归一化到 `[0,180)`，不能直接作为头尾方向。
- Pose：每个实例的检测框及按名称组织的关键点原图坐标和模型置信度。**模型点置信度不是标注的 0/1/2 可见性码**。
- 两次推理的实例列表独立，不能直接按列表索引配对。需要联合结果时，按类别与几何重叠/中心约束进行一对一匹配，并拒绝歧义关联；高密度重叠场景宜采用 OBB 裁剪后 Pose 的两阶段流程，该关联/ROI 生产模块尚未实现。

如固定结构的语义关键点已经足以给出工件位置与方向，也可仅部署 Pose 模型以减少延迟；OBB 模型是否必要，应由数据和节拍实验决定。

## 7. 生产验收

不要只用 mAP 判断定位是否可用。独立测试集至少统计：

| 指标 | 方法 |
|---|---|
| 工件漏检/误检 | 按场景、工件类型、批次分别计算；同时统计整图误报 |
| 关键点位置误差 | 正确匹配实例后，按点计算欧氏像素误差的均值、中位数、P95、最大值；另报未检测/低置信度拒识率，不能只统计成功点 |
| 方向误差 | OBB 按模 180° 的最短角差；有语义头尾的方向按模 360°，两者分开 |
| 透视稳定性 | 不同倾斜程度、位置和工作距离独立分层；必要时看重投影误差 |
| 物理定位误差 | 标定映射后与独立测量基准对比，单位 mm；统计整条成像/标定/预测链路 |
| 性能 | 目标生产硬件上的端到端 P50/P95 延迟、内存/显存、异常恢复 |

生产决策增加点置信度、几何一致性、可见点数量和歧义判定，未达到阈值应拒识/复拍。先通过图像质量、标注一致性与亚像素局部边缘/孔拟合评估误差下限，再决定是否需要更大网络。

后续每轮收集误检、漏检、大误差和低置信度样本，人工修正后进入新训练版本；不要把测试集持续回灌训练后仍沿用同一个测试成绩。

## 8. 开发验证

```powershell
npx tsx scripts/check-labelme-images.ts
python -m unittest discover -s tools/localization -v
```

这些检查覆盖图像匹配、旋转矩形几何、Labelme 往返、实例点关联、可见性、格式输出、负样本、泄漏和拒绝覆盖。它们不替代真实 Ultralytics 训练与现场精度测试。

## 9. 本机实图验收记录（2026-11-03）

使用用户指定的 `G:\datasets\_samples_data\RAW A1F A`，共 **23 张 BMP**（调色板模式、各图尺寸略有不同）。在本机 Edge 浏览器自动化和隔离 SQLite 测试实例中完成：

- 全部 23 张图加载，逐张将浏览器记录尺寸与原文件尺寸核对。
- 实图上三点绘制 OBB、放置实例关键点、缩放后拖动点、遮挡/不可确定状态、撤销、切图隔离。
- 经过实际 API 保存、重新打开、再次关联原图，点坐标完全一致。
- 导出 Labelme，再通过不同名 JSON 的 `imagePath` 匹配重新导入，类别、点位模板、坐标和可见性保留。
- 未审核的项目被训练数据准备工具拒绝。另用 3 张实图制作明确标记为 `FORMAT TEST ONLY` 的测试标签与人为划分，检查 OBB（9 列）和三点 Pose（14 列）的完整输出。
- 格式归一化后还原到像素的最大数值差约为 **1.04×10⁻⁶ px**。这是序列化精度，**不是人工标注精度或模型预测精度**。
- 测试前后逐文件 SHA-256 一致，没有改动源图片。
- 接口另验证了未登录拒绝、非管理员拒绝、管理员之间项目隔离、无效路径拒绝及保存恢复。

本地验收产物在 `temp/localization-smoke/real-data/`（不提交 Git）：
`real-annotation.png` 为实图操作截图，`report.json` 为检查结果，
`raw-project-draft.json` 为 23 张图的工具验收草稿，
`labelme-roundtrip-project.json` 为单图往返项目，
`format-export-final/` 为三张图的 OBB/Pose 格式测试输出。
这些测试标签和划分**不得用来报告生产训练成绩**。

编译、改动文件 ESLint、TypeScript 几何/导入检查和 Python 数据准备检查通过；
全仓 TypeScript 检查仍为既有 7 项错误，没有新增。
仓库已有 3000 端口开发实例，第二开发实例被锁阻止，因此浏览器与接口验收使用 3010 端口的构建后应用和独立测试数据库，未改动现有业务数据库。

本轮没有训练真实定位模型、评估推理精度或完成设备坐标标定。已查看样图主要是相近视角的 ABS 表面结构局部图，可以用于讨论结构点定义；跨场景定位还需要不同视角、背景、照明和批次，整件 OBB 定位应补充能清楚定义完整工件范围的图像。

## 10. 系统训练和模型辅助标注

1. Settings 配置 `torchPath`，并分别给 `TorchOBB`、`TorchPose` 设置可用 CUDA 的 Python 解释器。安装 `tools/localization/requirements.txt`；ONNX 导出另需 `onnx`，部署检查可用 `onnxruntime`。Python 3.8 环境本次验证使用 torch 2.4.1+cu118、Ultralytics 8.3.232、polars 1.8.2、onnx 1.16.2。
2. 保存标注项目，在 **Datasets → 导入定位标注** 选择项目、服务器图片根目录和任务。未标注图片被排除；有实例但未审核的图片会阻止导入。Pose 要求每个实例逐点处理，不能把漏标点转换成不可见点。
3. 预览类别、实例数、排除清单和分组划分，再提交创建数据集及三段配置。导入保存源文件 SHA256、标注快照、随机种子和划分报告。已有明确划分可以保留；自动划分可用采集组或文件名末尾编号。
4. 在 Models / Configurations 调整预训练权重、输入尺寸、batch、epochs 和学习率，再在 **Jobs** 创建并开始任务。当前适配器支持单张 CUDA GPU。真实 runner 写入训练日志和指标，最佳权重为 `best_model/model.pt`，最终权重为 `model_final/model.pt`。
5. 在 **Validation** 选择检查点执行评估或推理；评估配置 `localization.evaluation.split` 可选 `val` / `test`。结果包含 mAP、漏检/误检、匹配实例的像素误差和模 180° 方向误差。像素误差使用置信度 0.25、同类 IoU≥0.5 的一对一匹配；漏检不计入误差均值，必须同时看匹配数。Pose 使用 OKS 和命名点误差。
6. 检查点导出生成 ONNX 或 TorchScript 包，包含模型、`schema.json` 和 `export.json`。导出的原始网络输出仍需按对应任务做缩放还原和 NMS；不能直接把输出张量当作设备坐标。
7. 回到 **Annotation → 模型预测预标注**，选择完成训练的模型。默认只处理未标注且未确认的图片；替换当前图会提示确认，支持撤销。预测直接显示在原图上，保留模型来源并标为待审核。Pose 中置信度低于 0.5 的点留空，等待人工处理。
8. 选择实例后，拖动方形角点/边手柄缩放，黄色圆形手柄旋转（Shift 吸附 15°），拖动框整体平移，拖动关键点修正。缩放/旋转只调整框；平移同时移动关联点。点击“确认本图标注完成”并保存后，该图才可导入下一版训练数据。再次编辑会撤回确认状态。

### gen5/corner/abs 实机记录

源目录 `G:/datasets/sliderlocation/gen5/corner/abs`：48 张图，36 张人工确认标注，208 个 `slider_abs` 旋转框；12 张未标注图片排除。`tapring` 没有实例。关键点只有 4 个完整三点实例及 1 个两点实例，完整 Pose 导入被拒绝。

本次使用末尾编号分组、seed 42：train 17 张/101 个实例，val 12 张/65 个实例，test 7 张/42 个实例。同编号不跨集合。编号代表同一次采集是基于文件命名的假设，不能视为已核实的独立生产批次；这批同场景小样本不代表跨场景生产精度。

隔离验收库通过系统真实任务接口，在 RTX 3090 上训练 3 个 epoch（640、batch 2），完成了真实评估、ONNX 导出和预标注。浏览器验收验证了真实预测显示、鼠标缩放/旋转、撤销、逐图确认、保存恢复；测试修改保存在隔离账户，原始人工标签不受影响。截图位于 `temp/localization-smoke/pipeline/model-assisted-annotation.png`。

业务配置已导入 `.env` 实际使用的 `D:/home/z/my-project/db/custom.db`，项目为 `slider-gen5-corner-obb-d4f84092`。两个现有数据库均已备份至 `autotrain_db_backups/before-localization-*-20261105-184045.db`，未切换 `.env`。正式配置为 YOLO11n-OBB、1280、batch 4、最多 100 epochs，计划使用 RTX 4090；运行结果应以系统真实任务和后续验收记录为准。

### 正式 GPU 训练结果（2026-11-06）

任务 `slider-gen5-obb-gpu-1793932021415`（ID `cmw4ct36n0001ui48h72z2heo`）由已登录的标注所有者通过系统任务接口创建并启动。RTX 4090 / CUDA 11.8 实际执行 86 epochs，连续 30 epochs 无提升触发早停；任务状态 completed。数据库记录了训练日志、最佳检查点以及独立 val / test 验证任务。

| 指标 | val（12 张、65 目标） | test（7 张、42 目标） |
|---|---:|---:|
| mAP50 | 95.24% | 97.08% |
| mAP50–95 | 87.17% | 88.37% |
| 匹配 / 漏检 / 误检 | 59 / 6 / 2 | 39 / 3 / 3 |
| 中心误差 mean / P95 | 12.18 / 26.20 px | 11.84 / 26.16 px |
| 长边角度误差 mean / P95 | 8.05° / 25.93° | 3.64° / 9.61° |

匹配采用置信度 0.25、同类 IoU≥0.5；误差只覆盖匹配目标，方向统计排除长短边比<1.1 的近正方形框。AP 的 precision/recall 使用其自身曲线阈值，不等于上述固定阈值计数。该测试集规模小、场景相近，不能据此推断跨场景或毫米级定位效果。

任务目录：`H:/Autotrain_user_database/Hansen Liu/jobs/slider-gen5-obb-gpu-1793932021415/`。

- `best_model/model.pt`：已训练最佳模型，可在 Annotation 选择该任务预标注。
- `runtime.json` / `autotrain-console.log`：CUDA、GPU 名称、完整配置和原始真实日志。
- `validation/eval-1d694fc225d3/metrics.json`：验证集结果。
- `validation/eval-b7041ac932cc/metrics.json`：保留测试集结果。
- `export_model/obb_onnx/model.onnx`：本次正式导出的模型，旁边为 schema 和导出说明。

后续从网页导出时可选择 ONNX / TorchScript，导出按检查点分目录保存，避免不同 epoch 的导出相互覆盖。OBB 和 Pose 导出均通过真实图片的 ONNX Runtime/PyTorch 张量对比；这验证导出一致性，不代表定位精度。Pose 仅用 4 个真实完整实例裁剪图做了隔离 GPU 适配器测试，未把不完整的全图 Pose 标注用于正式训练。48 张原图及原始标注 JSON 的 SHA256 均保持不变。

最终验收包括两种格式的实际导出及 ZIP 下载、Pose 点位数量校验与 GPU 选择、浏览器格式切换及命令预览。生产构建和修改文件 ESLint 通过；全仓类型检查仍有 5 项已知错误（缺 socket.io/socket.io-client、mini-service stderr、dashboard Date|null），本次修改无新增错误。正式 ONNX 的固定输入为 `1×3×1280×1280`，原始输出为 `1×6×33600`；实际图片推理的最大张量绝对差为 0.013672（PyTorch 对 ONNX Runtime），通过约定的数值容差。原始输出需按模型包说明解码、执行旋转 NMS 并映射回原图。
