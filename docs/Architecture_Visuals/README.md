# HawkeyePlus 平台架构 · 领导汇报版

本目录介绍 **HawkeyePlus 工业视觉模型训练平台**。参照 SwiftFlow 图册的图片组合方式、WD 模板和可编辑 PPT 形式，内容依据本仓库的真实架构、框架注册表和功能实现。

**两平台结合汇报**见 [Joint_Platform/index.html](Joint_Platform/index.html)：以“总体概念 → 为什么 → 标准流程 → SwiftFlow → HawkeyePlus → 协同交付”为主线，中英文每版 20 页，含正式封面、可点击目录、四章过渡页、总结与结束页，解释从硬件设计到检测/测量交付的统一工程方法。

打开 `index.html` 可预览与下载。**中文 / 英文 × 深蓝 / 明色，共四版，每版 5 页**：

1. **HawkeyePlus 平台总体架构**：数据与标注资产 → 可视化训练中枢 → 验证与模型交付；下层为多框架训练能力、GPU/Python 环境与数据资产存储。
2. **从数据到模型的可视化闭环**：准备数据 → 组合配置 → 训练监控 → 验证效果 → 模型交付，展示四类视觉任务共用同一平台。
3. **Detection 工业缺陷检测**：检测框示意、PaddleDetection 的 7 组模型、TorchDet 的 4 组模型及 COCO / mAP 工作流。
4. **分割模型**：PaddleSeg 的 7 个预设、TorchSeg 的 4 个预设与项目 TEM 示例。
5. **分类与异常检测**：PaddleClas 的 9 个变体和 TorchAnomaly 的 5 个预设。

| 文件 | 内容 |
| --- | --- |
| `HawkeyePlus_Architecture_Dark_Editable.pptx` | 深蓝科技版，推荐用于屏幕汇报 |
| `HawkeyePlus_Architecture_Light_Editable.pptx` | 明色汇报版 |
| `HawkeyePlus_Architecture_EN_Dark_Editable.pptx` | 英文深蓝科技版，5 页，含英文演讲者备注 |
| `HawkeyePlus_Architecture_EN_Light_Editable.pptx` | 英文明色汇报版，5 页，含英文演讲者备注 |
| `png/01_Architecture_dark.png` | 首选架构主图，3840 × 2160 |
| `png/03_Detection_en_dark.png` | 英文 Detection 专页，3840 × 2160 |
| `png/` | 5 页 × 2 种语言 × 2 种配色，共 20 张 4K PNG |
| `svg/` | 对应 SVG 源图 |
| `HawkeyePlus_Architecture_*.pdf` | 四版 PowerPoint 导出的固定版式 |
| `Supported_Models.md` | 中英双语完整模型、配置值、可选骨干与支持边界 |
| `model_catalog.json` | 从当前源码导出的模型预设快照 |
| `assets/` | 可替换的独立插画与界面配图 |
| `Preview.jpg` / `Preview_en_dark.jpg` 等 | 四版本总览 / 单版本总览 |
| `manifest.json` | 来源与 SHA-256、每页原生对象数量 |
| `verification/report.json` | 平台名称、关键架构内容、可编辑对象与文字越界检查 |

## 编辑方式

文字、模块、标签与连接线为 PowerPoint 原生对象，模块按语义分组；可以整体移动，也可以进入组修改文字。配图为独立图片，可移动、缩放和替换，插画内部像素不作为原生形状编辑。PPT 与 SVG 均嵌入全部图片，打开时不依赖旁边的素材目录。

## 图片与汇报口径

- 原图、分割叠加、标注工具、训练配置与曲线图片来自本项目 `docs/tem_report/assets/`。创建任务截图只截取配置控件，去掉背景列表和命令路径。
- napari 是本项目配套的桌面分割标注工具；Web 内置矩形框标注与它是不同工具。
- 神经网络、框架模块、模型包与版本激活图为原创概念插画；通用存储、主机和屏幕造型与 WD 模板来自用户指定的参考图册。
- Detection 检测框、分类图、异常热力图为原创工业零件示意，已在对应页标明；它们不是真实训练预测或效果指标。架构和研发流程也加入了 Detection 配图。
- 模型列表直接读取 `src/lib/model-yaml.ts`；TorchDet 预设及骨干与 `torchtrain/torchtrain/det/models.py` 核对。数量是各框架下的界面选项，不是跨框架去重后的算法数量，不代表本机已逐一训练验收。
- 特别标注了 **TorchSeg 的 DeepLabV3P / DeepLabV3+ 界面名称实际使用 torchvision DeepLabV3**；SSD/SSDLite 是同一 TorchDet SSD 预设的两个骨干变体。完整说明见 `Supported_Models.md`。
- PaddlePaddle 对应检测、分类、分割接入；PyTorch 对应 TorchDet/TorchSeg；anomalib 是基于 PyTorch 的异常检测后端，并非第三种独立运行时。
- 三份配置合成训练任务；平台调用框架训练，提供任务管理与监控。图中未承诺分布式调度或自动模型搜索。
- 导出格式与支持范围取决于框架。模型发布是登记配置与权重、管理产品版本和激活模型；外部推理服务可按 API 查询，图中未承诺自动部署现场设备。
- 验证回路表示使用者结合结果更新数据与配置。示例曲线不作为平台性能基准。
- 以当前 `ARCHITECTURE.md`、`FEATURES.md` 与源码为事实来源，避免沿用旧 Overview 中尚未实现的 Socket.io 通道描述。

## 重新生成

使用安装了 `Pillow`、`python-pptx`、`playwright` 的 Python，以及仓库现有的 Node.js / TypeScript 依赖：

```powershell
python docs/Architecture_Visuals/build.py
python docs/Architecture_Visuals/verify_powerpoint.py
```

构建脚本默认从 `G:\projects\AOIPlatform\HwkUltra_g\docs\Architecture_Visuals` 读取原有 WD 模板、通用绘图函数及 SVG → 原生 PPT 转换器，不修改参考目录。可用 `--reference-dir` 指定其它位置。

第二步需要 Windows、Microsoft PowerPoint 与 `pywin32`，会实际打开此次生成的文件，检查可编辑对象与文字边界，导出 PDF 和最终 4K PNG；不会退出用户原本打开的 PowerPoint。

## English delivery

Open `index.html` and choose **English · Dark** or **English · Light**. Each edition contains five slides covering platform architecture, the visual model lifecycle, object detection, segmentation, and classification/anomaly detection. Text, panels and connectors are native editable PowerPoint objects; illustrations are separate replaceable pictures. The English slides and speaker notes are checked for untranslated Chinese text.

`Supported_Models.md` provides the full model/backbone list, generated from the current UI model tables. Custom YAML is supported as an authoring path; it is not a promise that every upstream model is compatible with the selected backend.
