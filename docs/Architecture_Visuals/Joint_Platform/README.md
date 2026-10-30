# SwiftFlow + HawkeyePlus — 标准化检测与测量工程平台

本汇报以**一套标准工程方法、两个互补平台、可复用交付资产**组织内容，适合向管理层解释平台建设目的与分工。它不是两个独立 PPT 的简单拼接，也不是已经完成自动连接器开发的声明。

打开 `index.html` 浏览并下载。中文、英文各有深蓝与明色两种配色，**每份 PPT 20 页**。正式标题为：

**SwiftFlow + HawkeyePlus｜工业检测与测量标准化平台：架构、流程与协同交付**

**SwiftFlow + HawkeyePlus | Standardized Inspection & Measurement: Architecture, Workflows & Integrated Delivery**

本版补齐正式封面、可点击目录、四个章节过渡页、总结与讨论结束页；原有 12 页主体内容完整保留。文字、模块和连接线可编辑，配图可独立替换。另附 PDF、4K PNG 和 SVG。

## 汇报顺序

| 页码 | 内容 | 要回答的问题 |
| --- | --- | --- |
| 01 | 正式封面 | 完整标题、平台名称、汇报主题与日期 |
| 02 | 汇报目录 | 四章结构；点击目录卡片可跳转到对应章节首页 |
| 03 | 第一章过渡页：总体概念与建设目的 | 明确本章主题、引导问题和三个要点 |
| 04–06 | 总体关系、建设目的、标准工程流程 | 为什么结合？用户如何从硬件、流程与算法走到交付？ |
| 07 | 第二章过渡页：SwiftFlow | 设备、流程与算法执行 |
| 08–10 | SwiftFlow 架构、硬件到流程、检测与测量 | 如何接入设备、组织流程、输出结果？ |
| 11 | 第三章过渡页：HawkeyePlus | 工业视觉模型研发与验证 |
| 12–16 | HawkeyePlus 架构、研发闭环与全部模型介绍 | 如何准备数据、训练验证、管理模型？ |
| 17 | 第四章过渡页：协同交付 | 从研发效果到现场可用 |
| 18 | 双平台衔接与交付基线 | 如何适配、联合验收与交付？ |
| 19 | 汇报总结 | 统一方法、两个平台、标准交付 |
| 20 | 结束与讨论 | 谢谢 / Thank You，讨论优先场景、验收与资产复用 |

全部页面使用连续的 `页码 / 20`。正文顶部标明所属章节。PPT 同时带有 **6 个原生 PowerPoint 分节**（开场、四章、总结），方便在缩略图视图中折叠和整理。

## 核心分工

- **工程人员**：定义检测/测量目标，设计光机电方案，确定成像、标定与验收要求。
- **SwiftFlow**：承载设备接入与调试、流程设计、传统视觉/模型推理/测量算法的执行、结果记录。
- **HawkeyePlus**：承载训练数据、模型配置、训练监控、独立验证、模型导出和版本管理。
- **共同工程方法**：约定每阶段输入、输出、检查点和交付包，再以联合验证连接研发效果与现场可用性。

传统视觉和几何/激光测量可以直接在 SwiftFlow 中执行，不要求所有任务都经过 AI 训练。AI 可辅助定位或分割；物理测量仍依赖标定、单位与参考验证。

## 交付物

| 文件 | 用途 |
| --- | --- |
| `SwiftFlow_HawkeyePlus_Integrated_ZH_Dark_Editable.pptx` | 中文深蓝版，推荐主汇报文件 |
| `SwiftFlow_HawkeyePlus_Integrated_EN_Dark_Editable.pptx` | 英文深蓝版，含英文备注 |
| `SwiftFlow_HawkeyePlus_Integrated_ZH_Light_Editable.pptx` | 中文明色版 |
| `SwiftFlow_HawkeyePlus_Integrated_EN_Light_Editable.pptx` | 英文明色版 |
| `*.pdf` | 四套固定版式文件 |
| `png/01_Cover_zh_dark.png` / `png/01_Cover_en_dark.png` | 中英文正式封面 |
| `png/` / `svg/` | 本版 20 页 × 4 个版本，共 80 张当前页面及对应 SVG；早期单页文件保留以兼容旧链接 |
| `Preview_zh_dark.jpg` 等 | 20 页总览 |
| `verification/report.json` | 章节顺序、目录链接、原生分节、页码、可编辑对象、英文完整性与文字边界检查 |
| `manifest.json` | 两个仓库的引用来源与哈希 |

HawkeyePlus 章节保留了上一版全部五页，包括 Detection、分割、分类与异常检测。完整模型/骨干清单见 [../Supported_Models.md](../Supported_Models.md)。

## 两个平台如何衔接

推荐交付模型产物、标签映射、预处理/输入约定、输出解释、阈值、版本和验收证据，并与 SwiftFlow 的机型、流程、标定及工艺配置关联。

这一建议有现有实现依据，但**不是“所有训练模型都能直接在 SwiftFlow 中运行”**：

- HawkeyePlus 按框架提供导出能力，并提供按产品登记/激活模型与查询当前版本的 API。
- SwiftFlow `InferenceSessionFactory` 当前实现 ONNX（包含 YOLO 路径）与 Paddle 后端；TorchSharp 为保留项。
- SwiftFlow 的输入形状、通道顺序、归一化、标签、阈值和输出解码均须与模型匹配；格式、算子及运行环境需要逐项验证。训练检查点不能直接当作推理交付物。
- 当前 `InferenceTask` 为 Detection、Classification、Segmentation，没有独立异常检测解码任务；异常模型的现场接入需要额外适配。
- 当前审阅范围未见 SwiftFlow 自动读取 HawkeyePlus 模型注册 API 的连接器。PPT 中的样本反馈、模型交付与标准包是工程协同路径，不宣称自动同步或自动部署已经实现。

硬件示意、检测框和部分模型图片为概念插画；TEM 与流程界面来自已有项目资料。图中没有未经测量的提效、节拍、精度或良率承诺。

## 重建

先确保上一级 HawkeyePlus 图册及参考 SwiftFlow 图册已生成。构建只读取其素材与可编辑 SVG，不修改两套原图册：

```powershell
python docs/Architecture_Visuals/Joint_Platform/build.py
python docs/Architecture_Visuals/Joint_Platform/verify_powerpoint.py
```

依赖与上级相同：Python、Pillow、python-pptx、Playwright、本机 Edge；PowerPoint 验证需要 pywin32 和 Microsoft PowerPoint。不会启动检测软件、训练任务或真实设备。

## English delivery

Each edition is a complete **20-slide presentation** with a formal title slide, clickable agenda, four chapter dividers, twelve content slides, a summary and a discussion slide. It also includes native PowerPoint sections, continuous page numbering and complete speaker notes.

The deck explains **one shared engineering method supported by two complementary platforms**. Its four chapters are Vision & Rationale, SwiftFlow, HawkeyePlus, and Integrated Delivery.

SwiftFlow handles equipment, workflow execution and results. HawkeyePlus handles training data, model development, evaluation and versions. Hardware design remains an engineering activity. Model delivery requires runtime adaptation and joint acceptance; the deck does not claim an automatic cross-platform connector or universal model compatibility.
