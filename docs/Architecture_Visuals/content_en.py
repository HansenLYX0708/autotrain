"""English copy for the shared executive layouts; no Chinese text in EN slides."""
COPY = {
    "HawkeyePlus 平台总体架构": "HawkeyePlus Platform Architecture",
    "从工业图像到可交付模型，贯通数据、训练、验证与发布": "Connect industrial images, model training, validation and release in one platform",
    "数据与标注资产": "Data & Annotations",
    "工业图像  +  标签 / 掩码": "Images + Labels / Masks",
    "上传导入": "Import", "标注转换": "Annotate", "样本预览": "Preview",
    "可视化训练中枢": "Visual Training Hub",
    "数据集配置": "Dataset", "模型配置": "Model", "训练参数": "Training",
    "任务编排": "Job Setup", "GPU 选择": "GPU Selection", "训练监控": "Training Monitor",
    "验证与模型交付": "Validation & Delivery",
    "评估与推理": "Evaluate & Infer", "指标评估 · 结果预览": "Metrics · Prediction preview",
    "推理模型导出": "Model Export", "模型转换 · 打包下载": "Convert · Package · Download",
    "模型版本发布": "Model Registry", "按产品管理 · 激活切换": "Product versions · Activation",
    "多框架训练能力": "Multi-framework Training",
    "运行与资产底座": "Compute & Asset Foundation",
    "检测 · 分类 · 分割": "Detect · Classify · Segment",
    "检测 · 分割": "Detect · Segment",
    "无监督异常检测": "Anomaly Detection",
    "算力与环境": "Compute", "数据与资产": "Data & Assets",
    "图像 · 权重 · 日志": "Images · Weights · Logs",
    "统一管理": "Management",
    "平台统一支撑": "Platform Management",
    "项目归属     ·     用户权限     ·     环境检查     ·     GPU 状态     ·     存储配额     ·     操作记录":
        "Project ownership   ·   User roles   ·   Environment checks   ·   GPU status   ·   Storage quotas   ·   Activity logs",
    "HawkeyePlus 从数据到模型的可视化闭环": "HawkeyePlus Visual Model Lifecycle",
    "从数据到模型的可视化闭环": "A Visual Lifecycle: From Data to Models",
    "将分散的准备、训练与验证环节，组织为统一的模型研发工作流":
        "Bring preparation, training and validation into a shared model-development workflow",
    "准备数据": "Prepare Data", "组合配置": "Configure",
    "导入图像与标注": "Import images & labels", "转换格式 · 检查样本": "Convert formats · Review samples",
    "数据 + 模型 + 参数": "Dataset + Model + Parameters", "模板复用 · 可视化配置": "Reusable templates · Visual setup",
    "启动任务 · 查看曲线": "Run jobs · Track training", "进度 · 损失 · 评估指标": "Progress · Loss · Metrics",
    "验证效果": "Validate", "独立评估 · 图像推理": "Evaluate · Preview predictions",
    "预览结果 · 判断可用性": "Inspect results · Assess readiness",
    "模型交付": "Deliver Models", "导出下载 · 注册版本": "Export · Register versions",
    "按产品切换激活模型": "Select the active product model",
    "依据验证结果，迭代数据与配置": "Use validation results to refine data and configuration",
    "四类视觉任务，共用一套平台能力": "Four vision tasks. One shared training platform.",
    "目标检测": "Object Detection", "发现缺陷与目标位置": "Locate objects and defects",
    "图像分类": "Image Classification", "识别类别与质量状态": "Recognize categories and quality",
    "语义分割": "Semantic Segmentation", "刻画区域与结构边界": "Map regions and boundaries",
    "利用正常样本学习异常": "Learn from normal samples",
    "降低使用门槛   /   复用训练经验   /   沉淀模型资产":
        "Simplify adoption   /   Reuse training knowledge   /   Build model assets",
    "Detection｜工业缺陷检测": "Object Detection | Industrial Defect Localization",
    "目标框、类别与置信度，让缺陷位置与类型一目了然":
        "Locate defects and identify their classes with bounding boxes and confidence scores",
    "缺陷定位与识别": "Locate & Classify Defects",
    "检测结果示意，非真实模型预测": "Illustrative output, not a model prediction",
    "输出：目标框 · 类别 · 置信度": "Output: Boxes · Classes · Confidence",
    "单阶段 / 轻量模型": "One-stage / compact families",
    "Transformer / 两阶段模型": "Transformer / two-stage families",
    "COCO 数据与标注": "COCO Images & Labels",
    "训练与指标跟踪": "Training & Metric Tracking",
    "检测结果预览": "Detection Preview",
    "框标注 · 类别管理": "Box labels · Classes",
    "mAP / mAP50 · 训练曲线": "mAP / mAP50 · Training curves",
    "定位缺陷 · 查看类别": "Locate defects · Inspect classes",
    "SSD：VGG16 → SSD300；MobileNetV3 → SSDLite320":
        "SSD variants: VGG16 → SSD300; MobileNetV3 → SSDLite320",
    "预设按框架区分；PP-YOLOE、RT-DETR 等不属于 TorchDet。":
        "Presets are backend-specific; PP-YOLOE and RT-DETR are not TorchDet models.",
    "分割模型｜像素级结构与区域识别": "Segmentation Models | Pixel-level Understanding",
    "以 TEM 分割示例说明任务；模型覆盖轻量网络、编码器解码器与 Transformer":
        "TEM images illustrate segmentation across compact, encoder-decoder and Transformer architectures",
    "像素级预测示例": "Pixel-level Prediction Example",
    "图像区域 · 结构边界 · 标签掩码": "Regions · Boundaries · Label masks",
    "项目内 TEM 分割示例": "TEM segmentation example from this project",
    "PaddleSeg｜7 个预设": "PaddleSeg | 7 Presets",
    "TorchSeg｜4 个预设": "TorchSeg | 4 Presets",
    "TorchSeg 的 DeepLabV3P 使用 torchvision DeepLabV3，并非 V3+ 解码器。":
        "TorchSeg maps DeepLabV3P to torchvision DeepLabV3, not a V3+ decoder.",
    "数据组织": "Dataset Layout",
    "图像 + 掩码 + 训练 / 验证划分": "Images + masks + train / validation splits",
    "评估指标": "Evaluation Metrics",
    "mIoU · Dice · 准确率": "mIoU · Dice · Accuracy",
    "业务用途": "Applications",
    "缺陷区域 · 微观结构 · 边界分析": "Defect regions · Microstructure · Boundaries",
    "分类与异常检测｜补齐视觉任务覆盖": "Classification & Anomaly Detection",
    "从整图类别判断，到基于正常样本的异常识别，复用同一套模型研发流程":
        "From image-level labels to learning normal appearance, reuse the same model-development workflow",
    "PaddleClas｜图像分类": "PaddleClas | Image Classification",
    "TorchAnomaly｜异常检测": "TorchAnomaly | Anomaly Detection",
    "按类别学习，输出整图分类结果": "Learn labeled categories; predict image-level classes",
    "基于正常样本，输出异常分数与热力图": "Learn normal appearance; produce anomaly scores and heatmaps",
    "分类输出示意": "Illustrative classification",
    "异常热力图示意": "Illustrative anomaly heatmap",
    "9 个分类变体": "9 Classification Variants",
    "5 个 anomalib 预设": "5 anomalib Presets",
    "仅列当前界面预设；自定义 YAML 的兼容性取决于所选后端。":
        "Current UI presets only; custom YAML compatibility depends on the selected backend.",
}

NOTES_EN = [
    "HawkeyePlus combines dataset, model and training configurations, then starts the selected Python training backend. "
    "It manages jobs, monitoring, validation and model assets rather than implementing a new training algorithm. "
    "PaddlePaddle covers PaddleDetection/PaddleClas/PaddleSeg; PyTorch covers TorchDet/TorchSeg; anomalib is the PyTorch-based "
    "TorchAnomaly adapter. Compute selection does not imply distributed cluster scheduling.",
    "The workflow combines repository examples and conceptual detection graphics. Napari is the companion desktop segmentation "
    "annotation tool, separate from the web rectangle annotation tool. Export support varies by backend; PaddleClas has no "
    "platform export entrypoint. Model release registers configurations and weights, versions and the active model; it does not "
    "automatically deploy to production machines. The feedback path represents a user revising data and configuration.",
    "Detection models are read directly from DETECTION_PRESETS and TORCH_DET_PRESETS. PaddleDetection has seven preset families; "
    "TorchDet has four. SSD/VGG16 uses SSD300; SSD/MobileNetV3-Large uses SSDLite320. TorchDet supports COCO detector-head transfer "
    "for Faster R-CNN, RetinaNet and FCOS; SSD falls back to ImageNet backbone initialization. PP-YOLOE, YOLOv3, PicoDet, RT-DETR "
    "and CenterNet are not TorchDet families. Defect boxes are illustrative, not inference results or benchmark evidence.",
    "PaddleSeg has seven current UI architectures. TorchSeg has four UI presets. The TorchSeg UI calls one preset DeepLabV3+ "
    "(config key DeepLabV3P), but the runtime uses torchvision DeepLabV3 without a V3+ decoder. DeepLabV3 is also a runtime alias, "
    "not a fifth UI preset. TEM images are project examples, not a benchmark for all models.",
    "The nine PaddleClas choices are listed by exact configuration name in Supported_Models.md. The five TorchAnomaly presets "
    "are PatchCore, PaDiM, STFPM, EfficientAD and SuperSimpleNet. Training uses normal images. ReverseDistillation is not a current "
    "UI preset. Classification and anomaly images are conceptual illustrations. Environment/version compatibility still applies.",
]
