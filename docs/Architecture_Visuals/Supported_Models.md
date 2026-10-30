# HawkeyePlus — Supported Training Models / 当前训练模型清单

Scope: current platform-selectable presets, derived from `src/lib/model-yaml.ts`. Bundled TorchDet names and backbone variants are cross-checked against the Python builder. This is not a claim that every preset has been trained or benchmarked on the current machine.

范围：当前平台界面可选择的模型预设，直接从源码生成；TorchDet 名称与骨干变体已与训练器构建表核对。不把上游框架的全部算法或未经核实的自定义 YAML 算作内置预设，也不表示本机已逐一训练验收。

| Framework | UI entries / 预设数 | Models / 模型 |
| --- | ---: | --- |
| PaddleDetection | 7 | PP-YOLOE; YOLOv3; PicoDet; RT-DETR; Faster R-CNN; RetinaNet; CenterNet |
| TorchDet | 4 | Faster R-CNN; RetinaNet; FCOS; SSD / SSDLite |
| PaddleSeg | 7 | PP-LiteSeg; BiSeNetV2; OCRNet; UNet; DeepLabV3+; SegFormer-B0; FCN |
| TorchSeg | 4 | UNet; DeepLabV3 (UI: DeepLabV3+); FCN; LR-ASPP |
| PaddleClas | 9 | ResNet18; ResNet34; ResNet50; ResNet101; MobileNetV3_small_x1_0; MobileNetV3_large_x1_0; PPLCNet_x1_0; PPHGNet_small; SwinTransformer_tiny_patch4_window7_224 |
| TorchAnomaly | 5 | PatchCore; PaDiM; STFPM; EfficientAD; SuperSimpleNet |

Counts are framework-specific entries; shared families such as Faster R-CNN and UNet are not unique models across runtimes.
数量按框架分别统计；跨框架重复的 Faster R-CNN、UNet 等不算不同算法家族。

## Detection: what it does / 检测任务介绍

Detection predicts bounding boxes, classes and confidence scores. The platform supports COCO data, training, mAP/mAP50 monitoring, independent evaluation and inference previews. The new industrial-defect diagram is illustrative, not an actual model prediction.

目标检测输出目标框、类别与置信度；平台支持 COCO 数据、训练、mAP/mAP50 监控、独立评估和推理预览。图册中的工业缺陷检测图为概念示意，不是训练后的真实预测结果。

## PaddleDetection

| Display / 展示名称 | Config key / 配置值 | Selectable backbones / 可选骨干 |
| --- | --- | --- |
| PP-YOLOE | `PP-YOLOE` | `CSPResNet` |
| YOLOv3 | `YOLOv3` | `DarkNet`, `MobileNetV1`, `MobileNetV3`, `ResNet` |
| PicoDet | `PicoDet` | `LCNet`, `ESNet` |
| RT-DETR | `RT-DETR` | `ResNet`, `HGNetv2` |
| Faster R-CNN | `FasterRCNN` | `ResNet`, `ResNeXt` |
| RetinaNet | `RetinaNet` | `ResNet` |
| CenterNet | `CenterNet` | `DLA`, `ResNet` |

## TorchDet

| Display / 展示名称 | Config key / 配置值 | Selectable backbones / 可选骨干 |
| --- | --- | --- |
| Faster R-CNN | `FasterRCNN` | `ResNet50-FPN`, `ResNet50-FPN-v2`, `MobileNetV3-Large-FPN`, `MobileNetV3-Large-320-FPN` |
| RetinaNet | `RetinaNet` | `ResNet50-FPN`, `ResNet50-FPN-v2` |
| FCOS | `FCOS` | `ResNet50-FPN` |
| SSD / SSDLite | `SSD` | `VGG16`, `MobileNetV3-Large` |

## PaddleSeg

| Display / 展示名称 | Config key / 配置值 | Selectable backbones / 可选骨干 |
| --- | --- | --- |
| PP-LiteSeg | `PPLiteSeg` | `STDC1`, `STDC2` |
| BiSeNetV2 | `BiSeNetV2` | Architecture-defined / 架构内置 |
| OCRNet | `OCRNet` | `HRNet_W18`, `HRNet_W48` |
| UNet | `UNet` | Architecture-defined / 架构内置 |
| DeepLabV3+ | `DeepLabV3P` | `ResNet50_vd`, `ResNet101_vd` |
| SegFormer-B0 | `SegFormer_B0` | Architecture-defined / 架构内置 |
| FCN | `FCN` | `HRNet_W18`, `HRNet_W48` |

## TorchSeg

| Display / 展示名称 | Config key / 配置值 | Selectable backbones / 可选骨干 |
| --- | --- | --- |
| UNet | `UNet` | Architecture-defined / 架构内置 |
| DeepLabV3 (UI: DeepLabV3+) | `DeepLabV3P` | `ResNet50`, `ResNet101`, `MobileNetV3-Large` |
| FCN | `FCN` | `ResNet50`, `ResNet101` |
| LR-ASPP | `LRASPP` | `MobileNetV3-Large` |

## PaddleClas

| Display / 展示名称 | Config key / 配置值 | Selectable backbones / 可选骨干 |
| --- | --- | --- |
| ResNet18 | `ResNet18` | Architecture-defined / 架构内置 |
| ResNet34 | `ResNet34` | Architecture-defined / 架构内置 |
| ResNet50 | `ResNet50` | Architecture-defined / 架构内置 |
| ResNet101 | `ResNet101` | Architecture-defined / 架构内置 |
| MobileNetV3_small_x1_0 | `MobileNetV3_small_x1_0` | Architecture-defined / 架构内置 |
| MobileNetV3_large_x1_0 | `MobileNetV3_large_x1_0` | Architecture-defined / 架构内置 |
| PPLCNet_x1_0 | `PPLCNet_x1_0` | Architecture-defined / 架构内置 |
| PPHGNet_small | `PPHGNet_small` | Architecture-defined / 架构内置 |
| SwinTransformer_tiny_patch4_window7_224 | `SwinTransformer_tiny_patch4_window7_224` | Architecture-defined / 架构内置 |

## TorchAnomaly

| Display / 展示名称 | Config key / 配置值 | Selectable backbones / 可选骨干 |
| --- | --- | --- |
| PatchCore | `Patchcore` | `wide_resnet50_2`, `resnet18`, `resnet50` |
| PaDiM | `Padim` | `resnet18`, `wide_resnet50_2` |
| STFPM | `Stfpm` | `resnet18`, `wide_resnet50_2` |
| EfficientAD | `EfficientAd` | Architecture-defined / 架构内置 |
| SuperSimpleNet | `Supersimplenet` | `wide_resnet50_2.tv_in1k`, `resnet18.tv_in1k` |

## Support boundaries / 支持边界

- **PaddleDetection vs TorchDet:** PP-YOLOE, YOLOv3, PicoDet, RT-DETR and CenterNet are PaddleDetection presets; they are not supported TorchDet families. / 这些预设不能直接当作 TorchDet 模型使用。
- **SSD / SSDLite:** TorchDet exposes the `SSD` family. `VGG16` selects `ssd300_vgg16`; `MobileNetV3-Large` selects `ssdlite320_mobilenet_v3_large`. They are two variants of one UI preset.
- **Transfer learning:** TorchDet supports COCO detector-head replacement for Faster R-CNN, RetinaNet and FCOS. SSD requests fall back to ImageNet backbone initialization; no uniform COCO transfer promise is made.
- **TorchSeg naming:** the UI's `DeepLabV3P` / “DeepLabV3+” uses torchvision **DeepLabV3**. The runtime also accepts `DeepLabV3`, but it is not an extra UI preset. / 不把该兼容名称当作真正的 V3+ 解码器。
- **PaddleClas variants:** MobileNetV3 small/large are x1.0; PP-LCNet is x1.0; PP-HGNet is small; Swin Transformer is tiny/patch4/window7/224. Exact configuration names are listed above.
- **Anomaly detection:** the five presets are PatchCore, PaDiM, STFPM, EfficientAD and SuperSimpleNet. ReverseDistillation is not a current UI preset. Training uses normal images.
- **Custom YAML:** users can import or edit YAML, but compatibility depends on the chosen backend and installed version. It is not blanket support for every upstream architecture.
- **Environment:** external Paddle repos and matching Python environments must be configured. Model export formats depend on the backend; PaddleClas has no platform export entrypoint.

Sources: `src/lib/model-yaml.ts`, `src/lib/frameworks.ts`, `torchtrain/torchtrain/det/models.py`, `torchtrain/torchtrain/seg/models.py`.
