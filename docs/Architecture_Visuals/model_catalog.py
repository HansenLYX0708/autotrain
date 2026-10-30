"""Read the actual UI model tables without loading any training runtime."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TABLES = {
    "PaddleDetection": "DETECTION_PRESETS",
    "TorchDet": "TORCH_DET_PRESETS",
    "PaddleSeg": "SEG_ARCHITECTURES",
    "TorchSeg": "TORCH_SEG_ARCHITECTURES",
    "PaddleClas": "CLAS_ARCHITECTURES",
    "TorchAnomaly": "ANOMALY_PRESETS",
}
DISPLAY = {
    "FasterRCNN": "Faster R-CNN", "PPLiteSeg": "PP-LiteSeg",
    "DeepLabV3P": "DeepLabV3+", "SegFormer_B0": "SegFormer-B0",
    "LRASPP": "LR-ASPP", "Patchcore": "PatchCore", "Padim": "PaDiM",
    "Stfpm": "STFPM", "EfficientAd": "EfficientAD",
    "Supersimplenet": "SuperSimpleNet",
    "SSD": "SSD / SSDLite",
}


def python_constant(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing runtime table: " + name)


def load_catalog():
    script = """
const fs=require('fs'),ts=require('typescript'),Module=require('module'),path=require('path');
const f=path.resolve('src/lib/model-yaml.ts'),m=new Module(f);m.paths=module.paths;
m._compile(ts.transpileModule(fs.readFileSync(f,'utf8'),
    {compilerOptions:{module:ts.ModuleKind.CommonJS}}).outputText,f);
const names=%s;
console.log(JSON.stringify(Object.fromEntries(names.map(n=>[n,m.exports[n]]))));
""" % json.dumps(list(TABLES.values()))
    result = subprocess.run(["node", "-e", script], cwd=str(REPO), capture_output=True,
                            text=True, encoding="utf-8", check=True)
    raw = json.loads(result.stdout)
    catalog = {}
    for framework, table in TABLES.items():
        values = raw[table]
        if isinstance(values, list):
            items = [(v if isinstance(v, str) else v["value"], {} if isinstance(v, str) else v) for v in values]
        else:
            items = list(values.items())
        catalog[framework] = [
            dict(key=key, display=DISPLAY.get(key, key), **{
                "backbones": info.get("backbones", []),
                "source_label": info.get("label", key),
                "architecture": info.get("architecture", key),
                "class_path": info.get("classPath"),
                "supports_coco_transfer": info.get("supportsCocoTransfer"),
            })
            for key, info in items
        ]
    # Verify the bundled detection variants against the runtime builder table.
    det = python_constant(REPO/"torchtrain/torchtrain/det/models.py", "DET_ARCHITECTURES")
    assert set(det) == {item["key"] for item in catalog["TorchDet"]}
    for item in catalog["TorchDet"]:
        assert set(item["backbones"]) == set(det[item["key"]])
        item["runtime_builders"] = det[item["key"]]
    seg = python_constant(REPO/"torchtrain/torchtrain/seg/models.py", "SEG_ARCHITECTURES")
    assert {item["key"] for item in catalog["TorchSeg"]}.issubset(seg)
    for item in catalog["TorchSeg"]:
        if item["key"] == "DeepLabV3P":
            item["display"] = "DeepLabV3 (UI: DeepLabV3+)"
            item["runtime_note"] = "DeepLabV3P is a config alias for torchvision DeepLabV3; it is not a DeepLabV3+ decoder."
    return catalog


def write_catalog(catalog):
    source = REPO/"src/lib/model-yaml.ts"
    data = {
        "scope": "Current platform UI presets, not the entire upstream framework catalog or a record of completed training runs.",
        "source": "src/lib/model-yaml.ts",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "frameworks": catalog,
    }
    (HERE/"model_catalog.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# HawkeyePlus — Supported Training Models / 当前训练模型清单", "",
        "Scope: current platform-selectable presets, derived from `src/lib/model-yaml.ts`. "
        "Bundled TorchDet names and backbone variants are cross-checked against the Python builder. "
        "This is not a claim that every preset has been trained or benchmarked on the current machine.", "",
        "范围：当前平台界面可选择的模型预设，直接从源码生成；TorchDet 名称与骨干变体已与训练器构建表核对。"
        "不把上游框架的全部算法或未经核实的自定义 YAML 算作内置预设，也不表示本机已逐一训练验收。", "",
        "| Framework | UI entries / 预设数 | Models / 模型 |",
        "| --- | ---: | --- |",
    ]
    for framework, items in catalog.items():
        lines.append("| {} | {} | {} |".format(framework, len(items), "; ".join(i["display"] for i in items)))
    lines += [
        "", "Counts are framework-specific entries; shared families such as Faster R-CNN and UNet are not unique models across runtimes.",
        "数量按框架分别统计；跨框架重复的 Faster R-CNN、UNet 等不算不同算法家族。", "",
        "## Detection: what it does / 检测任务介绍", "",
        "Detection predicts bounding boxes, classes and confidence scores. The platform supports COCO data, "
        "training, mAP/mAP50 monitoring, independent evaluation and inference previews. "
        "The new industrial-defect diagram is illustrative, not an actual model prediction.", "",
        "目标检测输出目标框、类别与置信度；平台支持 COCO 数据、训练、mAP/mAP50 监控、独立评估和推理预览。"
        "图册中的工业缺陷检测图为概念示意，不是训练后的真实预测结果。", "",
    ]
    for framework, items in catalog.items():
        lines += ["## " + framework, "", "| Display / 展示名称 | Config key / 配置值 | Selectable backbones / 可选骨干 |",
                  "| --- | --- | --- |"]
        for item in items:
            lines.append("| {} | `{}` | {} |".format(
                item["display"], item["key"],
                ", ".join("`"+b+"`" for b in item["backbones"]) or "Architecture-defined / 架构内置"))
        lines.append("")
    lines += [
        "## Support boundaries / 支持边界", "",
        "- **PaddleDetection vs TorchDet:** PP-YOLOE, YOLOv3, PicoDet, RT-DETR and CenterNet are PaddleDetection presets; "
        "they are not supported TorchDet families. / 这些预设不能直接当作 TorchDet 模型使用。",
        "- **SSD / SSDLite:** TorchDet exposes the `SSD` family. `VGG16` selects `ssd300_vgg16`; "
        "`MobileNetV3-Large` selects `ssdlite320_mobilenet_v3_large`. They are two variants of one UI preset.",
        "- **Transfer learning:** TorchDet supports COCO detector-head replacement for Faster R-CNN, RetinaNet and FCOS. "
        "SSD requests fall back to ImageNet backbone initialization; no uniform COCO transfer promise is made.",
        "- **TorchSeg naming:** the UI's `DeepLabV3P` / “DeepLabV3+” uses torchvision **DeepLabV3**. "
        "The runtime also accepts `DeepLabV3`, but it is not an extra UI preset. / 不把该兼容名称当作真正的 V3+ 解码器。",
        "- **PaddleClas variants:** MobileNetV3 small/large are x1.0; PP-LCNet is x1.0; PP-HGNet is small; "
        "Swin Transformer is tiny/patch4/window7/224. Exact configuration names are listed above.",
        "- **Anomaly detection:** the five presets are PatchCore, PaDiM, STFPM, EfficientAD and SuperSimpleNet. "
        "ReverseDistillation is not a current UI preset. Training uses normal images.",
        "- **Custom YAML:** users can import or edit YAML, but compatibility depends on the chosen backend and "
        "installed version. It is not blanket support for every upstream architecture.",
        "- **Environment:** external Paddle repos and matching Python environments must be configured. "
        "Model export formats depend on the backend; PaddleClas has no platform export entrypoint.",
        "",
        "Sources: `src/lib/model-yaml.ts`, `src/lib/frameworks.ts`, "
        "`torchtrain/torchtrain/det/models.py`, `torchtrain/torchtrain/seg/models.py`.",
    ]
    (HERE/"Supported_Models.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
