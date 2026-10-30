"""Build HawkeyePlus executive visuals using the SwiftFlow visual conventions.

Content and screenshots come from this repository. The reference contributes
only the WD template, generic illustration primitives, and editable SVG exporter.
"""
import argparse
import base64
import hashlib
import html
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageFont
from playwright.sync_api import sync_playwright
from content_en import COPY, NOTES_EN
from model_catalog import load_catalog, write_catalog

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ASSETS = HERE / "assets"
DEFAULT_REFERENCE = Path(r"G:\projects\AOIPlatform\HwkUltra_g\docs\Architecture_Visuals")
SCREENSHOTS = REPO / "docs/tem_report/assets"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
W, H = 1920, 1080
COLORS = {
    "dark": dict(bg="#061A2C", panel="#0B273D", inset="#12364F",
                 line="#28516A", ink="#F2F8FF", muted="#ACC7DA",
                 cyan="#59DAF5", blue="#7BAAFF", teal="#5AE7CE", tag="#153F58"),
    "light": dict(bg="#F5F9FF", panel="#FFFFFF", inset="#E9F1FC",
                  line="#C6D8EB", ink="#082D48", muted="#4C6B83",
                  cyan="#007D9D", blue="#2564CD", teal="#087F79", tag="#E5EFFB"),
}
SOURCES = [
    "README.md", "docs/ARCHITECTURE.md", "docs/FEATURES.md",
    "src/lib/frameworks.ts", "src/lib/yaml-merge.ts",
    "src/app/api/training-jobs/route.ts",
    "src/app/api/checkpoints/export/route.ts",
    "src/app/api/deployed-models/route.ts",
    "src/app/api/deployed-models/active/route.ts",
    "src/lib/model-yaml.ts",
    "torchtrain/torchtrain/det/models.py",
    "torchtrain/torchtrain/seg/models.py",
    "docs/tem_report/assets/01_raw.png",
    "docs/tem_report/assets/03_added.png",
    "docs/tem_report/assets/11_napari_ui.png",
    "docs/tem_report/assets/12b_train_job_redacted.png",
    "docs/tem_report/assets/13_train_curve.png",
]
NAMES = [(1, "Architecture"), (2, "Lifecycle"), (3, "Detection"),
         (4, "Segmentation"), (5, "Classification_Anomaly")]
LANGUAGES = ("zh", "en")
CATALOG = {}
NOTES = [
    "HawkeyePlus 是工业视觉模型训练平台。数据集、训练参数、模型三份配置"
    "深度合并成任务配置，由平台启动对应 Python 框架的训练子进程。"
    "平台负责配置、启停、监控、验证与模型资产管理，并非另一个训练算法框架。"
    "PaddlePaddle 对应 PaddleDetection/PaddleClas/PaddleSeg，PyTorch 对应"
    "TorchDet/TorchSeg，anomalib 是基于 PyTorch 的 TorchAnomaly 后端。"
    "三者是接入能力的概括，不代表三个完全独立的运行时家族。"
    "运行期是单机服务加训练子进程；图中的任务编排和 GPU 选择不代表分布式集群调度。",
    "页面用项目内已有的 TEM 标注、创建训练任务、训练曲线和分割结果示例"
    "串起工作流。napari 是本仓库配套的桌面分割标注工具，内置 Web 标注"
    "是矩形框工具，二者不混为同一个浏览器页面。"
    "模型导出格式随框架支持情况而定；PaddleClas 无平台导出入口。"
    "发布指登记配置与权重、按产品管理版本并切换激活版本，外部推理服务"
    "可通过 API 查询当前版本，不代表平台自动把模型部署到现场设备。"
    "回路表示人工结合验证结果更新数据与配置，不是自动学习回流。",
    "检测模型直接读取 DETECTION_PRESETS/TORCH_DET_PRESETS。PaddleDetection 有七组预设，"
    "TorchDet 有四组。SSD 的 VGG16 变体使用 SSD300，MobileNetV3-Large 变体使用 SSDLite320。"
    "PP-YOLOE、YOLOv3、PicoDet、RT-DETR、CenterNet 不属于 TorchDet。示意检测框不是实际模型预测。",
    "PaddleSeg 有七个界面预设，TorchSeg 有四个。TorchSeg 中的 DeepLabV3P / DeepLabV3+ "
    "界面名称实际对应 torchvision DeepLabV3，不含 V3+ 解码器。运行器也接受 DeepLabV3 "
    "别名，但它不是额外的第五个界面预设。TEM 图来自当前项目示例。",
    "分类与异常检测预设从源码提取。分类模型按具体变体统计共九项；TorchAnomaly 五项为 "
    "PatchCore、PaDiM、STFPM、EfficientAD、SuperSimpleNet。ReverseDistillation 不是当前界面预设。"
    "图片为概念示意。完整配置值和可选骨干见 Supported_Models.md。",
]


def file_suffix(lang, mode):
    return mode if lang == "zh" else "en_" + mode


def deck_stem(lang, mode):
    return "HawkeyePlus_Architecture_" + ("" if lang == "zh" else "EN_") + mode.title()


@lru_cache(maxsize=256)
def metric_font(size, cjk, bold=False):
    filename = "simhei.ttf" if cjk else "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype("C:/Windows/Fonts/" + filename, round(size*4))


def data_uri(path):
    path = Path(path)
    mime = "image/svg+xml" if path.suffix == ".svg" else "image/png"
    return "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii")


class Canvas:
    def __init__(self, mode, title, lang="zh"):
        self.mode, self.c, self.lang = mode, COLORS[mode], lang
        title = self.translate(title)
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
            "<title>" + html.escape(title) + "</title>",
            '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">'
            f'<path d="M0 0 L8 4.5 L0 9" fill="none" stroke="{self.c["cyan"]}" stroke-width="1.4"/>'
            '</marker></defs><g font-family="Arial, SimHei, sans-serif">',
        ]
        self.rect(0, 0, W, H, self.c["bg"])

    def rect(self, x, y, w, h, fill, stroke="none", radius=0, sw=1):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def translate(self, value):
        if self.lang == "zh":
            return value
        if value in COPY:
            return COPY[value]
        if re.search(r"[\u4e00-\u9fff]", value):
            raise ValueError("Missing English translation: " + value)
        return value

    def text(self, x, y, value, size=28, color=None, weight=400, width=1600, anchor="start"):
        value = self.translate(value)
        cjk = bool(re.search(r"[\u4e00-\u9fff]", value))
        measured = metric_font(size, cjk, weight >= 600).getlength(value)/4
        if measured > width*.96:
            size = round(size*width*.96/measured, 2)
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color or self.c["ink"]}" text-anchor="{anchor}" data-max-width="{width}">'
            + html.escape(value) + "</text>")

    def path(self, d, color=None, sw=2, arrow=False, opacity=1):
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color or self.c["line"]}" stroke-width="{sw}" '
            f'opacity="{opacity}" stroke-linecap="round" stroke-linejoin="round"'
            + (' marker-end="url(#arrow)"' if arrow else "") + "/>")

    def circle(self, x, y, r, color):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')

    def image(self, name, x, y, w, h):
        self.parts.append(f'<image href="{data_uri(ASSETS / (self.mode + "_" + name + ".png"))}" x="{x}" y="{y}" width="{w}" height="{h}"/>')

    def group(self, label):
        self.parts.append("<g><title>" + html.escape(self.translate(label)) + "</title>")

    def end(self):
        self.parts.append("</g>")

    def badge(self, x, y, value, w=150):
        self.rect(x, y, w, 38, self.c["tag"], radius=10)
        self.text(x+w/2, y+26, value, 23, self.c["cyan"], 600, w-12, "middle")

    def header(self, title, subtitle, page):
        p = self.c
        for x in [1350, 1510, 1670]:
            self.path(f"M{x} 0 V45 L{x+44} 89 H1856", p["line"], 1, opacity=.4)
            self.circle(x+44, 89, 3, p["line"])
        self.rect(64, 54, 6, 30, p["cyan"], radius=3)
        self.text(86, 77, "HAWKEYEPLUS  /  INDUSTRIAL VISION AI", 23, p["muted"], 600, 1200)
        self.image("logo", 1742, 50, 111, 47)
        self.text(64, 146, title, 58, weight=600, width=1710)
        self.text(67, 193, subtitle, 28, p["muted"], width=1740)
        self.path("M64 1027 H1856", p["line"], 1)
        self.text(65, 1055, "HawkeyePlus  ·  AI Model Training Platform", 16, p["muted"], width=1050)
        self.text(1748, 1055, "WD CONFIDENTIAL", 15, p["muted"], width=400, anchor="end")
        self.text(1854, 1055, str(page).zfill(2), 16, p["muted"], width=50, anchor="end")

    def save(self, path):
        Path(path).write_text("".join(self.parts) + "</g></svg>", encoding="utf-8")


def material_defs(mode):
    return """
<linearGradient id="metal" x2=".9" y2="1"><stop stop-color="#d6e6ef"/><stop offset=".36" stop-color="#8aaac3"/><stop offset="1" stop-color="#3d5d79"/></linearGradient>
<linearGradient id="darkMetal" x2=".9" y2="1"><stop stop-color="#456681"/><stop offset=".5" stop-color="#213a54"/><stop offset="1" stop-color="#10243b"/></linearGradient>
<linearGradient id="blueMetal" x2=".8" y2="1"><stop stop-color="#48c3f0"/><stop offset=".4" stop-color="#3c8cff"/><stop offset="1" stop-color="#204265"/></linearGradient>
<linearGradient id="chip" x2="1" y2="1"><stop stop-color="#144f91"/><stop offset=".5" stop-color="#2155ae"/><stop offset="1" stop-color="#806cff"/></linearGradient>
<linearGradient id="paper" x2=".3" y2="1"><stop stop-color="#edf6fa"/><stop offset="1" stop-color="#a2c3d5"/></linearGradient>
<radialGradient id="halo"><stop stop-color="#2DA7FF" stop-opacity=".25"/><stop offset="1" stop-color="#2DA7FF" stop-opacity="0"/></radialGradient>
"""


def wrap_asset(body, mode, width=240, height=220):
    if mode == "light":
        body = body.replace('fill="#020b18" opacity=".6"', 'fill="#00263e" opacity=".12"')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width*4}" height="{height*4}" '
            f'viewBox="0 0 {width} {height}"><defs>{material_defs(mode)}</defs>'
            f'<g font-family="Arial,sans-serif">{body}</g></svg>')


def neural_core(mode):
    """A new neural-network training illustration, with no SwiftFlow elements."""
    edge = "#318DE0" if mode == "light" else "#75E5FF"
    s = ['<ellipse cx="300" cy="156" rx="278" ry="121" fill="url(#halo)"/>']
    for y, rx, ry in [(191, 184, 59), (173, 173, 55)]:
        s.append(f'<path d="M{300-rx} {y} L300 {y+ry} {300+rx} {y} V{y+16} L300 {y+ry+16} {300-rx} {y+16} Z" fill="url(#darkMetal)" stroke="#437BAB"/>')
        s.append(f'<path d="M300 {y-ry} L{300+rx} {y} 300 {y+ry} {300-rx} {y} Z" fill="url(#chip)" stroke="{edge}" stroke-width="1.8"/>')
    nodes = []
    for col, ys in enumerate([[91, 127, 163], [58, 99, 140, 181], [58, 99, 140, 181], [91, 127, 163]]):
        nodes.append([(200 + col*66, y) for y in ys])
    for left, right in zip(nodes, nodes[1:]):
        for x1, y1 in left:
            for x2, y2 in right:
                s.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{edge}" stroke-opacity=".32" stroke-width="1.3"/>')
    for col in nodes:
        for x, y in col:
            s.append(f'<circle cx="{x}" cy="{y}" r="8" fill="#216CC0" stroke="#A2F2FF" stroke-width="2"/>')
            s.append(f'<circle cx="{x}" cy="{y}" r="3" fill="#C6FCFF"/>')
    for side in [-1, 1]:
        for i in range(4):
            x = 300 + side * (172 + i*5)
            y = 172 + i*13
            ex = x + side*60
            s.append(f'<path d="M{x} {y} L{ex} {y+17} H{ex+side*23}" stroke="{edge}" fill="none" stroke-width="1.4"/>')
            s.append(f'<circle cx="{ex+side*23}" cy="{y+17}" r="3" fill="{edge}"/>')
    return wrap_asset("".join(s), mode, 600, 280)


def dataset_art(mode):
    # Show two different task types rather than a segmentation-only platform.
    raw = data_uri(SCREENSHOTS/"03_added.png")
    overlay = data_uri(ASSETS/f"{mode}_detection_scene.png")
    s = '<ellipse cx="210" cy="239" rx="191" ry="14" fill="#00263e" opacity=".16"/>'
    s += '<rect x="21" y="25" width="210" height="186" rx="12" fill="#274C68" stroke="#89C6DE" stroke-width="2"/>'
    s += f'<image href="{raw}" x="29" y="34" width="194" height="169" preserveAspectRatio="xMidYMid slice"/>'
    s += '<rect x="174" y="66" width="210" height="177" rx="12" fill="#1E4C71" stroke="#6FDDE6" stroke-width="2"/>'
    s += f'<image href="{overlay}" x="182" y="74" width="194" height="161" preserveAspectRatio="xMidYMid slice"/>'
    return wrap_asset(s, mode, 420, 260)


def industrial_scene(mode, task="detection"):
    """Original industrial-part concept art; labels never imply a real prediction."""
    s = [
        '<defs><linearGradient id="steel" x2=".7" y2="1"><stop stop-color="#CADBE8"/>'
        '<stop offset=".4" stop-color="#859FB5"/><stop offset=".7" stop-color="#B7CDD9"/>'
        '<stop offset="1" stop-color="#63849D"/></linearGradient>'
        '<radialGradient id="heat"><stop stop-color="#FFFF84"/><stop offset=".3" stop-color="#FF9B44"/>'
        '<stop offset=".6" stop-color="#FF4F53" stop-opacity=".8"/><stop offset="1" stop-color="#AA5EFF" stop-opacity="0"/></radialGradient></defs>',
        '<rect x="4" y="4" width="632" height="382" rx="20" fill="#102B43" stroke="#5189A7" stroke-width="2"/>',
        '<path d="M52 56 H588 M52 332 H588 M52 56 V332 M588 56 V332" stroke="#2B4B64" fill="none"/>',
        '<ellipse cx="320" cy="319" rx="219" ry="33" fill="#061826" opacity=".65"/>',
        '<path d="M110 98 H526 V288 L508 304 H109 Z" fill="#41617A" stroke="#A3B8C6"/>',
        '<path d="M110 85 H510 L526 98 H126 Z" fill="#D0E1EC"/>',
        '<rect x="110" y="85" width="400" height="205" rx="13" fill="url(#steel)" stroke="#DAEBF2" stroke-width="2"/>',
    ]
    for y in range(101, 281, 12):
        s.append(f'<path d="M133 {y} H487" stroke="#365874" stroke-opacity=".16"/>')
    for x, y in [(133, 108), (487, 108), (133, 266), (487, 266)]:
        s.append(f'<circle cx="{x}" cy="{y}" r="10" fill="#344B60" stroke="#D4E5EC" stroke-width="2"/>')
        s.append(f'<path d="M{x-5} {y} H{x+5}" stroke="#A9C0D0" stroke-width="2"/>')
    s.append('<rect x="185" y="137" width="245" height="101" rx="22" fill="#49677E" stroke="#D7E6EE" stroke-width="3"/>')
    s.append('<rect x="196" y="147" width="223" height="80" rx="15" fill="#193248"/>')
    s.append('<path d="M262 111 L289 129 M267 109 L292 126 M272 108 L295 123" stroke="#EFF7FA" stroke-width="2"/>')
    s.append('<path d="M463 282 L474 264 L486 276 L490 290 H462 Z" fill="#1C384C" stroke="#E0E7E9"/>')
    if task == "detection":
        for x, y, w, h, label in [(241, 88, 75, 53, "SCRATCH"), (449, 248, 56, 55, "EDGE CHIP")]:
            s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="none" stroke="#5BF0C2" stroke-width="3"/>')
            s.append(f'<rect x="{x}" y="{y-23}" width="{max(86,len(label)*11)}" height="23" rx="3" fill="#134F49"/>')
            s.append(f'<text x="{x+6}" y="{y-7}" fill="#BFFFE4" font-size="14" font-weight="700">{label}</text>')
    elif task == "anomaly":
        s.append('<ellipse cx="278" cy="119" rx="63" ry="48" fill="url(#heat)"/>')
        s.append('<ellipse cx="475" cy="278" rx="50" ry="43" fill="url(#heat)"/>')
    else:
        s.append('<rect x="431" y="31" width="177" height="38" rx="8" fill="#124D45" stroke="#5BE5B8"/>')
        s.append('<text x="519" y="57" text-anchor="middle" font-size="21" font-weight="700" fill="#B2FFE1">DEFECT CLASS</text>')
    return wrap_asset("".join(s), mode, 640, 390)


def model_art(mode, registry=False):
    s = '<ellipse cx="120" cy="194" rx="100" ry="12" fill="#00263e" opacity=".14"/>'
    s += '<path d="M45 69 L119 34 L196 69 L120 105 Z" fill="#68B4DF" stroke="#B4E5F3"/>'
    s += '<path d="M45 69 L120 105 V187 L45 149 Z" fill="#245883" stroke="#88C5E0"/>'
    s += '<path d="M120 105 L196 69 V150 L120 187 Z" fill="#193D66" stroke="#88C5E0"/>'
    s += '<path d="M82 52 L157 88 V125 L139 135 V97 L64 61 Z" fill="#B4CFE7" opacity=".75"/>'
    if registry:
        s += '<circle cx="181" cy="152" r="32" fill="#0C695E" stroke="#78EDC8" stroke-width="3"/>'
        s += '<path d="M166 152 L177 163 L197 140" stroke="#B9FFDF" stroke-width="5" fill="none"/>'
    else:
        s += '<path d="M168 173 V124 H153 L180 97 L207 124 H192 V173 Z" fill="#5FDCF1" stroke="#BEF7FF" stroke-width="2"/>'
    return wrap_asset(s, mode)


def runtime_art(mode, kind):
    palette = {"paddle": "#58C4F1", "torch": "#F6A078", "anomaly": "#A394FF"}
    accent = palette[kind]
    s = '<ellipse cx="120" cy="193" rx="99" ry="14" fill="#00263e" opacity=".14"/>'
    s += '<path d="M29 70 L118 24 L208 70 V155 L120 199 L29 155 Z" fill="#193B60" stroke="#7CA5C8" stroke-width="2"/>'
    s += f'<path d="M29 70 L119 115 L208 70 L118 24 Z" fill="url(#chip)" stroke="{accent}" stroke-width="2"/>'
    s += f'<path d="M119 115 V199 M29 144 L119 187 L208 144" stroke="{accent}" fill="none" stroke-width="2"/>'
    points = [(78, 67), (119, 47), (161, 67), (119, 87)]
    if kind == "anomaly":
        s += f'<ellipse cx="119" cy="69" rx="31" ry="18" fill="none" stroke="{accent}" stroke-width="4"/>'
        s += '<circle cx="132" cy="65" r="7" fill="#FC9B7C"/>'
    else:
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]:
            x1, y1 = points[a]
            x2, y2 = points[b]
            s += f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{accent}" stroke-width="2"/>'
        for x, y in points:
            s += f'<circle cx="{x}" cy="{y}" r="5" fill="{accent}"/>'
    return wrap_asset(s, mode)


def render(page, svg_path, png_path, width, height, transparent=False):
    page.set_viewport_size({"width": width, "height": height})
    page.set_content('<html><head><style>html,body{margin:0;background:transparent}svg{display:block;width:100%;height:100%}</style></head><body>'
                     + Path(svg_path).read_text(encoding="utf-8") + "</body></html>")
    page.evaluate("document.fonts.ready")
    page.evaluate("()=>Promise.all(Array.from(document.images).map(i=>i.decode().catch(()=>{})))")
    page.screenshot(path=str(png_path), omit_background=transparent)


def prepare_assets(browser, reference, art):
    page = browser.new_page(device_scale_factor=1)
    # Use only the job dialog's controls; exclude local paths and background rows.
    with Image.open(SCREENSHOTS/"12b_train_job_redacted.png") as im:
        im.crop((326, 105, 836, 536)).save(ASSETS/"job_controls.png")
    with Image.open(SCREENSHOTS/"11_napari_ui.png") as im:
        im.crop((0, 25, im.width, im.height-22)).save(ASSETS/"annotation_screen.png")
    for mode in COLORS:
        for name, task in [("detection_scene", "detection"), ("classification_scene", "classification"),
                           ("anomaly_scene", "anomaly")]:
            source = ASSETS/f"{mode}_{name}.svg"
            source.write_text(industrial_scene(mode, task), encoding="utf-8")
            render(page, source, source.with_suffix(".png"), 1280, 780, transparent=True)
        bodies = {
            "server": art.server(), "storage": art.database(), "folder": art.folder(),
            "monitor": art.monitor(data_uri(SCREENSHOTS/"13_train_curve.png")),
            "validation": art.monitor(data_uri(ASSETS/f"{mode}_detection_scene.png")),
            "segmentation": art.monitor(data_uri(SCREENSHOTS/"03_added.png")),
            "annotation": art.monitor(data_uri(ASSETS/"annotation_screen.png")),
            "config": art.monitor(data_uri(ASSETS/"job_controls.png")),
        }
        files = {name: wrap_asset(body, mode) for name, body in bodies.items()}
        files.update({
            "data": dataset_art(mode), "core": neural_core(mode),
            "export": model_art(mode), "registry": model_art(mode, True),
            **{name: runtime_art(mode, name) for name in ["paddle", "torch", "anomaly"]},
        })
        for name, svg in files.items():
            source = ASSETS / f"{mode}_{name}.svg"
            source.write_text(svg, encoding="utf-8")
            size = (1680, 1040) if name == "data" else (1800, 840) if name == "core" else (960, 880)
            render(page, source, source.with_suffix(".png"), *size, transparent=True)
        render(page, reference/f"template_reference/logo_{mode}.svg",
               ASSETS/f"{mode}_logo.png", 450, 180, transparent=True)
    page.close()


def architecture(mode, lang="zh"):
    c = Canvas(mode, "HawkeyePlus 平台总体架构", lang)
    p = c.c
    c.header("HawkeyePlus 平台总体架构", "从工业图像到可交付模型，贯通数据、训练、验证与发布", 1)
    c.group("数据与标注资产")
    c.rect(64, 237, 472, 438, p["panel"], p["line"], 22)
    c.text(91, 280, "01", 22, p["cyan"], 600, 60)
    c.text(145, 282, "数据与标注资产", 32, weight=600, width=365)
    c.image("data", 83, 303, 434, 265)
    c.text(300, 582, "工业图像  +  标签 / 掩码", 27, p["ink"], 600, 420, "middle")
    for x, title in [(91, "上传导入"), (236, "标注转换"), (381, "样本预览")]:
        c.badge(x, 604, title, 127)
    c.end()

    c.group("可视化训练中枢")
    c.rect(581, 237, 717, 438, p["panel"], p["line"], 22)
    c.rect(795, 237, 288, 3, p["cyan"], radius=1)
    c.text(940, 282, "可视化训练中枢", 36, weight=600, width=650, anchor="middle")
    for x, label in [(619, "数据集配置"), (838, "模型配置"), (1057, "训练参数")]:
        c.rect(x, 311, 203, 49, p["inset"], p["line"], 10)
        c.text(x+101.5, 344, label, 26, width=191, anchor="middle")
    c.path("M720 362 V380 H1158 V362 M940 362 V399", p["cyan"], 1.8)
    c.image("core", 631, 378, 618, 243)
    for x, label in [(610, "任务编排"), (840, "GPU 选择"), (1070, "训练监控")]:
        c.badge(x, 620, label, 200)
    c.end()

    c.group("验证与模型交付")
    c.rect(1343, 237, 513, 438, p["panel"], p["line"], 22)
    c.text(1370, 280, "02", 22, p["cyan"], 600, 60)
    c.text(1424, 282, "验证与模型交付", 32, weight=600, width=400)
    for i, (image, title, desc) in enumerate([
        ("validation", "评估与推理", "指标评估 · 结果预览"),
        ("export", "推理模型导出", "模型转换 · 打包下载"),
        ("registry", "模型版本发布", "按产品管理 · 激活切换"),
    ]):
        y = 318 + 113*i
        c.image(image, 1364, y-10, 123, 109)
        c.text(1498, y+30, title, 29, weight=600, width=328)
        c.text(1498, y+69, desc, 23, p["muted"], width=328)
        if i < 2:
            c.path(f"M1498 {y+90} H1827", p["line"], 1)
    c.end()
    c.path("M541 448 H573", p["cyan"], 2.5, True)
    c.path("M1306 448 H1336", p["cyan"], 2.5, True)
    c.path("M940 677 V702", p["cyan"], 2.5, True)

    c.group("多框架训练能力")
    c.rect(64, 710, 1068, 51, p["inset"], p["line"], 12)
    c.text(91, 745, "多框架训练能力", 28, p["cyan"], 600, 900)
    c.end()
    c.group("运行与资产底座")
    c.rect(1160, 710, 696, 51, p["inset"], p["line"], 12)
    c.text(1187, 745, "运行与资产底座", 28, p["cyan"], 600, 620)
    c.end()
    for x, image, title, desc in [
        (64, "paddle", "PaddlePaddle", "检测 · 分类 · 分割"),
        (420, "torch", "PyTorch", "检测 · 分割"),
        (776, "anomaly", "anomalib", "无监督异常检测"),
    ]:
        c.group(title)
        c.image(image, x+8, 780, 151, 141)
        c.text(x+165, 829, title, 28, weight=600, width=191)
        c.text(x+165, 871, desc, 22, p["muted"], width=192)
        c.end()
    for x, image, title, desc in [
        (1160, "server", "算力与环境", "GPU / Python"),
        (1508, "storage", "数据与资产", "图像 · 权重 · 日志"),
    ]:
        c.group(title)
        c.image(image, x+4, 779, 143, 142)
        c.text(x+151, 829, title, 27, weight=600, width=188)
        c.text(x+151, 871, desc, 22, p["muted"], width=188)
        c.end()
    c.group("平台统一支撑")
    c.rect(64, 952, 1792, 55, p["panel"], p["line"], 12)
    c.circle(92, 980, 5, p["teal"])
    c.text(113, 990, "统一管理", 24, p["teal"], 600, 200)
    c.text(299, 990, "项目归属     ·     用户权限     ·     环境检查     ·     GPU 状态     ·     存储配额     ·     操作记录", 23, p["muted"], width=1510)
    c.end()
    return c


def lifecycle(mode, lang="zh"):
    c = Canvas(mode, "HawkeyePlus 从数据到模型的可视化闭环", lang)
    p = c.c
    c.header("从数据到模型的可视化闭环", "将分散的准备、训练与验证环节，组织为统一的模型研发工作流", 2)
    cards = [
        ("准备数据", "annotation", "导入图像与标注", "转换格式 · 检查样本"),
        ("组合配置", "config", "数据 + 模型 + 参数", "模板复用 · 可视化配置"),
        ("训练监控", "monitor", "启动任务 · 查看曲线", "进度 · 损失 · 评估指标"),
        ("验证效果", "validation", "独立评估 · 图像推理", "预览结果 · 判断可用性"),
        ("模型交付", "registry", "导出下载 · 注册版本", "按产品切换激活模型"),
    ]
    for i, (title, image, desc1, desc2) in enumerate(cards):
        x = 64 + i*364
        c.group(title)
        c.rect(x, 243, 336, 438, p["panel"], p["line"], 20)
        c.text(x+25, 285, f"0{i+1}", 23, p["cyan"], 600, 60)
        c.text(x+82, 287, title, 32, weight=600, width=228)
        c.image(image, x+22, 313, 292, 238)
        c.path(f"M{x+25} 563 H{x+311}", p["line"], 1)
        c.text(x+168, 608, desc1, 24, width=314, anchor="middle")
        c.text(x+168, 646, desc2, 23, p["muted"], width=314, anchor="middle")
        c.end()
        if i < 4:
            c.path(f"M{x+341} 458 H{x+359}", p["cyan"], 2, True)
    # Human-directed iteration; this is not an automatic training feedback loop.
    c.path("M1324 685 V726 H232 V688", p["cyan"], 2, True)
    c.rect(480, 704, 608, 42, p["bg"], radius=5)
    c.text(784, 733, "依据验证结果，迭代数据与配置", 25, p["cyan"], width=590, anchor="middle")
    c.text(64, 801, "四类视觉任务，共用一套平台能力", 33, weight=600, width=1700)
    for x, title, desc in [
        (64, "目标检测", "发现缺陷与目标位置"),
        (519, "图像分类", "识别类别与质量状态"),
        (974, "语义分割", "刻画区域与结构边界"),
        (1429, "无监督异常检测", "利用正常样本学习异常"),
    ]:
        c.group(title)
        c.rect(x, 827, 427, 113, p["inset"], p["line"], 14)
        c.rect(x+20, 851, 4, 60, p["cyan"], radius=2)
        c.text(x+42, 873, title, 28, weight=600, width=366)
        c.text(x+42, 915, desc, 24, p["muted"], width=366)
        c.end()
    c.text(960, 993, "降低使用门槛   /   复用训练经验   /   沉淀模型资产", 28, p["teal"], 600, 1700, "middle")
    return c


def draw_model_names(c, framework, x, y, width, size=32, spacing=47):
    """Wrap the actual model list; each family remains a complete name."""
    lines, line = [], ""
    for item in CATALOG[framework]:
        name = item["display"]
        trial = line + ("  ·  " if line else "") + name
        if line and metric_font(size, False, True).getlength(trial)/4 > width*.96:
            lines.append(line)
            line = name
        else:
            line = trial
    if line:
        lines.append(line)
    for i, line in enumerate(lines):
        c.text(x, y+spacing*i, line, size, weight=600, width=width)
    return len(lines)


def feature_strip(c, entries, top=811):
    for i, (image, title, desc) in enumerate(entries):
        x = 64+i*607
        c.group(title)
        c.rect(x, top, 578, 137, c.c["panel"], c.c["line"], 18)
        c.image(image, x+12, top+8, 128, 117)
        c.text(x+157, top+51, title, 27, weight=600, width=399)
        c.text(x+157, top+98, desc, 24, c.c["muted"], width=399)
        c.end()


def detection(mode, lang="zh"):
    c = Canvas(mode, "Detection｜工业缺陷检测", lang)
    p = c.c
    c.header("Detection｜工业缺陷检测", "目标框、类别与置信度，让缺陷位置与类型一目了然", 3)
    c.group("Detection")
    c.rect(64, 237, 660, 529, p["panel"], p["line"], 22)
    c.text(92, 286, "缺陷定位与识别", 33, weight=600, width=600)
    c.image("detection_scene", 84, 321, 620, 378)
    c.text(394, 722, "输出：目标框 · 类别 · 置信度", 27, p["cyan"], 600, 620, "middle")
    c.text(394, 751, "检测结果示意，非真实模型预测", 20, p["muted"], width=620, anchor="middle")
    c.end()
    for framework, top in [("PaddleDetection", 237), ("TorchDet", 509)]:
        c.group(framework)
        c.rect(755, top, 1101, 257, p["panel"], p["line"], 22)
        c.text(787, top+49, framework, 35, p["cyan"], 600, 790)
        c.badge(1651, top+23, f"{len(CATALOG[framework])} presets", 174)
        draw_model_names(c, framework, 787, top+106, 1031, 34, 49)
        if framework == "TorchDet":
            c.text(787, top+229, "SSD：VGG16 → SSD300；MobileNetV3 → SSDLite320", 24, p["muted"], width=1031)
        c.end()
    feature_strip(c, [
        ("folder", "COCO 数据与标注", "框标注 · 类别管理"),
        ("monitor", "训练与指标跟踪", "mAP / mAP50 · 训练曲线"),
        ("validation", "检测结果预览", "定位缺陷 · 查看类别"),
    ])
    c.text(960, 993, "预设按框架区分；PP-YOLOE、RT-DETR 等不属于 TorchDet。", 24, p["muted"], width=1792, anchor="middle")
    return c


def segmentation(mode, lang="zh"):
    c = Canvas(mode, "分割模型｜像素级结构与区域识别", lang)
    p = c.c
    c.header("分割模型｜像素级结构与区域识别",
             "以 TEM 分割示例说明任务；模型覆盖轻量网络、编码器解码器与 Transformer", 4)
    c.group("Segmentation example")
    c.rect(64, 237, 660, 529, p["panel"], p["line"], 22)
    c.text(92, 286, "像素级预测示例", 33, weight=600, width=600)
    c.image("segmentation", 89, 301, 610, 374)
    c.text(394, 713, "图像区域 · 结构边界 · 标签掩码", 26, p["cyan"], 600, 620, "middle")
    c.text(394, 750, "项目内 TEM 分割示例", 21, p["muted"], width=620, anchor="middle")
    c.end()
    for framework, top in [("PaddleSeg", 237), ("TorchSeg", 509)]:
        c.group(framework)
        c.rect(755, top, 1101, 257, p["panel"], p["line"], 22)
        c.text(787, top+49, framework, 35, p["cyan"], 600, 790)
        c.badge(1651, top+23, f"{len(CATALOG[framework])} presets", 174)
        draw_model_names(c, framework, 787, top+110, 1031, 32, 49)
        if framework == "TorchSeg":
            # A UI alias must not be advertised as a different decoder.
            c.text(787, top+230, "TorchSeg 的 DeepLabV3P 使用 torchvision DeepLabV3，并非 V3+ 解码器。",
                   23, p["muted"], width=1031)
        c.end()
    feature_strip(c, [
        ("folder", "数据组织", "图像 + 掩码 + 训练 / 验证划分"),
        ("monitor", "评估指标", "mIoU · Dice · 准确率"),
        ("segmentation", "业务用途", "缺陷区域 · 微观结构 · 边界分析"),
    ])
    return c


def classification_anomaly(mode, lang="zh"):
    c = Canvas(mode, "分类与异常检测｜补齐视觉任务覆盖", lang)
    p = c.c
    c.header("分类与异常检测｜补齐视觉任务覆盖",
             "从整图类别判断，到基于正常样本的异常识别，复用同一套模型研发流程", 5)
    for x, framework, title, subtitle, asset, caption in [
        (64, "PaddleClas", "PaddleClas｜图像分类", "按类别学习，输出整图分类结果",
         "classification_scene", "分类输出示意"),
        (974, "TorchAnomaly", "TorchAnomaly｜异常检测", "基于正常样本，输出异常分数与热力图",
         "anomaly_scene", "异常热力图示意"),
    ]:
        c.group(framework)
        c.rect(x, 237, 882, 711, p["panel"], p["line"], 22)
        c.text(x+29, 287, title, 34, weight=600, width=824)
        c.text(x+29, 333, subtitle, 26, p["muted"], width=824)
        c.image(asset, x+169, 355, 544, 300)
        c.text(x+441, 681, caption, 21, p["muted"], width=820, anchor="middle")
        c.path(f"M{x+29} 705 H{x+853}", p["line"], 1)
        c.text(x+29, 749, "9 个分类变体" if framework == "PaddleClas" else "5 个 anomalib 预设",
               29, p["cyan"], 600, 800)
        if framework == "PaddleClas":
            # Compact grouping keeps all nine exact variants visible in the source catalog.
            expected = {"ResNet18", "ResNet34", "ResNet50", "ResNet101",
                        "MobileNetV3_small_x1_0", "MobileNetV3_large_x1_0", "PPLCNet_x1_0",
                        "PPHGNet_small", "SwinTransformer_tiny_patch4_window7_224"}
            assert {v["key"] for v in CATALOG[framework]} == expected
            rows = ["ResNet18 · ResNet34 · ResNet50 · ResNet101",
                    "MobileNetV3 Small / Large (x1.0)",
                    "PP-LCNet x1.0 · PP-HGNet Small",
                    "Swin Transformer Tiny (patch4 / window7 / 224)"]
            for i, row in enumerate(rows):
                c.text(x+29, 794+i*40, row, 27, weight=600, width=824)
        else:
            draw_model_names(c, framework, x+29, 805, 824, 32, 58)
        c.end()
    c.text(960, 993, "仅列当前界面预设；自定义 YAML 的兼容性取决于所选后端。",
           24, p["muted"], width=1792, anchor="middle")
    return c


def build_deck(mode, lang, files, reference, exporter):
    style = json.loads((reference/"template_reference/style.json").read_text(encoding="utf-8"))
    exporter.CACHE = ASSETS/"ppt_cache"
    exporter.ensure_assets(files)
    prs, layout = exporter.new_presentation(style, mode)
    prs.core_properties.title = "HawkeyePlus Platform Architecture & Training Models" if lang == "en" else "HawkeyePlus 平台架构与训练模型"
    prs.core_properties.subject = "Industrial vision: data, training, validation and model portfolio"
    prs.core_properties.author = "HawkeyePlus"
    stats = []
    for i, path in enumerate(files):
        slide = prs.slides.add_slide(layout)
        stats.append(exporter.add_editable_svg(prs, slide, path))
        slide.notes_slide.notes_text_frame.text = (
            (NOTES_EN[i] + "\nEditable native text, panels and lines; independent illustration images. "
             "UI presets do not imply completed hardware validation. Full model/backbone list: Supported_Models.md.\nSources:\n"
             if lang == "en" else NOTES[i] + "\n文字、模块、连接线可编辑，配图可替换。"
             "当前预设不等于已逐一在本机训练验证；完整模型与骨干列表见 Supported_Models.md。\n来源：\n")
            + "\n".join(SOURCES))
    output = HERE/f"{deck_stem(lang, mode)}_Editable.pptx"
    prs.save(output)
    return {"file": output.name, "language": lang, "theme": mode, "slides": stats}


def gallery():
    sections = []
    titles = {
        "zh": ["平台总体架构", "模型研发闭环", "Detection：工业缺陷检测", "分割模型", "分类与异常检测"],
        "en": ["Platform Architecture", "Visual Model Lifecycle", "Object Detection", "Segmentation Models", "Classification & Anomaly Detection"],
    }
    for lang in ["en", "zh"]:
        for mode, theme in [("dark", "Dark / 深蓝"), ("light", "Light / 明色")]:
            label = ("English" if lang == "en" else "中文") + " · " + theme
            deck = deck_stem(lang, mode)
            sections.append(f'<section id="{lang}_{mode}"><h2>{label}</h2><p><a href="{deck}_Editable.pptx">Editable PPT / 可编辑 PPT</a>'
                            f' · <a href="{deck}.pdf">PDF</a></p><div class="grid">')
            for i, name in NAMES:
                stem = f"{i:02d}_{name}_{file_suffix(lang, mode)}"
                title = titles[lang][i-1]
                sections.append(f'<article><a href="png/{stem}.png"><img loading="lazy" src="png/{stem}.png" alt="{title}"></a>'
                                f'<h3>{i:02d} · {title}</h3><p><a href="png/{stem}.png">4K PNG</a> · <a href="svg/{stem}.svg">SVG</a></p></article>')
            sections.append("</div></section>")
    catalog_rows = "".join(
        "<tr><th>"+html.escape(framework)+"</th><td>"+html.escape("; ".join(m["display"] for m in models))+"</td></tr>"
        for framework, models in CATALOG.items())
    (HERE/"index.html").write_text("""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>HawkeyePlus Architecture & Training Models</title>
<style>*{box-sizing:border-box}body{margin:0;background:#061a2c;color:#f2f8ff;font:16px/1.6 'Segoe UI','Microsoft YaHei',sans-serif}
main{max-width:1480px;margin:56px auto;padding:0 32px}small{color:#59daf5;letter-spacing:3px}
h1{font-size:42px;margin:12px 0}header p{color:#acc7da}h2{font-size:26px;margin:42px 0 8px}
a{color:#70d5ff;text-decoration:none}a:hover{text-decoration:underline}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}
article{overflow:hidden;border:1px solid #28516a;border-radius:16px;background:#0b273d}img{width:100%;display:block}
nav{display:flex;gap:24px;flex-wrap:wrap}table{border-collapse:collapse;width:100%;margin-top:24px}th,td{border-bottom:1px solid #28516a;padding:14px;text-align:left}th{color:#59daf5;width:190px}
article h3,article p{margin:15px 22px}footer{margin:40px 0;color:#acc7da}@media(max-width:850px){.grid{grid-template-columns:1fr}h1{font-size:30px}}</style>
<main><header><small>HAWKEYEPLUS / INDUSTRIAL VISION AI</small><h1>Architecture & Training Models</h1>
<p>中英文架构与模型图册 · English / Chinese · 5 slides per edition · 4K PNG · Editable PowerPoint</p>
<nav><a href="Joint_Platform/">SwiftFlow + HawkeyePlus 结合汇报</a><a href="#en_dark">English · Dark</a><a href="#en_light">English · Light</a><a href="#zh_dark">中文 · 深蓝</a><a href="#zh_light">中文 · 明色</a><a href="#models">Model Catalog</a></nav>
</header>""" + "".join(sections) + """
<section id="models"><h2>Current UI Model Presets / 当前模型预设</h2>
<p><a href="Supported_Models.md">完整模型与骨干清单 / Full Model & Backbone List</a> · <a href="model_catalog.json">Source-derived JSON</a></p>
<table>""" + catalog_rows + """</table>
<p>Backend-specific UI presets; custom YAML depends on the installed backend. See the full list for the TorchSeg DeepLabV3 naming caveat.</p></section>
<footer>Native editable text, panels and connectors; independent replaceable images.<br>
Detection, classification and anomaly illustrations are conceptual. TEM screenshots are project examples, not benchmarks.<br>
文字、模块与连线可编辑；检测、分类和异常图为概念示意，TEM 示例不作为性能基准。</footer></main></html>""", encoding="utf-8")


def contact_sheet():
    sheet = Image.new("RGB", (2400, 1080), "#061A2C")
    for l, lang in enumerate(["en", "zh"]):
        for m, mode in enumerate(COLORS):
            edition = Image.new("RGB", (1920, 720), "#061A2C")
            for x, (i, name) in enumerate(NAMES):
                with Image.open(HERE/"png"/f"{i:02d}_{name}_{file_suffix(lang, mode)}.png") as im:
                    sheet.paste(im.convert("RGB").resize((480, 270), Image.Resampling.LANCZOS), (480*x, 270*(l*2+m)))
                    edition.paste(im.convert("RGB").resize((640, 360), Image.Resampling.LANCZOS), ((x%3)*640, (x//3)*360))
            edition.save(HERE/f"Preview_{lang}_{mode}.jpg", quality=94)
    sheet.save(HERE/"Preview.jpg", quality=94)


def main():
    global CATALOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    reference = args.reference_dir.resolve()
    sys.path.insert(0, str(reference))
    import editable_ppt
    import illustrations
    for path in [ASSETS, HERE/"png", HERE/"svg", HERE/"verification"]:
        path.mkdir(exist_ok=True)
    CATALOG = load_catalog()
    write_catalog(CATALOG)
    decks = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=EDGE, headless=True)
        prepare_assets(browser, reference, illustrations)
        page = browser.new_page(device_scale_factor=2)
        for lang in LANGUAGES:
            for mode in COLORS:
                files = []
                for (i, name), fn in zip(NAMES, [architecture, lifecycle, detection, segmentation, classification_anomaly]):
                    svg = HERE/"svg"/f"{i:02d}_{name}_{file_suffix(lang, mode)}.svg"
                    fn(mode, lang).save(svg)
                    files.append(svg)
                    render(page, svg, HERE/"png"/(svg.stem+".png"), W, H)
                decks.append(build_deck(mode, lang, files, reference, editable_ppt))
        browser.close()
    source_manifest = [{"path": p, "sha256": hashlib.sha256((REPO/p).read_bytes()).hexdigest()} for p in SOURCES]
    (HERE/"manifest.json").write_text(json.dumps({
        "platform": "HawkeyePlus", "decks": decks, "sources": source_manifest,
        "style_reference": str(reference), "image_size": [3840, 2160],
        "asset_notes": "HawkeyePlus screenshots; original AI/data/model illustrations; reference WD template and generic art primitives.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    gallery()
    contact_sheet()
    print(json.dumps(decks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
