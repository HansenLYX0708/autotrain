"""Compose the joint SwiftFlow + HawkeyePlus leadership presentation.

The narrative describes a shared engineering method and explicit handoffs.
It does not implement a software connector or execute hardware/training jobs.
"""
import hashlib
import html
import importlib.util
import json
import shutil
import sys
import uuid
from datetime import date
from pathlib import Path

from lxml import etree as ET
from PIL import Image
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
REPO = PARENT.parent.parent
SF_VISUALS = Path(r"G:\projects\AOIPlatform\HwkUltra_g\docs\Architecture_Visuals")
SF_REPO = SF_VISUALS.parent.parent
ASSETS = HERE/"assets"
sys.path.insert(0, str(PARENT))
spec = importlib.util.spec_from_file_location("hawkeye_visuals", PARENT/"build.py")
hp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hp)

LANGUAGES = ("zh", "en")
THEMES = ("dark", "light")
SVG = "http://www.w3.org/2000/svg"
SLIDES = [
    ("01_Shared_Vision", "一套标准流程，贯通检测与测量", "One Standard Workflow for Inspection & Measurement", "concept"),
    ("02_Why_Standardize", "为什么要建立统一的工程方法", "Why a Shared Engineering Method Matters", "concept"),
    ("03_Standard_Workflow", "从工程目标到标准化交付", "From Engineering Goals to Standardized Delivery", "concept"),
    ("04_SwiftFlow_Foundation", "SwiftFlow｜设备与流程执行平台", "SwiftFlow | Equipment & Process Execution", "swiftflow"),
    ("05_Hardware_to_Flow", "从硬件方案到可执行检测流程", "From Hardware Design to Executable Workflows", "swiftflow"),
    ("06_Inspection_Measurement", "以同一套流程，实现检测或测量", "One Workflow Foundation for Inspection & Measurement", "swiftflow"),
    ("07_HawkeyePlus_Architecture", "HawkeyePlus｜工业视觉模型研发", "HawkeyePlus | Industrial Vision Model Development", "hawkeye"),
    ("08_HawkeyePlus_Lifecycle", "从数据到模型的可视化闭环", "A Visual Lifecycle: From Data to Models", "hawkeye"),
    ("09_Detection_Models", "Detection｜工业缺陷检测", "Object Detection | Industrial Defect Localization", "hawkeye"),
    ("10_Segmentation_Models", "分割模型｜像素级结构与区域识别", "Segmentation Models | Pixel-level Understanding", "hawkeye"),
    ("11_Classification_Anomaly", "分类与异常检测｜补齐视觉任务覆盖", "Classification & Anomaly Detection", "hawkeye"),
    ("12_Handoff_Delivery", "以标准交付包，衔接研发与现场", "Connect Development and Production with a Standard Handoff", "handoff"),
]
DECK_TITLE = (
    "SwiftFlow + HawkeyePlus｜工业检测与测量标准化平台：架构、流程与协同交付",
    "SwiftFlow + HawkeyePlus | Standardized Inspection & Measurement: Architecture, Workflows & Integrated Delivery",
)
CHAPTERS = [
    dict(key="concept", title=("总体概念与建设目的", "Vision & Rationale"),
         subtitle=("建立统一的工程方法", "Establish a shared engineering method"),
         question=("如何让硬件、工艺与算法形成统一方法？", "How do hardware, workflows and algorithms work as one method?"),
         topics=[("平台关系", "Platform roles"), ("建设目的", "Why standardize"), ("标准流程", "Standard workflow")],
         asset="sf:optical", first=3, last=6),
    dict(key="swiftflow", title=("SwiftFlow", "SwiftFlow"),
         subtitle=("设备、流程与算法执行", "Equipment, workflows and algorithm execution"),
         question=("如何把设备与算法组织成可执行流程？", "How can equipment and algorithms become an executable workflow?"),
         topics=[("设备接入", "Equipment integration"), ("流程设计", "Workflow design"), ("检测与测量", "Inspection & measurement")],
         asset="sf:machine", first=7, last=10),
    dict(key="hawkeye", title=("HawkeyePlus", "HawkeyePlus"),
         subtitle=("工业视觉模型研发与验证", "Industrial vision model development and validation"),
         question=("如何从数据出发，形成可验证、可交付的模型？", "How do we turn data into validated, deliverable models?"),
         topics=[("模型研发流程", "Model development"), ("当前模型能力", "Model portfolio"), ("验证与版本管理", "Validation & versions")],
         asset="hp:core", first=11, last=16),
    dict(key="handoff", title=("协同交付", "Integrated Delivery"),
         subtitle=("从研发效果到现场可用", "From development results to production readiness"),
         question=("如何用交付包与联合验证连接两个平台？", "How do delivery packages and joint validation connect both platforms?"),
         topics=[("适配与验收", "Adaptation & acceptance"), ("统一交付资产", "Shared delivery assets"), ("结果与版本追溯", "Results & version traceability")],
         asset="hp:registry", first=17, last=18),
]
# Content IDs preserve the twelve existing diagrams and their source notes.
# Page numbers belong to the complete presentation, including its framing pages.
PAGES = [
    dict(stem="01_Cover", kind="cover", chapter=None, title=("封面｜工业检测与测量标准化平台", "Title | Standardized Inspection & Measurement")),
    dict(stem="02_Agenda", kind="agenda", chapter=None, title=("汇报目录", "Presentation Roadmap")),
    dict(stem="03_Chapter_Overview", kind="divider", chapter=0, title=CHAPTERS[0]["title"]),
    dict(stem="04_Shared_Vision", kind="content", chapter=0, content=1, title=SLIDES[0][1:3]),
    dict(stem="05_Why_Standardize", kind="content", chapter=0, content=2, title=SLIDES[1][1:3]),
    dict(stem="06_Standard_Workflow", kind="content", chapter=0, content=3, title=SLIDES[2][1:3]),
    dict(stem="07_Chapter_SwiftFlow", kind="divider", chapter=1, title=CHAPTERS[1]["title"]),
    dict(stem="08_SwiftFlow_Foundation", kind="content", chapter=1, content=4, title=SLIDES[3][1:3]),
    dict(stem="09_Hardware_to_Flow", kind="content", chapter=1, content=5, title=SLIDES[4][1:3]),
    dict(stem="10_Inspection_Measurement", kind="content", chapter=1, content=6, title=SLIDES[5][1:3]),
    dict(stem="11_Chapter_HawkeyePlus", kind="divider", chapter=2, title=CHAPTERS[2]["title"]),
    dict(stem="12_HawkeyePlus_Architecture", kind="content", chapter=2, content=7, title=SLIDES[6][1:3]),
    dict(stem="13_HawkeyePlus_Lifecycle", kind="content", chapter=2, content=8, title=SLIDES[7][1:3]),
    dict(stem="14_Detection_Models", kind="content", chapter=2, content=9, title=SLIDES[8][1:3]),
    dict(stem="15_Segmentation_Models", kind="content", chapter=2, content=10, title=SLIDES[9][1:3]),
    dict(stem="16_Classification_Anomaly", kind="content", chapter=2, content=11, title=SLIDES[10][1:3]),
    dict(stem="17_Chapter_Delivery", kind="divider", chapter=3, title=CHAPTERS[3]["title"]),
    dict(stem="18_Handoff_Delivery", kind="content", chapter=3, content=12, title=SLIDES[11][1:3]),
    dict(stem="19_Summary", kind="summary", chapter=None, title=("总结｜标准方法、平台协同、可复用交付", "Summary | Method, Platforms and Reusable Delivery")),
    dict(stem="20_Closing", kind="closing", chapter=None, title=("结束页｜讨论与交流", "Closing | Questions & Discussion")),
]
NATIVE_SECTIONS = [
    (1, ("开场与目录", "Opening & Roadmap")),
    *((c["first"], (f"{i+1:02d} "+c["title"][0], f"{i+1:02d} "+c["title"][1])) for i, c in enumerate(CHAPTERS)),
    (19, ("总结与交流", "Summary & Discussion")),
]
AGENDA_LINKS = [
    dict(x=64+(i%2)*918, y=245+(i//2)*330, w=874, h=300, target=c["first"])
    for i, c in enumerate(CHAPTERS)
]
SF_SOURCES = [
    "docs/Architecture.md", "docs/Flow_Engine_Internals.md", "docs/Configuration_Reference.md",
    "HWKUltra.Inference/Implementations/InferenceSessionFactory.cs",
    "HWKUltra.Inference/Implementations/ModelInstanceConfig.cs",
    "HWKUltra.Inference/Abstractions/InferenceTask.cs",
]
HP_SOURCES = hp.SOURCES
NOTES = {
    1: (
        "这是一套统一工程方法下的两个互补平台。SwiftFlow 将设备、流程、传统视觉、模型推理和结果组织为现场执行链；"
        "HawkeyePlus 负责训练数据、模型研发、评估与版本管理。工程目标先确定检测缺陷或测量对象，再决定硬件和算法路线。"
        "双向箭头表示经过人工整理的样本/需求以及经过适配验证的模型交付，不表示现有软件已实现自动数据回流或模型同步。",
        "Two complementary platforms support one engineering method. SwiftFlow organizes equipment, workflows, classical vision, "
        "inference and results into execution. HawkeyePlus manages training data, model development, evaluation and versions. "
        "The arrows represent curated samples and validated model handoffs, not an already implemented automatic connector."),
    2: (
        "统一方法的目的，是减少硬件、工艺与算法在衔接处的反复工作。这里描述缺少标准时的工程风险和预期收益，"
        "不是对某团队的现状调查，也没有未经测量的效率数字。标准化输入、可复用配置和联合验收，使参与者知道"
        "每一步需要什么、交付什么以及如何判断完成。",
        "A shared method reduces repeated handoff work between hardware, process and algorithm engineering. "
        "These are design motivations, not measured claims about a team's productivity. Common inputs, reusable assets and "
        "joint acceptance make the required inputs, outputs and completion criteria clear."),
    3: (
        "本页提出推荐标准流程，不宣称已上线一个自动化审批或交付系统。硬件设计和光机电选型由工程人员完成；"
        "SwiftFlow 提供设备接入、调试、标定与流程执行。算法开发可以选择传统视觉/数值测量，也可以由 HawkeyePlus 训练 AI。"
        "联合验证需要独立样本、误检漏检与节拍检查；测量任务还需要单位、标定与参考值一致性。各阶段交付物形成建议基线。",
        "This is a recommended engineering workflow, not a claim of an automated release system. Engineers design the hardware "
        "and imaging setup; SwiftFlow provides integration, commissioning, calibration and execution. Algorithms may be classical "
        "vision/numerical measurement or AI models developed in HawkeyePlus. Joint acceptance uses independent samples, detection "
        "errors and cycle-time checks, plus calibration, units and reference agreement for measurement."),
    4: (
        "SwiftFlow 是 HawkeyeUltra 的流程与执行底座。流程定义是数据，可以组合设备、视觉、AI 推理和结果节点，支持子流程、"
        "并行协同和运行控制。设备通过厂商无关接口与多实例路由接入。日志、计时、存储与结果支撑全流程。"
        "本图不表示所有设备或模型都已完成现场验收，也不代表分布式调度平台。",
        "SwiftFlow is the workflow and execution foundation within HawkeyeUltra. Data-defined workflows combine equipment, "
        "vision, inference and result nodes, including subflows, parallel coordination and run control. Vendor-neutral interfaces "
        "and instance routing connect devices. Logs, timing, storage and results support execution."),
    5: (
        "用户从相机、镜头、光源、运动和测量器件的工程方案开始。机型配置将物理设备映射到逻辑实例，调试与标定确认输入质量。"
        "再用流程节点组织定位、采集、分析、判定与结果。交付的不只是一个能运行的界面，而是设备映射、标定和工艺配置等可复用资产。"
        "SwiftFlow 不替代 CAD、光学设计或机械工程工具。",
        "Engineers start with camera, optics, lighting, motion and measurement-device design. Machine configuration maps physical "
        "devices to logical instances; commissioning and calibration establish input quality. Workflow nodes then organize positioning, "
        "acquisition, analysis and decisions. Deliverables include mappings, calibration and reusable process configuration. "
        "SwiftFlow is not a CAD or optical-design tool."),
    6: (
        "检测与测量可以共用采集、编排和结果追溯底座，但输出不同。检测输出位置、类别与判定；测量输出有定义的尺寸、高度或几何指标。"
        "AI 可以帮助定位或分割，物理尺寸仍依赖标定、单位和测量算法，不能把像素掩码直接当作尺寸。"
        "传统视觉或激光/几何测量无需先训练 AI 模型。插图只说明方法，不作为精度或量产能力证明。",
        "Inspection and measurement share acquisition, orchestration and traceability, but produce different outputs. Inspection "
        "produces locations, classes and decisions; measurement produces defined dimensions, heights or geometric metrics. "
        "AI may assist localization or segmentation, but physical measurements still require calibration, units and measurement logic. "
        "Classical vision and laser/geometric measurement do not require AI training."),
    12: (
        "推荐以交付包加联合验收衔接两个平台。HawkeyePlus 可导出受支持框架的推理产物，并按产品登记模型版本、提供激活模型查询 API。"
        "SwiftFlow 的当前工厂实现 ONNX（含 YOLO 路径）和 Paddle 后端，模型配置包含标签、尺寸、通道顺序、归一化、阈值和输出解码。"
        "训练支持列表不等于 SwiftFlow 可直接加载列表：需逐项核对格式、算子、输入输出和预后处理。Torch 训练检查点不是可直接交付的推理模型；"
        "TorchSharp 是保留后端；当前任务枚举为检测、分类、分割，没有独立异常检测解码任务。异常模型接入需要额外适配验证。"
        "未发现现有 SwiftFlow 服务自动调用 HawkeyePlus 模型注册 API 的证据，本报告不声称自动发布/自动回流已打通。"
        "建议交付资产是工程约定，不是新增软件功能。",
        "Use a model handoff package and joint acceptance. HawkeyePlus exports supported inference artifacts and registers product "
        "versions with an active-model query API. SwiftFlow currently implements ONNX (including the YOLO path) and Paddle backends. "
        "Its configuration includes labels, input shape, channel order, normalization, thresholds and output decoding. "
        "Training support is not direct runtime compatibility: verify operators, I/O and preprocessing/postprocessing per model. "
        "Torch checkpoints are not direct inference artifacts; TorchSharp is reserved, and the current task enum has detection, "
        "classification and segmentation, but no dedicated anomaly decoder. Anomaly deployment needs additional adaptation. "
        "No existing automatic registry synchronization or feedback connector is assumed. The delivery baseline is a proposed engineering convention."),
}


def suffix(lang, mode):
    return lang+"_"+mode


def deck_stem(lang, mode):
    return f"SwiftFlow_HawkeyePlus_Integrated_{lang.upper()}_{mode.title()}"


class Canvas(hp.Canvas):
    def translate(self, value):
        if isinstance(value, tuple):
            return value[0 if self.lang == "zh" else 1]
        return value

    def image(self, name, x, y, w, h):
        prefix, asset = name.split(":")
        path = ASSETS/f"{self.mode}_{prefix}_{asset}.png"
        self.parts.append(f'<image href="{hp.data_uri(path)}" x="{x}" y="{y}" width="{w}" height="{h}"/>')

    def header(self, title, subtitle, page, section="concept"):
        p = self.c
        names = {
            "concept": ("总体概念 / 标准工程方法", "CONCEPT / A SHARED ENGINEERING METHOD"),
            "swiftflow": ("第一部分 / SWIFTFLOW", "PART 01 / SWIFTFLOW"),
            "hawkeye": ("第二部分 / HAWKEYEPLUS", "PART 02 / HAWKEYEPLUS"),
            "handoff": ("协同交付 / 从研发到现场", "HANDOFF / FROM DEVELOPMENT TO EXECUTION"),
        }
        self.rect(64, 54, 6, 30, p["cyan"], radius=3)
        self.text(86, 77, names[section], 23, p["muted"], 600, 1500)
        self.image("hp:logo", 1742, 50, 111, 47)
        self.text(64, 146, title, 57, weight=600, width=1780)
        self.text(67, 193, subtitle, 28, p["muted"], width=1770)
        self.footer(page)

    def footer(self, page):
        p = self.c
        self.path("M64 1027 H1856", p["line"], 1)
        self.text(65, 1055, "SwiftFlow + HawkeyePlus  |  Industrial Vision Engineering", 16, p["muted"], width=1300)
        self.text(1748, 1055, "WD CONFIDENTIAL", 15, p["muted"], width=400, anchor="end")
        self.text(1854, 1055, f"{page:02d} / {len(PAGES):02d}", 16, p["muted"], width=100, anchor="end")

    def frame(self, kicker, page):
        self.rect(64, 54, 6, 30, self.c["cyan"], radius=3)
        self.text(86, 77, kicker, 23, self.c["muted"], 600, 1500)
        self.image("hp:logo", 1742, 50, 111, 47)
        self.footer(page)

    def mini_card(self, x, y, w, h, asset, title, detail):
        self.group(title)
        self.rect(x, y, w, h, self.c["panel"], self.c["line"], 17)
        self.image(asset, x+10, y+8, 132, h-16)
        self.text(x+155, y+51, title, 29, weight=600, width=w-177)
        self.text(x+155, y+98, detail, 24, self.c["muted"], width=w-177)
        self.end()


def setup_canvas(mode, lang, number, subtitle):
    meta = SLIDES[number-1]
    c = Canvas(mode, (meta[1], meta[2]), lang)
    c.header((meta[1], meta[2]), subtitle, number, meta[3])
    return c


def cover(mode, lang, page):
    c = Canvas(mode, DECK_TITLE, lang)
    p = c.c
    c.frame(("平台架构与标准流程汇报", "PLATFORM ARCHITECTURE & STANDARDIZED WORKFLOWS"), page)
    c.text(79, 231, "SwiftFlow + HawkeyePlus", 83, weight=600, width=1765)
    c.text(83, 337, ("工业检测与测量标准化平台", "Standardized Inspection & Measurement"),
           63, weight=600, width=1765)
    c.text(87, 413, ("架构、流程与协同交付", "Architecture, Workflows & Integrated Delivery"),
           38, p["cyan"], 600, 1740)
    c.path("M88 451 H1832", p["line"], 1.5)
    for x, name, asset, detail, color in [
        (112, "SwiftFlow", "sf:machine", ("设备、工艺与现场执行", "Equipment, processes and execution"), p["cyan"]),
        (1124, "HawkeyePlus", "hp:core", ("数据、模型与研发验证", "Data, models and validation"), p["teal"]),
    ]:
        c.group(name)
        c.image(asset, x, 482, 650, 330)
        c.text(x+325, 852, name, 41, color, 600, 650, "middle")
        c.text(x+325, 902, detail, 28, p["muted"], width=650, anchor="middle")
        c.end()
    c.path("M791 649 H1121", p["line"], 2)
    c.circle(958, 649, 48, p["inset"])
    c.text(958, 663, "+", 49, p["cyan"], 600, 88, "middle")
    c.text(87, 989, ("从硬件设计到算法研发与现场应用", "From hardware design to algorithms and production workflows"),
           29, p["muted"], width=1510)
    c.text(1832, 988, date.today().strftime("%Y.%m"), 23, p["muted"], width=170, anchor="end")
    return c


def agenda(mode, lang, page):
    c = Canvas(mode, ("汇报目录", "Presentation Roadmap"), lang)
    p = c.c
    c.header(("汇报目录", "Presentation Roadmap"),
             ("先建立统一工程方法，再展开两个平台及其协同交付",
              "Start with a shared method, then explore the two platforms and their handoff"),
             page)
    for i, (chapter, bounds) in enumerate(zip(CHAPTERS, AGENDA_LINKS)):
        x, y, w, h = (bounds[k] for k in ["x", "y", "w", "h"])
        c.group(chapter["title"])
        c.rect(x, y, w, h, p["panel"], p["line"], 20)
        c.text(x+27, y+59, f"{i+1:02d}", 43, p["cyan"], 600, 96)
        c.image(chapter["asset"], x+16, y+84, 226, 185)
        c.text(x+258, y+69, chapter["title"], 36, weight=600, width=585)
        c.text(x+258, y+125, chapter["subtitle"], 27, p["muted"], width=585)
        c.text(x+258, y+202, chapter["topics"][0], 26, p["cyan"], width=420)
        c.text(x+258, y+246, chapter["topics"][1], 26, p["cyan"], width=420)
        c.badge(x+w-163, y+h-61, f"{chapter['first']:02d}–{chapter['last']:02d}", 132)
        c.end()
    c.rect(64, 930, 1792, 73, p["inset"], p["line"], 14)
    c.text(93, 976, ("最后：总结与交流", "Closing: Summary & Discussion"), 29, p["teal"], 600, 1300)
    c.text(1823, 974, "19–20", 26, p["teal"], width=160, anchor="end")
    return c


def chapter_divider(mode, lang, page, chapter_index):
    chapter = CHAPTERS[chapter_index]
    c = Canvas(mode, chapter["title"], lang)
    p = c.c
    number = f"{chapter_index+1:02d}"
    c.frame((f"章节导览 / 第 {number} 章", f"CHAPTER {number} / OVERVIEW"), page)
    c.text(78, 426, number, 174, p["cyan"], 600, 270)
    c.text(347, 346, chapter["title"], 59, weight=600, width=925)
    c.text(350, 423, chapter["subtitle"], 34, p["muted"], width=800)
    c.path("M82 486 H1086", p["line"], 1.8)
    c.text(82, 555, chapter["question"], 31, p["ink"], width=1040)
    c.image(chapter["asset"], 1200, 273, 605, 421)
    for i, topic in enumerate(chapter["topics"]):
        x = 64+i*607
        c.group(topic)
        c.rect(x, 784, 578, 155, p["panel"], p["line"], 18)
        c.text(x+27, 834, f"{number}.{i+1}", 24, p["cyan"], 600, 110)
        c.text(x+27, 894, topic, 32, weight=600, width=523)
        c.end()
    c.text(82, 993, ("SwiftFlow + HawkeyePlus｜一套标准方法，两个互补平台",
                     "SwiftFlow + HawkeyePlus | One shared method, two complementary platforms"),
           25, p["muted"], width=1690)
    return c


def summary_slide(mode, lang, page):
    c = Canvas(mode, ("把能力连接成可复用的标准流程", "Build a Reusable Standard Workflow"), lang)
    p = c.c
    c.header(("把能力连接成可复用的标准流程", "Build a Reusable Standard Workflow"),
             ("用共同的方法组织工程工作，以两个平台承载能力，以交付基线保证衔接",
              "Organize engineering around a shared method, complementary platforms and a defined delivery baseline"),
             page)
    for i, (asset, title, lines) in enumerate([
        ("sf:optical", ("统一工程方法", "One engineering method"),
         [("从检测或测量目标出发", "Start with inspection or measurement goals"),
          ("明确硬件、流程与算法的输入输出", "Define inputs and outputs across disciplines")]),
        ("sf:flow", ("两个平台协同", "Two complementary platforms"),
         [("SwiftFlow 承载设备与流程执行", "SwiftFlow connects equipment and execution"),
          ("HawkeyePlus 承载模型研发与验证", "HawkeyePlus develops and validates models")]),
        ("hp:registry", ("形成标准交付", "A standard delivery baseline"),
         [("配置、模型与验收证据共同交付", "Deliver configuration, models and evidence"),
          ("适配验证后进入现场，保留版本与结果", "Validate on site and retain versions and results")]),
    ]):
        x = 64+i*607
        c.group(title)
        c.rect(x, 245, 578, 566, p["panel"], p["line"], 22)
        c.text(x+29, 298, title, 34, weight=600, width=520)
        c.image(asset, x+107, 331, 364, 269)
        c.path(f"M{x+29} 635 H{x+548}", p["line"], 1)
        for j, line in enumerate(lines):
            c.text(x+289, 692+j*66, line, 26, p["muted"], width=518, anchor="middle")
        c.end()
    c.rect(64, 853, 1792, 139, p["inset"], p["line"], 18)
    c.text(960, 910, ("目标明确 → 工程实现 → 联合验证 → 可追溯交付",
                       "Define the goal → Engineer the solution → Validate jointly → Deliver traceably"),
           34, p["teal"], 600, 1715, "middle")
    c.text(960, 965, ("最终目标：方便用户以一致的方法，实现检测或测量目的",
                       "The goal: help users deliver inspection and measurement through a consistent engineering approach"),
           27, p["muted"], width=1715, anchor="middle")
    return c


def closing(mode, lang, page):
    c = Canvas(mode, ("讨论与交流", "Questions & Discussion"), lang)
    p = c.c
    c.frame("SWIFTFLOW + HAWKEYEPLUS", page)
    c.group(("讨论与交流", "Questions & Discussion"))
    c.text(83, 289, ("谢谢", "Thank You"), 100, p["cyan"], 600, 1050)
    c.text(86, 392, ("讨论与交流", "Questions & Discussion"), 52, weight=600, width=1080)
    c.text(89, 471, ("从硬件设计到检测与测量交付", "From hardware design to inspection and measurement delivery"),
           30, p["muted"], width=1040)
    questions = [
        ("如何确定首个应用场景？", "Which use case should we prioritize?"),
        ("如何定义验收与交付基线？", "How should we define acceptance and delivery?"),
        ("哪些流程、模型与经验可以优先复用？", "Which workflows, models and knowledge can we reuse?"),
    ]
    for i, question in enumerate(questions):
        c.circle(99, 625+i*83, 5, p["teal"])
        c.text(121, 636+i*83, question, 28, p["muted"], width=932)
    c.end()
    c.group("SwiftFlow + HawkeyePlus")
    c.image("sf:machine", 1165, 247, 628, 364)
    c.image("hp:core", 1233, 641, 535, 250)
    c.end()
    c.text(86, 978, ("一套标准方法 · 两个互补平台 · 可复用的工程交付",
                     "One shared method · Two complementary platforms · Reusable engineering deliverables"),
           28, p["teal"], 600, 1740)
    return c


def vision(mode, lang):
    c = setup_canvas(mode, lang, 1, (
        "以 SwiftFlow 与 HawkeyePlus 协同，连接硬件设计、检测流程与算法研发",
        "Connect hardware design, inspection workflows and algorithm development through two complementary platforms"))
    p = c.c
    for x, title, asset, lines, detail in [
        (64, "SwiftFlow", "sf:machine",
         [("设备接入", "Equipment integration"), ("流程编排", "Workflow orchestration"), ("现场执行", "Inspection execution")],
         ("把硬件、算法和工艺组织成可执行流程", "Turn hardware, algorithms and process logic into executable workflows")),
        (996, "HawkeyePlus", "hp:core",
         [("数据与标注", "Data & annotations"), ("模型训练", "Model training"), ("验证与发布", "Validation & release")],
         ("把工业图像沉淀为可验证、可交付的模型", "Turn industrial images into validated, deliverable models")),
    ]:
        c.group(title)
        c.rect(x, 240, 860, 466, p["panel"], p["line"], 23)
        c.text(x+32, 296, title, 46, p["cyan"] if x == 64 else p["teal"], 600, 795)
        c.image(asset, x+25, 324, 417, 281)
        for i, line in enumerate(lines):
            c.circle(x+474, 385+i*74, 5, p["cyan"])
            c.text(x+493, 396+i*74, line, 30, width=334)
        c.text(x+430, 662, detail, 27, p["muted"], width=790, anchor="middle")
        c.end()
    c.text(960, 402, ("样本", "Samples"), 19, p["cyan"], width=70, anchor="middle")
    c.path("M933 423 H986", p["cyan"], 2.5, True)
    c.text(960, 505, ("模型包", "Models"), 19, p["teal"], width=70, anchor="middle")
    c.path("M986 526 H933", p["teal"], 2.5, True)
    c.rect(64, 739, 1792, 64, p["inset"], p["line"], 14)
    c.text(960, 781, ("共同方法：明确目标 → 设计实现 → 适配验证 → 标准交付",
                       "Shared method: Define the goal → Design & build → Adapt & validate → Standardize delivery"),
           29, p["cyan"], 600, 1720, "middle")
    for x, image, title, detail in [
        (64, "hp:validation", ("检测", "Inspection"), ("缺陷位置 · 类型 · 判定", "Defect location · Type · Decision")),
        (671, "sf:laser", ("测量", "Measurement"), ("尺寸 · 高度 · 几何指标", "Dimensions · Height · Geometry")),
        (1278, "sf:report", ("可追溯交付", "Traceable delivery"), ("配置 · 版本 · 验收结果", "Configuration · Version · Evidence")),
    ]:
        c.mini_card(x, 838, 578, 153, image, title, detail)
    return c


def why(mode, lang):
    c = setup_canvas(mode, lang, 2, (
        "让参与者知道每一步需要什么、交付什么，以及如何判断完成",
        "Make inputs, deliverables and acceptance criteria clear at every step"))
    p = c.c
    items = [
        ("sf:optical", ("减少衔接歧义", "Align engineering inputs"),
         ("硬件、成像与算法各有约束", "Align hardware, imaging and algorithm needs"),
         ("统一设计输入", "Common design inputs"),
         [("检测 / 测量目标", "Inspection / measurement goals"), ("成像质量与标定要求", "Image quality and calibration")]),
        ("sf:flow", ("减少重复搭建", "Reuse proven work"),
         ("成熟经验应成为可复用资产", "Turn proven knowledge into reusable assets"),
         ("统一交付资产", "Reusable deliverables"),
         [("设备与流程配置", "Device and workflow configuration"), ("模型与算法参数", "Models and algorithm parameters")]),
        ("sf:shield", ("减少交付不确定性", "Make acceptance explicit"),
         ("研发效果需要在现场条件下复现", "Reproduce results on production equipment"),
         ("统一验收口径", "Shared acceptance criteria"),
         [("独立样本与参考结果", "Independent samples and references"), ("效果、节拍与结果留痕", "Quality, timing and traceable results")]),
    ]
    for i, (asset, title, detail, standard, checks) in enumerate(items):
        x = 64+i*607
        c.group(title)
        c.rect(x, 242, 578, 603, p["panel"], p["line"], 22)
        c.text(x+28, 294, title, 33, weight=600, width=522)
        c.image(asset, x+116, 315, 346, 241)
        c.text(x+289, 593, detail, 25, p["muted"], width=516, anchor="middle")
        c.path(f"M{x+28} 623 H{x+550}", p["line"], 1)
        c.text(x+28, 673, standard, 31, p["cyan"], 600, 520)
        for j, check in enumerate(checks):
            c.circle(x+32, 720+j*62, 4, p["teal"])
            c.text(x+49, 731+j*62, check, 27, width=501)
        c.end()
    c.rect(64, 883, 1792, 108, p["inset"], p["line"], 17)
    c.text(960, 926, ("标准化的价值：让项目交接、方案复用与后续维护有共同依据",
                       "Standardization gives handoffs, reuse and maintenance a shared baseline"),
           32, p["teal"], 600, 1710, "middle")
    c.text(960, 970, ("形成工程方法与交付约定，再由两个平台承载具体能力",
                       "Define the method and delivery agreements; use both platforms to execute them"),
           26, p["muted"], width=1710, anchor="middle")
    return c


def workflow(mode, lang):
    c = setup_canvas(mode, lang, 3, (
        "推荐标准流程：每一步都有明确输入、负责内容和可检查的交付物",
        "Recommended workflow: clear inputs, engineering responsibilities and reviewable deliverables"))
    p = c.c
    stages = [
        ("sf:report", ("明确目标", "Define goals"),
         [("检测什么 / 测量什么", "What to inspect or measure"), ("质量、节拍与约束", "Quality, timing and constraints")],
         ("需求与验收基线", "Requirements & acceptance")),
        ("sf:optical", ("硬件与成像", "Design hardware"),
         [("相机、镜头、光源", "Camera, optics and lighting"), ("运动、测高与标定", "Motion, sensing and calibration")],
         ("硬件方案与样本", "Hardware design & samples")),
        ("sf:flow", ("检测流程", "Design workflows"),
         [("定位、采集与时序", "Positioning, capture and timing"), ("异常路径与结果组织", "Exceptions and result handling")],
         ("流程与设备配置", "Flow & device configuration")),
        ("hp:core", ("算法开发", "Develop algorithms"),
         [("传统视觉 / 几何测量", "Classical vision / geometry"), ("AI 训练与验证", "AI training and validation")],
         ("算法参数 / 模型包", "Parameters / model package")),
        ("sf:calibration", ("联合验证", "Validate jointly"),
         [("现场样本与参考值", "Site samples and references"), ("效果、测量与节拍", "Quality, metrology and timing")],
         ("验证记录与结论", "Acceptance evidence")),
        ("hp:registry", ("标准交付", "Deliver a baseline"),
         [("配置、模型与版本", "Configuration, models, versions"), ("结果归档与迭代", "Traceable results and iteration")],
         ("可复用工程资产", "Reusable engineering assets")),
    ]
    for i, (asset, title, lines, output) in enumerate(stages):
        x = 64+i*302
        c.group(title)
        c.rect(x, 245, 282, 507, p["panel"], p["line"], 18)
        c.text(x+23, 285, f"0{i+1}", 22, p["cyan"], 600, 100)
        c.text(x+141, 331, title, 29, weight=600, width=244, anchor="middle")
        c.image(asset, x+27, 353, 228, 185)
        for j, line in enumerate(lines):
            c.text(x+141, 578+j*43, line, 23, p["muted"], width=246, anchor="middle")
        c.path(f"M{x+22} 649 H{x+260}", p["line"], 1)
        c.text(x+141, 697, output, 24, p["teal"], 600, 247, "middle")
        c.end()
        if i < 5:
            c.path(f"M{x+286} 465 H{x+299}", p["cyan"], 1.8, True)
    for y, label, text, color in [
        (786, "SwiftFlow", ("承载设备接入、调试标定、流程执行与结果输出",
                           "Equipment integration, commissioning, calibration, workflow execution and results"), p["cyan"]),
        (866, "HawkeyePlus", ("承载训练数据、模型研发、评估验证与版本管理",
                             "Training data, model development, evaluation and model version management"), p["teal"]),
    ]:
        c.rect(64, y, 1792, 63, p["inset"], p["line"], 13)
        c.text(92, y+42, label, 30, color, 600, 270)
        c.text(392, y+41, text, 26, width=1420)
    c.text(960, 991, ("共同交付：硬件基线 + 流程基线 + 算法基线 + 验收证据",
                       "Shared deliverables: hardware + workflow + algorithm baselines + acceptance evidence"),
           27, p["teal"], 600, 1760, "middle")
    return c


def swiftflow(mode, lang):
    c = setup_canvas(mode, lang, 4, (
        "用可配置的流程，把设备、传统视觉、模型推理和结果组织成执行链",
        "Use configurable workflows to connect equipment, classical vision, inference and results"))
    p = c.c
    for x, w, title, asset, labels in [
        (64, 472, ("流程设计", "Workflow design"), "sf:flow",
         [("可视化编排", "Visual authoring"), ("工艺与子流程复用", "Reusable process flows")]),
        (581, 717, ("统一执行引擎", "Shared execution engine"), "sf:core",
         [("串并行协同 · 共享数据", "Sequential / parallel execution · Shared data"),
          ("运行控制 · 异常处理", "Run control · Failure handling")]),
        (1343, 513, ("算法与结果", "Algorithms & results"), "sf:chip",
         [("传统视觉 + 模型推理", "Classical vision + Model inference"),
          ("测量计算 + 结果归档", "Measurement + Result storage")]),
    ]:
        c.group(title)
        c.rect(x, 241, w, 428, p["panel"], p["line"], 22)
        c.text(x+w/2, 291, title, 34, weight=600, width=w-45, anchor="middle")
        c.image(asset, x+30, 306, w-60, 254)
        for i, label in enumerate(labels):
            c.text(x+w/2, 600+i*44, label, 25, p["cyan"] if i == 0 else p["muted"], width=w-44, anchor="middle")
        c.end()
    c.path("M541 455 H574", p["cyan"], 2.5, True)
    c.path("M1305 455 H1336", p["cyan"], 2.5, True)
    c.rect(64, 705, 1792, 53, p["inset"], p["line"], 13)
    c.text(94, 742, ("统一设备接入", "Unified equipment integration"), 30, p["cyan"], 600, 1000)
    c.text(1825, 741, ("标准接口 · 多实例 · 厂商适配", "Standard interfaces · Multiple instances · Vendor adapters"),
           25, p["muted"], width=960, anchor="end")
    for i, (asset, title) in enumerate([
        ("camera", ("工业相机", "Industrial cameras")), ("motion", ("精密运动", "Motion control")),
        ("light", ("受控照明", "Controlled lighting")), ("laser", ("激光测高", "Height sensing")),
        ("optical", ("光学 / 测量", "Optics / metrology")), ("network", ("IO / 产线通信", "IO / Line communication")),
    ]):
        x = 64+i*300
        c.group(title)
        c.image("sf:"+asset, x+48, 774, 190, 142)
        c.text(x+145, 944, title, 25, width=283, anchor="middle")
        c.end()
    c.text(960, 996, ("全链路支撑：日志告警 · 运行计时 · 结果追溯 · 存储管理 · 仿真调试",
                       "Cross-cutting support: logs & alarms · timing · traceability · storage · simulation"),
           25, p["teal"], width=1770, anchor="middle")
    return c


def hardware_flow(mode, lang):
    c = setup_canvas(mode, lang, 5, (
        "将硬件选型、设备调试和工艺经验，转化为可复用的配置与流程",
        "Translate hardware choices, commissioning and process knowledge into reusable configuration and workflows"))
    p = c.c
    c.group(("工程设计输入", "Engineering design inputs"))
    c.rect(64, 241, 732, 558, p["panel"], p["line"], 22)
    c.text(96, 293, ("工程设计输入", "Engineering design inputs"), 35, weight=600, width=664)
    c.image("sf:machine", 95, 315, 669, 338)
    c.text(430, 697, ("相机 / 镜头 / 光源 / 运动 / 传感器", "Camera / Optics / Lighting / Motion / Sensors"),
           28, p["cyan"], 600, 668, "middle")
    c.text(430, 753, ("明确视野、分辨率、节拍与标定要求", "Define field of view, resolution, timing and calibration"),
           26, p["muted"], width=668, anchor="middle")
    c.end()
    c.path("M807 515 H843", p["cyan"], 2.5, True)
    c.group(("设备配置与流程实现", "Configuration & workflow implementation"))
    c.rect(856, 241, 1000, 558, p["panel"], p["line"], 22)
    c.text(889, 293, ("设备配置与流程实现", "Configuration & workflow implementation"), 35, weight=600, width=933)
    c.image("sf:flow", 906, 315, 383, 246)
    c.text(1320, 386, ("设备映射", "Map device instances"), 29, p["cyan"], 600, 500)
    c.text(1320, 442, ("调试与标定", "Commission & calibrate"), 29, p["cyan"], 600, 500)
    c.text(1320, 498, ("参数与流程复用", "Reuse parameters & flows"), 29, p["cyan"], 600, 500)
    for i, (asset, title) in enumerate([
        ("calibration", ("定位", "Locate")), ("camera", ("采集", "Capture")),
        ("chip", ("分析", "Analyze")), ("report", ("判定与记录", "Decide & record")),
    ]):
        x = 890+i*237
        c.image("sf:"+asset, x+42, 581, 147, 124)
        c.text(x+114, 749, title, 26, width=220, anchor="middle")
        if i < 3:
            c.path(f"M{x+211} 647 H{x+236}", p["cyan"], 1.8, True)
    c.end()
    c.rect(64, 842, 1792, 149, p["inset"], p["line"], 18)
    c.text(96, 891, ("交付的是可复用的工程资产", "Deliver reusable engineering assets"), 33, p["teal"], 600, 1700)
    c.text(96, 951, ("设备映射与配置   ·   标定参数   ·   检测流程   ·   工艺参数   ·   结果定义",
                     "Device mappings & configuration   ·   Calibration   ·   Workflows   ·   Process parameters   ·   Result definitions"),
           28, width=1700)
    return c


def inspection_measurement(mode, lang):
    c = setup_canvas(mode, lang, 6, (
        "传统视觉、AI 与测量算法按任务组合，共享采集、编排和结果追溯能力",
        "Combine classical vision, AI and metrology as needed, sharing acquisition, orchestration and traceability"))
    p = c.c
    for x, title, asset, chain, result in [
        (64, ("检测：找到缺陷并判定", "Inspection: locate defects and decide"), "hp:detection_scene",
         ("图像 → 传统视觉 / AI → 判定", "Image → Classical vision / AI → Decision"),
         ("输出：位置、类别与合格判定", "Output: location, class and pass/fail")),
        (974, ("测量：获得定义明确的指标", "Measurement: produce defined metrics"), "sf:laser",
         ("标定 → 采集 / 分割 → 测量计算", "Calibration → Capture / segmentation → Measurement"),
         ("输出：尺寸、高度与几何指标", "Output: dimensions, height and geometry")),
    ]:
        c.group(title)
        c.rect(x, 241, 882, 588, p["panel"], p["line"], 22)
        c.text(x+28, 293, title, 35, weight=600, width=825)
        c.image(asset, x+126, 325, 630, 322)
        c.text(x+441, 699, chain, 29, p["cyan"], 600, 816, "middle")
        c.text(x+441, 770, result, 28, p["muted"], width=816, anchor="middle")
        c.end()
    c.rect(64, 867, 1792, 124, p["inset"], p["line"], 18)
    c.text(960, 915, ("共同底座：数据采集 → 算法执行 → 规则判定 → 结果归档",
                       "Shared foundation: acquire data → run algorithms → apply rules → retain results"),
           31, p["teal"], 600, 1720, "middle")
    c.text(960, 967, ("测量结果需要标定、单位与参考验证；AI 可以辅助定位或分割",
                       "Measurement requires calibration, units and reference validation; AI can assist localization or segmentation"),
           26, p["muted"], width=1720, anchor="middle")
    return c


def handoff(mode, lang):
    c = setup_canvas(mode, lang, 12, (
        "训练成果经过适配与联合验证，再进入现场检测或测量流程",
        "Adapt and jointly validate model deliverables before using them in inspection or measurement workflows"))
    p = c.c
    for x, name, image, desc in [
        (64, "HawkeyePlus", "hp:registry", ("验证后的模型与版本", "Validated model & version")),
        (1418, "SwiftFlow", "sf:flow", ("流程中的推理与判定", "Inference & decisions in the workflow")),
    ]:
        c.group(name)
        c.rect(x, 245, 438, 430, p["panel"], p["line"], 20)
        c.text(x+219, 301, name, 39, p["teal"] if x == 64 else p["cyan"], 600, 392, "middle")
        c.image(image, x+45, 329, 348, 243)
        c.text(x+219, 629, desc, 27, width=392, anchor="middle")
        c.end()
    c.path("M511 466 H541", p["cyan"], 2.5, True)
    c.path("M1375 466 H1408", p["cyan"], 2.5, True)
    c.group(("适配与联合验收", "Adaptation & joint acceptance"))
    c.rect(550, 245, 816, 430, p["inset"], p["line"], 20)
    c.text(958, 300, ("适配与联合验收", "Adaptation & joint acceptance"), 36, weight=600, width=748, anchor="middle")
    checks = [
        (("输入一致", "Match the input"), ("图像尺寸、通道与预处理", "Image shape, channels and preprocessing")),
        (("输出匹配", "Match the output"), ("标签、阈值与结果解释", "Labels, thresholds and output decoding")),
        (("现场验证", "Validate on site"), ("独立样本、测量参考与节拍", "Independent samples, measurement references and timing")),
    ]
    for i, (title, desc) in enumerate(checks):
        y = 363+i*99
        c.circle(585, y-8, 5, p["teal"])
        c.text(607, y+2, title, 28, p["cyan"], 600, 706)
        c.text(607, y+42, desc, 25, p["muted"], width=706)
    c.end()
    c.text(64, 737, ("建议统一交付资产", "Recommended delivery baseline"), 34, weight=600, width=1750)
    for i, (image, title, detail) in enumerate([
        ("sf:camera", ("硬件与标定", "Hardware baseline"), ("配置 · 标定 · 参考", "Setup · Calibration · References")),
        ("sf:flow", ("检测流程", "Workflow"), ("步骤 · 参数 · 规则", "Steps · Parameters · Rules")),
        ("hp:export", ("算法与模型", "Model package"), ("版本 · 标签 · 阈值", "Version · Labels · Thresholds")),
        ("sf:report", ("验收与追溯", "Acceptance"), ("样本 · 结论 · 记录", "Samples · Evidence · Records")),
    ]):
        x = 64+i*455
        c.group(title)
        c.rect(x, 766, 427, 228, p["panel"], p["line"], 17)
        c.image(image, x+14, 785, 139, 126)
        c.text(x+169, 824, title, 27, weight=600, width=238)
        c.text(x+213, 956, detail, 24, p["muted"], width=389, anchor="middle")
        c.end()
    return c


def prepare_assets():
    sf_root = SF_VISUALS/"leadership_foundation/assets"
    sf_names = ["machine", "optical", "flow", "core", "camera", "motion", "light", "laser",
                "chip", "calibration", "network", "report", "shield"]
    hp_names = ["logo", "core", "validation", "registry", "detection_scene", "export"]
    for mode in THEMES:
        for name in sf_names:
            shutil.copy2(sf_root/f"{mode}_{name}.png", ASSETS/f"{mode}_sf_{name}.png")
        for name in hp_names:
            shutil.copy2(PARENT/"assets"/f"{mode}_{name}.png", ASSETS/f"{mode}_hp_{name}.png")


def reuse_hawkeye(page, lang, mode, output):
    original_num = page-6
    original_name = hp.NAMES[original_num-1][1]
    path = PARENT/"svg"/f"{original_num:02d}_{original_name}_{hp.file_suffix(lang, mode)}.svg"
    root = ET.fromstring(path.read_bytes())
    for node in root.iter("{"+SVG+"}text"):
        x, y = node.get("x"), node.get("y")
        if x == "86" and y == "77":
            node.text = "第二部分 / HAWKEYEPLUS" if lang == "zh" else "PART 02 / HAWKEYEPLUS"
        if x == "65" and y == "1055":
            node.text = "SwiftFlow + HawkeyePlus  |  Industrial Vision Engineering"
            node.set("data-max-width", "1300")
        if x == "1854" and y == "1055":
            node.text = f"{page:02d}"
        if page == 7 and x == "64" and y == "146":
            node.text = SLIDES[page-1][1 if lang == "zh" else 2]
            node.set("font-size", "54")
    output.write_bytes(ET.tostring(root, encoding="utf-8"))


def speaker_notes(page, lang):
    if page in NOTES:
        text = NOTES[page][0 if lang == "zh" else 1]
    else:
        text = hp.NOTES[page-7] if lang == "zh" else hp.NOTES_EN[page-7]
        text += ("\n本页展示训练平台能力；接入 SwiftFlow 前仍需按目标推理后端适配和验收。"
                 if lang == "zh" else
                 "\nThese are training-platform capabilities. Integration into SwiftFlow requires runtime adaptation and acceptance.")
    return text + ("\n\n依据与说明：" if lang == "zh" else "\n\nSources and interpretation:") + (
        "\nHawkeyePlus: docs/ARCHITECTURE.md; docs/FEATURES.md; src/lib/model-yaml.ts; model registry routes."
        "\nSwiftFlow: docs/Architecture.md; docs/Flow_Engine_Internals.md; InferenceSessionFactory.cs; ModelInstanceConfig.cs."
        "\n" + ("配图包含概念插画与既有项目示例，不作为性能、精度或量产验收证明。所有文字、模块与连线可编辑，图片可独立替换。"
                 if lang == "zh" else
                 "Illustrations and project examples are not performance, accuracy or production-acceptance evidence. "
                 "Native text, panels and lines are editable; pictures are independently replaceable."))


def page_notes(meta, lang):
    if meta["kind"] == "content":
        return speaker_notes(meta["content"], lang)
    copy = {
        "cover": (
            "完整汇报标题为《SwiftFlow + HawkeyePlus：工业检测与测量标准化平台——架构、流程与协同交付》。"
            "先说明总体概念与标准化目的，再分别介绍设备与流程执行平台、模型研发平台，最后说明联合验收和交付基线。",
            "Introduce the complete title: SwiftFlow + HawkeyePlus — Standardized Inspection & Measurement: "
            "Architecture, Workflows & Integrated Delivery. Explain the shared concept and rationale, then the execution "
            "platform and model-development platform, followed by joint acceptance and a delivery baseline."),
        "agenda": (
            "全篇分四章：总体概念与建设目的、SwiftFlow、HawkeyePlus、协同交付，最后总结交流。"
            "目录卡片可在 PowerPoint 放映时点击，跳到对应章节首页。",
            "The presentation has four chapters: Vision & Rationale, SwiftFlow, HawkeyePlus, and Integrated Delivery, "
            "followed by a summary and discussion. Each agenda card links to its chapter divider in slideshow mode."),
        "summary": (
            "收束为三点：统一工程方法、两个互补平台、标准化交付。强调每一步的输入输出与验收依据，"
            "将配置、模型和证据形成可复用资产。再次区分推荐工程方法与已实现自动连接器，不引入新的产品能力承诺。",
            "Recap three ideas: one engineering method, two complementary platforms and a standard delivery baseline. "
            "Emphasize clear inputs, outputs and acceptance evidence, with reusable configuration and model assets. "
            "The recommended engineering method is distinct from an automatic software connector."),
        "closing": (
            "结束正式介绍，围绕优先应用场景、验收与交付基线、可复用流程及模型展开交流。"
            "讨论问题不代表已经确定实施范围或量产承诺。",
            "Close the presentation and discuss priority use cases, acceptance and delivery baselines, and reusable workflows "
            "and models. The discussion prompts do not imply an agreed implementation scope or production commitment."),
    }
    if meta["kind"] == "divider":
        chapter = CHAPTERS[meta["chapter"]]
        return (
            f"章节过渡：{chapter['title'][0]}。本章重点：{chapter['subtitle'][0]}。"
            f"引导问题：{chapter['question'][0]}。"
            if lang == "zh" else
            f"Chapter transition: {chapter['title'][1]}. Focus: {chapter['subtitle'][1]}. "
            f"Guiding question: {chapter['question'][1]}."
        )
    return copy[meta["kind"]][0 if lang == "zh" else 1]


def reframe_page(path, page_number, meta, lang):
    """Apply final slide numbering and chapter labels after composing content."""
    root = ET.fromstring(path.read_bytes())
    for node in root.iter("{"+SVG+"}text"):
        x, y = node.get("x"), node.get("y")
        if x == "1854" and y == "1055":
            node.text = f"{page_number:02d} / {len(PAGES):02d}"
            node.set("data-max-width", "100")
        if x == "86" and y == "77":
            if meta["kind"] == "content":
                chapter_index = meta["chapter"]
                title = CHAPTERS[chapter_index]["title"][0 if lang == "zh" else 1]
                node.text = (f"第 {chapter_index+1:02d} 章 / {title}" if lang == "zh"
                             else f"CHAPTER {chapter_index+1:02d} / {title.upper()}")
            elif meta["kind"] == "agenda":
                node.text = "汇报导览 / SWIFTFLOW + HAWKEYEPLUS" if lang == "zh" else "ROADMAP / SWIFTFLOW + HAWKEYEPLUS"
            elif meta["kind"] == "summary":
                node.text = "总结 / 核心要点" if lang == "zh" else "SUMMARY / KEY TAKEAWAYS"
    path.write_bytes(ET.tostring(root, encoding="utf-8"))


def add_native_sections(prs, lang):
    """Write PowerPoint's native section list, in addition to visual dividers."""
    ns_p = "http://schemas.openxmlformats.org/presentationml/2006/main"
    ns_p14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
    extension_uri = "{521415D9-36F7-43E2-AB2F-B90AF26B5E84}"
    root = prs.part._element
    ext_list = root.find("{"+ns_p+"}extLst")
    if ext_list is None:
        ext_list = ET.SubElement(root, "{"+ns_p+"}extLst")
    for child in list(ext_list):
        if child.get("uri") == extension_uri:
            ext_list.remove(child)
    extension = ET.SubElement(ext_list, "{"+ns_p+"}ext", uri=extension_uri)
    sections = ET.SubElement(extension, "{"+ns_p14+"}sectionLst", nsmap={"p14": ns_p14})
    ids = [s.get("id") for s in prs.slides._sldIdLst]
    for i, (first, names) in enumerate(NATIVE_SECTIONS):
        last = NATIVE_SECTIONS[i+1][0]-1 if i+1 < len(NATIVE_SECTIONS) else len(PAGES)
        name = names[0 if lang == "zh" else 1]
        section_id = "{"+str(uuid.uuid5(uuid.NAMESPACE_URL, "swiftflow-hawkeyeplus/"+lang+"/"+name)).upper()+"}"
        section = ET.SubElement(sections, "{"+ns_p14+"}section", name=name, id=section_id)
        slides = ET.SubElement(section, "{"+ns_p14+"}sldIdLst")
        for slide_id in ids[first-1:last]:
            ET.SubElement(slides, "{"+ns_p14+"}sldId", id=slide_id)


def build_deck(lang, mode, files):
    sys.path.insert(0, str(SF_VISUALS))
    import editable_ppt
    editable_ppt.CACHE = ASSETS/"ppt_cache"
    editable_ppt.ensure_assets(files)
    style = json.loads((SF_VISUALS/"template_reference/style.json").read_text(encoding="utf-8"))
    prs, layout = editable_ppt.new_presentation(style, mode)
    prs.core_properties.title = DECK_TITLE[0 if lang == "zh" else 1]
    prs.core_properties.subject = "Complete leadership presentation: vision, four chapters, standardized workflow and delivery"
    prs.core_properties.author = "SwiftFlow + HawkeyePlus"
    records = []
    for i, path in enumerate(files, 1):
        slide = prs.slides.add_slide(layout)
        records.append(editable_ppt.add_editable_svg(prs, slide, path))
        slide.notes_slide.notes_text_frame.text = page_notes(PAGES[i-1], lang)
    navigation = []
    for link in AGENDA_LINKS:
        navigation.append({
            "target_page": link["target"],
            "object_ids": editable_ppt.link_objects(prs.slides[1], link, prs,
                                                   destination=prs.slides[link["target"]-1]),
        })
    add_native_sections(prs, lang)
    target = HERE/f"{deck_stem(lang, mode)}_Editable.pptx"
    prs.save(target)
    return {"file": target.name, "language": lang, "theme": mode, "slides": records,
            "agenda_navigation": navigation, "native_sections": len(NATIVE_SECTIONS)}


def gallery():
    sections = []
    for lang in LANGUAGES:
        for mode in THEMES:
            key = suffix(lang, mode)
            title = ("中文" if lang == "zh" else "English") + (" · 深蓝 / Dark" if mode == "dark" else " · 明色 / Light")
            name = deck_stem(lang, mode)
            sections.append(f'<section id="{key}"><h2>{title}</h2><p><a href="{name}_Editable.pptx">可编辑 PPT / Editable PPT</a> · <a href="{name}.pdf">PDF</a></p><div class="grid">')
            for i, meta in enumerate(PAGES, 1):
                stem = meta["stem"]+"_"+key
                caption = meta["title"][0 if lang == "zh" else 1]
                sections.append(f'<article><a href="png/{stem}.png"><img loading="lazy" src="png/{stem}.png" alt="{html.escape(caption)}"></a>'
                                f'<h3>{i:02d} · {html.escape(caption)}</h3><p><a href="png/{stem}.png">4K PNG</a> · <a href="svg/{stem}.svg">SVG</a></p></article>')
            sections.append("</div></section>")
    (HERE/"index.html").write_text("""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SwiftFlow + HawkeyePlus | Integrated Platform</title>
<style>*{box-sizing:border-box}body{margin:0;background:#061a2c;color:#f2f8ff;font:16px/1.6 'Segoe UI','Microsoft YaHei',sans-serif}
main{max-width:1500px;margin:48px auto;padding:0 30px}small{color:#59daf5;letter-spacing:2px}h1{font-size:40px;margin:12px 0}
header p{color:#acc7da}h2{margin:48px 0 8px}a{color:#70d5ff;text-decoration:none}a:hover{text-decoration:underline}
nav{display:flex;gap:24px;flex-wrap:wrap}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}
article{border:1px solid #28516a;border-radius:15px;overflow:hidden;background:#0b273d}img{width:100%;display:block}
article h3,article p{margin:15px 22px}footer{margin:40px 0;color:#acc7da}@media(max-width:850px){.grid{grid-template-columns:1fr}h1{font-size:30px}}
</style><main><header><small>SWIFTFLOW + HAWKEYEPLUS</small><h1>工业检测与测量标准化平台</h1>
<p>Architecture, Workflows & Integrated Delivery — a complete leadership presentation.<br>
正式封面 → 可点击目录 → 四章过渡与正文 → 总结 → 讨论交流。20 页 / 20 slides per edition.<br>
四个章节：总体概念与建设目的 / SwiftFlow / HawkeyePlus / 协同交付。</p>
<nav><a href="#zh_dark">中文 · 深蓝</a><a href="#en_dark">English · Dark</a><a href="#zh_light">中文 · 明色</a><a href="#en_light">English · Light</a>
<a href="../Supported_Models.md">模型清单 / Model catalog</a></nav></header>""" + "".join(sections) + """
<footer>Native editable text, panels and connectors; independently replaceable images.<br>
The joint workflow is an engineering method and proposed delivery baseline. Model handoff requires runtime adaptation and acceptance.</footer></main></html>""", encoding="utf-8")


def contact_sheets():
    for lang in LANGUAGES:
        for mode in THEMES:
            height = ((len(PAGES)+3)//4)*270
            sheet = Image.new("RGB", (1920, height), "#061A2C")
            for i, meta in enumerate(PAGES):
                with Image.open(HERE/"png"/(meta["stem"]+"_"+suffix(lang, mode)+".png")) as image:
                    sheet.paste(image.convert("RGB").resize((480, 270), Image.Resampling.LANCZOS),
                                ((i%4)*480, (i//4)*270))
            sheet.save(HERE/f"Preview_{suffix(lang, mode)}.jpg", quality=94)


def main():
    for path in [ASSETS, HERE/"png", HERE/"svg", HERE/"verification"]:
        path.mkdir(exist_ok=True)
    prepare_assets()
    builders = {1: vision, 2: why, 3: workflow, 4: swiftflow, 5: hardware_flow, 6: inspection_measurement, 12: handoff}
    decks = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=hp.EDGE, headless=True)
        page = browser.new_page(device_scale_factor=2)
        for lang in LANGUAGES:
            for mode in THEMES:
                files = []
                for number, meta in enumerate(PAGES, 1):
                    target = HERE/"svg"/(meta["stem"]+"_"+suffix(lang, mode)+".svg")
                    if meta["kind"] == "content":
                        content = meta["content"]
                        if content in builders:
                            builders[content](mode, lang).save(target)
                        else:
                            reuse_hawkeye(content, lang, mode, target)
                    elif meta["kind"] == "divider":
                        chapter_divider(mode, lang, number, meta["chapter"]).save(target)
                    else:
                        fn = {"cover": cover, "agenda": agenda, "summary": summary_slide, "closing": closing}[meta["kind"]]
                        fn(mode, lang, number).save(target)
                    reframe_page(target, number, meta, lang)
                    hp.render(page, target, HERE/"png"/(target.stem+".png"), hp.W, hp.H)
                    files.append(target)
                decks.append(build_deck(lang, mode, files))
        browser.close()
    sources = []
    for root, paths, platform in [(REPO, HP_SOURCES, "HawkeyePlus"), (SF_REPO, SF_SOURCES, "SwiftFlow")]:
        for path in paths:
            sources.append({"platform": platform, "path": path, "sha256": hashlib.sha256((root/path).read_bytes()).hexdigest()})
    (HERE/"manifest.json").write_text(json.dumps({
        "title": DECK_TITLE, "slides_per_deck": len(PAGES), "pages": PAGES, "chapters": CHAPTERS,
        "decks": decks, "sources": sources, "image_size": [3840, 2160],
        "handoff_scope": "Proposed engineering method and delivery baseline; runtime adaptation and joint acceptance required. No automatic connector is claimed.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    gallery()
    contact_sheets()
    print(json.dumps([{"file": d["file"], "slides": len(d["slides"])} for d in decks], indent=2))


if __name__ == "__main__":
    main()
