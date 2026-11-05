"""Render a Chinese Markdown report and an offline HTML report from audit results."""
import argparse
import base64
import collections
import json
import math
import re
from pathlib import Path

import markdown
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def pct(x):
    return "%.1f%%" % (100*x)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"]*len(headers)) + " |"]
                     + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def charts(output, audit, metrics, details):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f7f9fc", "axes.facecolor": "#f7f9fc"})
    categories = metrics["categories"]
    ids = sorted(int(c) for c in categories)
    names = [categories[str(c)] for c in ids]
    raw = {}
    for split, meta in audit["provenance"].items():
        d = json.loads(Path(meta["path"]).read_text(encoding="utf-8-sig"))
        raw[split] = {
            c: len({a["image_id"] for a in d["annotations"] if a["category_id"] == c})
            for c in ids}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    y = np.arange(len(ids))
    for ax, split, color in zip(axes, ("train", "val"), ("#2563eb", "#f59e0b")):
        values = [raw[split][c] for c in ids]
        ax.barh(y, values, color=color, height=.65)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values)*1.18)
        ax.set_title("Original %s: positive images" % split)
        for n, value in enumerate(values):
            ax.text(value+max(values)*.015, n, str(value), va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(output/"class_counts.png"), dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4))
    a, b = metrics["current"]["classes"], metrics["candidate"]["classes"]
    x = np.arange(len(ids))
    ax.bar(x-.18, [100*c["coverage"] for c in a], width=.36, label="Original", color="#94a3b8")
    ax.bar(x+.18, [100*c["coverage"] for c in b], width=.36, label="Grouped fold 0", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Reference coverage C (%)")
    ax.set_title("Within-class coverage; train-fitted features for each split")
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(str(output/"coverage.png"), dpi=160)
    plt.close(fig)
    support = metrics["fold_support"]
    matrix = np.array([[next(r["positive_images"] for r in support
                             if r["fold"] == f and r["category_id"] == c) for c in ids] for f in range(5)])
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.imshow(np.log1p(matrix), cmap="Blues", aspect="auto")
    for f in range(5):
        for c in range(len(ids)):
            ax.text(c, f, str(matrix[f, c]), ha="center", va="center",
                    color="white" if matrix[f, c] > 180 else "#152438")
    ax.set_xticks(np.arange(len(ids)))
    ax.set_xticklabels(names)
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(["Fold %d" % f for f in range(5)])
    ax.set_title("Proposed grouped 5-fold plan: positive images (color uses log scale)")
    fig.tight_layout()
    fig.savefig(str(output/"folds.png"), dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 7, figsize=(14, 5))
    for ax, c, name in zip(axes.flat, ids, names):
        recs = [r for r in details["records"] if r["category_id"] == c and r["split"] == "train"
                and r["eligible"]]
        counts, _, _ = np.histogram2d([r["cy"] for r in recs], [r["cx"] for r in recs],
                                      bins=6, range=[[0, 1], [0, 1]])
        ax.imshow(counts/max(counts.sum(), 1), cmap="YlOrRd", vmin=0, vmax=.35, origin="upper")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    axes.flat[-1].axis("off")
    fig.suptitle("Defect centers in normalized image coordinates (NOT registered product coordinates)")
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(str(output/"positions.png"), dpi=160)
    plt.close(fig)


def crop(root, record, size=(280, 150)):
    path = root / record["split"] / record["file_name"]
    with Image.open(path) as original:
        image = original.convert("RGB")
    x, y, w, h = record["bbox"]
    pad = max(16, .5*max(w, h))
    l, t = max(0, int(x-pad)), max(0, int(y-pad))
    r, b = min(image.width, int(math.ceil(x+w+pad))), min(image.height, int(math.ceil(y+h+pad)))
    tile = image.crop((l, t, r, b))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((x-l, y-t, x+w-l, y+h-t), outline=(245, 80, 70), width=1)
    tile.thumbnail(size)
    canvas = Image.new("RGB", size, "#172033")
    canvas.paste(tile, ((size[0]-tile.width)//2, (size[1]-tile.height)//2))
    return canvas


def galleries(output, audit, metrics, details):
    root = Path(audit["root"])
    records = details["records"]
    # One representative from each class's largest insufficiently covered mode.
    gaps = []
    for c in sorted(int(k) for k in metrics["categories"]):
        choices = [g for g in metrics["current"]["gaps"] if g["category_id"] == c]
        if choices:
            gaps.append(max(choices, key=lambda g: g["train_images"]))
    panels = []
    for g in gaps:
        r = records[g["representative_record"]]
        label = metrics["categories"][str(g["category_id"])]
        panels.append((r, "%s mode %d | train %d / val %d images" %
                       (label, g["mode"], g["train_images"], g["val_images"])))
    if panels:
        canvas = Image.new("RGB", (900, math.ceil(len(panels)/3)*195), "#172033")
        draw = ImageDraw.Draw(canvas)
        for n, (r, label) in enumerate(panels):
            x, y = (n%3)*300, (n//3)*195
            canvas.paste(crop(root, r), (x+10, y+30))
            draw.text((x+10, y+7), label, fill="white")
            draw.text((x+10, y+180), "image %s / ann %s" % (r["image_key"], r["annotation_id"]), fill="#bcccdc")
        canvas.save(output/"coverage_gaps.jpg", quality=92)
    pairs = []
    for c in sorted(int(k) for k in metrics["categories"]):
        choices = [p for p in metrics["current"]["nearest_pairs"] if p["category_id"] == c]
        if choices:
            pairs.append(choices[0])
    canvas = Image.new("RGB", (900, len(pairs)*175), "#172033")
    draw = ImageDraw.Draw(canvas)
    for n, p in enumerate(pairs):
        y = n*175
        draw.text((10, y+12), "Class %s | d=%.4f" %
                  (metrics["categories"][str(p["category_id"])], p["distance"]), fill="white")
        draw.text((10, y+32), "Same source: %s" % p["same_group"], fill="#bcccdc")
        for col, field in enumerate(("record_a", "record_b")):
            r = records[p[field]]
            canvas.paste(crop(root, r, (300, 145)), (270+col*310, y+25))
            draw.text((270+col*310, y+6), "%s / ann %s" % (r["image_key"], r["annotation_id"]), fill="white")
    canvas.save(output/"nearest_pairs.jpg", quality=91)
    worst = min(metrics["candidate"]["classes"], key=lambda c: c["coverage"])
    cid = worst["category_id"]
    images = {i["key"]: i for i in details["images"]}
    canvas = Image.new("RGB", (900, 4*195), "#172033")
    draw = ImageDraw.Draw(canvas)
    for side in range(2):
        rs = sorted([r for r in records if r["eligible"] and r["category_id"] == cid
                     and (images[r["image_key"]]["fold"] == 0) == bool(side)],
                    key=lambda r: r["area_fraction"])
        selected = [rs[int(q*(len(rs)-1))] for q in (.1, .25, .4, .6, .75, .9)]
        for n, r in enumerate(selected):
            x, y = (n%3)*300, (side*2+n//3)*195
            canvas.paste(crop(root, r), (x+10, y+28))
            draw.text((x+10, y+5), "%s | %s | %s" %
                      (metrics["categories"][str(cid)], "VAL" if side else "TRAIN", r["source_group"]),
                      fill="white")
            draw.text((x+10, y+180), "area %.3f%% | ann %s" %
                      (100*r["area_fraction"], r["annotation_id"]), fill="#bcccdc")
    canvas.save(output/"source_shift.jpg", quality=92)


def render(output, audit, metrics, details):
    o, current, candidate = metrics["overview"], metrics["current"], metrics["candidate"]
    categories = metrics["categories"]
    raw = {s: json.loads(Path(p["path"]).read_text(encoding="utf-8-sig"))
           for s, p in audit["provenance"].items()}
    raw_counts = {}
    for cid in sorted(int(c) for c in categories):
        raw_counts[cid] = []
        for split in ("train", "val"):
            anns = [a for a in raw[split]["annotations"] if a["category_id"] == cid]
            raw_counts[cid] += [len(anns), len({a["image_id"] for a in anns})]
    categories_short = [categories[str(c["category_id"])] for c in current["classes"]
                        if c["val_unique_images"] < 100 or c["val_effective_groups"] < 5]
    pixel_overlaps = o["cross_split_duplicate_groups"]
    image_issues = "；".join("%s=%s" % (k, v) for k, v in o["image_issue_counts"].items()) or "无"
    bbox_issues = "；".join("%s=%s" % (k, v) for k, v in o["annotation_issue_counts"].items()) or "无"
    mismatch_groups = []
    for component in sorted({i["component"] for i in details["images"]}):
        sources = sorted({i["source_group"] for i in details["images"] if i["component"] == component})
        if len(sources) > 1:
            mismatch_groups.append((component, sources))
    sections = []
    def add(text):
        sections.append(text.strip())
    add("""# Slider 缺陷数据集多维覆盖审计

**审计对象：** `PHO_slider_abs/all_product/format3/annotations3_20261102`  
**报告生成时间：** %s  
**审计范围：** 全量标注、全量可读取图片、缺陷局部及上下文特征、图片内容哈希、来源分组、候选五折评估。  
**执行边界：** 原始图片与标注未修改；本轮未训练检测模型，也未自动删除视觉相似样本。
""" % o["created_at"])
    add("""## 结论与优先级

当前划分的**审计指数 Q 为 %.1f/100**：可靠性分项 R 为 %.1f/100，参考覆盖分项 C 为 %.1f/100。
这是一套公开规则下的“评估准备度”指标，**不是模型准确率、数据真实价值或生产全面性的百分比**。

当前结论：**现有验证集尚不适合作为新来源泛化或低漏检率的充分证据。**
主要依据是来源交叉、少数类支持不足及缺陷模式覆盖缺口。按保守来源连通组重划分的 fold 0
获得 Q=%.1f，但仍需确认来源编号含义、补足稀有类与正常品、完成模型验证，不能因分数提高而直接验收。

1. **P0，修复数据与确认来源。** 全量图像检查发现 %s；标注几何问题为 %s。
   图片内容完全相同的组有 %d 个，额外重复图片 %d 张，其中标注冲突组 %d 个。
   候选划分暂时隔离 %d 张需复核图片，保留原始文件。
2. **P0，重建来源隔离的评估。** 现有验证集有 %d/%d 张图的来源连通组同时出现在训练集。
   精确像素重复跨集合组数为 %d。来源交叉与精确重复是不同问题，不能混为一谈。
3. **P1，补足少数类和缺失模式。** 未达到每类 100 张唯一内容验证图、5 个有效来源组的诊断门槛：
   %s。优先补充其独立新来源，而不是从现有来源中复制或增强。
4. **P1，建立正常品及难负样本集。** 标注文件仅有 %d 张无标注图片，无标注不等于确认正常。
   当前不能评价正常品误报率，也不能证明低误报条件下的缺陷召回。
5. **P2，再开展样本压缩试验。** 先人工校准局部特征相似性，保护稀有模式、来源和困难背景，
   再比较完整集、分层随机子集和代表子集的实际漏检、误报及训练成本。
""" % (current["audit_index"], 100*current["reliability"], 100*current["coverage"],
       candidate["audit_index"], image_issues, bbox_issues, o["duplicate_pixel_groups"],
       o["redundant_pixel_images"], o["annotation_conflict_groups"], o["excluded_images"],
       current["group_overlap_val_images"], current["val_images"], pixel_overlaps,
       "、".join(categories_short), o["unannotated_images"]))
    add("""## 数据来源与完整性

目录名是一个数据版本标签，不能代替文件内容版本。以下 SHA-256 记录用于复现：
""")
    add(table(["集合", "图片", "缺陷实例", "标注文件修改时间", "SHA-256"],
              [[s, p["images"], p["annotations"], p["mtime"], "`"+p["sha256"]+"`"]
               for s, p in audit["provenance"].items()]))
    add("""
共 %s 张图、%s 个标注；成功读取 %s 张图片并提取 %s 个缺陷的特征。
可用于候选划分的图片为 %s 张。因缺图、无法解码、尺寸不符、无效/越界框或精确同图标注冲突，
整张图片进入待复核清单，避免把未处理的缺陷当作背景。被隔离图片与具体原因见
[issues.csv](issues.csv)，原始标注级错误见 [annotation_issues.csv](annotation_issues.csv)。

数据完整性检查不等于标注语义审查：本轮没有确认每个缺陷是否漏标、错类或框得准确。
矩形 segmentation 也不提供真实缺陷轮廓；本轮没有将框当作轮廓计算圆度等指标。
""" % tuple(format(o[k], ",") for k in
            ("images", "annotations", "readable_images", "extracted_defects", "eligible_images")))
    add("## 数量、来源和少数类支持\n\n![原始类别图片数量](class_counts.png)")
    add(table(["类别", "训练实例", "训练图片", "验证实例", "验证图片",
               "有效验证来源组", "候选 fold0 验证图片 / 组"],
              [[categories[str(c["category_id"])]] + raw_counts[c["category_id"]] +
               [c["val_groups"], "%d / %d" % (v["val_positive_images"], v["val_groups"])]
               for c, v in zip(current["classes"], candidate["classes"])]))
    add("""一张图片可以含多个类别，表中图片数不能直接相加。“有效验证来源组”一列是排除待复核图后的实际组数，
不等于下文考虑集中度的“有效来源组数”。实际组数即使达到 5，也可能绝大多数样本来自其中一组。
来源组是保守的文件名/内容连通组，不是已经验证的物理批次。新增来源的真实独立性仍需业务确认。

100 张图、5 个组仅为本报告的诊断目标，不代表工业验收充分性。即使独立样本零漏检，
100 个缺陷样本的单侧 95%% 漏检率上界仍约为 2.95%%；要支持漏检率低于 1%%、0.1%%、0.01%%，
分别约需 299、2,995、29,956 个独立且有代表性的零漏检样本。相关样本会削弱这些结论。
""".replace("%%", "%"))
    add("""## 指标定义：可靠性 R、参考覆盖 C 与审计指数 Q

所有分项归一到 0–1，最后乘 100 展示。类别间采用等权宏平均，避免大类掩盖少数类。
指标权重、聚类粒度和门槛是**工程诊断规则**，尚未用真实漏检损失校准；禁止把它们当作普适质量定律。

**可靠性 R = 0.30 I + 0.15 U + 0.35 G + 0.20 A**

- **I，完整性：** 可进入候选划分的图片数 / 输入图片总数；一图存在未解决问题则整图待复核。
- **U，非重复比例：** 可读取图片的唯一解码像素哈希数 / 可读取图片数；不代表近重复比例。
- **G，来源隔离：** 1 − 验证集中来源连通组见于训练集的图片比例。
- **A，验证支持：** 每类 `sqrt(min(唯一内容验证图片数/100,1) × min(有效来源组数/5,1))` 的宏平均。
  令 p_g 为该类验证图片在来源组 g 的比例（精确像素内容去重后），
  **有效来源组数 = 1 / Σ p_g²**。5 个均匀来源得 5；几乎全部来自单一来源则接近 1。
  这是来源集中度指标，不是已证实的统计独立样本数。

**参考覆盖 C = 0.45 M + 0.35 J + 0.20 F**

- **M，形态代理模式覆盖：** 每类只在其训练集上拟合特征缩放和聚类。训练模式至少包含 5 张图、
  2 个来源组才进入主分母；验证集至少有 3 张图、2 个来源组落入该模式才算覆盖。
  稀疏模式仍全部列入 modes.csv，未纳入主分母不表示可以删除。
- **J，位置×尺度×对比度联合覆盖：** 3×3 归一化位置格、训练面积三分位、
  训练绝对局部对比度三分位，共最多 81 个组合。训练中至少 5 张图、2 个组的格子进入分母；
  验证集至少有 1 张图落入才算覆盖。这是粗覆盖，不保证每格统计精度。
- **F，局部特征覆盖：** 对每个训练缺陷求到验证同类缺陷的最近距离。
  阈值取训练集中“排除相同图像内容后的同类最近邻距离”的 P90；统计距离不超过阈值的比例，
  再按类别宏平均。该阈值只衡量相对接近，尚未校准为业务上的可互相代表。

**Q = 100 × 2RC / (R+C)**，即可靠性和参考覆盖的调和平均。
I、U 在前后对照中固定使用同一原始输入，防止仅通过移除问题记录就虚增这两个分项。

**评分之外的必要门槛：** 来源定义确认、训练/验证无来源交叉、少数类支持足够、正常品评估齐备、
标注复核以及实际模型验证。任一关键门槛缺失，仍标记“待验证”，不以 Q 的高低替代门槛。
生产中的未知缺陷覆盖、未来批次覆盖及正常品误报性能在本轮均为“未测”，不能计算绝对全面性分数。
""")
    terms = [("完整性 I", "integrity"), ("非重复比例 U", "unique_fraction"),
             ("来源隔离 G", "group_isolation"), ("验证支持 A", "adequacy"),
             ("模式覆盖 M", "mode_coverage"), ("联合覆盖 J", "joint_coverage"),
             ("局部特征覆盖 F", "feature_coverage"), ("可靠性 R", "reliability"),
             ("参考覆盖 C", "coverage")]
    add(table(["分项", "当前划分", "分组候选 fold0"],
              [[label, pct(current[key]), pct(candidate[key])] for label, key in terms]
              + [["审计指数 Q", "%.1f / 100" % current["audit_index"],
                  "%.1f / 100" % candidate["audit_index"]]]))
    add(table(["类别", "当前实际 / 有效来源组", "当前最大来源占比",
               "候选实际 / 有效来源组", "候选最大来源占比"],
              [[categories[str(c["category_id"])],
                "%d / %.2f" % (c["val_groups"], c["val_effective_groups"]),
                pct(c["val_top_group_fraction"]),
                "%d / %.2f" % (v["val_groups"], v["val_effective_groups"]),
                pct(v["val_top_group_fraction"])]
               for c, v in zip(current["classes"], candidate["classes"])]))
    add("""## 局部特征与空间覆盖

每个缺陷使用原始分辨率裁剪，另取四周扩展 `max(12px, 0.5×框长边)` 的上下文。
局部和上下文分别计算低频 DCT、分区梯度方向、LBP、灰度直方图，共各 79 维；
另有 8 维几何和 8 维光度特征，总计 174 维。框周围环形区域只作为局部背景代理，
可能包含其他缺陷；残差配准和正常模板特征尚未建立。

每类缩放参数仅用对应训练部分的中位数与 P10–P90 范围拟合，再截断到 ±5。
特征块距离权重为局部 45%、上下文 25%、几何 15%、光度 15%，各块按维数归一。
使用确定种子的 MiniBatchKMeans；每类模式数约 `round(sqrt(训练实例数/10))`，限制在 2–20。
聚类表示相似性代理，不能直接视为真实缺陷亚型。

![归一化缺陷位置](positions.png)

位置坐标尚未进行产品结构配准，存在不同布局混合的可能。集中在少数位置既可能是缺陷机理，
也可能是采集偏差；不能以“位置不均匀”直接判定数据差。
后续应确认产品型号/模板，并按功能区及距结构边缘的距离重算覆盖。

![各类别参考覆盖](coverage.png)
""")
    add(table(["类别", "当前模式覆盖", "联合覆盖", "局部覆盖", "当前 C", "候选 C", "候选验证新颖比例"],
              [[categories[str(c["category_id"])], "%d/%d" % (c["covered_modes"], c["eligible_modes"]),
                pct(c["joint_coverage"]), pct(c["feature_coverage"]), pct(c["coverage"]),
                pct(v["coverage"]), pct(v["val_novelty_fraction"])]
               for c, v in zip(current["classes"], candidate["classes"])]))
    add("""“验证新颖比例”是验证缺陷到训练同类最近距离超出训练经验阈值的比例。
它提示分布差异，**不等同于异常率、错标率或预计漏检率**。当前与候选划分各自仅用训练部分拟合，
因此模式编号、距离单位及阈值不能跨划分直接比较；可对照的是同一规则下的归一化覆盖。
严格分组后 C 可能下降，这通常暴露真实的来源差异，不应为了提高 C 重新混合来源。
""")
    add(table(["聚类数量倍率（当前划分）", "模式覆盖 M", "参考覆盖 C"],
              [["0.7×", pct(metrics["sensitivity"][0]["mode_coverage"]), pct(metrics["sensitivity"][0]["coverage"])],
               ["1.0×", pct(current["mode_coverage"]), pct(current["coverage"])],
               ["1.3×（上限仍为20）", pct(metrics["sensitivity"][1]["mode_coverage"]),
                pct(metrics["sensitivity"][1]["coverage"])]]))
    add("聚类粒度敏感性只检查这一项参数；特征权重、背景窗口和阈值仍需人工样本对校准。")
    worst = min(candidate["classes"], key=lambda c: c["coverage"])
    cid = worst["category_id"]
    imap = {i["key"]: i for i in details["images"]}
    train_rs = [r for r in details["records"] if r["category_id"] == cid and r["eligible"]
                and imap[r["image_key"]]["fold"] != 0]
    val_rs = [r for r in details["records"] if r["category_id"] == cid and r["eligible"]
              and imap[r["image_key"]]["fold"] == 0]
    sources = collections.Counter(r["source_group"] for r in val_rs)
    top_source, top_count = sources.most_common(1)[0]
    add("""### 重点发现：%s 的来源与缺陷表现强关联

候选 fold0 中，该类验证有 %d 个缺陷，其中 %d 个来自 `%s`，占 %.1f%%。
这些是缺陷实例占比；上表来源集中度使用去重后的图片占比，两者分母不同。
下面列出同类缺陷的实际分布差异：
""" % (categories[str(cid)], len(val_rs), top_count, top_source, 100*top_count/len(val_rs)))
    add(table(["属性（中位数）", "分组训练", "分组验证 fold0"],
              [[label, fmt % (np.median([r[key] for r in train_rs])*factor),
                fmt % (np.median([r[key] for r in val_rs])*factor)]
               for label, key, factor, fmt in [
                   ("框面积占整图", "area_fraction", 100, "%.4f%%"),
                   ("框短边 / px", "short_px", 1, "%.1f"),
                   ("中心纵坐标 / 图高", "cy", 1, "%.3f"),
                   ("局部平均灰度 / 255", "gray_mean", 1, "%.3f")]]))
    add("""![同类跨来源差异](source_shift.jpg)

上两行是分组训练、下两行是 fold0 验证，分别按各自面积分位抽取六个局部。
这支持“来源隔离会暴露外观、尺度和位置偏移”的判断；它本身不能判定类别标注错误或模型必然漏检。
应检查该类别是否包含多种缺陷亚型、不同型号或成像条件，并为每种有效表现补充多个独立来源。
不能为追求分数而把这个来源重新拆入训练集。逐类来源画像见
[group_profiles.csv](group_profiles.csv)。
""")
    if (output/"coverage_gaps.jpg").exists():
        add("""### 当前验证支持不足的模式示例

每类展示训练图片数最多的一个支持不足模式的中心代表，红框为原始缺陷框。
这表示对应模式的验证样本/来源未达到本报告门槛，并不表示图中缺陷没有训练数据。
每个模式的图片数、组数和代表记录见 [coverage_gaps.csv](coverage_gaps.csv)、
[modes.csv](modes.csv)，代表记录可通过 defects.csv 的 feature_row 查找。

![覆盖缺口示例](coverage_gaps.jpg)
""")
    add("""### 局部相似候选：需要人工判断

下图每类展示一组高相似的不同图像内容中的局部缺陷。它们并非已确认的可删除重复。
特别检查背景位置、亮暗极性、尺寸以及微小结构是否存在重要差异。
[similarity_review.csv](similarity_review.csv) 提供每类最多 8 对候选及完整文件路径。

![局部相似候选](nearest_pairs.jpg)
""")
    add("""## 来源隔离的候选五折评估

分组规则：解析文件名中的 `[LR]-数字` 作为来源代理；相同主体/位置标识、
或完全相同的解码像素连接在一起，形成不可跨折拆分的连通组。
未知来源不能获得真实独立性的保证。本次未知编号图片 %d 张，原始编号组 %d 个，
合并后连通组 %d 个；其中跨编号合并的连通组 %d 个。

按类别的**图片级多标签**计数和总图片数进行五折平衡；使用固定种子 %d，尝试 400 个贪心候选，
目标仅涉及标签数量与组支持，不使用检测模型结果。图片内所有类别及标注共同随图分配。
fold 0 是预先编号的示例验证折，没有按模型成绩挑选。

![五折类别支持](folds.png)

候选 fold0 训练 %d 张、验证 %d 张。已自动断言：
训练/验证图片键互斥、来源连通组互斥、解码像素哈希互斥，五折验证集并集覆盖全部候选图片。

- [grouped_folds.csv](grouped_folds.csv)：每张候选图的五折编号、来源组、相对路径和类别。
- [fold_support.csv](fold_support.csv)：每折每类的图片与来源组支持。
- [candidate_fold0_train.coco.json](candidate_fold0_train.coco.json)：候选训练 COCO。
- [candidate_fold0_val.coco.json](candidate_fold0_val.coco.json)：候选验证 COCO。

COCO 图片根目录应设置为原 `format3`，`file_name` 已带原始 `train/` 或 `val/` 前缀，
ID 已重新唯一编号，完整图内标注保留。原 train/val 文件夹无需移动。
待复核图不在候选中；一致标注的精确重复图目前仍保留，但被强制分入同一折。

**使用方式：** 分别训练五次，每次使用其他四折训练、当前折评估，合并 out-of-fold 预测，
按类别及来源统计召回、定位误差和置信区间。bootstrap 应以来源组/实物为单位，
不能把同图多个框当成独立观测。五折用于开发评估，不同时冒充独立最终测试。
正式验收另外保留未来新批次；用正常验证集确定阈值后，在最终测试集固定阈值评价。
每折必须从同一通用预训练初始化重新训练，不能从已用原训练集或全集训练过的业务权重继续训练，
因为新验证折含有原训练图片。若用交叉验证反复选择超参数，最终效果仍需未来独立批次确认。

**尚未执行：** 检测器训练、五折性能评估、未来批次测试。这里“零交叉”只对已定义的来源代理、
主体标识和精确像素成立，不证明不同编号没有业务上的关联。
""" % (o["unknown_source_images"], o["source_groups"], o["connected_groups"], len(mismatch_groups),
       metrics["seed"], candidate["train_images"], candidate["val_images"]))
    add("""### 五折特征审计结果

已分别对五个候选验证折完成特征审计，每一折的缩放、聚类和经验阈值都只在其训练部分拟合。
以下是同一规则下的分折结果；均值和范围描述划分敏感性，不是模型性能置信区间。
""")
    add(table(["折", "训练 / 验证图片", "R", "C", "Q", "有效来源支持 A"],
              [[n, "%d / %d" % (e["train_images"], e["val_images"]),
                "%.1f" % (100*e["reliability"]), "%.1f" % (100*e["coverage"]),
                "%.1f" % e["audit_index"], pct(e["adequacy"])]
               for n, e in enumerate(metrics["grouped_evaluations"])]))
    g = metrics["grouped_summary"]
    add("五折平均 **Q=%.1f**（范围 %.1f–%.1f），平均参考覆盖 **C=%.1f%%**（范围 %.1f%%–%.1f%%）。"
        % (g["audit_index"]["mean"], g["audit_index"]["min"], g["audit_index"]["max"],
           100*g["coverage"]["mean"], 100*g["coverage"]["min"], 100*g["coverage"]["max"]))
    class_rows = []
    for index, c in enumerate(current["classes"]):
        cs = [e["classes"][index] for e in metrics["grouped_evaluations"]]
        values = [v["coverage"] for v in cs]
        class_rows.append([categories[str(c["category_id"])], pct(c["coverage"]),
                           pct(float(np.mean(values))), "%s–%s" % (pct(min(values)), pct(max(values))),
                           "%.2f" % min(v["val_effective_groups"] for v in cs)])
    add(table(["类别", "当前 C", "分组五折平均 C", "分组五折 C 范围", "五折最低有效来源组数"], class_rows))
    weakest = sorted(class_rows, key=lambda row: float(row[2].rstrip("%")))[:3]
    add("分组后的覆盖优先补强类别为 **%s**。五折平衡仍受到来源不可拆分的约束；例如某折的少数类"
        "可能只有个位数图片。它不会因为实施了交叉验证就自动获得可靠的单折漏检率估计。"
        % "、".join(row[0] for row in weakest))
    add("""## 改进行动与验收证据
""")
    add(table(["优先级", "行动", "需要形成的证据"],
              [["P0", "确认 L/R 编号、位置后缀、重拍与前缀含义；补充真实来源映射",
                "同一实物/同一批次关联表；据此重算分组，确认来源隔离"],
               ["P0", "恢复缺失图、修正无效框、复核同图标注冲突",
                "issues.csv 每条有处理记录；重新审计无未解释结构错误"],
               ["P1", "补采少数类和 coverage_gaps 中模式，优先新来源",
                "增加有效验证图片/独立组；模式覆盖提升；不是增强后的复制计数"],
               ["P1", "建立经确认的正常品和难负样本集",
                "设备/型号/光照覆盖；固定误报率下的缺陷召回和置信区间"],
               ["P1", "建立产品模板或功能区映射",
                "配准误差检查；按结构区域重算位置和上下文覆盖"],
               ["P2", "人工校准近邻与远邻；比较领域模型局部特征",
                "同类、跨来源的相似/不相似样本对及复核结论；误删风险可量化"],
               ["P2", "在可靠划分上做 70%/50%/30% 子集试验",
                "对照分层随机抽样；控制训练步数，报告每类漏检和误报变化"]]))
    add("""## 可复现性与文件索引

执行代码在 `tools/dataset_audit/`，运行方法见该目录 README。图片提取缓存位于
`temp/slider-audit/`；缓存记录输入标注 SHA-256 与每张图片的文件/像素 SHA-256。
本次提取是全量扫描，不是只对前一轮提到的 79 组候选重复进行检查。
若图片或标注发生变化，应重新运行提取，不能复用旧缓存。

| 文件 | 内容 |
| --- | --- |
| [metrics.json](metrics.json) | 全部分项、每类指标、门槛、随机种子、版本和限制 |
| [images.csv](images.csv) | 每张图的哈希、来源、问题、可用状态与候选折号 |
| [defects.csv](defects.csv) | 每个可提取缺陷的位置、尺寸、对比度及特征行索引 |
| [duplicates.csv](duplicates.csv) | 精确像素重复组、跨集合情况和标注冲突 |
| [issues.csv](issues.csv) | 需要复核的图片 |
| [modes.csv](modes.csv) | 所有代理模式，包括稀疏模式 |
| [coverage_gaps.csv](coverage_gaps.csv) | 达到训练支持、但验证支持不足的模式 |
| [similarity_review.csv](similarity_review.csv) | 待人工判断的局部相似候选 |
| [group_profiles.csv](group_profiles.csv) | 逐类别、来源的数量、尺度、位置和光度画像 |
| [grouped_folds.csv](grouped_folds.csv) | 可复现的五折图片分配 |

CSV 使用 UTF-8 BOM，方便本机 Excel 打开。HTML 报告图像已内嵌，可单文件查看正文；
下载表格与候选 COCO 时需保持旁边的文件可用。
""")
    text = "\n\n".join(sections)+"\n"
    (output/"report.md").write_text(text, encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "toc", "fenced_code"])
    for path in output.glob("*"):
        if path.suffix in (".png", ".jpg"):
            mime = "image/png" if path.suffix == ".png" else "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            body = body.replace('src="%s"' % path.name, 'src="data:%s;base64,%s"' % (mime, encoded))
    nav = "".join('<a href="#%s">%s</a>' % (anchor, re.sub("<.*?>", "", title))
                  for anchor, title in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body))
    cards = '<div class="cards">' + "".join(
        '<div class="card"><span>%s</span><strong>%s</strong><small>%s</small></div>' % item
        for item in [
            ("审计指数 Q", "%.1f" % current["audit_index"], "当前划分 / 100"),
            ("可靠性 R", "%.1f" % (100*current["reliability"]), "来源交叉限制评价"),
            ("参考覆盖 C", "%.1f" % (100*current["coverage"]), "相对于已观测训练数据"),
            ("可用图片", format(o["eligible_images"], ","), "%d 张进入复核清单" % o["excluded_images"])
        ]) + "</div>"
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Slider 数据集覆盖审计</title><style>
:root{color-scheme:light;--ink:#172438;--blue:#2563eb;--muted:#596b81}
*{box-sizing:border-box}body{margin:0;background:#f1f5fa;color:var(--ink);font:15px/1.85 "Microsoft YaHei","Segoe UI",sans-serif}
aside{position:fixed;left:0;top:0;width:245px;height:100vh;overflow:auto;background:#12233b;color:#fff;padding:30px 20px}
aside b{font-size:20px;display:block;margin-bottom:8px}aside small{color:#b6c7df}aside a{display:block;color:#d7e4f8;text-decoration:none;padding:9px 3px;border-bottom:1px solid #ffffff14;font-size:13px}
main{max-width:1320px;margin:0 auto 0 245px;padding:30px 46px 80px}article{background:#fff;padding:32px 38px;border:1px solid #dee5ef;border-radius:16px}
h1{font-size:30px;line-height:1.4;letter-spacing:-.5px}h2{font-size:23px;margin-top:52px;border-top:1px solid #e2e8f0;padding-top:26px;scroll-margin-top:25px}
h3{font-size:18px;margin-top:32px}p,li{max-width:100%;overflow-wrap:anywhere}a{color:#1d5ed5}
table{border-collapse:collapse;width:100%;font-size:12px;display:block;overflow-x:auto;margin:22px 0}
th,td{text-align:left;border-bottom:1px solid #dfe7f1;padding:9px 11px;min-width:65px}
th{background:#eaf0fa;color:#183453;white-space:nowrap}tr:nth-child(even){background:#f7f9fc}
td code{font-size:10px;word-break:break-all}code{background:#f0f3f8;padding:2px 4px;border-radius:4px}
img{max-width:100%;height:auto;display:block;margin:20px auto;border-radius:8px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:4px 0 24px}.card{padding:20px;background:white;border:1px solid #dce5f0;border-radius:13px}
.card span,.card small{display:block;color:var(--muted)}.card strong{font-size:36px;color:var(--blue);display:block;line-height:1.5}
.notice{background:#fff4d6;border-left:4px solid #e3a52b;padding:12px 18px;margin-bottom:22px}
@media(max-width:1000px){aside{position:static;width:auto;height:auto}aside a{display:inline-block;margin-right:14px}main{margin:0;padding:18px}.cards{grid-template-columns:repeat(2,1fr)}article{padding:20px}}
@media print{aside{display:none}main{margin:0;padding:0}article{border:0;padding:0}.cards{grid-template-columns:repeat(4,1fr)}body{background:white;font-size:10pt}h2{break-before:auto}img,table,.cards{break-inside:avoid}a{color:inherit}}
</style></head><body><aside><b>SLIDER DATA AUDIT</b><small>局部特征 · 来源隔离 · 可复现证据</small>__NAV__</aside>
<main>__CARDS__<div class="notice">状态：待验证。分数用于定位改进方向，不能替代来源确认、正常品测试与实际模型性能。</div><article>__BODY__</article></main></body></html>"""
    document = document.replace("__NAV__", nav).replace("__CARDS__", cards).replace("__BODY__", body)
    (output/"report.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.cache/"audit.json").read_text(encoding="utf-8"))
    metrics = json.loads((args.output/"metrics.json").read_text(encoding="utf-8"))
    details = json.loads((args.cache/"analysis_details.json").read_text(encoding="utf-8"))
    charts(args.output, audit, metrics, details)
    galleries(args.output, audit, metrics, details)
    render(args.output, audit, metrics, details)
    print("Report: %s" % (args.output/"report.html"), flush=True)


if __name__ == "__main__":
    main()
