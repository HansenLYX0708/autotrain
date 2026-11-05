# 缺陷局部覆盖与分组评估审计

面向 COCO 工业缺陷数据的只读审计工具。输出图片/标注问题、精确重复、
局部特征相似候选、类别与模式覆盖、候选来源隔离五折及中文 HTML/Markdown 报告。
不启动训练、不修改原始数据、不自动删除近重复样本。

当前实测报告：[Slider 20261102](../../docs/dataset-audits/slider-20261102/report.html)。
指标的公式、权重、阈值与限制均写入报告及 `metrics.json`。

## 环境

Python 3.8 及以上。当前机器已有依赖，无需重新安装；新环境可执行：

```bash
python -m pip install -r tools/dataset_audit/requirements.txt
```

默认数据布局：

```text
format3/
  train/<file_name>
  val/<file_name>
  annotations3_20261102/
    instance_train_step8.json
    instance_val_step6.json
```

只支持本轮使用的 8-bit 可解码图像和 COCO bbox；文件名来源规则为 `[LR]-数字`。
别的数据集须先调整/确认来源解析，不能将这个正则当作通用批次定义。
类别 ID/名称映射必须一致，集合内 image/annotation/category ID 不得重复。
目前分类别分析要求训练至少两张不同图片、验证至少一条可用缺陷；不足会显式报错。

## 运行

Windows PowerShell 示例：

```powershell
$env:OPENBLAS_NUM_THREADS = "2"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"

python tools/dataset_audit/extract.py `
  --root "G:\datasets\PHO_slider_abs\all_product\format3" `
  --cache "temp/slider-audit" --workers 6

python tools/dataset_audit/analyze.py `
  --cache "temp/slider-audit" `
  --output "docs/dataset-audits/slider-20261102"

python tools/dataset_audit/report.py `
  --cache "temp/slider-audit" `
  --output "docs/dataset-audits/slider-20261102"

python tools/dataset_audit/verify.py `
  --cache "temp/slider-audit" `
  --output "docs/dataset-audits/slider-20261102"
```

可用 `extract.py --annotations --train --val` 指定其他标注目录及文件名。
缓存和输出目录中的同名生成文件会覆盖；原始数据始终只读。
若图片或标注变化，重新提取。分析阶段会验证标注 SHA-256，**不会重读全量图片确认缓存时效**。

## 方法与解释边界

- 全量文件 SHA-256、解码像素 SHA-256。后者包含像素数组形状/类型，不把任意近似图当作精确重复。
- 174 维可解释特征：局部与上下文的 DCT/梯度/LBP/灰度直方图，几何及局部光度。
  局部内容先裁剪再缩放，实际尺度另行保留；无神经网络权重下载。
- 图片存在未解决的完整性问题、任一无效/越界框或精确同图标注冲突，整图暂不进入候选。
  仅以 1e-6 容差忽略浮点边界误差。标注一致性比较包含类别、bbox、segmentation、iscrowd，
  浮点数保留三位小数；不对同图不同标注自动合并。
- 同来源编号、主体位置标识、精确像素形成连通组，再以多标签图片计数分五折。
  精确重复即使标注一致也不跨折；候选 COCO 保留图片的完整原标注。
- 训练/验证各划分只使用对应训练部分拟合缩放、聚类、面积/对比度分位与距离阈值。
  不使用验证特征拟合；分折平衡使用全部已有标签计数，因而候选折属于开发评估设计，
  不能声称是从未接触过的最终测试。
- 每折需重新从通用预训练初始化训练，不能复用见过新验证折图片的业务权重。
  候选划分的构建与特征审计不等于已经执行了五折检测器训练。
- 当前 Q 是公开启发式规则下的评估准备度，不是绝对数据质量或生产覆盖百分比。
  未确认来源语义、缺乏正常品、未经人工校准的特征及未训练模型都作为明确限制保留。
- 稀疏模式保留在 `modes.csv` 中，即便不进入主要覆盖分母，也不得据此删除。
- 归一化位置不是配准后的产品坐标。没有真实掩码，故不报告真实缺陷圆度等轮廓指标。

## 文件与校验

`extract.py` 保存原始统计和特征缓存；`analyze.py` 输出 CSV/JSON、候选 COCO；
`report.py` 生成中文报告和图示。HTML 正文内嵌图像，表格下载链接仍指向相邻文件。

`verify.py` 检查计数与公式、候选五折覆盖/互斥、来源与像素零交叉、
候选 COCO 图内完整标注、报告链接与内嵌图片，以及核心分组/近邻逻辑的回归样例。
这些校验不包含检测器训练或模型效果验证。
